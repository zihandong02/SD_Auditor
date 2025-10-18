from openai import OpenAI, RateLimitError, APIError
import pandas as pd
import re, os, argparse, time, random, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------- Client ----------
def create_client(api_key: str, provider: str = "openai"):
    """Initialize an OpenAI-compatible client for OpenAI or DeepSeek."""
    base_url = "https://api.deepseek.com" if provider.lower() == "deepseek" else "https://api.openai.com/v1"
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"[INFO] Using provider: {provider} ({base_url})")
    return client

# ---------- Prompt Builders ----------
def row_to_prompt(row):
    """Use multiple fields to help the model score."""
    text = (
        f"Country: {row.get('country', '')}\n"
        f"Province: {row.get('province', '')}\n"
        f"Region_1: {row.get('region_1', '')}\n"
        f"Winery: {row.get('winery', '')}\n"
        f"Variety: {row.get('variety', '')}\n"
        f"Designation: {row.get('designation', '')}\n"
        f"Price: {row.get('price', '')}\n"
        f"Description: {row.get('description', '')}"
    )
    return text

def desc_only_prompt(row):
    return str(row.get("description", ""))

# ---------- Inference ----------
def ask_llm(client, text: str, model_name: str = "gpt-4o") -> int:
    """Call the model and return an integer in [80, 100], with retries for 429."""
    prompt = f"""
Give a wine quality score strictly between 80 and 100 (integer only).
Do not output anything except the number.

Description: {text}
""".strip()

    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=60,
            )
            s = r.choices[0].message.content.strip()
            m = re.search(r"\d+", s)
            x = int(m.group()) if m else 90
            return min(100, max(80, x))
        except RateLimitError:
            wait = min(2 ** attempt, 10) + random.uniform(0, 0.5)
            print(f"[429 rate limit] retry in {wait:.2f}s (attempt {attempt+1}/6)")
            time.sleep(wait)
        except APIError as e:
            wait = 1.5 + random.uniform(0, 0.5)
            print(f"[API error] retry in {wait:.2f}s (attempt {attempt+1}/6): {e}")
            time.sleep(wait)
        except Exception as e:
            print(f"[ask_llm error] {e}")
            return 90
    return 90

# ---------- Utility ----------
def stable_row_id(row) -> str:
    """
    Build a stable ID across runs to support resume.
    Uses md5(title||description) primarily; falls back to description if title missing.
    """
    title = str(row.get("title", ""))
    desc  = str(row.get("description", ""))
    key = f"{title}||{desc}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]

def chunk_indices(n, batch_size):
    """Yield index ranges [start, end) for batching."""
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)

# ---------- Final writer ----------
def _write_pretty_final(df_input_all: pd.DataFrame, checkpoint_csv_path: str, original_cols: list):
    """
    Read checkpoint (with _rid + predicted_score) and merge back to the full input df.
    Produce a clean final file whose columns are: original columns + predicted_score at last (no _rid).
    File name: <checkpoint_name>_final.csv
    """
    final_path = os.path.splitext(checkpoint_csv_path)[0] + "_final.csv"

    if not os.path.exists(checkpoint_csv_path):
        print("[INFO] No checkpoint file yet; skip final export.")
        return

    try:
        checkpoint = pd.read_csv(checkpoint_csv_path)
        if "_rid" not in checkpoint.columns or "predicted_score" not in checkpoint.columns:
            print("[WARN] Checkpoint missing _rid or predicted_score; skip final export.")
            return

        # Ensure input df has _rid (should already)
        if "_rid" not in df_input_all.columns:
            df_input_all = df_input_all.copy()
            df_input_all["_rid"] = df_input_all.apply(stable_row_id, axis=1)

        merged = df_input_all.merge(checkpoint[["_rid", "predicted_score"]], on="_rid", how="left")

        ordered_cols = original_cols + ["predicted_score"]
        merged[ordered_cols].to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"[FINAL] Wrote clean file with predicted_score at last: {final_path}")
    except Exception as e:
        print(f"[WARN] Failed to write final file: {e}")

# ---------- Main ----------
def main(
    api_key: str,
    provider: str = "openai",
    model_name: str = "gpt-4o",
    input_csv: str = "US_wine_removed.csv",
    out_csv: str = "wine_scores_with_gpt_4o.csv",
    batch_size: int = 200,
    sleep_between_batches: float = 3.0,
    max_workers: int = 5,
    desc_only: bool = False,
):
    client = create_client(api_key, provider)

    # Load input
    df = pd.read_csv(input_csv)
    original_cols = list(df.columns)  # record original column order
    # Build stable IDs for resume
    df["_rid"] = df.apply(stable_row_id, axis=1)

    # Load existing output (resume)
    done_ids = set()
    if os.path.exists(out_csv):
        try:
            prev = pd.read_csv(out_csv)
            if "_rid" in prev.columns:
                done_ids = set(prev["_rid"].astype(str))
                print(f"[INFO] Resuming: found {len(done_ids)} already completed rows in {out_csv}")
        except Exception as e:
            print(f"[WARN] Could not read existing output file: {e}")

    # Filter remaining rows
    todo_df = df[~df["_rid"].astype(str).isin(done_ids)].copy()
    total_remaining = len(todo_df)
    print(f"[INFO] Total rows: {len(df)}, remaining to process: {total_remaining}")

    if total_remaining == 0:
        print("[INFO] Nothing to do. All rows already processed.")
        if os.path.exists(out_csv):
            _write_pretty_final(df, out_csv, original_cols)
        return

    # Choose prompt builder
    build_prompt = desc_only_prompt if desc_only else row_to_prompt

    # Ensure checkpoint file has header if creating new
    write_header = not os.path.exists(out_csv)

    # Process in batches
    todo_df = todo_df.reset_index(drop=True)
    for (start, end) in chunk_indices(total_remaining, batch_size):
        batch = todo_df.iloc[start:end].copy()
        print(f"[BATCH] Processing rows {start}..{end-1} ({len(batch)} rows)")

        # Parallel scoring for this batch
        results = [None] * len(batch)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(ask_llm, client, build_prompt(row), model_name): i
                       for i, (_, row) in enumerate(batch.iterrows())}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring batch"):
                i = futures[fut]
                results[i] = fut.result()

        batch["predicted_score"] = results

        # --- Checkpoint append: _rid + original columns + predicted_score ---
        # make sure every original column exists
        for c in original_cols:
            if c not in batch.columns:
                batch[c] = pd.NA
        to_write = batch[["_rid"] + original_cols + ["predicted_score"]].copy()

        to_write.to_csv(out_csv, mode="a", index=False, encoding="utf-8-sig", header=write_header)
        write_header = False
        print(f"[BATCH] Appended {len(to_write)} rows to {out_csv}")

        # --- Update clean final file after each batch ---
        _write_pretty_final(df, out_csv, original_cols)

        # cooldown between batches
        if end < total_remaining and sleep_between_batches > 0:
            print(f"[BATCH] Sleeping {sleep_between_batches}s before next batch...")
            time.sleep(sleep_between_batches)

    print("[Done] All remaining rows processed and saved.")
    _write_pretty_final(df, out_csv, original_cols)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched wine scoring with resume & checkpointing.")
    parser.add_argument("--key", required=True, help="API key for OpenAI or DeepSeek")
    parser.add_argument("--provider", choices=["openai", "deepseek"], default="openai")
    parser.add_argument("--model", default="gpt-4o", help="Model name (e.g., 'gpt-4o' or 'deepseek-chat')")
    parser.add_argument("--input-csv", default="US_wine_removed.csv")
    parser.add_argument("--out-csv", default="wine_scores_with_gpt_4o_1.csv")  # checkpoint file (with _rid)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--sleep-between-batches", type=float, default=3.0)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--desc-only", action="store_true", help="Use only the description column as input")
    args = parser.parse_args()

    main(
        api_key=args.key,
        provider=args.provider,
        model_name=args.model,
        input_csv=args.input_csv,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
        sleep_between_batches=args.sleep_between_batches,
        max_workers=args.max_workers,
        desc_only=args.desc_only,
    )
