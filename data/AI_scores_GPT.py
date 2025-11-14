from openai import OpenAI
import pandas as pd, re, os, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def create_client(api_key: str, provider: str = "openai"):
    """Initialize an OpenAI-compatible client for OpenAI or DeepSeek."""
    if provider.lower() == "deepseek":
        base_url = "https://api.deepseek.com"
    else:
        base_url = "https://api.openai.com/v1"  # Default OpenAI endpoint

    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"[INFO] Using provider: {provider} ({base_url})")
    return client


def ask_llm(client, desc: str, model_name: str = "gpt-4o") -> int:
    """Send a prompt to the model and return an integer score between 80 and 100."""
    prompt = f"""
Give a wine quality score strictly between 80 and 100 (integer only).
Do not output anything except the number.

Description: {desc}"""
    r = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    s = r.choices[0].message.content.strip()
    print("Raw model output:", s)
    m = re.search(r"\d+", s)
    x = int(m.group()) if m else 90
    return min(100, max(80, x))


def row_to_prompt(row):
    """Concatenate all useful fields into a single string for the model."""
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


def main(api_key: str, provider: str = "openai", model_name: str = "gpt-4o"):
    client = create_client(api_key, provider)
    OUT_CSV = "wine_scores_with_gpt.csv"

    df = pd.read_csv("US_wine_removed.csv")

    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Use this line if you want GPT to consider all columns
        futures = {executor.submit(ask_llm, client, row_to_prompt(row), model_name): i for i, row in df.iterrows()}
        # Use this line if you want GPT to consider only the description column
        # futures = {executor.submit(ask_llm, client, d, model_name): i for i, d in enumerate(df["description"].astype(str))}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring wines"):
            i = futures[future]
            results[i] = future.result()

    df["predicted_score"] = results
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(df[["description", "predicted_score"]])
    print("[Done] saved as:", OUT_CSV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wine quality prediction using OpenAI or DeepSeek.")
    parser.add_argument("--key", required=True, help="API key for OpenAI or DeepSeek")
    parser.add_argument("--provider", choices=["openai", "deepseek"], default="openai", help="Choose API provider")
    parser.add_argument("--model", default="gpt-4o", help="Model name, e.g. 'gpt-4o-mini' or 'deepseek-chat'")
    args = parser.parse_args()

    main(args.key, args.provider, args.model)