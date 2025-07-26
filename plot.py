# plot_single_run.py

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json

def plot_from_dir(run_dir: Path):
    df_path = run_dir / "results.csv"
    params_path = run_dir / "params.json"

    if not df_path.exists() or not params_path.exists():
        raise FileNotFoundError("Missing results.csv or params.json in specified directory.")

    df = pd.read_csv(df_path, index_col="tau")

    with open(params_path) as f:
        args = json.load(f)["args"]

    alpha_level = float(args["alpha_level"])

    # Plot CI length
    plt.figure(figsize=(7, 3.5))
    plt.plot(df.index, df["mean_len_opt"],  "o-", label="opt-alpha")
    plt.plot(df.index, df["mean_len_base"], "o-", label="base-alpha")
    plt.plot(df.index, df["mean_len_ols"], "o-", label="ols")
    plt.xlabel(r"$\tau$")
    plt.ylabel("CI length (θ₁)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "ci_length_vs_tau.pdf")
    plt.close()

    # Plot Coverage
    plt.figure(figsize=(7, 3.5))
    plt.plot(df.index, df["covg_opt"],  "o-", label="opt-alpha")
    plt.plot(df.index, df["covg_base"], "o-", label="base-alpha")
    plt.plot(df.index, df["covg_ols"], "o-", label="ols")
    plt.axhline(1 - alpha_level, ls="--", color="gray")
    plt.ylim(0.5, 1.05)
    plt.xlabel(r"$\tau$")
    plt.ylabel("Coverage (θ₁)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "coverage_vs_tau.pdf")
    plt.close()

    print(f"[INFO] Plots saved to {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to single run result directory (e.g., results/2024-07-23T15-00-00)")
    args = parser.parse_args()

    plot_from_dir(Path(args.run_dir))
