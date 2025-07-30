#!/usr/bin/env python3
"""
plot.py

Load a CSV of experiment results (one row per tau) and
create a combined CI‐length vs τ and coverage vs τ figure.
"""
import argparse
import ast
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_ci_and_cov_vs_tau(
    df: pd.DataFrame,
    alpha_level: float,
    out_dir: Path,
    prefix: str = "ci_cov"
):
    """
    Plot CI length and coverage vs tau in a single figure.

    df:           DataFrame indexed by tau, containing at least columns
                  'mean_len_*' for CI lengths and 'covg_*' for coverages.
    alpha_level:  significance level for coverage reference line
    out_dir:      directory in which to save the figure
    prefix:       filename prefix
    """
    # pick vibrant colors and distinct markers
    colors  = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    markers = ["o", "s", "^", "D"]
    labels  = ["MCAR", "MAR", "Real label", "Real label only"]

    # auto-detect length and coverage columns by prefix
    length_cols   = [c for c in df.columns if c.startswith("mean_len_")]
    coverage_cols = [c for c in df.columns if c.startswith("covg_")]

    # create a 1×2 subplot
    fig, (ax_len, ax_cov) = plt.subplots(1, 2, figsize=(12, 4))

    # -----------------------
    # Plot CI length vs τ
    # -----------------------
    for col, c, m, lab in zip(length_cols, colors, markers, labels):
        ax_len.plot(df.index, df[col],
                    marker=m, linestyle="-", color=c, label=lab)
    ax_len.set_xlabel(r"$\tau$")
    ax_len.set_ylabel("CI length")
    ax_len.grid(True, linestyle="--", alpha=0.5)
    ax_len.legend(title="Method")

    # --------------------------
    # Plot coverage vs τ
    # --------------------------
    for col, c, m, lab in zip(coverage_cols, colors, markers, labels):
        ax_cov.plot(df.index, df[col],
                    marker=m, linestyle="-", color=c, label=lab)
    ax_cov.axhline(
        1 - alpha_level,
        linestyle="--", color="gray",
        label=f"$1-\\alpha = {1-alpha_level:.2f}$"
    )
    ax_cov.set_xlabel(r"$\tau$")
    ax_cov.set_ylabel("CI coverage")
    ax_cov.set_ylim(0.5, 1.05)
    ax_cov.grid(True, linestyle="--", alpha=0.5)
    ax_cov.legend(title="Method", loc="lower right")

    # adjust layout and save
    fig.tight_layout()
    savepath = out_dir / f"{prefix}_vs_tau.pdf"
    fig.savefig(savepath)
    plt.close(fig)
    print(f"[INFO] saved combined plot to {savepath}")


def main():
    parser = argparse.ArgumentParser(description="Plot CI length & coverage vs τ")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the CSV file produced by main.py"
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory in which to save the figure"
    )
    parser.add_argument(
        "--alpha_level",
        type=float,
        default=0.10,
        help="Significance level (default: 0.10)"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # if alpha_opt exists and is a string-form list, parse it
    if "alpha_opt" in df.columns:
        df["alpha_opt"] = df["alpha_opt"].apply(ast.literal_eval)

    # set tau as the index
    df = df.set_index("tau").sort_index()
    print(f"[INFO] τ values: {df.index.tolist()}")

    plot_ci_and_cov_vs_tau(
        df,
        alpha_level=args.alpha_level,
        out_dir=out_dir
    )


if __name__ == "__main__":
    main()
