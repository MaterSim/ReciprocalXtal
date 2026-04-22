from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BUCKET_ORDER = ["small", "medium", "large"]
BUCKET_LABELS = {
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
}


def load_summary(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def style_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )


def bucket_from_summary(summary: dict) -> str:
    return summary["selection"]["size_bucket"]


def extract_runtime_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        bucket = bucket_from_summary(summary)
        pnl_query_time = summary["descriptor_build_seconds"]["reciprocal_power_spectrum"]["query_mean"]
        sm_medium_time = next(
            row["mean_query_runtime_seconds"]
            for row in summary["structurematcher"]
            if row["setting"] == "medium"
        )
        rows.append(
            {
                "bucket": bucket,
                "pnl_query_mean": float(pnl_query_time),
                "structurematcher_medium_mean": float(sm_medium_time),
                "n_queries": int(summary["n_queries"]),
            }
        )
    return sorted(rows, key=lambda row: BUCKET_ORDER.index(row["bucket"]))


def plot_runtime_comparison(rows: list[dict], output_dir: Path) -> None:
    x = np.arange(len(rows))
    width = 0.34

    pnl_times = [row["pnl_query_mean"] for row in rows]
    sm_times = [row["structurematcher_medium_mean"] for row in rows]
    labels = [BUCKET_LABELS[row["bucket"]] for row in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.bar(x - width / 2, pnl_times, width=width, color="#1f4e79", label="Reciprocal P_nl")
    ax.bar(x + width / 2, sm_times, width=width, color="#7a8fa6", label="StructureMatcher (medium)")

    for xpos, value in zip(x - width / 2, pnl_times):
        ax.text(xpos, value * 1.08, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + width / 2, sm_times):
        ax.text(xpos, value * 1.08, f"{value:.3g}", ha="center", va="bottom", fontsize=8)

    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean query time (seconds, log scale)")
    ax.set_title("Runtime Scaling Across Structure Size")
    ax.legend(frameon=False, loc="upper left")

    fig.text(
        0.5,
        0.01,
        "P_nl uses reciprocal descriptor query build mean; StructureMatcher uses medium-setting mean query runtime.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    png_path = output_dir / "runtime_comparison_by_bucket.png"
    pdf_path = output_dir / "runtime_comparison_by_bucket.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure: {png_path}")
    print(f"Wrote figure: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot P_nl vs StructureMatcher runtime across small/medium/large benchmark summaries."
    )
    parser.add_argument(
        "summary_jsons",
        nargs="+",
        type=Path,
        help="Paths to size-bucket benchmark_summary.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/results/figures"),
        help="Directory for exported figures.",
    )
    args = parser.parse_args()

    style_matplotlib()
    summaries = [load_summary(path) for path in args.summary_jsons]
    rows = extract_runtime_rows(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_runtime_comparison(rows, args.output_dir)


if __name__ == "__main__":
    main()
