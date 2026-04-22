from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_benchmark_summary import (
    METHOD_COLORS,
    METHOD_LABELS,
    SETTING_LABELS,
    SETTING_MARKERS,
    bucket_label,
    get_threshold_rows,
    padded_limits,
    plot_f1_panel,
    plot_precision_recall_panel,
    style_matplotlib,
)


def load_summary(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a single 3-row combined benchmark summary figure."
    )
    parser.add_argument("small_summary", type=Path)
    parser.add_argument("medium_summary", type=Path)
    parser.add_argument("large_summary", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/results/figures"),
        help="Directory for exported figures.",
    )
    args = parser.parse_args()

    summaries = [
        load_summary(args.small_summary),
        load_summary(args.medium_summary),
        load_summary(args.large_summary),
    ]

    style_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))

    all_recalls: list[float] = []
    all_precisions: list[float] = []
    threshold_rows_by_summary = [get_threshold_rows(summary) for summary in summaries]
    for rows in threshold_rows_by_summary:
        all_recalls.extend([row["evaluation_recall"] for row in rows])
        all_precisions.extend([row["evaluation_precision"] for row in rows])

    shared_xlim = padded_limits(all_recalls, pad=0.18)
    shared_ylim = padded_limits(all_precisions, pad=0.25)

    for col_idx, (summary, rows) in enumerate(zip(summaries, threshold_rows_by_summary)):
        plot_f1_panel(axes[0, col_idx], rows)
        plot_precision_recall_panel(
            axes[1, col_idx],
            rows,
            xlim=shared_xlim,
            ylim=shared_ylim,
        )
        label = bucket_label(summary)
        axes[0, col_idx].set_title(label)
        axes[1, col_idx].set_title(label)
        axes[0, col_idx].set_ylabel("" if col_idx > 0 else "Evaluation F1")
        axes[1, col_idx].set_ylabel("" if col_idx > 0 else "Precision")
        legend = axes[0, col_idx].get_legend()
        if legend is not None:
            legend.remove()
        legend = axes[1, col_idx].get_legend()
        if legend is not None:
            legend.remove()

    for col_idx in range(3):
        axes[0, col_idx].set_xlabel("")
        axes[1, col_idx].set_xlabel("Recall")

    fig.suptitle("Benchmark Summary Across Structure Sizes", fontsize=16, y=0.965)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.91, wspace=0.22, hspace=0.34)

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], linewidth=3, label=METHOD_LABELS[method])
        for method in ["reciprocal_power_spectrum", "raw_gd"]
    ]
    setting_handles = [
        Line2D(
            [0],
            [0],
            marker=SETTING_MARKERS[setting],
            color="#555555",
            linestyle="None",
            markersize=7,
            label=SETTING_LABELS[setting],
        )
        for setting in ["strict", "medium", "loose"]
    ]

    axes[0, 0].legend(method_handles, [handle.get_label() for handle in method_handles], loc="upper right", frameon=False)
    axes[1, 0].legend(setting_handles, [handle.get_label() for handle in setting_handles], loc="lower left", frameon=False, title="StructureMatcher")

    png_path = args.output_dir / "benchmark_summary_combined.png"
    pdf_path = args.output_dir / "benchmark_summary_combined.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote figure: {png_path}")
    print(f"Wrote figure: {pdf_path}")


if __name__ == "__main__":
    main()
