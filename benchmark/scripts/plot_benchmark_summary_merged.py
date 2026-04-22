from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_benchmark_summary import (
    METHOD_LABELS,
    SETTING_LABELS,
    SETTING_MARKERS,
    SETTING_ORDER,
    padded_limits,
    style_matplotlib,
)


SIZE_ORDER = ["small", "medium", "large"]
SIZE_LABELS = {
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
}
SHORT_METHOD_LABELS = {
    "reciprocal_power_spectrum": "P_nl",
    "raw_gd": "G(d)",
}
SIZE_COLORS = {
    "small": "#0f766e",
    "medium": "#288cad",
    "large": "#6a9c6e",
}
METHOD_HATCH = {
    "reciprocal_power_spectrum": "",
    "raw_gd": "/",
}
METHOD_LINESTYLE = {
    "reciprocal_power_spectrum": "-",
    "raw_gd": "--",
}


def load_summary(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def summary_bucket(summary: dict) -> str:
    return summary["selection"]["size_bucket"]


def threshold_rows(summary: dict) -> list[dict]:
    rows = summary["threshold_pairwise"]["summary_rows"]
    return sorted(rows, key=lambda row: (SETTING_ORDER.index(row["structurematcher_setting"]), row["method"]))


def plot_merged_f1(ax: plt.Axes, rows_by_size: dict[str, list[dict]]) -> None:
    x = np.arange(len(SETTING_ORDER))
    width = 0.12
    offsets = {
        ("small", "reciprocal_power_spectrum"): -2.5 * width,
        ("small", "raw_gd"): -1.5 * width,
        ("medium", "reciprocal_power_spectrum"): -0.5 * width,
        ("medium", "raw_gd"): 0.5 * width,
        ("large", "reciprocal_power_spectrum"): 1.5 * width,
        ("large", "raw_gd"): 2.5 * width,
    }

    for size in SIZE_ORDER:
        rows = rows_by_size[size]
        for method in ["reciprocal_power_spectrum", "raw_gd"]:
            heights = [
                next(
                    row["evaluation_f1"]
                    for row in rows
                    if row["method"] == method and row["structurematcher_setting"] == setting
                )
                for setting in SETTING_ORDER
            ]
            ax.bar(
                x + offsets[(size, method)],
                heights,
                width=width,
                color=SIZE_COLORS[size] if method == "reciprocal_power_spectrum" else "white",
                hatch=METHOD_HATCH[method],
                edgecolor=SIZE_COLORS[size],
                linewidth=1.6,
            )

    ax.set_xticks(x, [SETTING_LABELS[setting] for setting in SETTING_ORDER])
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Evaluation F1")
    ax.set_title("F1 Score Comparison")


def plot_merged_tradeoff(ax: plt.Axes, rows_by_size: dict[str, list[dict]]) -> None:
    all_recalls: list[float] = []
    all_precisions: list[float] = []

    for size in SIZE_ORDER:
        rows = rows_by_size[size]
        for method in ["reciprocal_power_spectrum", "raw_gd"]:
            method_rows = [row for row in rows if row["method"] == method]
            recalls = [row["evaluation_recall"] for row in method_rows]
            precisions = [row["evaluation_precision"] for row in method_rows]
            all_recalls.extend(recalls)
            all_precisions.extend(precisions)
            ax.plot(
                recalls,
                precisions,
                color=SIZE_COLORS[size],
                linestyle=METHOD_LINESTYLE[method],
                linewidth=2.0,
                alpha=0.95,
            )
            for row in method_rows:
                ax.scatter(
                    row["evaluation_recall"],
                    row["evaluation_precision"],
                    facecolors=SIZE_COLORS[size] if method == "reciprocal_power_spectrum" else "white",
                    edgecolors=SIZE_COLORS[size],
                    marker=SETTING_MARKERS[row["structurematcher_setting"]],
                    s=60,
                    linewidths=1.8,
                    zorder=3,
                )

            label_x = recalls[-1]
            label_y = precisions[-1]
            label = f"{SHORT_METHOD_LABELS[method]} ({SIZE_LABELS[size].lower()})"
            x_offset = 0.02
            y_offset = 0.0
            ha = "left"
            if size == "small" and method == "reciprocal_power_spectrum":
                x_offset = 0.03
                y_offset = 0.01
            elif size == "small" and method == "raw_gd":
                x_offset = 0.03
                y_offset = 0.01
            elif size == "medium" and method == "reciprocal_power_spectrum":
                x_offset = 0.02
                y_offset = 0.015
            elif size == "medium" and method == "raw_gd":
                x_offset = 0.02
                y_offset = -0.03
            elif size == "large" and method == "reciprocal_power_spectrum":
                x_offset = -0.01
                y_offset = 0.01
                ha = "right"
            elif size == "large" and method == "raw_gd":
                x_offset = -0.01
                y_offset = -0.01
                ha = "right"

            ax.text(
                label_x + x_offset,
                label_y + y_offset,
                label,
                color=SIZE_COLORS[size],
                fontsize=9,
                ha=ha,
                va="center",
            )

    ax.set_xlim(*padded_limits(all_recalls, pad=0.18))
    ax.set_ylim(*padded_limits(all_precisions, pad=0.30, upper=1.08))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Score Tradeoff")


def plot_runtime_panel(ax: plt.Axes, summaries_by_size: dict[str, dict]) -> None:
    x = np.arange(len(SIZE_ORDER))
    width = 0.32

    pnl_times = [
        summaries_by_size[size]["descriptor_build_seconds"]["reciprocal_power_spectrum"]["query_mean"]
        for size in SIZE_ORDER
    ]
    sm_times = [
        next(
            row["mean_query_runtime_seconds"]
            for row in summaries_by_size[size]["structurematcher"]
            if row["setting"] == "medium"
        )
        for size in SIZE_ORDER
    ]

    pnl_bars = ax.bar(x - width / 2, pnl_times, width=width, color="#0f766e", label="P_nl")
    sm_bars = ax.bar(x + width / 2, sm_times, width=width, color="#94a3b8", label="StructureMatcher")

    def format_seconds(value: float) -> str:
        if value >= 1.0:
            return f"{value:.2f} s"
        if value >= 0.1:
            return f"{value:.2f} s"
        if value >= 0.01:
            return f"{value:.3f} s"
        return f"{value:.2e} s"

    def annotate_bars(bars, values: list[float], color: str) -> None:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.18,
                format_seconds(value),
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
            )

    annotate_bars(pnl_bars, pnl_times, "#0f766e")
    annotate_bars(sm_bars, sm_times, "#475569")

    ax.set_yscale("log")
    all_times = pnl_times + sm_times
    ax.set_ylim(min(all_times) / 1.6, max(all_times) * 2.2)
    ax.set_xticks(x, [SIZE_LABELS[size] for size in SIZE_ORDER])
    ax.set_ylabel("Mean query time (s, log)")
    ax.set_title("Runtime Comparison")
    ax.legend(frameon=False, loc="upper left")


def shared_descriptor_setting(
    summaries_by_size: dict[str, dict],
    key: str,
) -> str:
    values = {
        str(summaries_by_size[size]["descriptor_settings"].get(key))
        for size in SIZE_ORDER
    }
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def add_summary_table(ax: plt.Axes, summaries_by_size: dict[str, dict]) -> None:
    ax.axis("off")
    first = summaries_by_size[SIZE_ORDER[0]]
    formulas = ", ".join(first["selection"]["formulas"])
    descriptor_line = (
        f"dmax={shared_descriptor_setting(summaries_by_size, 'dmax')}, "
        f"n={shared_descriptor_setting(summaries_by_size, 'nmax')}, "
        f"l={shared_descriptor_setting(summaries_by_size, 'lmax')}"
    )
    basis_line = f"basis={shared_descriptor_setting(summaries_by_size, 'rbasis')}"
    noisy_counts = {
        size: sum(
            1
            for value in summaries_by_size[size]["query_generation"]["combined_noise"]
        ) * summaries_by_size[size]["n_references"]
        for size in SIZE_ORDER
    }
    ax.set_title("Dataset And Evaluation Setup", pad=2)

    sections = [
        (
            "Dataset",
            [
                f"Compositions: {formulas}",
                "Cell-size bins:",
                "  Small  < 20 atoms",
                "  Medium 20-50 atoms",
                "  Large  > 50 atoms",
            ],
        ),
        (
            "StructureMatcher",
            [
                "Strict  ltol=0.1, stol=0.1, angle=2",
                "Medium ltol=0.2, stol=0.3, angle=5",
                "Loose   ltol=0.3, stol=0.5, angle=10",
            ],
        ),
        (
            "Descriptor",
            [
                descriptor_line,
                basis_line,
            ],
        ),
        (
            "Counts",
            [
                (
                    "References: "
                    f"small={summaries_by_size['small']['n_references']}, "
                    f"medium={summaries_by_size['medium']['n_references']}, "
                    f"large={summaries_by_size['large']['n_references']}"
                ),
                (
                    "Noisy queries: "
                    f"small={noisy_counts['small']}, "
                    f"medium={noisy_counts['medium']}, "
                    f"large={noisy_counts['large']}"
                ),
            ],
        ),
    ]

    y = 0.88
    for header, lines in sections:
        ax.text(
            0.02,
            y,
            header,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            color="#1f2937",
        )
        y -= 0.052
        for line in lines:
            ax.text(
                0.04,
                y,
                line,
                transform=ax.transAxes,
                fontsize=10.5,
                va="top",
                ha="left",
                color="#374151",
                family="monospace" if "ltol" in line or "dmax=" in line or "basis=" in line else None,
            )
            y -= 0.048
        y -= 0.032


def add_legends(ax_f1: plt.Axes, ax_tradeoff: plt.Axes) -> None:
    size_handles = [
        Patch(facecolor=SIZE_COLORS[size], edgecolor=SIZE_COLORS[size], label=SIZE_LABELS[size])
        for size in SIZE_ORDER
    ]
    method_handles = [
        Patch(
            facecolor="#9ca3af" if method == "reciprocal_power_spectrum" else "white",
            edgecolor="#666666",
            hatch=METHOD_HATCH[method],
            linewidth=1.4,
            label=METHOD_LABELS[method],
        )
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
        for setting in SETTING_ORDER
    ]

    legend_sizes = ax_f1.legend(
        handles=size_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.6,
        frameon=False,
        title="Size",
    )
    ax_f1.add_artist(legend_sizes)
    ax_f1.legend(handles=method_handles, loc="upper right", frameon=False, title="Descriptor")
    ax_tradeoff.legend(handles=setting_handles, loc="lower left", frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a merged benchmark figure with small/medium/large encoded by color."
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

    summaries = [load_summary(args.small_summary), load_summary(args.medium_summary), load_summary(args.large_summary)]
    rows_by_size = {summary_bucket(summary): threshold_rows(summary) for summary in summaries}
    summaries_by_size = {summary_bucket(summary): summary for summary in summaries}

    style_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))
    plot_merged_f1(axes[0, 0], rows_by_size)
    plot_merged_tradeoff(axes[0, 1], rows_by_size)
    plot_runtime_panel(axes[1, 0], summaries_by_size)
    add_summary_table(axes[1, 1], summaries_by_size)
    add_legends(axes[0, 0], axes[0, 1])
    axes[0, 0].grid(False)
    axes[1, 0].grid(False)

    fig.suptitle("Benchmarking Reciprocal-Space Matching Across Crystal Size Regimes", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.90, wspace=0.28, hspace=0.32)

    png_path = args.output_dir / "benchmark_summary_merged.png"
    pdf_path = args.output_dir / "benchmark_summary_merged.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote figure: {png_path}")
    print(f"Wrote figure: {pdf_path}")


if __name__ == "__main__":
    main()
