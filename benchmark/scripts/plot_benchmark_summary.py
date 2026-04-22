from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


METHOD_LABELS = {
    "reciprocal_power_spectrum": "Reciprocal P_nl",
    "raw_gd": "Raw G(d)",
}

METHOD_COLORS = {
    "reciprocal_power_spectrum": "#1f4e79",
    "raw_gd": "#b85c38",
}

SETTING_LABELS = {
    "strict": "Strict",
    "medium": "Medium",
    "loose": "Loose",
}

SETTING_ORDER = ["strict", "medium", "loose"]
SETTING_MARKERS = {
    "strict": "o",
    "medium": "s",
    "loose": "^",
}


def load_summary(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def bucket_label(summary: dict) -> str:
    bucket = summary.get("selection", {}).get("size_bucket")
    if not bucket:
        return "Overview"
    return f"{bucket.capitalize()} Cell"


def style_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )


def get_threshold_rows(summary: dict) -> list[dict]:
    rows = summary["threshold_pairwise"]["summary_rows"]
    return sorted(rows, key=lambda row: (SETTING_ORDER.index(row["structurematcher_setting"]), row["method"]))


def padded_limits(
    values: list[float],
    *,
    pad: float = 0.04,
    lower: float = 0.0,
    upper: float = 1.0,
) -> tuple[float, float]:
    vmin = min(values)
    vmax = max(values)
    if np.isclose(vmin, vmax):
        span = 0.05
    else:
        span = vmax - vmin
    lo = max(lower, vmin - pad * span - 0.02)
    hi = min(upper, vmax + pad * span + 0.02)
    return lo, hi


def plot_f1_panel(ax: plt.Axes, rows: list[dict]) -> None:
    x = np.arange(len(SETTING_ORDER))
    width = 0.34

    for offset, method in [(-width / 2, "reciprocal_power_spectrum"), (width / 2, "raw_gd")]:
        y = [
            next(
                row["evaluation_f1"]
                for row in rows
                if row["method"] == method and row["structurematcher_setting"] == setting
            )
            for setting in SETTING_ORDER
        ]
        ax.bar(
            x + offset,
            y,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

    ax.set_xticks(x, [SETTING_LABELS[setting] for setting in SETTING_ORDER])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Evaluation F1")
    ax.set_title("Descriptor Agreement With StructureMatcher")
    ax.legend(frameon=False, loc="upper right")


def plot_precision_recall_panel(
    ax: plt.Axes,
    rows: list[dict],
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    all_xs: list[float] = []
    all_ys: list[float] = []
    for method in ["reciprocal_power_spectrum", "raw_gd"]:
        method_rows = [row for row in rows if row["method"] == method]
        xs = [row["evaluation_recall"] for row in method_rows]
        ys = [row["evaluation_precision"] for row in method_rows]
        all_xs.extend(xs)
        all_ys.extend(ys)
        ax.plot(
            xs,
            ys,
            linewidth=2.0,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        for row in method_rows:
            setting = row["structurematcher_setting"]
            ax.scatter(
                row["evaluation_recall"],
                row["evaluation_precision"],
                s=64,
                marker=SETTING_MARKERS[setting],
                color=METHOD_COLORS[method],
                edgecolors="none",
                zorder=3,
            )

    auto_xlim = padded_limits(all_xs, pad=0.18)
    auto_ylim = padded_limits(all_ys, pad=0.25)
    ax.set_xlim(*(xlim or auto_xlim))
    ax.set_ylim(*(ylim or auto_ylim))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Tradeoff")
    ax.legend(frameon=False, loc="lower left")


def add_setting_legend(ax: plt.Axes, *, loc: str = "lower right") -> None:
    handles = [
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
    ax.legend(handles=handles, frameon=False, loc=loc, title="StructureMatcher")


def add_figure_caption(fig: plt.Figure, summary: dict) -> None:
    selection = summary["selection"]
    formulas = selection.get("formulas") or selection.get("curated_reference_labels") or selection.get("material_ids")
    formula_text = ", ".join(formulas) if formulas else "custom selection"
    caption = (
        f"Source: {summary['source']} | References: {summary['n_references']} | "
        f"Queries: {summary['n_queries']} | Selection: {formula_text}"
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9, color="#444444")


def add_suptitle(fig: plt.Figure, summary: dict) -> None:
    fig.suptitle(f"{bucket_label(summary)} Benchmark Summary", fontsize=14, y=1.03)


def write_topline_table(summary: dict, rows: list[dict], output_dir: Path) -> None:
    best_row = max(rows, key=lambda row: row["evaluation_f1"])
    lines = [
        "setting,method,evaluation_f1,evaluation_precision,evaluation_recall,evaluation_roc_auc,threshold",
    ]
    for row in rows:
        lines.append(
            ",".join(
                [
                    row["structurematcher_setting"],
                    METHOD_LABELS[row["method"]],
                    f"{row['evaluation_f1']:.4f}",
                    f"{row['evaluation_precision']:.4f}",
                    f"{row['evaluation_recall']:.4f}",
                    f"{row['evaluation_roc_auc']:.4f}",
                    f"{row['threshold']:.6g}",
                ]
            )
        )
    lines.append("")
    lines.append(
        f"best_overall,{SETTING_LABELS[best_row['structurematcher_setting']]},"
        f"{METHOD_LABELS[best_row['method']]},F1={best_row['evaluation_f1']:.4f}"
    )
    (output_dir / "benchmark_visual_summary.csv").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create paper-ready summary plots from benchmark_summary.json."
    )
    parser.add_argument(
        "summary_json",
        type=Path,
        help="Path to benchmark_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for exported figures. Defaults to a 'figures' folder next to the JSON file.",
    )
    parser.add_argument("--pr-xmin", type=float, default=None, help="Optional precision-recall x-axis minimum.")
    parser.add_argument("--pr-xmax", type=float, default=None, help="Optional precision-recall x-axis maximum.")
    parser.add_argument("--pr-ymin", type=float, default=None, help="Optional precision-recall y-axis minimum.")
    parser.add_argument("--pr-ymax", type=float, default=None, help="Optional precision-recall y-axis maximum.")
    args = parser.parse_args()

    summary = load_summary(args.summary_json)
    output_dir = args.output_dir or (args.summary_json.parent / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    style_matplotlib()
    rows = get_threshold_rows(summary)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)
    plot_f1_panel(axes[0], rows)
    xlim = None if args.pr_xmin is None or args.pr_xmax is None else (args.pr_xmin, args.pr_xmax)
    ylim = None if args.pr_ymin is None or args.pr_ymax is None else (args.pr_ymin, args.pr_ymax)
    plot_precision_recall_panel(axes[1], rows, xlim=xlim, ylim=ylim)
    add_suptitle(fig, summary)
    add_figure_caption(fig, summary)

    bucket_slug = summary.get("selection", {}).get("size_bucket") or "overview"
    png_path = output_dir / f"benchmark_summary_{bucket_slug}.png"
    pdf_path = output_dir / f"benchmark_summary_{bucket_slug}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    write_topline_table(summary, rows, output_dir)
    print(f"Wrote figure: {png_path}")
    print(f"Wrote figure: {pdf_path}")
    print(f"Wrote table:  {output_dir / 'benchmark_visual_summary.csv'}")


if __name__ == "__main__":
    main()
