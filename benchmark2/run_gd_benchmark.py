from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from benchmark2.core import (
    DEFAULT_G_BIN_WIDTH,
    DEFAULT_MATCH_PROFILE,
    build_descriptor_pair_rows,
    compute_gd_descriptors,
    evaluate_descriptor_pairwise_matching,
    load_prepared_dataset,
    path_token,
    preprocess_descriptor_map,
    progress,
    summarize_descriptor_runtime,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the benchmark2 G(d) benchmark on a prepared dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Prepared dataset directory containing dataset_manifest.csv and threshold_split.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write results. Defaults under <dataset>/results/gd/<tag>.",
    )
    parser.add_argument("--dmax", type=float, default=10.0)
    parser.add_argument("--g-bin-width", type=float, default=DEFAULT_G_BIN_WIDTH)
    parser.add_argument(
        "--continuous-match-profile",
        type=str,
        default=DEFAULT_MATCH_PROFILE,
        choices=["raw", "normalized", "shape"],
    )
    parser.add_argument(
        "--calibration-source",
        type=str,
        default="queries",
        choices=["queries", "references"],
        help=(
            "Use dataset queries or reference-vs-reference pairs when fitting "
            "the held-out distance threshold."
        ),
    )
    return parser.parse_args()


def default_output_dir(dataset_dir: Path, args: argparse.Namespace) -> Path:
    tag = (
        f"dmax{path_token(args.dmax)}_gbin{path_token(args.g_bin_width)}_"
        f"profile{path_token(args.continuous_match_profile)}_"
        f"calib_{path_token(args.calibration_source)}"
    )
    return dataset_dir / "results" / "gd" / tag


def main() -> None:
    args = parse_args()
    dataset = load_prepared_dataset(args.dataset_dir)
    output_dir = args.output_dir or default_output_dir(dataset.dataset_dir, args)
    progress(
        f"Running G(d) benchmark | dataset={dataset.dataset_dir} | "
        f"calibration={args.calibration_source} | "
        f"profile={args.continuous_match_profile} | g_bin_width={args.g_bin_width}"
    )

    descriptors, build_times = compute_gd_descriptors(
        dataset.all_entries(),
        dmax=args.dmax,
        bin_width=args.g_bin_width,
    )
    processed = preprocess_descriptor_map(
        descriptors,
        method="gd",
        match_profile=args.continuous_match_profile,
        pnl_first_weight=1.0,
    )
    reference_descriptors = {
        entry.entry_id: processed[entry.entry_id] for entry in dataset.references
    }
    query_descriptors = {
        entry.entry_id: processed[entry.entry_id] for entry in dataset.queries
    }

    pair_rows = build_descriptor_pair_rows(
        method="gd",
        references=dataset.references,
        queries=dataset.queries,
        reference_descriptors=reference_descriptors,
        query_descriptors=query_descriptors,
    )
    calibration_pair_rows = None
    if args.calibration_source == "references":
        calibration_pair_rows = build_descriptor_pair_rows(
            method="gd",
            references=dataset.references,
            queries=dataset.references,
            reference_descriptors=reference_descriptors,
            query_descriptors=reference_descriptors,
        )
    prediction_rows, threshold_summary_rows = evaluate_descriptor_pairwise_matching(
        pair_rows=pair_rows,
        split_policy=dataset.threshold_split,
        method="gd",
        match_profile=args.continuous_match_profile,
        pnl_first_weight=1.0,
        calibration_pair_rows=calibration_pair_rows,
        calibration_source=args.calibration_source,
    )
    runtime_rows = summarize_descriptor_runtime(
        method="gd",
        references=dataset.references,
        queries=dataset.queries,
        build_times=build_times,
        pair_rows=pair_rows,
    )

    write_csv(output_dir / "pairwise_distances.csv", pair_rows)
    write_csv(output_dir / "pairwise_predictions.csv", prediction_rows)
    write_csv(output_dir / "threshold_summary.csv", threshold_summary_rows)
    write_csv(output_dir / "runtime_summary.csv", runtime_rows)
    summary = {
        "method": "gd",
        "dataset_dir": str(dataset.dataset_dir),
        "settings": {
            "dmax": args.dmax,
            "g_bin_width": args.g_bin_width,
            "continuous_match_profile": args.continuous_match_profile,
            "calibration_source": args.calibration_source,
        },
        "reference_count": len(dataset.references),
        "query_count": len(dataset.queries),
        "pair_count": len(pair_rows),
        "threshold_summary": threshold_summary_rows,
        "runtime_summary": runtime_rows,
    }
    write_json(output_dir / "benchmark_summary.json", summary)

    print("=== G(d) benchmark complete ===")
    print(f"Dataset:    {dataset.dataset_dir}")
    print(f"Calibration:{args.calibration_source}")
    print(f"Output:     {output_dir.resolve()}")
    for row in threshold_summary_rows:
        print(
            f"{row['bucket']}: eval F1={row['evaluation_f1']:.4f}, "
            f"precision={row['evaluation_precision']:.4f}, "
            f"recall={row['evaluation_recall']:.4f}, "
            f"threshold={row['threshold']:.6g}"
        )


if __name__ == "__main__":
    main()
