from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from benchmark2.core import (
    DEFAULT_MATCH_PROFILE,
    compute_pnl_descriptors,
    build_descriptor_pair_rows,
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
        description="Run the benchmark2 P_nl benchmark on a prepared dataset."
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
        help="Directory to write results. Defaults under <dataset>/results/pnl/<tag>.",
    )
    parser.add_argument("--dmax", type=float, default=10.0)
    parser.add_argument("--nmax", type=int, default=10)
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument(
        "--rbasis",
        type=str,
        default="bessel",
        choices=["bessel", "chebyshev", "gto"],
    )
    parser.add_argument("--normalize-reciprocal", action="store_true")
    parser.add_argument(
        "--continuous-match-profile",
        type=str,
        default=DEFAULT_MATCH_PROFILE,
        choices=["raw", "normalized", "shape"],
    )
    parser.add_argument(
        "--pnl-first-weight",
        type=float,
        default=0.1,
        help="Weight applied to the first P_nl component before distance comparison.",
    )
    return parser.parse_args()


def default_output_dir(dataset_dir: Path, args: argparse.Namespace) -> Path:
    tag = (
        f"dmax{path_token(args.dmax)}_nmax{path_token(args.nmax)}_lmax{path_token(args.lmax)}_"
        f"rbasis{path_token(args.rbasis)}_profile{path_token(args.continuous_match_profile)}_"
        f"pnlw{path_token(args.pnl_first_weight)}_norm{int(args.normalize_reciprocal)}"
    )
    return dataset_dir / "results" / "pnl" / tag


def main() -> None:
    args = parse_args()
    if args.pnl_first_weight < 0.0:
        raise ValueError("--pnl-first-weight must be non-negative.")

    dataset = load_prepared_dataset(args.dataset_dir)
    output_dir = args.output_dir or default_output_dir(dataset.dataset_dir, args)
    progress(
        f"Running P_nl benchmark | dataset={dataset.dataset_dir} | "
        f"profile={args.continuous_match_profile} | rbasis={args.rbasis}"
    )

    descriptors, build_times = compute_pnl_descriptors(
        dataset.all_entries(),
        dmax=args.dmax,
        nmax=args.nmax,
        lmax=args.lmax,
        rbasis=args.rbasis,
        normalize_reciprocal=args.normalize_reciprocal,
    )
    processed = preprocess_descriptor_map(
        descriptors,
        method="pnl",
        match_profile=args.continuous_match_profile,
        pnl_first_weight=args.pnl_first_weight,
    )
    reference_descriptors = {
        entry.entry_id: processed[entry.entry_id] for entry in dataset.references
    }
    query_descriptors = {
        entry.entry_id: processed[entry.entry_id] for entry in dataset.queries
    }

    pair_rows = build_descriptor_pair_rows(
        method="pnl",
        references=dataset.references,
        queries=dataset.queries,
        reference_descriptors=reference_descriptors,
        query_descriptors=query_descriptors,
    )
    prediction_rows, threshold_summary_rows = evaluate_descriptor_pairwise_matching(
        pair_rows=pair_rows,
        split_policy=dataset.threshold_split,
        method="pnl",
        match_profile=args.continuous_match_profile,
        pnl_first_weight=args.pnl_first_weight,
    )
    runtime_rows = summarize_descriptor_runtime(
        method="pnl",
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
        "method": "pnl",
        "dataset_dir": str(dataset.dataset_dir),
        "settings": {
            "dmax": args.dmax,
            "nmax": args.nmax,
            "lmax": args.lmax,
            "rbasis": args.rbasis,
            "normalize_reciprocal": args.normalize_reciprocal,
            "continuous_match_profile": args.continuous_match_profile,
            "pnl_first_weight": args.pnl_first_weight,
        },
        "reference_count": len(dataset.references),
        "query_count": len(dataset.queries),
        "pair_count": len(pair_rows),
        "threshold_summary": threshold_summary_rows,
        "runtime_summary": runtime_rows,
    }
    write_json(output_dir / "benchmark_summary.json", summary)

    print("=== P_nl benchmark complete ===")
    print(f"Dataset:    {dataset.dataset_dir}")
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
