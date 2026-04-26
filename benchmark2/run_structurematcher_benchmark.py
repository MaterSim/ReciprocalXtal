from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from benchmark2.core import (
    load_prepared_dataset,
    progress,
    run_structure_matcher,
    summarize_structure_matcher_results,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the benchmark2 StructureMatcher benchmark on a prepared dataset."
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
        help="Directory to write results. Defaults under <dataset>/results/structurematcher/default.",
    )
    parser.add_argument(
        "--pair-timeout-seconds",
        type=float,
        default=30.0,
        help="Per-pair timeout for StructureMatcher.fit. Timed out pairs are recorded as non-matches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_prepared_dataset(args.dataset_dir)
    output_dir = args.output_dir or (
        dataset.dataset_dir / "results" / "structurematcher" / "default"
    )
    progress(f"Running StructureMatcher benchmark | dataset={dataset.dataset_dir}")

    pair_rows, query_rows = run_structure_matcher(
        dataset.references,
        dataset.queries,
        pair_timeout_seconds=args.pair_timeout_seconds,
    )
    summary_rows = summarize_structure_matcher_results(pair_rows, query_rows)

    write_csv(output_dir / "pairwise_results.csv", pair_rows)
    write_csv(output_dir / "query_summary.csv", query_rows)
    write_csv(output_dir / "aggregate_summary.csv", summary_rows)
    summary = {
        "method": "structurematcher",
        "dataset_dir": str(dataset.dataset_dir),
        "settings": [
            {"name": "strict", "ltol": 0.1, "stol": 0.1, "angle_tol": 2.0},
            {"name": "medium", "ltol": 0.2, "stol": 0.3, "angle_tol": 5.0},
            {"name": "loose", "ltol": 0.3, "stol": 0.5, "angle_tol": 10.0},
        ],
        "pair_timeout_seconds": args.pair_timeout_seconds,
        "reference_count": len(dataset.references),
        "query_count": len(dataset.queries),
        "aggregate_summary": summary_rows,
    }
    write_json(output_dir / "benchmark_summary.json", summary)

    print("=== StructureMatcher benchmark complete ===")
    print(f"Dataset:    {dataset.dataset_dir}")
    print(f"Output:     {output_dir.resolve()}")
    for row in summary_rows:
        print(
            f"{row['bucket']} / {row['setting']}: pairwise F1={row['pairwise_f1']:.4f}, "
            f"query true-parent match rate={row['true_parent_match_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
