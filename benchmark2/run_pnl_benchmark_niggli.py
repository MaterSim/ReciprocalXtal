from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from benchmark2.core import (
    DEFAULT_MATCH_PROFILE,
    DEFAULT_MAX_AXIS_MULTIPLIER,
    DatasetBundle,
    BenchmarkEntry,
    build_descriptor_pair_rows,
    canonicalize_reference_structure,
    choose_reference_structure_for_bucket,
    compute_pnl_descriptors,
    evaluate_descriptor_pairwise_matching,
    load_prepared_dataset,
    make_niggli_structure,
    path_token,
    preprocess_descriptor_map,
    progress,
    spacegroup_info,
    summarize_descriptor_runtime,
    write_csv,
    write_json,
)

CANONICALIZATION_MODES = (
    "stored",
    "niggli",
    "primitive_niggli",
    "bucketed_primitive_niggli",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the benchmark2 P_nl benchmark after applying a selectable "
            "canonicalization pipeline to references and queries."
        )
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
        help="Directory to write results. Defaults under <dataset>/results/pnl_niggli/<tag>.",
    )
    parser.add_argument(
        "--canonicalization-mode",
        type=str,
        default="niggli",
        choices=CANONICALIZATION_MODES,
        help=(
            "Structure canonicalization pipeline to apply before descriptor generation. "
            "Use 'bucketed_primitive_niggli' to primitive-standardize first, rebuild a "
            "deterministic bucket-matching supercell, then Niggli-reduce."
        ),
    )
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument(
        "--max-axis-multiplier",
        type=int,
        default=DEFAULT_MAX_AXIS_MULTIPLIER,
        help="Largest axis repeat to consider in bucketed primitive canonicalization.",
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
        f"dmax{path_token(args.dmax)}_nmax{path_token(args.nmax)}_lmax{path_token(args.lmax)}_"
        f"rbasis{path_token(args.rbasis)}_profile{path_token(args.continuous_match_profile)}_"
        f"pnlw{path_token(args.pnl_first_weight)}_norm{int(args.normalize_reciprocal)}_"
        f"calib_{path_token(args.calibration_source)}_"
        f"canon_{path_token(args.canonicalization_mode)}"
    )
    return dataset_dir / "results" / "pnl_niggli" / tag


def canonicalize_structure(
    entry: BenchmarkEntry,
    *,
    canonicalization_mode: str,
    symprec: float,
    max_axis_multiplier: int,
):
    if canonicalization_mode == "stored":
        return entry.structure.copy()

    current = entry.structure.copy()
    if canonicalization_mode in {"primitive_niggli", "bucketed_primitive_niggli"}:
        current = canonicalize_reference_structure(current, symprec=symprec)

    if canonicalization_mode == "bucketed_primitive_niggli":
        bucketed, _ = choose_reference_structure_for_bucket(
            current,
            bucket=entry.bucket,
            max_axis_multiplier=max_axis_multiplier,
        )
        if bucketed is None:
            raise RuntimeError(
                "Bucketed primitive canonicalization could not find a matching supercell "
                f"for entry {entry.entry_id} in bucket '{entry.bucket}'."
            )
        current = bucketed

    if canonicalization_mode in {"niggli", "primitive_niggli", "bucketed_primitive_niggli"}:
        reduced = make_niggli_structure(current)
        if reduced is None:
            raise RuntimeError(
                f"Failed to build canonicalized structure for entry {entry.entry_id} "
                f"using mode '{canonicalization_mode}'."
            )
        current = reduced

    return current


def canonicalize_entry(
    entry: BenchmarkEntry,
    *,
    canonicalization_mode: str,
    symprec: float,
    max_axis_multiplier: int,
) -> BenchmarkEntry:
    structure = canonicalize_structure(
        entry,
        canonicalization_mode=canonicalization_mode,
        symprec=symprec,
        max_axis_multiplier=max_axis_multiplier,
    )
    symbol, number = spacegroup_info(structure, symprec=symprec)
    return replace(
        entry,
        structure=structure,
        nsites=len(structure),
        spacegroup_symbol=symbol,
        spacegroup_number=number,
    )


def make_canonicalized_dataset(dataset: DatasetBundle, args: argparse.Namespace) -> DatasetBundle:
    return DatasetBundle(
        dataset_dir=dataset.dataset_dir,
        references=[
            canonicalize_entry(
                entry,
                canonicalization_mode=args.canonicalization_mode,
                symprec=args.symprec,
                max_axis_multiplier=args.max_axis_multiplier,
            )
            for entry in dataset.references
        ],
        queries=[
            canonicalize_entry(
                entry,
                canonicalization_mode=args.canonicalization_mode,
                symprec=args.symprec,
                max_axis_multiplier=args.max_axis_multiplier,
            )
            for entry in dataset.queries
        ],
        threshold_split=dataset.threshold_split,
        metadata=dataset.metadata,
    )


def main() -> None:
    args = parse_args()
    if args.pnl_first_weight < 0.0:
        raise ValueError("--pnl-first-weight must be non-negative.")

    dataset = make_canonicalized_dataset(load_prepared_dataset(args.dataset_dir), args)
    output_dir = args.output_dir or default_output_dir(dataset.dataset_dir, args)
    method_name = f"pnl_{args.canonicalization_mode}"
    progress(
        f"Running canonicalized P_nl benchmark | dataset={dataset.dataset_dir} | "
        f"mode={args.canonicalization_mode} | calibration={args.calibration_source} | "
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
        method=method_name,
        references=dataset.references,
        queries=dataset.queries,
        reference_descriptors=reference_descriptors,
        query_descriptors=query_descriptors,
    )
    calibration_pair_rows = None
    if args.calibration_source == "references":
        calibration_pair_rows = build_descriptor_pair_rows(
            method=method_name,
            references=dataset.references,
            queries=dataset.references,
            reference_descriptors=reference_descriptors,
            query_descriptors=reference_descriptors,
        )
    prediction_rows, threshold_summary_rows = evaluate_descriptor_pairwise_matching(
        pair_rows=pair_rows,
        split_policy=dataset.threshold_split,
        method=method_name,
        match_profile=args.continuous_match_profile,
        pnl_first_weight=args.pnl_first_weight,
        calibration_pair_rows=calibration_pair_rows,
        calibration_source=args.calibration_source,
    )
    runtime_rows = summarize_descriptor_runtime(
        method=method_name,
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
        "method": method_name,
        "dataset_dir": str(dataset.dataset_dir),
        "settings": {
            "canonicalization_mode": args.canonicalization_mode,
            "symprec": args.symprec,
            "max_axis_multiplier": args.max_axis_multiplier,
            "dmax": args.dmax,
            "nmax": args.nmax,
            "lmax": args.lmax,
            "rbasis": args.rbasis,
            "normalize_reciprocal": args.normalize_reciprocal,
            "continuous_match_profile": args.continuous_match_profile,
            "pnl_first_weight": args.pnl_first_weight,
            "calibration_source": args.calibration_source,
        },
        "reference_count": len(dataset.references),
        "query_count": len(dataset.queries),
        "pair_count": len(pair_rows),
        "threshold_summary": threshold_summary_rows,
        "runtime_summary": runtime_rows,
    }
    write_json(output_dir / "benchmark_summary.json", summary)

    print("=== Canonicalized P_nl benchmark complete ===")
    print(f"Dataset:    {dataset.dataset_dir}")
    print(f"Mode:       {args.canonicalization_mode}")
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
