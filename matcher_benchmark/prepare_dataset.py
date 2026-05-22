from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from core import (
    DEFAULT_COORD_NOISE_LEVELS,
    DEFAULT_FORMULAS,
    DEFAULT_LATTICE_NOISE_LEVELS,
    DEFAULT_MAX_AXIS_MULTIPLIER,
    DEFAULT_MAX_PER_FORMULA,
    DEFAULT_MAX_TOTAL_STRUCTURES,
    DEFAULT_MIN_REFERENCE_PARENTS,
    DEFAULT_SEED,
    build_queries_from_references,
    build_threshold_split_policy,
    fetch_mp_references_for_bucket,
    filter_formulas_with_min_parents,
    get_mp_api_key,
    progress,
    save_prepared_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a shared benchmark2 dataset from Materials Project references "
            "for one target size bucket."
        )
    )
    parser.add_argument(
        "--bucket",
        choices=["small", "medium", "large"],
        required=True,
        help="Target size bucket for prepared references and derived queries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the prepared dataset. Defaults to benchmark2/datasets/<bucket>.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Materials Project API key. If omitted, MP_API_KEY is used.",
    )
    parser.add_argument(
        "--formula",
        nargs="*",
        default=DEFAULT_FORMULAS,
        help="Formula list for automatic Materials Project selection.",
    )
    parser.add_argument(
        "--material-id",
        action="append",
        default=[],
        help="Explicit MP material ID to include. Repeat to pin custom parents.",
    )
    parser.add_argument(
        "--max-per-formula",
        type=int,
        default=DEFAULT_MAX_PER_FORMULA,
        help="Maximum accepted parents per formula after bucket preparation.",
    )
    parser.add_argument(
        "--max-total-structures",
        type=int,
        default=DEFAULT_MAX_TOTAL_STRUCTURES,
        help="Maximum accepted parent references across all formulas.",
    )
    parser.add_argument(
        "--min-reference-parents-per-formula",
        type=int,
        default=DEFAULT_MIN_REFERENCE_PARENTS,
        help="Minimum parent count required per formula after bucket preparation.",
    )
    parser.add_argument(
        "--coord-noise",
        type=float,
        nargs="+",
        metavar="COORD_A",
        default=None,
        help=(
            "Coordinate perturbation sigmas in Angstrom. Pass as a space-separated "
            "list to override the default three-level coordinate-noise ladder."
        ),
    )
    parser.add_argument(
        "--lattice-noise",
        type=float,
        nargs="+",
        metavar="LATTICE_EPS",
        default=None,
        help=(
            "Lattice perturbation sigmas. Pass as a space-separated list to override "
            "the default three-level lattice-noise ladder."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument(
        "--max-axis-multiplier",
        type=int,
        default=DEFAULT_MAX_AXIS_MULTIPLIER,
        help="Largest axis repeat to consider when building medium/large reference supercells.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("benchmark2") / "datasets" / args.bucket
    coord_noise_levels = (
        [float(value) for value in args.coord_noise]
        if args.coord_noise
        else list(DEFAULT_COORD_NOISE_LEVELS)
    )
    lattice_noise_levels = (
        [float(value) for value in args.lattice_noise]
        if args.lattice_noise
        else list(DEFAULT_LATTICE_NOISE_LEVELS)
    )

    progress(
        f"Preparing benchmark2 dataset | bucket={args.bucket} | formulas={args.formula} | "
        f"max_per_formula={args.max_per_formula}"
    )
    api_key = get_mp_api_key(args.api_key)
    references = fetch_mp_references_for_bucket(
        api_key=api_key,
        formulas=args.formula,
        explicit_material_ids=args.material_id,
        bucket=args.bucket,
        max_per_formula=args.max_per_formula,
        max_total_structures=args.max_total_structures,
        symprec=args.symprec,
        max_axis_multiplier=args.max_axis_multiplier,
    )
    progress(f"Fetched {len(references)} prepared reference candidates before formula filtering")
    references = filter_formulas_with_min_parents(
        references,
        min_unique_parents=args.min_reference_parents_per_formula,
    )
    progress(f"Retained {len(references)} references after formula parent-count filtering")

    rng = np.random.default_rng(args.seed)
    queries = build_queries_from_references(
        references,
        symprec=args.symprec,
        rng=rng,
        coord_noise_levels=coord_noise_levels,
        lattice_noise_levels=lattice_noise_levels,
    )
    progress(f"Generated {len(queries)} query structures")

    threshold_split = build_threshold_split_policy(references, seed=args.seed)
    metadata = {
        "bucket": args.bucket,
        "reference_count": len(references),
        "query_count": len(queries),
        "selection": {
            "formulas": list(args.formula),
            "material_ids": list(args.material_id),
            "max_per_formula": args.max_per_formula,
            "max_total_structures": args.max_total_structures,
            "min_reference_parents_per_formula": args.min_reference_parents_per_formula,
            "max_axis_multiplier": args.max_axis_multiplier,
        },
        "query_generation": {
            "coord_noise_levels_angstrom": list(coord_noise_levels),
            "lattice_noise_levels_epsilon": list(lattice_noise_levels),
            "paired_noise_levels": [
                {"coord_noise_angstrom": coord, "lattice_noise_epsilon": lattice}
                for lattice in lattice_noise_levels
                for coord in coord_noise_levels
            ],
            "seed": args.seed,
            "variant_families": [
                "equivalent_transform",
                "combined_perturbation",
            ],
        },
    }
    save_prepared_dataset(
        dataset_dir=output_dir,
        references=references,
        queries=queries,
        threshold_split=threshold_split,
        metadata=metadata,
    )
    progress(f"Wrote prepared dataset to {output_dir.resolve()}")
    print("=== Dataset ready ===")
    print(f"Bucket:     {args.bucket}")
    print(f"References: {len(references)}")
    print(f"Queries:    {len(queries)}")
    print(f"Output:     {output_dir.resolve()}")


if __name__ == "__main__":
    main()
