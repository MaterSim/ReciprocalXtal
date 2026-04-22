from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from run_structurematcher_vs_reciprocal import (
    BenchmarkEntry,
    DEFAULT_G_BIN_WIDTH,
    PRIMARY_POOL_SCOPE,
    RECP,
    build_descriptor_pair_rows,
    build_manifest_rows,
    build_threshold_split_policy,
    compute_descriptors,
    evaluate_threshold_pairwise_matching,
    preprocess_descriptor_map,
    progress,
    write_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PNL_FIRST_WEIGHT = 0.1
CACHED_STRUCTUREMATCHER_FILES = (
    "structurematcher_pairs.csv",
    "structurematcher_query_summary.csv",
    "structurematcher_aggregate.csv",
)


def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_cached_summary(cached_results_dir: Path) -> dict | None:
    summary_path = cached_results_dir / "benchmark_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open() as handle:
        return json.load(handle)


def parse_optional_int(value: str | None) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(value)


def parse_aggregate_rows(rows: Sequence[dict]) -> List[dict]:
    parsed: List[dict] = []
    for row in rows:
        parsed.append(
            {
                "setting": row["setting"],
                "true_parent_match_rate": float(row["true_parent_match_rate"]),
                "total_false_positives": int(row["total_false_positives"]),
                "total_false_negatives": int(row["total_false_negatives"]),
                "mean_query_runtime_seconds": float(row["mean_query_runtime_seconds"]),
            }
        )
    return parsed


def resolve_cached_input_path(
    raw_path: str,
    *,
    manifest_path: Path,
    cached_results_dir: Path,
) -> Path:
    path = Path(raw_path)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                REPO_ROOT / path,
                manifest_path.parent / path,
                cached_results_dir / path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve cached structure path '{raw_path}' from {manifest_path}."
    )


def load_cached_entries(
    cached_results_dir: Path,
) -> Tuple[List[BenchmarkEntry], List[BenchmarkEntry]]:
    manifest_path = cached_results_dir / "dataset_manifest.csv"
    manifest_rows = read_csv_rows(manifest_path)
    Structure = __import__("pymatgen.core", fromlist=["Structure"]).Structure

    references: List[BenchmarkEntry] = []
    queries: List[BenchmarkEntry] = []

    for row in manifest_rows:
        raw_path = row.get("path", "")
        if not raw_path:
            raise RuntimeError(
                f"Manifest row for {row.get('entry_id', '<unknown>')} is missing a CIF path."
            )

        structure_path = resolve_cached_input_path(
            raw_path,
            manifest_path=manifest_path,
            cached_results_dir=cached_results_dir,
        )
        structure = Structure.from_file(str(structure_path))
        entry = BenchmarkEntry(
            entry_id=row["entry_id"],
            parent_id=row["parent_id"],
            material_id=row["material_id"],
            formula=row["formula"],
            display_name=row["display_name"],
            family_name=row["family_name"],
            variant=row["variant"],
            variant_family=row["variant_family"],
            structure=structure,
            source=row["source"],
            nsites=int(row["nsites"]),
            spacegroup_symbol=row["spacegroup_symbol"],
            spacegroup_number=parse_optional_int(row.get("spacegroup_number")),
            path=structure_path,
        )
        if row["split"] == "reference":
            references.append(entry)
        elif row["split"] == "query":
            queries.append(entry)
        else:
            raise RuntimeError(f"Unexpected manifest split '{row['split']}' in {manifest_path}.")

    if not references or not queries:
        raise RuntimeError(
            f"Cached manifest at {manifest_path} did not contain both references and queries."
        )
    return references, queries


def copy_cached_structurematcher_outputs(
    cached_results_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in CACHED_STRUCTUREMATCHER_FILES:
        source_path = cached_results_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing cached StructureMatcher output: {source_path}")
        shutil.copy2(source_path, output_dir / filename)


def copy_cached_dataset_tree(
    cached_results_dir: Path,
    output_dir: Path,
    entries: Sequence[BenchmarkEntry],
) -> bool:
    source_dataset_dir = cached_results_dir / "dataset"
    if not source_dataset_dir.exists():
        return False

    destination_dataset_dir = output_dir / "dataset"
    shutil.copytree(source_dataset_dir, destination_dataset_dir, dirs_exist_ok=True)

    source_dataset_dir = source_dataset_dir.resolve()
    for entry in entries:
        if entry.path is None:
            continue
        try:
            relative_path = entry.path.resolve().relative_to(source_dataset_dir)
        except ValueError:
            continue
        entry.path = destination_dataset_dir / relative_path

    return True


def ensure_structurematcher_rows_cover_dataset(
    sm_pair_rows: Sequence[dict],
    references: Sequence[BenchmarkEntry],
    queries: Sequence[BenchmarkEntry],
) -> None:
    reference_ids = {entry.entry_id for entry in references}
    query_ids = {entry.entry_id for entry in queries}

    for row in sm_pair_rows:
        if row["query_id"] not in query_ids:
            raise RuntimeError(
                f"Cached StructureMatcher rows reference unknown query '{row['query_id']}'."
            )
        if row["reference_id"] not in reference_ids:
            raise RuntimeError(
                f"Cached StructureMatcher rows reference unknown reference '{row['reference_id']}'."
            )


def resolve_effective_settings(
    args: argparse.Namespace,
    cached_summary: dict | None,
) -> dict:
    cached_descriptor_settings = {}
    cached_query_generation = {}
    if cached_summary:
        cached_descriptor_settings = cached_summary.get("descriptor_settings", {})
        cached_query_generation = cached_summary.get("query_generation", {})

    return {
        "seed": int(args.seed if args.seed is not None else cached_query_generation.get("seed", 7)),
        "dmax": float(args.dmax if args.dmax is not None else cached_descriptor_settings.get("dmax", 10.0)),
        "nmax": int(args.nmax if args.nmax is not None else cached_descriptor_settings.get("nmax", 10)),
        "lmax": int(args.lmax if args.lmax is not None else cached_descriptor_settings.get("lmax", 10)),
        "rbasis": str(args.rbasis if args.rbasis is not None else cached_descriptor_settings.get("rbasis", "Bessel")),
        "normalize_reciprocal": bool(
            args.normalize_reciprocal
            if args.normalize_reciprocal is not None
            else cached_descriptor_settings.get("normalize_reciprocal", False)
        ),
        "g_bin_width": float(
            args.g_bin_width
            if args.g_bin_width is not None
            else cached_descriptor_settings.get("g_bin_width", DEFAULT_G_BIN_WIDTH)
        ),
        "continuous_match_profile": str(
            args.continuous_match_profile
            if args.continuous_match_profile is not None
            else cached_descriptor_settings.get("continuous_match_profile", "normalized")
        ),
        "pnl_first_weight": float(
            args.pnl_first_weight
            if args.pnl_first_weight is not None
            else cached_descriptor_settings.get("pnl_first_weight", DEFAULT_PNL_FIRST_WEIGHT)
        ),
    }


def summarize_selection_from_entries(references: Sequence[BenchmarkEntry]) -> dict:
    formulas = sorted({entry.formula for entry in references})
    material_ids = sorted({entry.material_id for entry in references})
    return {
        "formulas": formulas,
        "material_ids": material_ids,
        "curated_reference_ids": [],
        "curated_reference_labels": [],
        "max_per_formula": None,
        "max_total_structures": len(references),
        "size_bucket": None,
        "min_sites": min(entry.nsites for entry in references),
        "max_sites": max(entry.nsites for entry in references),
        "min_reference_parents_per_formula": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse a saved benchmark dataset and cached StructureMatcher outputs, "
            "then rerun only the reciprocal/G(d) side with new hyperparameters."
        )
    )
    parser.add_argument(
        "--cached-results-dir",
        type=Path,
        required=True,
        help="Existing benchmark result directory containing dataset_manifest.csv and structurematcher_*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the new descriptor outputs and copied cached artifacts.",
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--copy-dataset",
        dest="copy_dataset",
        action="store_true",
        help="Copy the cached dataset tree into the new output directory (default).",
    )
    dataset_group.add_argument(
        "--reuse-dataset-in-place",
        dest="copy_dataset",
        action="store_false",
        help="Keep the original cached dataset files in place and reuse their manifest paths.",
    )
    parser.set_defaults(copy_dataset=True)

    parser.add_argument("--seed", type=int, default=None, help="Override the held-out split seed.")
    parser.add_argument("--dmax", type=float, default=None)
    parser.add_argument("--nmax", type=int, default=None)
    parser.add_argument("--lmax", type=int, default=None)
    parser.add_argument(
        "--rbasis",
        type=str,
        default=None,
        choices=["Bessel", "chebyshev", "gto"],
    )
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize-reciprocal",
        dest="normalize_reciprocal",
        action="store_true",
        help="Override cached settings and L2-normalize the reciprocal descriptor before P_nl computation.",
    )
    normalize_group.add_argument(
        "--no-normalize-reciprocal",
        dest="normalize_reciprocal",
        action="store_false",
        help="Override cached settings and leave the reciprocal descriptor unnormalized before P_nl computation.",
    )
    parser.set_defaults(normalize_reciprocal=None)
    parser.add_argument("--g-bin-width", type=float, default=None)
    parser.add_argument(
        "--continuous-match-profile",
        type=str,
        default=None,
        choices=["raw", "normalized", "shape"],
    )
    parser.add_argument(
        "--pnl-first-weight",
        type=float,
        default=None,
        help="Override the first-component weighting used before descriptor matching.",
    )
    args = parser.parse_args()

    if args.pnl_first_weight is not None and args.pnl_first_weight < 0.0:
        raise ValueError("--pnl-first-weight must be non-negative.")

    cached_results_dir = args.cached_results_dir
    output_dir = args.output_dir
    if cached_results_dir.resolve() == output_dir.resolve():
        raise RuntimeError("--output-dir must be different from --cached-results-dir.")

    cached_summary = load_cached_summary(cached_results_dir)
    effective_settings = resolve_effective_settings(args, cached_summary)

    progress(f"Loading cached dataset from {cached_results_dir}")
    references, queries = load_cached_entries(cached_results_dir)
    progress(f"Loaded cached entries | references={len(references)} | queries={len(queries)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_cached_structurematcher_outputs(cached_results_dir, output_dir)

    dataset_mode = "reused_in_place"
    if args.copy_dataset:
        dataset_copied = copy_cached_dataset_tree(
            cached_results_dir,
            output_dir,
            list(references) + list(queries),
        )
        if dataset_copied:
            write_csv(output_dir / "dataset_manifest.csv", build_manifest_rows(references, queries))
            dataset_mode = "copied"
            progress(f"Copied cached dataset into {output_dir / 'dataset'}")
        else:
            shutil.copy2(cached_results_dir / "dataset_manifest.csv", output_dir / "dataset_manifest.csv")
            progress("Cached dataset tree not found; reusing the original manifest paths in place")
    else:
        shutil.copy2(cached_results_dir / "dataset_manifest.csv", output_dir / "dataset_manifest.csv")
        progress("Reusing cached dataset files in place")

    sm_pair_rows = read_csv_rows(cached_results_dir / "structurematcher_pairs.csv")
    ensure_structurematcher_rows_cover_dataset(sm_pair_rows, references, queries)
    sm_aggregate_rows = read_csv_rows(cached_results_dir / "structurematcher_aggregate.csv")

    recp = RECP(
        dmax=effective_settings["dmax"],
        nmax=effective_settings["nmax"],
        lmax=effective_settings["lmax"],
        rbasis=effective_settings["rbasis"],
    )
    all_entries = list(references) + list(queries)
    progress(
        "Computing descriptors from cached dataset "
        f"| dmax={effective_settings['dmax']} | nmax={effective_settings['nmax']} "
        f"| lmax={effective_settings['lmax']} | rbasis={effective_settings['rbasis']}"
    )
    descriptors, build_times = compute_descriptors(
        all_entries,
        recp,
        normalize_reciprocal=effective_settings["normalize_reciprocal"],
        g_bin_width=effective_settings["g_bin_width"],
    )

    descriptor_build_summary: Dict[str, dict] = {}
    descriptor_pair_rows_by_method: Dict[str, List[dict]] = {}
    for method_name, method_desc in descriptors.items():
        progress(f"Building cached pair rows for {method_name}")
        processed_desc = preprocess_descriptor_map(
            method_desc,
            method=method_name,
            match_profile=effective_settings["continuous_match_profile"],
            pnl_first_weight=effective_settings["pnl_first_weight"],
        )
        ref_desc = {entry.entry_id: processed_desc[entry.entry_id] for entry in references}
        query_desc = {entry.entry_id: processed_desc[entry.entry_id] for entry in queries}
        descriptor_pair_rows_by_method[method_name] = build_descriptor_pair_rows(
            method_name,
            references,
            queries,
            ref_desc,
            query_desc,
        )

        times = build_times[method_name]
        descriptor_build_summary[method_name] = {
            "reference_total": float(sum(times[entry.entry_id] for entry in references)),
            "query_total": float(sum(times[entry.entry_id] for entry in queries)),
            "reference_mean": float(np.mean([times[entry.entry_id] for entry in references])),
            "query_mean": float(np.mean([times[entry.entry_id] for entry in queries])),
        }

    progress("Evaluating descriptor thresholds against cached StructureMatcher labels")
    threshold_prediction_rows: List[dict] = []
    threshold_summary_rows: List[dict] = []
    threshold_status = "ok"
    threshold_error = None
    threshold_split_policy = None
    try:
        threshold_split_policy = build_threshold_split_policy(
            references,
            seed=effective_settings["seed"],
        )
        threshold_prediction_rows, threshold_summary_rows = evaluate_threshold_pairwise_matching(
            descriptor_pair_rows_by_method=descriptor_pair_rows_by_method,
            sm_pair_rows=sm_pair_rows,
            references=references,
            split_policy=threshold_split_policy,
            match_profile=effective_settings["continuous_match_profile"],
            pnl_first_weight=effective_settings["pnl_first_weight"],
        )
    except RuntimeError as exc:
        threshold_status = "skipped"
        threshold_error = str(exc)
        progress(f"Skipping threshold evaluation: {threshold_error}")

    write_csv(output_dir / "pairwise_threshold_predictions.csv", threshold_prediction_rows)
    write_csv(output_dir / "pairwise_threshold_summary.csv", threshold_summary_rows)

    if cached_summary:
        selection = cached_summary.get("selection", summarize_selection_from_entries(references))
        query_generation = dict(cached_summary.get("query_generation", {}))
    else:
        selection = summarize_selection_from_entries(references)
        query_generation = {}
    query_generation["seed"] = effective_settings["seed"]

    descriptor_settings = {
        "dmax": effective_settings["dmax"],
        "nmax": effective_settings["nmax"],
        "lmax": effective_settings["lmax"],
        "rbasis": effective_settings["rbasis"],
        "normalize_reciprocal": effective_settings["normalize_reciprocal"],
        "g_bin_width": effective_settings["g_bin_width"],
        "continuous_match_profile": effective_settings["continuous_match_profile"],
        "pnl_first_weight": effective_settings["pnl_first_weight"],
        "primary_pool_scope": PRIMARY_POOL_SCOPE,
    }

    summary = {
        "source": "cached_benchmark_reuse",
        "cached_results_dir": str(cached_results_dir),
        "cache_reuse": {
            "dataset_mode": dataset_mode,
            "structurematcher_reused": True,
            "copied_structurematcher_files": list(CACHED_STRUCTUREMATCHER_FILES),
        },
        "preset": None if not cached_summary else cached_summary.get("preset"),
        "n_references": len(references),
        "n_queries": len(queries),
        "selection": selection,
        "query_generation": query_generation,
        "descriptor_settings": descriptor_settings,
        "descriptor_build_seconds": descriptor_build_summary,
        "threshold_pairwise": {
            "status": threshold_status,
            "error": threshold_error,
            "split_policy": threshold_split_policy,
            "summary_rows": threshold_summary_rows,
        },
        "structurematcher": (
            cached_summary.get("structurematcher")
            if cached_summary and cached_summary.get("structurematcher") is not None
            else parse_aggregate_rows(sm_aggregate_rows)
        ),
    }

    summary_path = output_dir / "benchmark_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    progress(f"Wrote reused benchmark summary to {summary_path}")

    print("=== Cached benchmark reuse complete ===")
    print(f"Cached source: {cached_results_dir.resolve()}")
    print(f"Output dir:    {output_dir.resolve()}")
    print(f"Dataset mode:  {dataset_mode}")
    print(f"References:    {len(references)}")
    print(f"Queries:       {len(queries)}")
    print(f"Primary pool:  {PRIMARY_POOL_SCOPE}")
    print(f"Match profile: {effective_settings['continuous_match_profile']}")
    print(f"P_nl first-component weight: {effective_settings['pnl_first_weight']}")
    if threshold_status == "ok":
        print("Primary threshold benchmark:")
        for row in threshold_summary_rows:
            print(
                f"  {row['method']} vs SM-{row['structurematcher_setting']}: "
                f"tau={row['threshold']:.6g}, eval F1={row['evaluation_f1']:.4f}, "
                f"precision={row['evaluation_precision']:.4f}, recall={row['evaluation_recall']:.4f}"
            )
    else:
        print(f"Primary threshold benchmark: skipped ({threshold_error})")


if __name__ == "__main__":
    main()
