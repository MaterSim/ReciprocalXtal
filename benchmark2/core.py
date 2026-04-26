from __future__ import annotations

import csv
import json
import os
import re
import signal
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

if TYPE_CHECKING:
    from reciprocal import RECP


DEFAULT_FORMULAS = ["C", "SiO2", "TiO2"]
DEFAULT_COORD_NOISE_LEVELS: List[float] = [0.002, 0.015, 0.050]
DEFAULT_LATTICE_NOISE_LEVELS: List[float] = [0.003, 0.020, 0.060]
DEFAULT_G_BIN_WIDTH = 0.02
DEFAULT_MATCH_PROFILE = "normalized"
DEFAULT_ORIGIN_SHIFT = np.array([0.173, 0.257, 0.389], dtype=float)
DEFAULT_MIN_REFERENCE_PARENTS = 2
DEFAULT_MAX_PER_FORMULA = 20
DEFAULT_MAX_TOTAL_STRUCTURES = 60
DEFAULT_MAX_AXIS_MULTIPLIER = 6
DEFAULT_SEED = 7

CELL_SIZE_BUCKETS: Dict[str, dict] = {
    "small": {"min_sites": 1, "max_sites": 39},
    "medium": {"min_sites": 40, "max_sites": 99},
    "large": {"min_sites": 100, "max_sites": None},
}


@dataclass
class BenchmarkEntry:
    entry_id: str
    parent_id: str
    material_id: str
    formula: str
    display_name: str
    family_name: str
    variant: str
    variant_family: str
    bucket: str
    structure: Any
    source: str
    nsites: int
    spacegroup_symbol: str
    spacegroup_number: int | None
    path: Path | None = None


@dataclass
class DatasetBundle:
    dataset_dir: Path
    references: List[BenchmarkEntry]
    queries: List[BenchmarkEntry]
    threshold_split: dict
    metadata: dict | None = None

    def all_entries(self) -> List[BenchmarkEntry]:
        return list(self.references) + list(self.queries)


@dataclass
class MatcherSetting:
    name: str
    ltol: float
    stol: float
    angle_tol: float


class StructureMatcherPairTimeoutError(TimeoutError):
    pass


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _raise_structure_matcher_timeout(signum: int, frame: Any) -> None:
    raise StructureMatcherPairTimeoutError("StructureMatcher pair evaluation timed out.")


def run_with_timeout(
    timeout_seconds: float | None,
    func: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if timeout_seconds is None or timeout_seconds <= 0:
        return func(*args, **kwargs)
    if not hasattr(signal, "setitimer"):
        return func(*args, **kwargs)

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_structure_matcher_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _structure_cls() -> Any:
    return __import__("pymatgen.core", fromlist=["Structure"]).Structure


def _structure_matcher_cls() -> Any:
    return __import__(
        "pymatgen.analysis.structure_matcher",
        fromlist=["StructureMatcher"],
    ).StructureMatcher


def _spacegroup_analyzer_cls() -> Any:
    return __import__(
        "pymatgen.symmetry.analyzer",
        fromlist=["SpacegroupAnalyzer"],
    ).SpacegroupAnalyzer


def _mp_rester_cls() -> Any:
    return __import__("pymatgen.ext.matproj", fromlist=["MPRester"]).MPRester


def sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def path_token(value: float | int | str) -> str:
    text = f"{value}"
    return sanitize_token(text.replace(".", "p"))


def normalize_rbasis_name(value: str) -> str:
    lowered = str(value).strip().lower()
    if lowered not in {"bessel", "chebyshev", "gto"}:
        raise ValueError(f"Unsupported radial basis: {value}")
    return lowered


def bucket_contains(bucket: str, nsites: int) -> bool:
    spec = CELL_SIZE_BUCKETS[bucket]
    if nsites < int(spec["min_sites"]):
        return False
    max_sites = spec["max_sites"]
    if max_sites is not None and nsites > int(max_sites):
        return False
    return True


def bucket_label_for_nsites(nsites: int) -> str:
    for bucket in ("small", "medium", "large"):
        if bucket_contains(bucket, nsites):
            return bucket
    raise ValueError(f"No bucket matched site count {nsites}")


def get_mp_api_key(cli_value: str | None) -> str:
    api_key = cli_value or os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Materials Project access requires an API key. Pass --api-key or set MP_API_KEY."
        )
    return api_key


def spacegroup_info(structure: Any, symprec: float) -> Tuple[str, int | None]:
    SpacegroupAnalyzer = _spacegroup_analyzer_cls()
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        symbol = analyzer.get_space_group_symbol()
        number = analyzer.get_space_group_number()
        return str(symbol), int(number)
    except Exception:
        return "unknown", None


def canonicalize_reference_structure(structure: Any, symprec: float) -> Any:
    SpacegroupAnalyzer = _spacegroup_analyzer_cls()
    try:
        return SpacegroupAnalyzer(
            structure,
            symprec=symprec,
        ).get_primitive_standard_structure()
    except Exception:
        return structure.copy()


def make_entry(
    *,
    parent_id: str,
    material_id: str,
    formula: str,
    display_name: str,
    family_name: str,
    variant: str,
    variant_family: str,
    bucket: str,
    structure: Any,
    source: str,
    symprec: float,
    path: Path | None = None,
) -> BenchmarkEntry:
    spacegroup_symbol, spacegroup_number = spacegroup_info(structure, symprec=symprec)
    entry_id = f"{bucket}::{parent_id}::{variant}"
    return BenchmarkEntry(
        entry_id=entry_id,
        parent_id=parent_id,
        material_id=material_id,
        formula=formula,
        display_name=display_name,
        family_name=family_name,
        variant=variant,
        variant_family=variant_family,
        bucket=bucket,
        structure=structure,
        source=source,
        nsites=len(structure),
        spacegroup_symbol=spacegroup_symbol,
        spacegroup_number=spacegroup_number,
        path=path,
    )


def structure_to_ase_atoms(structure: Any) -> Any:
    if hasattr(structure, "to_ase_atoms"):
        return structure.to_ase_atoms()
    adaptor = __import__("pymatgen.io.ase", fromlist=["AseAtomsAdaptor"]).AseAtomsAdaptor()
    return adaptor.get_atoms(structure)


def sorted_supercell_multipliers(max_axis_multiplier: int) -> List[Tuple[int, int, int]]:
    candidates: List[Tuple[int, int, int]] = []
    for a in range(1, max_axis_multiplier + 1):
        for b in range(1, a + 1):
            for c in range(1, b + 1):
                candidates.append((a, b, c))
    candidates.sort(
        key=lambda triple: (
            triple[0] * triple[1] * triple[2],
            (triple[0] - triple[1]) + (triple[1] - triple[2]),
            -triple[0],
            -triple[1],
            -triple[2],
        )
    )
    return candidates


def make_supercell_structure(structure: Any, multiplier: Sequence[int]) -> Any:
    supercell = structure.copy()
    supercell.make_supercell(list(multiplier))
    return supercell


def choose_reference_structure_for_bucket(
    structure: Any,
    *,
    bucket: str,
    max_axis_multiplier: int,
) -> Tuple[Any | None, Tuple[int, int, int]]:
    if bucket_contains(bucket, len(structure)):
        return structure.copy(), (1, 1, 1)

    if bucket == "small":
        return None, (1, 1, 1)

    for multiplier in sorted_supercell_multipliers(max_axis_multiplier):
        if multiplier == (1, 1, 1):
            continue
        candidate = make_supercell_structure(structure, multiplier)
        if bucket_contains(bucket, len(candidate)):
            return candidate, multiplier
    return None, (1, 1, 1)


def filter_formulas_with_min_parents(
    references: Sequence[BenchmarkEntry],
    *,
    min_unique_parents: int,
) -> List[BenchmarkEntry]:
    formula_to_parent_ids: Dict[str, set[str]] = defaultdict(set)
    for reference in references:
        formula_to_parent_ids[reference.formula].add(reference.parent_id)

    allowed_formulas = {
        formula
        for formula, parent_ids in formula_to_parent_ids.items()
        if len(parent_ids) >= min_unique_parents
    }
    filtered = [reference for reference in references if reference.formula in allowed_formulas]
    if not filtered:
        raise RuntimeError(
            "No formula retained enough polymorph parents for held-out evaluation. "
            f"Need at least {min_unique_parents} parent structures per formula."
        )
    return filtered


def fetch_mp_references_for_bucket(
    *,
    api_key: str,
    formulas: Sequence[str],
    explicit_material_ids: Sequence[str],
    bucket: str,
    max_per_formula: int,
    max_total_structures: int,
    symprec: float,
    max_axis_multiplier: int,
) -> List[BenchmarkEntry]:
    MPRester = _mp_rester_cls()
    references: List[BenchmarkEntry] = []

    def try_prepare(
        material_id: str,
        display_name: str,
        source: str,
        structure: Any,
    ) -> BenchmarkEntry | None:
        canonical = canonicalize_reference_structure(structure, symprec=symprec)
        prepared, multiplier = choose_reference_structure_for_bucket(
            canonical,
            bucket=bucket,
            max_axis_multiplier=max_axis_multiplier,
        )
        if prepared is None:
            return None
        multiplier_label = "x".join(str(value) for value in multiplier)
        prepared_source = source if multiplier == (1, 1, 1) else f"{source}_supercell_{multiplier_label}"
        formula = prepared.composition.reduced_formula
        return make_entry(
            parent_id=material_id,
            material_id=material_id,
            formula=formula,
            display_name=display_name,
            family_name=formula,
            variant="reference",
            variant_family="reference",
            bucket=bucket,
            structure=prepared,
            source=prepared_source,
            symprec=symprec,
        )

    with MPRester(api_key) as rester:
        if explicit_material_ids:
            for material_id in explicit_material_ids:
                structure = rester.get_structure_by_material_id(material_id)
                entry = try_prepare(
                    material_id=material_id,
                    display_name=material_id,
                    source="materials_project",
                    structure=structure,
                )
                if entry is not None:
                    references.append(entry)
            if not references:
                raise RuntimeError(
                    "None of the explicit Materials Project IDs could be prepared "
                    f"for the '{bucket}' bucket."
                )
            return references[:max_total_structures]

        for formula in formulas:
            accepted_for_formula = 0
            material_ids = sorted(dict.fromkeys(rester.get_materials_ids(formula)))
            for material_id in material_ids:
                if len(references) >= max_total_structures:
                    break
                if accepted_for_formula >= max_per_formula:
                    break
                structure = rester.get_structure_by_material_id(material_id)
                entry = try_prepare(
                    material_id=material_id,
                    display_name=material_id,
                    source="materials_project",
                    structure=structure,
                )
                if entry is None:
                    continue
                references.append(entry)
                accepted_for_formula += 1
            if len(references) >= max_total_structures:
                break

    if not references:
        raise RuntimeError(
            "No Materials Project structures were selected. "
            "Try a different formula set, bucket, or larger supercell search."
        )
    return references


def make_conventional_structure(structure: Any, symprec: float) -> Any | None:
    SpacegroupAnalyzer = _spacegroup_analyzer_cls()
    try:
        return SpacegroupAnalyzer(
            structure,
            symprec=symprec,
        ).get_conventional_standard_structure()
    except Exception:
        return None


def make_niggli_structure(structure: Any) -> Any | None:
    try:
        return structure.get_reduced_structure(reduction_algo="niggli")
    except Exception:
        return None


def make_origin_shifted_structure(structure: Any, shift: np.ndarray) -> Any:
    shifted = structure.copy()
    shifted.translate_sites(
        indices=range(len(shifted)),
        vector=shift.tolist(),
        frac_coords=True,
        to_unit_cell=True,
    )
    return shifted


def make_permuted_structure(structure: Any, rng: np.random.Generator) -> Any:
    Structure = _structure_cls()
    sites = list(structure.sites)
    order = rng.permutation(len(sites))
    return Structure.from_sites([sites[i] for i in order])


def make_coordinate_perturbed_structure(
    structure: Any,
    sigma_angstrom: float,
    rng: np.random.Generator,
) -> Any:
    Structure = _structure_cls()
    displaced = structure.cart_coords + rng.normal(
        loc=0.0,
        scale=sigma_angstrom,
        size=(len(structure), 3),
    )
    return Structure(
        lattice=structure.lattice,
        species=structure.species,
        coords=displaced,
        coords_are_cartesian=True,
        to_unit_cell=True,
        site_properties=structure.site_properties,
        labels=getattr(structure, "labels", None),
    )


def make_lattice_perturbed_structure(
    structure: Any,
    epsilon: float,
    rng: np.random.Generator,
) -> Any:
    Structure = _structure_cls()
    scales = 1.0 + rng.normal(loc=0.0, scale=epsilon, size=3)
    perturbed_lattice = np.asarray(structure.lattice.matrix, dtype=float) @ np.diag(scales)
    return Structure(
        lattice=perturbed_lattice,
        species=structure.species,
        coords=structure.frac_coords,
        coords_are_cartesian=False,
        to_unit_cell=True,
        site_properties=structure.site_properties,
        labels=getattr(structure, "labels", None),
    )


def build_queries_from_references(
    references: Sequence[BenchmarkEntry],
    *,
    symprec: float,
    rng: np.random.Generator,
    coord_noise_levels: Sequence[float],
    lattice_noise_levels: Sequence[float],
) -> List[BenchmarkEntry]:
    if len(coord_noise_levels) != len(lattice_noise_levels):
        raise ValueError(
            "coord_noise_levels and lattice_noise_levels must have the same length "
            "so the perturbation ladder stays aligned."
        )

    queries: List[BenchmarkEntry] = []

    for reference in references:
        base = reference.structure
        equivalent_variants: List[Tuple[str, str, Any | None]] = [
            (
                "conventional_standard",
                "equivalent_transform",
                make_conventional_structure(base, symprec),
            ),
            ("niggli_reduced", "equivalent_transform", make_niggli_structure(base)),
            (
                "origin_shifted",
                "equivalent_transform",
                make_origin_shifted_structure(base, DEFAULT_ORIGIN_SHIFT),
            ),
            ("permuted_sites", "equivalent_transform", make_permuted_structure(base, rng)),
        ]

        for variant, family, structure in equivalent_variants:
            if structure is None:
                continue
            queries.append(
                make_entry(
                    parent_id=reference.parent_id,
                    material_id=reference.material_id,
                    formula=reference.formula,
                    display_name=reference.display_name,
                    family_name=reference.family_name,
                    variant=variant,
                    variant_family=family,
                    bucket=reference.bucket,
                    structure=structure,
                    source=reference.source,
                    symprec=symprec,
                )
            )

        for sigma_angstrom, epsilon in zip(coord_noise_levels, lattice_noise_levels):
            lattice_perturbed = make_lattice_perturbed_structure(base, epsilon, rng)
            combined = make_coordinate_perturbed_structure(lattice_perturbed, sigma_angstrom, rng)
            queries.append(
                make_entry(
                    parent_id=reference.parent_id,
                    material_id=reference.material_id,
                    formula=reference.formula,
                    display_name=reference.display_name,
                    family_name=reference.family_name,
                    variant=f"combined_noise_{sigma_angstrom:.3f}A_{epsilon:.3f}",
                    variant_family="combined_perturbation",
                    bucket=reference.bucket,
                    structure=combined,
                    source=reference.source,
                    symprec=symprec,
                )
            )

    if not queries:
        raise RuntimeError("No query structures were generated from the reference structures.")
    return queries


def save_entries_as_cifs(
    entries: Sequence[BenchmarkEntry],
    *,
    dataset_dir: Path,
    split: str,
) -> None:
    split_dir = dataset_dir / "structures" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        filename = (
            f"{sanitize_token(entry.bucket)}__{sanitize_token(entry.parent_id)}__"
            f"{sanitize_token(entry.variant)}__{sanitize_token(entry.material_id)}.cif"
        )
        path = split_dir / filename
        entry.structure.to(filename=str(path))
        entry.path = path.relative_to(dataset_dir)


def build_manifest_rows(
    references: Sequence[BenchmarkEntry],
    queries: Sequence[BenchmarkEntry],
) -> List[dict]:
    rows: List[dict] = []
    for split, entries in (("reference", references), ("query", queries)):
        for entry in entries:
            rows.append(
                {
                    "split": split,
                    "bucket": entry.bucket,
                    "entry_id": entry.entry_id,
                    "parent_id": entry.parent_id,
                    "material_id": entry.material_id,
                    "formula": entry.formula,
                    "display_name": entry.display_name,
                    "family_name": entry.family_name,
                    "variant": entry.variant,
                    "variant_family": entry.variant_family,
                    "nsites": entry.nsites,
                    "spacegroup_symbol": entry.spacegroup_symbol,
                    "spacegroup_number": entry.spacegroup_number,
                    "source": entry.source,
                    "path": "" if entry.path is None else entry.path.as_posix(),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def save_prepared_dataset(
    *,
    dataset_dir: Path,
    references: Sequence[BenchmarkEntry],
    queries: Sequence[BenchmarkEntry],
    threshold_split: dict,
    metadata: dict,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_entries_as_cifs(references, dataset_dir=dataset_dir, split="references")
    save_entries_as_cifs(queries, dataset_dir=dataset_dir, split="queries")
    write_csv(dataset_dir / "dataset_manifest.csv", build_manifest_rows(references, queries))
    write_json(dataset_dir / "threshold_split.json", threshold_split)
    write_json(dataset_dir / "dataset_summary.json", metadata)


def load_prepared_dataset(dataset_dir: Path) -> DatasetBundle:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "dataset_manifest.csv"
    split_path = dataset_dir / "threshold_split.json"
    summary_path = dataset_dir / "dataset_summary.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing threshold split: {split_path}")

    references: List[BenchmarkEntry] = []
    queries: List[BenchmarkEntry] = []
    seen_entry_ids: set[str] = set()

    for row in read_csv(manifest_path):
        structure_path = dataset_dir / row["path"]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"Issues encountered while parsing CIF: .* fractional coordinates "
                    r"rounded to ideal values to avoid issues with finite precision\."
                ),
                category=UserWarning,
            )
            structure = _structure_cls().from_file(str(structure_path))
        entry = BenchmarkEntry(
            entry_id=row["entry_id"],
            parent_id=row["parent_id"],
            material_id=row["material_id"],
            formula=row["formula"],
            display_name=row["display_name"],
            family_name=row["family_name"],
            variant=row["variant"],
            variant_family=row["variant_family"],
            bucket=row["bucket"],
            structure=structure,
            source=row["source"],
            nsites=int(row["nsites"]),
            spacegroup_symbol=row["spacegroup_symbol"],
            spacegroup_number=(
                None
                if row["spacegroup_number"] in {"", "None"}
                else int(row["spacegroup_number"])
            ),
            path=Path(row["path"]),
        )
        if entry.entry_id in seen_entry_ids:
            raise RuntimeError(f"Duplicate entry_id in manifest: {entry.entry_id}")
        seen_entry_ids.add(entry.entry_id)
        if row["split"] == "reference":
            references.append(entry)
        elif row["split"] == "query":
            queries.append(entry)
        else:
            raise RuntimeError(f"Unsupported split in manifest: {row['split']}")

    with split_path.open() as handle:
        threshold_split = json.load(handle)
    metadata = None
    if summary_path.exists():
        with summary_path.open() as handle:
            metadata = json.load(handle)

    return DatasetBundle(
        dataset_dir=dataset_dir,
        references=references,
        queries=queries,
        threshold_split=threshold_split,
        metadata=metadata,
    )


def get_matcher_settings() -> List[MatcherSetting]:
    return [
        MatcherSetting("strict", ltol=0.1, stol=0.1, angle_tol=2.0),
        MatcherSetting("medium", ltol=0.2, stol=0.3, angle_tol=5.0),
        MatcherSetting("loose", ltol=0.3, stol=0.5, angle_tol=10.0),
    ]


def build_threshold_split_policy(
    references: Sequence[BenchmarkEntry],
    *,
    seed: int,
) -> dict:
    bucket_to_formula_to_parent_ids: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for reference in references:
        bucket_to_formula_to_parent_ids[reference.bucket][reference.formula].append(reference.parent_id)

    rng = np.random.default_rng(seed)
    policy_by_bucket: Dict[str, dict] = {}

    for bucket, formula_to_parent_ids in sorted(bucket_to_formula_to_parent_ids.items()):
        calibration_by_formula: Dict[str, List[str]] = {}
        evaluation_by_formula: Dict[str, List[str]] = {}
        fallback_reasons: Dict[str, str | None] = {}
        mode_by_formula: Dict[str, str] = {}

        for formula, parent_ids in sorted(formula_to_parent_ids.items()):
            unique_parent_ids = sorted(set(parent_ids))
            if len(unique_parent_ids) < 2:
                raise RuntimeError(
                    "Threshold-based held-out evaluation requires at least two reference "
                    f"parents for each formula. Bucket '{bucket}' formula '{formula}' only "
                    f"has {len(unique_parent_ids)}."
                )

            shuffled = list(np.asarray(unique_parent_ids)[rng.permutation(len(unique_parent_ids))])
            n_eval = max(1, len(unique_parent_ids) // 2)
            if len(unique_parent_ids) - n_eval < 1:
                n_eval = len(unique_parent_ids) - 1

            evaluation_ids = sorted(shuffled[:n_eval])
            calibration_ids = sorted(shuffled[n_eval:])
            calibration_by_formula[formula] = calibration_ids
            evaluation_by_formula[formula] = evaluation_ids

            strict_quarantine_feasible = len(calibration_ids) > 1
            mode_by_formula[formula] = (
                "held_out_query_parents_with_reference_quarantine"
                if strict_quarantine_feasible
                else "held_out_query_parents"
            )
            fallback_reasons[formula] = (
                None
                if strict_quarantine_feasible
                else (
                    "Strict reference quarantine would leave the calibration split with no "
                    "cross-parent same-formula negatives. Falling back to a held-out query-parent "
                    "split so thresholds remain fit-able on two-reference-per-formula datasets."
                )
            )

        policy_by_bucket[bucket] = {
            "calibration_parent_ids_by_formula": calibration_by_formula,
            "evaluation_parent_ids_by_formula": evaluation_by_formula,
            "mode_by_formula": mode_by_formula,
            "fallback_reason_by_formula": fallback_reasons,
        }

    return {"seed": seed, "by_bucket": policy_by_bucket}


def assign_threshold_split(
    *,
    bucket: str,
    query_parent_id: str,
    reference_parent_id: str,
    formula: str,
    split_policy: dict,
) -> str | None:
    bucket_policy = split_policy["by_bucket"][bucket]
    calibration_ids = set(bucket_policy["calibration_parent_ids_by_formula"][formula])
    evaluation_ids = set(bucket_policy["evaluation_parent_ids_by_formula"][formula])
    mode = bucket_policy["mode_by_formula"][formula]

    if query_parent_id in evaluation_ids:
        return "evaluation"
    if query_parent_id in calibration_ids:
        if mode == "held_out_query_parents_with_reference_quarantine":
            return "calibration" if reference_parent_id in calibration_ids else None
        return "calibration"
    return None


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_binary_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
) -> dict:
    label_array = np.asarray(labels, dtype=int)
    prediction_array = np.asarray(predictions, dtype=int)

    tp = int(np.sum((label_array == 1) & (prediction_array == 1)))
    fp = int(np.sum((label_array == 0) & (prediction_array == 1)))
    tn = int(np.sum((label_array == 0) & (prediction_array == 0)))
    fn = int(np.sum((label_array == 1) & (prediction_array == 0)))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_count": int(np.sum(label_array == 1)),
        "negative_count": int(np.sum(label_array == 0)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": safe_divide(tp + tn, len(label_array)),
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def compute_distance_roc_auc(
    labels: Sequence[int],
    distances: Sequence[float],
) -> float:
    label_array = np.asarray(labels, dtype=int)
    distance_array = np.asarray(distances, dtype=float)
    positive_mask = label_array == 1
    negative_mask = label_array == 0
    n_pos = int(np.sum(positive_mask))
    n_neg = int(np.sum(negative_mask))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    scores = -distance_array
    ranks = rankdata(scores, method="average")
    sum_positive_ranks = float(np.sum(ranks[positive_mask]))
    auc = (sum_positive_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def fit_distance_threshold(
    distances: Sequence[float],
    labels: Sequence[int],
) -> dict:
    distance_array = np.asarray(distances, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if len(distance_array) == 0:
        raise RuntimeError("Cannot fit a threshold without calibration pairs.")

    unique_distances = np.sort(np.unique(distance_array))
    epsilon = max(1e-12, np.finfo(float).eps * max(1.0, float(unique_distances[0])))
    candidate_thresholds = np.concatenate(([unique_distances[0] - epsilon], unique_distances))

    best_result: dict | None = None
    best_key: tuple[float, float, float] | None = None

    for threshold in candidate_thresholds:
        predictions = (distance_array <= threshold).astype(int)
        metrics = compute_binary_metrics(label_array, predictions)
        key = (metrics["f1"], metrics["precision"], -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_result = {
                "threshold": float(threshold),
                "metrics": metrics,
                "roc_auc": compute_distance_roc_auc(label_array, distance_array),
            }

    if best_result is None:
        raise RuntimeError("Threshold fitting did not produce any candidate.")
    return best_result


def classification_tag(label: int, prediction: int) -> str:
    if label == 1 and prediction == 1:
        return "TP"
    if label == 0 and prediction == 1:
        return "FP"
    if label == 0 and prediction == 0:
        return "TN"
    return "FN"


def summarize_split_evaluation(
    distances: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> dict:
    distance_array = np.asarray(distances, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    predictions = (distance_array <= threshold).astype(int)
    metrics = compute_binary_metrics(label_array, predictions)
    metrics["roc_auc"] = compute_distance_roc_auc(label_array, distance_array)
    return metrics


def compute_gd_histogram(
    ds: np.ndarray,
    vals: np.ndarray,
    *,
    dmax: float,
    bin_width: float,
) -> np.ndarray:
    bin_edges = np.arange(0.0, dmax + bin_width, bin_width, dtype=float)
    hist, _ = np.histogram(ds, bins=bin_edges, weights=vals)
    return hist.astype(float)


def compute_pnl_descriptors(
    entries: Sequence[BenchmarkEntry],
    *,
    dmax: float,
    nmax: int,
    lmax: int,
    rbasis: str,
    normalize_reciprocal: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    from reciprocal import RECP

    recp = RECP(
        dmax=dmax,
        nmax=nmax,
        lmax=lmax,
        rbasis=normalize_rbasis_name(rbasis),
    )
    descriptors: Dict[str, np.ndarray] = {}
    runtimes: Dict[str, float] = {}

    for entry in entries:
        atoms = structure_to_ase_atoms(entry.structure)
        start = time.perf_counter()
        coords, vals, _ = recp.build_reciprocal(atoms)
        descriptor = recp.compute_sph_torch(coords, vals, norm=normalize_reciprocal)
        if hasattr(descriptor, "detach"):
            descriptor = descriptor.detach().cpu().numpy()
        descriptors[entry.entry_id] = np.asarray(descriptor, dtype=float)
        runtimes[entry.entry_id] = time.perf_counter() - start

    return descriptors, runtimes


def compute_gd_descriptors(
    entries: Sequence[BenchmarkEntry],
    *,
    dmax: float,
    bin_width: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    from reciprocal import RECP

    recp = RECP(dmax=dmax)
    descriptors: Dict[str, np.ndarray] = {}
    runtimes: Dict[str, float] = {}

    for entry in entries:
        atoms = structure_to_ase_atoms(entry.structure)
        start = time.perf_counter()
        _, vals, ds = recp.build_reciprocal(atoms)
        descriptor = compute_gd_histogram(ds, vals, dmax=dmax, bin_width=bin_width)
        descriptors[entry.entry_id] = np.asarray(descriptor, dtype=float)
        runtimes[entry.entry_id] = time.perf_counter() - start

    return descriptors, runtimes


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm <= eps:
        return np.zeros_like(array, dtype=float)
    return array / norm


def apply_pnl_first_weight(vector: np.ndarray, pnl_first_weight: float) -> np.ndarray:
    array = np.asarray(vector, dtype=float).copy()
    if array.size > 0:
        array[0] *= pnl_first_weight
    return array


def preprocess_continuous_descriptor(
    vector: np.ndarray,
    *,
    method: str,
    match_profile: str,
    pnl_first_weight: float,
) -> np.ndarray:
    array = np.asarray(vector, dtype=float)

    if match_profile == "shape" and method == "pnl":
        array = np.log1p(np.maximum(array, 0.0))

    if method == "pnl":
        array = apply_pnl_first_weight(array, pnl_first_weight)

    if match_profile == "raw":
        return array
    if match_profile in {"normalized", "shape"}:
        return l2_normalize(array)

    raise ValueError(f"Unsupported continuous match profile: {match_profile}")


def preprocess_descriptor_map(
    descriptors: Dict[str, np.ndarray],
    *,
    method: str,
    match_profile: str,
    pnl_first_weight: float,
) -> Dict[str, np.ndarray]:
    return {
        entry_id: preprocess_continuous_descriptor(
            vector,
            method=method,
            match_profile=match_profile,
            pnl_first_weight=pnl_first_weight,
        )
        for entry_id, vector in descriptors.items()
    }


def select_reference_pool(
    query: BenchmarkEntry,
    references: Sequence[BenchmarkEntry],
) -> List[BenchmarkEntry]:
    pool = [
        ref
        for ref in references
        if ref.formula == query.formula and ref.bucket == query.bucket
    ]
    if not pool:
        raise RuntimeError(
            "No same-formula reference structures available for query "
            f"{query.entry_id} in bucket '{query.bucket}'."
        )
    return pool


def build_descriptor_pair_rows(
    *,
    method: str,
    references: Sequence[BenchmarkEntry],
    queries: Sequence[BenchmarkEntry],
    reference_descriptors: Dict[str, np.ndarray],
    query_descriptors: Dict[str, np.ndarray],
) -> List[dict]:
    pair_rows: List[dict] = []

    for query in queries:
        query_vector = query_descriptors[query.entry_id]
        candidate_refs = select_reference_pool(query, references)

        for ref in candidate_refs:
            pair_start = time.perf_counter()
            distance = float(np.linalg.norm(query_vector - reference_descriptors[ref.entry_id]))
            pair_runtime = time.perf_counter() - pair_start
            label = int(query.parent_id == ref.parent_id)
            pair_rows.append(
                {
                    "method": method,
                    "bucket": query.bucket,
                    "query_id": query.entry_id,
                    "query_parent_id": query.parent_id,
                    "query_formula": query.formula,
                    "query_display_name": query.display_name,
                    "query_family_name": query.family_name,
                    "query_variant": query.variant,
                    "query_variant_family": query.variant_family,
                    "reference_id": ref.entry_id,
                    "reference_parent_id": ref.parent_id,
                    "reference_formula": ref.formula,
                    "reference_display_name": ref.display_name,
                    "reference_family_name": ref.family_name,
                    "reference_variant": ref.variant,
                    "distance": distance,
                    "label": label,
                    "pair_runtime_seconds": pair_runtime,
                }
            )
    return pair_rows


def evaluate_descriptor_pairwise_matching(
    *,
    pair_rows: Sequence[dict],
    split_policy: dict,
    method: str,
    match_profile: str,
    pnl_first_weight: float,
) -> Tuple[List[dict], List[dict]]:
    prediction_rows: List[dict] = []
    summary_rows: List[dict] = []

    rows_by_bucket: Dict[str, List[dict]] = defaultdict(list)
    for row in pair_rows:
        rows_by_bucket[row["bucket"]].append(row)

    for bucket, bucket_rows in sorted(rows_by_bucket.items()):
        calibration_records: List[dict] = []
        evaluation_records: List[dict] = []

        for pair_row in bucket_rows:
            split = assign_threshold_split(
                bucket=bucket,
                query_parent_id=pair_row["query_parent_id"],
                reference_parent_id=pair_row["reference_parent_id"],
                formula=pair_row["reference_formula"],
                split_policy=split_policy,
            )
            if split is None:
                continue

            record = {
                **pair_row,
                "split": split,
                "match_profile": match_profile,
            }
            if split == "calibration":
                calibration_records.append(record)
            else:
                evaluation_records.append(record)

        if not calibration_records:
            raise RuntimeError(
                f"No calibration pairs were available for method '{method}' in bucket '{bucket}'."
            )
        if not evaluation_records:
            raise RuntimeError(
                f"No evaluation pairs were available for method '{method}' in bucket '{bucket}'."
            )

        fitted = fit_distance_threshold(
            [record["distance"] for record in calibration_records],
            [record["label"] for record in calibration_records],
        )
        threshold = fitted["threshold"]
        calibration_metrics = fitted["metrics"]
        evaluation_metrics = summarize_split_evaluation(
            [record["distance"] for record in evaluation_records],
            [record["label"] for record in evaluation_records],
            threshold,
        )

        for record in calibration_records + evaluation_records:
            prediction = int(record["distance"] <= threshold)
            prediction_rows.append(
                {
                    **record,
                    "pnl_first_weight": pnl_first_weight,
                    "threshold": threshold,
                    "predicted_match": prediction,
                    "classification_tag": classification_tag(record["label"], prediction),
                }
            )

        summary_rows.append(
            {
                "method": method,
                "bucket": bucket,
                "match_profile": match_profile,
                "pnl_first_weight": pnl_first_weight,
                "threshold": threshold,
                "calibration_pairs": len(calibration_records),
                "calibration_positives": calibration_metrics["positive_count"],
                "calibration_negatives": calibration_metrics["negative_count"],
                "calibration_precision": calibration_metrics["precision"],
                "calibration_recall": calibration_metrics["recall"],
                "calibration_f1": calibration_metrics["f1"],
                "calibration_balanced_accuracy": calibration_metrics["balanced_accuracy"],
                "calibration_roc_auc": fitted["roc_auc"],
                "evaluation_pairs": len(evaluation_records),
                "evaluation_positives": evaluation_metrics["positive_count"],
                "evaluation_negatives": evaluation_metrics["negative_count"],
                "evaluation_tp": evaluation_metrics["tp"],
                "evaluation_fp": evaluation_metrics["fp"],
                "evaluation_tn": evaluation_metrics["tn"],
                "evaluation_fn": evaluation_metrics["fn"],
                "evaluation_precision": evaluation_metrics["precision"],
                "evaluation_recall": evaluation_metrics["recall"],
                "evaluation_f1": evaluation_metrics["f1"],
                "evaluation_balanced_accuracy": evaluation_metrics["balanced_accuracy"],
                "evaluation_roc_auc": evaluation_metrics["roc_auc"],
                "descriptor_mean_pair_runtime_seconds": float(
                    np.mean([record["pair_runtime_seconds"] for record in evaluation_records])
                ),
            }
        )

    return prediction_rows, summary_rows


def summarize_descriptor_runtime(
    *,
    method: str,
    references: Sequence[BenchmarkEntry],
    queries: Sequence[BenchmarkEntry],
    build_times: Dict[str, float],
    pair_rows: Sequence[dict],
) -> List[dict]:
    rows: List[dict] = []
    for bucket in sorted({entry.bucket for entry in list(references) + list(queries)}):
        bucket_refs = [entry for entry in references if entry.bucket == bucket]
        bucket_queries = [entry for entry in queries if entry.bucket == bucket]
        bucket_pairs = [row for row in pair_rows if row["bucket"] == bucket]
        rows.append(
            {
                "method": method,
                "bucket": bucket,
                "reference_count": len(bucket_refs),
                "query_count": len(bucket_queries),
                "reference_total_seconds": float(
                    sum(build_times[entry.entry_id] for entry in bucket_refs)
                ),
                "query_total_seconds": float(
                    sum(build_times[entry.entry_id] for entry in bucket_queries)
                ),
                "reference_mean_seconds": float(
                    np.mean([build_times[entry.entry_id] for entry in bucket_refs])
                )
                if bucket_refs
                else 0.0,
                "query_mean_seconds": float(
                    np.mean([build_times[entry.entry_id] for entry in bucket_queries])
                )
                if bucket_queries
                else 0.0,
                "pair_mean_runtime_seconds": float(
                    np.mean([row["pair_runtime_seconds"] for row in bucket_pairs])
                )
                if bucket_pairs
                else 0.0,
            }
        )
    return rows


def run_structure_matcher(
    references: Sequence[BenchmarkEntry],
    queries: Sequence[BenchmarkEntry],
    *,
    pair_timeout_seconds: float = 30.0,
) -> Tuple[List[dict], List[dict]]:
    StructureMatcher = _structure_matcher_cls()
    settings = get_matcher_settings()
    pair_rows: List[dict] = []
    query_summary_rows: List[dict] = []

    for setting in settings:
        matcher = StructureMatcher(
            ltol=setting.ltol,
            stol=setting.stol,
            angle_tol=setting.angle_tol,
            primitive_cell=True,
            scale=True,
            attempt_supercell=True,
            allow_subset=False,
        )

        for query in queries:
            query_start = time.perf_counter()
            matched_reference_ids: List[str] = []
            false_positives = 0
            has_true_match = False
            timeout_count = 0
            candidate_refs = select_reference_pool(query, references)

            for ref in candidate_refs:
                pair_start = time.perf_counter()
                timed_out = False
                try:
                    is_match = bool(
                        run_with_timeout(
                            pair_timeout_seconds,
                            matcher.fit,
                            query.structure,
                            ref.structure,
                        )
                    )
                except StructureMatcherPairTimeoutError:
                    timed_out = True
                    timeout_count += 1
                    is_match = False
                pair_runtime = time.perf_counter() - pair_start
                label = int(query.parent_id == ref.parent_id)

                if is_match:
                    matched_reference_ids.append(ref.entry_id)
                    if label:
                        has_true_match = True
                    else:
                        false_positives += 1

                pair_rows.append(
                    {
                        "setting": setting.name,
                        "bucket": query.bucket,
                        "query_id": query.entry_id,
                        "query_parent_id": query.parent_id,
                        "query_formula": query.formula,
                        "query_display_name": query.display_name,
                        "query_family_name": query.family_name,
                        "query_variant": query.variant,
                        "query_variant_family": query.variant_family,
                        "reference_id": ref.entry_id,
                        "reference_parent_id": ref.parent_id,
                        "reference_formula": ref.formula,
                        "reference_display_name": ref.display_name,
                        "reference_family_name": ref.family_name,
                        "reference_variant": ref.variant,
                        "label": label,
                        "is_match": int(is_match),
                        "timed_out": int(timed_out),
                        "pair_runtime_seconds": pair_runtime,
                    }
                )

            query_runtime = time.perf_counter() - query_start
            query_summary_rows.append(
                {
                    "setting": setting.name,
                    "bucket": query.bucket,
                    "query_id": query.entry_id,
                    "query_parent_id": query.parent_id,
                    "query_display_name": query.display_name,
                    "query_family_name": query.family_name,
                    "query_variant": query.variant,
                    "query_variant_family": query.variant_family,
                    "true_parent_matched": int(has_true_match),
                    "candidate_count": len(candidate_refs),
                    "num_matches": len(matched_reference_ids),
                    "timeout_count": timeout_count,
                    "false_positives": false_positives,
                    "false_negatives": 0 if has_true_match else 1,
                    "matched_reference_ids": "|".join(matched_reference_ids),
                    "query_runtime_seconds": query_runtime,
                }
            )

    return pair_rows, query_summary_rows


def summarize_structure_matcher_results(
    pair_rows: Sequence[dict],
    query_rows: Sequence[dict],
) -> List[dict]:
    summary_rows: List[dict] = []
    pair_groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    query_groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    for row in pair_rows:
        pair_groups[(row["setting"], row["bucket"])].append(row)
    for row in query_rows:
        query_groups[(row["setting"], row["bucket"])].append(row)

    for key in sorted(pair_groups):
        setting, bucket = key
        grouped_pairs = pair_groups[key]
        grouped_queries = query_groups[key]
        pair_metrics = compute_binary_metrics(
            [row["label"] for row in grouped_pairs],
            [row["is_match"] for row in grouped_pairs],
        )
        summary_rows.append(
            {
                "setting": setting,
                "bucket": bucket,
                "pairwise_precision": pair_metrics["precision"],
                "pairwise_recall": pair_metrics["recall"],
                "pairwise_f1": pair_metrics["f1"],
                "pairwise_balanced_accuracy": pair_metrics["balanced_accuracy"],
                "pairwise_tp": pair_metrics["tp"],
                "pairwise_fp": pair_metrics["fp"],
                "pairwise_tn": pair_metrics["tn"],
                "pairwise_fn": pair_metrics["fn"],
                "pairwise_mean_runtime_seconds": float(
                    np.mean([row["pair_runtime_seconds"] for row in grouped_pairs])
                ),
                "pairwise_timeout_count": int(sum(row["timed_out"] for row in grouped_pairs)),
                "query_count": len(grouped_queries),
                "true_parent_match_rate": float(
                    np.mean([row["true_parent_matched"] for row in grouped_queries])
                )
                if grouped_queries
                else 0.0,
                "query_timeout_count": int(sum(row["timeout_count"] for row in grouped_queries)),
                "total_false_positives": int(
                    sum(row["false_positives"] for row in grouped_queries)
                ),
                "total_false_negatives": int(
                    sum(row["false_negatives"] for row in grouped_queries)
                ),
                "mean_query_runtime_seconds": float(
                    np.mean([row["query_runtime_seconds"] for row in grouped_queries])
                )
                if grouped_queries
                else 0.0,
            }
        )

    return summary_rows
