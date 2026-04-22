from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

# Allow importing reciprocal.py from repository root.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from reciprocal import RECP  # noqa: E402


DEFAULT_FORMULAS = ["C", "SiO2", "TiO2"]
DEFAULT_COORD_NOISE = [0.01, 0.02]
DEFAULT_LATTICE_NOISE = [0.2]
DEFAULT_G_BIN_WIDTH = 0.02
DEFAULT_ORIGIN_SHIFT = np.array([0.173, 0.257, 0.389], dtype=float)
PRIMARY_POOL_SCOPE = "same_formula"
CELL_SIZE_BUCKETS: Dict[str, dict] = {
    "small": {"min_sites": 1, "max_sites": 19},
    "medium": {"min_sites": 20, "max_sites": 50},
    "large": {"min_sites": 51, "max_sites": None},
}


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@dataclass(frozen=True)
class ReferenceSpec:
    material_id: str
    parent_id: str
    display_name: str
    family_name: str


CURATED_PRESETS: Dict[str, dict] = {
    "polymorph_small": {
        "references": [
            ReferenceSpec("mp-66", "diamond", "diamond", "C"),
            ReferenceSpec("mp-48", "graphite", "graphite", "C"),
            ReferenceSpec("mp-7000", "alpha_quartz", "alpha_quartz", "SiO2"),
            ReferenceSpec("mp-6945", "alpha_cristobalite", "alpha_cristobalite", "SiO2"),
            ReferenceSpec("mp-12787", "coesite", "coesite", "SiO2"),
            ReferenceSpec("mp-390", "anatase", "anatase", "TiO2"),
            ReferenceSpec("mp-2657", "rutile", "rutile", "TiO2"),
            ReferenceSpec("mp-1840", "brookite", "brookite", "TiO2"),
        ],
        "coord_noise": [0.02, 0.05, 0.10],
        "lattice_noise": [0.01, 0.02, 0.05],
        "include_supercell": True,
        "max_sites": 48,
        "max_supercell_sites": 96,
    }
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
    structure: Any
    source: str
    nsites: int
    spacegroup_symbol: str
    spacegroup_number: int | None
    path: Path | None = None


@dataclass
class MatcherSetting:
    name: str
    ltol: float
    stol: float
    angle_tol: float


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


def build_combined_noise_values(
    coord_noise_levels: Sequence[float],
    lattice_noise_levels: Sequence[float],
) -> List[str]:
    return [
        f"{float(coord_noise):g}:{float(lattice_noise):g}"
        for coord_noise in coord_noise_levels
        for lattice_noise in lattice_noise_levels
    ]


def build_combined_noise_levels(
    coord_noise_levels: Sequence[float],
    lattice_noise_levels: Sequence[float],
) -> List[Tuple[float, float]]:
    return [
        (float(coord_noise), float(lattice_noise))
        for coord_noise in coord_noise_levels
        for lattice_noise in lattice_noise_levels
    ]


def spacegroup_info(structure: Any, symprec: float) -> Tuple[str, int | None]:
    SpacegroupAnalyzer = _spacegroup_analyzer_cls()
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        symbol, number = analyzer.get_space_group_symbol(), analyzer.get_space_group_number()
        return str(symbol), int(number)
    except Exception:
        return "unknown", None


def canonicalize_reference_structure(structure: Any, symprec: float) -> Any:
    SpacegroupAnalyzer = _spacegroup_analyzer_cls()
    try:
        return SpacegroupAnalyzer(structure, symprec=symprec).get_primitive_standard_structure()
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
    structure: Any,
    source: str,
    symprec: float,
    path: Path | None = None,
) -> BenchmarkEntry:
    spacegroup_symbol, spacegroup_number = spacegroup_info(structure, symprec=symprec)
    entry_id = f"{parent_id}::{variant}"
    return BenchmarkEntry(
        entry_id=entry_id,
        parent_id=parent_id,
        material_id=material_id,
        formula=formula,
        display_name=display_name,
        family_name=family_name,
        variant=variant,
        variant_family=variant_family,
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


def save_entries_as_cifs(entries: Sequence[BenchmarkEntry], output_dir: Path, split: str) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        filename = (
            f"{sanitize_token(entry.parent_id)}__{sanitize_token(entry.variant)}"
            f"__{sanitize_token(entry.material_id)}.cif"
        )
        path = split_dir / filename
        entry.structure.to(filename=str(path))
        entry.path = path


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
                    "path": "" if entry.path is None else str(entry.path),
                }
            )
    return rows


def infer_local_parent_id(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("_reference", "_optimized", "_perturbed"):
        stem = stem.replace(token, "")
    stem = re.sub(r"_pert\d+(\.\d+)?", "", stem)
    stem = re.sub(r"\s+", "_", stem)
    return stem


def discover_local_entries(dataset_root: Path, symprec: float) -> Tuple[List[BenchmarkEntry], List[BenchmarkEntry]]:
    Structure = _structure_cls()
    files = sorted(dataset_root.rglob("*.cif"))
    references: List[BenchmarkEntry] = []
    queries: List[BenchmarkEntry] = []

    for file in files:
        structure = Structure.from_file(str(file))
        parent_id = infer_local_parent_id(file)
        formula = structure.composition.reduced_formula
        variant = "reference" if "reference" in file.stem.lower() else sanitize_token(file.stem)
        variant_family = "reference" if variant == "reference" else "local_query"
        entry = make_entry(
            parent_id=parent_id,
            material_id=parent_id,
            formula=formula,
            display_name=parent_id,
            family_name=formula,
            variant=variant,
            variant_family=variant_family,
            structure=structure,
            source="local",
            symprec=symprec,
            path=file,
        )
        if "reference" in file.stem.lower():
            references.append(entry)
        else:
            queries.append(entry)

    if not references:
        raise RuntimeError(f"No reference CIF files found under: {dataset_root}")
    if not queries:
        raise RuntimeError(f"No query CIF files found under: {dataset_root}")
    return references, queries


def get_mp_api_key(cli_value: str | None) -> str:
    api_key = cli_value or os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Materials Project access requires an API key. Pass --api-key or set MP_API_KEY."
        )
    return api_key


def fetch_mp_references(
    *,
    api_key: str,
    formulas: Sequence[str],
    explicit_material_ids: Sequence[str],
    reference_specs: Sequence[ReferenceSpec],
    max_per_formula: int,
    max_total_structures: int,
    min_sites: int | None,
    max_sites: int | None,
    symprec: float,
) -> List[BenchmarkEntry]:
    MPRester = _mp_rester_cls()
    references: List[BenchmarkEntry] = []

    effective_min_sites = 0 if min_sites is None else int(min_sites)

    def within_site_window(structure: Any) -> bool:
        nsites = len(structure)
        if nsites < effective_min_sites:
            return False
        if max_sites is not None and nsites > max_sites:
            return False
        return True

    with MPRester(api_key) as rester:
        if reference_specs:
            for spec in reference_specs:
                structure = canonicalize_reference_structure(
                    rester.get_structure_by_material_id(spec.material_id),
                    symprec=symprec,
                )
                if not within_site_window(structure):
                    raise RuntimeError(
                        "Curated reference "
                        f"{spec.material_id} does not satisfy the site-count filter "
                        f"({effective_min_sites} <= nsites <= {max_sites})."
                    )
                references.append(
                    make_entry(
                        parent_id=spec.parent_id,
                        material_id=spec.material_id,
                        formula=structure.composition.reduced_formula,
                        display_name=spec.display_name,
                        family_name=spec.family_name,
                        variant="reference",
                        variant_family="reference",
                        structure=structure,
                        source="materials_project_curated",
                        symprec=symprec,
                    )
                )
            return references

        if explicit_material_ids:
            for material_id in explicit_material_ids[:max_total_structures]:
                structure = canonicalize_reference_structure(
                    rester.get_structure_by_material_id(material_id),
                    symprec=symprec,
                )
                if not within_site_window(structure):
                    continue
                formula = structure.composition.reduced_formula
                references.append(
                    make_entry(
                        parent_id=material_id,
                        material_id=material_id,
                        formula=formula,
                        display_name=material_id,
                        family_name=formula,
                        variant="reference",
                        variant_family="reference",
                        structure=structure,
                        source="materials_project",
                        symprec=symprec,
                    )
                )
            if not references:
                raise RuntimeError("No usable Materials Project structures were fetched.")
            return references

        for formula in formulas:
            accepted_for_formula = 0
            for material_id in sorted(dict.fromkeys(rester.get_materials_ids(formula))):
                if len(references) >= max_total_structures:
                    break
                if accepted_for_formula >= max_per_formula:
                    break
                structure = canonicalize_reference_structure(
                    rester.get_structure_by_material_id(material_id),
                    symprec=symprec,
                )
                if not within_site_window(structure):
                    continue
                references.append(
                    make_entry(
                        parent_id=material_id,
                        material_id=material_id,
                        formula=structure.composition.reduced_formula,
                        display_name=material_id,
                        family_name=structure.composition.reduced_formula,
                        variant="reference",
                        variant_family="reference",
                        structure=structure,
                        source="materials_project",
                        symprec=symprec,
                    )
                )
                accepted_for_formula += 1
            if len(references) >= max_total_structures:
                break

    if not references:
        raise RuntimeError(
            "No Materials Project reference structures were selected. "
            "Try relaxing the site-count filter or passing explicit --material-id values."
        )
    return references


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


def make_conventional_structure(structure: Any, symprec: float) -> Any | None:
    SpacegroupAnalyzer = _spacegroup_analyzer_cls()
    try:
        return SpacegroupAnalyzer(structure, symprec=symprec).get_conventional_standard_structure()
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


def make_supercell_structure(structure: Any) -> Any:
    supercell = structure.copy()
    supercell.make_supercell([2, 1, 1])
    return supercell


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


def build_mp_queries(
    references: Sequence[BenchmarkEntry],
    *,
    symprec: float,
    rng: np.random.Generator,
    include_supercell: bool,
    max_supercell_sites: int,
    coord_noise_levels: Sequence[float],
    lattice_noise_levels: Sequence[float],
    combined_noise_levels: Sequence[Tuple[float, float]],
) -> List[BenchmarkEntry]:
    queries: List[BenchmarkEntry] = []

    for reference in references:
        base = reference.structure
        equivalent_variants: List[Tuple[str, str, Any | None]] = [
            ("conventional_standard", "equivalent_transform", make_conventional_structure(base, symprec)),
            ("niggli_reduced", "equivalent_transform", make_niggli_structure(base)),
            ("origin_shifted", "equivalent_transform", make_origin_shifted_structure(base, DEFAULT_ORIGIN_SHIFT)),
            ("permuted_sites", "equivalent_transform", make_permuted_structure(base, rng)),
        ]

        if include_supercell and len(base) * 2 <= max_supercell_sites:
            equivalent_variants.append(
                ("supercell_2x1x1", "equivalent_transform", make_supercell_structure(base))
            )

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
                    structure=structure,
                    source=reference.source,
                    symprec=symprec,
                )
            )

        for sigma_angstrom in coord_noise_levels:
            queries.append(
                make_entry(
                    parent_id=reference.parent_id,
                    material_id=reference.material_id,
                    formula=reference.formula,
                    display_name=reference.display_name,
                    family_name=reference.family_name,
                    variant=f"coord_noise_{sigma_angstrom:.3f}A",
                    variant_family="coordinate_perturbation",
                    structure=make_coordinate_perturbed_structure(base, sigma_angstrom, rng),
                    source=reference.source,
                    symprec=symprec,
                )
            )

        for epsilon in lattice_noise_levels:
            queries.append(
                make_entry(
                    parent_id=reference.parent_id,
                    material_id=reference.material_id,
                    formula=reference.formula,
                    display_name=reference.display_name,
                    family_name=reference.family_name,
                    variant=f"lattice_noise_{epsilon:.3f}",
                    variant_family="lattice_perturbation",
                    structure=make_lattice_perturbed_structure(base, epsilon, rng),
                    source=reference.source,
                    symprec=symprec,
                )
            )

        for sigma_angstrom, epsilon in combined_noise_levels:
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
                    structure=combined,
                    source=reference.source,
                    symprec=symprec,
                )
            )

    if not queries:
        raise RuntimeError("No query structures were generated from the reference structures.")
    return queries


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


def compute_descriptors(
    entries: Sequence[BenchmarkEntry],
    recp: RECP,
    *,
    normalize_reciprocal: bool,
    g_bin_width: float,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
    descriptors: Dict[str, Dict[str, np.ndarray]] = {"reciprocal_power_spectrum": {}, "raw_gd": {}}
    runtimes: Dict[str, Dict[str, float]] = {"reciprocal_power_spectrum": {}, "raw_gd": {}}

    for entry in entries:
        atoms = structure_to_ase_atoms(entry.structure)
        coords, vals, ds = recp.build_reciprocal(atoms)

        start = time.perf_counter()
        reciprocal = recp.compute_sph_torch(coords, vals, norm=normalize_reciprocal)
        if hasattr(reciprocal, "detach"):
            reciprocal = reciprocal.detach().cpu().numpy()
        descriptors["reciprocal_power_spectrum"][entry.entry_id] = np.asarray(reciprocal, dtype=float)
        runtimes["reciprocal_power_spectrum"][entry.entry_id] = time.perf_counter() - start

        start = time.perf_counter()
        raw_gd = compute_gd_histogram(ds, vals, dmax=recp.dmax, bin_width=g_bin_width)
        descriptors["raw_gd"][entry.entry_id] = raw_gd
        runtimes["raw_gd"][entry.entry_id] = time.perf_counter() - start

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

    if match_profile == "shape":
        if method == "reciprocal_power_spectrum":
            # Compress the dynamic range so a few dominant low-order terms do not
            # overwhelm the shape comparison.
            array = np.log1p(np.maximum(array, 0.0))

    if method == "reciprocal_power_spectrum":
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
    formula_to_parent_ids: Dict[str, List[str]] = defaultdict(list)
    for reference in references:
        formula_to_parent_ids[reference.formula].append(reference.parent_id)

    rng = np.random.default_rng(seed)
    calibration_by_formula: Dict[str, List[str]] = {}
    evaluation_by_formula: Dict[str, List[str]] = {}

    for formula, parent_ids in sorted(formula_to_parent_ids.items()):
        unique_parent_ids = sorted(set(parent_ids))
        if len(unique_parent_ids) < 2:
            raise RuntimeError(
                "Threshold-based held-out evaluation requires at least two reference "
                f"parents for each formula. Formula '{formula}' only has "
                f"{len(unique_parent_ids)}."
            )

        shuffled = list(np.asarray(unique_parent_ids)[rng.permutation(len(unique_parent_ids))])
        n_eval = max(1, len(unique_parent_ids) // 2)
        if len(unique_parent_ids) - n_eval < 1:
            n_eval = len(unique_parent_ids) - 1

        evaluation_ids = sorted(shuffled[:n_eval])
        calibration_ids = sorted(shuffled[n_eval:])
        calibration_by_formula[formula] = calibration_ids
        evaluation_by_formula[formula] = evaluation_ids

    strict_quarantine_feasible = any(
        len(calibration_ids) > 1 for calibration_ids in calibration_by_formula.values()
    )

    return {
        "mode": (
            "held_out_query_parents_with_reference_quarantine"
            if strict_quarantine_feasible
            else "held_out_query_parents"
        ),
        "seed": seed,
        "calibration_parent_ids_by_formula": calibration_by_formula,
        "evaluation_parent_ids_by_formula": evaluation_by_formula,
        "fallback_reason": (
            None
            if strict_quarantine_feasible
            else (
                "Strict reference quarantine would leave the calibration split with no "
                "cross-parent same-formula negatives. Falling back to a held-out query-parent "
                "split so thresholds remain fit-able on two-reference-per-formula datasets."
            )
        ),
    }


def assign_threshold_split(
    *,
    query_parent_id: str,
    reference_parent_id: str,
    formula: str,
    split_policy: dict,
) -> str | None:
    calibration_ids = set(split_policy["calibration_parent_ids_by_formula"][formula])
    evaluation_ids = set(split_policy["evaluation_parent_ids_by_formula"][formula])

    if query_parent_id in evaluation_ids:
        return "evaluation"
    if query_parent_id in calibration_ids:
        if split_policy["mode"] == "held_out_query_parents_with_reference_quarantine":
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
    f1 = safe_divide(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "accuracy": safe_divide(tp + tn, len(label_array)),
        "positive_count": int(np.sum(label_array == 1)),
        "negative_count": int(np.sum(label_array == 0)),
    }


def compute_distance_roc_auc(
    labels: Sequence[int],
    distances: Sequence[float],
) -> float | None:
    label_array = np.asarray(labels, dtype=int)
    distance_array = np.asarray(distances, dtype=float)
    n_pos = int(np.sum(label_array == 1))
    n_neg = int(np.sum(label_array == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    # Lower distance means more likely positive, so rank by negative distance.
    ranks = rankdata(-distance_array, method="average")
    positive_ranks = float(np.sum(ranks[label_array == 1]))
    auc = (positive_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
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


def evaluate_threshold_pairwise_matching(
    *,
    descriptor_pair_rows_by_method: Dict[str, List[dict]],
    sm_pair_rows: Sequence[dict],
    references: Sequence[BenchmarkEntry],
    split_policy: dict,
    match_profile: str,
    pnl_first_weight: float,
) -> tuple[List[dict], List[dict]]:
    reference_formula_by_parent = {reference.parent_id: reference.formula for reference in references}
    sm_pair_lookup: Dict[tuple[str, str, str], dict] = {
        (row["setting"], row["query_id"], row["reference_id"]): row for row in sm_pair_rows
    }

    prediction_rows: List[dict] = []
    summary_rows: List[dict] = []

    for method_name, descriptor_pair_rows in descriptor_pair_rows_by_method.items():
        for setting in get_matcher_settings():
            calibration_records: List[dict] = []
            evaluation_records: List[dict] = []

            for pair_row in descriptor_pair_rows:
                sm_row = sm_pair_lookup[(setting.name, pair_row["query_id"], pair_row["reference_id"])]
                formula = reference_formula_by_parent[pair_row["reference_parent_id"]]
                split = assign_threshold_split(
                    query_parent_id=pair_row["query_parent_id"],
                    reference_parent_id=pair_row["reference_parent_id"],
                    formula=formula,
                    split_policy=split_policy,
                )
                if split is None:
                    continue

                record = {
                    "split": split,
                    "method": method_name,
                    "continuous_match_profile": match_profile,
                    "structurematcher_setting": setting.name,
                    "query_id": pair_row["query_id"],
                    "query_parent_id": pair_row["query_parent_id"],
                    "query_display_name": pair_row["query_display_name"],
                    "query_family_name": pair_row["query_family_name"],
                    "query_variant": pair_row["query_variant"],
                    "query_variant_family": pair_row["query_variant_family"],
                    "reference_id": pair_row["reference_id"],
                    "reference_parent_id": pair_row["reference_parent_id"],
                    "reference_display_name": pair_row["reference_display_name"],
                    "reference_family_name": pair_row["reference_family_name"],
                    "reference_variant": pair_row["reference_variant"],
                    "formula": formula,
                    "distance": float(pair_row["distance"]),
                    "structurematcher_label": int(sm_row["is_match"]),
                    "descriptor_pair_runtime_seconds": float(pair_row["pair_runtime_seconds"]),
                    "structurematcher_pair_runtime_seconds": float(sm_row["pair_runtime_seconds"]),
                }

                if split == "calibration":
                    calibration_records.append(record)
                else:
                    evaluation_records.append(record)

            if not calibration_records:
                raise RuntimeError(
                    f"No calibration pairs were available for method '{method_name}' and setting '{setting.name}'."
                )
            if not evaluation_records:
                raise RuntimeError(
                    f"No evaluation pairs were available for method '{method_name}' and setting '{setting.name}'."
                )

            fitted = fit_distance_threshold(
                [record["distance"] for record in calibration_records],
                [record["structurematcher_label"] for record in calibration_records],
            )
            threshold = fitted["threshold"]
            calibration_metrics = fitted["metrics"]
            evaluation_metrics = summarize_split_evaluation(
                [record["distance"] for record in evaluation_records],
                [record["structurematcher_label"] for record in evaluation_records],
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
                        "classification_tag": classification_tag(
                            record["structurematcher_label"],
                            prediction,
                        ),
                    }
                )

            summary_rows.append(
                {
                    "method": method_name,
                    "continuous_match_profile": match_profile,
                    "pnl_first_weight": pnl_first_weight,
                    "structurematcher_setting": setting.name,
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
                        np.mean([record["descriptor_pair_runtime_seconds"] for record in evaluation_records])
                    ),
                    "structurematcher_mean_pair_runtime_seconds": float(
                        np.mean([record["structurematcher_pair_runtime_seconds"] for record in evaluation_records])
                    ),
                }
            )

    return prediction_rows, summary_rows


def select_reference_pool(
    query: BenchmarkEntry,
    references: Sequence[BenchmarkEntry],
) -> List[BenchmarkEntry]:
    pool = [ref for ref in references if ref.formula == query.formula]
    if not pool:
        raise RuntimeError(
            f"No same-formula reference structures available for query {query.entry_id}."
        )
    return pool


def build_descriptor_pair_rows(
    method: str,
    references: Sequence[BenchmarkEntry],
    queries: Sequence[BenchmarkEntry],
    ref_desc: Dict[str, np.ndarray],
    query_desc: Dict[str, np.ndarray],
) -> List[dict]:
    pair_rows: List[dict] = []

    for query in queries:
        qv = query_desc[query.entry_id]
        candidate_refs = select_reference_pool(query, references)

        for ref in candidate_refs:
            pair_start = time.perf_counter()
            distance = float(np.linalg.norm(qv - ref_desc[ref.entry_id]))
            pair_runtime = time.perf_counter() - pair_start
            is_true_parent = int(query.parent_id == ref.parent_id)
            pair_rows.append(
                {
                    "method": method,
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
                    "is_true_parent": is_true_parent,
                    "pair_runtime_seconds": pair_runtime,
                }
            )
    return pair_rows


def run_structure_matcher(
    references: Sequence[BenchmarkEntry],
    queries: Sequence[BenchmarkEntry],
) -> Tuple[List[dict], List[dict], List[dict]]:
    StructureMatcher = _structure_matcher_cls()
    settings = get_matcher_settings()

    pair_rows: List[dict] = []
    query_summary_rows: List[dict] = []
    aggregate_rows: List[dict] = []

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

        n_true_parent_matched = 0
        total_query_runtime = 0.0
        total_false_positives = 0
        total_false_negatives = 0

        for query in queries:
            query_start = time.perf_counter()
            matched_reference_ids: List[str] = []
            false_positives = 0
            has_true_match = False
            candidate_refs = select_reference_pool(query, references)

            for ref in candidate_refs:
                pair_start = time.perf_counter()
                is_match = bool(matcher.fit(query.structure, ref.structure))
                pair_runtime = time.perf_counter() - pair_start
                is_true_parent = int(query.parent_id == ref.parent_id)

                if is_match:
                    matched_reference_ids.append(ref.entry_id)
                    if is_true_parent:
                        has_true_match = True
                    else:
                        false_positives += 1

                pair_rows.append(
                    {
                        "setting": setting.name,
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
                        "is_true_parent": is_true_parent,
                        "is_match": int(is_match),
                        "pair_runtime_seconds": pair_runtime,
                    }
                )

            query_runtime = time.perf_counter() - query_start
            false_negative = 0 if has_true_match else 1
            n_true_parent_matched += int(has_true_match)
            total_query_runtime += query_runtime
            total_false_positives += false_positives
            total_false_negatives += false_negative

            query_summary_rows.append(
                {
                    "setting": setting.name,
                    "query_id": query.entry_id,
                    "query_parent_id": query.parent_id,
                    "query_display_name": query.display_name,
                    "query_family_name": query.family_name,
                    "query_variant": query.variant,
                    "query_variant_family": query.variant_family,
                    "true_parent_matched": int(has_true_match),
                    "candidate_count": len(candidate_refs),
                    "num_matches": len(matched_reference_ids),
                    "false_positives": false_positives,
                    "false_negatives": false_negative,
                    "matched_reference_ids": "|".join(matched_reference_ids),
                    "query_runtime_seconds": query_runtime,
                }
            )

        aggregate_rows.append(
                {
                    "setting": setting.name,
                    "true_parent_match_rate": n_true_parent_matched / max(len(queries), 1),
                    "total_false_positives": total_false_positives,
                    "total_false_negatives": total_false_negatives,
                "mean_query_runtime_seconds": total_query_runtime / max(len(queries), 1),
            }
        )

    return pair_rows, query_summary_rows, aggregate_rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def apply_preset_defaults(args: argparse.Namespace) -> tuple[argparse.Namespace, List[ReferenceSpec]]:
    if not args.preset:
        return args, []

    preset = CURATED_PRESETS[args.preset]
    if args.coord_noise == DEFAULT_COORD_NOISE:
        args.coord_noise = list(preset["coord_noise"])
    if args.lattice_noise == DEFAULT_LATTICE_NOISE:
        args.lattice_noise = list(preset["lattice_noise"])
    if args.max_sites is None:
        args.max_sites = int(preset["max_sites"])
    if args.max_supercell_sites == 120:
        args.max_supercell_sites = int(preset["max_supercell_sites"])
    if args.skip_supercell and preset["include_supercell"]:
        pass
    return args, list(preset["references"])


def apply_size_bucket_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.size_bucket:
        return args
    bucket = CELL_SIZE_BUCKETS[args.size_bucket]
    args.min_sites = int(bucket["min_sites"])
    args.max_sites = None if bucket["max_sites"] is None else int(bucket["max_sites"])
    return args


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark reciprocal power spectrum P_nl, raw G(d), "
            "and pymatgen StructureMatcher on small MP or local crystal datasets."
        )
    )
    parser.add_argument(
        "--source",
        choices=["mp", "local"],
        default="mp",
        help="Use Materials Project references ('mp') or the existing local CIF layout ('local').",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(CURATED_PRESETS),
        default=None,
        help="Curated Materials Project benchmark preset with pinned polymorph IDs.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Fig-6_reconstruction"),
        help="Root directory containing local CIF files when --source=local.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/results"),
        help="Directory to write dataset manifests, CSV outputs, and summary JSON.",
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
        help="Formula list for automatic MP reference selection.",
    )
    parser.add_argument(
        "--size-bucket",
        choices=sorted(CELL_SIZE_BUCKETS),
        default=None,
        help="Reference-structure site-count bucket: small (<20), medium (20-50), or large (>50).",
    )
    parser.add_argument(
        "--material-id",
        action="append",
        default=[],
        help="Explicit MP material ID to benchmark. Repeat to pin a custom set.",
    )
    parser.add_argument("--max-per-formula", type=int, default=20)
    parser.add_argument("--max-total-structures", type=int, default=60)
    parser.add_argument("--min-sites", type=int, default=None)
    parser.add_argument("--max-sites", type=int, default=None)
    parser.add_argument(
        "--min-reference-parents-per-formula",
        type=int,
        default=2,
        help="Minimum number of reference parents a formula must retain for held-out evaluation.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--coord-noise", type=float, nargs="*", default=DEFAULT_COORD_NOISE)
    parser.add_argument("--lattice-noise", type=float, nargs="*", default=DEFAULT_LATTICE_NOISE)
    parser.add_argument("--skip-supercell", action="store_true")
    parser.add_argument("--max-supercell-sites", type=int, default=120)
    parser.add_argument("--dmax", type=float, default=10.0)
    parser.add_argument("--nmax", type=int, default=10)
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument(
        "--rbasis",
        type=str,
        default="Bessel",
        choices=["Bessel", "chebyshev", "gto"],
    )
    parser.add_argument("--normalize-reciprocal", action="store_true")
    parser.add_argument("--g-bin-width", type=float, default=DEFAULT_G_BIN_WIDTH)
    parser.add_argument(
        "--continuous-match-profile",
        type=str,
        default="normalized",
        choices=["raw", "normalized", "shape"],
        help=(
            "How to post-process continuous descriptors before L2 pairwise "
            "comparison. 'raw' reproduces the old plain-L2 benchmark. "
            "'normalized' applies per-vector L2 normalization and is the "
            "default fair comparison. 'shape' applies log1p + L2 "
            "normalization for P_nl, and L2 normalization for G(d)."
        ),
    )
    parser.add_argument(
        "--pnl-first-weight",
        type=float,
        default=0.1,
        help=(
            "Weight applied to the first reciprocal power-spectrum component "
            "before continuous matching. Use 1.0 to recover the unweighted "
            "behavior."
        ),
    )
    args = parser.parse_args()
    if args.pnl_first_weight < 0.0:
        raise ValueError("--pnl-first-weight must be non-negative.")

    args, reference_specs = apply_preset_defaults(args)
    args = apply_size_bucket_defaults(args)
    rng = np.random.default_rng(args.seed)
    combined_noise_values = build_combined_noise_values(args.coord_noise, args.lattice_noise)
    combined_noise_levels = build_combined_noise_levels(args.coord_noise, args.lattice_noise)
    progress(
        f"Starting benchmark | source={args.source} | size_bucket={args.size_bucket} | "
        f"coord_noise={args.coord_noise} | lattice_noise={args.lattice_noise}"
    )

    if args.source == "local":
        progress(f"Loading local dataset from {args.dataset_root}")
        references, queries = discover_local_entries(args.dataset_root, symprec=args.symprec)
    else:
        progress("Fetching Materials Project references")
        api_key = get_mp_api_key(args.api_key)
        references = fetch_mp_references(
            api_key=api_key,
            formulas=args.formula,
            explicit_material_ids=args.material_id,
            reference_specs=reference_specs,
            max_per_formula=args.max_per_formula,
            max_total_structures=args.max_total_structures,
            min_sites=args.min_sites,
            max_sites=args.max_sites,
            symprec=args.symprec,
        )
        references = filter_formulas_with_min_parents(
            references,
            min_unique_parents=args.min_reference_parents_per_formula,
        )
        progress(f"Selected {len(references)} reference structures; generating query variants")
        queries = build_mp_queries(
            references,
            symprec=args.symprec,
            rng=rng,
            include_supercell=not args.skip_supercell,
            max_supercell_sites=args.max_supercell_sites,
            coord_noise_levels=args.coord_noise,
            lattice_noise_levels=args.lattice_noise,
            combined_noise_levels=combined_noise_levels,
        )
        dataset_dir = args.output_dir / "dataset"
        save_entries_as_cifs(references, dataset_dir, split="references")
        save_entries_as_cifs(queries, dataset_dir, split="queries")
        progress(f"Saved generated dataset under {dataset_dir}")

    manifest_rows = build_manifest_rows(references, queries)
    write_csv(args.output_dir / "dataset_manifest.csv", manifest_rows)
    progress(f"Prepared dataset manifest | references={len(references)} | queries={len(queries)}")

    recp = RECP(dmax=args.dmax, nmax=args.nmax, lmax=args.lmax, rbasis=args.rbasis)
    all_entries = list(references) + list(queries)
    progress("Computing reciprocal and G(d) descriptors")
    descriptors, build_times = compute_descriptors(
        all_entries,
        recp,
        normalize_reciprocal=args.normalize_reciprocal,
        g_bin_width=args.g_bin_width,
    )

    descriptor_build_summary: Dict[str, dict] = {}
    descriptor_pair_rows_by_method: Dict[str, List[dict]] = {}

    for method_name, method_desc in descriptors.items():
        progress(f"Building descriptor pair rows for {method_name}")
        processed_desc = preprocess_descriptor_map(
            method_desc,
            method=method_name,
            match_profile=args.continuous_match_profile,
            pnl_first_weight=args.pnl_first_weight,
        )
        ref_desc = {entry.entry_id: processed_desc[entry.entry_id] for entry in references}
        query_desc = {entry.entry_id: processed_desc[entry.entry_id] for entry in queries}
        pair_rows = build_descriptor_pair_rows(
            method_name,
            references,
            queries,
            ref_desc,
            query_desc,
        )
        descriptor_pair_rows_by_method[method_name] = pair_rows

        times = build_times[method_name]
        descriptor_build_summary[method_name] = {
            "reference_total": float(sum(times[e.entry_id] for e in references)),
            "query_total": float(sum(times[e.entry_id] for e in queries)),
            "reference_mean": float(np.mean([times[e.entry_id] for e in references])) if references else 0.0,
            "query_mean": float(np.mean([times[e.entry_id] for e in queries])) if queries else 0.0,
        }

    progress("Running StructureMatcher baselines")
    sm_pair_rows, sm_query_rows, sm_aggregate_rows = run_structure_matcher(
        references,
        queries,
    )
    write_csv(args.output_dir / "structurematcher_pairs.csv", sm_pair_rows)
    write_csv(args.output_dir / "structurematcher_query_summary.csv", sm_query_rows)
    write_csv(args.output_dir / "structurematcher_aggregate.csv", sm_aggregate_rows)

    progress("Evaluating threshold-based descriptor matching against StructureMatcher")
    threshold_prediction_rows: List[dict] = []
    threshold_summary_rows: List[dict] = []
    threshold_status = "ok"
    threshold_error = None
    threshold_split_policy = None
    try:
        threshold_split_policy = build_threshold_split_policy(references, seed=args.seed)
        threshold_prediction_rows, threshold_summary_rows = evaluate_threshold_pairwise_matching(
            descriptor_pair_rows_by_method=descriptor_pair_rows_by_method,
            sm_pair_rows=sm_pair_rows,
            references=references,
            split_policy=threshold_split_policy,
            match_profile=args.continuous_match_profile,
            pnl_first_weight=args.pnl_first_weight,
        )
    except RuntimeError as exc:
        threshold_status = "skipped"
        threshold_error = str(exc)
        progress(f"Skipping threshold evaluation: {threshold_error}")

    write_csv(args.output_dir / "pairwise_threshold_predictions.csv", threshold_prediction_rows)
    write_csv(args.output_dir / "pairwise_threshold_summary.csv", threshold_summary_rows)

    summary_formulas = list(args.formula) if not args.material_id and not reference_specs else []

    summary = {
        "source": args.source,
        "preset": args.preset,
        "n_references": len(references),
        "n_queries": len(queries),
        "selection": {
            "formulas": summary_formulas,
            "material_ids": args.material_id,
            "curated_reference_ids": [spec.material_id for spec in reference_specs],
            "curated_reference_labels": [spec.display_name for spec in reference_specs],
            "max_per_formula": args.max_per_formula,
            "max_total_structures": args.max_total_structures,
            "size_bucket": args.size_bucket,
            "min_sites": args.min_sites,
            "max_sites": args.max_sites,
            "min_reference_parents_per_formula": args.min_reference_parents_per_formula,
        },
        "query_generation": {
            "coord_noise": args.coord_noise,
            "lattice_noise": args.lattice_noise,
            "combined_noise": combined_noise_values,
            "include_supercell": not args.skip_supercell,
            "max_supercell_sites": args.max_supercell_sites,
            "seed": args.seed,
        },
        "descriptor_settings": {
            "dmax": args.dmax,
            "nmax": args.nmax,
            "lmax": args.lmax,
            "rbasis": args.rbasis,
            "normalize_reciprocal": args.normalize_reciprocal,
            "g_bin_width": args.g_bin_width,
            "continuous_match_profile": args.continuous_match_profile,
            "pnl_first_weight": args.pnl_first_weight,
            "primary_pool_scope": PRIMARY_POOL_SCOPE,
        },
        "descriptor_build_seconds": descriptor_build_summary,
        "threshold_pairwise": {
            "status": threshold_status,
            "error": threshold_error,
            "split_policy": threshold_split_policy,
            "summary_rows": threshold_summary_rows,
        },
        "structurematcher": sm_aggregate_rows,
    }

    summary_path = args.output_dir / "benchmark_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    progress(f"Wrote summary JSON to {summary_path}")

    print("=== Benchmark complete ===")
    print(f"Source:      {args.source}")
    print(f"Preset:      {args.preset}")
    print(f"Size bucket: {args.size_bucket}")
    print(f"References:  {len(references)}")
    print(f"Queries:     {len(queries)}")
    print(f"Primary pool: {PRIMARY_POOL_SCOPE}")
    print(f"Continuous match profile: {args.continuous_match_profile}")
    print(f"P_nl first-component weight: {args.pnl_first_weight}")
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
    for row in sm_aggregate_rows:
        print(
            f"StructureMatcher {row['setting']} true-parent match rate: "
            f"{row['true_parent_match_rate']:.4f}"
        )
    print(f"Wrote outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
