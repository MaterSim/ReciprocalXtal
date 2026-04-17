from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Allow importing reciprocal.py from repository root.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from reciprocal import RECP  # noqa: E402


DEFAULT_FORMULAS = ["C", "SiO2", "TiO2"]
DEFAULT_COORD_NOISE = [0.01, 0.02]
DEFAULT_LATTICE_NOISE = [0.01]
DEFAULT_COMBINED_NOISE = ["0.02:0.01"]
DEFAULT_G_SIGMAS = [0.03, 0.05, 0.10]
DEFAULT_G_BIN_WIDTH = 0.02
DEFAULT_ORIGIN_SHIFT = np.array([0.173, 0.257, 0.389], dtype=float)
PRIMARY_POOL_SCOPE = "same_formula"
ALL_REFERENCES_SCOPE = "all_references"


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
        "combined_noise": ["0.02:0.01", "0.05:0.02", "0.10:0.05"],
        "include_supercell": True,
        "max_sites": 48,
        "max_supercell_sites": 96,
        "g_sigmas": DEFAULT_G_SIGMAS,
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


def parse_combined_noise(values: Sequence[str]) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(
                f"Invalid combined noise specifier '{value}'. Use '<coord_A>:<lattice_fraction>'."
            )
        coord_str, lattice_str = value.split(":", 1)
        pairs.append((float(coord_str), float(lattice_str)))
    return pairs


def format_sigma_tag(sigma: float) -> str:
    return f"{sigma:.3f}".rstrip("0").rstrip(".").replace(".", "p")


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
    max_sites: int,
    symprec: float,
) -> List[BenchmarkEntry]:
    MPRester = _mp_rester_cls()
    references: List[BenchmarkEntry] = []

    with MPRester(api_key) as rester:
        if reference_specs:
            for spec in reference_specs:
                structure = canonicalize_reference_structure(
                    rester.get_structure_by_material_id(spec.material_id),
                    symprec=symprec,
                )
                if len(structure) > max_sites:
                    raise RuntimeError(
                        f"Curated reference {spec.material_id} exceeds --max-sites={max_sites}."
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
                if len(structure) > max_sites:
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
                if len(structure) > max_sites:
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
            "Try relaxing --max-sites or passing explicit --material-id values."
        )
    return references


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
    g_sigmas: Sequence[float],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
    descriptors: Dict[str, Dict[str, np.ndarray]] = {"reciprocal_power_spectrum": {}, "raw_gd": {}}
    runtimes: Dict[str, Dict[str, float]] = {"reciprocal_power_spectrum": {}, "raw_gd": {}}

    for sigma in g_sigmas:
        method = f"smoothed_gd_sigma_{format_sigma_tag(sigma)}"
        descriptors[method] = {}
        runtimes[method] = {}

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

        for sigma in g_sigmas:
            method = f"smoothed_gd_sigma_{format_sigma_tag(sigma)}"
            start = time.perf_counter()
            sigma_bins = sigma / g_bin_width
            smoothed = gaussian_filter1d(raw_gd, sigma=sigma_bins, mode="nearest")
            descriptors[method][entry.entry_id] = smoothed.astype(float)
            runtimes[method][entry.entry_id] = time.perf_counter() - start

    return descriptors, runtimes


def select_reference_pool(
    query: BenchmarkEntry,
    references: Sequence[BenchmarkEntry],
    *,
    pool_scope: str,
) -> List[BenchmarkEntry]:
    if pool_scope == PRIMARY_POOL_SCOPE:
        pool = [ref for ref in references if ref.formula == query.formula]
    elif pool_scope == ALL_REFERENCES_SCOPE:
        pool = list(references)
    else:
        raise ValueError(f"Unsupported pool scope: {pool_scope}")

    if not pool:
        raise RuntimeError(
            f"No reference structures available for query {query.entry_id} under pool scope '{pool_scope}'."
        )
    return pool


def run_continuous_retrieval(
    method: str,
    references: Sequence[BenchmarkEntry],
    queries: Sequence[BenchmarkEntry],
    ref_desc: Dict[str, np.ndarray],
    query_desc: Dict[str, np.ndarray],
    *,
    pool_scope: str,
) -> Tuple[List[dict], List[dict], dict]:
    pair_rows: List[dict] = []
    query_rows: List[dict] = []
    n_top1 = 0
    n_top5 = 0
    total_runtime = 0.0
    same_distances: List[float] = []
    different_distances: List[float] = []

    for query in queries:
        query_start = time.perf_counter()
        distances: List[Tuple[float, BenchmarkEntry]] = []
        qv = query_desc[query.entry_id]
        candidate_refs = select_reference_pool(query, references, pool_scope=pool_scope)

        for ref in candidate_refs:
            distance = float(np.linalg.norm(qv - ref_desc[ref.entry_id]))
            distances.append((distance, ref))
            is_true_parent = int(query.parent_id == ref.parent_id)
            pair_rows.append(
                {
                    "method": method,
                    "pool_scope": pool_scope,
                    "query_id": query.entry_id,
                    "query_parent_id": query.parent_id,
                    "query_display_name": query.display_name,
                    "query_family_name": query.family_name,
                    "query_variant": query.variant,
                    "query_variant_family": query.variant_family,
                    "reference_id": ref.entry_id,
                    "reference_parent_id": ref.parent_id,
                    "reference_display_name": ref.display_name,
                    "reference_family_name": ref.family_name,
                    "reference_variant": ref.variant,
                    "distance": distance,
                    "is_true_parent": is_true_parent,
                }
            )
            if is_true_parent:
                same_distances.append(distance)
            else:
                different_distances.append(distance)

        distances.sort(key=lambda item: item[0])
        query_runtime = time.perf_counter() - query_start
        total_runtime += query_runtime

        true_parent_rank = None
        nearest_wrong_distance = None
        for idx, (distance, ref) in enumerate(distances, start=1):
            if ref.parent_id == query.parent_id and true_parent_rank is None:
                true_parent_rank = idx
            if ref.parent_id != query.parent_id and nearest_wrong_distance is None:
                nearest_wrong_distance = distance
            if true_parent_rank is not None and nearest_wrong_distance is not None:
                break

        top_k = min(5, len(distances))
        top1_distance, top1_ref = distances[0]
        top1_correct = int(top1_ref.parent_id == query.parent_id)
        top5_correct = int(any(ref.parent_id == query.parent_id for _, ref in distances[:top_k]))
        n_top1 += top1_correct
        n_top5 += top5_correct

        query_rows.append(
            {
                "method": method,
                "pool_scope": pool_scope,
                "query_id": query.entry_id,
                "query_parent_id": query.parent_id,
                "query_display_name": query.display_name,
                "query_family_name": query.family_name,
                "query_variant": query.variant,
                "query_variant_family": query.variant_family,
                "top1_reference_id": top1_ref.entry_id,
                "top1_reference_parent_id": top1_ref.parent_id,
                "top1_reference_display_name": top1_ref.display_name,
                "top1_distance": top1_distance,
                "nearest_wrong_distance": nearest_wrong_distance,
                "true_parent_rank": true_parent_rank,
                "candidate_count": len(candidate_refs),
                "correct_top1": top1_correct,
                "correct_top5": top5_correct,
                "query_runtime_seconds": query_runtime,
            }
        )

    summary = {
        "method": method,
        "pool_scope": pool_scope,
        "n_queries": len(queries),
        "top1_accuracy": n_top1 / max(len(queries), 1),
        "top5_accuracy": n_top5 / max(len(queries), 1),
        "mean_query_runtime_seconds": total_runtime / max(len(queries), 1),
        "same_distance_mean": float(np.mean(same_distances)) if same_distances else None,
        "different_distance_mean": float(np.mean(different_distances)) if different_distances else None,
        "same_distance_min": float(np.min(same_distances)) if same_distances else None,
        "different_distance_min": float(np.min(different_distances)) if different_distances else None,
    }
    return pair_rows, query_rows, summary


def run_structure_matcher(
    references: Sequence[BenchmarkEntry],
    queries: Sequence[BenchmarkEntry],
    *,
    pool_scope: str,
) -> Tuple[List[dict], List[dict], List[dict]]:
    StructureMatcher = _structure_matcher_cls()
    settings = [
        MatcherSetting("strict", ltol=0.1, stol=0.1, angle_tol=2.0),
        MatcherSetting("medium", ltol=0.2, stol=0.3, angle_tol=5.0),
        MatcherSetting("loose", ltol=0.3, stol=0.5, angle_tol=10.0),
    ]

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
            candidate_refs = select_reference_pool(query, references, pool_scope=pool_scope)

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
                        "pool_scope": pool_scope,
                        "query_id": query.entry_id,
                        "query_parent_id": query.parent_id,
                        "query_display_name": query.display_name,
                        "query_family_name": query.family_name,
                        "query_variant": query.variant,
                        "query_variant_family": query.variant_family,
                        "reference_id": ref.entry_id,
                        "reference_parent_id": ref.parent_id,
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
                    "pool_scope": pool_scope,
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
                "pool_scope": pool_scope,
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
    if args.combined_noise == DEFAULT_COMBINED_NOISE:
        args.combined_noise = list(preset["combined_noise"])
    if args.max_sites == 40:
        args.max_sites = int(preset["max_sites"])
    if args.max_supercell_sites == 80:
        args.max_supercell_sites = int(preset["max_supercell_sites"])
    if args.g_sigma == DEFAULT_G_SIGMAS:
        args.g_sigma = list(preset["g_sigmas"])
    if args.skip_supercell and preset["include_supercell"]:
        pass
    return args, list(preset["references"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark reciprocal power spectrum P_nl, raw G(d), smoothed G(d), "
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
        "--material-id",
        action="append",
        default=[],
        help="Explicit MP material ID to benchmark. Repeat to pin a custom set.",
    )
    parser.add_argument("--max-per-formula", type=int, default=2)
    parser.add_argument("--max-total-structures", type=int, default=6)
    parser.add_argument("--max-sites", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--coord-noise", type=float, nargs="*", default=DEFAULT_COORD_NOISE)
    parser.add_argument("--lattice-noise", type=float, nargs="*", default=DEFAULT_LATTICE_NOISE)
    parser.add_argument("--combined-noise", type=str, nargs="*", default=DEFAULT_COMBINED_NOISE)
    parser.add_argument("--skip-supercell", action="store_true")
    parser.add_argument("--max-supercell-sites", type=int, default=80)
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
    parser.add_argument("--g-sigma", type=float, nargs="*", default=DEFAULT_G_SIGMAS)
    parser.add_argument(
        "--include-cross-composition",
        action="store_true",
        help="Also report mixed-composition retrieval across the full reference set as a secondary screen.",
    )
    args = parser.parse_args()

    args, reference_specs = apply_preset_defaults(args)
    rng = np.random.default_rng(args.seed)
    combined_noise_levels = parse_combined_noise(args.combined_noise)

    if args.source == "local":
        references, queries = discover_local_entries(args.dataset_root, symprec=args.symprec)
    else:
        api_key = get_mp_api_key(args.api_key)
        references = fetch_mp_references(
            api_key=api_key,
            formulas=args.formula,
            explicit_material_ids=args.material_id,
            reference_specs=reference_specs,
            max_per_formula=args.max_per_formula,
            max_total_structures=args.max_total_structures,
            max_sites=args.max_sites,
            symprec=args.symprec,
        )
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

    manifest_rows = build_manifest_rows(references, queries)
    write_csv(args.output_dir / "dataset_manifest.csv", manifest_rows)

    recp = RECP(dmax=args.dmax, nmax=args.nmax, lmax=args.lmax, rbasis=args.rbasis)
    all_entries = list(references) + list(queries)
    descriptors, build_times = compute_descriptors(
        all_entries,
        recp,
        normalize_reciprocal=args.normalize_reciprocal,
        g_bin_width=args.g_bin_width,
        g_sigmas=args.g_sigma,
    )

    continuous_summaries: Dict[str, dict] = {}
    cross_composition_summaries: Dict[str, dict] = {}
    descriptor_build_summary: Dict[str, dict] = {}

    for method_name, method_desc in descriptors.items():
        ref_desc = {entry.entry_id: method_desc[entry.entry_id] for entry in references}
        query_desc = {entry.entry_id: method_desc[entry.entry_id] for entry in queries}
        pair_rows, query_rows, summary = run_continuous_retrieval(
            method_name,
            references,
            queries,
            ref_desc,
            query_desc,
            pool_scope=PRIMARY_POOL_SCOPE,
        )
        write_csv(args.output_dir / f"{method_name}_pairwise.csv", pair_rows)
        write_csv(args.output_dir / f"{method_name}_retrieval.csv", query_rows)
        continuous_summaries[method_name] = summary

        if args.include_cross_composition:
            cross_pair_rows, cross_query_rows, cross_summary = run_continuous_retrieval(
                method_name,
                references,
                queries,
                ref_desc,
                query_desc,
                pool_scope=ALL_REFERENCES_SCOPE,
            )
            write_csv(
                args.output_dir / f"{method_name}_{ALL_REFERENCES_SCOPE}_pairwise.csv",
                cross_pair_rows,
            )
            write_csv(
                args.output_dir / f"{method_name}_{ALL_REFERENCES_SCOPE}_retrieval.csv",
                cross_query_rows,
            )
            cross_composition_summaries[method_name] = cross_summary

        times = build_times[method_name]
        descriptor_build_summary[method_name] = {
            "reference_total": float(sum(times[e.entry_id] for e in references)),
            "query_total": float(sum(times[e.entry_id] for e in queries)),
            "reference_mean": float(np.mean([times[e.entry_id] for e in references])) if references else 0.0,
            "query_mean": float(np.mean([times[e.entry_id] for e in queries])) if queries else 0.0,
        }

    sm_pair_rows, sm_query_rows, sm_aggregate_rows = run_structure_matcher(
        references,
        queries,
        pool_scope=PRIMARY_POOL_SCOPE,
    )
    write_csv(args.output_dir / "structurematcher_pairs.csv", sm_pair_rows)
    write_csv(args.output_dir / "structurematcher_query_summary.csv", sm_query_rows)
    write_csv(args.output_dir / "structurematcher_aggregate.csv", sm_aggregate_rows)

    summary = {
        "source": args.source,
        "preset": args.preset,
        "n_references": len(references),
        "n_queries": len(queries),
        "selection": {
            "formulas": args.formula,
            "material_ids": args.material_id,
            "curated_reference_ids": [spec.material_id for spec in reference_specs],
            "curated_reference_labels": [spec.display_name for spec in reference_specs],
            "max_per_formula": args.max_per_formula,
            "max_total_structures": args.max_total_structures,
            "max_sites": args.max_sites,
        },
        "query_generation": {
            "coord_noise": args.coord_noise,
            "lattice_noise": args.lattice_noise,
            "combined_noise": args.combined_noise,
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
            "g_sigma": args.g_sigma,
            "primary_pool_scope": PRIMARY_POOL_SCOPE,
            "include_cross_composition": args.include_cross_composition,
        },
        "descriptor_build_seconds": descriptor_build_summary,
        "continuous_methods": continuous_summaries,
        "cross_composition_continuous_methods": cross_composition_summaries,
        "structurematcher": sm_aggregate_rows,
    }

    summary_path = args.output_dir / "benchmark_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("=== Benchmark complete ===")
    print(f"Source:      {args.source}")
    print(f"Preset:      {args.preset}")
    print(f"References:  {len(references)}")
    print(f"Queries:     {len(queries)}")
    print(f"Primary pool: {PRIMARY_POOL_SCOPE}")
    for method_name, method_summary in continuous_summaries.items():
        print(
            f"{method_name} top-1/top-5: "
            f"{method_summary['top1_accuracy']:.4f}/{method_summary['top5_accuracy']:.4f}"
        )
    if args.include_cross_composition:
        for method_name, method_summary in cross_composition_summaries.items():
            print(
                f"{method_name} ({ALL_REFERENCES_SCOPE}) top-1/top-5: "
                f"{method_summary['top1_accuracy']:.4f}/{method_summary['top5_accuracy']:.4f}"
            )
    for row in sm_aggregate_rows:
        print(
            f"StructureMatcher {row['setting']} true-parent match rate: "
            f"{row['true_parent_match_rate']:.4f}"
        )
    print(f"Wrote outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
