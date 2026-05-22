from __future__ import division
import numpy as np
from ase.neighborlist import NeighborList
from scipy.ndimage import gaussian_filter1d


class RDFCalculator:
    """
    RDF-only calculator extracted from SO3 implementation.

    Provides element-wise, pairwise, and mean RDFs given an ASE Atoms object.
    """

    def __init__(self, rcut=3.5):
        self.rcut = float(rcut)
        self._atoms = None
        self.neighborlist = None
        self.neighbor_indices = None

    def init_atoms(self, atoms, atom_ids=None):
        """Initialize atoms and neighbor list."""
        self._atoms = atoms
        self.natoms = len(atoms)
        self.build_neighbor_list(atom_ids)

    def build_neighbor_list(self, atom_ids=None):
        """Build neighbor list vectors and index pairs within cutoff."""
        atoms = self._atoms
        cell_matrix = atoms.get_cell()
        neighbors = []
        neighbor_indices = []

        if atom_ids is None:
            atom_ids = range(len(atoms))

        cutoffs = [self.rcut] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
        nl.update(atoms)

        for i in atom_ids:
            center_atom = atoms.positions[i]
            indices, offsets = nl.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                pos = atoms.positions[j] + offset @ cell_matrix - center_atom
                if np.sum(np.abs(pos)) < 1e-3:
                    continue
                neighbors.append(pos)
                neighbor_indices.append([i, j])

        self.neighborlist = np.array(neighbors, dtype=np.float64)
        neighbor_indices = np.asarray(neighbor_indices, dtype=int)
        if neighbor_indices.size == 0:
            neighbor_indices = np.empty((0, 2), dtype=int)
        self.neighbor_indices = neighbor_indices

    def compute_rdf(self, atoms, atom_ids=None, mode="mean", num_bins=50, sigma=2.0):
        """
        Compute RDFs for selected atoms.

        mode: "mean" (mean over all centers), "pairwise" (center/neighbor element pairs),
              or "element" (mean per center element).
        Returns (rdf_array, labels) where labels may be empty for mean mode.
        """
        if atom_ids is None:
            atom_ids = list(range(len(atoms)))
        self.init_atoms(atoms, atom_ids)

        # Collect neighbor distances per center
        centers = np.array(atom_ids, dtype=int)
        center_to_rows = {cid: idx for idx, cid in enumerate(centers)}
        per_center = [[] for _ in centers]
        for vec, (ci, nj) in zip(self.neighborlist, self.neighbor_indices):
            if ci in center_to_rows:
                per_center[center_to_rows[ci]].append(np.linalg.norm(vec))

        max_cn = max((len(r) for r in per_center), default=0)
        dists = np.zeros((len(centers), max_cn))
        for idx, row in enumerate(per_center):
            dists[idx, :len(row)] = row

        if mode == "pairwise":
            rdf_array, labels = self.dist2rdf_pairwise(dists, centers, num_bins=num_bins, sigma=sigma)
        elif mode == "element":
            rdf_array = self.dist2rdf_elementwise(dists, centers, num_bins=num_bins, sigma=sigma)
            labels = getattr(self, "rdf_labels", [])
        else:  # mean
            rdf_array = self.dist2rdf(dists, num_bins=num_bins, sigma=sigma)
            rdf_array = np.mean(rdf_array, axis=0)
            labels = []
        return rdf_array, labels

    def dist2rdf(self, dists, num_bins=30, sigma=0.5):
        """Simple RDF histogram with Gaussian smoothing."""
        N, _ = dists.shape
        dr = self.rcut / num_bins
        rdf = np.zeros((N, num_bins))
        valid_mask = (dists > 0) & (dists < self.rcut)
        bin_indices = (dists * valid_mask / dr).astype(int)

        for i in range(N):
            valid_idx = valid_mask[i]
            bincount = np.bincount(bin_indices[i][valid_idx], minlength=num_bins)
            rdf[i, :len(bincount)] += bincount

        rdf = gaussian_filter1d(rdf, sigma=sigma, axis=1, mode='nearest')
        return rdf

    def dist2rdf_pairwise(self, dists, atom_ids, num_bins=20, sigma=2):
        """Pair-resolved RDF averaged by center element."""
        dr = self.rcut / num_bins
        center_numbers = np.array([self._atoms[i].number for i in atom_ids])
        unique_centers = np.unique(center_numbers)
        all_neighbor_numbers = np.unique(self._atoms.numbers)

        def z2sym(z):
            try:
                idx = np.where(self._atoms.numbers == int(z))[0]
                if idx.size > 0:
                    return self._atoms[idx[0]].symbol
                return f"Z{int(z)}"
            except Exception:
                return f"Z{z}"

        pair_hist = {(zc, zn): np.zeros(num_bins) for zc in unique_centers for zn in all_neighbor_numbers}
        center_counts = {zc: 0 for zc in unique_centers}

        max_cn = dists.shape[1]
        for row_idx, center_id in enumerate(atom_ids):
            center_z = center_numbers[row_idx]
            center_counts[center_z] += 1

            mask = self.neighbor_indices[:, 0] == center_id
            neighbors = self.neighbor_indices[mask]
            if len(neighbors) == 0:
                continue

            truncate = min(len(neighbors), max_cn)
            neighbor_ids = neighbors[:truncate, 1]
            neighbor_zs = np.array([self._atoms[j].number for j in neighbor_ids])
            neighbor_dists = dists[row_idx, :truncate]

            valid_mask = (neighbor_dists > 0) & (neighbor_dists < self.rcut)
            if not np.any(valid_mask):
                continue

            bins = (neighbor_dists[valid_mask] / dr).astype(int)
            bins = np.clip(bins, 0, num_bins - 1)

            for b, nz in zip(bins, neighbor_zs[valid_mask]):
                pair_hist[(center_z, nz)][b] += 1.0

        labels = []
        rdf_rows = []
        for zc in sorted(unique_centers):
            count = max(center_counts.get(zc, 1), 1)
            for zn in sorted(all_neighbor_numbers):
                hist = pair_hist[(zc, zn)] / count
                hist = gaussian_filter1d(hist, sigma=sigma, mode='nearest')
                rdf_rows.append(hist)
                labels.append(f"{z2sym(zc)}-{z2sym(zn)}")

        rdf_array = np.vstack(rdf_rows) if rdf_rows else np.zeros((0, num_bins))
        return rdf_array, labels

    def dist2rdf_elementwise(self, dists, atom_ids, num_bins=50, sigma=2):
        """Element-wise RDF averaged over centers of each element."""
        dr = self.rcut / num_bins
        atom_numbers = np.array([self._atoms[i].number for i in atom_ids])
        unique_elements = np.unique(atom_numbers)

        def compute_single_rdf(dists_filtered):
            n_rows = dists_filtered.shape[0]
            rdf = np.zeros((n_rows, num_bins))
            valid_mask = (dists_filtered > 0) & (dists_filtered < self.rcut)
            bin_indices = (dists_filtered * valid_mask / dr).astype(int)
            for i in range(n_rows):
                valid_idx = valid_mask[i]
                bincount = np.bincount(bin_indices[i][valid_idx], minlength=num_bins)
                rdf[i, :len(bincount)] += bincount
            rdf = gaussian_filter1d(rdf, sigma=sigma, axis=1, mode='nearest')
            return rdf

        rdf_rows = []
        labels = []
        for element_number in sorted(unique_elements):
            mask = atom_numbers == element_number
            dists_filtered = dists[mask, :]
            if dists_filtered.size == 0:
                continue
            rdf_elem = compute_single_rdf(dists_filtered)
            rdf_rows.append(np.mean(rdf_elem, axis=0))
            labels.append(f"Z{element_number}")
        rdf_array = np.vstack(rdf_rows) if rdf_rows else np.zeros((0, num_bins))
        self.rdf_labels = labels
        return rdf_array


__all__ = ["RDFCalculator"]


