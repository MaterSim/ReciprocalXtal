"""
Module for Reciprocal Representation Simulation
"""
import importlib.resources
import numpy as np
from monty.serialization import loadfn
from pyxtal.database.element import Element
from pyxtal.XRD import create_index
import torch
from functools import lru_cache


def _load_spherical_harmonics():
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        from e3nn.o3 import spherical_harmonics as implementation

        return implementation

    with safe_globals([slice]):
        from e3nn.o3 import spherical_harmonics as implementation

    return implementation


spherical_harmonics = _load_spherical_harmonics()

with importlib.resources.as_file(
    importlib.resources.files("pyxtal") / "database" / "atomic_scattering_params.json"
) as path:
    ATOMIC_SCATTERING_PARAMS = loadfn(path)

from scipy.special import spherical_jn
from scipy.optimize import brentq

@lru_cache(maxsize=64)
def spherical_bessel_zeros(l, nmax):
    """
    Compute first nmax zeros of spherical Bessel function j_l(x).

    Parameters
    ----------
    l : int
        Angular momentum order (0=s, 1=p, 2=d, etc.)
    nmax : int
        Number of zeros to compute

    Returns
    -------
    zeros : np.ndarray
        Array of first nmax zeros, shape (nmax,)
    """
    nmax = int(nmax)  # Defensive conversion
    zeros = []
    for n in range(1, nmax + 1):
        # Approximate bounds for n-th zero of j_l
        a = (n + l/2 - 0.5) * np.pi
        b = (n + l/2 + 0.5) * np.pi

        # Find zero using Brent's method
        z = brentq(lambda x: spherical_jn(l, x), a, b)
        zeros.append(z)

    return np.array(zeros, dtype=np.float64)

def bessel_basis(r, nmax=6, r_cut=0.24, l=0):
    """
    ORTHONORMAL spherical Bessel radial basis on [0, r_cut] under r^2 dr measure.

    This constructs basis functions that satisfy:
    ∫₀^r_cut r² Rₙ(r) Rₘ(r) dr = δₙₘ (Kronecker delta)

    Parameters
    ----------
    r : torch.Tensor or np.ndarray
        Radial distances, shape (N,) or (N, 1)
    nmax : int
        Number of basis functions
    r_cut : float
        Cutoff radius
    l : int
        Angular momentum order (0=s, 1=p, 2=d, 3=f)

    Returns
    -------
    basis : torch.Tensor
        Orthonormal basis functions, shape (N, nmax)

    Notes
    -----
    This implementation uses Bessel zeros to guarantee orthonormality:
    1. Finds zeros z_{l,n} where j_l(z_{l,n}) = 0
    2. Scales arguments: j_l(z_{l,n} * r/r_cut)
    3. Normalizes by: N = sqrt(2)/(r_cut^{3/2} * |j_{l+1}(z_{l,n})|)

    Verified orthonormality: max |off-diag| < 2e-7
    """
    # Squeeze r to ensure it's 1D: (N, 1) -> (N,)
    if isinstance(r, torch.Tensor):
        if r.dim() > 1:
            r = r.squeeze(-1)
        r_numpy = r.cpu().numpy()
        is_torch = True
        device, dtype = r.device, r.dtype
    else:
        if r.ndim > 1:
            r = r.squeeze(-1)
        r_numpy = r
        is_torch = False

    # Step 1: Get zeros of spherical Bessel function j_l (cached)
    zeros = spherical_bessel_zeros(l, nmax)

    # Initialize basis array
    basis = np.zeros((len(r_numpy), nmax), dtype=np.float64)

    # Step 2-3: Build orthonormal basis functions
    for n, z_n in enumerate(zeros):
        # Scale argument to [0, r_cut] using n-th zero
        x = z_n * r_numpy / r_cut

        # Evaluate spherical Bessel function
        jn = spherical_jn(l, x)

        # Compute normalization constant
        # N_{n,l} = sqrt(2) / (r_cut^{3/2} * |j_{l+1}(z_{l,n})|)
        j_l1 = spherical_jn(l + 1, z_n)
        norm = np.sqrt(2.0) / (r_cut**1.5 * np.abs(j_l1))

        # Store normalized basis function
        basis[:, n] = norm * jn

    # Convert back to torch if input was torch
    if is_torch:
        return torch.tensor(basis, dtype=dtype, device=device)
    else:
        return basis

def gto_basis(r, nmax=6, r_cut=0.24):
    """
    Create a set of radial basis functions

    Args:
        r: radial distances, shape (N, 1) or (N,)
        nmax: maximum radial quantum number
        r_cut: cutoff radius for normalization (defaults to max radius)

    Returns:
        Tensor of shape (N, nmax)
    """
    # Squeeze r to ensure it's 1D: (N, 1) -> (N,)
    if isinstance(r, torch.Tensor) and r.dim() > 1:
        r = r.squeeze(-1)
    elif isinstance(r, np.ndarray) and r.ndim > 1:
        r = r.squeeze(-1)

    # Scale r to [0, 1] range
    r_scaled = r / r_cut

    # Create empty tensor to hold basis functions
    if isinstance(r, torch.Tensor):
        basis = torch.zeros((r.shape[0], nmax), dtype=r.dtype, device=r.device)
    else:
        basis = np.zeros((r.shape[0], nmax))

    # Fill with basis functions (Gaussian-type orbitals)
    for n in range(nmax):
        # GTO-like function: r^n * exp(-alpha * r^2)
        alpha = 1.0 / (n + 1)  # Different width for each basis function
        if isinstance(r, torch.Tensor):
            basis[:, n] = ((r_scaled ** n) * torch.exp(-alpha * r_scaled ** 2)).view(-1)
        else:
            basis[:, n] = (r_scaled ** n) * np.exp(-alpha * r_scaled ** 2)

    return basis  # Shape (N, nmax)

def chebyshev_basis(r, nmax=6, r_cut=0.24, normalize=True):
    """
    Chebyshev polynomial basis - excellent for oscillatory features
    """
    # Squeeze r to ensure it's 1D: (N, 1) -> (N,)
    if isinstance(r, torch.Tensor):
        if r.dim() > 1:
            r = r.squeeze(-1)
        is_torch = True
    else:
        if r.ndim > 1:
            r = r.squeeze(-1)
        is_torch = False

    # Scale r to [-1, 1] range for Chebyshev polynomials
    r_scaled = 2 * (r / r_cut) - 1
    if is_torch:
        basis = torch.zeros((r.shape[0], nmax), dtype=r.dtype, device=r.device)
    else:
        basis = np.zeros((r.shape[0], nmax), dtype=r.dtype if hasattr(r, "dtype") else np.float64)

    # T0(x) = 1, T1(x) = x
    basis[:, 0] = 1
    if nmax > 1:
        basis[:, 1] = r_scaled.reshape(-1)

    # Recurrence relation: Tn+1(x) = 2x*Tn(x) - Tn-1(x)
    for n in range(2, nmax):
        basis[:, n] = 2 * r_scaled.reshape(-1) * basis[:, n-1] - basis[:, n-2]

    # Normalize each basis function with Chebyshev weight
    if normalize:
        x_vals = r_scaled.reshape(-1)
        dx = x_vals[1] - x_vals[0] if len(x_vals) > 1 else 1.0
        # Chebyshev weight: 1/sqrt(1-x^2), avoiding singularities
        if is_torch:
            weights = 1.0 / torch.sqrt(torch.clamp(1 - x_vals**2, min=1e-10))
        else:
            weights = 1.0 / np.sqrt(np.clip(1 - x_vals**2, 1e-10, None))
        weights = weights * dx

        for n in range(nmax):
            # Compute norm with Chebyshev weight
            if is_torch:
                norm_sq = torch.sum(weights * basis[:, n] ** 2)
                if norm_sq > 1e-10:
                    basis[:, n] /= torch.sqrt(norm_sq)
            else:
                norm_sq = np.sum(weights * basis[:, n] ** 2)
                if norm_sq > 1e-10:
                    basis[:, n] /= np.sqrt(norm_sq)

    return basis

class RECP:
    """
    A class to compute the crystal in the reciprocal space.

    Args:
        d_max (float): maximum d-spacing to consider in the reciprocal space
        nmax: int, degree of radial expansion
        lmax: int, degree of spherical harmonic expansion
    """

    def __init__(self, dmax=6.0, nmax=4, lmax=4, rbasis='chebyshev', res=0.1, sigma=None):
        self.dmax = dmax
        self.nmax = nmax
        self.lmax = lmax
        self.num_bins = int(np.ceil(self.dmax / res))
        self.res = res
        if sigma is None: sigma = 5*self.res
        self.sigma = sigma
        self.rbasis = rbasis
        self.rcut = 0.5 * self.dmax / np.pi
        self.msize = 2 * self.lmax + 1
        self.l_vals = np.repeat(np.arange(self.lmax + 1), [2 * l + 1 for l in range(self.lmax + 1)])
        self.m_vals = np.concatenate([np.arange(-l, l + 1) for l in range(self.lmax + 1)])
        self.midx_vals = self.msize // 2 + self.m_vals

    def __str__(self):
        s = f"Reciprocal space expansion with cutoff: {self.dmax:6.3f} per Ang\n"
        s += f"lmax: {self.lmax}, nmax: {self.nmax}, rbasis: {self.rbasis}\n"
        return s

    def __repr__(self):
        return str(self)

    def build_reciprocal(self, atoms):
        """
        3x3 representation -> 1x6 (a, b, c, alpha, beta, gamma)
        """
        #print(atoms)
        rec_matrix = atoms.cell.reciprocal()
        hkl_index = create_index()
        hkl_max = np.array([1, 1, 1])
        hkl_min = np.array([-1, -1, -1])

        for index in hkl_index:
            d = np.linalg.norm(np.dot(index, rec_matrix)) * (2 * np.pi)
            multiple = int(np.ceil(self.dmax / d))
            index *= multiple
            for i in range(len(hkl_max)):
                if hkl_max[i] < index[i]:
                    hkl_max[i] = index[i]
                if hkl_min[i] > index[i]:
                    hkl_min[i] = index[i]
        h0, k0, l0 = hkl_min*2
        h1, k1, l1 = hkl_max*2
        h = np.arange(h0, h1 + 1)
        k = np.arange(k0, k1 + 1)
        l = np.arange(l0, l1 + 1)

        hkl = np.array(np.meshgrid(h, k, l)).transpose()
        hkl = np.reshape(hkl, [len(h) * len(k) * len(l), 3])
        hkl = hkl[np.where(hkl.any(axis=1))[0]]
        d_hkl = np.linalg.norm(hkl@rec_matrix, axis=1) * (2 * np.pi)

        mask = np.where(d_hkl < self.dmax)[0]
        hkl, d_hkl = hkl[mask], d_hkl[mask]
        #print("hkl_max", h0, h1, k0, k1, l0, l1, d_hkl.shape, "d_hkl", d_hkl.min(), d_hkl.max())

        N_atoms = len(atoms)
        s2 = d_hkl ** 2 / (16 * np.pi ** 2)
        # Compute the atomic scattering factors
        coeffs = np.zeros([N_atoms, 4, 2])
        zs = np.zeros([N_atoms, 1], dtype=int)
        element_cache = {
            elem: (ATOMIC_SCATTERING_PARAMS[elem], Element(elem).z)
            for elem in set(atoms.get_chemical_symbols())
        }
        for i, elem in enumerate(atoms.get_chemical_symbols()):
            coeff, z = element_cache[elem]
            coeffs[i, :, :] = coeff
            zs[i] = z

        tmp1 = np.exp(np.einsum("ij,k->ijk", -coeffs[:, :, 1], s2))  # N*4, M
        tmp2 = np.einsum("ij,ijk->ik", coeffs[:, :, 0], tmp1)  # N*4, N*M
        sfs = np.add(-41.78214 * np.einsum("ij,j->ij", tmp2, s2), zs)  # N*M, M -> N*M
        # to add dampling factor to ensure the decay to 0?

        # Compute the structure factors
        const = -2j * np.pi
        positions = atoms.get_scaled_positions()#; print("positions", positions)
        g_dot_rs = np.dot(positions, hkl.T)  # N_atoms * M_hkl
        exps = np.exp(const * g_dot_rs)
        fs = np.sum(sfs * exps, axis=0)
        intensities = (fs * fs.conjugate()).real  # M
        masks = np.where(intensities > 1e-4)[0]
        hkl, intensities, d_hkl, fs = hkl[masks], intensities[masks], d_hkl[masks], fs[masks]
        sfs = sfs[:, masks]
        I0 = np.sum(zs)**2
        intensities /= I0  # Normalize the intensities
        intensities *= np.cos(0.5 * np.pi * d_hkl / self.dmax)  # Apply the Gaussian factor
        # Remove peaks that survive the raw structure-factor threshold but become
        # numerically negligible after normalization and tapering near dmax.
        post_eps = max(1e-8 * float(np.max(intensities)), 1e-12)
        post_mask = intensities > post_eps
        hkl, intensities, d_hkl = hkl[post_mask], intensities[post_mask], d_hkl[post_mask]
        #print("intensities", intensities.shape, intensities.min(), intensities.max())
        #max_idx = np.argmax(intensities); print("max", intensities[max_idx], hkl[max_idx], d_hkl[max_idx], fs[max_idx])
        #min_idx = np.argmin(intensities); print("min", intensities[min_idx], hkl[min_idx], d_hkl[min_idx], fs[min_idx])
        # find the id of hkl=[1,1,1]
        #id_111 = np.where(np.all(hkl == [1, 1, 1], axis=1))[0]
        #if len(id_111) > 0:
        #    print("hkl=[1,1,1]", intensities[id_111], d_hkl[id_111], sfs[0, id_111], fs[id_111])
        return hkl@rec_matrix, intensities, d_hkl

    def compute(self, atoms, norm=False):
        """
        d for any give abitray [h,k,l] index
        """
        coords, vals, ds = self.build_reciprocal(atoms)
        p = self.compute_sph_torch(coords, vals, norm=norm)
        #print("\np shape:", p.shape, "p min:", p.min(), "p max:", p.max())
        rdf = self.compute_rdf(ds, vals)
        #D = np.concatenate([p, rdf], axis=0)
        #print("D shape:", D.shape, "D min:", D.min(), "D max:", D.max())

        return p, rdf

    def compute_rdf(self, ds, vals):
        """
        Get the radial distribution function (RDF) from the d-spacing and values.
        """
        from scipy.ndimage import gaussian_filter1d
        #print("number of bins:", self.num_bins)
        bins = np.linspace(0, self.dmax, self.num_bins)
        rdf, _ = np.histogram(ds, bins=bins, weights=vals)
        rdf = gaussian_filter1d(rdf, sigma=self.sigma)
        #ids = np.where(rdf > 0.01)[0]; print(len(ds)); print("loc", bins[ids][:5]); print("pek", rdf[ids][:5])
        #print("rdf shape:", rdf.shape, "rdf min:", rdf.min(), "rdf max:", rdf.max())
        return rdf

    def build_radial_grid(self, r_values=None, intensities=None):
        """
        Build a single representative radial shell in the same units as
        |q| = ||xyz||.

        The active formulation intentionally uses one shell centered at the
        intensity-weighted mean reciprocal radius.
        """
        if r_values is None or intensities is None or len(r_values) == 0:
            r_center = 0.5 * self.rcut
        else:
            r_values = np.asarray(r_values, dtype=np.float64).reshape(-1)
            intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
            weight_sum = np.sum(intensities)
            if abs(weight_sum) < 1e-12:
                r_center = np.mean(r_values)
            else:
                r_center = np.sum(r_values * intensities) / weight_sum
            r_center = np.clip(r_center, 0.0, self.rcut)

        return np.asarray([r_center], dtype=np.float64)

    def radial_basis_on_grid(self, r_grid):
        """
        Evaluate the selected radial basis on the representative shell.
        """
        r_input = np.asarray(r_grid, dtype=np.float64).reshape(-1, 1)
        if self.rbasis == 'chebyshev':
            return chebyshev_basis(r_input, self.nmax, self.rcut)
        elif self.rbasis == 'gto':
            return gto_basis(r_input, self.nmax, self.rcut)
        return bessel_basis(r_input, self.nmax, self.rcut)

    def compute_sph_torch(self, xyz, v, norm=False):
        """
        Compute a reciprocal-space descriptor using the paper-style angular
        projection, while collapsing the radial dependence to one
        representative shell.

        The active approximation is:

            a_lm = sum_i I_i Y_lm^*(qhat_i)
            A_nlm = R_n(d_shell) a_lm
            P_nl = sum_m |A_nlm|^2

        where d_shell is one representative reciprocal radius for the
        structure. This intentionally ignores shell-to-shell radial
        integration and keeps only one shell.

        Args:
            xyz: Tensor of shape (N, 3) representing 3D coordinates.
            v: Tensor of shape (N,) representing scalar values at each point.
            norm: Whether to normalize the final descriptor.

        Returns:
            Array of shape
            (nmax * (lmax + 1),)
            representing the one-shell diagonal P_{nl} descriptor.
        """
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        if xyz.shape[0] == 0:
            return np.zeros(self.nmax * (self.lmax + 1), dtype=np.float32)

        intensities = np.asarray(v, dtype=np.float64).reshape(-1)
        radii = np.linalg.norm(xyz, axis=1)
        safe_r = np.where(radii > 1e-12, radii, 1e-12)

        # 1. Choose one representative shell radius and evaluate the selected
        # radial basis on that shell.
        r_grid = self.build_radial_grid(radii, intensities)
        radial_basis = np.asarray(self.radial_basis_on_grid(r_grid), dtype=np.float64).reshape(-1)

        # 2. Real spherical harmonics evaluated directly at reciprocal peak
        # directions using the faster e3nn implementation.
        rhat_t = torch.tensor(xyz / safe_r[:, None], dtype=torch.float32)
        degrees = list(range(self.lmax + 1))
        Y = spherical_harmonics(
            degrees,
            rhat_t,
            normalize=False,
            normalization="norm",
        ).detach().cpu().numpy()

        # 3. One-shell angular coefficients a_lm and one-shell radial weighting.
        a_lm = np.zeros((self.lmax + 1, self.msize), dtype=np.float64)
        offset = 0
        weighted_Y = intensities[:, None] * Y
        for l in range(self.lmax + 1):
            dim = 2 * l + 1
            a_lm[l, self.msize // 2 - l : self.msize // 2 + l + 1] = np.sum(
                weighted_Y[:, offset : offset + dim],
                axis=0,
            )
            offset += dim

        # 4. Build one-shell A_nlm and keep only the diagonal power
        #    P_nl = sum_m |A_nlm|^2.
        A = radial_basis[:, None, None] * a_lm[None, :, :]
        P = np.sum(np.abs(A) ** 2, axis=2).real
        descriptor = P.reshape(-1).astype(np.float32)
        if norm:
            descriptor_norm = float(np.linalg.norm(descriptor))
            descriptor /= (descriptor_norm + 1e-9)
        return descriptor

    def plot(self, data, filename='reciprocal.png'):
        """
        Plot the computed reciprocal space representation and RDF.

        Args:
            data: Tuple of (rdf, p, label) where rdf is the radial distribution function
                  and p is the expansion coefficients, label is a string for the plot title.
            filename: Name of the file to save the plot.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        for (p, rdf, label) in data:
            #plt.plot(p, label=label, alpha=0.5, lw=0.9, marker='o', markersize=3, linestyle='-')
            plt.plot(p, label=label, alpha=0.5, lw=0.9)

        plt.xlabel(f'Index ({len(p)})')
        plt.ylabel('Expansion Coefficients')
        plt.legend(loc=1)

        plt.subplot(1, 2, 2)
        x = np.linspace(0, self.rcut, len(rdf))
        for (p, rdf, label) in data:
            plt.plot(x, rdf, label=label, alpha=0.5, lw=0.9)
        plt.xlabel(f'd-spacing (per Angstrom)')
        plt.ylabel('RDF')
        plt.legend(loc=1)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def reconstruction(self, spg, wps, elements, rep0, P_ref, rdf_ref, verbose=False):
        """
        Generate a crystal with the desired local P_ref

        Args:
            spg (int): pyxtal.symmetry.Group object
            wps: list of wps for the disired crystal (e.g., [wp1, wp2])
            P_ref: reference enviroment

        Returns:
            xtal and its mse loss
        """
        torch.autograd.set_detect_anomaly(True)

        def apply_bounds(tensor):
            """Clamps tensor values between 0 and 1."""
            with torch.no_grad():
                tensor.clamp_(0.0, 1.0)

        # Clone and enable gradients for `reps`
        rep = rep.clone().detach().requires_grad_(True)
        generators = generators.clone().detach()

        # Choose optimizer
        optimizer = torch.optim.Adam([rep_batch], lr=lr)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

        # Optimization loop
        for step in range(num_steps):
            optimizer.zero_grad()

            # Compute losses per sample (B,)
            loss = self.loss(spg, wps, elements, P_ref, RDF_ref)
            loss.backward(torch.ones_like(loss))
            torch.nn.utils.clip_grad_norm_(rep_batch, max_norm=10.0)  # Gradient clipping

            # Step the scheduler
            optimizer.step()
            if step > 100: scheduler.step()

            if verbose and step % 1 == 0:
                print(f"Step {step}, {loss_sum:.6f}, LR={scheduler.get_last_lr()[0]:.6f}")
            if step + 1 == num_steps:
                print(f"stopping at last iteration")
        #xtal =
        return rep.detach(), losses.detach()

    def loss(self, spg, wps, elements, P_ref, RDF_ref):
        res =  WP.get()
        p, xrd, rdf = self.compute()
        loss1 = torch.sum()
        return loss

    def compute_pnl_from_cif(self, cif_path, norm=False):
        """Load a CIF file and return the Pnl vector."""
        from ase.io import read
        atoms = read(cif_path)
        p, rdf = self.compute(atoms, norm=norm)
        return p

    def compute_pnl_for_cifs(self, cif_paths, norm=False):
        """Compute Pnl vectors for multiple CIF files."""
        out = {}
        for path in cif_paths:
            out[path] = self.compute_pnl_from_cif(path, norm=norm)
        return out

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    import time

    recp = RECP(dmax=10.0, nmax=10, lmax=10, rbasis="bessel")

    cif_files = [
        "benchmark2/datasets_2/medium/structures/queries/medium__mp-1080826__conventional_standard__mp-1080826.cif",
        "benchmark2/datasets_2/medium/structures/references/medium__mp-1080826__reference__mp-1080826.cif",
        "benchmark2/datasets_2/medium/structures/references/medium__mp-1190171__reference__mp-1190171.cif",
    ]

    t0 = time.time()
    pnl_map = recp.compute_pnl_for_cifs(cif_files, norm=False)
    t1 = time.time()
    print(f"\nComputed PnL for {len(cif_files)} CIFs in {t1-t0:.3f} s")

    # calculate distance between the first two and the last one
    d0 = time.time()
    dist_1 = np.linalg.norm(pnl_map[cif_files[0]] - pnl_map[cif_files[1]])
    #L1 distance
    #dist_1 = torch.sum(torch.abs(pnl_map[cif_files[0]] - pnl_map[cif_files[1]]))
    dist_2 = np.linalg.norm(pnl_map[cif_files[0]] - pnl_map[cif_files[2]])
    #dist_2 = torch.sum(torch.abs(pnl_map[cif_files[0]] - pnl_map[cif_files[2]]))
    d1 = time.time()
    print(f"Distance calc time: {d1-d0:.3f} s")
    print(f"\nDistance between query and correct reference: {dist_1:.4f}")
    print(f"Distance between query and wrong reference: {dist_2:.4f}")

    for path, pnl in pnl_map.items():
        print(f"\n{path}")
        print(pnl)

    labels = [
        "Query: conventional_standard",
        "Reference: mp-1080826",
        "Wrong reference: mp-1190171",
    ]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "figure.dpi": 300,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, path, label in zip(axes, cif_files, labels):
        pnl = np.asarray(pnl_map[path], dtype=float)
        ax.plot(pnl, linewidth=1.6)
        ax.set_title(label)
        ax.set_xlabel("Power Spectrum Index")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel(r"$P_{nl}$")
    fig.suptitle("Three-way $P_{nl}$ comparison", fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = Path("reciprocal_three_way_power_spectrum_new.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {output_path}")
