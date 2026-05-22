from pathlib import Path
import sys

# Allow importing project-level modules when running as a script
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import math

# Set environment variables before importing numpy/torch for deterministic BLAS.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import importlib.resources
from copy import deepcopy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from pyxtal import pyxtal

from reciprocal_torch import RECP
from batch_sym import Symmetry


def to_numpy_1d(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


class TorchRDF:
    """
    Simple differentiable RDF via Gaussian-smoothed pair-distance histogram.

    This is used only inside the optimization objective so the entire
    reconstruction path remains differentiable. It is not intended to exactly
    reproduce the older NumPy RDF implementation.
    """

    def __init__(self, rcut=2.1, num_bins=160, sigma=0.04, device=None, dtype=torch.float64):
        self.rcut = float(rcut)
        self.num_bins = int(num_bins)
        self.sigma = float(sigma)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.centers = torch.linspace(0.0, self.rcut, self.num_bins, device=self.device, dtype=self.dtype)

    def compute(self, coords):
        if coords.shape[0] < 2:
            return torch.zeros(self.num_bins, dtype=self.dtype, device=coords.device)

        dmat = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
        iu = torch.triu_indices(dmat.shape[0], dmat.shape[1], offset=1, device=coords.device)
        dists = dmat[iu[0], iu[1]]
        dists = dists[dists < self.rcut]
        if dists.numel() == 0:
            return torch.zeros(self.num_bins, dtype=self.dtype, device=coords.device)

        diff = dists[:, None] - self.centers[None, :]
        rdf = torch.exp(-0.5 * (diff / self.sigma) ** 2).sum(dim=0)
        return rdf / rdf.sum().clamp_min(1e-12)


class InverseOptimizer:
    """
    Fully differentiable inverse optimization for the diamond scripts.

    The optimization variables are the compact normalized representation
    returned by `batch_sym.get_batch_from_rows`, which matches pyxtal's 1D
    representation ordering for the supported prototypes here.
    """

    def __init__(
        self,
        reference_xtal,
        recp_params=None,
        weight_descriptor=2.0,
        weight_rdf=1.0,
        lr=3e-2,
        steps=800,
        device=None,
    ):
        self.ref_xtal = deepcopy(reference_xtal)
        self.spg = int(reference_xtal.group.number)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

        self.lattice_norm_factor = 35.0
        self.angle_norm_factor = 180.0
        self.lattice_bounds = (2.0, 7.0)
        self.angle_bounds = (30.0, 150.0)
        self.wyckoff_bounds = (0.0, 1.0)

        self.weight_p = float(weight_descriptor)
        self.weight_rdf = float(weight_rdf)
        self.lr = float(lr)
        self.steps = int(steps)

        recp_params = dict(recp_params or {})
        self.recp = RECP(
            dmax=recp_params.get("dmax", 10.0),
            nmax=recp_params.get("nmax", 10),
            lmax=recp_params.get("lmax", 10),
            rbasis=recp_params.get("rbasis", "bessel"),
        )
        self.rdf_calculator = TorchRDF(
            rcut=recp_params.get("rcut", 2.1),
            num_bins=recp_params.get("rdf_bins", 160),
            sigma=recp_params.get("rdf_sigma", 0.04),
            device=self.device,
            dtype=self.dtype,
        )

        with importlib.resources.as_file(
            importlib.resources.files("pyxtal") / "database" / "wyckoff_list.csv"
        ) as wyckoff_csv:
            self.symmetry = Symmetry(str(wyckoff_csv))

        self.ref_row = torch.tensor(
            self.ref_xtal.get_tabular_representation(),
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(0)
        (
            self.spg_batch,
            self.wps_batch,
            rep_ref_batch,
        ) = self.symmetry.get_batch_from_rows(
            self.ref_row.clone(),
            radian=True,
            normalize_in=False,
            normalize_out=True,
            max_rep=30,
        )
        self.spg_batch = self.spg_batch.to(self.device)
        self.wps_batch = self.wps_batch.to(self.device)
        self.rep_template = rep_ref_batch[0].to(self.device)
        self.valid_mask = self.rep_template >= 0
        self.wps_list = [int(v.item()) for v in self.wps_batch[:, 0]]

        self.ref_rep = self._normalize_1d_rep(self.ref_xtal.get_1d_rep_x())
        self.ref_rep = torch.tensor(self.ref_rep, dtype=self.dtype, device=self.device)
        if self.ref_rep.shape[0] != int(self.valid_mask.sum().item()):
            raise RuntimeError(
                "Compact batch_sym representation does not match pyxtal 1D representation length."
            )

        # Sanity-check that the compact torch representation matches pyxtal's 1D rep.
        ref_batch_compact = self.rep_template[self.valid_mask]
        max_diff = torch.max(torch.abs(ref_batch_compact - self.ref_rep)).item()
        if max_diff > 1e-6:
            raise RuntimeError(
                f"Compact representation ordering mismatch (max diff={max_diff:.3e})."
            )

        (
            _,
            _,
            _,
            _,
            _,
            self.generators_ref,
            self.g_map_ref,
            self.xyz_map_ref,
        ) = self.symmetry.get_tuple_from_batch(
            self.spg_batch,
            self.wps_batch,
            rep_ref_batch.clone().to(self.device),
            normalize=True,
        )
        self.generators_ref = self.generators_ref.to(self.device)
        self.g_map_ref = self.g_map_ref.to(self.device)
        self.xyz_map_ref = self.xyz_map_ref.to(self.device)

        ref_state = self._state_from_compact_rep(self.ref_rep)
        self.hkl_grid = self.recp.build_hkl_grid(ref_state["cell"]).to(self.device, dtype=self.dtype)

        self.p_ref = self._descriptor_from_state(ref_state, norm=False)
        self.rdf_ref = self.rdf_calculator.compute(ref_state["coords"])
        self.p_ref_numpy = to_numpy_1d(self.p_ref)
        self.rdf_ref_numpy = to_numpy_1d(self.rdf_ref)
        self.loss_history = []
        self.loss_p_history = []
        self.loss_rdf_history = []

        print(f"Reference descriptor size: {self.p_ref.numel()}")
        print(f"Reference RDF size: {self.rdf_ref.numel()}")

    def _cell_dofs(self):
        if self.spg <= 2:
            return 3, 3
        if self.spg <= 15:
            return 3, 1
        if self.spg <= 74:
            return 3, 0
        if self.spg <= 194:
            return 2, 0
        return 1, 0

    def _normalize_1d_rep(self, rep):
        rep = np.asarray(rep, dtype=np.float64).copy()
        n_abc, n_ang = self._cell_dofs()
        rep[:n_abc] /= self.lattice_norm_factor
        if n_ang > 0:
            rep[n_abc : n_abc + n_ang] /= self.angle_norm_factor
        return rep

    def _denormalize_1d_rep(self, rep):
        rep = np.asarray(rep, dtype=np.float64).copy()
        n_abc, n_ang = self._cell_dofs()
        rep[:n_abc] *= self.lattice_norm_factor
        if n_ang > 0:
            rep[n_abc : n_abc + n_ang] *= self.angle_norm_factor
        return rep

    def _compact_bounds(self):
        n_abc, n_ang = self._cell_dofs()
        total = int(self.valid_mask.sum().item())
        lower = torch.full((total,), self.wyckoff_bounds[0], dtype=self.dtype, device=self.device)
        upper = torch.full((total,), self.wyckoff_bounds[1], dtype=self.dtype, device=self.device)

        lat_lo = self.lattice_bounds[0] / self.lattice_norm_factor
        lat_hi = self.lattice_bounds[1] / self.lattice_norm_factor
        lower[:n_abc] = lat_lo
        upper[:n_abc] = lat_hi
        if n_ang > 0:
            ang_lo = self.angle_bounds[0] / self.angle_norm_factor
            ang_hi = self.angle_bounds[1] / self.angle_norm_factor
            lower[n_abc : n_abc + n_ang] = ang_lo
            upper[n_abc : n_abc + n_ang] = ang_hi
        return lower, upper

    def _build_full_rep(self, rep_compact):
        rep_full = self.rep_template.clone()
        rep_full[self.valid_mask] = rep_compact
        return rep_full.unsqueeze(0)

    def _state_from_compact_rep(self, rep_compact):
        rep_batch = self._build_full_rep(rep_compact)
        cell, coords, numbers, _, _ = self.symmetry.get_tuple_from_batch_opt(
            self.spg_batch,
            rep_batch,
            self.generators_ref,
            self.g_map_ref,
            self.xyz_map_ref,
            normalize=True,
        )
        valid = numbers[0] > 0
        return {
            "cell": cell[0],
            "coords": coords[0, valid],
            "numbers": numbers[0, valid],
        }

    def _descriptor_from_state(self, state, norm=False):
        q_xyz, intensities, _ = self.recp.build_reciprocal_from_structure_torch(
            state["cell"], state["coords"], state["numbers"], hkl=self.hkl_grid
        )
        return self.recp.compute_sph_descriptor_torch(q_xyz, intensities, norm=norm)

    def objective_torch(self, rep_compact):
        state = self._state_from_compact_rep(rep_compact)
        p_current = self._descriptor_from_state(state, norm=False)
        rdf_current = self.rdf_calculator.compute(state["coords"])

        loss_p = torch.linalg.vector_norm(p_current - self.p_ref) / math.sqrt(p_current.numel())
        loss_rdf = torch.linalg.vector_norm(rdf_current - self.rdf_ref) / math.sqrt(rdf_current.numel())
        loss = self.weight_p * loss_p + self.weight_rdf * loss_rdf
        return loss, loss_p, loss_rdf, state, p_current, rdf_current

    def optimize(self, rep0):
        rep0_t = torch.tensor(rep0, dtype=self.dtype, device=self.device)
        params = torch.nn.Parameter(rep0_t.clone())
        lower, upper = self._compact_bounds()

        optimizer = AdamW([params], lr=self.lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=max(self.steps // 20, 1), gamma=0.5)
        best = {"loss": float("inf"), "rep": None}

        print("\nStarting differentiable AdamW optimization...")
        for step in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            loss, loss_p, loss_rdf, *_ = self.objective_torch(params)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([params], max_norm=5.0)
            optimizer.step()
            with torch.no_grad():
                params.clamp_(lower, upper)
            scheduler.step()

            loss_value = float(loss.item())
            self.loss_history.append(loss_value)
            self.loss_p_history.append(float(loss_p.item()))
            self.loss_rdf_history.append(float(loss_rdf.item()))
            if loss_value < best["loss"]:
                best["loss"] = loss_value
                best["rep"] = params.detach().clone()

            if step == 0 or (step + 1) % 25 == 0:
                print(
                    f"  step {step+1:4d}/{self.steps} | "
                    f"loss={loss_value:.6e} | "
                    f"loss_p={loss_p.item():.6e} | "
                    f"loss_rdf={loss_rdf.item():.6e} | "
                    f"lr={scheduler.get_last_lr()[0]:.3e}"
                )

        final_rep = best["rep"] if best["rep"] is not None else params.detach()
        print(f"\nOptimization complete. Best loss: {best['loss']:.6e}")
        return final_rep.cpu().numpy()

    def perturb_structure(self, d_lat=0.5, d_coor=0.5):
        xtal_pert = deepcopy(self.ref_xtal)
        if hasattr(xtal_pert, "random_state") and hasattr(xtal_pert.lattice, "random_state"):
            xtal_pert.lattice.random_state = xtal_pert.random_state.spawn(1)[0]
        xtal_pert.apply_perturbation(d_lat=d_lat, d_coor=d_coor)
        rep0 = self._normalize_1d_rep(xtal_pert.get_1d_rep_x())
        return xtal_pert, rep0, d_lat, d_coor

    def get_optimized_structure(self, rep_opt):
        xtal_opt = deepcopy(self.ref_xtal)
        xtal_opt.update_from_1d_rep(self._denormalize_1d_rep(rep_opt))
        return xtal_opt

    def evaluate_numpy(self, rep):
        rep_t = torch.tensor(rep, dtype=self.dtype, device=self.device)
        _, _, _, state, p, rdf = self.objective_torch(rep_t)
        return state, to_numpy_1d(p), to_numpy_1d(rdf)


def normalize_excluding_first(p, clip=None, eps=1e-12):
    p = np.array(p, dtype=float).copy()
    if clip is not None:
        p = np.clip(p, -clip, clip)
    if p.size <= 1:
        return p
    scale = np.max(np.abs(p[1:])) + eps
    return p / scale


def main(prototype="diamond", d_lat=0.5, d_coor=0.5):
    set_global_seed(42)

    print("=" * 70)
    print(f"RECIPROCAL-SPACE INVERSE OPTIMIZATION ({prototype})")
    print("=" * 70)

    if prototype == "diamond":
        xtal_ref = pyxtal(random_state=42)
        xtal_ref.from_spg_wps_rep(166, ["6c"], [2.522, 6.178, 0.125])
    elif prototype == "h-diamond":
        hdia = pyxtal(random_state=42)
        hdia.from_prototype("h-diamond")
        xtal_ref = hdia.subgroup_once(H=193, eps=0)
    else:
        raise ValueError(f"Unsupported prototype: {prototype}")

    recp_params = {"dmax": 10.0, "nmax": 10, "lmax": 10, "rbasis": "bessel", "rcut": 2.1}
    optimizer = InverseOptimizer(
        xtal_ref,
        recp_params=recp_params,
        weight_descriptor=2.0,
        weight_rdf=1.0,
        lr=3e-2,
        steps=3000,
    )

    print("\n" + "-" * 70)
    print(f"PERTURBATION STEP (lat={int(d_lat*100)}%, coor={d_coor:.2f}Å)")
    print("-" * 70)
    xtal_pert, rep0, d_lat, d_coor = optimizer.perturb_structure(d_lat=d_lat, d_coor=d_coor)
    _, p_pert_numpy, rdf_pert_numpy = optimizer.evaluate_numpy(rep0)

    print("\n" + "-" * 70)
    print("OPTIMIZATION STEP (AdamW)")
    print("-" * 70)
    rep_opt = optimizer.optimize(rep0)
    xtal_opt = optimizer.get_optimized_structure(rep_opt)
    _, p_opt_numpy, rdf_opt_numpy = optimizer.evaluate_numpy(rep_opt)

    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)
    print("\nReference structure:")
    print(xtal_ref)
    print("\nOptimized structure:")
    print(xtal_opt)

    p_ref_plot = normalize_excluding_first(optimizer.p_ref_numpy)
    p_pert_plot = normalize_excluding_first(p_pert_numpy)
    p_opt_plot = normalize_excluding_first(p_opt_numpy)

    os.makedirs("cifs", exist_ok=True)
    os.makedirs("fig", exist_ok=True)
    name = prototype.replace("-", "_")
    pert_str = f"lat{int(d_lat*100):02d}_coor{int(d_coor*100):02d}"

    ref_file = f"cifs/{name}_reference.cif"
    pert_file = f"cifs/{name}_{pert_str}_perturbed.cif"
    opt_file = f"cifs/{name}_{pert_str}_optimized.cif"
    xtal_ref.to_file(ref_file)
    xtal_pert.to_file(pert_file)
    xtal_opt.to_file(opt_file)
    print(f"\nSaved:\n  {ref_file}\n  {pert_file}\n  {opt_file}")

    fig, axs = plt.subplots(2, 2, figsize=(7.5, 6.5))
    font_size_label = 12
    font_size_title = 11
    font_size_legend = 10
    font_size_tick = 9
    x_rdf = np.linspace(0, optimizer.rdf_calculator.rcut, optimizer.rdf_calculator.num_bins)

    axs[0, 0].plot(p_ref_plot, label="Reference", alpha=0.8, lw=1.0)
    axs[0, 0].plot(p_pert_plot, label="Perturbed", alpha=0.8, lw=1.0)
    axs[0, 0].set_yscale("symlog", linthresh=1e-1)
    axs[0, 0].set_ylabel("$P_{nl}$", fontsize=font_size_label)
    axs[0, 0].set_title("Power Spectrum", fontsize=font_size_title)
    axs[0, 0].legend(fontsize=font_size_legend, loc="upper left", frameon=True, fancybox=True)
    axs[0, 0].tick_params(labelsize=font_size_tick)

    axs[0, 1].plot(x_rdf, optimizer.rdf_ref_numpy, label="Reference", alpha=0.8, lw=1.0)
    axs[0, 1].plot(x_rdf, rdf_pert_numpy, label="Perturbed", alpha=0.8, lw=1.0, linestyle="--")
    axs[0, 1].set_ylabel("$G(r)$", fontsize=font_size_label)
    axs[0, 1].set_title("RDF", fontsize=font_size_title)
    axs[0, 1].legend(fontsize=font_size_legend, loc="upper left", frameon=True, fancybox=True)
    axs[0, 1].tick_params(labelsize=font_size_tick)

    axs[1, 0].plot(p_ref_plot, label="Reference", alpha=0.8, lw=1.0)
    axs[1, 0].plot(p_opt_plot, label="Optimized", alpha=0.8, lw=1.0)
    axs[1, 0].set_yscale("symlog", linthresh=1e-1)
    axs[1, 0].set_xlabel("Power Spectrum Index", fontsize=font_size_label)
    axs[1, 0].set_ylabel("$P_{nl}$", fontsize=font_size_label)
    axs[1, 0].legend(fontsize=font_size_legend, loc="upper left", frameon=True, fancybox=True)
    axs[1, 0].tick_params(labelsize=font_size_tick)

    axs[1, 1].plot(x_rdf, optimizer.rdf_ref_numpy, label="Reference", alpha=0.8, lw=1.0)
    axs[1, 1].plot(x_rdf, rdf_opt_numpy, label="Optimized", alpha=0.8, lw=1.0, linestyle="--")
    axs[1, 1].set_xlabel("r (Å)", fontsize=font_size_label)
    axs[1, 1].set_ylabel("$G(r)$", fontsize=font_size_label)
    axs[1, 1].legend(fontsize=font_size_legend, loc="upper left", frameon=True, fancybox=True)
    axs[1, 1].tick_params(labelsize=font_size_tick)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_file = f"fig/{name}_{pert_str}_reconstruction.png"
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"  {fig_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fully differentiable inverse optimization of crystal structure using reciprocal-space descriptors"
    )
    parser.add_argument("--prototype", type=str, default="h-diamond", help="Structure prototype")
    parser.add_argument("--d-lat", type=float, default=1, help="Relative lattice perturbation")
    parser.add_argument("--d-coor", type=float, default=2, help="Coordinate perturbation magnitude (Å)")
    args = parser.parse_args()

    main(prototype=args.prototype, d_lat=args.d_lat, d_coor=args.d_coor)
