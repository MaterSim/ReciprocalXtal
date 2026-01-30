"""
Inverse optimization for quartz structures: lattice and Wyckoff coordinate recovery.

Perturbs α-quartz and β-quartz structures, then optimizes lattice parameters and
Wyckoff coordinates to recover the reference reciprocal-space descriptor using
combined reciprocal-space (RECP) and real-space (pairwise RDF) descriptors.
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from scipy.optimize import minimize
from pyxtal import pyxtal
from reciprocal import RECP
from copy import deepcopy
from rdf_real import RDFCalculator
import random


def set_global_seed(seed: int):
    """Set seeds for numpy, python `random`, and torch (if available)."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass

class InverseOptimizer:
    """
    Optimize lattice and Wyckoff-site free variables to match a target descriptor.
    """

    def __init__(self, reference_xtal, recp_params=None, weight_descriptor=1.0, weight_rdf=2.0, pair_weights=None):
        """
        Args:
            reference_xtal: pyxtal object (reference structure)
            recp_params: dict of RECP parameters (dmax, nmax, lmax, rbasis)
            weight_descriptor: weight for descriptor loss in combined objective
            weight_rdf: weight for RDF loss in combined objective
        """
        self.ref_xtal = deepcopy(reference_xtal)
        self.spg = reference_xtal.group.number
        
        # Normalization factors and bounds for lattice parameters (configurable)
        self.lattice_norm_factor = 7.0  
        self.angle_norm_factor = 180.0  # for angles
        
        # Bounds in unnormalized space
        self.lattice_bounds = (3.0, 7.0)  # Angstroms
        self.angle_bounds = (30.0, 150.0)  # degrees
        self.wyckoff_bounds = (0.0, 1.0)  # fractional coordinates
        
        self.recp = RECP(dmax=recp_params['dmax'],nmax=recp_params['nmax'],lmax=recp_params['lmax'],rbasis=recp_params['rbasis'])
        self.rdf_calc = RDFCalculator(rcut=recp_params['rcut'])
        
        # Compute reference descriptor and RDF
        coords, vals, ds = self.recp.build_reciprocal(self.ref_xtal.to_ase())
        self.p_ref = self.recp.compute_sph_torch(coords, vals, norm=False)
        
        # Compute reference real-space RDF (pairwise)
        self.rdf_ref, self.rdf_labels = self.rdf_calc.compute_rdf(
            self.ref_xtal.to_ase(), mode='pairwise', num_bins=50, sigma=2.0
        )
        print(f"Pair RDF labels: {self.rdf_labels}")
        
        self.p_ref_numpy = self.p_ref.detach().numpy()
        self.rdf_ref_numpy = self.rdf_ref
        
        self.p_size = len(self.p_ref_numpy)
        self.rdf_size = self.rdf_ref_numpy.shape[0]
        
        print(f"Reference descriptor size: {self.p_size}")
        print(f"Reference RDF size: {self.rdf_size}")
        print(f"  (Real-space RDF shape: {self.rdf_ref_numpy.shape} = {self.rdf_ref_numpy.shape[0]} pairs × {self.rdf_ref_numpy.shape[1]} bins)")
        
        self.weight_p = weight_descriptor
        self.weight_rdf = weight_rdf
        self.pair_weights = pair_weights if pair_weights is not None else {}
        
        self.loss_history = []
        self.n_calls = 0
        
        print(f"Reference descriptor shape: {self.p_ref_numpy.shape}")
        print(f"Reference RDF shape: {self.rdf_ref_numpy.shape}")
        
    def _get_lattice_dofs(self):
        from pyxtal.lattice import Lattice
        return Lattice.get_dofs(self.ref_xtal.lattice.ltype)

    def _normalize_rep(self, rep):
        """Normalize 1D representation by dividing lattice and angle components."""
        repn = rep.copy()
        N_abc, N_ang = self._get_lattice_dofs()
        if len(repn) >= N_abc:
            repn[:N_abc] = repn[:N_abc] / self.lattice_norm_factor
        if len(repn) >= N_abc + N_ang:
            repn[N_abc:N_abc+N_ang] = repn[N_abc:N_abc+N_ang] / self.angle_norm_factor
        return repn

    def _denormalize_rep(self, rep):
        """Denormalize 1D representation by multiplying lattice and angle components."""
        repd = rep.copy()
        N_abc, N_ang = self._get_lattice_dofs()
        if len(repd) >= N_abc:
            repd[:N_abc] = repd[:N_abc] * self.lattice_norm_factor
        if len(repd) >= N_abc + N_ang:
            repd[N_abc:N_abc+N_ang] = repd[N_abc:N_abc+N_ang] * self.angle_norm_factor
        return repd
    
    def objective(self, rep):
        """
        Objective function: combined descriptor + RDF loss.
        
        Args:
            rep: 1D representation of free variables from pyxtal.get_1d_rep_x()
        
        Returns:
            loss: scalar (weighted sum of descriptor and RDF losses)
        """
        xtal_current = deepcopy(self.ref_xtal)
        rep_denorm = self._denormalize_rep(rep)
        xtal_current.update_from_1d_rep(rep_denorm)
        
        coords_cur, vals_cur, ds_cur = self.recp.build_reciprocal(xtal_current.to_ase())
        p_current = self.recp.compute_sph_torch(coords_cur, vals_cur, norm=False)
        p_current_numpy = p_current.detach().numpy()
        
        loss_p = np.linalg.norm(p_current_numpy - self.p_ref_numpy) / np.sqrt(len(p_current_numpy))
        
        rdf_current, rdf_current_labels = self.rdf_calc.compute_rdf(
            xtal_current.to_ase(), mode='pairwise', num_bins=50, sigma=2.0
        )
        
        pair_losses = []
        n_pairs = rdf_current.shape[0]
        for pair_idx in range(n_pairs):
            rdf_ref_elem = self.rdf_ref_numpy[pair_idx, :]
            rdf_cur_elem = rdf_current[pair_idx, :]
            elem_l2 = np.linalg.norm(rdf_cur_elem - rdf_ref_elem) / np.sqrt(len(rdf_ref_elem))
            label = rdf_current_labels[pair_idx] if pair_idx < len(rdf_current_labels) else None
            multiplier = float(self.pair_weights.get(label, 1.0)) if label else 1.0
            pair_losses.append(multiplier * elem_l2)
        
        loss_rdf = np.sum(pair_losses)
        loss = self.weight_p * loss_p + self.weight_rdf * loss_rdf
        
        if self.n_calls % 10 == 1:
            print(f"Iteration {self.n_calls}: loss={loss:.6e}, loss_p={loss_p:.6e}, sum_pair_L2={loss_rdf:.6e}")
        
        self.loss_history.append(loss)
        self.n_calls += 1
        
        if not hasattr(self, 'loss_p_history'):
            self.loss_p_history = []
            self.loss_rdf_history = []
        self.loss_p_history.append(loss_p)
        self.loss_rdf_history.append(loss_rdf)
        
        return loss
    
    def optimize(self, rep0):
        """
        Sequential optimization with bounds: Nelder-Mead -> L-BFGS-B stages
        
        Args:
            rep0: initial 1D representation (perturbed)
        
        Returns:
            result: final OptimizeResult object
        """
        from pyxtal.lattice import Lattice
        
        # Get lattice DOFs
        [N_abc, N_ang] = Lattice.get_dofs(self.ref_xtal.lattice.ltype)
        
        # Compute normalized bounds from class attributes
        lattice_bounds_norm = (self.lattice_bounds[0] / self.lattice_norm_factor, 
                              self.lattice_bounds[1] / self.lattice_norm_factor)
        angle_bounds_norm = (self.angle_bounds[0] / self.angle_norm_factor,
                            self.angle_bounds[1] / self.angle_norm_factor)
        
        # Set bounds in normalized space
        bounds = [lattice_bounds_norm] * N_abc + [angle_bounds_norm] * N_ang
        bounds += [self.wyckoff_bounds] * (len(rep0) - N_abc - N_ang)

        # Ensure initial values within normalized bounds
        rep = rep0.copy()
        print(f"\nInitial representation {rep}")  # normalized values
        for i in range(N_abc):
            rep[i] = np.clip(rep[i], lattice_bounds_norm[0], lattice_bounds_norm[1])
        for i in range(N_abc, N_abc + N_ang):
            rep[i] = np.clip(rep[i], angle_bounds_norm[0], angle_bounds_norm[1])
        for i in range(N_abc + N_ang, len(rep)):
            rep[i] = np.clip(rep[i], self.wyckoff_bounds[0], self.wyckoff_bounds[1])
        
        print(f"\nStarting optimization...")
        print(f"Initial loss: {self.objective(rep):.6e}")
        
        # Stage 1: Nelder-Mead (300)
        print("--- Stage 1: Nelder-Mead (300) ---")
        result = minimize(self.objective, rep, method='Nelder-Mead',
             bounds=bounds, options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
        rep = result.x
        print(f"  Loss: {result.fun:.6e}, Status: {result.message}")
                
        # Stage 2: L-BFGS-B (300) with tighter tolerances
        print("--- Stage 2: L-BFGS-B (300) ---")
        result = minimize(self.objective, rep, method='L-BFGS-B',
                 bounds=bounds, options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        rep = result.x
        print(f"  Loss: {result.fun:.6e}, Status: {result.message}")
        
        # Stage 3: L-BFGS-B (100)
        print("--- Stage 3: L-BFGS-B (200) ---")
        result = minimize(self.objective, rep, method='L-BFGS-B',
                 bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        rep = result.x
        print(f"  Loss: {result.fun:.6e}, Status: {result.message}")
        
        # Stage 4: L-BFGS-B (100)
        print("--- Stage 4: L-BFGS-B (100) ---")
        result = minimize(self.objective, rep, method='L-BFGS-B',
                 bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        # Show loss component evolution
        if hasattr(self, 'loss_p_history'):
            print(f"\n  Loss components (normalized):")
            print(f"    Descriptor (P): {self.loss_p_history[0]:.6e} → {self.loss_p_history[-1]:.6e} (ratio: {self.loss_p_history[0]/(self.loss_p_history[-1]+1e-12):.1f}x)")
            print(f"    RDF: {self.loss_rdf_history[0]:.6e} → {self.loss_rdf_history[-1]:.6e} (ratio: {self.loss_rdf_history[0]/(self.loss_rdf_history[-1]+1e-12):.1f}x)")
            print(f"    Min RDF loss reached: {min(self.loss_rdf_history):.6e} at iteration {np.argmin(self.loss_rdf_history)}")
            print(f"    Min P loss reached: {min(self.loss_p_history):.6e} at iteration {np.argmin(self.loss_p_history)}")
        
        
        print(f"\nOptimization complete.")
        print(f"  Final loss: {result.fun:.6e}")
        print(f"  Final status: {result.message}")
        print(f"  Success: {result.success}")
        print(f"  Total function calls: {self.n_calls}")
        
        return result
    
    def perturb_structure(self, d_lat=0.1, d_coor=0.9):
        """
        Create a perturbed 1D representation.
        
        Args:
            d_lat: lattice perturbation magnitude (relative)
            d_coor: coordinate perturbation magnitude (Å)
        
        Returns:
            xtal_perturbed: pyxtal object with perturbed structure
            rep0: perturbed 1D representation
            d_lat: lattice perturbation used
            d_coor: coordinate perturbation used
        """
        xtal_pert = deepcopy(self.ref_xtal)
        if hasattr(xtal_pert, 'random_state') and hasattr(xtal_pert.lattice, 'random_state'):
            xtal_pert.lattice.random_state = xtal_pert.random_state.spawn(1)[0]
        xtal_pert.apply_perturbation(d_lat=d_lat, d_coor=d_coor)
        #xtal_pert.subgroup_once(H=154, eps=perturbation)
        # return normalized rep for optimization
        #xtal_pert = pyxtal()
        #xtal_pert.from_seed('final_21jan/a_quartz_pert200_perturbed.cif')
        rep0= self._normalize_rep(xtal_pert.get_1d_rep_x())
        
        print(f"Perturbed structure: {xtal_pert}")
        p_pert, rdf_pert = self.recp.compute(xtal_pert.to_ase(), norm=False)
        print(f"Perturbed descriptor shape: {p_pert.shape}")
        
        return xtal_pert, rep0, d_lat, d_coor
    
    def get_optimized_structure(self, rep_opt):
        """
        Reconstruct the optimized structure from 1D representation.
        """
        xtal_opt = deepcopy(self.ref_xtal)
        xtal_opt.update_from_1d_rep(self._denormalize_rep(rep_opt))
        return xtal_opt


def main(prototype='a-quartz', d_lat=0.1, d_coor=0.9):
    """Inverse optimization workflow: reference → perturb → optimize → compare."""
    set_global_seed(42)
    
    print("=" * 70)
    print(f"RECIPROCAL-SPACE INVERSE OPTIMIZATION ({prototype})")
    print("=" * 70)
    
    xtal_ref = pyxtal(random_state=42)
    xtal_ref.from_prototype(prototype)
    print(f"\nReference structure: {xtal_ref}")
    
    #print(xtal_ref);exit()
    #xtal_ref.from_seed('cifs/a_quartz_pert30_perturbed.cif')
    
    # Initialize optimizer
    recp_params = {'dmax': 10, 'nmax': 10, 'lmax': 10, 'rbasis': 'bessel', 'rcut': 2.1}
    optimizer = InverseOptimizer(
        xtal_ref,
        recp_params=recp_params,
        weight_descriptor=2.0,
        weight_rdf=0.1,
        pair_weights={'Si-Si': 2.0, 'Si-O': 1.0, 'O-O': 2.0, 'O-Si': 1.0}
    )
    
    # Perturb and get initial point
    print("\n" + "-" * 70)
    print(f"PERTURBATION STEP (lat={int(d_lat*100)}%, coor={d_coor:.2f}Å)")
    print("-" * 70)
    xtal_pert, rep0, d_lat, d_coor = optimizer.perturb_structure(d_lat=d_lat, d_coor=d_coor)
    
    # Compute perturbed descriptor and RDF
    p_pert, _ = optimizer.recp.compute(xtal_pert.to_ase(), norm=False)
    #print(f"Perturbed descriptor shape: {p_pert}");exit()
    p_pert_numpy = p_pert.detach().numpy()
    
    # Compute reference and perturbed RDFs for plotting
    rdf_ref_real, rdf_labels = optimizer.rdf_calc.compute_rdf(
        xtal_ref.to_ase(), mode='pairwise', num_bins=50, sigma=2.0
    )
    rdf_pert_real, _ = optimizer.rdf_calc.compute_rdf(
        xtal_pert.to_ase(), mode='pairwise', num_bins=50, sigma=2.0
    )
    
    # Run optimization
    print("\n" + "-" * 70)
    print("OPTIMIZATION STEP")
    print("-" * 70)
    result = optimizer.optimize(rep0)
    result = optimizer.optimize(result.x)  # refine further
    # Get optimized structure
    xtal_opt = optimizer.get_optimized_structure(result.x)
    
    
    # Compute final descriptor and RDF
    p_opt, _ = optimizer.recp.compute(xtal_opt.to_ase(), norm=False)
    rdf_opt_real, _ = optimizer.rdf_calc.compute_rdf(
        xtal_opt.to_ase(), mode='pairwise', num_bins=50, sigma=2.0
    )
    
    # Compare
    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)
    print(f"\nReference structure:")
    print(xtal_ref)
    print(f"\nOptimized structure:")
    print(xtal_opt)
    
    p_ref_numpy = optimizer.p_ref_numpy
    p_opt_numpy = p_opt.detach().numpy()

    # ---- Option A (visualization): normalize by max(|p[1:]|) so the tail is visible ----
    def normalize_excluding_first(p, clip=None, eps=1e-12):
        p = np.array(p, dtype=float).copy()
        if clip is not None:
            p = np.clip(p, -clip, clip)
        if p.size <= 1:
            return p
        scale = np.max(np.abs(p[1:])) + eps
        return p / scale

    # Use these *only for plotting* (do not change the raw descriptors used in the loss)
    p_ref_plot  = normalize_excluding_first(p_ref_numpy,  clip=None)
    p_pert_plot = normalize_excluding_first(p_pert_numpy, clip=None)
    p_opt_plot  = normalize_excluding_first(p_opt_numpy,  clip=None)

    # Optionally report the dominant term for context
    print(f"P[0] raw (ref/pert/opt): {p_ref_numpy[0]:.3e} / {p_pert_numpy[0]:.3e} / {p_opt_numpy[0]:.3e}")
    
    # Save structures
    print(f"\nSaving structures...")
    import os
    os.makedirs('cifs', exist_ok=True)
    os.makedirs('fig', exist_ok=True)
    
    name = prototype.replace('-', '_')
    pert_str = f"lat{int(d_lat*100):02d}_coor{int(d_coor*100):02d}"
    
    ref_file = f'cifs/{name}_reference.cif'
    pert_file = f'cifs/{name}_{pert_str}_perturbed.cif'
    opt_file = f'cifs/{name}_{pert_str}_optimized.cif'
    
    xtal_ref.to_file(ref_file)
    xtal_pert.to_file(pert_file)
    xtal_opt.to_file(opt_file)
    print(f"  {ref_file}")
    print(f"  {pert_file}")
    print(f"  {opt_file}")
    
    # Plot comparison: upper (ref vs perturbed), lower (ref vs optimized)
    import matplotlib.pyplot as plt
    
    # Manuscript-quality figure (half-page width ~3.5 inches, full height)
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 6.5))
    
    # Font sizes for journal manuscript (half-page figure)
    font_size_label = 12
    font_size_title = 11
    font_size_legend = 10
    font_size_tick = 9
    
    # Create x-axis for Real RDFs
    x_rdf_real = np.linspace(0, optimizer.rdf_calc.rcut, rdf_ref_real.shape[1])
    
    # Upper row: Reference vs Perturbed
    # Column 0: Power Spectrum
    axs[0, 0].plot(p_ref_plot, label='Reference', alpha=0.8, lw=1.0)
    axs[0, 0].plot(p_pert_plot, label='Perturbed', alpha=0.8, lw=1.0)
    # Use a symmetric log scale to reveal small components while keeping the dominant first term
    axs[0, 0].set_yscale('symlog', linthresh=1e-1)
    axs[0, 0].set_ylabel('$P_{nl}$', fontsize=font_size_label)
    axs[0, 0].set_title('Power Spectrum', fontsize=font_size_title)
    axs[0, 0].legend(fontsize=font_size_legend, loc='upper left', frameon=True, fancybox=True)
    axs[0, 0].tick_params(labelsize=font_size_tick)
    #axs[0, 0].grid(True, alpha=0.3, linestyle='--')

    # Column 1: RDF (pair-wise)
    colors_ref = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors_pert = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']
    for pair_idx, label in enumerate(rdf_labels):
        if pair_idx < rdf_ref_real.shape[0]:
            c_ref = colors_ref[pair_idx % len(colors_ref)]
            c_pert = colors_pert[pair_idx % len(colors_pert)]
            axs[0, 1].plot(x_rdf_real, rdf_ref_real[pair_idx], 
                    label=f'{label} Ref', alpha=0.8, lw=1.0, color=c_ref)
            axs[0, 1].plot(x_rdf_real, rdf_pert_real[pair_idx], 
                    label=f'{label} Pert', alpha=0.8, lw=1.0, linestyle='--', color=c_pert)
            axs[0, 1].set_ylabel('G(r)', fontsize=font_size_label)
    #axs[0, 1].set_xlabel('r (Å)', fontsize=font_size_label, fontweight='bold')
    axs[0, 1].set_title('RDF', fontsize=font_size_title)
    axs[0, 1].legend(fontsize=font_size_legend-1, loc='upper left', frameon=True, fancybox=True, ncol=2)
    axs[0, 1].tick_params(labelsize=font_size_tick)
    #axs[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Lower row: Reference vs Optimized
    # Column 0: Power Spectrum
    axs[1, 0].plot(p_ref_plot, label='Reference', alpha=0.8, lw=1.0)
    axs[1, 0].plot(p_opt_plot, label='Optimized', alpha=0.8, lw=1.0)
    # Use a symmetric log scale to reveal small components while keeping the dominant first term
    axs[1, 0].set_yscale('symlog', linthresh=1e-1)
    axs[1, 0].set_xlabel('Power Spectrum Index', fontsize=font_size_label)
    axs[1, 0].set_ylabel('$P_{nl}$', fontsize=font_size_label)
    #axs[1, 0].set_title('(c) Power Spectrum: Ref vs Opt', fontsize=font_size_title, fontweight='bold', loc='left')
    axs[1, 0].legend(fontsize=font_size_legend, loc='upper left', frameon=True, fancybox=True)
    axs[1, 0].tick_params(labelsize=font_size_tick)
    #axs[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Column 1: RDF (pair-wise)
    for pair_idx, label in enumerate(rdf_labels):
        if pair_idx < rdf_opt_real.shape[0]:
            c_ref = colors_ref[pair_idx % len(colors_ref)]
            c_opt = colors_pert[pair_idx % len(colors_pert)]
            axs[1, 1].plot(x_rdf_real, rdf_ref_real[pair_idx], 
                    label=f'{label} Ref', alpha=0.8, lw=1.0, color=c_ref)
            axs[1, 1].plot(x_rdf_real, rdf_opt_real[pair_idx], 
                    label=f'{label} Opt', alpha=0.8, lw=1.0, linestyle='--', color=c_opt)
            axs[1, 1].set_xlabel('r (Å)', fontsize=font_size_label)
            axs[1, 1].set_ylabel('G(r)', fontsize=font_size_label)
    #axs[1, 1].set_title('(d) RDF: Ref vs Opt', fontsize=font_size_title, fontweight='bold', loc='left')
    #axs[1, 1].legend(fontsize=font_size_legend-1, loc='upper right', frameon=True, fancybox=True, ncol=2)
    axs[1, 1].tick_params(labelsize=font_size_tick)
    #axs[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Add overall figure title
    #fig.suptitle(f'{prototype.upper()}', 
    #             fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_file = f'fig/{name}_{pert_str}.png'
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"  {fig_file}")


if __name__ == '__main__':
    set_global_seed(42)
    
    import argparse
    parser = argparse.ArgumentParser(description='Inverse Optimization of Crystal Structure using RECIPROCAL-space Descriptors')
    parser.add_argument('--prototype', type=str, default='a-quartz', help='Structure prototype (default: a-quartz)')
    parser.add_argument('--d-lat', type=float, default=0.1, help='Lattice perturbation magnitude (relative)')
    parser.add_argument('--d-coor', type=float, default=0.9, help='Coordinate perturbation magnitude (Å)')
    args = parser.parse_args()

    main(prototype=args.prototype, d_lat=args.d_lat, d_coor=args.d_coor)
