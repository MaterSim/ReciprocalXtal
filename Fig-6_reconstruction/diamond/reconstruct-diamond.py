from pathlib import Path
import sys

# Allow importing project-level reciprocal.py when running as a script
sys.path.append(str(Path(__file__).resolve().parents[2]))


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

    def __init__(self, reference_xtal, recp_params=None, weight_descriptor=1.0, weight_rdf=2.0):
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
        self.lattice_bounds = (2.0, 7.0)  # Angstroms
        self.angle_bounds = (30.0, 150.0)  # degrees
        self.wyckoff_bounds = (0.0, 1.0)  # fractional coordinates
        
        self.recp = RECP(dmax=recp_params['dmax'],nmax=recp_params['nmax'],lmax=recp_params['lmax'],rbasis=recp_params['rbasis'])
        self.rdf_calculator = RDFCalculator(rcut=recp_params['rcut'])
        
        # Compute reference descriptor and RDF (normalized)
        coords, vals, ds = self.recp.build_reciprocal(self.ref_xtal.to_ase())
        
        self.p_ref = self.recp.compute_sph_torch(coords, vals, norm=False)
        #pp=self.p_ref[self.p_ref >1e-4]
        #print(f"pp : {pp} ")
        #print(f"Reference descriptor shape: {self.p_ref}");exit()
        
        # Compute real-space RDF via RDFCalculator
        self.rdf_ref, self.rdf_labels = self.rdf_calculator.compute_rdf(self.ref_xtal.to_ase(), mode='mean')
        if self.rdf_labels is not None:
            print(f"Pair RDF labels: {self.rdf_labels}")
        
        self.p_ref_numpy = self.p_ref.detach().numpy()
        self.rdf_ref_numpy = self.rdf_ref
        
        # Store sizes for MAE calculation
        self.p_size = len(self.p_ref_numpy)
        if self.rdf_ref_numpy.ndim == 2:
            self.rdf_size = self.rdf_ref_numpy.shape[0]  # num_pairs for pairwise RDF
        else:
            self.rdf_size = len(self.rdf_ref_numpy)  # num_bins for mean RDF
        
        print(f"Reference descriptor size: {self.p_size}")
        print(f"Reference RDF size: {self.rdf_size}")
        if self.rdf_ref_numpy.ndim == 2:
            print(f"  (Real-space RDF shape: {self.rdf_ref_numpy.shape} = {self.rdf_ref_numpy.shape[0]} pairs × {self.rdf_ref_numpy.shape[1]} bins)")
        else:
            print(f"  (Real-space RDF shape: {self.rdf_ref_numpy.shape} = {len(self.rdf_ref_numpy)} bins)")
        
        self.weight_p = weight_descriptor
        self.weight_rdf = weight_rdf
        # per-pair multipliers (e.g. {'Si-Si': 2.0, 'O-O': 2.0})
        
        # Storage for optimization
        self.loss_history = []
        self.n_calls = 0
        
        #print(f"Reference structure: {self.ref_xtal}")
        print(f"Reference descriptor shape: {self.p_ref_numpy.shape}")
        print(f"Reference RDF shape: {self.rdf_ref_numpy.shape}")
        
    def _get_lattice_dofs(self):
        from pyxtal.lattice import Lattice
        return Lattice.get_dofs(self.ref_xtal.lattice.ltype)

    def _normalize_rep(self, rep):
        repn = np.array(rep).copy()  # Convert to numpy array if it's a list
        N_abc, N_ang = self._get_lattice_dofs()
        if len(repn) >= N_abc:
            repn[:N_abc] = repn[:N_abc] / self.lattice_norm_factor
        if len(repn) >= N_abc + N_ang:
            repn[N_abc:N_abc+N_ang] = repn[N_abc:N_abc+N_ang] / self.angle_norm_factor
        return repn

    def _denormalize_rep(self, rep):
        repd = np.array(rep).copy()  # Convert to numpy array if it's a list
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
        # Create a copy and update the 1D representation (denormalize first)
        xtal_current = deepcopy(self.ref_xtal)
        rep_denorm = self._denormalize_rep(rep)
        xtal_current.update_from_1d_rep(rep_denorm)
        
        # Compute descriptor P
        coords_cur, vals_cur, ds_cur = self.recp.build_reciprocal(xtal_current.to_ase())
        p_current = self.recp.compute_sph_torch(coords_cur, vals_cur, norm=False)
        p_current_numpy = p_current.detach().numpy()
        
        # Descriptor loss (normalized L2 norm / RMSE)
        loss_p = np.linalg.norm(p_current_numpy - self.p_ref_numpy) / np.sqrt(len(p_current_numpy))
        
        # RDF loss - real-space, pair-wise RDF via RDFCalculator
        loss_rdf = 0.0
        rdf_current, _ = self.rdf_calculator.compute_rdf(xtal_current.to_ase(), mode='mean')
        #print(f"rdf_current shape: {rdf_current.shape} ");exit()
        if hasattr(rdf_current, 'shape') and rdf_current.ndim == 2:
            n_pairs = rdf_current.shape[0]
            for pair_idx in range(n_pairs):
                rdf_ref_elem = self.rdf_ref_numpy[pair_idx, :]
                rdf_cur_elem = rdf_current[pair_idx, :]
                # L2 norm normalized by the reference integral (better for peak shifts)
                ref_integral = np.sum(rdf_ref_elem)
                if ref_integral > 0:
                    elem_l2 = np.linalg.norm(rdf_cur_elem - rdf_ref_elem) / ref_integral
                else:
                    elem_l2 = np.linalg.norm(rdf_cur_elem - rdf_ref_elem)
                loss_rdf += elem_l2
        else:
            # fallback: treat RDF as 1D vector
            #print(f"rdf_current shape: {rdf_current} ")
            #print(f"rdf_ref shape: {self.rdf_ref_numpy} "); exit()
            loss_rdf = np.linalg.norm(rdf_current - self.rdf_ref_numpy)

        # Combined loss (MAE - both terms already normalized)
        loss = self.weight_p * loss_p + self.weight_rdf * loss_rdf
        # Debug: print per-element losses occasionally
        if self.n_calls % 10 == 1:
            print(f"Iteration {self.n_calls}: loss={loss:.6e}, loss_p={loss_p:.6e}, sum_pair_L2={loss_rdf:.6e} \n rep={rep} ")
        
        self.loss_history.append(loss)
        self.n_calls += 1
        
        #if self.n_calls % 10 == 0 or self.n_calls == 1:
        #    print(f"  Iteration {self.n_calls}: loss={loss:.6e}, loss_p={loss_p:.6e}, loss_rdf={loss_rdf:.6e}")
        
        # Store individual losses for analysis
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
    ''' 
    def perturb_structure(self, d_lat, d_coor):
        """
        Generate perturbed structure with lattice and coordinate noise.
        
        Args:
            d_lat: relative lattice perturbation magnitude
            d_coor: Cartesian coordinate displacement magnitude (Å)
        
        Returns:
            xtal_perturbed: perturbed pyxtal object
            rep0: normalized 1D representation for optimization
        """
        xtal_pert = deepcopy(self.ref_xtal)
        if hasattr(xtal_pert, 'random_state') and hasattr(xtal_pert.lattice, 'random_state'):
            xtal_pert.lattice.random_state = xtal_pert.random_state.spawn(1)[0]
        
        xtal_pert.apply_perturbation(d_lat=d_lat, d_coor=d_coor)
        rep0 = self._normalize_rep(xtal_pert.get_1d_rep_x())
        
        return xtal_pert, rep0
    '''
    def perturb_structure(self,prototype='diamond'):
        """
        Create a perturbed 1D representation.
        
        Args:
            perturbation: relative perturbation (0.05 = 5%)
        
        Returns:
            xtal_perturbed: pyxtal object with perturbed structure
            rep0: perturbed 1D representation
        """
        # Get reference 1D representation
        
        print(f"ref xtal before perturbation: {self.ref_xtal}")
        xtal_pert = pyxtal()
        if prototype=='diamond':
            xtal_pert.from_seed('perturbed/diamond_perturbed.cif')
        elif prototype=='h-diamond':
            xtal_pert.from_seed('perturbed/h_diamond_perturbed.cif')
        
        print(f"Perturbed structure: {xtal_pert}") 
        
        # return normalized rep for optimization
        rep0= self._normalize_rep(xtal_pert.get_1d_rep_x())
        
        print(f"Perturbed structure: {xtal_pert}") ##;exit()
        
        return xtal_pert, rep0
    
    def get_optimized_structure(self, rep_opt):
        """
        Reconstruct the optimized structure from 1D representation.
        """
        xtal_opt = deepcopy(self.ref_xtal)
        xtal_opt.update_from_1d_rep(self._denormalize_rep(rep_opt))
        return xtal_opt


def main(prototype='diamond'):
    # fix all random seeds for reproducibility
    set_global_seed(42)
    """
    Main workflow: create reference → perturb → optimize → compare.
    
    Args:
        prototype: structure prototype name (e.g., 'diamond', 'a-quartz', 'graphite')
    """
    # Create reference structure
    print("=" * 70)
    print(f"RECIPROCAL-SPACE INVERSE OPTIMIZATION ({prototype})")
    print("=" * 70)
    
    if prototype == 'diamond':
        xtal_ref = pyxtal(random_state=42)
        xtal_ref.from_spg_wps_rep(166,['6c'],[2.522, 6.178, 0.125])
        #xtal_ref.from_prototype('diamond')
       #print(f"\nReference structure: {xtal_ref}");exit()
    elif prototype == 'h-diamond':
        hdia = pyxtal()
        hdia.from_prototype('h-diamond')
        xtal_ref=hdia.subgroup_once(H=193, eps=0)
        #print(f"\nReference structure: {xtal_ref}");exit()
        #xtal_ref.from_spg_wps_rep(194,['4b'],[2.522, 4.119, 0.0])
    else:
        exit()
        
    
    # Initialize optimizer
    recp_params = {'dmax': 10, 'nmax': 10, 'lmax': 10, 'rbasis': 'bessel', 'rcut': 2.1, 'alpha': 2.0}
    optimizer = InverseOptimizer(
        xtal_ref,
        recp_params=recp_params,
        weight_descriptor=1.0,
        weight_rdf=1
    )
    
    # Perturb and get initial point
    print("\n" + "-" * 70)
    print("-" * 70)
    xtal_pert, rep0 = optimizer.perturb_structure(prototype=prototype)
    
    # Compute perturbed descriptor (reciprocal) and both RDF types
    p_pert, _ = optimizer.recp.compute(xtal_pert.to_ase(), norm=False)
    
    p_pert_numpy = p_pert.detach().numpy()
    
    # Compute reference RDF (real-space)
    rdf_ref_real_full, rdf_labels = optimizer.rdf_calculator.compute_rdf(xtal_ref.to_ase(), mode='mean')
    # If labels not available, create default labels based on number of RDF components
    if rdf_labels is None and hasattr(rdf_ref_real_full, 'shape'):
        rdf_labels = [f'Pair {i}' for i in range(rdf_ref_real_full.shape[0])]
    
    
    # Compute perturbed RDF (real-space)
    rdf_pert_real_full, _ = optimizer.rdf_calculator.compute_rdf(xtal_pert.to_ase(), mode='mean')
    
    # Run optimization
    print("\n" + "-" * 70)
    print("OPTIMIZATION STEP")
    print("-" * 70)
    result = optimizer.optimize(rep0)
    result = optimizer.optimize(result.x)  # extra run for good measure
    
    # Get optimized structure
    xtal_opt = optimizer.get_optimized_structure(result.x)
    
    
    # Compute final descriptor (reciprocal) and both RDF types
    p_opt, _ = optimizer.recp.compute(xtal_opt.to_ase(), norm=False)
    
    # Compute optimized RDF (real-space)
    rdf_opt_real_full, _ = optimizer.rdf_calculator.compute_rdf(xtal_opt.to_ase(), mode='mean')
    
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
    
    # Save structures
    print(f"\nSaving structures...")
    import os
    os.makedirs('cifs', exist_ok=True)
    os.makedirs('fig', exist_ok=True)
    
    name = prototype.replace('-', '_')
    
    ref_file = f'cifs/{name}_reference.cif'
    pert_file = f'cifs/{name}_perturbed.cif'
    opt_file = f'cifs/{name}_reconstructed.cif'
    
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
    font_size_title = 12
    font_size_legend = 10
    font_size_tick = 9
    
    # Create x-axis for real-space RDF
    if rdf_ref_real_full.ndim == 2:
        num_bins = rdf_ref_real_full.shape[1]
    else:
        num_bins = len(rdf_ref_real_full)
    x_rdf_real = np.linspace(0, optimizer.rdf_calculator.rcut, num_bins)
    
    # Upper row: Reference vs Perturbed
    # Column 0: Power Spectrum
    axs[0, 0].plot(p_ref_numpy, label='Reference', alpha=0.8, lw=1.0)
    axs[0, 0].plot(p_pert_numpy, label='Perturbed', alpha=0.8, lw=1.0)
    axs[0, 0].set_ylabel('$P_{nl}$', fontsize=font_size_label)
    axs[0, 0].set_title('(a) Power Spectrum', fontsize=font_size_title, fontweight='bold')
    axs[0, 0].legend(fontsize=font_size_legend, loc='upper left', frameon=True, fancybox=True)
    axs[0, 0].tick_params(labelsize=font_size_tick)
    #axs[0, 0].grid(True, alpha=0.3, linestyle='--')

    # Column 1: RDF-Real
    if rdf_ref_real_full.ndim == 2:
        rdf_ref_mean = np.mean(rdf_ref_real_full, axis=0)
        rdf_pert_mean = np.mean(rdf_pert_real_full, axis=0)
    else:
        rdf_ref_mean = rdf_ref_real_full
        rdf_pert_mean = rdf_pert_real_full
    axs[0, 1].plot(x_rdf_real, rdf_ref_mean, label='Reference', alpha=0.8, lw=1.0)
    axs[0, 1].plot(x_rdf_real, rdf_pert_mean, label='Perturbed', alpha=0.8, lw=1.0, linestyle='--')
    axs[0, 1].set_ylabel('RDF', fontsize=font_size_label)
    axs[0, 1].set_title('(b) RDF', fontsize=font_size_title, fontweight='bold')
    axs[0, 1].legend(fontsize=font_size_legend, loc='upper left', frameon=True, fancybox=True)
    axs[0, 1].tick_params(labelsize=font_size_tick)
    #axs[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Lower row: Reference vs Optimized
    # Column 0: Power Spectrum
    axs[1, 0].plot(p_ref_numpy, label='Reference', alpha=0.8, lw=1.0)
    axs[1, 0].plot(p_opt_numpy, label='Optimized', alpha=0.8, lw=1.0)
    axs[1, 0].set_xlabel('Power Spectrum Index', fontsize=font_size_label)
    axs[1, 0].set_ylabel('$P_{nl}$', fontsize=font_size_label)
    #axs[1, 0].set_title('(c) Power Spectrum: Ref vs Opt', fontsize=font_size_title, fontweight='bold', loc='left')
    axs[1, 0].legend(fontsize=font_size_legend, loc='upper left', frameon=True, fancybox=True)
    axs[1, 0].tick_params(labelsize=font_size_tick)
    #axs[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Column 1: RDF-Real
    if rdf_opt_real_full.ndim == 2:
        rdf_opt_mean = np.mean(rdf_opt_real_full, axis=0)
    else:
        rdf_opt_mean = rdf_opt_real_full
    axs[1, 1].plot(x_rdf_real, rdf_ref_mean, label='Reference', alpha=0.8, lw=1.0)
    axs[1, 1].plot(x_rdf_real, rdf_opt_mean, label='Optimized', alpha=0.8, lw=1.0, linestyle='--')
    axs[1, 1].set_xlabel('r (Å)', fontsize=font_size_label)
    axs[1, 1].set_ylabel('RDF', fontsize=font_size_label)
    axs[1, 1].legend(fontsize=font_size_legend, loc='upper left', frameon=True, fancybox=True)
    axs[1, 1].tick_params(labelsize=font_size_tick)
    
    # Add overall figure title
    #fig.suptitle(f'{prototype.upper()}', 
    #             fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_file = f'fig/{name}_inverse_optimization_plot.png'
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"  {fig_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inverse Optimization of Crystal Structure using RECIPROCAL-space Descriptors')
    parser.add_argument('--prototype', type=str, default='diamond', help='Structure prototype (default: diamond)')
    args = parser.parse_args()

    main(prototype=args.prototype)