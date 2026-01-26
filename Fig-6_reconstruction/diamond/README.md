# Crystal Structure Reconstruction from Perturbed Configurations

## Overview

This module reconstructs elemental crystal structures from perturbed configurations using inverse optimization with reciprocal-space descriptors and real-space RDF matching.

## Case Studies

Two crystal systems are examined:

1. **Diamond** (Space group 166, R-3m)
2. **Hexagonal Diamond** (Space group 193, P63/mcm)

## Perturbation

Structural perturbations are generated using PyXtal's `apply_perturbation()` function:

```python
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
```

**Perturbation parameters:**
- **Diamond**: $(d_{\text{lat}}, d_{\text{coor}}) = (0.1, 0.3)$
- **Hexagonal Diamond**: $(d_{\text{lat}}, d_{\text{coor}}) = (0.15, 0.45)$

## Real-Space RDF Calculation

For single-element crystals (diamond and hexagonal diamond), all atoms are crystallographically equivalent. The real-space radial distribution function is computed as a histogram of neighbor distances using `RDFCalculator.dist2rdf()` with Gaussian smoothing, then averaged across all center atoms via `mode="mean"`.

## Optimization

Inverse optimization minimizes the combined objective function using SciPy's multi-stage approach (Nelder-Mead → L-BFGS-B refinement stages).

**Limitations:** For increased perturbation magnitudes beyond the given parameters, structures become trapped in local minima despite the multi-stage optimization strategy. More sophisticated methods (e.g., basin-hopping, global optimization) may be required to handle larger perturbations.

## References

- PyXtal: [https://github.com/qzhu2017/pyxtal](https://github.com/qzhu2017/pyxtal)
- SciPy Optimization: [https://docs.scipy.org/doc/scipy/reference/optimize.html](https://docs.scipy.org/doc/scipy/reference/optimize.html)