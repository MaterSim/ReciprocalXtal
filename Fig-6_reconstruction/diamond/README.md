# Crystal Structure Reconstruction from Perturbed Configurations

## Overview

This module reconstructs elemental crystal structures from perturbed configurations using inverse optimization with reciprocal-space descriptors and real-space RDF matching.

## Case Studies

Two crystal systems are examined:

1. **Diamond** (Space group 166, R-3m)
2. **Hexagonal Diamond** (Space group 193, P63/mcm)

## Perturbation

Structural perturbations are generated using PyXtal's `apply_perturbation()` function with lattice perturbation magnitude (`d_lat`, relative) and coordinate displacement magnitude (`d_coor`, in Å).

Default parameters in `reconstruct-diamond.py`:
- **Diamond**: $d_{\text{lat}}=0.5$, $d_{\text{coor}}=0.5$ Å
- **Hexagonal Diamond**: $d_{\text{lat}}=0.2$, $d_{\text{coor}}=1.5$ Å

## Real-Space RDF Calculation

For single-element crystals (diamond and hexagonal diamond), all atoms are crystallographically equivalent. The real-space radial distribution function is computed as a histogram of neighbor distances using `RDFCalculator.dist2rdf()` with Gaussian smoothing, then averaged across all center atoms via `mode="mean"`.

## Optimization

Inverse optimization minimizes the combined objective function using SciPy's multi-stage approach (Nelder-Mead → L-BFGS-B refinement stages).

## How to Run

```bash
# Default: diamond with d_lat=0.5, d_coor=0.5 Å
python reconstruct-diamond.py --prototype diamond

# Hexagonal diamond with custom perturbation
python reconstruct-diamond.py --prototype h-diamond --d-lat 0.2 --d-coor 1.5

# Custom perturbation parameters
python reconstruct-diamond.py --prototype diamond --d-lat 0.3 --d-coor 0.8
```

**Command-line options:**
- `--prototype`: Structure prototype (`diamond` or `h-diamond`; default: `diamond`)
- `--d-lat`: Lattice perturbation magnitude, relative (default: 0.5)
- `--d-coor`: Coordinate perturbation magnitude in Å (default: 0.5)

Outputs are written to `cifs/` (reference, perturbed, optimized CIFs) and `fig/` (comparison plot).

**Limitations:** For increased perturbation magnitudes beyond the given parameters, structures may become trapped in local minima despite the multi-stage optimization strategy. More sophisticated methods (e.g., basin-hopping, global optimization) may be required to handle larger perturbations.

## References

- PyXtal: [https://github.com/qzhu2017/pyxtal](https://github.com/qzhu2017/pyxtal)
- SciPy Optimization: [https://docs.scipy.org/doc/scipy/reference/optimize.html](https://docs.scipy.org/doc/scipy/reference/optimize.html)
