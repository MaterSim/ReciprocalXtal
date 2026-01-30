# Quartz Reconstruction from Perturbed Configurations

## Overview

This module reconstructs α- and β-quartz structures from perturbed configurations using inverse optimization that combines reciprocal-space descriptors (RECP) and pairwise real-space RDF matching.

## Case Studies

- **α quartz** (`a-quartz`, trigonal)
- **β quartz** (`b-quartz`, hexagonal)

## Perturbation

Perturbations are generated with PyXtal's `apply_perturbation()` on the reference structure, then normalized for optimization. Current settings in `reconstruct_quartz.py`:

- α quartz: $d_{lat}=0.1$, $d_{coor}=0.9$ Å
- β quartz: $d_{lat}=0.2$, $d_{coor}=1.8$ Å

## Real-Space RDF Calculation

Pair-resolved RDFs are computed with `RDFCalculator.compute_rdf(..., mode="pairwise")` using 50 bins, Gaussian smoothing ($\sigma=2.0$ Å), and a cutoff of 2.1 Å. Pair labels (e.g., `Si-Si`, `Si-O`, `O-O`) are kept to apply per-pair weights in the loss.

## Optimization

The objective is a weighted sum of descriptor RMSE and RDF RMSE. Optimization runs a staged SciPy pipeline (Nelder-Mead followed by several L-BFGS-B refinements) with bounds on lattice, angles, and Wyckoff coordinates.

## How to Run

```bash
# Default: α-quartz with d_lat=0.1, d_coor=0.9 Å
python reconstruct_quartz.py --prototype a-quartz

# β-quartz with custom perturbation
python reconstruct_quartz.py --prototype b-quartz --d-lat 0.2 --d-coor 1.8

# Custom perturbation parameters
python reconstruct_quartz.py --prototype a-quartz --d-lat 0.15 --d-coor 1.2
```

**Command-line options:**
- `--prototype`: Structure prototype (`a-quartz` or `b-quartz`; default: `a-quartz`)
- `--d-lat`: Lattice perturbation magnitude, relative (default: 0.1)
- `--d-coor`: Coordinate perturbation magnitude in Å (default: 0.9)

Outputs are written to `cifs/` (reference, perturbed, optimized CIFs) and `fig/` (comparison plot).

## References

- PyXtal: https://github.com/qzhu2017/pyxtal
- SciPy Optimization: https://docs.scipy.org/doc/scipy/reference/optimize.html