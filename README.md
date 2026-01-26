

# ReciprocalXtal

This repository accompanies the paper  **“Crystal Representation in the Reciprocal Space”**.

It contains scripts used to generate the  figures and demonstrations in the manuscript, focusing on a rotation- and translation-invariant crystal representation built in reciprocal space.

## Repository Structure

```
.
├── reciprocal.py        # Core routines for reciprocal-space representation and P_nl computation
├── rdf_real.py          # Real-space RDF Calculation
├── README.md
├── Fig-4_Pnl_calc/
│   ├── compute_and_plot__Pnl.py   # Computes and visualizes the power spectrum P_nl
├── Fig-5_noise_stability/
│   ├── plot_diamond_noise_robustness.py   # Noise-robustness test (diamond)
└── Fig-6_reconstruction/                   # Reconstruct from noisy structure
    ├── diamond/
    └── quartz/
```

## What This Code Does

- Builds a four-dimensional reciprocal-space representation from structure factors.
- Converts it into a rotationally invariant power spectrum using spherical harmonics.
- Demonstrates:
  - Power-spectrum calculation (Fig. 4),
  - Robustness against noise and symmetry-equivalent perturbations (Fig. 5),
  - Symmetry-constrained reconstruction of crystal structures (Fig. 6).

The scripts are intentionally lightweight and meant to illustrate the core ideas rather than provide a full production framework.



## Set Up Conda Environment

We recommend using a dedicated Conda environment to reproduce the results.

```bash
conda create -n xtal python=3.10.8
conda activate xtal

# Install PyXtal 
pip install --upgrade git+https://github.com/MaterSim/PyXtal.git@master

```

This setup provides the minimal environment required to run the example scripts in this repository.

