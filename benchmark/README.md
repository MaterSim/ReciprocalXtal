# Benchmark: StructureMatcher vs Reciprocal Power Spectrum

This folder contains a runnable benchmark script for comparing:

- **Our approach**: reciprocal-space power spectrum (`RECP`) + L2 retrieval
- **Baseline**: `pymatgen` `StructureMatcher` (strict/medium/loose tolerances)

## Script

- `scripts/run_structurematcher_vs_reciprocal.py`

## What it does

1. Scans CIF files under `Fig-6_reconstruction/**/cifs/*.cif`
2. Uses `*reference*.cif` files as reference database
3. Uses all non-reference CIF files as queries
4. Computes top-1 retrieval for reciprocal descriptor distances
5. Runs pairwise `StructureMatcher.fit()` for strict/medium/loose settings
6. Writes CSV outputs in `benchmark/results/`

## Usage

From repository root:

```bash
python benchmark/scripts/run_structurematcher_vs_reciprocal.py
```

Optional flags:

- `--dataset-root Fig-6_reconstruction`
- `--output-dir benchmark/results`
- `--dmax 10 --nmax 10 --lmax 10 --rbasis Bessel`

## Output files

- `benchmark/results/reciprocal_retrieval.csv`
- `benchmark/results/structurematcher_pairs.csv`
- `benchmark/results/structurematcher_query_summary.csv`
- `benchmark/results/benchmark_summary.json`

## Dependencies

Install if missing:

```bash
pip install numpy ase pymatgen pyxtal torch e3nn scipy monty
```
