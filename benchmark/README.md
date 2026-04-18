# Benchmark: Threshold Matching vs StructureMatcher

This folder contains a runnable benchmark script for comparing:

- **Our approach**: reciprocal-space descriptors with configurable post-processing
  and threshold-based pairwise matching
- **Baseline**: `pymatgen` `StructureMatcher` (strict/medium/loose tolerances)

The continuous descriptors currently benchmarked are:

- reciprocal-space power spectrum `P_nl`
- raw reciprocal radial profile `G(d)`

## Script

- `scripts/run_structurematcher_vs_reciprocal.py`

## What it does

1. Scans CIF files under `Fig-6_reconstruction/**/cifs/*.cif`
2. Uses `*reference*.cif` files as reference database
3. Uses all non-reference CIF files as queries
4. Runs pairwise `StructureMatcher.fit()` for strict/medium/loose settings
5. Computes same-formula pair distances for `P_nl` and raw `G(d)`
6. Fits descriptor thresholds against those labels on a held-out parent split
7. Writes pairwise threshold predictions and benchmark summaries

## Usage

From repository root:

```bash
python benchmark/scripts/run_structurematcher_vs_reciprocal.py
```

Optional flags:

- `--dataset-root Fig-6_reconstruction`
- `--output-dir benchmark/results`
- `--dmax 10 --nmax 10 --lmax 10 --rbasis Bessel`
- `--continuous-match-profile normalized`
- `--pnl-first-weight 0.1`

## Output files

- `benchmark/results/dataset_manifest.csv`
- `benchmark/results/pairwise_threshold_summary.csv`
- `benchmark/results/pairwise_threshold_predictions.csv`
- `benchmark/results/structurematcher_pairs.csv`
- `benchmark/results/structurematcher_query_summary.csv`
- `benchmark/results/structurematcher_aggregate.csv`
- `benchmark/results/benchmark_summary.json`

Generated benchmark outputs under `benchmark/results/` are intentionally ignored
by git so the folder stays clean between runs.

## Dependencies

Install if missing:

```bash
pip install numpy ase pymatgen pyxtal torch e3nn scipy monty
```
