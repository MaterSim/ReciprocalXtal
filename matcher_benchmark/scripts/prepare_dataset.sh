#!/usr/bin/env bash

#SBATCH --job-name="matcher_prepare_dataset"
#SBATCH --partition=Apus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

source /users/oridwan/miniconda3/etc/profile.d/conda.sh
conda activate xtal

# Edit the bucket once here.
BUCKET="small"

# Edit these defaults for direct `sbatch matcher_benchmark/scripts/prepare_dataset.sh`.
PY_ARGS=(
  --bucket "${BUCKET}"
  --output-dir "matcher_benchmark/dataset_2/${BUCKET}"
  --formula C SiO2 TiO2
  --max-per-formula 20
  --max-total-structures 60
  --min-reference-parents-per-formula 2
  --coord-noise 0.002 0.015 0.050
  --lattice-noise 0.003 0.010 0.030
  --seed 7
  --symprec 0.01
  --max-axis-multiplier 6
)

cd /users/oridwan/Github/ReciprocalXtal
python /users/oridwan/Github/ReciprocalXtal/matcher_benchmark/prepare_dataset.py "${PY_ARGS[@]}"
