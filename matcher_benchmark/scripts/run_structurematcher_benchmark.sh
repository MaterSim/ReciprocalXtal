#!/usr/bin/env bash

#SBATCH --job-name="benchmark2_structurematcher"
#SBATCH --partition=Apus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

source /users/oridwan/miniconda3/etc/profile.d/conda.sh
conda activate xtal

# Edit these defaults for direct `sbatch benchmark2/scripts/run_structurematcher_benchmark.sh`.

BUCKET="large"
PY_ARGS=(
  --dataset-dir "/users/oridwan/Github/ReciprocalXtal/matcher_benchmark/dataset/${BUCKET}"
)

log_dir="/users/oridwan/Github/ReciprocalXtal/matcher_benchmark/logs"
log_file="${log_dir}/run_structurematcher_benchmark.log"

cd /users/oridwan/Github/ReciprocalXtal
mkdir -p "${log_dir}"
printf "[%s] %s\n" "$(date +"%H:%M:%S")" "Log file: ${log_file}"
python /users/oridwan/Github/ReciprocalXtal/matcher_benchmark/run_structurematcher_benchmark.py "${PY_ARGS[@]}" "$@" 2>&1 | tee "${log_file}"
