#!/usr/bin/env bash

#SBATCH --job-name="benchmark2_pnl"
#SBATCH --partition=Apus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PYTHON_BIN="/users/oridwan/miniconda3/envs/xtal/bin/python"

BUCKET="large"
dataset_dir="/users/oridwan/Github/ReciprocalXtal/matcher_benchmark/dataset/${BUCKET}"

# Edit these defaults for direct `sbatch matcher_benchmark/scripts/run_pnl_benchmark.sh`.
PY_ARGS=(
  --dataset-dir "${dataset_dir}"
  --dmax 10
  --nmax 10
  --lmax 10
  --rbasis bessel
  --continuous-match-profile shape
  --pnl-first-weight 0.1
  --calibration-source queries
  --threshold-policy strict_medium_loose
  --calibration-grouping all_noise_same_parent


)

log_dir="/users/oridwan/Github/ReciprocalXtal/matcher_benchmark/logs"
log_file="${log_dir}/run_pnl_benchmark.log"

cd /users/oridwan/Github/ReciprocalXtal
mkdir -p "${log_dir}"
printf "[%s] %s\n" "$(date +"%H:%M:%S")" "Log file: ${log_file}"
start_time=$(date +%s)
"${PYTHON_BIN}" /users/oridwan/Github/ReciprocalXtal/matcher_benchmark/run_pnl_benchmark.py "${PY_ARGS[@]}" "$@" 2>&1 | tee "${log_file}"
end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "[%s] Total time taken: %d seconds\n" "$(date +"%H:%M:%S")" "$elapsed" | tee -a "${log_file}"
