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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SH="${SCRIPT_DIR}/common.sh"
if [[ ! -f "${COMMON_SH}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  COMMON_SH="${SLURM_SUBMIT_DIR%/}/matcher_benchmark/scripts/common.sh"
fi
# shellcheck disable=SC1090
source "${COMMON_SH}"

BUCKET="small"
dataset_dir="${BENCHMARK2_DIR}/dataset/${BUCKET}"

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

log_dir="${BENCHMARK2_DIR}/logs"
log_file="${log_dir}/run_pnl_benchmark.log"

mkdir -p "${log_dir}"
printf "[%s] %s\n" "$(date +"%H:%M:%S")" "Log file: ${log_file}"
start_time=$(date +%s)
cd "${REPO_ROOT}"
python "${BENCHMARK2_DIR}/run_pnl_benchmark.py" "${PY_ARGS[@]}" "$@" 2>&1 | tee "${log_file}"
end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "[%s] Total time taken: %d seconds\n" "$(date +"%H:%M:%S")" "$elapsed" | tee -a "${log_file}"
