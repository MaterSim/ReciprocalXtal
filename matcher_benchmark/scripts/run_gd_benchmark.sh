#!/bin/bash

#SBATCH --job-name="benchmark2_gd"
#SBATCH --partition=Apus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/dev/null

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SH="${SCRIPT_DIR}/common.sh"
if [[ ! -f "${COMMON_SH}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  COMMON_SH="${SLURM_SUBMIT_DIR%/}/benchmark2/scripts/common.sh"
fi
source "${COMMON_SH}"
activate_xtal_env

for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    echo "benchmark2/scripts/run_gd_benchmark.sh"
    echo "Wraps benchmark2/run_gd_benchmark.py and computes a default output directory from the benchmark arguments."
    echo
    python "${BENCHMARK2_DIR}/run_gd_benchmark.py" --help
    exit 0
  fi
done

args=("$@")
dataset_dir=""
explicit_output_dir=""
dmax="10.0"
g_bin_width="0.02"
continuous_match_profile="normalized"

i=0
while [[ $i -lt $# ]]; do
  arg="${args[$i]}"
  case "$arg" in
    --dataset-dir)
      ((i+=1))
      dataset_dir="${args[$i]}"
      ;;
    --output-dir)
      ((i+=1))
      explicit_output_dir="${args[$i]}"
      ;;
    --dmax)
      ((i+=1))
      dmax="${args[$i]}"
      ;;
    --g-bin-width)
      ((i+=1))
      g_bin_width="${args[$i]}"
      ;;
    --continuous-match-profile)
      ((i+=1))
      continuous_match_profile="${args[$i]}"
      ;;
  esac
  ((i+=1))
done

if [[ -z "${dataset_dir}" ]]; then
  echo "--dataset-dir is required." >&2
  exit 1
fi

benchmark_output_dir="${explicit_output_dir}"
if [[ -z "${benchmark_output_dir}" ]]; then
  gd_tag="dmax$(path_token "${dmax}")"
  gd_tag+="_gbin$(path_token "${g_bin_width}")"
  gd_tag+="_profile_$(path_token "${continuous_match_profile}")"
  benchmark_output_dir="${dataset_dir%/}/results/gd/${gd_tag}"
fi

log_file="${benchmark_output_dir%/}/run_gd_benchmark.log"

cmd=(python "${BENCHMARK2_DIR}/run_gd_benchmark.py" "${args[@]}")
if [[ -z "${explicit_output_dir}" ]]; then
  cmd+=(--output-dir "${benchmark_output_dir}")
fi

cd "${REPO_ROOT}"
mkdir -p "${benchmark_output_dir}"
log "Dataset dir: ${dataset_dir}"
log "Output dir: ${benchmark_output_dir}"
log "Log file: ${log_file}"
"${cmd[@]}" | tee "${log_file}"
