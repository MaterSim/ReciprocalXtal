#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    echo "benchmark2/scripts/run_pnl_benchmark.sh"
    echo "Wraps benchmark2/run_pnl_benchmark.py and computes a default output directory from the benchmark arguments."
    echo
    python "${BENCHMARK2_DIR}/run_pnl_benchmark.py" --help
    exit 0
  fi
done

args=("$@")
dataset_dir=""
explicit_output_dir=""
dmax="10.0"
nmax="10"
lmax="10"
rbasis="bessel"
continuous_match_profile="normalized"
pnl_first_weight="0.1"
normalize_reciprocal=0

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
    --nmax)
      ((i+=1))
      nmax="${args[$i]}"
      ;;
    --lmax)
      ((i+=1))
      lmax="${args[$i]}"
      ;;
    --rbasis)
      ((i+=1))
      rbasis="${args[$i]}"
      ;;
    --continuous-match-profile)
      ((i+=1))
      continuous_match_profile="${args[$i]}"
      ;;
    --pnl-first-weight)
      ((i+=1))
      pnl_first_weight="${args[$i]}"
      ;;
    --normalize-reciprocal)
      normalize_reciprocal=1
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
  pnl_tag="dmax$(path_token "${dmax}")"
  pnl_tag+="_nmax$(path_token "${nmax}")"
  pnl_tag+="_lmax$(path_token "${lmax}")"
  pnl_tag+="_rbasis_$(path_token "${rbasis}")"
  pnl_tag+="_profile_$(path_token "${continuous_match_profile}")"
  pnl_tag+="_pnlw$(path_token "${pnl_first_weight}")"
  if [[ "${normalize_reciprocal}" -eq 1 ]]; then
    pnl_tag+="_norm1"
  fi
  benchmark_output_dir="${dataset_dir%/}/results/pnl/${pnl_tag}"
fi

log_file="${benchmark_output_dir%/}/run_pnl_benchmark.log"

cmd=(python "${BENCHMARK2_DIR}/run_pnl_benchmark.py" "${args[@]}")
if [[ -z "${explicit_output_dir}" ]]; then
  cmd+=(--output-dir "${benchmark_output_dir}")
fi

cd "${REPO_ROOT}"
mkdir -p "${benchmark_output_dir}"
log "Dataset dir: ${dataset_dir}"
log "Output dir: ${benchmark_output_dir}"
log "Log file: ${log_file}"
"${cmd[@]}" | tee "${log_file}"
