#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    echo "benchmark2/scripts/run_structurematcher_benchmark.sh"
    echo "Wraps benchmark2/run_structurematcher_benchmark.py and computes the default StructureMatcher output directory."
    echo
    python "${BENCHMARK2_DIR}/run_structurematcher_benchmark.py" --help
    exit 0
  fi
done

args=("$@")
dataset_dir=""
explicit_output_dir=""

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
  esac
  ((i+=1))
done

if [[ -z "${dataset_dir}" ]]; then
  echo "--dataset-dir is required." >&2
  exit 1
fi

benchmark_output_dir="${explicit_output_dir}"
if [[ -z "${benchmark_output_dir}" ]]; then
  benchmark_output_dir="${dataset_dir%/}/results/structurematcher/default"
fi

log_file="${benchmark_output_dir%/}/run_structurematcher_benchmark.log"

cmd=(python "${BENCHMARK2_DIR}/run_structurematcher_benchmark.py" "${args[@]}")
if [[ -z "${explicit_output_dir}" ]]; then
  cmd+=(--output-dir "${benchmark_output_dir}")
fi

cd "${REPO_ROOT}"
mkdir -p "${benchmark_output_dir}"
log "Dataset dir: ${dataset_dir}"
log "Output dir: ${benchmark_output_dir}"
log "Log file: ${log_file}"
"${cmd[@]}" | tee "${log_file}"
