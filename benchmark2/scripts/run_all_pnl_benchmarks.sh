#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    cat <<'EOF'
benchmark2/scripts/run_all_pnl_benchmarks.sh
Runs P_nl sequentially for small, medium, and large datasets.

Options specific to this wrapper:
  --dataset-root PATH   Base directory that contains small/, medium/, large/
                        Default: benchmark2/datasets
  --dataset-name NAME   Optional dataset subdirectory name inside each bucket

All other arguments are forwarded to benchmark2/scripts/run_pnl_benchmark.sh
except --dataset-dir and --output-dir, which are computed per bucket.
EOF
    exit 0
  fi
done

args=("$@")
dataset_root="benchmark2/datasets_2"
dataset_name=""
forward_args=()

i=0
while [[ $i -lt $# ]]; do
  arg="${args[$i]}"
  case "$arg" in
    --dataset-root)
      ((i+=1))
      dataset_root="${args[$i]}"
      ;;
    --dataset-name)
      ((i+=1))
      dataset_name="${args[$i]}"
      ;;
    --dataset-dir|--output-dir)
      echo "${arg} is not supported by this wrapper. Use --dataset-root and optional --dataset-name." >&2
      exit 1
      ;;
    *)
      forward_args+=("$arg")
      ;;
  esac
  ((i+=1))
done

cd "${REPO_ROOT}"

for bucket in small medium large; do
  dataset_dir="${dataset_root%/}/${bucket}"
  if [[ -n "${dataset_name}" ]]; then
    dataset_dir+="/${dataset_name}"
  fi

  log "Running P_nl for ${bucket}"
  cmd=("${BENCHMARK2_SCRIPTS_DIR}/run_pnl_benchmark.sh" --dataset-dir "${dataset_dir}")
  if [[ ${#forward_args[@]} -gt 0 ]]; then
    cmd+=("${forward_args[@]}")
  fi
  "${cmd[@]}"
done
