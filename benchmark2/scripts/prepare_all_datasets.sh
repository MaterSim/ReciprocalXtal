#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    cat <<'EOF'
benchmark2/scripts/prepare_all_datasets.sh
Runs dataset preparation sequentially for small, medium, and large buckets.

Options specific to this wrapper:
  --dataset-root PATH   Base directory for bucket outputs.
                        Default: benchmark2/datasets

All other arguments are forwarded to benchmark2/scripts/prepare_dataset.sh
except --bucket and --output-dir, which are computed per bucket.
EOF
    exit 0
  fi
done

args=("$@")
dataset_root="benchmark2/datasets_2"
forward_args=()

i=0
while [[ $i -lt $# ]]; do
  arg="${args[$i]}"
  case "$arg" in
    --dataset-root)
      ((i+=1))
      dataset_root="${args[$i]}"
      ;;
    --bucket|--output-dir)
      echo "${arg} is not supported by this wrapper. Use --dataset-root instead." >&2
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
  log "Preparing dataset for ${bucket}"
  cmd=(
    "${BENCHMARK2_SCRIPTS_DIR}/prepare_dataset.sh"
    --bucket "${bucket}"
    --output-dir "${dataset_root%/}/${bucket}"
  )
  if [[ ${#forward_args[@]} -gt 0 ]]; then
    cmd+=("${forward_args[@]}")
  fi
  "${cmd[@]}"
done
