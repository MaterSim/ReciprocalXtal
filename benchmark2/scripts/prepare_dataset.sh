#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    echo "benchmark2/scripts/prepare_dataset.sh"
    echo "Wraps benchmark2/prepare_dataset.py and computes a default output directory from the passed dataset arguments."
    echo
    python "${BENCHMARK2_DIR}/prepare_dataset.py" --help
    exit 0
  fi
done

args=("$@")
bucket=""
explicit_output_dir=""
max_per_formula="20"
max_total_structures="60"
min_reference_parents="2"
seed="7"
symprec="0.01"
max_axis_multiplier="6"
formulas=()
material_ids=()
coord_noise_levels=()
lattice_noise_levels=()

i=0
while [[ $i -lt $# ]]; do
  arg="${args[$i]}"
  case "$arg" in
    --bucket)
      ((i+=1))
      bucket="${args[$i]}"
      ;;
    --output-dir)
      ((i+=1))
      explicit_output_dir="${args[$i]}"
      ;;
    --formula)
      formulas=()
      ((i+=1))
      while [[ $i -lt $# && "${args[$i]}" != --* ]]; do
        formulas+=("${args[$i]}")
        ((i+=1))
      done
      ((i-=1))
      ;;
    --material-id)
      ((i+=1))
      material_ids+=("${args[$i]}")
      ;;
    --max-per-formula)
      ((i+=1))
      max_per_formula="${args[$i]}"
      ;;
    --max-total-structures)
      ((i+=1))
      max_total_structures="${args[$i]}"
      ;;
    --min-reference-parents-per-formula)
      ((i+=1))
      min_reference_parents="${args[$i]}"
      ;;
    --coord-noise)
      ((i+=1))
      coord_noise_levels+=("${args[$i]}")
      ;;
    --lattice-noise)
      ((i+=1))
      lattice_noise_levels+=("${args[$i]}")
      ;;
    --seed)
      ((i+=1))
      seed="${args[$i]}"
      ;;
    --symprec)
      ((i+=1))
      symprec="${args[$i]}"
      ;;
    --max-axis-multiplier)
      ((i+=1))
      max_axis_multiplier="${args[$i]}"
      ;;
  esac
  ((i+=1))
done

if [[ -z "${bucket}" ]]; then
  echo "--bucket is required." >&2
  exit 1
fi

if [[ ${#formulas[@]} -eq 0 ]]; then
  formulas=(C SiO2 TiO2)
fi

if [[ ${#coord_noise_levels[@]} -eq 0 ]]; then
  coord_noise_levels=("0.002" "0.015" "0.050")
fi

if [[ ${#lattice_noise_levels[@]} -eq 0 ]]; then
  lattice_noise_levels=("0.003" "0.020" "0.060")
fi

if [[ ${#coord_noise_levels[@]} -ne ${#lattice_noise_levels[@]} ]]; then
  echo "--coord-noise and --lattice-noise must be repeated the same number of times." >&2
  exit 1
fi

dataset_output_dir="${explicit_output_dir}"
if [[ -z "${dataset_output_dir}" ]]; then
  if [[ ${#material_ids[@]} -gt 0 ]]; then
    selection_token="ids_$(join_by "-" "${material_ids[@]}")"
  else
    selection_token="formulas_$(join_by "-" "${formulas[@]}")"
  fi

  coord_tokens=()
  for coord_noise in "${coord_noise_levels[@]}"; do
    coord_tokens+=("$(path_token "${coord_noise}")")
  done

  lattice_tokens=()
  for lattice_noise in "${lattice_noise_levels[@]}"; do
    lattice_tokens+=("$(path_token "${lattice_noise}")")
  done

  dataset_tag="$(path_token "${selection_token}")"
  dataset_tag+="_mpf$(path_token "${max_per_formula}")"
  dataset_tag+="_mts$(path_token "${max_total_structures}")"
  dataset_tag+="_mrp$(path_token "${min_reference_parents}")"
  dataset_tag+="_coord_$(join_by "__" "${coord_tokens[@]}")"
  dataset_tag+="_lattice_$(join_by "__" "${lattice_tokens[@]}")"
  dataset_tag+="_seed$(path_token "${seed}")"
  dataset_tag+="_sym$(path_token "${symprec}")"
  dataset_tag+="_amax$(path_token "${max_axis_multiplier}")"
  dataset_output_dir="benchmark2/datasets/${bucket}/${dataset_tag}"
fi

log_file="${dataset_output_dir%/}/prepare_dataset.log"

cmd=(python "${BENCHMARK2_DIR}/prepare_dataset.py" "${args[@]}")
if [[ -z "${explicit_output_dir}" ]]; then
  cmd+=(--output-dir "${dataset_output_dir}")
fi

cd "${REPO_ROOT}"
mkdir -p "${dataset_output_dir}"
log "Bucket: ${bucket}"
log "Output dir: ${dataset_output_dir}"
log "Log file: ${log_file}"
"${cmd[@]}" | tee "${log_file}"
