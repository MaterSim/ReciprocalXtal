#!/usr/bin/env bash

set -euo pipefail

BENCHMARK2_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK2_DIR="$(cd "${BENCHMARK2_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BENCHMARK2_DIR}/.." && pwd)"

SLURM_PARTITION="Apus"
SLURM_NODES="1"
SLURM_NTASKS_PER_NODE="1"
SLURM_CPUS_PER_TASK="96"
SLURM_TIME="120:00:00"
SLURM_MEM_PER_CPU="2G"
SLURM_OUTPUT="/dev/null"

path_token() {
  local value="$1"
  value="${value// /_}"
  value="${value//\//-}"
  value="${value//:/-}"
  value="${value//./p}"
  value="${value//+/plus}"
  value="${value//,/__}"
  echo "$value"
}

join_by() {
  local separator="$1"
  shift

  local out=""
  local first=1
  local item
  for item in "$@"; do
    if [[ $first -eq 1 ]]; then
      out="$item"
      first=0
    else
      out+="${separator}${item}"
    fi
  done
  printf "%s" "$out"
}

log() {
  printf "[%s] %s\n" "$(date +"%H:%M:%S")" "$*"
}

activate_xtal_env() {
  local conda_sh="/users/oridwan/miniconda3/etc/profile.d/conda.sh"
  if [[ ! -f "${conda_sh}" ]]; then
    echo "Conda init script not found: ${conda_sh}" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate xtal
}

submit_sbatch() {
  local job_name="$1"
  local dependency="$2"
  local script_path="$3"
  shift 3

  local -a cmd=(
    sbatch
    --parsable
    --job-name "${job_name}"
    --partition "${SLURM_PARTITION}"
    --nodes "${SLURM_NODES}"
    --ntasks-per-node "${SLURM_NTASKS_PER_NODE}"
    --cpus-per-task "${SLURM_CPUS_PER_TASK}"
    --time "${SLURM_TIME}"
    --mem-per-cpu "${SLURM_MEM_PER_CPU}"
    --output "${SLURM_OUTPUT}"
  )

  if [[ -n "${dependency}" ]]; then
    cmd+=(--dependency "${dependency}")
  fi

  cmd+=("${script_path}" "$@")

  printf "[%s] Submitting %s\n" "$(date +"%H:%M:%S")" "${job_name}" >&2
  local job_id
  job_id="$("${cmd[@]}")"
  printf "[%s] Submitted %s as job %s\n" "$(date +"%H:%M:%S")" "${job_name}" "${job_id}" >&2
  printf "%s" "${job_id}"
}
