#!/usr/bin/env bash

set -euo pipefail

BENCHMARK2_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK2_DIR="$(cd "${BENCHMARK2_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BENCHMARK2_DIR}/.." && pwd)"

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
