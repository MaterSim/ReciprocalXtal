#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

timestamp() {
  date +"%H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

path_token() {
  local value="$1"
  value="${value// /_}"
  value="${value//\//-}"
  value="${value//:/-}"
  value="${value//./p}"
  echo "$value"
}

PYTHON_BIN="${PYTHON_BIN:-/Users/oridwan/miniconda3/envs/xtal/bin/python}"
CACHED_RESULTS_ROOT="${CACHED_RESULTS_ROOT:-benchmark/results}"
SEED="${SEED:-7}"
DMAX="${DMAX:-20.0}"
NMAX="${NMAX:-5}"
LMAX="${LMAX:-5}"
RBASIS="${RBASIS:-Bessel}"
G_BIN_WIDTH="${G_BIN_WIDTH:-0.02}"
CONTINUOUS_MATCH_PROFILE="${CONTINUOUS_MATCH_PROFILE:-normalized}"
PNL_FIRST_WEIGHT="${PNL_FIRST_WEIGHT:-0.1}"
NORMALIZE_RECIPROCAL="${NORMALIZE_RECIPROCAL:-0}"
COPY_DATASET="${COPY_DATASET:-1}"
SIZE_BUCKETS=(${SIZE_BUCKETS:-small medium large})

OUTPUT_ROOT_DEFAULT="benchmark/results_cached_reuse_dmax$(path_token "${DMAX}")_nmax$(path_token "${NMAX}")_lmax$(path_token "${LMAX}")_pnlw$(path_token "${PNL_FIRST_WEIGHT}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_ROOT_DEFAULT}}"
FIGURE_DIR="${FIGURE_DIR:-${OUTPUT_ROOT}/figures}"

SIZE_BUCKETS_STR="${SIZE_BUCKETS[*]:-}"

mkdir -p "$OUTPUT_ROOT"
cat > "${OUTPUT_ROOT}/run_settings.txt" <<EOF
PYTHON_BIN=${PYTHON_BIN}
CACHED_RESULTS_ROOT=${CACHED_RESULTS_ROOT}
OUTPUT_ROOT=${OUTPUT_ROOT}
FIGURE_DIR=${FIGURE_DIR}
SIZE_BUCKETS=${SIZE_BUCKETS_STR}
SEED=${SEED}
DMAX=${DMAX}
NMAX=${NMAX}
LMAX=${LMAX}
RBASIS=${RBASIS}
G_BIN_WIDTH=${G_BIN_WIDTH}
CONTINUOUS_MATCH_PROFILE=${CONTINUOUS_MATCH_PROFILE}
PNL_FIRST_WEIGHT=${PNL_FIRST_WEIGHT}
NORMALIZE_RECIPROCAL=${NORMALIZE_RECIPROCAL}
COPY_DATASET=${COPY_DATASET}
EOF

COMMON_ARGS=()
if [[ "${COPY_DATASET}" == "0" ]]; then
  COMMON_ARGS+=(--reuse-dataset-in-place)
else
  COMMON_ARGS+=(--copy-dataset)
fi
if [[ -n "${SEED}" ]]; then
  COMMON_ARGS+=(--seed "${SEED}")
fi
if [[ -n "${DMAX}" ]]; then
  COMMON_ARGS+=(--dmax "${DMAX}")
fi
if [[ -n "${NMAX}" ]]; then
  COMMON_ARGS+=(--nmax "${NMAX}")
fi
if [[ -n "${LMAX}" ]]; then
  COMMON_ARGS+=(--lmax "${LMAX}")
fi
if [[ -n "${RBASIS}" ]]; then
  COMMON_ARGS+=(--rbasis "${RBASIS}")
fi
if [[ -n "${G_BIN_WIDTH}" ]]; then
  COMMON_ARGS+=(--g-bin-width "${G_BIN_WIDTH}")
fi
if [[ -n "${CONTINUOUS_MATCH_PROFILE}" ]]; then
  COMMON_ARGS+=(--continuous-match-profile "${CONTINUOUS_MATCH_PROFILE}")
fi
if [[ -n "${PNL_FIRST_WEIGHT}" ]]; then
  COMMON_ARGS+=(--pnl-first-weight "${PNL_FIRST_WEIGHT}")
fi
if [[ "${NORMALIZE_RECIPROCAL}" == "1" ]]; then
  COMMON_ARGS+=(--normalize-reciprocal)
elif [[ "${NORMALIZE_RECIPROCAL}" == "0" ]]; then
  COMMON_ARGS+=(--no-normalize-reciprocal)
fi

log "Cached results root: ${CACHED_RESULTS_ROOT}"
log "New output root: ${OUTPUT_ROOT}"

for size_bucket in "${SIZE_BUCKETS[@]}"; do
  cache_dir="${CACHED_RESULTS_ROOT}/${size_bucket}_cell"
  output_dir="${OUTPUT_ROOT}/${size_bucket}_cell"

  if [[ ! -d "${cache_dir}" ]]; then
    log "Skipping ${size_bucket} cell because ${cache_dir} does not exist"
    continue
  fi

  log "Recomputing reciprocal outputs for ${size_bucket} cell from cache"
  "${PYTHON_BIN}" benchmark/scripts/rerun_reciprocal_from_cache.py \
    --cached-results-dir "${cache_dir}" \
    --output-dir "${output_dir}" \
    "${COMMON_ARGS[@]}"
done

summary_args=()
for size_bucket in small medium large; do
  summary_path="${OUTPUT_ROOT}/${size_bucket}_cell/benchmark_summary.json"
  if [[ -f "${summary_path}" ]]; then
    summary_args+=("${summary_path}")
  fi
done

if [[ ${#summary_args[@]} -eq 3 ]]; then
  log "Generating merged plot for reused benchmark outputs"
  "${PYTHON_BIN}" benchmark/scripts/plot_benchmark_summary_merged.py \
    "${summary_args[@]}" \
    --output-dir "${FIGURE_DIR}"
  log "Wrote merged figure under: ${FIGURE_DIR}"
else
  log "Skipping merged plot because one or more size-bucket summaries are missing"
fi

log "Wrote reused benchmark outputs under: ${OUTPUT_ROOT}"
log "Wrote run settings under: ${OUTPUT_ROOT}/run_settings.txt"
