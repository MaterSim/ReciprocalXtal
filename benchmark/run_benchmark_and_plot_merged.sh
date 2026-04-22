#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

timestamp() {
  date +"%H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark/results_dmax10}" # Output dir
FIGURE_DIR="${FIGURE_DIR:-${OUTPUT_ROOT}/figures}"

FORMULAS=(${FORMULAS:-SiO2 TiO2})
SIZE_BUCKETS=(${SIZE_BUCKETS:-small medium large})
MATERIAL_IDS=(${MATERIAL_IDS:-})

SEED="${SEED:-7}"
SYMPREC="${SYMPREC:-0.01}"
MAX_PER_FORMULA="${MAX_PER_FORMULA:-20}"
MAX_TOTAL_STRUCTURES="${MAX_TOTAL_STRUCTURES:-60}"
MAX_SUPERCELL_SITES="${MAX_SUPERCELL_SITES:-120}"

COORD_NOISE=(${COORD_NOISE:-0.02 0.05 0.10})
LATTICE_NOISE=(${LATTICE_NOISE:-0.01 0.02 0.05})

DMAX="${DMAX:-10.0}"  #dmax
NMAX="${NMAX:-10}"
LMAX="${LMAX:-10}"
RBASIS="${RBASIS:-Bessel}"
G_BIN_WIDTH="${G_BIN_WIDTH:-0.02}"
CONTINUOUS_MATCH_PROFILE="${CONTINUOUS_MATCH_PROFILE:-normalized}"
PNL_FIRST_WEIGHT="${PNL_FIRST_WEIGHT:-0.0}"

FORMULAS_STR="${FORMULAS[*]:-}"
SIZE_BUCKETS_STR="${SIZE_BUCKETS[*]:-}"
MATERIAL_IDS_STR="${MATERIAL_IDS[*]:-}"
COORD_NOISE_STR="${COORD_NOISE[*]:-}"
LATTICE_NOISE_STR="${LATTICE_NOISE[*]:-}"

mkdir -p "$OUTPUT_ROOT"
cat > "${OUTPUT_ROOT}/run_settings.txt" <<EOF
PYTHON_BIN=${PYTHON_BIN}
SOURCE=mp
OUTPUT_ROOT=${OUTPUT_ROOT}
FIGURE_DIR=${FIGURE_DIR}
FORMULAS=${FORMULAS_STR}
SIZE_BUCKETS=${SIZE_BUCKETS_STR}
MATERIAL_IDS=${MATERIAL_IDS_STR}
SEED=${SEED}
SYMPREC=${SYMPREC}
MAX_PER_FORMULA=${MAX_PER_FORMULA}
MAX_TOTAL_STRUCTURES=${MAX_TOTAL_STRUCTURES}
MAX_SUPERCELL_SITES=${MAX_SUPERCELL_SITES}
COORD_NOISE=${COORD_NOISE_STR}
LATTICE_NOISE=${LATTICE_NOISE_STR}
DMAX=${DMAX}
NMAX=${NMAX}
LMAX=${LMAX}
RBASIS=${RBASIS}
G_BIN_WIDTH=${G_BIN_WIDTH}
CONTINUOUS_MATCH_PROFILE=${CONTINUOUS_MATCH_PROFILE}
PNL_FIRST_WEIGHT=${PNL_FIRST_WEIGHT}
PRESET=${PRESET:-}
SKIP_SUPERCELL=${SKIP_SUPERCELL:-0}
NORMALIZE_RECIPROCAL=${NORMALIZE_RECIPROCAL:-0}
EOF

log "Source: mp"
log "Output root: ${OUTPUT_ROOT}"
log "Figure dir: ${FIGURE_DIR}"

COMMON_ARGS=(
  --source mp
  --seed "$SEED"
  --symprec "$SYMPREC"
  --coord-noise "${COORD_NOISE[@]}"
  --lattice-noise "${LATTICE_NOISE[@]}"
  --max-supercell-sites "$MAX_SUPERCELL_SITES"
  --dmax "$DMAX"
  --nmax "$NMAX"
  --lmax "$LMAX"
  --rbasis "$RBASIS"
  --g-bin-width "$G_BIN_WIDTH"
  --continuous-match-profile "$CONTINUOUS_MATCH_PROFILE"
  --pnl-first-weight "$PNL_FIRST_WEIGHT"
  --api-key "${MP_API_KEY:-}"
  --formula "${FORMULAS[@]}"
  --max-per-formula "$MAX_PER_FORMULA"
  --max-total-structures "$MAX_TOTAL_STRUCTURES"
)

if [[ -n "${PRESET:-}" ]]; then
  COMMON_ARGS+=(--preset "$PRESET")
fi
if [[ "${SKIP_SUPERCELL:-0}" == "1" ]]; then
  COMMON_ARGS+=(--skip-supercell)
fi
if [[ "${NORMALIZE_RECIPROCAL:-0}" == "1" ]]; then
  COMMON_ARGS+=(--normalize-reciprocal)
fi
if [[ ${#MATERIAL_IDS[@]} -gt 0 ]]; then
  for material_id in "${MATERIAL_IDS[@]}"; do
    [[ -n "$material_id" ]] && COMMON_ARGS+=(--material-id "$material_id")
  done
fi

log "Generating datasets from Materials Project"
for size_bucket in "${SIZE_BUCKETS[@]}"; do
  log "Running benchmark for ${size_bucket} cell"
  "$PYTHON_BIN" benchmark/scripts/run_structurematcher_vs_reciprocal.py \
    "${COMMON_ARGS[@]}" \
    --size-bucket "$size_bucket" \
    --output-dir "${OUTPUT_ROOT}/${size_bucket}_cell"
  log "Finished ${size_bucket} cell"
done

log "Generating merged plot"
"$PYTHON_BIN" benchmark/scripts/plot_benchmark_summary_merged.py \
  "${OUTPUT_ROOT}/small_cell/benchmark_summary.json" \
  "${OUTPUT_ROOT}/medium_cell/benchmark_summary.json" \
  "${OUTPUT_ROOT}/large_cell/benchmark_summary.json" \
  --output-dir "$FIGURE_DIR"

log "Wrote benchmark outputs under: ${OUTPUT_ROOT}"
log "Wrote merged figure under: ${FIGURE_DIR}"
log "Wrote run settings under: ${OUTPUT_ROOT}/run_settings.txt"
