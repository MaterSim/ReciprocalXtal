#!/usr/bin/env bash
set -e

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE=${SOURCE:-mp}
PRESET=${PRESET:-}
FORMULAS=${FORMULAS:-"SiO2 TiO2"}
SIZE_BUCKETS=${SIZE_BUCKETS:-"small medium large"}
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-benchmark/results_1st-one-0}
MAX_PER_FORMULA=${MAX_PER_FORMULA:-20}
MAX_TOTAL_STRUCTURES=${MAX_TOTAL_STRUCTURES:-60}
NOISE_TIER_COUNT=${NOISE_TIER_COUNT:-3}
NOISE_SEARCH_COMBINED=${NOISE_SEARCH_COMBINED:-"0.002:0.002 0.005:0.005 0.01:0.01 0.015:0.015 0.02:0.02 0.03:0.03 0.05:0.05"}

OPTIONAL_FLAGS=()
if [[ -n "$PRESET" ]]; then
	OPTIONAL_FLAGS+=(--preset "$PRESET")
fi
if [[ "${SKIP_SUPERCELL:-0}" == "1" ]]; then
	OPTIONAL_FLAGS+=(--skip-supercell)
fi
if [[ "${NORMALIZE_RECIPROCAL:-0}" == "1" ]]; then
	OPTIONAL_FLAGS+=(--normalize-reciprocal)
fi

FORMULA_ARGS=()
for formula in ${FORMULAS}; do
	FORMULA_ARGS+=("$formula")
done

MATERIAL_ID_ARGS=()
if [[ -n "${MATERIAL_IDS:-}" ]]; then
	for mid in ${MATERIAL_IDS}; do
		MATERIAL_ID_ARGS+=(--material-id "$mid")
	done
fi

NOISE_SEARCH_ARGS=()
for level in ${NOISE_SEARCH_COMBINED}; do
	NOISE_SEARCH_ARGS+=("$level")
done

for size_bucket in ${SIZE_BUCKETS}; do
	"$PYTHON_BIN" benchmark/scripts/run_structurematcher_vs_reciprocal.py \
		--source "$SOURCE" \
		--dataset-root Fig-6_reconstruction \
		--output-dir "${BASE_OUTPUT_DIR}/${size_bucket}_cell" \
		--api-key "${MP_API_KEY:-}" \
		"${OPTIONAL_FLAGS[@]}" \
		--formula "${FORMULA_ARGS[@]}" \
		--size-bucket "$size_bucket" \
		"${MATERIAL_ID_ARGS[@]}" \
		--max-per-formula "$MAX_PER_FORMULA" \
		--max-total-structures "$MAX_TOTAL_STRUCTURES" \
		--seed 7 \
		--symprec 0.01 \
		--auto-combined-noise-tiers \
		--noise-tier-count "$NOISE_TIER_COUNT" \
		--noise-search-combined "${NOISE_SEARCH_ARGS[@]}" \
		--max-supercell-sites 120 \
		--dmax 10.0 \
		--nmax 10 \
		--lmax 10 \
		--rbasis Bessel \
		--g-bin-width 0.02 \
		--continuous-match-profile normalized \
		--pnl-first-weight 0.0 \
		"$@"
done

