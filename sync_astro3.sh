#!/usr/bin/env bash
# =============================================================================
# sync_astro3.sh — Push code + bias_grid input data from local Mac to astro3.
#
# rsync is incremental by default: only changed bytes are transferred, so this
# is the "update only when changed" mechanism. Safe to re-run any time.
#
# Pulls results back when called as: ./sync_astro3.sh pull
# =============================================================================

set -euo pipefail

REMOTE="${ASTRO3_REMOTE:-roeyovadia@astro3.tau.ac.il}"

LOCAL_REPO="${HOME}/Roey/Masters/Reasearch/Scripts"
LOCAL_TABLES="${HOME}/Roey/Masters/Reasearch/Ostars_article/tables"
LOCAL_MASS="${HOME}/Documents/Data/BLOeM_Data/mass_bloem.csv"
LOCAL_RV_DIR="${HOME}/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_neb_div_from_coadded"
LOCAL_SB2_DIR="${HOME}/Documents/Data/BLOeM_Data/BLOeM_DR5.0_Combined_perStar_sb2"
LOCAL_RESULTS="${HOME}/Roey/Masters/Reasearch/scriptsOut/bias_grid"

push() {
    echo "[1/5] Ensuring remote dirs exist on ${REMOTE}..."
    ssh "${REMOTE}" "mkdir -p ~/Scripts ~/data ~/tables ~/bias_grid_results"

    echo "[2/5] Syncing code -> ~/Scripts/"
    rsync -avz --delete \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='scriptsOut/' \
        --exclude='.claude/' \
        --exclude='.venv/' \
        --exclude='.idea/' \
        "${LOCAL_REPO}/" \
        "${REMOTE}:~/Scripts/"

    echo "[3/5] Syncing LaTeX tables + catalog -> ~/tables/"
    rsync -avz \
        "${LOCAL_TABLES}/sb1_solutions.tex" \
        "${LOCAL_TABLES}/sb2_solutions.tex" \
        "${LOCAL_TABLES}/ostar_catalog.csv" \
        "${REMOTE}:~/tables/"

    echo "      Syncing mass catalog -> ~/data/"
    rsync -avz "${LOCAL_MASS}" "${REMOTE}:~/data/mass_bloem.csv"

    echo "[4/5] Syncing per-star RV CSVs -> ~/data/rv_csvs/"
    rsync -avz \
        --include='*/' \
        --include='*_CCF_RVs.csv' \
        --exclude='*' \
        "${LOCAL_RV_DIR}/" \
        "${REMOTE}:~/data/rv_csvs/"

    echo "[5/5] Syncing SB2 per-star dirs -> ~/data/sb2_per_star/"
    rsync -avz \
        --include='*/' \
        --include='rv_final_for_mcmc.csv' \
        --include='rv_corrected.csv' \
        --include='rv_extracted.csv' \
        --exclude='*' \
        "${LOCAL_SB2_DIR}/" \
        "${REMOTE}:~/data/sb2_per_star/"

    echo "Done. Next: ssh ${REMOTE} and run setup_astro3.sh (first time only)."
}

pull() {
    mkdir -p "${LOCAL_RESULTS}"
    echo "Pulling ~/bias_grid_results -> ${LOCAL_RESULTS}"
    rsync -avz "${REMOTE}:~/bias_grid_results/" "${LOCAL_RESULTS}/"
}

case "${1:-push}" in
    push) push ;;
    pull) pull ;;
    *) echo "Usage: $0 [push|pull]"; exit 1 ;;
esac
