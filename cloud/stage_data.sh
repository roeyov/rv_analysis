#!/usr/bin/env bash
# =============================================================================
# cloud/stage_data.sh — Stage bias_grid input data into the Docker build
# context (<repo>/cloud_data/, gitignored) for the Cloud Batch image.
#
# Mirrors the include-filters of sync_astro3.sh EXACTLY — the RV source dir
# is 174 GB and the SB2 dir 3.5 GB; only the filtered CSVs (a few MB total)
# are staged. Safe to re-run (rsync incremental).
#
# The Dockerfile then does: COPY cloud_data/ /app/data/
# matching the /app/data/... paths in configs/params_bias_cloud*.yaml.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${REPO_ROOT}/cloud_data"

LOCAL_TABLES="${HOME}/Roey/Masters/Reasearch/Ostars_article/tables"
LOCAL_MASS="${HOME}/Documents/Data/BLOeM_Data/mass_bloem.csv"
LOCAL_RV_DIR="${HOME}/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_neb_div_from_coadded"
LOCAL_SB2_DIR="${HOME}/Documents/Data/BLOeM_Data/BLOeM_DR5.0_Combined_perStar_sb2"

mkdir -p "${DEST}/tables"

echo "[1/4] LaTeX tables + catalog -> cloud_data/tables/"
rsync -av \
    "${LOCAL_TABLES}/sb1_solutions.tex" \
    "${LOCAL_TABLES}/sb2_solutions.tex" \
    "${LOCAL_TABLES}/ostar_catalog.csv" \
    "${DEST}/tables/"

echo "[2/4] Mass catalog -> cloud_data/mass_bloem.csv"
rsync -av "${LOCAL_MASS}" "${DEST}/mass_bloem.csv"

echo "[3/4] Per-star RV CSVs -> cloud_data/rv_csvs/  (filtered: *_CCF_RVs.csv)"
rsync -av \
    --include='*/' \
    --include='*_CCF_RVs.csv' \
    --exclude='*' \
    --prune-empty-dirs \
    "${LOCAL_RV_DIR}/" \
    "${DEST}/rv_csvs/"

echo "[4/4] SB2 per-star files -> cloud_data/sb2_per_star/"
rsync -av \
    --include='*/' \
    --include='rv_final_for_mcmc.csv' \
    --include='rv_corrected.csv' \
    --include='rv_extracted.csv' \
    --exclude='*' \
    --prune-empty-dirs \
    "${LOCAL_SB2_DIR}/" \
    "${DEST}/sb2_per_star/"

echo "Done. Staged size: $(du -sh "${DEST}" | cut -f1)"
echo "Next: python -m cloud.run_bias_grid_batch --run-id <id> --build"
