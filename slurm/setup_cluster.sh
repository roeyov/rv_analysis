#!/bin/bash
# =============================================================================
# First-time setup for the TAU HPC SLURM cluster.
#
# Run this ONCE on the cluster login node (slurmlogin.tau.ac.il)
# after syncing your code. It:
#   1. Loads the system conda module
#   2. Creates a conda environment with pipeline dependencies
#   3. Creates directory structure for data and results
#
# Prerequisites:
#   - SSH access to slurmlogin.tau.ac.il
#   - "power" group membership (request via HPC admins)
#   - Code synced to ~/Scripts/
#
# Usage:
#   ssh roeyovadia@slurmlogin.tau.ac.il
#   bash ~/Scripts/slurm/setup_cluster.sh
# =============================================================================

set -euo pipefail

CONDA_ENV="tau_binary"

echo "========================================"
echo "TAU HPC Cluster Setup for Bias Grid"
echo "========================================"

# --- 1. Load system conda module ---
echo ""
echo "[1/3] Loading system conda module..."
module load miniconda/miniconda3-4.7.12-environmentally

# Set cache dirs to home (shared storage)
export CONDA_PKGS_DIRS=$HOME/.conda/pkgs
export CONDA_ENVS_DIRS=$HOME/.conda/envs
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS"

echo "      Conda loaded. Envs will be stored in ${CONDA_ENVS_DIRS}"

# --- 2. Create conda environment ---
ENV_PATH="${CONDA_ENVS_DIRS}/${CONDA_ENV}"
if [ -d "${ENV_PATH}" ]; then
    echo ""
    echo "[2/3] Conda environment '${CONDA_ENV}' already exists at ${ENV_PATH}"
    echo "      To recreate: rm -rf ${ENV_PATH} && re-run this script"
else
    echo ""
    echo "[2/3] Creating conda environment '${CONDA_ENV}'..."
    conda create -y --prefix "${ENV_PATH}" python=3.11

    conda activate "${ENV_PATH}"

    # Install pipeline dependencies (excluding cloud packages)
    pip install numpy pandas scipy matplotlib plotly astropy \
                lmfit emcee corner tqdm PyYAML numba

    echo "      Environment '${CONDA_ENV}' created and packages installed."
fi

# --- 3. Create directory structure ---
echo ""
echo "[3/3] Creating directory structure..."

mkdir -p "${HOME}/data"
mkdir -p "${HOME}/tables"
mkdir -p "${HOME}/bias_grid_results"
mkdir -p "${HOME}/Scripts/logs"

echo "      Directories created."

# --- Summary ---
echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "Next steps — sync data from your Mac:"
echo ""
echo "  # From your local machine, run:"
echo "  rsync -avz ~/Roey/Masters/Reasearch/Scripts/ roeyovadia@slurmlogin.tau.ac.il:~/Scripts/"
echo "  rsync -avz ~/Roey/Masters/Reasearch/Ostars_article/tables/sb{1,2}_solutions.tex roeyovadia@slurmlogin.tau.ac.il:~/tables/"
echo "  rsync ~/Documents/Data/BLOeM_Data/mass_bloem.csv roeyovadia@slurmlogin.tau.ac.il:~/data/"
echo ""
echo "  # Per-star RV CSVs (needed for rv_err, gamma, field assignment):"
echo "  rsync -avz ~/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_neb_div_from_coadded/ roeyovadia@slurmlogin.tau.ac.il:~/data/rv_csvs/"
echo ""
echo "Then submit the grid search:"
echo ""
echo "  cd ~/Scripts"
echo "  sbatch slurm/submit_bias_grid.slurm"
echo "========================================"
