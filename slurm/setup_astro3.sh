#!/usr/bin/env bash
# =============================================================================
# setup_astro3.sh — One-time bootstrap on astro3.tau.ac.il.
#
# astro3 has GLIBC 2.17 (CentOS 7-era), too old for the latest Miniconda
# installer (needs GLIBC >= 2.28). We install micromamba — a single static
# binary with no GLIBC dependency — then create the `tau_binary` env.
#
# Idempotent: re-running it skips already-installed steps and just runs
# `pip install -U` for the dependency list.
#
# Usage (on astro3, after code has been rsynced to ~/Scripts/):
#   bash ~/Scripts/slurm/setup_astro3.sh
# =============================================================================

set -euo pipefail

ENV_NAME="tau_binary"
MAMBA_ROOT_PREFIX="${HOME}/micromamba"
MAMBA_BIN="${HOME}/.local/bin/micromamba"

echo "========================================"
echo "astro3 setup — bias_grid"
echo "========================================"

# --- 1. Install micromamba if missing ---
if [ ! -x "${MAMBA_BIN}" ]; then
    echo "[1/3] Installing micromamba..."
    mkdir -p "${HOME}/.local/bin"
    cd "${HOME}"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj bin/micromamba
    mv "${HOME}/bin/micromamba" "${MAMBA_BIN}"
    rmdir "${HOME}/bin" 2>/dev/null || true

    if ! grep -q 'micromamba shell init' "${HOME}/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${HOME}/.bashrc"
        "${MAMBA_BIN}" shell init -s bash -r "${MAMBA_ROOT_PREFIX}" >/dev/null
        echo "      Shell init added to ~/.bashrc — open a new shell after this script."
    fi
else
    echo "[1/3] micromamba already installed at ${MAMBA_BIN}"
fi

export MAMBA_ROOT_PREFIX

# --- 2. Create or update the env ---
#
# astro3 has GCC 4.8.5 and GLIBC 2.17. PyPI wheels for numpy/scipy/etc. now
# target manylinux_2_28 (GLIBC >= 2.28), so pip would try to build from source
# and fail. conda-forge still publishes linux-64 binaries against the older
# GLIBC 2.17 baseline, so we install all native deps via micromamba and only
# pure-Python packages via pip.
if "${MAMBA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[2/3] Env '${ENV_NAME}' exists — updating dependencies."
else
    echo "[2/3] Creating env '${ENV_NAME}' (python=3.11)..."
    "${MAMBA_BIN}" create -y -n "${ENV_NAME}" -c conda-forge python=3.11
fi

# Native deps (binaries from conda-forge — built against GLIBC 2.17).
"${MAMBA_BIN}" install -y -n "${ENV_NAME}" -c conda-forge \
    numpy pandas scipy matplotlib astropy numba

# Pure-Python deps (small, safe to pip-install).
"${MAMBA_BIN}" run -n "${ENV_NAME}" pip install --upgrade \
    plotly lmfit emcee corner tqdm PyYAML

# --- 3. Create directory structure ---
echo "[3/3] Creating data/result directories..."
mkdir -p "${HOME}/data" "${HOME}/tables" "${HOME}/bias_grid_results"

echo ""
echo "========================================"
echo "Setup complete."
echo ""
echo "Activate the env:    micromamba activate ${ENV_NAME}"
echo "Smoke test:          python -c 'import numpy, lmfit, emcee, numba; print(\"ok\")'"
echo "Run bias_grid:       cd ~/Scripts && python -m simulations.bias_grid --config configs/params_bias_astro3.yaml"
echo "========================================"
