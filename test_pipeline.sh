#!/usr/bin/env bash
# =============================================================================
# test_pipeline.sh — Run each pipeline stage on one star to verify refactoring
#
# Usage:
#   cd /Users/roeyovadia/Roey/Masters/Reasearch/Scripts/.claude/worktrees/unruffled-yalow
#   conda activate tau_binary
#   bash test_pipeline.sh
#
# Each stage can also be run independently by copying the relevant section.
# =============================================================================

set -euo pipefail

PYTHON="/Users/roeyovadia/miniconda3/envs/tau_binary/bin/python"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# --- Paths (adjust if needed) ---
STAR="BLOeM_8-020"
FITS_DIR="/Users/roeyovadia/Documents/Data/BLOeM_Data/BLOeM_DR5.0_Combined"
CCF_OUT="/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_neb_div_from_coadded"
LMFIT_OUT="${CCF_OUT}/second"
MCMC_OUT="${LMFIT_OUT}/mcmc_test"
PARAMS_YAML="${ROOT}/configs/params.yaml"

echo "========================================"
echo "  Pipeline Verification — ${STAR}"
echo "  Root: ${ROOT}"
echo "========================================"
echo ""

# =============================================================================
# STAGE 0: Import smoke test (fast — no data needed)
# =============================================================================
echo "=== STAGE 0: Import smoke test ==="

$PYTHON -c "
from orbital.kepler import true_anomaly, kepler_newton
from orbital.fitting import lmfit_on_sample
from orbital.statistics import calculate_binary_probability, DecsionFlags
from orbital.plotting import print_lmfit_result

from period_search.periodogram import ls, pdc_opt
from period_search.candidates import find_periods

from pipeline.data_loading import load_final_data_from_ccf_out
from pipeline.config import PARAM_FILE, load_args
from pipeline.evaluator import main_single, main_multiple

from mcmc.models import rv_model, log_probability
from mcmc.runner import run_mcmc_ecc, run_mcmc_circ, run_mcmc_null
from mcmc.analysis import lucy_sweeney_significant, summarise_chain
from mcmc.mcmc_plotting import make_corner, plot_orbit_with_band_phase
from mcmc.batch import run_mcmc_batch, run_mcmc_single_star
from mcmc.selector_app import get_best_row

from spectroscopy.ccf_core import CCF, cross_cor
from spectroscopy.equivalent_width import calculate_equivalent_width2
from spectroscopy.broadening import rotational_broadening
from spectroscopy.coaddition import create_coadded_spectra
from spectroscopy.plotting import plot_rv_vs_mjd
from spectroscopy.ccf_main import main as ccf_main

from utils.roche_lobe import compute_min_period_row
from utils.constants import SNR_PPL

from simulations.common import (
    BLOEM_MJD_ARRAYS, simulate_system_refined, simulate_system,
    sample_gamma, RV12, nu_func, get_rv_amplitudes,
    uniform_random_sample, sine_inclination_sample,
    ostar_radius_series_from_mass,
)
from simulations.create_binary_simulations import generate_binary_rv_at_mjds
from simulations.make_rvs import out_multiple_and_dump

# Verify YAML config loads correctly
cfg = load_args()
assert 'pipeline_io' in cfg, 'pipeline_io missing from YAML'
assert 'mcmc_params' in cfg, 'mcmc_params missing from YAML'
assert 'rv_input_dir' in cfg['mcmc_params'], 'MCMC I/O missing from YAML'

# Quick simulation sanity check
import numpy as _np
_mjds = BLOEM_MJD_ARRAYS[0]
_orb = {'t0': 0.0, 'period': 30.0, 'ecc': 0.3, 'omega': 1.0,
        'k1': 15.0, 'k2': 7.0, 'gamma': 168.0}
_rvs, _sigs = generate_binary_rv_at_mjds(_mjds, _orb)
assert _rvs is not None, 'Binary sim returned None'
assert len(_rvs) == len(_mjds), 'Binary sim length mismatch'

print('All imports OK')
print(f'Default config: {PARAM_FILE}')
"

echo "  ✓ All imports passed"
echo ""

# =============================================================================
# STAGE 1: CCF spectral analysis → RV extraction
# =============================================================================
echo "=== STAGE 1: CCF — Radial velocity extraction ==="
echo "  Entry point: python -m spectroscopy.ccf_main --input_file spectroscopy/ccf_input.yaml"
echo "  Input:  FITS spectra from ${FITS_DIR}"
echo "  Output: *_CCF_RVs.csv, *_CoAdded.csv"
echo ""
echo "  Command:"
echo "    $PYTHON -m spectroscopy.ccf_main --input_file spectroscopy/ccf_input.yaml"
echo ""
echo "  NOTE: Requires FITS data on disk. Skipping auto-run."
echo ""
echo "  Verify output exists from previous run:"

if [ -f "${CCF_OUT}/${STAR}_CCF_RVs.csv" ]; then
    echo "    ✓ ${STAR}_CCF_RVs.csv exists ($(wc -l < "${CCF_OUT}/${STAR}_CCF_RVs.csv") lines)"
    echo "    Columns: $(head -1 "${CCF_OUT}/${STAR}_CCF_RVs.csv" | tr ',' '\n' | wc -l)"
else
    echo "    ✗ ${STAR}_CCF_RVs.csv not found — run Stage 1 first"
fi
echo ""

# =============================================================================
# STAGE 2: Period search + orbital fitting
# =============================================================================
echo "=== STAGE 2: Period search + orbital fitting ==="
echo "  Entry point: python -m pipeline.evaluator --config configs/params.yaml"
echo "  Input:  *_CCF_RVs.csv + configs/params.yaml (pipeline_io section)"
echo "  Output: {star}/lmfit_summary.csv, periodogram/, lmfit_solutions/"
echo ""
echo "  Command:"
echo "    $PYTHON -m pipeline.evaluator --config ${PARAMS_YAML}"
echo ""
echo "  Verify output exists from previous run:"

if [ -f "${LMFIT_OUT}/${STAR}/lmfit_summary.csv" ]; then
    N_SOLUTIONS=$(tail -n +2 "${LMFIT_OUT}/${STAR}/lmfit_summary.csv" | wc -l)
    echo "    ✓ lmfit_summary.csv exists (${N_SOLUTIONS} solution rows)"
    echo "    ✓ Columns: $(head -1 "${LMFIT_OUT}/${STAR}/lmfit_summary.csv" | tr ',' '\n' | wc -l)"
else
    echo "    ✗ lmfit_summary.csv not found — run Stage 2 first"
fi

if [ -d "${LMFIT_OUT}/${STAR}/lmfit_solutions" ]; then
    echo "    ✓ lmfit_solutions/ has $(ls "${LMFIT_OUT}/${STAR}/lmfit_solutions/" | wc -l) files"
fi

if [ -d "${LMFIT_OUT}/${STAR}/periodogram" ]; then
    echo "    ✓ periodogram/ has $(ls "${LMFIT_OUT}/${STAR}/periodogram/" | wc -l) files"
fi
echo ""

# =============================================================================
# STAGE 3: Interactive threshold assessment (Streamlit)
# =============================================================================
echo "=== STAGE 3: Interactive threshold assessment (Streamlit) ==="
echo "  Entry point: mcmc.selector_app"
echo "  Input:  {star}/lmfit_summary.csv directory"
echo ""
echo "  Command:"
echo "    streamlit run ${ROOT}/mcmc/selector_app.py"
echo ""
echo "  This opens a browser UI. Set 'Solution directory' to:"
echo "    ${LMFIT_OUT}"
echo ""
echo "  Quick non-interactive test (get_best_row on one star):"

$PYTHON -c "
import pandas as pd
from mcmc.selector_app import get_best_row

df = pd.read_csv('${LMFIT_OUT}/${STAR}/lmfit_summary.csv')
print(f'  Loaded {len(df)} rows for ${STAR}')

best = get_best_row(
    df,
    filter_expr=(
        \"~candidate_method.str.contains('MANUAL')\"
        \"& ~candidate_method.str.contains('null')\"
        \"& candidate_method.str.contains('jitter')\"
    ),
    field_to_check='bic',
    take_min=True,
)
if best is not None:
    print(f'  ✓ Best row: P={best.get(\"Period_value\", \"?\"):.4f} d, '
          f'e={best.get(\"Eccentricity_value\", \"?\"):.4f}, '
          f'K1={best.get(\"K1_value\", \"?\"):.2f} km/s, '
          f'BIC={best.get(\"bic\", \"?\"):.2f}')
else:
    print('  ✗ No valid row found with these filters')
"
echo ""

# =============================================================================
# STAGE 4: MCMC posterior sampling
# =============================================================================
echo "=== STAGE 4: MCMC posterior sampling ==="
echo "  Entry point: python -m mcmc.batch --config configs/params.yaml"
echo "  Input:  *_CCF_RVs.csv + lmfit_summary.csv + configs/params.yaml (mcmc_params section)"
echo "  Output: {star}/corner_ecc.png, phase_ecc.png, results_ecc.txt, ..."
echo ""
echo "  Command:"
echo "    $PYTHON -m mcmc.batch --config ${PARAMS_YAML}"
echo ""
echo "  NOTE: MCMC takes ~5-15 min per star (2000 steps x 32 walkers)."
echo "  For a quick test, reduce steps in configs/params.yaml:"
echo "    mcmc_params.steps: 200"
echo "    mcmc_params.walkers: 16"
echo ""

# Check existing MCMC output
if [ -d "${LMFIT_OUT}/${STAR}/mcmc" ]; then
    echo "  Existing MCMC output:"
    ls "${LMFIT_OUT}/${STAR}/mcmc/" 2>/dev/null | head -10 | sed 's/^/    /'
fi
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "========================================"
echo "  Pipeline Stage Summary"
echo "========================================"
echo ""
echo "  Stage 1 (CCF):       python -m spectroscopy.ccf_main --input_file spectroscopy/ccf_input.yaml"
echo "  Stage 2 (Fit):       python -m pipeline.evaluator    --config configs/params.yaml"
echo "  Stage 3 (Select):    streamlit run mcmc/selector_app.py"
echo "  Stage 4 (MCMC):      python -m mcmc.batch            --config configs/params.yaml"
echo ""
echo "  All commands assume:  cd ${ROOT}"
echo "                        conda activate tau_binary"
echo ""
echo "Done."
