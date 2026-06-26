#!/bin/bash
# Run the full CCF + pipeline workflow for a single star.
#
# Usage:
#   ./run_star.sh 4-043              # HeII + all HeI lines
#   ./run_star.sh 4-043 --nebular    # HeII only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CCF_FIRST="$SCRIPT_DIR/spectroscopy/ccf_input_first.yaml"
CCF_SECOND="$SCRIPT_DIR/spectroscopy/ccf_input_second.yaml"
PARAMS="$SCRIPT_DIR/configs/params.yaml"

# --- Parse arguments ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <star_name> [--nebular]"
    echo "  e.g. $0 4-043"
    echo "  e.g. $0 4-043 --nebular"
    echo "  e.g. $0 4-043 --balmer"
    echo "  e.g. $0 4-043 --he_1       # HeI only (no HeII)"
    exit 1
fi

STAR="$1"
shift
NEBULAR=false
BALMER=false
HE1_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --nebular) NEBULAR=true ;;
        --balmer)  BALMER=true ;;
        --he_1)    HE1_ONLY=true ;;
    esac
done

echo "Star: $STAR  |  Nebular: $NEBULAR  |  Balmer: $BALMER  |  HeI-only: $HE1_ONLY"

# --- Helper: toggle HeI lines in a CCF yaml ---
toggle_hei_lines() {
    local file="$1"
    local hei_keys=("HeI_4471" "HeI_4143" "HeI_4388" "HeI+HeII_4026")

    for key in "${hei_keys[@]}"; do
        if $NEBULAR; then
            # Comment out: only if not already commented
            sed -i '' "/${key}/s/^  \"/  # \"/" "$file"
        else
            # Uncomment: only if currently commented
            sed -i '' "/${key}/s/^  # \"/  \"/" "$file"
        fi
    done
}

# Toggle Balmer (H_*) lines: uncomment if --balmer, comment out otherwise
toggle_balmer_lines() {
    local file="$1"
    local balmer_keys=("H_Gamma" "H_Delta" "H_Epsilon")

    for key in "${balmer_keys[@]}"; do
        if $BALMER; then
            # Uncomment: only if currently commented
            sed -i '' "/${key}/s/^  # \"/  \"/" "$file"
        else
            # Comment out: only if not already commented
            sed -i '' "/${key}/s/^  \"/  # \"/" "$file"
        fi
    done
}

# When --he_1: uncomment HeI_4143/4388, comment out everything else
toggle_he1_only() {
    local file="$1"
    local hei_keys=("HeI_4143" "HeI_4388")
    local other_keys=("HeI_4471" "HeI+HeII_4026" "HeII_4542" "HeII_4200")

    # Uncomment the 3 HeI lines
    for key in "${hei_keys[@]}"; do
        sed -i '' "/${key}/s/^  # \"/  \"/" "$file"
    done
    # Comment out HeII and HeI+HeII lines
    for key in "${other_keys[@]}"; do
        sed -i '' "/${key}/s/^  \"/  # \"/" "$file"
    done
}

# --- Update CCF YAML files ---
for ccf_file in "$CCF_FIRST" "$CCF_SECOND"; do
    sed -i '' "s/list_of_objects: \[.*\]/list_of_objects: [\"BLOeM_${STAR}\"]/" "$ccf_file"
    if $HE1_ONLY; then
        toggle_he1_only "$ccf_file"
    else
        toggle_hei_lines "$ccf_file"
    fi
    toggle_balmer_lines "$ccf_file"
    echo "Updated $(basename "$ccf_file"): object=BLOeM_${STAR}"
done

# --- Run CCF ---
echo ""
echo ">>> Running CCF first pass..."
python -m spectroscopy.ccf_main --input_file "$CCF_FIRST"

echo ""
echo ">>> Running CCF second pass..."
python -m spectroscopy.ccf_main --input_file "$CCF_SECOND"

# --- Update configs/params.yaml ---
sed -i '' "s/object_list: \[.*\]/object_list: ['${STAR}']/" "$PARAMS"
echo ""
echo "Updated configs/params.yaml: object_list=['${STAR}']"

# --- Run pipeline ---
echo ""
echo ">>> Running pipeline evaluator..."
python -m pipeline.evaluator

# --- Show best solution from lmfit_summary.csv ---
echo ""
echo ">>> Best solution from lmfit_summary.csv:"
python -c "
import yaml, pandas as pd, sys

with open('$PARAMS') as f:
    cfg = yaml.safe_load(f)

base_dir = cfg['base_dir']
lmfit_subdir = cfg.get('lmfit_subdir', '')
star = 'BLOeM_${STAR}'
csv_path = f'{base_dir}/{lmfit_subdir}/{star}/lmfit_summary.csv'

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f'lmfit_summary.csv not found at {csv_path}')
    sys.exit(0)

mcmc_cfg = cfg.get('mcmc_params', {})
expr = mcmc_cfg.get('filter_expression', '')
field = mcmc_cfg.get('field_to_check', 'bicc')
take_min = mcmc_cfg.get('take_min', True)

if expr:
    expr = ' '.join(expr.split())
    filtered = df.loc[df.eval(expr)]
else:
    filtered = df

if filtered.empty:
    print('No rows match the filter expression.')
    print(f'Total rows in CSV: {len(df)}')
    sys.exit(0)

if take_min:
    best = filtered.sort_values(field).iloc[0]
else:
    best = filtered.sort_values(field, ascending=False).iloc[0]

sid = int(best['solution_id'])
sol_dir = f'{base_dir}/{lmfit_subdir}/{star}/lmfit_solutions'
png = f'{sol_dir}/{star}_sid-{sid}_phase_residuals.png'

cols = [c for c in best.index if c.endswith('_value') or c.endswith('_iter_fap') or c == 'prob_bicc']
print(f'Rows matching filter: {len(filtered)} / {len(df)}')
print(f'Best by {field} ({"min" if take_min else "max"}):')
print(best[cols].to_string())
print(f'\nPhase plot: {png}')

import subprocess, os
if os.path.exists(png):
    subprocess.Popen(['open', png])
else:
    print(f'PNG not found: {png}')
"

# --- Run MCMC on best solution ---
echo ""
echo ">>> Running MCMC..."
python -c "
import yaml, numpy as np, pandas as pd, sys, os
sys.path.insert(0, '$SCRIPT_DIR')

from pipeline.config import load_args
from mcmc.batch import run_mcmc_single_star, run_mcmc_null_full, load_rv_csv
from mcmc.selector_app import get_best_row

args_dict = load_args('$PARAMS')

with open('$PARAMS') as f:
    import yaml
    cfg = yaml.safe_load(f)

base_dir = cfg['base_dir']
lmfit_subdir = cfg.get('lmfit_subdir', '')
star = 'BLOeM_${STAR}'

csv_path = f'{base_dir}/{lmfit_subdir}/{star}/lmfit_summary.csv'
rv_path = f'{base_dir}/{star}_CCF_RVs.csv'
mcmc_out = f'{base_dir}/{lmfit_subdir}/{star}/mcmc'

results_df = pd.read_csv(csv_path)
MJDs, rv_obs, rv_sigmas = load_rv_csv(rv_path)

mcmc_cfg = cfg.get('mcmc_params', {})
expr = mcmc_cfg.get('filter_expression', '')
field = mcmc_cfg.get('field_to_check', 'bicc')
take_min = mcmc_cfg.get('take_min', True)
if expr:
    expr = ' '.join(expr.split())

best_row = get_best_row(results_df, expr, field, take_min)

os.makedirs(mcmc_out, exist_ok=True)

if best_row is None:
    print('No best row found, running null MCMC...')
    null_row = results_df[
        results_df.candidate_method.str.contains('null_hyp_jitter')
    ].iloc[0]
    run_mcmc_null_full(args_dict, null_row, MJDs, rv_obs, rv_sigmas,
                       mcmc_out, star)
else:
    print(f'Running MCMC for {star} (P={best_row.get(\"Period_value\", \"?\")})...')
    run_mcmc_single_star(args_dict, best_row, MJDs, rv_obs, rv_sigmas,
                         mcmc_out, star)

print(f'MCMC results saved to {mcmc_out}')
"

echo ""
echo "Done: $STAR"
