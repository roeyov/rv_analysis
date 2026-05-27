#!/bin/bash
# Run N closure seeds end-to-end: generate synthetic catalog -> bias_grid ->
# closure_compare. Skips seeds whose grid_run already has a closure_summary.csv
# (resumable). Logs each step to a master log + per-seed sub-logs.
#
# Usage:
#   bash simulations/run_closure_seeds.sh [BASE_DIR] [SEED_START] [SEED_END] [E_SCORE_MODE]
#
# Defaults: BASE_DIR=$SCRIPTS_OUT/simulation_pipeline/bias_grid_closure/sana2012_10seeds_n134,
#           seeds 42..51, E_SCORE_MODE=split.
# Outputs land outside the repo (see CLAUDE.md "Outputs Convention").
#
# bias_grid.py is YAML-only — this script copies configs/params_bias.yaml per seed
# and patches the bias_grid: block with per-seed knobs (seed, output_dir,
# sb1/sb2 paths, e_score_mode, parallelism). closure_generate reads the
# same patched YAML so it pulls the 134-star sample.

set -euo pipefail

SCRIPTS_OUT="${SCRIPTS_OUT:-/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut}"
BASE_DIR="${1:-$SCRIPTS_OUT/simulation_pipeline/bias_grid_closure/sana2012_10seeds_n134}"
SEED_START="${2:-42}"
SEED_END="${3:-51}"
E_SCORE_MODE="${4:-split}"

CONFIG_BASE="${CONFIG_BASE:-configs/params_bias.yaml}"
TRUTH_PI="${TRUTH_PI:--0.55}"
TRUTH_KAPPA="${TRUTH_KAPPA:--0.10}"
TRUTH_ETA="${TRUTH_ETA:--0.45}"
TRUTH_FBIN="${TRUTH_FBIN:-0.69}"
PRESET="${PRESET:-final3_10min2}"
DETECT_METHOD="${DETECT_METHOD:-rv_threshold}"
N_WORKERS="${N_WORKERS:-8}"

mkdir -p "$BASE_DIR"
MASTER_LOG="$BASE_DIR/run.log"
echo "=== Closure 10-seed sweep starting $(date) ===" | tee -a "$MASTER_LOG"
echo "BASE_DIR=$BASE_DIR  seeds=$SEED_START..$SEED_END  preset=$PRESET  e_score_mode=$E_SCORE_MODE" | tee -a "$MASTER_LOG"
echo "CONFIG_BASE=$CONFIG_BASE  detect=$DETECT_METHOD  n_workers=$N_WORKERS" | tee -a "$MASTER_LOG"
echo "TRUTH: pi=$TRUTH_PI kappa=$TRUTH_KAPPA eta=$TRUTH_ETA fbin=$TRUTH_FBIN" | tee -a "$MASTER_LOG"

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    SEED_DIR="$BASE_DIR/seed_$SEED"
    GRID_DIR="$SEED_DIR/grid_run"
    SUMMARY="$GRID_DIR/closure_report/closure_summary.csv"
    SEED_YAML="$SEED_DIR/params_seed.yaml"

    if [ -f "$SUMMARY" ]; then
        echo "[seed $SEED] already complete: $SUMMARY" | tee -a "$MASTER_LOG"
        continue
    fi

    mkdir -p "$SEED_DIR"

    # Emit per-seed YAML: copy base, patch bias_grid: block.
    SEED="$SEED" \
    PRESET="$PRESET" \
    DETECT_METHOD="$DETECT_METHOD" \
    E_SCORE_MODE="$E_SCORE_MODE" \
    N_WORKERS="$N_WORKERS" \
    SEED_DIR="$SEED_DIR" \
    GRID_DIR="$GRID_DIR" \
    CONFIG_BASE="$CONFIG_BASE" \
    SEED_YAML="$SEED_YAML" \
    python - <<'PY'
import os, yaml
with open(os.environ["CONFIG_BASE"]) as f:
    cfg = yaml.safe_load(f)
bg = cfg.setdefault("bias_grid", {}) or {}
bg["seed"] = int(os.environ["SEED"])
bg["preset"] = os.environ["PRESET"]
bg["detect_method"] = os.environ["DETECT_METHOD"]
bg["e_score_mode"] = os.environ["E_SCORE_MODE"]
bg["parallel_grid"] = True
bg["n_workers"] = int(os.environ["N_WORKERS"])
bg["sb1_tex"] = os.path.join(os.environ["SEED_DIR"], "sb1_solutions.tex")
bg["sb2_tex"] = os.path.join(os.environ["SEED_DIR"], "sb2_solutions.tex")
bg["output_dir"] = os.environ["GRID_DIR"]
cfg["bias_grid"] = bg
with open(os.environ["SEED_YAML"], "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    echo "[seed $SEED] $(date) generating closure..." | tee -a "$MASTER_LOG"
    python -m simulations.bias_grid_closure_generate \
        --truth-pi "$TRUTH_PI" --truth-kappa "$TRUTH_KAPPA" \
        --truth-eta "$TRUTH_ETA" --truth-fbin "$TRUTH_FBIN" \
        --seed "$SEED" \
        --config "$SEED_YAML" \
        --output-dir "$SEED_DIR" \
        > "$SEED_DIR/closure_gen.log" 2>&1

    echo "[seed $SEED] $(date) running bias_grid..." | tee -a "$MASTER_LOG"
    python -m simulations.bias_grid \
        --config "$SEED_YAML" \
        > "$SEED_DIR/grid_run.log" 2>&1

    echo "[seed $SEED] $(date) closure compare..." | tee -a "$MASTER_LOG"
    python -m simulations.bias_grid_closure_compare \
        --grid-dir "$GRID_DIR" \
        --truth-yaml "$SEED_DIR/truth_params.yaml" \
        2>&1 | tee -a "$MASTER_LOG"
done

echo "=== Closure 10-seed sweep finished $(date) ===" | tee -a "$MASTER_LOG"
echo "Aggregate with: python -m simulations.aggregate_closure_seeds --base-dir $BASE_DIR"
