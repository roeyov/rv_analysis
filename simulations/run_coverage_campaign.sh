#!/bin/bash
# =============================================================================
# run_coverage_campaign.sh — Tier-4 mock-survey coverage campaign (option B).
#
# Outer loop over truth points, inner loop over mock seeds. Per mock:
#   generate synthetic catalog (bias_grid_closure_generate, mock seed)
#     -> anchor the binomial on the mock's own detection count
#        (closure_anchor; before 2026-09-29 every mock used the real 77)
#     -> bias_grid inference (grid seed stays the base config's, e.g. 42)
#     -> closure_compare (argmax diagnostics)
#     -> closure_coverage per-run (68/95 equal-tail containment row).
# Then per-truth aggregate_closure_seeds + campaign-wide coverage table.
#
# Detection-method consistency is enforced BY CONSTRUCTION: the mock
# generator receives the detect_method read from the base config, and the
# grid runs from the same per-seed YAML.
#
# Resumable: a seed with .seed_done is skipped; a crashed grid resumes from
# its det shards on relaunch. Safety rails: mkdir lock, busy-check with
# self-PID exclusion, thread-oversubscription env vars.
#
# Usage (astro3, from ~/Scripts, inside nohup):
#   CONFIG_BASE=configs/params_bias_astro3_ext3_rv10.yaml \
#   nohup ./simulations/run_coverage_campaign.sh \
#       /jonathan_storage/roeyovadia/bias_grid_results/tier4 \
#       > /jonathan_storage/roeyovadia/bias_grid_results/launcher_tier4.log 2>&1 &
#
# Local smoke (sequential; macOS parallel_grid hangs):
#   CONFIG_BASE=configs/params_bias_local_tier4_smoke.yaml \
#   TRUTH_TAGS=smoke_node SEED_START=101 SEED_END=102 \
#   ./simulations/run_coverage_campaign.sh $SCRIPTS_OUT/simulation_pipeline/tier4_smoke
#
# Env knobs: CONFIG_BASE (required), TRUTH_TAGS ("adopted kappa_flat
# kappa_lmc"), SEED_START/SEED_END (101/125), VARIANT, TEST (cvm),
# PRESET (override grid preset), PARALLEL_GRID/N_WORKERS (override),
# CLEAN_SHARDS=1 (default: delete det shards after a seed completes —
# mock shards are expendable, cubes+coverage rows are kept).
# =============================================================================
set -u

CAMPAIGN_DIR="${1:?usage: run_coverage_campaign.sh CAMPAIGN_DIR}"
CONFIG_BASE="${CONFIG_BASE:?set CONFIG_BASE to the base bias YAML}"
TRUTH_TAGS="${TRUTH_TAGS:-adopted kappa_flat kappa_lmc}"
SEED_START="${SEED_START:-101}"
SEED_END="${SEED_END:-125}"
VARIANT="${VARIANT:-eccentric_only__numerical__lucyT}"
TEST="${TEST:-cvm}"
PRESET="${PRESET:-}"
PARALLEL_GRID="${PARALLEL_GRID:-}"
N_WORKERS="${N_WORKERS:-}"
CLEAN_SHARDS="${CLEAN_SHARDS:-1}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

if [ -z "${PY:-}" ]; then
    if command -v micromamba >/dev/null 2>&1; then
        PY="micromamba run -n tau_binary python -u"
    elif command -v conda >/dev/null 2>&1; then
        PY="conda run -n tau_binary python -u"
    else
        PY="python -u"
    fi
fi

mkdir -p "$CAMPAIGN_DIR"
LOG="$CAMPAIGN_DIR/campaign.log"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- Safety rails -----------------------------------------------------------
LOCK="$CAMPAIGN_DIR/.campaign.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    say "BUSY: lock $LOCK exists (another campaign running or crashed; rmdir if stale)"
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# Busy-check with self-PID exclusion (an ssh launch string can match itself).
BUSY=$(pgrep -f "python -u -m simulations.bias_grid" 2>/dev/null | grep -vw "$$" | grep -vw "$PPID" || true)
if [ -n "$BUSY" ]; then
    say "BUSY: bias_grid already running (PIDs: $BUSY) — aborting"
    exit 1
fi

DETECT=$($PY -c "import yaml; print((yaml.safe_load(open('$CONFIG_BASE')).get('bias_grid') or {}).get('detect_method', 'pipeline_batched'))")

say "=== Tier-4 coverage campaign start ==="
say "CAMPAIGN_DIR=$CAMPAIGN_DIR"
say "CONFIG_BASE=$CONFIG_BASE  detect_method=$DETECT  variant=$VARIANT  test=$TEST"
say "truths=[$TRUTH_TAGS]  seeds=$SEED_START..$SEED_END  preset_override=${PRESET:-none}"
say "parallel_override=${PARALLEL_GRID:-none}/${N_WORKERS:-none}  clean_shards=$CLEAN_SHARDS"

FAILED=0
for TAG in $TRUTH_TAGS; do
    case "$TAG" in
        adopted)     TPI=-0.60; TKAPPA=-0.33; TETA=-0.44; TFBIN=0.613 ;;
        kappa_flat)  TPI=-0.60; TKAPPA=0.0;   TETA=-0.44; TFBIN=0.613 ;;
        kappa_lmc)   TPI=-0.60; TKAPPA=-1.0;  TETA=-0.44; TFBIN=0.613 ;;
        custom|smoke_node)
            TPI="${TRUTH_PI:?custom/smoke_node tag needs TRUTH_PI}"
            TKAPPA="${TRUTH_KAPPA:?}"; TETA="${TRUTH_ETA:?}"; TFBIN="${TRUTH_FBIN:?}" ;;
        *) say "ABORT: unknown truth tag '$TAG'"; exit 2 ;;
    esac
    TRUTH_DIR="$CAMPAIGN_DIR/$TAG"
    mkdir -p "$TRUTH_DIR"
    say "--- truth $TAG: pi=$TPI kappa=$TKAPPA eta=$TETA fbin=$TFBIN ---"

    for SEED in $(seq "$SEED_START" "$SEED_END"); do
        SEED_DIR="$TRUTH_DIR/seed_$SEED"
        GRID_DIR="$SEED_DIR/grid_run"
        if [ -f "$SEED_DIR/.seed_done" ]; then
            say "[$TAG/seed $SEED] done sentinel present — skip"
            continue
        fi
        mkdir -p "$SEED_DIR"
        SEED_YAML="$SEED_DIR/params_seed.yaml"

        # Per-seed YAML: patch tables + output dir; grid seed NOT touched.
        cat > "$SEED_DIR/_patch_yaml.py" <<'PY'
import os, yaml
with open(os.environ["CONFIG_BASE"]) as f:
    cfg = yaml.safe_load(f)
bg = cfg.setdefault("bias_grid", {}) or {}
bg["sb1_tex"] = os.path.join(os.environ["SEED_DIR"], "sb1_solutions.tex")
bg["sb2_tex"] = os.path.join(os.environ["SEED_DIR"], "sb2_solutions.tex")
bg["output_dir"] = os.environ["GRID_DIR"]
if os.environ.get("PRESET"):
    bg["preset"] = os.environ["PRESET"]
if os.environ.get("PARALLEL_GRID"):
    bg["parallel_grid"] = os.environ["PARALLEL_GRID"].lower() == "true"
if os.environ.get("N_WORKERS"):
    bg["n_workers"] = int(os.environ["N_WORKERS"])
cfg["bias_grid"] = bg
with open(os.environ["SEED_YAML"], "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
        CONFIG_BASE="$CONFIG_BASE" SEED_DIR="$SEED_DIR" GRID_DIR="$GRID_DIR" \
        SEED_YAML="$SEED_YAML" PRESET="$PRESET" PARALLEL_GRID="$PARALLEL_GRID" \
        N_WORKERS="$N_WORKERS" $PY "$SEED_DIR/_patch_yaml.py" || { say "[$TAG/seed $SEED] YAML patch FAILED"; FAILED=$((FAILED+1)); continue; }

        if [ -f "$SEED_DIR/mock_meta.yaml" ]; then
            say "[$TAG/seed $SEED] mock tables already generated — reuse"
        else
            say "[$TAG/seed $SEED] generating mock catalog ($DETECT)..."
            $PY -m simulations.bias_grid_closure_generate \
                --truth-pi "$TPI" --truth-kappa "$TKAPPA" \
                --truth-eta "$TETA" --truth-fbin "$TFBIN" \
                --seed "$SEED" --detect-method "$DETECT" \
                --config "$SEED_YAML" --output-dir "$SEED_DIR" \
                > "$SEED_DIR/closure_gen.log" 2>&1 \
                || { say "[$TAG/seed $SEED] generation FAILED (see closure_gen.log)"; FAILED=$((FAILED+1)); continue; }
        fi

        # Anchor the binomial on THIS mock's detections (n_stars and
        # n_detected + n_false_positive from truth_params.yaml). Without this
        # step every mock is scored against the real catalogue's 77 (the
        # 2026-09-29 F2 bug; see simulations/closure_anchor.py).
        ANCHOR=$($PY -m simulations.closure_anchor --seed-yaml "$SEED_YAML" \
                    --truth-yaml "$SEED_DIR/truth_params.yaml" 2>>"$SEED_DIR/closure_gen.log") \
            || { say "[$TAG/seed $SEED] binomial-anchor patch FAILED"; FAILED=$((FAILED+1)); continue; }
        say "[$TAG/seed $SEED] $ANCHOR"

        say "[$TAG/seed $SEED] running bias_grid..."
        $PY -m simulations.bias_grid --config "$SEED_YAML" \
            > "$SEED_DIR/grid_run.log" 2>&1
        RC=$?
        if [ $RC -ne 0 ] || [ ! -f "$GRID_DIR/grid_cubes.npz" ]; then
            say "[$TAG/seed $SEED] bias_grid FAILED rc=$RC (see grid_run.log) — no sentinel, will resume on relaunch"
            FAILED=$((FAILED+1)); continue
        fi

        say "[$TAG/seed $SEED] closure compare + coverage row..."
        $PY -m simulations.bias_grid_closure_compare \
            --grid-dir "$GRID_DIR" --truth-yaml "$SEED_DIR/truth_params.yaml" \
            --test "$TEST" --variant "$VARIANT" \
            >> "$SEED_DIR/closure_gen.log" 2>&1 \
            || { say "[$TAG/seed $SEED] closure_compare FAILED"; FAILED=$((FAILED+1)); continue; }
        $PY -m simulations.closure_coverage per-run \
            --grid-dir "$GRID_DIR" --truth-yaml "$SEED_DIR/truth_params.yaml" \
            --truth-tag "$TAG" --seed "$SEED" \
            --variant "$VARIANT" --test "$TEST" \
            >> "$SEED_DIR/closure_gen.log" 2>&1 \
            || { say "[$TAG/seed $SEED] closure_coverage FAILED"; FAILED=$((FAILED+1)); continue; }

        touch "$SEED_DIR/.seed_done"
        if [ "$CLEAN_SHARDS" = "1" ]; then
            rm -rf "$GRID_DIR/det_shards" "$GRID_DIR/checkpoint_cubes.npz" \
                   "$GRID_DIR/checkpoint_hists.npz" "$GRID_DIR/checkpoint_results.csv"
        fi
        say "[$TAG/seed $SEED] DONE"
    done

    say "[$TAG] per-truth argmax aggregate:"
    $PY -m simulations.aggregate_closure_seeds --base-dir "$TRUTH_DIR" \
        --test "$TEST" --variant "$VARIANT" \
        --out "$TRUTH_DIR/per_seed_summary.csv" 2>&1 | tee -a "$LOG" || true
done

say "=== campaign coverage table ==="
$PY -m simulations.closure_coverage aggregate \
    --campaign-dir "$CAMPAIGN_DIR" --out "$CAMPAIGN_DIR/coverage_table.csv" \
    2>&1 | tee -a "$LOG"

say "=== Tier-4 coverage campaign finished (failed steps: $FAILED) ==="
exit $([ "$FAILED" -eq 0 ] && echo 0 || echo 1)
