#!/bin/bash
# =============================================================================
# run_variant_rescore_chain.sh — 2026-09 bias-grid variant campaign (astro3).
#
# Sequentially re-scores the ed_hoc_rlof_check_ext3 det shards under every
# offline variant (fresh dir + det_shards symlink -> the engine's resume
# "gap re-score" recomputes all cubes with ZERO injections, ~15 min each),
# then runs the rv_threshold quartet (fresh short runs, ~5-10 min each).
#
# Safety rails (agreed before any science variant ran; hardened after the
# 2026-09-02 adversarial review):
#   * mkdir-based LOCK + busy-check: one chain at a time, no races.
#   * pre-launch preflight: the config's preset grid must match ext3's AND
#     the config must be parallel_grid + n_workers>1 (the sequential engine
#     path would REWRITE shards; the engine itself also hard-refuses both
#     cases on a symlinked det_shards since the same review).
#   * post-run assertion that the forward pass had NOTHING to inject.
#   * completion sentinel .chain_verified: a step is SKIPped only when its
#     sentinel exists. grid_cubes.npz without a sentinel = failed/unverified
#     attempt -> quarantined to grid_cubes.FAILED.npz and the step reruns.
#     (grid_cubes.npz is written BEFORE the verify gate runs, so its mere
#     existence proves nothing.)
#   * never resume a re-score dir: stale checkpoint files are wiped before
#     each attempt (a crashed re-score is rerun from scratch, it's cheap).
#   * parity gates first: the unchanged-settings re-score plus the
#     manual==elbow re-score must PASS verify_rescore_parity (bit-identity)
#     before any science variant is scored.
#
# NOTE: the ~24 h ext3_qmin01 companion run is NOT part of this chain — it
# is launched separately (announced) after the chain completes.
#
# Usage (from ~/Scripts on astro3):
#   nohup ./simulations/run_variant_rescore_chain.sh \
#       > ~/bias_grid_results/launcher_variant_chain.log 2>&1 &
# =============================================================================
set -u

BASE=/jonathan_storage/roeyovadia/bias_grid_results
REF=$BASE/ed_hoc_rlof_check_ext3
LOCK=$BASE/.variant_chain.lock
cd "$(dirname "$0")/.." || exit 1

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "BUSY: lock $LOCK exists (another chain running or crashed; rmdir it if stale)"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

if pgrep -f "python -u -m simulations.bias_grid" >/dev/null; then
  echo "BUSY: a bias_grid process is already running — aborting"
  exit 1
fi
if [ ! -f "$REF/grid_cubes.npz" ]; then
  echo "ABORT: $REF/grid_cubes.npz missing — ext3 not finished/aggregated"
  exit 1
fi

PY="micromamba run -n tau_binary python -u"

preflight_check () {
  # Refuse to launch a re-score whose config (a) resolves to a different
  # grid than the reference run (the forward pass would INJECT new points
  # and write their shards through the det_shards symlink into ext3's real
  # shard dir), or (b) is not in parallel_grid/n_workers>1 mode (the
  # sequential engine path re-injects and REWRITES every shard). The
  # engine refuses both on a symlinked det_shards too — this preflight
  # just fails before any compute is wasted.
  local config=$1
  $PY - "$config" "$REF/run_config.yaml" <<'EOF'
import sys, yaml, numpy as np
from simulations.bias_config import GRID_PRESETS
cfg = yaml.safe_load(open(sys.argv[1]))
ref = yaml.safe_load(open(sys.argv[2]))
bg = cfg.get("bias_grid") or {}
preset = bg.get("preset")
if preset not in GRID_PRESETS:
    sys.exit("preflight: unknown preset %r" % preset)
if not bg.get("parallel_grid", False):
    sys.exit("preflight: parallel_grid must be true for a re-score "
             "(sequential mode rewrites shards)")
nw = bg.get("n_workers")
if nw is not None and int(nw) <= 1:
    sys.exit("preflight: n_workers must be > 1 for a re-score")
p = GRID_PRESETS[preset]
rp = ref["preset"]
for axis in ("pi", "kappa", "eta", "fbin"):
    a = np.asarray(p[axis], dtype=float)
    b = np.asarray(rp[axis], dtype=float)
    if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-12):
        sys.exit("preflight: %s axis differs from reference grid "
                 "(%s vs %s pts) — would inject through the symlink"
                 % (axis, a.shape, b.shape))
print("preflight: grid matches reference (%s), parallel mode ok" % preset)
EOF
}

quarantine () {
  local dir=$1
  if [ -f "$dir/grid_cubes.npz" ]; then
    mv "$dir/grid_cubes.npz" "$dir/grid_cubes.FAILED.npz"
    echo "  quarantined $dir/grid_cubes.npz -> grid_cubes.FAILED.npz"
  fi
  rm -f "$dir/.chain_verified"
}

run_rescore () {
  # $1 = variant name (config configs/params_bias_astro3_<name>.yaml,
  #      output dir $BASE/<name>)
  # $2 = verify mode for verify_rescore_parity ('' = skip verification)
  local name=$1 mode=${2:-}
  local dir=$BASE/$name
  local log=$BASE/launcher_${name}.log
  if [ -f "$dir/.chain_verified" ]; then
    echo "SKIP $name (.chain_verified sentinel present)"
    return 0
  fi
  if [ -f "$dir/grid_cubes.npz" ]; then
    echo "STALE $name: grid_cubes.npz without sentinel (failed/unverified "
    echo "  earlier attempt) — quarantining and rerunning"
    quarantine "$dir"
  fi
  if ! preflight_check "configs/params_bias_astro3_${name}.yaml"; then
    echo "ABORT $name: preflight failed — NOT launching"
    return 2
  fi
  mkdir -p "$dir"
  # Never resume: wipe any partial state, keep only the shard symlink.
  rm -f "$dir/checkpoint_cubes.npz" "$dir/checkpoint_results.csv" \
        "$dir/det_index.npz" "$dir/grid_hists.npz" \
        "$dir/checkpoint_hists.npz" "$dir/run_config.yaml"
  if [ -e "$dir/det_shards" ]; then
    if [ "$(readlink "$dir/det_shards")" != "$REF/det_shards" ]; then
      echo "ABORT $name: $dir/det_shards exists but is not a symlink to "
      echo "  $REF/det_shards — refusing to touch it"
      return 2
    fi
  else
    ln -s "$REF/det_shards" "$dir/det_shards" || return 1
  fi
  echo "=== $(date '+%F %T') re-scoring $name ==="
  $PY -m simulations.bias_grid \
      --config "configs/params_bias_astro3_${name}.yaml" > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAIL $name: bias_grid exited rc=$rc (see $log)"
    quarantine "$dir"
    return 1
  fi
  if ! grep -q "Parallel grid mode: 0 grid points to compute" "$log"; then
    echo "ABORT $name: forward pass wanted to inject — preset/grid mismatch"
    echo "  (the engine's symlink guard should have refused the writes;"
    echo "   verify $REF/det_shards integrity before rerunning ANYTHING)"
    quarantine "$dir"
    return 2
  fi
  if [ -n "$mode" ]; then
    if ! $PY -m simulations.verify_rescore_parity \
          --ref "$REF" --test "$dir" --mode "$mode"; then
      echo "VERIFY FAIL $name (mode=$mode)"
      quarantine "$dir"
      return 1
    fi
  fi
  touch "$dir/.chain_verified"
  echo "=== $(date '+%F %T') $name done ==="
  return 0
}

run_forward () {
  # Fresh short runs (rv_threshold quartet): normal launches, own shards,
  # no symlink. Same sentinel discipline.
  local name=$1
  local dir=$BASE/$name
  local log=$BASE/launcher_${name}.log
  if [ -f "$dir/.chain_verified" ]; then
    echo "SKIP $name (.chain_verified sentinel present)"
    return 0
  fi
  echo "=== $(date '+%F %T') launching $name ==="
  $PY -m simulations.bias_grid \
      --config "configs/params_bias_astro3_${name}.yaml" > "$log" 2>&1
  local rc=$?
  echo "=== $(date '+%F %T') $name exited rc=$rc ==="
  if [ $rc -eq 0 ] && [ -f "$dir/grid_cubes.npz" ]; then
    touch "$dir/.chain_verified"
    return 0
  fi
  quarantine "$dir"
  return 1
}

# ---- Parity gates (both must pass before any science variant) --------------
run_rescore ext3_rescore_parity       parity        || exit 1
run_rescore ext3_rescore_pstar_elbow  manual_parity || exit 1

# ---- Science re-scores -----------------------------------------------------
# rc=2 (abort: preflight/grid-mismatch/foreign det_shards) hard-stops the
# chain; rc=1 (soft failure, quarantined) accumulates and is reported.
FAILED=""
step () {
  run_rescore "$1" "$2"
  local rc=$?
  if [ $rc -eq 2 ]; then
    echo "HARD STOP: $1 aborted on a safety guard"
    exit 2
  fi
  [ $rc -ne 0 ] && FAILED="$FAILED $1"
  return 0
}
step ext3_rescore_env5         env
step ext3_rescore_binom_none   binom
step ext3_rescore_binom_sana   ''
step ext3_rescore_binom_n72    ''
step ext3_rescore_weakK1_k10   ''
step ext3_rescore_weakK1_c1    ''
step ext3_rescore_pstar_2d     ''
step ext3_rescore_pstar_5d     ''
step ext3_rescore_sig010       ''
step ext3_rescore_sig020       ''
if [ -f /jonathan_storage/roeyovadia/tables/sb1_solutions_inner5.tex ]; then
  step ext3_rescore_inner5     ''
else
  echo "SKIP ext3_rescore_inner5 (inner-orbit tables not synced yet)"
fi

# ---- rv_threshold quartet (fresh short runs) -------------------------------
for rv in ext3_rv05 ext3_rv10 ext3_rv15 ext3_rv20; do
  run_forward "$rv" || FAILED="$FAILED $rv"
done

# ---- q-mask re-score (light; own script, not the engine) -------------------
if [ ! -f "$BASE/ext3_qmask01/grid_cubes.npz" ]; then
  echo "=== $(date '+%F %T') rescore_qmask q>=0.1 ==="
  $PY -m simulations.rescore_qmask \
      --run-dir "$REF" --q-cut 0.1 \
      --output-dir "$BASE/ext3_qmask01" --n-workers 16 \
      > "$BASE/launcher_ext3_qmask01.log" 2>&1 \
      || FAILED="$FAILED qmask01"
fi

echo "=== $(date '+%F %T') chain done ==="
echo "REMINDER: launch ext3_qmin01 (~24 h) separately after reviewing this log."
if [ -n "$FAILED" ]; then
  echo "FAILED variants:$FAILED"
  exit 1
fi
echo "ALL VARIANTS OK"
