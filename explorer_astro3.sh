#!/bin/bash
# explorer_astro3.sh — open the bias_grid explorer for an astro3 run in ONE step.
#
# Usage:  ./explorer_astro3.sh <run-name-or-abs-path>
#
# Does everything the old multi-step flow did, in one invocation:
#   * verifies the run dir (grid_cubes.npz) on astro3
#   * REUSES an existing explorer server already serving that dir, else picks a
#     port free on BOTH ends (850x) and launches a DETACHED streamlit server
#   * starts the Mac-side per-port persistent reconnect tunnel (nohup, disowned)
#     unless one is already listening
#   * verifies end-to-end health and prints:  OPEN http://localhost:<PORT>
#
# Requires the TAU VPN. Never pulls any run data to the Mac.
set -u
REMOTE=roeyovadia@astro3.tau.ac.il
RUN="${1:?usage: explorer_astro3.sh <run-name-or-abs-path>}"
case "$RUN" in /*|"~"*) DIR="$RUN" ;; *) DIR="~/bias_grid_results/$RUN" ;; esac

# Ports already LISTENing locally (the tunnel binds the same number both ends,
# so the remote port choice must avoid these too).
LOCAL_USED=$(lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | grep -oE '850[0-9]' | sort -u | tr '\n' ' ')

# ---- ONE remote call: verify + reuse-or-launch + wait-healthy --------------
# Script goes over stdin (bash -s), so none of these patterns appear in any
# process cmdline -> no pgrep/pkill self-match hazards.
OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE" bash -s -- "$DIR" "$LOCAL_USED" <<'REMOTE_EOF'
DIR="$1"; LOCAL_USED="${2:-}"
RD=$(eval echo "$DIR")                       # expand ~ remotely
if [ ! -f "$RD/grid_cubes.npz" ]; then
  echo "MISSING $RD"; ls ~/bias_grid_results/; exit 3
fi
# Reuse a server already serving this dir.
for p in 8501 8502 8503 8504 8505 8506 8507 8508 8509 8510; do
  [ "$(curl -s -m3 http://127.0.0.1:$p/_stcore/health 2>/dev/null)" = ok ] || continue
  # Anchor to end-of-cmdline: a bare prefix match would wrongly reuse e.g.
  # <run>_ext_fast's server when asked for <run>.
  if pgrep -af "server.port $p" | grep -qE -- "output-dir ${RD%/}/?\$"; then
    echo "REUSE $p"; exit 0
  fi
done
# Pick a port free on both ends.
REMOTE_USED=$( (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -oE '850[0-9]' | sort -u)
PORT=""
for p in 8501 8502 8503 8504 8505 8506 8507 8508 8509 8510; do
  printf '%s\n%s\n' "$REMOTE_USED" "$LOCAL_USED" | tr ' ' '\n' | grep -qx "$p" && continue
  PORT=$p; break
done
[ -n "$PORT" ] || { echo NOFREEPORT; exit 4; }
# setsid + subshell: fully sever from the ssh session, otherwise micromamba's
# children hold the channel open and the ssh (and this script) never returns.
cd ~/Scripts && ( setsid ~/.local/bin/micromamba run -n tau_binary \
  streamlit run simulations/bias_grid_explorer.py \
  --server.headless true --server.port "$PORT" --server.address 127.0.0.1 \
  --browser.gatherUsageStats false \
  -- --output-dir "$RD/" > ~/bias_grid_results/explorer_"$PORT".log 2>&1 </dev/null & )
for i in $(seq 1 30); do
  [ "$(curl -s -m5 http://127.0.0.1:$PORT/_stcore/health 2>/dev/null)" = ok ] \
    && { echo "READY $PORT"; exit 0; }
  sleep 2
done
echo "DOWN $PORT"; tail -5 ~/bias_grid_results/explorer_"$PORT".log; exit 5
REMOTE_EOF
)
RC=$?
echo "$OUT"
[ $RC -eq 0 ] || { echo "FAILED (remote stage, rc=$RC)"; exit $RC; }

PORT=$(echo "$OUT" | awk '/^(REUSE|READY) /{print $2; exit}')
[ -n "$PORT" ] || { echo "FAILED to parse port"; exit 7; }

# ---- Mac-side per-port persistent tunnel (skip if already listening) -------
if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  nohup bash -c "while true; do
      ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
          -o ServerAliveCountMax=4 -o ConnectTimeout=15 -o BatchMode=yes \
          -L $PORT:127.0.0.1:$PORT $REMOTE
      sleep 5
    done" >/dev/null 2>&1 </dev/null &
  disown
  echo "tunnel loop started for port $PORT"
else
  echo "tunnel already up for port $PORT"
fi

# ---- End-to-end verify -----------------------------------------------------
for i in $(seq 1 12); do
  [ "$(curl -s -m5 http://localhost:"$PORT"/_stcore/health 2>/dev/null)" = ok ] \
    && { echo "OPEN http://localhost:$PORT"; exit 0; }
  sleep 2
done
echo "TUNNEL FAILED for port $PORT (server is healthy on astro3)"; exit 6
