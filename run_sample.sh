#!/bin/bash
# Run the SB1 chain over a list of stars, stage by stage, for a run folder made
# by pipeline/make_run.py:
#   ccf          CCF pass 1 + pass 2, then the per-line quality report
#   corrections  per-field, per-epoch RV zero-point corrections (fixed Mean RV)
#   fit          period search + orbital fitting (pipeline.evaluator; resumes)
#   mcmc         MCMC on each star's best solution (mcmc.batch)
#   summary      detections_summary.csv
#
# Usage:
#   caffeinate -i ./run_sample.sh <run_dir>                    # all stages
#   caffeinate -i ./run_sample.sh <run_dir> --to ccf           # stop after CCF (check lines first)
#   caffeinate -i ./run_sample.sh <run_dir> --from corrections # continue after the check
#   caffeinate -i ./run_sample.sh <run_dir> --from fit         # resume a stopped fit
#
# Rerunning 'ccf' rewrites the pass-2 CSVs (any manual edits there are lost);
# 'corrections' must always follow it.
set -euo pipefail

STAGES=(ccf corrections fit mcmc summary)
usage() { sed -n '2,17p' "$0"; exit 1; }
[ $# -ge 1 ] || usage
RUN_DIR="$(cd "$1" && pwd)"; shift
FROM=ccf; TO=summary
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --to)   TO="$2"; shift 2 ;;
        *) usage ;;
    esac
done
idx() { local i; for i in "${!STAGES[@]}"; do [ "${STAGES[$i]}" = "$1" ] && { echo "$i"; return; }; done
        echo "unknown stage '$1' (${STAGES[*]})" >&2; exit 1; }
I_FROM=$(idx "$FROM"); I_TO=$(idx "$TO")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
CFG="$RUN_DIR/configs"
for f in "$CFG/ccf_pass1.yaml" "$CFG/ccf_pass2.yaml" "$CFG/params.yaml" "$RUN_DIR/run.env"; do
    [ -f "$f" ] || { echo "missing $f (make the run with python -m pipeline.make_run)"; exit 1; }
done
# shellcheck source=/dev/null
source "$RUN_DIR/run.env"
python -c "import lmfit, emcee, numba, PDC.pdc_func" 2>/dev/null || {
    echo "python ($(command -v python)) lacks the pipeline packages: conda activate tau_binary"; exit 1; }

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$RUN_DIR/run.log"; }
want() { local i; i=$(idx "$1"); [ "$i" -ge "$I_FROM" ] && [ "$i" -le "$I_TO" ]; }

log "=== run_sample.sh $RUN_DIR (stages $FROM..$TO) ==="
if want ccf; then
    log "CCF pass 1 -> ccf.log"
    python -m spectroscopy.ccf_main --input_file "$CFG/ccf_pass1.yaml" >> "$RUN_DIR/ccf.log" 2>&1
    log "CCF pass 2 -> ccf.log"
    python -m spectroscopy.ccf_main --input_file "$CFG/ccf_pass2.yaml" >> "$RUN_DIR/ccf.log" 2>&1
    log "line-quality report -> line_quality.txt"
    python -m spectroscopy.ccf_line_quality "$RUN_DIR/pass2" > "$RUN_DIR/line_quality.txt"
fi
if want corrections; then
    log "RV zero-point corrections -> corrections.log"
    python -m spectroscopy.apply_rv_corrections \
        --corrections "$RV_CORRECTIONS" --input-dir "$RUN_DIR/pass2" >> "$RUN_DIR/corrections.log" 2>&1
fi
if want fit; then
    log "period search + fit -> fit.log  (~1.5 min per star)"
    python -m pipeline.evaluator --config "$CFG/params.yaml" >> "$RUN_DIR/fit.log" 2>&1
fi
if want mcmc; then
    log "MCMC -> mcmc.log"
    python -m mcmc.batch --config "$CFG/params.yaml" >> "$RUN_DIR/mcmc.log" 2>&1
fi
if want summary; then
    log "detection summary -> summary.txt"
    python -m pipeline.detection_summary --config "$CFG/params.yaml" > "$RUN_DIR/summary.txt"
    head -1 "$RUN_DIR/summary.txt" | tee -a "$RUN_DIR/run.log"
fi
log "done"
