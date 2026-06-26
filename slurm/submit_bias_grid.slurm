#!/bin/bash
# =============================================================================
# SLURM array job for bias correction grid search.
#
# Splits the 4D grid across array tasks. Each task processes a contiguous
# chunk of grid points using all allocated CPUs for per-star parallelism.
#
# Usage:
#   cd ~/Scripts
#   mkdir -p logs
#   sbatch slurm/submit_bias_grid.slurm
#
# To override defaults (e.g. smaller test run):
#   sbatch --array=0-3 --export=PRESET=quick,TOTAL_GRID_POINTS=81,N_TASKS=4 \
#          slurm/submit_bias_grid.slurm
# =============================================================================

#SBATCH --job-name=bias_grid
#SBATCH --partition=power-general-shared-pool
#SBATCH --account=public-users_v2
#SBATCH --qos=public
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-47
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# --- Configuration (override via --export or edit here) ---
: "${TOTAL_GRID_POINTS:=2304}"   # preset D: 8x8x6x6
: "${N_TASKS:=48}"               # must match --array count
: "${PRESET:=D}"
: "${SEED:=42}"
: "${N_INJECT:=}"                # empty = use preset default
: "${CONDA_ENV:=tau_binary}"

# --- Paths ---
SCRIPTS_DIR="${HOME}/Scripts"
CONFIG="${SCRIPTS_DIR}/configs/params_bias_slurm.yaml"
OUTPUT_BASE="${HOME}/bias_grid_results/${PRESET}"

# Data paths (synced from local machine via setup_cluster.sh)
SB1_TEX="${HOME}/tables/sb1_solutions.tex"
SB2_TEX="${HOME}/tables/sb2_solutions.tex"
MASS_FILE="${HOME}/data/mass_bloem.csv"
RV_DIR="${HOME}/data/rv_csvs"

# --- Compute this task's grid range ---
CHUNK_SIZE=$(( (TOTAL_GRID_POINTS + N_TASKS - 1) / N_TASKS ))
GRID_START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
GRID_END=$(( GRID_START + CHUNK_SIZE ))
if [ "${GRID_END}" -gt "${TOTAL_GRID_POINTS}" ]; then
    GRID_END=${TOTAL_GRID_POINTS}
fi

# --- Environment ---
module load miniconda/miniconda3-4.7.12-environmentally
conda activate "${HOME}/.conda/envs/${CONDA_ENV}"

cd "${SCRIPTS_DIR}"

# Workers: use all allocated CPUs minus 1 for overhead
N_WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))
if [ "${N_WORKERS}" -lt 1 ]; then N_WORKERS=1; fi

# --- Build command ---
CMD="python -m simulations.bias_grid \
    --config ${CONFIG} \
    --preset ${PRESET} \
    --seed ${SEED} \
    --output-dir ${OUTPUT_BASE} \
    --n-workers ${N_WORKERS} \
    --grid-start ${GRID_START} \
    --grid-end ${GRID_END} \
    --sb1-tex ${SB1_TEX} \
    --sb2-tex ${SB2_TEX} \
    --mass-file ${MASS_FILE} \
    --rv-dir ${RV_DIR}"

if [ -n "${N_INJECT}" ]; then
    CMD="${CMD} --n-inject ${N_INJECT}"
fi

echo "========================================"
echo "SLURM Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Grid range: [${GRID_START}, ${GRID_END})"
echo "Workers: ${N_WORKERS}"
echo "Output: ${OUTPUT_BASE}/task_${GRID_START}_${GRID_END}/"
echo "========================================"
echo ""

${CMD}

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} finished with exit code $?"
