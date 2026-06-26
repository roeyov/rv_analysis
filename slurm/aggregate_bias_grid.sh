#!/bin/bash
# =============================================================================
# Aggregate partial results from SLURM array tasks.
#
# Run this after all array tasks complete. Can be run:
#   1. Interactively on the login node:
#        bash slurm/aggregate_bias_grid.sh
#
#   2. As a SLURM job with dependency:
#        JOB_ID=$(sbatch --parsable slurm/submit_bias_grid.slurm)
#        sbatch --dependency=afterok:${JOB_ID} slurm/aggregate_bias_grid.sh
# =============================================================================

#SBATCH --job-name=bias_agg
#SBATCH --partition=power-general-shared-pool
#SBATCH --account=public-users_v2
#SBATCH --qos=public
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/bias_aggregate_%j.out
#SBATCH --error=logs/bias_aggregate_%j.err

: "${PRESET:=D}"
: "${CONDA_ENV:=tau_binary}"

SCRIPTS_DIR="${HOME}/Scripts"
OUTPUT_BASE="${HOME}/bias_grid_results/${PRESET}"

# Activate environment
module load miniconda/miniconda3-4.7.12-environmentally
conda activate "${HOME}/.conda/envs/${CONDA_ENV}"

cd "${SCRIPTS_DIR}"

echo "========================================"
echo "Aggregating results from: ${OUTPUT_BASE}"
echo "========================================"

python -m simulations.bias_grid --aggregate "${OUTPUT_BASE}"

echo ""
echo "Aggregation finished with exit code $?"
echo "Results:"
ls -lh "${OUTPUT_BASE}"/grid_*.{csv,npz,pdf} 2>/dev/null
