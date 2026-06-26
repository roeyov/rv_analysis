# SLURM Bias Grid — TAU HPC Cluster Guide

> ## ⚠️ DEPRECATED — do not use SLURM
>
> The SLURM path (`slurmlogin.tau.ac.il` / `slurm-client.cs.tau.ac.il`,
> `submit_bias_grid.slurm`, `--grid-start/--grid-end/--aggregate`) is **no
> longer used**. The live workflow runs the bias grid on a **single shared
> machine, `astro3.tau.ac.il`, reached over the TAU VPN**.
>
> **astro3 bias-grid workflow (canonical):**
> - Requires the **TAU VPN**. Env is **micromamba** `tau_binary` (not conda):
>   `micromamba run -n tau_binary …`. Code at `~/Scripts`, inputs at
>   `~/tables` + `~/data`, outputs at `~/bias_grid_results/<run_name>/`.
> - **Sync from the Mac:** `./sync_astro3.sh` (push) / `./sync_astro3.sh pull`.
> - **Launch a run** (background, survives disconnect). The run name/suffix,
>   preset, and all knobs come from the YAML `bias_grid` block — `output_dir`
>   sets the run directory:
>   ```bash
>   cd ~/Scripts
>   nohup micromamba run -n tau_binary python -u -m simulations.bias_grid \
>       --config configs/params_bias_astro3<_variant>.yaml \
>       > ~/bias_grid_results/launcher_<tag>.log 2>&1 </dev/null &
>   ```
> - **Monitor:** `tail -f ~/bias_grid_results/launcher_<tag>.log`, or the
>   per-run `~/bias_grid_results/<run_name>/bias_grid.log`.
> - **Timing:** the full `final4_10min` preset (150 000 points, 20×25×20×15,
>   adaptive injection, 94 workers, `parallel_grid: true`) takes **≈ 2 h**.
> - **Explore:** SSH-tunnel the Streamlit explorer (see the
>   `astro3-explorer-tunnel` memory).
>
> Everything below is retained for historical reference only.

## Overview

Run the bias correction grid search on the TAU HPC SLURM cluster using
array jobs. Each array task processes a chunk of grid points, using all
allocated CPUs for per-star injection-recovery.

**Cluster**: `slurmlogin.tau.ac.il`
**Partition**: `power-general-shared-pool` (public, preemptible)
**Account**: `public-users_v2`
**QoS**: `public`

Jobs use idle time on shared nodes — they may wait or be suspended when
node owners submit their own work.

---

## First-Time Setup

### 1. Get cluster access

- You need "power" group membership — contact HPC admins or check
  https://hpcguide.tau.ac.il
- SSH: `ssh roeyovadia@slurmlogin.tau.ac.il`

### 2. Sync code from your Mac

```bash
# Run from your LOCAL machine:
rsync -avz ~/Roey/Masters/Reasearch/Scripts/ \
    roeyovadia@slurmlogin.tau.ac.il:~/Scripts/
```

### 3. Run the setup script

```bash
# On the cluster login node:
bash ~/Scripts/slurm/setup_cluster.sh
```

This loads the system conda module and creates the `tau_binary` environment
under `~/.conda/envs/`.

### 4. Sync data files

```bash
# From your LOCAL machine:

# LaTeX solution tables
rsync -avz ~/Roey/Masters/Reasearch/Ostars_article/tables/sb1_solutions.tex \
    roeyovadia@slurmlogin.tau.ac.il:~/tables/
rsync -avz ~/Roey/Masters/Reasearch/Ostars_article/tables/sb2_solutions.tex \
    roeyovadia@slurmlogin.tau.ac.il:~/tables/

# Mass catalog
rsync ~/Documents/Data/BLOeM_Data/mass_bloem.csv \
    roeyovadia@slurmlogin.tau.ac.il:~/data/

# Per-star RV CSVs (for rv_err, gamma, field assignment)
rsync -avz ~/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_neb_div_from_coadded/ \
    roeyovadia@slurmlogin.tau.ac.il:~/data/rv_csvs/
```

---

## Running the Grid Search

### 1. Test with a single grid point first

```bash
cd ~/Scripts
mkdir -p logs

sbatch --array=0 \
    --export=PRESET=single,TOTAL_GRID_POINTS=1,N_TASKS=1 \
    slurm/submit_bias_grid.slurm
```

Check output: `cat logs/bias_grid_<JOB_ID>_0.out`

### 2. Test with the quick preset (81 points)

```bash
sbatch --array=0-3 \
    --export=PRESET=quick,TOTAL_GRID_POINTS=81,N_TASKS=4 \
    slurm/submit_bias_grid.slurm
```

### 3. Full run — preset D (2304 points)

```bash
# Submit array job (48 tasks, ~48 points each)
JOB_ID=$(sbatch --parsable slurm/submit_bias_grid.slurm)
echo "Submitted: $JOB_ID"

# Submit aggregation to run after all tasks finish
sbatch --dependency=afterok:${JOB_ID} slurm/aggregate_bias_grid.sh
```

### Resource notes

Public queue gives 4 CPUs per task (shared nodes). Each task uses 3
workers for per-star parallelism. With fewer CPUs than CS cluster, tasks
run longer but you get free compute.

| Tasks | Points/task | Est. time/task (4 CPUs) |
|-------|-------------|-------------------------|
| 48    | 48          | ~4-8 hours              |
| 96    | 24          | ~2-4 hours              |
| 192   | 12          | ~1-2 hours              |

---

## Monitoring

```bash
squeue --me                    # all your jobs
squeue -j <JOB_ID>            # specific array job

# Summary of all tasks
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS

# Show only failed tasks
sacct -j <JOB_ID> --format=JobID,State,ExitCode | grep -v COMPLETED

# Tail a task log
tail -f logs/bias_grid_<JOB_ID>_<TASK_ID>.out

# Count completed task directories
ls -d ~/bias_grid_results/D/task_*/ 2>/dev/null | wc -l
```

---

## Handling Failures

### Suspended/preempted tasks

Public queue jobs can be suspended when node owners submit work.
Each task checkpoints after every grid point, so resubmitting resumes
from where it left off:

```bash
sbatch slurm/submit_bias_grid.slurm
```

### Out-of-memory

Increase `--mem-per-cpu` in `submit_bias_grid.slurm`:

```
#SBATCH --mem-per-cpu=8G     # was 4G
```

---

## Aggregation

After all tasks complete:

```bash
# Interactive on login node
bash slurm/aggregate_bias_grid.sh

# Or already submitted with --dependency (see step 3)
```

Output in `~/bias_grid_results/D/`:
- `grid_search_results.csv` — all grid points sorted by GMF
- `grid_cubes.npz` — 4D result cubes
- `grid_1d_posteriors.pdf` — 1D marginalized posteriors
- `grid_2d_gmf.pdf` — 2D marginalized heatmaps

### Pull results back to your Mac

```bash
rsync -avz roeyovadia@slurmlogin.tau.ac.il:~/bias_grid_results/D/grid_*.{csv,npz,pdf} \
    ~/Roey/Masters/Reasearch/scriptsOut/bias_grid/D/
```

---

## File Reference

| File | Purpose |
|------|---------|
| `slurm/submit_bias_grid.slurm` | SLURM array job submission script |
| `slurm/aggregate_bias_grid.sh` | Merge partial results + plot |
| `slurm/setup_cluster.sh` | First-time conda/directory setup |
| `configs/params_bias_slurm.yaml` | Pipeline config with cluster paths |
| `simulations/bias_grid.py` | Main grid search code |
| `simulations/bias_config.py` | Grid presets and default config |

---

## Useful SLURM Commands

```bash
sinfo                          # available partitions
squeue --me                    # your jobs
scancel <JOB_ID>               # cancel a job
scancel <JOB_ID>_<TASK_ID>     # cancel one array task
sacct -j <JOB_ID> -l           # detailed job accounting
```
