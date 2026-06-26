# CLAUDE.md — Binary Star Pipeline

## Project Summary
Binary star detection & orbital characterization pipeline for the **BLOeM survey** (Binarity at LOw Metallicity). Processes radial velocity (RV) data from VLT/FLAMES observations of ~134 O-type stars across ~25 epochs over ~2 years.

**Data flow**: FITS spectra → CCF RV extraction → Periodogram period search → lmfit orbital fitting → MCMC posterior sampling

## Directory Structure

```
Scripts/
├── spectroscopy/     # Stage 1: CCF radial velocity extraction from FITS spectra
├── pipeline/         # Stage 2: Period search + orbital fitting (main orchestrator)
├── period_search/    # Periodogram (Lomb-Scargle + PDC) & permutation FAP tests
├── orbital/          # Kepler physics, lmfit fitting, statistical tests
├── mcmc/             # Stage 4: emcee posterior sampling, Lucy-Sweeney test
├── cloud/            # Google Cloud Batch infrastructure for simulations
├── simulations/      # Synthetic binary RV generation & population synthesis
├── PDC/              # Phase-Distance Correlation (Numba-accelerated)
├── utils/            # Constants, plotting style, Roche lobe calculations
├── configs/          # All pipeline / bias-grid / cloud / SLURM YAML configs
│   ├── params.yaml       # Main pipeline config (local)
│   ├── params_cloud.yaml # Cloud pipeline config (container paths)
│   ├── params_bias*.yaml # Bias-grid presets (local / astro3 / SLURM / smoke)
```

## Entry Points

```bash
# Stage 1: CCF RV extraction
python -m spectroscopy.ccf_main --input_file spectroscopy/ccf_input.yaml

# Stage 2: Period search + orbital fitting
python -m pipeline.evaluator --config configs/params.yaml

# Stage 3: Interactive threshold selection (Streamlit)
streamlit run mcmc/selector_app.py

# Stage 4: MCMC posterior sampling
python -m mcmc.batch --config configs/params.yaml

# Cloud batch simulations
python -m cloud.run_cloud_batch --n-per-field 10 --run-id my-run-001
```

## Key Files

| File | Purpose |
|------|---------|
| `pipeline/evaluator.py` | Main orchestrator — `main_single()` / `main_multiple()` |
| `pipeline/config.py` | YAML config loading, path resolution, search region setup |
| `pipeline/data_loading.py` | CSV loading, column standardization |
| `orbital/kepler.py` | Kepler equation solvers (iterative + Newton) |
| `orbital/fitting.py` | lmfit parameter setup, chi-squared objectives, jitter model |
| `orbital/statistics.py` | AIC/BIC, F-test, phase coverage, binary probability |
| `orbital/plotting.py` | Phase-folded RV plots (plotly + matplotlib) |
| `period_search/candidates.py` | Peak extraction, FWHM bounds, candidate table building |
| `period_search/periodogram.py` | LS + PDC periodograms |
| `period_search/permutation.py` | Permutation-based FAP (multiprocessing) |
| `mcmc/batch.py` | Batch MCMC runner, eccentric vs circular model selection |
| `mcmc/models.py` | Keplerian RV model, log-prior/likelihood for emcee |
| `mcmc/runner.py` | emcee wrappers |
| `mcmc/analysis.py` | Lucy-Sweeney eccentricity significance test |
| `utils/constants.py` | Column names, parameter keys, shared constants |
| `simulations/common.py` | Real BLOeM MJD arrays, noise models, Keplerian physics |

## Internal Conventions

### Column Names (utils/constants.py)
- `ts` = MJD timestamps, `rvs` = radial velocities (km/s), `errs` = RV errors
- Orbital params: `Period`, `K1`, `GAMMA`, `Eccentricity`, `OMEGA_rad`, `T0`, `ln_sigmaJ`

### Best Solution Selection (from lmfit_summary.csv)
```python
df.query(
    "~candidate_method.str.contains('MANUAL')"
    " & ~candidate_method.str.contains('null')"
    " & candidate_method.str.contains('jitter')"
    " & (prob_bicc > 0.5)"
    " & (mass_flag_peri == 0)"
).sort_values("bicc").iloc[0]
```

### Output Structure (per star)
```
{star_name}/
├── lmfit_summary.csv          # All candidate solutions with statistics
├── lmfit_solutions/           # Plots per solution (phase + time residuals)
├── periodogram/               # LS + PDC periodograms, permutation results
└── mcmc/                      # Corner plots, phase bands, chain summaries
```

## Config (configs/params.yaml) Key Sections
- `base_dir` — Root output directory
- `lmfit_subdir` — Stage 2 output subdirectory name
- `mcmc_subdir` — Stage 4 output subdirectory name
- `pipeline_io.object_list` — List of stars to process (empty = all)
- `periodogram_params` — Period range, FAP method, permutation settings
- `lmfit_params.search_region` — Parameter bounds for differential_evolution
- `mcmc_params` — Walker count, burn-in, steps, filter expression

## Design Patterns
- **Multiprocessing**: Candidate fitting + LS permutations use `multiprocessing.Pool` / `ProcessPoolExecutor`
- **Numba JIT**: PDC calculations use `@njit` for performance
- **Deep copy isolation**: lmfit Parameters are deepcopied before parallel workers
- **Deterministic seeding**: Cloud tasks use `sha256(run_id + task_index)` for reproducibility
- **Hierarchical paths**: All I/O resolves relative to `base_dir` from config

## Dependencies
numpy, pandas, scipy, matplotlib, plotly, astropy, lmfit, emcee, corner, tqdm, PyYAML, numba, google-cloud-storage, google-cloud-batch

**Conda env**: `tau_binary`

## Outputs Convention
**All script outputs go outside this repo**, under
`/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/` (the `SCRIPTS_OUT`
env var). The `Scripts/` working directory holds **code and configs only** —
never write generated artifacts (CSVs, PDFs, NPZ, logs, HDF5, plots, MCMC
chains, simulation grids, etc.) into it.

- For any new script (ad hoc or recurring), default output paths to a
  subdirectory of `$SCRIPTS_OUT`, organized by pipeline stage:
  - `scriptsOut/CCF/` — Stage 1 CCF RV extraction outputs
  - `scriptsOut/simulation_pipeline/` — population synthesis, bias grids, closure tests
  - `scriptsOut/spectrasDrawer/`, `scriptsOut/orbitalGifs/`, `scriptsOut/RVDataGen/`, etc.
- Configs should expose `output_dir` / `base_dir` and resolve them relative
  to `$SCRIPTS_OUT`. Shell drivers should set
  `SCRIPTS_OUT="${SCRIPTS_OUT:-/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut}"`.
- `.gitignore` still lists historical output directories (e.g.
  `bias_grid_closure/`) as a safety net against accidental in-repo writes.

## Astro3 Sync
- Sync code + bias-grid inputs to astro3 with `./sync_astro3.sh` (push) or
  `./sync_astro3.sh pull` (results back). Don't `rsync` individual files —
  the script is the canonical entry point and excludes `.git/`, `__pycache__/`,
  `scriptsOut/`, `.claude/`, `.venv/`, `.idea/`.
- Remote layout: code at `/jonathan_storage/roeyovadia/Scripts/`, inputs at
  `/jonathan_storage/roeyovadia/{data,tables}/`, outputs at
  `/jonathan_storage/roeyovadia/bias_grid_results/`.

## Related Repos
- Paper: `/Users/roeyovadia/Roey/Masters/Reasearch/Ostars_article` (GitHub: `roeyov/Ostars-Multiplicity-paper`)
- Pipeline output root: `/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/`
- CCF reference output: `scriptsOut/CCF/dr5_neb_div_from_coadded/second/`
- Bias-grid closure outputs: `scriptsOut/simulation_pipeline/bias_grid_closure/`

## User Preferences
- Prefers concise, direct communication
- Working on MSc thesis — binary star multiplicity in the SMC
- Uses macOS, VSCode, conda (`tau_binary` env)
