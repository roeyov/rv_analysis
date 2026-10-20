# Running the single-star chain (`run_star.sh`) on a list of stars

This covers the **SB1 chain only**: iterative CCF RV extraction → per-epoch RV
zero-point correction → period search + orbital fitting → best-solution
selection → MCMC on the best solution.
SB2 systems use `sb2_pipeline/` instead. Bias grids and simulations are not
covered here.

## What `run_star.sh` does

`./run_star.sh 4-043` runs, for one star:

| Step | Command | Reads | Writes |
|---|---|---|---|
| 1. CCF pass 1 | `python -m spectroscopy.ccf_main --input_file spectroscopy/ccf_input_first.yaml` | FITS spectra, per-star model templates (`template_path`), v sin i table | `<pass1_out>/BLOeM_<ID>_CCF_RVs.csv`, `_CoAdded.csv` |
| 2. CCF pass 2 | same, with `ccf_input_second.yaml` | FITS spectra, **pass-1 co-added spectrum as template** | `<pass2_out>/BLOeM_<ID>_CCF_RVs.csv`, `_CoAdded.csv` |
| 3. RV corrections | `python -m spectroscopy.apply_rv_corrections --stars BLOeM_<ID>` | pass-2 `_CCF_RVs.csv`, corrections table | adds `fixed Mean RV`, `fixed Mean RVsig` to the pass-2 `_CCF_RVs.csv` |
| 4. Fit | `python -m pipeline.evaluator` (`configs/params.yaml`) | `fixed Mean RV(sig)` from the pass-2 `_CCF_RVs.csv`, mass table | `<base_dir>/<lmfit_subdir>/BLOeM_<ID>/lmfit_summary.csv` + plots |
| 5. Best solution | inline Python | `lmfit_summary.csv`, `mcmc_params.filter_expression` | prints the row, opens its phase plot |
| 6. MCMC | inline Python (`mcmc.batch.run_mcmc_single_star`) | best row (or the null-model row if no row passes the filter) | `<base_dir>/<lmfit_subdir>/BLOeM_<ID>/mcmc/` |

It does this by editing the three YAMLs in place with `sed` (the star name, and
the line list for `--nebular` / `--balmer` / `--he_1`), then calling the stages.

Step 3 subtracts the per-field, per-epoch RV zero-point offset from each RV,
`fixed Mean RV = Mean RV − rv_correction`, and adds the correction error in
quadrature, `fixed Mean RVsig = sqrt(Mean RVsig² + Error²)`. Corrections are
matched to epochs by nearest MJD within 0.005 d. `pipeline_io.rv_col` /
`rverr_col` in `params.yaml` make the fit and the MCMC use these columns.
`params.yaml` also uses the same fit settings as the paper's Stage-2 run
(`configs/params_dr5_fifth_fixed_rvs.yaml`).

**There is no batch wrapper for `run_star.sh`.** You don't need one, because
every stage already accepts a list. There are two ways to run a list:

- **Route A (recommended): run each stage once over the whole list.** Faster to
  set up, and the fitting stage can resume after a crash.
- **Route B: loop `run_star.sh` over the list.** You get exactly the per-star
  output layout of `run_star.sh`, but there are caveats (see below).

---

## 0. One-time setup

### Environment

```bash
conda create -n tau_binary python=3.12
conda activate tau_binary
cd Scripts
pip install -r requirements.txt -c constraints.txt
pip install kaleido streamlit     # not in requirements.txt: kaleido = PNG export of plotly plots, streamlit = Stage 3 app
```

`constraints.txt` pins the versions the pipeline was validated with (numpy < 2).
Run everything from the `Scripts/` directory with `tau_binary` active
(`run_star.sh` changes into `Scripts/` itself). `run_star.sh` calls whatever
`python` is on your `PATH`.

### PDC module (currently missing from the repo)

`period_search/` imports `PDC.pdc_func` (`calc_pdc`, `pdc_window`,
`calc_pdc_optimized`). `PDC/` is recorded in git as a submodule pointer with no
URL, so a fresh clone gets an **empty `PDC/` directory**, and the upstream PDC
code does not contain `calc_pdc_optimized`. Until this is fixed in the repo,
get `pdc_func.py` from Roey and put it at `PDC/pdc_func.py`. Check it with:

```bash
python -c "from PDC.pdc_func import calc_pdc, pdc_window, calc_pdc_optimized; print('PDC OK')"
```

### Input data (not in the repo)

Ask Roey where to get these:

| Input | YAML key | Notes |
|---|---|---|
| BLOeM DR5 combined spectra | `path_to_observations` (both CCF YAMLs) | searched recursively; files named `BLOeM_<ID>_<epoch>_Combined.fits` |
| Per-star model templates | `template_path` (`ccf_input_first.yaml` only) | one file per star, filename containing `BLOeM_<ID>` |
| v sin i table | `path_to_meta_data_csv` (`ccf_input_first.yaml` only) | columns `ID, vsini, evsini`; used to broaden the templates |
| Mass table | `pipeline_io.mass_file` (`params.yaml`) | needed for the Roche-lobe flag `mass_flag_peri` that the default best-solution filter uses |
| RV zero-point corrections (`RVcorrectionsDR5.csv`) | `RV_CORRECTIONS` env var for `run_star.sh`; `--corrections` for `apply_rv_corrections` | columns `config, MJD, rv_correction, Error`, with `config = F<field>_E<epoch>`. `run_star.sh` defaults to `$SCRIPTS_OUT/CCF/dr5_neb_div_from_coadded/RVcorrectionsDR5.csv` |

### Paths you must edit

All paths in the YAMLs are absolute paths on Roey's machine. Environment
variables are **not** expanded, so write literal absolute paths. Outputs go
outside the repo (convention: under `scriptsOut/`). The paths have to chain
together like this:

```
ccf_input_first.yaml   path_to_output  = <OUT>/pass1/
ccf_input_second.yaml  template_path   = <OUT>/pass1/      ← must equal pass-1 output
ccf_input_second.yaml  path_to_output  = <OUT>/pass2/
configs/params.yaml    base_dir        = <OUT>/pass2       ← must equal pass-2 output
configs/params.yaml    lmfit_subdir    = e.g. "fit_v1"     (fit outputs: <base_dir>/<lmfit_subdir>/)
configs/params.yaml    mcmc_subdir     = e.g. "mcmc"       (Route A MCMC outputs: <base_dir>/<lmfit_subdir>/<mcmc_subdir>/)
```

In `ccf_input_second.yaml`, keep `path_to_meta_data_csv: ""`. The pass-1
co-added spectrum is already the star's own spectrum, so it isn't broadened again.

### Spectral lines

The lines used are the uncommented entries under `cross_cor_ranges`. Keep the
same set in both CCF YAMLs. The `run_star.sh` flags correspond to:

| Flag | Lines |
|---|---|
| (none) | HeII 4542, HeII 4200, HeI 4471, HeI 4143, HeI 4388, HeI+HeII 4026 |
| `--nebular` | HeII 4542, HeII 4200 only |
| `--balmer` | adds H_Gamma, H_Delta, H_Epsilon |
| `--he_1` | HeI 4143, HeI 4388 only |

The flags rewrite the YAMLs, and the change stays after the run. In a batch run
the same line set applies to every star in the list. If some stars need
different lines, split them into separate lists and use separate output
directories.

### Star list file

A plain text file with one star per line, written as `BLOeM_<field>-<number>`.
Only the first word of each line is read:

```
BLOeM_1-012
BLOeM_1-023
BLOeM_4-043
```

Use the `BLOeM_` prefix. The output filenames are built from this string, and
star IDs are matched by substring.

---

## Route A: stage by stage (recommended)

### A1. CCF pass 1, then pass 2, over the whole list

In **both** `spectroscopy/ccf_input_first.yaml` and `ccf_input_second.yaml` set:

```yaml
path_to_list_of_objects: "/abs/path/to/my_stars.txt"   # takes precedence over list_of_objects
```

Then run pass 1 to completion before you start pass 2, because pass 2 needs
every pass-1 `_CoAdded.csv`:

```bash
python -m spectroscopy.ccf_main --input_file spectroscopy/ccf_input_first.yaml
python -m spectroscopy.ccf_main --input_file spectroscopy/ccf_input_second.yaml
```

Always pass `--input_file`. Without it, `ccf_main` looks for `./ccf_input.yaml`.
If you see `No template found for BLOeM_X-YYY. using first MJD sample as template`,
the template file is missing or its name doesn't contain the star ID.

### A1b. RV zero-point corrections over the whole pass-2 directory

```bash
python -m spectroscopy.apply_rv_corrections \
    --corrections /abs/path/to/RVcorrectionsDR5.csv --input-dir <OUT>/pass2/ --dry-run
python -m spectroscopy.apply_rv_corrections \
    --corrections /abs/path/to/RVcorrectionsDR5.csv --input-dir <OUT>/pass2/
```

Without `--stars`, it processes every `*_CCF_RVs.csv` in the directory. It
checks every file before writing any, so if one file fails (e.g. an epoch with
no correction within 0.005 d), nothing is written. Rerunning it just rewrites
the `fixed` columns from `Mean RV`, so it is safe to repeat. Run it again
whenever pass 2 is rerun, because pass 2 rewrites the CSVs without the `fixed`
columns.

### A2. Period search + orbital fitting

In `configs/params.yaml`:

```yaml
pipeline_io:
  object_list: []     # [] = every *_CCF_RVs.csv in base_dir; or e.g. ['1-012', '4-043'] (substring match)
```

```bash
python -m pipeline.evaluator --config configs/params.yaml
```

This is the long stage. The ~110-star O-star sample took about 3 h on a Mac
laptop (Aug 2026). For long runs, keep the machine awake and log to a file:

```bash
caffeinate -i nohup python -m pipeline.evaluator --config configs/params.yaml > fit.log 2>&1 &
```

**Resume and skip rules** (`pipeline/evaluator.py`, `main_single`):
- If a star already has `lmfit_summary.csv` in `<base_dir>/<lmfit_subdir>/`, it
  is **not refitted**. After a crash, rerun the same command and the run
  continues where it stopped.
- If you change fit settings and want new results, use a **new `lmfit_subdir`**
  or delete the old per-star folders. Otherwise the old results are kept
  without any message telling you so.
- `BLOeM_6-107` is always skipped (hardcoded in `main_multiple`).

### A3. (Optional) Inspect the best-solution selection

The best solution per star is chosen in `mcmc_params` of `params.yaml`: rows
passing `filter_expression`, minimum of `field_to_check` (`bicc`). To look at
the effect of thresholds interactively:

```bash
streamlit run mcmc/selector_app.py
```

### A4. MCMC on the best solution of every star

```bash
caffeinate -i nohup python -m mcmc.batch --config configs/params.yaml > mcmc.log 2>&1 &
```

For each star, it picks the best row with the same filter and runs the
eccentric MCMC, plus a circular MCMC when the Lucy–Sweeney test finds the
eccentricity not significant. If no row passes the filter, it runs the null
(constant RV + jitter) model instead. It is fast: with the default 32 walkers ×
2000 steps, 30 B dwarfs (13 orbits, 17 null models) took 68 s in total
(Oct 2026). 2000 steps can be short for long periods, though: check the
autocorrelation time `tau` stored in `chains_ecc.npz` (aim for nsteps ≳ 50 τ).

Differences from `run_star.sh` to be aware of:
- **Output location**: `<base_dir>/<lmfit_subdir>/<mcmc_subdir>/BLOeM_<ID>/`, plus a
  combined `mcmc_params.csv` in `<base_dir>/<lmfit_subdir>/<mcmc_subdir>/`.
  (`run_star.sh` writes to `<base_dir>/<lmfit_subdir>/BLOeM_<ID>/mcmc/`.)
- **It ignores `object_list`**: it runs on every `*_CCF_RVs.csv` in `base_dir`
  that has an `lmfit_summary.csv`. To restrict it to your list, give the list
  its own `base_dir`.
- **No resume**: if you rerun it, every star is sampled again.

---

## Route B: loop `run_star.sh`

```bash
conda activate tau_binary
cd Scripts
LOGDIR=/abs/path/to/logs; mkdir -p "$LOGDIR"

while read -r star _; do
  [ -z "$star" ] && continue
  id="${star#BLOeM_}"                      # run_star.sh takes the ID without the prefix
  ./run_star.sh "$id" < /dev/null > "$LOGDIR/$star.log" 2>&1 || echo "FAILED: $star"
done < /abs/path/to/my_stars.txt
```

To use a line flag, append it to every call (e.g. `./run_star.sh "$id" --nebular`).
If the corrections table is not at the default location, run
`export RV_CORRECTIONS=/abs/path/to/RVcorrectionsDR5.csv` before the loop.
`run_star.sh` checks that `params.yaml`'s `base_dir` equals the pass-2
`path_to_output` and stops if it doesn't.

Caveats:
- **`path_to_list_of_objects` must be `""` in both CCF YAMLs.** It takes
  precedence over the `list_of_objects` that `run_star.sh` rewrites, so if it
  is set, every iteration reprocesses the whole list.
- **It opens a plot window per star** (`open <phase plot>.png`). On Linux there
  is no `open` command, so step 4 fails and the star aborts before MCMC.
- **macOS only as written**: it uses BSD `sed -i ''`. On Linux, change these to `sed -i`.
- **It edits the YAMLs in place**. After the loop they point at the last star
  and the last line set. Check `git diff spectroscopy/ configs/` before you commit.
- **Skips**: if a star already has `lmfit_summary.csv`, the fit is skipped, but
  CCF, the corrections and MCMC still rerun. To redo a star from scratch,
  delete `<base_dir>/<lmfit_subdir>/BLOeM_<ID>/`. `BLOeM_6-107` is skipped by the
  evaluator, so its MCMC step then fails (logged as `FAILED`).

---

## Output layout (per star, under `<base_dir>/<lmfit_subdir>/BLOeM_<ID>/`)

```
lmfit_summary.csv      all candidate solutions + statistics (bicc, prob_bicc, FAPs, flags)
lmfit_solutions/       phase-folded + residual plots per solution (sid-<n>)
periodogram/           LS + PDC periodograms, permutation FAPs
mcmc/                  Route B only: corner plots, phase bands, chain summaries
```

The RVs themselves are in `<base_dir>/BLOeM_<ID>_CCF_RVs.csv` (pass 2) and
diagnostic plots are in `<base_dir>/visuals/`.

## Troubleshooting

- **Fitting stops at 0 % CPU and nothing changes for many minutes.** On macOS, a
  plotly `write_image` call (kaleido) occasionally deadlocks a worker, roughly
  once every few dozen stars. Kill the run and rerun the same command; finished
  stars are skipped. Setting `plot: false` in `params.yaml` avoids this, but
  then no plots are written. `lmfit_summary.csv` is still written.
- **`No rows match the filter expression`.** No candidate passed the detection
  criteria, so the star is treated as having no orbital solution and gets the
  null-model MCMC.
- **`ModuleNotFoundError: PDC.pdc_func`.** See *PDC module* above.
- **`KeyError: ... missing columns ['fixed Mean RV', 'fixed Mean RVsig']`.** The
  correction step didn't run on that file, or pass 2 was rerun afterwards. Run
  `apply_rv_corrections` again (A1b).
- **`apply_rv_corrections`: `rows beyond tolerance` / `no corrections for field N`.**
  An epoch in the CSV has no entry in the corrections table. Don't raise
  `--tolerance`: the closest epochs in field 8 are only 0.0185 d apart, so a
  larger tolerance can match the wrong epoch. Check which table version you have.
- **`lmfit_summary.csv not found` / `FileNotFoundError` on `_CCF_RVs.csv`.**
  `base_dir` doesn't match the pass-2 `path_to_output`, or pass 2 didn't run for
  that star.
