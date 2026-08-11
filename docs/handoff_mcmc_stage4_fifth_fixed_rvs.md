# Handoff: Stage-4 MCMC on `fifth_fixed_rvs`

*Written 2026-08-11 while the Stage-2 run was still in progress. Audience: the
session that will run `python -m mcmc.batch` on the fifth generation.*

This note covers two things you need before running Stage 4:

1. **Stuck-walker pruning** — a new, default-OFF mechanism now wired through the
   MCMC stack. It has **not been smoke-tested end to end**; see §3.
2. **What Stage 2 changed** in the fifth run, because it alters the input table
   the batch runner reads and adds columns you may want to use.

---

## 1. Stuck-walker pruning — what exists now

The recipe adopted in the 2026-07 clean reruns
(`Ostars_article/proposed_phase_figures/scripts/rerun_circ_mcmc_*.py`) is now a
library function wired through the pipeline. **Sampling is never affected** —
pruning only filters what the runner *returns*, and the saved `.npz` always
keeps the full raw chain plus the mask, so it is reversible and auditable.

### `mcmc/analysis.py::prune_stuck_walkers(chain, log_prob, delta_mean=10.0, delta_competitive=5.0)`

- Input is the **walker-resolved** post-burn chain `(n_steps, n_walkers, ndim)`
  and `log_prob (n_steps, n_walkers)`.
- A walker is **stuck** when its *mean* ln-posterior is more than `delta_mean`
  (10) below the best walker's mean.
- **Competitive-mode guard**: a flagged walker whose *single best* sample is
  within `delta_competitive` (5) of the global maximum is kept — it is sampling
  a real secondary mode, and dropping it would hide genuine multimodality.
  Implementation is `keep = (walker_mean >= walker_mean.max() - delta_mean) |
  competitive`, so the guard always wins.
- Returns `(keep_mask, info)` and prints what it dropped/kept.
- Thresholds are **function arguments, not config keys** — copied verbatim from
  the adopted scripts. Promote them to YAML if you want them tunable.

### `mcmc/runner.py::run_mcmc_generic(..., prune=False)`

The single choke point. It now pulls the walker-resolved chain instead of
`flat=True`, computes the keep-mask when `prune=True` (all-True otherwise), and
builds the returned flats from kept walkers only:

```python
flat_samples = chain[:, keep, :].reshape(-1, ndim)
flat_logprob = log_prob[:, keep].reshape(-1)
```

With `prune=False` this step-major/walker-minor reshape is intended to be
**bit-identical** to emcee's `flat=True` ordering. All four wrappers
(`run_mcmc_ecc`, `run_mcmc_circ`, `run_mcmc_null`, `run_mcmc_double_kepler`)
take and forward `prune=False`.

### `mcmc/chain_io.py` — auditability

- `save_chain(..., keep_walkers=None, prune_info=None)`: the npz **always**
  stores the full raw chain; the boolean `keep_walkers` mask is a stored field
  and the prune parameters + dropped indices go into `meta_json` under
  `"prune"`.
- `load_chain(...)` backfills an all-True mask for pre-pruning files
  (`mcmc_out4` and earlier) and returns **both** `flat_samples`/`flat_logprob`
  (raw) and `flat_samples_pruned`/`flat_logprob_pruned`.

### Config control

- New key `PRUNE_STUCK_WALKERS = "prune_stuck_walkers"` in `utils/constants.py`.
- All **three** batch call sites (null / ecc / circ) pass
  `prune=args_dict[MCMC_PARAMS].get(PRUNE_STUCK_WALKERS, False)`
  (`mcmc/batch.py` lines ~95, ~153, ~202).
- **Default is off, and no config in `configs/` sets it** (verified). Enable with
  one line under `mcmc_params:` in the config you run:

```yaml
mcmc_params:
  prune_stuck_walkers: true
```

### Downstream consequences when the flag is ON

- `summarise_chain` receives pruned flats → `results_*.txt`,
  `percentiles_*.csv`, and the aggregate `mcmc_params.csv` all become
  pruned-based.
- Corner and phase-band plots receive the same pruned flats.
- **The Lucy–Sweeney gate consumes the pruned `e` statistics**, so the
  ecc/circ decision itself becomes pruning-aware.

---

## 2. What Stage 2 changed for the fifth generation

Stage 4 reads `.../dr5_neb_div_from_coadded/fifth_fixed_rvs/*/lmfit_summary.csv`
and writes to `fifth_fixed_rvs/mcmc_out/`. Run it with the same config as
Stage 2 so paths and knobs match:

```
python -m mcmc.batch --config configs/params_dr5_fifth_fixed_rvs.yaml
```

Relevant differences vs `fourth_fixed_rvs`:

- **RV columns are unchanged**: `rv_col: "fixed Mean RV"`,
  `rverr_col: "fixed Mean RVsig"` (same corrected inputs, not regenerated).
- **`lmfit_summary.csv` now has 116 columns** (was 103). The 13 new ones are a
  per-candidate **circular fit** (e=0, ω=π/2 fixed, 5 free params):
  `circ_bicc, circ_llh, circ_chisqr, circ_redchi, circ_nfev, circ_nvarys,
  circ_Period_value, circ_K1_value, circ_GAMMA_value, circ_T0_value,
  circ_ln_sigmaJ_value, p_bicc_kep_vs_circ, prob_bicc_circ`.
  `p_bicc_kep_vs_circ → 1` means the eccentric model is preferred;
  `→ 0` means circular is. The `filter_expression` is untouched, so best-row
  selection behaves exactly as before — the new columns are informational
  unless you choose to use them.
- **Period prior is now configurable**: `mcmc_params.dP_frac` (box half-width)
  and `mcmc_params.sigma_P_frac` (Gaussian width), both fractions of the lmfit
  period. The fifth config sets them to the historical values **0.01 / 0.002**,
  so Stage-4 sampling is unchanged by default.
  **Note the bug this fixed:** `sigma_P_frac` was previously never threaded
  through `log_probability`, so the ±0.2 % Gaussian was hard-wired and any
  attempt to widen it was silently ignored. It now takes effect. For the
  long-period tail (P ≳ baseline/2 ≈ 400 d) the posterior width is
  prior-dominated; widening `sigma_P_frac` there is now actually possible, but
  it is a science decision — the periods quoted for those stars should be
  treated as prior-bounded until it is made deliberately.
- **`fwhm_window_factor: 1.5`** caps each candidate's fit window at
  `[P/1.5, 1.5P]`. Chosen empirically over a 7-star sweep: it recovers
  BLOeM_2-035's better orbit (P=397.5 d, e=0.63, BICc 66.71 → 50.97, jitter
  0.80 → 0.20 km/s) with ≤0.07 BICc cost elsewhere and **zero** effect on
  well-determined systems. **2-035's best row will differ from `mcmc_out4`.**
- **`de_seed: 562`** with per-fit derivation, so Stage-2 fits are now
  reproducible (they were not in the fourth run). Expect small differences vs
  `fourth_fixed_rvs` on marginal stars for this reason alone.

### A free convergence diagnostic you should use

The circular model is a strict subset of the Keplerian, so it can never
legitimately fit better. Any row with `circ_llh < llh` therefore proves the
**Keplerian** DE missed its global optimum (`llh` is a cost — lower is better).
During the fifth run this fired on ~4 % of candidate rows, worst case ~18.7 llh
units, but on **none** of the winning rows sampled mid-run.

Before running Stage 4, audit it on the finished table:

```python
import pandas as pd, glob
F = ("~candidate_method.str.contains('MANUAL') & ~candidate_method.str.contains('null')"
     " & candidate_method.str.contains('jitter') & (prob_bicc > 0.5) & (mass_flag_peri == 0)"
     " & ((candidate_method.str.contains('LS') & (LS_iter_fap < 0.2))"
     " | (candidate_method.str.contains('PDC') & (PDC_iter_fap < 0.2)))")
for f in sorted(glob.glob('.../fifth_fixed_rvs/*/lmfit_summary.csv')):
    df = pd.read_csv(f)
    b = df.query(F).sort_values('bicc')
    if not b.empty and b.iloc[0]['llh'] - b.iloc[0]['circ_llh'] > 1e-6:
        print(f, 'BEST ROW under-converged')
```

Any star printed here has an unreliable lmfit solution, and MCMC seeded from it
will faithfully sample the **wrong** basin — MCMC cannot recover from a bad
lmfit period. Fix the fit before trusting its posterior.

---

## 3. Things to check before production use

1. **Not smoke-tested end to end.** The wired batch → wrapper → generic path
   with `prune=True` has never been executed (the 8-073 test run was cancelled).
   The recipe itself was validated on saved chains, but **run one single star
   with the flag on before any production use.**
2. **`prune=False` regression.** The claim that the reshape is bit-identical to
   `flat=True` was verified for the earlier `chain_io` work but not re-confirmed
   after this refactor. Cheap to check on one star:
   compare `run_mcmc_*(..., prune=False)` flats against
   `sampler.get_chain(discard=nburn, thin=thin, flat=True)`.
3. **Lucy–Sweeney flips.** With pruning on, borderline stars can reclassify.
   On the current chains nothing flips (3-046 at 2.43 stays circular), but watch
   the gate printout on the next real run.
4. **Null models get pruned too** — 2-parameter fits with acceptance ~0.7. The
   competitive guard should make this a no-op; eyeball a couple of null prune
   printouts to confirm.
5. **Double-Kepler**: the wrapper has the passthrough, but the out-of-band 4-043
   scripts do not use it. Intentional — the adopted solution is untouched.
6. **Double clipping in corners**: `make_corner` still applies its own
   display-level `dlogp_clip=15.0` on top of pruned samples. Benign, but know
   it is there.
7. **Thresholds** (Δmean = 10, Δcompetitive = 5) are verbatim from the adopted
   scripts and are function arguments — expose them in YAML if you want them
   configurable per run.
8. **`mcmc/chain_io.py` is untracked** but imported at module level by
   `mcmc/runner.py`. Committing the other changes without
   `git add mcmc/chain_io.py` breaks `import mcmc.runner` and every Stage-4
   entry point.

## 4. Where this leaves the original goal

The mechanism is in the pipeline, but **the current paper still reads the
adopted overrides**: the `mcmc_out4` summaries on disk are unpruned, so 8-073
and 3-014 remain contaminated in the tables, and the five stuck-walker
adoptions still live in `proposed_phase_figures/`.

Two ways to resolve it, Roey's call:

- **In-place post-processing** of the saved `mcmc_out4` chains (they are raw and
  reloadable via `load_chain`, so summaries can be regenerated without
  re-sampling), or
- **Fold pruning into the fifth-generation Stage-4 run** by setting
  `prune_stuck_walkers: true` alongside the σ_P work — which is the natural
  moment, since the fifth run's lmfit inputs have changed anyway.

The second is cleaner if the paper is moving to fifth-generation numbers; the
first is the only option if the fourth-generation tables must stand.
