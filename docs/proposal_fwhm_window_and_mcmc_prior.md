# Proposal: FWHM fit-window cap & configurable MCMC period prior

*2026-08-06 — arising from the BLOeM_2-035 constrained re-fit (fourth_fixed_rvs) and the
follow-up re-fits of 1-068 / 1-076 / 4-041. Status: awaiting approval, nothing implemented.*

## Background

With the zero-point-corrected RVs, a MANUAL lmfit of **2-035** confined to P ∈ [200, 500] d
found P = 397.5 d, e = 0.63, K1 = 6.2, **BICc = 50.97** — decisively better (ΔBICc ≈ 15.7)
than the pipeline's winning ~968 d quasi-circular family (BICc = 66.7), with the jitter
dropping from 0.80 to 0.21 km/s. The pipeline had the right candidates (418.9 d and 468.8 d
PDC peaks, ~5% and ~18% from the true period) but both *escaped* to the 968 d 2P-alias basin
because their fit windows were effectively **347–10000 d**.

Diagnostic re-fits of the other long-period reshuffle stars showed the pathology does **not**
repeat there: 1-068's 894 d solution re-wins inside a capped window (identical BICc from an
independent DE start), and 1-076 / 4-041 have flat likelihoods above ~1000 d (ΔBICc < 1 from
973 d to 5796 d; every constrained fit pins at its window boundary) — genuine
"lower-limit-on-P" systems, not window artifacts.

---

## Change 1 — Cap and sort the FWHM period windows

**Where:** `period_search/candidates.py` — `_maybe_add_fwhm` inside `append_period_results`
(lines ~113–127), fed by `_fwhm_bounds` (lines ~32–52).

**Current code:**

```python
def _maybe_add_fwhm(rec, method_name, idx):
    if not include_fwhm:
        return
    fL, fR = _fwhm_bounds(
        _freq_arr(method_name),
        periodogram_res[method_name]["pow"],
        idx
    )
    rec.update({
        "fwhm_freq_low": fL,
        "fwhm_freq_high": fR,
        "fwhm_per_low": 1.0 / fR if np.isfinite(fR) and fR > 0 else np.nan,
        "fwhm_per_high": 1.0 / fL if np.isfinite(fL) and fL > 0 else np.nan,
    })
```

**Two defects, one edit:**

1. **Unbounded windows.** In `_fwhm_bounds`, when the peak's half-power point runs off the
   frequency grid (`i == 0` → `fL = freq[0]`, or `j == len(power)-1` → `fR = freq[-1]`), the
   period bound becomes the grid edge, i.e. `max_period` = 10000 d. That is exactly how
   2-035's 418.9/468.8 d candidates received a 347–10000 d window (factor ~29) and fell into
   the 968 d basin, hiding the ΔBICc ≈ 16 better orbit at 397.5 d.
2. **PDC low/high inversion.** `per_low = 1/fR`, `per_high = 1/fL` is correct for the LS
   frequency ordering but inverted for PDC's, so PDC rows carry `fwhm_per_low > fwhm_per_high`
   (e.g. low = 10000, high = 347). It only works today because lmfit silently reorders
   `min > max` bounds (verified empirically: `add(min=10000, max=347.4)` → `min=347.4,
   max=10000`).

**Proposed replacement:**

```python
def _maybe_add_fwhm(rec, method_name, idx, period_value):
    if not include_fwhm:
        return
    fL, fR = _fwhm_bounds(_freq_arr(method_name),
                          periodogram_res[method_name]["pow"], idx)
    pers = [1.0 / f for f in (fL, fR) if np.isfinite(f) and f > 0]
    lo, hi = (min(pers), max(pers)) if len(pers) == 2 else (np.nan, np.nan)
    # cap the fit window when the half-power point runs off the grid
    lo = max(lo, period_value / fwhm_window_factor)
    hi = min(hi, period_value * fwhm_window_factor)
    rec.update({"fwhm_freq_low": fL, "fwhm_freq_high": fR,
                "fwhm_per_low": lo, "fwhm_per_high": hi})
```

**Threading (default-preserving):**

- `append_period_results(..., fwhm_window_factor=np.inf)` — new keyword, `inf` = today's
  behavior for every existing caller (incl. bias grid / fast_detect).
- In `find_periods`: `fwhm_window_factor = peri_params.get(PERI_FWHM_WINDOW_FACTOR, np.inf)`;
  new constant `PERI_FWHM_WINDOW_FACTOR = "fwhm_window_factor"` in `utils/constants.py`.
- Set **`fwhm_window_factor: 2.0`** in `configs/params_dr5_fourth_fixed_rvs.yaml`
  (`periodogram_params` block) for future runs.

**Why ×2 is safe:** the P/2 and 2P aliases remain reachable through the explicit
`include_harmonics` candidates — that is the intended mechanism; window bleed is not.

**Evidence of scope:** fixes the demonstrated 2-035 failure; leaves 1-068 / 1-076 / 4-041
scientifically unchanged (verified by the six constrained MANUAL re-fits, sids 10–11 in each
star's `fourth_fixed_rvs` directory).

---

## Change 2 — Configurable MCMC period prior (currently ±1% box **plus** a ±0.2% Gaussian)

**Where the prior lives:** `mcmc/models.py:23-43`:

```python
def log_prior(theta, P_center, T0_center, dT0_days, K1_center, omega_center=np.pi,
              dP_frac=0.01, sigma_P_frac=0.002, add_jitter=False):
    ...
    # Period narrow prior (Gaussian around P_center)
    Pmin = P_center * (1 - dP_frac)
    Pmax = P_center * (1 + dP_frac)
    if not (Pmin < P < Pmax):
        return -np.inf
    sigma_P = sigma_P_frac * P_center
    lp = -0.5 * ((P - P_center) / sigma_P)**2
```

**Hard-coded call sites of `dP_frac=0.01`:**

- `mcmc/batch.py:139` and `:187` (`run_mcmc_single_star`, ecc and circ calls)
- `pipeline/evaluator.py:599` and `:661` (`run_mcmc`)
- defaults in `mcmc/runner.py:51` (`run_mcmc_ecc`) and `:84` (`run_mcmc_circ`)

**Hidden bug on top:** `sigma_P_frac` is not threaded at all — `log_probability`
(`mcmc/models.py:75-85`) accepts only `dP_frac` and calls `log_prior` without
`sigma_P_frac`, so the Gaussian is always 0.2% regardless of anything upstream. The posterior
period width is therefore **prior-dominated** whenever the data allow more than ~±0.5%.

**Measured consequences:**

- 1-076 / 4-041: data cannot distinguish 973 d from 5796 d (ΔBICc < 1), yet
  `mcmc_params.csv` reports sub-percent period errors.
- The "335σ / 377σ period shifts" flagged in the run-comparison report were σ-inflated by
  this prior — the σ-scaled between-run comparisons are only meaningful for orbits the data
  actually constrain.
- MCMC can never recover from a wrong lmfit period: for 2-035 it faithfully sampled ±1%
  around the wrong 968 d solution.

**Proposed minimal change (default-preserving):**

1. New YAML knobs under `mcmc_params`: `dP_frac` (default 0.01) and `sigma_P_frac`
   (default 0.002); constants in `utils/constants.py`.
2. Read them in `run_mcmc_single_star` (`mcmc/batch.py`) and `run_mcmc`
   (`pipeline/evaluator.py`); pass through `run_mcmc_ecc` / `run_mcmc_circ`.
3. Fix the threading: add `sigma_P_frac` to `log_probability`'s signature and to the
   positional `logprob_args` tuples in `mcmc/runner.py:72-73` and `:105`, and forward it to
   `log_prior`.

**Optional stronger variant (for discussion, not part of the minimal change):** derive the
period box per star from the candidate's (capped) FWHM window instead of a fixed fraction, so
the posterior sees the same period range the fit searched — long-P systems would then report
honest, wide period posteriors automatically.

**Paper implication either way:** period uncertainties in `mcmc_params.csv` for the
long-period tail (P ≳ baseline/2 ≈ 400 d) are prior-limited and should be quoted as lower
limits / prior-bounded until those stars are rerun with a wide prior.

---

