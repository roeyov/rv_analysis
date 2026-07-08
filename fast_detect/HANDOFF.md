# fast_detect — Handoff

Batched, backend-pluggable re-implementation of the BLOeM binary-**detection**
decision, built to accelerate `detect_method` in the bias grid. The original
production stack (`pipeline/`, `period_search/`, `orbital/`, `PDC/`,
`utils/roche_lobe.py`) is **untouched** and serves as the validation oracle.

Full design + rationale: `~/.claude/plans/now-after-plenty-iterations-sharded-sphinx.md`.

---

## 1. Why this exists

`detect_method: pipeline` runs the full periodogram + permutation-FAP + lmfit +
BICc stack **per injected system** (~8.5 s/detection). At 134 stars × n_inject
per grid cell, full `final5_cvm_1e3` (380,160 cells) at n_inject≈100 is **~1
year** on astro3 — infeasible. Goal: reproduce the *same detection decision* over
a whole population of injections at once (batched linear algebra), on NumPy/BLAS
(astro3 + Mac CPU) or MLX/Metal (Mac GPU).

### Two correctness findings that shaped everything
1. **The true detection criterion is the `filter_expression`** (configs/params.yaml:96):
   detected iff some LS/PDC candidate satisfies
   `prob_bicc > 0.5  AND  mass_flag_peri == 0  AND  iter_fap < 0.2`
   (iter_fap = the **permutation** empirical FAP → permutations are REQUIRED).
2. **The current bias-grid `detect_from_arrays` is unfaithful**: it decides on
   `prob_bicc > 0.5` **alone** — it runs the permutations and computes
   `LS_iter_fap`/`PDC_iter_fap` then **discards them**, and never applies the
   Roche-lobe (`mass_flag_peri`) cut. So **past pipeline-mode bias grids
   over-count detections.** `fast_detect` implements the full 3-condition gate.

Profiling of one detection (8.5 s): **64% is the permutation FAP** run via a
`ProcessPoolExecutor` *per detection* → nested multiprocessing inside 94 grid
workers (the `select.poll` = 6.9 s tottime). Batching turns that worst
bottleneck into the most GPU-friendly kernel (a GEMM).

---

## 2. What was built (all in `fast_detect/`)

| Module | Role | Validation vs original |
|---|---|---|
| `backends.py` | array-module abstraction `get_xp("numpy"\|"mlx"\|"auto")`; MLX optional w/ graceful fallback | — |
| `batched_ls.py` | batched floating-mean Lomb-Scargle (`normalization='model'`): peak period, peak power, **FWHM period window** (`peak_with_fwhm`) | peak period **exact**, power 39.580 vs 39.582 (astropy), spectrum corr **0.99995**; FWHM bounds **exact** (e.g. [4.7077,4.7134]=pipeline) |
| `batched_pdc.py` | batched PDC periodogram (U-centered value-dist `A` × shared phase-dist `B(f)` → GEMM), peak + FWHM | peak period/power **exact**, spectrum corr **1.00000** vs `pdc_opt` |
| `batched_perm.py` | batched permutation null → `iter_fap` (`iter_fap_fn` generic over LS/PDC) | `iter_fap` **0.090 = 0.090** vs astropy shuffle loop; null max-power dist matches |
| `batched_fit.py` | batched orbital fit (P free in FWHM; linear in γ,K; **jitter profiled in closed form**) + null → `orb_bicc`/`null_bicc`/`prob_bicc` | `prob_bicc` decision **22/22** vs lmfit DE |
| `rlof.py` | batched `mass_flag_peri` — **reuses** `utils.roche_lobe.compute_min_period_row` | **bit-exact** (same code); guards brentq-NaN on degenerate fits |
| `detector.py` | `detect_batch()` — the full `filter_expression` OR-gate `(LS ‖ PDC)` | end-to-end: 18/18 binaries, 0/4 singles |
| `oracle_parity.py` | per-cell P_det parity harness + **faithful all-candidate oracle** | see §3 |

Key idea that makes the fit batchable: with **uniform per-star errors** the LS
'model' power is error-independent and the jitter σ_j **profiles out in closed
form** (`sig2* = clip(SSR/N, ...)`), so the orbital fit collapses to *minimising
the linear-LS residual SSR over (P,e,ω,T0)* — a batched Keplerian template match.

Reused (not reimplemented): `simulations/bias_grid_lib/physics.py`
(`compute_K1`, `kepler_E_batch`, `rv_model_jit`, `powerlaw_draw`),
`orbital/statistics.py` (BICc + `calculate_binary_probability`),
`period_search/periodogram.py` (`make_pdc_freq_grid`), `utils/roche_lobe.py`.

---

## 3. Results

**Per-kernel validation: all exact** (see table above).

**Per-cell detection-fraction parity** vs the *faithful all-candidate* oracle
(fits every LS/PDC candidate, applies `filter_expression` per row), NPERM=100,
n=40 injections/cell:

| Cell | P_det oracle | P_det fast | 2σ MC | per-inj agree |
|---|---|---|---|---|
| π=−0.7,κ=−0.5,η=−0.5 | 0.550 | 0.625 | ±0.16 | 93% |
| π=−1.2,κ=−1.0,η=−0.2 | 0.325 | 0.225 | ±0.15 | 85% |

Both **within 2σ**; residual is **MC-noise-limited** (small n + the oracle's own
un-seed-fixed permutation `iter_fap`, ~±0.05 run-to-run), not a structural bias.
Adding the PDC branch raised cell-1 agreement 88→93% and closed the "missing PDC"
gap identified in the LS-only diagnosis.

**Benchmark** (real config pmin 0.4 / pmax 10000 → LS 96k freqs, PDC 19k;
n_perm=10):

| Path | per-injection | speedup | full `final5` @ n_inject=100 (5.1e9 inj) |
|---|---|---|---|
| pipeline (orig) | 8.5 s | 1× | ~1 year (94 cores) |
| batched numpy | **27.5 ms** | **310×** | ~16 days (astro3, 96 cores) |
| batched MLX (M4 GPU) | ~3–5 ms (periodogram-limited) | ~10× on periodogram | ~2–4 days (Mac only) |

MLX `max_power`: **0.18 ms/curve vs 1.79 ms** (numpy) = 10×, **99.8%** peak
agreement (0.2% float32 peak-flips on near-degenerate spectra).

---

## 3b. Phase 3 — WIRED INTO THE BIAS GRID (done)

`detect_method: "pipeline_batched"` is registered
([detection.py](../simulations/bias_grid_lib/detection.py)) and dispatched via
`_WORKER_BY_METHOD` in both the parallel and sequential engine paths
([parallel.py](../simulations/bias_grid_lib/parallel.py),
[engine.py](../simulations/bias_grid_lib/engine.py)). The batched worker
`_worker_star_injections_batched`
([injection.py](../simulations/bias_grid_lib/injection.py)) draws the same
population as the vectorized worker and calls `fast_detect.detect_batch` once per
star (LS/PDC periodograms cached by MJDs in `detector._get_pgrams`). **Verified
end-to-end**: an 81-point `quick` run with the single-variant/CvM filter produced
a correct schema-5 cube (`variants=['eccentric_only__numerical__lucyT']`,
`metrics=['cvm']`), explorer-compatible.

**How to launch (CPU) — MUST thread-pin:**
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python -m simulations.bias_grid --config <cfg with bias_grid.detect_method: pipeline_batched>
```
Config knobs: `bias_grid.detect_method: pipeline_batched`; top-level
`fast_detect_backend: numpy|mlx` (default numpy). Reuse the single-variant/CvM
`variant_filter` + `scored_tests` for a clean cube.

## 4. Open ends / caveats

- **Thread-pinning is mandatory for the CPU path.** `parallel_grid`×`n_workers`
  parallelizes grid points AND each worker's numpy/BLAS GEMM spawns ~N_core
  threads → they multiply. Un-pinned, an 8-worker Mac run hit **load 114 on 16
  cores and stalled (0 points in 28 min)**; pinned (env above) → load 9, healthy.
  Same rule as the astro3 runs.
- **Per-star batching is inefficient at small n_inject.** The worker batches only
  `n_inject` injections per `detect_batch` call, so the batched throughput is
  wasted when n_inject is tiny (the smoke's n_inject=2 → 81 pts took 25 min).
  n_inject=100 amortizes much better; the real optimization is **field-cell
  batching** — group all stars of a field into one `detect_batch` call
  (batch≈13,400) — which needs a per-field task shape rather than per-star.
- **RLOF fidelity**: the batched worker builds a zero-error `mass_row`
  (`injection._zero_err_mass_row`) because the task tuple carries only M1/R1
  scalars, not the ±catalog errors → `mass_flag_peri` uses central M/R (a small
  approximation; thread the error columns through the task for exactness).
- **Scale**: 310× (CPU) still leaves n_inject=100 full grid at ~days, because the
  work is 5.1e9 injections. The real levers are **lower n_inject** (orig pipeline
  used n_inject=2; n=10 ≈ 1.6 days astro3-CPU / ~4 h Mac-GPU) and/or a **coarser
  grid** (`final5_cvm_1e3_fast`, 6× fewer) — a user trade-off, not yet chosen.
- **MLX float32**: 0.2% peak-flips near degenerate spectra (within MC noise but a
  fidelity caveat). Fix: float64 accumulation for the peak argmax, or a tie-break.
- **GPU efficiency**: `detector`/`batched_ls`/`batched_pdc` transfer power to host
  per freq-chunk (`to_numpy`) and do argmax on host; `_value_distance_uc`,
  `_local_power`, and `batched_fit` are pure host-numpy. So MLX only accelerates
  the periodogram GEMM today. Full GPU throughput needs on-device running-argmax
  and moving the value-distance / fit to the backend.
- **Parity precision**: only n=40 × 2 cells measured. A large-N sweep (n≈200,
  seed-fixed oracle `iter_fap`) is needed to pin the residual sub-percent.
- **fit grid**: `fit_batch` uses (n_e=10, n_omega=12, n_phase=12, n_period=3);
  may over-fit high-e/long-P aliases (a suspected contributor to cell-1's +0.075).
  Tunable; validate BICc against DE on eccentric cases.
- **FFT-batched LS**: flagged in the plan but low-value here (N=26 epochs → dense
  per-freq is already ~O(Nf·N) with tiny N).
- `mlx` was `pip install`ed into the `tau_binary` env (Apple-only; absent on
  astro3, which uses the numpy backend).

---

## 5. What to do next (prioritized)

1. **Field-cell batching** (biggest throughput win): restructure the injection
   task from per-star to per-(field-cell) so one `detect_batch` call handles all
   stars' injections sharing a field's MJDs (batch≈13,400 vs ~n_inject now). This
   is what realizes the batched/GPU throughput; per-star at n_inject=2 wasted it.
2. **Set the n_inject × grid vs wall-time target** (user decision) — decides
   whether it's hours or days. Benchmark: numpy 27.5 ms/inj; n_inject=10 on the
   fast grid ≈ hours.
3. **MLX fidelity + throughput**: float64 argmax tie-break (0.2% peak-flips);
   on-device running-max; move value-distance/fit to the backend. GPU mode is
   `parallel_grid: false` + `backend: mlx` (one process drives the GPU — do NOT
   combine with N workers).
4. **Per-cell P_det parity** of a `pipeline_batched` grid vs a `pipeline` grid on
   the same seed (integration-level parity); + large-N kernel parity (n≈200).
5. **(Optional) fix the existing slow path**: make `detect_from_arrays` apply the
   `filter_expression` it already computes so non-batched `pipeline` runs are
   faithful too — a zero-extra-cost correctness fix.

---

## 6b. Field-cell batching — DONE (implemented + verified)

Implemented: `_worker_field_injections_batched`
([injection.py](../simulations/bias_grid_lib/injection.py)) pools all of a
field's stars into one `detect_batch` call; a shared `_build_injection_tasks` +
`_FIELD_WORKER_BY_METHOD` ([parallel.py](../simulations/bias_grid_lib/parallel.py))
emits one task per (field, cell) for `pipeline_batched` at both dispatch sites
(engine + parallel). **Verified**: injected population bit-identical to the
per-star path (same per-star seeds); detection differs only by permutation-FAP MC
noise (deterministic field-wide `det_rng`); end-to-end thread-pinned smoke
produced the correct single-variant/CvM cube. **Throughput: 6.37 ms/injection**
(81 pts × 134 stars × 20 inj, 8 pinned workers) vs **~70 ms/inj** for the per-star
path (n_inject=2) → **~11× faster per injection.** The spec below is retained for
reference.

**Motivation.** The bias grid's work unit was one **star**, so the
`pipeline_batched` worker calls `detect_batch` with only that star's `n_inject`
injections — a tiny batch that barely amortizes the periodogram design + GEMM
(the n_inject=2 smoke took 25 min/81 pts). But **all stars in a field share one
of only 8 MJD arrays**, and the LS/PDC periodogram design depends *only* on the
MJDs — so pooling **every star's injections within a field** into a single
`detect_batch` call (batch ≈ stars_in_field × n_inject ≈ 1k–13k) is what actually
realizes the batched/GPU throughput. This is the single biggest speed lever left.
**Hard correctness rule: batch strictly by field** — never mix stars from
different fields in one call, or the shared periodogram design (built for one MJD
grid) is wrong.

**Instructions.**
1. **New worker** `_worker_field_injections_batched(field_task)` in
   [injection.py](../simulations/bias_grid_lib/injection.py), beside
   `_worker_star_injections_batched`. The `field_task` carries the field's shared
   `MJDs` + per-star arrays (`star_seeds`, `rv_err`, `M1`, `R1`, `gamma`) for all
   stars in ONE field, plus `n_inject, f_bin, pi, kappa, eta, cfg, args_dict`.
   Per star, run the *same* draws as `_worker_star_injections_batched` (reuse that
   body per star — seed each star with its own `star_seed` for reproducibility);
   collect per-injection `(logP,e,K1,q, rv_obs, err, mass_row)` for **binaries**,
   and the noise-only `(rv_obs, err, mass_row)` for **singles**, across all stars.
   Then **concatenate all binaries → one `(n_epoch × N_bin)` matrix** (+ `errs`,
   `mass_rows`) and call `detect_batch` **once**; same for singles. Aggregate to
   **one** result dict with the existing per-star schema (summed
   `n_physical/n_detected/n_false_positive`, concatenated det/nondet arrays,
   summed `hist_total`/`hist_det`).
2. **Build per-FIELD tasks** when `detect_method == "pipeline_batched"` at BOTH
   task-building sites — `engine._run_one_grid_point`
   ([engine.py:124-156](../simulations/bias_grid_lib/engine.py)) and
   `parallel._worker_grid_point`
   ([parallel.py:156-177](../simulations/bias_grid_lib/parallel.py)): group star
   indices by field, emit one task per (field, cell) with that field's per-star
   arrays, and route to `_worker_field_injections_batched`. Keep the per-star path
   for all other methods (extend `_WORKER_BY_METHOD` or branch on the method).
   The result-aggregation loops downstream already sum over a *list of result
   dicts*, so returning per-field dicts (fewer, larger) needs no change there.
3. **No detector change** — `detect_batch(mjds, rv_matrix, errs, mass_rows, cfg,
   backend, rng)` already takes per-injection `errs`/`mass_rows`, and
   `detector._get_pgrams` caches the periodogram objects by MJDs (built once per
   field, reused across cells).
4. Keep the **thread-pin** launch requirement (§3b) and the numpy/mlx backend
   switch unchanged.

**Verify.** Re-run the thread-pinned `pipeline_batched` smoke (quick preset) →
confirm it still produces the correct schema-5 single-variant/CvM cube, and
measure wall-time vs the per-star version (expect several×). Optionally confirm
per-cell P_det is statistically consistent with the per-star run on the same seed.

## 6. How to run

```bash
# per-cell parity vs faithful oracle (slow: runs the production DE per candidate)
conda run -n tau_binary python -m fast_detect.oracle_parity

# standalone detect_batch: see fast_detect/detector.py::detect_batch
#   detect_batch(mjds, rv_matrix[n_epoch,N], errs[N], mass_rows[N], cfg,
#                backend="numpy"|"mlx", rng=..., fap_thresh=0.2)
```
Kernel-level validations (LS/PDC/perm/fit vs pipeline) were run as ad-hoc scripts;
re-create from the numbers in §3 if needed. The bias configs set
`periodogram_params.window_iterations` (= n_perm, default 10) — keep identical to
the pipeline for apples-to-apples `iter_fap`.
