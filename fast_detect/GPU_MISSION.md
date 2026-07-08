# Mission: fully GPU-optimize `fast_detect` on the M4 Max, then benchmark vs astro3

**Goal.** Make the MLX (Apple-Metal) backend of `fast_detect` deliver its *real*
end-to-end throughput on the M4 Max GPU, with results that match the numpy/CPU
backend, then **benchmark the full `detect_batch` on the M4 GPU against a 94-core
astro3 run** to settle: *is the optimized Mac GPU faster than astro3's 94 CPU
cores for this workload?*

Read `fast_detect/HANDOFF.md` first for the full package background (what the
detector is, every kernel, how it's validated, and how it's wired into the bias
grid as `detect_method: pipeline_batched`). This doc is the mission spec; that doc
is the reference.

Env for everything: `conda run -n tau_binary python …` with
`PYTHONPATH=/Users/roeyovadia/Roey/Masters/Reasearch/Scripts`. `mlx` 0.31.2 is
installed in `tau_binary` (Apple-only; absent on astro3). Write all outputs under
`$SCRATCH` (a scratchpad dir), NEVER into the repo. Nothing is committed.

---

## Why this is needed (current state, measured)

The MLX backend *works* but is **un-optimized**, so a GPU benchmark today would
badly understate it:
- **Transfer-throttled**: `BatchedLS.peak`/`BatchedPDC` compute the power for each
  frequency chunk on the GPU, then `xp.to_numpy(P)` copies it to host and does the
  `argmax`/running-max in **numpy on the CPU** (~24 chunks × copies for ~96k LS
  freqs). The reduction is not on the GPU.
- **Only the periodogram GEMM is on the GPU.** `batched_pdc._value_distance_uc`,
  the FWHM `_local_power`, and the **entire `batched_fit`** (Kepler template SSR)
  are pure host-numpy. So the fit — a big fraction of per-injection cost — never
  touches the GPU; on the Mac it runs on ~12 CPU cores.
- **float32 peak-flips**: MLX defaults to float32; the `argmax` over ~96k
  near-degenerate LS peaks flips on ~**0.2%** of curves vs numpy (a true period vs
  an alias round the wrong way). Affects results, not timing.

Measured so far: MLX `max_power` = **0.18 ms/curve** vs numpy **1.79 ms/curve**
(10×, but transfer-throttled, and the numpy baseline was unpinned = all 16 Mac
cores). Field-batched numpy = **6.37 ms/injection** (8 thread-pinned cores).
Rough (soft) extrapolation put the optimized GPU at ~1.5–3× astro3 — but that
rests on one throttled kernel, hence this mission.

**Which part gates the benchmark:** `2b` (on-device + fit-on-GPU) is what makes a
GPU throughput number meaningful. `2a` (float32 fidelity) is for correctness, not
speed. Do `2b` for the benchmark; do `2a` before trusting/using GPU results.

---

## The work

### 2b — on-device reductions + GPU-resident kernels (the throughput fix)
Files: `fast_detect/backends.py`, `batched_ls.py`, `batched_pdc.py`,
`batched_fit.py`.

- **2b-0 (foundation): extend the MLX shim** in `backends.py` (`_MlxBackend`) with
  whatever the on-device reductions need — `argmax(axis=...)`, `take_along_axis`,
  `maximum`/`where` for a running max, and a documented float32/float64 policy
  (see 2a). Verify each op exists in `mlx.core` (some names/semantics differ from
  numpy). Everyone else depends on this — do it first.
- **2b-i: `batched_ls`** — keep the running **max + argmax over frequencies ON the
  backend** across chunks (maintain `best_pow`/`best_idx` as backend arrays,
  update per chunk with backend `where`), and transfer only the final `n_batch`
  peaks + indices at the end. Applies to `_scan`/`peak`/`max_power`; `peak_with_fwhm`'s
  `_local_power` can stay host (small) or move too.
- **2b-ii: `batched_pdc`** — same on-device freq reduction. `_value_distance_uc`
  (uses `scipy.special.erf`) is computed **once per batch**, not per freq — leave
  it on host and upload `A`, unless `mlx.core.erf` exists and it's easy. The
  per-freq `B` GEMM + reduction is the hot path → on GPU.
- **2b-iii: `batched_fit`** — move the Kepler **template-SSR scan** onto the
  backend. The (e, ω, phase) template grid is shared; per injection only P (and
  the linear γ,K solve) vary. Options: (a) reimplement the Kepler solve +
  `_ssr_linear` in backend ops (MLX has no numba; a fixed-iteration Newton for E
  is fine since e<0.95); or (b) precompute the shared templates on host and do
  only the batched projections/SSR on the GPU. **Profile first** (after 2b-i/ii):
  if the periodogram now dominates and the fit is small, 2b-iii may not be worth
  it — measure before investing.

### 2a — float32 fidelity (correctness)
Files: `batched_ls.py`, `batched_pdc.py` (the peak/argmax sites).
- The ~0.2% flips are float32 rounding on near-equal peaks. Fix so the GPU peak
  index matches the CPU: do the **power normalization + argmax in float64** at the
  reduction (MLX float64 may fall back to CPU — verify; if too slow, an
  alternative is a deterministic tie-break, or re-evaluating the top-K candidate
  frequencies in float64). Note the tension with 2b (on-device float32 reduction)
  — resolve by keeping the GEMM float32 but the final compare/argmax float64.

### 3 — the benchmark (the actual question)
- **GPU run**: full `detect_batch` / a `pipeline_batched` bias-grid sub-grid on the
  Mac with `fast_detect_backend: mlx`, `parallel_grid: false` (ONE process drives
  the GPU — never combine with N workers). Measure ms/injection.
- **astro3 run**: the SAME sub-grid + n_inject with `backend: numpy`,
  `parallel_grid: true`, thread-pinned
  (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1`),
  `n_workers` ≈ 94. Needs TAU VPN + `./sync_astro3.sh` (syncs `fast_detect/` +
  configs; astro3 uses the numpy backend since mlx is Apple-only). Run per the
  astro3 recipe in `~/.claude` memories (`astro3-thread-oversubscription`,
  `slurm-bias-grid`).
- **Report**: ms/inj GPU vs astro3, and extrapolate `final5_cvm_1e3` @ n_inject=100
  wall-time for each. That answers the question.

---

## Validation gates (do NOT skip)
After each of 2b-i / 2b-ii / 2b-iii / 2a, assert **numpy ↔ mlx parity** on a
reference batch (a few hundred synthetic RV curves at a real BLOeM MJD array):
- periodogram: peak **period** identical, peak **power** ~4 sig figs;
- `iter_fap`, `prob_bicc`: match within MC/float tolerance;
- final `detect_batch` `detected` mask: agreement ≥ 99.8% (2a should push LS/PDC
  peak agreement to ~100%).
Reference patterns exist in `fast_detect/oracle_parity.py` and the (ephemeral,
scratchpad) `mlxbench.py` — recreate a small parity script; keep it under `$SCRATCH`.
Also re-run the thread-pinned `pipeline_batched` smoke (quick preset, n_inject≈20)
after wiring to confirm the cube is still correct.

## Gotchas
- GPU mode = `parallel_grid: false` + `backend: mlx`; CPU mode = thread-pinned +
  `backend: numpy`. Never mix (8 processes contending for one GPU; or BLAS
  oversubscription — an un-pinned 8-worker CPU run hit load 114 and stalled).
- MLX is lazy/async — force eval (`mx.eval` or a `to_numpy`) before timing.
- `detector._get_pgrams` caches BatchedLS/BatchedPDC by MJDs; the cache key
  includes the backend, so numpy and mlx don't collide.
- The LS grid is ~96k freqs regardless of `pmax` (density = baseline×samples_per_peak).

---

## Suggested multi-agent plan (for speed + less context loss)

The three sub-tasks of 2b touch **different files**, so after the shared
foundation they parallelize cleanly. Recommended structure (orchestrate from the
main chat; each agent returns a short structured result, not file dumps):

1. **Sequential foundation (orchestrator or 1 agent):** `2b-0` — extend
   `backends.py` MLX shim (+ decide the float64 policy for 2a). Everything depends
   on it, so land it first and confirm it imports.
2. **Fan out 3 agents IN PARALLEL** (disjoint files → no conflicts), each doing its
   module's 2b + 2a slice AND its own numpy↔mlx parity check:
   - Agent B → `batched_ls.py` (2b-i on-device reduction + 2a float64 argmax).
   - Agent C → `batched_pdc.py` (2b-ii + 2a).
   - Agent D → `batched_fit.py` (2b-iii) — but instruct it to **profile first** and
     only move the fit to GPU if it's a real fraction of cost; else report that.
   Give each agent: this doc, `HANDOFF.md`, the exact functions to change, and the
   parity-gate requirement. Each returns: what changed + its parity numbers.
3. **Sequential integration (orchestrator or 1 agent):** end-to-end
   numpy↔mlx `detect_batch` parity + the thread-pinned cube smoke + a Mac-GPU
   throughput number. Fix any cross-module seams.
4. **Sequential, human-in-loop:** the astro3 benchmark (needs VPN + sync + a long
   run) — do this from the main chat, not a fire-and-forget agent, so the VPN/auth
   and the 94-core run are supervised.

Why this shape: the parallel B/C/D is where the wall-clock savings are (three
delicate numerical rewrites at once), the disjoint-files rule prevents edit
conflicts, and each agent self-validates so the orchestrator only merges *verified*
modules — minimizing context bloat in the main thread. If using the `Workflow`
tool: one phase for 2b-0, a `parallel()` phase for B/C/D (each gated by its parity
schema), then an integration phase; keep D's "profile-first" branch so it can bail
cheaply.

Start each fresh session by having the agents re-read this file + `HANDOFF.md`
rather than relying on prior context.
