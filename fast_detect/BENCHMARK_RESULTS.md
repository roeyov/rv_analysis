# fast_detect — GPU vs astro3 benchmark results

Answers the GPU_MISSION question: **is the MLX-optimized M4 Max GPU faster than
astro3's 94 CPU cores for the full `detect_batch` workload?**
**No — astro3's 94 cores are ~2.5× faster end-to-end.**

Measured 2026-07-02, after landing the on-device (2b) reductions in `batched_ls`
/ `batched_pdc` (running max+argmax kept on the backend across freq chunks; single
host transfer at the end). Fit port (2b-iii) **skipped** — profiling showed the
fit is only 6.9% of wall-time; the permutation FAP (84.5%) dominates and is just
repeated periodogram `max_power`, which the on-device reduction already accelerates.

## Throughput (detect_batch, real n_perm=10, 26 BLOeM epochs)

Direct `fast_detect.bench_detect` batch sweep:

| batch N | Mac M4 GPU (mlx) | astro3 1 core (numpy, pinned) |
|---|---|---|
| 100  | 10.84 ms/inj (92/s)  | 90.79 ms/inj (11/s) |
| 300  | 8.53 ms/inj (117/s)  | 123.58 ms/inj (8/s) |
| 1000 | **7.47 ms/inj (134/s)** | 125.48 ms/inj (8/s) |
| 2000 | 8.66 ms/inj (116/s)  | — |
| 4000 | 10.62 ms/inj (94/s)  | — |

GPU throughput **peaks at ~134 inj/s (N≈1000) then degrades** — the un-ported host
kernels (`_value_distance_uc`, `batched_fit`, FWHM `_local_power`) scale ∝N and
dominate at large batches. GPU is **periodogram-limited**, exactly as HANDOFF §4 warned.

## Head-to-head

| config | throughput | note |
|---|---|---|
| Mac M4 GPU, 1 process (best) | **134 inj/s** | direct bench, no harness overhead |
| Mac M4 single CPU core (pinned) | 38 inj/s | 26 ms/inj — = HANDOFF's "27.5 ms batched numpy" |
| astro3 single CPU core (pinned, idle box) | 8 inj/s | 125 ms/inj — **~4.8× slower per core than M4** |
| astro3 core under 94-way load | 3.7 inj/s | 267 ms/inj — 2.1× memory-bandwidth slowdown |
| astro3 94-worker aggregate | **~345 inj/s** | 341 (real grid, 108,540 binaries/318 s) ≈ 352 (clean 94-parallel) |

- **1 M4 GPU ≈ 17 astro3 cores** (≈ 3.5 M4 CPU cores). The GPU accelerates the
  periodogram/permutation GEMM ~10–17×; host-bound value-distance/fit/FWHM cap it.
- astro3 94-worker **parallel efficiency ≈ 47%** of ideal 94× — the box is
  **memory-bandwidth-limited** (94 BLAS processes; per-core 125→267 ms/inj, and
  `load` stays ~33 with 94 processes running = they stall on memory, not CPU). So
  the "perfect-scaling" ceiling is not reachable; ~345 inj/s is the real number.
- **astro3-94 ≈ 2.5× the single GPU**, and the comparison *handicaps* astro3 (the
  GPU number excludes the harness overhead astro3 carries) — so 2.5× is conservative.

## final5_cvm_1e3 @ n_inject=100 (≈5.1e9 injections)

| | throughput | wall-time |
|---|---|---|
| Mac GPU (1 process) | 134 inj/s | ~440 days |
| astro3 94-core | ~345 inj/s | **~170 days** |

**Correcting HANDOFF §3's "~16 days (astro3)":** that assumed 27.5 ms/inj *per
astro3 core* with *perfect* 96× scaling (→ ~3,700 inj/s). But 27.5 ms/inj is the
**Mac M4** single-core rate (measured 26 ms here); a real astro3 core is 125 ms/inj
(idle) / 267 ms/inj (under 94-way load), and real scaling is 47%. Measured astro3 =
~345 inj/s → **~170 days, not 16.** Likewise HANDOFF's "MLX ~2–4 days" assumed
~15,000 inj/s (periodogram-GEMM-only); the true faithful-gate end-to-end MLX rate
is 134 inj/s (~440 days). Both are infeasible at n_inject=100 — the real lever is
**lower n_inject / coarser grid** (HANDOFF §4), not the hardware.

## What would flip the result to the GPU

1. Move `_value_distance_uc` (PDC), `batched_fit`, and the FWHM `_local_power` onto
   the backend so the GPU isn't host-bound at large N (HANDOFF "GPU efficiency" item).
2. Larger per-batch problem (more epochs, or pooling many fields) to better fill the GPU.
3. On-device running-argmax already done; the remaining transfer is the per-freq
   value-distance/`B_flat` upload (PDC) and all-host fit.

Until then, **for production bias grids, astro3's 94 cores are the faster path**;
the MLX backend is useful on a Mac-only machine (no astro3 access) where it is
~3.2× a single Mac numpy process.

## Reproduce

```bash
# Mac GPU:
conda run -n tau_binary python -m fast_detect.bench_detect --backend mlx \
  --mass ~/Documents/Data/BLOeM_Data/mass_bloem.csv --sizes 100,300,1000,2000,4000
# astro3 single core:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 micromamba run -n tau_binary python -m fast_detect.bench_detect \
  --backend numpy --mass ~/data/mass_bloem.csv --sizes 100,300,1000 --reps 2
# astro3 94-worker aggregate: pipeline_batched + fast_detect_backend:numpy + parallel_grid:true
```
