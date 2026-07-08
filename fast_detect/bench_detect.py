"""Portable detect_batch throughput benchmark: ms/injection vs batch size.

The SAME script answers the GPU-vs-astro3 question on both machines:
  - Mac (Apple GPU):  python -m fast_detect.bench_detect --backend mlx  --mass <mac mass csv>
  - astro3 (1 core):  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                      NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
                      python -m fast_detect.bench_detect --backend numpy --mass ~/data/mass_bloem.csv

A GPU processes a whole batch of B injections roughly in parallel, so its
ms/injection FALLS as B grows; a single pinned CPU core is ~flat. The question
"is 1 Mac GPU faster than astro3's N cores?" reduces to:
    GPU wins iff  t_cpu_1core(B) / t_gpu(B)  >  N_cores
(each of astro3's N cores runs an independent detect_batch, so aggregate CPU
throughput = N_cores * B / t_cpu_1core). Prints ms/inj at each B; feed the
single-core numbers + N_cores into the comparison.

No file writes (stdout only)."""
import argparse, time, os
import numpy as np
import pandas as pd
import yaml

from simulations.common import BLOEM_MJD_ARRAYS
from fast_detect.detector import detect_batch


def build_batch(mjds, N, rng, err=6.0, frac_bin=0.7):
    ne = mjds.size
    P = 10 ** rng.uniform(0.2, 1.8, N)
    K = rng.uniform(8, 45, N)
    ph = rng.uniform(0, 1, N)
    is_bin = rng.random(N) < frac_bin
    Y = rng.normal(0.0, err, (ne, N))
    Y += is_bin[None, :] * K[None, :] * np.sin(
        2 * np.pi * (mjds[:, None] / P[None, :] + ph[None, :]))
    return Y, np.full(N, err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="numpy", choices=["numpy", "mlx"])
    ap.add_argument("--mass", required=True, help="path to mass_bloem.csv")
    ap.add_argument("--config", default="configs/params_bias.yaml")
    ap.add_argument("--sizes", default="100,300,1000,2000",
                    help="comma-separated batch sizes (injections)")
    ap.add_argument("--nperm", type=int, default=10, help="permutation iters (= window_iterations)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--field", type=int, default=0, help="BLOEM_MJD_ARRAYS index")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["periodogram_params"]["max_period"] = 1000.0
    cfg["periodogram_params"]["window_iterations"] = args.nperm
    mjds = np.array(BLOEM_MJD_ARRAYS[args.field], float)
    mass_row = pd.read_csv(args.mass).iloc[10].to_dict()
    sizes = [int(s) for s in args.sizes.split(",")]

    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    print(f"backend={args.backend}  n_epoch={mjds.size}  n_perm={args.nperm}  "
          f"reps={args.reps}  OMP_NUM_THREADS={threads}")
    print(f"{'batch N':>8} {'total ms':>10} {'ms/inj':>9} {'inj/s':>10}")
    for N in sizes:
        rng = np.random.default_rng(0)
        Y, errs = build_batch(mjds, N, rng)
        mass_rows = [mass_row] * N
        # warm-up (build periodogram grids, Metal/JIT compile; forces host eval)
        detect_batch(mjds, Y, errs, mass_rows, cfg, backend=args.backend,
                     rng=np.random.default_rng(1))
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            det, _ = detect_batch(mjds, Y, errs, mass_rows, cfg,
                                  backend=args.backend, rng=np.random.default_rng(1))
            ts.append(time.perf_counter() - t0)
        t = min(ts)
        print(f"{N:8d} {t*1e3:10.1f} {t*1e3/N:9.3f} {N/t:10.0f}")


if __name__ == "__main__":
    main()
