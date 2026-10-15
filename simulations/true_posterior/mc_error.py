"""
simulations.true_posterior.mc_error — Monte-Carlo error of the posterior
quantiles from the finite injection set.

The injection set is split into n disjoint per-star blocks
(``InjectionSet(subset=(k, n))``); the posterior is evaluated on each
block for the same observed table. The block-to-block scatter of every
reported quantile gives the MC error of a single block, and sd / sqrt(n)
the MC error of the full-set result (the blocks are independent iid
subsamples). Each quantile's MC error is also reported relative to the
68% half-width of the full-set posterior when ``--full-dir`` is given.

    python -u -m simulations.true_posterior.mc_error \\
        --injset <pipeline set> --obs <obs_catalog.csv> --out-dir <dir> \\
        --n-subsets 5 --model split [--full-dir <headline run dir>] \\
        [--kappa-grid ...] [--split-f fg_lo,fg_hi,fl_hi,d] [--n-workers 16]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

from simulations.true_posterior.injection_set import load_injection_set
from simulations.true_posterior.likelihood import InjectionSet
from simulations.true_posterior.run_posterior import parse_c_nodes, run

AXES_MIX = ("pi", "kappa", "eta", "fbin", "c1", "c2")
AXES_SPLIT = ("pi", "kappa", "eta", "f_tot", "f_gt", "f_lt", "log10_rho",
              "c1", "c2")
QS = ("p16", "p50", "p84")


def summarize_blocks(summaries, axes, full=None):
    rows = []
    n = len(summaries)
    for ax in axes:
        for q in QS:
            v = np.array([s[ax][q] for s in summaries], float)
            rec = {"axis": ax, "quantile": q, "n_blocks": n,
                   "block_mean": float(v.mean()),
                   "block_sd": float(v.std(ddof=1)) if n > 1 else np.nan,
                   "mc_err_full": (float(v.std(ddof=1) / np.sqrt(n))
                                   if n > 1 else np.nan)}
            if full is not None and ax in full:
                hw = 0.5 * (full[ax]["p84"] - full[ax]["p16"])
                rec["full_value"] = float(full[ax][q])
                rec["hw68_full"] = float(hw)
                rec["mc_err_over_hw"] = (rec["mc_err_full"] / hw if hw > 0
                                         else np.nan)
            rows.append(rec)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--injset", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-subsets", type=int, default=5)
    ap.add_argument("--model", choices=["mix", "split"], default="split")
    ap.add_argument("--full-dir", default=None,
                    help="finished full-set run (posterior_summary.yaml) "
                         "for the half-width normalization")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--f-grid", default=None)
    ap.add_argument("--pi-grid", default=None)
    ap.add_argument("--kappa-grid", default=None)
    ap.add_argument("--eta-grid", default=None)
    ap.add_argument("--q-min", type=float, default=None)
    ap.add_argument("--split-f", default=None, metavar="fg_lo,fg_hi,fl_hi,d",
                    help="split-model f lattice of the full-set run (see "
                         "run_posterior --split-f)")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--below-shape", choices=["k1", "k1e"], default=None,
                    help="as run_posterior --below-shape (must equal the "
                         "full-set run's setting)")
    ap.add_argument("--below-pcirc-as-circular", action="store_true",
                    help="as run_posterior --below-pcirc-as-circular")
    ap.add_argument("--c-nodes", default=None, metavar="c0,c1,...",
                    help="as run_posterior --c-nodes (must equal the "
                         "full-set run's lattice)")
    a = ap.parse_args(argv)
    c_nodes = parse_c_nodes(a.c_nodes)

    t0 = time.time()
    data, stars, man = load_injection_set(a.injset)
    print("[mc] loaded %s (%d records) in %.0f s"
          % (a.injset, data["star"].size, time.time() - t0), flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    summaries = []
    for k in range(a.n_subsets):
        inj = InjectionSet(data, stars, man, subset=(k, a.n_subsets))
        bdir = os.path.join(a.out_dir, "block_%d" % k)
        t1 = time.time()
        summ = run(None, a.obs, bdir, inj_cache=inj, model=a.model,
                   preset=a.preset, f_grid=a.f_grid, pi_grid=a.pi_grid,
                   kappa_grid=a.kappa_grid, eta_grid=a.eta_grid,
                   q_min=a.q_min, split_f=a.split_f,
                   n_workers=a.n_workers, log=lambda *x: None,
                   below_shape=a.below_shape,
                   below_pcirc_ecc=("as_circular"
                                    if a.below_pcirc_as_circular
                                    else "raise"),
                   c_grid=c_nodes)
        summaries.append(summ)
        axes = AXES_SPLIT if a.model == "split" else AXES_MIX
        print("[mc] block %d/%d (%d inj/star) %.0f s: %s"
              % (k + 1, a.n_subsets, int(inj.N.min()), time.time() - t1,
                 "  ".join("%s %+.3f" % (ax, summ[ax]["p50"])
                           for ax in axes)), flush=True)
        del inj
    full = None
    if a.full_dir:
        with open(os.path.join(a.full_dir, "posterior_summary.yaml")) as fh:
            full = yaml.safe_load(fh)
    table = summarize_blocks(summaries, axes, full)
    table.to_csv(os.path.join(a.out_dir, "mc_error.csv"), index=False)
    with open(os.path.join(a.out_dir, "mc_error_config.yaml"), "w") as fh:
        yaml.safe_dump({"injset": os.path.abspath(a.injset),
                        "obs": os.path.abspath(a.obs),
                        "n_subsets": a.n_subsets, "model": a.model,
                        "full_dir": a.full_dir, "grids": {
                            "pi": a.pi_grid, "kappa": a.kappa_grid,
                            "eta": a.eta_grid, "f": a.f_grid,
                            "preset": a.preset}, "q_min": a.q_min,
                        "split_f": a.split_f,
                        "below_shape": a.below_shape,
                        "below_pcirc_as_circular":
                            bool(a.below_pcirc_as_circular),
                        "c_nodes": (None if c_nodes is None
                                    else list(map(float, c_nodes)))},
                       fh, sort_keys=False)
    with pd.option_context("display.width", 140):
        print(table.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("[mc] wrote %s/mc_error.csv (%.0f s)" % (a.out_dir,
                                                   time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
