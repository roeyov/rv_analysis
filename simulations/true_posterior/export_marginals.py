"""
simulations.true_posterior.export_marginals — write the paper-side
``tables/bias_posterior_marginals_<fit>.npz`` of a finished posterior run.

The paper scripts (``Ostars_article/scripts/make_param_histograms_models.py``,
``make_q_distribution_models.py``, ...) read, per axis pi/kappa/eta/f_bin,
the grid, the 1-D marginal normalised to a peak of 1 and the cell-centred
16/50/84 per cent quantiles (``bias_grid_lib.marginals.grid_quantiles``),
plus, for the split model, ``logP_cut`` and the f_gt, f_lt, log10_rho, c1
and c2 quantiles of ``posterior_summary.yaml``. When the run directory has a
``run_posterior.yaml``, its ``q_min`` and ``k1_floor`` are stored too (NaN
when the run did not set them), so a reader can check which fit it holds.

    python -m simulations.true_posterior.export_marginals \
        --run <dir with grid_cubes.npz + posterior_summary.yaml> \
        --out <npz> [--test lik|cvm] [--tag <variant>] [--logp-cut 0.5266]

``--logp-cut`` is needed only when the run directory has no
``obs_catalog.csv`` (its ``logP_cut`` column is used otherwise).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

from simulations.bias_grid_lib.cube_io import read_gmf_cube
from simulations.bias_grid_lib.marginals import grid_quantiles

ESTIMATOR = "cell-centred (bias_grid_lib.marginals.grid_quantiles)"
AXES = (("pi", "pi_grid"), ("kappa", "kappa_grid"), ("eta", "eta_grid"),
        ("f_bin", "fbin_grid"))
SPLIT_EXTRA = ("f_gt", "f_lt", "log10_rho", "c1", "c2")


def export(run_dir, out, test="lik", tag=None, logp_cut=None):
    run_dir = os.path.abspath(run_dir)
    z = np.load(os.path.join(run_dir, "grid_cubes.npz"), allow_pickle=True)
    cube, tag = read_gmf_cube(z, test=test, tag=tag)
    prob = np.nan_to_num(np.exp(np.asarray(cube, float) - np.nanmax(cube)))
    rec = {"variant": tag, "metric": test, "source": run_dir,
           "estimator": ESTIMATOR}
    for ax, (name, key) in enumerate(AXES):
        grid = np.asarray(z[key], float)
        post = prob.sum(axis=tuple(a for a in range(4) if a != ax))
        q16, q50, q84 = grid_quantiles(post, grid, (0.16, 0.5, 0.84))
        rec[name + "_grid"] = grid
        rec[name + "_post"] = post / post.max()
        rec[name + "_p16"], rec[name + "_p50"], rec[name + "_p84"] = q16, q50, q84
    summ_path = os.path.join(run_dir, "posterior_summary.yaml")
    summ = yaml.safe_load(open(summ_path)) if os.path.exists(summ_path) else {}
    if "split" in str(tag):
        obs = os.path.join(run_dir, "obs_catalog.csv")
        if logp_cut is None and os.path.exists(obs):
            logp_cut = float(pd.read_csv(obs)["logP_cut"].iloc[0])
        if logp_cut is None:
            raise SystemExit("split run without obs_catalog.csv: pass --logp-cut")
        rec["logP_cut"] = float(logp_cut)
        for name in SPLIT_EXTRA:
            if name not in summ:
                raise SystemExit("posterior_summary.yaml lacks %s" % name)
            for q in ("p16", "p50", "p84"):
                rec["%s_%s" % (name, q)] = float(summ[name][q])
    meta_path = os.path.join(run_dir, "run_posterior.yaml")
    if os.path.exists(meta_path):
        meta = yaml.safe_load(open(meta_path)) or {}
        for key in ("q_min", "k1_floor"):
            v = meta.get(key)
            rec[key] = float("nan") if v is None else float(v)
    # cross-check the quantiles against the run's own summary
    for name, key in AXES:
        s = summ.get("f_tot" if name == "f_bin" else name)
        if s is None:
            s = summ.get("fbin") if name == "f_bin" else None
        if s is not None:
            for q in ("p16", "p50", "p84"):
                if abs(float(s[q]) - rec["%s_%s" % (name, q)]) > 1e-9:
                    raise SystemExit("quantile mismatch for %s %s: %r vs %r"
                                     % (name, q, s[q], rec["%s_%s" % (name, q)]))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    np.savez(out, **rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", default="lik", choices=["lik", "cvm"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--logp-cut", type=float, default=None)
    a = ap.parse_args(argv)
    rec = export(a.run, a.out, test=a.test, tag=a.tag, logp_cut=a.logp_cut)
    print("wrote %s  (%s, %s)" % (a.out, rec["variant"], rec["metric"]))
    for name, _ in AXES:
        print("  %-6s p50 %+.4f [%+.4f, %+.4f]" % (
            name, rec[name + "_p50"], rec[name + "_p16"], rec[name + "_p84"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
