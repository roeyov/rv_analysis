"""
simulations.true_posterior.kq_null — how often does chance give a
difference between kappa below and above a period split?

Runs the two-kappa slice of ``two_kappa`` (pi and eta fixed, the fractions
and circular fractions free) on every mock survey of a campaign drawn from a
ONE-kappa population, for several split periods, and on the real catalogue
with the identical recipe. For each catalogue and split it records the
posterior of dkappa = kappa_short - kappa_long (median, 16-84%, P(dkappa <=
0)) and 2 dlnZ between the best node and the best node on the diagonal
kappa_short = kappa_long. The null distribution of these statistics, and of
their maximum over the splits (the freedom to choose the split), calibrates
the significance of the real value.

The mocks are the split-model mocks of ``make_mocks`` (synthetic_obs.csv
holds the true K1 of every detected binary on both sides of the cut), read
with ``mock_catalog.build`` at a fixed cut. The per-star weight sums are
computed once per split and shared by all catalogues.

    python -u -m simulations.true_posterior.kq_null --injset <set> \\
        --campaign-dir <mocks> --cut 0.5266248325033714 --obs <real catalogue> \\
        --pi -0.8035 --eta -0.2880 --splits cut,5,10,20 \\
        --kappa-short-grid=-1.20,1.00,45 --kappa-long-grid=-1.80,0.00,37 \\
        --split-f=0.30,0.95,0.30,0.01 --below-shape k1e --out-dir <dir> \\
        [--n-workers 40] [--max-mocks 200]

Output: <out-dir>/null_stats.csv (one row per catalogue and split; the real
catalogue has mock = "real") and null_summary.yaml.
"""

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

from simulations.true_posterior import mock_catalog
from simulations.true_posterior.injection_set import load_injection_set
from simulations.true_posterior.likelihood import InjectionSet, ObservedData
from simulations.true_posterior.run_posterior import (parse_grid,
                                                      split_f_lattice)
from simulations.true_posterior.two_kappa import (evaluate, range_tensors,
                                                  summarize2)

_G = {}


def _stats(table, name):
    """Rows (one per split) for one catalogue table."""
    g = _G
    inj = g["inj"]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs = ObservedData(table, inj.stars, g["cut"], inj.p_circ, g["p_env"],
                           inj.bounds["e_max"],
                           below_pcirc_ecc="as_circular",
                           below_shape=g["below_shape"])
        for tag, lq in g["splits"]:
            t0 = time.time()
            res = evaluate(inj, obs, g["pi"], g["ks"], g["kl"], g["eta"], lq,
                           n_workers=1, log=lambda *a: None,
                           f_lattice=g["lat"], tensors=g["T"][tag])
            summ, lnJ, _ = summarize2(res, g["pi"], g["ks"], g["kl"],
                                      g["eta"])
            t = summ["tests"]
            rows.append({
                "mock": name, "split": tag, "logp_q": lq,
                "n_dx": int(obs.dx_pos.size),
                "n_dx_short": int((obs.dx_logP < lq).sum()),
                "n_below": int(obs.below.dx_pos.size
                               if obs.below is not None else 0),
                "ks_p16": summ["kappa_short"]["p16"],
                "ks_p50": summ["kappa_short"]["p50"],
                "ks_p84": summ["kappa_short"]["p84"],
                "kl_p16": summ["kappa_long"]["p16"],
                "kl_p50": summ["kappa_long"]["p50"],
                "kl_p84": summ["kappa_long"]["p84"],
                "dk_p16": summ["dkappa"]["p16"],
                "dk_p50": summ["dkappa"]["p50"],
                "dk_p84": summ["dkappa"]["p84"],
                "dk_mean": summ["dkappa"]["mean"],
                "p_dk_le0": t["p_dkappa_le0"],
                "two_dlnZ": t.get("two_dlnZ_best_vs_diagonal", np.nan),
                "ks_edge": summ["kappa_short"]["edge_node_mass"],
                "kl_edge": summ["kappa_long"]["edge_node_mass"],
                "ft_edge": summ["f_tot"]["edge_node_mass"],
                "seconds": time.time() - t0})
    return rows


def _one(seed_dir):
    ids = [str(s) for s in _G["inj"].stars["ID"]]
    table = mock_catalog.build(seed_dir, ids, cut=_G["cut"])
    name = "%s/%s" % (os.path.basename(os.path.dirname(seed_dir)),
                      os.path.basename(seed_dir))
    return _stats(table, name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--injset", required=True)
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--obs", default=None, help="real obs_catalog.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cut", type=float, required=True)
    ap.add_argument("--pi", type=float, required=True)
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--splits", default="cut,5,10,20",
                    help="comma list of split periods in days ('cut' = the "
                         "period cut)")
    ap.add_argument("--kappa-short-grid", required=True)
    ap.add_argument("--kappa-long-grid", required=True)
    ap.add_argument("--split-f", default=None)
    ap.add_argument("--below-shape", choices=["k1", "k1e"], default="k1e")
    ap.add_argument("--p-env", type=float, default=5.0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--max-mocks", type=int, default=None)
    a = ap.parse_args(argv)

    t0 = time.time()
    data, stars, man = load_injection_set(a.injset)
    inj = InjectionSet(data, stars, man, p_env=a.p_env)
    del data
    pi = np.array([a.pi])
    eta = np.array([a.eta])
    ks, kl = parse_grid(a.kappa_short_grid), parse_grid(a.kappa_long_grid)
    splits = [(s, a.cut if s == "cut" else float(np.log10(float(s))))
              for s in a.splits.split(",")]
    T = {}
    for tag, lq in splits:
        T[tag] = range_tensors(inj, pi, ks, kl, eta, a.cut, lq)
        print("[null] tensors for split %s (logP_q = %.4f): %.0f s"
              % (tag, lq, time.time() - t0), flush=True)
    _G.update(inj=inj, cut=a.cut, p_env=a.p_env, pi=pi, eta=eta, ks=ks, kl=kl,
              splits=splits, T=T, lat=split_f_lattice(a.split_f),
              below_shape=a.below_shape)
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    if a.obs:
        rows += _stats(pd.read_csv(a.obs), "real")
        for r in rows:
            print("[null] real, split %-3s: kappa_short %+.3f [%+.3f, %+.3f] "
                  "kappa_long %+.3f [%+.3f, %+.3f] dkappa %+.3f "
                  "P(dk<=0) %.4f 2dlnZ %.2f (%.0f s)"
                  % (r["split"], r["ks_p50"], r["ks_p16"], r["ks_p84"],
                     r["kl_p50"], r["kl_p16"], r["kl_p84"], r["dk_p50"],
                     r["p_dk_le0"], r["two_dlnZ"], r["seconds"]), flush=True)
    seeds = sorted(glob.glob(os.path.join(a.campaign_dir, "*", "seed_*")))
    if a.max_mocks:
        seeds = seeds[: a.max_mocks]
    print("[null] %d mocks" % len(seeds), flush=True)
    done = 0
    if a.n_workers > 1 and sys.platform.startswith("linux"):
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.n_workers) as pool:
            for out in pool.imap_unordered(_one, seeds, chunksize=1):
                rows += out
                done += 1
                if done % 20 == 0:
                    print("[null] %d/%d mocks (%.0f s)"
                          % (done, len(seeds), time.time() - t0), flush=True)
                    pd.DataFrame(rows).to_csv(
                        os.path.join(a.out_dir, "null_stats.partial.csv"),
                        index=False)
    else:
        for sd in seeds:
            rows += _one(sd)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, "null_stats.csv"), index=False)
    summ = {"n_mocks": int(len(seeds)), "pi": a.pi, "eta": a.eta,
            "cut": a.cut, "splits": {s: float(lq) for s, lq in splits},
            "kappa_short_grid": a.kappa_short_grid,
            "kappa_long_grid": a.kappa_long_grid, "split_f": a.split_f,
            "below_shape": a.below_shape, "injset": os.path.abspath(a.injset),
            "campaign_dir": os.path.abspath(a.campaign_dir), "per_split": {}}
    mocks = df[df["mock"] != "real"]
    real = df[df["mock"] == "real"].set_index("split")
    for tag, _ in splits:
        m = mocks[mocks["split"] == tag]
        rec = {"null_dk_p50_mean": float(m["dk_p50"].mean()),
               "null_dk_p50_sd": float(m["dk_p50"].std(ddof=1)),
               "null_two_dlnZ_q": [float(v) for v in np.percentile(
                   m["two_dlnZ"].dropna(), [50, 84, 95, 99])],
               "null_edge_limited": int(((m["ks_edge"] > 1e-4)
                                         | (m["kl_edge"] > 1e-4)).sum())}
        if tag in real.index:
            r = real.loc[tag]
            rec.update(real_dk_p50=float(r["dk_p50"]),
                       real_two_dlnZ=float(r["two_dlnZ"]),
                       real_p_dk_le0=float(r["p_dk_le0"]),
                       frac_null_dk_ge_real=float(np.mean(
                           m["dk_p50"] >= r["dk_p50"])),
                       frac_null_absdk_ge_real=float(np.mean(
                           m["dk_p50"].abs() >= abs(r["dk_p50"]))),
                       frac_null_two_dlnZ_ge_real=float(np.mean(
                           m["two_dlnZ"] >= r["two_dlnZ"])))
        summ["per_split"][tag] = rec
    # the freedom to choose the split: maximum of 2 dlnZ over the splits
    mx = mocks.groupby("mock")["two_dlnZ"].max()
    summ["max_over_splits"] = {
        "null_q": [float(v) for v in np.percentile(mx.dropna(),
                                                   [50, 84, 95, 99])]}
    if len(real):
        rmax = float(real["two_dlnZ"].max())
        summ["max_over_splits"].update(
            real=rmax, frac_null_ge_real=float(np.mean(mx >= rmax)))
    with open(os.path.join(a.out_dir, "null_summary.yaml"), "w") as fh:
        yaml.safe_dump(summ, fh, sort_keys=False)
    print(yaml.safe_dump(summ, sort_keys=False))
    print("[null] wrote %s/null_stats.csv (%.0f s)" % (a.out_dir,
                                                       time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
