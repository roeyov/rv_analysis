"""
simulations.true_posterior.boot_coverage — interval coverage of the
likelihood posterior from injection-set replicates at a chosen truth
(typically the real data's posterior medians).

No new detector runs are needed. Replicate catalogues are drawn from a
held-out block of the labelled injection set: per star, with probability
f_gt the star is a binary with logP >= logP*, resampled from the star's
held-out injections on that side with weights w_j(theta, c); with
probability f_lt one below logP*; otherwise a single, falsely detected
with probability beta_s. A drawn binary is detected iff its injection was.
Detected binaries above logP* are DX with the injection's (logP, e, K1)
(circular draws and P < p_circ are LS-circular with e_lim = 0); below
logP* they are DO (below_cut) and carry the same three columns, which the
likelihood uses only with ``--below-shape``; false positives are DO
(false_positive). The likelihood is then evaluated with a disjoint block of
the same set, so the replicate's detection outcomes and the inference's
alpha_s share no Monte-Carlo draws. logP* is fixed at the truth's cut for
every replicate.

``--below-shape k1|k1e`` evaluates the variant in which the below-cut
orbits contribute their amplitude (and eccentricity) density
(``likelihood`` docstring); the truth should then be that variant's fit.

Output layout matches a Tier-4 campaign, so ``closure_coverage.aggregate``
reads it: <out-dir>/boot/seed_<r>/grid_run/{mock_catalog.csv,
grid_cubes.npz, closure_report/coverage_row.csv} and
<out-dir>/boot/seed_<r>/truth_params.yaml. Rows cover pi, kappa, eta,
f_bin and, for the split model, f_gt, f_lt, log10_rho, c1, c2.

    python -u -m simulations.true_posterior.boot_coverage \\
        --injset <pipeline_batched set> --truth-from <real split run dir> \\
        --out-dir <dir> --n-rep 500 --model split \\
        [--gen-subset 1,2 --inf-subset 0,2] [--kappa-grid -1.30,1.80,63] \\
        [--below-shape k1e] [--method linear] [--n-workers 16]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

from simulations.closure_coverage import aggregate, coverage_rows_for_run
from simulations.true_posterior.injection_set import load_injection_set
from simulations.true_posterior.likelihood import InjectionSet
from simulations.true_posterior.population import powerlaw_cdf
from simulations.true_posterior.run_posterior import (
    TAG, TAG_SPLIT, METRIC, make_grids, parse_c_nodes, run, shared_tensors)

TRUTH_KEYS = ("pi", "kappa", "eta", "f_gt", "f_lt", "c1", "c2", "cut")
_G = {}


def truth_from_run(run_dir):
    """Truth = posterior medians of a finished split run + its cut."""
    with open(os.path.join(run_dir, "posterior_summary.yaml")) as fh:
        summ = yaml.safe_load(fh)
    obs = pd.read_csv(os.path.join(run_dir, "obs_catalog.csv"))
    t = {k: float(summ[k]["p50"]) for k in ("pi", "kappa", "eta", "f_gt",
                                             "f_lt", "c1", "c2")}
    t["cut"] = float(obs["logP_cut"].iloc[0])
    return t


def truth_record(t, bounds):
    """Full truth block (with derived f_bin and log10 rho) for the yaml."""
    F_gt = 1.0 - float(powerlaw_cdf(t["cut"], t["pi"], bounds["log_p_min"],
                                    bounds["log_p_max"]))
    rho = (t["f_lt"] / (1.0 - F_gt)) / (t["f_gt"] / F_gt)
    return {"pi": t["pi"], "kappa": t["kappa"], "eta": t["eta"],
            "fbin": t["f_gt"] + t["f_lt"], "f_gt": t["f_gt"],
            "f_lt": t["f_lt"], "log10_rho": float(np.log10(rho)),
            "c1": t["c1"], "c2": t["c2"], "cut": t["cut"], "F_gt": F_gt}


class ReplicateSampler:
    """Draw replicate catalogues from a held-out injection set at a truth."""

    def __init__(self, H, truth):
        self.H = H
        self.t = truth
        w = H.weights(truth["pi"], truth["kappa"], truth["eta"],
                      truth["c1"], truth["c2"])
        above = H.logP >= truth["cut"]
        self.cum = {}
        for side, mask in (("above", above), ("below", ~above)):
            per_star = []
            for s in range(H.n_stars):
                ws = (w * mask)[H.seg[s]:H.seg_end[s]]
                tot = ws.sum()
                if not tot > 0:
                    raise ValueError("star %d has no held-out injections %s "
                                     "the cut" % (s, side))
                per_star.append(np.cumsum(ws) / tot)
            self.cum[side] = per_star
        self.e_eff = np.where(H.short | H.circ_comp, 0.0, H.e)
        self.ids = [str(x) for x in H.stars["ID"]]

    def draw(self, rng):
        """One replicate: DataFrame in the obs_catalog schema + summary."""
        H, t = self.H, self.t
        rows, fp_ids = [], []
        n_dx = n_below = n_bin = 0
        for s in range(H.n_stars):
            rec = {"star_id": self.ids[s], "outcome": "ND", "reason": "",
                   "logP": np.nan, "e": np.nan, "e_is_upper_limit": False,
                   "K1": np.nan}
            u = rng.random()
            if u < t["f_gt"] + t["f_lt"]:
                n_bin += 1
                side = "above" if u < t["f_gt"] else "below"
                j = H.seg[s] + int(np.searchsorted(self.cum[side][s],
                                                   rng.random()))
                j = min(j, H.seg_end[s] - 1)
                if H.det[j]:
                    e = float(self.e_eff[j])
                    rec.update(logP=float(H.logP[j]), e=e,
                               e_is_upper_limit=bool(e < 1e-6),
                               K1=float(H.K1[j]))
                    if side == "above":
                        rec.update(outcome="DX")
                        n_dx += 1
                    else:
                        rec.update(outcome="DO", reason="below_cut")
                        n_below += 1
            elif rng.random() < H.beta[s]:
                rec.update(outcome="DO", reason="false_positive")
                fp_ids.append(self.ids[s])
            rows.append(rec)
        df = pd.DataFrame(rows)
        df["logP_cut"] = float(t["cut"])
        summ = {"n_stars": int(H.n_stars), "n_physical": n_bin,
                "n_detected": n_dx + n_below + len(fp_ids),
                "n_dx": n_dx, "n_below_cut": n_below,
                "n_false_positive": len(fp_ids),
                "false_positive_ids": fp_ids}
        return df, summ


def _one(r):
    out_root, opts = _G["out"], _G["opts"]
    seed_dir = os.path.join(out_root, "boot", "seed_%d" % r)
    grid_dir = os.path.join(seed_dir, "grid_run")
    row_path = os.path.join(grid_dir, "closure_report", "coverage_row.csv")
    if os.path.exists(row_path):
        return r, "skip", 0.0
    t0 = time.time()
    obs_csv = os.path.join(grid_dir, "mock_catalog.csv")
    run(None, obs_csv, grid_dir, inj_cache=_G["inj"], log=lambda *a: None,
        base_tensors=_G["base"], **opts)
    rows = coverage_rows_for_run(grid_dir,
                                 os.path.join(seed_dir, "truth_params.yaml"),
                                 "boot", r,
                                 variant=(TAG_SPLIT if opts["model"] == "split"
                                          else TAG),
                                 test=METRIC, method=_G["method"])
    os.makedirs(os.path.dirname(row_path), exist_ok=True)
    rows.to_csv(row_path, index=False)
    return r, "done", time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--injset", required=True)
    ap.add_argument("--out-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--truth-from", default=None,
                   help="finished split run dir: truth = its medians + cut")
    g.add_argument("--truth-yaml", default=None,
                   help="yaml with pi, kappa, eta, f_gt, f_lt, c1, c2, cut")
    ap.add_argument("--n-rep", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--gen-subset", default="1,2",
                    help="k,n block of the set used to draw replicates")
    ap.add_argument("--inf-subset", default="0,2",
                    help="k,n block of the set used for inference")
    ap.add_argument("--max-per-star", type=int, default=None,
                    help="cap on the inference block per star")
    ap.add_argument("--model", choices=["mix", "split"], default="split")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--f-grid", default=None)
    ap.add_argument("--pi-grid", default=None)
    ap.add_argument("--kappa-grid", default=None)
    ap.add_argument("--eta-grid", default=None)
    ap.add_argument("--c-grid", type=int, default=None)
    ap.add_argument("--c-nodes", default=None, metavar="c0,c1,...",
                    help="as run_posterior --c-nodes")
    ap.add_argument("--below-shape", choices=["k1", "k1e"], default=None,
                    help="split model: the below-cut orbits contribute their "
                         "amplitude (k1) or amplitude and eccentricity (k1e) "
                         "density")
    ap.add_argument("--method", default=None, choices=["cell", "linear"])
    ap.add_argument("--n-workers", type=int, default=1)
    a = ap.parse_args(argv)
    if a.below_shape and a.model != "split":
        raise SystemExit("--below-shape needs --model split")
    if a.c_nodes is not None and a.c_grid is not None:
        raise SystemExit("--c-nodes excludes --c-grid")
    c_nodes = parse_c_nodes(a.c_nodes)

    gen_sub = tuple(int(v) for v in a.gen_subset.split(","))
    inf_sub = tuple(int(v) for v in a.inf_subset.split(","))
    if gen_sub == inf_sub:
        raise SystemExit("--gen-subset and --inf-subset must differ")
    t0 = time.time()
    data, stars, man = load_injection_set(a.injset)
    H = InjectionSet(data, stars, man, subset=gen_sub)
    inj = InjectionSet(data, stars, man, subset=inf_sub,
                       max_per_star=a.max_per_star)
    del data
    print("[boot] injection set: %s; replicates from block %s (%d/star), "
          "inference from block %s (%d/star); %.0f s"
          % (a.injset, gen_sub, int(H.N.min()), inf_sub, int(inj.N.min()),
             time.time() - t0), flush=True)

    truth = (truth_from_run(a.truth_from) if a.truth_from
             else {k: float(v) for k, v in
                   yaml.safe_load(open(a.truth_yaml)).items()})
    missing = [k for k in TRUTH_KEYS if k not in truth]
    if missing:
        raise SystemExit("truth lacks %s" % missing)
    trec = truth_record(truth, inj.bounds)
    print("[boot] truth %s" % {k: round(v, 4) for k, v in trec.items()},
          flush=True)

    # replicate catalogues (cheap; drawn up front, deterministically)
    os.makedirs(a.out_dir, exist_ok=True)
    sampler = ReplicateSampler(H, truth)
    stats = []
    for r in range(a.n_rep):
        seed_dir = os.path.join(a.out_dir, "boot", "seed_%d" % r)
        grid_dir = os.path.join(seed_dir, "grid_run")
        os.makedirs(grid_dir, exist_ok=True)
        cat_path = os.path.join(grid_dir, "mock_catalog.csv")
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, r]))
        df, summ = sampler.draw(rng)
        if not os.path.exists(cat_path):
            df.to_csv(cat_path, index=False)
            with open(os.path.join(seed_dir, "truth_params.yaml"), "w") as fh:
                yaml.safe_dump({"truth": trec, "seed": int(a.seed),
                                "replicate": r, "summary": summ,
                                "gen_subset": list(gen_sub),
                                "inf_subset": list(inf_sub),
                                "injset": os.path.abspath(a.injset),
                                "p_circ": float(inj.p_circ)},
                               fh, sort_keys=False)
        stats.append(summ)
    st = pd.DataFrame(stats)
    print("[boot] %d replicates: N_D %.1f +- %.1f, DX %.1f, below-cut %.1f, "
          "FP %.2f (means)"
          % (a.n_rep, st["n_detected"].mean(), st["n_detected"].std(),
             st["n_dx"].mean(), st["n_below_cut"].mean(),
             st["n_false_positive"].mean()), flush=True)
    del sampler, H

    grids = make_grids(a.preset, a.pi_grid, a.kappa_grid, a.eta_grid,
                       a.f_grid)
    t0 = time.time()
    base = shared_tensors(inj, grids, cut=truth["cut"])
    print("[boot] shared tensors (fixed cut %.4f): %.0f s"
          % (truth["cut"], time.time() - t0), flush=True)
    _G.update(inj=inj, out=a.out_dir, base=base, method=a.method,
              opts={"preset": a.preset, "f_grid": a.f_grid, "c_n": a.c_grid,
                    "pi_grid": a.pi_grid, "kappa_grid": a.kappa_grid,
                    "eta_grid": a.eta_grid, "model": a.model,
                    "below_shape": a.below_shape, "c_grid": c_nodes})
    with open(os.path.join(a.out_dir, "boot_config.yaml"), "w") as fh:
        yaml.safe_dump({"truth": trec, "n_rep": a.n_rep, "seed": a.seed,
                        "gen_subset": list(gen_sub),
                        "inf_subset": list(inf_sub),
                        "max_per_star": a.max_per_star, "model": a.model,
                        "below_shape": a.below_shape,
                        "c_nodes": (None if c_nodes is None
                                    else list(map(float, c_nodes))),
                        "grids": {"pi": a.pi_grid, "kappa": a.kappa_grid,
                                  "eta": a.eta_grid, "f": a.f_grid,
                                  "preset": a.preset},
                        "method": a.method, "injset": os.path.abspath(a.injset),
                        "truth_from": a.truth_from, "truth_yaml": a.truth_yaml},
                       fh, sort_keys=False)
    reps = list(range(a.n_rep))
    t0 = time.time()
    if a.n_workers > 1 and sys.platform.startswith("linux"):
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_one, reps)):
                if res[1] == "done" and (i % 10 == 0 or i == len(reps) - 1):
                    print("[boot] %d/%d rep %d %.0f s (elapsed %.0f s)"
                          % (i + 1, len(reps), res[0], res[2],
                             time.time() - t0), flush=True)
    else:
        for i, r in enumerate(reps):
            res = _one(r)
            print("[boot] %d/%d rep %d %s %.0f s"
                  % (i + 1, len(reps), r, res[1], res[2]), flush=True)
    aggregate(a.out_dir, out_csv=os.path.join(a.out_dir, "coverage_table.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
