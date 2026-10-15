"""
simulations.true_posterior.coverage — interval coverage of the true
(likelihood) posterior on a Tier-4 mock campaign.

For every <campaign>/<truth_tag>/seed_<n>/ mock (written by
bias_grid_closure_generate), the driver:
  1. builds the per-star table (``mock_catalog.build``), using logP* from
     the mock's own tables exactly as the CvM grid did;
  2. evaluates the true posterior against ONE shared injection set for
     the mock's detector (no per-mock grid simulation);
  3. writes the coverage row with the shared cell-centred estimator
     (``closure_coverage.coverage_rows_for_run``, test ``lik``);
then aggregates everything into coverage_table.csv.

The injection set must use the same detector and sampling bounds as the
mocks. The driver checks detect_method and the bounds against
mock_meta.yaml.

    python -u -m simulations.true_posterior.coverage \
        --injset <injset dir> --campaign-dir <tier4 dir> --out-dir <dir> \
        [--fp-source <dir of regenerated mocks with false_positive_ids>] \
        [--model split] [--cut 0.5266] [--no-count] [--c-zero] \
        [--below-shape k1e] [--method linear] [--n-workers 16]

``--cut`` fixes logP* for every mock (instead of each mock's own elbow);
``--no-count`` drops the count terms (mix model); ``--c-zero`` fixes
c1 = c2 = 0; ``--method`` picks the quantile estimator of the rows.
``--below-shape k1|k1e`` (split model) evaluates the variant in which the
below-cut orbits contribute their amplitude (and eccentricity) density
(``likelihood`` docstring); the mocks carry the true K1 and e of every
detected binary on both sides of the cut.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

from simulations.closure_coverage import aggregate, coverage_rows_for_run
from simulations.true_posterior import mock_catalog
from simulations.true_posterior.injection_set import load_injection_set
from simulations.true_posterior.likelihood import InjectionSet
from simulations.true_posterior.run_posterior import (
    TAG, TAG_SPLIT, METRIC, make_grids, parse_c_nodes, run, shared_tensors)

_G = {}


def _seed_dirs(campaign_dir):
    return sorted(glob.glob(os.path.join(campaign_dir, "*", "seed_*")))


def _check_compat(seed_dir, inj):
    with open(os.path.join(seed_dir, "mock_meta.yaml")) as fh:
        meta = yaml.safe_load(fh)
    man = inj.manifest
    if meta.get("detect_method") != man.get("detect_method"):
        raise ValueError("%s: mock detector %r != injection set %r"
                         % (seed_dir, meta.get("detect_method"),
                            man.get("detect_method")))
    sb = meta.get("sampling_bounds") or {}
    for k, v in man["bounds"].items():
        if k in sb and abs(float(sb[k]) - float(v)) > 1e-12:
            raise ValueError("%s: sampling bound %s mock %s != injset %s"
                             % (seed_dir, k, sb[k], v))


def _one(seed_dir):
    inj, out_root, fp_src, opts = (_G["inj"], _G["out"], _G["fp_src"],
                                   _G["opts"])
    tag = os.path.basename(os.path.dirname(seed_dir))
    seed = int(os.path.basename(seed_dir).split("_")[1])
    grid_dir = os.path.join(out_root, tag, "seed_%d" % seed, "grid_run")
    row_path = os.path.join(grid_dir, "closure_report", "coverage_row.csv")
    if os.path.exists(row_path):
        return tag, seed, "skip", 0.0
    # atomic claim, so several driver instances (e.g. one in --reverse
    # order) can share one out-dir without computing the same mock twice
    os.makedirs(grid_dir, exist_ok=True)
    try:
        fd = os.open(os.path.join(grid_dir, ".claim"),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return tag, seed, "claimed", 0.0
    t0 = time.time()
    fp_ids = None
    if fp_src:
        with open(os.path.join(fp_src, tag, "seed_%d" % seed,
                               "truth_params.yaml")) as fh:
            fp_ids = (yaml.safe_load(fh).get("summary") or {}).get(
                "false_positive_ids")
    ids = [str(s) for s in inj.stars["ID"]]
    table = mock_catalog.build(seed_dir, ids, fp_ids=fp_ids, cut=_G["cut"])
    os.makedirs(grid_dir, exist_ok=True)
    obs_csv = os.path.join(grid_dir, "mock_catalog.csv")
    table.to_csv(obs_csv, index=False)
    run(None, obs_csv, grid_dir, inj_cache=inj, log=lambda *a: None,
        base_tensors=_G["base"], **opts)
    rows = coverage_rows_for_run(grid_dir,
                                 os.path.join(seed_dir, "truth_params.yaml"),
                                 tag, seed,
                                 variant=(TAG_SPLIT if opts.get("model")
                                          == "split" else TAG),
                                 test=METRIC, method=_G["method"])
    os.makedirs(os.path.dirname(row_path), exist_ok=True)
    rows.to_csv(row_path, index=False)
    return tag, seed, "done", time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--injset", required=True)
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fp-source", default=None,
                    help="campaign-shaped dir whose truth_params.yaml carry "
                         "false_positive_ids (regenerated mocks)")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--f-grid", default=None)
    ap.add_argument("--c-grid", type=int, default=None)
    ap.add_argument("--c-nodes", default=None, metavar="c0,c1,...",
                    help="as run_posterior --c-nodes")
    ap.add_argument("--max-per-star", type=int, default=None,
                    help="subsample the injection set per star")
    ap.add_argument("--model", choices=["mix", "split"], default="mix")
    ap.add_argument("--pi-grid", default=None, help="lo,hi,n override")
    ap.add_argument("--kappa-grid", default=None, help="lo,hi,n override")
    ap.add_argument("--eta-grid", default=None, help="lo,hi,n override")
    ap.add_argument("--subset", default=None,
                    help="k,n: block k of n disjoint per-star blocks")
    ap.add_argument("--cut", type=float, default=None,
                    help="fixed logP* for every mock (default: each mock's "
                         "own numerical elbow)")
    ap.add_argument("--no-count", action="store_true",
                    help="shape-only likelihood (mix model)")
    ap.add_argument("--c-zero", action="store_true",
                    help="fix c1 = c2 = 0")
    ap.add_argument("--below-shape", choices=["k1", "k1e"], default=None,
                    help="split model: the below-cut orbits contribute their "
                         "amplitude (k1) or amplitude and eccentricity (k1e) "
                         "density")
    ap.add_argument("--method", default=None, choices=["cell", "linear"],
                    help="quantile estimator for the coverage rows")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--reverse", action="store_true",
                    help="process the mocks in reverse order (a second "
                         "instance sharing the out-dir)")
    a = ap.parse_args(argv)
    if a.below_shape and a.model != "split":
        raise SystemExit("--below-shape needs --model split")
    if a.c_nodes is not None and (a.c_grid is not None or a.c_zero):
        raise SystemExit("--c-nodes excludes --c-grid and --c-zero")
    c_nodes = parse_c_nodes(a.c_nodes)

    subset = tuple(int(v) for v in a.subset.split(",")) if a.subset else None
    data, stars, man = load_injection_set(a.injset)
    inj = InjectionSet(data, stars, man, max_per_star=a.max_per_star,
                       subset=subset)
    del data
    seeds = _seed_dirs(a.campaign_dir)
    if not seeds:
        raise SystemExit("no seed dirs under %s" % a.campaign_dir)
    if a.reverse:
        seeds = seeds[::-1]
    for s in seeds:
        _check_compat(s, inj)
    grids = make_grids(a.preset, a.pi_grid, a.kappa_grid, a.eta_grid,
                       a.f_grid)
    t0 = time.time()
    base = shared_tensors(inj, grids, cut=a.cut)
    print("[tp-cov] shared tensors: %.0f s%s"
          % (time.time() - t0,
             "  (fixed cut %.4f)" % a.cut if a.cut is not None else ""),
          flush=True)
    _G.update(inj=inj, out=a.out_dir, fp_src=a.fp_source, base=base,
              cut=a.cut, method=a.method,
              opts={"preset": a.preset, "f_grid": a.f_grid, "c_n": a.c_grid,
                    "pi_grid": a.pi_grid, "kappa_grid": a.kappa_grid,
                    "eta_grid": a.eta_grid, "model": a.model,
                    "count": not a.no_count,
                    "c_grid": (np.array([0.0]) if a.c_zero else c_nodes),
                    "below_shape": a.below_shape})
    print("[tp-cov] %d mocks, injection set %s (%s, %d injections)"
          % (len(seeds), a.injset, man.get("detect_method"), inj.logP.size),
          flush=True)
    t0 = time.time()
    if a.n_workers > 1 and sys.platform.startswith("linux"):
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_one, seeds)):
                print("[tp-cov] %d/%d %s/seed_%d %s %.0f s  (elapsed %.0f s)"
                      % ((i + 1, len(seeds)) + res[:3] + (res[3],
                                                          time.time() - t0)),
                      flush=True)
    else:
        for i, s in enumerate(seeds):
            res = _one(s)
            print("[tp-cov] %d/%d %s/seed_%d %s %.0f s"
                  % ((i + 1, len(seeds)) + res), flush=True)
    aggregate(a.out_dir, out_csv=os.path.join(a.out_dir, "coverage_table.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
