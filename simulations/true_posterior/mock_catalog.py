"""
simulations.true_posterior.mock_catalog — per-star observed table for a
mock survey, in the obs_catalog schema, so ``run_posterior`` treats mocks
and real data identically.

Input is a mock seed directory written by bias_grid_closure_generate:
``synthetic_obs.csv`` (one row per detected binary, with star_id), the
mock's ``sb1_solutions.tex`` / ``sb2_solutions.tex`` (used only to compute
logP* exactly as the CvM grid did for this mock), and ``truth_params.yaml``
(summary, including ``false_positive_ids`` for mocks generated since
2026-09-29).

Outcomes: stars in synthetic_obs.csv are D, DX if logP >= logP* else DO
(below_cut). False-positive stars are D/DO (false_positive). All other
stars are ND. A mock orbit is LS-circular when e < 1e-6 (e is forced to 0
below p_circ), with e_lim = 0. This matches the generator's
``\\leq 0`` flag.

Older mocks carry only a false-positive count. Their false-positive stars
cannot be placed and are left ND; ``build`` warns, and the effect is at
most one or two stars' count terms.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

from simulations.bias_grid_lib.cutoffs import _compute_logP_cutoff
from simulations.bias_grid_lib.obs_io import load_observed_from_tex


def mock_cut(seed_dir, smooth_sigma=0.15):
    obs = load_observed_from_tex(os.path.join(seed_dir, "sb1_solutions.tex"),
                                 os.path.join(seed_dir, "sb2_solutions.tex"))
    return _compute_logP_cutoff(obs["logP"], "numerical",
                                smooth_sigma=smooth_sigma)


def build(seed_dir, star_ids, cut=None, smooth_sigma=0.15, fp_ids=None):
    """DataFrame in the obs_catalog schema for one mock.

    ``fp_ids`` overrides the truth yaml's false-positive star IDs (e.g. taken
    from a deterministic regeneration of an older mock).
    """
    syn = pd.read_csv(os.path.join(seed_dir, "synthetic_obs.csv"))
    with open(os.path.join(seed_dir, "truth_params.yaml")) as fh:
        meta = yaml.safe_load(fh)
    summ = meta.get("summary") or {}
    if fp_ids is None:
        fp_ids = summ.get("false_positive_ids")
    n_fp = int(summ.get("n_false_positive") or 0)
    if fp_ids is not None and len(fp_ids) != n_fp:
        raise ValueError("%s: %d false-positive IDs but n_false_positive=%d"
                         % (seed_dir, len(fp_ids), n_fp))
    if fp_ids is None:
        fp_ids = []
        if n_fp:
            warnings.warn("%s: %d false positive(s) without star IDs — "
                          "left as ND" % (seed_dir, n_fp))
    if cut is None:
        cut = mock_cut(seed_dir, smooth_sigma)
    syn["star_id"] = syn["star_id"].astype(str)
    if syn["star_id"].duplicated().any():
        raise ValueError("%s: more than one detected binary per star"
                         % seed_dir)
    unknown = sorted(set(syn["star_id"]) - set(star_ids))
    if unknown:
        raise ValueError("%s: stars not in the sample: %s"
                         % (seed_dir, unknown))
    det = syn.set_index("star_id")
    p_circ = float(meta.get("p_circ", 2.26))
    rows = []
    for sid in star_ids:
        rec = {"star_id": sid, "outcome": "ND", "reason": "",
               "logP": np.nan, "e": np.nan, "e_is_upper_limit": False,
               "K1": np.nan}
        if sid in det.index:
            r = det.loc[sid]
            circ = (float(r["e"]) < 1e-6) or (float(r["P"]) < p_circ)
            rec.update(logP=float(r["logP"]),
                       e=0.0 if circ else float(r["e"]),
                       e_is_upper_limit=bool(circ), K1=float(r["K1"]))
            if rec["logP"] >= cut:
                rec["outcome"] = "DX"
            else:
                rec.update(outcome="DO", reason="below_cut")
        elif sid in fp_ids:
            rec.update(outcome="DO", reason="false_positive")
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["logP_cut"] = float(cut)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seed-dir", required=True)
    ap.add_argument("--stars-csv", required=True,
                    help="injection set stars.csv (defines the sample)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cut", type=float, default=None)
    a = ap.parse_args(argv)
    stars = pd.read_csv(a.stars_csv)
    df = build(a.seed_dir, [str(s) for s in stars["ID"]], cut=a.cut)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    c = df["outcome"].value_counts().to_dict()
    print("mock catalog: cut=%.4f  %s  -> %s"
          % (df["logP_cut"].iloc[0], c, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
