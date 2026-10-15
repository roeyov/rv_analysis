"""
simulations.true_posterior.make_mocks — end-to-end mock surveys at a split
truth, with the real detector and false positives.

Each mock is one realization of the survey: for every catalogue star,
independently, a binary with logP >= logP* with probability f_gt, a binary
with logP < logP* with probability f_lt, else a single. Binary orbits come
from the split model's population (logP from the power law pi truncated to
the component's range, q from q^kappa, e from e^eta with the p_circ rule
and circular fractions c1 on [p_circ, P_env) and c2 above P_env), the
nuisances and noise from ``injection._star_binary_rvs`` (as the grid and
the injection set), and every curve — singles included — goes through the
configured detector (``pipeline_batched`` or ``rv_threshold``). Detected
singles are the mock's false positives, recorded by star.

Each mock directory follows the Tier-4 layout that ``mock_catalog.build``
and ``coverage.py`` read: synthetic_obs.csv (one row per detected binary),
sb1/sb2_solutions.tex (so the numerical elbow can be recomputed exactly as
for the real tables), truth_params.yaml (truth incl. f_gt, f_lt, log10_rho,
cut, fbin = f_gt + f_lt; summary incl. false_positive_ids), mock_meta.yaml
(detector and sampling bounds) and mock_truth.csv (every star's component,
orbit and detection flag).

    python -u -m simulations.true_posterior.make_mocks \\
        --config configs/params_bias_astro3_ed_hoc_rlof_check_ext3.yaml \\
        --truth-from <real split run dir> --out-dir <campaign dir> \\
        --tag adopted --n-mocks 200 [--seed 20260930] [--n-workers 16]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

from simulations.bias_grid_closure_generate import (
    _write_sb1_table, _write_sb2_table, split_sb1_sb2)
from simulations.bias_grid_lib.injection import _star_binary_rvs
from simulations.common import BLOEM_MJD_ARRAYS
from simulations.true_posterior.boot_coverage import (
    TRUTH_KEYS, truth_from_run, truth_record)
from simulations.true_posterior.injection_set import (
    load_run_config, load_stars)
from simulations.true_posterior.population import powerlaw_ppf

_G = {}


def draw_population(stars, truth, bounds, p_circ, p_env, seed, mock_index):
    """Per-star component and orbit draws for one mock.

    Returns a DataFrame (star order) with component (0 single, 1 above the
    cut, 2 below), logP, q, e_draw (before the p_circ rule) and circ flag.
    Seeds: SeedSequence([seed, mock_index, star_index]).
    """
    rows = []
    for _, st in stars.iterrows():
        s = int(st["star_index"])
        rng = np.random.default_rng(np.random.SeedSequence([seed, mock_index,
                                                           s]))
        u = rng.random()
        if u < truth["f_gt"]:
            comp, lo, hi = 1, truth["cut"], bounds["log_p_max"]
        elif u < truth["f_gt"] + truth["f_lt"]:
            comp, lo, hi = 2, bounds["log_p_min"], truth["cut"]
        else:
            comp = 0
        rec = {"star_index": s, "star_id": str(st["ID"]), "component": comp,
               "logP": np.nan, "q": np.nan, "e_draw": np.nan, "circ": False}
        if comp:
            logP = float(powerlaw_ppf(rng.random(), truth["pi"], lo, hi))
            q = float(powerlaw_ppf(rng.random(), truth["kappa"],
                                   bounds["q_min"], bounds["q_max"]))
            P = 10.0 ** logP
            c = 0.0 if P < p_circ else (truth["c1"] if P < p_env
                                        else truth["c2"])
            circ = rng.random() < c
            e = 0.0 if circ else float(powerlaw_ppf(rng.random(),
                                                    truth["eta"], 1e-6,
                                                    bounds["e_max"]))
            rec.update(logP=logP, q=q, e_draw=e, circ=bool(circ))
        rec["_rng_state"] = rng     # continues into the nuisance draws
        rows.append(rec)
    return rows


def _one(mock_index):
    (stars, truth, bounds, cfg, args_dict, detect_method, out_root, tag,
     seed, sb2_q) = (_G[k] for k in ("stars", "truth", "bounds", "cfg",
                                     "args_dict", "detect_method", "out",
                                     "tag", "seed", "sb2_q"))
    seed_dir = os.path.join(out_root, tag, "seed_%d" % mock_index)
    if os.path.exists(os.path.join(seed_dir, "truth_params.yaml")):
        return mock_index, "skip", 0.0
    t0 = time.time()
    p_circ, p_env = float(cfg["p_circ"]), float(_G["p_env"])
    pop = draw_population(stars, truth, bounds, p_circ, p_env, seed,
                          mock_index)
    by_field = {}
    for rec, (_, st) in zip(pop, stars.iterrows()):
        by_field.setdefault(int(st["field"]), []).append((rec, st))

    det_all = {}
    K1_all, e_all = {}, {}
    for field, items in sorted(by_field.items()):
        mjds = np.asarray(BLOEM_MJD_ARRAYS[field], dtype=np.float64)
        n_ep = mjds.size
        curves, errs, owners = [], [], []
        for rec, st in items:
            rng = rec.pop("_rng_state")
            rv_err = float(st["rv_err"])
            if rec["component"]:
                e_arr = np.array([rec["e_draw"]])
                (_lp, e_arr, K1, _q, rv, _m, _r, _i, _o, _t) = \
                    _star_binary_rvs(mjds, n_ep, rv_err, float(st["Mspec"]),
                                     float(st["R_star"]), float(st["gamma"]),
                                     np.array([rec["logP"]]),
                                     np.array([rec["q"]]), e_arr, cfg, rng)
                K1_all[rec["star_index"]] = float(K1[0])
                e_all[rec["star_index"]] = float(e_arr[0])
                curves.append(rv[0])
            else:
                curves.append(float(st["gamma"])
                              + rng.normal(0.0, rv_err, n_ep))
            errs.append(rv_err)
            owners.append(rec["star_index"])
        rv_all = np.stack(curves)
        err_all = np.asarray(errs, float)
        if detect_method == "pipeline_batched":
            from fast_detect.detector import detect_batch
            det_rng = np.random.default_rng(
                np.random.SeedSequence([seed, mock_index, 10 ** 6 + field]))
            det, _info = detect_batch(mjds, rv_all.T, err_all, None, cfg,
                                      backend=args_dict.get(
                                          "fast_detect_backend", "numpy"),
                                      rng=det_rng)
            det = np.asarray(det, dtype=bool)
        else:
            drv = float(args_dict.get("drv_threshold", 20.0))
            sig = float(args_dict.get("significance_threshold", 4.0))
            eff = np.maximum(drv, sig * err_all * np.sqrt(2.0))
            det = (rv_all.max(axis=1) - rv_all.min(axis=1)) > eff
        for o, d in zip(owners, det):
            det_all[o] = bool(d)

    detections, fp_ids, truth_rows = [], [], []
    for rec in pop:
        s = rec["star_index"]
        d = det_all[s]
        row = {"star_id": rec["star_id"], "component": rec["component"],
               "logP": rec["logP"], "q": rec["q"],
               "e": e_all.get(s, np.nan), "K1": K1_all.get(s, np.nan),
               "circ": rec["circ"], "detected": int(d)}
        truth_rows.append(row)
        if not d:
            continue
        if rec["component"]:
            detections.append({"star_id": rec["star_id"],
                               "logP": float(rec["logP"]),
                               "P": float(10.0 ** rec["logP"]),
                               "e": float(e_all[s]), "K1": float(K1_all[s]),
                               "q": float(rec["q"])})
        else:
            fp_ids.append(rec["star_id"])
    n_comp = np.bincount([r["component"] for r in pop], minlength=3)
    summary = {"n_stars": int(len(pop)), "n_physical": int(n_comp[1]
                                                            + n_comp[2]),
               "n_above": int(n_comp[1]), "n_below": int(n_comp[2]),
               "n_detected": int(len(detections)),
               "n_detected_above": int(sum(d["logP"] >= truth["cut"]
                                           for d in detections)),
               "n_false_positive": int(len(fp_ids)),
               "false_positive_ids": fp_ids}

    os.makedirs(seed_dir, exist_ok=True)
    sb1, sb2 = split_sb1_sb2(detections, sb2_q)
    _write_sb1_table(os.path.join(seed_dir, "sb1_solutions.tex"), sb1, p_circ)
    _write_sb2_table(os.path.join(seed_dir, "sb2_solutions.tex"), sb2, p_circ)
    pd.DataFrame(detections, columns=["star_id", "logP", "P", "e", "K1",
                                      "q"]).assign(
        role=lambda d: np.where(d["q"] >= sb2_q, "SB2", "SB1")
    )[["star_id", "role", "P", "logP", "e", "K1", "q"]].to_csv(
        os.path.join(seed_dir, "synthetic_obs.csv"), index=False)
    pd.DataFrame(truth_rows).to_csv(os.path.join(seed_dir, "mock_truth.csv"),
                                    index=False)
    trec = _G["trec"]
    with open(os.path.join(seed_dir, "truth_params.yaml"), "w") as fh:
        yaml.safe_dump({"truth": trec, "seed": int(seed),
                        "mock_index": int(mock_index),
                        "sb2_q_threshold": float(sb2_q), "summary": summary,
                        "n_sb1": len(sb1), "n_sb2": len(sb2),
                        "config": _G["config"], "p_circ": p_circ,
                        "p_env": p_env, "generator": "make_mocks"},
                       fh, sort_keys=False)
    with open(os.path.join(seed_dir, "mock_meta.yaml"), "w") as fh:
        yaml.safe_dump({"truth": trec, "seed": int(seed),
                        "mock_index": int(mock_index),
                        "detect_method": detect_method,
                        "n_inject_per_star": 1, "summary": summary,
                        "sampling_bounds": {**{k: float(bounds[k])
                                               for k in bounds},
                                            "p_circ": p_circ},
                        "config": _G["config"], "generator": "make_mocks"},
                       fh, sort_keys=False)
    return mock_index, "done", time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", required=True, help="bias YAML (detector, "
                    "sampling bounds, sample paths)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--truth-from", default=None)
    g.add_argument("--truth-yaml", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="adopted")
    ap.add_argument("--n-mocks", type=int, default=200)
    ap.add_argument("--first", type=int, default=0,
                    help="index of the first mock (seed_<index>)")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--p-env", type=float, default=5.0)
    ap.add_argument("--sb2-q-threshold", type=float, default=0.5)
    ap.add_argument("--n-workers", type=int, default=1)
    a = ap.parse_args(argv)

    args_dict, cfg, bg, detect_method = load_run_config(a.config)
    stars = load_stars(bg)
    bounds = {k: float(cfg[k]) for k in ("log_p_min", "log_p_max", "q_min",
                                         "q_max", "e_max")}
    truth = (truth_from_run(a.truth_from) if a.truth_from
             else {k: float(v) for k, v in
                   yaml.safe_load(open(a.truth_yaml)).items()})
    missing = [k for k in TRUTH_KEYS if k not in truth]
    if missing:
        raise SystemExit("truth lacks %s" % missing)
    if not (bounds["log_p_min"] < truth["cut"] < bounds["log_p_max"]):
        raise SystemExit("cut %.4f outside the logP bounds" % truth["cut"])
    if 10.0 ** truth["cut"] <= float(cfg["p_circ"]):
        raise SystemExit("cut must lie above p_circ")
    trec = truth_record(truth, bounds)
    print("[mocks] %s, %d stars, detector %s, truth %s"
          % (a.config, len(stars), detect_method,
             {k: round(v, 4) for k, v in trec.items()}), flush=True)
    os.makedirs(os.path.join(a.out_dir, a.tag), exist_ok=True)
    with open(os.path.join(a.out_dir, a.tag, "campaign_truth.yaml"),
              "w") as fh:
        yaml.safe_dump({"truth": trec, "seed": a.seed, "n_mocks": a.n_mocks,
                        "first": a.first, "config": os.path.abspath(a.config),
                        "detect_method": detect_method,
                        "truth_from": a.truth_from,
                        "truth_yaml": a.truth_yaml, "p_env": a.p_env},
                       fh, sort_keys=False)
    _G.update(stars=stars, truth=truth, trec=trec, bounds=bounds, cfg=cfg,
              args_dict=args_dict, detect_method=detect_method,
              out=a.out_dir, tag=a.tag, seed=a.seed, sb2_q=a.sb2_q_threshold,
              config=os.path.abspath(a.config), p_env=a.p_env)
    idx = list(range(a.first, a.first + a.n_mocks))
    t0 = time.time()
    n_det, n_fp, n_done = [], [], 0
    if a.n_workers > 1 and sys.platform.startswith("linux"):
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_one, idx)):
                if res[1] == "done":
                    n_done += 1
                if (i + 1) % 20 == 0 or i == len(idx) - 1:
                    print("[mocks] %d/%d (elapsed %.0f s)"
                          % (i + 1, len(idx), time.time() - t0), flush=True)
    else:
        for i, m in enumerate(idx):
            res = _one(m)
            print("[mocks] %d/%d seed_%d %s %.1f s"
                  % (i + 1, len(idx), m, res[1], res[2]), flush=True)
    for m in idx:
        with open(os.path.join(a.out_dir, a.tag, "seed_%d" % m,
                               "truth_params.yaml")) as fh:
            s = yaml.safe_load(fh)["summary"]
        n_det.append(s["n_detected"] + s["n_false_positive"])
        n_fp.append(s["n_false_positive"])
    print("[mocks] done: %d mocks, N_D %.1f +- %.1f (FP %.2f per mock), "
          "%.0f s" % (len(idx), np.mean(n_det), np.std(n_det), np.mean(n_fp),
                      time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
