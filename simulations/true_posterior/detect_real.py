"""
simulations.true_posterior.detect_real — run the fast detector that labels
the injection set (``fast_detect.detector.detect_batch``, ext3 settings) on
the REAL radial-velocity curves of the detected O-star binaries, to test
its sensitivity against the production detections at the low-K1 end.

Curves: SB1 orbits from the CCF tables (``fixed Mean RV``); SB2 primaries
from the fixed-RV SB2 tree (``rv1`` with the ``use1`` mask); 3-042 from
``rv2`` of its SB2 file; 4-080 B from its own sb3 file. Each curve is run
on its own MJD set (a subset of the field's array), with the per-star scalar
RV error of the injection set (``stars.csv``) unless ``--err-source real``
(median of the curve's own errors). The 10-permutation FAP gate makes a
decision stochastic, so every curve is run with ``--n-seeds`` detector seeds
and the recovery fraction is reported.

    python -m simulations.true_posterior.detect_real \
        --config configs/params_bias_astro3_ed_hoc_rlof_check_ext3.yaml \
        --out-dir $SCRIPTS_OUT/simulation_pipeline/true_posterior/detect_real
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

from fast_detect.detector import detect_batch
from simulations.common import BLOEM_MJD_ARRAYS
from simulations.true_posterior.injection_set import (
    _det_branch, load_run_config)

HOME = os.path.expanduser("~")
SCRIPTS_OUT = os.environ.get(
    "SCRIPTS_OUT", os.path.join(HOME, "Roey/Masters/Reasearch/scriptsOut"))
PAPER = os.path.join(HOME, "Roey/Masters/Reasearch/Ostars_article")
DEFAULTS = {
    "sb1": os.path.join(PAPER, "tables/sb1_solutions.csv"),
    "sb2": os.path.join(PAPER, "tables/sb2_solutions.csv"),
    "ccf_dir": os.path.join(SCRIPTS_OUT, "CCF/dr5_neb_div_from_coadded"),
    "sb2_dir": os.path.join(SCRIPTS_OUT, "sb2_fixed_rvs"),
    "stars": os.path.join(SCRIPTS_OUT, "simulation_pipeline/true_posterior/"
                          "injset_smoke_rv/stars.csv"),
    "sb3_4080": os.path.join(HOME, "Documents/Data/BLOeM_Data/"
                             "BLOeM_DR5.0_Combined_perStar_sb2/BLOeM_4-080/"
                             "sb2_analysis/20260303_193228/data/4-080_sb3.csv"),
    "out_dir": os.path.join(SCRIPTS_OUT, "simulation_pipeline/true_posterior/"
                            "detect_real"),
}


def _num(v):
    """float from a table cell; None for '---', limits and blanks."""
    s = str(v).strip()
    if s in ("", "---", "nan") or s.startswith((">=", "<=", "<", ">")):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def star_of(label):
    m = re.search(r"\d-\d{3}", str(label))
    return m.group(0) if m else None


def load_rows(sb1_csv, sb2_csv):
    rows = []
    for table, path in (("SB1", sb1_csv), ("SB2", sb2_csv)):
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            P = _num(r["P [d]"])
            if P is None:
                continue
            rows.append({"label": str(r["BLOeM ID"]).strip(), "table": table,
                         "star": star_of(r["BLOeM ID"]), "P": P,
                         "K1": _num(r["K1 [km/s]"]), "e": _num(r["e"])})
    return rows


def read_curve(row, a):
    """(mjd, rv, err) of the real curve behind one table row."""
    lab, star = row["label"], row["star"]
    if row["table"] == "SB2" or lab.startswith("3-042"):
        f = os.path.join(a.sb2_dir, "BLOeM_%s" % star, "data",
                         "rv_sb2_for_mcmc.csv")
        d = pd.read_csv(f, comment="#")
        comp = "2" if lab.startswith("3-042") else "1"
        use = d["use" + comp].astype(bool).to_numpy() if "use" + comp in d \
            else np.ones(len(d), bool)
        d = d[use]
        return (d["mjd"].to_numpy(float), d["rv" + comp].to_numpy(float),
                d["rv%s_err" % comp].to_numpy(float), f)
    if lab.startswith("4-080 B"):
        d = pd.read_csv(a.sb3_4080)
        return (d["MJD"].to_numpy(float), d["rv"].to_numpy(float),
                d["rv_err"].to_numpy(float), a.sb3_4080)
    f = os.path.join(a.ccf_dir, "BLOeM_%s_CCF_RVs.csv" % star)
    d = pd.read_csv(f)
    d = d.dropna(subset=["MJD", "fixed Mean RV", "fixed Mean RVsig"])
    return (d["MJD"].to_numpy(float), d["fixed Mean RV"].to_numpy(float),
            d["fixed Mean RVsig"].to_numpy(float), f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", required=True, help="bias YAML (ext3)")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    ap.add_argument("--err-source", choices=["stars", "real"], default="stars")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261004)
    a = ap.parse_args(argv)

    args_dict, cfg, _bg, det_method = load_run_config(a.config)
    if det_method != "pipeline_batched" or cfg.get("apply_rlof_gate", True):
        raise SystemExit("expected pipeline_batched with the RLOF gate off")
    backend = args_dict.get("fast_detect_backend", "numpy")
    stars = pd.read_csv(a.stars).set_index("ID")
    rows = load_rows(a.sb1, a.sb2)
    curves = []
    for r in rows:
        mjd, rv, err, src = read_curve(r, a)
        ok = np.isfinite(mjd) & np.isfinite(rv) & np.isfinite(err)
        mjd, rv, err = mjd[ok], rv[ok], err[ok]
        o = np.argsort(mjd)
        mjd, rv, err = mjd[o], rv[o], err[o]
        s = stars.loc[r["star"]]
        field = int(s["field"])
        arr = np.asarray(BLOEM_MJD_ARRAYS[field], float)
        if not np.all(np.isin(mjd, arr)):
            raise SystemExit("%s: MJDs are not a subset of field array %d"
                             % (r["label"], field))
        r.update({"field": field, "n_ep": int(mjd.size),
                  "rv_err_stars": float(s["rv_err"]),
                  "rv_err_real": float(np.median(err)), "source": src})
        curves.append((mjd, rv))
    print("%d curves (%d SB1 rows, %d SB2 rows)" % (
        len(rows), sum(r["table"] == "SB1" for r in rows),
        sum(r["table"] == "SB2" for r in rows)))

    # group by the exact MJD set (one detect_batch call per group and seed)
    groups = {}
    for i, (mjd, _rv) in enumerate(curves):
        groups.setdefault(tuple(np.round(mjd, 6)), []).append(i)
    print("%d distinct MJD sets" % len(groups))
    n = len(rows)
    det = np.zeros((a.n_seeds, n), bool)
    kill = np.zeros((a.n_seeds, n), np.int8)
    branch = np.zeros((a.n_seeds, n), np.int8)
    pfit = np.full((a.n_seeds, n), np.nan)
    pb = np.full((a.n_seeds, n), np.nan)
    fap = np.full((a.n_seeds, n), np.nan)
    for sidx in range(a.n_seeds):
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, sidx]))
        for key, idx in groups.items():
            mjds = np.asarray(key, float)
            Y = np.stack([curves[i][1] for i in idx], axis=0)      # (n, ep)
            errs = np.array([rows[i]["rv_err_" + a.err_source] for i in idx])
            d, info = detect_batch(mjds, Y.T, errs, None, cfg,
                                   backend=backend, rng=rng)
            br = _det_branch(info, d)
            det[sidx, idx] = d
            kill[sidx, idx] = info["kill_flag"]
            branch[sidx, idx] = br
            pfit[sidx, idx] = np.where(br == 1, info["P_fit_ls"],
                                       info["P_fit_pdc"])
            pb[sidx, idx] = np.maximum(info["prob_bicc_ls"],
                                       info["prob_bicc_pdc"])
            fap[sidx, idx] = np.minimum(info["LS_iter_fap"],
                                        info["PDC_iter_fap"])
        print("seed %2d: %d/%d detected" % (sidx, int(det[sidx].sum()), n),
              flush=True)

    out = []
    for i, r in enumerate(rows):
        dd = det[:, i]
        pf = pfit[:, i][dd]
        out.append({**{k: r[k] for k in ("label", "table", "star", "P", "K1",
                                         "e", "field", "n_ep", "rv_err_stars",
                                         "rv_err_real")},
                    "det_frac": float(dd.mean()),
                    "kill_mode": int(np.bincount(kill[:, i]).argmax()),
                    "branch_mode": int(np.bincount(branch[:, i]).argmax()),
                    "P_fit_median": float(np.median(pf)) if pf.size else np.nan,
                    "P_fit_over_P": (float(np.median(pf) / r["P"])
                                     if pf.size else np.nan),
                    "prob_bicc_median": float(np.nanmedian(pb[:, i])),
                    "fap_median": float(np.nanmedian(fap[:, i])),
                    "source": r["source"]})
    df = pd.DataFrame(out).sort_values("K1")
    os.makedirs(a.out_dir, exist_ok=True)
    tag = "stars" if a.err_source == "stars" else "realerr"
    path = os.path.join(a.out_dir, "detect_real_%s.csv" % tag)
    with open(path, "w") as fh:
        fh.write("# config=%s err_source=%s n_seeds=%d seed=%d\n"
                 % (a.config, a.err_source, a.n_seeds, a.seed))
        df.to_csv(fh, index=False)
    print("wrote", path)
    print("\nrecovery by K1 bin (mean detection fraction over seeds):")
    for lo, hi in ((0, 25), (25, 50), (50, 100), (100, 1e9)):
        s = df[(df.K1 >= lo) & (df.K1 < hi)]
        print("  K1 [%3d, %4s): N = %2d  mean det_frac %.3f  always detected %2d"
              "  never %2d" % (lo, "inf" if hi > 1e8 else "%d" % hi, len(s),
                              s.det_frac.mean(), int((s.det_frac == 1).sum()),
                              int((s.det_frac == 0).sum())))
    print("overall: %.3f of %d curves recovered on average" % (df.det_frac.mean(), n))
    miss = df[df.det_frac < 1.0]
    if len(miss):
        print("\ncurves not recovered in every seed:")
        print(miss[["label", "table", "P", "K1", "e", "n_ep", "rv_err_stars",
                    "det_frac", "kill_mode", "P_fit_over_P", "prob_bicc_median",
                    "fap_median"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
