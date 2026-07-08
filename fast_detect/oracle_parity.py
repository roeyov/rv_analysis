"""Per-cell detection-fraction parity: fast detector vs the FAITHFUL oracle.

The faithful oracle is the production stack run per injection
(``pipeline.evaluator.detect_from_arrays`` with permutations ON) with the *true*
``filter_expression`` applied: detected iff

    prob_bicc > 0.5  AND  mass_flag_peri == 0  AND  iter_fap < 0.2

(the current ``detect_method: pipeline`` applies only prob_bicc > 0.5, so it is
NOT the faithful oracle — we recompute mass_flag_peri / iter_fap from its info).

Acceptance: |P_det_fast - P_det_oracle| within Monte-Carlo noise sqrt(p(1-p)/N).

Reuses the bias-grid injection physics so populations match the real grid.
Run: python -m fast_detect.oracle_parity
"""

import os
# Force the production LS-permutation FAP to run sequentially (its per-call
# ProcessPoolExecutor otherwise re-imports the harness under spawn). The bias
# grid sets this same flag inside its workers.
os.environ.setdefault("_BIAS_GRID_SUBPROCESS", "1")

import copy
import numpy as np
import pandas as pd

from simulations.bias_grid_lib.physics import (
    powerlaw_draw, compute_K1, rv_model_jit)
from utils.roche_lobe import compute_min_period_row
from utils.constants import (TIME_STAMPS, RADIAL_VELS, ERRORS, PERIOD, K1_STR, T, ECC)
from orbital.fitting import lmfit_on_sample
from orbital.statistics import (calculate_statistical_flags,
                                calculate_binary_probability)
from pipeline.config import change_search_region_default
from pipeline.evaluator import detect_from_arrays
from period_search.candidates import find_periods
from fast_detect.detector import detect_batch


def make_binary_population(cell, star, mjds, cfg, n_inject, rng):
    """Draw n_inject binaries for a grid cell on one star (reuses grid physics).

    cell : dict(pi,kappa,eta,fbin)   star : dict(M1,R1,err,mass_row)
    Returns (rv_matrix[n_epoch,n_inject], errs[n_inject], truth dict).
    """
    bg = cfg.get("bias_grid", {})
    lpmin = float(bg.get("log_p_min", 0.046)); lpmax = float(bg.get("log_p_max", 3.5))
    qmin, qmax = 0.1, 1.0; emax = 0.95
    t = np.asarray(mjds, float); ne = t.size
    logP = powerlaw_draw(n_inject, cell["pi"], lpmin, lpmax, rng)
    P = 10.0 ** logP
    q = powerlaw_draw(n_inject, cell["kappa"], qmin, qmax, rng)
    e = powerlaw_draw(n_inject, cell["eta"], 1e-6, emax, rng)
    incl = np.arccos(rng.uniform(0, 1, n_inject))
    omega = rng.uniform(0, 2 * np.pi, n_inject)
    T0 = rng.uniform(0, P)
    M1 = np.clip(rng.normal(star["M1"], 0.3 * star["M1"], n_inject), 7.0, 80.0)
    rvm = np.empty((ne, n_inject))
    K1 = np.empty(n_inject)
    for i in range(n_inject):
        K1[i] = compute_K1(P[i], e[i], q[i], M1[i], incl[i])
        rvm[:, i] = rv_model_jit(t, P[i], T0[i], omega[i], e[i], K1[i], star["gamma"])
    rvm = rvm + rng.normal(0.0, star["err"], (ne, n_inject))
    errs = np.full(n_inject, star["err"])
    return rvm, errs, dict(P=P, e=e, K1=K1)


def faithful_oracle(t, rv, err_epoch, mass_row, cfg):
    """Detected? per the true filter_expression, via the production pipeline."""
    det, info = detect_from_arrays(t, rv, err_epoch, cfg, use_fwhm=True)
    pb = float(info.get("prob_bicc", np.nan))
    lsf = float(info.get("LS_iter_fap", np.nan))
    pdf = float(info.get("PDC_iter_fap", np.nan))
    fap_ok = (np.isfinite(lsf) and lsf < 0.2) or (np.isfinite(pdf) and pdf < 0.2)
    try:
        mr = compute_min_period_row(mass_row, P_fit=float(info.get("period", np.nan)),
                                    K1=float(info.get("K1", 0.0)),
                                    e=float(info.get("ecc", 0.0)))
        mass_ok = int(mr["mass_flag_peri"]) == 0
    except Exception:
        mass_ok = True
    return bool(pb > 0.5 and fap_ok and mass_ok)


def faithful_oracle_full(t, rv, err_epoch, mass_row, cfg):
    """The TRUE detection criterion: fit EVERY LS/PDC candidate with the
    production DE fitter and apply filter_expression per row (detected if any
    candidate satisfies prob_bicc>0.5 & iter_fap<0.2 & mass_flag_peri==0).

    This is the faithful analogue of production's get_best_row(results_df) —
    unlike detect_from_arrays, which collapses to the single best-redchi row.
    """
    t = np.asarray(t, float); rv = np.asarray(rv, float); err_epoch = np.asarray(err_epoch, float)
    data = {TIME_STAMPS: pd.Series(t), RADIAL_VELS: pd.Series(rv), ERRORS: pd.Series(err_epoch)}
    out = find_periods(rv, t, err_epoch, cfg, "val", out_dir=None, use_fwhm=True)
    cdf = out[-1]
    if cdf is None or cdf.empty:
        return False
    null_args = copy.deepcopy(cfg)
    change_search_region_default(null_args, PERIOD, 0, -0.1, 0.1, False)
    change_search_region_default(null_args, K1_STR, 0, -0.1, 0.1, False)
    try:
        null_res = lmfit_on_sample(null_args, data, null_hyp=True, use_jitter=True)
        null_bicc = float(calculate_statistical_flags(data, null_res, is_null=True)["bicc"])
    except Exception:
        return False
    tmin = float(t.min())
    for cand in cdf.itertuples(index=False):
        method = str(getattr(cand, "method", ""))
        ifap = (getattr(cand, "LS_iter_fap", np.nan) if "LS" in method
                else getattr(cand, "PDC_iter_fap", np.nan))
        if not (np.isfinite(ifap) and ifap < 0.2):
            continue
        P = float(cand.period)
        fa = copy.deepcopy(cfg)
        change_search_region_default(fa, PERIOD, P, float(cand.fwhm_per_low),
                                     float(cand.fwhm_per_high), True)
        change_search_region_default(fa, T, tmin, tmin - P, tmin + P, True)
        try:
            res = lmfit_on_sample(fa, data, use_jitter=True)
            st = calculate_statistical_flags(data, res, is_null=False)
            pb = calculate_binary_probability(float(st["bicc"]), null_bicc)
        except Exception:
            continue
        if pb <= 0.5:
            continue
        p = res.params
        try:
            mf = compute_min_period_row(mass_row, P_fit=float(p[PERIOD].value),
                                        K1=abs(float(p[K1_STR].value)),
                                        e=float(p[ECC].value))["mass_flag_peri"]
        except Exception:
            mf = 0
        if int(mf) == 0:
            return True
    return False


def parity_cell(cell, star, mjds, cfg, n_inject=30, seed=0, backend="numpy",
                oracle="full"):
    rng = np.random.default_rng(seed)
    rvm, errs, truth = make_binary_population(cell, star, mjds, cfg, n_inject, rng)
    err_epoch = np.full(len(mjds), star["err"])
    orc_fn = faithful_oracle_full if oracle == "full" else faithful_oracle
    # oracle (slow): per injection
    orc = np.array([orc_fn(np.asarray(mjds, float), rvm[:, i], err_epoch,
                           star["mass_row"], cfg) for i in range(n_inject)])
    # fast (batched)
    det, _ = detect_batch(mjds, rvm, errs, [star["mass_row"]] * n_inject, cfg,
                          backend=backend, rng=np.random.default_rng(seed + 1))
    p_o, p_f = orc.mean(), det.mean()
    mc = np.sqrt(max(p_o * (1 - p_o), 1e-6) / n_inject)
    agree = (orc == det).mean()
    return dict(P_det_oracle=p_o, P_det_fast=p_f, mc_noise=mc,
                within=abs(p_o - p_f) <= 2 * mc, per_inj_agree=agree, n=n_inject)


if __name__ == "__main__":
    import yaml, pandas as pd
    from simulations.common import BLOEM_MJD_ARRAYS
    cfg = yaml.safe_load(open("configs/params_bias.yaml"))
    cfg["periodogram_params"]["max_period"] = 1000.0
    mass = pd.read_csv("/Users/roeyovadia/Documents/Data/BLOeM_Data/mass_bloem.csv")
    row = mass.iloc[10].to_dict()
    star = dict(M1=float(row["Mspec"]), R1=float(row["R_star"]), err=6.0,
                gamma=120.0, mass_row=row)
    mjds = np.array(BLOEM_MJD_ARRAYS[0], float)
    cells = [dict(pi=-0.7, kappa=-0.5, eta=-0.5, fbin=1.0),
             dict(pi=-1.2, kappa=-1.0, eta=-0.2, fbin=1.0)]
    print("%-32s %8s %8s %8s %6s" % ("cell", "P_orcl", "P_fast", "mc", "agree"))
    for c in cells:
        r = parity_cell(c, star, mjds, cfg, n_inject=30, seed=1)
        print("pi=%.1f k=%.1f eta=%.1f      %8.3f %8.3f %8.3f %6.2f"
              % (c["pi"], c["kappa"], c["eta"], r["P_det_oracle"],
                 r["P_det_fast"], r["mc_noise"], r["per_inj_agree"]))
