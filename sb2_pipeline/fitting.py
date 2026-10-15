"""
sb2_pipeline.fitting — joint lmfit objectives and the per-candidate fit loop.

The joint objective mirrors orbital.fitting.chisqr_with_jitter over both
components (cost = -2 lnL up to a constant):
    sig_i^2 = err_i^2 + exp(ln_sigmaJ_i)^2
    cost = sum(log sig1^2) + sum(r1^2/sig1^2)
         + sum(log sig2^2) + sum(r2^2/sig2^2)

The null model is two independent constants (GAMMA1, GAMMA2) plus the two
jitters: for a genuine SB2 the components straddle gamma at every epoch, so a
shared-gamma null would be rejected trivially and prob_bicc would be a no-op;
against two flat lines the gate actually tests Keplerian phasing.
"""

import copy

import numpy as np
import lmfit

from utils.constants import (
    LMFIT_PARAMS, SEARCH_REGION, MINI_METHOD, MAX_NFEV, DE_SEED,
    FIT_CIRCULAR_COMPARISON, MIN_VAL, MAX_VAL,
    PERIOD, GAMMA, K1_STR, K2_STR, OMEGA, ECC, T,
    LN_SIGMA_JITTER, LN_SIGMA_JITTER_1, LN_SIGMA_JITTER_2,
)
from orbital.fitting import define_search_param
from orbital.kepler import nus1
from orbital.statistics import summarize_result, calculate_binary_probability
from simulations.common import RV12
from pipeline.config import change_search_region_default
from pipeline.evaluator import _derive_de_seed

from sb2_pipeline.constants import (
    SB2_TS1, SB2_TS2, SB2_RV1, SB2_ERR1, SB2_RV2, SB2_ERR2, GAMMA1, GAMMA2,
    JITTER_MODE,
)
from sb2_pipeline.statistics import sb2_statistical_flags, enforce_k_convention_row

_DATA_KEYS = (SB2_TS1, SB2_RV1, SB2_ERR1, SB2_TS2, SB2_RV2, SB2_ERR2)


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def sb2_joint_cost(p, **kws):
    """Scalar joint-likelihood cost for the 9-parameter SB2 model."""
    gamma = float(p[GAMMA].value)
    K1 = float(p[K1_STR].value)
    K2 = float(p[K2_STR].value)
    omega = float(p[OMEGA].value)
    ecc = float(p[ECC].value)
    T0 = float(p[T].value)
    P = float(p[PERIOD].value)
    sj1 = float(np.exp(p[LN_SIGMA_JITTER_1].value))
    sj2 = float(np.exp(p[LN_SIGMA_JITTER_2].value))

    ts1 = np.asarray(kws[SB2_TS1], float)
    rv1 = np.asarray(kws[SB2_RV1], float)
    err1 = np.asarray(kws[SB2_ERR1], float)
    ts2 = np.asarray(kws[SB2_TS2], float)
    rv2 = np.asarray(kws[SB2_RV2], float)
    err2 = np.asarray(kws[SB2_ERR2], float)

    if not (np.isfinite(P) and P > 0 and np.isfinite(ecc) and 0 <= ecc < 1
            and np.isfinite(sj1) and sj1 >= 0
            and np.isfinite(sj2) and sj2 >= 0):
        return 2e12
    try:
        v1, _ = RV12(nus1(ts1, P, T0, ecc), gamma, K1, K2, omega, ecc)
        _, v2 = RV12(nus1(ts2, P, T0, ecc), gamma, K1, K2, omega, ecc)
        sig1 = err1 ** 2 + sj1 ** 2
        sig2 = err2 ** 2 + sj2 ** 2
        res = (np.log(sig1).sum() + ((v1 - rv1) ** 2 / sig1).sum()
               + np.log(sig2).sum() + ((v2 - rv2) ** 2 / sig2).sum())
    except Exception:
        res = 2e12
    return float(res)


def sb2_joint_cost_sharedJ(p, **kws):
    """Scalar joint cost with ONE shared jitter (8-parameter SB2 model)."""
    gamma = float(p[GAMMA].value)
    K1 = float(p[K1_STR].value)
    K2 = float(p[K2_STR].value)
    omega = float(p[OMEGA].value)
    ecc = float(p[ECC].value)
    T0 = float(p[T].value)
    P = float(p[PERIOD].value)
    sj = float(np.exp(p[LN_SIGMA_JITTER].value))

    ts1 = np.asarray(kws[SB2_TS1], float)
    rv1 = np.asarray(kws[SB2_RV1], float)
    err1 = np.asarray(kws[SB2_ERR1], float)
    ts2 = np.asarray(kws[SB2_TS2], float)
    rv2 = np.asarray(kws[SB2_RV2], float)
    err2 = np.asarray(kws[SB2_ERR2], float)

    if not (np.isfinite(P) and P > 0 and np.isfinite(ecc) and 0 <= ecc < 1
            and np.isfinite(sj) and sj >= 0):
        return 2e12
    try:
        v1, _ = RV12(nus1(ts1, P, T0, ecc), gamma, K1, K2, omega, ecc)
        _, v2 = RV12(nus1(ts2, P, T0, ecc), gamma, K1, K2, omega, ecc)
        sig1 = err1 ** 2 + sj ** 2
        sig2 = err2 ** 2 + sj ** 2
        res = (np.log(sig1).sum() + ((v1 - rv1) ** 2 / sig1).sum()
               + np.log(sig2).sum() + ((v2 - rv2) ** 2 / sig2).sum())
    except Exception:
        res = 2e12
    return float(res)


def sb2_null_cost(p, **kws):
    """Null model: two independent constant velocities + two jitters."""
    g1 = float(p[GAMMA1].value)
    g2 = float(p[GAMMA2].value)
    sj1 = float(np.exp(p[LN_SIGMA_JITTER_1].value))
    sj2 = float(np.exp(p[LN_SIGMA_JITTER_2].value))

    rv1 = np.asarray(kws[SB2_RV1], float)
    err1 = np.asarray(kws[SB2_ERR1], float)
    rv2 = np.asarray(kws[SB2_RV2], float)
    err2 = np.asarray(kws[SB2_ERR2], float)

    sig1 = err1 ** 2 + sj1 ** 2
    sig2 = err2 ** 2 + sj2 ** 2
    res = (np.log(sig1).sum() + ((rv1 - g1) ** 2 / sig1).sum()
           + np.log(sig2).sum() + ((rv2 - g2) ** 2 / sig2).sum())
    return float(res)


def sb2_null_cost_sharedJ(p, **kws):
    """Null model: two independent constants + ONE shared jitter."""
    g1 = float(p[GAMMA1].value)
    g2 = float(p[GAMMA2].value)
    sj = float(np.exp(p[LN_SIGMA_JITTER].value))

    rv1 = np.asarray(kws[SB2_RV1], float)
    err1 = np.asarray(kws[SB2_ERR1], float)
    rv2 = np.asarray(kws[SB2_RV2], float)
    err2 = np.asarray(kws[SB2_ERR2], float)

    sig1 = err1 ** 2 + sj ** 2
    sig2 = err2 ** 2 + sj ** 2
    res = (np.log(sig1).sum() + ((rv1 - g1) ** 2 / sig1).sum()
           + np.log(sig2).sum() + ((rv2 - g2) ** 2 / sig2).sum())
    return float(res)


def jitter_mode_of(args_dict):
    """"separate" (default) or "shared" from lmfit_params.jitter_mode."""
    mode = args_dict[LMFIT_PARAMS].get(JITTER_MODE, "separate")
    if mode not in ("separate", "shared"):
        raise ValueError(f"Unknown jitter_mode {mode!r}")
    return mode


# ---------------------------------------------------------------------------
# lmfit wrapper
# ---------------------------------------------------------------------------

def _weighted_mean(v, err):
    w = 1.0 / np.asarray(err, float) ** 2
    return float(np.sum(np.asarray(v, float) * w) / np.sum(w))


def lmfit_sb2(args_dict, data, null_hyp=False):
    """Joint SB2 fit (or two-constant null when null_hyp). Mirrors lmfit_on_sample."""
    lmfit_params_dict = args_dict[LMFIT_PARAMS]
    mini_method = lmfit_params_dict[MINI_METHOD]
    max_nfev = int(lmfit_params_dict.get(MAX_NFEV, 200000))
    search_params = lmfit_params_dict[SEARCH_REGION]

    de_seed = lmfit_params_dict.get(DE_SEED, None)
    minimize_kws = {}
    if de_seed is not None and mini_method == "differential_evolution":
        minimize_kws["seed"] = int(de_seed)

    fcn_kws = {k: data[k] for k in _DATA_KEYS}
    params = lmfit.Parameters()
    shared = jitter_mode_of(args_dict) == "shared"

    if null_hyp:
        gmin = search_params[GAMMA][MIN_VAL]
        gmax = search_params[GAMMA][MAX_VAL]
        params.add(GAMMA1, value=_weighted_mean(data[SB2_RV1], data[SB2_ERR1]),
                   min=gmin, max=gmax, vary=True)
        params.add(GAMMA2, value=_weighted_mean(data[SB2_RV2], data[SB2_ERR2]),
                   min=gmin, max=gmax, vary=True)
        if shared:
            define_search_param(params, search_params, LN_SIGMA_JITTER)
            mini = lmfit.Minimizer(sb2_null_cost_sharedJ, params,
                                   fcn_kws=fcn_kws)
        else:
            define_search_param(params, search_params, LN_SIGMA_JITTER_1)
            define_search_param(params, search_params, LN_SIGMA_JITTER_2)
            mini = lmfit.Minimizer(sb2_null_cost, params, fcn_kws=fcn_kws)
    else:
        jitter_fields = ((LN_SIGMA_JITTER,) if shared
                         else (LN_SIGMA_JITTER_1, LN_SIGMA_JITTER_2))
        for field in (PERIOD, GAMMA, K1_STR, K2_STR, OMEGA, ECC, T) \
                + jitter_fields:
            define_search_param(params, search_params, field)
        objective = sb2_joint_cost_sharedJ if shared else sb2_joint_cost
        mini = lmfit.Minimizer(objective, params, fcn_kws=fcn_kws)

    return mini.minimize(method=mini_method, max_nfev=max_nfev, **minimize_kws)


# ---------------------------------------------------------------------------
# Circular comparison
# ---------------------------------------------------------------------------

_CIRC_COLS = [
    "circ_bicc", "circ_llh", "circ_chisqr", "circ_redchi",
    "circ_nfev", "circ_nvarys",
    "circ_Period_value", "circ_K1_value", "circ_K2_value",
    "circ_GAMMA_value", "circ_T0_value",
    "circ_ln_sigmaJ1_value", "circ_ln_sigmaJ2_value",
    "p_bicc_kep_vs_circ",
]


def run_circular_comparison_sb2(local_args_dict, data, kep_row):
    """
    Circular (e=0, omega=pi/2 fixed) counterpart of an SB2 candidate fit —
    7 free parameters. Returns circ_* columns (all-NaN on failure), never raises.
    """
    out = {k: np.nan for k in _CIRC_COLS}

    circ_args = copy.deepcopy(local_args_dict)
    change_search_region_default(circ_args, ECC, 0.0, 0.0, 0.9999, False)
    change_search_region_default(circ_args, OMEGA, np.pi / 2, 0.0, 6.4, False)
    circ_args[LMFIT_PARAMS][DE_SEED] = _derive_de_seed(
        local_args_dict[LMFIT_PARAMS].get(DE_SEED, None), "circ")

    try:
        circ_res = lmfit_sb2(circ_args, data)
    except Exception as e:
        print(f"SB2 circular comparison failed for period "
              f"{kep_row.get('candidate_period')}: {e}")
        return out

    stats = sb2_statistical_flags(data, circ_res, is_null=False)
    p = circ_res.params
    out.update({
        "circ_bicc": stats["bicc"],
        "circ_llh": stats["llh"],
        "circ_chisqr": stats["chisqr"],
        "circ_redchi": stats["redchi"],
        "circ_nfev": circ_res.nfev,
        "circ_nvarys": circ_res.nvarys,
        "circ_Period_value": p[PERIOD].value,
        "circ_K1_value": p[K1_STR].value,
        "circ_K2_value": p[K2_STR].value,
        "circ_GAMMA_value": p[GAMMA].value,
        "circ_T0_value": p[T].value,
        "p_bicc_kep_vs_circ": calculate_binary_probability(
            kep_row["bicc"], stats["bicc"]),
    })
    if LN_SIGMA_JITTER in p:
        out["circ_ln_sigmaJ_value"] = p[LN_SIGMA_JITTER].value
    else:
        out["circ_ln_sigmaJ1_value"] = p[LN_SIGMA_JITTER_1].value
        out["circ_ln_sigmaJ2_value"] = p[LN_SIGMA_JITTER_2].value

    kep_llh = kep_row.get("llh", np.nan)
    tol = max(1e-3 * abs(kep_llh), 1e-6) if np.isfinite(kep_llh) else np.inf
    if np.isfinite(kep_llh) and stats["llh"] < kep_llh - tol:
        print(f"WARNING: circular llh ({stats['llh']:.4f}) < Keplerian llh "
              f"({kep_llh:.4f}) for period {kep_row.get('candidate_period')} "
              f"-> Keplerian fit under-converged")
    return out


# ---------------------------------------------------------------------------
# Candidate loop (serial by design — plotting inside workers is the known
# kaleido/backend hazard, and 30 stars don't need a Pool)
# ---------------------------------------------------------------------------

def fit_sb2_candidates(args_dict, data, candidates_df, star_name, out_dir,
                       use_fwhm=True, plot=True):
    """Fit every candidate period; returns a list of summary-row dicts."""
    from sb2_pipeline.plotting import plot_sb2_lmfit_solution

    fit_circ = args_dict[LMFIT_PARAMS].get(FIT_CIRCULAR_COMPARISON, False)
    base_seed = args_dict[LMFIT_PARAMS].get(DE_SEED, None)
    ts_min = float(min(np.min(data[SB2_TS1]), np.min(data[SB2_TS2])))

    rows = []
    for sid, cand in enumerate(candidates_df.itertuples()):
        local_args = copy.deepcopy(args_dict)
        chosen_period = float(cand.period)

        fwhm_lo = float(getattr(cand, "fwhm_per_low", np.nan))
        fwhm_hi = float(getattr(cand, "fwhm_per_high", np.nan))
        if use_fwhm and np.isfinite(fwhm_lo) and np.isfinite(fwhm_hi) \
                and fwhm_lo < chosen_period < fwhm_hi:
            change_search_region_default(
                local_args, PERIOD, chosen_period, fwhm_lo, fwhm_hi, True)
        else:
            # NaN-FWHM candidates under use_fwhm still refine within +-10%
            # (vary=True); with use_fwhm off the SB1 convention fixes the
            # period at the periodogram value (vary=False).
            change_search_region_default(
                local_args, PERIOD, chosen_period,
                chosen_period * 0.9, chosen_period * 1.1, use_fwhm)

        change_search_region_default(
            local_args, T, ts_min,
            ts_min - chosen_period, ts_min + chosen_period, True)

        local_args[LMFIT_PARAMS][DE_SEED] = _derive_de_seed(
            base_seed, star_name, getattr(cand, "method", ""),
            f"{chosen_period:.10g}")

        try:
            result = lmfit_sb2(local_args, data)
        except Exception as e:
            print(f"  SB2 fit failed for period {chosen_period}: {e}")
            continue

        row = summarize_result(result, star_name)
        row.update(sb2_statistical_flags(data, result, is_null=False))
        row.update({
            "candidate_method": str(getattr(cand, "method", "")) + "_jitter",
            "candidate_period": chosen_period,
            "solution_id": sid,
            "is_best": False,
            "LS_power": float(getattr(cand, "LS_power", np.nan)),
            "LS_fap": float(getattr(cand, "LS_fap", np.nan)),
            "PDC_power": float(getattr(cand, "PDC_power", np.nan)),
            "PDC_fap": float(getattr(cand, "PDC_fap", np.nan)),
            "fwhm_per_low": fwhm_lo,
            "fwhm_per_high": fwhm_hi,
        })

        if fit_circ:
            row.update(run_circular_comparison_sb2(local_args, data, row))

        # Figure first (from the as-fitted result, matching the as-loaded
        # data), then relabel to the K1 < K2 convention.
        if plot and out_dir:
            row["phs"] = plot_sb2_lmfit_solution(
                data, result, args_dict, star_name, sid, out_dir)
        else:
            row["phs"] = ""
        enforce_k_convention_row(row)

        rows.append(row)
    return rows


def fit_sb2_null(args_dict, data, star_name, solution_id):
    """Two-constant null fit -> summary-row dict (candidate_method null_hyp_jitter)."""
    local_args = copy.deepcopy(args_dict)
    local_args[LMFIT_PARAMS][DE_SEED] = _derive_de_seed(
        args_dict[LMFIT_PARAMS].get(DE_SEED, None), star_name, "null")

    result = lmfit_sb2(local_args, data, null_hyp=True)
    row = summarize_result(result, star_name)
    row.update(sb2_statistical_flags(data, result, is_null=True))
    row.update({
        "candidate_method": "null_hyp_jitter",
        "candidate_period": 0.0,
        "solution_id": solution_id,
        "is_best": False,
        "k_swapped": False,
        "phs": "",
    })
    return row
