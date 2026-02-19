"""
Statistical tests and diagnostics for orbital fitting.

This module provides:
  - AIC/BIC-based binary probability
  - Likelihood-ratio test
  - F-test for nested models
  - RV significance and threshold tests
  - Phase coverage quality checks
  - Statistical flags from lmfit results
  - Result summarization helpers
  - Decision flag enum
"""

import numpy as np
from enum import Enum
from scipy.stats import chi2, f as f_dist

from utils.constants import (
    TIME_STAMPS, RADIAL_VELS, ERRORS,
    PERIOD, GAMMA, K1_STR, OMEGA, ECC, T,
    LN_SIGMA_JITTER,
)
from orbital.kepler import nus1, v1mod
from orbital.fitting import _safe_errors, extract_observations


# ---------------------------------------------------------------------------
# Decision flags
# ---------------------------------------------------------------------------

class DecsionFlags(Enum):
    DELTA_RV_BIN     = 0
    LOMB_SCARGLE_BIN = 1
    PDC_BIN          = 2


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def calculate_binary_probability(AIC_binary, AIC_single):
    """
    Calculate the probability that the binary model is correct based on two AIC values.

    Parameters:
        AIC_binary (float): AIC of the binary model.
        AIC_single (float): AIC of the single-star model.

    Returns:
        Probability (float) of the binary model being correct.
    """
    delta_AIC = AIC_single - AIC_binary
    probability = 1 / (1 + np.exp(-0.5 * delta_AIC))
    return probability


def lr_test_prob(chi2_full, chi2_null, k_full, k_null, N):
    """
    Return (p_value, prob_full) where:
      - p_value = P(delta_chi2 >= observed | null)
      - prob_full = 1 - p_value
    Inputs:
      chi2_full, chi2_null : floats
         the chi-squared of the full and null models
      k_full, k_null : ints
         number of fitted parameters in full vs null model
      N : int
         number of data points
    """
    nu_null = N - k_null
    nu_full = N - k_full
    delta_chi2 = chi2_null - chi2_full
    df_diff    = (nu_null - nu_full)

    p_value = chi2.sf(delta_chi2, df_diff)
    prob_full = 1 - p_value
    return p_value, prob_full


def f_test(data, kep_row, null_row):
    N = len(data[TIME_STAMPS])

    p_orb = kep_row.get("nvarys", None)
    chi2_orb = kep_row.get("chisqr", None)

    p_null = null_row.get("nvarys", 1)
    if "chisqr" in null_row:
        chi2_null = null_row["chisqr"]
    else:
        chi2_null = null_row["redchi"] * (N - p_null)

    ok = (p_orb is not None) and (chi2_orb is not None) and (N > p_orb) and (p_orb > p_null)
    if ok and (chi2_null > chi2_orb):
        dfn = p_orb - p_null
        dfd = N - p_orb

        Fstat = ((chi2_null - chi2_orb) / dfn) / (chi2_orb / dfd)
        pF = 1.0 - f_dist.cdf(Fstat, dfn=dfn, dfd=dfd)
        Fcrit99 = f_dist.ppf(0.99, dfn=dfn, dfd=dfd)
        return {
            'F_stat': float(Fstat),
            'F_pvalue': float(pF),
            'F_crit_99': float(Fcrit99),
            'chi2_null': float(chi2_null),
            'chi2_orb': float(chi2_orb),
            'df_num': int(dfn),
            'df_den': int(dfd),
            'bin_flag_Ftest': int(Fstat > Fcrit99)
        }
    else:
        return {
            'F_stat': 0,
            'F_pvalue': 0,
            'F_crit_99': 0,
            'chi2_null': 0,
            'chi2_orb': 0,
            'df_num': 0,
            'df_den': 0,
            'bin_flag_Ftest': 0
        }


# ---------------------------------------------------------------------------
# RV significance
# ---------------------------------------------------------------------------

def rv_significance(rvs, err_vs):
    np_rvs = np.array(rvs).flatten()
    err_vs = np.array(err_vs).flatten()
    err_vs_sq = err_vs * err_vs

    pairwise_diff = np.abs(np_rvs[:, None] - np_rvs[None, :])
    pairwise_sigma_sq = np.sqrt(err_vs_sq[:, None] + err_vs_sq[None, :])

    signif = pairwise_diff / pairwise_sigma_sq
    return signif


def binary_rv_threshold(rvs, err_vs, drv_tresh=20, sign_threshold=4):
    np_rvs = np.array(rvs).flatten()
    err_vs = np.array(err_vs).flatten()
    pairwise_diff = np.abs(np_rvs[:, None] - np_rvs[None, :])

    signif = rv_significance(rvs, err_vs)

    return np.any((signif > sign_threshold) & (pairwise_diff > drv_tresh))


# ---------------------------------------------------------------------------
# Statistical flags from lmfit results
# ---------------------------------------------------------------------------

def calculate_statistical_flags(data, mini_results, is_null):
    rvs = data[RADIAL_VELS]
    ts = data[TIME_STAMPS]
    rv_errs = _safe_errors(data[ERRORS])
    p = mini_results.params
    Gamma1 = float(p[GAMMA].value)
    sigma_j = float(np.exp(p[LN_SIGMA_JITTER].value))
    if not is_null:
        K1     = float(p[K1_STR].value)
        Omega  = float(p[OMEGA].value)
        ecc    = float(p[ECC].value)
        T0     = float(p[T].value)
        P      = float(p[PERIOD].value)
        nu = nus1(ts, P, T0, ecc)
        v1 = v1mod(nu, Gamma1, K1, Omega, ecc)
    else:
        v1 = Gamma1

    sig2 = rv_errs ** 2 + sigma_j ** 2
    resid = rvs - v1
    calc_chisqr = np.sum(resid**2 / sig2)

    ndata  = rvs.size
    nvarys = mini_results.nvarys
    dof    = max(ndata - nvarys, 1)

    redchi = calc_chisqr / dof

    llh = calc_chisqr + np.sum(np.log(sig2))
    aic = 2*nvarys + llh
    bic = nvarys*np.log(ndata) + llh

    bicc = nvarys*np.log(ndata)*(ndata/(ndata-nvarys-2)) + llh

    ev_bic = nvarys*np.log(ndata) + llh*(1-1/ndata)
    ret_dict = {
        "llh": llh,
        "chisqr": calc_chisqr,
        "aic": aic,
        "bic": bic,
        "bicc": bicc,
        "ev_bic": ev_bic,
        "redchi": redchi,
        "ndata": ndata,
    }
    return ret_dict


# ---------------------------------------------------------------------------
# Result summarization
# ---------------------------------------------------------------------------

def summarize_result(result, star_name):
    """
    Turn a single lmfit.MinimizerResult into a flat dict.
    """
    row = {}
    row['star_name'] = star_name
    row['method'] = result.method
    row['nfev'] = result.nfev
    row['ndata'] = result.ndata
    row['nvarys'] = result.nvarys
    row['chisqr'] = result.chisqr
    row['redchi'] = result.redchi
    row['aic'] = result.aic
    row['bic'] = result.bic

    for name, par in result.params.items():
        init = result.init_values.get(name, None)
        row[f'{name}_init'] = init
        row[f'{name}_value'] = par.value
        row[f'{name}_vary'] = par.vary
        row[f'{name}_stderr'] = par.stderr

    return row


def compute_orbital_params(result):
    params = result.params
    return (
        params[GAMMA].value,
        params[K1_STR].value,
        params[OMEGA].value,
        params[ECC].value,
        params[PERIOD].value,
        params[T].value
    )


# ---------------------------------------------------------------------------
# Phase coverage quality
# ---------------------------------------------------------------------------

def coverage_qc(gaps_dict, rv_values):
    """
    gaps_dict = {
        "max_phase_gap": ...,
        "top_max_rv_gap": ...,
        "bottom_max_rv_gap": ...
    }
    rv_values: array-like of measured RVs (km/s)
    """
    rv = np.asarray(rv_values, dtype=float)
    p10, p90 = np.percentile(rv, [10, 90])
    rv_span = max(p90 - p10, 1e-9)

    g_phase = float(gaps_dict["max_phase_gap"])
    gtop_n  = float(gaps_dict["top_max_rv_gap"])
    gbot_n  = float(gaps_dict["bottom_max_rv_gap"])

    phase_ok = (g_phase <= 0.25)
    rv_ok    = (gtop_n <= 0.50) and (gbot_n <= 0.50)

    score = 1.0 - (0.30*g_phase + 0.35*gtop_n + 0.35*gbot_n)

    passed = phase_ok and rv_ok and (score >= 0.60)

    return {
        "ph_gtop_norm": round(gtop_n, 3),
        "ph_gbot_norm": round(gbot_n, 3),
        "ph_phase_ok": phase_ok,
        "ph_rv_ok": rv_ok,
        "ph_score": round(score, 3),
        "ph_pass": passed,
        "ph_tier": "good" if score >= 0.70 else ("ok" if score >= 0.60 else "needs_work"),
    }


def calculate_phase_criterias(data, result):
    """
    Returns:
        max_phase_gap         in [0,1]
        top_max_rv_gap        in [0,1]  (distance from model crest)
        bottom_max_rv_gap     in [0,1]  (distance from model trough)
    """
    # Import here to avoid circular imports (plotting uses statistics)
    from orbital.plotting import plot_phase_folded_with_residuals

    hjds, vels, errs = extract_observations(data)
    Gamma, K, Omega, ecc, P, T0 = compute_orbital_params(result)

    phs_data, phase_grid, rv_phase, residuals = plot_phase_folded_with_residuals(
        hjds, vels, errs, P, T0, Gamma, K, Omega, ecc, '', plot=False
    )

    # --- Max phase gap on the circle ---
    phs = np.asarray(phs_data, dtype=float) % 1.0
    phs = np.sort(phs)
    if phs.size >= 2:
        diffs = np.diff(phs, append=phs[0] + 1.0)
        max_phase_gap = float(np.nanmax(diffs))
    else:
        max_phase_gap = 1.0

    # --- RV crest/trough headroom ---
    v_obs_max = float(np.nanmax(vels))
    v_obs_min = float(np.nanmin(vels))
    v_mod_max = float(np.nanmax(rv_phase))
    v_mod_min = float(np.nanmin(rv_phase))

    crest_headroom   = v_mod_max - v_obs_max
    trough_headroom  = v_obs_min - v_mod_min

    crest_scale  = max(v_mod_max - Gamma, 0.0)
    trough_scale = max(Gamma - v_mod_min, 0.0)

    if crest_scale > 0:
        top_max_rv_gap = crest_headroom / crest_scale
    else:
        top_max_rv_gap = np.nan

    if trough_scale > 0:
        bottom_max_rv_gap = trough_headroom / trough_scale
    else:
        bottom_max_rv_gap = np.nan

    def _clip01(x):
        return float(np.clip(x, 0.0, 1.0)) if np.isfinite(x) else np.nan

    max_phase_gap      = _clip01(max_phase_gap)
    top_max_rv_gap     = _clip01(top_max_rv_gap)
    bottom_max_rv_gap  = _clip01(bottom_max_rv_gap)

    return {
        "max_phase_gap": np.round(max_phase_gap, 2),
        "top_max_rv_gap": np.round(top_max_rv_gap, 2),
        "bottom_max_rv_gap": np.round(bottom_max_rv_gap, 2),
    }
