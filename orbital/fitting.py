"""
lmfit-based orbital fitting: objectives, parameter setup, and fitting wrappers.

This module provides:
  - Parameter setup helpers for lmfit search regions
  - Weighted chi-squared objective functions (with/without jitter)
  - Null-hypothesis (constant RV) objective
  - High-level fitting wrapper (lmfit_on_sample)
"""

import sys
import numpy as np
import lmfit
import matplotlib.pylab as pylab
from types import SimpleNamespace

from utils.constants import (
    TIME_STAMPS, RADIAL_VELS, ERRORS,
    LMFIT_PARAMS, SEARCH_REGION, MINI_METHOD, MAX_NFEV,
    PERIOD, GAMMA, K1_STR, OMEGA, ECC, T,
    LN_SIGMA_JITTER, INIT_VAL, MIN_VAL, MAX_VAL, VARY,
)
from orbital.kepler import nus1, v1mod, rv_double_kepler_from_times


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _finite_or_default(x, default=1.0):
    """Return array with non-finite values replaced by default."""
    x = np.asarray(x, float)
    bad = ~np.isfinite(x)
    if np.any(bad):
        x = x.copy()
        x[bad] = default
    return x


def _safe_errors(errs):
    """Replace non-finite / non-positive errors by a small positive value."""
    errs = np.asarray(errs, float)
    errs = np.where(np.isfinite(errs), errs, np.nan)
    # smallest sensible sigma from finite positives or fallback
    pos = errs[np.isfinite(errs) & (errs > 0)]
    floor = np.nanmedian(pos) * 1e-3 if pos.size else 1e-3
    floor = max(floor, 1e-6)
    errs = np.where(errs > 0, errs, floor)
    return errs


# ---------------------------------------------------------------------------
# Parameter setup
# ---------------------------------------------------------------------------

def define_search_param(params, search_params_dict, field, radius=0):
    """
    Defines a search parameter for a given field and adds it to the parameters object.

    Args:
        params (Parameters): The parameters object to which the search parameter will be added.
        search_params_dict (dict): A dictionary containing search parameter details with the following structure:
            - field (str): The name of the field for which the search parameter is defined.
            - INIT_VAL (float): The initial value of the parameter.
            - MIN_VAL (float): The minimum value of the parameter.
            - MAX_VAL (float): The maximum value of the parameter.
            - VARY (bool): A flag indicating whether the parameter should vary during the search.
        field (str): The specific field for which the search parameter is being defined.
        radius (float, optional): An optional radius value to adjust the min and max values of the parameter. Default is 0.

    Returns:
        None: The function modifies the `params` object in place.
    """
    params.add(field, value=search_params_dict[field][INIT_VAL],
               min=search_params_dict[field][MIN_VAL] - radius/2,
               max=search_params_dict[field][MAX_VAL] + radius/2,
               vary=search_params_dict[field][VARY])


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_observations(data):
    hjds = np.array(data[TIME_STAMPS])
    vels = np.array(data[RADIAL_VELS])
    errs = np.abs(data[ERRORS])
    return hjds, vels, errs


def get_rv_weighted_mean(data):
    v1s     = np.array(data[RADIAL_VELS])
    errv1s  = np.abs(data[ERRORS])
    weights = 1.0 / errv1s ** 2
    gamma0 = np.sum(v1s * weights) / np.sum(weights)
    return gamma0


# ---------------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------------

def compute_AIC_BIC(y_obs, errors, chi_sqr, num_params):
    """
    Compute the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

    Parameters:
        y_obs (array): Observed data.
        errors (array): Observational errors (standard deviation).
        chi_sqr (float): Chi-squared value.
        num_params (int): Number of parameters in the model (k).

    Returns:
        AIC, BIC
    """
    n = len(y_obs)
    log_likelihood_term = np.sum(np.log(2 * np.pi * errors**2))
    AIC = 2 * num_params + chi_sqr + log_likelihood_term
    BIC = num_params * np.log(n) + chi_sqr + log_likelihood_term
    return AIC, BIC


# ---------------------------------------------------------------------------
# Objective functions (residuals / scalar cost)
# ---------------------------------------------------------------------------

def chisqr_func(p, **kws):
    """
    Computes weighted residuals (model - data) / sigma for lmfit minimization.
    """
    Gamma1 = p[GAMMA].value
    K1 = p[K1_STR].value
    Omega = p[OMEGA].value
    ecc = p[ECC].value
    T0 = p[T].value
    P = p[PERIOD].value
    v1 = v1mod(nus1(kws[TIME_STAMPS], P, T0, ecc), Gamma1, K1, Omega, ecc)
    return (v1 - kws[RADIAL_VELS]) / kws[ERRORS]


def null_resid_with_jitter(params, **kws):
    """Null (constant RV) model objective with jitter term."""
    gamma = params[GAMMA].value
    ln_sj = params[LN_SIGMA_JITTER].value
    sj = np.exp(ln_sj)
    sig2 = kws[ERRORS] ** 2 + sj ** 2

    res_chi_sqrd = (kws[RADIAL_VELS] - gamma) * (kws[RADIAL_VELS] - gamma) / sig2
    res_ln = np.log(sig2)
    res = res_ln.sum() + res_chi_sqrd.sum()
    return float(res)


def chisqr_with_jitter(p, **kws):
    """
    Full orbital model objective with jitter term.

    Returns scalar log-likelihood-like cost for lmfit scalar minimizers.
    """
    Gamma1 = float(p[GAMMA].value)
    K1     = float(p[K1_STR].value)
    Omega  = float(p[OMEGA].value)
    ecc    = float(p[ECC].value)
    T0     = float(p[T].value)
    P      = float(p[PERIOD].value)
    sigmaJ = float(np.exp(p[LN_SIGMA_JITTER].value))

    hjd = np.asarray(kws[TIME_STAMPS], float)
    rv  = np.asarray(kws[RADIAL_VELS], float)
    err = np.asarray(kws[ERRORS], float)

    if not (np.isfinite(P) and P > 0 and np.isfinite(ecc) and 0 <= ecc < 1 and np.isfinite(sigmaJ) and sigmaJ >= 0):
        return 2e12
    try:
        nu  = nus1(hjd, P, T0, ecc)
        v1  = v1mod(nu, Gamma1, K1, Omega, ecc)
        sig2 = err**2 + sigmaJ**2
        res_chi_sqrd = (v1 - rv) * (v1 - rv) / sig2
        res_ln = np.log(sig2)
        res = res_ln.sum() + res_chi_sqrd.sum()
    except Exception:
        res = 2e12
    return float(res)


# ---------------------------------------------------------------------------
# High-level fitting wrapper
# ---------------------------------------------------------------------------

def lmfit_on_sample(args_dict, data, null_hyp=False, use_jitter=False):
    """
    Fit an orbital model to RV vs. MJD data, or—if null_hyp=True—fit
    the null hypothesis of a constant velocity.

    Parameters
    ----------
    args_dict : dict
        Dictionary holding your LMFIT_PARAMS entry, etc.
    data : dict-like
        Must contain keys TIME_STAMPS, RADIAL_VELS, and ERRORS.
    null_hyp : bool, optional
        If True, skip the orbital fit and instead compute reduced chi-sq of
        a constant-velocity model (weighted mean). Default is False.
    use_jitter : bool, optional
        If True, include a jitter term in the fit.

    Returns
    -------
    If null_hyp is False:
        result : lmfit.MinimizerResult
    If null_hyp is True (no jitter):
        SimpleNamespace with GAMMA, redchi, aic, bic, chisqr
    If null_hyp is True (with jitter):
        lmfit.MinimizerResult
    """
    # ---- style & recursion setup ----
    pylab_params = {
        'legend.fontsize': 'large',
        'figure.figsize': (12, 4),
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large'
    }
    pylab.rcParams.update(pylab_params)
    sys.setrecursionlimit(int(1e6))

    # ---- extract data arrays ----
    hjds1   = np.array(data[TIME_STAMPS])
    v1s     = np.array(data[RADIAL_VELS])
    errv1s  = np.abs(data[ERRORS])
    lmfit_params_dict = args_dict[LMFIT_PARAMS]
    mini_method       = lmfit_params_dict[MINI_METHOD]
    max_nfev          = int(lmfit_params_dict.get(MAX_NFEV, 200000))
    params            = lmfit.Parameters()
    search_params     = lmfit_params_dict[SEARCH_REGION]

    # ---- null hypothesis: constant velocity fit ----
    if null_hyp:
        if use_jitter:
            params = lmfit.Parameters()
            gamma0 = get_rv_weighted_mean(data)
            params.add(GAMMA, value=gamma0,
                       min=search_params[GAMMA][MIN_VAL],
                       max=search_params[GAMMA][MAX_VAL], vary=True)
            define_search_param(params, search_params, LN_SIGMA_JITTER)
            mini = lmfit.Minimizer(
                null_resid_with_jitter,
                params,
                fcn_kws={TIME_STAMPS: hjds1,
                         RADIAL_VELS: v1s,
                         ERRORS: errv1s}
            )
            result = mini.minimize(method=mini_method, max_nfev=max_nfev)
            return result
        else:
            gamma0   = get_rv_weighted_mean(data)
            chisqr    = np.sum(((v1s - gamma0) / errv1s)**2)
            dof      = len(v1s) - 1
            redchi   = chisqr / dof
            aic, bic = compute_AIC_BIC(v1s, errv1s, chisqr, 1)
            return SimpleNamespace(GAMMA=gamma0, redchi=redchi,
                                   aic=aic, bic=bic, chisqr=chisqr)

    # ---- full orbital fit ----
    define_search_param(params, search_params, PERIOD)
    define_search_param(params, search_params, GAMMA)
    define_search_param(params, search_params, K1_STR)
    define_search_param(params, search_params, OMEGA)
    define_search_param(params, search_params, ECC)
    define_search_param(params, search_params, T)
    if not use_jitter:
        mini = lmfit.Minimizer(
            chisqr_func, params,
            fcn_kws={TIME_STAMPS: hjds1,
                     RADIAL_VELS: v1s,
                     ERRORS: errv1s}
        )
    else:
        define_search_param(params, search_params, LN_SIGMA_JITTER)
        mini = lmfit.Minimizer(
            chisqr_with_jitter, params,
            fcn_kws={TIME_STAMPS: hjds1,
                     RADIAL_VELS: v1s,
                     ERRORS: errv1s}
        )
    result = mini.minimize(method=mini_method, max_nfev=max_nfev)
    return result


# ---------------------------------------------------------------------------
# Double-Keplerian (hierarchical triple) fitting
# ---------------------------------------------------------------------------

# Parameter name constants for outer orbit
_P_IN = "Period_in"
_K_IN = "K1_in"
_OMEGA_IN = "OMEGA_in"
_ECC_IN = "Ecc_in"
_T0_IN = "T0_in"
_P_OUT = "Period_out"
_K_OUT = "K1_out"
_OMEGA_OUT = "OMEGA_out"
_ECC_OUT = "Ecc_out"
_T0_OUT = "T0_out"


def chisqr_double_kepler_with_jitter(p, **kws):
    """
    Double-Keplerian objective with jitter for hierarchical triple systems.

    Returns scalar log-likelihood-like cost (same convention as
    chisqr_with_jitter).
    """
    # Inner orbit
    P_in = float(p[_P_IN].value)
    K_in = float(p[_K_IN].value)
    omega_in = float(p[_OMEGA_IN].value)
    ecc_in = float(p[_ECC_IN].value)
    T0_in = float(p[_T0_IN].value)
    # Outer orbit
    P_out = float(p[_P_OUT].value)
    K_out = float(p[_K_OUT].value)
    omega_out = float(p[_OMEGA_OUT].value)
    ecc_out = float(p[_ECC_OUT].value)
    T0_out = float(p[_T0_OUT].value)
    # Shared
    gamma = float(p[GAMMA].value)
    sigmaJ = float(np.exp(p[LN_SIGMA_JITTER].value))

    hjd = np.asarray(kws[TIME_STAMPS], float)
    rv = np.asarray(kws[RADIAL_VELS], float)
    err = np.asarray(kws[ERRORS], float)

    if not (np.isfinite(P_in) and P_in > 0
            and np.isfinite(P_out) and P_out > 0
            and np.isfinite(ecc_in) and 0 <= ecc_in < 1
            and np.isfinite(ecc_out) and 0 <= ecc_out < 1
            and np.isfinite(sigmaJ) and sigmaJ >= 0):
        return 2e12
    try:
        model = rv_double_kepler_from_times(
            hjd, P_in, T0_in, omega_in, ecc_in, K_in,
            P_out, T0_out, omega_out, ecc_out, K_out, gamma)
        sig2 = err**2 + sigmaJ**2
        res_chi_sqrd = (model - rv)**2 / sig2
        res_ln = np.log(sig2)
        res = res_ln.sum() + res_chi_sqrd.sum()
    except Exception:
        res = 2e12
    return float(res)


def lmfit_double_kepler(data, inner_params, outer_params,
                        method="differential_evolution", max_nfev=500000):
    """
    Fit a double-Keplerian (hierarchical triple) model.

    Parameters
    ----------
    data : dict-like
        Must contain keys TIME_STAMPS, RADIAL_VELS, ERRORS.
    inner_params : dict
        Inner orbit seeds: {P, K1, omega, ecc, T0} with keys
        'value', 'min', 'max', 'vary' for each.
    outer_params : dict
        Outer orbit seeds: same structure.
    method : str
        lmfit minimization method.
    max_nfev : int
        Maximum function evaluations.

    Returns
    -------
    lmfit.MinimizerResult
    """
    sys.setrecursionlimit(int(1e6))

    hjds = np.array(data[TIME_STAMPS])
    v1s = np.array(data[RADIAL_VELS])
    errv1s = np.abs(data[ERRORS])

    params = lmfit.Parameters()

    # Inner orbit
    for name, cfg in [(_P_IN, inner_params["Period"]),
                      (_K_IN, inner_params["K1"]),
                      (_OMEGA_IN, inner_params["omega"]),
                      (_ECC_IN, inner_params["ecc"]),
                      (_T0_IN, inner_params["T0"])]:
        params.add(name, value=cfg["value"], min=cfg["min"],
                   max=cfg["max"], vary=cfg["vary"])

    # Outer orbit
    for name, cfg in [(_P_OUT, outer_params["Period"]),
                      (_K_OUT, outer_params["K1"]),
                      (_OMEGA_OUT, outer_params["omega"]),
                      (_ECC_OUT, outer_params["ecc"]),
                      (_T0_OUT, outer_params["T0"])]:
        params.add(name, value=cfg["value"], min=cfg["min"],
                   max=cfg["max"], vary=cfg["vary"])

    # Shared
    params.add(GAMMA, value=inner_params["gamma"]["value"],
               min=inner_params["gamma"]["min"],
               max=inner_params["gamma"]["max"], vary=True)
    params.add(LN_SIGMA_JITTER, value=inner_params["ln_sigmaJ"]["value"],
               min=inner_params["ln_sigmaJ"]["min"],
               max=inner_params["ln_sigmaJ"]["max"], vary=True)

    mini = lmfit.Minimizer(
        chisqr_double_kepler_with_jitter, params,
        fcn_kws={TIME_STAMPS: hjds, RADIAL_VELS: v1s, ERRORS: errv1s}
    )
    result = mini.minimize(method=method, max_nfev=max_nfev)
    return result
