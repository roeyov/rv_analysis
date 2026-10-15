"""
sb2_pipeline.mcmc_model — log-prior / log-likelihood / log-probability for the
joint SB2 model (eccentric 9-parameter and circular 7-parameter thetas).

Theta orderings (see sb2_pipeline.constants):
    ecc : [P, T0, omega, e, K1, K2, gamma, log_sj1, log_sj2]
    circ: [P, T0, K1, K2, gamma, log_sj1, log_sj2]
The first four eccentric slots match the SB1 models (mcmc.models.log_prior),
keeping omega at column 2 for the corner-plot degree conversion and P/T0
first for phase folding. Priors mirror the SB1 ones: box+Gaussian on P,
T0 box +-dT0, 0<=e<0.99, 0<K<2*K_center per component, omega in
omega_center+-pi, log_sj ~ N(0,3) each, gamma unbounded. No K1<K2 constraint
— that would truncate the posterior exactly when K1~K2; label swaps are
tracked downstream via ecc_swap_fraction instead.
"""

import numpy as np

from orbital.kepler import true_anomaly
from simulations.common import RV12

Max_e = 0.99


def sb2_rv_model_mcmc(t, P, T0, omega, e, K1, K2, gamma):
    """(rv1, rv2) model with the fast Newton Kepler solver (MCMC hot path)."""
    nu = true_anomaly(t, P, T0, e, solver="newton")
    return RV12(nu, gamma, K1, K2, omega, e)


# ======================================================================
# Eccentric model
# ======================================================================

def log_prior_sb2(theta, P_center, T0_center, dT0_days, K1_center, K2_center,
                  omega_center, dP_frac=0.01, sigma_P_frac=0.002):
    P, T0, omega, e, K1, K2, gamma, log_sj1, log_sj2 = theta

    if not (P_center * (1 - dP_frac) < P < P_center * (1 + dP_frac)):
        return -np.inf
    sigma_P = sigma_P_frac * P_center
    lp = -0.5 * ((P - P_center) / sigma_P) ** 2

    if not (T0_center - dT0_days < T0 < T0_center + dT0_days):
        return -np.inf
    if not (0.0 <= e < Max_e):
        return -np.inf
    if not (0 < K1 < K1_center * 2):
        return -np.inf
    if not (0 < K2 < K2_center * 2):
        return -np.inf
    if not (omega_center - np.pi <= omega <= omega_center + np.pi):
        return -np.inf

    lp += -0.5 * (log_sj1 / 3.0) ** 2
    lp += -0.5 * (log_sj2 / 3.0) ** 2
    return lp


def log_likelihood_sb2(theta, t1, rv1, err1, t2, rv2, err2):
    P, T0, omega, e, K1, K2, gamma, log_sj1, log_sj2 = theta
    var1 = err1 ** 2 + np.exp(log_sj1) ** 2
    var2 = err2 ** 2 + np.exp(log_sj2) ** 2

    v1, _ = sb2_rv_model_mcmc(t1, P, T0, omega, e, K1, K2, gamma)
    _, v2 = sb2_rv_model_mcmc(t2, P, T0, omega, e, K1, K2, gamma)
    return -0.5 * (np.sum((rv1 - v1) ** 2 / var1 + np.log(2 * np.pi * var1))
                   + np.sum((rv2 - v2) ** 2 / var2 + np.log(2 * np.pi * var2)))


def log_probability_sb2(theta, t1, rv1, err1, t2, rv2, err2,
                        P_center, T0_center, K1_center, K2_center,
                        omega_center, dP_frac=0.01, dT0_days=2.0,
                        sigma_P_frac=0.002):
    lp = log_prior_sb2(theta, P_center, T0_center, dT0_days,
                       K1_center, K2_center, omega_center,
                       dP_frac=dP_frac, sigma_P_frac=sigma_P_frac)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_sb2(theta, t1, rv1, err1, t2, rv2, err2)


# ======================================================================
# Circular model (e = 0, omega = pi/2 — dropped from theta entirely)
# ======================================================================

# ---- shared-jitter variant (theta 8): one log_sj for both components ----

def log_prior_sb2_sharedJ(theta, P_center, T0_center, dT0_days,
                          K1_center, K2_center, omega_center,
                          dP_frac=0.01, sigma_P_frac=0.002):
    P, T0, omega, e, K1, K2, gamma, log_sj = theta

    if not (P_center * (1 - dP_frac) < P < P_center * (1 + dP_frac)):
        return -np.inf
    sigma_P = sigma_P_frac * P_center
    lp = -0.5 * ((P - P_center) / sigma_P) ** 2

    if not (T0_center - dT0_days < T0 < T0_center + dT0_days):
        return -np.inf
    if not (0.0 <= e < Max_e):
        return -np.inf
    if not (0 < K1 < K1_center * 2):
        return -np.inf
    if not (0 < K2 < K2_center * 2):
        return -np.inf
    if not (omega_center - np.pi <= omega <= omega_center + np.pi):
        return -np.inf

    lp += -0.5 * (log_sj / 3.0) ** 2
    return lp


def log_likelihood_sb2_sharedJ(theta, t1, rv1, err1, t2, rv2, err2):
    P, T0, omega, e, K1, K2, gamma, log_sj = theta
    sj2 = np.exp(log_sj) ** 2
    var1 = err1 ** 2 + sj2
    var2 = err2 ** 2 + sj2

    v1, _ = sb2_rv_model_mcmc(t1, P, T0, omega, e, K1, K2, gamma)
    _, v2 = sb2_rv_model_mcmc(t2, P, T0, omega, e, K1, K2, gamma)
    return -0.5 * (np.sum((rv1 - v1) ** 2 / var1 + np.log(2 * np.pi * var1))
                   + np.sum((rv2 - v2) ** 2 / var2 + np.log(2 * np.pi * var2)))


def log_probability_sb2_sharedJ(theta, t1, rv1, err1, t2, rv2, err2,
                                P_center, T0_center, K1_center, K2_center,
                                omega_center, dP_frac=0.01, dT0_days=2.0,
                                sigma_P_frac=0.002):
    lp = log_prior_sb2_sharedJ(theta, P_center, T0_center, dT0_days,
                               K1_center, K2_center, omega_center,
                               dP_frac=dP_frac, sigma_P_frac=sigma_P_frac)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_sb2_sharedJ(theta, t1, rv1, err1, t2, rv2, err2)


def log_prior_sb2_circ(theta, P_center, T0_center, dT0_days,
                       K1_center, K2_center,
                       dP_frac=0.01, sigma_P_frac=0.002):
    P, T0, K1, K2, gamma, log_sj1, log_sj2 = theta

    if not (P_center * (1 - dP_frac) < P < P_center * (1 + dP_frac)):
        return -np.inf
    sigma_P = sigma_P_frac * P_center
    lp = -0.5 * ((P - P_center) / sigma_P) ** 2

    if not (T0_center - dT0_days < T0 < T0_center + dT0_days):
        return -np.inf
    if not (0 < K1 < K1_center * 2):
        return -np.inf
    if not (0 < K2 < K2_center * 2):
        return -np.inf

    lp += -0.5 * (log_sj1 / 3.0) ** 2
    lp += -0.5 * (log_sj2 / 3.0) ** 2
    return lp


def log_likelihood_sb2_circ(theta, t1, rv1, err1, t2, rv2, err2):
    P, T0, K1, K2, gamma, log_sj1, log_sj2 = theta
    var1 = err1 ** 2 + np.exp(log_sj1) ** 2
    var2 = err2 ** 2 + np.exp(log_sj2) ** 2

    v1, _ = sb2_rv_model_mcmc(t1, P, T0, np.pi / 2, 0.0, K1, K2, gamma)
    _, v2 = sb2_rv_model_mcmc(t2, P, T0, np.pi / 2, 0.0, K1, K2, gamma)
    return -0.5 * (np.sum((rv1 - v1) ** 2 / var1 + np.log(2 * np.pi * var1))
                   + np.sum((rv2 - v2) ** 2 / var2 + np.log(2 * np.pi * var2)))


def log_probability_sb2_circ(theta, t1, rv1, err1, t2, rv2, err2,
                             P_center, T0_center, K1_center, K2_center,
                             dP_frac=0.01, dT0_days=2.0, sigma_P_frac=0.002):
    lp = log_prior_sb2_circ(theta, P_center, T0_center, dT0_days,
                            K1_center, K2_center,
                            dP_frac=dP_frac, sigma_P_frac=sigma_P_frac)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_sb2_circ(theta, t1, rv1, err1, t2, rv2, err2)


def log_prior_sb2_circ_sharedJ(theta, P_center, T0_center, dT0_days,
                               K1_center, K2_center,
                               dP_frac=0.01, sigma_P_frac=0.002):
    P, T0, K1, K2, gamma, log_sj = theta

    if not (P_center * (1 - dP_frac) < P < P_center * (1 + dP_frac)):
        return -np.inf
    sigma_P = sigma_P_frac * P_center
    lp = -0.5 * ((P - P_center) / sigma_P) ** 2

    if not (T0_center - dT0_days < T0 < T0_center + dT0_days):
        return -np.inf
    if not (0 < K1 < K1_center * 2):
        return -np.inf
    if not (0 < K2 < K2_center * 2):
        return -np.inf

    lp += -0.5 * (log_sj / 3.0) ** 2
    return lp


def log_likelihood_sb2_circ_sharedJ(theta, t1, rv1, err1, t2, rv2, err2):
    P, T0, K1, K2, gamma, log_sj = theta
    sj2 = np.exp(log_sj) ** 2
    var1 = err1 ** 2 + sj2
    var2 = err2 ** 2 + sj2

    v1, _ = sb2_rv_model_mcmc(t1, P, T0, np.pi / 2, 0.0, K1, K2, gamma)
    _, v2 = sb2_rv_model_mcmc(t2, P, T0, np.pi / 2, 0.0, K1, K2, gamma)
    return -0.5 * (np.sum((rv1 - v1) ** 2 / var1 + np.log(2 * np.pi * var1))
                   + np.sum((rv2 - v2) ** 2 / var2 + np.log(2 * np.pi * var2)))


def log_probability_sb2_circ_sharedJ(theta, t1, rv1, err1, t2, rv2, err2,
                                     P_center, T0_center, K1_center,
                                     K2_center, dP_frac=0.01, dT0_days=2.0,
                                     sigma_P_frac=0.002):
    lp = log_prior_sb2_circ_sharedJ(theta, P_center, T0_center, dT0_days,
                                    K1_center, K2_center,
                                    dP_frac=dP_frac,
                                    sigma_P_frac=sigma_P_frac)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_sb2_circ_sharedJ(theta, t1, rv1, err1,
                                                t2, rv2, err2)
