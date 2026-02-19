"""
mcmc.models — Log-prior, log-likelihood, log-probability for eccentric, circular, and null models.
"""

import numpy as np

from orbital.kepler import true_anomaly

# Maximum eccentricity to probe
Max_e = 0.99


def rv_model(t, P, T0, omega, e, K1, gamma):
    """RV model from timestamps (wraps orbital.kepler functions)."""
    nu = true_anomaly(t, P, T0, e, solver="newton")
    return gamma + K1 * (np.cos(nu + omega) + e * np.cos(omega))


# ======================================================================
# Eccentric model
# ======================================================================

def log_prior(theta, P_center, T0_center, dT0_days, K1_center, omega_center=np.pi,
              dP_frac=0.01, sigma_P_frac=0.002, add_jitter=False):
    """
    If add_jitter=False:
        theta = [P, T0, omega, e, K1, gamma]
    If add_jitter=True:
        theta = [P, T0, omega, e, K1, gamma, log_sj]
        with a weak prior on log_sj.
    """
    if add_jitter:
        P, T0, omega, e, K1, gamma, log_sj = theta
    else:
        P, T0, omega, e, K1, gamma = theta

    # Period narrow prior (Gaussian around P_center)
    Pmin = P_center * (1 - dP_frac)
    Pmax = P_center * (1 + dP_frac)
    if not (Pmin < P < Pmax):
        return -np.inf
    sigma_P = sigma_P_frac * P_center
    lp = -0.5 * ((P - P_center) / sigma_P)**2

    # Simple box priors for others
    if not (T0_center - dT0_days < T0 < T0_center + dT0_days):
        return -np.inf
    if not (0.0 <= e < Max_e):
        return -np.inf
    if not (0 < K1 <  K1_center*2):
        return -np.inf
    if not (omega_center-np.pi <= omega <= omega_center + np.pi):
        return -np.inf

    # Weak prior on log sigma_jit if enabled (Normal(0, 3))
    if add_jitter:
        lp += -0.5 * (log_sj / 3.0)**2

    return lp


def log_likelihood(theta, t, rv, rv_err, add_jitter=False):
    if add_jitter:
        P, T0, omega, e, K1, gamma, log_sj = theta
        s_jit = np.exp(log_sj)
        var = rv_err**2 + s_jit**2
    else:
        P, T0, omega, e, K1, gamma = theta
        var = rv_err**2

    model_rv = rv_model(t, P, T0, omega, e, K1, gamma)
    return -0.5 * np.sum((rv - model_rv)**2 / var + np.log(2*np.pi*var))


def log_probability(theta, t, rv, rv_err,
                    P_center, T0_center, K1_center, omega_center=np.pi,
                    dP_frac=0.1, dT0_days=2.0, add_jitter=False):
    """
    Wrapper that combines prior and likelihood and passes add_jitter through.
    """
    lp = log_prior(theta, P_center, T0_center, dT0_days, K1_center,omega_center=omega_center,
                   dP_frac=dP_frac, add_jitter=add_jitter)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, rv, rv_err, add_jitter=add_jitter)


# ======================================================================
# Circular model
# ======================================================================

def log_prior_circ(theta, P_center, T0_center, dT0_days, K1_center,
                   dP_frac=0.01, sigma_P_frac=0.002, add_jitter=False):
    """
    If add_jitter=False:
        theta = [P, T0, K1, gamma]
    If add_jitter=True:
        theta = [P, T0, K1, gamma, log_sj]
    """
    if add_jitter:
        P, T0, K1, gamma, log_sj = theta
    else:
        P, T0, K1, gamma = theta

    Pmin = P_center * (1 - dP_frac)
    Pmax = P_center * (1 + dP_frac)
    if not (Pmin < P < Pmax):
        return -np.inf
    sigma_P = sigma_P_frac * P_center
    lp = -0.5 * ((P - P_center) / sigma_P)**2

    if not (T0_center - dT0_days < T0 < T0_center + dT0_days):
        return -np.inf
    if not (0 < K1 <  K1_center*2):
        return -np.inf

    if add_jitter:
        lp += -0.5 * (log_sj / 3.0)**2

    return lp


def log_likelihood_circ(theta, t, rv, rv_err, add_jitter=False):
    if add_jitter:
        P, T0, K1, gamma, log_sj = theta
        s_jit = np.exp(log_sj)
        var = rv_err**2 + s_jit**2
    else:
        P, T0, K1, gamma = theta
        var = rv_err**2

    model_rv = rv_model(t, P, T0, np.pi/2, 0.0, K1, gamma)
    return -0.5 * np.sum((rv - model_rv)**2 / var + np.log(2*np.pi*var))


def log_probability_circ(theta, t, rv, rv_err,
                         P_center, T0_center, K1_center,
                         dP_frac=0.01, dT0_days=2.0, add_jitter=False):
    lp = log_prior_circ(theta, P_center, T0_center, dT0_days, K1_center,
                        dP_frac=dP_frac, add_jitter=add_jitter)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_circ(theta, t, rv, rv_err, add_jitter=add_jitter)


# ======================================================================
# Null model (constant RV)
# ======================================================================

def log_prior_null(theta, add_jitter=False):
    """
    If add_jitter=False:
        theta = [P, T0, K1, gamma]
    If add_jitter=True:
        theta = [P, T0, K1, gamma, log_sj]
    """
    if add_jitter:
        gamma, log_sj = theta
    else:
        gamma = theta
    lp = 0
    if add_jitter:
        lp += -0.5 * (log_sj / 3.0)**2
    return lp


def log_likelihood_null(theta, t, rv, rv_err, add_jitter=False):
    if add_jitter:
        gamma, log_sj = theta
        s_jit = np.exp(log_sj)
        var = rv_err**2 + s_jit**2
    else:
        gamma = theta
        var = rv_err**2

    model_rv = np.ones(rv.shape)*gamma
    return -0.5 * np.sum((rv - model_rv)**2 / var + np.log(2*np.pi*var))


def log_probability_null(theta, t, rv, rv_err,  add_jitter=False):
    lp = log_prior_null(theta, add_jitter=add_jitter)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_null(theta, t, rv, rv_err, add_jitter=add_jitter)
