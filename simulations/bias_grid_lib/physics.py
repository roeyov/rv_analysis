"""Keplerian physics, Roche-lobe check, and power-law sampler.

Includes the numba-accelerated single and batched Kepler solvers used by
both the per-injection path and the vectorized RV-threshold worker.
"""

import numpy as np
from numba import njit

from simulations.bias_grid_lib.constants import (
    G_CGS, MSUN, RSUN, DAY, KM, TWOPI,
)
from simulations.bias_grid_lib.logging_utils import logger


# ---------------------------------------------------------------------------
# Numba-accelerated Kepler solver & RV model
# ---------------------------------------------------------------------------

@njit(cache=True)
def kepler_E(M, e, tol=1e-10, maxiter=100):
    """Solve Kepler's equation M = E - e sin(E) via Newton-Raphson."""
    E = np.copy(M)
    for i in range(len(M)):
        for _ in range(maxiter):
            dE = -(E[i] - e * np.sin(E[i]) - M[i]) / (1.0 - e * np.cos(E[i]))
            E[i] += dE
            if np.abs(dE) < tol:
                break
    return E


@njit(cache=True)
def rv_model_jit(t, P, T0, omega, e, K1, gamma):
    """Compute Keplerian RV curve (numba-accelerated)."""
    phase = ((t - T0) / P) % 1.0
    M = TWOPI * phase
    if e < 1e-8:
        return gamma + K1 * np.cos(M + omega)
    E = kepler_E(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                           np.sqrt(1.0 - e) * np.cos(E / 2.0))
    return gamma + K1 * (np.cos(nu + omega) + e * np.cos(omega))


@njit(cache=True)
def kepler_E_batch(M_2d, e_1d, tol=1e-10, maxiter=100):
    """Solve Kepler's equation for a batch of systems.

    Parameters
    ----------
    M_2d : ndarray, shape (n_sys, n_epochs)
        Mean anomalies.
    e_1d : ndarray, shape (n_sys,)
        Eccentricities (one per system).

    Returns
    -------
    E : ndarray, shape (n_sys, n_epochs)
        Eccentric anomalies.
    """
    n_sys, n_ep = M_2d.shape
    E = np.copy(M_2d)
    for s in range(n_sys):
        ecc = e_1d[s]
        if ecc < 1e-8:
            continue  # circular: E = M (already copied)
        for i in range(n_ep):
            for _ in range(maxiter):
                dE = -(E[s, i] - ecc * np.sin(E[s, i]) - M_2d[s, i]) / \
                     (1.0 - ecc * np.cos(E[s, i]))
                E[s, i] += dE
                if abs(dE) < tol:
                    break
    return E


def compute_K1_batch(P_days, e, q, M1_Msun, incl):
    """Vectorized K1 [km/s] computation. All inputs shape (n,)."""
    P_s = P_days * DAY
    M1 = M1_Msun * MSUN
    M2 = q * M1
    a = (G_CGS * (M1 + M2) * P_s ** 2 / (4 * np.pi ** 2)) ** (1.0 / 3.0)
    a1 = a * M2 / (M1 + M2)
    return TWOPI * a1 * np.sin(incl) / (P_s * np.sqrt(1 - e ** 2)) / KM


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def compute_K1(P_days, e, q, M1_Msun, incl):
    """Compute K1 [km/s] from physical parameters."""
    P_s = P_days * DAY
    M1 = M1_Msun * MSUN
    M2 = q * M1
    a = (G_CGS * (M1 + M2) * P_s ** 2 / (4 * np.pi ** 2)) ** (1.0 / 3.0)
    a1 = a * M2 / (M1 + M2)
    K1 = TWOPI * a1 * np.sin(incl) / (P_s * np.sqrt(1 - e ** 2)) / KM
    logger.debug("compute_K1: P=%.2f q=%.3f M1=%.1f incl=%.2f → K1=%.2f",
                 P_days, q, M1_Msun, incl, K1)
    return K1


def roche_lobe_check(P_days, e, q, M1_Msun, R1_Rsun):
    """Return True if the primary fits inside its Roche lobe at periastron."""
    if q <= 0 or P_days <= 0:
        return False
    P_s = P_days * DAY
    M1 = M1_Msun * MSUN
    M2 = q * M1_Msun * MSUN
    a = (G_CGS * (M1 + M2) * P_s ** 2 / (4 * np.pi ** 2)) ** (1.0 / 3.0)
    r_peri = a * (1.0 - e)
    q_don = 1.0 / q
    q23 = q_don ** (2.0 / 3.0)
    rl = 0.49 * q23 / (0.6 * q23 + np.log(1.0 + q_don ** (1.0 / 3.0)))
    result = (R1_Rsun * RSUN) < rl * r_peri
    logger.debug("roche_lobe_check: P=%.2f e=%.3f q=%.3f M1=%.1f R1=%.1f → %s",
                 P_days, e, q, M1_Msun, R1_Rsun, result)
    return result


def powerlaw_draw(n, alpha, xmin, xmax, rng):
    """Draw n samples from p(x) ∝ x^alpha on [xmin, xmax]."""
    a = alpha + 1.0
    u = rng.uniform(0, 1, n)
    xmin = max(float(xmin), 1e-10)
    xmax = max(float(xmax), xmin + 1e-10)
    if abs(a) < 1e-8:
        return np.asarray(xmin * (xmax / xmin) ** u, dtype=np.float64)
    base = xmin ** a + u * (xmax ** a - xmin ** a)
    base = np.maximum(np.real(base), 1e-300)
    samples = np.real(np.exp(np.log(base) / a)).astype(np.float64)
    logger.debug("powerlaw_draw: n=%d alpha=%.2f [%.3f,%.3f] → [%.3f,%.3f]",
                 n, alpha, xmin, xmax, samples.min(), samples.max())
    return samples
