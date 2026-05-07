"""
Canonical Kepler equation solvers and radial-velocity model functions.

This module provides the single source of truth for:
  - Solving Kepler's equation  M = E - e*sin(E)
  - Computing true anomalies from timestamps and orbital elements
  - Evaluating the single-lined spectroscopic binary RV model

Two solver implementations are available:
  - kepler_iterative: robust for broad parameter searches (lmfit)
  - kepler_newton:    fast Newton-Raphson for well-conditioned inputs (MCMC)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Kepler equation solvers
# ---------------------------------------------------------------------------

def kepler_iterative(M, ecc, tol=1e-7, max_resets=30, max_iter_per_reset=10000):
    """
    Solve Kepler's equation iteratively (Halley-like fixed-point iteration).

    Robust for broad parameter searches where initial guesses may be poor.
    Falls back to random restarts if convergence stalls.

    Parameters
    ----------
    M : array_like
        Mean anomalies (radians).
    ecc : float
        Orbital eccentricity (0 <= ecc < 1).
    tol : float
        Convergence tolerance on |E_{n+1} - E_n|.
    max_resets : int
        Maximum number of random restarts before returning NaN.
    max_iter_per_reset : int
        Maximum iterations per restart attempt.

    Returns
    -------
    E : ndarray
        Eccentric anomalies (radians).
    """
    M = np.asarray(M, dtype=float)
    E = np.full_like(M, np.pi)
    counter = 0
    reset_count = 0

    while True:
        if counter > max_iter_per_reset:
            E = np.random.rand(len(M)) * np.pi
            counter = 0
            reset_count += 1
            if reset_count > max_resets:
                return np.full_like(M, np.nan)

        E_new = (M - ecc * (E * np.cos(E) - np.sin(E))) / (1.0 - ecc * np.cos(E))
        if np.all(np.abs(E_new - E) < tol):
            return E_new
        E = E_new
        counter += 1


def kepler_newton(M, ecc, tol=1e-10, maxiter=100):
    """
    Solve Kepler's equation via Newton-Raphson iteration.

    Fast and accurate when the initial guess (E0 = M) is reasonable,
    which is the case inside MCMC samplers where parameters are
    constrained by priors.

    Parameters
    ----------
    M : array_like
        Mean anomalies (radians).
    ecc : float
        Orbital eccentricity (0 <= ecc < 1).
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    E : ndarray
        Eccentric anomalies (radians).
    """
    M = np.asarray(M, dtype=float)
    E = M.copy()
    for _ in range(maxiter):
        f = E - ecc * np.sin(E) - M
        fprime = 1.0 - ecc * np.cos(E)
        dE = -f / fprime
        E = E + dE
        if np.all(np.abs(dE) < tol):
            break
    return E


# ---------------------------------------------------------------------------
# True anomaly computation
# ---------------------------------------------------------------------------

def true_anomaly_from_E(E, ecc):
    """
    Convert eccentric anomaly E to true anomaly nu.

    Parameters
    ----------
    E : array_like
        Eccentric anomaly (radians).
    ecc : float
        Orbital eccentricity.

    Returns
    -------
    nu : ndarray
        True anomaly (radians).
    """
    eccfac = np.sqrt((1.0 + ecc) / (1.0 - ecc))
    return 2.0 * np.arctan(eccfac * np.tan(0.5 * E))


def true_anomaly(t, P, T0, ecc, solver="iterative"):
    """
    Compute true anomalies from timestamps and orbital elements.

    Parameters
    ----------
    t : array_like
        Observation timestamps (e.g. MJD).
    P : float
        Orbital period (same units as t).
    T0 : float
        Time of periastron passage (same units as t).
    ecc : float
        Orbital eccentricity.
    solver : str
        Which Kepler solver to use: "iterative" (robust) or "newton" (fast).

    Returns
    -------
    nu : ndarray
        True anomalies (radians).
    """
    t = np.asarray(t, dtype=float)
    phases = (t - T0) / P - ((t - T0) / P).astype(int)
    M = 2.0 * np.pi * phases

    if solver == "newton":
        E = kepler_newton(M, ecc)
    else:
        E = kepler_iterative(M, ecc)

    return true_anomaly_from_E(E, ecc)


# Backward-compatible aliases used by existing code
def nus1(hjds, P, T0, ecc):
    """Backward-compatible alias for true_anomaly (iterative solver)."""
    return true_anomaly(hjds, P, T0, ecc, solver="iterative")


def Kepler(E_init, M, ecc):
    """Backward-compatible alias for kepler_iterative."""
    return kepler_iterative(M, ecc)


# ---------------------------------------------------------------------------
# Radial velocity model
# ---------------------------------------------------------------------------

def rv_model(nu, gamma, K1, omega, ecc):
    """
    Compute radial velocity for a single-lined spectroscopic binary.

    Parameters
    ----------
    nu : array_like
        True anomaly (radians).
    gamma : float
        Systemic (centre-of-mass) velocity [km/s].
    K1 : float
        RV semi-amplitude of the primary [km/s].
    omega : float
        Argument of periastron (radians).
    ecc : float
        Orbital eccentricity.

    Returns
    -------
    rv : ndarray
        Radial velocity [km/s].
    """
    return gamma + K1 * (np.cos(omega + nu) + ecc * np.cos(omega))


# Backward-compatible alias
v1mod = rv_model


def rv_model_from_times(t, P, T0, omega, ecc, K1, gamma, solver="iterative"):
    """
    Compute radial velocity directly from timestamps.

    Convenience function combining true_anomaly + rv_model.
    """
    nu = true_anomaly(t, P, T0, ecc, solver=solver)
    return rv_model(nu, gamma, K1, omega, ecc)


def rv_double_kepler_from_times(t, P_in, T0_in, omega_in, ecc_in, K1_in,
                                 P_out, T0_out, omega_out, ecc_out, K1_out,
                                 gamma, solver="iterative"):
    """
    Compute radial velocity for a hierarchical triple (double Keplerian).

    RV(t) = gamma
          + K1_in  [cos(omega_in  + nu_in(t))  + ecc_in  cos(omega_in)]
          + K1_out [cos(omega_out + nu_out(t)) + ecc_out cos(omega_out)]

    Parameters
    ----------
    t : array_like
        Observation timestamps (e.g. MJD).
    P_in, T0_in, omega_in, ecc_in, K1_in : float
        Inner-orbit Keplerian elements.
    P_out, T0_out, omega_out, ecc_out, K1_out : float
        Outer-orbit Keplerian elements.
    gamma : float
        True systemic velocity [km/s].
    solver : str
        Kepler equation solver ("iterative" or "newton").

    Returns
    -------
    rv : ndarray
        Radial velocity [km/s].
    """
    nu_in = true_anomaly(t, P_in, T0_in, ecc_in, solver=solver)
    nu_out = true_anomaly(t, P_out, T0_out, ecc_out, solver=solver)
    return (gamma
            + K1_in * (np.cos(omega_in + nu_in) + ecc_in * np.cos(omega_in))
            + K1_out * (np.cos(omega_out + nu_out) + ecc_out * np.cos(omega_out)))
