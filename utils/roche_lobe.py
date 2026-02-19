# utils/period_min.py
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from typing import Dict

# --- constants (kept identical to your script) ---
C1 = 9651121.605
C2 = 365.2563356
RSUN2AU = 0.00465047

def roche_f(q: float) -> float:
    return 0.49 * q**(2/3) / (0.6 * q**(2/3) + np.log1p(q**(1/3)))

def P_massfunc(M1: float, q: float, K1: float, e: float, sini3: float) -> float:
    # returns P (days)
    return C1 * (M1 * sini3) / (q * (1 + q)**2) * K1**-3 * (1 - e**2)**(-1.5)

def P_roche(M1: float, R_AU: float, q: float, e_eff: float, alpha: float) -> float:
    # returns P (days); e_eff = (1-e) for periastron, (1+e) for apastron
    f_q = roche_f(q)
    return C2 * R_AU**1.5 / ((alpha * f_q)**1.5 * e_eff**1.5 * np.sqrt(M1 * (1 + 1/q)))

def _find_root(residual, q_lo=1e-5, q_hi=2e3) -> float:
    try:
        return brentq(residual, q_lo, q_hi)
    except ValueError:
        for hi in (5e3, 1e4):
            try:
                return brentq(residual, q_lo, hi)
            except ValueError:
                pass
        raise

def _solve_Pmin_peri(M1: float, R_Rsun: float, K1: float, e: float, alpha: float, sini3: float):
    R_AU = R_Rsun * RSUN2AU
    residual = lambda q: P_roche(M1, R_AU, q, 1 - e, alpha) - P_massfunc(M1, q, K1, e, sini3)
    q_root = _find_root(residual)
    return P_massfunc(M1, q_root, K1, e, sini3), q_root

def _solve_Pmin_apa(M1: float, R_Rsun: float, K1: float, e: float, alpha: float, sini3: float):
    R_AU = R_Rsun * RSUN2AU
    residual = lambda q: P_roche(M1, R_AU, q, 1 + e, alpha) - P_massfunc(M1, q, K1, e, sini3)
    q_root = _find_root(residual)
    return P_massfunc(M1, q_root, K1, e, sini3), q_root

def _best_case_peri(M1: float, R_Rsun: float, e: float, alpha: float) -> float:
    f_inf = 0.49 / 0.6
    R_AU = R_Rsun * RSUN2AU
    return C2 * R_AU**1.5 / ((alpha * f_inf)**1.5 * (1 - e)**1.5 * np.sqrt(M1))

def compute_min_period_row(
    mass_row, *,
    P_fit: float,
    K1: float,
    e: float,
    alpha_peri: float = 1.2,
    alpha_apa: float = 1.0,
    i_deg: float = 90.0,
) -> Dict[str, float]:
    """
    Compute minimum-period constraints vs the Roche lobe, using one row from mass_bloem.csv.

    Parameters
    ----------
    mass_row : pandas.Series or mapping
        Must contain: Mspec, Mspec_er_plus, Mspec_er_minus, R_star, R_star_er_plus, R_star_er_minus
    P_fit : float
        The fitted orbital period (days) to compare against.
    K1 : float
        Fitted K1 (same units used in your original formula).
    e : float
        Eccentricity.
    alpha_peri : float
        Roche scale factor at periastron (default 1.2).
    alpha_apa : float
        Roche scale factor at apastron (default 1.0).
    i_deg : float
        Inclination in degrees; default 90° (edge-on).

    Returns
    -------
    dict with keys:
      M1, Rstar, mass_flag_peri, mass_flag_apa,
      Pmin_peri_central, Pmin_peri_upper, Pmin_peri_lower,
      Pmin_apa_central,  Pmin_apa_upper,  Pmin_apa_lower,
      q_at_min_peri, Pmin_peri_best, q_at_min_apa
    """
    # pull nominal values + ±errors
    if mass_row is None:
        return {
            "M1": 0,
            "Rstar": 0,
            "mass_flag_peri": 0,
            "mass_flag_apa": 0,
            "Pmin_peri_central": 0,
            "Pmin_peri_upper":   0,
            "Pmin_peri_lower":   0,
            "Pmin_apa_central":  0,
            "Pmin_apa_upper":    0,
            "Pmin_apa_lower":    0,
            "q_at_min_peri": 0,
            "Pmin_peri_best": 0,
            "q_at_min_apa": 0,
        }
    mspec    = float(mass_row["Mspec"])
    dM_plus  = float(mass_row["Mspec_er_plus"])
    dM_minus = float(mass_row["Mspec_er_minus"])
    rspec    = float(mass_row["R_star"])
    dR_plus  = float(mass_row["R_star_er_plus"])
    dR_minus = float(mass_row["R_star_er_minus"])

    M1_array = np.clip(np.array([mspec - dM_minus, mspec, mspec + dM_plus], dtype=float), 0.1, None)
    R_array  = np.clip(np.array([rspec - dR_minus, rspec, rspec + dR_plus], dtype=float), 0.1, None)

    sini3 = float(np.sin(np.deg2rad(i_deg))**3)

    # Periastron grid
    P_peri_grid = np.empty((3, 3), dtype=float)
    Q_peri_grid = np.empty((3, 3), dtype=float)
    for iM, M1 in enumerate(M1_array):
        for iR, R in enumerate(R_array):
            P_peri_grid[iM, iR], Q_peri_grid[iM, iR] = _solve_Pmin_peri(M1, R, K1, e, alpha_peri, sini3)

    P_peri_central = float(P_peri_grid[1, 1])
    P_peri_upper   = float(np.max(P_peri_grid))
    P_peri_lower   = float(np.min(P_peri_grid))
    q_peri         = float(Q_peri_grid[1, 1])

    # Apastron grid
    P_apa_grid = np.empty((3, 3), dtype=float)
    Q_apa_grid = np.empty((3, 3), dtype=float)
    for iM, M1 in enumerate(M1_array):
        for iR, R in enumerate(R_array):
            P_apa_grid[iM, iR], Q_apa_grid[iM, iR] = _solve_Pmin_apa(M1, R, K1, e, alpha_apa, sini3)

    P_apa_central = float(P_apa_grid[1, 1])
    P_apa_upper   = float(np.max(P_apa_grid))
    P_apa_lower   = float(np.min(P_apa_grid))
    q_apa         = float(Q_apa_grid[1, 1])

    # Best-case periastron (q → ∞)
    Pmin_peri_best = float(_best_case_peri(M1_array[1], R_array[1], e, alpha_peri))

    # Flags relative to the fitted period
    mass_flag_peri = int(P_peri_lower > P_fit)
    mass_flag_apa  = int(P_apa_lower  > P_fit)

    return {
        "M1": mspec,
        "Rstar": rspec,
        "mass_flag_peri": mass_flag_peri,
        "mass_flag_apa": mass_flag_apa,
        "Pmin_peri_central": P_peri_central,
        "Pmin_peri_upper":   P_peri_upper,
        "Pmin_peri_lower":   P_peri_lower,
        "Pmin_apa_central":  P_apa_central,
        "Pmin_apa_upper":    P_apa_upper,
        "Pmin_apa_lower":    P_apa_lower,
        "q_at_min_peri": q_peri,
        "Pmin_peri_best": Pmin_peri_best,
        "q_at_min_apa": q_apa,
    }

