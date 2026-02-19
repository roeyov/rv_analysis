"""
simulations.create_binary_simulations — Generate binary-star RV simulations
using explicit BLOeM MJD timing arrays, outputting pipeline-compatible CSVs.

Mirrors createSingleSimulations.py but adds Keplerian binary orbital signals.

Output format matches Stage 2 pipeline input:
    filename: SIMuLaTioN_NNNNNN_CCF_RVs.csv
    columns:  Mean RV, Mean RVsig, MJD, SNR_PPL, SNR

Usage:
    python -m simulations.create_binary_simulations
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from simulations.common import (
    BLOEM_MJD_ARRAYS,
    SIGMA_SHAPE, SIGMA_LOC, SIGMA_SCALE,
    simulate_system_refined,
    sample_gamma,
    RV12, nu_func,
    get_rv_amplitudes,
    uniform_random_sample,
    sine_inclination_sample,
)


# ---------------------------------------------------------------------------
# Default orbital parameter distributions
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "log_period_range": (0, 4),        # log10(P/days), uniform
    "ecc_range": (0, 0.999),           # eccentricity, uniform
    "omega_range": (-np.pi, np.pi),    # argument of periastron [rad], uniform
    "m1_range": (15, 80),              # primary mass [M_sun], uniform
    "q_range": (0, 1),                 # mass ratio, uniform
    "inc_range": (0, np.pi / 2),       # inclination [rad], sin(i) prior
    "mean_gamma": 168,                 # systemic velocity [km/s]
    "std_gamma": 35,                   # systemic velocity scatter [km/s]
}


# ---------------------------------------------------------------------------
# Sampling one binary system's orbital parameters
# ---------------------------------------------------------------------------

def sample_orbital_params(rng=None, params=None):
    """
    Sample a single set of orbital parameters from the prior distributions.

    Parameters
    ----------
    rng : numpy.random.Generator or None
        Random number generator (for reproducibility).
    params : dict or None
        Override default distribution ranges. Keys match DEFAULT_PARAMS.

    Returns
    -------
    dict with keys: t0, period, ecc, omega, k1, k2, gamma, m1, q, inc
    """
    if params is None:
        params = DEFAULT_PARAMS
    if rng is None:
        rng = np.random.default_rng()

    log_p = rng.uniform(*params["log_period_range"])
    period = 10 ** log_p

    ecc = rng.uniform(*params["ecc_range"])
    # Zero eccentricity for very short periods (tidal circularization)
    if period <= 5:
        ecc = 0.0

    omega = rng.uniform(*params["omega_range"])
    m1 = rng.uniform(*params["m1_range"])
    q = rng.uniform(*params["q_range"])

    # Sin(i) prior via rejection sampling (fast for single draws)
    while True:
        inc_candidate = rng.uniform(*params["inc_range"])
        if rng.random() < np.sin(inc_candidate):
            inc = inc_candidate
            break

    # T0 as fraction of period
    t0 = (rng.uniform(0, 1) - 0.5) * period

    # Physical K1, K2 from mass, period, q, ecc, inclination
    k1, k2 = get_rv_amplitudes(m1, period, q, ecc, inc)

    # Systemic velocity
    gamma = norm.rvs(loc=params["mean_gamma"], scale=params["std_gamma"])

    return {
        "t0": float(t0),
        "period": float(period),
        "ecc": float(ecc),
        "omega": float(omega),
        "k1": float(k1),
        "k2": float(k2),
        "gamma": float(gamma),
        "m1": float(m1),
        "q": float(q),
        "inc": float(inc),
    }


# ---------------------------------------------------------------------------
# Core: generate one binary RV observation at explicit MJDs
# ---------------------------------------------------------------------------

def generate_binary_rv_at_mjds(mjds, orbital_params,
                                sigma_shape=SIGMA_SHAPE,
                                sigma_loc=SIGMA_LOC,
                                sigma_scale=SIGMA_SCALE,
                                intra_cv=0.25):
    """
    Compute Keplerian binary RVs at explicit MJD times, then add realistic noise.

    Parameters
    ----------
    mjds : array_like
        Observation times (MJD).
    orbital_params : dict
        Must contain: t0, period, ecc, omega, k1, k2, gamma.
    sigma_shape, sigma_loc, sigma_scale : float
        Log-normal noise parameters.
    intra_cv : float
        Per-observation flutter coefficient of variation.

    Returns
    -------
    rvs : ndarray
        Noisy observed RVs.
    sigmas : ndarray
        Per-observation noise sigmas.
    """
    mjds = np.asarray(mjds, dtype=float)
    t0 = orbital_params["t0"]
    period = orbital_params["period"]
    ecc = orbital_params["ecc"]
    omega = orbital_params["omega"]
    k1 = orbital_params["k1"]
    k2 = orbital_params["k2"]
    gamma = orbital_params["gamma"]

    # Compute orbital phases and true anomalies
    phases = (mjds - t0) / period - ((mjds - t0) / period).astype(int)
    nus = nu_func(phases, ecc)
    if nus is None:
        return None, None

    # Noise-free Keplerian RVs (primary only)
    rv_model = RV12(nus, gamma, k1, k2, omega, ecc)[0]

    # Add realistic noise using the same model as createSingleSimulations
    rvs, sigmas = simulate_system_refined(rv_model, sigma_shape, sigma_loc,
                                           sigma_scale, intra_cv=intra_cv)

    return rvs, sigmas


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_binary_csvs(output_path, mjd_arrays=None, n_per_field=5,
                          params=None, random_seed=None):
    """
    Generate pipeline-compatible CSV files with binary RV simulations.

    Produces two outputs:
      1. Per-system CSV files: ``SIMuLaTioN_NNNNNN_CCF_RVs.csv``
         (pipeline-compatible, columns: Mean RV, Mean RVsig, MJD, SNR_PPL, SNR)
      2. A single truth table: ``orbital_params_truth.csv``
         (one row per system with all sampled orbital parameters)

    Parameters
    ----------
    output_path : str
        Directory to write CSV files.
    mjd_arrays : list of tuple/array or None
        Explicit MJD timing arrays. Default: BLOEM_MJD_ARRAYS.
    n_per_field : int
        Number of binary realizations per MJD array.
    params : dict or None
        Override default orbital parameter distributions.
    random_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with generation statistics and path to truth table.
    """
    if mjd_arrays is None:
        mjd_arrays = BLOEM_MJD_ARRAYS
    if params is None:
        params = DEFAULT_PARAMS

    rng = np.random.default_rng(random_seed)
    os.makedirs(output_path, exist_ok=True)

    total_systems = n_per_field * len(mjd_arrays)
    digits = len(str(total_systems))
    saved = 0
    failed = 0
    truth_rows = []

    for field_idx, mjds in enumerate(mjd_arrays):
        for _ in range(n_per_field):
            # Sample new orbital parameters for each realization
            orb = sample_orbital_params(rng=rng, params=params)

            rvs, sigmas = generate_binary_rv_at_mjds(mjds, orb)

            if rvs is None:
                failed += 1
                continue

            # --- RV data CSV (pipeline-compatible) ---
            df = pd.DataFrame({
                "Mean RV": rvs,
                "Mean RVsig": sigmas,
                "MJD": mjds,
                "SNR_PPL": 100 * np.ones_like(rvs),
                "SNR": 100 * np.ones_like(rvs),
            })

            filename = f"SIMuLaTioN_{saved:0{digits}d}_CCF_RVs.csv"
            df.to_csv(os.path.join(output_path, filename), index=False)

            # --- Collect truth row ---
            truth_rows.append({
                "sim_id": saved,
                "filename": filename,
                "field_idx": field_idx,
                "n_obs": len(mjds),
                "T0": orb["t0"],
                "Period": orb["period"],
                "Eccentricity": orb["ecc"],
                "OMEGA_rad": orb["omega"],
                "OMEGA_deg": np.degrees(orb["omega"]),
                "K1": orb["k1"],
                "K2": orb["k2"],
                "GAMMA": orb["gamma"],
                "Mass1": orb["m1"],
                "MassRatio": orb["q"],
                "Inclination_rad": orb["inc"],
                "Inclination_deg": np.degrees(orb["inc"]),
            })

            saved += 1

    # --- Save truth table ---
    truth_path = os.path.join(output_path, "orbital_params_truth.csv")
    if truth_rows:
        truth_df = pd.DataFrame(truth_rows)
        truth_df.to_csv(truth_path, index=False)

    return {
        "saved": saved,
        "failed": failed,
        "total_attempted": total_systems,
        "truth_table": truth_path,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    N = 5
    output_path = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/simulations_binary/"

    print(f"Generating binary simulations → {output_path}")
    print(f"  Fields: {len(BLOEM_MJD_ARRAYS)}")
    print(f"  Realizations per field: {N}")
    print(f"  Total: {N * len(BLOEM_MJD_ARRAYS)}")

    stats = generate_binary_csvs(
        output_path=output_path,
        n_per_field=N,
        random_seed=42,
    )

    print(f"\nDone: {stats['saved']} CSVs written, {stats['failed']} Kepler failures")
    print(f"Truth table: {stats['truth_table']}")


if __name__ == "__main__":
    main()
