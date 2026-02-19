# Short script to create mock RVs from a given binary (Roey Ovadia, Tomer Shenar, tshenar@tau.ac.il)
import os
import sys

from utils.roche_lobe import compute_min_period_row

# Add the Scripts directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from utils.constants import *
from tmps.sigmaSimDistribution import sample_method1, sample_method2
from orbital.kepler import kepler_iterative, true_anomaly_from_E

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import strftime
import tqdm
from scipy import constants
from scipy.integrate import quad
from scipy.interpolate import interp1d

np.set_printoptions(precision=2)

failed_exec_dict = {
    "M1": 0,                  # stellar mass [Msun]
    "Rstar": 0,               # stellar radius [Rsun]
    "mass_flag_peri": 0,     # Roche-lobe / mass consistency flag at periastron
    "mass_flag_apa": 0,      # same at apastron
    "Pmin_peri_central": 0,   # central min period (days) at periastron
    "Pmin_peri_upper": 0,     # +error
    "Pmin_peri_lower": 0,     # -error
    "Pmin_apa_central": 0,    # central min period (days) at apastron
    "Pmin_apa_upper": 0,      # +error
    "Pmin_apa_lower": 0,      # -error
    "q_at_min_peri": 0,       # mass ratio q at periastron limit
    "Pmin_peri_best": 0,      # best min-period value (could be same as central)
    "q_at_min_apa": 0,        # mass ratio q at apastron limit
}

######################################
# # # # # # #FUNCTIONS # # # # #
######################################

def Kepler(E, M, ecc):
    """Wrapper around canonical kepler_iterative; returns None on non-convergence."""
    result = kepler_iterative(M, ecc, tol=1e-10, max_resets=0, max_iter_per_reset=990)
    if np.any(np.isnan(result)):
        return None
    return result


# Returns RVs for primary and secondary as function of nu (true anomaly) and orbital parameters
def RV12(nu, gamma, k1, k2, omega, ecc):
    v1 = gamma + k1 * (np.cos(omega + nu) + ecc * np.cos(omega))
    v2 = gamma + k2 * (np.cos(np.pi + omega + nu) + ecc * np.cos(np.pi + omega))
    return v1, v2


# Returns true anomaly from phases
def nu_func(phi, ecc):
    M = 2 * np.pi * phi
    E = Kepler(1., M, ecc)
    if E is None:
        return E
    return true_anomaly_from_E(E, ecc)


######################################
# # # # # # #END FUNCTIONS   # # # # #
######################################

def get_data(t0, p, ecc, omega, k1, k2, gamma, nrv, sig_rv):
    # Generate random array of observation times between 0 & P
    # ts = np.array([ 74.5177001 ,  97.64613847, 103.9829237 , 183.06141608,
    #        228.11894284, 238.69681915, 258.76892231, 295.83020203,
    #        338.11024624, 356.48782477, 409.01431527, 432.39915669,
    #        450.18902947, 453.56448127, 467.40229203, 477.5505655 ,
    #        490.73553866, 514.63869076, 521.51148672, 603.3569005 ,
    #        644.44251155, 649.57725408, 660.60566377, 663.63165303,
    #        683.56149292])

    ts = np.sort(uniform_random_sample((0, 1), nrv)) * NUM_OF_DAYS
    # Generate corresponding phases
    phases = (ts - t0) / p - ((ts - t0) / p).astype(int)
    # Generate mean anomalies
    # ms = 2 * np.pi * phases
    # # Generate mean anomalies
    # es = Kepler(1., ms, ecc)
    # Generate true anomalies
    nus = nu_func(phases, ecc)
    if nus is None:
        return None, None, None, None
    # Generate true RVs for primary
    rvs_1_true = RV12(nus, gamma, k1, k2, omega, ecc)[0]

    # Generate errors from normal distribution
    # sig = 3
    # sig_rv_arr = np.array([sig_rv] * nrv)
    # errs_v1 = np.array([np.random.normal(0, sig) for sig in sig_rv_arr])


    # Generate errors from fitted log normal distribution
    shape, loc, scale = 0.8702, 0.0, 3.4602
    sigma_arr, errs_v1_arr = sample_method1(shape=shape, loc=loc, scale=scale,n_objects=1, n_measurements=nrv)
    sigma = sigma_arr[0]
    errs_v1 = errs_v1_arr[0]
    sig_rv_arr = sigma
    # Generate realistic RVs for primary (this is what you should store)
    rvs_1 = rvs_1_true + errs_v1
    return rvs_1, ts, sig_rv_arr, errs_v1


def out_single_and_plot(t0, p, ecc, omega, k1, k2, gamma, nrv, sig_rv, plot=False):
    rvs_1, ts, sig_rv_arr, errs_v1 = get_data(t0, p, ecc, omega, k1, k2, gamma, nrv, sig_rv)
    if rvs_1 is None:
        return None, None, None
    # For testing purposes:
    # Plot continuous RV curve
    if plot:
        ts_dense = np.linspace(0., NUM_OF_DAYS, num=10000)
        phis_dense = (ts_dense - t0) / p - ((ts_dense - t0) / p).astype(int)
        nus_dense = nu_func(phis_dense, ecc)
        if nus_dense is None:
            return None, None, None
        rvs_dense = RV12(nus_dense, gamma, k1, k2, omega, ecc)[0]

        plt.plot(ts_dense, rvs_dense, label='RV curve')
        # Plot "measurements" with error bars
        plt.errorbar(ts, rvs_1, yerr=sig_rv_arr, label='RV measurements', fmt='.')
        plt.xlabel('Time [days]')
        plt.ylabel('RV [km/s]')
        plt.show()
    return rvs_1, ts, errs_v1


def uniform_random_sample(tuple_range, n_of_samples):
    return np.random.uniform(low=tuple_range[0], high=tuple_range[1], size=n_of_samples)


def sine_inclination_sample(tuple_range, n_of_samples):
    def pdf(x):
        return np.sin(x)

    x = np.linspace(tuple_range[0], tuple_range[1], 1000)
    # Calculate the CDF by numerical integration
    cdf = np.array([quad(pdf, tuple_range[0], xi)[0] for xi in x])
    # Normalize the CDF
    cdf = cdf / cdf[-1]
    # Interpolate the inverse CDF (percent point function, PPF)
    inverse_cdf = interp1d(cdf, x, kind='linear')
    # Generate uniform random samples
    uniform_samples = np.random.rand(n_of_samples)
    # Transform uniform samples using the inverse CDF
    return inverse_cdf(uniform_samples)


def get_rv_amplitudes(m1, p, q, e, i):
    G = constants.G
    p_sec = p * 86400
    m1_kg = m1 * 1.989e30
    a_cubed = (G * m1_kg * p_sec * p_sec) / (4 * np.pi * np.pi)
    a = np.cbrt(a_cubed)

    k2 = np.cbrt((2 * np.pi * G * m1_kg) / (p_sec * (1 + q) * (1 + q))) * np.sin(i) / np.sqrt(1 - (e * e))
    k1 = q * k2

    return k1 / 1000, k2 / 1000

import math

import math
import pandas as pd

def ostar_radius_series_from_mass(
    M_Msun: float,
    relation: str = "eker2018",
    sigma_logR_dex: float = 0.10
) -> pd.Series:
    """
    Estimate MAIN-SEQUENCE O-star radius [Rsun] from mass [Msun] and return
    a pandas.Series with the fields:
        Mspec, Mspec_er_plus, Mspec_er_minus,
        R_star, R_star_er_plus, R_star_er_minus

    Parameters
    ----------
    M_Msun : float
        Stellar mass [Msun].
    relation : str, optional
        Either "eker2018" (default) or "demircan1991".
    sigma_logR_dex : float, optional
        Intrinsic scatter in log10(R). Default 0.10 dex (~±25%).

    Returns
    -------
    pandas.Series
    """
    if M_Msun <= 0:
        raise ValueError("Mass must be positive.")

    rel = relation.lower()
    if rel.startswith("eker"):
        # log10(R/Rsun) = 0.64 log10(M/Msun) + 0.011
        a = 0.64
        c = 10**0.011  # ≈ 1.026
        R = c * (M_Msun ** a)
    elif rel.startswith("demir"):
        # R/Rsun = 1.33 * (M/Msun)^0.555
        a = 0.555
        c = 1.33
        R = c * (M_Msun ** a)
    else:
        raise ValueError("relation must be 'eker2018' or 'demircan1991'")

    # Convert scatter (dex) to ln-space sigma
    sigma_lnR = sigma_logR_dex * math.log(10.0)
    R_plus_err  = R * (math.exp(sigma_lnR) - 1.0)
    R_minus_err = R * (1.0 - math.exp(-sigma_lnR))

    return pd.Series({
        "Mspec": M_Msun,
        "Mspec_er_plus": 0.0,
        "Mspec_er_minus": 0.0,
        "R_star": R,
        "R_star_er_plus": R_plus_err,
        "R_star_er_minus": R_minus_err,
    })


def out_single(t0, p, ecc, omega, m1, q, gamma, inc, nrv, sig_rv, plot=False):
    k1, k2 = get_rv_amplitudes(m1, p, q, ecc, inc)
    print("T0: {:.2f}, P: {:.2f}, ecc:{:.2f}, omega: {:.2f}, k1: {:.2f}, "
          "k2: {:.2f}, gamma: {:.2f}, , M: {:.2f}, Q: {:.2f}, Inc: {:.2f}".format(
        t0, p, ecc, omega, k1, k2, gamma, m1, q, inc))
    rvs_1, ts, errs_v1 = out_single_and_plot(t0, p, ecc, omega, k1, k2, gamma, nrv, sig_rv, plot)

    return rvs_1, ts, errs_v1


def out_multiple_and_dump(args_dict, nrv, sig_rv, n_of_samples, out_dir, are_binaries):
    """
    Generate simulated RV samples, features, and labels; merge Roche-lobe outputs per row;
    and write batched Parquet files.

    Parameters
    ----------
    args_dict : dict
        Must contain per-parameter config dicts with RANGE and/or SAMPLES.
        Expected keys (constants): T, ECC, OMEGA, K1_STR, K2_STR, GAMMA, PERIOD, M1, Q, INC,
                                   LOG_PERIOD (for sampling), MAX_MIN_DIFF placeholder, etc.
    nrv : int
        Number of RV epochs to simulate per system.
    sig_rv : float
        Per-epoch RV uncertainty (km/s).
    n_of_samples : int
        Number of systems to simulate.
    out_dir : str
        Format string path for output parquet files, e.g. "/path/to/chunk_{:03d}.parquet".
    are_binaries : bool or int
        Label for this batch (1 for binaries, 0 for singles).

    Notes
    -----
    - Uses dict-rows; Pandas unions keys into columns automatically.
    - Converts any numpy arrays stored in a single cell to lists (Parquet-friendly).
    - Merges `roche_lobe_dict` (from `compute_min_period_row`) into each row.
    - Forces keyword-only for compute_min_period_row arguments (per your function signature).
    """

    # ---- Pre-sampling for each parameter ----
    n_of_samples_in_files = int(1e4)
    base_keys = [
        T, ECC, OMEGA, K1_STR, K2_STR, GAMMA, PERIOD, M1, Q, INC,
        RADIAL_VELS, TIME_STAMPS, ERRORS, FEATURES, LABELS
    ]

    # Draw samples
    for key in list(args_dict.keys()):
        if key == INC:
            args_dict[key][SAMPLES] = sine_inclination_sample(args_dict[key][RANGE], n_of_samples)
            continue
        if RANGE in args_dict[key]:
            args_dict[key][SAMPLES] = uniform_random_sample(args_dict[key][RANGE], n_of_samples)

    # Derived samples
    args_dict[PERIOD][SAMPLES] = 10 ** args_dict[LOG_PERIOD][SAMPLES]
    args_dict[T][SAMPLES] = (args_dict[T][SAMPLES] - 0.5) * args_dict[PERIOD][SAMPLES]

    # Zero eccentricities for very short periods (scalarwise threshold)
    short_mask = args_dict[PERIOD][SAMPLES] <= 5
    args_dict[ECC][SAMPLES][short_mask] = 0

    # RV semi-amplitudes from physical params (vectorized)
    k1_vec, k2_vec = get_rv_amplitudes(
        args_dict[M1][SAMPLES],
        args_dict[PERIOD][SAMPLES],
        args_dict[Q][SAMPLES],
        args_dict[ECC][SAMPLES],
        args_dict[INC][SAMPLES],
    )
    args_dict[K1_STR][SAMPLES] = k1_vec
    args_dict[K2_STR][SAMPLES] = k2_vec

    # ---- Main loop ----
    data = []
    files_counter = 1
    non_converging = 0
    non_roche_lobing = 0
    min_max_diff_list = []

    # Ensure MAX_MIN_DIFF dict exists for compatibility with your later code
    if MAX_MIN_DIFF not in args_dict:
        args_dict[MAX_MIN_DIFF] = {}
    # initialize so caller can inspect even if loop short-circuits
    args_dict[MAX_MIN_DIFF][SAMPLES] = min_max_diff_list

    for i in tqdm.tqdm(range(n_of_samples)):
        # Per-row scalars
        t0_i     = float(args_dict[T][SAMPLES][i])
        per_i    = float(args_dict[PERIOD][SAMPLES][i])
        ecc_i    = float(args_dict[ECC][SAMPLES][i])
        om_i     = float(args_dict[OMEGA][SAMPLES][i])
        k1_i     = float(args_dict[K1_STR][SAMPLES][i])
        k2_i     = float(args_dict[K2_STR][SAMPLES][i])
        gam_i    = float(args_dict[GAMMA][SAMPLES][i])
        m1_i     = float(args_dict[M1][SAMPLES][i])
        q_i      = float(args_dict[Q][SAMPLES][i])
        inc_i    = float(args_dict[INC][SAMPLES][i])

        # Roche-lobe & radius info (merge later)
        mass_series = ostar_radius_series_from_mass(m1_i)  # returns Series with Mspec, R_star, etc.
        try:
            if not are_binaries:
                raise ValueError
            roche_lobe_dict = compute_min_period_row(
                mass_series,
                P_fit=per_i,
                K1=k1_i,
                e=ecc_i,
            )
        except ValueError as e:
            non_roche_lobing += 1
            roche_lobe_dict = failed_exec_dict.copy()

        # Simulate RVs
        rvs_1, ts, errs_v1 = out_single_and_plot(
            t0_i, per_i, ecc_i, om_i, k1_i, k2_i, gam_i,
            nrv, sig_rv, plot=False
        )

        if rvs_1 is None:
            non_converging += 1
            continue

        # Features
        min_max_diff_1 = float(np.max(rvs_1) - np.min(rvs_1))
        calced_features = [float(sig_rv), min_max_diff_1, float(np.mean(rvs_1)), float(np.std(rvs_1))]
        features = np.concatenate([rvs_1, ts, np.ediff1d(ts), calced_features])

        # Row dict (convert arrays to lists for Parquet friendliness)
        row = {
            'sim_id':i,
            T: t0_i,
            ECC: ecc_i,
            OMEGA: om_i,
            K1_STR: k1_i,
            K2_STR: k2_i,
            GAMMA: gam_i,
            PERIOD: per_i,
            M1: m1_i,
            Q: q_i,
            INC: inc_i,
            RADIAL_VELS: rvs_1.tolist(),
            TIME_STAMPS: ts.tolist(),
            ERRORS: (np.ones_like(rvs_1) * sig_rv).tolist(),
            FEATURES: features.tolist(),
            LABELS: int(are_binaries),
        }
        row.update(roche_lobe_dict)  # merge Roche-lobe outputs

        min_max_diff_list.append(min_max_diff_1)
        data.append(row)

        # Batch flush
        if (i + 1) % n_of_samples_in_files == 0:
            df = pd.DataFrame(data)
            df.to_parquet(out_dir.format(files_counter))
            files_counter += 1
            data = []

    # Save remainder
    if data:
        df = pd.DataFrame(data)
        df.to_parquet(out_dir.format(files_counter))

    # Persist min-max list in args_dict for downstream plots/diagnostics
    args_dict[MAX_MIN_DIFF][SAMPLES] = min_max_diff_list

    return {
        "written_files": files_counter if not data else files_counter,
        "non_converging": non_converging,
        "non_roche_lobing": non_roche_lobing,
        "n_rows_written": sum(1 for _ in min_max_diff_list),
    }



def main():
    SINGLE = False
    if SINGLE:
        ######################################
        # # # # # # # USER INPUT # # # # # # #
        ######################################
        # Orbit Pars:
        # Time of periastron
        T0 = -85.79
        # Period
        P = 335.66
        # Eccentricity
        e = 0.4
        # Argument of periastron
        Omega = 60. * np.pi / 180.  # omega in radians
        # Primary RV semi-amplitude
        K1 = 100.
        # secondary RV semi-amplitude
        K2 = 20.
        # Primary Mass
        m1 = 30.
        # Mass Ratio
        q = 0.4
        # secondary RV semi-amplitude
        inc = 0.5
        # systemic velocity
        Gamma = 3.
        # Number of RVs
        NRV = 25
        # RV error (RVERR = array)
        sig_RV = 3.
        # out_single_and_plot(T0, P, e, Omega, K1, K2, Gamma, NRV, sig_RV, True)

        out_single(T0, P, e, Omega, m1, q, Gamma, inc, NRV, sig_RV, True)
    else:
        generate_trues = True
        T0_RANGE = (0, 1)  # flat times pi
        LOG_PERIODS_RANGE = (0, 4)  # flat log space
        ECC_RANGE = (0, 0.999)  # flat space
        OMEGA_RANGE = (-np.pi, np.pi)  # flat space
        INC_RANGE = (0, np.pi / 2) if generate_trues else (0, 0)  # sine space
        M1_RANGE = (15, 80)  # flat on mass space
        Q_RANGE = (0, 1)  # flat space
        GAMMA_RANGE = (0, 50)  # flat space
        NRV = 25

        sig_RV = 3.
        ARGS_DICT = {T: {RANGE: T0_RANGE},
                     ECC: {RANGE: ECC_RANGE},
                     OMEGA: {RANGE: OMEGA_RANGE},
                     M1: {RANGE: M1_RANGE},
                     Q: {RANGE: Q_RANGE},
                     GAMMA: {RANGE: GAMMA_RANGE},
                     LOG_PERIOD: {RANGE: LOG_PERIODS_RANGE},
                     INC: {RANGE: INC_RANGE},
                     PERIOD: {},
                     K1_STR: {},
                     K2_STR: {},
                     }
        N_OF_SAMPS = int(1e5)
        dataset_name = "simulations"
        # np.random.seed(42)
        if generate_trues:
            timestr = strftime("{}_{}".format(dataset_name, str(N_OF_SAMPS)))
            OUTDIR = r"//Users/roeyovadia/Documents/Data/simulatedData/22_10_25/{}/".format(timestr)
            os.makedirs(OUTDIR, exist_ok=True)
            out_fp_format = OUTDIR + r"/true_{}.parquet"
            print(out_multiple_and_dump(ARGS_DICT, NRV, sig_RV, N_OF_SAMPS, out_fp_format, 1))
        else:
            timestr = strftime("{}_{}".format(dataset_name, str(N_OF_SAMPS)))
            OUTDIR = r"//Users/roeyovadia/Documents/Data/simulatedData/22_10_25/{}/".format(timestr)
            os.makedirs(OUTDIR, exist_ok=True)
            out_fp_format = OUTDIR + r"/false_{}.parquet"
            print(out_multiple_and_dump(ARGS_DICT, NRV, sig_RV, N_OF_SAMPS, out_fp_format, 0))


if __name__ == "__main__":
    main()
