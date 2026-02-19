# Script to create mock RVs from a given binary (Roey Ovadia, Tomer Shenar, tshenar@tau.ac.il)
import os
import sys

from utils.roche_lobe import compute_min_period_row

# Add the repo root to sys.path so top-level packages are importable
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
from utils.constants import *
from tmps.sigmaSimDistribution import sample_method1, sample_method2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import strftime
import tqdm

from simulations.common import (
    RV12, nu_func, get_rv_amplitudes, ostar_radius_series_from_mass,
    uniform_random_sample, sine_inclination_sample,
)

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


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def get_data(t0, p, ecc, omega, k1, k2, gamma, nrv, sig_rv):
    """Generate noisy binary RV data with random uniform time sampling."""
    ts = np.sort(uniform_random_sample((0, 1), nrv)) * NUM_OF_DAYS
    phases = (ts - t0) / p - ((ts - t0) / p).astype(int)
    nus = nu_func(phases, ecc)
    if nus is None:
        return None, None, None, None
    rvs_1_true = RV12(nus, gamma, k1, k2, omega, ecc)[0]

    # Generate errors from fitted log normal distribution
    shape, loc, scale = 0.8702, 0.0, 3.4602
    sigma_arr, errs_v1_arr = sample_method1(shape=shape, loc=loc, scale=scale,
                                             n_objects=1, n_measurements=nrv)
    sigma = sigma_arr[0]
    errs_v1 = errs_v1_arr[0]
    sig_rv_arr = sigma
    rvs_1 = rvs_1_true + errs_v1
    return rvs_1, ts, sig_rv_arr, errs_v1


def out_single_and_plot(t0, p, ecc, omega, k1, k2, gamma, nrv, sig_rv, plot=False):
    """Generate one binary system and optionally plot."""
    rvs_1, ts, sig_rv_arr, errs_v1 = get_data(t0, p, ecc, omega, k1, k2, gamma, nrv, sig_rv)
    if rvs_1 is None:
        return None, None, None
    if plot:
        ts_dense = np.linspace(0., NUM_OF_DAYS, num=10000)
        phis_dense = (ts_dense - t0) / p - ((ts_dense - t0) / p).astype(int)
        nus_dense = nu_func(phis_dense, ecc)
        if nus_dense is None:
            return None, None, None
        rvs_dense = RV12(nus_dense, gamma, k1, k2, omega, ecc)[0]

        plt.plot(ts_dense, rvs_dense, label='RV curve')
        plt.errorbar(ts, rvs_1, yerr=sig_rv_arr, label='RV measurements', fmt='.')
        plt.xlabel('Time [days]')
        plt.ylabel('RV [km/s]')
        plt.show()
    return rvs_1, ts, errs_v1


def out_single(t0, p, ecc, omega, m1, q, gamma, inc, nrv, sig_rv, plot=False):
    """Generate one binary system from physical parameters."""
    k1, k2 = get_rv_amplitudes(m1, p, q, ecc, inc)
    print("T0: {:.2f}, P: {:.2f}, ecc:{:.2f}, omega: {:.2f}, k1: {:.2f}, "
          "k2: {:.2f}, gamma: {:.2f}, , M: {:.2f}, Q: {:.2f}, Inc: {:.2f}".format(
        t0, p, ecc, omega, k1, k2, gamma, m1, q, inc))
    rvs_1, ts, errs_v1 = out_single_and_plot(t0, p, ecc, omega, k1, k2, gamma, nrv, sig_rv, plot)
    return rvs_1, ts, errs_v1


# ---------------------------------------------------------------------------
# Batch population synthesis (Parquet output)
# ---------------------------------------------------------------------------

def out_multiple_and_dump(args_dict, nrv, sig_rv, n_of_samples, out_dir, are_binaries):
    """
    Generate simulated RV samples, features, and labels; merge Roche-lobe outputs per row;
    and write batched Parquet files.

    Parameters
    ----------
    args_dict : dict
        Must contain per-parameter config dicts with RANGE and/or SAMPLES.
    nrv : int
        Number of RV epochs to simulate per system.
    sig_rv : float
        Per-epoch RV uncertainty (km/s).
    n_of_samples : int
        Number of systems to simulate.
    out_dir : str
        Format string path for output parquet files.
    are_binaries : bool or int
        Label for this batch (1 for binaries, 0 for singles).
    """
    n_of_samples_in_files = int(1e4)

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

    # Zero eccentricities for very short periods
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

    # Main loop
    data = []
    files_counter = 1
    non_converging = 0
    non_roche_lobing = 0
    min_max_diff_list = []

    if MAX_MIN_DIFF not in args_dict:
        args_dict[MAX_MIN_DIFF] = {}
    args_dict[MAX_MIN_DIFF][SAMPLES] = min_max_diff_list

    for i in tqdm.tqdm(range(n_of_samples)):
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

        mass_series = ostar_radius_series_from_mass(m1_i)
        try:
            if not are_binaries:
                raise ValueError
            roche_lobe_dict = compute_min_period_row(
                mass_series, P_fit=per_i, K1=k1_i, e=ecc_i,
            )
        except ValueError:
            non_roche_lobing += 1
            roche_lobe_dict = failed_exec_dict.copy()

        rvs_1, ts, errs_v1 = out_single_and_plot(
            t0_i, per_i, ecc_i, om_i, k1_i, k2_i, gam_i,
            nrv, sig_rv, plot=False
        )

        if rvs_1 is None:
            non_converging += 1
            continue

        min_max_diff_1 = float(np.max(rvs_1) - np.min(rvs_1))
        calced_features = [float(sig_rv), min_max_diff_1, float(np.mean(rvs_1)), float(np.std(rvs_1))]
        features = np.concatenate([rvs_1, ts, np.ediff1d(ts), calced_features])

        row = {
            'sim_id': i,
            T: t0_i, ECC: ecc_i, OMEGA: om_i,
            K1_STR: k1_i, K2_STR: k2_i, GAMMA: gam_i,
            PERIOD: per_i, M1: m1_i, Q: q_i, INC: inc_i,
            RADIAL_VELS: rvs_1.tolist(),
            TIME_STAMPS: ts.tolist(),
            ERRORS: (np.ones_like(rvs_1) * sig_rv).tolist(),
            FEATURES: features.tolist(),
            LABELS: int(are_binaries),
        }
        row.update(roche_lobe_dict)

        min_max_diff_list.append(min_max_diff_1)
        data.append(row)

        if (i + 1) % n_of_samples_in_files == 0:
            df = pd.DataFrame(data)
            df.to_parquet(out_dir.format(files_counter))
            files_counter += 1
            data = []

    if data:
        df = pd.DataFrame(data)
        df.to_parquet(out_dir.format(files_counter))

    args_dict[MAX_MIN_DIFF][SAMPLES] = min_max_diff_list

    return {
        "written_files": files_counter if not data else files_counter,
        "non_converging": non_converging,
        "non_roche_lobing": non_roche_lobing,
        "n_rows_written": sum(1 for _ in min_max_diff_list),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    SINGLE = False
    if SINGLE:
        T0 = -85.79
        P = 335.66
        e = 0.4
        Omega = 60. * np.pi / 180.
        K1 = 100.
        K2 = 20.
        m1 = 30.
        q = 0.4
        inc = 0.5
        Gamma = 3.
        NRV = 25
        sig_RV = 3.
        out_single(T0, P, e, Omega, m1, q, Gamma, inc, NRV, sig_RV, True)
    else:
        generate_trues = True
        T0_RANGE = (0, 1)
        LOG_PERIODS_RANGE = (0, 4)
        ECC_RANGE = (0, 0.999)
        OMEGA_RANGE = (-np.pi, np.pi)
        INC_RANGE = (0, np.pi / 2) if generate_trues else (0, 0)
        M1_RANGE = (15, 80)
        Q_RANGE = (0, 1)
        GAMMA_RANGE = (0, 50)
        NRV = 25
        sig_RV = 3.

        ARGS_DICT = {
            T: {RANGE: T0_RANGE},
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
