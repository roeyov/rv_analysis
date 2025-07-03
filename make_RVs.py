# Short script to create mock RVs from a given binary (Roey Ovadia, Tomer Shenar, tshenar@tau.ac.il)
import os
import sys

# Add the Scripts directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from utils.constants import *
from tmps.sigmaSimDistribution import sample_method1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import strftime
import tqdm
from scipy import constants
from scipy.integrate import quad
from scipy.interpolate import interp1d

np.set_printoptions(precision=2)


######################################
# # # # # # #FUNCTIONS # # # # #
######################################

def Kepler(E, M, ecc):
    counter = 0
    loop_E = E
    while True:
        if counter > 990:
            return None
        E2 = (M - ecc * (loop_E * np.cos(loop_E) - np.sin(loop_E))) / (1. - ecc * np.cos(loop_E))
        eps = np.abs(E2 - loop_E)
        if np.all(eps < 1E-10):
            return E2
        else:
            loop_E = E2
            counter += 1


# Returns RVs for primary and secondary as function of nu (true anomaly) and orbital parameters
def RV12(nu, gamma, k1, k2, omega, ecc):
    v1 = gamma + k1 * (np.cos(omega + nu) + ecc * np.cos(omega))
    v2 = gamma + k2 * (np.cos(np.pi + omega + nu) + ecc * np.cos(np.pi + omega))
    return v1, v2


# Returns true anomaly from phases
def nu_func(phi, ecc):
    e_fac = np.sqrt((1 + ecc) / (1 - ecc))
    M = 2 * np.pi * phi
    E = Kepler(1., M, ecc)
    if E is None:
        return E
    return 2. * np.arctan(e_fac * np.tan(0.5 * E))


######################################
# # # # # # #END FUNCTIONS   # # # # #
######################################

def get_data(t0, p, ecc, omega, k1, k2, gamma, nrv, sig_rv):
    # Generate random array of observation times between 0 & P
    ts = np.array([ 74.5177001 ,  97.64613847, 103.9829237 , 183.06141608,
           228.11894284, 238.69681915, 258.76892231, 295.83020203,
           338.11024624, 356.48782477, 409.01431527, 432.39915669,
           450.18902947, 453.56448127, 467.40229203, 477.5505655 ,
           490.73553866, 514.63869076, 521.51148672, 603.3569005 ,
           644.44251155, 649.57725408, 660.60566377, 663.63165303,
           683.56149292])

    # ts = np.sort(uniform_random_sample((0, 1), nrv)) * NUM_OF_DAYS
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
    sig_rv_arr = [sigma]*nrv
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

    x = np.linspace(tuple_range[0], tuple_range[1], n_of_samples)
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


def out_single(t0, p, ecc, omega, m1, q, gamma, inc, nrv, sig_rv, plot=False):
    k1, k2 = get_rv_amplitudes(m1, p, q, ecc, inc)
    print("T0: {:.2f}, P: {:.2f}, ecc:{:.2f}, omega: {:.2f}, k1: {:.2f}, "
          "k2: {:.2f}, gamma: {:.2f}, , M: {:.2f}, Q: {:.2f}, Inc: {:.2f}".format(
        t0, p, ecc, omega, k1, k2, gamma, m1, q, inc))
    rvs_1, ts, errs_v1 = out_single_and_plot(t0, p, ecc, omega, k1, k2, gamma, nrv, sig_rv, plot)

    return rvs_1, ts, errs_v1


def out_multiple_and_dump(args_dict, nrv, sig_rv, n_of_samples, out_dir, are_binaries):
    n_of_samples_in_files = int(1e4)
    columns = [T, ECC, OMEGA, K1_STR, K2_STR, GAMMA, PERIOD, M1, Q, INC, RADIAL_VELS, TIME_STAMPS, ERRORS,FEATURES,LABELS]
    for key in args_dict.keys():
        if key == INC:
            args_dict[key][SAMPLES] = sine_inclination_sample(args_dict[key][RANGE], n_of_samples)
            continue
        if RANGE in args_dict[key]:
            args_dict[key][SAMPLES] = uniform_random_sample(args_dict[key][RANGE], n_of_samples)

    args_dict[PERIOD][SAMPLES] = 10 ** args_dict[LOG_PERIOD][SAMPLES]
    args_dict[T][SAMPLES] = (args_dict[T][SAMPLES] - 0.5) * args_dict[PERIOD][SAMPLES]

    args_dict[ECC][SAMPLES][args_dict[PERIOD][SAMPLES] <= 5] = 0

    args_dict[K1_STR][SAMPLES], args_dict[K2_STR][SAMPLES] = get_rv_amplitudes(args_dict[M1][SAMPLES],
                                                                               args_dict[PERIOD][SAMPLES],
                                                                               args_dict[Q][SAMPLES],
                                                                               args_dict[ECC][SAMPLES],
                                                                               args_dict[INC][SAMPLES])
    args_dict[MAX_MIN_DIFF] = {}
    min_max_diff = []

    data = []
    files_counter = 1
    non_converging = 0
    # f = open(out_dir.format(files_counter), 'w')

    for i in tqdm.tqdm(range(n_of_samples)):
        # print("T0: {:.2f}, P: {:.2f}, ecc:{:.2f}, omega: {:.2f}, k1: {:.2f}, "
        #       "k2: {:.2f}, gamma: {:.2f}, , M: {:.2f}, Q: {:.2f}, Inc: {:.2f}".format(
        #                                           args_dict[T][SAMPLES][i],
        #                                           args_dict[PERIOD][SAMPLES][i],
        #                                           args_dict[ECC][SAMPLES][i],
        #                                           args_dict[OMEGA][SAMPLES][i],
        #                                           args_dict[K1_STR][SAMPLES][i],
        #                                           args_dict[K2_STR][SAMPLES][i],
        #                                           args_dict[GAMMA][SAMPLES][i],
        #                                           args_dict[M1][SAMPLES][i],
        #                                           args_dict[Q][SAMPLES][i],
        #                                           args_dict[INC][SAMPLES][i]))

        rvs_1, ts, errs_v1 = out_single_and_plot(args_dict[T][SAMPLES][i],
                                                 args_dict[PERIOD][SAMPLES][i],
                                                 args_dict[ECC][SAMPLES][i],
                                                 args_dict[OMEGA][SAMPLES][i],
                                                 args_dict[K1_STR][SAMPLES][i],
                                                 args_dict[K2_STR][SAMPLES][i],
                                                 args_dict[GAMMA][SAMPLES][i],
                                                 nrv, sig_rv,
                                                 plot=False)

        if rvs_1 is None:
            non_converging += 1
            # print(float(non_converging)/i)
            continue
        min_max_diff_1 =  max(rvs_1) - min(rvs_1)
        calced_features = [sig_rv,min_max_diff_1, np.mean(rvs_1), np.std(rvs_1)]
        features = np.concatenate([rvs_1, ts, np.ediff1d(ts), calced_features])
        labels = int(are_binaries)
        min_max_diff.append(min_max_diff_1)
        data.append([args_dict[T][SAMPLES][i],
                     args_dict[ECC][SAMPLES][i],
                     args_dict[OMEGA][SAMPLES][i],
                     args_dict[K1_STR][SAMPLES][i],
                     args_dict[K2_STR][SAMPLES][i],
                     args_dict[GAMMA][SAMPLES][i],
                     args_dict[PERIOD][SAMPLES][i],
                     args_dict[M1][SAMPLES][i],
                     args_dict[Q][SAMPLES][i],
                     args_dict[INC][SAMPLES][i],
                     rvs_1, ts, np.ones(rvs_1.shape) * sig_rv,
                     features,
                     labels])
        if (i + 1) % n_of_samples_in_files == 0:
            df = pd.DataFrame(data, columns=columns)
            df.to_parquet(out_dir.format(files_counter))
            files_counter += 1

            data = []
    args_dict[MAX_MIN_DIFF][SAMPLES] = min_max_diff


    # histogram plots
    # for key in args_dict:
    #     # if SAMPLES not in args_dict[key]:
    #     # if key != MAX_MIN_DIFF:
    #     #     continue
    #     try:
    #         plt.hist(np.log10(args_dict[key][SAMPLES]), bins=100, alpha=0.5)
    #         plt.title("log_" + key + ' '+ str(np.sum(np.array(args_dict[key][SAMPLES])<20)))
    #
    #         # plt.hist(args_dict[key][SAMPLES], bins=50, alpha=0.5)
    #         # plt.title(key)
    #
    #         plt.show()
    #     except ValueError:
    #         continue
    if len(data):
        df = pd.DataFrame(data, columns=columns)
        df.to_parquet(out_dir.format(files_counter))
    # f.close()


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
        generate_trues = False
        T0_RANGE = (0, 1)  # flat times pi
        LOG_PERIODS_RANGE = (0, 4)  # flat log space
        ECC_RANGE = (0, 0.97)  # flat space
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
        N_OF_SAMPS = int(1e7)
        dataset_name = "new_sigma"
        # np.random.seed(42)
        if generate_trues:
            timestr = strftime("{}_{}_Trues".format(dataset_name, str(N_OF_SAMPS)))
            OUTDIR = r"//Users/roeyovadia/Documents/Data/simulatedData_new_noise/{}/".format(timestr)
            os.makedirs(OUTDIR, exist_ok=True)
            out_fp_format = OUTDIR + r"/{}.parquet"
            out_multiple_and_dump(ARGS_DICT, NRV, sig_RV, N_OF_SAMPS, out_fp_format, 1)
        else:
            timestr = strftime("{}_{}_Falses".format(dataset_name, str(N_OF_SAMPS)))
            OUTDIR = r"/Users/roeyovadia/Documents/Data/simulatedData_new_noise/{}/".format(timestr)
            os.makedirs(OUTDIR, exist_ok=True)
            out_fp_format = OUTDIR + r"/{}.parquet"
            out_multiple_and_dump(ARGS_DICT, NRV, sig_RV, N_OF_SAMPS, out_fp_format, 0)


if __name__ == "__main__":
    main()
