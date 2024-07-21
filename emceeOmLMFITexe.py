#

# import emcee
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# from astropy.time import Time
# import astropy.units as u
# import thejoker as tj

import lmfit
import matplotlib.pylab as pylab
import argparse
import corner
import make_RVs
import json

SIM_PARAMS = "simulated_params"
T = "T0"
N_OF_PS = "NumOfPeriods"
LOG_PERIOD = "LogPeriod"
PERIOD = "Period"
ECC = "Eccentricity"
OMEGA = "OMEGA_rad"
K1_STR = "K1"
K2_STR = "K2"
GAMMA = "GAMMA"
RANGE = "Range"
SAMPLES = "Samples"
RVS = "RadialVelocities"
ERR_RVS = "RVErrors"
TS = "TimeStamps"
TS_DIFF = TS + "Diff"
M1 = "Mass1"
Q = "MassRatio"
INC = "Inclination"
NOISE_SIG = "noise_sigma"
NRV = "number_of_samples"
PLOT = "plot"

LMFIT_PARAMS = "lmfit_params"
SEARCH_REGION = "search_region"
MINI_METHOD = "minimization_method"
BIN_TIME = "BinTime"
INIT_VAL = "init_value"
MIN_VAL = "min_value"
MAX_VAL = "max_value"
VARY = "vary"

CORNER_PARAMS = "corner_params"
TRUTHS = "truths"
LN_SIGMA = "__lnsigma"
CONRER_METHOD = "method"
NAN_POLICY = "nan_policy"
BURN = "burn"
STEPS = "steps"
THIN = "thin"

TIME_STAMPS = "ts"
TRUE_ERRORS = "true_errs"
ERRORS = "errs"
RADIAL_VELS = "rvs"


def get_simulated_data(args_dict):
    parameters_dict = args_dict[SIM_PARAMS]
    rvs_1, ts, errs_v1 = make_RVs.out_single_and_plot(parameters_dict[T], parameters_dict[PERIOD], parameters_dict[ECC],
                                                      parameters_dict[OMEGA], parameters_dict[K1_STR],
                                                      parameters_dict[K2_STR],
                                                      parameters_dict[GAMMA], parameters_dict[NRV],
                                                      parameters_dict[NOISE_SIG], args_dict[PLOT])
    # astropy_rvs = u.quantity.Quantity(rvs_1, unit=u.km / u.s)
    # astropy_err_rvs = u.quantity.Quantity(np.ones(parameters_dict[NRV]) * parameters_dict, unit=u.km / u.s)
    # t = Time(ts + 60347, format="mjd", scale="tcb")
    # data = tj.RVData(t=t, rv=astropy_rvs, rv_err=astropy_err_rvs)
    # if args_dict[PLOT]:
    #     _ = data.plot()
    #     plt.show()
    ret_dict = {TIME_STAMPS: ts.tolist(),
                RADIAL_VELS: rvs_1.tolist(),
                TRUE_ERRORS: errs_v1.tolist(),
                ERRORS: [ parameters_dict[NOISE_SIG]] * parameters_dict[NRV] }
    return ret_dict


def define_search_param(params, search_params_dict, field, radius=0):
    params.add(field, value=search_params_dict[field][INIT_VAL] ,
               min=search_params_dict[field][MIN_VAL] - radius/2,
               max=search_params_dict[field][MAX_VAL] + radius/2,
               vary=search_params_dict[field][VARY])


def lmfit_on_sample(args_dict, data):
    lmfit_params_dict = args_dict[LMFIT_PARAMS]
    pylab_params = {'legend.fontsize': 'large',
                    'figure.figsize': (12, 4),
                    'axes.labelsize': 'x-large',
                    'axes.titlesize': 'x-large',
                    'xtick.labelsize': 'x-large',
                    'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(pylab_params)
    locleg = 'upper left'
    sys.setrecursionlimit(int(1E6))

    # *********************** USER INPUT ***********************

    hjds1 = np.array(data[TIME_STAMPS])
    v1s = np.array(data[RADIAL_VELS])
    errv1s = np.abs(data[ERRORS])
    mini_method = lmfit_params_dict[MINI_METHOD]

    # Please note initial values for parameters, bounds, and whether they should vary or not
    params = lmfit.Parameters()
    search_params_dict = lmfit_params_dict[SEARCH_REGION]
    define_search_param(params, search_params_dict, PERIOD)
    define_search_param(params, search_params_dict, GAMMA)
    define_search_param(params, search_params_dict, K1_STR)
    define_search_param(params, search_params_dict, OMEGA)
    define_search_param(params, search_params_dict, ECC)
    t0_search_region = search_params_dict[PERIOD][MAX_VAL]
    define_search_param(params, search_params_dict, T, t0_search_region)

    # *********************** END OF USER INPUT ***********************
    # Initialization
    mini = lmfit.Minimizer(chisqr, params, fcn_kws={TIME_STAMPS: hjds1, RADIAL_VELS: v1s, ERRORS: errv1s})
    result = mini.minimize(method=mini_method)
    print(lmfit.fit_report(result))
    print(lmfit.fit_report)  # will print you a fit report from your model
    Gamma1_res = result.params[GAMMA].value
    K1_res = result.params[K1_STR].value
    Omega_res = result.params[OMEGA].value
    ecc_res = result.params[ECC].value
    P_res = result.params[PERIOD].value
    T0_res = result.params[T].value

    JDsplot = np.arange(min(hjds1) - 0.5 * P_res, max(hjds1) + 0.5 * P_res, .1)
    phsplot = (JDsplot - T0_res) / P_res - ((JDsplot - T0_res) / P_res).astype(int)
    Ms = 2 * np.pi * phsplot
    Es = Kepler(np.pi, Ms, ecc_res)
    eccfac = np.sqrt((1 + ecc_res) / (1 - ecc_res))
    nusdata = 2. * np.arctan(eccfac * np.tan(0.5 * Es))
    plt.errorbar(hjds1, v1s, yerr=errv1s, fmt='o', color='black')
    plt.plot(JDsplot, v1mod(nusdata, Gamma1_res, K1_res, Omega_res, ecc_res), color='black', label='primary')
    plt.xlabel('MJD')
    plt.ylabel(r'RV [${\rm km}\,{\rm s}^{-1}$]')
    plt.legend(loc=locleg)
    plt.show()
    # Create continous phase grid
    phs = np.linspace(0., 1., num=1000)
    Ms = 2 * np.pi * phs
    Es = Kepler(np.pi, Ms, ecc_res)
    eccfac = np.sqrt((1 + ecc_res) / (1 - ecc_res))
    nus = 2. * np.arctan(eccfac * np.tan(0.5 * Es))

    # Create data phase grid
    phsdata = (hjds1 - T0_res) / P_res - ((hjds1 - T0_res) / P_res).astype(int)
    phsdata[phsdata < 0] = phsdata[phsdata < 0] + 1.
    Ms = 2 * np.pi * phsdata
    Es = Kepler(np.pi, Ms, ecc_res)
    eccfac = np.sqrt((1 + ecc_res) / (1 - ecc_res))
    nusdata = 2. * np.arctan(eccfac * np.tan(0.5 * Es))
    rms1 = (np.sum((v1mod(nusdata, Gamma1_res, K1_res, Omega_res, ecc_res) - v1s) ** 2) / len(v1s)) ** 0.5

    if args_dict[PLOT]:
        fig, ax = plt.subplots(figsize=[6, 3])
        plt.errorbar(phsdata, v1s, yerr=errv1s, fmt='o', color='red')

        plt.plot(phs, v1mod(nus, Gamma1_res, K1_res, Omega_res, ecc_res), color='red', label=r'Primary')
        plt.plot([0., 1., ], [Gamma1_res, Gamma1_res], color='black')
        plt.xlim(0., 1.)
        plt.legend()
        plt.ylabel(r'RV $[{\rm km}\,{\rm s}^{-1}]$')
        plt.xlabel(r'phase')
        plt.show()

    print("RMS1 is ", rms1)
    print("mean Error: ", np.mean(errv1s))

    print("multiply errors1 by: ", rms1 / np.mean(errv1s))
    return result


# For converting Mean anomalies to eccentric anomalies (M-->E)
def Kepler(E, M, ecc):
    counter = 0
    conversion_count = 1
    loop_E = E
    while True:
        if counter > 990:
            loop_E = np.random.rand(len(M)) * np.pi
            counter = 0
            print("did not converge {} times".format(conversion_count))
            conversion_count += 1
        E2 = (M - ecc * (loop_E * np.cos(loop_E) - np.sin(loop_E))) / (1. - ecc * np.cos(loop_E))
        eps = np.abs(E2 - loop_E)
        if np.all(eps < 1E-7):
            return E2
        else:
            loop_E = E2
            counter += 1


# Given true anomaly nu and parameters, function returns an 2-col array of modeled (Q,U)
def v1mod(nu, gamma1, k1, omega, ecc):
    v1 = gamma1 + k1 * (np.cos(omega + nu) + ecc * np.cos(omega))
    return v1


def nus1(hjds, P, T0, ecc):
    phis = (hjds - T0) / P - ((hjds - T0) / P).astype(int)
    phis[phis < 0] = phis[phis < 0] + 1.
    Ms = 2 * np.pi * phis
    Es = Kepler(np.pi, Ms, ecc)
    eccfac = np.sqrt((1 + ecc) / (1 - ecc))
    nusdata = 2. * np.arctan(eccfac * np.tan(0.5 * Es))
    return nusdata


# Target function (returns differences between model array with parameter set p and data)
def chisqr(p, **kws):
    Gamma1 = p[GAMMA].value
    K1 = p[K1_STR].value
    Omega = p[OMEGA].value
    ecc = p[ECC].value
    T0 = p[T].value
    P = p[PERIOD].value
    v1 = v1mod(nus1(kws[TIME_STAMPS], P, T0, ecc), Gamma1, K1, Omega, ecc)
    return (v1 - kws[RADIAL_VELS]) / kws[ERRORS]


def corner_plot(args_dict, data, mini_results, truths=None):
    corner_params = args_dict[CORNER_PARAMS]
    mini_results.params.add(LN_SIGMA, value=np.log(corner_params[LN_SIGMA][INIT_VAL]),
                            min=np.log(corner_params[LN_SIGMA][MIN_VAL]),
                            max=np.log(corner_params[LN_SIGMA][MAX_VAL]))
    res = lmfit.minimize(chisqr,
                         kws = data,
                         method=corner_params[CONRER_METHOD],
                         nan_policy=corner_params[NAN_POLICY],
                         burn=corner_params[BURN],
                         steps=corner_params[STEPS],
                         thin=corner_params[THIN],
                         params=mini_results.params, is_weighted=False, progress=False)

    # result_emcee = mini.minimize(params=emcee_params)
    print(res.var_names)
    print(list(res.params.valuesdict().values()))

    if truths:
        # truths[T] = truths[T] % truths[PERIOD]
        emcee_plot = corner.corner(res.flatchain, labels=res.var_names,
                                   truths=truths)
    else:
        emcee_plot = corner.corner(res.flatchain, labels=res.var_names)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="A simple example of argparse")

    # Add arguments
    parser.add_argument('--plot', help='should plot', action='store_true')
    parser.add_argument('--output_dir', '-o', type=str, help='The output dir', required=True)
    parser.add_argument('--json_params_file', '-j', type=str, help='json format parameters', required=True)
    parser.add_argument('--generate_sample', '-g', help='should Generate', action='store_true')
    parser.add_argument('--load_sample', '-l', type=str, help='directory to samples json file')
    parser.add_argument('--seed', '-s', type=int, help='seed for consistent outputs for debug', default=1234)
    # Parse the arguments
    args = parser.parse_args()

    np.random.seed(args.seed)
    np.random.default_rng(seed=args.seed)

    if os.path.isdir(args.output_dir):
        if not os.listdir(args.output_dir):
            print("outDir is not empty. choose different out")
    else:
        os.makedirs(args.output_dir)

    with open(args.json_params_file, 'r') as json_file:
        args_dict = json.load(json_file)

    if args.generate_sample:
        print("Generating Sample by Json Params from: {}".format(args.json_params_file))
        print("Given Orbital Params: {}".format(args_dict[SIM_PARAMS]))

        data = get_simulated_data(args_dict)
        truths = [args_dict[SIM_PARAMS][PERIOD],
                  args_dict[SIM_PARAMS][GAMMA],
                  args_dict[SIM_PARAMS][K1_STR],
                  args_dict[SIM_PARAMS][OMEGA],
                  args_dict[SIM_PARAMS][ECC],
                  args_dict[SIM_PARAMS][T],
                  0]
        data["truths"] = truths
        samples_path = os.path.join(args.output_dir, "sample.json")
        print(data)
        with open(samples_path, 'w') as json_file:
            json.dump(data, json_file)
    elif args.load_sample:
        samples_path = args.load_sample
    else:
        print("no data were chosen. Either generate sample or provide a sample json path")
        exit(1)

    print("Uploading Sample from file: {}".format(samples_path))
    with open(samples_path, 'r') as json_file:
        data = json.load(json_file)

    mini_results = lmfit_on_sample(args_dict, data)
    #todo add best prediction plot
    if TRUTHS in data.keys():
        corner_plot(args_dict, data, mini_results, data[TRUTHS])
    else:
        corner_plot(args_dict, data, mini_results)
    exit(0)


if __name__ == '__main__':
    main()
