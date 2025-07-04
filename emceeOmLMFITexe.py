import sys
import os
import io
import contextlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lmfit
import matplotlib.pylab as pylab
import argparse
import corner
import make_RVs
import json
from types import SimpleNamespace


from Spectroscopy.constants import MJD_MID
from utils.constants import *
from utils.utils import make_runlog


def get_simulated_data(args_dict):
    """
    Generates simulated radial velocity data based on provided parameters.

    Args:
        args_dict (dict): A dictionary containing the simulation parameters and plot settings. It should have the following structure:
            - SIM_PARAMS: A dictionary containing the specific simulation parameters:
                - T (float): Some time parameter.
                - PERIOD (float): The period of the system.
                - ECC (float): The eccentricity of the orbit.
                - OMEGA (float): The argument of periapsis.
                - K1_STR (float): Radial velocity semi-amplitude of the primary component.
                - K2_STR (float): Radial velocity semi-amplitude of the secondary component.
                - GAMMA (float): Systemic velocity.
                - NRV (int): Number of radial velocity data points.
                - NOISE_SIG (float): Standard deviation of the noise to be added to the data.
            - PLOT (bool): A flag indicating whether to generate and display plots.

    Returns:
        dict: A dictionary containing the simulated data with the following keys:
            - TIME_STAMPS (list): List of time stamps of the observations.
            - RADIAL_VELS (list): List of simulated radial velocities.
            - TRUE_ERRORS (list): List of true errors in the simulated data.
            - ERRORS (list): List of noise standard deviations for the radial velocities.

    Notes:
        - The function uses an external function `make_RVs.out_single_and_plot` to generate the radial velocities and errors.
        - The plotting section of the code is currently commented out.
    """
    parameters_dict = args_dict[SIM_PARAMS]
    rvs_1, ts, errs_v1 = make_RVs.out_single_and_plot(parameters_dict[T], parameters_dict[PERIOD], parameters_dict[ECC],
                                                      parameters_dict[OMEGA], parameters_dict[K1_STR],
                                                      parameters_dict[K2_STR],
                                                      parameters_dict[GAMMA], parameters_dict[NRV],
                                                      parameters_dict[NOISE_SIG], args_dict[PLOT])
    ret_dict = {TIME_STAMPS: ts.tolist(),
                RADIAL_VELS: rvs_1.tolist(),
                TRUE_ERRORS: errs_v1.tolist(),
                ERRORS: [ parameters_dict[NOISE_SIG]] * parameters_dict[NRV] }
    return ret_dict


def define_search_param(params, search_params_dict, field, radius=0):
    """
    Defines a search parameter for a given field and adds it to the parameters object.

    Args:
        params (Parameters): The parameters object to which the search parameter will be added.
        search_params_dict (dict): A dictionary containing search parameter details with the following structure:
            - field (str): The name of the field for which the search parameter is defined.
            - INIT_VAL (float): The initial value of the parameter.
            - MIN_VAL (float): The minimum value of the parameter.
            - MAX_VAL (float): The maximum value of the parameter.
            - VARY (bool): A flag indicating whether the parameter should vary during the search.
        field (str): The specific field for which the search parameter is being defined.
        radius (float, optional): An optional radius value to adjust the min and max values of the parameter. Default is 0.

    Returns:
        None: The function modifies the `params` object in place.

    Notes:
        - The function adjusts the minimum and maximum values of the parameter by subtracting and adding half of the radius, respectively.
        - The `vary` flag determines whether the parameter will vary during the search process.
    """
    params.add(field, value=search_params_dict[field][INIT_VAL] ,
               min=search_params_dict[field][MIN_VAL] - radius/2,
               max=search_params_dict[field][MAX_VAL] + radius/2,
               vary=search_params_dict[field][VARY])


def summarize_result(result, star_name):
    """
    Turn a single lmfit.MinimizerResult into a flat dict.
    """
    row = {}
    # 1) Global fit statistics
    row['star_name'] = star_name
    row['method'] = result.method
    row['nfev'] = result.nfev  # function evals
    row['ndata'] = result.ndata  # data points
    row['nvarys'] = result.nvarys  # fitted vars
    row['chisqr'] = result.chisqr
    row['redchi'] = result.redchi
    row['aic'] = result.aic
    row['bic'] = result.bic

    # 2) Per-parameter summaries
    # result.params is an OrderedDict of Parameter objects
    for name, par in result.params.items():
        init = result.init_values.get(name, None)  # initial guess
        row[f'{name}_init'] = init
        row[f'{name}_value'] = par.value
        row[f'{name}_vary'] = par.vary
        # if stderr is None → warning that uncertainty wasn't estimated
        row[f'{name}_stderr'] = par.stderr

    return row



def extract_observations(data):
    hjds = np.array(data[TIME_STAMPS])
    vels = np.array(data[RADIAL_VELS])
    errs = np.abs(data[ERRORS])
    return hjds, vels, errs


def get_fit_report(result, star_name, out_dir=None):
    report = lmfit.fit_report(result)
    print(report)
    if out_dir:
        path = os.path.join(out_dir, f"{star_name}_report.txt")
        with open(path, 'w') as f:
            f.write(report)
        print(f"[{star_name}] Fit report saved to {path}")


def compute_orbital_params(result):
    params = result.params
    return (
        params[GAMMA].value,
        params[K1_STR].value,
        params[OMEGA].value,
        params[ECC].value,
        params[PERIOD].value,
        params[T].value
    )


def compute_rv_curve_time(hjds, P, T0, ecc, Gamma, K, Omega, step=0.1):
    """
    Returns time_grid and corresponding RV by calling v1mod on nu from Kepler.
    """
    tmin, tmax = hjds.min() - 0.5*P, hjds.max() + 0.5*P
    time_grid = np.arange(tmin, tmax, step)
    phases = ((time_grid - T0) / P) % 1
    M = 2*np.pi*phases
    E = Kepler(np.pi, M, ecc)
    nu = 2*np.arctan2(np.sqrt(1+ecc)*np.sin(E/2), np.sqrt(1-ecc)*np.cos(E/2))
    rv = v1mod(nu, Gamma, K, Omega, ecc)
    return time_grid, rv


def compute_rv_curve_phase(P, T0, ecc, Gamma, K, Omega, n_points=1000):
    """
    Returns phase_grid and RV by calling v1mod on nu from Kepler at each dense phase.
    """
    phase_grid = np.linspace(0,1,n_points)
    M = 2*np.pi*phase_grid
    E = Kepler(np.pi, M, ecc)
    if np.isnan(E.all()):
        return phase_grid, np.full(phase_grid.shape, np.nan)
    nu = 2*np.arctan2(np.sqrt(1+ecc)*np.sin(E/2), np.sqrt(1-ecc)*np.cos(E/2))
    rv = v1mod(nu, Gamma, K, Omega, ecc)
    return phase_grid, rv


def plot_time_series_with_residuals(hjds, vels, errs, time_grid, rv_model,
                                    Gamma, K, Omega, ecc, P, star_name,
                                    out_dir=None, figsize=(10,10)):
    vmod_data = np.interp(hjds, time_grid, rv_model)
    residuals = vels - vmod_data

    fig, (ax1, ax1r) = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={'height_ratios':[3,1]}
    )
    ax1.errorbar(hjds, vels, yerr=errs, fmt='o', color='black')
    ax1.plot(time_grid, rv_model, color='black', label='Model')
    ax1.set_ylabel(r'RV [{\\rm km}\\,{\rm s}^{-1}]')
    ax1.set_title(f"Orbital fit {star_name}: P={P:.2f}d, ecc={ecc:.3f}")
    ax1.legend(loc='upper left')

    ax1r.errorbar(hjds, residuals, yerr=errs, fmt='o', color='gray')
    ax1r.axhline(0, color='black', lw=0.8)
    ax1r.set_xlabel('MJD')
    ax1r.set_ylabel('O–C')

    fig.tight_layout()
    if out_dir:
        fig.savefig(os.path.join(out_dir, f"{star_name}_time_residuals.png"))
    plt.show()
    return residuals

def calculate_phase_criterias(data, result):
    hjds, vels, errs = extract_observations(data)
    Gamma, K, Omega, ecc, P, T0 = compute_orbital_params(result)
    phs_data, phase_grid, rv_phase, residuals = plot_phase_folded_with_residuals(
        hjds, vels, errs, P, T0,
        Gamma, K, Omega, ecc,
        '',plot=False
    )
    phs_data.sort()
    max_phase_gap = np.max((phs_data - np.roll(phs_data, 1))%1)
    top_max_rv_gap, bottom_max_rv_gap = (np.max(rv_phase) - np.max(vels))/(np.max(rv_phase)-Gamma), (np.min(rv_phase) - np.min(vels))/(Gamma - np.min(rv_phase))

    return np.round(max_phase_gap,2), np.round(top_max_rv_gap,2), np.round(bottom_max_rv_gap,2)



def plot_phase_folded_with_residuals(hjds, vels, errs, P, T0,
                                     Gamma, K, Omega, ecc,
                                     star_name, out_dir=None,plot=True,
                                     figsize=(10,10)):
    phs_data = ((hjds - T0) / P) % 1
    phase_grid, rv_phase = compute_rv_curve_phase(P, T0, ecc, Gamma, K, Omega)
    vmod_data = np.interp(phs_data, phase_grid, rv_phase)
    residuals = vels - vmod_data
    if plot:
        fig, (ax2, ax2r) = plt.subplots(
            2, 1, sharex=True, figsize=figsize,
            gridspec_kw={'height_ratios':[3,1]}
        )
        ax2.errorbar(phs_data, vels, yerr=errs, fmt='o', color='red')
        ax2.plot(phase_grid, rv_phase, color='red', label='Model')
        ax2.axhline(Gamma, color='black', linestyle='--')
        ax2.set_xlim(0,1)
        ax2.set_ylabel(r'RV [{\\rm km}\\,{\rm s}^{-1}]')
        ax2.set_title(f"Folded fit {star_name}")
        ax2.legend()

        ax2r.errorbar(phs_data, residuals, yerr=errs, fmt='o', color='gray')
        ax2r.axhline(0, color='black', lw=0.8)
        ax2r.set_xlabel('Phase')
        ax2r.set_ylabel('O–C')

        fig.tight_layout()
        if out_dir:
            fig.savefig(os.path.join(out_dir, f"{star_name}_phase_residuals.png"))
        plt.show()
    return phs_data, phase_grid, rv_phase, residuals


def print_lmfit_result(data, args_dict, star_name, result, out_dir=None):
    sys.setrecursionlimit(int(1e6))
    hjds, vels, errs = extract_observations(data)
    get_fit_report(result, star_name, out_dir)
    Gamma, K, Omega, ecc, P, T0 = compute_orbital_params(result)

    time_grid, rv_time = compute_rv_curve_time(hjds, P, T0, ecc, Gamma, K, Omega)
    plot_time_series_with_residuals(
        hjds, vels, errs, time_grid, rv_time,
        Gamma, K, Omega, ecc, P, star_name, out_dir
    )

    phs_data, phase_grid, rv_phase, residuals = plot_phase_folded_with_residuals(
        hjds, vels, errs, P, T0,
        Gamma, K, Omega, ecc,
        star_name, out_dir
    )

    return phs_data

def get_rv_weighted_mean(data):
    v1s     = np.array(data[RADIAL_VELS])
    errv1s  = np.abs(data[ERRORS])
    weights = 1.0 / errv1s ** 2
    gamma0 = np.sum(v1s * weights) / np.sum(weights)
    return gamma0

def compute_AIC_BIC(y_obs, errors,chi_sqr, num_params):
    """
    Compute the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

    Parameters:
        y_obs (array): Observed data.
        y_model (array): Model predictions.
        errors (array): Observational errors (standard deviation).
        num_params (int): Number of parameters in the model (k).

    Returns:
        AIC, BIC
    """
    n = len(y_obs)

    log_likelihood_term = np.sum(np.log(2 * np.pi * errors**2))

    # Compute AIC
    AIC = 2 * num_params + chi_sqr + log_likelihood_term

    # Compute BIC
    BIC = num_params * np.log(n) + chi_sqr + log_likelihood_term

    return AIC, BIC

def lmfit_on_sample(args_dict, output, data, star_name='', null_hyp=False):
    """
    Fit an orbital model to RV vs. MJD data, or—if null_hyp=True—fit
    the null hypothesis of a constant velocity and compute its reduced chi².

    Parameters
    ----------
    args_dict : dict
        Dictionary holding your LMFIT_PARAMS entry, etc.
    output : path or file-like
        Where to dump fit reports.
    data : dict-like
        Must contain keys TIME_STAMPS, RADIAL_VELS, and ERRORS.
    star_name : str, optional
        A label for printing/reporting.
    null_hyp : bool, optional
        If True, skip the orbital fit and instead compute reduced χ² of
        a constant–velocity model (weighted mean). Default is False.

    Returns
    -------
    If null_hyp is False:
        result : lmfit.MinimizerResult
            The full orbital fit result.
    If null_hyp is True:
        dict with keys
            'null_gamma'  : float
                Best-fit constant velocity.
            'null_redchi' : float
                Reduced χ² of the constant model.
    """
    # ---- style & recursion setup ----
    pylab_params = {
        'legend.fontsize': 'large',
        'figure.figsize': (12, 4),
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large'
    }
    pylab.rcParams.update(pylab_params)
    sys.setrecursionlimit(int(1e6))

    # ---- extract data arrays ----
    hjds1   = np.array(data[TIME_STAMPS])
    v1s     = np.array(data[RADIAL_VELS])
    errv1s  = np.abs(data[ERRORS])

    # ---- null hypothesis: constant velocity fit ----
    if null_hyp:
        # weighted mean as best-fit constant velocity
        gamma0   = get_rv_weighted_mean(data)
        # compute total chi²
        chisq    = np.sum(((v1s - gamma0) / errv1s)**2)
        dof      = len(v1s) - 1   # one fitted parameter (gamma)
        redchi   = chisq / dof
        aic,bic = compute_AIC_BIC(v1s, errv1s, chisq, 1)
        # report
        # print(f"[{star_name}] Null hypothesis fit → γ = {gamma0:.5g}, "
        #       f"χ²_red = {redchi:.3f}")
        return SimpleNamespace(gamma=gamma0, redchi=redchi, aic=aic, bic=bic)

    # ---- full orbital fit ----
    lmfit_params_dict = args_dict[LMFIT_PARAMS]
    mini_method       = lmfit_params_dict[MINI_METHOD]
    params            = lmfit.Parameters()
    search_params     = lmfit_params_dict[SEARCH_REGION]

    # define your five orbital parameters
    define_search_param(params, search_params, PERIOD)
    define_search_param(params, search_params, GAMMA)
    define_search_param(params, search_params, K1_STR)
    define_search_param(params, search_params, OMEGA)
    define_search_param(params, search_params, ECC)
    t0_max = search_params[PERIOD][MAX_VAL]
    define_search_param(params, search_params, T, t0_max)

    mini = lmfit.Minimizer(
        chisqr, params,
        fcn_kws={TIME_STAMPS: hjds1,
                 RADIAL_VELS: v1s,
                 ERRORS: errv1s}
    )
    result = mini.minimize(method=mini_method)

    dump_minimization_report(result, output, "lmfit")
    return result


def Kepler(E, M, ecc):
    """
    Converts mean anomalies to eccentric anomalies using the Kepler's equation.

    Args:
        E (numpy.ndarray): Initial guess for the eccentric anomalies.
        M (numpy.ndarray): Mean anomalies.
        ecc (float): Eccentricity of the orbit.

    Returns:
        numpy.ndarray: Calculated eccentric anomalies.

    Notes:
        - The function uses an iterative method to solve Kepler's equation: M = E - ecc * sin(E).
        - If the solution does not converge within 990 iterations, the function resets the initial guesses and continues.
        - A convergence message is printed if the solution does not converge after a reset.
        - Convergence is achieved when the change in eccentric anomaly is less than 1E-7.
    """
    counter = 0
    conversion_count = 1
    loop_E = E
    while True:
        if counter > 10000:
            loop_E = np.random.rand(len(M)) * np.pi
            counter = 0
            print("did not converge {} times".format(conversion_count))
            if conversion_count > 30:
                return np.full(loop_E.shape, np.nan)
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
    """
    Calculates the modeled radial velocity (v1) given the true anomaly and orbital parameters.

    Args:
        nu (float or numpy.ndarray): True anomaly.
        gamma1 (float): Systemic velocity.
        k1 (float): Radial velocity semi-amplitude.
        omega (float): Argument of periapsis.
        ecc (float): Orbital eccentricity.

    Returns:
        float or numpy.ndarray: Modeled radial velocity (v1).

    Notes:
        - The function calculates the radial velocity using the formula:
          v1 = gamma1 + k1 * (cos(omega + nu) + ecc * cos(omega)).
        - The input `nu` can be a single value or an array of values.
    """
    v1 = gamma1 + k1 * (np.cos(omega + nu) + ecc * np.cos(omega))
    return v1


def nus1(hjds, P, T0, ecc):
    """
    Calculates the true anomalies (nu) from heliocentric Julian dates (hjds) and orbital parameters.

    Args:
        hjds (numpy.ndarray): Array of heliocentric Julian dates.
        P (float): Orbital period.
        T0 (float): Time of periastron passage.
        ecc (float): Orbital eccentricity.

    Returns:
        numpy.ndarray: Array of true anomalies (nu).

    Notes:
        - The function calculates the orbital phase (phis) from the input dates.
        - Mean anomalies (Ms) are derived from the phases.
        - Eccentric anomalies (Es) are obtained using the Kepler function.
        - True anomalies (nusdata) are calculated from the eccentric anomalies.
    """
    phis = (hjds - T0) / P - ((hjds - T0) / P).astype(int)
    phis[phis < 0] = phis[phis < 0] + 1.
    Ms = 2 * np.pi * phis
    Es = Kepler(np.pi, Ms, ecc)
    eccfac = np.sqrt((1 + ecc) / (1 - ecc))
    nusdata = 2. * np.arctan(eccfac * np.tan(0.5 * Es))
    return nusdata


# Target function (returns differences between model array with parameter set p and data)
def chisqr(p, **kws):
    """
    Computes the chi-squared values, which represent the differences between the modeled and observed radial velocities.

    Args:
        p (Parameters): A parameters object containing the model parameters. It should include:
            - GAMMA: Systemic velocity parameter.
            - K1_STR: Radial velocity semi-amplitude parameter.
            - OMEGA: Argument of periapsis parameter.
            - ECC: Eccentricity parameter.
            - T: Time of periastron passage parameter.
            - PERIOD: Orbital period parameter.
        **kws: Additional keyword arguments containing the data and error arrays:
            - TIME_STAMPS (numpy.ndarray): Array of observation time stamps.
            - RADIAL_VELS (numpy.ndarray): Array of observed radial velocities.
            - ERRORS (numpy.ndarray): Array of errors in the observed radial velocities.

    Returns:
        numpy.ndarray: Array of normalized differences between the modeled and observed radial velocities.

    Notes:
        - The function extracts the necessary parameters from the `p` object.
        - It calculates the modeled radial velocities using the `v1mod` function and the true anomalies obtained from the `nus1` function.
        - The differences between the modeled and observed radial velocities are normalized by the observational errors.
    """
    Gamma1 = p[GAMMA].value
    K1 = p[K1_STR].value
    Omega = p[OMEGA].value
    ecc = p[ECC].value
    T0 = p[T].value
    P = p[PERIOD].value
    v1 = v1mod(nus1(kws[TIME_STAMPS], P, T0, ecc), Gamma1, K1, Omega, ecc)
    return (v1 - kws[RADIAL_VELS]) / kws[ERRORS]


def dump_minimization_report(res, output, file_prefix):
    lines = []
    best_fit_params = res.params
    lines.append("Best-fit parameter values:\n")
    for name, param in best_fit_params.items():
        lines.append(f"{name}: {param.value}\n")

    # Minimized value of the objective function
    minimized_value = res.chisqr
    lines.append(f"Minimized chi-square value: {minimized_value}\n")

    # Reduced chi-square value
    reduced_chi_square = res.redchi
    lines.append(f"Reduced chi-square value: {reduced_chi_square}\n")

    # Success of the optimization
    success = res.success
    lines.append(f"Optimization success: {success}\n")

    # Fit statistics
    aic = res.aic
    bic = res.bic
    lines.append(f"AIC: {aic}, BIC: {bic}\n")

    # Covariance matrix
    covariance_matrix = res.covar
    lines.append(f"Covariance matrix: \n{covariance_matrix}\n")

    # Detailed fit report
    lines.append("Fit report:\n")
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        lmfit.report_fit(res)
    lines.append(report.getvalue())

    with open(os.path.join(output, "{}_fit_report.txt".format(file_prefix)), "w") as file:
        file.writelines(lines)



def corner_plot(args_dict, data, mini_results,output, truths=None):
    """
    Generates a corner plot for the given data and fit results using the specified parameters.

    Args:
        args_dict (dict): A dictionary containing the corner plot parameters. It should have the following structure:
            - CORNER_PARAMS: A dictionary containing parameters for the corner plot and minimization process:
                - LN_SIGMA: A dictionary with the following keys:
                    - INIT_VAL (float): Initial value for the log of sigma.
                    - MIN_VAL (float): Minimum value for the log of sigma.
                    - MAX_VAL (float): Maximum value for the log of sigma.
                - CORNER_METHOD (str): The minimization method to be used.
                - NAN_POLICY (str): Policy for handling NaN values.
                - BURN (int): Number of burn-in steps.
                - STEPS (int): Number of MCMC steps.
                - THIN (int): Thinning factor for MCMC sampling.
        data (dict): A dictionary containing the data for the minimization process.
        mini_results (MinimizerResult): The initial minimization results object.
        truths (dict, optional): A dictionary containing the true values of the parameters for reference in the corner plot. Default is None.

    Returns:
        None: The function generates and displays a corner plot.

    Notes:
        - The function first adds the log of sigma parameter to the minimization results.
        - It performs the minimization using the `lmfit.minimize` function.
        - The resulting chain of parameter samples is used to generate a corner plot.
        - If `truths` is provided, it will be used to indicate the true parameter values on the corner plot.
    """
    corner_params = args_dict[CORNER_PARAMS]
    corner_params_obj = mini_results.params.copy()

    corner_params_obj.add(LN_SIGMA, value=np.log(corner_params[LN_SIGMA][INIT_VAL]),
                            min=np.log(corner_params[LN_SIGMA][MIN_VAL]),
                            max=np.log(corner_params[LN_SIGMA][MAX_VAL]))

    # Set the specific parameter's vary to True
    corner_params_obj[PERIOD].set(vary=True)
    corner_params_obj[PERIOD].stderr = corner_params_obj[PERIOD].value * 0.05
    # for name, p in corner_params_obj.items():
    #     if not p.vary:
    #         p.set(vary=True)
    #     # Expand bounds around best value using stderr (3σ as an example)
    #     if p.stderr is not None and p.stderr > 0:
    #         p_min = p.value - 3 * p.stderr
    #         p_max = p.value + 3 * p.stderr
    #         p.set(min=p_min, max=p_max)
    #     elif name != ECC:
    #         # If no stderr is available, provide a fallback region
    #         p.set(min=p.value *0.95, max=p.value *1.05)
    # for name, p in corner_params_obj.items():
    #     print(f"{name}: value={p.value:.4f}, stderr={p.stderr}, bounds=({p.min}, {p.max})")

    # Then use this updated Parameters object
    res = lmfit.minimize(
        chisqr,
        kws=data,
        method=corner_params[CONRER_METHOD],
        nan_policy=corner_params[NAN_POLICY],
        burn=corner_params[BURN],
        steps=corner_params[STEPS],
        thin=corner_params[THIN],
        params=corner_params_obj,
        is_weighted=True,
        progress=True
    )


    dump_minimization_report(res, output, "corner")

    prediction = [res.params[PERIOD].value,
                  res.params[GAMMA].value,
                  res.params[K1_STR].value,
                  res.params[OMEGA].value,
                  res.params[ECC].value,
                  res.params[T].value,
                  0]

    figure = corner.corner(res.flatchain, truths=prediction, truth_color='red', labels=res.var_names)
    best_fit_params = res.params.copy()
    best_fit_list = []
    # Get the flat MCMC chain
    chain = res.flatchain  # shape: (n_samples, n_parameters)
    var_names = res.var_names

    # Loop through parameters and update with median and stderr
    for name in var_names:
        samples = chain[name]  # samples for this parameter
        median = np.median(samples)
        stderr = np.std(samples, ddof=1)
        best_fit_params[name].set(value=median, stderr=stderr)
        best_fit_list.append(best_fit_params[name].value)
    # Add the second set of truths manually
    num_of_params = len(res.var_names)
    axes = np.array(figure.axes).reshape((num_of_params, num_of_params))
    for i in range(num_of_params):
        for j in range(i + 1):
            ax = axes[i, j]
            ax.axvline(best_fit_list[j], color='blue', linestyle='--')
            if i != j:
                ax.axhline(best_fit_list[i], color='blue', linestyle='--')
                ax.plot(best_fit_list[j], best_fit_list[i], 'bo')

    # Add the second set of truths manually
    if truths:
        num_of_params = len(res.var_names)
        axes = np.array(figure.axes).reshape((num_of_params, num_of_params))
        for i in range(num_of_params):
            for j in range(i + 1):
                ax = axes[i, j]
                ax.axvline(truths[j], color='blue', linestyle='--')
                if i != j:
                    ax.axhline(truths[i], color='blue', linestyle='--')
                    ax.plot(truths[j], truths[i], 'bo')

    # Add a title to the corner plot
    figure.suptitle("MCMC corner plot\nred cross is prediction\nblue cross Are the Truths")

    # Show the plot
    plt.show()


def main():
    """
    Main function to run the script, which can generate or load sample data, and perform analysis.

    This script uses argparse to handle command-line arguments. It can either generate sample data
    based on JSON parameters or load existing sample data from a JSON file, and then perform
    minimization and generate corner plots.

    Command-line Arguments:
        --output_dir, -o: The directory where output files will be saved. Required.
        --json_params_file, -j: Path to the JSON file containing parameters. Required.
        --generate_sample, -g: If specified, generates sample data based on the JSON parameters.
        --load_sample, -l: Directory path to the JSON file containing sample data to load.
        --seed, -s: Seed for random number generation to ensure consistent outputs for debugging. Default is 1234.

    Workflow:
        1. Parses command-line arguments.
        2. Sets the random seed for reproducibility.
        3. Checks if the output directory exists and is empty or creates it.
        4. Loads parameters from the specified JSON file.
        5. Generates or loads sample data as specified by the arguments.
        6. Performs minimization on the sample data.
        7. Generates and displays a corner plot of the results.
    """
    parser = argparse.ArgumentParser(description="A simple example of argparse")

    # Add arguments
    parser.add_argument('--output_dir', '-o', type=str, help='The output dir', required=True)
    parser.add_argument('--json_params_file', '-j', type=str, help='json format parameters', required=True)
    parser.add_argument('--generate_sample', '-g', help='should Generate', action='store_true')
    parser.add_argument('--load_sample', '-l', type=str, help='directory to samples json file')
    parser.add_argument('--seed', '-s', type=int, help='seed for consistent outputs for debug', default=1234)
    # Parse the arguments
    args = parser.parse_args()

    np.random.seed(args.seed)
    np.random.default_rng(seed=args.seed)

    # if os.path.isdir(args.output_dir):
    #
    #     if  len(os.listdir(args.output_dir)):
    #         print("outDir is not empty. choose different out")
    #         exit(1)
    # else:
    #     os.makedirs(os.path.join(args.output_dir))

    make_runlog(args.output_dir, [args.json_params_file])

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

        with open(samples_path, 'w') as json_file:
            json.dump(data, json_file)
    elif args.load_sample:
        samples_path = args.load_sample
    else:
        print("no data was chosen. Either generate sample or provide a sample json path")
        exit(1)

    print("Uploading Sample from file: {}".format(samples_path))
    if samples_path.endswith(".json"):
        with open(samples_path, 'r') as json_file:
            data = json.load(json_file)
    elif samples_path.endswith(".csv"):
        data = pd.read_csv(samples_path, sep=' ')
        data.rename(columns={'MJD': TIME_STAMPS,
                           'Mean RV': RADIAL_VELS,
                           'Mean RVsig': ERRORS}, inplace=True)

    else:
        print("Unknown input data format")
        exit(1)
    mini_results = lmfit_on_sample(args_dict, args.output_dir, data)

    if TRUTHS in data.keys():
        corner_plot(args_dict, data, mini_results, args.output_dir, data[TRUTHS])
    else:
        corner_plot(args_dict, data, mini_results, args.output_dir)
    exit(0)


if __name__ == '__main__':
    main()

