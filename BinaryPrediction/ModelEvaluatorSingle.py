from idlelib.run import flush_stdout

import numpy as np
import pandas as pd
import re
import os
import json
import glob
from scipy.signal import find_peaks

from h5py.h5f import flush

from BinaryPrediction.Models import significance
from utils.constants import *
from utils.periodagram import ls, pdc, plotls
from emceeOmLMFITexe import lmfit_on_sample, corner_plot, print_lmfit_result, summarize_result, get_rv_weighted_mean

from enum import Enum

class DecsionFlags(Enum):
    DELTA_RV_BIN   = 0
    LOMB_SCARGLE_BIN = 1
    PDC_BIN  = 2


def load_final_data_from_ccf_out(data):
    rvs = data['Mean RV']
    mjds = data['MJD']
    err_vs = data['Mean RVsig']
    return rvs, mjds, err_vs

def find_periods(rvs, mjds, err_vs, star_name):
    pmin = 1.0
    pmax = 1000.0

    period, fap, fal, freq, pow = ls(mjds, rvs,data_err=err_vs, pmin=pmin, pmax=pmax)
    ls_sig_periods = significant_periods(1/freq,pow)
    if period <= pmin or period >= pmax:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    best_period1, fap1, freq, pdc_power_reg = pdc(mjds  , rvs, data_err=err_vs, pmin=pmin, pmax=pmax)

    pdc_sig_periods = significant_periods(1/freq,pdc_power_reg)
    return ls_sig_periods + pdc_sig_periods

def find_period(rvs, mjds, err_vs, star_name):
    pmin = 1.0
    pmax = 1000.0

    period, fap, fal, freq, pow = ls(mjds, rvs,data_err=err_vs, pmin=pmin, pmax=pmax)
    if period > pmin and period < pmax:
        plotls(freq, pow, fal, pmin=pmin, pmax=pmax, star_id=star_name +'_LS')
    else:
        period =  fap = fal = np.nan
    best_period1, fap1, freq, pdc_power_reg = pdc(mjds  , rvs, data_err=err_vs, pmin=pmin, pmax=pmax)
    if best_period1 > pmin and best_period1 < pmax:
        plotls(freq, pdc_power_reg, fal=[] , pmin=pmin, pmax=pmax, star_id=star_name+'_PDC')
    else:
        best_period1 =  fap1 = np.nan
    return period, fap, fal, best_period1, fap1

def get_bloem_object_name(path_to_csv):
    pattern = r'\d-\d{3}'
    match = re.search(pattern, path_to_csv)
    if match:
        return match.group(0)
    else:
        return 'No_named_object'


def significant_periods(periods, powers,
                        max_periods=5,
                        min_separation=1.2):
    """
    Identify up to `max_periods` significant periods from a power spectrum,
    ensuring selected periods differ by at least a factor `min_separation`.

    Separation check is done in log-space:
        |ln P_i - ln P_j| >= ln(min_separation)

    Parameters
    ----------
    periods : array_like, shape (N,)
        Array of period values (need not be evenly spaced).
    powers : array_like, shape (N,)
        Corresponding power values.
    max_periods : int, optional
        Maximum number of periods to return (default: 5).
    min_separation : float, optional
        Minimum multiplicative separation between any two returned periods
        (default: 1.2 means at least ±20% apart).

    Returns
    -------
    selected_periods : list of float
        The most significant periods, sorted by descending power, pruned
        to enforce the minimum log-space separation criterion.
    """
    periods = np.asarray(periods)
    powers = np.asarray(powers)

    # 1. Find all local peaks
    peak_idx, _ = find_peaks(powers)
    pk_periods = periods[peak_idx]
    pk_powers = powers[peak_idx]

    # 2. Sort peaks by descending power
    order = np.argsort(pk_powers)[::-1]

    # 3. Precompute logs and threshold
    log_periods = np.log(pk_periods)
    log_thresh = np.log(min_separation)

    # 4. Greedily pick peaks, enforcing log-space separation
    selected = []
    selected_logs = []
    for i in order:
        lp = log_periods[i]
        # check against all already-selected peaks
        if all(abs(lp - sl) >= log_thresh for sl in selected_logs):
            selected.append(pk_periods[i])
            selected_logs.append(lp)
            if len(selected) >= max_periods:
                break

    return selected

def binary_rv_threshold(rvs, err_vs, drv_tresh=20, sign_threshold=4):
    np_rvs = np.array(rvs).flatten()
    err_vs = np.array(err_vs).flatten()
    err_vs_sq = err_vs * err_vs

    pairwise_diff = np.abs(np_rvs[:, None] - np_rvs[None, :])
    pairwise_sigma_sq = np.sqrt(err_vs_sq[:, None] + err_vs_sq[None, :])

    signif = pairwise_diff/pairwise_sigma_sq

    return np.any((signif > sign_threshold) & (pairwise_diff > drv_tresh))

def change_search_region_default(args_dict, field_name, init_val, min_val, max_val, vary):
    args_dict[LMFIT_PARAMS][SEARCH_REGION][field_name][INIT_VAL] = init_val
    args_dict[LMFIT_PARAMS][SEARCH_REGION][field_name][MIN_VAL] = min_val
    args_dict[LMFIT_PARAMS][SEARCH_REGION][field_name][MAX_VAL] = max_val
    args_dict[LMFIT_PARAMS][SEARCH_REGION][field_name][VARY] = vary



def main_single(path_to_csv):
    JSON_PARAM_FILE = '/Users/roeyovadia/Roey/Masters/Reasearch/Scripts/params.json'
    data = pd.read_csv(path_to_csv, sep=' ')
    star_name = get_bloem_object_name(path_to_csv)

    rvs, mjds, err_vs = load_final_data_from_ccf_out(data)
    ls_p, ls_fap, ls_fal, pdc_p, pdc_fap = find_period(rvs, mjds, err_vs, star_name)
    out_dir = os.path.dirname(path_to_csv)
    with open(JSON_PARAM_FILE, 'r') as json_file:
        args_dict = json.load(json_file)
    possible_periods = find_periods(rvs, mjds, err_vs, star_name)
    cur_red_chi = 1e9
    best_period = best_result = None
    data.rename(columns={'MJD': TIME_STAMPS,
                         'Mean RV': RADIAL_VELS,
                         'Mean RVsig': ERRORS}, inplace=True)

    change_search_region_default(args_dict, GAMMA, get_rv_weighted_mean(data), min(rvs), max(rvs), True)

    for chosen_period in possible_periods:
        if np.isnan(ls_fap):
            break
        change_search_region_default(args_dict, PERIOD, chosen_period, chosen_period - 0.1, chosen_period + 0.1, False)

        mini_results = lmfit_on_sample(args_dict, out_dir, data, star_name)
        if abs(mini_results.redchi -1) < abs(cur_red_chi-1):
            cur_red_chi = mini_results.redchi
            best_period = chosen_period
            best_result = mini_results
    ## attempt Null Hypothesis
    change_search_region_default(args_dict, PERIOD, 0, -0.1,  0.1, False)
    change_search_region_default(args_dict, K1_STR, 0, -0.1,  0.1, False)
    mini_results = lmfit_on_sample(args_dict, out_dir, data, star_name, null_hyp=True)
    if abs(mini_results.redchi - 1) < abs(cur_red_chi - 1):
        cur_red_chi = mini_results.redchi
        best_period = 0
        best_result = mini_results
    ## end Null Hypothesis

    if hasattr(best_result, 'redchi') and not hasattr(best_result, 'params'):
        # it’s the null-hyp branch
        row = {
            'star_name': star_name,
            'method':    'null_hyp',
            'nfev':      None,
            'ndata':     len(data[TIME_STAMPS]),
            'nvarys':    1,
            'chisqr':    (best_result.redchi * (len(data[TIME_STAMPS]) - 1)),
            'redchi':    best_result.redchi,
            'aic':       None,
            'bic':       None,
            'gamma_init': None,
            'gamma_value': best_result.gamma,
            'gamma_vary':  False,
            'gamma_stderr': None,
        }
        print(best_result)
    else:
        row = summarize_result(best_result, star_name)
        print_lmfit_result(data,args_dict,star_name,best_result)
    # print(row)
    is_binary_th = binary_rv_threshold(rvs, err_vs, drv_tresh=20, sign_threshold=4)
    is_binary_periodic_ls = ls_fap < 0.001
    is_binary_periodic_pdc = pdc_fap < 0.001
    if not is_binary_th and (is_binary_periodic_ls or is_binary_periodic_pdc):
        print(f"{star_name} is passed due to periodogram")

    return  row, (is_binary_periodic_pdc << DecsionFlags.PDC_BIN.value) | \
            (is_binary_periodic_ls <<  DecsionFlags.LOMB_SCARGLE_BIN.value) | \
            (is_binary_th << DecsionFlags.DELTA_RV_BIN.value)


def main_multiple(path_to_ccf_out_dir):
    all_rows = []
    binarity_counter =[0,0,0,0,0,0,0,0]
    decision_results = []
    list_of_csvs = glob.glob(os.path.join(path_to_ccf_out_dir, '*_CCF_RVs.csv'))
    for csv in list_of_csvs:
        print(csv)

        path_to_csv = os.path.join(path_to_ccf_out_dir, csv)
        row, ret_val = main_single(path_to_csv)
        print(ret_val)
        binarity_counter[ret_val] += 1
        decision_results.append([get_bloem_object_name(path_to_csv),ret_val])
        all_rows.append(row)
    print(binarity_counter)
    df = pd.DataFrame(all_rows)

    # example: move star_name to front
    cols = ['star_name'] + [c for c in df.columns if c != 'star_name']
    df = df[cols]

    # save to CSV
    df.to_csv("lmfit_summary.csv", index=False)
    return pd.DataFrame(decision_results, columns=['BLOEM OBJECT', 'DECISION RESULTS'])


if __name__ == '__main__':

    import sys

    # Open a file for writing
    log_file = open("output.log", "w")

    # Redirect stdout to the file
    # sys.stdout = log_file


    path_to_input = '/Users/roeyovadia/Roey/Masters//Reasearch/scriptsOut/CCF/ostars_sb1_from_coAdded/'
    # path_to_input = '/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/sb1_ostars_coAdded/BLOeM_5-097_CCF_RVs.csv'
    if os.path.isdir(path_to_input):
        main_multiple(path_to_input)
    elif os.path.isfile(path_to_input):
        print("Decision Flag Is: ",main_single(path_to_input))

    else:
        print('Please provide a valid path')
        exit(1)

    # Don't forget to close the file when you're done
    log_file.close()
