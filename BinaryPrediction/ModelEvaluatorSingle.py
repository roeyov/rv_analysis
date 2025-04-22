import numpy as np
import pandas as pd
import re
import os
import json
import glob

from BinaryPrediction.Models import significance
from utils.constants import *
from utils.periodagram import ls, pdc, plotls
from emceeOmLMFITexe import lmfit_on_sample, corner_plot

def load_final_data_from_ccf_out(data):
    rvs = data['Mean RV']
    mjds = data['MJD']
    err_vs = data['Mean RVsig']
    return rvs, mjds, err_vs

def find_period(rvs, mjds, err_vs, star_name):
    pmin = 1.0
    pmax = 1000.0

    period, fap, fal, freq, pow = ls(mjds, rvs,data_err=err_vs, pmin=pmin, pmax=pmax)
    if period <= pmin or period >= pmax:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    plotls(freq, pow, fal, pmin=pmin, pmax=pmax, star_id=star_name +'_LS')
    best_period1, fap1, freq, pdc_power_reg = pdc(mjds  , rvs, data_err=err_vs, pmin=pmin, pmax=pmax)
    plotls(freq, pdc_power_reg, fal=[] , pmin=pmin, pmax=pmax, star_id=star_name+'_PDC')
    return period, fap, fal, best_period1, fap1

def get_bloem_object_name(path_to_csv):
    pattern = r'\d-\d{3}'
    match = re.search(pattern, path_to_csv)
    if match:
        return match.group(0)
    else:
        return 'No_named_object'

def binary_rv_threshold(rvs, err_vs, drv_tresh=20, sign_threshold=4):
    np_rvs = np.array(rvs).flatten()
    err_vs = np.array(err_vs).flatten()
    err_vs_sq = err_vs * err_vs

    pairwise_diff = np.abs(np_rvs[:, None] - np_rvs[None, :])
    pairwise_sigma_sq = np.sqrt(err_vs_sq[:, None] + err_vs_sq[None, :])

    signif = pairwise_diff/pairwise_sigma_sq

    return np.any((signif > sign_threshold) & (pairwise_diff > drv_tresh))

def main_single(path_to_csv):
    JSON_PARAM_FILE = '/Users/roeyovadia/Roey/Masters/Reasearch/Scripts/params.json'
    data = pd.read_csv(path_to_csv, sep=' ')
    star_name = get_bloem_object_name(path_to_csv)

    rvs, mjds, err_vs = load_final_data_from_ccf_out(data)
    ls_p, ls_fap, ls_fal, pdc_p, pdc_fap = find_period(rvs, mjds, err_vs, star_name)
    if np.isnan(ls_fap):
        print("no period found")
        return 0
    out_dir = os.path.dirname(path_to_csv)
    with open(JSON_PARAM_FILE, 'r') as json_file:
        args_dict = json.load(json_file)

    chosen_period = pdc_p

    args_dict[LMFIT_PARAMS][SEARCH_REGION][PERIOD][INIT_VAL] = chosen_period
    args_dict[LMFIT_PARAMS][SEARCH_REGION][PERIOD][MIN_VAL] = chosen_period - 0.1
    args_dict[LMFIT_PARAMS][SEARCH_REGION][PERIOD][MAX_VAL] = chosen_period + 0.1
    args_dict[LMFIT_PARAMS][SEARCH_REGION][PERIOD][VARY] = False

    data.rename(columns={'MJD': TIME_STAMPS,
                         'Mean RV': RADIAL_VELS,
                         'Mean RVsig': ERRORS}, inplace=True)

    mini_results = lmfit_on_sample(args_dict, out_dir, data, star_name)
    is_binary_th = binary_rv_threshold(rvs, err_vs, drv_tresh=20, sign_threshold=4)
    is_binary_periodic_ls = ls_fap < 0.001
    is_binary_periodic_pdc = pdc_fap < 0.001

    return  (is_binary_periodic_pdc << 2) | (is_binary_periodic_ls << 1) | is_binary_th


def main_multiple(path_to_ccf_out_dir):
    binarity_counter =[0,0,0,0]
    decision_results = []
    list_of_csvs = glob.glob(os.path.join(path_to_ccf_out_dir, '*_CCF_RVs.csv'))
    for csv in list_of_csvs:
        print(csv)

        path_to_csv = os.path.join(path_to_ccf_out_dir, csv)
        ret_val = main_single(path_to_csv)
        binarity_counter[ret_val] += 1
        decision_results.append([get_bloem_object_name(path_to_csv),ret_val])
    print(binarity_counter)
    return pd.DataFrame(decision_results, columns=['BLOEM OBJECT', 'DECISION RESULTS'])


if __name__ == '__main__':

    import sys

    # Open a file for writing
    log_file = open("output.log", "w")

    # Redirect stdout to the file
    sys.stdout = log_file


    # path_to_input = '/Users/roeyovadia/Roey/Masters//Reasearch/scriptsOut/CCF/ostars_sb1_from_coAdded/'
    path_to_input = '/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/sb1_ostars_coAdded/BLOeM_5-097_CCF_RVs.csv'
    if os.path.isdir(path_to_input):
        main_multiple(path_to_input)
    elif os.path.isfile(path_to_input):
        print("Decision Flag Is: ",main_single(path_to_input))

    else:
        print('Please provide a valid path')
        exit(1)

    # Don't forget to close the file when you're done
    log_file.close()
