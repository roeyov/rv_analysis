"""
pipeline.data_loading — CSV loading, column renaming, and data preparation.
"""

import re

import numpy as np
import pandas as pd

from utils.constants import RADIAL_VELS, TIME_STAMPS, ERRORS, SNR_PPL


def load_final_data_from_ccf_out(data):
    """Extract RV, time, and error arrays from a DataFrame with internal column names."""
    rvs = data[RADIAL_VELS]
    mjds = data[TIME_STAMPS]
    err_vs = data[ERRORS]
    return rvs, mjds, err_vs


def get_bloem_object_name(path_to_csv):
    """Derive BLOeM object name from a CSV file path."""
    # Pattern for original objects (e.g., 1-123)
    pattern_original = r'\d-\d{3}'
    # Pattern for simulation objects (e.g., SIMuLaTioN_42)
    pattern_sim = r'SIMuLaTioN_(\d+)'

    match_orig = re.search(pattern_original, path_to_csv)
    match_sim = re.search(pattern_sim, path_to_csv)

    if match_orig:
        return 'BLOeM_' + match_orig.group(0)
    elif match_sim:
        # group(1) returns just the digits inside the parentheses
        return 'BLOeM_SIM_' + match_sim.group(1)
    else:
        return 'No_named_object'


def _load_and_clean_csv(path_to_csv):
    """Read CSV, drop rows with missing RV/time/sigma, return cleaned df and dropped info."""
    data = pd.read_csv(path_to_csv, sep=',')
    mask = data[['Mean RV', 'MJD', 'Mean RVsig']].isna().any(axis=1)
    if mask.any():
        print("Dropped indices:", data.index[mask].tolist())
        print("Causes:\n", data.loc[mask, ['Mean RV', 'MJD', 'Mean RVsig']].isna())
    data = data.dropna(subset=['Mean RV', 'MJD', 'Mean RVsig']).reset_index(drop=True)
    # data = data[data.MJD <60300]
    # data = data.iloc[9:].reset_index(drop=True)
    return data


def _rename_to_internal_cols(data):
    """Standardize column names to internal constants."""
    data.rename(columns={'MJD': TIME_STAMPS,
                         'Mean RV': RADIAL_VELS,
                         'Mean RVsig': ERRORS}, inplace=True)


def _prepare_inputs_for_period_search(data):
    """Return arrays used across the pipeline (RV, time, errors, SNR medians)."""
    rvs, mjds, err_vs = load_final_data_from_ccf_out(data)
    median_snr_calc = np.median(data["SNR"])
    median_snr_obs  = np.median(data[SNR_PPL])
    return rvs, mjds, err_vs, median_snr_calc, median_snr_obs
