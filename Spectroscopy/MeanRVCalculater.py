#!/usr/bin/env python3
"""
weighted_rv_module.py

This module converts a dictionary of MJD keys with spectral line RV measurements
(and their uncertainties) into a DataFrame and then computes, for each MJD, a
weighted mean RV. Outliers are identified per spectral line if the measurement
deviates from the initial weighted mean by more than sigma_clip times its individual
RV uncertainty. When the verbose flag is set, the module prints which lines were
flagged as outliers for each MJD.

Usage:
    - Import the functions from this module.
    - Or run the module directly to see an example with sample data.
"""

import pandas as pd
import numpy as np


def dict_to_df(data_dict):
    """
    Convert the input dictionary to a DataFrame.
    The keys (MJDs) become a column named "MJD".
    """
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df = df.reset_index().rename(columns={'index': 'MJD'})
    return df


def compute_weighted_mean_row(row, lines, sigma_clip=3, verbose=False):
    """
    For a single row (one MJD), compute the weighted mean of the RV measurements
    from the specified spectral lines.

    Each measurement is weighted by 1/(RVsig). An initial weighted mean is computed,
    and then any measurement whose absolute difference from that mean exceeds
    sigma_clip * (its own uncertainty) is flagged as an outlier and excluded.

    Parameters:
      row        : A row from the DataFrame corresponding to one MJD.
      lines      : List of spectral line names (without the " RV" and " RVsig" suffix).
      sigma_clip : The clipping threshold (default is 3).
      verbose    : If True, prints out which lines were flagged as outliers.

    Returns:
      wm           : The recomputed weighted mean using non-outlier measurements.
      wm_error     : An estimated error on the weighted mean.
      outlier_lines: List of spectral line names flagged as outliers.
    """
    rvs = []
    errors = []
    line_names = []

    for line in lines:
        rv_key = f"{line} RV"
        err_key = f"{line} RVsig"
        ew_key = f"{line} EW"
        if rv_key in row and err_key in row and ew_key in row:
            # Only include if both RV and uncertainty are not NaN.

            if pd.notnull(row[rv_key]) and pd.notnull(row[err_key]) and pd.notnull(row[ew_key]):
                # if row[ew_key] < 0.1: continue
                rvs.append(row[rv_key])
                errors.append(row[err_key])
                line_names.append(line)

    if not rvs:
        return None, None, []

    rvs = np.array(rvs)
    errors = np.array(errors)

    # Compute initial weighted mean with weights = 1 / error.
    weights = 1.0 / errors
    wm_initial = np.sum(rvs * weights) / np.sum(weights)
    # wm_initial = np.nanmedian(rvs)
    # Identify outliers: measurements that deviate more than sigma_clip times their own error.
    mask = np.abs(rvs - wm_initial) <= sigma_clip * errors
    outlier_lines = [line_names[i] for i in range(len(mask)) if not mask[i]]

    # If all measurements are flagged, fall back to using all data.
    if np.sum(mask) == 0:
        mask = np.ones_like(rvs, dtype=bool)
        outlier_lines = []

    rvs_clean = rvs[mask]
    errors_clean = errors[mask]
    weights_clean = 1.0 / errors_clean

    # Recalculate weighted mean with non-outlier data.
    wm = np.sum(rvs_clean * weights_clean) / np.sum(weights_clean)

    # Estimate error on the weighted mean:
    # Compute effective number of measurements.
    n_eff = (np.sum(weights_clean) ** 2) / np.sum(weights_clean ** 2)
    weighted_std = np.sqrt(np.sum(weights_clean * (rvs_clean - wm) ** 2) / np.sum(weights_clean))
    wm_error = weighted_std / np.sqrt(n_eff) if n_eff > 0 else np.nan

    return wm, wm_error, outlier_lines


def process_df(df, sigma_clip=3, verbose=False):
    """
    Process the entire DataFrame to compute the weighted mean RV for each MJD.

    Parameters:
      df         : DataFrame with a column "MJD" and spectral line measurements.
      sigma_clip : Sigma clipping threshold for outlier rejection.
      verbose    : If True, prints which spectral lines were flagged as outliers for each MJD.

    Returns:
      result_df  : A DataFrame with one row per MJD containing:
                   - "MJD"
                   - "weighted_mean_RV"
                   - "weighted_RV_error"
                   - "outliers" (list of spectral lines flagged as outliers)
    """
    # Define the spectral lines (without suffixes) to use.
    lines = ["He I + He II 4026", "He II 4200", "He I 4388", "He I 4471", "He II 4542", "H Gamma","H Delta","H Epsilon"]
    results = []

    for idx, row in df.iterrows():
        mjd = row["MJD"]
        wm, wm_err, outlier_lines = compute_weighted_mean_row(row, lines, sigma_clip, verbose)
        if verbose and outlier_lines:
            print(f"MJD {mjd}: Outliers flagged for lines: {', '.join(outlier_lines)}")
        results.append({
            "MJD": mjd,
            "weighted_mean_RV": wm,
            "weighted_RV_error": wm_err,
            "outliers": outlier_lines
        })

    result_df = pd.DataFrame(results)
    return result_df


if __name__ == "__main__":
    # Example input dictionary in the correct format.
    data = {
        60246.12028475: {
            'He I + He II 4026 RV': 165.9564822111413,
            'He I + He II 4026 RVsig': 11.193757760975938,
            'He II 4200 RV': 138.4306197193985,
            'He II 4200 RVsig': 99.47144700668612,
            'He I 4388 RV': 158.4119210199751,
            'He I 4388 RVsig': 15.05380549035133,
            'He I 4471 RV': 148.15337022507413,
            'He I 4471 RVsig': 10.76720011327219,
            'He II 4542 RV': -7.729746392688202,
            'He II 4542 RVsig': 58.67017259803256,
            'merged RV': 155.61109984358148,
            'merged RVsig': 6.794008596771086
        },
        60248.1702796: {
            'He I + He II 4026 RV': 236.6063665234032,
            'He I + He II 4026 RVsig': 22.579788189829856,
            'He II 4200 RV': 334.1774580171801,
            'He II 4200 RVsig': 63.313003311877914,
            'He I 4388 RV': 297.03339109262726,
            'He I 4388 RVsig': 29.0220972249562,
            'He I 4471 RV': 294.00060729291465,
            'He I 4471 RVsig': 13.726635694082024,
            'He II 4542 RV': 476.1366816570362,
            'He II 4542 RVsig': 40.5581331673549,
            'merged RV': 273.9233058180814,
            'merged RVsig': 10.370221966281585
        },
        60256.1754323: {
            'He I + He II 4026 RV': 151.6887237191828,
            'He I + He II 4026 RVsig': 12.718837382123025,
            'He II 4200 RV': 78.96401658076336,
            'He II 4200 RVsig': 87.53300003852038,
            'He I 4388 RV': 147.6009203005802,
            'He I 4388 RVsig': 16.549016182576874,
            'He I 4471 RV': 121.4263953284827,
            'He I 4471 RVsig': 9.478777550830303,
            'He II 4542 RV': 23.58607051622731,
            'He II 4542 RVsig': 69.17661650201921,
            'merged RV': 135.88715126795924,
            'merged RVsig': 7.167506641453741
        },
        # ... Add additional MJD entries as needed.
    }

    # Convert the dictionary to a DataFrame.
    df = dict_to_df(data)

    # Process the DataFrame to compute weighted means (with sigma clipping and verbose output).
    result_df = process_df(df, sigma_clip=3, verbose=True)

    print("\nWeighted Mean RV per MJD:")
    print(result_df)
