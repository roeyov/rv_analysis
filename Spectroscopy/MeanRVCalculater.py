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


import pandas as pd
import numpy as np


def calculate_weighted_rv_with_flags(df):
    """
    For each MJD (each row in the dataframe), calculate the weighted mean RV
    (and its uncertainty) using the following steps:

      1. For each measurement i, ignore the measurement if its equivalent width (EW)
         is "consistent with zero" (i.e. if the interval EW_i ± EW_i_sig contains zero).
         Flag these measurements with a flag value of 2.
      2. Using the remaining measurements, compute a first-iteration weighted mean RV,
         where the weight for measurement i is 1/(RV_i_sig)^2.
      3. Flag any measurement (from the candidate list) that deviates by more than
         3σ from the first-iteration weighted mean with a flag value of 1.
      4. Compute the final weighted mean RV and its uncertainty using only the measurements
         that were not flagged by sigma clipping (i.e. that have flag 0).
      5. Add two new columns to the dataframe: "RV_Mean" and "RV_Meansig" which hold the
         final weighted mean RV and its uncertainty, respectively.
      6. For each measurement sample (i), add a new column "RV_{i}_flag" that indicates
         whether that measurement was:
             0 - used (valid)
             1 - dropped due to sigma clipping
             2 - dropped because EW is consistent with zero (or missing)

    The input dataframe is expected to have the following columns:
        - "MJD"
        - For each measurement index i (from 0 to N):
            "RV_{i}", "RV_{i}sig", "EW_{i}", "EW_{i}sig"

    Args:
        df (pd.DataFrame): Input dataframe with the required columns.

    Returns:
        pd.DataFrame: The dataframe with new columns "RV_Mean", "RV_Meansig" and
                      "RV_{i}_flag" for each measurement.
    """
    # Determine measurement indices by scanning for columns starting with "RV_"
    # (excluding our output columns that may contain "Mean" or "flag").
    measurement_indices = []
    for col in df.columns:
        if col.endswith(" RV") and ("Mean" not in col) and ("flag" not in col) and ('merged' not in col):
            try:
                idx = col.split(' ')[0]
                measurement_indices.append(idx)
            except (IndexError, ValueError):
                continue
    measurement_indices = sorted(set(measurement_indices))

    # Initialize dictionaries to store the flag for each measurement index (per row)
    flags_dict = {i: [] for i in measurement_indices}

    # Lists to hold final weighted mean results per row.
    rv_mean_list = []
    rv_meansig_list = []
    rv_median_list = []

    # Process each row (each MJD)
    for _, row in df.iterrows():
        row_flags = {}  # Will hold the flag for each measurement in this row.
        candidates = []  # List of candidates: tuples of (i, rv, rvsig)

        # Step 1: Process each measurement and flag based on EW.
        for i in measurement_indices:
            rv_val = row.get(f"{i} RV")
            rvsig_val = row.get(f"{i} RVsig")
            ew_val = row.get(f"{i} EW")
            ewsig_val = row.get(f"{i} EWsig")

            # If any value is missing, treat it as an invalid measurement (flag 2)
            if pd.isna(rv_val) or pd.isna(rvsig_val) or pd.isna(ew_val) or pd.isna(ewsig_val):
                row_flags[i] = 2
                continue

            # Check if the equivalent width is "consistent with zero":
            # If the interval [EW - EWsig, EW + EWsig] contains zero, drop it.
            if (ew_val - ewsig_val) <= 0 <= (ew_val + 2*ewsig_val):
                row_flags[i] = 2
                continue

            # Otherwise, mark as tentatively valid (flag 0) and add to candidate list.
            row_flags[i] = 0
            candidates.append((i, rv_val, rvsig_val))

        # If no candidates survive the EW filter, record NaN for the weighted mean.
        if len(candidates) == 0:
            rv_mean_list.append(np.nan)
            rv_meansig_list.append(np.nan)
            rv_median_list.append(np.nan)
        else:
            # Step 2: First iteration weighted mean using candidate measurements.
            weights = [1 / (rvsig ** 2) for (_, rv, rvsig) in candidates]
            weighted_mean = sum(rv * w for (_, rv, rvsig), w in zip(candidates, weights)) / sum(weights)
            median = np.median([rv for (_, rv, _), w in zip(candidates, weights)])
            weighted_error = np.sqrt(1 / sum(weights))

            # Step 3: Sigma clipping: flag measurements deviating more than 3 sigma.
            candidates_clipped = []
            for (i, rv, rvsig) in candidates:
                if abs(rv - weighted_mean) > 3 * rvsig:
                    row_flags[i] = 1  # Dropped due to sigma clipping.
                else:
                    candidates_clipped.append((i, rv, rvsig))

            # Step 4: Calculate final weighted mean from the non-clipped candidates.
            if len(candidates_clipped) == 0:
                final_mean = np.nan
                final_median = np.nan
                final_error = np.nan
            else:
                final_weights = [1 / (rvsig ** 2) for (_, rv, rvsig) in candidates_clipped]
                final_mean = sum(rv * w for (_, rv, rvsig), w in zip(candidates_clipped, final_weights)) / sum(
                    final_weights)
                # Calculate the weighted mean
                average = final_mean
                # Calculate the weighted variance
                values =  np.array([rv for (_, rv, _) in candidates_clipped])
                variance = np.average((values - average) ** 2, weights=final_weights)
                # Return the square root of the variance
                final_error = np.sqrt(variance)
                # final_error = np.sqrt(1 / sum(final_weights))

                final_median = np.median([rv for (_, rv, _), w in zip(candidates_clipped, final_weights)])
            rv_mean_list.append(final_mean)
            rv_meansig_list.append(final_error)
            rv_median_list.append(final_median)


            # rv_mean_list.append(weighted_mean)
            # rv_meansig_list.append(weighted_error)
            # rv_median_list.append(median)

        # Append the computed flags for this row into the flags_dict.
        for i in measurement_indices:
            # If a measurement index is missing from row_flags, default to flag 2.
            flags_dict[i].append(row_flags.get(i, 2))

    # Step 5: Add the final weighted mean columns.
    df["Mean RV"] = rv_mean_list
    df["Median RV"] = rv_median_list
    df["Mean RVsig"] = rv_meansig_list

    # Add flag columns for each measurement sample.
    for i in measurement_indices:
        df[f"{i} flag"] = flags_dict[i]

    return df
# Example usage:
# Assuming you have a dataframe `df` with the proper columns:
# df = pd.read_csv("your_file.csv")
# df = calculate_weighted_rv(df)
# df.to_csv("output_with_weighted_rv.csv", index=False)
