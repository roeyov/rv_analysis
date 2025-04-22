import os
import glob
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_csv_files(csv_dir):
    """
    Retrieve all CSV file paths in the given directory.

    Args:
        csv_dir (str): Path to the directory containing CSV files.

    Returns:
        list: List of file paths matching '*.csv'.
    """
    return glob.glob(os.path.join(csv_dir, '*_CCF_RVs.csv'))


def get_group_from_filename(csv_path):
    """
    Determine the group for a CSV file based on the filename.
    It checks for substrings '_{i}-' for i in 1 to 8.

    For example, the file "BLOeM_1-012_CCF_RVs.csv" will be assigned to group "1"
    because it contains the substring "_1-".

    Args:
        csv_path (str): Path (or filename) of the CSV file.

    Returns:
        str: The group as a string if a match is found, otherwise "Unknown".
    """
    filename = os.path.basename(csv_path)
    for i in range(1, 9):
        if f"_{i}-" in filename:
            return str(i)
    return "Unknown"


def process_csv_file(csv_file):
    """
    Process a single CSV file: calculate mean and median of 'RV' column and determine grouping.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        dict: Dictionary with filename, group, mean_RV, and median_RV.
    """
    df = pd.read_csv(csv_file, sep=' ')
    # Compute the mean and median of the 'RV' column
    try:
        mean_rv = df['Mean RV'].mean()
        median_rv = df['Median RV'].median()
    except KeyError:
        print("{} does not contain 'RV' column.".format(csv_file))
        return

    group = get_group_from_filename(csv_file)

    return {
        'filename': os.path.basename(csv_file),
        'group': group,
        'mean_RV': mean_rv,
        'median_RV': median_rv
    }


def process_all_csv(csv_dir):
    """
    Process all CSV files in a directory and compute statistics with grouping.

    Args:
        csv_dir (str): Directory containing CSV files.

    Returns:
        pd.DataFrame: DataFrame with columns for filename, group, mean_RV, and median_RV.
    """
    csv_files = get_csv_files(csv_dir)
    results = []

    for csv_file in csv_files:
        result = process_csv_file(csv_file)
        results.append(result)

    return pd.DataFrame(results)


def plot_histograms(results_df):
    """
    Plot histograms for mean and median RV values overall and by group.

    This function creates subplots for the overall histogram (combining all CSVs) 
    and separate histograms for each group derived from the filenames.

    Args:
        results_df (pd.DataFrame): DataFrame containing 'mean_RV', 'median_RV', and 'group' columns.
    """
    # Overall data
    all_mean = results_df['mean_RV']
    all_median = results_df['median_RV']

    # Unique groups and calculate subplot layout
    groups = results_df['group'].unique()
    num_groups = len(groups)
    num_plots = num_groups + 1  # +1 for overall histogram
    ncols = 2
    nrows = math.ceil(num_plots / ncols)

    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 4))
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16,9))
    axes = axes

    # Overall histogram
    ax = axes
    _,bins_edges,_ = ax.hist(all_mean, alpha=0.5, label='Mean RV', bins=20)
    ax.hist(all_median, alpha=0.5, label='Median RV', bins=bins_edges)
    mean_all_mean = np.mean(all_mean)
    mean_all_median = np.mean(all_median)
    ax.axvline(mean_all_mean, color='blue', linestyle='dashed', linewidth=1,
               label=f"Mean of Mean RV: {mean_all_mean:.2f}")
    ax.axvline(mean_all_median, color='orange', linestyle='dashed', linewidth=1,
               label=f"Mean of Median RV: {mean_all_median:.2f}")

    ax.set_title("Overall Histogram (All CSVs)")
    ax.set_xlabel("RV values")
    ax.set_ylabel("Frequency")
    ax.set_xlim(50,300)
    ax.legend()

    # # Histograms per group
    # for i, grp in enumerate(groups, start=1):
    #     ax = axes[i]
    #     grp_data = results_df[results_df['group'] == grp]
    #     ax.hist(grp_data['mean_RV'], alpha=0.5, label='Mean RV', bins=bins_edges)
    #     ax.hist(grp_data['median_RV'], alpha=0.5, label='Median RV', bins=bins_edges)
    #     mean_all_mean = np.mean(grp_data['mean_RV'])
    #     mean_all_median = np.mean(grp_data['median_RV'])
    #     ax.axvline(mean_all_mean, color='blue', linestyle='dashed', linewidth=1,
    #                label=f"Mean of Mean RV: {mean_all_mean:.2f}")
    #     ax.axvline(mean_all_median, color='orange', linestyle='dashed', linewidth=1,
    #                label=f"Mean of Median RV: {mean_all_median:.2f}")
    #     ax.set_title(f"Field {grp} : {grp_data.shape[0]} templated SB1s")
    #     ax.set_xlabel("RV values")
    #     ax.set_ylabel("Frequency")
    #     ax.set_xlim(50, 300)
    #     ax.legend()

    # Remove any unused subplots
    # for j in range(num_plots, len(axes)):
    #     fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to parse command line arguments, process CSV files, and plot histograms.
    """
    parser = argparse.ArgumentParser(
        description="Process CSV files to calculate mean and median 'RV' values, group results by filename substring, and plot histograms."
    )
    parser.add_argument("csv_dir", type=str, help="Directory containing the CSV files.")

    args = parser.parse_args()

    # Process CSV files and compute statistics
    results_df = process_all_csv(args.csv_dir)

    # Check if any CSVs were processed
    if results_df.empty:
        print("No CSV files found or processed in the given directory.")
        return

    # Plot the results
    plot_histograms(results_df)
    print(results_df, results_df.shape)


if __name__ == '__main__':
    main()
