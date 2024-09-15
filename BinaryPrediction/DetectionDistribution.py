import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# from Scripts.utils.periodagram import *
from sklearn.metrics import roc_curve, auc
import torch


directory_out_fmt = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\scriptsOut\ModelEvaluator\periodogram_comp\{}\model_output"
directory_in_fmt = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\scriptsOut\RVDataGen\{}"
dir_name_false = "same_ts_10000_Falses"
dir_name_true = "same_ts_10000_Trues"


def load_and_predict_models(df,models_dir):
    paths = [p for p in os.listdir(models_dir) if p.endswith("pth")]
    for p_model in paths:
        model = torch.load(os.path.join(models_dir,p_model))
        df[p_model.split('.')[0]] = model()


def load_parquet(fp, file_names=None):
    # List to hold individual DataFrames
    df_list = []
    # Loop through all files in the directory
    iter_file_names = os.listdir(fp) if file_names is None else file_names
    for file_name in iter_file_names:
        # Full path to the file
        file_path = os.path.join(fp, file_name)
        # Read the Parquet file into a DataFrame
        df = pd.read_parquet(file_path)
        # Append the DataFrame to the list
        df_list.append(df)
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df, iter_file_names


def unity(array):
    return array


def plot_roc_by_field(some_df, field_to_roc, hist_scaler=unity):
    plt.hist(hist_scaler(some_df[some_df.labels == 0][field_to_roc]), bins=200, alpha=0.6, density=True, color='r',
             label='Falses')
    plt.hist(hist_scaler(some_df[some_df.labels == 1][field_to_roc]), bins=200, alpha=0.6, density=True, color='b',
             label='Trues')

    # Add a vertical line at x=0.114
    plt.axvline(x=0.114, color='green', linestyle='--', linewidth=2)

    # Add a label near the vertical line
    plt.text(0.114 + 0.01, plt.ylim()[1] * 0.9, 'FAP by chi2.sf', color='green', fontsize=12, rotation=90)

    # Add legend
    plt.legend()

    plt.show()

    fpr, tpr, thresholds = roc_curve(some_df.labels, some_df[field_to_roc])

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

    # Plot the diagonal line (no-skill classifier)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')

    # Set labels, title, and legend
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, weight='bold')
    plt.legend(loc='lower right')

    # Show grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.show()


ORBITAL_PARAMS = ['T0', 'Eccentricity', 'OMEGA_rad', 'K1', 'K2', 'GAMMA', 'Period',
                  'Mass1', 'MassRatio', 'Inclination']


def get_worst_trues(row, field_to_check, threshold):
    return row[field_to_check] <= threshold


def plot_field_distribution(some_df, fields_to_hist=ORBITAL_PARAMS, boolean_func_to_apply=(lambda x: True), args=()):
    # Apply the boolean function to filter the DataFrame
    boolean_array = some_df.apply(boolean_func_to_apply, axis=1, args=args)

    sub_df = some_df[boolean_array]

    # Determine the number of fields to plot
    num_plots = len(fields_to_hist)

    # Create subplots in a grid layout
    fig, axes = plt.subplots(nrows=(num_plots + 2) // 3, ncols=3, figsize=(15, 5 * ((num_plots + 2) // 3)))
    axes = axes.flatten()  # Flatten axes for easier iteration

    plot_idx = 0

    # Loop through the specified fields and plot histograms
    for field_to_hist in fields_to_hist:
        ax = axes[plot_idx]  # Get the current axis
        try:
            ax.hist(sub_df[field_to_hist], bins=100, alpha=0.5, density=True)
            ax.hist(some_df[field_to_hist], bins=100, alpha=0.5, density=True)
            ax.set_title(field_to_hist)
            plot_idx += 1
        except ValueError:
            print(f"Skipping field {field_to_hist} due to a ValueError.")
            continue

    # Hide unused subplots if any
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])

    num_rows = len(sub_df)
    fig.text(0.5, 0.01, f"Number of rows in filtered DataFrame: {num_rows}", ha='center', fontsize=12)

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.savefig("dist_pdc.png")  # Save as PNG, but you can use other formats like PDF, SVG, etc.

    plt.show()



def plot_dist_worst_trues(df, field_to_check, tnr=0.95):
    falses = df[df.labels == 0]
    threshold = np.percentile(falses[field_to_check], tnr * 100)
    plot_field_distribution(df[df.labels == 1], boolean_func_to_apply=get_worst_trues, args=(field_to_check, threshold))


combined_df_false, files_found_false = load_parquet(directory_out_fmt.format(dir_name_false))
combined_df_true, files_found_true = load_parquet(directory_out_fmt.format(dir_name_true))
combined_df_in_false, _ = load_parquet(directory_in_fmt.format(dir_name_false), files_found_false)
combined_df_in_true, _ = load_parquet(directory_in_fmt.format(dir_name_true), files_found_true)

one = pd.concat([combined_df_in_false, combined_df_false], axis=1)
two = pd.concat([combined_df_in_true, combined_df_true], axis=1)
final = pd.concat([one, two], axis=0)

falses = final[final.labels == 0]
threshold = np.percentile(falses["bestPeriodPower"], 95)
final['PassedLombScargle'] = ~final.apply(get_worst_trues, axis=1, args=("bestPeriodPower",threshold))

threshold = np.percentile(falses["bestPeriodPowerPDC"], 95)
final['PassedPDC'] = ~final.apply(get_worst_trues, axis=1, args=("bestPeriodPowerPDC",threshold))
final['PassedMaxMin'] = final.maxMinDiff > 20

# final.loc[final.maxMinDiff > 20, "bestPeriodPowerPDC"] = 1
# final.loc[final.maxMinDiff > 20, "bestPeriodPower"] = 100

print(final.columns)
# bestPeriodPowerPDC

# rocs and distributions
# plot_roc_by_field(final ,"bestPeriodPower",np.log10)
# plot_dist_worst_trues(final, "bestPeriodPower")


plot_roc_by_field(final, "bestPeriodPowerPDC")
plot_dist_worst_trues(final, "bestPeriodPowerPDC")

# Counts by test passes
trues = final[final.labels == 1]
combination_counts = trues.groupby(['PassedMaxMin', 'PassedPDC', 'PassedLombScargle']).size().reset_index(name='count')
print(combination_counts)

print(trues.groupby(['PassedMaxMin', 'PassedPDC', 'PassedLombScargle']))

