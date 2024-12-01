import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.periodagram import *
from sklearn.metrics import roc_curve, auc
import torch
from utils.constants import *
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

directory_out_fmt = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\scriptsOut\ModelEvaluator\periodogram_comp\{}\model_output"
directory_in_fmt = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\scriptsOut\RVDataGen\{}"
dir_name_false = "same_ts_10000_Falses{}"
dir_name_true = "same_ts_10000_Trues{}"


def load_and_predict_models(df, models_dir):
    paths = [p for p in os.listdir(models_dir) if p.endswith("pth")]
    for p_model in paths:
        model = torch.load(os.path.join(models_dir, p_model))
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


def plot_roc_by_field_dep(some_df, field_to_roc, hist_scaler=unity):
    plt.hist(hist_scaler(some_df[some_df.labels == 0][field_to_roc]), bins=200, alpha=0.6, density=True, color='r',
             label='Falses')
    plt.hist(hist_scaler(some_df[some_df.labels == 1][field_to_roc]), bins=200, alpha=0.6, density=True, color='b',
             label='Trues')

    # Add a vertical line at x=0.114
    plt.axvline(x=0.15365, color='green', linestyle='--', linewidth=2)

    # Add a label near the vertical line
    plt.text(0.15365 + 0.01, plt.ylim()[1] * 0.9, 'FAP by chi2.sf', color='green', fontsize=12, rotation=90)

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


ORBITAL_PARAMS = {
                  ECC: unity,
                  OMEGA: unity,
                  PERIOD: np.log10,
                  M1: unity,
                  Q: unity,
                  INC:unity}

def find_closest_index(tpr, value):
    tpr = np.asarray(tpr)  # Ensure tpr is a numpy array for efficient operations
    index = (np.abs(tpr - value)).argmin()  # Find the index of the minimum difference
    return index

def plot_roc_by_field(some_df, fields_to_rocs, hist_scalers=[lambda x: x]):
    num_fields = len(fields_to_rocs)
    fig = plt.figure(figsize=(24, 13))
    gs = GridSpec(num_fields, 2, width_ratios=[1, num_fields])  # Adjusting width ratios to make right plot larger
    update_dict = {}

    threshold_lines = [None for _ in fields_to_rocs]

    for i, field_to_roc in enumerate(fields_to_rocs):
        ax =  fig.add_subplot(gs[i, 0])
        hist_scaler = hist_scalers[i] if len(hist_scalers) > 1 else hist_scalers[0]
        # Plot histograms for each field
        ax.hist(hist_scaler(some_df[some_df.labels == 0][field_to_roc]), bins=200, alpha=0.6, density=True, color='r',
                label='Single Sim')
        ax.hist(hist_scaler(some_df[some_df.labels == 1][field_to_roc]), bins=200, alpha=0.6, density=True, color='b',
                label='Binary Sim')
        update_dict[i] = {"t_line" : ax.axvline(x=0, color='magenta',linewidth=2,  linestyle='--', label='Threshold'), "scaler" : hist_scaler}

        # Set labels and legend for the histograms
        # ax.set_title(f'{field_to_roc if field_to_roc != BEST_PERIOD_POWER else BEST_PERIOD_POWER + "LS"}', fontsize=14)
        ax.set_xlabel(f'{field_to_roc if field_to_roc== PDC_BEST_PERIOD_POWER else "log_10({})".format(field_to_roc)}', fontsize=12)
        ax.set_ylabel('Density', fontsize=16)
        ax.legend(fontsize=16)


    # Now, plot the ROC curves for each field on the same figure
    ax4 = fig.add_subplot(gs[:, 1])  # Right plot spanning both rows

    for i, field_to_roc in enumerate(fields_to_rocs):
        # Calculate ROC curve for the current field
        fpr, tpr, thresholds = roc_curve(some_df.labels, some_df[field_to_roc])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        ax4.plot(fpr, tpr, lw=2, label=f'{field_to_roc if field_to_roc != BEST_PERIOD_POWER else BEST_PERIOD_POWER + "LS"} (area = {roc_auc:.2f})')
        update_dict[i]["roc_point"], = ax4.plot([], [], 'ro', markersize=10, color='magenta')  # Initial red point on ROC curve
        update_dict[i]["roc"] = roc_curve(some_df.labels, some_df[field_to_roc])

    # Plot the diagonal line (no-skill classifier)
    ax4.plot([0, 1], [0, 1], color='grey', linestyle='--')

    # Set labels, title, and legend for the ROC curves
    ax4.set_xlabel('False Positive Rate', fontsize=20)
    ax4.set_ylabel('True Positive Rate', fontsize=20)
    ax4.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, weight='bold')
    ax4.legend(loc='lower right', fontsize=26)
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Show grid
    ax4.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Update function for animation
    def update(i):
        # Update ROC point
        items = []

        print((i+60)/100.0)
        for j, field_to_roc_f in enumerate(fields_to_rocs):
            fpr_f, tpr_f, thresholds_f = update_dict[j]["roc"]
            index = find_closest_index(tpr_f, ((i+60)/100.0))
            if index >= len(thresholds_f): continue
            update_dict[j]["roc_point"].set_data(fpr_f[index], tpr_f[index])
            scaler =  update_dict[j]['scaler']
        # Update threshold line on histogram
            update_dict[j]["t_line"].set_xdata([scaler(thresholds_f[index]), scaler(thresholds_f[index])])
            items.extend([update_dict[j]["roc_point"], update_dict[j]["t_line"]])
        return items

    frames = max(len(update_dict[j]["roc"][2]) for j, field_to_roc in enumerate(fields_to_rocs))
    print("frames ", frames)
    ani = FuncAnimation(fig, update, frames=40, blit=True, repeat=False, interval=100)
    ani.save("roc_curve_animation_4.gif", writer="pillow", fps=5)
    # Show the ROC plot
    plt.show()


def plot_roc_by_field_dep2(some_df, fields_to_rocs, hist_scalers=[lambda x: x]):
    num_fields = len(fields_to_rocs)

    # Create subplots for histograms
    fig, axes = plt.subplots(num_fields, 1, figsize=(10, 4 * num_fields), constrained_layout=True)

    # If there's only one field, axes won't be an array, so wrap it in a list
    if num_fields == 1:
        axes = [axes]

    for i, field_to_roc in enumerate(fields_to_rocs):
        ax = axes[i]
        hist_scaler = hist_scalers[i] if len(hist_scalers) > 1 else hist_scalers[0]
        # Plot histograms for each field
        ax.hist(hist_scaler(some_df[some_df.labels == 0][field_to_roc]), bins=200, alpha=0.6, density=True, color='r',
                label='Single Sim')
        ax.hist(hist_scaler(some_df[some_df.labels == 1][field_to_roc]), bins=200, alpha=0.6, density=True, color='b',
                label='Binary Sim')

        # Set labels and legend for the histograms
        ax.set_title(f'{field_to_roc if field_to_roc != BEST_PERIOD_POWER else BEST_PERIOD_POWER + "LS"}', fontsize=14)
        ax.set_xlabel(f'{field_to_roc if field_to_roc== PDC_BEST_PERIOD_POWER else "log_10({})".format(field_to_roc)}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()

    plt.show()

    # Now, plot the ROC curves for each field on the same figure
    plt.figure(figsize=(10, 6))

    for field_to_roc in fields_to_rocs:
        # Calculate ROC curve for the current field
        fpr, tpr, thresholds = roc_curve(some_df.labels, some_df[field_to_roc])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{field_to_roc if field_to_roc != BEST_PERIOD_POWER else BEST_PERIOD_POWER + "LS"} (area = {roc_auc:.2f})')

    # Plot the diagonal line (no-skill classifier)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')

    # Set labels, title, and legend for the ROC curves
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, weight='bold')
    plt.legend(loc='lower right')

    # Show grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the ROC plot
    plt.show()


def get_worst_trues(row, field_to_check, threshold):
    return row[field_to_check] <= threshold


def plot_field_distribution(some_df, str_to_file='', fields_to_hist=ORBITAL_PARAMS,
                            boolean_func_to_apply=(lambda x: True), args=()):
    # Apply the boolean function to filter the DataFrame
    boolean_array = some_df.apply(boolean_func_to_apply, axis=1, args=args)

    sub_df = some_df[boolean_array]

    # Set the overall style
    sns.set(style="whitegrid")

    # Determine the number of fields to plot
    num_plots = len(fields_to_hist)

    # Create subplots in a grid layout with high-quality size
    fig, axes = plt.subplots(nrows=(num_plots + 2) // 3, ncols=3, figsize=(15, 5 * ((num_plots + 2) // 3)))
    axes = axes.flatten()  # Flatten axes for easier iteration
    fig.suptitle('GenPop VS badTruesPop {} Decision'.format(str_to_file))
    plot_idx = 0

    # Loop through the specified fields and plot histograms
    for field_to_hist, func in fields_to_hist.items():
        ax = axes[plot_idx]  # Get the current axis
        try:
            # Plot filtered data
            sns.histplot(func(sub_df[field_to_hist]), bins=30, ax=ax, kde=False, stat='density', color='blue', alpha=0.6,
                         label='Filtered')
            # Plot original data
            sns.histplot(func(some_df[field_to_hist]), bins=30, ax=ax, kde=False, stat='density', color='orange', alpha=0.4,
                         label='Original')

            # Set title and labels
            ax.set_title(f'Distribution of {field_to_hist}', fontsize=14, weight='bold')
            ax.set_xlabel(f'{field_to_hist}', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)

            # Add legend
            ax.legend(fontsize=10)

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

    # Save as a high-resolution image
    plt.savefig("bad_trues_for_{}_decision.png".format(str_to_file), dpi=300,
                bbox_inches='tight')  # High resolution for article use

    plt.show()


def plot_dist_worst_trues(df, field_to_check, tnr=0.95):
    falses = df[df.labels == 0]
    threshold = np.percentile(falses[field_to_check], tnr * 100)
    plot_field_distribution(df[df.labels == 1], str_to_file=field_to_check, boolean_func_to_apply=get_worst_trues,
                            args=(field_to_check, threshold))


def calc_field_loss(df, predicted_fields, true_field):
    results = {}
    true_values = np.log10(df[true_field])
    for prediction in predicted_fields:
        predicted_field = np.log10(df[prediction])
        MSE = np.square(np.subtract(true_values, predicted_field)).mean()
        results[prediction] = MSE
    return results


combined_df_false, files_found_false = load_parquet(directory_out_fmt.format(dir_name_false.format("")))
combined_df_true, files_found_true = load_parquet(directory_out_fmt.format(dir_name_true.format("")))
combined_df_in_false, _ = load_parquet(directory_in_fmt.format(dir_name_false.format("")), files_found_false)
combined_df_in_true, _ = load_parquet(directory_in_fmt.format(dir_name_true.format("")), files_found_true)
combined_df_false_nn, files_found_false = load_parquet(directory_out_fmt.format(dir_name_false.format("_nn3")))
combined_df_true_nn, files_found_true = load_parquet(directory_out_fmt.format(dir_name_true.format("_nn3")))

one = pd.concat([combined_df_in_false, combined_df_false, combined_df_false_nn], axis=1)
two = pd.concat([combined_df_in_true, combined_df_true, combined_df_true_nn], axis=1)
final = pd.concat([one, two], axis=0)

p_value = 95

falses = final[final.labels == 0]
threshold = np.percentile(falses[BEST_PERIOD_POWER], p_value)
final['PassedLombScargle'] = ~final.apply(get_worst_trues, axis=1, args=(BEST_PERIOD_POWER, threshold))

threshold = np.percentile(falses[PDC_BEST_PERIOD_POWER], p_value)
final['PassedPDC'] = ~final.apply(get_worst_trues, axis=1, args=(PDC_BEST_PERIOD_POWER, threshold))
threshold = np.percentile(falses[MAX_MIN_DIFF], p_value)
print("threshold herrrrreee!!!: ", threshold)
final['PassedMaxMin'] = final[MAX_MIN_DIFF] > 20
threshold = np.percentile(falses[NN_DECISION_PRE], p_value)
final['PassedNN'] = ~final.apply(get_worst_trues, axis=1, args=(NN_DECISION_PRE, threshold))

final = final[final[PERIOD] < 1000]
final = final[final[PERIOD] > 2]

# final = final[final.maxMinDiff < 15]
# print(np.max(final[PERIOD]))

# final.loc[final.maxMinDiff > 20, "bestPeriodPowerPDC"] = 1
# final.loc[final.maxMinDiff > 20, "bestPeriodPower"] = 100

print(final.columns, final.shape)
# bestPeriodPowerPDC

# rocs and distributions
# plot_roc_by_field(final ,"bestPeriodPower",np.log10)
# plot_dist_worst_trues(final, "bestPeriodPower")


plot_roc_by_field(final, [MAX_MIN_DIFF, BEST_PERIOD_POWER, PDC_BEST_PERIOD_POWER],
                  [ np.log10, np.log10, unity])
# plot_dist_worst_trues(final, PDC_BEST_PERIOD_POWER, tnr=p_value/100)
# plot_dist_worst_trues(final, MAX_MIN_DIFF, tnr=p_value/100)
# plot_dist_worst_trues(final, BEST_PERIOD_POWER, tnr=p_value/100)
# plot_dist_worst_trues(final, NN_DECISION_PRE, tnr=p_value/100)
for i in range(3):
    print(sum(np.array(final[final[LABELS] == 0][BEST_PERIOD_POWER]) > np.array([fal[i] for fal in final[final[LABELS] == 0][FALSE_ALARM_LEVELS]])))
print(len(final[final[LABELS] == 0]))
trues = final[final[LABELS] == 1]

# calc Period loss (by log)
period_loss = calc_field_loss(trues, [PDC_BEST_PERIOD, BEST_PERIOD], PERIOD)
print(period_loss)

period_loss = calc_field_loss(trues[trues['PassedPDC']], [PDC_BEST_PERIOD], PERIOD)
print(period_loss)
period_loss = calc_field_loss(trues[trues['PassedNN']], [NN_PERIOD], PERIOD)
print(period_loss)
period_loss = calc_field_loss(trues[trues['PassedLombScargle']], [BEST_PERIOD], PERIOD)
print(period_loss)

# Counts by test passes
combination_counts = final.groupby(['PassedMaxMin', 'PassedPDC', 'PassedLombScargle']).size().reset_index(name='count')
# combination_counts = trues.groupby(['PassedMaxMin', "PassedNN"]).size().reset_index(name='count')
# print(trues.groupby(['PassedMaxMin', 'PassedPDC', 'PassedLombScargle', "PassedNN"]))
print(combination_counts)
print(len(trues))

# trues = trues[trues[PERIOD] < 350]
#
# index = 1
# print(list(trues[ERRORS].iloc[index]))
# print(list(trues[RADIAL_VELS].iloc[index]))
# print(list(trues[TIME_STAMPS].iloc[index]))
#
# best_period1, fap1, freq, pdc_power_reg = pdc(trues[TIME_STAMPS].iloc[index], trues[RADIAL_VELS].iloc[index], data_err=trues[ERRORS].iloc[index], pmin=1.0, pmax=5000)
#
#
# plt.figure(figsize=(13, 5))
#
# plt.xlabel("Frequency")
# plt.ylabel("Power")
# print("period: ",trues[PERIOD].iloc[index])
# plt.axvline(1/trues[PERIOD].iloc[index], color='red', alpha=0.15, linewidth=4)
# plt.plot(freq, pdc_power_reg, color='black', linewidth=0.75)
# plt.show()
print( )