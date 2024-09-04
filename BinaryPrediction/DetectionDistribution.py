import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from Scripts.utils.periodagram import *
from sklearn.metrics import roc_curve, auc

directory_out_fmt = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\scriptsOut\ModelEvaluator\{}\model_output"
directory_in_fmt = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\scriptsOut\RVDataGen\{}"
dir_name_false = "same_ts_100000_Falses"
dir_name_true = "same_ts_10000_Trues"

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


combined_df_false, files_found_false = load_parquet(directory_out_fmt.format(dir_name_false))
combined_df_true, files_found_true = load_parquet(directory_out_fmt.format(dir_name_true))
combined_df_in_false,_ = load_parquet(directory_in_fmt.format(dir_name_false), files_found_false)
combined_df_in_true,_ = load_parquet(directory_in_fmt.format(dir_name_true),files_found_true)
# print(np.unique( combined_df_false['falseAlarmLevels'].apply(lambda x: x[0] if len(x) > 2 else None),return_counts=True))
# print(np.unique( combined_df_true['falseAlarmLevels'].apply(lambda x: x[0] if len(x) > 2 else None),return_counts=True))
arr = np.log10([row.bestPeriodPower / row.falseAlarmLevels[2] for i,row in combined_df_false.iterrows()])
plt.figure(figsize=(10, 6))
plt.hist(
    np.log10(combined_df_false.bestPeriodPower),
    bins=200,
    color='blue',
    alpha=0.7,
    # edgecolor='black',
    density=True,
    # cumulative=True
)

sub_log_period = (0.3 < np.log10(combined_df_in_true.Period)) & (np.log10(combined_df_in_true.Period) < 3)
plt.hist(
    np.log10(combined_df_true[sub_log_period].bestPeriodPower),
    bins=200,
    color='red',
    alpha=0.7,
    # edgecolor='black',
    density=True
    # cumulative=-1
)
# plt.hist(
#     arr,
#     bins=200,
#     color='blue',
#     alpha=0.7,
#     edgecolor='black'
# )

# Adding labels, title, and grid
plt.title('Histogram of Best Period Power', fontsize=16, weight='bold')
plt.xlabel('Best Period Power', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Adding a text box for any additional details
plt.text(0.95, 0.95, 'N = {}'.format(len(combined_df_false.bestPeriodPower)),
         horizontalalignment='right',
         verticalalignment='top',
         transform=plt.gca().transAxes,
         fontsize=12)
plt.text(0.95, 0.95, 'N Falses = {}\nN Trues = {}'.format(len(combined_df_false.bestPeriodPower),
                                                          len(combined_df_true[sub_log_period].bestPeriodPower)),
         horizontalalignment='right',
         verticalalignment='top',
         transform=plt.gca().transAxes,
         fontsize=12)

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()

# Calculate the ROC curve
print(combined_df_in_false.columns)
y_test = np.concatenate([combined_df_in_false.labels, combined_df_in_true[sub_log_period].labels])
y_probs = np.concatenate([np.log10(combined_df_false.bestPeriodPower), np.log10(combined_df_true[sub_log_period].bestPeriodPower)])
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

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
# combine_max_df = pd.concat(
#     [combined_df[combined_df.bestPeriodPower > np.array([row[2] for row in combined_df.falseAlarmLevels])],
#      combined_df_in[combined_df.bestPeriodPower > np.array([row[2] for row in combined_df.falseAlarmLevels])]], axis=1)
# plt.hist(
#     np.log10(combined_df[combined_df.bestPeriodPower > np.array([row[0] for row in combined_df.falseAlarmLevels])].bestPeriod),
#     bins=50,
#     color='blue',
#     alpha=0.7,
#     edgecolor='black'
# )
#
# # Adding title and axis labels
# plt.title('Histogram of Log-Best Periods that passed 0.5')
# plt.xlabel('Log(Best Period)')
# plt.ylabel('Frequency')
#
# # Adding a grid
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()
#
# plt.hist(
#     np.log10(combined_df[combined_df.bestPeriodPower > np.array([row[1] for row in combined_df.falseAlarmLevels])].bestPeriod),
#     bins=50,
#     color='blue',
#     alpha=0.7,
#     edgecolor='black'
# )
#
# # Adding title and axis labels
# plt.title('Histogram of Log-Best Periods that passed 0.01')
# plt.xlabel('Log(Best Period)')
# plt.ylabel('Frequency')
#
# # Adding a grid
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()
#
# plt.hist(
#     np.log10(combined_df[combined_df.bestPeriodPower > np.array([row[2] for row in combined_df.falseAlarmLevels])].bestPeriod),
#     bins=50,
#     color='blue',
#     alpha=0.7,
#     edgecolor='black'
# )
#
# # Adding title and axis labels
# plt.title('Histogram of Log-Best Periods that passed 0.001')
# plt.xlabel('Log(Best Period)')
# plt.ylabel('Frequency')
#
# # Adding a grid
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()
# row = combine_max_df.iloc[1]
# period1, fap1, fal1, freq1, pow1 = ls(np.array(row.ts), np.array(row.rvs), data_err=np.array(row.errs), pmin=1, pmax=1100)
# x = np.linspace(0, 730, 1000)
# # Array of positions where the Dirac delta functions should be located
# positions = np.array(row.ts)
# # Initialize the array for the delta function sum
# delta_sum = np.zeros_like(x)
# # Create the sum of Dirac delta functions
# for pos in positions:
#     # Find the closest index in x corresponding to the position of the delta
#     index = np.argmin(np.abs(x - pos))
#     delta_sum[index] += 1
# print(list(row.ts))
# period2, fap2, fal2, freq2, pow2 = ls(np.array(x), np.array(delta_sum),  pmin=1, pmax=1100)
# plotls(freq1, pow1, fal1, pmin=1, pmax=1100, star_id=r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\Scripts\BinaryPrediction\ls_2")
# plotls(freq2, pow2, fal2, pmin=1, pmax=1100, star_id=r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\Scripts\BinaryPrediction\ts_2")


# Create a figure with 2 rows and 2 columns of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Flatten the axes array for easier iteration
axes = axes.flatten()
# Loop through each subplot
for i, ax in enumerate(axes):
    y = np.log10(combined_df_true[(i+1 > np.log10(combined_df_in_true.Period)) & (np.log10(combined_df_in_true.Period) >i) & (np.log10(combined_df_in_true.Period) >0.3)].bestPeriodPower)  # Generate some data with a phase shift
    ax.hist(y, label=f'Subplot {i+1}',bins=200,density=True)
    ax.set_title(f'Subplot {i+1}, N = {len(y)}')
    ax.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()