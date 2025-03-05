import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

input_path = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\scriptsOut\ModelEvaluator\low_power_trues_same_ts.parquet"
sub_df = pd.read_parquet(input_path)
directory = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\scriptsOut\RVDataGen\same_ts_10000_Trues"

# Initialize an empty list to store individual DataFrames
dataframes = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    # Read the Parquet file and append the DataFrame to the list
    iter_df = pd.read_parquet(file_path)
    dataframes.append(iter_df)

# Concatenate all DataFrames into a single DataFrame
df = pd.concat(dataframes, ignore_index=True)

# Now, combined_df contains data from all Parquet files in the directory

sub_df = sub_df[(sub_df.Period<1000) & (sub_df.Period>2) ]
df = df[(df.Period<1000) & (df.Period>2) ]


def is_list_or_array(obj):
    return isinstance(obj, (list, tuple, np.ndarray))

num_plots = sum(1 for key in df.columns if not is_list_or_array(df[key].iloc[0]))

# Create a figure with subplots
fig, axes = plt.subplots(nrows=(num_plots + 2) // 3, ncols=3, figsize=(15, 5 * ((num_plots + 2) // 3)))
axes = axes.flatten()  # Flatten in case of a grid to iterate easily

plot_idx = 0

for key in df.columns:
    if is_list_or_array(df[key].iloc[0]):
        continue
    try:
        ax = axes[plot_idx]  # Select the subplot axis
        # ax.hist(np.log10(df[key]), bins=50, alpha=0.5, label="all", density=True)
        # ax.hist(np.log10(sub_df[key]), bins=50, alpha=0.5, label="sub", density=True)
        # ax.set_title("log10_"+key)
        ax.hist(df[key], bins=50, alpha=0.5, label="all", density=True)
        ax.hist(sub_df[key], bins=50, alpha=0.5, label="sub", density=True)
        ax.set_title(key)
        ax.legend()
        plot_idx += 1
    except ValueError:
        continue

# Hide any unused subplots
for i in range(plot_idx, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout to avoid overlap
# plt.tight_layout()
plt.show()

## histogram plots
# for key in df.columns:
#     if is_list_or_array(df[key].iloc[0]):
#         continue
#     try:
#         # plt.hist(np.log10(args_dict[key][SAMPLES]), bins=50, alpha=0.5)
#         # plt.title("log_" + key)
#         #
#         plt.hist(df[key], bins=100, alpha=0.5)
#         plt.title(key)
#
#         plt.show()
#     except ValueError:
#         continue
