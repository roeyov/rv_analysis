import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Directory containing your space-separated CSVs
DIR = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/ostars_sb1_new_list_from_coadded"
# Find all files ending with CCF_RVs.csv
pattern = os.path.join(DIR, "*CCF_RVs.csv")
files = glob.glob(pattern)
# Collect all 'Mean RVsig' values
mean_rvsig_list = []
for filepath in files:
    try:
        df = pd.read_csv(filepath, sep=' ')
    except Exception as e:
        print(f"Could not read {filepath}: {e}")
        continue
    if "Mean RVsig" in df.columns:
        mean_rvsig_list.append(df["Mean RVsig"].dropna())
    else:
        print(f"'Mean RVsig' column not found in {os.path.basename(filepath)}")
# Concatenate into one Series
if not mean_rvsig_list:
    raise RuntimeError("No 'Mean RVsig' data was found in any file.")
all_mean_rvsig = pd.concat(mean_rvsig_list)
# Compute statistics
mean_val   = all_mean_rvsig.mean()
median_val = all_mean_rvsig.median()
std_val    = all_mean_rvsig.std()
# Plot histogram
plt.figure(figsize=(8, 5))
from scipy.stats import lognorm
import numpy as np
data = all_mean_rvsig.values
# find the smallest positive value
min_nonzero = data[data > 0].min()
# replace zeros with that value
data[data == 0] = min_nonzero
# 1. Fit lognormal (fix loc=0)

shape, loc, scale = lognorm.fit(data, floc=0)

# 2. Plot histogram
plt.figure(figsize=(8,5))
counts, bins, patches = plt.hist(data, bins=250, edgecolor='black', alpha=0.6)

# 3. Overlay fitted PDF
bw = bins[1] - bins[0]
x = np.linspace(bins[0], bins[-1], 1000)
pdf = lognorm.pdf(x, shape, loc, scale)
mu_ln    = np.log(scale)
sigma_ln = shape

print("Fitted Log-Normal Distribution Parameters:")
print(f"  • shape (σ of ln-X)  : {sigma_ln:.4f}")
print(f"  • location (loc)     : {loc:.4e}")
print(f"  • scale (exp(μ) of X): {scale:.4f}")
print(f"  → underlying normal μ: {mu_ln:.4f}")
print(f"  → underlying normal σ: {sigma_ln:.4f}")
plt.plot(x, pdf * len(data) * bw, 'r-', lw=2, label='Lognormal fit')
# Identify the bin with the maximum count
max_bin_idx = counts.argmax()
max_count   = counts[max_bin_idx]
bin_low     = bins[max_bin_idx]
bin_high    = bins[max_bin_idx + 1]
avg_max_bin = (bin_low+bin_high)/2.0
# Color bins: highlight the max-count bin
for idx, patch in enumerate(patches):
    patch.set_facecolor('orange' if idx == max_bin_idx else 'gray')
# Add vertical lines for mean and median, and keep references
line_mean = plt.axvline(
    mean_val, color='red', linestyle='--', linewidth=1.5,
    label=f'Mean: {mean_val:.2f}'
)
line_med  = plt.axvline(
    median_val, color='blue', linestyle='-.', linewidth=1.5,
    label=f'Median: {median_val:.2f}'
)
line_max_bin  = plt.axvline(
    avg_max_bin, color='green', linestyle=':', linewidth=1.5,
    label=f'MaxBin: {avg_max_bin:.2f}'
)
fit_mode = loc + scale * np.exp(-shape**2)
line_fit_mode  = plt.axvline(
    fit_mode, color='magenta', linestyle='-', linewidth=1.5,
    label=f'fitMode: {fit_mode:.2f}'
)
# Annotate standard deviation on the plot
x_text = plt.xlim()[1] * 0.7
y_text = plt.ylim()[1] * 0.9
plt.text(
    x_text, y_text,
    f"Std Dev: {std_val:.2f}",
    fontsize=10,
    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7)
)
# Create a patch for the max-count bin
max_patch = mpatches.Patch(
    color='orange',
    label=f'Max bin [{bin_low:.2f}, {bin_high:.2f}): {int(max_count)}'
)
# Combine into one legend
plt.legend(
    handles=[line_mean, line_med, max_patch, line_max_bin,line_fit_mode],
    loc='best'
)
plt.xlabel("Mean RVsig")
plt.ylabel("Counts")
plt.title("Histogram of Mean RVsig\n(highlighted max-count bin)")
plt.tight_layout()
# (Re–add your mean/median lines, annotations, etc., as before)
plt.legend()
plt.show()
