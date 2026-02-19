import pandas as pd


SANA_DECISION = "/Users/roeyovadia/Roey/Masters/Reasearch/presentaions/Seminar/BLOeM_O_Binaries_Sana.txt"
SB2_ZEHAVA_LIST = "/Users/roeyovadia/Documents/Data/BLOeM_Data/lists/ostars_sb2.txt"
SB1_ZEHAVA_LIST = "/Users/roeyovadia/Documents/Data/BLOeM_Data/lists/ostars_sb1.txt"

mcmc_solution = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/only_He_from_coAdded2/new_for_seminar/mcmc_min_withNull/mcmc_params.csv"


epochs_9_res = pd.read_csv(SANA_DECISION, sep="\t",)
epochs_21_sb2 = pd.read_csv(SB2_ZEHAVA_LIST, header=None)
epochs_21_sb1 = pd.read_csv(SB1_ZEHAVA_LIST,header=None)
epochs_21_sb1.rename(columns={0:'BLOeM_ID'}, inplace=True)
epochs_21_sb2.rename(columns={0:'BLOeM_ID'}, inplace=True)
epochs_21_sb2['Type'] = 'SB2'

mcmc_solution_df = pd.read_csv(mcmc_solution)

sb1_single_split = mcmc_solution_df[["star_name","null_gamma_mode"]].copy()
sb1_single_split["Type"] = "SB1"
sb1_single_split.loc[~sb1_single_split["null_gamma_mode"].isna(), "Type"] = "single"
sb1_single_split.drop(columns=["null_gamma_mode"], inplace=True)
# Ensure matching column names
sb1_single_split = sb1_single_split.rename(columns={"star_name": "BLOeM_ID"})

# Concatenate
combined_df = pd.concat([sb1_single_split, epochs_21_sb2], ignore_index=True)

# Find ids in combined_df not present in epochs_9_res
missing_ids = combined_df[~combined_df["BLOeM_ID"].isin(epochs_9_res["BLOeM_ID"])]

# Force their type to 'single'
missing_rows = missing_ids.copy()
missing_rows["Type"] = "single"

# Append
epochs_9_res_updated = pd.concat([epochs_9_res, missing_rows], ignore_index=True)
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Merge on common BLOeM_IDs
# ------------------------------------------------------------
compare_df = pd.merge(
    epochs_9_res_updated,
    combined_df,
    on="BLOeM_ID",
    suffixes=("_sana", "_mcmc")
)

# ------------------------------------------------------------
# 2. Agreement flag
# ------------------------------------------------------------
compare_df["is_agreement"] = (
    compare_df["Type_sana"] == compare_df["Type_mcmc"]
)

# ------------------------------------------------------------
# 3. Summary printout
# ------------------------------------------------------------
n_total = len(compare_df)
n_agree = compare_df["is_agreement"].sum()
n_disagree = n_total - n_agree

print("===== SANA vs MCMC Classification Comparison =====")
print(f"Total common BLOeM_IDs : {n_total}")
print(f"Agreements             : {n_agree}")
print(f"Disagreements          : {n_disagree}")
print(f"Agreement Rate         : {100*n_agree/n_total:.1f}%")

# ------------------------------------------------------------
# 4. Disagreement classes
# ------------------------------------------------------------
disagree_df = compare_df[~compare_df["is_agreement"]].copy()
disagree_df["disagreement_class"] = (
    disagree_df["Type_sana"] + " → " + disagree_df["Type_mcmc"]
)

print("\n===== Disagreement Breakdown (SANA → MCMC) =====")
if len(disagree_df) > 0:
    print(disagree_df["disagreement_class"].value_counts())
else:
    print("No disagreements found.")

print("\n===== Example Disagreements =====")
print(
    disagree_df[["BLOeM_ID","Type_sana","Type_mcmc","disagreement_class"]]
    .head(40)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 5. Plot: agreement vs disagreement (with bar labels)
# ------------------------------------------------------------
plt.figure(figsize=(6,4))

counts = compare_df["is_agreement"].value_counts().rename(index={
    True: "agreement",
    False: "disagreement"
})

ax = counts.plot(kind="bar")

# Add labels
for i, v in enumerate(counts.values):
    ax.text(i, v - 10, str(v), ha='center', fontsize=12)

plt.ylabel("Number of Stars")
plt.title("9 epoch vs 21 epoch: Agreement Summary")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# 6. Plot: disagreement classes (with bar labels)
# ------------------------------------------------------------
if len(disagree_df) > 0:

    plt.figure(figsize=(8,6))

    class_counts = disagree_df["disagreement_class"].value_counts()
    ax = class_counts.plot(kind="bar")

    # Add labels above each bar
    for i, v in enumerate(class_counts.values):
        ax.text(i, v - 0.5, str(v), ha='center', fontsize=12)

    plt.ylabel("Number of Stars")
    plt.title("9 epoch vs 21 epoch: Disagreement Classes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

else:
    print("No disagreement classes to plot.")
