import os.path
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# --- constants ---
PERIOD                   = "Period"
LABELS                   = "labels"
ECC                      = "Eccentricity"
PDC_BEST_PERIOD_POWER    = "bestPeriodPowerPDC"
BEST_PERIOD_POWER        = "bestPeriodPower"
MAX_MIN_DIFF             = "maxMinDiff"
SIGNIFICANCE             = "significance"
K1_STR                   = "K1"
SNR                      = "SNR"
SVM_SCORE                = "svm_score"
FEATURES                 = "features"

# --- helpers to map between TPR/FPR and thresholds ---
def tpr_from_threshold(roc, thresh):
    idx = np.argmin(np.abs(roc["thresholds"] - thresh))
    return float(roc["tpr"][idx])

def threshold_from_tpr(roc, target_tpr):
    idx = np.argmin(np.abs(roc["tpr"] - target_tpr))
    return float(roc["thresholds"][idx])

def fpr_from_threshold(roc, thresh):
    idx = np.argmin(np.abs(roc["thresholds"] - thresh))
    return float(roc["fpr"][idx])

def threshold_from_fpr(roc, target_fpr):
    idx = np.argmin(np.abs(roc["fpr"] - target_fpr))
    return float(roc["thresholds"][idx])

# --- data loading ---
@st.cache_data
def load_data(sample_dir, results_dir,input_pattern, debug=False, debug_limit=5):
    """
    Loads and merges parquet files from periodogram and sample directories.
    Uses a single glob-based loop for efficiency.
    """
    frames = []
    # pattern to match all result files
    pattern = os.path.join(results_dir, input_pattern , '*.parquet')
    all_res_files = sorted(glob.glob(pattern))
    if debug:
        all_res_files = all_res_files[:debug_limit]

    for res_path in all_res_files:
        noise_dir = os.path.basename(os.path.dirname(res_path))
        file_name = os.path.basename(res_path)
        sam_path = os.path.join(sample_dir, noise_dir, file_name)
        parts = noise_dir.split('_')
        # expected format: ['noise', 'analysis', '<amplitude>', '10000', '<label>']
        try:
            amplitude = float(parts[2])
        except (IndexError, ValueError):
            amplitude = np.nan
        # Read result data
        df_res = pd.read_parquet(
            res_path,
            columns=[PDC_BEST_PERIOD_POWER, BEST_PERIOD_POWER, MAX_MIN_DIFF, SIGNIFICANCE]
        )
        # Read sample data
        df_sam = pd.read_parquet(
            sam_path,
            columns=[PERIOD, LABELS, ECC, K1_STR,FEATURES]
        )

        # Combine and add metadata
        df = pd.concat(
            [df_res.reset_index(drop=True), df_sam.reset_index(drop=True)],
            axis=1
        )
        df.loc[df[SIGNIFICANCE] < 4, MAX_MIN_DIFF] = 2
        df['noise_dir'] = noise_dir
        df['noise_amplitude'] = amplitude
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

# --- ROC computation ---
@st.cache_data
def calculate_rocs(df):
    classifiers = [MAX_MIN_DIFF, PDC_BEST_PERIOD_POWER, BEST_PERIOD_POWER,SVM_SCORE]
    roc_dict = {}
    for clf in classifiers:
        if clf not in df.columns:
            continue
        fpr, tpr, thresholds = roc_curve(df[LABELS], df[clf])
        roc_dict[clf] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
    return roc_dict

# --- main app ---
st.title("Interactive ROC & Heatmaps for Binary Classifiers")

# input_pattern = '*'
input_pattern = 'noise_analysis_*'
samples_in = "/Users/roeyovadia/Documents/Data/simulatedData/noise_full/samples"
results_in = "/Users/roeyovadia/Documents/Data/simulatedData/noise_full/periodogram"
data = load_data(samples_in,results_in, input_pattern=input_pattern)
n_pos = int((data[LABELS] == 1).sum())
n_neg = int((data[LABELS] == 0).sum())
st.markdown(f"**Loaded {len(data)} examples:** {n_pos} positive, {n_neg} negative")

# Remove directories
all_dirs = sorted(data['noise_dir'].unique())
to_remove = st.sidebar.multiselect(
    "Remove noise directories from analysis", options=all_dirs, default=[]
)
if to_remove:
    data = data[~data['noise_dir'].isin(to_remove)]

# Log-transform scores
data[MAX_MIN_DIFF]       = np.log10(data[MAX_MIN_DIFF])
data[PERIOD]       = np.log10(data[PERIOD])
data[BEST_PERIOD_POWER]  = np.log10(data[BEST_PERIOD_POWER])
data[SNR]  = data[K1_STR]/data['noise_amplitude']

# Sidebar: SNR filter for positives only (min & max)
snr_min = st.sidebar.number_input(
    "Min SNR (apply to positives)",
    min_value=0.0,
    max_value=float(data[SNR].max()),
    value=0.0,
    step=0.1,
    format="%.3f",
    key="snr_min"
)
snr_max = st.sidebar.number_input(
    "Max SNR (apply to positives)",
    min_value=0.0,
    max_value=float(data[SNR].max()),
    value=float(data[SNR].max()),
    step=0.1,
    format="%.3f",
    key="snr_max"
)
# apply filter: keep positives with SNR between min and max
mask_snr = (data[LABELS] == 1) & ((data[SNR] < snr_min) | (data[SNR] > snr_max))
# drop those outside the range
data = data[~mask_snr]

features = [MAX_MIN_DIFF, PDC_BEST_PERIOD_POWER, BEST_PERIOD_POWER]
X = data[features].values
y = data[LABELS].values

# # 2) Split into train/test (optional but recommended)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, stratify=y, test_size=0.2, random_state=42
# )
#
# # 3) Build a pipeline: standardize → SVM with probability outputs
# svm_clf = make_pipeline(
#     StandardScaler(),
#     SVC(kernel='rbf', probability=True, random_state=42)
# )
#
# # 4) Train
# svm_clf.fit(X_train, y_train)
#
# # 5) Compute the “meta‐score” for all your data
# #    (this is P(label=1) from the SVM)
# data['svm_score'] = svm_clf.predict_proba(data[features])[:, 1]
#

roc_results = calculate_rocs(data)

# Initialize session state
for clf in [MAX_MIN_DIFF, PDC_BEST_PERIOD_POWER, BEST_PERIOD_POWER]:
    th_key = f"{clf}_th"
    if th_key not in st.session_state:
        st.session_state[th_key] = (data[clf].max() - data[clf].min()) / 2
if "mode" not in st.session_state:
    st.session_state.mode = 'TPR'
if "common_target" not in st.session_state:
    st.session_state.common_target = 0.5

# Callbacks to sync thresholds and target
def on_target_change():
    mode = st.session_state.mode
    tgt  = st.session_state.common_target
    for clf, roc in roc_results.items():
        if mode == 'TPR':
            st.session_state[f"{clf}_th"] = threshold_from_tpr(roc, tgt)
        else:
            st.session_state[f"{clf}_th"] = threshold_from_fpr(roc, tgt)

def on_threshold_change(changed_clf):
    mode = st.session_state.mode
    th   = st.session_state[f"{changed_clf}_th"]
    roc  = roc_results[changed_clf]
    if mode == 'TPR':
        new_tgt = tpr_from_threshold(roc, th)
    else:
        new_tgt = fpr_from_threshold(roc, th)
    st.session_state.common_target = new_tgt
    # update other thresholds
    for clf, roc in roc_results.items():
        if mode == 'TPR':
            st.session_state[f"{clf}_th"] = threshold_from_tpr(roc, new_tgt)
        else:
            st.session_state[f"{clf}_th"] = threshold_from_fpr(roc, new_tgt)

def main():
    # Sidebar controls
    st.sidebar.header("Control Panel")
    st.sidebar.radio(
        "Threshold Mode:", options=['TPR', 'FPR'], key='mode', on_change=on_target_change
    )
    st.sidebar.number_input(
        "Common Target", min_value=0.0000, max_value=1.000, step=0.001,
        key="common_target", on_change=on_target_change, format="%.3f"
    )
    for clf, label in [
        (MAX_MIN_DIFF, "RV Threshold"),
        (PDC_BEST_PERIOD_POWER, "PDC Threshold"),
        (BEST_PERIOD_POWER, "LS Threshold"),
        # (SVM_SCORE, "SVM Threshold"),
    ]:
        st.sidebar.number_input(
            label,
            min_value=float(data[clf].min()),
            max_value=float(data[clf].max()),
            step=(data[clf].max() - data[clf].min()) / 1000,
            key=f"{clf}_th",
            on_change=on_threshold_change,
            args=(clf,),
            format="%.3f"
        )

    # Plot ROC curves with dynamic points
    st.subheader("ROC Curves")
    roc_fig = go.Figure()
    colors = {MAX_MIN_DIFF: 'red', PDC_BEST_PERIOD_POWER: 'green', BEST_PERIOD_POWER: 'blue', SVM_SCORE: 'magenta'}
    for clf, roc in roc_results.items():
        roc_fig.add_trace(go.Scatter(
            x=roc['fpr'], y=roc['tpr'], mode='lines', name=clf, line=dict(color=colors[clf])
        ))
        th = st.session_state[f"{clf}_th"]
        idx = np.argmin(np.abs(roc['thresholds'] - th))
        roc_fig.add_trace(go.Scatter(
            x=[roc['fpr'][idx]], y=[roc['tpr'][idx]],
            mode='markers', marker=dict(size=12, symbol='x', color=colors[clf]),
            name=f"{clf} @ {th:.3f}"
        ))
    roc_fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    st.plotly_chart(roc_fig)

    # Heatmaps recalculation
    if st.button("Apply Thresholds and Recalculate Heatmaps"):
        period_bins = np.linspace(data[PERIOD].min(), data[PERIOD].max(), 20)
        ecc_bins = np.linspace(data[ECC].min(), data[ECC].max(), 20)

        def compute_heatmap(df, score_col, th):
            df['pred'] = df[score_col] >= th
            M, N = len(ecc_bins) - 1, len(period_bins) - 1
            hm = np.full((M, N), np.nan)
            text = np.empty((M, N), dtype=object)
            for i in range(M):
                for j in range(N):
                    bucket = df[
                        (df[ECC] >= ecc_bins[i]) & (df[ECC] < ecc_bins[i + 1]) &
                        (df[PERIOD] >= period_bins[j]) & (df[PERIOD] < period_bins[j + 1])
                        ]
                    # positives in bucket
                    pos = bucket[bucket[LABELS] == 1]
                    # true positives
                    tp = pos[pos['pred']].shape[0]
                    # false negatives
                    fn = pos.shape[0] - tp
                    # TPR calculation
                    hm[i, j] = tp / pos.shape[0] if pos.shape[0] > 0 else np.nan
                    # text annotation
                    text[i, j] = f"TP: {tp}, FN: {fn}, N: {pos.shape[0]}"
            return hm, text

        for clf in roc_results:
            th = st.session_state[f"{clf}_th"]
            hm, text = compute_heatmap(data.copy(), clf, th)
            fig = go.Figure(go.Heatmap(
                z=hm,
                x=period_bins,
                y=ecc_bins,
                colorscale='Viridis',
                colorbar=dict(title='TPR'),
                text=text,
                hovertemplate='%{text}<extra></extra>'
            ))
            fig.update_layout(
                title=f"{clf.split('_')[0]} TPR Heatmap @ {th:.3f}",
                xaxis_title=PERIOD,
                yaxis_title=ECC,
                height=500
            )
            st.plotly_chart(fig)

            # 1D: Detection vs Period
            df = data.copy()
            df['pred'] = df[clf] >= th
            per_tp = []
            per_counts = []
            centers_p = []
            for j in range(len(period_bins) - 1):
                mask = (df[PERIOD] >= period_bins[j]) & (df[PERIOD] < period_bins[j + 1])
                bucket = df[mask & (df[LABELS] == 1)]
                centers_p.append((period_bins[j] + period_bins[j + 1]) / 2)
                per_counts.append(len(bucket))
                per_tp.append(bucket['pred'].sum() / len(bucket) if len(bucket) > 0 else np.nan)
            per_fig = go.Figure()
            per_fig.add_trace(go.Bar(x=centers_p, y=per_tp, name='TPR'))
            per_fig.update_layout(
                title=f"{clf.split('_')[0]} TPR vs Period @ {th:.3f}",
                xaxis_title=PERIOD, yaxis_title='TPR', height=400
            )
            st.plotly_chart(per_fig)

            # 1D: Detection vs Eccentricity
            ecc_tp = []
            ecc_counts = []
            centers_e = []
            for i in range(len(ecc_bins) - 1):
                mask = (df[ECC] >= ecc_bins[i]) & (df[ECC] < ecc_bins[i + 1])
                bucket = df[mask & (df[LABELS] == 1)]
                centers_e.append((ecc_bins[i] + ecc_bins[i + 1]) / 2)
                ecc_counts.append(len(bucket))
                ecc_tp.append(bucket['pred'].sum() / len(bucket) if len(bucket) > 0 else np.nan)
            ecc_fig = go.Figure()
            ecc_fig.add_trace(go.Bar(x=centers_e, y=ecc_tp, name='TPR'))
            ecc_fig.update_layout(
                title=f"{clf.split('_')[0]} TPR vs Eccentricity @ {th:.3f}",
                xaxis_title=ECC, yaxis_title='TPR', height=400
            )
            st.plotly_chart(ecc_fig)

            # 3) Score distributions: true vs. false
            true_scores = data.loc[data[LABELS] == 1, clf]
            false_scores = data.loc[data[LABELS] == 0, clf]

            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(
                x=true_scores,
                name='True (label=1)',
                opacity=0.6,
                nbinsx=40
            ))
            hist_fig.add_trace(go.Histogram(
                x=false_scores,
                name='False (label=0)',
                opacity=0.6,
                nbinsx=40
            ))

            hist_fig.update_layout(
                barmode='overlay',
                title=f"{clf} Score Distribution",
                xaxis_title=clf,
                yaxis_title="Count",
                height=400
            )

            st.plotly_chart(hist_fig)

if __name__ == '__main__':
    main()