"""
2D scatter and heatmap of LS_iter_fap vs prob_bicc from simulation results.
Highlights the detection region: prob_bicc > 0.5 AND LS_iter_fap < 0.2.
"""

import sys, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plot_style import INTERACTIVE as S, apply_style_to_layout

# ── data ────────────────────────────────────────────────────────────────
CSV = (
    "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/"
    "CCF/simulations/first/stacked_lmfit_summary.csv"
)
OUT_DIR = os.path.dirname(CSV)

df = pd.read_csv(CSV, usecols=["LS_iter_fap", "prob_bicc"]).dropna()
x = df["LS_iter_fap"].values
y = df["prob_bicc"].values

# ── detection-region thresholds ─────────────────────────────────────────
FAP_THR = 0.2
PROB_THR = 0.5

REGION_RECT = dict(
    type="rect",
    x0=0, x1=FAP_THR, y0=PROB_THR, y1=1.0,
    fillcolor="rgba(0,200,0,0.10)",
    line=dict(color="green", width=2, dash="dash"),
)
REGION_LABEL = dict(
    x=FAP_THR / 2, y=0.75,
    text="Detection<br>region",
    showarrow=False,
    font=dict(size=S.font_annotation, color="green"),
)

# ── 1. scatter plot ─────────────────────────────────────────────────────
fig_sc = go.Figure()
fig_sc.add_trace(go.Scattergl(
    x=x, y=y,
    mode="markers",
    marker=dict(size=5, color=S.color_data_primary, opacity=0.3),
    name="Simulations",
))
fig_sc.add_shape(**REGION_RECT)
fig_sc.add_annotation(**REGION_LABEL)
apply_style_to_layout(fig_sc, S)
fig_sc.update_layout(
    xaxis_title="LS iterative FAP",
    yaxis_title="P(binary | BIC)",
    xaxis=dict(range=[-0.02, 1.02]),
    yaxis=dict(range=[-0.02, 1.02]),
)

sc_path = os.path.join(OUT_DIR, "scatter_iter_fap_vs_prob_bicc.html")
fig_sc.write_html(sc_path)
print(f"Scatter  → {sc_path}")

# ── 2. 2D heatmap ──────────────────────────────────────────────────────
NBINS = 60
counts, xedges, yedges = np.histogram2d(x, y, bins=NBINS,
                                         range=[[0, 1], [0, 1]])
# transpose so axes match (histogram2d returns [x, y] but Heatmap wants [y, x])
counts = counts.T
# use NaN for zero bins so they appear white
counts[counts == 0] = np.nan

fig_hm = go.Figure()
fig_hm.add_trace(go.Heatmap(
    z=counts,
    x=xedges[:-1],
    y=yedges[:-1],
    colorscale="Viridis",
    colorbar=dict(title="Count"),
    hoverongaps=False,
))
fig_hm.add_shape(**REGION_RECT)
fig_hm.add_annotation(**REGION_LABEL)
apply_style_to_layout(fig_hm, S)
fig_hm.update_layout(
    xaxis_title="LS iterative FAP",
    yaxis_title="P(binary | BIC)",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
)

hm_path = os.path.join(OUT_DIR, "heatmap_iter_fap_vs_prob_bicc.html")
fig_hm.write_html(hm_path)
print(f"Heatmap  → {hm_path}")
