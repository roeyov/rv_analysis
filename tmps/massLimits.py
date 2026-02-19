import sympy as sp
import pandas as pd
import re
import os
from pathlib import Path

from utils.constants import SNR_PPL

# calculates the min mass of the companion from the measured P, e, K1 and the mass of the star from the mass_bloem.csv file
mass_file = '/Users/roeyovadia/Documents/Data/mass_bloem.csv'
best_fits_csv = '/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/with_snr_2_andBalmer/me1/finalDecisionBest.csv'
parent_dir = os.path.dirname(best_fits_csv)

# ===== Friendly name placeholder for CALC score =====
SCORE_FRIENDLY_NAME = "Luminosity-weighted SNR" # e.g. "Luminosity-weighted SNR", "CCF×SNR Visibility", "Flux-weighted detectability"


def _right_side_after_first_underscore(s: str) -> str:
    s = str(s)
    return s.split("_", 1)[1] if "_" in s else s


def q_ratio(m1,m2):
    return m2/m1

def get_L_in_sun_units(m):
    if m < 0.43:
        return 0.23*(m**2.3)
    if m < 2:
        return m**4
    if m < 55:
        return 1.4 * (m**3.5)
    else:
        return 32000*m

def L_ratio(m1,m2):
    m1_L = get_L_in_sun_units(m1)
    m2_L = get_L_in_sun_units(m2)
    return m2_L/m1_L

results = []
best_fits = pd.read_csv(best_fits_csv)
for i,row in best_fits.iterrows():
    print(f"--- Processing {row.star_name} ---")  # Added for better logging
    # This is the corrected code
    try:
        P = row["Period_value"]
        k1 = row["K1_value"]
        ecc = row["Eccentricity_value"]
        masdf = pd.read_csv(mass_file, header=0)

        mspec = masdf.loc[masdf['ID'] == f'{row.star_name}', 'Mspec'].iloc[0]
        print(f"Mspec({row.star_name}) = {mspec:.5f}")

        mspec_err_plus = masdf.loc[masdf['ID'] == f'{row.star_name}', 'Mspec_er_plus'].iloc[0]
        mspec_err_min = masdf.loc[masdf['ID'] == f'{row.star_name}', 'Mspec_er_minus'].iloc[0]  # Assuming variable name is 'Mspec_er_minus'

        # Calculate the mass function once
        f_m = (P * k1 ** 3 * (1 - ecc ** 2) ** (3 / 2)) * 1.036149e-7
        m2 = sp.symbols('M2', positive=True)

        eq = f_m * (mspec + m2) ** 2 - (m2 ** 3)
        eq2 = f_m * ((mspec + mspec_err_plus) + m2) ** 2 - (m2 ** 3)
        eq3 = f_m * ((mspec + mspec_err_min) + m2) ** 2 - (m2 ** 3)

        M2 = sp.nsolve(eq, mspec)  # A good initial guess is the primary mass itself

        # Now, use that result as the starting point for the others
        M2_plus = sp.nsolve(eq2, M2)
        M2_minus = sp.nsolve(eq3, M2)

        print(f"M2 min({row.star_name}) = {M2:.5f}")
        print(f"M2 min upper limit({row.star_name}) = {M2_plus:.5f}")
        print(f"M2 min lower limit({row.star_name}) = {M2_minus:.5f}")
        cur_res_dict = {
            "system": row.star_name,
            "M1_Msun": float(mspec),
            "M1_plus_Msun": float(mspec + mspec_err_plus),
            "M1_minus_Msun": float(mspec + mspec_err_min),
            "M2_Msun": float(M2),
            "M2_plus_Msun": float(M2_plus),
            "M2_minus_Msun": float(M2_minus),
            "P_days": float(P),
            "K1_kms": float(k1),
        }
        for j in ['','plus_','minus_']:
            one = cur_res_dict[f"M{1}_{j}Msun"]
            two = cur_res_dict[f"M{2}_{j}Msun"]
            cur_res_dict[f"q_{j}ratio"] = q_ratio(one,two)
            cur_res_dict[f"L_{j}ratio"] = L_ratio(one,two)
            cur_res_dict[SNR_PPL] = row[SNR_PPL]
            cur_res_dict["SNR_CALC"] = row["SNR"]
            cur_res_dict[f"CALC_{j}score"] = row["SNR"]*cur_res_dict[f"L_{j}ratio"]
            cur_res_dict[f"PPL_{j}score"] = cur_res_dict[SNR_PPL] *cur_res_dict[f"L_{j}ratio"]
        results.append(cur_res_dict)
    except (KeyError, IndexError) as e:
        print(f"Could not process {row.star_name}. Error: {e}")
    except ValueError as e:
        print(f"Numerical solver failed for {row.star_name}. Error: {e}")


df = pd.DataFrame(results)
df.to_csv(os.path.join(parent_dir, "binary_masses.csv"),index=False)
import matplotlib.pyplot as plt
# === Plots: M2 vs score (CALC and PPL), with +/- error bars and x=2 line ===
import numpy as np
import matplotlib.pyplot as plt

def _plot_score_vs_M2(
    df,
    score_prefix: str,
    out_path: str,
    dpi: int = 200,
    log: bool = False,
    x_thresh: float = 2.0,   # vertical threshold (e.g., M2 = 2)
    y_thresh: float = 1.0,   # horizontal threshold
    label_col: str = "system"
):
    """
    Plots M2 (x) vs <score_prefix>_score (y) with correct min/max asymmetric error bars,
    computed safely regardless of whether plus/minus swapped order.
    Adds vertical line at x_thresh and horizontal line at y_thresh.
    Annotates points that are strictly above both thresholds with their system name.
    """
    needed_cols = [
        "M2_Msun", "M2_plus_Msun", "M2_minus_Msun",
        f"{score_prefix}_score", f"{score_prefix}_plus_score", f"{score_prefix}_minus_score",
    ]

    # Try to attach labels; tolerate if missing
    label_source = None
    if label_col in df.columns:
        label_source = label_col
    elif "star_name" in df.columns:
        label_source = "star_name"

    use_cols = needed_cols + ([label_source] if label_source else [])
    sub = df[use_cols].apply(pd.to_numeric, errors="coerce") if not label_source else df[use_cols].copy()
    sub = sub.dropna()
    if sub.empty:
        print(f"[WARN] No valid rows for {score_prefix}. Skipping plot.")
        return

    # Core arrays
    x  = pd.to_numeric(sub["M2_Msun"], errors="coerce").to_numpy()
    x1 = pd.to_numeric(sub["M2_plus_Msun"], errors="coerce").to_numpy()
    x2 = pd.to_numeric(sub["M2_minus_Msun"], errors="coerce").to_numpy()

    y  = pd.to_numeric(sub[f"{score_prefix}_score"], errors="coerce").to_numpy()
    y1 = pd.to_numeric(sub[f"{score_prefix}_plus_score"], errors="coerce").to_numpy()
    y2 = pd.to_numeric(sub[f"{score_prefix}_minus_score"], errors="coerce").to_numpy()

    # Compute true lower/upper limits
    x_min = np.minimum(x1, x2)
    x_max = np.maximum(x1, x2)
    y_min = np.minimum(y1, y2)
    y_max = np.maximum(y1, y2)

    # Proper asymmetric errors (always positive)
    xerr_lower = np.maximum(0.0, x - x_min)
    xerr_upper = np.maximum(0.0, x_max - x)
    yerr_lower = np.maximum(0.0, y - y_min)
    yerr_upper = np.maximum(0.0, y_max - y)

    # PPT aesthetics
    plt.rcParams.update({
        "axes.titlesize": 24,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.titlesize": 26,
    })

    fig, ax = plt.subplots(figsize=(13.33, 7.5), constrained_layout=True)

    ax.errorbar(
        x, y,
        xerr=[xerr_lower, xerr_upper],
        yerr=[yerr_lower, yerr_upper],
        fmt="o", ms=4, elinewidth=0.9, capsize=2, alpha=0.9
    )

    # Threshold lines
    ax.axvline(x_thresh, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(y_thresh, linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xlabel(r"$M_2\ \mathrm{[M_\odot]}$")
    ax.set_ylabel(f"{score_prefix} score")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"M2 vs {score_prefix} score")

    if log:
        ax.set_yscale("log")
        ax.set_xscale("log")

    # ---- Annotate systems above BOTH thresholds ----
    if label_source:
        labels = sub[label_source].astype(str).to_numpy()
        mask = (x > x_thresh) & (y > y_thresh)
        # Slight alternating offsets to reduce label overlap
        dx_dy = [(5, 5), (5, -5), (-5, 5), (-5, -5)]
        k = 0
        for xi, yi, lab in zip(x[mask], y[mask], labels[mask]):
            dx, dy = dx_dy[k % len(dx_dy)]
            k += 1
            ax.annotate(
                lab,
                xy=(xi, yi),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=12,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.8)
            )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# Output paths (adjust as desired)
_calc_out = os.path.join(parent_dir, "M2_vs_CALC.png")
_ppl_out  = os.path.join(parent_dir, "M2_vs_PPL.png")

_plot_score_vs_M2(df, "CALC", _calc_out, log=False, x_thresh=2.0, y_thresh=1.0)
_plot_score_vs_M2(df, "PPL",  _ppl_out,  log=False, x_thresh=2.0, y_thresh=1.0)

# ========= Plotly interactive HTML: M2 vs score with hover info =========
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def _pick_col(df: pd.DataFrame, *cands):
    """Return the first column name that exists in df among candidates."""
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns found: {cands}")

def plotly_score_vs_M2_html(
    df: pd.DataFrame,
    score_prefix: str,
    out_html: str,
    log_x: bool = False,
    log_y: bool = False,
    title: str | None = None,
):
    """
    Interactive HTML scatter:
      x = M2, y = <score_prefix>_score
      Asymmetric error bars from +/- variants (order-agnostic).
      Hover shows: system (star_name), M1 ±, M2 ±, q ±.
    """

    # Resolve column names (tolerate single-underscore & double-underscore cases)
    c_system = _pick_col(df, "system", "star_name")

    c_M1      = _pick_col(df, "M1_Msun", "M1__Msun")
    c_M1_plus = _pick_col(df, "M1_plus_Msun", "M1_plus__Msun")
    c_M1_min  = _pick_col(df, "M1_minus_Msun", "M1_minus__Msun")

    c_M2      = _pick_col(df, "M2_Msun", "M2__Msun")
    c_M2_plus = _pick_col(df, "M2_plus_Msun", "M2_plus__Msun")
    c_M2_min  = _pick_col(df, "M2_minus_Msun", "M2_minus__Msun")

    # score columns
    c_score      = _pick_col(df, f"{score_prefix}_score", f"{score_prefix}__score")
    c_score_plus = _pick_col(df, f"{score_prefix}_plus_score", f"{score_prefix}_plus__score")
    c_score_min  = _pick_col(df, f"{score_prefix}_minus_score", f"{score_prefix}_minus__score")

    # q columns
    c_q      = _pick_col(df, "q_ratio", "q__ratio")
    c_q_plus = _pick_col(df, "q_plus_ratio", "q_plus__ratio")
    c_q_min  = _pick_col(df, "q_minus_ratio", "q_minus__ratio")

    needed = [
        c_system,
        c_M1, c_M1_plus, c_M1_min,
        c_M2, c_M2_plus, c_M2_min,
        c_score, c_score_plus, c_score_min,
        c_q, c_q_plus, c_q_min,
    ]

    sub = df[needed].apply(pd.to_numeric, errors="ignore").dropna()
    if sub.empty:
        print(f"[WARN] No valid rows for {score_prefix}. Skipping Plotly plot.")
        return

    # Core arrays
    x  = pd.to_numeric(sub[c_M2], errors="coerce").to_numpy()
    x1 = pd.to_numeric(sub[c_M2_plus], errors="coerce").to_numpy()
    x2 = pd.to_numeric(sub[c_M2_min], errors="coerce").to_numpy()

    y  = pd.to_numeric(sub[c_score], errors="coerce").to_numpy()
    y1 = pd.to_numeric(sub[c_score_plus], errors="coerce").to_numpy()
    y2 = pd.to_numeric(sub[c_score_min], errors="coerce").to_numpy()

    # Asymmetric errors (order-agnostic)
    x_min = np.minimum(x1, x2)
    x_max = np.maximum(x1, x2)
    y_min = np.minimum(y1, y2)
    y_max = np.maximum(y1, y2)

    xerr_minus = np.maximum(0.0, x - x_min)
    xerr_plus  = np.maximum(0.0, x_max - x)
    yerr_minus = np.maximum(0.0, y - y_min)
    yerr_plus  = np.maximum(0.0, y_max - y)

    # Customdata for hover
    system = sub[c_system].astype(str).to_numpy()
    M1      = pd.to_numeric(sub[c_M1], errors="coerce").to_numpy()
    M1p     = pd.to_numeric(sub[c_M1_plus], errors="coerce").to_numpy()
    M1m     = pd.to_numeric(sub[c_M1_min], errors="coerce").to_numpy()
    M2      = x
    M2p     = x_max  # use resolved max/min to reflect direction-agnostic reporting
    M2m     = x_min

    q       = pd.to_numeric(sub[c_q], errors="coerce").to_numpy()
    qp      = pd.to_numeric(sub[c_q_plus], errors="coerce").to_numpy()
    qm      = pd.to_numeric(sub[c_q_min], errors="coerce").to_numpy()

    # ---- inside plotly_score_vs_M2_html() ----

    # Precompute delta values for hover display
    M1_err_minus = M1 - M1m
    M1_err_plus = M1p - M1
    M2_err_minus = M2 - M2m
    M2_err_plus = M2p - M2
    q_err_minus = q - qm
    q_err_plus = qp - q

    # customdata columns for hover:
    # (system, M1, M1-, M1+, M1_err-, M1_err+, M2, M2_err-, M2_err+, q, q_err-, q_err+)
    custom = np.column_stack([
        system, M1, M1m, M1p, M1_err_minus, M1_err_plus,
        M2, M2_err_minus, M2_err_plus,
        q, q_err_minus, q_err_plus
    ])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=7, opacity=0.9),
            error_x=dict(visible=True, type="data", array=xerr_plus, arrayminus=xerr_minus, thickness=1),
            error_y=dict(visible=True, type="data", array=yerr_plus, arrayminus=yerr_minus, thickness=1),
            customdata=custom,
            hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "M1: %{customdata[1]:.3f} [−%{customdata[4]:.3f}, +%{customdata[5]:.3f}] M⊙<br>"
                    "M2: %{customdata[6]:.3f} [−%{customdata[7]:.3f}, +%{customdata[8]:.3f}] M⊙<br>"
                    "q:  %{customdata[9]:.3f} [−%{customdata[10]:.3f}, +%{customdata[11]:.3f}]<br>"
                    + f"{score_prefix} score: " + "%{y:.3f}<extra></extra>"
            ),
            name=f"{score_prefix} score",
        )
    )

    # Vertical line at M2 = 2
    fig.add_vline(x=2.0, line_dash="dash", line_width=1.2, opacity=0.7)
    fig.add_hline(1.0, line_dash="dash", line_width=1.2, opacity=0.7)

    fig.update_layout(
        title=title or f"M2 vs {score_prefix} score",
        xaxis_title="M2 [M⊙]",
        yaxis_title=f"{score_prefix} score",
        width=1200, height=650,
        template="plotly_white",
        margin=dict(l=70, r=30, t=70, b=70),
    )

    fig.update_xaxes(type="log" if log_x else "linear")
    fig.update_yaxes(type="log" if log_y else "linear")

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved interactive HTML: {out_html.resolve()}")

# ---- Calls (mirror your PNGs) ----
_calc_html = os.path.join(parent_dir,  "M2_vs_CALC.html")
_ppl_html  = os.path.join(parent_dir,  "M2_vs_PPL.html")

plotly_score_vs_M2_html(df, "CALC", _calc_html, log_x=True, log_y=True)
plotly_score_vs_M2_html(df, "PPL",  _ppl_html,  log_x=True, log_y=True)

import numpy as np
import plotly.graph_objects as go

def _right_side_after_first_underscore(s: str) -> str:
    if "_" in s:
        return s.split("_", 1)[1]
    return s

def plotly_M2_vs_P_with_K1(
    df: pd.DataFrame,
    out_html_linear: str,
    out_html_logy: str,
    title_base: str = "M2 vs Period"
):
    """
    Scatter: x = Period [days] (log scale), y = M2 [M⊙] with asymmetric error bars,
    marker color = K1 [km/s].
    Adds horizontal lines at 3 and 5 M⊙ (with legend), and red shaded region for M2 >= 3 M⊙
    labeled 'Black holes region'. Each point is annotated by the right side of '_' from 'system'.
    Saves two HTML files: linear y and log y.
    """
    # Required columns
    needed = [
        "system",
        "P_days",
        "M1_Msun",
        "M2_Msun", "M2_plus_Msun", "M2_minus_Msun",
    ]
    sub = df[needed].copy()
    sub = sub.dropna()
    if sub.empty:
        print("[WARN] No valid rows for M2 vs Period plot.")
        return

    # Core arrays
    xP   = pd.to_numeric(sub["P_days"], errors="coerce").to_numpy()
    # K1   = pd.to_numeric(sub["K1_kms"], errors="coerce").to_numpy()
    M1   = pd.to_numeric(sub["M1_Msun"], errors="coerce").to_numpy()

    yM2  = pd.to_numeric(sub["M2_Msun"], errors="coerce").to_numpy()
    y1   = pd.to_numeric(sub["M2_plus_Msun"], errors="coerce").to_numpy()
    y2   = pd.to_numeric(sub["M2_minus_Msun"], errors="coerce").to_numpy()

    # Asymmetric (order-agnostic) errors
    y_min = np.minimum(y1, y2)
    y_max = np.maximum(y1, y2)
    yerr_minus = np.maximum(0.0, yM2 - y_min)
    yerr_plus  = np.maximum(0.0, y_max - yM2)

    # Labels from right side after first underscore
    labels = [ _right_side_after_first_underscore(s) for s in sub["system"].astype(str) ]

    def _make_figure(log_y: bool):
        fig = go.Figure()

        # Main scatter with K1 colormap + y error bars
        fig.add_trace(
            go.Scatter(
                x=xP, y=yM2, mode="markers+text",
                text=labels,
                textposition="middle right",
                textfont=dict(size=10),
                # marker=dict(
                #     size=8,
                #     color=K1,
                #     colorscale="Viridis",
                #     colorbar=dict(title="K1 [km/s]", x=1.03),  # move colorbar a bit right
                #     showscale=True,
                # ),
                marker=dict(
                    size=8,
                    color=M1,
                    colorscale="Viridis",
                    colorbar=dict(title="M1 [M⊙]", x=1.03),  # move colorbar a bit right
                    showscale=True,
                ),
                error_y=dict(visible=True, type="data", array=yerr_plus, arrayminus=yerr_minus, thickness=1),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "System: %{customdata[0]}<br>"
                    "P: %{x:.4g} d<br>"
                    "M2: %{y:.3f} M⊙ [−%{customdata[1]:.3f}, +%{customdata[2]:.3f}]<br>"
                    "K1: %{marker.color:.3f} km/s<extra></extra>"
                ),
                customdata=np.column_stack([sub["system"].astype(str).to_numpy(), yerr_minus, yerr_plus]),
                name="Systems",
            )
        )

        # Horizontal lines
        fig.add_trace(go.Scatter(
            x=[np.nanmin(xP), np.nanmax(xP)], y=[3, 3],
            mode="lines", line=dict(dash="dash"), name="M2 = 3 M⊙", hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=[np.nanmin(xP), np.nanmax(xP)], y=[5, 5],
            mode="lines", line=dict(dash="dot"), name="M2 = 5 M⊙", hoverinfo="skip"
        ))

        # Red shaded region
        y_top = max(np.nanmax(y_max) * 1.05, 6.0)
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=np.nanmin(xP), x1=np.nanmax(xP),
            y0=3.0, y1=y_top,
            fillcolor="red", opacity=0.09, line=dict(width=0), layer="below"
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color="red", opacity=0.3),
            name="Black holes region (M2 ≥ 3 M⊙)"
        ))

        # ✅ Updated layout: legend below, colorbar spaced, compact
        fig.update_layout(
            title=f"{title_base}",
            xaxis_title="Period [days] (log scale)",
            yaxis_title="M2 [M⊙]",
            template="plotly_white",
            width=800, height=650,
            margin=dict(l=60, r=60, t=70, b=90),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            ),
        )

        fig.update_xaxes(type="log", showgrid=True)
        fig.update_yaxes(type="log" if log_y else "linear", showgrid=True)
        return fig

    # Save linear-y
    fig_linear = _make_figure(log_y=False)
    fig_linear.write_html(out_html_linear, include_plotlyjs="cdn")
    print(f"Saved interactive HTML (linear y): {Path(out_html_linear).resolve()}")

    # Save log-y
    fig_logy = _make_figure(log_y=True)
    fig_logy.write_html(out_html_logy, include_plotlyjs="cdn")
    print(f"Saved interactive HTML (log y): {Path(out_html_logy).resolve()}")

# ---- Calls ----
_m2p_linear = os.path.join(parent_dir, "M2_vs_Period_K1_linearY.html")
_m2p_logy   = os.path.join(parent_dir, "M2_vs_Period_K1_logY.html")

plotly_M2_vs_P_with_K1(df, _m2p_linear, _m2p_logy)


import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def plotly_M2_vs_score_pretty(
    df: pd.DataFrame,
    score_prefix: str,              # "CALC"
    out_html_linear: str,
    out_html_logy: str,
    x_bands=( (0,2,"M2 < 2 M⊙","rgba(0,0,0,0.03)"),
              (2,3,"2 ≤ M2 < 3 M⊙ (NS/BH cand.)","rgba(255,165,0,0.08)"),
              (3,None,"M2 ≥ 3 M⊙ (BH region)","rgba(255,0,0,0.09)") ),
    y_thresh=1.0,                   # score threshold
    show_y_band=True                # horizontal band for high scores
):
    """
    Conference-style M2 vs <score> plot:
    - x = M2 [M⊙] (linear), y = <score> (linear/log)
    - vertical bands for M2 zones (with legend)
    - optional horizontal band for y >= y_thresh
    - error bars thinner & semi-transparent
    - labels are the right part of '_' in 'system'
    - legend below; clearer gridlines
    Produces two HTML files: linear y and log y.
    """
    # Resolve columns (re-use your existing names)
    c_system = "system" if "system" in df.columns else "star_name"
    needed = [
        c_system,
        "M2_Msun", "M2_plus_Msun", "M2_minus_Msun",
        f"{score_prefix}_score", f"{score_prefix}_plus_score", f"{score_prefix}_minus_score"
    ]
    sub = df[needed].copy().dropna()
    if sub.empty:
        print(f"[WARN] No valid rows for {score_prefix}. Skipping pretty plot.")
        return

    # Core arrays
    x  = pd.to_numeric(sub["M2_Msun"], errors="coerce").to_numpy()
    xp = pd.to_numeric(sub["M2_plus_Msun"], errors="coerce").to_numpy()
    xm = pd.to_numeric(sub["M2_minus_Msun"], errors="coerce").to_numpy()

    y  = pd.to_numeric(sub[f"{score_prefix}_score"], errors="coerce").to_numpy()
    yp = pd.to_numeric(sub[f"{score_prefix}_plus_score"], errors="coerce").to_numpy()
    ym = pd.to_numeric(sub[f"{score_prefix}_minus_score"], errors="coerce").to_numpy()

    # Order-agnostic asymmetric errors
    x_min = np.minimum(xp, xm); x_max = np.maximum(xp, xm)
    y_min = np.minimum(yp, ym); y_max = np.maximum(yp, ym)
    xerr_minus = np.maximum(0.0, x - x_min); xerr_plus  = np.maximum(0.0, x_max - x)
    yerr_minus = np.maximum(0.0, y - y_min); yerr_plus  = np.maximum(0.0, y_max - y)

    labels = [_right_side_after_first_underscore(s) for s in sub[c_system].astype(str)]
    score_name = SCORE_FRIENDLY_NAME

    def _make_fig(log_y: bool):
        fig = go.Figure()

        # ---- Vertical M2 bands (as shapes) + legend dummies ----
        x_lo = float(np.nanmin(x_min)) if np.isfinite(x_min).any() else float(np.nanmin(x))
        x_hi = float(np.nanmax(x_max)) if np.isfinite(x_max).any() else float(np.nanmax(x))
        # Expand slightly for aesthetics
        x_lo_plot = max(0.0, x_lo - 0.05*(x_hi - x_lo))
        x_hi_plot = x_hi + 0.05*(x_hi - x_lo)

        for i, (x0, x1, name, rgba) in enumerate(x_bands):
            x0p = x0 if x0 is not None else x_lo_plot
            x1p = x1 if x1 is not None else x_hi_plot
            fig.add_shape(
                type="rect", xref="x", yref="paper",
                x0=x0p, x1=x1p, y0=0, y1=1,
                fillcolor=rgba, line=dict(width=0), layer="below"
            )
            # Dummy for legend
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=rgba),
                name=name, showlegend=True, hoverinfo="skip"
            ))

        # ---- Optional horizontal high-score band ----
        if show_y_band and np.isfinite(y).any():
            y_hi = float(np.nanmax(y_max))
            y_lo = float(np.nanmin(y_min))
            fig.add_shape(
                type="rect", xref="paper", yref="y",
                x0=0, x1=1, y0=y_thresh, y1=y_hi + 0.05*(y_hi - y_lo),
                fillcolor="rgba(0,128,0,0.06)", line=dict(width=0), layer="below"
            )
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color="rgba(0,128,0,0.25)"),
                name=f"{score_name} ≥ {y_thresh:g}", showlegend=True, hoverinfo="skip"
            ))

        # ---- Threshold reference lines ----
        # common lines at M2=2,3,5; and y = y_thresh
        for xv, dash, nm in [(2.0,"dash","M2 = 2 M⊙"),
                             (3.0,"dash","M2 = 3 M⊙"),
                             (5.0,"dot","M2 = 5 M⊙")]:
            fig.add_vline(x=xv, line_dash=dash, line_width=1.2, opacity=0.7)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nm, showlegend=True, hoverinfo="skip"))

        fig.add_hline(y=y_thresh, line_dash="dash", line_width=1.2, opacity=0.7)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"{score_name} = {y_thresh:g}", showlegend=True, hoverinfo="skip"))

        # ---- Main scatter (subtle error bars; labels) ----
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers+text",
                text=labels, textposition="middle right",
                textfont=dict(size=10, color="rgba(0,0,0,0.85)"),
                marker=dict(size=8, color="rgba(0,0,0,0.75)",
                            line=dict(width=1, color="rgba(255,255,255,0.9)")),
                error_x=dict(visible=True, type="data",
                             array=xerr_plus, arrayminus=xerr_minus,
                             thickness=0.7, color="rgba(0,0,0,0.25)"),
                error_y=dict(visible=True, type="data",
                             array=yerr_plus, arrayminus=yerr_minus,
                             thickness=0.7, color="rgba(0,0,0,0.25)"),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "System: %{customdata[0]}<br>"
                    "M2: %{x:.3f} M⊙ [−%{customdata[1]:.3f}, +%{customdata[2]:.3f}]<br>"
                    f"{score_prefix} score: " + "%{y:.3f} [−%{customdata[3]:.3f}, +%{customdata[4]:.3f}]"
                    "<extra></extra>"
                ),
                customdata=np.column_stack([sub[c_system].astype(str).to_numpy(),
                                            xerr_minus, xerr_plus, yerr_minus, yerr_plus]),
                name="Systems"
            )
        )

        # ---- Layout polish ----
        fig.update_layout(
            title=f"M2 vs {score_name} — {'log-y' if log_y else 'linear y'}",
            xaxis_title="M2 [M⊙]",
            yaxis_title=score_name,
            template="plotly_white",
            width=1200, height=650,
            margin=dict(l=70, r=60, t=70, b=100),
            legend=dict(
                orientation="h", x=0.5, xanchor="center",
                y=-0.25, yanchor="top",
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="rgba(0,0,0,0.15)", borderwidth=1
            )
        )

        # Clearer gridlines (visible under bands)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.25)", gridwidth=1.0)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.25)", gridwidth=1.0,
                         type="log" if log_y else "linear")

        # Expand x range to include band padding
        fig.update_xaxes(range=[x_lo_plot, x_hi_plot])

        return fig

    # ---- Write outputs ----
    out_html_linear = Path(out_html_linear)
    out_html_logy   = Path(out_html_logy)
    out_html_linear.parent.mkdir(parents=True, exist_ok=True)

    fig_lin = _make_fig(log_y=False)
    fig_lin.write_html(out_html_linear, include_plotlyjs="cdn")
    print(f"Saved: {out_html_linear.resolve()}")

    fig_log = _make_fig(log_y=True)
    fig_log.write_html(out_html_logy, include_plotlyjs="cdn")
    print(f"Saved: {out_html_logy.resolve()}")

_calc_pretty_linear = os.path.join(parent_dir, "M2_vs_CALC_pretty_linearY.html")
_calc_pretty_logy   =  os.path.join(parent_dir, "M2_vs_CALC_pretty_logY.html")

plotly_M2_vs_score_pretty(df, "CALC", _calc_pretty_linear, _calc_pretty_logy, y_thresh=1.0)
