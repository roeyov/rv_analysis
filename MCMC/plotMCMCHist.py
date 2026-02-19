# =========================
# Clean imports (once)
# =========================
import os
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# 1) Combined + individual ecc_*_mode histograms (fixed)
# =========================================================

def _label_with_units(param: str, is_log: bool) -> str:
    """
    Return axis label with physical units in square brackets.
    Uses log₁₀(...) when is_log=True.
    """
    # units in square brackets
    unit_map = {
        "P": "days",
        "T0": "days",          # or "MJD" if that's what you use; keep as days unless you prefer
        "omega": "rad",
        "e": "",               # dimensionless
        "K1": "km s⁻¹",
        "gamma": "km s⁻¹",
        "log_sj": "km s⁻¹",    # σ_J units; even if param name is log_sj we show σ_J
        "sj": "km s⁻¹",
        "sigmaJ": "km s⁻¹",
    }

    # pretty symbol overrides (optional)
    pretty = {
        "P": "P",
        "T0": "T₀",
        "omega": "ω",
        "e": "e",
        "K1": "K₁",
        "gamma": "γ",
        "log_sj": "σ_J",
        "sj": "σ_J",
        "sigmaJ": "σ_J",
    }

    sym = pretty.get(param, param)
    unit = unit_map.get(param, "")

    if is_log:
        # log label still carries the same physical unit in brackets
        if unit:
            return f"log₁₀({sym}) [{unit}]"
        return f"log₁₀({sym})"
    else:
        if unit:
            return f"{sym} [{unit}]"
        return f"{sym}"

def _xbins_for_exact_n(x, nbins):
    x = np.asarray(x)
    if x.size == 0:
        return None
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmin == xmax:
        eps = 1e-6 if xmin == 0 else abs(xmin) * 1e-6
        xmin -= eps
        xmax += eps
    size = (xmax - xmin) / nbins
    return dict(start=xmin, end=xmax, size=size)

def plot_eccentric_mode_histograms_physics_save_all(
    df,
    outdir=".",
    outname_prefix="ecc_mode_hist",
    nbins=45,
    ncols=3,
    log_params=("P",),
    add_quantile_lines=True,
    qlo=0.16,
    qhi=0.84,
    add_median_line=True,
    add_mean_to_titles=True,
    export=("html", "png"),
    image_scale=3,
):
    """
    Combined grid + individual histograms for ecc_{param}_mode.
    Each parameter has a fixed color shared between grid and individual plots.

    Fixes:
      - Removed undefined variable 'c' usage in individual plots
      - Robustly excludes ecc_log_sj_mode only (no accidental prefix pitfalls)
      - Guards log transforms (values must be > 0)
      - Cleans infs
    """

    os.makedirs(outdir, exist_ok=True)

    PARAM_COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    # Only exclude the specific unwanted column (most robust)
    ecc_mode_cols = sorted([
        c for c in df.columns
        if c.startswith("ecc_") and c.endswith("_mode") and c != "ecc_log_sj_mode"
    ])

    if not ecc_mode_cols:
        raise ValueError("No ecc_*_mode columns found (after excluding ecc_log_sj_mode).")

    # ---- collect plotted data ----
    plotted = []
    for i, col in enumerate(ecc_mode_cols):
        param = col.replace("ecc_", "").replace("_mode", "")
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if vals.size == 0:
            continue

        if param in log_params:
            vals = vals[vals > 0]
            if vals.size == 0:
                continue
            x_vals = np.log10(vals)
            x_label = _label_with_units(param, is_log=True)
        else:
            x_vals = vals
            x_label = _label_with_units(param, is_log=False)

        plotted.append(dict(
            param=param,
            x_vals=x_vals,
            x_label=x_label,
            mean=float(np.mean(x_vals)),
            n=int(x_vals.size),
            color=PARAM_COLORS[i % len(PARAM_COLORS)]
        ))

    if len(plotted) == 0:
        raise ValueError("All ecc_*_mode columns were empty after cleaning/log-guards.")

    # ---- helpers ----
    def _export(fig, base):
        export_set = set(export)
        if "html" in export_set:
            fig.write_html(base + ".html", include_plotlyjs="cdn", include_mathjax="cdn")
        for fmt in ("png", "pdf", "svg"):
            if fmt in export_set:
                fig.write_image(base + f".{fmt}", scale=image_scale)

    def _style_axes(fig, r, c):
        fig.update_xaxes(
            row=r, col=c,
            showline=True, linewidth=1.2,
            ticks="outside", ticklen=6,
            mirror=True,
            showgrid=True, gridwidth=1,
            zeroline=False,
        )
        fig.update_yaxes(
            row=r, col=c,
            showline=True, linewidth=1.2,
            ticks="outside", ticklen=6,
            mirror=True,
            showgrid=True, gridwidth=1,
            zeroline=False,
        )

    # =========================================================
    # Combined grid
    # =========================================================
    nparams = len(plotted)
    nrows = math.ceil(nparams / ncols)

    titles = [
        f"{_label_with_units(d['param'],is_log=False)} (mean={d['mean']:.3g}, N={d['n']})"
        if add_mean_to_titles else f"{_label_with_units(d['param'],is_log=False)} (N={d['n']})"
        for d in plotted
    ]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.14,
    )

    fig.update_layout(
        template="simple_white",
        font=dict(family="Times New Roman", size=18),
        title=dict(
            text="Eccentric Keplerian MCMC — posterior distributions",
            x=0.5, xanchor="center"
        ),
        width=1300,
        height=300 * nrows,
        margin=dict(l=60, r=40, t=90, b=60),
    )

    for i, d in enumerate(plotted):
        r = i // ncols + 1
        c = i % ncols + 1
        xb = _xbins_for_exact_n(d["x_vals"], nbins)

        fig.add_trace(
            go.Histogram(
                x=d["x_vals"],
                xbins=xb,
                marker=dict(color=d["color"]),
                showlegend=False,
            ),
            row=r, col=c
        )

        if add_quantile_lines and d["n"] > 10:
            lo = np.quantile(d["x_vals"], qlo)
            hi = np.quantile(d["x_vals"], qhi)
            fig.add_vline(x=lo, row=r, col=c, line_dash="dot", line_width=2)
            fig.add_vline(x=hi, row=r, col=c, line_dash="dot", line_width=2)

        if add_median_line and d["n"] > 10:
            fig.add_vline(x=np.median(d["x_vals"]), row=r, col=c, line_width=2)

        if r == nrows:
            fig.update_xaxes(title_text=d["x_label"], row=r, col=c)
        else:
            fig.update_xaxes(title_text="", row=r, col=c)

        fig.update_yaxes(title_text="N of Systems" if c == 1 else "", row=r, col=c)
        _style_axes(fig, r, c)

    fig.update_annotations(font=dict(size=16, family="Times New Roman"))
    _export(fig, os.path.join(outdir, f"{outname_prefix}_grid"))

    # =========================================================
    # Individual histograms (same colors)
    # =========================================================
    indiv_figs = {}

    for d in plotted:
        f = go.Figure()
        xb = _xbins_for_exact_n(d["x_vals"], nbins)

        f.add_trace(
            go.Histogram(
                x=d["x_vals"],
                xbins=xb,
                marker=dict(color=d["color"]),
                showlegend=False,
            )
        )

        if add_quantile_lines and d["n"] > 10:
            f.add_vline(x=np.quantile(d["x_vals"], qlo), line_dash="dot", line_width=2)
            f.add_vline(x=np.quantile(d["x_vals"], qhi), line_dash="dot", line_width=2)

        if add_median_line and d["n"] > 10:
            f.add_vline(x=np.median(d["x_vals"]), line_width=2)

        title = f"{d['param']} distribution"
        if add_mean_to_titles:
            title += f" (mean={d['mean']:.3g}, N={d['n']})"

        f.update_layout(
            template="simple_white",
            font=dict(family="Times New Roman", size=20),
            title=dict(text=title, x=0.5, xanchor="center"),
            width=900,
            height=650,
            margin=dict(l=80, r=40, t=80, b=80),
        )

        f.update_xaxes(
            title=d["x_label"],
            showline=True, linewidth=1.6, mirror=True,
            ticks="outside", ticklen=6,
            showgrid=True, gridwidth=1,
            zeroline=False,
        )

        # FIX: 'c' was undefined here; always title for single plots
        f.update_yaxes(
            title="N of Systems",
            showline=True, linewidth=1.6, mirror=True,
            ticks="outside", ticklen=6,
            showgrid=True, gridwidth=1,
            zeroline=False,
        )

        base = os.path.join(outdir, f"{outname_prefix}_{d['param']}")
        _export(f, base)
        indiv_figs[d["param"]] = f

    return fig, indiv_figs


# =========================================================
# 2) Build star_gamma df (fixed robust field parsing)
# =========================================================
def build_star_gamma_df(
    df,
    star_col="star_name",
    null_gamma_col="null_gamma_mode",
    ecc_gamma_col="ecc_gamma_mode",
):
    """
    Build a dataframe with columns:
        - star_name
        - gamma
        - is_ecc (bool)
        - field_id (int)

    Rules:
        - If null_gamma_mode is not NaN -> take it (is_ecc = False)
        - Else if ecc_gamma_mode is not NaN -> take it (is_ecc = True)
        - No row may have BOTH gamma values non-NaN (raises ValueError)
        - Rows with neither gamma are dropped

    star_name accepted formats:
        - BLOeM_{field_id}-{star_id}
        - BLOeM_{field_id}_{star_id}
    """

    for c in (star_col, null_gamma_col, ecc_gamma_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    s = df[[star_col, null_gamma_col, ecc_gamma_col]].copy()
    s = s.replace([np.inf, -np.inf], np.nan)

    null_ok = s[null_gamma_col].notna()
    ecc_ok  = s[ecc_gamma_col].notna()

    both = null_ok & ecc_ok
    if both.any():
        bad = s.loc[both, [star_col, null_gamma_col, ecc_gamma_col]].head(10)
        raise ValueError(
            f"Found {both.sum()} rows with BOTH {null_gamma_col} and {ecc_gamma_col} non-NaN.\n"
            f"Example rows:\n{bad}"
        )

    s["gamma"] = np.where(null_ok, s[null_gamma_col], s[ecc_gamma_col])
    s["is_ecc"] = ecc_ok
    s = s.dropna(subset=["gamma"]).copy()

    # FIX: robust regex supports '-' or '_' after field_id
    field_ids = s[star_col].astype(str).str.extract(r"BLOeM_(\d+)[-_]", expand=False)
    if field_ids.isna().any():
        bad_names = s.loc[field_ids.isna(), star_col].head(5)
        raise ValueError(
            "Failed to extract field_id from some star_name values.\n"
            f"Examples:\n{bad_names}"
        )

    s["field_id"] = field_ids.astype(int)

    out = s[[star_col, "gamma", "is_ecc", "field_id"]].copy()
    # keep your original removal
    return out[(out[star_col] != "BLOeM_4-040") & (out[star_col] != "BLOeM_4-041")]


# =========================================================
# 3) Gamma histogram (single)
# =========================================================
def plot_gamma_histogram_physics(
    star_gamma_df,
    outpath_base="gamma_hist",
    nbins=45,
    add_quantile_lines=True,
    qlo=0.16,
    qhi=0.84,
    add_median_line=True,
    add_mean_to_title=True,
    export=("html", "png"),
    image_scale=3,
    gamma_color="#8c564b",
):
    x = star_gamma_df["gamma"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if x.size == 0:
        raise ValueError("No gamma values to plot after dropping NaNs.")

    mean_val = float(np.mean(x))
    med_val  = float(np.median(x))

    title = "γ mode distribution"
    if add_mean_to_title:
        title += f" (mean={mean_val:.3g}, N={x.size})"

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x, nbinsx=nbins,
        marker=dict(color=gamma_color),
        showlegend=False,
    ))

    if add_quantile_lines and x.size > 10:
        lo = float(np.quantile(x, qlo))
        hi = float(np.quantile(x, qhi))
        fig.add_vline(x=lo, line_dash="dot", line_width=2)
        fig.add_vline(x=hi, line_dash="dot", line_width=2)

    if add_median_line and x.size > 10:
        fig.add_vline(x=med_val, line_width=2)

    fig.update_layout(
        template="simple_white",
        font=dict(family="Times New Roman", size=20),
        title=dict(text=title, x=0.5, xanchor="center"),
        width=900,
        height=650,
        margin=dict(l=80, r=40, t=80, b=80),
    )
    fig.update_xaxes(
        title="γ (km s⁻¹)",
        showline=True, linewidth=1.6, mirror=True,
        ticks="outside", ticklen=6,
        showgrid=True, gridwidth=1,
        zeroline=False,
    )
    fig.update_yaxes(
        title="N of systems",
        showline=True, linewidth=1.6, mirror=True,
        ticks="outside", ticklen=6,
        showgrid=True, gridwidth=1,
        zeroline=False,
    )

    export_set = set(export)
    if "html" in export_set:
        fig.write_html(outpath_base + ".html", include_plotlyjs="cdn", include_mathjax="cdn")
        print(f"Saved HTML: {outpath_base}.html")
    for fmt in ("png", "pdf", "svg"):
        if fmt in export_set:
            fig.write_image(outpath_base + f".{fmt}", scale=image_scale)
            print(f"Saved {fmt.upper()}: {outpath_base}.{fmt}")

    return fig


# =========================================================
# 4) Gamma stacked histograms (ok, just kept)
# =========================================================
def plot_gamma_histograms_stacked(
    star_gamma_df,
    outpath_base="gamma_hist",
    nbins=45,
    export=("html", "png"),
    image_scale=3,
    color_ecc="#d62728",
    color_null="#1f77b4",
):
    if "gamma" not in star_gamma_df.columns:
        raise ValueError("star_gamma_df must contain a 'gamma' column")
    if "is_ecc" not in star_gamma_df.columns:
        raise ValueError("star_gamma_df must contain an 'is_ecc' column")
    if "field_id" not in star_gamma_df.columns:
        raise ValueError("star_gamma_df must contain a 'field_id' column")

    df = star_gamma_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["gamma"]).copy()

    def _export(fig, base):
        export_set = set(export)
        if "html" in export_set:
            fig.write_html(base + ".html", include_plotlyjs="cdn", include_mathjax="cdn")
            print(f"Saved HTML: {base}.html")
        for fmt in ("png", "pdf", "svg"):
            if fmt in export_set:
                fig.write_image(base + f".{fmt}", scale=image_scale)
                print(f"Saved {fmt.upper()}: {base}.{fmt}")

    def _style(fig, title_text):
        fig.update_layout(
            template="simple_white",
            font=dict(family="Times New Roman", size=20),
            title=dict(text=title_text, x=0.5, xanchor="center"),
            width=950,
            height=650,
            margin=dict(l=80, r=40, t=80, b=80),
            barmode="stack",
            legend=dict(
                x=0.02, y=0.98,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
        )
        fig.update_xaxes(
            title="γ (km s⁻¹)",
            showline=True, linewidth=1.6, mirror=True,
            ticks="outside", ticklen=6,
            showgrid=True, gridwidth=1,
            zeroline=False,
        )
        fig.update_yaxes(
            title="N of Systems",
            showline=True, linewidth=1.6, mirror=True,
            ticks="outside", ticklen=6,
            showgrid=True, gridwidth=1,
            zeroline=False,
        )

    # 1) stacked by is_ecc
    fig1 = go.Figure()
    df_null = df[df["is_ecc"] == False]
    df_ecc  = df[df["is_ecc"] == True]

    fig1.add_trace(go.Histogram(
        x=df_ecc["gamma"],
        nbinsx=nbins,
        name=f"SB1 (N={len(df_ecc)})",
        marker=dict(color=color_ecc),
        opacity=0.95,
    ))
    fig1.add_trace(go.Histogram(
        x=df_null["gamma"],
        nbinsx=nbins,
        name=f"Observed Single (N={len(df_null)})",
        marker=dict(color=color_null),
        opacity=0.95,
    ))

    mean_g = float(df["gamma"].mean())
    std_g  = float(df["gamma"].std())
    _style(fig1, f"γ mode distribution — stacked by class (mean={mean_g:.3g}±{std_g:.3g} km/s)")
    _export(fig1, outpath_base + "_by_is_ecc")

    # 2) stacked by field_id
    fig2 = go.Figure()
    fields = sorted(df["field_id"].dropna().unique().tolist())
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    for i, fid in enumerate(fields):
        dff = df[df["field_id"] == fid]
        if len(dff) == 0:
            continue
        mean_f = float(np.mean(dff["gamma"]))
        std_f  = float(np.std(dff["gamma"]))
        name = f"Field {int(fid)} (mean={mean_f:.3g}±{std_f:.3g}, N={len(dff)})"

        fig2.add_trace(go.Histogram(
            x=dff["gamma"],
            nbinsx=nbins,
            name=name,
            marker=dict(color=palette[i % len(palette)]),
            opacity=0.95,
        ))

    _style(fig2, "γ mode distribution — stacked by field_id")
    _export(fig2, outpath_base + "_by_field")

    return fig1, fig2


# =========================================================
# 5) logP vs e scatter (your two versions unchanged)
# =========================================================
def plot_logP_e_scatter_with_circ_as_e0(
    df,
    outpath_base="logP_vs_e_K1_circ_as_e0",
    ecc_p_col="ecc_P_median",
    ecc_p_errm_col="ecc_P_errm",
    ecc_p_errp_col="ecc_P_errp",
    ecc_e_col="ecc_e_median",
    ecc_e_errm_col="ecc_e_errm",
    ecc_e_errp_col="ecc_e_errp",
    ecc_k1_col="ecc_K1_median",
    circ_p_col="circ_P_median",
    circ_p_errm_col="circ_P_errm",
    circ_p_errp_col="circ_P_errp",
    circ_k1_col="circ_K1_median",
    point_size=8,
    png_scale=3,
):
    # Identify rows that have any circular-solution columns
    circ_cols_any = [c for c in df.columns if c.startswith("circ_")]
    if len(circ_cols_any) > 0:
        has_circ_any = df[circ_cols_any].notna().any(axis=1)
    else:
        has_circ_any = np.zeros(len(df), dtype=bool)

    # --- Eccentric solutions: keep only rows with NO circular solution ---
    ecc_cols = [
        ecc_p_col, ecc_p_errm_col, ecc_p_errp_col,
        ecc_e_col, ecc_e_errm_col, ecc_e_errp_col,
        ecc_k1_col,
    ]
    df_ecc = df[ecc_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    df_ecc["has_circ"] = has_circ_any.loc[df_ecc.index].to_numpy()
    df_ecc = df_ecc[~df_ecc["has_circ"]].copy()

    df_ecc = df_ecc[
        (df_ecc[ecc_p_col] > 0) &
        (df_ecc[ecc_p_col] - df_ecc[ecc_p_errm_col] > 0) &
        (df_ecc[ecc_e_col] >= 0) & (df_ecc[ecc_e_col] <= 1)
    ].copy()

    # x = actual period (days), y = e
    P_ecc = df_ecc[ecc_p_col].to_numpy()
    P_ecc_errm = df_ecc[ecc_p_errm_col].to_numpy()
    P_ecc_errp = df_ecc[ecc_p_errp_col].to_numpy()

    e_ecc = df_ecc[ecc_e_col].to_numpy()
    e_ecc_errm = df_ecc[ecc_e_errm_col].to_numpy()
    e_ecc_errp = df_ecc[ecc_e_errp_col].to_numpy()
    K1_ecc = df_ecc[ecc_k1_col].to_numpy()

    # --- Circular solutions: keep only rows that DO have circular solution ---
    circ_cols = [circ_p_col, circ_p_errm_col, circ_p_errp_col, circ_k1_col]
    df_circ = df[circ_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    df_circ["has_circ"] = has_circ_any.loc[df_circ.index].to_numpy()
    df_circ = df_circ[df_circ["has_circ"]].copy()

    df_circ = df_circ[
        (df_circ[circ_p_col] > 0) &
        (df_circ[circ_p_col] - df_circ[circ_p_errm_col] > 0)
    ].copy()

    P_circ = df_circ[circ_p_col].to_numpy()
    P_circ_errm = df_circ[circ_p_errm_col].to_numpy()
    P_circ_errp = df_circ[circ_p_errp_col].to_numpy()

    # Circular: e = 0 with zero error bars
    e_circ = np.zeros_like(P_circ)
    e_circ_errm = np.zeros_like(P_circ)
    e_circ_errp = np.zeros_like(P_circ)
    K1_circ = df_circ[circ_k1_col].to_numpy()

    def _err(arr_p, arr_m):
        """
        Helper to construct asymmetric errorbar dicts for Plotly in *linear* space.
        arr_p, arr_m are the positive/negative errors in the same units as x or y.
        """
        return dict(
            type="data",
            symmetric=False,
            array=arr_p,
            arrayminus=arr_m,
            thickness=1.2,
            width=3,
        )

    fig = go.Figure()

    # --- Eccentric points ---
    fig.add_trace(go.Scatter(
        x=P_ecc,
        y=e_ecc,
        mode="markers",
        name="Significant e (Lucy-Sweeney)",
        marker=dict(
            symbol="circle", size=point_size,
            color=K1_ecc, colorscale="Viridis",
            colorbar=dict(title=dict(text="K₁ (km s⁻¹)")),
            line=dict(width=0.6, color="black"),
        ),
        error_x=_err(P_ecc_errp, P_ecc_errm),
        error_y=_err(e_ecc_errp, e_ecc_errm),
        hovertemplate=(
            "P: %{x:.3f} d<br>"
            "e: %{y:.3f}<br>"
            "K₁: %{marker.color:.2f} km/s<br>"
            "<extra></extra>"
        ),
    ))

    # --- Circular points (e = 0) ---
    fig.add_trace(go.Scatter(
        x=P_circ,
        y=e_circ,
        mode="markers",
        name="Insignificant e (Lucy-Sweeney)",
        marker=dict(
            symbol="triangle-down", size=point_size + 2,
            color=K1_circ, colorscale="Viridis",
            showscale=False,
            line=dict(width=0.8, color="black"),
        ),
        error_x=_err(P_circ_errp, P_circ_errm),
        error_y=_err(e_circ_errp, e_circ_errm),
        hovertemplate=(
            "P: %{x:.3f} d<br>"
            "e: 0 (circular)<br>"
            "K₁: %{marker.color:.2f} km/s<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        template="simple_white",
        width=950, height=700,
        font=dict(family="Times New Roman", size=20),
        title=dict(
            text="P vs e — eccentric-only vs circular-at-e=0 (colored by K₁)",
            x=0.5, xanchor="center",
        ),
        margin=dict(l=80, r=40, t=80, b=80),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        title="P (days)",
        type="log",  # <-- log10 scale on x, but x values are P
        showline=True, linewidth=1.6, mirror=True,
        ticks="outside", ticklen=6,
        showgrid=True, gridwidth=1,
        zeroline=False,
    )
    fig.update_yaxes(
        title="e",
        range=[-0.05, 1.02],
        showline=True, linewidth=1.6, mirror=True,
        ticks="outside", ticklen=6,
        showgrid=True, gridwidth=1,
        zeroline=False,
    )

    html_path = outpath_base if outpath_base.endswith(".html") else outpath_base + ".html"
    png_path = os.path.splitext(html_path)[0] + ".png"

    fig.write_html(html_path, include_plotlyjs="cdn", include_mathjax="cdn")
    print(f"Saved HTML: {html_path}")

    fig.write_image(png_path, scale=png_scale)
    print(f"Saved PNG: {png_path}")

    return fig



def plot_logP_e_scatter_with_errors(
    df,
    outpath_base="logP_vs_e_K1",
    p_col="ecc_P_median",
    p_errm_col="ecc_P_errm",
    p_errp_col="ecc_P_errp",
    e_col="ecc_e_median",
    e_errm_col="ecc_e_errm",
    e_errp_col="ecc_e_errp",
    k1_col="ecc_K1_median",
    point_size=8,
    png_scale=3,
):
    # Identify if each row has any circular-solution columns
    circ_cols = [c for c in df.columns if c.startswith("circ_")]
    if len(circ_cols) > 0:
        has_circ = df[circ_cols].notna().any(axis=1)
    else:
        has_circ = np.zeros(len(df), dtype=bool)

    cols = [p_col, p_errm_col, p_errp_col, e_col, e_errm_col, e_errp_col, k1_col]
    dfp = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    dfp["has_circ"] = has_circ.loc[dfp.index].to_numpy()

    dfp = dfp[
        (dfp[p_col] > 0) &
        (dfp[p_col] - dfp[p_errm_col] > 0) &
        (dfp[e_col] >= 0) & (dfp[e_col] <= 1)
    ].copy()

    if len(dfp) == 0:
        raise ValueError(
            "No valid rows after cleaning/guards (check NaNs, P<=0, or errm>=P)."
        )

    # x = actual period (days), with asymmetric errors in linear space
    P = dfp[p_col].to_numpy()
    P_errm = dfp[p_errm_col].to_numpy()
    P_errp = dfp[p_errp_col].to_numpy()

    e = dfp[e_col].to_numpy()
    e_errm = dfp[e_errm_col].to_numpy()
    e_errp = dfp[e_errp_col].to_numpy()
    K1 = dfp[k1_col].to_numpy()
    mask = dfp["has_circ"].to_numpy()

    def _err(arr_p, arr_m):
        return dict(
            type="data",
            symmetric=False,
            array=arr_p,
            arrayminus=arr_m,
            thickness=1.2,
            width=3,
        )

    fig = go.Figure()

    # --- Eccentric (no circular solution) ---
    fig.add_trace(go.Scatter(
        x=P[~mask],
        y=e[~mask],
        mode="markers",
        name="Significant e (Lucy-Sweeney)",
        marker=dict(
            symbol="circle", size=point_size,
            color=K1[~mask], colorscale="Viridis",
            colorbar=dict(title=dict(text="K₁ (km s⁻¹)")),
            line=dict(width=0.6, color="black"),
        ),
        error_x=_err(P_errp[~mask], P_errm[~mask]),
        error_y=_err(e_errp[~mask], e_errm[~mask]),
        hovertemplate=(
            "P: %{x:.3f} d<br>"
            "e: %{y:.3f}<br>"
            "K₁: %{marker.color:.2f} km/s<br>"
            "<extra></extra>"
        ),
    ))

    # --- Rows that *also* have circular solutions ---
    fig.add_trace(go.Scatter(
        x=P[mask],
        y=e[mask],
        mode="markers",
        name="Insignificant e (Lucy-Sweeney)",
        marker=dict(
            symbol="triangle-down", size=point_size + 2,
            color=K1[mask], colorscale="Viridis",
            showscale=False,
            line=dict(width=0.8, color="black"),
        ),
        error_x=_err(P_errp[mask], P_errm[mask]),
        error_y=dict(
            type="data",
            symmetric=False,
            array=np.zeros_like(e_errm[mask]),
            arrayminus=e_errm[mask],
            thickness=1.2,
            width=3,
        ),
        hovertemplate=(
            "P: %{x:.3f} d<br>"
            "e: %{y:.3f}<br>"
            "K₁: %{marker.color:.2f} km/s<br>"
            "circ: yes<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        template="simple_white",
        width=950, height=700,
        font=dict(family="Times New Roman", size=20),
        title=dict(
            text="P vs e — eccentric MCMC solutions (colored by K₁)",
            x=0.5, xanchor="center",
        ),
        margin=dict(l=80, r=40, t=80, b=80),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        title="P (days)",
        type="log",  # <-- log10 scale, actual P on axis
        showline=True, linewidth=1.6, mirror=True,
        ticks="outside", ticklen=6,
        showgrid=True, gridwidth=1,
        zeroline=False,
    )
    fig.update_yaxes(
        title="e",
        range=[-0.02, 1.02],
        showline=True, linewidth=1.6, mirror=True,
        ticks="outside", ticklen=6,
        showgrid=True, gridwidth=1,
        zeroline=False,
    )

    html_path = outpath_base if outpath_base.endswith(".html") else outpath_base + ".html"
    png_path = os.path.splitext(html_path)[0] + ".png"

    fig.write_html(html_path, include_plotlyjs="cdn", include_mathjax="cdn")
    print(f"Saved HTML: {html_path}")

    fig.write_image(png_path, scale=png_scale)
    print(f"Saved PNG: {png_path}")

    return fig



# =========================================================
# 6) Subplots helper (fixed exclusions + log guard)
# =========================================================
def plot_eccentric_mode_histograms_subplots(
    df,
    outpath="ecc_mode_histograms.html",
    nbins=50,
    log_params=("P",),
    save_png=False,
    png_scale=2,
    ncols=3
):
    """
    Plot one histogram per ecc_{param}_mode using Plotly subplots.

    Fixes:
      - Removed trailing space in ecc_log_sj filter
      - Robustly excludes only ecc_log_sj_mode
      - Guards log10 transform for > 0 only
    """

    ecc_mode_cols = sorted([
        c for c in df.columns
        if c.startswith("ecc_") and c.endswith("_mode") and c != "ecc_log_sj_mode"
    ])

    if len(ecc_mode_cols) == 0:
        raise ValueError("No ecc_*_mode columns found (after excluding ecc_log_sj_mode).")

    nparams = len(ecc_mode_cols)
    nrows = math.ceil(nparams / ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[c.replace("ecc_", "").replace("_mode", "") for c in ecc_mode_cols],
        horizontal_spacing=0.07,
        vertical_spacing=0.10,
    )

    for i, col in enumerate(ecc_mode_cols):
        param = col.replace("ecc_", "").replace("_mode", "")
        values = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(values) == 0:
            continue

        if param in log_params:
            values = values[values > 0]
            if len(values) == 0:
                continue
            values = np.log10(values)
            x_title = _label_with_units(param, is_log=True)
        else:
            x_title = _label_with_units(param, is_log=False)

        row = i // ncols + 1
        col_idx = i % ncols + 1

        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=nbins,
            showlegend=False,
            marker=dict(line=dict(width=0)),
        ), row=row, col=col_idx)

        fig.update_xaxes(title_text=x_title, row=row, col=col_idx)

    fig.update_layout(
        title="MCMC eccentric solution — parameter mode distributions",
        template="plotly_white",
        width=1100,
        height=350 * nrows,
    )

    fig.write_html(outpath)
    print(f"Saved interactive histogram to: {outpath}")

    if save_png:
        png_path = os.path.splitext(outpath)[0] + ".png"
        fig.write_image(png_path, scale=png_scale)
        print(f"Saved PNG to: {png_path}")

    return fig


# =========================================================
# Example main (your original, unchanged paths)
# =========================================================
if __name__ == "__main__":
    mcmc_sol_path = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/neb_div_from_coadded/new_for_seminar/mcmc_min_withNull/mcmc_params.csv"
    directory = os.path.dirname(mcmc_sol_path)

    df = pd.read_csv(mcmc_sol_path)
    # Combined + individual histograms
    df["ecc_omega_mode"] = np.mod(df["ecc_omega_mode"], np.pi)

    grid_fig, indiv_figs = plot_eccentric_mode_histograms_physics_save_all(
        df,
        outdir=directory + "/figures_noMean1",
        outname_prefix="ecc_mode",
        nbins=20,
        ncols=3,
        log_params=("P", "K1"),
        add_quantile_lines=True,
        add_median_line=True,
        add_mean_to_titles=True,
        export=("html", "png"),
        image_scale=3
    )

    fig = plot_logP_e_scatter_with_errors(
        df,
        outpath_base=directory + "/logP_vs_e_K12",
        png_scale=3
    )

    fig = plot_logP_e_scatter_with_circ_as_e0(
        df,
        outpath_base=directory + "/logP_vs_e_K1_02",
        png_scale=3
    )

    star_gamma = build_star_gamma_df(df, null_gamma_col="null_gamma_mode", ecc_gamma_col="ecc_gamma_mode")

    # fig = plot_gamma_histogram_physics(
    #     star_gamma,
    #     outpath_base=directory + "/gamma_hist",
    #     nbins=30,
    #     export=("html", "png"),
    #     image_scale=3,
    # )
    #
    # fig_is_ecc, fig_field = plot_gamma_histograms_stacked(
    #     star_gamma,
    #     outpath_base=directory + "/gamma_hist_stacked",
    #     nbins=30,
    #     export=("html", "png"),
    #     image_scale=3
    # )
