"""
spectroscopy.plotting — RV vs MJD plots (Matplotlib + Plotly).

All functions accept *marker_dict* as an explicit parameter
instead of relying on module-level global state.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ---------- helpers ----------

def _select_rv_columns(df, plot_only=None):
    if plot_only is None:
        return [
            col for col in df.columns
            if ("RV" in col) and ("RVsig" not in col)
            and ("flag" not in col) and (col != "MJD")
        ]
    return [key + " RV" for key in plot_only]


def _apply_filter(df, rv_col, do_filter):
    flag_col = rv_col.replace("RV", "flag")
    if do_filter and flag_col in df.columns:
        return df[df[flag_col] == 0]
    return df


_MPL_TO_PLOTLY = {
    'o': 'circle', 's': 'square', '^': 'triangle-up',
    'v': 'triangle-down', '<': 'triangle-left', '>': 'triangle-right',
    'D': 'diamond', '*': 'star', 'x': 'x', '+': 'cross',
    '.': 'circle-open',
}


# ---------- combined interactive HTML ----------

def plot_rv_vs_mjd_combined(cc_result_df, mean_calc_df, name, out,
                            marker_dict=None):
    """
    One interactive HTML with buttons to switch between All / Final / Significant RVs.
    """
    if marker_dict is None:
        marker_dict = {}

    os.makedirs(os.path.join(out, name), exist_ok=True)

    fig = go.Figure()
    trace_sections = []

    # ---- Section 1: All_RVS ----
    all_indices = []
    rv_cols_all = _select_rv_columns(cc_result_df, plot_only=None)
    for rv_col in rv_cols_all:
        marker, color = marker_dict.get(rv_col, ('o', '#1f77b4'))
        symbol = _MPL_TO_PLOTLY.get(marker, 'circle')
        dfp = _apply_filter(cc_result_df, rv_col, False)
        x = dfp["MJD"].to_numpy()
        y = dfp[rv_col].to_numpy()
        unc = rv_col.replace("RV", "RVsig")
        kw = dict(x=x, y=y, mode="markers", name=rv_col,
                  marker=dict(symbol=symbol, color=color, opacity=0.9, size=8),
                  hovertemplate=f"MJD=%{{x}}<br>RV=%{{y}} km/s<extra>{rv_col}</extra>",
                  visible=True)
        if unc in dfp.columns:
            kw["error_y"] = dict(type="data", array=dfp[unc].to_numpy(), visible=True)
        fig.add_trace(go.Scatter(**kw))
        all_indices.append(len(fig.data) - 1)
    trace_sections.append(("All_RVS", all_indices))

    # ---- Section 2: Final_RVS ----
    final_indices = []
    rv_cols_final = _select_rv_columns(mean_calc_df, plot_only=["Mean"])
    for rv_col in rv_cols_final:
        marker, color = marker_dict.get(rv_col, ('o', '#1f77b4'))
        symbol = _MPL_TO_PLOTLY.get(marker, 'circle')
        dfp = _apply_filter(mean_calc_df, rv_col, True)
        x = dfp["MJD"].to_numpy()
        y = dfp[rv_col].to_numpy()
        unc = rv_col.replace("RV", "RVsig")
        kw = dict(x=x, y=y, mode="markers", name=f"{rv_col} (final)",
                  marker=dict(symbol=symbol, color=color, opacity=0.95, size=10),
                  hovertemplate=f"MJD=%{{x}}<br>RV=%{{y}} km/s<extra>{rv_col}</extra>",
                  visible=False)
        if unc in dfp.columns:
            kw["error_y"] = dict(type="data", array=dfp[unc].to_numpy(), visible=True)
        fig.add_trace(go.Scatter(**kw))

        nan_mask = mean_calc_df["Mean RV"].isna()
        nan_mjds = mean_calc_df.loc[nan_mask, "MJD"]
        for mjd in nan_mjds:
            fig.add_vline(x=mjd,
                          line=dict(color="red", dash="dash", width=1),
                          annotation_text=f"MJD={mjd:.2f}",
                          annotation_position="top",
                          annotation=dict(font=dict(color="red")))

        n_total = len(mean_calc_df)
        n_valid = mean_calc_df["Mean RV"].notna().sum()
        fig.update_layout(
            annotations=[dict(
                text=f"{n_valid}/{n_total} epochs with valid Mean RV",
                x=0.5, xref="paper", y=-0.10, yref="paper",
                showarrow=False, font=dict(size=12, color="black"),
            )]
        )
        final_indices.append(len(fig.data) - 1)
    trace_sections.append(("Final_RVS", final_indices))

    # ---- Section 3: Significant_RVS ----
    sig_indices = []
    rv_cols_sig = _select_rv_columns(mean_calc_df, plot_only=None)
    for rv_col in rv_cols_sig:
        marker, color = marker_dict.get(rv_col, ('o', '#1f77b4'))
        symbol = _MPL_TO_PLOTLY.get(marker, 'circle')
        dfp = _apply_filter(mean_calc_df, rv_col, True)
        if dfp.empty:
            continue
        x = dfp["MJD"].to_numpy()
        y = dfp[rv_col].to_numpy()
        unc = rv_col.replace("RV", "RVsig")
        kw = dict(x=x, y=y, mode="markers", name=f"{rv_col} (signif.)",
                  marker=dict(symbol=symbol, color=color, opacity=0.9, size=8),
                  hovertemplate=f"MJD=%{{x}}<br>RV=%{{y}} km/s<extra>{rv_col}</extra>",
                  visible=False)
        if unc in dfp.columns:
            kw["error_y"] = dict(type="data", array=dfp[unc].to_numpy(), visible=True)
        fig.add_trace(go.Scatter(**kw))
        sig_indices.append(len(fig.data) - 1)
    trace_sections.append(("Significant_RVS", sig_indices))

    # ---- Buttons ----
    def vis_mask(active_name):
        mask = [False] * len(fig.data)
        for sect_name, idxs in trace_sections:
            if sect_name == active_name:
                for i in idxs:
                    mask[i] = True
        return mask

    buttons = [
        dict(label=sect_name, method="update",
             args=[{"visible": vis_mask(sect_name)},
                   {"title": f"RV vs MJD {name} | {sect_name}"}])
        for sect_name, _ in trace_sections
    ]

    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="right",
                          x=0.5, xanchor="center", y=1.15, yanchor="top",
                          buttons=buttons, showactive=True,
                          pad={"r": 10, "t": 5})],
        title=f"RV vs MJD {name} | All_RVS",
        xaxis_title="MJD [d]", yaxis_title="RV [km/s]",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center",
                    yanchor="top"),
        margin=dict(l=60, r=20, t=100, b=100),
    )

    safe_name = "".join(
        c if c.isalnum() or c in "-_ " else "_" for c in str(name)
    ).strip().replace(" ", "_")
    html_path = os.path.join(out, name, f"{safe_name}_rv_vs_mjd_combined.html")
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    return html_path


# ---------- single-section convenience ----------

def plot_rv_vs_mjd(df, name, plot_only=None, filter=False, out=None,
                   out_suffix='', marker_dict=None):
    """
    Matplotlib show + optional Plotly HTML save.
    """
    _plot_rv_vs_mjd_mpl(df, name, plot_only=plot_only, filter=filter,
                        marker_dict=marker_dict)
    if out is not None:
        os.makedirs(out, exist_ok=True)
        return _plot_rv_vs_mjd_plotly_save(
            df, name, plot_only=plot_only, filter=filter,
            out=out, out_suffix=out_suffix, marker_dict=marker_dict,
        )
    return None


def _plot_rv_vs_mjd_mpl(df, name, plot_only=None, filter=False,
                        marker_dict=None):
    if marker_dict is None:
        marker_dict = {}
    rv_columns = _select_rv_columns(df, plot_only)

    plt.figure(figsize=(8, 6))
    for rv_col in rv_columns:
        marker, color = marker_dict.get(rv_col, ('o', '#1f77b4'))
        uncertainty_col = rv_col.replace("RV", "RVsig")
        df_to_plot = _apply_filter(df, rv_col, filter)

        if uncertainty_col in df.columns:
            plt.errorbar(df_to_plot["MJD"], df_to_plot[rv_col],
                         yerr=df_to_plot[uncertainty_col],
                         fmt=marker, linestyle='', color=color,
                         label=rv_col, capsize=3, alpha=0.8)
        else:
            plt.plot(df_to_plot["MJD"], df_to_plot[rv_col],
                     marker=marker, linestyle='', color=color,
                     label=rv_col, alpha=0.8)

    plt.xlabel("MJD [d]")
    plt.ylabel("RV [km/s]")
    plt.title(f"RV vs MJD {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_rv_vs_mjd_plotly_save(df, name, plot_only=None, filter=False,
                                out='.', out_suffix='', marker_dict=None):
    if marker_dict is None:
        marker_dict = {}
    rv_columns = _select_rv_columns(df, plot_only)

    fig = go.Figure()
    for rv_col in rv_columns:
        marker, color = marker_dict.get(rv_col, ('o', '#1f77b4'))
        symbol = _MPL_TO_PLOTLY.get(marker, 'circle')
        uncertainty_col = rv_col.replace("RV", "RVsig")
        df_to_plot = _apply_filter(df, rv_col, filter)

        x = df_to_plot["MJD"].to_numpy()
        y = df_to_plot[rv_col].to_numpy()
        kw = dict(x=x, y=y, mode="markers", name=rv_col,
                  marker=dict(symbol=symbol, color=color, opacity=0.9, size=8),
                  hovertemplate=f"MJD=%{{x}}<br>RV=%{{y}} km/s<extra>{rv_col}</extra>")
        if uncertainty_col in df.columns:
            kw["error_y"] = dict(type="data",
                                 array=df_to_plot[uncertainty_col].to_numpy(),
                                 visible=True)
        fig.add_trace(go.Scatter(**kw))

    fig.update_layout(
        title=f"RV vs MJD {name}",
        xaxis_title="MJD [d]", yaxis_title="RV [km/s]",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center",
                    yanchor="top"),
        margin=dict(l=60, r=20, t=60, b=100),
    )

    safe_name = "".join(
        c if c.isalnum() or c in "-_ " else "_" for c in str(name)
    ).strip().replace(" ", "_")
    suffix = (f"_{out_suffix}" if out_suffix else "")
    out_dir = os.path.join(out, name)
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"{safe_name}_rv_vs_mjd{suffix}.html")
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    return html_path
