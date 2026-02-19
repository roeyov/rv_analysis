import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _to_native_f64(x):
    """Return a native-endian float64 NumPy array."""
    arr = np.asarray(x)
    if arr.dtype.kind == 'f':
        # If big-endian (">" or "!"), or non-native, swap
        if arr.dtype.byteorder in ('>', '!') or (arr.dtype.byteorder == '=' and not np.little_endian):
            arr = arr.byteswap().newbyteorder()
    return arr.astype(np.float64, copy=False)

def _build_mjd_to_spectrum_map(a, mjd_col, wl_col, flux_col):
    """
    Accepts either:
      - a single DataFrame with multiple MJDs (group by mjd_col), or
      - a list of DataFrames each containing a single MJD.
    Returns: dict[MJD] -> DataFrame[[wl_col, flux_col]]
    """
    if isinstance(a, list):
        mjd_map = {}
        for df in a:
            if mjd_col not in df.columns:
                raise ValueError(f"Expected column {mjd_col} in spectra list item")
            mjd_val = np.unique(df[mjd_col].values)
            if len(mjd_val) != 1:
                raise ValueError("Each spectra DataFrame should have exactly one unique MJD")
            mjd_map[float(mjd_val[0])] = df[[wl_col, flux_col]].copy()
        return mjd_map

    # assume single concatenated DataFrame
    if mjd_col not in a.columns:
        raise ValueError(f"Expected column {mjd_col} in spectra DataFrame")
    mjd_map = {}
    for mjd_val, g in a.groupby(mjd_col):
        mjd_map[float(mjd_val)] = g[[wl_col, flux_col]].copy()
    return mjd_map


def _nearest_mjd(target, mjds, atol=1e-6):
    """Return the MJD from 'mjds' closest to 'target' (within atol if provided)."""
    mjds_arr = np.asarray(sorted(mjds), dtype=float)
    idx = int(np.argmin(np.abs(mjds_arr - target)))
    if atol is not None and np.abs(mjds_arr[idx] - target) > atol:
        # still return the nearest; caller may decide to warn/skip
        pass
    return float(mjds_arr[idx])

def spec_to_df(d):
    arrays_only = {k: v for k, v in d.items() if hasattr(v, "__len__") and not isinstance(v, (str, bytes))}
    # make into DataFrame
    df = pd.DataFrame(arrays_only)
    return df

import os
import plotly.graph_objects as go

import os
import plotly.graph_objects as go

def save_html_fig_plotly(
    wl_min, fl_min, wl_max, fl_max, mjd_min, mjd_max, star, out_dir,
    title_prefix="all Spectra | min vs max RV",
    lines_to_windows=None,
    mjd_map_for_all=None,         # NEW: dict[MJD] -> DataFrame with wavelength/flux
    wl_col="WAVELENGTH", flux_col="SCI_NORM"
):
    """
    Save an interactive HTML overlay with TWO modes:
      1) All spectra (each epoch as a trace, label = rounded MJD)
      2) Min/Max only (the two selected spectra)
    Switch modes with top buttons.
    """
    os.makedirs(out_dir, exist_ok=True)

    fig = go.Figure()
    trace_sections = []  # list of (name, [indices])

    # ---------- Section A: All spectra ----------
    all_indices = []
    if mjd_map_for_all:
        for mjd in sorted(mjd_map_for_all.keys()):
            sdf = spec_to_df(mjd_map_for_all[mjd])
            if wl_col not in sdf.columns or flux_col not in sdf.columns:
                continue
            wl = _to_native_f64(sdf[wl_col].values)
            fl = _to_native_f64(sdf[flux_col].values)
            # label by rounded MJD
            label = f"MJD {float(mjd):.5f}"
            fig.add_trace(go.Scatter(
                x=wl, y=fl, mode="lines",
                name=label,
                line=dict(width=1),
                opacity=0.65,
                hovertemplate="λ=%{x:.5f}<br>Flux=%{y:.4f}<extra>" + label + "</extra>",
                visible=False,  # default OFF; we start on Min/Max
            ))
            all_indices.append(len(fig.data) - 1)
    trace_sections.append(("All spectra", all_indices))

    # ---------- Section B: Min/Max only ----------
    minmax_indices = []
    fig.add_trace(go.Scatter(
        x=_to_native_f64(wl_min), y=_to_native_f64(fl_min),
        mode="lines",
        name=f"min RV @ {mjd_min:.5f}",
        line=dict(width=2),
        hovertemplate="λ=%{x:.5f}<br>Flux=%{y:.4f}<extra>min RV</extra>",
        visible=True
    ))
    minmax_indices.append(len(fig.data) - 1)

    fig.add_trace(go.Scatter(
        x=_to_native_f64(wl_max), y=_to_native_f64(fl_max),
        mode="lines",
        name=f"max RV @ {mjd_max:.5f}",
        line=dict(width=2),
        hovertemplate="λ=%{x:.5f}<br>Flux=%{y:.4f}<extra>max RV</extra>",
        visible=True
    ))
    minmax_indices.append(len(fig.data) - 1)

    trace_sections.append(("Min/Max only", minmax_indices))

    # ---------- Layout (legend below) ----------
    fig.update_layout(
        title=f"{star} | {title_prefix}",
        xaxis_title="Wavelength",
        yaxis_title="Normalized Flux",
        legend=dict(
            orientation="h",
            y=-0.2, x=0.5, xanchor="center", yanchor="top"
        ),
        margin=dict(l=60, r=20, t=60, b=100),
    )

    # ---------- Window-edge vertical lines + centered labels ----------
    if lines_to_windows:
        for line_name, window in lines_to_windows.items():
            try:
                lam_lo, lam_hi = float(window[0]), float(window[1])
            except Exception:
                continue
            lam_center = 0.5 * (lam_lo + lam_hi)
            fig.add_vline(x=lam_lo, line_width=1, line_dash="dot", opacity=0.5)
            fig.add_vline(x=lam_hi, line_width=1, line_dash="dot", opacity=0.5)
            fig.add_annotation(
                x=lam_center,
                y=1.02, yref="paper",
                showarrow=False,
                text=line_name,
                font=dict(size=10, color="black")
            )

    # ---------- Buttons to toggle sections ----------
    def vis_mask(active_name):
        mask = [False] * len(fig.data)
        for sect_name, idxs in trace_sections:
            if sect_name == active_name:
                for i in idxs:
                    mask[i] = True
        return mask

    buttons = []
    for sect_name, _ in trace_sections:
        buttons.append(dict(
            label=sect_name,
            method="update",
            args=[{"visible": vis_mask(sect_name)},
                  {"title": f"{star} | {title_prefix} — {sect_name}"}]
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5, xanchor="center",
            y=1.15, yanchor="top",
            buttons=buttons,
            showactive=True,
            pad={"r": 10, "t": 5}
        )]
    )

    # ---------- Save ----------
    html_name = f"{star}_all_spectra_minmax_overlay.html"
    html_path = os.path.join(out_dir, html_name)
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    return html_path

def save_html_fig(wl_min,fl_min,wl_max,fl_max,mjd_min,mjd_max, star, out_dir):
    import mpld3
    from mpld3 import plugins

    fig, ax = plt.subplots()
    ax.plot(wl_min, fl_min, label=f"min RV @ {mjd_min:.5f}")
    ax.plot(wl_max, fl_max, label=f"max RV @ {mjd_max:.5f}")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"{star} | all Spectra | min vs max RV")
    ax.legend()
    fig.tight_layout()

    # connect interactive plugins (toolbar buttons)
    plugins.connect(fig, plugins.Reset(), plugins.Zoom(), plugins.BoxZoom())

    # save as standalone HTML
    html_name = f"{star}_all_spectra_minmax_overlay.html"
    html_path = os.path.join(out_dir, html_name)
    mpld3.save_html(fig, html_path)

    plt.close(fig)

def save_minmax_overlay_plots(a, mean_calc, lines_to_windows, star, out_root,
                              mjd_col="MJD_mid", wl_col="WAVELENGTH", flux_col="SCI_NORM"):
    """
    For each spectral window in lines_to_windows, save a plot overlaying
    the spectra at min(Mean RV) and max(Mean RV). All plots are saved into:
    {out_root}/extrema_overlays/{star}/
    """
    if out_root is None or out_root == "":
        return

    # columns expected in mean_calc: 'Mean RV' and a time column compatible with 'a'
    if "Mean RV" not in mean_calc.columns:
        raise ValueError("mean_calc must have a 'Mean RV' column")

    # pick a time column to match spectra 'a'
    time_col_candidates = [mjd_col, "MJD", "HJD", "HJD_mid", "MJD_MID", "MJD_mid"]
    time_col = next((c for c in time_col_candidates if c in mean_calc.columns), None)
    if time_col is None:
        raise ValueError(f"mean_calc must include one of {time_col_candidates}")

    # find min/max RV epochs
    valid = mean_calc[["Mean RV", time_col]].dropna()
    if valid.empty:
        return

    idx_min = valid["Mean RV"].idxmin()
    idx_max = valid["Mean RV"].idxmax()
    mjd_min = float(valid.loc[idx_min, time_col])
    mjd_max = float(valid.loc[idx_max, time_col])

    # map spectra by MJD
    mjd_map = a
    if not mjd_map:
        return

    # robust match to nearest available MJD in spectra
    mjd_min_mapped = _nearest_mjd(mjd_min, mjd_map.keys())
    mjd_max_mapped = _nearest_mjd(mjd_max, mjd_map.keys())
    spec_min = spec_to_df(mjd_map.get(mjd_min_mapped))
    spec_max = spec_to_df(mjd_map.get(mjd_max_mapped))
    if spec_min is None or spec_max is None:
        return

    # output directory (common folder + per-star subfolder)
    out_dir = os.path.join(out_root, star)
    os.makedirs(out_dir, exist_ok=True)
    # Convert to native-endian float64 arrays
    wl_min = _to_native_f64(spec_min[wl_col].values)
    fl_min = _to_native_f64(spec_min[flux_col].values)
    wl_max = _to_native_f64(spec_max[wl_col].values)
    fl_max = _to_native_f64(spec_max[flux_col].values)

    save_html_fig_plotly(wl_min, fl_min,
                         wl_max, fl_max,
                         mjd_min,mjd_max,
                         star, out_dir,
                         lines_to_windows=lines_to_windows,
                         mjd_map_for_all=a)

