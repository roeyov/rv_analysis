"""
Orbital fit plotting: time-series, phase-folded, and residual plots.

This module provides matplotlib and plotly visualizations for:
  - RV time series with residuals
  - Phase-folded RV curves with residuals
  - Null (constant-RV) model plots
  - Fit report generation
"""

import sys
import os
import io
import contextlib

import matplotlib.pyplot as plt
import numpy as np
import lmfit
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils.constants import (
    GAMMA, K1_STR, OMEGA, ECC, PERIOD, T,
    LN_SIGMA_JITTER, PLOT_STYLE,
)
from orbital.kepler import Kepler, v1mod, true_anomaly_from_E
from orbital.fitting import extract_observations
from orbital.statistics import compute_orbital_params
from utils.plot_style import get_style, apply_style_to_layout


# ---------------------------------------------------------------------------
# RV curve computation helpers
# ---------------------------------------------------------------------------

def compute_rv_curve_time(hjds, P, T0, ecc, Gamma, K, Omega, step=0.1):
    """
    Returns time_grid and corresponding RV by calling v1mod on nu from Kepler.
    """
    tmin, tmax = hjds.min() - 0.5*P, hjds.max() + 0.5*P
    time_grid = np.arange(tmin, tmax, step)
    phases = ((time_grid - T0) / P) % 1
    M = 2*np.pi*phases
    E = Kepler(np.pi, M, ecc)
    nu = 2*np.arctan2(np.sqrt(1+ecc)*np.sin(E/2), np.sqrt(1-ecc)*np.cos(E/2))
    rv = v1mod(nu, Gamma, K, Omega, ecc)
    return time_grid, rv


def compute_rv_curve_phase(P, T0, ecc, Gamma, K, Omega, n_points=1000):
    """
    Returns phase_grid and RV by calling v1mod on nu from Kepler at each dense phase.
    """
    phase_grid = np.linspace(0, 1, n_points)
    M = 2*np.pi*phase_grid
    E = Kepler(np.pi, M, ecc)
    if np.isnan(E.all()):
        return phase_grid, np.full(phase_grid.shape, np.nan)
    nu = 2*np.arctan2(np.sqrt(1+ecc)*np.sin(E/2), np.sqrt(1-ecc)*np.cos(E/2))
    rv = v1mod(nu, Gamma, K, Omega, ecc)
    return phase_grid, rv


def compute_phase_residuals(hjds, vels, P, T0, ecc, Gamma, K, Omega):
    """Compute phase-folded data, model curve, and residuals (no plotting)."""
    phs_data = ((np.asarray(hjds, dtype=float) - float(T0)) / float(P)) % 1.0
    phase_grid, rv_phase = compute_rv_curve_phase(P, T0, ecc, Gamma, K, Omega)
    phase_grid = np.asarray(phase_grid, dtype=float)
    rv_phase = np.asarray(rv_phase, dtype=float)
    vmod_data = np.interp(phs_data, phase_grid, rv_phase)
    residuals = np.asarray(vels, dtype=float) - vmod_data
    return phs_data, phase_grid, rv_phase, residuals


def compute_null_rv_curve_time(hjds, Gamma, ngrid=500, pad_frac=0.05):
    """
    Build a simple time grid and RV curve for the null (constant-velocity) model.
    """
    hjds = np.asarray(hjds, dtype=float)
    if hjds.size == 0:
        return np.array([]), np.array([])

    tmin, tmax = hjds.min(), hjds.max()
    dt = tmax - tmin
    if dt <= 0:
        time_grid = hjds.copy()
    else:
        tmin -= pad_frac * dt
        tmax += pad_frac * dt
        time_grid = np.linspace(tmin, tmax, ngrid)

    rv_model = np.full_like(time_grid, float(Gamma))
    return time_grid, rv_model


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in str(name)).strip().replace(" ", "_")


def get_fit_report(result, star_name, solution_id=0, out_dir=None):
    report = lmfit.fit_report(result)
    if out_dir:
        path = os.path.join(out_dir, f"{star_name}_sid-{solution_id}_report.txt")
        with open(path, 'w') as f:
            f.write(report)
        print(f"[{star_name}] Fit report saved to {path}")


def dump_minimization_report(res, output, file_prefix):
    lines = []
    best_fit_params = res.params
    lines.append("Best-fit parameter values:\n")
    for name, param in best_fit_params.items():
        lines.append(f"{name}: {param.value}\n")

    minimized_value = res.chisqr
    lines.append(f"Minimized chi-square value: {minimized_value}\n")

    reduced_chi_square = res.redchi
    lines.append(f"Reduced chi-square value: {reduced_chi_square}\n")

    success = res.success
    lines.append(f"Optimization success: {success}\n")

    aic = res.aic
    bic = res.bic
    lines.append(f"AIC: {aic}, BIC: {bic}\n")

    covariance_matrix = res.covar
    lines.append(f"Covariance matrix: \n{covariance_matrix}\n")

    lines.append("Fit report:\n")
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        lmfit.report_fit(res)
    lines.append(report.getvalue())

    with open(os.path.join(output, "{}_fit_report.txt".format(file_prefix)), "w") as file:
        file.writelines(lines)


# ---------------------------------------------------------------------------
# Matplotlib plots
# ---------------------------------------------------------------------------

def plot_time_series_with_residuals(hjds, vels, errs, time_grid, rv_model,
                                    Gamma, K, Omega, ecc, P, star_name, solution_id=0,
                                    out_dir=None, figsize=(10, 10)):
    vmod_data = np.interp(hjds, time_grid, rv_model)
    residuals = vels - vmod_data

    if not out_dir:
        return residuals

    fig, (ax1, ax1r) = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax1.errorbar(hjds, vels, yerr=errs, fmt='o', color='black')
    ax1.plot(time_grid, rv_model, color='black', label='Model')
    ax1.set_ylabel(r'RV [{\\rm km}\\,{\rm s}^{-1}]')
    if (P is None) or (not np.isfinite(P)):
        title = f"Null (constant RV) model {star_name}"
    else:
        title = f"Orbital fit {star_name}: P={P:.2f}d, ecc={ecc:.3f}"
    ax1.set_title(title)
    ax1.legend(loc='upper left')

    ax1r.errorbar(hjds, residuals, yerr=errs, fmt='o', color='gray')
    ax1r.axhline(0, color='black', lw=0.8)
    ax1r.set_xlabel('MJD')
    ax1r.set_ylabel('O\u2013C')

    fig.tight_layout()
    if out_dir:
        fig.savefig(os.path.join(out_dir, f"{star_name}_sid-{solution_id}_time_residuals.png"))
    plt.show()
    return residuals


def plot_phase_folded_with_residuals(hjds, vels, errs, P, T0,
                                     Gamma, K, Omega, ecc,
                                     star_name, solution_id=0, out_dir=None,
                                     figsize=(10, 10)):
    phs_data, phase_grid, rv_phase, residuals = compute_phase_residuals(
        hjds, vels, P, T0, ecc, Gamma, K, Omega,
    )

    if not out_dir:
        return phs_data, phase_grid, rv_phase, residuals

    fig, (ax2, ax2r) = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax2.errorbar(phs_data, vels, yerr=errs, fmt='o', color='red')
    ax2.plot(phase_grid, rv_phase, color='red', label='Model')
    ax2.axhline(Gamma, color='black', linestyle='--')
    ax2.set_xlim(0, 1)
    ax2.set_ylabel('RV [km s$^{-1}$]')
    ax2.set_title(f"Folded fit {star_name}")
    ax2.legend()

    ax2r.errorbar(phs_data, residuals, yerr=errs, fmt='o', color='gray')
    ax2r.axhline(0, color='black', lw=0.8)
    ax2r.set_xlabel('Phase')
    ax2r.set_ylabel('O\u2013C')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{star_name}_sid-{solution_id}_phase_residuals.png"))
    plt.show()
    return phs_data, phase_grid, rv_phase, residuals


# ---------------------------------------------------------------------------
# Plotly plots
# ---------------------------------------------------------------------------

def plot_time_series_with_residuals_plotly(
    hjds, vels, errs, time_grid, rv_model,
    Gamma, K, Omega, ecc, P, star_name, solution_id=0, jitter=0.0,
    out_dir=None, figsize=(1000, 700),
    style="interactive",
):
    s = get_style(style)
    hjds = np.asarray(hjds, dtype=float)
    vels = np.asarray(vels, dtype=float)
    errs = np.asarray(errs, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    rv_model = np.asarray(rv_model, dtype=float)

    vmod_data = np.interp(hjds, time_grid, rv_model)
    residuals = vels - vmod_data

    if not out_dir:
        return residuals

    # ── title ─────────────────────────────────────────────────────────
    title = ""
    if s.show_title:
        if (P is None) or (not np.isfinite(P)):
            title = f"Null (constant RV) model for {star_name}"
        else:
            title = (f"Orbital fit for {star_name}: "
                     f"P = {P:.2f} d, e = {ecc:.3f}")

    # ── labels (concise in paper mode) ────────────────────────────────
    if s.show_title:
        lbl_data = "Measurements (\u00b11\u03c3, statistical errors)"
        lbl_jit = "Measurements (\u00b11\u03c3, including jitter)"
        lbl_model = "Best-fit model"
        lbl_res = "Residuals (\u00b11\u03c3, statistical errors)"
        lbl_res_jit = "Residuals (\u00b11\u03c3, including jitter)"
        x_label = "Time (Modified Julian Date)"
        y_label_rv = "Radial velocity [km s\u207b\u00b9]"
        y_label_res = "Residuals [km s\u207b\u00b9]"
        sub_titles = ("Radial velocity versus time", "Residuals")
    else:
        lbl_data = "Data"
        lbl_jit = "Data (incl. jitter)"
        lbl_model = "Best fit"
        lbl_res = "O\u2013C"
        lbl_res_jit = "O\u2013C (incl. jitter)"
        x_label = "MJD"
        y_label_rv = "RV [km s\u207b\u00b9]"
        y_label_res = "O\u2013C [km s\u207b\u00b9]"
        sub_titles = None

    # ── subplots ──────────────────────────────────────────────────────
    vspacing = 0.06 if s.show_title else 0.10
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=vspacing,
        subplot_titles=sub_titles,
    )

    # Data
    fig.add_trace(
        go.Scatter(
            x=hjds, y=vels, mode="markers", name=lbl_data,
            marker=dict(size=s.marker_size, opacity=0.7, color=s.color_data_primary),
            error_y=dict(type="data", array=errs, visible=True,
                         color=s.color_data_primary),
            hovertemplate="MJD=%{x:.4f}<br>RV=%{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    if jitter:
        errs2 = np.sqrt(errs ** 2.0 + jitter ** 2.0)
        fig.add_trace(
            go.Scatter(
                x=hjds, y=vels, mode="markers", name=lbl_jit,
                marker=dict(size=s.marker_size, opacity=0.5,
                            color=s.color_jitter, symbol="circle-open"),
                error_y=dict(type="data", array=errs2, visible=True,
                             color=s.color_jitter),
                hovertemplate="MJD=%{x:.4f}<br>RV=%{y:.3f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Model
    fig.add_trace(
        go.Scatter(
            x=time_grid, y=rv_model, mode="lines", name=lbl_model,
            line=dict(width=s.line_width_model, color=s.color_model),
            hovertemplate="MJD=%{x:.4f}<br>Model=%{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Residuals
    fig.add_trace(
        go.Scatter(
            x=hjds, y=residuals, mode="markers", name=lbl_res,
            marker=dict(size=s.marker_size, opacity=0.7, color=s.color_residuals),
            error_y=dict(type="data", array=errs, visible=True,
                         color=s.color_residuals),
            hovertemplate="MJD=%{x:.4f}<br>O\u2013C=%{y:.3f}<extra></extra>",
        ),
        row=2, col=1,
    )

    if jitter:
        errs2 = np.sqrt(errs ** 2.0 + jitter ** 2.0)
        fig.add_trace(
            go.Scatter(
                x=hjds, y=residuals, mode="markers", name=lbl_res_jit,
                marker=dict(size=s.marker_size, opacity=0.5,
                            color=s.color_jitter, symbol="circle-open"),
                error_y=dict(type="data", array=errs2, visible=True,
                             color=s.color_jitter),
                hovertemplate="MJD=%{x:.4f}<br>O\u2013C=%{y:.3f}<extra></extra>",
            ),
            row=2, col=1,
        )

    fig.add_hline(y=0, line_width=s.line_width_reference, line_dash="dot",
                  line_color=s.color_residuals, row=2, col=1)

    # ── axes ──────────────────────────────────────────────────────────
    frame_kw = dict(showline=True, linewidth=1, linecolor="black",
                    mirror=True) if s.show_axis_frame else {}
    fig.update_xaxes(title_text=x_label, row=2, col=1,
                     title_font=dict(size=s.font_axis_title),
                     tickfont=dict(size=s.font_tick),
                     showgrid=s.show_grid, gridcolor=s.grid_color, **frame_kw)
    fig.update_yaxes(title_text=y_label_rv, row=1, col=1,
                     title_font=dict(size=s.font_axis_title),
                     tickfont=dict(size=s.font_tick),
                     showgrid=s.show_grid, gridcolor=s.grid_color, **frame_kw)
    fig.update_yaxes(title_text=y_label_res, row=2, col=1,
                     title_font=dict(size=s.font_axis_title),
                     tickfont=dict(size=s.font_tick),
                     showgrid=s.show_grid, gridcolor=s.grid_color, **frame_kw)
    # top subplot x-axis also needs frame
    fig.update_xaxes(row=1, col=1,
                     showgrid=s.show_grid, gridcolor=s.grid_color, **frame_kw)

    # ── layout ────────────────────────────────────────────────────────
    w = s.width
    h = s.height
    apply_style_to_layout(fig, s)
    fig.update_layout(
        width=w, height=h,
        title=title if title else None,
        title_font=dict(size=s.font_title),
    )

    # ── save ──────────────────────────────────────────────────────────
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        safe_star = _sanitize_filename(star_name)
        base = f"{safe_star}_sid-{solution_id}_time_residuals"

        fig.write_html(os.path.join(out_dir, f"{base}.html"),
                       include_plotlyjs="cdn", full_html=True)
        fig.write_image(os.path.join(out_dir, f"{base}.png"),
                        width=w, height=h, scale=s.scale)
        if s.export_pdf:
            fig.write_image(os.path.join(out_dir, f"{base}.pdf"),
                            format="pdf", width=w, height=h, scale=s.scale)

    return residuals


def plot_phase_folded_with_residuals_plotly(
    hjds, vels, errs, P, T0,
    Gamma, K, Omega, ecc,
    star_name, solution_id=0, jitter=0, out_dir=None, plot=True,
    figsize=(1000, 700),
    style="interactive",
):
    s = get_style(style)

    phs_data, phase_grid, rv_phase, residuals = compute_phase_residuals(
        hjds, vels, P, T0, ecc, Gamma, K, Omega,
    )
    vels = np.asarray(vels, dtype=float)
    errs = np.asarray(errs, dtype=float)

    if plot and out_dir:
        # ── labels ────────────────────────────────────────────────────
        if s.show_title:
            lbl_data = "Measurements (\u00b11\u03c3, statistical errors)"
            lbl_jit = "Measurements (\u00b11\u03c3, including jitter)"
            lbl_model = "Best-fit model"
            lbl_res = "Residuals (\u00b11\u03c3, statistical errors)"
            lbl_res_jit = "Residuals (\u00b11\u03c3, including jitter)"
            x_label = "Orbital phase (cycle fraction)"
            y_label_rv = "Radial velocity [km s\u207b\u00b9]"
            y_label_res = "Residuals [km s\u207b\u00b9]"
            sub_titles = ("Phase-folded radial-velocity curve",
                          "Phase-folded residuals")
            title = (f"Phase-folded RV curve for {star_name}: "
                     f"P = {P:.2f} d, e = {ecc:.3f}")
        else:
            lbl_data = "Data"
            lbl_jit = "Data (incl. jitter)"
            lbl_model = "Best fit"
            lbl_res = "O\u2013C"
            lbl_res_jit = "O\u2013C (incl. jitter)"
            x_label = "Phase"
            y_label_rv = "RV [km s\u207b\u00b9]"
            y_label_res = "O\u2013C [km s\u207b\u00b9]"
            sub_titles = None
            title = ""

        # ── subplots ──────────────────────────────────────────────────
        vspacing = 0.06 if s.show_title else 0.10
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=vspacing,
            subplot_titles=sub_titles,
        )

        # Data
        fig.add_trace(
            go.Scatter(
                x=phs_data, y=vels, mode="markers", name=lbl_data,
                marker=dict(size=s.marker_size, opacity=0.7,
                            color=s.color_data_secondary),
                error_y=dict(type="data", array=errs, visible=True,
                             color=s.color_data_secondary),
                hovertemplate="Phase=%{x:.4f}<br>RV=%{y:.3f}<extra></extra>",
            ),
            row=1, col=1,
        )

        if jitter:
            errs2 = np.sqrt(errs ** 2.0 + jitter ** 2.0)
            fig.add_trace(
                go.Scatter(
                    x=phs_data, y=vels, mode="markers", name=lbl_jit,
                    marker=dict(size=s.marker_size, opacity=0.5,
                                color=s.color_jitter, symbol="circle-open"),
                    error_y=dict(type="data", array=errs2, visible=True,
                                 color=s.color_jitter),
                    hovertemplate="Phase=%{x:.4f}<br>RV=%{y:.3f}<extra></extra>",
                ),
                row=1, col=1,
            )

        # Model
        fig.add_trace(
            go.Scatter(
                x=phase_grid, y=rv_phase, mode="lines", name=lbl_model,
                line=dict(width=s.line_width_model, color=s.color_model),
                hovertemplate="Phase=%{x:.4f}<br>Model=%{y:.3f}<extra></extra>",
            ),
            row=1, col=1,
        )

        # Gamma line
        fig.add_hline(y=float(Gamma), line_dash="dash",
                      line_color=s.color_gamma,
                      line_width=s.line_width_reference, row=1, col=1)

        # Residuals
        fig.add_trace(
            go.Scatter(
                x=phs_data, y=residuals, mode="markers", name=lbl_res,
                marker=dict(size=s.marker_size, opacity=0.8,
                            color=s.color_residuals),
                error_y=dict(type="data", array=errs, visible=True,
                             color=s.color_residuals),
                hovertemplate="Phase=%{x:.4f}<br>O\u2013C=%{y:.3f}<extra></extra>",
            ),
            row=2, col=1,
        )

        if jitter:
            errs2 = np.sqrt(errs ** 2.0 + jitter ** 2.0)
            fig.add_trace(
                go.Scatter(
                    x=phs_data, y=residuals, mode="markers", name=lbl_res_jit,
                    marker=dict(size=s.marker_size, opacity=0.5,
                                color=s.color_jitter, symbol="circle-open"),
                    error_y=dict(type="data", array=errs2, visible=True,
                                 color=s.color_jitter),
                    hovertemplate="Phase=%{x:.4f}<br>O\u2013C=%{y:.3f}<extra></extra>",
                ),
                row=2, col=1,
            )

        fig.add_hline(y=0, line_width=s.line_width_reference, line_dash="dot",
                      line_color=s.color_residuals, row=2, col=1)

        # ── axes ──────────────────────────────────────────────────────
        frame_kw = dict(showline=True, linewidth=1, linecolor="black",
                        mirror=True) if s.show_axis_frame else {}
        fig.update_xaxes(title_text=x_label, range=[0, 1], row=2, col=1,
                         title_font=dict(size=s.font_axis_title),
                         tickfont=dict(size=s.font_tick),
                         showgrid=s.show_grid, gridcolor=s.grid_color, **frame_kw)
        fig.update_yaxes(title_text=y_label_rv, row=1, col=1,
                         title_font=dict(size=s.font_axis_title),
                         tickfont=dict(size=s.font_tick),
                         showgrid=s.show_grid, gridcolor=s.grid_color, **frame_kw)
        fig.update_yaxes(title_text=y_label_res, row=2, col=1,
                         title_font=dict(size=s.font_axis_title),
                         tickfont=dict(size=s.font_tick),
                         showgrid=s.show_grid, gridcolor=s.grid_color, **frame_kw)
        fig.update_xaxes(row=1, col=1,
                         showgrid=s.show_grid, gridcolor=s.grid_color, **frame_kw)

        # ── layout ────────────────────────────────────────────────────
        w = s.width
        h = s.height
        apply_style_to_layout(fig, s)
        fig.update_layout(
            width=w, height=h,
            title=title if title else None,
            title_font=dict(size=s.font_title),
        )

        # ── save ──────────────────────────────────────────────────────
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            safe_star = _sanitize_filename(star_name)
            base = f"{safe_star}_sid-{solution_id}_phase_residuals"

            fig.write_html(os.path.join(out_dir, f"{base}.html"),
                           include_plotlyjs="cdn", full_html=True)
            fig.write_image(os.path.join(out_dir, f"{base}.png"),
                            width=w, height=h, scale=s.scale)
            if s.export_pdf:
                fig.write_image(os.path.join(out_dir, f"{base}.pdf"),
                                format="pdf", width=w, height=h, scale=s.scale)

    return phs_data, phase_grid, rv_phase, residuals


# ---------------------------------------------------------------------------
# High-level result printing
# ---------------------------------------------------------------------------

def print_lmfit_result(data, args_dict, star_name, result, solution_id=0, out_dir=None):
    sys.setrecursionlimit(int(1e6))
    plot_style = args_dict.get(PLOT_STYLE, "interactive") if args_dict else "interactive"
    hjds, vels, errs = extract_observations(data)
    get_fit_report(result, star_name, solution_id, out_dir)
    Gamma, K, Omega, ecc, P, T0 = compute_orbital_params(result)
    sig_jit = 0
    if LN_SIGMA_JITTER in result.params.keys():
        sig_jit = np.exp(result.params[LN_SIGMA_JITTER].value)
    time_grid, rv_time = compute_rv_curve_time(hjds, P, T0, ecc, Gamma, K, Omega)
    plot_time_series_with_residuals_plotly(
        hjds, vels, errs, time_grid, rv_time,
        Gamma, K, Omega, ecc, P, star_name, solution_id, jitter=sig_jit,
        out_dir=out_dir, style=plot_style,
    )
    phs_data, phase_grid, rv_phase, residuals = compute_phase_residuals(
        hjds, vels, P, T0, ecc, Gamma, K, Omega,
    )
    plot_phase_folded_with_residuals_plotly(
        hjds, vels, errs, P, T0,
        Gamma, K, Omega, ecc,
        star_name, solution_id, jitter=sig_jit, out_dir=out_dir, plot=True,
        style=plot_style,
    )

    return phs_data


def print_lmfit_result_null(
    data, args_dict, star_name, result, solution_id=0,
    out_dir=None, use_jitter=False
):
    """
    Dump the fit report and produce time-series plots for the null (constant-RV) model.
    """
    sys.setrecursionlimit(int(1e6))
    hjds, vels, errs = extract_observations(data)

    Gamma_val = None
    sig_jit = 0.0

    if hasattr(result, "params"):
        get_fit_report(result, star_name, solution_id, out_dir)

        if GAMMA in result.params:
            Gamma_val = result.params[GAMMA].value
        else:
            Gamma_val = getattr(result, GAMMA, np.nan)

        if use_jitter and (LN_SIGMA_JITTER in result.params):
            sig_jit = np.exp(result.params[LN_SIGMA_JITTER].value)
    else:
        Gamma_val = getattr(result, GAMMA, np.nan)

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            report_path = os.path.join(out_dir, f"{star_name}_sid-{solution_id}_report.txt")
            with open(report_path, "w") as f:
                f.write("Null (constant-RV) model\n")
                f.write(f"gamma = {Gamma_val:.6f}\n")
                f.write(f"redchi = {getattr(result, 'redchi', np.nan):.6g}\n")
                f.write(f"AIC = {getattr(result, 'aic', np.nan):.6g}\n")
                f.write(f"BIC = {getattr(result, 'bic', np.nan):.6g}\n")

    time_grid, rv_time = compute_null_rv_curve_time(hjds, Gamma_val)

    plot_style = args_dict.get(PLOT_STYLE, "interactive") if args_dict else "interactive"
    plot_time_series_with_residuals_plotly(
        hjds, vels, errs, time_grid, rv_time,
        Gamma_val, 0.0, 0.0, 0.0, None,
        star_name, solution_id, jitter=sig_jit, out_dir=out_dir,
        style=plot_style,
    )
