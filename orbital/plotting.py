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
    LN_SIGMA_JITTER,
)
from orbital.kepler import Kepler, v1mod, true_anomaly_from_E
from orbital.fitting import extract_observations
from orbital.statistics import compute_orbital_params


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
                                     star_name, solution_id=0, out_dir=None, plot=True,
                                     figsize=(10, 10)):
    phs_data = ((hjds - T0) / P) % 1
    phase_grid, rv_phase = compute_rv_curve_phase(P, T0, ecc, Gamma, K, Omega)
    vmod_data = np.interp(phs_data, phase_grid, rv_phase)
    residuals = vels - vmod_data
    if plot:
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
        if out_dir:
            fig.savefig(os.path.join(out_dir, f"{star_name}_sid-{solution_id}_phase_residuals.png"))
        plt.show()
    return phs_data, phase_grid, rv_phase, residuals


# ---------------------------------------------------------------------------
# Plotly plots
# ---------------------------------------------------------------------------

def plot_time_series_with_residuals_plotly(
    hjds, vels, errs, time_grid, rv_model,
    Gamma, K, Omega, ecc, P, star_name, solution_id=0, jitter=0.0,
    out_dir=None, figsize=(1000, 700)
):
    hjds = np.asarray(hjds, dtype=float)
    vels = np.asarray(vels, dtype=float)
    errs = np.asarray(errs, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    rv_model = np.asarray(rv_model, dtype=float)

    vmod_data = np.interp(hjds, time_grid, rv_model)
    residuals = vels - vmod_data

    if (P is None) or (not np.isfinite(P)):
        title = f"Null (constant radial-velocity) model for {star_name}"
    else:
        title = (
            f"Orbital fit for {star_name}: "
            f"Period P = {P:.2f} d, eccentricity e = {ecc:.3f}"
        )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.06,
        subplot_titles=("Radial velocity versus time", "Residuals")
    )

    fig.add_trace(
        go.Scatter(
            x=hjds, y=vels, mode="markers",
            name="Measurements (\u00b11\u03c3, statistical errors)",
            marker=dict(size=8, opacity=0.7, color="royalblue"),
            error_y=dict(type="data", array=errs, visible=True, color="royalblue"),
            hovertemplate=(
                "Time (MJD) = %{x:.6f}<br>"
                "Radial velocity = %{y:.4f} km/s"
                "<br>\u00b11\u03c3 statistical uncertainty"
                "<extra></extra>"
            ),
        ),
        row=1, col=1
    )

    if jitter:
        errs2 = np.sqrt(errs**2.0 + jitter**2.0)
        fig.add_trace(
            go.Scatter(
                x=hjds, y=vels, mode="markers",
                name="Measurements (\u00b11\u03c3, including jitter)",
                marker=dict(size=8, opacity=0.5, color="orange", symbol="circle-open"),
                error_y=dict(type="data", array=errs2, visible=True, color="orange"),
                hovertemplate=(
                    "Time (MJD) = %{x:.6f}<br>"
                    "Radial velocity = %{y:.4f} km/s"
                    "<br>\u00b11\u03c3 including jitter term"
                    "<extra></extra>"
                ),
                showlegend=True
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=time_grid, y=rv_model, mode="lines", name="Best-fit model",
            line=dict(width=3),
            hovertemplate=(
                "Time (MJD) = %{x:.6f}<br>"
                "Model radial velocity = %{y:.4f} km/s"
                "<extra>Model</extra>"
            ),
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=hjds, y=residuals, mode="markers",
            name="Residuals (\u00b11\u03c3, statistical errors)",
            marker=dict(size=8, opacity=0.7),
            error_y=dict(type="data", array=errs, visible=True),
            hovertemplate=(
                "Time (MJD) = %{x:.6f}<br>"
                "Residuals = %{y:.4f} km/s"
                "<br>\u00b11\u03c3 statistical uncertainty"
                "<extra>Residuals</extra>"
            ),
        ),
        row=2, col=1
    )

    if jitter:
        errs2 = np.sqrt(errs**2.0 + jitter**2.0)
        fig.add_trace(
            go.Scatter(
                x=hjds, y=residuals, mode="markers",
                name="Residuals (\u00b11\u03c3, including jitter)",
                marker=dict(size=8, opacity=0.5, color="orange", symbol="circle-open"),
                error_y=dict(type="data", array=errs2, visible=True, color="orange"),
                hovertemplate=(
                    "Time (MJD) = %{x:.6f}<br>"
                    "Residuals = %{y:.4f} km/s"
                    "<br>\u00b11\u03c3 including jitter term"
                    "<extra>Residuals</extra>"
                ),
                showlegend=True
            ),
            row=2, col=1
        )

    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_xaxes(title_text="Time (Modified Julian Date)", row=2, col=1,
                     title_font=dict(size=20), tickfont=dict(size=16))
    fig.update_yaxes(title_text="Radial velocity [km s\u207b\u00b9]", row=1, col=1,
                     title_font=dict(size=20), tickfont=dict(size=16))
    fig.update_yaxes(title_text="Residuals [km s\u207b\u00b9]", row=2, col=1,
                     title_font=dict(size=20), tickfont=dict(size=16))

    fig.update_layout(
        width=figsize[0], height=figsize[1],
        template="plotly_white", title=title,
        title_font=dict(size=22), font=dict(size=18),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", yanchor="top",
                    font=dict(size=16)),
        margin=dict(l=80, r=20, t=80, b=90),
    )

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        safe_star = _sanitize_filename(star_name)
        html_path = os.path.join(out_dir, f"{safe_star}_sid-{solution_id}_time_residuals.html")
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        png_path = os.path.join(out_dir, f"{safe_star}_sid-{solution_id}_time_residuals.png")
        fig.write_image(png_path, scale=4)

    return residuals


def plot_phase_folded_with_residuals_plotly(
    hjds, vels, errs, P, T0,
    Gamma, K, Omega, ecc,
    star_name, solution_id=0, jitter=0, out_dir=None, plot=True,
    figsize=(1000, 700)
):
    phs_data = ((np.asarray(hjds, dtype=float) - float(T0)) / float(P)) % 1.0
    phase_grid, rv_phase = compute_rv_curve_phase(P, T0, ecc, Gamma, K, Omega)
    phase_grid = np.asarray(phase_grid, dtype=float)
    rv_phase = np.asarray(rv_phase, dtype=float)

    vmod_data = np.interp(phs_data, phase_grid, rv_phase)
    vels = np.asarray(vels, dtype=float)
    errs = np.asarray(errs, dtype=float)
    residuals = vels - vmod_data

    if plot:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.06,
            subplot_titles=("Phase-folded radial-velocity curve", "Phase-folded residuals")
        )

        fig.add_trace(
            go.Scatter(
                x=phs_data, y=vels, mode="markers",
                name="Measurements (\u00b11\u03c3, statistical errors)",
                marker=dict(size=8, opacity=0.7, color="crimson"),
                error_y=dict(type="data", array=errs, visible=True),
                hovertemplate=(
                    "Orbital phase = %{x:.5f}<br>"
                    "Radial velocity = %{y:.4f} km/s"
                    "<br>\u00b11\u03c3 statistical uncertainty"
                    "<extra>Data</extra>"
                ),
            ),
            row=1, col=1
        )

        if jitter:
            errs2 = np.sqrt(errs**2.0 + jitter**2.0)
            fig.add_trace(
                go.Scatter(
                    x=phs_data, y=vels, mode="markers",
                    name="Measurements (\u00b11\u03c3, including jitter)",
                    marker=dict(size=8, opacity=0.5, color="orange", symbol="circle-open"),
                    error_y=dict(type="data", array=errs2, visible=True, color="orange"),
                    hovertemplate=(
                        "Orbital phase = %{x:.5f}<br>"
                        "Radial velocity = %{y:.4f} km/s"
                        "<br>\u00b11\u03c3 including jitter term"
                        "<extra>Data</extra>"
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=phase_grid, y=rv_phase, mode="lines",
                name="Best-fit model",
                line=dict(width=3, color="crimson"),
                hovertemplate=(
                    "Orbital phase = %{x:.5f}<br>"
                    "Model radial velocity = %{y:.4f} km/s"
                    "<extra>Model</extra>"
                ),
            ),
            row=1, col=1
        )

        fig.add_hline(y=float(Gamma), line_dash="dash", line_color="gray", row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=phs_data, y=residuals, mode="markers",
                name="Residuals (\u00b11\u03c3, statistical errors)",
                marker=dict(size=8, opacity=0.8, color="gray"),
                error_y=dict(type="data", array=errs, visible=True),
                hovertemplate=(
                    "Orbital phase = %{x:.5f}<br>"
                    "Residuals = %{y:.4f} km/s"
                    "<br>\u00b11\u03c3 statistical uncertainty"
                    "<extra>Residuals</extra>"
                ),
            ),
            row=2, col=1
        )

        if jitter:
            errs2 = np.sqrt(errs**2.0 + jitter**2.0)
            fig.add_trace(
                go.Scatter(
                    x=phs_data, y=residuals, mode="markers",
                    name="Residuals (\u00b11\u03c3, including jitter)",
                    marker=dict(size=8, opacity=0.5, color="orange", symbol="circle-open"),
                    error_y=dict(type="data", array=errs2, visible=True, color="orange"),
                    hovertemplate=(
                        "Orbital phase = %{x:.5f}<br>"
                        "Residuals = %{y:.4f} km/s"
                        "<br>\u00b11\u03c3 including jitter term"
                        "<extra>Residuals</extra>"
                    ),
                    showlegend=True
                ),
                row=2, col=1
            )

        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)

        fig.update_xaxes(title_text="Orbital phase (cycle fraction)", range=[0, 1],
                         row=2, col=1, title_font=dict(size=20), tickfont=dict(size=16))
        fig.update_yaxes(title_text="Radial velocity [km s\u207b\u00b9]", row=1, col=1,
                         title_font=dict(size=20), tickfont=dict(size=16))
        fig.update_yaxes(title_text="Residuals [km s\u207b\u00b9]", row=2, col=1,
                         title_font=dict(size=20), tickfont=dict(size=16))

        fig.update_layout(
            width=figsize[0], height=figsize[1],
            template="plotly_white",
            title=(f"Phase-folded radial-velocity curve for {star_name}: "
                   f"P = {P:.2f} d, e = {ecc:.3f}"),
            title_font=dict(size=22), font=dict(size=18),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", yanchor="top",
                        font=dict(size=16)),
            margin=dict(l=80, r=20, t=80, b=90),
        )

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            safe_star = _sanitize_filename(star_name)
            html_path = os.path.join(out_dir, f"{safe_star}_sid-{solution_id}_phase_residuals.html")
            fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
            png_path = os.path.join(out_dir, f"{safe_star}_sid-{solution_id}_phase_residuals.png")
            fig.write_image(png_path, scale=4)

    return phs_data, phase_grid, rv_phase, residuals


# ---------------------------------------------------------------------------
# High-level result printing
# ---------------------------------------------------------------------------

def print_lmfit_result(data, args_dict, star_name, result, solution_id=0, out_dir=None):
    sys.setrecursionlimit(int(1e6))
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
        out_dir=out_dir
    )
    phs_data, phase_grid, rv_phase, residuals = plot_phase_folded_with_residuals(
        hjds, vels, errs, P, T0,
        Gamma, K, Omega, ecc,
        star_name, solution_id, out_dir, plot=False
    )
    plot_phase_folded_with_residuals_plotly(
        hjds, vels, errs, P, T0,
        Gamma, K, Omega, ecc,
        star_name, solution_id, jitter=sig_jit, out_dir=out_dir, plot=True
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

    plot_time_series_with_residuals_plotly(
        hjds, vels, errs, time_grid, rv_time,
        Gamma_val, 0.0, 0.0, 0.0, None,
        star_name, solution_id, jitter=sig_jit, out_dir=out_dir
    )
