import sys
import os
import io
import contextlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lmfit
import matplotlib.pylab as pylab
import argparse
import corner
import json
from types import SimpleNamespace
from plotly.subplots import make_subplots
import plotly.graph_objects as go


from Spectroscopy.constants import MJD_MID
from utils.constants import *
from utils.utils import make_runlog

import numpy as np

def _finite_or_default(x, default=1.0):
    """Return array with non-finite values replaced by default."""
    x = np.asarray(x, float)
    bad = ~np.isfinite(x)
    if np.any(bad):
        x = x.copy()
        x[bad] = default
    return x

def _safe_errors(errs):
    """Replace non-finite / non-positive errors by a small positive value."""
    errs = np.asarray(errs, float)
    errs = np.where(np.isfinite(errs), errs, np.nan)
    # smallest sensible sigma from finite positives or fallback
    pos = errs[np.isfinite(errs) & (errs > 0)]
    floor = np.nanmedian(pos) * 1e-3 if pos.size else 1e-3
    floor = max(floor, 1e-6)
    errs = np.where(errs > 0, errs, floor)
    return errs

def define_search_param(params, search_params_dict, field, radius=0):
    """
    Defines a search parameter for a given field and adds it to the parameters object.

    Args:
        params (Parameters): The parameters object to which the search parameter will be added.
        search_params_dict (dict): A dictionary containing search parameter details with the following structure:
            - field (str): The name of the field for which the search parameter is defined.
            - INIT_VAL (float): The initial value of the parameter.
            - MIN_VAL (float): The minimum value of the parameter.
            - MAX_VAL (float): The maximum value of the parameter.
            - VARY (bool): A flag indicating whether the parameter should vary during the search.
        field (str): The specific field for which the search parameter is being defined.
        radius (float, optional): An optional radius value to adjust the min and max values of the parameter. Default is 0.

    Returns:
        None: The function modifies the `params` object in place.

    Notes:
        - The function adjusts the minimum and maximum values of the parameter by subtracting and adding half of the radius, respectively.
        - The `vary` flag determines whether the parameter will vary during the search process.
    """
    params.add(field, value=search_params_dict[field][INIT_VAL] ,
               min=search_params_dict[field][MIN_VAL] - radius/2,
               max=search_params_dict[field][MAX_VAL] + radius/2,
               vary=search_params_dict[field][VARY])

def calculate_statistical_flags(data, mini_results, is_null):
    rvs = data[RADIAL_VELS]
    # rv_errs = data[ERRORS]
    ts = data[TIME_STAMPS]
    rv_errs = _safe_errors(data[ERRORS])
    p = mini_results.params
    Gamma1 = float(p[GAMMA].value)
    sigma_j = float(np.exp(p[LN_SIGMA_JITTER].value))
    if not is_null:
        K1     = float(p[K1_STR].value)
        Omega  = float(p[OMEGA].value)
        ecc    = float(p[ECC].value)
        T0     = float(p[T].value)
        P      = float (p[PERIOD].value)
        nu = nus1(ts, P, T0, ecc)
        v1 = v1mod(nu, Gamma1, K1, Omega, ecc)
    else:
        v1 = Gamma1

    sig2 = rv_errs ** 2 + sigma_j ** 2
    # ---------- explicit statistics (matching lmfit conventions) ----------
    # weighted residuals
    resid = rvs - v1
    calc_chisqr = np.sum(resid**2 / sig2)

    ndata  = rvs.size
    nvarys = mini_results.nvarys   # number of free parameters actually varied
    dof    = max(ndata - nvarys, 1)

    redchi = calc_chisqr / dof

    llh = calc_chisqr + np.sum(np.log(sig2))
    aic = 2*nvarys + llh
    bic = nvarys*np.log(ndata) +llh

    bicc =  nvarys*np.log(ndata)*(ndata/(ndata-nvarys-2)) +llh

    ev_bic =  nvarys*np.log(ndata) +llh*(1-1/ndata)
    ret_dict = {
        "llh": llh,
        "chisqr": calc_chisqr,
        "aic": aic,
        "bic": bic,
        "bicc": bicc,
        "ev_bic": ev_bic,
        "redchi": redchi,
        "ndata": ndata,
    }
    return ret_dict

def summarize_result(result, star_name):
    """
    Turn a single lmfit.MinimizerResult into a flat dict.
    """
    row = {}
    # 1) Global fit statistics
    row['star_name'] = star_name
    row['method'] = result.method
    row['nfev'] = result.nfev  # function evals
    row['ndata'] = result.ndata  # data points
    row['nvarys'] = result.nvarys  # fitted vars
    row['chisqr'] = result.chisqr
    row['redchi'] = result.redchi
    row['aic'] = result.aic
    row['bic'] = result.bic

    # 2) Per-parameter summaries
    # result.params is an OrderedDict of Parameter objects
    for name, par in result.params.items():
        init = result.init_values.get(name, None)  # initial guess
        row[f'{name}_init'] = init
        row[f'{name}_value'] = par.value
        row[f'{name}_vary'] = par.vary
        # if stderr is None → warning that uncertainty wasn't estimated
        row[f'{name}_stderr'] = par.stderr

    return row



def extract_observations(data):
    hjds = np.array(data[TIME_STAMPS])
    vels = np.array(data[RADIAL_VELS])
    errs = np.abs(data[ERRORS])
    return hjds, vels, errs


def get_fit_report(result, star_name,solution_id=0, out_dir=None):
    report = lmfit.fit_report(result)
    if out_dir:
        path = os.path.join(out_dir, f"{star_name}_sid-{solution_id}_report.txt")
        with open(path, 'w') as f:
            f.write(report)
        print(f"[{star_name}] Fit report saved to {path}")


def compute_orbital_params(result):
    params = result.params
    return (
        params[GAMMA].value,
        params[K1_STR].value,
        params[OMEGA].value,
        params[ECC].value,
        params[PERIOD].value,
        params[T].value
    )


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
    phase_grid = np.linspace(0,1,n_points)
    M = 2*np.pi*phase_grid
    E = Kepler(np.pi, M, ecc)
    if np.isnan(E.all()):
        return phase_grid, np.full(phase_grid.shape, np.nan)
    nu = 2*np.arctan2(np.sqrt(1+ecc)*np.sin(E/2), np.sqrt(1-ecc)*np.cos(E/2))
    rv = v1mod(nu, Gamma, K, Omega, ecc)
    return phase_grid, rv


def plot_time_series_with_residuals(hjds, vels, errs, time_grid, rv_model,
                                    Gamma, K, Omega, ecc, P, star_name, solution_id=0,
                                    out_dir=None, figsize=(10,10)):
    vmod_data = np.interp(hjds, time_grid, rv_model)
    residuals = vels - vmod_data

    fig, (ax1, ax1r) = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={'height_ratios':[3,1]}
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
    ax1r.set_ylabel('O–C')

    fig.tight_layout()
    if out_dir:
        fig.savefig(os.path.join(out_dir, f"{star_name}_sid-{solution_id}_time_residuals.png"))
    plt.show()
    return residuals

def coverage_qc(gaps_dict, rv_values):
    """
    gaps_dict = {
        "max_phase_gap": ...,
        "top_max_rv_gap": ...,
        "bottom_max_rv_gap": ...
    }
    rv_values: array-like of measured RVs (km/s)
    """
    rv = np.asarray(rv_values, dtype=float)
    p10, p90 = np.percentile(rv, [10, 90])
    rv_span = max(p90 - p10, 1e-9)  # avoid divide-by-zero

    g_phase = float(gaps_dict["max_phase_gap"])
    gtop_n  = float(gaps_dict["top_max_rv_gap"])
    gbot_n  = float(gaps_dict["bottom_max_rv_gap"])

    phase_ok = (g_phase <= 0.25)
    rv_ok    = (gtop_n <= 0.50) and (gbot_n <= 0.50)

    score = 1.0 - (0.30*g_phase + 0.35*gtop_n + 0.35*gbot_n)

    passed = phase_ok and rv_ok and (score >= 0.60)

    return {
        "ph_gtop_norm": round(gtop_n, 3),
        "ph_gbot_norm": round(gbot_n, 3),
        "ph_phase_ok": phase_ok,
        "ph_rv_ok": rv_ok,
        "ph_score": round(score, 3),
        "ph_pass": passed,
        "ph_tier": "good" if score >= 0.70 else ("ok" if score >= 0.60 else "needs_work"),
    }

def calculate_phase_criterias(data, result):
    """
    Returns:
        max_phase_gap         in [0,1]
        top_max_rv_gap        in [0,1]  (distance from model crest)
        bottom_max_rv_gap     in [0,1]  (distance from model trough)
    """
    hjds, vels, errs = extract_observations(data)
    Gamma, K, Omega, ecc, P, T0 = compute_orbital_params(result)

    # phs_data: observed phases in [0,1), rv_phase: model RV at those phases (or model sampled at obs phases)
    phs_data, phase_grid, rv_phase, residuals = plot_phase_folded_with_residuals(
        hjds, vels, errs, P, T0, Gamma, K, Omega, ecc, '', plot=False
    )

    # --- Max phase gap on the circle ---
    phs = np.asarray(phs_data, dtype=float) % 1.0
    phs = np.sort(phs)
    if phs.size >= 2:
        diffs = np.diff(phs, append=phs[0] + 1.0)  # includes wrap-around gap
        max_phase_gap = float(np.nanmax(diffs))
    else:
        max_phase_gap = 1.0  # with 0/1 point, coverage is effectively unknown -> worst

    # --- RV crest/trough headroom (how far observed extrema are from model extrema) ---
    v_obs_max = float(np.nanmax(vels))
    v_obs_min = float(np.nanmin(vels))
    v_mod_max = float(np.nanmax(rv_phase))
    v_mod_min = float(np.nanmin(rv_phase))

    # Distances to model extrema
    crest_headroom   = v_mod_max - v_obs_max         # ≥0 if we didn’t reach crest
    trough_headroom  = v_obs_min - v_mod_min         # ≥0 if we didn’t reach trough

    # Normalization scales relative to Gamma
    crest_scale  = max(v_mod_max - Gamma, 0.0)
    trough_scale = max(Gamma - v_mod_min, 0.0)

    # Normalize; guard against zero scales
    if crest_scale > 0:
        top_max_rv_gap = crest_headroom / crest_scale
    else:
        top_max_rv_gap = np.nan  # flat model / pathological case

    if trough_scale > 0:
        bottom_max_rv_gap = trough_headroom / trough_scale
    else:
        bottom_max_rv_gap = np.nan

    # Clip to [0,1] (negative due to noise -> 0; >1 due to outliers -> 1)
    def _clip01(x):
        return float(np.clip(x, 0.0, 1.0)) if np.isfinite(x) else np.nan

    max_phase_gap      = _clip01(max_phase_gap)
    top_max_rv_gap     = _clip01(top_max_rv_gap)
    bottom_max_rv_gap  = _clip01(bottom_max_rv_gap)

    return {
        "max_phase_gap": np.round(max_phase_gap, 2),
        "top_max_rv_gap": np.round(top_max_rv_gap, 2),
        "bottom_max_rv_gap": np.round(bottom_max_rv_gap, 2),
    }


def _sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in str(name)).strip().replace(" ", "_")

def compute_null_rv_curve_time(hjds, Gamma, ngrid=500, pad_frac=0.05):
    """
    Build a simple time grid and RV curve for the null (constant-velocity) model.

    Parameters
    ----------
    hjds : array-like
        Observation timestamps.
    Gamma : float
        Systemic velocity of the null model.
    ngrid : int, optional
        Number of points in the time grid.
    pad_frac : float, optional
        Fractional padding added before/after the data time span.

    Returns
    -------
    time_grid : ndarray
        Time grid spanning the data.
    rv_model : ndarray
        Constant RV model evaluated on time_grid.
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


# ---------- Time series (RV vs MJD) + residuals ----------
def plot_time_series_with_residuals_plotly(
    hjds, vels, errs, time_grid, rv_model,
    Gamma, K, Omega, ecc, P, star_name, solution_id=0, jitter=0.0,
    out_dir=None, figsize=(1000, 700)  # (width, height) in px
):
    hjds = np.asarray(hjds, dtype=float)
    vels = np.asarray(vels, dtype=float)
    errs = np.asarray(errs, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    rv_model = np.asarray(rv_model, dtype=float)

    # Model sampled at data timestamps
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
        subplot_titles=(
            "Radial velocity versus time",
            "Residuals"
        )
    )

    # Top: data + error bars
    fig.add_trace(
        go.Scatter(
            x=hjds, y=vels, mode="markers",
            name="Measurements (±1σ, statistical errors)",
            marker=dict(size=8, opacity=0.7, color="royalblue"),
            error_y=dict(
                type="data",
                array=errs,
                visible=True,
                color="royalblue"
            ),
            hovertemplate=(
                "Time (MJD) = %{x:.6f}<br>"
                "Radial velocity = %{y:.4f} km/s"
                "<br>±1σ statistical uncertainty"
                "<extra></extra>"
            ),
        ),
        row=1, col=1
    )

    # Second error bars (e.g., including jitter)
    if jitter:
        errs2 = np.sqrt(errs**2.0 + jitter**2.0)
        fig.add_trace(
            go.Scatter(
                x=hjds, y=vels, mode="markers",
                name="Measurements (±1σ, including jitter)",
                marker=dict(size=8, opacity=0.5, color="orange", symbol="circle-open"),
                error_y=dict(
                    type="data",
                    array=errs2,
                    visible=True,
                    color="orange"
                ),
                hovertemplate=(
                    "Time (MJD) = %{x:.6f}<br>"
                    "Radial velocity = %{y:.4f} km/s"
                    "<br>±1σ including jitter term"
                    "<extra></extra>"
                ),
                showlegend=True
            ),
            row=1, col=1
        )

    # Top: model line
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

    # Bottom: residuals + error bars
    fig.add_trace(
        go.Scatter(
            x=hjds, y=residuals, mode="markers",
            name="Residuals (±1σ, statistical errors)",
            marker=dict(size=8, opacity=0.7),
            error_y=dict(type="data", array=errs, visible=True),
            hovertemplate=(
                "Time (MJD) = %{x:.6f}<br>"
                "Residuals = %{y:.4f} km/s"
                "<br>±1σ statistical uncertainty"
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
                name="Residuals (±1σ, including jitter)",
                marker=dict(size=8, opacity=0.5, color="orange", symbol="circle-open"),
                error_y=dict(
                    type="data",
                    array=errs2,
                    visible=True,
                    color="orange"
                ),
                hovertemplate=(
                    "Time (MJD) = %{x:.6f}<br>"
                    "Residuals = %{y:.4f} km/s"
                    "<br>±1σ including jitter term"
                    "<extra>Residuals</extra>"
                ),
                showlegend=True
            ),
            row=2, col=1
        )

    # Residuals zero line
    fig.add_hline(
        y=0, line_width=1, line_dash="dot", line_color="gray",
        row=2, col=1
    )

    # Axes labels
    fig.update_xaxes(
        title_text="Time (Modified Julian Date)",
        row=2, col=1,
        title_font=dict(size=20),
        tickfont=dict(size=16)
    )
    fig.update_yaxes(
        title_text="Radial velocity [km s⁻¹]",
        row=1, col=1,
        title_font=dict(size=20),
        tickfont=dict(size=16)
    )
    fig.update_yaxes(
        title_text="Residuals [km s⁻¹]",
        row=2, col=1,
        title_font=dict(size=20),
        tickfont=dict(size=16)
    )

    # Layout: global font sizes and legend
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        template="plotly_white",
        title=title,
        title_font=dict(size=22),
        font=dict(size=18),
        legend=dict(
            orientation="h",
            y=-0.15, x=0.5,
            xanchor="center", yanchor="top",
            font=dict(size=16)
        ),
        margin=dict(l=80, r=20, t=80, b=90),
    )

    # Save HTML (standalone)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        safe_star = _sanitize_filename(star_name)
        html_path = os.path.join(out_dir, f"{safe_star}_sid-{solution_id}_time_residuals.html")
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        png_path = os.path.join(out_dir, f"{safe_star}_sid-{solution_id}_time_residuals.png")
        fig.write_image(png_path,scale=4)

    return residuals



# ---------- Phase-folded + residuals ----------
def plot_phase_folded_with_residuals_plotly(
    hjds, vels, errs, P, T0,
    Gamma, K, Omega, ecc,
    star_name, solution_id=0, jitter=0, out_dir=None, plot=True,
    figsize=(1000, 700)  # (width, height) in px
):
    # Compute phases + model curve
    phs_data = ((np.asarray(hjds, dtype=float) - float(T0)) / float(P)) % 1.0
    phase_grid, rv_phase = compute_rv_curve_phase(P, T0, ecc, Gamma, K, Omega)
    phase_grid = np.asarray(phase_grid, dtype=float)
    rv_phase = np.asarray(rv_phase, dtype=float)

    # Interpolate model to data phases for residuals
    vmod_data = np.interp(phs_data, phase_grid, rv_phase)
    vels = np.asarray(vels, dtype=float)
    errs = np.asarray(errs, dtype=float)
    residuals = vels - vmod_data

    if plot:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.06,
            subplot_titles=(
                "Phase-folded radial-velocity curve",
                "Phase-folded residuals"
            )
        )

        # Top: data with errors + model + Gamma
        fig.add_trace(
            go.Scatter(
                x=phs_data, y=vels, mode="markers",
                name="Measurements (±1σ, statistical errors)",
                marker=dict(size=8, opacity=0.7, color="crimson"),
                error_y=dict(type="data", array=errs, visible=True),
                hovertemplate=(
                    "Orbital phase = %{x:.5f}<br>"
                    "Radial velocity = %{y:.4f} km/s"
                    "<br>±1σ statistical uncertainty"
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
                    name="Measurements (±1σ, including jitter)",
                    marker=dict(size=8, opacity=0.5, color="orange", symbol="circle-open"),
                    error_y=dict(
                        type="data",
                        array=errs2,
                        visible=True,
                        color="orange"
                    ),
                    hovertemplate=(
                        "Orbital phase = %{x:.5f}<br>"
                        "Radial velocity = %{y:.4f} km/s"
                        "<br>±1σ including jitter term"
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

        fig.add_hline(
            y=float(Gamma),
            line_dash="dash",
            line_color="gray",
            row=1, col=1
        )

        # Bottom: residuals + errors
        fig.add_trace(
            go.Scatter(
                x=phs_data, y=residuals, mode="markers",
                name="Residuals (±1σ, statistical errors)",
                marker=dict(size=8, opacity=0.8, color="gray"),
                error_y=dict(type="data", array=errs, visible=True),
                hovertemplate=(
                    "Orbital phase = %{x:.5f}<br>"
                    "Residuals = %{y:.4f} km/s"
                    "<br>±1σ statistical uncertainty"
                    "<extra>Residuals</extra>"
                ),
            ),
            row=2, col=1
        )

        if jitter:
            errs2 = np.sqrt(errs ** 2.0 + jitter**2.0)
            fig.add_trace(
                go.Scatter(
                    x=phs_data, y=residuals, mode="markers",
                    name="Residuals (±1σ, including jitter)",
                    marker=dict(size=8, opacity=0.5, color="orange", symbol="circle-open"),
                    error_y=dict(
                        type="data",
                        array=errs2,
                        visible=True,
                        color="orange"
                    ),
                    hovertemplate=(
                        "Orbital phase = %{x:.5f}<br>"
                        "Residuals = %{y:.4f} km/s"
                        "<br>±1σ including jitter term"
                        "<extra>Residuals</extra>"
                    ),
                    showlegend=True
                ),
                row=2, col=1
            )

        fig.add_hline(
            y=0, line_width=1, line_dash="dot", line_color="gray",
            row=2, col=1
        )

        # Axes
        fig.update_xaxes(
            title_text="Orbital phase (cycle fraction)",
            range=[0, 1],
            row=2, col=1,
            title_font=dict(size=20),
            tickfont=dict(size=16)
        )
        fig.update_yaxes(
            title_text="Radial velocity [km s⁻¹]",
            row=1, col=1,
            title_font=dict(size=20),
            tickfont=dict(size=16)
        )
        fig.update_yaxes(
            title_text="Residuals [km s⁻¹]",
            row=2, col=1,
            title_font=dict(size=20),
            tickfont=dict(size=16)
        )

        fig.update_layout(
            width=figsize[0],
            height=figsize[1],
            template="plotly_white",
            title=(
                f"Phase-folded radial-velocity curve for {star_name}: "
                f"P = {P:.2f} d, e = {ecc:.3f}"
            ),
            title_font=dict(size=22),
            font=dict(size=18),
            legend=dict(
                orientation="h",
                y=-0.15, x=0.5,
                xanchor="center", yanchor="top",
                font=dict(size=16)
            ),
            margin=dict(l=80, r=20, t=80, b=90),
        )

        # Save HTML (standalone)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            safe_star = _sanitize_filename(star_name)
            html_path = os.path.join(out_dir, f"{safe_star}_sid-{solution_id}_phase_residuals.html")
            fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
            png_path = os.path.join(out_dir, f"{safe_star}_sid-{solution_id}_phase_residuals.png")
            fig.write_image(png_path,scale=4)

    return phs_data, phase_grid, rv_phase, residuals


def plot_phase_folded_with_residuals(hjds, vels, errs, P, T0,
                                     Gamma, K, Omega, ecc,
                                     star_name, solution_id=0, out_dir=None,plot=True,
                                     figsize=(10,10)):
    phs_data = ((hjds - T0) / P) % 1
    phase_grid, rv_phase = compute_rv_curve_phase(P, T0, ecc, Gamma, K, Omega)
    vmod_data = np.interp(phs_data, phase_grid, rv_phase)
    residuals = vels - vmod_data
    if plot:
        fig, (ax2, ax2r) = plt.subplots(
            2, 1, sharex=True, figsize=figsize,
            gridspec_kw={'height_ratios':[3,1]}
        )
        ax2.errorbar(phs_data, vels, yerr=errs, fmt='o', color='red')
        ax2.plot(phase_grid, rv_phase, color='red', label='Model')
        ax2.axhline(Gamma, color='black', linestyle='--')
        ax2.set_xlim(0,1)
        ax2.set_ylabel('RV [km s$^{-1}$]')
        ax2.set_title(f"Folded fit {star_name}")
        ax2.legend()

        ax2r.errorbar(phs_data, residuals, yerr=errs, fmt='o', color='gray')
        ax2r.axhline(0, color='black', lw=0.8)
        ax2r.set_xlabel('Phase')
        ax2r.set_ylabel('O–C')

        fig.tight_layout()
        if out_dir:
            fig.savefig(os.path.join(out_dir, f"{star_name}_sid-{solution_id}_phase_residuals.png"))
        plt.show()
    return phs_data, phase_grid, rv_phase, residuals


def print_lmfit_result(data, args_dict, star_name, result,solution_id=0, out_dir=None):
    sys.setrecursionlimit(int(1e6))
    hjds, vels, errs = extract_observations(data)
    get_fit_report(result, star_name,solution_id, out_dir)
    Gamma, K, Omega, ecc, P, T0 = compute_orbital_params(result)
    sig_jit = 0
    if LN_SIGMA_JITTER in result.params.keys():
        sig_jit = np.exp(result.params[LN_SIGMA_JITTER].value)
    time_grid, rv_time = compute_rv_curve_time(hjds, P, T0, ecc, Gamma, K, Omega)
    plot_time_series_with_residuals_plotly(
        hjds, vels, errs, time_grid, rv_time,
        Gamma, K, Omega, ecc, P, star_name,solution_id, jitter=sig_jit,
        out_dir=out_dir  # (width, height) in px
    )
    phs_data, phase_grid, rv_phase, residuals = plot_phase_folded_with_residuals(
        hjds, vels, errs, P, T0,
        Gamma, K, Omega, ecc,
        star_name,solution_id, out_dir, plot=False
    )
    plot_phase_folded_with_residuals_plotly(
        hjds, vels, errs, P, T0,
        Gamma, K, Omega, ecc,
        star_name,solution_id,jitter=sig_jit, out_dir=out_dir, plot=True
        # (width, height) in px
    )

    return phs_data
def print_lmfit_result_null(
    data, args_dict, star_name, result, solution_id=0,
    out_dir=None, use_jitter=False
):
    """
    Dump the fit report and produce time-series plots for the null (constant-RV) model.

    - No orbital parameters are derived.
    - No phase-folded plots are produced.
    - Only RV vs. time + residuals are plotted (matplotlib + plotly).
    """
    sys.setrecursionlimit(int(1e6))
    hjds, vels, errs = extract_observations(data)

    Gamma_val = None
    sig_jit = 0.0

    if hasattr(result, "params"):
        # Full lmfit MinimizerResult (e.g. jittered null)
        get_fit_report(result, star_name, solution_id, out_dir)

        if GAMMA in result.params:
            Gamma_val = result.params[GAMMA].value
        else:
            Gamma_val = getattr(result, GAMMA, np.nan)

        if use_jitter and (LN_SIGMA_JITTER in result.params):
            sig_jit = np.exp(result.params[LN_SIGMA_JITTER].value)
    else:
        # SimpleNamespace from the analytic (no-jitter) null fit
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

    # Build simple constant-RV model on a time grid
    time_grid, rv_time = compute_null_rv_curve_time(hjds, Gamma_val)

    # Interactive plotly version (no phase folding)
    plot_time_series_with_residuals_plotly(
        hjds, vels, errs, time_grid, rv_time,
        Gamma_val, 0.0, 0.0, 0.0, None,
        star_name, solution_id, jitter=sig_jit, out_dir=out_dir
    )

def get_rv_weighted_mean(data):
    v1s     = np.array(data[RADIAL_VELS])
    errv1s  = np.abs(data[ERRORS])
    weights = 1.0 / errv1s ** 2
    gamma0 = np.sum(v1s * weights) / np.sum(weights)
    return gamma0

def compute_AIC_BIC(y_obs, errors,chi_sqr, num_params):
    """
    Compute the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

    Parameters:
        y_obs (array): Observed data.
        y_model (array): Model predictions.
        errors (array): Observational errors (standard deviation).
        num_params (int): Number of parameters in the model (k).

    Returns:
        AIC, BIC
    """
    n = len(y_obs)

    log_likelihood_term = np.sum(np.log(2 * np.pi * errors**2))

    # Compute AIC
    AIC = 2 * num_params + chi_sqr + log_likelihood_term

    # Compute BIC
    BIC = num_params * np.log(n) + chi_sqr + log_likelihood_term

    return AIC, BIC

def lmfit_on_sample(args_dict, data, null_hyp=False, use_jitter=False):
    """
    Fit an orbital model to RV vs. MJD data, or—if null_hyp=True—fit
    the null hypothesis of a constant velocity and compute its reduced chi².

    Parameters
    ----------
    args_dict : dict
        Dictionary holding your LMFIT_PARAMS entry, etc.
    output : path or file-like
        Where to dump fit reports.
    data : dict-like
        Must contain keys TIME_STAMPS, RADIAL_VELS, and ERRORS.
    star_name : str, optional
        A label for printing/reporting.
    null_hyp : bool, optional
        If True, skip the orbital fit and instead compute reduced χ² of
        a constant–velocity model (weighted mean). Default is False.

    Returns
    -------
    If null_hyp is False:
        result : lmfit.MinimizerResult
            The full orbital fit result.
    If null_hyp is True:
        dict with keys
            'null_gamma'  : float
                Best-fit constant velocity.
            'null_redchi' : float
                Reduced χ² of the constant model.
                :param use_jitter:
    """
    # ---- style & recursion setup ----
    pylab_params = {
        'legend.fontsize': 'large',
        'figure.figsize': (12, 4),
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large'
    }
    pylab.rcParams.update(pylab_params)
    sys.setrecursionlimit(int(1e6))

    # ---- extract data arrays ----
    hjds1   = np.array(data[TIME_STAMPS])
    v1s     = np.array(data[RADIAL_VELS])
    errv1s  = np.abs(data[ERRORS])
    lmfit_params_dict = args_dict[LMFIT_PARAMS]
    mini_method       = lmfit_params_dict[MINI_METHOD]
    params            = lmfit.Parameters()
    search_params     = lmfit_params_dict[SEARCH_REGION]
    # ---- null hypothesis: constant velocity fit ----
    if null_hyp:
        if use_jitter:
            # Null model WITH jitter: fit gamma and ln_sigma_jitter via lmfit
            params = lmfit.Parameters()
            # initial gamma = weighted mean without jitter
            gamma0 = get_rv_weighted_mean(data)
            params.add(GAMMA, value=gamma0, min=search_params[GAMMA][MIN_VAL],max=search_params[GAMMA][MAX_VAL],vary=True)

            # initial jitter ~ median error (or small fraction thereof)
            # you can tune bounds as you like
            define_search_param(params, search_params, LN_SIGMA_JITTER)
            mini = lmfit.Minimizer(
                null_resid_with_jitter,
                params,
                fcn_kws={TIME_STAMPS: hjds1,
                         RADIAL_VELS: v1s,
                         ERRORS: errv1s}
            )
            result = mini.minimize(method=mini_method, max_nfev=200000)
            return result
        else:
            # weighted mean as best-fit constant velocity
            gamma0   = get_rv_weighted_mean(data)
            # compute total chi²
            chisqr    = np.sum(((v1s - gamma0) / errv1s)**2)
            dof      = len(v1s) - 1   # one fitted parameter (gamma)
            redchi   = chisqr / dof
            aic,bic = compute_AIC_BIC(v1s, errv1s, chisqr, 1)
            # report
            # print(f"[{star_name}] Null hypothesis fit → γ = {gamma0:.5g}, "
            #       f"χ²_red = {redchi:.3f}")
            return SimpleNamespace(GAMMA=gamma0, redchi=redchi, aic=aic, bic=bic, chisqr=chisqr)

    # ---- full orbital fit ----


    # define your five orbital parameters
    define_search_param(params, search_params, PERIOD)
    define_search_param(params, search_params, GAMMA)
    define_search_param(params, search_params, K1_STR)
    define_search_param(params, search_params, OMEGA)
    define_search_param(params, search_params, ECC)
    define_search_param(params, search_params, T)
    if not use_jitter:
        mini = lmfit.Minimizer(
            chisqr_func, params,
            fcn_kws={TIME_STAMPS: hjds1,
                     RADIAL_VELS: v1s,
                     ERRORS: errv1s}
        )
    else:
        define_search_param(params, search_params, LN_SIGMA_JITTER)
        mini = lmfit.Minimizer(
            chisqr_with_jitter, params,
            fcn_kws={TIME_STAMPS: hjds1,
                     RADIAL_VELS: v1s,
                     ERRORS: errv1s}
        )
    result = mini.minimize(method=mini_method, max_nfev=200000)
    return result


def Kepler(E, M, ecc):
    """
    Converts mean anomalies to eccentric anomalies using the Kepler's equation.

    Args:
        E (numpy.ndarray): Initial guess for the eccentric anomalies.
        M (numpy.ndarray): Mean anomalies.
        ecc (float): Eccentricity of the orbit.

    Returns:
        numpy.ndarray: Calculated eccentric anomalies.

    Notes:
        - The function uses an iterative method to solve Kepler's equation: M = E - ecc * sin(E).
        - If the solution does not converge within 990 iterations, the function resets the initial guesses and continues.
        - A convergence message is printed if the solution does not converge after a reset.
        - Convergence is achieved when the change in eccentric anomaly is less than 1E-7.
    """
    counter = 0
    conversion_count = 1
    loop_E = E
    while True:
        if counter > 10000:
            loop_E = np.random.rand(len(M)) * np.pi
            counter = 0
            # print("did not converge {} times".format(conversion_count))
            if conversion_count > 30:
                return np.full(loop_E.shape, np.nan)
            conversion_count += 1
        E2 = (M - ecc * (loop_E * np.cos(loop_E) - np.sin(loop_E))) / (1. - ecc * np.cos(loop_E))
        eps = np.abs(E2 - loop_E)
        if np.all(eps < 1E-7):
            return E2
        else:
            loop_E = E2
            counter += 1


# Given true anomaly nu and parameters, function returns an 2-col array of modeled (Q,U)
def v1mod(nu, gamma1, k1, omega, ecc):
    """
    Calculates the modeled radial velocity (v1) given the true anomaly and orbital parameters.

    Args:
        nu (float or numpy.ndarray): True anomaly.
        gamma1 (float): Systemic velocity.
        k1 (float): Radial velocity semi-amplitude.
        omega (float): Argument of periapsis.
        ecc (float): Orbital eccentricity.

    Returns:
        float or numpy.ndarray: Modeled radial velocity (v1).

    Notes:
        - The function calculates the radial velocity using the formula:
          v1 = gamma1 + k1 * (cos(omega + nu) + ecc * cos(omega)).
        - The input `nu` can be a single value or an array of values.
    """
    v1 = gamma1 + k1 * (np.cos(omega + nu) + ecc * np.cos(omega))
    return v1


def nus1(hjds, P, T0, ecc):
    """
    Calculates the true anomalies (nu) from heliocentric Julian dates (hjds) and orbital parameters.

    Args:
        hjds (numpy.ndarray): Array of heliocentric Julian dates.
        P (float): Orbital period.
        T0 (float): Time of periastron passage.
        ecc (float): Orbital eccentricity.

    Returns:
        numpy.ndarray: Array of true anomalies (nu).

    Notes:
        - The function calculates the orbital phase (phis) from the input dates.
        - Mean anomalies (Ms) are derived from the phases.
        - Eccentric anomalies (Es) are obtained using the Kepler function.
        - True anomalies (nusdata) are calculated from the eccentric anomalies.
    """
    phis = (hjds - T0) / P - ((hjds - T0) / P).astype(int)
    # phis[phis < 0] = phis[phis < 0] + 1.
    Ms = 2 * np.pi * phis
    Es = Kepler(np.pi, Ms, ecc)
    eccfac = np.sqrt((1 + ecc) / (1 - ecc))
    nusdata = 2. * np.arctan(eccfac * np.tan(0.5 * Es))
    return nusdata


# Target function (returns differences between model array with parameter set p and data)
def chisqr_func(p, **kws):
    """
    Computes the chi-squared values, which represent the differences between the modeled and observed radial velocities.

    Args:
        p (Parameters): A parameters object containing the model parameters. It should include:
            - GAMMA: Systemic velocity parameter.
            - K1_STR: Radial velocity semi-amplitude parameter.
            - OMEGA: Argument of periapsis parameter.
            - ECC: Eccentricity parameter.
            - T: Time of periastron passage parameter.
            - PERIOD: Orbital period parameter.
        **kws: Additional keyword arguments containing the data and error arrays:
            - TIME_STAMPS (numpy.ndarray): Array of observation time stamps.
            - RADIAL_VELS (numpy.ndarray): Array of observed radial velocities.
            - ERRORS (numpy.ndarray): Array of errors in the observed radial velocities.

    Returns:
        numpy.ndarray: Array of normalized differences between the modeled and observed radial velocities.

    Notes:
        - The function extracts the necessary parameters from the `p` object.
        - It calculates the modeled radial velocities using the `v1mod` function and the true anomalies obtained from the `nus1` function.
        - The differences between the modeled and observed radial velocities are normalized by the observational errors.
    """
    Gamma1 = p[GAMMA].value
    K1 = p[K1_STR].value
    Omega = p[OMEGA].value
    ecc = p[ECC].value
    T0 = p[T].value
    P = p[PERIOD].value
    v1 = v1mod(nus1(kws[TIME_STAMPS], P, T0, ecc), Gamma1, K1, Omega, ecc)
    return (v1 - kws[RADIAL_VELS]) / kws[ERRORS]


def null_resid_with_jitter(params, **kws):
    gamma = params[GAMMA].value
    ln_sj = params[LN_SIGMA_JITTER].value
    sj = np.exp(ln_sj)
    sig2 = kws[ERRORS] ** 2 + sj ** 2
    # sig2 = np.maximum(sig2, 1e-30)

    # sigma_eff = np.sqrt(sig2)
    # res_chi = (kws[RADIAL_VELS] - gamma) / sigma_eff
    # res_log = np.sqrt(np.log(sig2))  # match chisqr_with_jitter
    # res = np.concatenate([res_chi, res_log])

    res_chi_sqrd = (kws[RADIAL_VELS] - gamma)* (kws[RADIAL_VELS] - gamma)/ sig2
    res_ln = np.log(sig2)
    res = res_ln.sum() + res_chi_sqrd.sum()
    return float(res)

    # return _finite_or_default(res, default=1e6)


# Target function (returns differences between model array with parameter set p and data)
def chisqr_with_jitter(p, **kws):
    """
    Computes the chi-squared values, which represent the differences between the modeled and observed radial velocities.

    Args:
        p (Parameters): A parameters object containing the model parameters. It should include:
            - GAMMA: Systemic velocity parameter.
            - K1_STR: Radial velocity semi-amplitude parameter.
            - OMEGA: Argument of periapsis parameter.
            - ECC: Eccentricity parameter.
            - T: Time of periastron passage parameter.
            - PERIOD: Orbital period parameter.
        **kws: Additional keyword arguments containing the data and error arrays:
            - TIME_STAMPS (numpy.ndarray): Array of observation time stamps.
            - RADIAL_VELS (numpy.ndarray): Array of observed radial velocities.
            - ERRORS (numpy.ndarray): Array of errors in the observed radial velocities.

    Returns:
        numpy.ndarray: Array of normalized differences between the modeled and observed radial velocities.

    Notes:
        - The function extracts the necessary parameters from the `p` object.
        - It calculates the modeled radial velocities using the `v1mod` function and the true anomalies obtained from the `nus1` function.
        - The differences between the modeled and observed radial velocities are normalized by the observational errors.
    """
    Gamma1 = float(p[GAMMA].value)
    K1     = float(p[K1_STR].value)
    Omega  = float(p[OMEGA].value)
    ecc    = float(p[ECC].value)
    T0     = float(p[T].value)
    P      = float(p[PERIOD].value)
    sigmaJ = float(np.exp(p[LN_SIGMA_JITTER].value))

    hjd = np.asarray(kws[TIME_STAMPS], float)
    rv  = np.asarray(kws[RADIAL_VELS], float)
    err = np.asarray(kws[ERRORS], float)

    if not (np.isfinite(P) and P > 0 and np.isfinite(ecc) and 0 <= ecc < 1 and np.isfinite(sigmaJ) and sigmaJ >= 0):
        return 2e12
    try:
        nu  = nus1(hjd, P, T0, ecc)
        v1  = v1mod(nu, Gamma1, K1, Omega, ecc)
        sig2 = err**2 + sigmaJ**2
        res_chi_sqrd = (v1 - rv)* (v1 - rv) / sig2
        res_ln = np.log(sig2)
        res = res_ln.sum() + res_chi_sqrd.sum()
    except Exception:
        res = 2e12
    return float(res)

def dump_minimization_report(res, output, file_prefix):
    lines = []
    best_fit_params = res.params
    lines.append("Best-fit parameter values:\n")
    for name, param in best_fit_params.items():
        lines.append(f"{name}: {param.value}\n")

    # Minimized value of the objective function
    minimized_value = res.chisqr
    lines.append(f"Minimized chi-square value: {minimized_value}\n")

    # Reduced chi-square value
    reduced_chi_square = res.redchi
    lines.append(f"Reduced chi-square value: {reduced_chi_square}\n")

    # Success of the optimization
    success = res.success
    lines.append(f"Optimization success: {success}\n")

    # Fit statistics
    aic = res.aic
    bic = res.bic
    lines.append(f"AIC: {aic}, BIC: {bic}\n")

    # Covariance matrix
    covariance_matrix = res.covar
    lines.append(f"Covariance matrix: \n{covariance_matrix}\n")

    # Detailed fit report
    lines.append("Fit report:\n")
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        lmfit.report_fit(res)
    lines.append(report.getvalue())

    with open(os.path.join(output, "{}_fit_report.txt".format(file_prefix)), "w") as file:
        file.writelines(lines)



def corner_plot(args_dict, data, mini_results,output, truths=None):
    """
    Generates a corner plot for the given data and fit results using the specified parameters.

    Args:
        args_dict (dict): A dictionary containing the corner plot parameters. It should have the following structure:
            - CORNER_PARAMS: A dictionary containing parameters for the corner plot and minimization process:
                - LN_SIGMA: A dictionary with the following keys:
                    - INIT_VAL (float): Initial value for the log of sigma.
                    - MIN_VAL (float): Minimum value for the log of sigma.
                    - MAX_VAL (float): Maximum value for the log of sigma.
                - CORNER_METHOD (str): The minimization method to be used.
                - NAN_POLICY (str): Policy for handling NaN values.
                - BURN (int): Number of burn-in steps.
                - STEPS (int): Number of MCMC steps.
                - THIN (int): Thinning factor for MCMC sampling.
        data (dict): A dictionary containing the data for the minimization process.
        mini_results (MinimizerResult): The initial minimization results object.
        truths (dict, optional): A dictionary containing the true values of the parameters for reference in the corner plot. Default is None.

    Returns:
        None: The function generates and displays a corner plot.

    Notes:
        - The function first adds the log of sigma parameter to the minimization results.
        - It performs the minimization using the `lmfit.minimize` function.
        - The resulting chain of parameter samples is used to generate a corner plot.
        - If `truths` is provided, it will be used to indicate the true parameter values on the corner plot.
    """
    corner_params = args_dict[CORNER_PARAMS]
    corner_params_obj = mini_results.params.copy()

    corner_params_obj.add(LN_SIGMA, value=np.log(corner_params[LN_SIGMA][INIT_VAL]),
                            min=np.log(corner_params[LN_SIGMA][MIN_VAL]),
                            max=np.log(corner_params[LN_SIGMA][MAX_VAL]))

    # Set the specific parameter's vary to True
    corner_params_obj[PERIOD].set(vary=True)
    corner_params_obj[PERIOD].stderr = corner_params_obj[PERIOD].value * 0.05
    # for name, p in corner_params_obj.items():
    #     if not p.vary:
    #         p.set(vary=True)
    #     # Expand bounds around best value using stderr (3σ as an example)
    #     if p.stderr is not None and p.stderr > 0:
    #         p_min = p.value - 3 * p.stderr
    #         p_max = p.value + 3 * p.stderr
    #         p.set(min=p_min, max=p_max)
    #     elif name != ECC:
    #         # If no stderr is available, provide a fallback region
    #         p.set(min=p.value *0.95, max=p.value *1.05)
    # for name, p in corner_params_obj.items():
    #     print(f"{name}: value={p.value:.4f}, stderr={p.stderr}, bounds=({p.min}, {p.max})")

    # Then use this updated Parameters object
    res = lmfit.minimize(
        chisqr_func,
        kws=data,
        method=corner_params[CONRER_METHOD],
        nan_policy=corner_params[NAN_POLICY],
        burn=corner_params[BURN],
        steps=corner_params[STEPS],
        thin=corner_params[THIN],
        params=corner_params_obj,
        is_weighted=True,
        progress=True
    )


    dump_minimization_report(res, output, "corner")

    prediction = [res.params[PERIOD].value,
                  res.params[GAMMA].value,
                  res.params[K1_STR].value,
                  res.params[OMEGA].value,
                  res.params[ECC].value,
                  res.params[T].value,
                  0]

    figure = corner.corner(res.flatchain, truths=prediction, truth_color='red', labels=res.var_names)
    best_fit_params = res.params.copy()
    best_fit_list = []
    # Get the flat MCMC chain
    chain = res.flatchain  # shape: (n_samples, n_parameters)
    var_names = res.var_names

    # Loop through parameters and update with median and stderr
    for name in var_names:
        samples = chain[name]  # samples for this parameter
        median = np.median(samples)
        stderr = np.std(samples, ddof=1)
        # best_fit_params[name].set(value=median, stderr=stderr)
        best_fit_list.append(best_fit_params[name].value)
    # Add the second set of truths manually
    num_of_params = len(res.var_names)
    axes = np.array(figure.axes).reshape((num_of_params, num_of_params))
    for i in range(num_of_params):
        for j in range(i + 1):
            ax = axes[i, j]
            ax.axvline(best_fit_list[j], color='blue', linestyle='--')
            if i != j:
                ax.axhline(best_fit_list[i], color='blue', linestyle='--')
                ax.plot(best_fit_list[j], best_fit_list[i], 'bo')

    # Add the second set of truths manually
    if truths:
        num_of_params = len(res.var_names)
        axes = np.array(figure.axes).reshape((num_of_params, num_of_params))
        for i in range(num_of_params):
            for j in range(i + 1):
                ax = axes[i, j]
                ax.axvline(truths[j], color='blue', linestyle='--')
                if i != j:
                    ax.axhline(truths[i], color='blue', linestyle='--')
                    ax.plot(truths[j], truths[i], 'bo')

    # Add a title to the corner plot
    figure.suptitle("MCMC corner plot\nred cross is prediction\nblue cross Are the Truths")

    # Show the plot
    plt.show()


