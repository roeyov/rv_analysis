"""
simulations.bias_grid_explorer — Streamlit GUI for exploring bias grid results.

Usage:
    streamlit run simulations/bias_grid_explorer.py -- --output-dir /path/to/results
"""

import os
import sys
import argparse

# Ensure the project root is on sys.path so 'simulations.*' imports work
# when launched via `streamlit run simulations/bias_grid_explorer.py`
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulations.bias_config import DEFAULT_BIAS_CFG
from simulations.bias_grid import (
    load_observed_from_tex, _HIST_BINS, _HIST_PAIRS, _HIST_NBINS,
    _DET_SHARDS_DIR, _load_det_shard,
)


# ---------------------------------------------------------------------------
# CLI argument parsing (Streamlit passes args after --)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sb1-tex", default=None)
    parser.add_argument("--sb2-tex", default=None)
    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_grid_data(output_dir):
    """Load grid_cubes.npz and optionally grid_detected.npz."""
    # Try grid_cubes.npz first, fall back to checkpoint_cubes.npz
    cubes_path = os.path.join(output_dir, "grid_cubes.npz")
    if not os.path.exists(cubes_path):
        cubes_path = os.path.join(output_dir, "checkpoint_cubes.npz")
    cubes = np.load(cubes_path, allow_pickle=True)

    data = {
        "pdet_cube": cubes["pdet_cube"],
        "pi_grid": cubes["pi_grid"],
        "kappa_grid": cubes["kappa_grid"],
        "eta_grid": cubes["eta_grid"],
        "fbin_grid": cubes["fbin_grid"],
    }

    # Load per-test GMF and p-value cubes
    test_names = ["ks", "ad", "cvm"]
    available_tests = []
    gmf_cubes = {}
    test_pval_cubes = {}
    best_fits = {}

    for tname in test_names:
        gmf_key = "gmf_%s_cube" % tname
        bf_key = "best_fit_%s" % tname
        if gmf_key in cubes.files:
            gmf_cubes[tname] = cubes[gmf_key]
            available_tests.append(tname)
            test_pval_cubes[tname] = {}
            for par in ("logP", "e", "K1", "e_circ"):
                tc_key = "%s_%s_cube" % (tname, par)
                if tc_key in cubes.files:
                    test_pval_cubes[tname][par] = cubes[tc_key]
            # Always recompute best fit from the loaded cube — stored
            # values may be stale from an incomplete checkpoint.
            gc = gmf_cubes[tname]
            idx = np.unravel_index(np.nanargmax(gc), gc.shape)
            best_fits[tname] = (
                float(data["pi_grid"][idx[0]]),
                float(data["kappa_grid"][idx[1]]),
                float(data["eta_grid"][idx[2]]),
                float(data["fbin_grid"][idx[3]]),
            )

    # Backward compat: if no per-test cubes, use legacy KS keys
    if "ks" not in gmf_cubes and "gmf_cube" in cubes.files:
        gmf_cubes["ks"] = cubes["gmf_cube"]
        available_tests.append("ks")
        test_pval_cubes["ks"] = {}
        for par, old_key in [("logP", "ks_logP_cube"),
                             ("e", "ks_e_cube"),
                             ("K1", "ks_K1_cube")]:
            if old_key in cubes.files:
                test_pval_cubes["ks"][par] = cubes[old_key]
        gc = gmf_cubes["ks"]
        idx = np.unravel_index(np.nanargmax(gc), gc.shape)
        best_fits["ks"] = (
            float(data["pi_grid"][idx[0]]),
            float(data["kappa_grid"][idx[1]]),
            float(data["eta_grid"][idx[2]]),
            float(data["fbin_grid"][idx[3]]),
        )

    data["available_tests"] = available_tests
    data["gmf_cubes"] = gmf_cubes
    data["test_pval_cubes"] = test_pval_cubes
    data["best_fits"] = best_fits

    # Recover the eccentricity scoring mode used to produce these cubes.
    # Modern files carry an explicit "e_score_mode" entry; legacy files
    # are inferred from the presence of *_e_circ_cube (split mode) vs.
    # absence (combined mode — the only other historical option).
    if "e_score_mode" in cubes.files:
        data["e_score_mode"] = str(cubes["e_score_mode"])
    else:
        has_e_circ = any("e_circ" in test_pval_cubes.get(t, {})
                         for t in test_pval_cubes)
        data["e_score_mode"] = "split" if has_e_circ else "combined"

    # Default gmf_cube / best_fit (KS for backward compat)
    default_test = available_tests[0] if available_tests else "ks"
    data["gmf_cube"] = gmf_cubes.get(default_test, cubes.get("gmf_cube"))
    data["best_fit"] = best_fits.get(default_test)
    # Legacy p-value cube keys (used by sidebar stats)
    data["ks_logP_cube"] = test_pval_cubes.get("ks", {}).get(
        "logP", np.zeros_like(data["pdet_cube"]))
    data["ks_e_cube"] = test_pval_cubes.get("ks", {}).get(
        "e", np.zeros_like(data["pdet_cube"]))
    data["ks_K1_cube"] = test_pval_cubes.get("ks", {}).get(
        "K1", np.zeros_like(data["pdet_cube"]))

    # --- Detect which storage format is available ---
    shard_dir = os.path.join(output_dir, _DET_SHARDS_DIR)
    det_index_path = os.path.join(output_dir, "det_index.npz")
    hists_path = os.path.join(output_dir, "grid_hists.npz")
    if not os.path.exists(hists_path):
        hists_path = os.path.join(output_dir, "checkpoint_hists.npz")

    # Legacy concatenated format
    det_legacy_path = os.path.join(output_dir, "grid_detected.npz")
    if not os.path.exists(det_legacy_path):
        det_legacy_path = os.path.join(output_dir, "checkpoint_detected.npz")

    if os.path.isdir(shard_dir) and os.path.exists(det_index_path):
        # New shard-based format: detected arrays loaded on demand
        idx_data = np.load(det_index_path)
        data["has_detected"] = True
        data["det_format"] = "shards"
        data["det_output_dir"] = output_dir
        data["step_to_ijkl"] = idx_data["step_to_ijkl"]

        # Load histograms from separate file
        data["hist_total"] = {}
        data["hist_det"] = {}
        if os.path.exists(hists_path):
            hdata = np.load(hists_path)
            for a, b in _HIST_PAIRS:
                key_t = "hist_total_%s_%s" % (a, b)
                key_d = "hist_det_%s_%s" % (a, b)
                if key_t in hdata.files:
                    data["hist_total"][(a, b)] = hdata[key_t]
                    data["hist_det"][(a, b)] = hdata[key_d]

    elif os.path.exists(det_legacy_path):
        # Legacy concatenated format
        det = np.load(det_legacy_path, allow_pickle=True)
        data["has_detected"] = True
        data["det_format"] = "legacy"
        data["det_logP"] = det["all_logP_det"]
        data["det_e"] = det["all_e_det"]
        data["det_K1"] = det["all_K1_det"]
        data["det_q"] = det["all_q_det"]
        data["det_offsets"] = det["offsets"]
        data["step_to_ijkl"] = det["step_to_ijkl"]
        data["hist_total"] = {}
        data["hist_det"] = {}
        for a, b in _HIST_PAIRS:
            key_t = "hist_total_%s_%s" % (a, b)
            key_d = "hist_det_%s_%s" % (a, b)
            if key_t in det.files:
                data["hist_total"][(a, b)] = det[key_t]
                data["hist_det"][(a, b)] = det[key_d]
    else:
        data["has_detected"] = False

    return data


@st.cache_data
def load_observed(sb1_tex, sb2_tex):
    return load_observed_from_tex(sb1_tex, sb2_tex)


# ---------------------------------------------------------------------------
# Posterior computation helpers
# ---------------------------------------------------------------------------

def gmf_to_prob(gmf_cube):
    """Convert log-GMF cube to normalized probability."""
    shifted = gmf_cube - np.nanmax(gmf_cube)
    prob = np.exp(shifted)
    return np.nan_to_num(prob, nan=0.0)


def get_1d_posterior(prob, grids, axis_idx):
    """Marginalize 4D probability to 1D for axis_idx."""
    axes_to_sum = tuple(i for i in range(4) if i != axis_idx)
    post = np.nansum(prob, axis=axes_to_sum)
    grid = grids[axis_idx]
    if np.sum(post) > 0 and len(grid) > 1:
        post = post / np.trapz(post, grid)
    return post


def get_2d_marginal(prob, i1, i2):
    """Marginalize 4D probability to 2D for axes (i1, i2)."""
    axes_to_sum = tuple(i for i in range(4) if i not in (i1, i2))
    return np.nansum(prob, axis=axes_to_sum)


def get_detected_for_point(data, i, j, k, l):
    """Get detected arrays for grid point (i,j,k,l).

    Supports both shard-based (new) and legacy concatenated formats.
    """
    if not data["has_detected"]:
        return None

    ijkl = np.array([i, j, k, l])

    if data.get("det_format") == "shards":
        # New format: find the step number, load its shard file on demand
        step_to_ijkl = data["step_to_ijkl"]
        matches = np.where(
            (step_to_ijkl[:, 1:] == ijkl).all(axis=1)
        )[0]
        if len(matches) == 0:
            return None
        step = int(step_to_ijkl[matches[0], 0])
        return _load_det_shard(data["det_output_dir"], step)

    else:
        # Legacy format: offset slicing into concatenated arrays
        matches = np.where(
            (data["step_to_ijkl"] == ijkl).all(axis=1)
        )[0]
        if len(matches) == 0:
            return None
        idx = matches[0]
        lo = int(data["det_offsets"][idx])
        hi = int(data["det_offsets"][idx + 1])
        return {
            "logP": data["det_logP"][lo:hi],
            "e": data["det_e"][lo:hi],
            "K1": data["det_K1"][lo:hi],
            "q": data["det_q"][lo:hi],
        }


def powerlaw_cdf(x, alpha, xmin, xmax):
    """Analytical CDF of p(x) ~ x^alpha on [xmin, xmax]."""
    a = alpha + 1.0
    if abs(a) < 1e-8:
        return np.log(x / xmin) / np.log(xmax / xmin)
    return (x**a - xmin**a) / (xmax**a - xmin**a)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

PARAM_NAMES = ["\u03c0", "\u03ba", "\u03b7", "f_bin"]
PARAM_LABELS = ["pi", "kappa", "eta", "f_bin"]
REF_VALS = [-0.55, -0.10, -0.45, 0.69]  # Sana+2012 Galactic
COLORS_1D = ["#4393c3", "#d6604d", "#5aae61", "#9970ab"]


def plot_corner(prob, grids, best_fit, current_vals, smooth=True):
    """Create a corner plot: 1D marginals on the diagonal, 2D marginals
    on the lower triangle. Upper triangle is hidden."""
    n = len(grids)
    fig = make_subplots(
        rows=n, cols=n,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.04, vertical_spacing=0.04,
    )

    first_heatmap = True
    legend_used = False

    for row_idx in range(n):
        for col_idx in range(n):
            row = row_idx + 1
            col = col_idx + 1

            if row_idx == col_idx:
                # Diagonal: 1D marginal posterior
                idx = row_idx
                grid = grids[idx]
                post = get_1d_posterior(prob, grids, idx)
                color = COLORS_1D[idx]

                # Density fill
                fig.add_trace(go.Scatter(
                    x=grid, y=post, mode="lines", fill="tozeroy",
                    line=dict(color=color, width=2),
                    fillcolor="rgba(%d,%d,%d,0.2)" % (
                        int(color[1:3], 16),
                        int(color[3:5], 16),
                        int(color[5:7], 16)),
                    name=PARAM_NAMES[idx], showlegend=False,
                ), row=row, col=col)

                # 68% CI band
                cdf = np.cumsum(post)
                if cdf[-1] > 0:
                    cdf = cdf / cdf[-1]
                    lo = np.interp(0.16, cdf, grid)
                    hi = np.interp(0.84, cdf, grid)
                    fig.add_vrect(x0=lo, x1=hi, fillcolor="crimson",
                                  opacity=0.1, line_width=0,
                                  row=row, col=col)

                # Best-fit star
                fig.add_trace(go.Scatter(
                    x=[best_fit[idx]],
                    y=[np.interp(best_fit[idx], grid, post)],
                    mode="markers",
                    marker=dict(symbol="star", size=14, color="gold",
                                line=dict(width=1, color="black")),
                    name="Best fit", showlegend=(not legend_used),
                ), row=row, col=col)

                # Current slider position circle
                fig.add_trace(go.Scatter(
                    x=[current_vals[idx]],
                    y=[np.interp(current_vals[idx], grid, post)],
                    mode="markers",
                    marker=dict(symbol="circle", size=10, color="red",
                                line=dict(width=1, color="black")),
                    name="Current", showlegend=(not legend_used),
                ), row=row, col=col)
                legend_used = True

                # Sana+2012 reference line + annotation
                fig.add_vline(x=REF_VALS[idx], line_dash="dot",
                              line_color="grey", row=row, col=col)
                axis_num = (row - 1) * n + col
                fig.add_annotation(
                    text="Sana+12: %s=%.2f" % (PARAM_NAMES[idx],
                                               REF_VALS[idx]),
                    x=REF_VALS[idx], y=1.0,
                    xref="x" if axis_num == 1 else "x%d" % axis_num,
                    yref=("y domain" if axis_num == 1
                          else "y%d domain" % axis_num),
                    xanchor="left", yanchor="top",
                    showarrow=False, font=dict(size=9, color="grey"),
                    row=row, col=col,
                )

            elif row_idx > col_idx:
                # Lower triangle: 2D marginal heatmap
                # x-axis = grids[col_idx], y-axis = grids[row_idx]
                marg = get_2d_marginal(prob, col_idx, row_idx)
                gx, gy = grids[col_idx], grids[row_idx]

                fig.add_trace(go.Heatmap(
                    z=marg.T, x=gx, y=gy,
                    colorscale="RdYlBu_r",
                    zsmooth="best" if smooth else False,
                    showscale=first_heatmap,
                    colorbar=dict(title="Marginal prob")
                             if first_heatmap else None,
                ), row=row, col=col)
                first_heatmap = False

                # Best-fit star
                fig.add_trace(go.Scatter(
                    x=[best_fit[col_idx]], y=[best_fit[row_idx]],
                    mode="markers",
                    marker=dict(symbol="star", size=14, color="gold",
                                line=dict(width=1.5, color="black")),
                    showlegend=False,
                ), row=row, col=col)

                # Current position circle
                fig.add_trace(go.Scatter(
                    x=[current_vals[col_idx]], y=[current_vals[row_idx]],
                    mode="markers",
                    marker=dict(symbol="circle", size=10, color="red",
                                line=dict(width=1.5, color="black")),
                    showlegend=False,
                ), row=row, col=col)

            else:
                # Upper triangle: hide axes
                fig.update_xaxes(visible=False, row=row, col=col)
                fig.update_yaxes(visible=False, row=row, col=col)
                continue

            # Axis titles: only bottom row gets x-titles,
            # only leftmost column gets y-titles.
            if row == n:
                fig.update_xaxes(title_text=PARAM_NAMES[col_idx],
                                 row=row, col=col)
            else:
                fig.update_xaxes(showticklabels=False, row=row, col=col)

            if col == 1:
                if row_idx == col_idx:
                    y_title = "Density" if row_idx == 0 else ""
                    fig.update_yaxes(title_text=y_title,
                                     showticklabels=(row_idx == 0),
                                     row=row, col=col)
                else:
                    fig.update_yaxes(title_text=PARAM_NAMES[row_idx],
                                     row=row, col=col)
            else:
                fig.update_yaxes(showticklabels=False, row=row, col=col)

    fig.update_layout(height=750, width=750, margin=dict(t=40, b=40))
    return fig


def plot_cdfs(det_params, obs, current_vals, ks_pvals, test_label="KS",
              use_empirical=False, p_det=None, e_score_mode="combined"):
    """Plot CDF + PDF comparison: observed vs synthetic detected vs intrinsic.

    Layout: 2 rows x 4 cols.
      Row 1: CDFs  for logP, e, K1, q
      Row 2: PDFs  for logP, e, K1, q
    Observed PDF/CDF shown for logP, e, K1 (not q).
    Intrinsic + sim. detected shown for all four.

    The detected/observed CDFs are scaled by the detection completeness so
    the intrinsic CDF reaches 1.0 and the gap shows undetected systems.

    If *use_empirical* is True, the intrinsic distribution is drawn from the
    full simulated population (det + nondet) instead of the analytical
    power-law CDF/PDF.
    """
    cfg = DEFAULT_BIAS_CFG
    pi_val, kappa_val, eta_val, fbin_val = current_vals

    # Build empirical "all injected" arrays when requested
    empirical_all = {}
    if use_empirical and det_params is not None:
        for key in ("logP", "e", "K1", "q"):
            nondet_key = "%s_nondet" % key
            det_arr = det_params.get(key, np.array([]))
            nondet_arr = det_params.get(nondet_key, np.array([]))
            if len(det_arr) or len(nondet_arr):
                empirical_all[key] = np.concatenate([det_arr, nondet_arr])

    # Only eccentric_only ignores circular systems entirely. 'split' still
    # scores them via the separate circular-fraction binomial, so they
    # belong in the CDF visualization.
    obs_e_arr = obs.get("e")
    if e_score_mode == "eccentric_only" and obs_e_arr is not None:
        obs_e_arr = np.asarray(obs_e_arr)
        obs_e_arr = obs_e_arr[obs_e_arr > 0]
        if "e" in empirical_all:
            empirical_all["e"] = empirical_all["e"][empirical_all["e"] > 0]

    # (key, obs_array_or_None, xlabel, color, alpha, xmin, xmax, n_bins)
    param_configs = [
        ("logP", obs["logP"], "log\u2081\u2080(P/d)", "#4393c3",
         pi_val, cfg["log_p_min"], cfg["log_p_max"], 15),
        ("e", obs_e_arr, "e", "#d6604d",
         eta_val, 1e-6, cfg["e_max"], 15),
        ("K1", obs["K1"], "K\u2081 [km/s]", "#5aae61",
         None, None, None, 15),
        ("q", None, "q", "#8073ac",
         kappa_val, cfg["q_min"], cfg["q_max"], 12),
    ]

    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=["log P", "Eccentricity", "K\u2081", "q",
                        "", "", "", "",
                        "", "", "", ""],
        row_heights=[0.4, 0.35, 0.25],
        vertical_spacing=0.10,
    )

    for col_idx, (key, obs_arr, xlabel, color, alpha, xmin, xmax, n_bins) \
            in enumerate(param_configs):
        col = col_idx + 1
        has_obs = obs_arr is not None and len(obs_arr) >= 2

        # --- Determine clipping range ---
        if has_obs:
            obs_s = np.sort(obs_arr)
        else:
            obs_s = None

        if key == "e":
            # Only eccentric_only mode excludes the circular spike from
            # sim/empirical CDFs and histograms.
            clip_lo = 1e-12 if e_score_mode == "eccentric_only" else 0.0
            clip_hi = 1.0
        elif has_obs:
            clip_lo, clip_hi = 0.0, float(obs_s[-1])
        else:
            clip_lo = xmin if xmin is not None else 0.0
            clip_hi = xmax if xmax is not None else 1.0

        # ===================== Row 1: CDF =====================

        # Completeness scaling so intrinsic CDF → [0, 1] and
        # detected/observed → [0, completeness].
        # Use empirical ratio when available, otherwise fall back to p_det.
        # For the e marginal in eccentric_only mode, restrict the ratio to
        # the e>0 subset so the displayed CDF can reach 1.0.
        completeness = 1.0
        if use_empirical and key in empirical_all:
            all_vals = empirical_all[key]
            n_all = len(all_vals)
            if n_all > 0 and det_params is not None:
                det_vals = det_params.get(key, np.array([]))
                if key == "e" and e_score_mode == "eccentric_only":
                    all_vals_subset = all_vals[all_vals > 0]
                    det_vals_subset = (
                        np.asarray(det_vals)[np.asarray(det_vals) > 0]
                        if len(det_vals) else det_vals
                    )
                    n_all_sub = len(all_vals_subset)
                    n_det_sub = len(det_vals_subset)
                    if n_all_sub > 0:
                        completeness = n_det_sub / n_all_sub
                else:
                    completeness = len(det_vals) / n_all
        elif p_det is not None and p_det < 1.0:
            completeness = p_det

        # Observed CDF
        if has_obs:
            obs_cdf = np.arange(1, len(obs_s) + 1) / len(obs_s)
            fig.add_trace(go.Scatter(
                x=obs_s, y=obs_cdf * completeness, mode="lines",
                line=dict(color="black", width=2, shape="hv"),
                name="Observed (n=%d)" % len(obs_s),
                showlegend=(col_idx == 0),
                legendgroup="obs",
            ), row=1, col=col)

        # Synthetic detected CDF (clipped to observed range)
        if det_params is not None and len(det_params.get(key, [])) >= 2:
            sim_raw = det_params[key]
            sim_raw = sim_raw[(sim_raw >= clip_lo) & (sim_raw <= clip_hi)]
            sim = np.sort(sim_raw)
            if len(sim) >= 2:
                sim_cdf = np.arange(1, len(sim) + 1) / len(sim)
                fig.add_trace(go.Scatter(
                    x=sim, y=sim_cdf * completeness, mode="lines",
                    line=dict(color=color, width=2, dash="dash", shape="hv"),
                    name="Sim. detected (n=%d)" % len(sim),
                    showlegend=(col_idx == 0),
                    legendgroup="sim",
                ), row=1, col=col)

        # Intrinsic CDF — empirical (det+nondet) or analytical power-law
        # Clip to same range as detected so CDFs are comparable.
        if use_empirical and key in empirical_all and len(empirical_all[key]) >= 2:
            all_vals = empirical_all[key]
            all_vals = all_vals[(all_vals >= clip_lo) & (all_vals <= clip_hi)]
            all_s = np.sort(all_vals)
            if len(all_s) >= 2:
                all_cdf = np.arange(1, len(all_s) + 1) / len(all_s)
                fig.add_trace(go.Scatter(
                    x=all_s, y=all_cdf, mode="lines",
                    line=dict(color="gray", width=2, dash="dot", shape="hv"),
                    name="Sim. all (n=%d)" % len(all_s),
                    showlegend=(col_idx == 0),
                    legendgroup="intr",
                ), row=1, col=col)
        elif alpha is not None and xmin is not None:
            # Analytical power-law CDF
            x_hi_cdf = clip_hi if key == "logP" and has_obs else xmax
            x_intr = np.linspace(xmin, x_hi_cdf, 200)
            y_intr = powerlaw_cdf(x_intr, alpha, xmin, x_hi_cdf)
            fig.add_trace(go.Scatter(
                x=x_intr, y=y_intr, mode="lines",
                line=dict(color="gray", width=2, dash="dot"),
                name="Intrinsic",
                showlegend=(col_idx == 0),
                legendgroup="intr",
            ), row=1, col=col)

        # p-value annotation
        ks_key = {"logP": "ks_logP", "e": "ks_e",
                  "K1": "ks_K1", "q": "ks_q"}.get(key)
        pval = ks_pvals.get(ks_key, np.nan) if ks_key else np.nan
        if not np.isnan(pval):
            xref = "x domain" if col == 1 else "x%d domain" % col
            yref = "y domain" if col == 1 else "y%d domain" % col
            if key == "e" and e_score_mode == "split":
                p_e_circ = ks_pvals.get("ks_e_circ", np.nan)
                ann_text = "p(e>0)=%.3f\np(f_circ)=%.3f" % (pval, p_e_circ)
            elif key == "e" and e_score_mode == "eccentric_only":
                ann_text = "p(e>0) = %.3f" % pval
            else:
                ann_text = "%s p = %.3f" % (test_label, pval)
            fig.add_annotation(
                text=ann_text,
                x=0.05, y=0.95, xref=xref, yref=yref,
                showarrow=False, font=dict(size=11),
                bgcolor="wheat", opacity=0.8,
            )

        fig.update_xaxes(title_text="", row=1, col=col)
        fig.update_yaxes(
            title_text="CDF" if col == 1 else "", row=1, col=col,
            range=[0, 1.05],
        )

        # ===================== Row 2: PDF (histogram) =====================

        bin_lo = clip_lo if has_obs else xmin
        bin_hi = clip_hi if has_obs else xmax
        bins = np.linspace(bin_lo, bin_hi, n_bins + 1)
        bw = float(bins[1] - bins[0])
        pdf_peak = 0.0  # track tallest bar for intrinsic scaling
        density_sim_arr = None
        density_all_arr = None

        # Observed PDF (scaled by completeness so gap vs intrinsic is visible)
        if has_obs:
            counts_obs, _ = np.histogram(obs_arr, bins=bins)
            density_obs = counts_obs / (counts_obs.sum() * bw) * completeness
            pdf_peak = max(pdf_peak, float(density_obs.max()))
            fig.add_trace(go.Bar(
                x=((bins[:-1] + bins[1:]) / 2).tolist(),
                y=density_obs.tolist(),
                width=bw * 0.95,
                marker_color="black", opacity=0.35,
                name="Obs. PDF",
                showlegend=(col_idx == 0),
                legendgroup="obs_pdf",
            ), row=2, col=col)

        # Synthetic detected PDF (clipped, scaled by completeness)
        if det_params is not None and len(det_params.get(key, [])) >= 2:
            sim_raw = det_params[key]
            sim_clipped = sim_raw[(sim_raw >= clip_lo) & (sim_raw <= clip_hi)]
            if len(sim_clipped) >= 2:
                counts_sim, _ = np.histogram(sim_clipped, bins=bins)
                density_sim = counts_sim / (counts_sim.sum() * bw) * completeness
                density_sim_arr = density_sim
                pdf_peak = max(pdf_peak, float(density_sim.max()))
                bin_centers = ((bins[:-1] + bins[1:]) / 2).tolist()
                fig.add_trace(go.Scatter(
                    x=bin_centers, y=density_sim.tolist(),
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    name="Sim. PDF",
                    showlegend=(col_idx == 0),
                    legendgroup="sim_pdf",
                ), row=2, col=col)

        # Intrinsic PDF — empirical histogram or analytical power-law
        # Clip to same range as detected so PDFs are comparable.
        # Intrinsic integrates to 1.0; detected/observed to completeness.
        if use_empirical and key in empirical_all and len(empirical_all[key]) >= 2:
            all_vals = empirical_all[key]
            all_vals = all_vals[(all_vals >= clip_lo) & (all_vals <= clip_hi)]
            if len(all_vals) >= 2:
                counts_all, _ = np.histogram(all_vals, bins=bins)
                density_all = counts_all / (counts_all.sum() * bw)
                density_all_arr = density_all
                bin_centers = ((bins[:-1] + bins[1:]) / 2).tolist()
                fig.add_trace(go.Scatter(
                    x=bin_centers, y=density_all.tolist(),
                    mode="lines",
                    line=dict(color="gray", width=2, dash="dot"),
                    name="Sim. all PDF",
                    showlegend=(col_idx == 0),
                    legendgroup="intr",
                ), row=2, col=col)
        elif alpha is not None and xmin is not None:
            # Analytical power-law PDF — clamp to pdf_peak so
            # divergent power-laws don't crush the histogram y-axis.
            a = alpha + 1.0
            x_pdf = np.linspace(max(xmin, 1e-6), xmax, 200)
            if abs(a) < 1e-8:
                y_pdf = 1.0 / (x_pdf * np.log(xmax / xmin))
            else:
                y_pdf = a * x_pdf**alpha / (xmax**a - xmin**a)
            if pdf_peak > 0:
                y_pdf = np.minimum(y_pdf, pdf_peak)
            fig.add_trace(go.Scatter(
                x=x_pdf.tolist(), y=y_pdf.tolist(), mode="lines",
                line=dict(color="gray", width=2, dash="dot"),
                name="Intrinsic",
                showlegend=False,
                legendgroup="intr",
            ), row=2, col=col)

        fig.update_xaxes(title_text="", row=2, col=col)
        fig.update_yaxes(
            title_text="PDF" if col == 1 else "", row=2, col=col,
        )

        # ===================== Row 3: Bias correction =====================
        # Ratio of detected PDF to intrinsic (sim all) PDF — per-bin
        # detection efficiency. Only meaningful when both are available.
        if density_sim_arr is not None and density_all_arr is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(density_all_arr > 0,
                                 density_sim_arr / density_all_arr, np.nan)
            bin_centers = ((bins[:-1] + bins[1:]) / 2).tolist()
            fig.add_trace(go.Scatter(
                x=bin_centers, y=ratio.tolist(),
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                name="Bias correction",
                showlegend=(col_idx == 0),
                legendgroup="bias",
            ), row=3, col=col)

        fig.update_xaxes(title_text=xlabel, row=3, col=col)
        fig.update_yaxes(
            title_text="Sim / Sim all" if col == 1 else "", row=3, col=col,
        )

    fig.update_layout(
        height=800, margin=dict(t=40, b=40),
        barmode="overlay",
    )
    return fig


def plot_detection_maps(data, smooth=True, contours=True):
    """Plot 2D detection probability heatmaps."""
    label_map = {
        "logP": "log\u2081\u2080(P/d)", "e": "e",
        "K1": "K\u2081 [km/s]", "q": "q",
    }

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=["%s vs %s" % (label_map[a], label_map[b])
                        for a, b in _HIST_PAIRS],
    )

    for plot_idx, (a, b) in enumerate(_HIST_PAIRS):
        row = plot_idx // 3 + 1
        col = plot_idx % 3 + 1

        ht = data["hist_total"].get((a, b))
        hd = data["hist_det"].get((a, b))
        if ht is None or hd is None:
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            pdet = np.where(ht > 0, hd / ht, np.nan)

        # Bin centers
        bins_a = _HIST_BINS[a]
        bins_b = _HIST_BINS[b]
        centers_a = 0.5 * (bins_a[:-1] + bins_a[1:])
        centers_b = 0.5 * (bins_b[:-1] + bins_b[1:])

        pdet_filled = np.nan_to_num(pdet, nan=0.0)

        if smooth or contours:
            fig.add_trace(go.Contour(
                z=pdet_filled.T, x=centers_a, y=centers_b,
                colorscale="Viridis", zmin=0, zmax=1,
                contours=dict(
                    showlines=contours,
                    coloring="heatmap",
                    start=0, end=1, size=0.1,
                ),
                line=dict(color="white", width=0.8) if contours else None,
                showscale=(plot_idx == 0),
                colorbar=dict(title="p_det") if plot_idx == 0 else None,
            ), row=row, col=col)
        else:
            fig.add_trace(go.Heatmap(
                z=pdet.T, x=centers_a, y=centers_b,
                colorscale="Viridis", zmin=0, zmax=1,
                zsmooth=False,
                showscale=(plot_idx == 0),
                colorbar=dict(title="p_det") if plot_idx == 0 else None,
            ), row=row, col=col)

        fig.update_xaxes(title_text=label_map[a], row=row, col=col)
        fig.update_yaxes(title_text=label_map[b], row=row, col=col)

    fig.update_layout(height=550, margin=dict(t=40, b=40))
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Bias Grid Explorer", layout="wide")
    st.title("Bias Correction Grid Search Explorer")

    args = parse_args()

    # Output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = st.text_input("Output directory", "")
    if not output_dir or not os.path.isdir(output_dir):
        st.warning("Please provide a valid output directory.")
        st.stop()

    st.caption(os.path.basename(output_dir.rstrip("/")))

    # Load data
    data = load_grid_data(output_dir)

    # Load observed distributions
    sb1 = args.sb1_tex or DEFAULT_BIAS_CFG.get("sb1_tex", "")
    sb2 = args.sb2_tex or DEFAULT_BIAS_CFG.get("sb2_tex", "")
    obs = None
    if sb1 and sb2 and os.path.exists(sb1) and os.path.exists(sb2):
        obs = load_observed(sb1, sb2)

    pi_grid = data["pi_grid"]
    kappa_grid = data["kappa_grid"]
    eta_grid = data["eta_grid"]
    fbin_grid = data["fbin_grid"]
    grids = [pi_grid, kappa_grid, eta_grid, fbin_grid]
    available_tests = data.get("available_tests", ["ks"])

    # --- Sidebar: scoring metric selector ---
    st.sidebar.header("Scoring Metric")
    test_labels = {"ks": "Kolmogorov-Smirnov",
                   "ad": "Anderson-Darling",
                   "cvm": "Cram\u00e9r-von Mises"}
    selected_test = st.sidebar.radio(
        "Select test",
        options=available_tests,
        format_func=lambda t: test_labels.get(t, t.upper()),
        index=0,
    )

    # Active GMF cube and best fit for selected test
    gmf_cube = data["gmf_cubes"][selected_test]
    best_fit = data["best_fits"][selected_test]
    pval_cubes = data["test_pval_cubes"].get(selected_test, {})

    # --- Sidebar: parameter sliders ---
    st.sidebar.header("Grid Point Selection")
    pi_val = st.sidebar.select_slider(
        "\u03c0 (period exponent)",
        options=[round(x, 3) for x in pi_grid.tolist()],
        value=round(float(best_fit[0]), 3),
    )
    kappa_val = st.sidebar.select_slider(
        "\u03ba (mass-ratio exponent)",
        options=[round(x, 3) for x in kappa_grid.tolist()],
        value=round(float(best_fit[1]), 3),
    )
    eta_val = st.sidebar.select_slider(
        "\u03b7 (eccentricity exponent)",
        options=[round(x, 3) for x in eta_grid.tolist()],
        value=round(float(best_fit[2]), 3),
    )
    fbin_val = st.sidebar.select_slider(
        "f_bin (binary fraction)",
        options=[round(x, 3) for x in fbin_grid.tolist()],
        value=round(float(best_fit[3]), 3),
    )
    current_vals = [pi_val, kappa_val, eta_val, fbin_val]

    # Grid indices for current slider position — use argmin(abs) instead
    # of searchsorted to avoid off-by-one from float rounding mismatches.
    i = int(np.argmin(np.abs(pi_grid - pi_val)))
    j = int(np.argmin(np.abs(kappa_grid - kappa_val)))
    k = int(np.argmin(np.abs(eta_grid - eta_val)))
    l = int(np.argmin(np.abs(fbin_grid - fbin_val)))

    # Sidebar stats
    test_label = test_labels.get(selected_test, selected_test.upper())
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Grid Point Stats")
    st.sidebar.caption("e_score_mode: %s" %
                       data.get("e_score_mode", "combined"))
    st.sidebar.metric("p_det", "%.4f" % data["pdet_cube"][i, j, k, l])
    gmf_val = gmf_cube[i, j, k, l]
    st.sidebar.metric("log GMF (%s)" % test_label,
                       "%.2f" % gmf_val if np.isfinite(gmf_val) else "-inf")
    if pval_cubes:
        st.sidebar.metric("p(logP)", "%.4f" % pval_cubes.get(
            "logP", np.zeros_like(data["pdet_cube"]))[i, j, k, l])
        e_mode = data.get("e_score_mode", "combined")
        if e_mode == "split":
            st.sidebar.metric("p(e>0)", "%.4f" % pval_cubes["e"][i, j, k, l])
            if "e_circ" in pval_cubes:
                st.sidebar.metric(
                    "p(f_circ)", "%.4f" % pval_cubes["e_circ"][i, j, k, l])
        elif e_mode == "eccentric_only":
            st.sidebar.metric("p(e>0)", "%.4f" % pval_cubes.get(
                "e", np.zeros_like(data["pdet_cube"]))[i, j, k, l])
        else:
            st.sidebar.metric("p(e)", "%.4f" % pval_cubes.get(
                "e", np.zeros_like(data["pdet_cube"]))[i, j, k, l])
        st.sidebar.metric("p(K1)", "%.4f" % pval_cubes.get(
            "K1", np.zeros_like(data["pdet_cube"]))[i, j, k, l])

    # Best fit info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Best Fit (%s)" % test_label)
    st.sidebar.text(
        "\u03c0=%.2f  \u03ba=%.2f  \u03b7=%.2f  f=%.2f" % best_fit
    )
    # Show all tests' best fits for comparison
    if len(available_tests) > 1:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Best Fits (all tests)")
        for tname in available_tests:
            bf = data["best_fits"][tname]
            st.sidebar.text(
                "%s: \u03c0=%.2f \u03ba=%.2f \u03b7=%.2f f=%.2f" % (
                    tname.upper(), *bf))

    # --- Main area ---
    prob = gmf_to_prob(gmf_cube)

    # Section 1: Marginalized Posterior Corner Plot
    st.header("Marginalized Posterior Corner (%s)" % test_label)
    smooth_corner = st.checkbox("Smooth interpolation", value=True,
                                key="smooth_corner")
    fig_corner = plot_corner(prob, grids, best_fit, current_vals,
                             smooth=smooth_corner)
    st.plotly_chart(fig_corner, use_container_width=True)

    # Section 3: CDF Comparison
    if data["has_detected"] and obs is not None:
        st.header("CDF Comparison")
        use_empirical = st.checkbox(
            "Use empirical intrinsic (det + nondet) instead of power-law",
            value=True, key="use_empirical",
        )
        det_params = get_detected_for_point(data, i, j, k, l)
        ks_pvals = {}
        if pval_cubes:
            ks_pvals = {
                "ks_logP": pval_cubes.get(
                    "logP", np.zeros_like(data["pdet_cube"]))[i, j, k, l],
                "ks_e": pval_cubes.get(
                    "e", np.zeros_like(data["pdet_cube"]))[i, j, k, l],
                "ks_K1": pval_cubes.get(
                    "K1", np.zeros_like(data["pdet_cube"]))[i, j, k, l],
            }
            if "e_circ" in pval_cubes:
                ks_pvals["ks_e_circ"] = pval_cubes["e_circ"][i, j, k, l]
        if det_params is not None and len(det_params["logP"]) >= 2:
            p_det = float(data["pdet_cube"][i, j, k, l])
            fig_cdf = plot_cdfs(det_params, obs, current_vals, ks_pvals,
                                test_label=test_label,
                                use_empirical=use_empirical,
                                p_det=p_det,
                                e_score_mode=data.get(
                                    "e_score_mode", "combined"))
            st.plotly_chart(fig_cdf, use_container_width=True)
        else:
            st.info("No detected systems for this grid point "
                    "(or fewer than 2 detections).")
    elif not data["has_detected"]:
        st.info("CDF comparison unavailable: grid_detected.npz not found. "
                "Re-run bias_grid with the updated code to generate it.")
    elif obs is None:
        st.info("CDF comparison unavailable: could not load observed "
                "distributions from LaTeX tables. Check --sb1-tex / --sb2-tex.")

    # Section 4: 2D Detection Probability Maps
    if data["has_detected"] and data.get("hist_total"):
        st.header("2D Detection Probability Maps")
        col_a, col_b = st.columns(2)
        smooth_det = col_a.checkbox("Smooth interpolation", value=True, key="smooth_det")
        contours_det = col_b.checkbox("Contour lines", value=True, key="contours_det")
        fig_det = plot_detection_maps(data, smooth=smooth_det, contours=contours_det)
        st.plotly_chart(fig_det, use_container_width=True)


if __name__ == "__main__":
    main()
