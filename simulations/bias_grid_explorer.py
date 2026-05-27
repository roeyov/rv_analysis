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

# numpy 2.x renamed trapz -> trapezoid
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulations.bias_config import DEFAULT_BIAS_CFG
from simulations.bias_grid import (
    _HIST_BINS, _HIST_PAIRS, _HIST_NBINS,
    _DET_SHARDS_DIR, _load_det_shard,
    CUBE_SCHEMA_VERSION,
)


# ---------------------------------------------------------------------------
# CLI argument parsing (Streamlit passes args after --)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


def _obs_e_from_cube(e_value, e_is_upper_limit, apply_lucy_sweeny_e):
    """Apply Lucy-Sweeney convention to raw cube obs e.

    Collapses upper-limit rows to e=0 when ``apply_lucy_sweeny_e`` is True;
    otherwise keeps the reported limit value. Mirrors the inline logic in
    ``bias_grid.load_observed_from_tex`` so explorer renders match what the
    runtime scored.
    """
    e = np.asarray(e_value, dtype=float).copy()
    mask = np.asarray(e_is_upper_limit, dtype=bool)
    if apply_lucy_sweeny_e:
        e[mask] = 0.0
    return e


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_grid_data(output_dir):
    """Load grid_cubes.npz (or checkpoint_cubes.npz) plus shard metadata.

    Enforces ``cube_schema_version >= CUBE_SCHEMA_VERSION``: older cubes
    don't carry the obs arrays the explorer now requires, so re-rendering
    them would silently fall back to outdated paths. Hard-fail instead.
    """
    cubes_path = os.path.join(output_dir, "grid_cubes.npz")
    if not os.path.exists(cubes_path):
        cubes_path = os.path.join(output_dir, "checkpoint_cubes.npz")
    cubes = np.load(cubes_path, allow_pickle=True)

    stored_version = (int(cubes["cube_schema_version"])
                      if "cube_schema_version" in cubes.files else 0)
    if stored_version < CUBE_SCHEMA_VERSION:
        st.error(
            "Cube schema mismatch: %s reports version %d, explorer "
            "requires >= %d. Re-run bias_grid.py on this output dir to "
            "regenerate the cubes with the obs arrays and extra "
            "metadata the explorer now needs."
            % (cubes_path, stored_version, CUBE_SCHEMA_VERSION)
        )
        st.stop()

    data = {
        "cube_schema_version": stored_version,
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

    # logP cutoff metadata. Schema v2+ guarantees the mode/value keys;
    # smooth_sigma is best-effort (only present for newly-rerun cubes).
    data["logP_cutoff"] = float(cubes["logP_cutoff"]) \
        if "logP_cutoff" in cubes.files else 0.0
    data["logP_cutoff_mode"] = str(cubes["logP_cutoff_mode"]) \
        if "logP_cutoff_mode" in cubes.files else "none"
    data["logP_cutoff_scope"] = str(cubes["logP_cutoff_scope"]) \
        if "logP_cutoff_scope" in cubes.files else "period_only"
    data["logP_cutoff_smooth_sigma"] = (
        float(cubes["logP_cutoff_smooth_sigma"])
        if "logP_cutoff_smooth_sigma" in cubes.files else float("nan"))

    data["apply_lucy_sweeny_e"] = bool(cubes["apply_lucy_sweeny_e"]) \
        if "apply_lucy_sweeny_e" in cubes.files else True

    # Source tex paths — purely informational, surfaced as captions.
    data["sb1_tex"] = str(cubes["sb1_tex"]) \
        if "sb1_tex" in cubes.files else ""
    data["sb2_tex"] = str(cubes["sb2_tex"]) \
        if "sb2_tex" in cubes.files else ""

    # Observed distributions. Stored as raw e_value + is_upper_limit mask
    # so the explorer can re-render under either Lucy-Sweeney convention.
    data["obs"] = {
        "logP": np.asarray(cubes["obs_logP"]),
        "e_value": np.asarray(cubes["obs_e_value"]),
        "e_is_upper_limit": np.asarray(cubes["obs_e_is_upper_limit"],
                                       dtype=bool),
        "K1": np.asarray(cubes["obs_K1"]),
        "q_sb2": np.asarray(cubes["obs_q_sb2"]),
        "n_sb1": int(cubes["obs_n_sb1"]),
        "n_sb2": int(cubes["obs_n_sb2"]),
    }

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
        post = post / _trapz(post, grid)
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
              use_empirical=False, p_det=None, e_score_mode="combined",
              logP_cutoff=0.0, restrict_e_to_positive=True):
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

    # The caller is responsible for any scope-aware obs filtering: under
    # 'exclude' it pre-masks P/e/K1 jointly; under 'period_only' it masks
    # only obs['logP']. Here we treat ``logP_cutoff`` as the *axis* clip
    # value for the logP panel only — it sets clip_lo for the P CDF.

    # Build empirical "all injected" arrays when requested
    empirical_all = {}
    if use_empirical and det_params is not None:
        for key in ("logP", "e", "K1", "q"):
            nondet_key = "%s_nondet" % key
            det_arr = det_params.get(key, np.array([]))
            nondet_arr = det_params.get(nondet_key, np.array([]))
            if len(det_arr) or len(nondet_arr):
                empirical_all[key] = np.concatenate([det_arr, nondet_arr])

    # Eccentric_only ALWAYS ignores e=0 on the simulated side — that's
    # what the fit did. The checkbox only affects how we *display* the
    # obs side and the y-axis of sim/intrinsic CDFs:
    #   restrict_e_to_positive=True  → hide e=0 from obs too (scored view)
    #   restrict_e_to_positive=False → show the obs e=0 spike and shift
    #     sim/intrinsic CDFs upward by f_circ_obs so they start where the
    #     obs CDF resumes after the spike.
    obs_e_arr = obs.get("e")
    f_circ_obs = 0.0
    if obs_e_arr is not None:
        obs_e_arr = np.asarray(obs_e_arr)
        if len(obs_e_arr):
            f_circ_obs = float(np.sum(obs_e_arr == 0) / len(obs_e_arr))
        if e_score_mode == "eccentric_only" and restrict_e_to_positive:
            obs_e_arr = obs_e_arr[obs_e_arr > 0]
    # In eccentric_only the simulated e=0 systems are always discarded
    # (matching the fit), independent of the display checkbox.
    if e_score_mode == "eccentric_only" and "e" in empirical_all:
        empirical_all["e"] = empirical_all["e"][empirical_all["e"] > 0]

    # When the e>0 restriction is off in eccentric_only mode, shift sim
    # and intrinsic CDFs upward so they start at y = f_circ_obs and the
    # obs e=0 spike sits below the resumed CDF.
    shift_e_to_full = (e_score_mode == "eccentric_only"
                       and not restrict_e_to_positive
                       and f_circ_obs > 0.0)

    # (key, obs_array_or_None, xlabel, color, alpha, xmin, xmax, n_bins)
    param_configs = [
        ("logP", obs["logP"], "log\u2081\u2080(P/d)", "#4393c3",
         pi_val, max(cfg["log_p_min"], logP_cutoff), cfg["log_p_max"], 15),
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
            # In eccentric_only the fit always filtered sim/empirical e=0
            # out — keep that floor on the sim side regardless of the
            # display checkbox. The obs-side window is handled separately
            # below so the obs e=0 spike can still show through.
            if e_score_mode == "eccentric_only":
                clip_lo = 1e-12
            else:
                clip_lo = 0.0
            clip_hi = 1.0
        elif key == "logP" and has_obs:
            clip_lo = max(0.0, logP_cutoff)
            clip_hi = float(obs_s[-1])
        elif has_obs:
            clip_lo, clip_hi = 0.0, float(obs_s[-1])
        else:
            clip_lo = xmin if xmin is not None else 0.0
            clip_hi = xmax if xmax is not None else 1.0

        # Obs-side window: filter obs_s only when we want to *hide* the
        # below-window region from the display. For logP-cutoff and the
        # "restricted" eccentric_only view (shift_e_to_full=False) we
        # filter; when shift_e_to_full we keep obs zeros so the spike
        # is visible.
        if has_obs and clip_lo > 0.0 and not shift_e_to_full:
            obs_s = obs_s[(obs_s >= clip_lo) & (obs_s <= clip_hi)]
            has_obs = len(obs_s) >= 2

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
                det_vals = np.asarray(det_params.get(key, np.array([])))
                # Match bias_grid scoring: when a window is active (logP
                # cutoff, or e>0 for eccentric_only-restricted) the
                # detection rate is computed within that window.
                if clip_lo > 0.0 or clip_hi < np.inf:
                    all_in = all_vals[(all_vals >= clip_lo)
                                      & (all_vals <= clip_hi)]
                    det_in = det_vals[(det_vals >= clip_lo)
                                      & (det_vals <= clip_hi)] \
                        if len(det_vals) else det_vals
                    if len(all_in) > 0:
                        completeness = len(det_in) / len(all_in)
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
                sim_y = sim_cdf * completeness
                if key == "e" and shift_e_to_full:
                    # Lift conditional sim CDF onto the full e axis:
                    # starts at f_circ_obs*completeness, ends at completeness.
                    sim_y = completeness * (
                        f_circ_obs + (1.0 - f_circ_obs) * sim_cdf)
                fig.add_trace(go.Scatter(
                    x=sim, y=sim_y, mode="lines",
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
                all_y = all_cdf
                if key == "e" and shift_e_to_full:
                    all_y = f_circ_obs + (1.0 - f_circ_obs) * all_cdf
                fig.add_trace(go.Scatter(
                    x=all_s, y=all_y, mode="lines",
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
            # Eccentric_only was fit to e>0 only. If the user is viewing
            # the full distribution, shift the power-law upward by the
            # observed circular fraction and prepend a vertical jump.
            if key == "e" and shift_e_to_full:
                y_intr = f_circ_obs + (1.0 - f_circ_obs) * y_intr
                x_intr = np.concatenate([[0.0, xmin], x_intr])
                y_intr = np.concatenate([[0.0, f_circ_obs], y_intr])
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

        # When showing the lifted eccentric_only view, extend bins down
        # to 0 so the obs e=0 spike is captured in the first bin.
        if key == "e" and shift_e_to_full:
            bin_lo = 0.0
        elif has_obs:
            bin_lo = clip_lo
        else:
            bin_lo = xmin
        bin_hi = clip_hi if has_obs else xmax
        bins = np.linspace(bin_lo, bin_hi, n_bins + 1)
        bw = float(bins[1] - bins[0])
        pdf_peak = 0.0  # track tallest bar for intrinsic scaling
        density_sim_arr = None
        density_all_arr = None
        # Conditional-to-unconditional scaling for sim/intrinsic PDFs
        # when lifting the eccentric_only view onto the full e axis.
        e_pdf_scale = (1.0 - f_circ_obs) if (
            key == "e" and shift_e_to_full) else 1.0

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
                density_sim = (counts_sim / (counts_sim.sum() * bw)
                               * completeness * e_pdf_scale)
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
                density_all = (counts_all / (counts_all.sum() * bw)
                               * e_pdf_scale)
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
            shift_pl_e_pdf = key == "e" and shift_e_to_full
            if shift_pl_e_pdf:
                y_pdf = y_pdf * (1.0 - f_circ_obs)
            if pdf_peak > 0:
                y_pdf = np.minimum(y_pdf, pdf_peak)
            fig.add_trace(go.Scatter(
                x=x_pdf.tolist(), y=y_pdf.tolist(), mode="lines",
                line=dict(color="gray", width=2, dash="dot"),
                name="Intrinsic",
                showlegend=False,
                legendgroup="intr",
            ), row=2, col=col)
            if shift_pl_e_pdf:
                # Delta-spike representation of the circular fraction:
                # a tall bar at e=0 with area = f_circ_obs.
                fig.add_trace(go.Bar(
                    x=[float(bins[0] + bw / 2)],
                    y=[f_circ_obs / bw],
                    width=bw * 0.95,
                    marker_color="gray", opacity=0.4,
                    name="Intrinsic e=0",
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

    # Load data. Obs lives in the cube now (schema v2+); apply Lucy-Sweeney
    # at display time per cube flag, with an in-page toggle to override.
    data = load_grid_data(output_dir)
    obs_cube = data["obs"]

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
    st.sidebar.caption("apply_lucy_sweeny_e: %s" %
                       data.get("apply_lucy_sweeny_e", True))
    st.sidebar.caption("logP_cutoff_mode: %s" %
                       data.get("logP_cutoff_mode", "none"))
    st.sidebar.caption("logP_cutoff_scope: %s" %
                       data.get("logP_cutoff_scope", "period_only"))
    _cut_val = float(data.get("logP_cutoff", 0.0))
    st.sidebar.caption("logP_cutoff: %.3f" % _cut_val)
    _smooth_sigma = data.get("logP_cutoff_smooth_sigma", float("nan"))
    if np.isfinite(_smooth_sigma):
        st.sidebar.caption("logP_cutoff_smooth_sigma: %.3f" % _smooth_sigma)
    if data.get("sb1_tex"):
        st.sidebar.caption("sb1_tex: %s" %
                           os.path.basename(data["sb1_tex"]))
    if data.get("sb2_tex"):
        st.sidebar.caption("sb2_tex: %s" %
                           os.path.basename(data["sb2_tex"]))
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
    if data["has_detected"]:
        st.header("CDF Comparison")
        use_empirical = st.checkbox(
            "Use empirical intrinsic (det + nondet) instead of power-law",
            value=True, key="use_empirical",
        )

        # "As scored" toggles match the filtering used to compute the
        # KS/AD/CvM p-values stored in the cubes.
        e_mode_data = data.get("e_score_mode", "combined")
        logP_cutoff_data = float(data.get("logP_cutoff", 0.0))
        logP_cutoff_mode = data.get("logP_cutoff_mode", "none")
        logP_cutoff_scope = data.get("logP_cutoff_scope", "period_only")
        cutoff_active = logP_cutoff_data > 0.0
        cube_lucy = bool(data.get("apply_lucy_sweeny_e", True))

        col_cb1, col_cb2, col_cb3 = st.columns(3)

        if logP_cutoff_scope == "period_only":
            cutoff_help = (
                "Drop obs systems with logP < cutoff from the P panel "
                "only — mirrors the runtime 'period_only' scope: e/K1 "
                "obs and sim/intrinsic distributions are unaffected.")
        else:
            cutoff_help = (
                "Drop obs systems with logP < cutoff jointly from the "
                "P/e/K1 panels — mirrors the runtime 'exclude' scope. "
                "Sim/intrinsic are already population-restricted by "
                "log_p_min override at scoring time.")
        apply_logP_cutoff = col_cb1.checkbox(
            "Apply logP cutoff (scope=%s, value=%.3f, mode=%s)" % (
                logP_cutoff_scope, logP_cutoff_data, logP_cutoff_mode),
            value=cutoff_active,
            disabled=(not cutoff_active),
            key="apply_logP_cutoff",
            help=cutoff_help,
        )
        if e_mode_data == "eccentric_only":
            restrict_e_to_positive = col_cb2.checkbox(
                "Restrict eccentricity to e>0 (as scored)",
                value=True,
                key="restrict_e_to_positive",
            )
        else:
            restrict_e_to_positive = True

        display_lucy = col_cb3.checkbox(
            "Apply Lucy-Sweeney (cube=%s)" % cube_lucy,
            value=cube_lucy,
            key="display_lucy",
            help="Collapse e upper-limit rows to e=0 (cube convention) vs "
                 "keep the reported limit value. Off-default diverges from "
                 "the persisted p-values.",
        )
        if display_lucy != cube_lucy:
            st.warning(
                "Display-only override: persisted p-values were computed "
                "with apply_lucy_sweeny_e=%s. CDFs below use %s." % (
                    cube_lucy, display_lucy))

        # Build the obs dict the rest of the section consumes: apply
        # display-time Lucy-Sweeney, then (if requested) the scope-aware
        # logP cutoff filter.
        obs = {
            "logP": np.asarray(obs_cube["logP"]),
            "e": _obs_e_from_cube(obs_cube["e_value"],
                                  obs_cube["e_is_upper_limit"],
                                  display_lucy),
            "K1": np.asarray(obs_cube["K1"]),
            "q_sb2": np.asarray(obs_cube["q_sb2"]),
            "n_sb1": obs_cube["n_sb1"],
            "n_sb2": obs_cube["n_sb2"],
        }
        if apply_logP_cutoff and cutoff_active:
            keep = obs["logP"] >= logP_cutoff_data
            if logP_cutoff_scope == "exclude":
                # Joint mask: drop the same rows from P, e, K1 — matches
                # bias_grid.py:1733-1735 under scope='exclude'.
                obs = {
                    "logP": obs["logP"][keep],
                    "e": obs["e"][keep],
                    "K1": obs["K1"][keep],
                    "q_sb2": obs["q_sb2"],
                    "n_sb1": obs["n_sb1"],
                    "n_sb2": obs["n_sb2"],
                }
            else:
                # period_only: filter only the P panel; e/K1 stay full.
                obs["logP"] = obs["logP"][keep]

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
            fig_cdf = plot_cdfs(
                det_params, obs, current_vals, ks_pvals,
                test_label=test_label,
                use_empirical=use_empirical,
                p_det=p_det,
                e_score_mode=e_mode_data,
                logP_cutoff=(logP_cutoff_data if apply_logP_cutoff else 0.0),
                restrict_e_to_positive=restrict_e_to_positive,
            )
            st.plotly_chart(fig_cdf, use_container_width=True)
        else:
            st.info("No detected systems for this grid point "
                    "(or fewer than 2 detections).")
    else:
        st.info("CDF comparison unavailable: detected arrays missing. "
                "Re-run bias_grid to generate per-grid-point shards.")

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
