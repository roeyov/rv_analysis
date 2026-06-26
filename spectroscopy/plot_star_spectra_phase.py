#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive multi-epoch spectra plotter — phase-ordered animation.

Like plot_star_spectra.py but sorts/animates by orbital phase computed from the
best solution in lmfit_summary.csv.  Includes a side panel with the phase-folded
RV curve and highlights the max/min RV epochs.
"""
import argparse, re, webbrowser
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

from orbital.kepler import Kepler, rv_model, true_anomaly_from_E

EPOCH_REGEX = re.compile(r'_(\d+)(?:[_\.])')


# ---------------------------------------------------------------------------
# Helpers shared with plot_star_spectra.py
# ---------------------------------------------------------------------------

def find_candidate_files(root: Path, star_name: str) -> List[Path]:
    star_lower = star_name.lower()
    return sorted([p for p in root.rglob("*.fits") if star_lower in p.name.lower()],
                  key=lambda p: p.name.lower())


def parse_epoch_from_name(fname: str) -> Optional[int]:
    m = EPOCH_REGEX.search(fname)
    if not m: return None
    try: return int(m.group(1))
    except ValueError: return None


def read_fits_spectrum(fp: Path, x_name: str, y_name: str, ext: int):
    with fits.open(fp, ignore_missing_simple=True) as hdul:
        hdr0 = dict(hdul[0].header)
        hdu = hdul[ext]
        hdr_ext = dict(hdu.header)
        data = hdu.data
        if hasattr(data, "columns"):
            x = np.array(data[x_name], dtype=float).flatten()
            y = np.array(data[y_name], dtype=float).flatten()
        else:
            naxis1 = data.shape[-1]
            crval1 = hdu.header.get("CRVAL1")
            cdelt1 = hdu.header.get("CDELT1")
            crpix1 = hdu.header.get("CRPIX1", 1.0)
            if crval1 is None or cdelt1 is None:
                raise ValueError(f"{fp.name}: missing CRVAL1/CDELT1 for wavelength axis.")
            pix = np.arange(1, naxis1 + 1, dtype=float)
            x = crval1 + (pix - crpix1) * cdelt1
            y = np.array(data, dtype=float).flatten()
        return x, y, {"primary_header": hdr0, "ext_header": hdr_ext}


def extract_mjd(headers: dict, preferred_key: str) -> Optional[float]:
    for k in [preferred_key, "MJD-OBS", "MJD", "HJD", "BMJD", "BJD"]:
        v = headers["primary_header"].get(k, headers["ext_header"].get(k))
        if v is not None:
            try: return float(v)
            except: pass
    return None


def load_template(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    sfx = path.suffix.lower()
    if sfx in {".fits", ".fit", ".fts"}:
        with fits.open(path, ignore_missing_simple=True) as hdul:
            pick = None
            for h in hdul[1:]:
                if hasattr(h.data, "columns") and h.data is not None and len(h.data.columns) >= 2:
                    pick = h; break
            if pick is None: raise ValueError("Template FITS must have a table HDU with >=2 columns.")
            return (np.array(pick.data.field(0), float).flatten(),
                    np.array(pick.data.field(1), float).flatten())
    arr = np.loadtxt(path, ndmin=2)
    if arr.shape[1] < 2: raise ValueError("Template text file must have >=2 columns.")
    return arr[:, 0], arr[:, 1]


# ---------------------------------------------------------------------------
# Best-solution loading from lmfit_summary.csv + params.yaml
# ---------------------------------------------------------------------------

def load_best_solution(csv_path: Path, config_path: Path) -> pd.Series:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mcmc = cfg["mcmc_params"]
    filter_expr = mcmc["filter_expression"]
    sort_field = mcmc.get("field_to_check", "bicc")
    take_min = mcmc.get("take_min", True)

    df = pd.read_csv(csv_path)
    filtered = df.query(filter_expr)
    if filtered.empty:
        raise SystemExit(f"[ERR] No rows pass the filter in {csv_path}")
    best = filtered.sort_values(sort_field, ascending=take_min).iloc[0]
    return best


def parse_numpy_str(s: str) -> np.ndarray:
    """Parse a numpy-style string array like '[ 1.0  2.0  3.0 ]'."""
    return np.fromstring(s.strip("[]"), sep=" ")


# ---------------------------------------------------------------------------
# Phase-based coloring
# ---------------------------------------------------------------------------

def phase_to_hex_color(phase: float) -> str:
    cmap = mcm.get_cmap("twilight_shifted")
    return mcolors.to_hex(cmap(phase))


# ---------------------------------------------------------------------------
# RV model curve on a dense phase grid
# ---------------------------------------------------------------------------

def compute_rv_curve_phase(ecc, gamma, K1, omega, n_points=1000):
    phase_grid = np.linspace(0, 1, n_points)
    M = 2 * np.pi * phase_grid
    E = Kepler(np.pi, M, ecc)
    nu = true_anomaly_from_E(E, ecc)
    rv = rv_model(nu, gamma, K1, omega, ecc)
    return phase_grid, rv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase-ordered multi-epoch spectra plotter")
    ap.add_argument("directory", type=str)
    ap.add_argument("star_name", type=str)
    ap.add_argument("--csv", required=True, type=str, help="Path to lmfit_summary.csv")
    ap.add_argument("--config", required=True, type=str, help="Path to params.yaml")
    ap.add_argument("--x-name", default="WAVELENGTH")
    ap.add_argument("--y-name", default="SCI_NORM")
    ap.add_argument("--ext", type=int, default=1)
    ap.add_argument("--time-key", default="MJD")
    ap.add_argument("--cadence-ms", type=int, default=200, help="Animation cadence per epoch (ms)")
    ap.add_argument("--template", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    # --- Load best orbital solution ---
    best = load_best_solution(Path(args.csv), Path(args.config))
    P = float(best["Period_value"])
    T0 = float(best["T0_value"])
    ecc = float(best["Eccentricity_value"])
    omega = float(best["OMEGA_rad_value"])
    K1 = float(best["K1_value"])
    gamma = float(best["GAMMA_value"])
    print(f"[INFO] Best solution: P={P:.4f} d, T0={T0:.4f}, e={ecc:.4f}, K1={K1:.2f}")

    # --- Parse RV data from CSV ---
    rvs_csv = parse_numpy_str(str(best["rvs"]))
    errs_csv = parse_numpy_str(str(best["errs"]))
    mjds_csv = parse_numpy_str(str(best["MJD"]))
    phases_csv = ((mjds_csv - T0) / P) % 1.0

    # Identify max/min RV indices
    idx_max_rv = int(np.argmax(rvs_csv))
    idx_min_rv = int(np.argmin(rvs_csv))
    mjd_max_rv = mjds_csv[idx_max_rv]
    mjd_min_rv = mjds_csv[idx_min_rv]

    # --- Find and read spectra ---
    root = Path(args.directory).expanduser().resolve()
    files = find_candidate_files(root, args.star_name)
    if not files:
        raise SystemExit(f"[ERR] No FITS files containing '{args.star_name}' under {root}")

    spectra = []
    for fp in files:
        try:
            x, y, headers = read_fits_spectrum(fp, args.x_name, args.y_name, args.ext)
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}")
            continue
        epoch = parse_epoch_from_name(fp.name)
        mjd = extract_mjd(headers, args.time_key)
        if mjd is None:
            print(f"[WARN] Skipping {fp.name}: no MJD found (needed for phase)")
            continue
        phase = ((mjd - T0) / P) % 1.0
        # Check if this epoch is max or min RV (closest MJD match)
        is_max_rv = abs(mjd - mjd_max_rv) < 0.01
        is_min_rv = abs(mjd - mjd_min_rv) < 0.01
        spectra.append({
            "path": fp, "epoch": epoch,
            "epoch_str": str(epoch) if epoch is not None else "NA",
            "mjd": mjd, "phase": phase, "x": x, "y": y,
            "is_max_rv": is_max_rv, "is_min_rv": is_min_rv,
        })

    if not spectra:
        raise SystemExit(f"[ERR] No valid spectra for '{args.star_name}'.")

    # Sort by orbital phase
    spectra.sort(key=lambda d: d["phase"])

    # --- Create side-by-side figure ---
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=["Spectra", "Phase-folded RV"],
        horizontal_spacing=0.08,
    )

    original_colors = []
    spectra_trace_indices = []  # track which trace indices are spectra (col 1)

    # --- Col 1: spectra traces ---
    for s in spectra:
        color = phase_to_hex_color(s["phase"])
        original_colors.append(color)

        tag = "         "  # 9 chars padding to match "[MAX RV] "
        lw = 2
        if s["is_max_rv"]:
            tag = " [MAX RV]"
        elif s["is_min_rv"]:
            tag = " [MIN RV]"

        customdata = np.column_stack([
            np.full_like(s["x"], s["mjd"], dtype=float),
            np.full_like(s["x"], s["epoch"] if s["epoch"] is not None else np.nan, dtype=float),
            np.full_like(s["x"], s["phase"], dtype=float),
        ])
        trace_idx = len(fig.data)
        spectra_trace_indices.append(trace_idx)
        fig.add_trace(go.Scatter(
            x=s["x"], y=s["y"], mode="lines",
            name=f"\u03c6={s['phase']:.3f} (ep {s['epoch_str']:>3s}){tag}",
            line=dict(width=lw, color=color),
            customdata=customdata,
            legendgroup=f"spec_{s['epoch_str']}",
            hovertemplate=(
                "\u03bb=%{x:.2f}<br>"
                "Flux=%{y:.4g}<br>"
                "MJD=%{customdata[0]:.5f}<br>"
                "Epoch=%{customdata[1]:.0f}<br>"
                "\u03c6=%{customdata[2]:.3f}"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    template_index = None
    if args.template:
        try:
            tx, ty = load_template(Path(args.template).expanduser().resolve())
            template_index = len(fig.data)
            fig.add_trace(go.Scatter(
                x=tx, y=ty, mode="lines", name="Template",
                line=dict(width=3, dash="dash", color="#000000"),
                hovertemplate="Template<extra></extra>"
            ), row=1, col=1)
            original_colors.append("#000000")
        except Exception as e:
            print(f"[WARN] Failed to load template: {e}")

    n_spectra_traces = len(fig.data)  # all col-1 traces so far

    # --- Col 2: Phase-folded RV curve ---
    # Model curve
    ph_grid, rv_grid = compute_rv_curve_phase(ecc, gamma, K1, omega)
    fig.add_trace(go.Scatter(
        x=ph_grid, y=rv_grid, mode="lines",
        name="RV model",
        line=dict(width=2, color="#333333"),
        showlegend=False,
        hovertemplate="\u03c6=%{x:.3f}<br>RV=%{y:.2f} km/s<extra></extra>",
    ), row=1, col=2)

    # Data points (regular)
    mask_regular = np.ones(len(rvs_csv), dtype=bool)
    mask_regular[idx_max_rv] = False
    mask_regular[idx_min_rv] = False
    fig.add_trace(go.Scatter(
        x=phases_csv[mask_regular], y=rvs_csv[mask_regular],
        mode="markers",
        name="RV data",
        marker=dict(size=8, color=[phase_to_hex_color(p) for p in phases_csv[mask_regular]],
                    line=dict(width=1, color="black")),
        error_y=dict(type="data", array=errs_csv[mask_regular], visible=True),
        showlegend=False,
        hovertemplate="\u03c6=%{x:.3f}<br>RV=%{y:.2f} km/s<extra></extra>",
    ), row=1, col=2)

    # Max RV point
    fig.add_trace(go.Scatter(
        x=[phases_csv[idx_max_rv]], y=[rvs_csv[idx_max_rv]],
        mode="markers",
        name="MAX RV",
        marker=dict(size=14, color="red", symbol="triangle-up",
                    line=dict(width=2, color="black")),
        error_y=dict(type="data", array=[errs_csv[idx_max_rv]], visible=True),
        showlegend=True,
        hovertemplate="MAX RV<br>\u03c6=%{x:.3f}<br>RV=%{y:.2f} km/s<extra></extra>",
    ), row=1, col=2)

    # Min RV point
    fig.add_trace(go.Scatter(
        x=[phases_csv[idx_min_rv]], y=[rvs_csv[idx_min_rv]],
        mode="markers",
        name="MIN RV",
        marker=dict(size=14, color="blue", symbol="triangle-down",
                    line=dict(width=2, color="black")),
        error_y=dict(type="data", array=[errs_csv[idx_min_rv]], visible=True),
        showlegend=True,
        hovertemplate="MIN RV<br>\u03c6=%{x:.3f}<br>RV=%{y:.2f} km/s<extra></extra>",
    ), row=1, col=2)

    # Phase marker (vertical line) — used during animation to show current phase
    # Starts at first spectrum's phase
    rv_ymin = float(np.min(rv_grid)) - 10
    rv_ymax = float(np.max(rv_grid)) + 10
    phase_marker_idx = len(fig.data)
    fig.add_trace(go.Scatter(
        x=[spectra[0]["phase"], spectra[0]["phase"]],
        y=[rv_ymin, rv_ymax],
        mode="lines",
        name="Current phase",
        line=dict(width=2, color="red", dash="dot"),
        showlegend=False,
        visible=False,  # only visible during animation
    ), row=1, col=2)

    n_total = len(fig.data)
    n_rv_traces = n_total - n_spectra_traces  # RV panel traces count

    n_spectra = len(spectra)
    anim_color = "#1f77b4"

    # -------- Animation frames: phase-ordered, one spectrum at a time --------
    frames = []
    for i in range(n_spectra):
        # Visibility: hide all spectra except i-th; keep all RV traces visible
        vis = [False] * n_total
        vis[spectra_trace_indices[i]] = True  # show this spectrum
        # Template always hidden during animation (or show if desired)
        # RV traces always visible
        for j in range(n_spectra_traces, n_total):
            vis[j] = True
        # Phase marker visible during animation
        vis[phase_marker_idx] = True

        # Update phase marker position
        frame_data = []
        for j in range(n_total):
            if j == phase_marker_idx:
                frame_data.append(go.Scatter(
                    x=[spectra[i]["phase"], spectra[i]["phase"]],
                    y=[rv_ymin, rv_ymax],
                    visible=True,
                ))
            else:
                frame_data.append(go.Scatter(visible=vis[j]))

        frames.append(go.Frame(
            name=f"\u03c6={spectra[i]['phase']:.3f} (ep {spectra[i]['epoch_str']})",
            data=frame_data,
        ))
    fig.frames = frames

    # --- helper lists for buttons ---
    all_true_default = [True] * n_spectra_traces + [True] * n_rv_traces
    # Phase marker hidden in default mode
    all_true_default[phase_marker_idx] = False

    anim_start_vis = [False] * n_total
    anim_start_vis[spectra_trace_indices[0]] = True
    for j in range(n_spectra_traces, n_total):
        anim_start_vis[j] = True

    # -------- UI menus --------
    hide_all_spectra = ["legendonly"] * n_spectra_traces + [True] * n_rv_traces
    hide_all_spectra[phase_marker_idx] = False

    anim_colors = [anim_color] * n_spectra
    if template_index is not None:
        anim_colors.append(fig.data[template_index].line.color)

    menu_default = dict(
        type="buttons", direction="left", x=0.0, y=1.15, xanchor="left", yanchor="top",
        showactive=False,
        buttons=[
            dict(label="Remove all", method="update",
                 args=[{"visible": hide_all_spectra}, {}]),
            dict(label="Reset all", method="update",
                 args=[{"visible": all_true_default}, {}]),
            dict(
                label="Enter animation",
                method="update",
                args=[
                    {"visible": anim_start_vis},
                    {
                        "updatemenus[0].visible": False,
                        "updatemenus[1].visible": True,
                        "sliders[0].visible": True,
                    }
                ]
            ),
        ]
    )

    play_args = [None, {"frame": {"duration": args.cadence_ms, "redraw": True}, "fromcurrent": False,
                        "transition": {"duration": 0}}]
    pause_args = [None, {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]

    menu_anim = dict(
        type="buttons", direction="left", x=0.0, y=1.15, xanchor="left", yanchor="top",
        visible=False, showactive=False,
        buttons=[
            dict(label="\u25b6 Play", method="animate", args=play_args),
            dict(label="\u23f8 Pause", method="animate", args=pause_args),
            dict(
                label="Exit animation",
                method="update",
                args=[
                    {"visible": all_true_default},
                    {
                        "updatemenus[0].visible": True,
                        "updatemenus[1].visible": False,
                        "sliders[0].visible": False,
                    }
                ]
            ),
        ]
    )

    slider = dict(
        visible=False, active=0, currentvalue={"prefix": "Phase: "},
        steps=[{"label": f.name, "method": "animate",
                "args": [[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0}}]}
               for f in fig.frames]
    )

    fig.update_layout(
        title=f"{args.star_name} \u2014 phase-ordered spectra (P={P:.2f} d, e={ecc:.3f})",
        hovermode="closest",
        legend_title="Phase (epoch)",
        legend=dict(font=dict(family="monospace")),
        uirevision="keep",
        updatemenus=[menu_default, menu_anim],
        sliders=[slider],
        margin=dict(l=60, r=20, t=90, b=60),
    )
    fig.update_xaxes(title_text="Wavelength", row=1, col=1)
    fig.update_yaxes(title_text="Flux", row=1, col=1)
    fig.update_xaxes(title_text="Phase", row=1, col=2)
    fig.update_yaxes(title_text="RV [km/s]", row=1, col=2)

    out_path = Path(args.out) if args.out else Path(f"{args.star_name}_spectra_phase.html")
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True, auto_play=False)
    print(f"[OK] Wrote interactive plot to: {out_path.resolve()}")
    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
