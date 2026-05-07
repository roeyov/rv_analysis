#!/usr/bin/env python3
"""
cloud.analyze_results — Analyze simulation pipeline results vs truth.

Produces diagnostic plots comparing recovered orbital parameters against
injected truth values, plus detection statistics.

Usage:
    python -m cloud.analyze_results --run-id 20240101_120000

Plots produced:
    1. Parameter recovery scatter (P, e, K1)
    2. Detection fraction vs Period, K1, Eccentricity
    3. Residual histograms (ΔP/P, Δe, ΔK1/K1)
    4. Detection maps: Period-K1, Period-Ecc (scatter)
    5. Detection probability density maps: P-K1, P-e, e-K1 (2D binned)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import glob as globmod

from mcmc.selector_app import get_best_row
from orbital.plotting import (
    compute_rv_curve_time,
    plot_time_series_with_residuals_plotly,
    plot_phase_folded_with_residuals_plotly,
)


LOCAL_OUTPUT = "/tmp/cloud_results"
DEFAULT_YAML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "params_cloud.yaml")


def load_results(run_dir):
    """Load the joined results+truth CSV."""
    joined_path = os.path.join(run_dir, "results_with_truth.csv")
    if not os.path.exists(joined_path):
        print(f"ERROR: {joined_path} not found. Run collect_results first.")
        sys.exit(1)
    return pd.read_csv(joined_path)


def _apply_filter(group, filter_expr, field_to_check):
    """Apply the same filter chain as ``get_best_row`` but return the
    surviving DataFrame (without the final min/max pick).

    Returns *None* if no rows survive.
    """
    fdf = group.copy()

    # Null-hyp reference
    null_field = None
    if field_to_check in fdf.columns and "candidate_method" in fdf.columns:
        null_mask = fdf["candidate_method"].str.contains("null_hyp_jitter", na=False)
        if null_mask.any():
            null_field = fdf.loc[null_mask, field_to_check].iloc[0]

    # User filter expression
    if filter_expr is not None and filter_expr.strip():
        try:
            clean_expr = " ".join(filter_expr.splitlines())
            fdf = fdf.query(clean_expr, engine="python")
        except Exception:
            return None

    if fdf.empty:
        return None

    # Field-based extra filtering
    if field_to_check not in fdf.columns:
        return None
    if field_to_check == "bic" and null_field is not None:
        fdf = fdf[fdf[field_to_check] < null_field]
    fdf = fdf[fdf[field_to_check].notna()]

    return fdf if not fdf.empty else None


def get_best_detections_standard(df, filter_expr, field_to_check, take_min):
    """Select the best detection row per simulation using **only** the
    filter expression + field optimisation (no distance preference).

    This is the baseline selection strategy.
    """
    best_rows = []
    for sim_id, group in df.groupby("sim_id"):
        best = get_best_row(group, filter_expr, field_to_check, take_min)
        if best is None:
            continue
        best_rows.append(best)
    if not best_rows:
        return pd.DataFrame()
    return pd.DataFrame(best_rows).reset_index(drop=True)


def get_best_detections(df, filter_expr, field_to_check, take_min):
    """Select the best detection row per simulation.

    For each sim_id the selection rule is:

    1. Apply *filter_expr* + field logic (same as ``get_best_row``).
    2. Among the rows that survive the filter, check whether the
       ``closest_to_truth`` row (lowest ``truth_distance``) is present.
       If so, prefer it over the min/max-field row — this way the oracle
       closest-to-truth solution is chosen whenever the filter would have
       accepted it anyway.
    3. Otherwise fall back to the standard min/max-field pick.
    """
    best_rows = []
    for sim_id, group in df.groupby("sim_id"):
        # Standard filter-based pick
        best = get_best_row(group, filter_expr, field_to_check, take_min)
        if best is None:
            continue

        # Check if the closest-to-truth row also passes the filter.
        # We replicate the filter logic on the group to get the surviving
        # index set, then check membership.
        closest_mask = group["closest_to_truth"]
        if closest_mask.any():
            closest_idx = closest_mask.idxmax()  # index of the True row
            # Build the same filtered subset that get_best_row produces
            filtered = _apply_filter(group, filter_expr, field_to_check)
            if filtered is not None and closest_idx in filtered.index:
                best = group.loc[closest_idx]

        best_rows.append(best)

    if not best_rows:
        return pd.DataFrame()

    return pd.DataFrame(best_rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Truth-distance scoring & chosen/closest marking
# ---------------------------------------------------------------------------

def _truth_distance_score(row):
    """Combined period + eccentricity distance to identify the closest solution.

    Score = |ΔP/P| + |Δe|

    Returns ``np.inf`` when Period truth is missing / zero (unusable row).
    """
    P_true, P_fit = row.get("Period"), row.get("Period_value")
    if not (pd.notna(P_true) and pd.notna(P_fit) and P_true > 0):
        return np.inf

    score = abs((P_fit - P_true) / P_true)

    e_true, e_fit = row.get("Eccentricity"), row.get("Eccentricity_value")
    if pd.notna(e_true) and pd.notna(e_fit):
        score += abs(e_fit - e_true)

    return score


def mark_chosen_and_closest(df, filter_expr, field_to_check, take_min):
    """Add boolean marker columns identifying the 'best' row per simulation.

    Columns added to *df* (in-place):

    chosen_filter
        True for the row selected by the YAML filter expression + BIC logic
        (same criteria used by ``get_best_detections``).
    closest_to_truth
        True for the row with the lowest weighted truth-distance score
        among non-null-hypothesis candidates.
    truth_distance
        The composite score itself (lower = closer to truth).
    """
    df["chosen_filter"] = False
    df["closest_to_truth"] = False
    df["truth_distance"] = np.nan

    # Compute truth distance for every non-null-hypothesis row
    non_null = ~df["candidate_method"].str.contains("null_hyp", na=False)
    if non_null.any():
        df.loc[non_null, "truth_distance"] = df.loc[non_null].apply(
            _truth_distance_score, axis=1
        )

    for sim_id, group in df.groupby("sim_id"):
        # 1. Chosen by filter expression + field optimisation
        best = get_best_row(group, filter_expr, field_to_check, take_min)
        if best is not None:
            df.loc[best.name, "chosen_filter"] = True

        # 2. Closest to injected truth (excluding null-hypothesis rows)
        cands = group[~group["candidate_method"].str.contains("null_hyp", na=False)]
        valid_dist = cands["truth_distance"].dropna()
        if not valid_dist.empty:
            df.loc[valid_dist.idxmin(), "closest_to_truth"] = True

    return df


def plot_parameter_recovery(det, truth_col, fit_col, label, logscale, ax):
    """Scatter plot of recovered vs true parameter with 1:1 line."""
    x = det[truth_col].values
    y = det[fit_col].values
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    ax.scatter(x, y, alpha=0.5, s=20, edgecolors="none")

    # 1:1 line
    lo = min(x.min(), y.min()) if len(x) > 0 else 0
    hi = max(x.max(), y.max()) if len(x) > 0 else 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="1:1")

    ax.set_xlabel(f"{label} (true)")
    ax.set_ylabel(f"{label} (recovered)")
    ax.set_title(f"{label} Recovery")

    if logscale and len(x) > 0:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.legend(fontsize=8)


def plot_detection_fraction(df, truth_col, label, n_bins, ax, det_sim_ids, logscale=False):
    """Detection fraction as a function of a truth parameter."""
    vals = df[truth_col].dropna()
    if logscale:
        vals = np.log10(vals)
        xlabel = f"log10({label})"
    else:
        xlabel = label

    bins = np.linspace(vals.min(), vals.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # "Detected" = sim_id is in the set selected by get_best_detections
    detected = df["sim_id"].isin(det_sim_ids)

    frac = []
    for i in range(n_bins):
        if logscale:
            in_bin = (np.log10(df[truth_col]) >= bins[i]) & (np.log10(df[truth_col]) < bins[i + 1])
        else:
            in_bin = (df[truth_col] >= bins[i]) & (df[truth_col] < bins[i + 1])
        total = in_bin.sum()
        det_count = (in_bin & detected).sum()
        frac.append(det_count / total if total > 0 else np.nan)

    ax.bar(bin_centers, frac, width=np.diff(bins), alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Detection Fraction")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.set_title(f"Detection Fraction vs {label}")


def plot_residual_histogram(det, truth_col, fit_col, label, fractional, ax):
    """Histogram of residuals (fit - true) or fractional residuals."""
    true_vals = det[truth_col].values
    fit_vals = det[fit_col].values
    valid = np.isfinite(true_vals) & np.isfinite(fit_vals) & (true_vals != 0)
    true_vals, fit_vals = true_vals[valid], fit_vals[valid]

    if fractional and len(true_vals) > 0:
        resid = (fit_vals - true_vals) / true_vals
        xlabel = f"Δ{label}/{label}"
    else:
        resid = fit_vals - true_vals
        xlabel = f"Δ{label}"

    if len(resid) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.hist(resid, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    median = np.median(resid)
    mad = np.median(np.abs(resid - median))
    ax.set_title(f"{label}: median={median:.3f}, MAD={mad:.3f}")


def plot_detection_map(truth_df, det_sim_ids, xcol, ycol, xlabel, ylabel, ax,
                       xlog=False, ylog=False):
    """Scatter plot of detected vs missed on a 2D parameter plane."""
    detected_mask = truth_df["sim_id"].isin(det_sim_ids)
    missed = truth_df[~detected_mask]
    found = truth_df[detected_mask]

    ax.scatter(missed[xcol], missed[ycol],
               c="red", alpha=0.4, s=30, label=f"Missed ({len(missed)})",
               edgecolors="none")
    ax.scatter(found[xcol], found[ycol],
               c="green", alpha=0.6, s=30, label=f"Detected ({len(found)})",
               edgecolors="none")

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def plot_detection_density(truth_df, det_sim_ids, xcol, ycol, xlabel, ylabel,
                           ax, nx=15, ny=15, xlog=False, ylog=False):
    """2D binned detection probability map (n_detected / n_total per bin).

    Bins are uniform in the displayed space (log if xlog/ylog).
    """
    xvals = truth_df[xcol].values.copy()
    yvals = truth_df[ycol].values.copy()
    detected = truth_df["sim_id"].isin(det_sim_ids).values

    # Work in log-space for log axes
    if xlog:
        xvals = np.log10(xvals)
    if ylog:
        yvals = np.log10(yvals)

    valid = np.isfinite(xvals) & np.isfinite(yvals)
    xvals, yvals, detected = xvals[valid], yvals[valid], detected[valid]

    if len(xvals) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    xedges = np.linspace(xvals.min(), xvals.max(), nx + 1)
    yedges = np.linspace(yvals.min(), yvals.max(), ny + 1)

    # Count total and detected per bin
    total, _, _ = np.histogram2d(xvals, yvals, bins=[xedges, yedges])
    det_count, _, _ = np.histogram2d(xvals[detected], yvals[detected],
                                     bins=[xedges, yedges])

    with np.errstate(invalid="ignore"):
        frac = np.where(total > 0, det_count / total, np.nan)

    # Plot as image (transpose because histogram2d returns [x, y])
    if xlog:
        xedges = 10**xedges
    if ylog:
        yedges = 10**yedges

    im = ax.pcolormesh(xedges, yedges, frac.T, cmap="RdYlGn",
                       vmin=0, vmax=1, shading="flat")
    cbar = plt.colorbar(im, ax=ax, label="Detection Probability")

    # Annotate bins with counts where total > 0
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    for i in range(nx):
        for j in range(ny):
            if total[i, j] > 0:
                txt = f"{int(det_count[i, j])}/{int(total[i, j])}"
                ax.text(xc[i], yc[j], txt, ha="center", va="center",
                        fontsize=5, color="black", alpha=0.7)

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# ---------------------------------------------------------------------------
# Orbital-fit plots (phase-folded + time-series) from CSV columns
# ---------------------------------------------------------------------------

def _parse_sim_ids(csv_string):
    """Parse a comma-separated string of sim IDs into a set of ints.

    Examples: ``"0042"`` → {42},  ``"0000,0001,0002"`` → {0, 1, 2}
    """
    if csv_string is None:
        return None
    return {int(tok.strip()) for tok in csv_string.split(",") if tok.strip()}


def _find_rv_csv(run_dir, sim_id):
    """Locate the simulated RV CSV for a given sim_id inside the run directory."""
    sim_tag = f"SIMuLaTioN_{int(sim_id):06d}"
    # Standard cloud layout: output/SIMuLaTioN_NNNNNN/SIMuLaTioN_NNNNNN_CCF_RVs.csv
    pattern = os.path.join(run_dir, "**", f"{sim_tag}_CCF_RVs.csv")
    hits = globmod.glob(pattern, recursive=True)
    return hits[0] if hits else None


def _plot_one_solution(hjds, vels, errs, P, T0, Gamma, K, Omega, ecc,
                       star_name, solution_id, jitter, out_dir):
    """Plot time-series + phase-folded for a single set of orbital parameters."""
    time_grid, rv_model = compute_rv_curve_time(hjds, P, T0, ecc, Gamma, K, Omega)
    plot_time_series_with_residuals_plotly(
        hjds, vels, errs, time_grid, rv_model,
        Gamma, K, Omega, ecc, P,
        star_name, solution_id=solution_id, jitter=jitter,
        out_dir=out_dir,
    )
    plot_phase_folded_with_residuals_plotly(
        hjds, vels, errs, P, T0,
        Gamma, K, Omega, ecc,
        star_name, solution_id=solution_id, jitter=jitter,
        out_dir=out_dir,
    )


def plot_orbital_fits(df, run_dir, out_dir, sim_id_filter=None):
    """Generate phase-folded and time-series Plotly plots for every candidate
    solution of the selected simulations, plus a separate plot from the
    injected truth parameters.

    Output is organised into per-star subdirectories::

        out_dir/{star_name}/
            {star}_sid-0_time_residuals.html
            {star}_sid-0_phase_residuals.html
            ...
            {star}_sid-truth_time_residuals.html   ← truth
            {star}_sid-truth_phase_residuals.html

    Parameters
    ----------
    df : DataFrame
        Full results_with_truth table (all rows, all candidate solutions).
    run_dir : str
        Root of the downloaded run directory (used to locate RV CSVs).
    out_dir : str
        Base directory for per-star plot subdirectories.
    sim_id_filter : set or None
        If given, only plot simulations whose sim_id is in this set.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_stars = 0

    for sim_id, group in df.groupby("sim_id"):
        sim_id = int(sim_id)
        if sim_id_filter is not None and sim_id not in sim_id_filter:
            continue

        # --- Locate the raw RV CSV ---
        csv_path = _find_rv_csv(run_dir, sim_id)
        if csv_path is None:
            print(f"  [sim {sim_id}] RV CSV not found, skipping.")
            continue

        rv_data = pd.read_csv(csv_path)
        hjds = rv_data["MJD"].values
        vels = rv_data["Mean RV"].values
        errs = np.abs(rv_data["Mean RVsig"].values)

        # Determine star name and create per-star output dir
        star_name = str(group["star_name"].iloc[0]) if "star_name" in group.columns else f"SIM_{sim_id:06d}"
        star_dir = os.path.join(out_dir, star_name)
        os.makedirs(star_dir, exist_ok=True)

        # --- Plot every non-null candidate solution ---
        candidates = group[~group["candidate_method"].str.contains("null_hyp", na=False)]
        n_sol = 0
        for _, row in candidates.iterrows():
            P     = row.get("Period_value")
            T0    = row.get("T0_value", row.get("T_value"))
            Gamma = row.get("GAMMA_value")
            K     = row.get("K1_value")
            Omega = row.get("OMEGA_rad_value")
            ecc   = row.get("Eccentricity_value")

            if any(pd.isna(v) for v in (P, T0, Gamma, K, Omega, ecc)):
                continue

            ln_sj = row.get("ln_sigmaJ_value", np.nan)
            jitter = float(np.exp(ln_sj)) if pd.notna(ln_sj) else 0.0
            solution_id = int(row.get("solution_id", n_sol))

            _plot_one_solution(hjds, vels, errs, P, T0, Gamma, K, Omega, ecc,
                               star_name, solution_id, jitter, star_dir)
            n_sol += 1

        # --- Plot the injected truth as its own "solution" ---
        first = group.iloc[0]
        P_true     = first.get("Period")
        T0_true    = first.get("T0")
        Gamma_true = first.get("GAMMA")
        K_true     = first.get("K1")
        Omega_true = first.get("OMEGA_rad")
        ecc_true   = first.get("Eccentricity")

        if all(pd.notna(v) for v in (P_true, T0_true, Gamma_true, K_true, Omega_true, ecc_true)):
            _plot_one_solution(hjds, vels, errs,
                               P_true, T0_true, Gamma_true, K_true, Omega_true, ecc_true,
                               star_name + "_TRUTH", "truth", 0.0, star_dir)

        print(f"  [sim {sim_id}] {star_name}: {n_sol} solutions + truth → {star_dir}")
        n_stars += 1

    print(f"Orbital-fit plots generated for {n_stars} simulations → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze simulation results vs truth.")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--output-dir", default=LOCAL_OUTPUT, help="Base output directory")
    parser.add_argument("--format", default="png", choices=["png", "pdf"],
                        help="Plot format (default: png)")
    parser.add_argument("--config", default=DEFAULT_YAML,
                        help=f"YAML config file (default: {DEFAULT_YAML})")
    parser.add_argument("--plot-orbits", action="store_true",
                        help="Generate phase-folded and time-series plots for detected solutions")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip diagnostic plot generation")
    parser.add_argument("--sim-ids", default=None,
                        help="Comma-separated sim IDs to plot, e.g. '0042' or '0000,0001,0002'")
    args = parser.parse_args()

    # Load detection criteria from YAML mcmc_params section
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mcmc_cfg = cfg.get("mcmc_params", {})
    filter_expr = mcmc_cfg.get("filter_expression", None)
    field_to_check = mcmc_cfg.get("field_to_check", "bic")
    take_min = mcmc_cfg.get("take_min", True)

    print(f"Detection criteria (from {os.path.basename(args.config)}):")
    print(f"  filter:  {filter_expr}")
    print(f"  field:   {field_to_check}  (min={take_min})")

    run_dir = os.path.join(args.output_dir, args.run_id)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    df = load_results(run_dir)
    print(f"Loaded {len(df)} rows, {df['sim_id'].nunique()} unique simulations")

    # --- Mark chosen (by filter) and closest-to-truth rows per sim ---
    mark_chosen_and_closest(df, filter_expr, field_to_check, take_min)
    df.to_csv(os.path.join(run_dir, "results_with_truth.csv"), index=False)
    print("Annotated with chosen_filter & closest_to_truth → results_with_truth.csv")

    # Get best detections — two strategies:
    #   det_std  = standard filter + min/max field (baseline)
    #   det_dist = prefer closest-to-truth when it passes the filter (oracle)
    det_std = get_best_detections_standard(df, filter_expr, field_to_check, take_min)
    det_dist = get_best_detections(df, filter_expr, field_to_check, take_min)
    det = det_dist  # default for detection-level plots (same sim set either way)
    print(f"Best-row detections (standard): {len(det_std)}")
    print(f"Best-row detections (distance):  {len(det_dist)}")

    # Also load truth for detection fraction (need all sims, not just detected)
    truth_path = os.path.join(run_dir, "orbital_params_truth.csv")
    if os.path.exists(truth_path):
        truth_df = pd.read_csv(truth_path)
        # Merge detection info onto truth
        det_ids = set(det["sim_id"].dropna().astype(int))
        truth_df["detected"] = truth_df["sim_id"].isin(det_ids)
    else:
        truth_df = None

    # =========================================================================
    # Diagnostic plots (skip with --no-plots)
    # =========================================================================
    if args.no_plots:
        print("Skipping diagnostic plots (--no-plots).")
    else:
        # =================================================================
        # Figure 1: Parameter recovery (2 rows × 3 cols)
        #   Top row:    standard selection (filter + min BIC)
        #   Bottom row: distance-based selection (prefer closest-to-truth)
        # =================================================================
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Top row — standard
        plot_parameter_recovery(det_std, "Period", "Period_value", "Period [d]", logscale=True, ax=axes[0, 0])
        plot_parameter_recovery(det_std, "Eccentricity", "Eccentricity_value", "Eccentricity", logscale=False, ax=axes[0, 1])
        plot_parameter_recovery(det_std, "K1", "K1_value", "K1 [km/s]", logscale=False, ax=axes[0, 2])
        for ax in axes[0]:
            ax.set_title("(Standard) " + ax.get_title(), fontsize=10)

        # Bottom row — distance-based
        plot_parameter_recovery(det_dist, "Period", "Period_value", "Period [d]", logscale=True, ax=axes[1, 0])
        plot_parameter_recovery(det_dist, "Eccentricity", "Eccentricity_value", "Eccentricity", logscale=False, ax=axes[1, 1])
        plot_parameter_recovery(det_dist, "K1", "K1_value", "K1 [km/s]", logscale=False, ax=axes[1, 2])
        for ax in axes[1]:
            ax.set_title("(Distance) " + ax.get_title(), fontsize=10)

        fig.suptitle("Orbital Parameter Recovery\nTop: Standard (filter+BIC)  —  Bottom: Distance-based",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"parameter_recovery.{args.format}"),
                    dpi=150, bbox_inches="tight")
        print(f"Saved: parameter_recovery.{args.format}")

        # =================================================================
        # Figure 2: Detection fraction (2 panels)
        # =================================================================
        if truth_df is not None:
            det_sim_ids = set(det["sim_id"].dropna().astype(int))

            fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))

            plot_detection_fraction(truth_df, "Period", "Period [d]", n_bins=10,
                                    ax=axes2[0], det_sim_ids=det_sim_ids, logscale=True)
            plot_detection_fraction(truth_df, "K1", "K1 [km/s]", n_bins=10,
                                    ax=axes2[1], det_sim_ids=det_sim_ids)
            plot_detection_fraction(truth_df, "Eccentricity", "Eccentricity", n_bins=10,
                                    ax=axes2[2], det_sim_ids=det_sim_ids)

            fig2.suptitle("Detection Fraction", fontsize=14, y=1.02)
            fig2.tight_layout()
            fig2.savefig(os.path.join(plots_dir, f"detection_fraction.{args.format}"),
                         dpi=150, bbox_inches="tight")
            print(f"Saved: detection_fraction.{args.format}")

        # =================================================================
        # Figure 3: Residual histograms (2 rows × 3 cols)
        #   Top row:    standard selection
        #   Bottom row: distance-based selection
        # =================================================================
        fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))

        # Top row — standard
        plot_residual_histogram(det_std, "Period", "Period_value", "P", fractional=True, ax=axes3[0, 0])
        plot_residual_histogram(det_std, "Eccentricity", "Eccentricity_value", "e", fractional=False, ax=axes3[0, 1])
        plot_residual_histogram(det_std, "K1", "K1_value", "K1", fractional=True, ax=axes3[0, 2])
        for ax in axes3[0]:
            ax.set_title("(Standard) " + ax.get_title(), fontsize=10)

        # Bottom row — distance-based
        plot_residual_histogram(det_dist, "Period", "Period_value", "P", fractional=True, ax=axes3[1, 0])
        plot_residual_histogram(det_dist, "Eccentricity", "Eccentricity_value", "e", fractional=False, ax=axes3[1, 1])
        plot_residual_histogram(det_dist, "K1", "K1_value", "K1", fractional=True, ax=axes3[1, 2])
        for ax in axes3[1]:
            ax.set_title("(Distance) " + ax.get_title(), fontsize=10)

        fig3.suptitle("Recovery Residuals\nTop: Standard (filter+BIC)  —  Bottom: Distance-based",
                      fontsize=14, y=1.02)
        fig3.tight_layout()
        fig3.savefig(os.path.join(plots_dir, f"residual_histograms.{args.format}"),
                     dpi=150, bbox_inches="tight")
        print(f"Saved: residual_histograms.{args.format}")

        # =================================================================
        # Figure 4: Detection maps (scatter) — Period-K1 and Period-Ecc
        # =================================================================
        if truth_df is not None:
            fig4, axes4 = plt.subplots(1, 3, figsize=(21, 6))

            plot_detection_map(truth_df, det_sim_ids,
                               "Period", "K1", "Period [d] (true)", "K1 [km/s] (true)",
                               ax=axes4[0], xlog=True)
            axes4[0].set_title("Detection Map: Period vs K1")

            plot_detection_map(truth_df, det_sim_ids,
                               "Period", "Eccentricity",
                               "Period [d] (true)", "Eccentricity (true)",
                               ax=axes4[1], xlog=True)
            axes4[1].set_title("Detection Map: Period vs Eccentricity")

            plot_detection_map(truth_df, det_sim_ids,
                               "Period", "MassRatio",
                               "Period [d] (true)", "q (true)",
                               ax=axes4[2], xlog=True)
            axes4[2].set_title("Detection Map: Period vs q")

            fig4.suptitle("Detection Maps", fontsize=14, y=1.02)
            fig4.tight_layout()
            fig4.savefig(os.path.join(plots_dir, f"detection_map.{args.format}"),
                         dpi=150, bbox_inches="tight")
            print(f"Saved: detection_map.{args.format}")

        # =================================================================
        # Figure 5: Detection probability density maps (2D binned)
        #   Period-K1, Period-Ecc, Ecc-K1
        # =================================================================
        if truth_df is not None:
            fig5, axes5 = plt.subplots(2, 2, figsize=(16, 12))

            plot_detection_density(truth_df, det_sim_ids,
                                   "Period", "K1",
                                   "Period [d]", "K1 [km/s]",
                                   ax=axes5[0, 0], nx=20, ny=20, xlog=True)
            axes5[0, 0].set_title("Detection Probability: Period vs K1")

            plot_detection_density(truth_df, det_sim_ids,
                                   "Period", "Eccentricity",
                                   "Period [d]", "Eccentricity",
                                   ax=axes5[0, 1], nx=20, ny=20, xlog=True)
            axes5[0, 1].set_title("Detection Probability: Period vs Ecc")

            plot_detection_density(truth_df, det_sim_ids,
                                   "Eccentricity", "K1",
                                   "Eccentricity", "K1 [km/s]",
                                   ax=axes5[1, 0], nx=20, ny=20)
            axes5[1, 0].set_title("Detection Probability: Ecc vs K1")

            plot_detection_density(truth_df, det_sim_ids,
                                   "Period", "MassRatio",
                                   "Period [d]", "q",
                                   ax=axes5[1, 1], nx=20, ny=20, xlog=True)
            axes5[1, 1].set_title("Detection Probability: Period vs q")

            fig5.suptitle("Detection Probability Density Maps", fontsize=14, y=1.02)
            fig5.tight_layout()
            fig5.savefig(os.path.join(plots_dir, f"detection_density.{args.format}"),
                         dpi=150, bbox_inches="tight")
            print(f"Saved: detection_density.{args.format}")

    # =========================================================================
    # Optional: orbital-fit plots (phase + time series)
    # =========================================================================
    if args.plot_orbits:
        sim_id_filter = _parse_sim_ids(args.sim_ids) if args.sim_ids else None
        orbit_plots_dir = os.path.join(plots_dir, "orbital_fits")
        plot_orbital_fits(df, run_dir, orbit_plots_dir, sim_id_filter=sim_id_filter)

    # =========================================================================
    # Summary statistics
    # =========================================================================
    print("\n" + "=" * 60)
    print("  Summary Statistics")
    print("=" * 60)

    n_total = df["sim_id"].nunique()
    print(f"  Total simulations:     {n_total}")
    print(f"  Detected (standard):   {len(det_std)}")
    print(f"  Detected (distance):   {len(det_dist)}")
    print(f"  Detection rate:        {len(det_std)/n_total:.1%}" if n_total > 0 else "  N/A")

    for label, det_set in [("STANDARD (filter+BIC)", det_std),
                            ("DISTANCE-BASED", det_dist)]:
        print(f"\n  --- {label} ---")
        if len(det_set) == 0:
            print("    No detections.")
            continue

        # Period recovery
        p_true = det_set["Period"].values
        p_fit = det_set["Period_value"].values
        valid_p = np.isfinite(p_true) & np.isfinite(p_fit) & (p_true > 0)
        if valid_p.sum() > 0:
            dp = (p_fit[valid_p] - p_true[valid_p]) / p_true[valid_p]
            print(f"    Period ΔP/P:        median={np.median(dp):.4f}, MAD={np.median(np.abs(dp - np.median(dp))):.4f}")

        # Eccentricity recovery
        e_true = det_set["Eccentricity"].values
        e_fit = det_set["Eccentricity_value"].values
        valid_e = np.isfinite(e_true) & np.isfinite(e_fit)
        if valid_e.sum() > 0:
            de = e_fit[valid_e] - e_true[valid_e]
            print(f"    Eccentricity Δe:    median={np.median(de):.4f}, MAD={np.median(np.abs(de - np.median(de))):.4f}")

        # K1 recovery
        k1_true = det_set["K1"].values
        k1_fit = det_set["K1_value"].values
        valid_k = np.isfinite(k1_true) & np.isfinite(k1_fit) & (k1_true > 0)
        if valid_k.sum() > 0:
            dk = (k1_fit[valid_k] - k1_true[valid_k]) / k1_true[valid_k]
            print(f"    K1 ΔK1/K1:          median={np.median(dk):.4f}, MAD={np.median(np.abs(dk - np.median(dk))):.4f}")

    # --- Chosen-filter vs closest-to-truth comparison ---
    n_chosen = df["chosen_filter"].sum()
    n_closest = df["closest_to_truth"].sum()
    n_agree = (df["chosen_filter"] & df["closest_to_truth"]).sum()

    print(f"\n  Chosen-filter solutions:       {n_chosen}")
    print(f"  Closest-to-truth solutions:    {n_closest}")
    print(f"  Same row (agree):              {n_agree}")
    print(f"  Different row (disagree):      {n_chosen - n_agree}")

    if n_chosen > 0:
        chosen_td = df.loc[df["chosen_filter"], "truth_distance"]
        closest_td = df.loc[df["closest_to_truth"], "truth_distance"]
        print(f"\n  Truth distance (chosen):   "
              f"median={chosen_td.median():.4f}, mean={chosen_td.mean():.4f}")
        print(f"  Truth distance (closest):  "
              f"median={closest_td.median():.4f}, mean={closest_td.mean():.4f}")

    print(f"\nPlots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
