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

from mcmc.selector_app import get_best_row


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


def get_best_detections(df, filter_expr, field_to_check, take_min):
    """Select the best detection row per simulation using YAML-defined criteria.

    Uses the same get_best_row logic as mcmc/selector_app.py:
      1. Apply filter_expression (pandas query)
      2. If field_to_check == 'bic', require bic < null_bic
      3. Pick row with min/max of field_to_check
    """
    best_rows = []
    for sim_id, group in df.groupby("sim_id"):
        row = get_best_row(group, filter_expr, field_to_check, take_min)
        if row is not None:
            best_rows.append(row)

    if not best_rows:
        return pd.DataFrame()

    return pd.DataFrame(best_rows).reset_index(drop=True)


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


def main():
    parser = argparse.ArgumentParser(description="Analyze simulation results vs truth.")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--output-dir", default=LOCAL_OUTPUT, help="Base output directory")
    parser.add_argument("--format", default="png", choices=["png", "pdf"],
                        help="Plot format (default: png)")
    parser.add_argument("--config", default=DEFAULT_YAML,
                        help=f"YAML config file (default: {DEFAULT_YAML})")
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

    # Get best detections using YAML criteria (same logic as mcmc/selector_app.py)
    det = get_best_detections(df, filter_expr, field_to_check, take_min)
    print(f"Best-row detections: {len(det)}")

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
    # Figure 1: Parameter recovery (3 panels)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_parameter_recovery(det, "Period", "Period_value", "Period [d]", logscale=True, ax=axes[0])
    plot_parameter_recovery(det, "Eccentricity", "Eccentricity_value", "Eccentricity", logscale=False, ax=axes[1])
    plot_parameter_recovery(det, "K1", "K1_value", "K1 [km/s]", logscale=False, ax=axes[2])

    fig.suptitle("Orbital Parameter Recovery", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"parameter_recovery.{args.format}"),
                dpi=150, bbox_inches="tight")
    print(f"Saved: parameter_recovery.{args.format}")

    # =========================================================================
    # Figure 2: Detection fraction (2 panels)
    # =========================================================================
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

    # =========================================================================
    # Figure 3: Residual histograms (3 panels)
    # =========================================================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

    plot_residual_histogram(det, "Period", "Period_value", "P", fractional=True, ax=axes3[0])
    plot_residual_histogram(det, "Eccentricity", "Eccentricity_value", "e", fractional=False, ax=axes3[1])
    plot_residual_histogram(det, "K1", "K1_value", "K1", fractional=True, ax=axes3[2])

    fig3.suptitle("Recovery Residuals", fontsize=14, y=1.02)
    fig3.tight_layout()
    fig3.savefig(os.path.join(plots_dir, f"residual_histograms.{args.format}"),
                 dpi=150, bbox_inches="tight")
    print(f"Saved: residual_histograms.{args.format}")

    # =========================================================================
    # Figure 4: Detection maps (scatter) — Period-K1 and Period-Ecc
    # =========================================================================
    if truth_df is not None:
        fig4, axes4 = plt.subplots(1, 2, figsize=(15, 6))

        plot_detection_map(truth_df, det_sim_ids,
                           "Period", "K1", "Period [d] (true)", "K1 [km/s] (true)",
                           ax=axes4[0], xlog=True)
        axes4[0].set_title("Detection Map: Period vs K1")

        plot_detection_map(truth_df, det_sim_ids,
                           "Period", "Eccentricity",
                           "Period [d] (true)", "Eccentricity (true)",
                           ax=axes4[1], xlog=True)
        axes4[1].set_title("Detection Map: Period vs Eccentricity")

        fig4.suptitle("Detection Maps", fontsize=14, y=1.02)
        fig4.tight_layout()
        fig4.savefig(os.path.join(plots_dir, f"detection_map.{args.format}"),
                     dpi=150, bbox_inches="tight")
        print(f"Saved: detection_map.{args.format}")

    # =========================================================================
    # Figure 5: Detection probability density maps (2D binned)
    #   Period-K1, Period-Ecc, Ecc-K1
    # =========================================================================
    if truth_df is not None:
        fig5, axes5 = plt.subplots(1, 3, figsize=(20, 6))

        plot_detection_density(truth_df, det_sim_ids,
                               "Period", "K1",
                               "Period [d]", "K1 [km/s]",
                               ax=axes5[0], xlog=True)
        axes5[0].set_title("Detection Probability: Period vs K1")

        plot_detection_density(truth_df, det_sim_ids,
                               "Period", "Eccentricity",
                               "Period [d]", "Eccentricity",
                               ax=axes5[1], xlog=True)
        axes5[1].set_title("Detection Probability: Period vs Ecc")

        plot_detection_density(truth_df, det_sim_ids,
                               "Eccentricity", "K1",
                               "Eccentricity", "K1 [km/s]",
                               ax=axes5[2])
        axes5[2].set_title("Detection Probability: Ecc vs K1")

        fig5.suptitle("Detection Probability Density Maps", fontsize=14, y=1.02)
        fig5.tight_layout()
        fig5.savefig(os.path.join(plots_dir, f"detection_density.{args.format}"),
                     dpi=150, bbox_inches="tight")
        print(f"Saved: detection_density.{args.format}")

    # =========================================================================
    # Summary statistics
    # =========================================================================
    print("\n" + "=" * 60)
    print("  Summary Statistics")
    print("=" * 60)

    n_total = df["sim_id"].nunique()
    n_detected = len(det)
    print(f"  Total simulations:     {n_total}")
    print(f"  Detected (best row):   {n_detected}")
    print(f"  Detection rate:        {n_detected/n_total:.1%}" if n_total > 0 else "  N/A")

    if len(det) > 0:
        # Period recovery
        p_true = det["Period"].values
        p_fit = det["Period_value"].values
        valid_p = np.isfinite(p_true) & np.isfinite(p_fit) & (p_true > 0)
        if valid_p.sum() > 0:
            dp = (p_fit[valid_p] - p_true[valid_p]) / p_true[valid_p]
            print(f"\n  Period ΔP/P:")
            print(f"    Median:  {np.median(dp):.4f}")
            print(f"    MAD:     {np.median(np.abs(dp - np.median(dp))):.4f}")

        # Eccentricity recovery
        e_true = det["Eccentricity"].values
        e_fit = det["Eccentricity_value"].values
        valid_e = np.isfinite(e_true) & np.isfinite(e_fit)
        if valid_e.sum() > 0:
            de = e_fit[valid_e] - e_true[valid_e]
            print(f"\n  Eccentricity Δe:")
            print(f"    Median:  {np.median(de):.4f}")
            print(f"    MAD:     {np.median(np.abs(de - np.median(de))):.4f}")

        # K1 recovery
        k1_true = det["K1"].values
        k1_fit = det["K1_value"].values
        valid_k = np.isfinite(k1_true) & np.isfinite(k1_fit) & (k1_true > 0)
        if valid_k.sum() > 0:
            dk = (k1_fit[valid_k] - k1_true[valid_k]) / k1_true[valid_k]
            print(f"\n  K1 ΔK1/K1:")
            print(f"    Median:  {np.median(dk):.4f}")
            print(f"    MAD:     {np.median(np.abs(dk - np.median(dk))):.4f}")

    print(f"\nPlots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
