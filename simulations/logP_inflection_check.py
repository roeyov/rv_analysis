"""
simulations.logP_inflection_check
=================================

Diagnostic for picking a low-period cutoff for the bias-grid period CDF
goodness-of-fit. The observed CDF over log10(P/d) has an inflection at low
periods (likely real attrition of short-period systems via mergers /
common-envelope), which is not in the simulated power-law model and biases
the inferred period exponent.

This script compares two algorithmic methods for locating that inflection:

1. **poly3**: fit a cubic polynomial to the empirical CDF and take the
   inflection of the cubic as the cutoff (`x* = -c2 / (3 * c3)`).
2. **numerical**: smooth the empirical CDF with a Gaussian kernel and find
   the first logP where the second derivative crosses from positive to
   negative (the elbow of the CDF). Repeated for several smoothing widths
   to expose σ-sensitivity.

For each candidate cutoff it also fits a power-law to the **conditional**
empirical CDF (P >= cutoff, re-normalized) and reports the best-fit
exponent α plus a KS p-value against the fitted power-law — this is what
tells us whether the cutoff actually yields a power-law-like residual.

Outputs a single PDF with four stacked panels under
`$SCRIPTS_OUT/simulation_pipeline/inflection_check/`.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from scipy.stats import ks_1samp

from simulations.bias_grid import (
    load_observed_from_tex,
    _numerical_logP_cutoff,
)
from simulations.bias_config import DEFAULT_BIAS_CFG

DEFAULT_SIGMAS = (0.05, 0.10, 0.15, 0.20)


def empirical_cdf(x):
    xs = np.sort(np.asarray(x, dtype=float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def poly3_inflection(x, y):
    """Fit a cubic to (x, y) and return its inflection point.

    Returns (x_inf, coeffs_natural_basis) where coeffs = [c0, c1, c2, c3]
    so that y ≈ c0 + c1*x + c2*x^2 + c3*x^3. Inflection at -c2 / (3*c3);
    returns np.nan if |c3| is numerically zero.
    """
    poly = np.polynomial.Polynomial.fit(x, y, 3).convert()
    c = poly.coef
    while len(c) < 4:
        c = np.append(c, 0.0)
    c0, c1, c2, c3 = c[:4]
    if abs(c3) < 1e-8:
        return np.nan, (c0, c1, c2, c3)
    return -c2 / (3.0 * c3), (c0, c1, c2, c3)


def numerical_inflection(x, y, sigma_dex, n_grid=2000):
    """Diagnostic-only variant: returns cutoff + the intermediate arrays.

    The cutoff value itself is computed by the shared
    `_numerical_logP_cutoff` in bias_grid.py so the diagnostic and the
    production scorer cannot drift apart. This wrapper additionally
    returns the smoothed CDF and its 2nd derivative on a uniform grid,
    which the diagnostic plot needs for panel C.
    """
    obs_logP = np.asarray(x, dtype=float)
    cutoff = _numerical_logP_cutoff(obs_logP, smooth_sigma=sigma_dex,
                                    n_grid=n_grid)
    x_grid = np.linspace(obs_logP.min(), obs_logP.max(), n_grid)
    cdf_interp = np.interp(x_grid, x, y)
    dx = x_grid[1] - x_grid[0]
    smoothed = gaussian_filter1d(cdf_interp, sigma=sigma_dex / dx,
                                 mode="nearest")
    d1 = np.gradient(smoothed, dx)
    d2 = np.gradient(d1, dx)
    # `_numerical_logP_cutoff` returns 0.0 when there's no crossing — map
    # that back to NaN here so the diagnostic plot can mark it as
    # "no crossing" instead of placing a vertical line at logP=0.
    if cutoff == 0.0:
        cutoff = np.nan
    return cutoff, x_grid, smoothed, d2


def powerlaw_cdf(x, alpha, xmin, xmax):
    """CDF of p(x) ∝ x^alpha on [xmin, xmax]. Matches simulations.bias_grid.

    Note: matches the convention used by `powerlaw_draw` in bias_grid.py,
    where the variable is logP (not P) and the exponent α is applied to
    logP. So the same function describes both the diagnostic fit and the
    bias-grid simulation draw.
    """
    a = alpha + 1.0
    if abs(a) < 1e-8:
        return np.log(x / xmin) / np.log(xmax / xmin)
    return (x ** a - xmin ** a) / (xmax ** a - xmin ** a)


def fit_alpha_to_conditional_cdf(x_kept, xmin, xmax):
    """Least-squares α for the empirical CDF of `x_kept` on [xmin, xmax]."""
    xs, ys = empirical_cdf(x_kept)

    def loss(alpha):
        model = powerlaw_cdf(xs, alpha, xmin, xmax)
        return float(np.sum((ys - model) ** 2))

    res = minimize_scalar(loss, bounds=(-5.0, 5.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def ks_pvalue_against_powerlaw(x_kept, alpha, xmin, xmax):
    """KS goodness-of-fit p-value for x_kept vs the fitted power-law CDF."""
    if len(x_kept) < 3:
        return np.nan
    cdf = lambda v: powerlaw_cdf(np.asarray(v, dtype=float), alpha, xmin, xmax)
    return float(ks_1samp(x_kept, cdf).pvalue)


def evaluate_cutoff(obs_logP, cutoff, xmax):
    """Return (n_kept, n_total, α_fit, ks_p) for a candidate cutoff."""
    n_total = len(obs_logP)
    if not np.isfinite(cutoff):
        return 0, n_total, np.nan, np.nan
    # Guard a degenerate xmin: powerlaw_cdf needs xmin > 0 strictly. obs_logP
    # in days has values ~0.4-3.5 so this is generally safe, but clamp anyway.
    xmin = max(float(cutoff), 1e-3)
    keep = obs_logP >= cutoff
    n_kept = int(keep.sum())
    if n_kept < 3:
        return n_kept, n_total, np.nan, np.nan
    x_kept = obs_logP[keep]
    alpha = fit_alpha_to_conditional_cdf(x_kept, xmin, xmax)
    ks_p = ks_pvalue_against_powerlaw(x_kept, alpha, xmin, xmax)
    return n_kept, n_total, alpha, ks_p


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sb1", default=DEFAULT_BIAS_CFG["sb1_tex"],
                   help="Path to sb1_solutions.tex")
    p.add_argument("--sb2", default=DEFAULT_BIAS_CFG["sb2_tex"],
                   help="Path to sb2_solutions.tex")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (defaults to "
                        "$SCRIPTS_OUT/simulation_pipeline/inflection_check)")
    p.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS),
                   help="Gaussian smoothing widths (dex) for the numerical method")
    return p.parse_args()


def resolve_out_dir(cli_out_dir):
    if cli_out_dir is not None:
        return cli_out_dir
    base = os.environ.get(
        "SCRIPTS_OUT",
        "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut")
    return os.path.join(base, "simulation_pipeline", "inflection_check")


def main():
    args = parse_args()
    out_dir = resolve_out_dir(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    obs = load_observed_from_tex(args.sb1, args.sb2)
    obs_logP = obs["logP"]
    xs, ys = empirical_cdf(obs_logP)
    xmax_global = float(xs.max())

    x_poly, poly_coeffs = poly3_inflection(xs, ys)

    numerical_results = []  # list of (sigma, cutoff, x_grid, smoothed, d2)
    for sigma in args.sigmas:
        cutoff, x_grid, smoothed, d2 = numerical_inflection(xs, ys, sigma)
        numerical_results.append((sigma, cutoff, x_grid, smoothed, d2))

    rows = []
    n_kept_poly, n_total, a_poly, ks_poly = evaluate_cutoff(
        obs_logP, x_poly, xmax_global)
    rows.append(("poly3", None, x_poly, n_kept_poly, n_total, a_poly, ks_poly))
    for sigma, cutoff, _, _, _ in numerical_results:
        n_kept, _, a_fit, ks_p = evaluate_cutoff(obs_logP, cutoff, xmax_global)
        rows.append(("numerical", sigma, cutoff, n_kept, n_total, a_fit, ks_p))

    # ---- stdout table -----------------------------------------------------
    print()
    print(f"Observed sample: {len(obs_logP)} systems  "
          f"(SB1={obs['n_sb1']}, SB2={obs['n_sb2']})")
    print(f"  logP range: [{xs.min():.3f}, {xs.max():.3f}]  "
          f"(P range: [{10**xs.min():.2f}, {10**xs.max():.0f}] d)")
    print(f"  poly3 coefficients (natural basis, [c0..c3]): "
          f"{[f'{c:+.4g}' for c in poly_coeffs]}")
    print()
    header = f"{'method':<10} {'σ':>6} {'cut_logP':>10} {'cut_P_d':>10} " \
             f"{'n_kept/n':>11} {'α_fit':>9} {'KS_p':>8}"
    print(header)
    print("-" * len(header))
    for method, sigma, cutoff, n_kept, n_total, a_fit, ks_p in rows:
        sigma_str = "-" if sigma is None else f"{sigma:.2f}"
        cut_str = f"{cutoff:.3f}" if np.isfinite(cutoff) else "nan"
        cutP_str = f"{10**cutoff:.2f}" if np.isfinite(cutoff) else "nan"
        a_str = f"{a_fit:+.3f}" if np.isfinite(a_fit) else "nan"
        ks_str = f"{ks_p:.3f}" if np.isfinite(ks_p) else "nan"
        print(f"{method:<10} {sigma_str:>6} {cut_str:>10} {cutP_str:>10} "
              f"{n_kept:>4d}/{n_total:<4d}  {a_str:>9} {ks_str:>8}")
    print()

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 11.5),
                             gridspec_kw={"hspace": 0.35})

    # Panel A: full empirical CDF with cubic + one smoothed overlay
    axA = axes[0]
    axA.step(xs, ys, where="post", color="k", lw=2,
             label=f"Empirical CDF (n={len(obs_logP)})")

    # Cubic curve (over the obs range)
    x_dense = np.linspace(xs.min(), xs.max(), 400)
    c0, c1, c2, c3 = poly_coeffs
    y_cubic = c0 + c1 * x_dense + c2 * x_dense ** 2 + c3 * x_dense ** 3
    axA.plot(x_dense, y_cubic, color="C1", lw=1.5, ls="-",
             label=f"Cubic fit (infl @ logP={x_poly:.3f})")

    # One smoothed curve (use the middle σ for visual clarity)
    mid = len(numerical_results) // 2
    sigma_mid, cut_mid, xg_mid, sm_mid, _ = numerical_results[mid]
    axA.plot(xg_mid, sm_mid, color="C2", lw=1.5, ls="--",
             label=f"Smoothed CDF (σ={sigma_mid:.2f})")

    if np.isfinite(x_poly):
        axA.axvline(x_poly, color="C1", lw=1.2, ls=":",
                    label=f"poly3 cutoff = {x_poly:.3f}")
    for sigma, cutoff, _, _, _ in numerical_results:
        if np.isfinite(cutoff):
            axA.axvline(cutoff, color="C2", lw=0.8, ls=":", alpha=0.7)

    axA.set_xlabel(r"$\log_{10}(P/\mathrm{d})$")
    axA.set_ylabel("CDF")
    axA.set_title("Panel A — observed empirical CDF + inflection candidates")
    axA.set_ylim(-0.02, 1.05)
    axA.legend(loc="lower right", fontsize=8)
    axA.grid(alpha=0.3)

    # Panel B: histogram of logP
    axB = axes[1]
    axB.hist(obs_logP, bins=20, color="0.55", edgecolor="k", alpha=0.85)
    if np.isfinite(x_poly):
        axB.axvline(x_poly, color="C1", lw=1.5, ls="--",
                    label=f"poly3 = {x_poly:.3f}")
    for sigma, cutoff, _, _, _ in numerical_results:
        if np.isfinite(cutoff):
            axB.axvline(cutoff, color="C2", lw=0.9, ls=":", alpha=0.7,
                        label=f"num σ={sigma:.2f} → {cutoff:.3f}")
    axB.set_xlabel(r"$\log_{10}(P/\mathrm{d})$")
    axB.set_ylabel("count / bin")
    axB.set_title("Panel B — observed logP histogram (PDF context)")
    axB.legend(loc="upper right", fontsize=7, ncol=2)
    axB.grid(alpha=0.3)

    # Panel C: numerical 2nd derivative for each σ
    axC = axes[2]
    cmap = plt.get_cmap("viridis")
    for i, (sigma, cutoff, x_grid, _, d2) in enumerate(numerical_results):
        color = cmap(i / max(len(numerical_results) - 1, 1))
        axC.plot(x_grid, d2, color=color, lw=1.4,
                 label=f"σ={sigma:.2f}  (cut={cutoff:.3f})"
                       if np.isfinite(cutoff) else f"σ={sigma:.2f}  (no cross)")
        if np.isfinite(cutoff):
            axC.axvline(cutoff, color=color, lw=0.7, ls=":", alpha=0.6)
    axC.axhline(0.0, color="k", lw=0.8, ls="-")
    axC.set_xlabel(r"$\log_{10}(P/\mathrm{d})$")
    axC.set_ylabel(r"$d^2\mathrm{CDF}/d(\log_{10}P)^2$")
    axC.set_title("Panel C — smoothed 2nd derivative (zero-crossing = elbow)")
    axC.legend(loc="upper right", fontsize=7)
    axC.grid(alpha=0.3)

    # Panel D: conditional CDF after each candidate cutoff + power-law fit
    axD = axes[3]
    # poly3
    if np.isfinite(x_poly) and n_kept_poly >= 3:
        xmin = max(x_poly, 1e-3)
        keep = obs_logP >= x_poly
        xs_k, ys_k = empirical_cdf(obs_logP[keep])
        axD.step(xs_k, ys_k, where="post", color="C1", lw=2,
                 label=f"poly3 (cut={x_poly:.3f}, n={n_kept_poly}, "
                       f"α={a_poly:+.2f}, KSp={ks_poly:.2f})")
        x_dense_d = np.linspace(xmin, xmax_global, 400)
        axD.plot(x_dense_d, powerlaw_cdf(x_dense_d, a_poly, xmin, xmax_global),
                 color="C1", lw=1.0, ls="--")
    # numerical (use the σ with best KSp on the conditional fit)
    best_idx = None
    best_kp = -np.inf
    for i, (sigma, cutoff, _, _, _) in enumerate(numerical_results):
        n_k, _, a_f, ks_p = evaluate_cutoff(obs_logP, cutoff, xmax_global)
        if np.isfinite(ks_p) and ks_p > best_kp:
            best_kp = ks_p
            best_idx = i
    if best_idx is not None:
        sigma_b, cut_b, _, _, _ = numerical_results[best_idx]
        n_k, _, a_f, ks_p = evaluate_cutoff(obs_logP, cut_b, xmax_global)
        xmin = max(cut_b, 1e-3)
        keep = obs_logP >= cut_b
        xs_k, ys_k = empirical_cdf(obs_logP[keep])
        axD.step(xs_k, ys_k, where="post", color="C2", lw=2,
                 label=f"num σ={sigma_b:.2f} (cut={cut_b:.3f}, n={n_k}, "
                       f"α={a_f:+.2f}, KSp={ks_p:.2f})")
        x_dense_d = np.linspace(xmin, xmax_global, 400)
        axD.plot(x_dense_d, powerlaw_cdf(x_dense_d, a_f, xmin, xmax_global),
                 color="C2", lw=1.0, ls="--")
    axD.set_xlabel(r"$\log_{10}(P/\mathrm{d})$")
    axD.set_ylabel("conditional CDF (re-normalized)")
    axD.set_title("Panel D — conditional CDF above cutoff vs fitted power-law "
                  "(dashed)")
    axD.set_ylim(-0.02, 1.05)
    axD.legend(loc="lower right", fontsize=8)
    axD.grid(alpha=0.3)

    fig.suptitle("logP inflection diagnostic — observed binary CDF",
                 fontsize=12, y=0.995)

    pdf_path = os.path.join(out_dir, "logP_inflection_check.pdf")
    png_path = os.path.join(out_dir, "logP_inflection_check.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
