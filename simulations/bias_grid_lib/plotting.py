"""End-of-run PDF plots + summary-DataFrame builder.

``plot_grid_results`` produces three PDFs (1D marginal posteriors, 2D
marginal heatmaps, best-fit CDF comparison) from the result dict the
engine returns; ``format_grid_summary`` flattens the per-grid-point
result list into a sortable summary DataFrame.
"""

import os
import numpy as np
import pandas as pd

from simulations.bias_grid_lib.constants import _trapz
from simulations.bias_grid_lib.logging_utils import logger
from simulations.bias_grid_lib.statistics import _SCORED_TESTS


def plot_grid_results(results, output_dir=None, obs_logP=None, obs_e=None,
                      obs_K1=None):
    """
    Plot grid search results: 2D marginalized GMF heatmaps,
    1D marginalized posteriors, best-fit CDF comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pi_grid = results["pi_grid"]
    kappa_grid = results["kappa_grid"]
    eta_grid = results["eta_grid"]
    fbin_grid = results["fbin_grid"]
    gmf = results["gmf_cube"]
    best = results["best_fit"]

    # Convert log GMF to probability
    gmf_shifted = gmf - np.nanmax(gmf)
    prob = np.exp(gmf_shifted)
    prob = np.nan_to_num(prob, nan=0.0)

    param_names = ["π", "κ", "η", "f_bin"]
    grids = [pi_grid, kappa_grid, eta_grid, fbin_grid]
    # Reference values (Sana+2012 Galactic)
    ref_vals = [-0.55, -0.10, -0.45, 0.69]

    # ====== 1D Marginalized Posteriors ======
    fig_1d, axes_1d = plt.subplots(1, 4, figsize=(18, 4))
    colors = ["#4393c3", "#d6604d", "#5aae61", "#9970ab"]

    for ax, axis_idx, label, color, ref in zip(
            axes_1d, range(4), param_names, colors, ref_vals):
        grid = grids[axis_idx]
        axes_to_sum = tuple(i for i in range(4) if i != axis_idx)
        post_1d = np.nansum(prob, axis=axes_to_sum)

        # Normalize
        if np.sum(post_1d) > 0 and len(grid) > 1:
            post_1d /= _trapz(post_1d, grid)

        ax.fill_between(grid, post_1d, alpha=0.3, color=color)
        ax.plot(grid, post_1d, color=color, lw=2)

        # Mode
        mode_idx = np.argmax(post_1d)
        mode = grid[mode_idx]
        ax.axvline(mode, color="crimson", lw=1.5, ls="--",
                   label=f"Mode = {mode:.2f}")

        # 68% CI
        cdf = np.cumsum(post_1d)
        if cdf[-1] > 0:
            cdf /= cdf[-1]
            lo = np.interp(0.16, cdf, grid)
            hi = np.interp(0.84, cdf, grid)
            ax.axvspan(lo, hi, alpha=0.12, color="crimson",
                       label=f"68% CI [{lo:.2f}, {hi:.2f}]")

        ax.axvline(ref, color="grey", lw=1, ls=":",
                   label=f"Sana+12: {ref}")
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Posterior density", fontsize=11)
        ax.legend(fontsize=7)

    plt.suptitle(
        f"Best fit: π={best[0]:.2f}, κ={best[1]:.2f}, "
        f"η={best[2]:.2f}, f_bin={best[3]:.2f}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_1d.savefig(os.path.join(output_dir, "grid_1d_posteriors.pdf"),
                       bbox_inches="tight")
        logger.info("Saved 1D posteriors to %s/grid_1d_posteriors.pdf",
                     output_dir)

    # ====== 2D Marginalized GMF Heatmaps ======
    pairs = [
        (0, 1, "π", "κ"),
        (0, 2, "π", "η"),
        (0, 3, "π", "f_bin"),
        (1, 2, "κ", "η"),
        (1, 3, "κ", "f_bin"),
        (2, 3, "η", "f_bin"),
    ]

    fig_2d, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (i1, i2, l1, l2) in zip(axes, pairs):
        axes_to_sum = tuple(i for i in range(4) if i not in (i1, i2))
        marg = np.nansum(prob, axis=axes_to_sum)

        g1 = grids[i1]
        g2 = grids[i2]

        im = ax.imshow(marg.T, origin="lower", aspect="auto",
                       extent=[g1[0], g1[-1], g2[0], g2[-1]],
                       cmap="RdYlBu_r", interpolation="bilinear")
        ax.set_xlabel(l1, fontsize=12)
        ax.set_ylabel(l2, fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Marginal prob")
        ax.plot(best[i1], best[i2], "w*", markersize=12,
                markeredgecolor="k", markeredgewidth=1)
        ax.set_title(f"{l1} vs {l2}", fontsize=11)

    plt.suptitle(
        f"Best fit: π={best[0]:.2f}, κ={best[1]:.2f}, "
        f"η={best[2]:.2f}, f_bin={best[3]:.2f}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if output_dir:
        fig_2d.savefig(os.path.join(output_dir, "grid_2d_gmf.pdf"),
                       bbox_inches="tight")
        logger.info("Saved 2D heatmaps to %s/grid_2d_gmf.pdf", output_dir)

    # ====== CDF comparison: best-fit detected vs observed ======
    fig_cdf = None
    best_res = None
    if obs_logP is not None:
        # Find the best-fit result entry (the one with highest log_gmf
        # that still has the detected arrays).
        best_pi, best_kappa, best_eta, best_fbin = best
        for r in results["results"]:
            if (np.isclose(r["pi"], best_pi)
                    and np.isclose(r["kappa"], best_kappa)
                    and np.isclose(r["eta"], best_eta)
                    and np.isclose(r["fbin"], best_fbin)):
                # Check that detected arrays survived (not stripped by
                # checkpoint round-trip).
                if isinstance(r.get("logP_det"), np.ndarray):
                    best_res = r
                break

    if best_res is not None and len(best_res["logP_det"]) >= 2:
        fig_cdf, axes_cdf = plt.subplots(1, 3, figsize=(16, 5))

        logP_cutoff = float(results.get("logP_cutoff", 0.0) or 0.0)
        logP_cutoff_mode = results.get("logP_cutoff_mode", "none")

        param_pairs = [
            ("logP_det", obs_logP, r"$\log_{10}(P/\mathrm{d})$", "#4393c3",
             logP_cutoff),
            ("e_det", obs_e, "$e$", "#d6604d", 0.0),
            ("K1_det", obs_K1, "$K_1$ [km/s]", "#5aae61", 0.0),
        ]

        for ax, (key, obs_arr, xlabel, color, lo) in zip(axes_cdf,
                                                          param_pairs):
            # When a lower cutoff is active (only logP), restrict both obs
            # and sim arrays before building the empirical CDF. ks_2samp
            # already renormalizes from the truncated samples, so this is
            # the same conditional view the scorer compared.
            sim_raw = np.asarray(best_res[key])
            obs_raw = np.asarray(obs_arr)
            sim = np.sort(sim_raw[sim_raw >= lo]) if lo > 0 else np.sort(sim_raw)
            obs_s = np.sort(obs_raw[obs_raw >= lo]) if lo > 0 else np.sort(obs_raw)

            # Empirical CDF
            sim_cdf = np.arange(1, len(sim) + 1) / len(sim)
            obs_cdf = np.arange(1, len(obs_s) + 1) / len(obs_s)

            ax.step(obs_s, obs_cdf, where="post", color="k", lw=2,
                    label=f"Observed (n={len(obs_s)})")
            ax.step(sim, sim_cdf, where="post", color=color, lw=2,
                    ls="--",
                    label=f"Simulated (n={len(sim)})")

            if lo > 0:
                ax.axvline(lo, color="0.4", lw=1.2, ls=":",
                           label=f"cutoff = {lo:.3f}\n({logP_cutoff_mode})")

            # KS p-value annotation
            ks_key = {"logP_det": "ks_p_logP", "e_det": "ks_p_e",
                      "K1_det": "ks_p_K1"}[key]
            pval = best_res.get(ks_key, np.nan)
            ax.text(0.05, 0.95, f"KS p = {pval:.3f}",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="wheat", alpha=0.5))

            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("CDF", fontsize=12)
            ax.legend(fontsize=9)
            ax.set_ylim(0, 1.05)

        plt.suptitle(
            f"Best fit CDF: π={best[0]:.2f}, κ={best[1]:.2f}, "
            f"η={best[2]:.2f}, f_bin={best[3]:.2f}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        if output_dir:
            fig_cdf.savefig(os.path.join(output_dir, "grid_cdf_bestfit.pdf"),
                            bbox_inches="tight")
            logger.info("Saved CDF comparison to %s/grid_cdf_bestfit.pdf",
                        output_dir)

    return {"fig_1d": fig_1d, "fig_2d": fig_2d, "fig_cdf": fig_cdf}


def format_grid_summary(results):
    """Create a summary DataFrame of all grid points, sorted by GMF."""
    rows = []
    for r in results["results"]:
        row = {
            "π": f"{r['pi']:.2f}",
            "κ": f"{r['kappa']:.2f}",
            "η": f"{r['eta']:.2f}",
            "f_bin": f"{r['fbin']:.2f}",
            "p_det": f"{r['p_det']:.3f}",
            "P_binom": f"{r['p_binom']:.4f}",
            "N_det": r["n_detected"],
        }
        for tname in _SCORED_TESTS:
            prefix = tname.upper()
            row["%s_logP" % prefix] = f"{r.get('%s_p_logP' % tname, 0):.3f}"
            row["%s_e" % prefix] = f"{r.get('%s_p_e' % tname, 0):.3f}"
            row["%s_K1" % prefix] = f"{r.get('%s_p_K1' % tname, 0):.3f}"
            lgmf = r.get("log_gmf_%s" % tname, r.get("log_gmf", -np.inf))
            row["log_GMF_%s" % prefix] = (
                f"{lgmf:.2f}" if np.isfinite(lgmf) else "-inf")
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_sort"] = [r.get("log_gmf_ks", r.get("log_gmf", -np.inf))
                   for r in results["results"]]
    df = df.sort_values("_sort", ascending=False).drop("_sort", axis=1)
    logger.info("format_grid_summary: %d rows", len(df))
    return df
