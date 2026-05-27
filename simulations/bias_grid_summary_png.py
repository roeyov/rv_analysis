"""
simulations.bias_grid_summary_png — Save PNG summary plots from bias-grid results.

For each scoring method available in the grid (KS / AD / CvM), produces one PNG:
  Row 1: 1D marginalized posteriors for pi, kappa, eta, f_bin
  Row 2: CDFs at the best-fit grid point for logP, e, K1, q
  Row 3: PDFs at the best-fit grid point for logP, e, K1, q

Usage:
    python -m simulations.bias_grid_summary_png \\
        --output-dir /path/to/grid/results \\
        [--output-png-dir /path/to/save/pngs] \\
        [--sb1-tex path.tex] [--sb2-tex path.tex]
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

# Make 'simulations.*' importable when invoked as a script.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from simulations.bias_config import DEFAULT_BIAS_CFG
from simulations.bias_grid import (
    load_observed_from_tex, _HIST_PAIRS,
    _DET_SHARDS_DIR, _load_det_shard,
)

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

PARAM_NAMES = [r"$\pi$", r"$\kappa$", r"$\eta$", r"$f_{\rm bin}$"]
REF_VALS = [-0.55, -0.10, -0.45, 0.69]  # Sana+2012 Galactic
COLORS_1D = ["#4393c3", "#d6604d", "#5aae61", "#9970ab"]
TEST_LABELS = {"ks": "Kolmogorov-Smirnov",
               "ad": "Anderson-Darling",
               "cvm": "Cramer-von Mises"}


# ---------------------------------------------------------------------------
# Data loading (same shape as bias_grid_explorer.load_grid_data, no streamlit)
# ---------------------------------------------------------------------------

def load_grid_data(output_dir):
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

    test_names = ["ks", "ad", "cvm"]
    available_tests = []
    gmf_cubes = {}
    test_pval_cubes = {}
    best_fits = {}

    for tname in test_names:
        gmf_key = "gmf_%s_cube" % tname
        if gmf_key in cubes.files:
            gmf_cubes[tname] = cubes[gmf_key]
            available_tests.append(tname)
            test_pval_cubes[tname] = {}
            for par in ("logP", "e", "K1", "e_circ"):
                tc_key = "%s_%s_cube" % (tname, par)
                if tc_key in cubes.files:
                    test_pval_cubes[tname][par] = cubes[tc_key]
            gc = gmf_cubes[tname]
            idx = np.unravel_index(np.nanargmax(gc), gc.shape)
            best_fits[tname] = (
                float(data["pi_grid"][idx[0]]),
                float(data["kappa_grid"][idx[1]]),
                float(data["eta_grid"][idx[2]]),
                float(data["fbin_grid"][idx[3]]),
            )

    # Legacy KS-only fallback
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

    if "logP_cutoff_scope" in cubes.files:
        data["logP_cutoff_scope"] = str(cubes["logP_cutoff_scope"])
    else:
        data["logP_cutoff_scope"] = "period_only"

    if "logP_cutoff_mode" in cubes.files:
        data["logP_cutoff_mode"] = str(cubes["logP_cutoff_mode"])
        data["logP_cutoff"] = float(cubes["logP_cutoff"])
    else:
        data["logP_cutoff_mode"] = "none"
        data["logP_cutoff"] = 0.0

    if "e_score_mode" in cubes.files:
        data["e_score_mode"] = str(cubes["e_score_mode"])
    else:
        has_e_circ = any("e_circ" in test_pval_cubes.get(t, {})
                         for t in test_pval_cubes)
        data["e_score_mode"] = "split" if has_e_circ else "combined"

    # Detected-systems storage discovery
    shard_dir = os.path.join(output_dir, _DET_SHARDS_DIR)
    det_index_path = os.path.join(output_dir, "det_index.npz")
    det_legacy_path = os.path.join(output_dir, "grid_detected.npz")
    if not os.path.exists(det_legacy_path):
        det_legacy_path = os.path.join(output_dir, "checkpoint_detected.npz")

    if os.path.isdir(shard_dir) and os.path.exists(det_index_path):
        idx_data = np.load(det_index_path)
        data["has_detected"] = True
        data["det_format"] = "shards"
        data["det_output_dir"] = output_dir
        data["step_to_ijkl"] = idx_data["step_to_ijkl"]
    elif os.path.exists(det_legacy_path):
        det = np.load(det_legacy_path, allow_pickle=True)
        data["has_detected"] = True
        data["det_format"] = "legacy"
        data["det_logP"] = det["all_logP_det"]
        data["det_e"] = det["all_e_det"]
        data["det_K1"] = det["all_K1_det"]
        data["det_q"] = det["all_q_det"]
        data["det_offsets"] = det["offsets"]
        data["step_to_ijkl"] = det["step_to_ijkl"]
    else:
        data["has_detected"] = False

    return data


def get_detected_for_point(data, i, j, k, l):
    if not data.get("has_detected"):
        return None

    ijkl = np.array([i, j, k, l])

    if data.get("det_format") == "shards":
        step_to_ijkl = data["step_to_ijkl"]
        matches = np.where((step_to_ijkl[:, 1:] == ijkl).all(axis=1))[0]
        if len(matches) == 0:
            return None
        step = int(step_to_ijkl[matches[0], 0])
        return _load_det_shard(data["det_output_dir"], step)

    matches = np.where((data["step_to_ijkl"] == ijkl).all(axis=1))[0]
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


# ---------------------------------------------------------------------------
# Posterior helpers
# ---------------------------------------------------------------------------

def gmf_to_prob(gmf_cube):
    shifted = gmf_cube - np.nanmax(gmf_cube)
    prob = np.exp(shifted)
    return np.nan_to_num(prob, nan=0.0)


def get_1d_posterior(prob, grids, axis_idx):
    axes_to_sum = tuple(i for i in range(4) if i != axis_idx)
    post = np.nansum(prob, axis=axes_to_sum)
    grid = grids[axis_idx]
    if np.sum(post) > 0 and len(grid) > 1:
        post = post / _trapz(post, grid)
    return post


def powerlaw_cdf(x, alpha, xmin, xmax):
    a = alpha + 1.0
    if abs(a) < 1e-8:
        return np.log(x / xmin) / np.log(xmax / xmin)
    return (x**a - xmin**a) / (xmax**a - xmin**a)


def powerlaw_pdf(x, alpha, xmin, xmax):
    a = alpha + 1.0
    if abs(a) < 1e-8:
        return 1.0 / (x * np.log(xmax / xmin))
    return a * x**alpha / (xmax**a - xmin**a)


# ---------------------------------------------------------------------------
# Plot construction
# ---------------------------------------------------------------------------

def _draw_marginals_row(axes_row, prob, grids, best_fit):
    for col in range(4):
        ax = axes_row[col]
        grid = grids[col]
        post = get_1d_posterior(prob, grids, col)
        color = COLORS_1D[col]

        ax.fill_between(grid, post, alpha=0.25, color=color)
        ax.plot(grid, post, color=color, lw=2)

        cdf = np.cumsum(post)
        if cdf[-1] > 0:
            cdf_n = cdf / cdf[-1]
            lo = float(np.interp(0.16, cdf_n, grid))
            hi = float(np.interp(0.84, cdf_n, grid))
            ax.axvspan(lo, hi, color="crimson", alpha=0.10,
                       label=r"68\% CI" if col == 0 else None)

        ax.axvline(best_fit[col], color="gold", lw=1.8, ls="--",
                   label="Best fit" if col == 0 else None)
        ax.axvline(REF_VALS[col], color="grey", lw=1.2, ls=":",
                   label="Sana+12" if col == 0 else None)

        ax.set_xlabel(PARAM_NAMES[col])
        if col == 0:
            ax.set_ylabel("Posterior density")
            ax.legend(fontsize=8, loc="best")
        ax.set_title("%s = %.2f (ref %.2f)" % (
            PARAM_NAMES[col], best_fit[col], REF_VALS[col]), fontsize=9)


def _draw_cdf_pdf_rows(axes_cdf, axes_pdf, det_params, obs, best_fit,
                       e_score_mode, logP_cutoff=0.0,
                       logP_cutoff_mode="none"):
    cfg = DEFAULT_BIAS_CFG
    pi_val, kappa_val, eta_val, fbin_val = best_fit

    obs_e_arr = obs.get("e") if obs is not None else None
    if e_score_mode == "eccentric_only" and obs_e_arr is not None:
        obs_e_arr = np.asarray(obs_e_arr)
        obs_e_arr = obs_e_arr[obs_e_arr > 0]

    empirical_all = {}
    for key in ("logP", "e", "K1", "q"):
        det_arr = np.asarray(det_params.get(key, np.array([])))
        nondet_arr = np.asarray(det_params.get("%s_nondet" % key, np.array([])))
        if len(det_arr) or len(nondet_arr):
            empirical_all[key] = np.concatenate([det_arr, nondet_arr])
    if e_score_mode == "eccentric_only" and "e" in empirical_all:
        empirical_all["e"] = empirical_all["e"][empirical_all["e"] > 0]

    param_configs = [
        ("logP", obs.get("logP") if obs else None, r"$\log_{10}(P/\mathrm{d})$",
         "#4393c3", pi_val, cfg["log_p_min"], cfg["log_p_max"], 15),
        ("e", obs_e_arr, "e", "#d6604d",
         eta_val, 1e-6, cfg["e_max"], 15),
        ("K1", obs.get("K1") if obs else None, r"$K_1$ [km/s]", "#5aae61",
         None, None, None, 15),
        ("q", None, "q", "#8073ac",
         kappa_val, cfg["q_min"], cfg["q_max"], 12),
    ]

    for col_idx, (key, obs_arr, xlabel, color, alpha, xmin, xmax, n_bins) \
            in enumerate(param_configs):
        ax_cdf = axes_cdf[col_idx]
        ax_pdf = axes_pdf[col_idx]
        has_obs = obs_arr is not None and len(obs_arr) >= 2
        obs_s = np.sort(np.asarray(obs_arr)) if has_obs else None

        if key == "e":
            clip_lo = 1e-12 if e_score_mode == "eccentric_only" else 0.0
            clip_hi = 1.0
        elif has_obs:
            clip_lo, clip_hi = 0.0, float(obs_s[-1])
        else:
            clip_lo = xmin if xmin is not None else 0.0
            clip_hi = xmax if xmax is not None else 1.0

        # Raise the logP lower bound to the inflection cutoff used by the
        # scorer (a no-op when logP_cutoff == 0.0). Re-sort the observed
        # array after the filter so the conditional empirical CDF is
        # computed on the same window the KS test saw.
        if key == "logP" and logP_cutoff > 0.0:
            clip_lo = max(clip_lo, float(logP_cutoff))
            if has_obs:
                obs_filt = np.asarray(obs_arr)
                obs_filt = obs_filt[obs_filt >= clip_lo]
                if len(obs_filt) >= 2:
                    obs_s = np.sort(obs_filt)
                else:
                    has_obs = False
                    obs_s = None

        # Completeness — restricts detected/observed CDF height.
        completeness = 1.0
        if key in empirical_all:
            all_vals = empirical_all[key]
            n_all = len(all_vals)
            if n_all > 0:
                det_vals = np.asarray(det_params.get(key, np.array([])))
                if key == "e" and e_score_mode == "eccentric_only":
                    n_det_sub = int((det_vals > 0).sum()) if len(det_vals) else 0
                    completeness = n_det_sub / n_all
                else:
                    completeness = len(det_vals) / n_all

        # ---------- CDF row ----------
        if has_obs:
            obs_cdf = np.arange(1, len(obs_s) + 1) / len(obs_s)
            ax_cdf.step(obs_s, obs_cdf * completeness, where="post",
                        color="black", lw=2,
                        label="Observed (n=%d)" % len(obs_s))

        sim_raw = np.asarray(det_params.get(key, np.array([])))
        if len(sim_raw) >= 2:
            sim_clipped = sim_raw[(sim_raw >= clip_lo) & (sim_raw <= clip_hi)]
            if len(sim_clipped) >= 2:
                sim = np.sort(sim_clipped)
                sim_cdf = np.arange(1, len(sim) + 1) / len(sim)
                ax_cdf.step(sim, sim_cdf * completeness, where="post",
                            color=color, lw=2, ls="--",
                            label="Sim. detected (n=%d)" % len(sim))

        if key in empirical_all and len(empirical_all[key]) >= 2:
            all_c = empirical_all[key]
            all_c = all_c[(all_c >= clip_lo) & (all_c <= clip_hi)]
            if len(all_c) >= 2:
                all_s = np.sort(all_c)
                all_cdf = np.arange(1, len(all_s) + 1) / len(all_s)
                ax_cdf.step(all_s, all_cdf, where="post",
                            color="gray", lw=2, ls=":",
                            label="Sim. all (n=%d)" % len(all_s))
        elif alpha is not None and xmin is not None:
            x_hi_cdf = clip_hi if key == "logP" and has_obs else xmax
            x_intr = np.linspace(xmin, x_hi_cdf, 200)
            y_intr = powerlaw_cdf(x_intr, alpha, xmin, x_hi_cdf)
            ax_cdf.plot(x_intr, y_intr, color="gray", lw=2, ls=":",
                        label="Intrinsic")

        if key == "logP" and logP_cutoff > 0.0:
            ax_cdf.axvline(logP_cutoff, color="0.4", lw=1.0, ls=":",
                           label="cutoff=%.3f (%s)" % (logP_cutoff,
                                                       logP_cutoff_mode))

        ax_cdf.set_ylim(0, 1.05)
        ax_cdf.set_xlabel(xlabel)
        if col_idx == 0:
            ax_cdf.set_ylabel("CDF")
        ax_cdf.legend(fontsize=7, loc="best")

        # ---------- PDF row ----------
        bin_lo = clip_lo if has_obs else xmin
        bin_hi = clip_hi if has_obs else xmax
        bins = np.linspace(bin_lo, bin_hi, n_bins + 1)
        bw = float(bins[1] - bins[0])
        centers = 0.5 * (bins[:-1] + bins[1:])
        pdf_peak = 0.0

        if has_obs:
            counts, _ = np.histogram(obs_arr, bins=bins)
            density_obs = counts / (counts.sum() * bw) * completeness
            pdf_peak = max(pdf_peak, float(density_obs.max()))
            ax_pdf.bar(centers, density_obs, width=bw * 0.95,
                       color="black", alpha=0.30, label="Obs. PDF")

        if len(sim_raw) >= 2:
            sim_clipped = sim_raw[(sim_raw >= clip_lo) & (sim_raw <= clip_hi)]
            if len(sim_clipped) >= 2:
                counts_sim, _ = np.histogram(sim_clipped, bins=bins)
                density_sim = counts_sim / (counts_sim.sum() * bw) * completeness
                pdf_peak = max(pdf_peak, float(density_sim.max()))
                ax_pdf.plot(centers, density_sim, color=color, lw=2, ls="--",
                            marker="o", ms=4, label="Sim. PDF")

        if key in empirical_all and len(empirical_all[key]) >= 2:
            all_c = empirical_all[key]
            all_c = all_c[(all_c >= clip_lo) & (all_c <= clip_hi)]
            if len(all_c) >= 2:
                counts_all, _ = np.histogram(all_c, bins=bins)
                density_all = counts_all / (counts_all.sum() * bw)
                ax_pdf.plot(centers, density_all, color="gray", lw=2, ls=":",
                            label="Sim. all PDF")
        elif alpha is not None and xmin is not None:
            x_pdf = np.linspace(max(xmin, 1e-6), xmax, 200)
            y_pdf = powerlaw_pdf(x_pdf, alpha, xmin, xmax)
            if pdf_peak > 0:
                y_pdf = np.minimum(y_pdf, pdf_peak * 1.05)
            ax_pdf.plot(x_pdf, y_pdf, color="gray", lw=2, ls=":",
                        label="Intrinsic PDF")

        ax_pdf.set_xlabel(xlabel)
        if col_idx == 0:
            ax_pdf.set_ylabel("PDF")
        ax_pdf.legend(fontsize=7, loc="best")


def make_summary_figure(data, obs, test_name, e_score_mode,
                        logP_cutoff=0.0, logP_cutoff_mode="none",
                        logP_cutoff_scope="period_only"):
    gmf_cube = data["gmf_cubes"][test_name]
    best_fit = data["best_fits"][test_name]
    grids = [data["pi_grid"], data["kappa_grid"],
             data["eta_grid"], data["fbin_grid"]]

    prob = gmf_to_prob(gmf_cube)

    # Best-fit indices for detected-systems lookup
    i = int(np.argmin(np.abs(grids[0] - best_fit[0])))
    j = int(np.argmin(np.abs(grids[1] - best_fit[1])))
    k = int(np.argmin(np.abs(grids[2] - best_fit[2])))
    l = int(np.argmin(np.abs(grids[3] - best_fit[3])))

    fig, axes = plt.subplots(3, 4, figsize=(24.46, 10.79))

    _draw_marginals_row(axes[0], prob, grids, best_fit)

    det_params = get_detected_for_point(data, i, j, k, l)
    if det_params is None or len(det_params.get("logP", [])) < 2:
        for row in (1, 2):
            for col in range(4):
                axes[row, col].text(
                    0.5, 0.5,
                    "No detected systems\nfor best-fit grid point",
                    ha="center", va="center",
                    transform=axes[row, col].transAxes, fontsize=10,
                )
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
    else:
        _draw_cdf_pdf_rows(axes[1], axes[2], det_params, obs, best_fit,
                           e_score_mode,
                           logP_cutoff=logP_cutoff,
                           logP_cutoff_mode=logP_cutoff_mode)

    p_det_val = float(data["pdet_cube"][i, j, k, l])
    cutoff_str = (", logP_cutoff=%.3f (%s/%s)" % (logP_cutoff,
                                                   logP_cutoff_mode,
                                                   logP_cutoff_scope)
                  if logP_cutoff > 0 else "")
    title = ("%s — best fit: pi=%.2f, kappa=%.2f, eta=%.2f, f_bin=%.2f "
             "  (p_det=%.3f, e_score_mode=%s%s)" % (
                 TEST_LABELS.get(test_name, test_name.upper()),
                 best_fit[0], best_fit[1], best_fit[2], best_fit[3],
                 p_det_val, e_score_mode, cutoff_str))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--output-dir", required=True,
                        help="Bias-grid results directory (contains "
                             "grid_cubes.npz)")
    parser.add_argument("--output-png-dir", default=None,
                        help="Where to save PNGs (default: --output-dir)")
    parser.add_argument("--sb1-tex", default=None)
    parser.add_argument("--sb2-tex", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    out_png_dir = args.output_png_dir or args.output_dir
    os.makedirs(out_png_dir, exist_ok=True)

    data = load_grid_data(args.output_dir)

    sb1 = args.sb1_tex or DEFAULT_BIAS_CFG.get("sb1_tex", "")
    sb2 = args.sb2_tex or DEFAULT_BIAS_CFG.get("sb2_tex", "")
    obs = None
    if sb1 and sb2 and os.path.exists(sb1) and os.path.exists(sb2):
        obs = load_observed_from_tex(sb1, sb2)
    else:
        print("Warning: SB1/SB2 LaTeX tables not found — observed curves "
              "will be omitted.", file=sys.stderr)

    e_mode = data.get("e_score_mode", "combined")
    logP_cutoff = float(data.get("logP_cutoff", 0.0) or 0.0)
    logP_cutoff_mode = data.get("logP_cutoff_mode", "none")
    logP_cutoff_scope = data.get("logP_cutoff_scope", "period_only")

    # Only filter obs arrays here when scope=="exclude" (below-cutoff
    # systems are removed everywhere, mirroring the runtime behavior).
    # In "period_only" scope, _draw_cdf_pdf_rows handles the per-panel
    # logP filtering and leaves obs_e/obs_K1 untouched.
    if (logP_cutoff > 0.0 and logP_cutoff_scope == "exclude"
            and obs is not None):
        obs_logP = np.asarray(obs.get("logP"))
        if obs_logP is not None and obs_logP.size:
            keep = obs_logP >= logP_cutoff
            obs = {k: (np.asarray(v)[keep]
                       if isinstance(v, (list, np.ndarray))
                          and len(v) == len(obs_logP) else v)
                   for k, v in obs.items()}

    for test_name in data["available_tests"]:
        fig = make_summary_figure(data, obs, test_name, e_mode,
                                  logP_cutoff=logP_cutoff,
                                  logP_cutoff_mode=logP_cutoff_mode,
                                  logP_cutoff_scope=logP_cutoff_scope)
        out_path = os.path.join(out_png_dir, "summary_%s.png" % test_name)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print("Saved: %s" % out_path)


if __name__ == "__main__":
    main()
