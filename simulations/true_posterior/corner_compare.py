"""
simulations.true_posterior.corner_compare — one corner plot overlaying
several (pi, kappa, eta, f_bin) posteriors of the same data.

Default overlay: the corrected CvM pseudo-posterior of the adopted ext3
run, the one-component likelihood posterior, and the split-model
likelihood posterior (f_bin = f_gt + f_lt). Diagonal panels show unit-area
1-D marginals; lower panels show the 68% (solid) and 95% (dotted)
highest-density contours of the 2-D marginals, each on its own grid.
MW and LMC reference values follow make_bias_corner_figure.py.

    python -m simulations.true_posterior.corner_compare \
        --run "CvM:cvm:eccentric_only__numerical__lucyT:<ext3 dir>" \
        --run "likelihood:lik:likmix__numerical__lucyT:<mix dir>" \
        --run "split:lik:likmixsplit__numerical__lucyT:<split dir>" \
        --out-dir <dir>
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
import numpy as np   # noqa: E402

from simulations.bias_grid_lib.cube_io import read_gmf_cube   # noqa: E402
from simulations.bias_grid_lib.marginals import grid_quantiles   # noqa: E402

AA_SINGLE_COL = 3.46
AA_RCPARAMS = {
    "font.family": "serif", "font.size": 9, "axes.labelsize": 10,
    "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True,
    "ytick.minor.visible": True, "axes.linewidth": 0.8,
    "lines.linewidth": 1.0, "lines.markersize": 4, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
}
LABELS = [r"$\pi$", r"$\kappa$", r"$\eta$", r"$f_\mathrm{bin}$"]
COLORS = ["#777777", "#EE7733", "#0077BB", "#009988"]
STYLES = ["--", "-", "-", "-"]
REF_MW = [-0.55, -0.10, -0.45, 0.69]       # Sana+2012
REF_LMC = [-0.1, 0.2, -0.50, 0.58]         # Almeida+2017 / Shenar+2022
C_MW, C_LMC = "#CC3311", "#009988"
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def load_posterior(run_dir, test, tag):
    z = np.load(os.path.join(run_dir, "grid_cubes.npz"), allow_pickle=True)
    cube, tag = read_gmf_cube(z, test=test, tag=tag)
    cube = np.asarray(cube, float)
    prob = np.nan_to_num(np.exp(cube - np.nanmax(cube)))
    grids = [np.asarray(z[k], float) for k in ("pi_grid", "kappa_grid",
                                               "eta_grid", "fbin_grid")]
    m1, q, mean = [], [], []
    for ax in range(4):
        post = prob.sum(axis=tuple(a for a in range(4) if a != ax))
        m1.append(post / _trapz(post, grids[ax]))
        q.append(grid_quantiles(post, grids[ax],
                                (0.005, 0.16, 0.5, 0.84, 0.995)))
        # posterior mean of the marginal (trapezoid-weighted average)
        mean.append(float(_trapz(grids[ax] * post, grids[ax])
                          / _trapz(post, grids[ax])))
    m2 = {}
    for r in range(4):
        for c in range(r):
            m2[(r, c)] = prob.sum(axis=tuple(a for a in range(4)
                                             if a not in (r, c)))
    return {"grids": grids, "m1": m1, "q": np.array(q), "m2": m2,
            "mean": mean, "tag": tag}


def hdr_levels(marg, fractions=(0.95, 0.68)):
    flat = np.sort(marg.ravel())[::-1]
    csum = np.cumsum(flat) / flat.sum()
    return sorted({float(flat[min(int(np.searchsorted(csum, f)),
                                  flat.size - 1)]) for f in fractions})


def mean_title(p, ax_i, label_tex):
    """``<label> = mean^{+(p84-mean)}_{-(mean-p16)}`` for one run."""
    m = p["mean"][ax_i]
    lo, hi = p["q"][ax_i, 1], p["q"][ax_i, 3]
    return r"%s $= %.2f^{+%.2f}_{-%.2f}$" % (label_tex, m, hi - m, m - lo)


def render(posts, labels, width, out_stem, colors=None, styles=None,
           include_refs=False, annotate=False, mean_lines=True):
    """``include_refs`` widens every axis to show the MW/LMC markers;
    ``annotate`` writes each run's posterior mean and the MW/LMC values
    in the diagonal panels; ``mean_lines`` marks each run's posterior
    (trapezoid-weighted) mean with a vertical dotted line in the diagonal
    panels and writes the mean with its 68% interval above each panel."""
    plt.rcParams.update(AA_RCPARAMS)
    colors = COLORS if colors is None else colors
    styles = STYLES if styles is None else styles
    n = 4
    fig, axes = plt.subplots(n, n, figsize=(width, width),
                             gridspec_kw=dict(hspace=0.08, wspace=0.08))
    small = width < 5
    lims = []
    for ax_i in range(4):
        lo = min(p["q"][ax_i, 0] for p in posts)
        hi = max(p["q"][ax_i, 4] for p in posts)
        if include_refs:
            lo = min(lo, REF_MW[ax_i], REF_LMC[ax_i])
            hi = max(hi, REF_MW[ax_i], REF_LMC[ax_i])
        pad = 0.08 * (hi - lo)
        lims.append((lo - pad, hi + pad))
    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            if c > r:
                ax.set_visible(False)
                continue
            if r == c:
                for p, col, ls in zip(posts, colors, styles):
                    ax.plot(p["grids"][r], p["m1"][r], color=col, ls=ls,
                            lw=1.0)
                ax.axvline(REF_MW[r], color=C_MW, lw=0.6, ls="--")
                ax.axvline(REF_LMC[r], color=C_LMC, lw=0.6, ls="-.")
                if mean_lines:
                    for i, (p, col) in enumerate(zip(posts, colors)):
                        ax.axvline(p["mean"][r], color=col, lw=0.8, ls=":")
                        # one title line per run, stacked upward above the
                        # panel (the space above a diagonal panel is empty)
                        ax.text(0.5, 1.04 + i * (0.19 if small else 0.15),
                                mean_title(p, r, LABELS[r]),
                                transform=ax.transAxes, ha="center",
                                va="bottom", color=col,
                                fontsize=6.5 if small else 8.5)
                ax.set_yticks([])
                ax.set_ylim(bottom=0)
                ax.tick_params(axis="y", which="both", left=False, right=False)
                if annotate:
                    fs = 5.5 if small else 7.5
                    lines = [("%s %+.2f" % (lab, p["mean"][r]), col)
                             for p, col, lab in zip(posts, colors, labels)]
                    lines += [("MW %+.2f" % REF_MW[r], C_MW),
                              ("LMC %+.2f" % REF_LMC[r], C_LMC)]
                    if r == 3:
                        lines = [(t.replace("+", ""), c) for t, c in lines]
                    # put the block on the emptier side of the panel
                    lo, hi = lims[r]
                    mid = 0.5 * (lo + hi)
                    mass_right = sum(
                        float(np.sum(p["m1"][r][p["grids"][r] > mid]))
                        / max(float(np.sum(p["m1"][r])), 1e-300)
                        for p in posts) / len(posts)
                    right = mass_right < 0.5
                    for i, (txt, col) in enumerate(lines):
                        ax.text(0.96 if right else 0.04,
                                0.95 - i * (0.13 if small else 0.10), txt,
                                transform=ax.transAxes,
                                ha="right" if right else "left", va="top",
                                fontsize=fs, color=col,
                                bbox=dict(facecolor="white", alpha=0.75,
                                          edgecolor="none", pad=0.6))
            else:
                for p, col, ls in zip(posts, colors, styles):
                    X, Y = np.meshgrid(p["grids"][c], p["grids"][r],
                                       indexing="xy")
                    marg = p["m2"][(r, c)].T           # (n_r, n_c) -> rows=y
                    lv = hdr_levels(marg)
                    if len(lv) == 2:
                        ax.contour(X, Y, marg, levels=lv, colors=[col, col],
                                   linewidths=[0.6, 1.0],
                                   linestyles=["dotted", ls])
                ax.plot([REF_MW[c]], [REF_MW[r]], marker="P", ms=4,
                        color=C_MW, ls="none", mew=0.6)
                ax.plot([REF_LMC[c]], [REF_LMC[r]], marker="X", ms=4,
                        color=C_LMC, ls="none", mew=0.6)
                ax.set_ylim(*lims[r])
            ax.set_xlim(*lims[c])
            if r < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(LABELS[c])
                if small:
                    ax.tick_params(axis="x", labelrotation=45)
            if c > 0 or r == 0:
                ax.set_yticklabels([])
            elif r > 0:
                ax.set_ylabel(LABELS[r])
    handles = [Line2D([], [], color=col, ls=ls, lw=1.2, label=lab)
               for col, ls, lab in zip(colors, styles, labels)]
    if mean_lines:
        handles.append(Line2D([], [], color="black", ls=":", lw=0.8,
                              label="posterior mean"))
    handles += [Line2D([], [], color=C_MW, marker="P", ls="none", ms=4,
                       label="MW (Sana+2012)"),
                Line2D([], [], color=C_LMC, marker="X", ls="none", ms=4,
                       label="LMC (Almeida+2017, Shenar+2022)")]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.97, 0.97), frameon=False,
               fontsize=7 if small else 9)
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (out_stem, ext))
    plt.close(fig)
    return ["%s.%s" % (out_stem, ext) for ext in ("pdf", "png")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", action="append", required=True,
                    help="label:test:tag:run_dir (repeatable)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="corner_compare_real")
    ap.add_argument("--colors", default=None,
                    help="comma-separated colours, one per --run")
    ap.add_argument("--styles", default=None,
                    help="comma-separated line styles, one per --run")
    ap.add_argument("--include-refs", action="store_true",
                    help="widen the axes to show the MW and LMC markers")
    ap.add_argument("--annotate", action="store_true",
                    help="write posterior means and literature values in "
                         "the diagonal panels")
    ap.add_argument("--no-mean-lines", action="store_true",
                    help="omit the vertical posterior-mean lines and the "
                         "mean +/- 68%% titles above the diagonal panels")
    a = ap.parse_args(argv)
    colors = a.colors.split(",") if a.colors else None
    styles = a.styles.split(",") if a.styles else None
    posts, labels = [], []
    for spec in a.run:
        label, test, tag, run_dir = spec.split(":", 3)
        posts.append(load_posterior(run_dir, test, tag))
        labels.append(label)
        q = posts[-1]["q"]
        print("%-28s pi %+.3f [%+.3f,%+.3f]  kappa %+.3f [%+.3f,%+.3f]  "
              "eta %+.3f [%+.3f,%+.3f]  fbin %.3f [%.3f,%.3f]  means %s"
              % (label, q[0, 2], q[0, 1], q[0, 3], q[1, 2], q[1, 1], q[1, 3],
                 q[2, 2], q[2, 1], q[2, 3], q[3, 2], q[3, 1], q[3, 3],
                 ", ".join("%+.3f" % m for m in posts[-1]["mean"])))
    os.makedirs(a.out_dir, exist_ok=True)
    files = render(posts, labels, AA_SINGLE_COL,
                   os.path.join(a.out_dir, a.stem), colors, styles,
                   include_refs=a.include_refs, annotate=a.annotate,
                   mean_lines=not a.no_mean_lines)
    files += render(posts, labels, 7.0,
                    os.path.join(a.out_dir, a.stem + "_large"), colors,
                    styles, include_refs=a.include_refs, annotate=a.annotate,
                    mean_lines=not a.no_mean_lines)
    for f in files:
        print("wrote", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
