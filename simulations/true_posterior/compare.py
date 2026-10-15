"""
simulations.true_posterior.compare — CvM pseudo-posterior vs true
(likelihood) posterior on the same data.

Both use the shared cell-centred quantile estimator (bias_grid_lib.
marginals). The CvM cube is read at its (variant, metric) and the
likelihood cube at (likmix__numerical__lucyT, lik). The script writes a
CSV and a Markdown table of p16/p50/p84 per axis, the shift of the median
in units of the CvM 68% half-width, the width ratio, and a PNG/PDF with
the 1-D marginals overlaid.

    python -m simulations.true_posterior.compare \
        --cvm-run <dir> [--cvm-variant eccentric_only__numerical__lucyT] \
        --lik-run <dir> --out-dir <dir>
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from simulations.bias_grid_lib.cube_io import read_gmf_cube
from simulations.bias_grid_lib.marginals import grid_quantiles

AXES = ("pi", "kappa", "eta", "fbin")
LABELS = {"pi": r"$\pi$", "kappa": r"$\kappa$", "eta": r"$\eta$",
          "fbin": r"$f_\mathrm{bin}$"}


def marginals(run_dir, test, tag):
    z = np.load(os.path.join(run_dir, "grid_cubes.npz"), allow_pickle=True)
    cube, tag = read_gmf_cube(z, test=test, tag=tag)
    cube = np.asarray(cube, float)
    prob = np.nan_to_num(np.exp(cube - np.nanmax(cube)))
    grids = [np.asarray(z[k], float) for k in ("pi_grid", "kappa_grid",
                                               "eta_grid", "fbin_grid")]
    out = {}
    for ax, name in enumerate(AXES):
        post = prob.sum(axis=tuple(a for a in range(4) if a != ax))
        q = grid_quantiles(post, grids[ax], (0.16, 0.5, 0.84))
        dens = post / np.trapz(post, grids[ax])
        out[name] = {"grid": grids[ax], "dens": dens, "p16": q[0],
                     "p50": q[1], "p84": q[2]}
    return out, tag


def compare(cvm_run, lik_run, cvm_variant, out_dir, cvm_metric="cvm"):
    cvm, ctag = marginals(cvm_run, cvm_metric, cvm_variant)
    lik, ltag = marginals(lik_run, "lik", "likmix__numerical__lucyT")
    rows = []
    for ax in AXES:
        c, l = cvm[ax], lik[ax]
        hw_c = 0.5 * (c["p84"] - c["p16"])
        hw_l = 0.5 * (l["p84"] - l["p16"])
        rows.append({"axis": ax,
                     "cvm_p16": c["p16"], "cvm_p50": c["p50"],
                     "cvm_p84": c["p84"],
                     "lik_p16": l["p16"], "lik_p50": l["p50"],
                     "lik_p84": l["p84"],
                     "shift_in_cvm_hw": (l["p50"] - c["p50"]) / hw_c,
                     "width_ratio_lik_over_cvm": hw_l / hw_c})
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "cvm_vs_lik.csv"), index=False)
    with open(os.path.join(out_dir, "cvm_vs_lik.md"), "w") as fh:
        fh.write("| axis | CvM p50 (p16, p84) | likelihood p50 (p16, p84) "
                 "| shift / CvM hw | width ratio |\n|---|---|---|---|---|\n")
        for r in rows:
            fh.write("| %s | %+.3f (%+.3f, %+.3f) | %+.3f (%+.3f, %+.3f) "
                     "| %+.2f | %.2f |\n"
                     % (r["axis"], r["cvm_p50"], r["cvm_p16"], r["cvm_p84"],
                        r["lik_p50"], r["lik_p16"], r["lik_p84"],
                        r["shift_in_cvm_hw"], r["width_ratio_lik_over_cvm"]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.9))
        for a, ax in zip(axs, AXES):
            for d, col, lab in ((cvm[ax], "#4477AA", "CvM (pseudo)"),
                                (lik[ax], "#CC6677", "likelihood")):
                a.plot(d["grid"], d["dens"], color=col, lw=1.2, label=lab)
                a.axvline(d["p50"], color=col, lw=0.8, ls="--")
            a.set_xlabel(LABELS[ax])
            a.set_yticks([])
        axs[0].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, "cvm_vs_lik.%s" % ext),
                        dpi=200)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 — the table is the product
        print("plot skipped: %s" % exc)
    print("CvM tag %s | likelihood tag %s" % (ctag, ltag))
    print(df.to_string(index=False, float_format=lambda v: "%+.3f" % v))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cvm-run", required=True)
    ap.add_argument("--cvm-variant", default="eccentric_only__numerical__lucyT")
    ap.add_argument("--cvm-metric", default="cvm")
    ap.add_argument("--lik-run", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    compare(a.cvm_run, a.lik_run, a.cvm_variant, a.out_dir,
            cvm_metric=a.cvm_metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
