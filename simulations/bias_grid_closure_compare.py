"""
simulations.bias_grid_closure_compare — read a finished bias_grid output
together with a truth manifest from bias_grid_closure_generate, and produce
recovery diagnostics (1D/2D marginals with truth markers + a one-row
summary CSV).

Usage:
    python -m simulations.bias_grid_closure_compare \
        --grid-dir $SCRIPTS_OUT/simulation_pipeline/bias_grid_closure/sana2012/grid_run/ \
        --truth-yaml $SCRIPTS_OUT/simulation_pipeline/bias_grid_closure/sana2012/truth_params.yaml \
        --output $SCRIPTS_OUT/simulation_pipeline/bias_grid_closure/sana2012/closure_report/ \
        [--test ks|ad|cvm]
"""

import argparse
import csv
import os
import sys

import numpy as np
import yaml

# numpy 2.x renamed trapz -> trapezoid
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARAM_NAMES = ["pi", "kappa", "eta", "fbin"]
PARAM_LABELS = {"pi": r"$\pi$", "kappa": r"$\kappa$",
                "eta": r"$\eta$", "fbin": r"$f_{\rm bin}$"}


def _marginalize_log(cube, keep_axes):
    """Marginalize a log-likelihood cube over all axes except keep_axes.

    Returns a log-likelihood array with shape matching cube along keep_axes.
    """
    sum_axes = tuple(i for i in range(cube.ndim) if i not in keep_axes)
    log_max = float(np.nanmax(cube))
    if not np.isfinite(log_max):
        log_max = 0.0
    L = np.exp(cube - log_max)
    L_marg = L.sum(axis=sum_axes)
    log_marg = np.log(np.maximum(L_marg, 1e-300)) + log_max
    return log_marg


def _nearest_index(grid, value):
    return int(np.argmin(np.abs(np.asarray(grid) - value)))


def _load_grid(grid_dir):
    path = os.path.join(grid_dir, "grid_cubes.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError("grid_cubes.npz not found in %s" % grid_dir)
    data = np.load(path, allow_pickle=True)
    return data


def _load_truth(truth_yaml):
    with open(truth_yaml, "r") as f:
        return yaml.safe_load(f)


def _plot_1d_marginals(grids, log_marg_1d, truth, recovered, out_path,
                       test_name):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, name in zip(axes, PARAM_NAMES):
        x = grids[name]
        log_y = log_marg_1d[name]
        y = np.exp(log_y - log_y.max())
        y /= _trapz(y, x) if len(x) > 1 else 1.0
        ax.plot(x, y, "-", color="black", lw=1.5)
        ax.axvline(truth[name], color="red", ls="--", lw=1.5,
                   label="truth=%.3f" % truth[name])
        ax.axvline(recovered[name], color="C0", ls="-", lw=1.2,
                   label="recovered=%.3f" % recovered[name])
        ax.set_xlabel(PARAM_LABELS[name])
        ax.set_ylabel("marg. posterior")
        ax.legend(fontsize=8)
    fig.suptitle("Closure 1D marginals (%s test)" % test_name.upper())
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_2d_marginals(grids, cube, truth, recovered, out_path, test_name):
    pairs = [("pi", "kappa"), ("pi", "eta"), ("pi", "fbin"),
             ("kappa", "eta"), ("kappa", "fbin"), ("eta", "fbin")]
    axis_of = {n: i for i, n in enumerate(PARAM_NAMES)}

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    for ax, (a, b) in zip(axes, pairs):
        keep = (axis_of[a], axis_of[b])
        marg = _marginalize_log(cube, keep)
        if axis_of[a] > axis_of[b]:
            marg = marg.T
        xs = grids[a]
        ys = grids[b]
        marg_disp = marg - np.nanmax(marg)
        im = ax.imshow(marg_disp.T, origin="lower", aspect="auto",
                       extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                       cmap="viridis")
        ax.plot(truth[a], truth[b], "rx", ms=12, mew=2, label="truth")
        ax.plot(recovered[a], recovered[b], "wo", ms=8,
                markeredgecolor="black", label="recovered")
        ax.set_xlabel(PARAM_LABELS[a])
        ax.set_ylabel(PARAM_LABELS[b])
        ax.legend(fontsize=8, loc="best")
        plt.colorbar(im, ax=ax, label="log L (rel.)")
    fig.suptitle("Closure 2D marginals (%s test)" % test_name.upper())
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True,
                    help="bias_grid output dir containing grid_cubes.npz.")
    ap.add_argument("--truth-yaml", required=True,
                    help="truth_params.yaml from bias_grid_closure_generate.")
    ap.add_argument("--output", default=None,
                    help="Where to write report (default: <grid-dir>/closure_report).")
    ap.add_argument("--test", choices=["ks", "ad", "cvm"], default="ks",
                    help="Which GMF cube to use for argmax/marginals.")
    args = ap.parse_args(argv)

    out_dir = args.output or os.path.join(args.grid_dir, "closure_report")
    os.makedirs(out_dir, exist_ok=True)

    grid = _load_grid(args.grid_dir)
    truth_meta = _load_truth(args.truth_yaml)
    truth = {k: float(v) for k, v in truth_meta["truth"].items()}

    grids = {
        "pi": np.asarray(grid["pi_grid"]),
        "kappa": np.asarray(grid["kappa_grid"]),
        "eta": np.asarray(grid["eta_grid"]),
        "fbin": np.asarray(grid["fbin_grid"]),
    }

    cube_key = "gmf_%s_cube" % args.test
    if cube_key not in grid.files:
        raise KeyError("%s not in %s; available: %s" %
                       (cube_key, args.grid_dir, list(grid.files)))
    cube = np.asarray(grid[cube_key])

    flat_idx = int(np.nanargmax(cube))
    idx = np.unravel_index(flat_idx, cube.shape)
    recovered = {
        "pi": float(grids["pi"][idx[0]]),
        "kappa": float(grids["kappa"][idx[1]]),
        "eta": float(grids["eta"][idx[2]]),
        "fbin": float(grids["fbin"][idx[3]]),
    }
    log_gmf_max = float(cube[idx])

    log_marg_1d = {
        n: _marginalize_log(cube, (i,))
        for i, n in enumerate(PARAM_NAMES)
    }

    truth_idx = {n: _nearest_index(grids[n], truth[n]) for n in PARAM_NAMES}
    log_gmf_at_truth = float(cube[
        truth_idx["pi"], truth_idx["kappa"],
        truth_idx["eta"], truth_idx["fbin"]
    ])

    _plot_1d_marginals(grids, log_marg_1d, truth, recovered,
                       os.path.join(out_dir, "closure_1d_marginals.pdf"),
                       args.test)
    _plot_2d_marginals(grids, cube, truth, recovered,
                       os.path.join(out_dir, "closure_2d_marginals.pdf"),
                       args.test)

    summary_row = {
        "test": args.test,
        "log_gmf_max": log_gmf_max,
        "log_gmf_at_truth": log_gmf_at_truth,
        "log_gmf_gap": log_gmf_max - log_gmf_at_truth,
    }
    for n in PARAM_NAMES:
        summary_row["truth_%s" % n] = truth[n]
        summary_row["recovered_%s" % n] = recovered[n]
        summary_row["delta_%s" % n] = recovered[n] - truth[n]
        steps = np.diff(grids[n])
        median_step = float(np.median(steps)) if len(steps) else float("nan")
        summary_row["grid_step_%s" % n] = median_step
        summary_row["delta_in_steps_%s" % n] = (
            (recovered[n] - truth[n]) / median_step
            if median_step and median_step > 0 else float("nan"))

    summary_path = os.path.join(out_dir, "closure_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        w.writeheader()
        w.writerow(summary_row)

    print("[closure-cmp] truth     :", truth)
    print("[closure-cmp] recovered :", recovered)
    print("[closure-cmp] delta     :",
          {n: recovered[n] - truth[n] for n in PARAM_NAMES})
    print("[closure-cmp] log_gmf max=%.3f at_truth=%.3f gap=%.3f" %
          (log_gmf_max, log_gmf_at_truth,
           log_gmf_max - log_gmf_at_truth))
    print("[closure-cmp] wrote: %s" % summary_path)
    print("[closure-cmp] wrote: %s/closure_1d_marginals.pdf" % out_dir)
    print("[closure-cmp] wrote: %s/closure_2d_marginals.pdf" % out_dir)


if __name__ == "__main__":
    sys.exit(main())
