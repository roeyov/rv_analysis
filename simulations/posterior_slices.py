"""Fixed-kappa conditional slices + marginal exports from a bias-grid cube.

Tier 3.2 + 3.3 support of the 2026-09 variant campaign (JV-023a, TSA-103):

* 1-D marginal pseudo-posteriors for (pi, kappa, eta, f_bin) with
  median / 16th / 84th percentiles — same math as
  make_bias_corner_figure.py (exp(logGMF - max), nansum-marginalise,
  trapz-normalise, percentiles from the cumulative trapezoid).
* kappa-CONDITIONAL slices: fix kappa at the node nearest each requested
  value, renormalise prob[:, j, :, :], and report conditional
  (pi, eta, f_bin) medians/percentiles.
* npz export for the paper-side estimators: the 2-D (kappa, f_bin) joint
  marginal (needed by the f_bin(q>=0.1) posterior-truncation estimator,
  which must respect the kappa<->f_bin correlation) + all 1-D marginals.

Deliberately does NOT import the simulations package's explorer (which
would pull in streamlit); the cube is read directly, mirroring
make_bias_corner_figure.py.

Usage:
  python -m simulations.posterior_slices \
      --run-dir /jonathan_storage/roeyovadia/bias_grid_results/ed_hoc_rlof_check_ext3 \
      --metric cvm --kappa-slices 0,-0.1,-1 \
      --dump /tmp/ext3_marginals_slices.npz
"""

import argparse
import os
import sys

import numpy as np

AXES = ("pi", "kappa", "eta", "f_bin")
GRID_KEYS = ("pi_grid", "kappa_grid", "eta_grid", "fbin_grid")


def load_cube(run_dir, variant, metric):
    path = None
    for name in ("grid_cubes.npz", "checkpoint_cubes.npz"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError("no cube npz in %s" % run_dir)
    z = np.load(path, allow_pickle=True)
    tags = [str(t) for t in np.atleast_1d(z["variants"])]
    if variant is None:
        if len(tags) != 1:
            raise ValueError("cube has %d variants (%s) — pass --variant"
                             % (len(tags), tags))
        variant = tags[0]
    key = "v__%s__gmf_%s_cube" % (variant, metric)
    if key not in z.files:
        raise KeyError("%s not in cube; gmf keys: %s"
                       % (key, [k for k in z.files if "gmf" in k]))
    grids = [np.asarray(z[g], dtype=float) for g in GRID_KEYS]
    return np.asarray(z[key], dtype=np.float64), grids, variant, path


def gmf_to_prob(cube):
    prob = np.exp(cube - np.nanmax(cube))
    prob[~np.isfinite(prob)] = 0.0
    return prob


def marginal_1d(prob, axis, grid):
    axes = tuple(i for i in range(4) if i != axis)
    m = np.nansum(prob, axis=axes)
    norm = np.trapz(m, grid)
    return m / norm if norm > 0 else m


def percentiles(m, grid, qs=(0.16, 0.50, 0.84)):
    cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (m[1:] + m[:-1]) * np.diff(grid))])
    if cdf[-1] <= 0:
        return [np.nan] * len(qs)
    cdf /= cdf[-1]
    return [float(np.interp(q, cdf, grid)) for q in qs]


def fmt(name, m, grid):
    p16, p50, p84 = percentiles(m, grid)
    mode = float(grid[int(np.nanargmax(m))])
    return ("%-6s median=%+.3f  (16-84%%: %+.3f .. %+.3f)  mode=%+.3f"
            % (name, p50, p16, p84, mode)), (p16, p50, p84, mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", default=None,
                    help="variant tag (default: the cube's single variant)")
    ap.add_argument("--metric", default="cvm",
                    choices=["ks", "ad", "cvm", "wass"])
    ap.add_argument("--kappa-slices", default="0,-0.1,-1",
                    help="comma-separated kappa values to condition on")
    ap.add_argument("--dump", default=None,
                    help="npz output path (marginals + 2-D (kappa,f_bin))")
    args = ap.parse_args()

    cube, grids, variant, path = load_cube(args.run_dir, args.variant,
                                           args.metric)
    print("cube: %s [%s, %s] shape=%s" % (path, variant, args.metric,
                                          cube.shape))
    prob = gmf_to_prob(cube)

    print("\n== Full 1-D marginal pseudo-posteriors ==")
    dump = {"variant": variant, "metric": args.metric}
    for ax, (name, grid) in enumerate(zip(AXES, grids)):
        m = marginal_1d(prob, ax, grid)
        line, (p16, p50, p84, mode) = fmt(name, m, grid)
        print("  " + line)
        dump["%s_grid" % name] = grid
        dump["%s_post" % name] = m
        dump["%s_p16" % name] = p16
        dump["%s_p50" % name] = p50
        dump["%s_p84" % name] = p84

    # 2-D (kappa, f_bin) joint marginal for the truncation estimator.
    m2 = np.nansum(prob, axis=(0, 2))          # -> (kappa, f_bin)
    dump["kappa_fbin_2d"] = m2 / m2.sum() if m2.sum() > 0 else m2

    kappa_grid = grids[1]
    print("\n== kappa-conditional slices (JV-023a) ==")
    slice_vals = [float(s) for s in args.kappa_slices.split(",")]
    for kv in slice_vals:
        j = int(np.argmin(np.abs(kappa_grid - kv)))
        sl = prob[:, j, :, :]
        tot = sl.sum()
        if tot <= 0:
            print("  kappa=%+.2f (node %+.3f): zero probability — skipped"
                  % (kv, kappa_grid[j]))
            continue
        sl = sl / tot
        print("  kappa=%+.2f -> nearest node %+.3f "
              "(P(node)/P(marginal max) = %.3g)"
              % (kv, kappa_grid[j],
                 tot / np.nansum(prob, axis=(0, 2, 3)).max()))
        for ax3, (name, grid) in enumerate(
                [(n, g) for i, (n, g) in enumerate(zip(AXES, grids))
                 if i != 1]):
            axes = tuple(i for i in range(3) if i != ax3)
            m = np.nansum(sl, axis=axes)
            norm = np.trapz(m, grid)
            if norm > 0:
                m = m / norm
            line, (p16, p50, p84, _mode) = fmt(name, m, grid)
            print("    " + line)
            tagk = "cond_k%+g_%s" % (kv, name)
            dump[tagk + "_post"] = m
            dump[tagk + "_p50"] = p50
            dump[tagk + "_p16"] = p16
            dump[tagk + "_p84"] = p84

    if args.dump:
        np.savez_compressed(args.dump, **dump)
        print("\nwrote %s" % args.dump)
    return 0


if __name__ == "__main__":
    sys.exit(main())
