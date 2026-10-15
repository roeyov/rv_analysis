"""
simulations.closure_coverage — score-derived interval coverage from mock
surveys (Tier-4 coverage campaign, option B).

Two modes:

per-run — read one finished mock grid (grid_cubes.npz) + its truth manifest,
compute the 1-D marginal equal-tail intervals at 68% and 95% with the shared
cell-centred grid estimator (``bias_grid_lib.marginals.grid_quantiles``, also
used by Ostars_article/scripts/make_bias_corner_figure.py `median_ci`), and
emit one CSV row per axis with the truth-containment flags:

    python -m simulations.closure_coverage per-run \
        --grid-dir <seed_dir>/grid_run --truth-yaml <seed_dir>/truth_params.yaml \
        --truth-tag adopted --seed 101 [--variant ...] [--test cvm] [--out ...]

Rows written before 2026-09-29 used the old cumsum step-CDF, which put every
interior quantile half a grid step low (see marginals.py). Recompute them with
``--row-name`` set to a new file name so the old rows are not overwritten.

aggregate — collect all per-run rows under a campaign dir laid out as
<campaign>/<truth_tag>/seed_<n>/grid_run/closure_report/<row-name> and
print/write the coverage table (per axis x per truth, 68 and 95, with Wilson
binomial intervals):

    python -m simulations.closure_coverage aggregate \
        --campaign-dir /jonathan_storage/roeyovadia/bias_grid_results/tier4 \
        [--row-name coverage_row.csv] [--out <campaign>/coverage_table.csv]
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
import yaml

from simulations.bias_grid_lib.cube_io import read_gmf_cube
from simulations.bias_grid_lib.marginals import c_marginals, grid_quantiles

AXES = ["pi", "kappa", "eta", "fbin"]
GRID_KEY = {"pi": "pi_grid", "kappa": "kappa_grid",
            "eta": "eta_grid", "fbin": "fbin_grid"}
QUANTILES = (0.025, 0.16, 0.50, 0.84, 0.975)
QCOLS = ["p2_5", "p16", "p50", "p84", "p97_5"]

DEFAULT_TEST = "cvm"
DEFAULT_VARIANT = "eccentric_only__numerical__lucyT"
DEFAULT_ROW_NAME = "coverage_row.csv"


# ---------------------------------------------------------------------------
# Posterior -> interval math (same as make_bias_corner_figure.py)
# ---------------------------------------------------------------------------

def gmf_to_prob(gmf_cube):
    """prob = exp(gmf - nanmax(gmf)), NaN -> 0."""
    shifted = gmf_cube - np.nanmax(gmf_cube)
    prob = np.exp(shifted)
    return np.nan_to_num(prob, nan=0.0)


_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def marginal_1d(prob, axis_idx, grid):
    """Marginalize the 4-D probability to 1-D for ``axis_idx``.

    Same steps as make_bias_corner_figure's get_1d_posterior (nansum, then
    trapezoid normalization). The normalization does not change the
    quantiles; it only keeps the returned density on the same scale as the
    paper script's.
    """
    axes_to_sum = tuple(i for i in range(prob.ndim) if i != axis_idx)
    post = np.nansum(prob, axis=axes_to_sum)
    if np.sum(post) > 0 and len(grid) > 1:
        post = post / _trapz(post, grid)
    return post


def cdf_quantiles(post, grid, qs=QUANTILES, method=None):
    """Equal-tail quantiles of a 1-D marginal (grid CDF; see marginals)."""
    return grid_quantiles(post, grid, qs, method=method)


def signed_z(truth, qv):
    """(truth - p50) in units of the 68% half-width on the truth's side.

    |z| <= 1 exactly when the truth lies inside [p16, p84]. NaN when the
    relevant half-width is zero.
    """
    p16, p50, p84 = qv[1], qv[2], qv[3]
    hw = (p84 - p50) if truth >= p50 else (p50 - p16)
    return float((truth - p50) / hw) if hw > 0 else float("nan")


def _row(truth_tag, seed, axis, truth, qv, grid=None, post=None):
    row = {"truth_tag": truth_tag, "seed": int(seed), "axis": axis,
           "truth": truth}
    row.update(dict(zip(QCOLS, qv)))
    row["in68"] = int(qv[1] <= truth <= qv[3])
    row["in95"] = int(qv[0] <= truth <= qv[4])
    row["z68"] = signed_z(truth, qv)
    row["hw68"] = 0.5 * (qv[3] - qv[1])
    if grid is not None and len(grid) > 1:
        row["grid_step"] = float(np.median(np.diff(grid)))
    if post is not None and np.sum(post) > 0:
        # posterior mass on the two end nodes: a prior-box truncation flag
        row["edge_mass"] = float((post[0] + post[-1]) / np.sum(post))
    return row


SPLIT_EXTRA_AXES = ("f_gt", "f_lt", "log10_rho", "c1", "c2")


def split_extra_rows(grid_dir, truth, truth_tag, seed, method=None):
    """Coverage rows for the split model's extra parameters.

    Rows are produced for every key of ``SPLIT_EXTRA_AXES`` present in
    ``truth`` when ``grid_cubes_fgt.npz`` exists in ``grid_dir``. The
    marginals follow run_posterior.summarize_split: f_gt from the f_gt cube,
    f_lt from the global (f_gt, f_lt) table, log10 rho from its binned
    table (underflow/overflow bins at the ends) and c1, c2 from the c table.
    """
    fgt_path = os.path.join(grid_dir, "grid_cubes_fgt.npz")
    if not os.path.exists(fgt_path):
        return []
    zf = np.load(fgt_path, allow_pickle=True)
    z = np.load(os.path.join(grid_dir, "grid_cubes.npz"), allow_pickle=True)
    rows = []
    if "f_gt" in truth:
        key = [k for k in zf.files if k.endswith("__gmf_lik_cube")][0]
        prob = gmf_to_prob(np.asarray(zf[key], dtype=np.float64))
        grid = np.asarray(zf["fg_grid"], float)
        post = prob.sum(axis=(0, 1, 2))
        rows.append(_row(truth_tag, seed, "f_gt", truth["f_gt"],
                         cdf_quantiles(post, grid, method=method), grid,
                         post))
    if "f_lt" in truth:
        fgfl = np.asarray(z["lnpost_fg_fl"], float)
        grid = np.asarray(z["fl_grid"], float)
        post = np.exp(fgfl - np.max(fgfl)).sum(axis=0)
        rows.append(_row(truth_tag, seed, "f_lt", truth["f_lt"],
                         cdf_quantiles(post, grid, method=method), grid,
                         post))
    if "log10_rho" in truth:
        lr = np.asarray(z["lnpost_rho"], float)
        edges = np.asarray(z["rho_edges_log10"], float)
        centres = np.concatenate([[edges[0] - 0.01],
                                  0.5 * (edges[1:] + edges[:-1]),
                                  [edges[-1] + 0.01]])
        post = np.exp(lr - np.max(lr))
        rows.append(_row(truth_tag, seed, "log10_rho", truth["log10_rho"],
                         cdf_quantiles(post, centres, method=method),
                         centres, post))
    # c marginals integrate the other fraction over its cells, as
    # run_posterior.summarize does (an uneven lattice is not a plain sum)
    marg = dict(zip(("c1", "c2"),
                    c_marginals(z["lnpost_c1c2"], z["c_grid"], z["c2_grid"])))
    for name, gkey in (("c1", "c_grid"), ("c2", "c2_grid")):
        if name in truth:
            grid = np.asarray(z[gkey], float)
            post = marg[name]
            rows.append(_row(truth_tag, seed, name, truth[name],
                             cdf_quantiles(post, grid, method=method), grid,
                             post))
    return rows


def coverage_rows_for_run(grid_dir, truth_yaml, truth_tag, seed,
                          variant=DEFAULT_VARIANT, test=DEFAULT_TEST,
                          method=None, extra=True):
    """One coverage row per axis for a finished mock grid.

    ``method`` selects the quantile estimator (see marginals.grid_quantiles;
    None = the package default). With ``extra`` (default), split-model runs
    whose truth yaml carries f_gt / f_lt / log10_rho / c1 / c2 also get rows
    for those parameters.
    """
    cubes_path = os.path.join(grid_dir, "grid_cubes.npz")
    z = np.load(cubes_path, allow_pickle=True)
    cube, resolved_tag = read_gmf_cube(z, test=test, tag=variant)
    prob = gmf_to_prob(np.asarray(cube, dtype=np.float64))

    with open(truth_yaml) as fh:
        truth_meta = yaml.safe_load(fh)
    truth = {k: float(v) for k, v in truth_meta["truth"].items()
             if v is not None}

    rows = []
    for i, ax in enumerate(AXES):
        grid = np.asarray(z[GRID_KEY[ax]], dtype=float)
        post = marginal_1d(prob, i, grid)
        qv = cdf_quantiles(post, grid, method=method)
        rows.append(_row(truth_tag, seed, ax, truth[ax], qv, grid, post))
    if extra and test == "lik" and "split" in str(resolved_tag):
        rows += split_extra_rows(grid_dir, truth, truth_tag, seed,
                                 method=method)
    for row in rows:
        row["variant"] = resolved_tag
        row["test"] = test
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def wilson_interval(k, n, z=1.0):
    """Wilson score interval for a binomial fraction (z=1 -> ~68% CI)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - half, center + half


def coverage_table(df):
    """Coverage table (per truth x axis, plus POOLED) from per-mock rows.

    Besides the 68/95 containment fractions with Wilson intervals, each
    group carries the mean and median of the signed z (when the rows have
    ``z68``: (truth - p50) in 68% half-widths, so a mean far from 0 flags an
    offset rather than a width error) and the mean 68% half-width.
    """
    out = []
    groups = list(df.groupby(["truth_tag", "axis"], sort=False))
    # pooled-over-truths rows per axis, in the rows' axis order
    for ax in list(dict.fromkeys(df["axis"])):
        sub = df[df["axis"] == ax]
        if len(sub):
            groups.append((("POOLED", ax), sub))
    for (tag, ax), sub in groups:
        n = len(sub)
        for level, col, nominal in (("68", "in68", 0.68), ("95", "in95", 0.95)):
            k = int(sub[col].sum())
            lo, hi = wilson_interval(k, n)
            rec = {"truth_tag": tag, "axis": ax, "level": level,
                   "n_mocks": n, "n_contained": k,
                   "coverage": k / n, "wilson_lo": lo, "wilson_hi": hi,
                   "nominal": nominal,
                   "consistent": int(lo <= nominal <= hi)}
            if "z68" in sub.columns:
                zz = sub["z68"].to_numpy(float)
                zz = zz[np.isfinite(zz)]
                rec["z_mean"] = float(zz.mean()) if zz.size else float("nan")
                rec["z_median"] = (float(np.median(zz)) if zz.size
                                   else float("nan"))
                rec["n_below"] = int(np.sum(zz < -1)) if level == "68" \
                    else int(np.sum(sub["truth"] < sub["p2_5"]))
                rec["n_above"] = int(np.sum(zz > 1)) if level == "68" \
                    else int(np.sum(sub["truth"] > sub["p97_5"]))
            if "hw68" in sub.columns:
                rec["hw68_mean"] = float(sub["hw68"].mean())
            out.append(rec)
    return pd.DataFrame(out)


def aggregate(campaign_dir, out_csv=None, row_name=DEFAULT_ROW_NAME):
    pattern = os.path.join(campaign_dir, "*", "seed_*", "grid_run",
                           "closure_report", row_name)
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("No coverage_row.csv files under %s" % pattern)
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    table = coverage_table(df)

    print("=" * 78)
    print("Interval coverage (%d mock rows, %d mocks/truth-axis max)" %
          (len(df), df.groupby(["truth_tag", "axis"]).size().max()))
    print("=" * 78)
    with pd.option_context("display.width", 120):
        print(table.to_string(index=False,
                              float_format=lambda v: "%.3f" % v))
    if out_csv:
        table.to_csv(out_csv, index=False)
        df.to_csv(os.path.splitext(out_csv)[0] + "_rows.csv", index=False)
        print("\nWrote %s (+ _rows.csv with all per-mock rows)" % out_csv)
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("per-run", help="one coverage row per axis for a "
                                        "finished mock grid")
    pr.add_argument("--grid-dir", required=True)
    pr.add_argument("--truth-yaml", required=True)
    pr.add_argument("--truth-tag", required=True)
    pr.add_argument("--seed", type=int, required=True)
    pr.add_argument("--variant", default=DEFAULT_VARIANT)
    pr.add_argument("--test", default=DEFAULT_TEST,
                    choices=["ks", "ad", "cvm", "wass", "lik"])
    pr.add_argument("--row-name", default=DEFAULT_ROW_NAME,
                    help="file name of the row CSV inside "
                         "<grid-dir>/closure_report/ (default %(default)s); "
                         "ignored when --out is given")
    pr.add_argument("--out", default=None,
                    help="CSV path (default <grid-dir>/closure_report/"
                         "<row-name>)")
    pr.add_argument("--method", default=None, choices=["cell", "linear"],
                    help="quantile estimator (default: marginals."
                         "QUANTILE_METHOD)")

    ag = sub.add_parser("aggregate", help="coverage table over a campaign dir")
    ag.add_argument("--campaign-dir", required=True)
    ag.add_argument("--row-name", default=DEFAULT_ROW_NAME,
                    help="per-run row file name to collect "
                         "(default %(default)s)")
    ag.add_argument("--out", default=None)

    args = ap.parse_args(argv)

    if args.mode == "per-run":
        rows = coverage_rows_for_run(args.grid_dir, args.truth_yaml,
                                     args.truth_tag, args.seed,
                                     variant=args.variant, test=args.test,
                                     method=args.method)
        out = args.out or os.path.join(args.grid_dir, "closure_report",
                                       args.row_name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        rows.to_csv(out, index=False)
        print(rows.to_string(index=False,
                             float_format=lambda v: "%.4f" % v))
        print("[coverage] wrote %s" % out)
    else:
        aggregate(args.campaign_dir, out_csv=args.out,
                  row_name=args.row_name)


if __name__ == "__main__":
    sys.exit(main())
