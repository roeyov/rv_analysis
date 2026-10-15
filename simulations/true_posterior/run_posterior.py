"""
simulations.true_posterior.run_posterior — evaluate the true (likelihood)
posterior on a grid and write a cube the existing tools can read.

Inputs:
  * a labelled injection set (``injection_set.py``) for the detector that
    produced the data;
  * a per-star observed table with the obs_catalog schema: the real data
    (``obs_catalog.py``) or a mock catalogue (``mock_catalog.py``).

Output ``<out-dir>/grid_cubes.npz`` holds pi/kappa/eta/fbin grids, a
``variants`` list and ``v__likmix__numerical__lucyT__gmf_lik_cube``, the
log posterior up to a constant with uniform priors on the grid box and the
circular fractions c1, c2 marginalized. make_bias_corner_figure.py
(``--metric lik``), closure_coverage.py (``--test lik``) and
posterior_slices.py read it as they read CvM cubes. The run also writes
``posterior_summary.yaml`` (marginal quantiles from the shared cell-centred
estimator, plus the c1/c2 posterior) and ``run_posterior.yaml``
(provenance).

    python -m simulations.true_posterior.run_posterior \
        --injset <dir> --obs <obs_catalog.csv> --out-dir <dir> \
        [--preset ed_hoc_rlof_check_ext2] [--f-grid 0.30,0.95,131] \
        [--c-grid 21 | --c-nodes 0,0.1,...,1] [--q-min 0.1] [--n-workers 8] \
        [--model split [--split-f fg_lo,fg_hi,fl_hi,d]] [--k1-floor 10]

Grid specs are ``lo,hi,n``. Attach a value that starts with a minus sign
with ``=`` (``--pi-grid=-2.0,0.15,44``), or argparse reads it as an option.
"""

import argparse
import datetime as _dt
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

from simulations.bias_config import GRID_PRESETS
from simulations.bias_grid_lib.marginals import c_marginals, grid_quantiles
from simulations.true_posterior.injection_set import load_injection_set
from simulations.true_posterior.likelihood import (
    InjectionSet, ObservedData, node_loglike, node_split_outputs,
    node_table_split, shape_precompute)
from simulations.true_posterior.population import powerlaw_cdf

TAG = "likmix__numerical__lucyT"
TAG_SPLIT = "likmixsplit__numerical__lucyT"
METRIC = "lik"

# Split model default f lattice (see ``split_f_lattice``): f_gt and f_lt share
# spacing 0.02 so f_tot = f_gt + f_lt lands on 0.20 + 0.02 n (41 nodes up to
# 1.0). ``--split-f`` overrides the three arrays for one run.
SPLIT_FG = np.round(np.arange(0.20, 0.9001, 0.02), 10)
SPLIT_FL = np.round(np.arange(0.00, 0.4001, 0.02), 10)
SPLIT_FT = np.round(np.arange(0.20, 1.0001, 0.02), 10)
SPLIT_C = (0.0, 0.03, 0.1, 0.2, 0.4, 0.7, 1.0)
RHO_EDGES = np.round(np.arange(-2.0, 1.0001, 0.02), 10)   # log10 rho
AXES = ("pi", "kappa", "eta", "fbin")


def split_f_lattice(spec=None):
    """(fg, fl, ft) node arrays of the split model's f lattice.

    ``spec`` is None for the module defaults (SPLIT_FG, SPLIT_FL, SPLIT_FT)
    or a string ``"fg_lo,fg_hi,fl_hi,d"``: f_gt on fg_lo..fg_hi and f_lt on
    0..fl_hi with one shared step d, so that f_tot = fg_lo + (g + l) d
    (``node_split_outputs`` maps cell (g, l) to f_tot index g + l). ft holds
    exactly the reachable f_tot nodes <= 1; cells with f_gt + f_lt > 1 are
    masked by the simplex prior in ``node_table_split``.
    """
    if spec is None:
        return SPLIT_FG, SPLIT_FL, SPLIT_FT
    try:
        fg_lo, fg_hi, fl_hi, d = (float(v) for v in str(spec).split(","))
    except ValueError:
        raise ValueError("--split-f wants fg_lo,fg_hi,fl_hi,d, got %r"
                         % (spec,)) from None
    if not (d > 0 and 0 < fg_lo < fg_hi <= 1 and 0 < fl_hi <= 1):
        raise ValueError("--split-f: need d > 0, 0 < fg_lo < fg_hi <= 1 and "
                         "0 < fl_hi <= 1, got %r" % (spec,))

    def n_nodes(span):
        n = span / d
        if abs(n - round(n)) > 1e-6:
            raise ValueError("--split-f: span %g is not a multiple of d=%g"
                             % (span, d))
        return int(round(n)) + 1

    n_fg, n_fl = n_nodes(fg_hi - fg_lo), n_nodes(fl_hi)
    # f_tot indices g + l with finite mass: <= n_fg + n_fl - 2, and f_tot <= 1
    n_ft = min(n_fg + n_fl - 1, int(np.floor((1.0 - fg_lo) / d + 1e-9)) + 1)

    def lat(lo, n):
        return np.round(lo + d * np.arange(n), 10)
    return lat(fg_lo, n_fg), lat(0.0, n_fl), lat(fg_lo, n_ft)

# Worker globals (fork start method: inherited, not pickled).
_G = {}


def _node_block(args):
    i, k = args
    T, shape, obs, beta, f_grid, c_grid, c2_grid, n_eta, count = (
        _G["T"], _G["shape"], _G["obs"], _G["beta"], _G["f_grid"],
        _G["c_grid"], _G["c2_grid"], _G["n_eta"], _G["count"])
    out = np.empty((n_eta, len(f_grid)))
    outc = np.empty((n_eta, len(c_grid), len(c2_grid)))
    for l in range(n_eta):
        out[l], outc[l] = node_loglike(T, shape, obs, beta, i, k, l,
                                       f_grid, c_grid, return_c=True,
                                       c2_grid=c2_grid, count=count)
    return i, k, out, outc


def _tensors(inj, base_tensors, pi_grid, kappa_grid, eta_grid, cut, q_min,
             cut_groups):
    """The alpha tensors for one data set.

    ``base_tensors`` (from ``alpha_tensors`` on the same grid and q_min) are
    shared across data sets: den and num_all never depend on the cut; the
    cut-dependent groups in ``cut_groups`` are reused too when the dict
    carries them together with ``"_cut"`` equal to this data set's cut
    (a fixed-cut campaign), else they are computed here.
    """
    if base_tensors is None:
        return inj.alpha_tensors(pi_grid, kappa_grid, eta_grid, cut,
                                 q_min=q_min,
                                 groups=("den", "num_all") + tuple(cut_groups))
    T = dict(base_tensors)
    same_cut = ("_cut" in T and cut is not None
                and abs(float(T["_cut"]) - float(cut)) < 1e-12)
    todo = tuple(g for g in cut_groups if not (same_cut and g in T))
    if todo:
        T.update(inj.alpha_tensors(pi_grid, kappa_grid, eta_grid, cut,
                                   q_min=q_min, groups=todo))
    return T


def shared_tensors(inj, grids, cut=None, q_min=None):
    """Base tensors for a campaign: den + num_all, plus the cut-dependent
    groups (tagged with ``_cut``) when every data set uses the same cut."""
    groups = ("den", "num_all")
    if cut is not None:
        groups += ("num_cut", "den_cut")
    T = inj.alpha_tensors(grids[0], grids[1], grids[2], cut, q_min=q_min,
                          groups=groups)
    if cut is not None:
        T["_cut"] = float(cut)
    return T


def evaluate_grid(inj, obs, pi_grid, kappa_grid, eta_grid, f_grid, c_grid,
                  q_min=None, n_workers=1, use_errors=False, log=print,
                  base_tensors=None, c2_grid=None, count=True):
    """ln L(pi, kappa, eta, f) and the c1/c2 table, summed over nodes.

    ``base_tensors`` (den + num_all, from ``alpha_tensors`` on the same grid
    and q_min) can be shared across data sets; then only the cut-dependent
    num_cut group is computed here (see ``_tensors``).
    """
    t0 = time.time()
    T = _tensors(inj, base_tensors, pi_grid, kappa_grid, eta_grid, obs.cut,
                 q_min, ("num_cut",))
    log("[post] alpha tensors: %.0f s" % (time.time() - t0))
    shape = shape_precompute(obs, inj, pi_grid, kappa_grid, eta_grid,
                             q_min=q_min, use_errors=use_errors)
    log("[post] shape tables: %.0f s" % (time.time() - t0))
    c2_grid = c_grid if c2_grid is None else c2_grid
    _G.update(T=T, shape=shape, obs=obs, beta=inj.beta, f_grid=f_grid,
              c_grid=c_grid, c2_grid=c2_grid, n_eta=len(eta_grid),
              count=count)
    npi, nka = len(pi_grid), len(kappa_grid)
    cube = np.empty((npi, nka, len(eta_grid), len(f_grid)))
    lnc_all = np.full((len(c_grid), len(c2_grid)), -np.inf)
    jobs = [(i, k) for i in range(npi) for k in range(nka)]
    if n_workers > 1 and sys.platform.startswith("linux"):
        import multiprocessing as mp
        with mp.get_context("fork").Pool(n_workers) as pool:
            results = pool.imap_unordered(_node_block, jobs, chunksize=2)
            for i, k, out, outc in results:
                cube[i, k] = out
                lnc_all = np.logaddexp(lnc_all, np.logaddexp.reduce(outc, 0))
    else:
        for job in jobs:
            i, k, out, outc = _node_block(job)
            cube[i, k] = out
            lnc_all = np.logaddexp(lnc_all, np.logaddexp.reduce(outc, 0))
    log("[post] grid ln L: %.0f s (%d nodes)"
        % (time.time() - t0, cube[..., 0].size))
    return cube, lnc_all


def summarize(cube, grids, c_grid, lnc, c2_grid=None):
    c2_grid = c_grid if c2_grid is None else c2_grid
    prob = np.exp(cube - np.nanmax(cube))
    out = {}
    for ax, name in enumerate(AXES):
        post = prob.sum(axis=tuple(a for a in range(4) if a != ax))
        p16, p50, p84 = grid_quantiles(post, grids[ax], (0.16, 0.5, 0.84))
        p025, p975 = grid_quantiles(post, grids[ax], (0.025, 0.975))
        edge = float((post[0] + post[-1]) / post.sum())
        out[name] = {"p2_5": p025, "p16": p16, "p50": p50, "p84": p84,
                     "p97_5": p975,
                     "mode": float(grids[ax][int(np.argmax(post))]),
                     "edge_node_mass": edge}
    # each c marginal integrates the other fraction over its cells (the flat
    # prior), not a plain sum over its nodes, which an uneven lattice skews
    m1, m2 = c_marginals(lnc, c_grid, c2_grid)
    for name, m, g in (("c1", m1, c_grid), ("c2", m2, c2_grid)):
        p16, p50, p84 = grid_quantiles(m, g, (0.16, 0.5, 0.84))
        out[name] = {"p16": p16, "p50": p50, "p84": p84}
    return out


def write_outputs(out_dir, cube, grids, lnc, c_grid, summary, meta,
                  c2_grid=None):
    c2_grid = c_grid if c2_grid is None else c2_grid
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "grid_cubes.npz"),
             cube_schema_version=np.array(5),
             variants=np.array([TAG]),
             pi_grid=grids[0], kappa_grid=grids[1], eta_grid=grids[2],
             fbin_grid=grids[3],
             **{"v__%s__gmf_%s_cube" % (TAG, METRIC): cube.astype(np.float64)},
             c_grid=np.asarray(c_grid, float),
             c2_grid=np.asarray(c2_grid, float), lnpost_c1c2=lnc)
    with open(os.path.join(out_dir, "posterior_summary.yaml"), "w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)
    with open(os.path.join(out_dir, "run_posterior.yaml"), "w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)


def _node_block_split(args):
    i, k = args
    g = _G
    n_eta = g["n_eta"]
    ft = np.empty((n_eta, g["n_ft"]))
    fgm = np.empty((n_eta, g["fg"].size))
    fgfl = np.full((g["fg"].size, g["fl"].size), -np.inf)
    rho = np.full(len(RHO_EDGES) + 1, -np.inf)
    lnc = np.full((g["c"].size, g["c"].size), -np.inf)
    for l in range(n_eta):
        tot = node_table_split(g["T"], g["shape"], g["obs"], g["beta"], i, k,
                               l, g["fg"], g["fl"], g["c"], g["c"],
                               g["F_gt"][i])
        a, b, c2d, r, cc = node_split_outputs(tot, g["fg"], g["fl"], g["c"],
                                              g["c"], g["n_ft"], g["F_gt"][i],
                                              RHO_EDGES)
        ft[l], fgm[l] = a, b
        fgfl = np.logaddexp(fgfl, c2d)
        rho = np.logaddexp(rho, r)
        lnc = np.logaddexp(lnc, cc)
    return i, k, ft, fgm, fgfl, rho, lnc


def evaluate_grid_split(inj, obs, pi_grid, kappa_grid, eta_grid, q_min=None,
                        n_workers=1, use_errors=False, log=print,
                        base_tensors=None, c_grid=None, f_lattice=None):
    """Split-model grid: ln L(pi, kappa, eta, f_tot), ln L(..., f_gt) and
    global (f_gt, f_lt), rho and c tables. ``c_grid`` (default SPLIT_C)
    holds the c1 = c2 nodes; a single node [0.0] fixes both to zero.
    ``f_lattice`` is a (fg, fl, ft) triple from ``split_f_lattice``
    (default: the module lattice)."""
    t0 = time.time()
    fg, fl, ft = split_f_lattice() if f_lattice is None else f_lattice
    T = _tensors(inj, base_tensors, pi_grid, kappa_grid, eta_grid, obs.cut,
                 q_min, ("num_cut", "den_cut"))
    shape = shape_precompute(obs, inj, pi_grid, kappa_grid, eta_grid,
                             q_min=q_min, use_errors=use_errors)
    b = inj.bounds
    F_gt = np.array([1.0 - float(powerlaw_cdf(obs.cut, p, b["log_p_min"],
                                              b["log_p_max"]))
                     for p in pi_grid])
    c = np.asarray(SPLIT_C if c_grid is None else c_grid, float)
    _G.update(T=T, shape=shape, obs=obs, beta=inj.beta, fg=fg, fl=fl, c=c,
              n_ft=ft.size, F_gt=F_gt, n_eta=len(eta_grid))
    npi, nka, net = len(pi_grid), len(kappa_grid), len(eta_grid)
    cube_ft = np.empty((npi, nka, net, ft.size))
    cube_fg = np.empty((npi, nka, net, fg.size))
    fgfl = np.full((fg.size, fl.size), -np.inf)
    rho = np.full(len(RHO_EDGES) + 1, -np.inf)
    lnc = np.full((c.size, c.size), -np.inf)
    jobs = [(i, k) for i in range(npi) for k in range(nka)]

    def fold(res):
        nonlocal fgfl, rho, lnc
        i, k, ft, fgm, a, r, cc = res
        cube_ft[i, k], cube_fg[i, k] = ft, fgm
        fgfl = np.logaddexp(fgfl, a)
        rho = np.logaddexp(rho, r)
        lnc = np.logaddexp(lnc, cc)
    if n_workers > 1 and sys.platform.startswith("linux"):
        import multiprocessing as mp
        with mp.get_context("fork").Pool(n_workers) as pool:
            for res in pool.imap_unordered(_node_block_split, jobs,
                                           chunksize=2):
                fold(res)
    else:
        for job in jobs:
            fold(_node_block_split(job))
    log("[post] split grid ln L: %.0f s (%d nodes)"
        % (time.time() - t0, npi * nka * net))
    return cube_ft, cube_fg, fgfl, rho, lnc


def summarize_split(cube_ft, cube_fg, grids, fgfl, rho, lnc, c_grid=None,
                    f_lattice=None):
    fg, fl, _ = split_f_lattice() if f_lattice is None else f_lattice
    c = np.asarray(SPLIT_C if c_grid is None else c_grid, float)
    out = summarize(cube_ft, grids, c, lnc)
    out["f_tot"] = out.pop("fbin")
    prob = np.exp(cube_fg - np.nanmax(cube_fg))
    post = prob.sum(axis=(0, 1, 2))
    q = grid_quantiles(post, fg, (0.16, 0.5, 0.84))
    out["f_gt"] = {"p16": q[0], "p50": q[1], "p84": q[2],
                   "edge_node_mass": float((post[0] + post[-1]) / post.sum())}
    pl = np.exp(fgfl - np.max(fgfl)).sum(axis=0)
    q = grid_quantiles(pl, fl, (0.16, 0.5, 0.84))
    out["f_lt"] = {"p16": q[0], "p50": q[1], "p84": q[2]}
    pr = np.exp(rho - np.max(rho))
    centres = np.concatenate([[RHO_EDGES[0] - 0.01],
                              0.5 * (RHO_EDGES[1:] + RHO_EDGES[:-1]),
                              [RHO_EDGES[-1] + 0.01]])
    q = grid_quantiles(pr, centres, (0.16, 0.5, 0.84))
    out["log10_rho"] = {"p16": q[0], "p50": q[1], "p84": q[2],
                        "underflow_mass": float(pr[0] / pr.sum())}
    return out


def write_outputs_split(out_dir, cube_ft, cube_fg, grids, fgfl, rho, lnc,
                        summary, meta, c_grid=None, f_lattice=None):
    os.makedirs(out_dir, exist_ok=True)
    fg, fl, ft = split_f_lattice() if f_lattice is None else f_lattice
    c = np.asarray(SPLIT_C if c_grid is None else c_grid, float)
    common = dict(cube_schema_version=np.array(5),
                  variants=np.array([TAG_SPLIT]),
                  pi_grid=grids[0], kappa_grid=grids[1], eta_grid=grids[2],
                  fg_grid=fg, fl_grid=fl, ft_grid=ft,
                  lnpost_fg_fl=fgfl, rho_edges_log10=RHO_EDGES,
                  lnpost_rho=rho, c_grid=c, c2_grid=c, lnpost_c1c2=lnc)
    key = "v__%s__gmf_%s_cube" % (TAG_SPLIT, METRIC)
    np.savez(os.path.join(out_dir, "grid_cubes.npz"), fbin_grid=ft,
             **{key: cube_ft}, **common)
    np.savez(os.path.join(out_dir, "grid_cubes_fgt.npz"), fbin_grid=fg,
             **{key: cube_fg}, **common)
    with open(os.path.join(out_dir, "posterior_summary.yaml"), "w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)
    with open(os.path.join(out_dir, "run_posterior.yaml"), "w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)


def parse_grid(spec):
    lo, hi, n = spec.split(",")
    return np.linspace(float(lo), float(hi), int(n))


def parse_c_nodes(spec):
    """``"0,0.02,...,1"`` -> the c1 = c2 node array (None passes through).

    The nodes must increase strictly and span the prior box [0, 1]."""
    if spec is None:
        return None
    c = np.array([float(v) for v in str(spec).split(",")], float)
    if c.size < 2 or np.any(np.diff(c) <= 0) or c[0] != 0.0 or c[-1] != 1.0:
        raise ValueError("c nodes must increase strictly from 0 to 1: %r"
                         % (spec,))
    return c


DEFAULT_GRIDS = {"pi_grid": "-1.55,0.10,23", "kappa_grid": "-1.30,1.80,32",
                 "eta_grid": "-1.25,0.55,46", "f_grid": "0.35,0.95,61"}
# c1, c2 nodes: dense near 0, where mock posteriors (no circular component)
# concentrate; trapezoid weights handle the non-uniform spacing.
DEFAULT_C_GRID = (0.0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0)


def make_grids(preset=None, pi_grid=None, kappa_grid=None, eta_grid=None,
               f_grid=None):
    """Axis grids: explicit lo,hi,n specs, else the preset, else defaults."""
    pr = GRID_PRESETS[preset] if preset else None

    def one(spec, key, name):
        if spec:
            return parse_grid(spec)
        if pr is not None and key in pr:
            return np.asarray(pr[key], float)
        return parse_grid(DEFAULT_GRIDS[name])
    return [one(pi_grid, "pi", "pi_grid"),
            one(kappa_grid, "kappa", "kappa_grid"),
            one(eta_grid, "eta", "eta_grid"),
            parse_grid(f_grid or DEFAULT_GRIDS["f_grid"])]


def run(injset, obs_csv, out_dir, preset=None, f_grid=None, c_n=None,
        q_min=None, p_env=5.0, n_workers=1, use_errors=False,
        inj_cache=None, log=print, pi_grid=None, kappa_grid=None,
        eta_grid=None, base_tensors=None, c_grid=None, c2_zero=False,
        count=True, below_pcirc_ecc="raise", model="mix", split_f=None,
        k1_floor=None, k1_unknown="raise", below_shape=None):
    """``k1_floor`` (km/s) evaluates the amplitude-floor survey (see
    ``InjectionSet``); with ``inj_cache`` the cached set must have been
    built with the same floor. ``k1_unknown`` is passed to
    ``ObservedData``. ``below_shape`` ("k1" or "k1e", split model): the
    below-cut orbits contribute their amplitude (and eccentricity) density
    instead of the plain detection term (``likelihood`` docstring)."""
    if split_f is not None and model != "split":
        raise ValueError("split_f is a split-model option (--model split)")
    if below_shape is not None and model != "split":
        raise ValueError("below_shape is a split-model option (--model "
                         "split)")
    k1_floor = None if k1_floor is None else float(k1_floor)
    if inj_cache is None:
        data, stars, man = load_injection_set(injset)
        inj = InjectionSet(data, stars, man, p_env=p_env, k1_floor=k1_floor)
    else:
        inj = inj_cache
        if getattr(inj, "k1_floor", None) != k1_floor:
            raise ValueError("inj_cache was built with k1_floor=%r but run() "
                             "was given k1_floor=%r"
                             % (getattr(inj, "k1_floor", None), k1_floor))
    table = pd.read_csv(obs_csv)
    cut = float(table["logP_cut"].iloc[0])
    obs = ObservedData(table, inj.stars, cut, inj.p_circ, p_env,
                       inj.bounds["e_max"], below_pcirc_ecc=below_pcirc_ecc,
                       k1_floor=k1_floor, k1_unknown=k1_unknown,
                       below_shape=below_shape)
    grids = make_grids(preset, pi_grid, kappa_grid, eta_grid, f_grid)
    c_grid_given = c_grid
    if c_grid is None:
        c_grid = (np.linspace(0.0, 1.0, int(c_n)) if c_n
                  else np.asarray(DEFAULT_C_GRID, float))
    c_grid = np.asarray(c_grid, float)
    c2_grid = np.array([0.0]) if c2_zero else c_grid
    log("[post] obs %s  cut=%.4f  injections=%d  stars=%d  model=%s"
        % (obs.summary(), cut, inj.logP.size, inj.n_stars, model))
    if model == "split":
        if c2_zero or not count:
            raise ValueError("c2_zero / count=False are mix-model options")
        # the split model shares one node set between c1 and c2: SPLIT_C
        # unless an explicit c_grid was given (e.g. [0.0] to fix both)
        c_split = None if c_n is None and c_grid_given is None else c_grid
        lat = split_f_lattice(split_f)
        fg, fl, ft = lat
        cube_ft, cube_fg, fgfl, rho, lnc = evaluate_grid_split(
            inj, obs, grids[0], grids[1], grids[2], q_min=q_min,
            n_workers=n_workers, use_errors=use_errors, log=log,
            base_tensors=base_tensors, c_grid=c_split, f_lattice=lat)
        summary = summarize_split(cube_ft, cube_fg, grids[:3] + [ft],
                                  fgfl, rho, lnc, c_grid=c_split,
                                  f_lattice=lat)
        meta = {"injset": os.path.abspath(injset) if injset else None,
                "obs_csv": os.path.abspath(obs_csv), "model": "split",
                "tag": TAG_SPLIT, "metric": METRIC, "q_min": q_min,
                "k1_floor": k1_floor, "below_shape": below_shape,
                "below_pcirc_ecc": below_pcirc_ecc,
                "c_grid": (list(map(float, c_split)) if c_split is not None
                           else list(SPLIT_C)),
                "split_f": split_f,
                "split_f_lattice": {
                    "fg": [float(fg[0]), float(fg[-1]), int(fg.size)],
                    "fl": [float(fl[0]), float(fl[-1]), int(fl.size)],
                    "ft": [float(ft[0]), float(ft[-1]), int(ft.size)],
                    "d": float(np.round(fg[1] - fg[0], 10))},
                "n_workers": int(n_workers),
                "p_env": float(p_env), "use_errors": bool(use_errors),
                "pi_grid": pi_grid, "kappa_grid": kappa_grid,
                "eta_grid": eta_grid, "preset": preset,
                "obs_summary": obs.summary(),
                "detect_method": inj.manifest.get("detect_method"),
                "written": _dt.datetime.now().isoformat(timespec="seconds")}
        write_outputs_split(out_dir, cube_ft, cube_fg,
                            grids[:3] + [ft], fgfl, rho, lnc, summary,
                            meta, c_grid=c_split, f_lattice=lat)
        for name in ("pi", "kappa", "eta", "f_tot", "f_gt", "f_lt",
                     "log10_rho", "c1", "c2"):
            v = summary[name]
            log("  %-9s p50 %+.3f  [p16 %+.3f, p84 %+.3f]"
                % (name, v["p50"], v["p16"], v["p84"]))
        return summary
    cube, lnc = evaluate_grid(inj, obs, grids[0], grids[1], grids[2],
                              grids[3], c_grid, q_min=q_min,
                              n_workers=n_workers, use_errors=use_errors,
                              log=log, base_tensors=base_tensors,
                              c2_grid=c2_grid, count=count)
    summary = summarize(cube, grids, c_grid, lnc, c2_grid)
    meta = {"injset": os.path.abspath(injset) if injset else None,
            "obs_csv": os.path.abspath(obs_csv), "preset": preset,
            "f_grid": f_grid, "c_grid": c_grid.tolist(),
            "c2_grid": c2_grid.tolist(), "count_term": bool(count),
            "q_min": q_min, "k1_floor": k1_floor,
            "n_workers": int(n_workers),
            "pi_grid": pi_grid, "kappa_grid": kappa_grid,
            "eta_grid": eta_grid,
            "p_env": float(p_env), "use_errors": bool(use_errors),
            "tag": TAG, "metric": METRIC, "obs_summary": obs.summary(),
            "detect_method": inj.manifest.get("detect_method"),
            "written": _dt.datetime.now().isoformat(timespec="seconds")}
    write_outputs(out_dir, cube, grids, lnc, c_grid, summary, meta,
                  c2_grid=c2_grid)
    for name in AXES + ("c1", "c2"):
        s = summary[name]
        log("  %-6s p50 %+.3f  [p16 %+.3f, p84 %+.3f]%s"
            % (name, s["p50"], s["p16"], s["p84"],
               "  edge %.1e" % s["edge_node_mass"]
               if "edge_node_mass" in s else ""))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--injset", required=True)
    ap.add_argument("--obs", required=True, help="per-star observed CSV")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--preset", default=None,
                    help="GRID_PRESETS entry for the pi/kappa/eta nodes "
                         "(default: the fine true-posterior grids)")
    ap.add_argument("--f-grid", default=None, help="lo,hi,n")
    ap.add_argument("--pi-grid", default=None,
                    help="lo,hi,n (overrides the preset's pi nodes)")
    ap.add_argument("--kappa-grid", default=None, help="lo,hi,n")
    ap.add_argument("--eta-grid", default=None, help="lo,hi,n")
    ap.add_argument("--c-grid", type=int, default=None,
                    help="uniform nodes on [0,1] for c1, c2 (default: "
                         "SPLIT_C for the split model, the 11-node grid "
                         "dense near 0 for mix)")
    ap.add_argument("--c-nodes", default=None, metavar="c0,c1,...",
                    help="explicit c1 = c2 nodes, increasing from 0 to 1 "
                         "(instead of --c-grid)")
    ap.add_argument("--q-min", type=float, default=None,
                    help="restrict the q domain (e.g. 0.1 for qmin01)")
    ap.add_argument("--p-env", type=float, default=5.0)
    ap.add_argument("--use-errors", action="store_true")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--max-per-star", type=int, default=None,
                    help="subsample the injection set per star")
    ap.add_argument("--c2-zero", action="store_true",
                    help="no circular component above P_env (env5 analogue)")
    ap.add_argument("--c-zero", action="store_true",
                    help="fix c1 = c2 = 0 (no circular component at all)")
    ap.add_argument("--subset", default=None,
                    help="k,n: use block k of n disjoint per-star blocks of "
                         "the injection set (MC-error runs)")
    ap.add_argument("--no-count", action="store_true",
                    help="shape-only likelihood (binom_none analogue)")
    ap.add_argument("--model", choices=["mix", "split"], default="mix",
                    help="mix: power law over the full period range; "
                         "split: power law above logP*, free binary "
                         "fraction below it")
    ap.add_argument("--split-f", default=None, metavar="fg_lo,fg_hi,fl_hi,d",
                    help="split-model f lattice: f_gt on fg_lo..fg_hi, f_lt "
                         "on 0..fl_hi, shared step d (default "
                         "0.20,0.90,0.40,0.02)")
    ap.add_argument("--k1-floor", type=float, default=None,
                    help="amplitude-floor survey: simulated binaries count "
                         "as detected only if K1 >= this value (km/s); the "
                         "observed table must list detected orbits below "
                         "it as ND")
    ap.add_argument("--k1-unknown-above-floor", action="store_true",
                    help="with --k1-floor: accept detected stars without a "
                         "K1 and score them as detections above the floor")
    ap.add_argument("--below-shape", choices=["k1", "k1e"], default=None,
                    help="split model: below-cut orbits contribute their "
                         "amplitude density (k1) or amplitude and "
                         "eccentricity densities (k1e) instead of the plain "
                         "detection term; their period density is not used")
    ap.add_argument("--below-pcirc-as-circular", action="store_true",
                    help="treat orbits with a significant e below p_circ as "
                         "circular (the model forces e = 0 there)")
    a = ap.parse_args(argv)
    if a.c_nodes is not None and (a.c_grid is not None or a.c_zero):
        ap.error("--c-nodes excludes --c-grid and --c-zero")
    inj = None
    subset = tuple(int(v) for v in a.subset.split(",")) if a.subset else None
    if a.max_per_star or subset:
        data, stars, man = load_injection_set(a.injset)
        inj = InjectionSet(data, stars, man, p_env=a.p_env,
                           max_per_star=a.max_per_star, subset=subset,
                           k1_floor=a.k1_floor)
    run(a.injset, a.obs, a.out_dir, preset=a.preset, f_grid=a.f_grid,
        c_n=a.c_grid, q_min=a.q_min, p_env=a.p_env, n_workers=a.n_workers,
        use_errors=a.use_errors, pi_grid=a.pi_grid,
        kappa_grid=a.kappa_grid, eta_grid=a.eta_grid, inj_cache=inj,
        c2_zero=a.c2_zero, count=not a.no_count, model=a.model,
        c_grid=(np.array([0.0]) if a.c_zero else parse_c_nodes(a.c_nodes)),
        split_f=a.split_f,
        k1_floor=a.k1_floor,
        k1_unknown=("above_floor" if a.k1_unknown_above_floor else "raise"),
        below_shape=a.below_shape,
        below_pcirc_ecc=("as_circular" if a.below_pcirc_as_circular
                         else "raise"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
