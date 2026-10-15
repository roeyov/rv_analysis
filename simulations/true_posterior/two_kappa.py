"""
simulations.true_posterior.two_kappa — the split model with a mass-ratio
index that differs below and above a fixed period P_q.

    f(q) ~ q^kappa_short   for binaries with P <  P_q,
    f(q) ~ q^kappa_long    for binaries with P >= P_q,

everything else as in the split model of ``run_posterior`` (period power law
above the cut logP*, free binary fraction below it, e law, circular
fractions c1, c2, the same detector and injection set). P_q >= P* is fixed
(``--kappa-split-p``, days). With ``--below-shape`` the orbits below the cut
contribute their amplitude (and eccentricity) density, through kappa_short.

Why it is cheap: every injection lies in one period range, so each per-star
weight sum is the sum of a term that depends on (pi, kappa_short, eta) and
one that depends on (pi, kappa_long, eta). Two tensor sets from
``InjectionSet.alpha_tensors(logp_range=...)`` are enough. Only the count
terms couple the two indices, which makes the node grid
(pi, kappa_short, kappa_long, eta).

Outputs in ``--out-dir``:
  grid_cubes.npz        the standard split-model schema with kappa :=
                        kappa_long (kappa_short summed out), plus
                        ``kappa_short_grid``, ``lnpost_node4`` (ln evidence
                        per (pi, kappa_short, kappa_long, eta) node, the f
                        lattice and c nodes summed out), ``lnpost_ks_kl``
                        (the joint kappa posterior) and ``logp_q``
  grid_cubes_kshort.npz the same with kappa := kappa_short
  grid_cubes_fgt.npz    ln L(pi, kappa_long, eta, f_gt)
  posterior_summary.yaml  pi, kappa_short, kappa_long, dkappa
                        (= kappa_short - kappa_long), eta, f_tot, f_gt,
                        f_lt, log10_rho, c1, c2; and ``tests``:
                        P(dkappa <= 0), and 2 dlnZ between the best node
                        and the best node on the diagonal kappa_short =
                        kappa_long (available when the two grids share
                        nodes)
  run_posterior.yaml    settings
  ACCEPTED / REJECTED   the edge-mass gate (1e-4) on every free axis; an
                        axis with a single node is fixed and exempt

    python -u -m simulations.true_posterior.two_kappa --injset <set> \\
        --obs <obs_catalog.csv> --out-dir <dir> --kappa-split-p 10 \\
        --pi-grid=-2.0,0.5,26 --eta-grid=-0.9,0.6,26 \\
        --kappa-short-grid=-1.20,1.00,45 --kappa-long-grid=-1.80,0.00,37 \\
        --split-f=0.30,0.95,0.30,0.01 [--below-shape k1e \\
        --below-pcirc-as-circular] [--n-workers 34]

A single-node pi and eta grid (lo = hi, n = 1) gives the conditional
posterior of (kappa_short, kappa_long) at that pi and eta: a quick slice.
"""

import argparse
import datetime as _dt
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

from simulations.bias_grid_lib.marginals import grid_quantiles
from simulations.true_posterior.injection_set import load_injection_set
from simulations.true_posterior.likelihood import (
    CAT_DOA, CAT_DOANY, CAT_DOB, CAT_ND, InjectionSet, ObservedData,
    _dx_lnp, _lin, node_split_outputs, shape_precompute)
from simulations.true_posterior.population import powerlaw_cdf
from simulations.true_posterior.run_posterior import (
    METRIC, RHO_EDGES, SPLIT_C, parse_grid, split_f_lattice, summarize,
    summarize_split)

TAG_SPLIT2K = "likmixsplit2k__numerical__lucyT"
EDGE_MASS_MAX = 1e-4


def node_table_split2(T_lo, T_hi, sh_s, sh_l, obs, beta, i, k1, k2, l, fg,
                      fl, c1, c2, F_gt, hi_dx):
    """ln L(f_gt, f_lt, c1, c2) at node (pi_i, kappa_short_k1, kappa_long_k2,
    eta_l): ``likelihood.node_table_split`` with the per-star sums split by
    period range (``T_lo`` on the kappa_short grid for P < P_q, ``T_hi`` on
    the kappa_long grid for P >= P_q) and each orbit's amplitude density
    taken at the kappa of its range (``hi_dx``: DX rows with P >= P_q; the
    below-cut rows of ``below_shape`` always use kappa_short)."""
    fg = np.asarray(fg, float)
    fl = np.asarray(fl, float)
    c1 = np.asarray(c1, float)
    c2 = np.asarray(c2, float)
    F_gt = float(F_gt)

    def both(group):
        return (_lin(T_lo[group], None, i, k1, l, c1, c2)
                + _lin(T_hi[group], None, i, k2, l, c1, c2))
    D, Dg, Ng, Na = both("den"), both("den_cut"), both("num_cut"), \
        both("num_all")
    with np.errstate(divide="ignore", invalid="ignore"):
        a_gt = np.where(Dg > 0, Ng / np.where(Dg > 0, Dg, 1.0), 0.0)
        Dl = D - Dg
        a_lt = np.where(Dl > 0, np.clip(Na - Ng, 0.0, None)
                        / np.where(Dl > 0, Dl, 1.0), 0.0)
    G = fg[:, None, None]
    L = fl[None, :, None]
    cat = obs.cat
    tot = np.zeros((fg.size, fl.size, c1.size, c2.size))
    m_nd, m_any = cat == CAT_ND, cat == CAT_DOANY
    with np.errstate(divide="ignore", invalid="ignore"):
        for i1 in range(c1.size):
            for i2 in range(c2.size):
                ag, al = a_gt[i1, i2], a_lt[i1, i2]
                if m_nd.any():
                    p = (G * ag[m_nd] + L * al[m_nd]
                         + (1.0 - G - L) * beta[m_nd])
                    tot[:, :, i1, i2] += np.log1p(-p).sum(axis=-1)
                if m_any.any():
                    p = (G * ag[m_any] + L * al[m_any]
                         + (1.0 - G - L) * beta[m_any])
                    tot[:, :, i1, i2] += np.log(p).sum(axis=-1)
        m = cat == CAT_DOB
        sb = sh_s.get("below")
        if sb is not None:
            m = m & ~obs.below.used
            tot += (obs.below.dx_pos.size * np.log(fl)[None, :, None, None]
                    + _dx_lnp(sb, obs.below, i, k1, l, c1, c2).sum(
                        axis=0)[None, None])
        if m.any():
            tot += (m.sum() * np.log(fl)[None, :, None, None]
                    + np.log(a_lt[..., m]).sum(axis=-1)[None, None])
        m = cat == CAT_DOA
        if m.any():
            tot += (m.sum() * np.log(fg)[:, None, None, None]
                    + np.log(a_gt[..., m]).sum(axis=-1)[None, None])
        n_dx = obs.dx_pos.size
        if n_dx:
            lnp = np.where(hi_dx[:, None, None],
                           _dx_lnp(sh_l, obs, i, k2, l, c1, c2),
                           _dx_lnp(sh_s, obs, i, k1, l, c1, c2))
            tot += (n_dx * (np.log(fg)[:, None, None, None] - np.log(F_gt))
                    + lnp.sum(axis=0)[None, None])
    simplex = (fg[:, None] + fl[None, :]) <= 1.0 + 1e-12
    return np.where(simplex[:, :, None, None], tot, -np.inf)


def _lse(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis,
                                            keepdims=True)), axis=axis)


_G = {}


def _job(args):
    """All (kappa_long, eta) nodes of one (pi, kappa_short) pair."""
    i, k1 = args
    g = _G
    n_kl, n_eta = g["n_kl"], g["n_eta"]
    ft = np.empty((n_kl, n_eta, g["n_ft"]))
    fgm = np.empty((n_kl, n_eta, g["fg"].size))
    fgfl = np.full((g["fg"].size, g["fl"].size), -np.inf)
    rho = np.full(len(RHO_EDGES) + 1, -np.inf)
    lnc = np.full((g["c"].size, g["c"].size), -np.inf)
    for k2 in range(n_kl):
        for l in range(n_eta):
            tot = node_table_split2(g["T_lo"], g["T_hi"], g["sh_s"],
                                    g["sh_l"], g["obs"], g["beta"], i, k1,
                                    k2, l, g["fg"], g["fl"], g["c"], g["c"],
                                    g["F_gt"][i], g["hi_dx"])
            a, b, c2d, r, cc = node_split_outputs(
                tot, g["fg"], g["fl"], g["c"], g["c"], g["n_ft"],
                g["F_gt"][i], RHO_EDGES)
            ft[k2, l], fgm[k2, l] = a, b
            fgfl = np.logaddexp(fgfl, c2d)
            rho = np.logaddexp(rho, r)
            lnc = np.logaddexp(lnc, cc)
    return i, k1, ft, fgm, fgfl, rho, lnc


def range_tensors(inj, pi_grid, ks_grid, kl_grid, eta_grid, cut, logp_q,
                  q_min=None):
    """(T_lo, T_hi): the per-star weight sums below and at-or-above P_q."""
    groups = ("den", "num_all", "num_cut", "den_cut")
    T_lo = inj.alpha_tensors(pi_grid, ks_grid, eta_grid, cut, q_min=q_min,
                             groups=groups, logp_range=(None, logp_q))
    T_hi = inj.alpha_tensors(pi_grid, kl_grid, eta_grid, cut, q_min=q_min,
                             groups=groups, logp_range=(logp_q, None))
    return T_lo, T_hi


def evaluate(inj, obs, pi_grid, ks_grid, kl_grid, eta_grid, logp_q,
             q_min=None, n_workers=1, use_errors=False, log=print,
             c_grid=None, f_lattice=None, tensors=None):
    """The two-kappa grid. Returns a dict of the accumulated tables.

    ``tensors`` = ``range_tensors(...)`` for the same grids, cut and P_q
    can be passed in when many catalogues share them (mock campaigns)."""
    t0 = time.time()
    if logp_q < obs.cut - 1e-12:
        raise ValueError("P_q must not lie below the period cut")
    fg, fl, ft = split_f_lattice() if f_lattice is None else f_lattice
    if tensors is None:
        tensors = range_tensors(inj, pi_grid, ks_grid, kl_grid, eta_grid,
                                obs.cut, logp_q, q_min=q_min)
    T_lo, T_hi = tensors
    log("[2k] alpha tensors: %.0f s" % (time.time() - t0))
    sh_s = shape_precompute(obs, inj, pi_grid, ks_grid, eta_grid,
                            q_min=q_min, use_errors=use_errors)
    sh_l = shape_precompute(obs, inj, pi_grid, kl_grid, eta_grid,
                            q_min=q_min, use_errors=use_errors)
    hi_dx = obs.dx_logP >= logp_q
    b = inj.bounds
    F_gt = np.array([1.0 - float(powerlaw_cdf(obs.cut, p, b["log_p_min"],
                                              b["log_p_max"]))
                     for p in pi_grid])
    c = np.asarray(SPLIT_C if c_grid is None else c_grid, float)
    npi, nks, nkl, net = (len(pi_grid), len(ks_grid), len(kl_grid),
                          len(eta_grid))
    _G.update(T_lo=T_lo, T_hi=T_hi, sh_s=sh_s, sh_l=sh_l, obs=obs,
              beta=inj.beta, fg=fg, fl=fl, c=c, n_ft=ft.size, F_gt=F_gt,
              n_eta=net, n_kl=nkl, hi_dx=hi_dx)
    node4 = np.full((npi, nks, nkl, net), -np.inf)
    ft_long = np.full((npi, nkl, net, ft.size), -np.inf)
    ft_short = np.full((npi, nks, net, ft.size), -np.inf)
    fg_long = np.full((npi, nkl, net, fg.size), -np.inf)
    acc = {"fgfl": np.full((fg.size, fl.size), -np.inf),
           "rho": np.full(len(RHO_EDGES) + 1, -np.inf),
           "lnc": np.full((c.size, c.size), -np.inf)}
    jobs = [(i, k1) for i in range(npi) for k1 in range(nks)]

    def fold(res):
        i, k1, ftab, fgm, a, r, cc = res
        node4[i, k1] = _lse(ftab, 2)
        ft_long[i] = np.logaddexp(ft_long[i], ftab)
        ft_short[i, k1] = _lse(ftab, 0)
        fg_long[i] = np.logaddexp(fg_long[i], fgm)
        acc["fgfl"] = np.logaddexp(acc["fgfl"], a)
        acc["rho"] = np.logaddexp(acc["rho"], r)
        acc["lnc"] = np.logaddexp(acc["lnc"], cc)
    if n_workers > 1 and sys.platform.startswith("linux"):
        import multiprocessing as mp
        with mp.get_context("fork").Pool(n_workers) as pool:
            for res in pool.imap_unordered(_job, jobs, chunksize=1):
                fold(res)
    else:
        for job in jobs:
            fold(_job(job))
    log("[2k] grid ln L: %.0f s (%d nodes; DX orbits below / at-or-above "
        "P_q: %d / %d)" % (time.time() - t0, npi * nks * nkl * net,
                           int((~hi_dx).sum()), int(hi_dx.sum())))
    return {"node4": node4, "ft_long": ft_long, "ft_short": ft_short,
            "fg_long": fg_long, "c": c, "lattice": (fg, fl, ft), **acc}


def _axis_summary(post, grid):
    p16, p50, p84 = grid_quantiles(post, grid, (0.16, 0.5, 0.84))
    p025, p975 = grid_quantiles(post, grid, (0.025, 0.975))
    return {"p2_5": p025, "p16": p16, "p50": p50, "p84": p84, "p97_5": p975,
            "mode": float(grid[int(np.argmax(post))]),
            "edge_node_mass": float((post[0] + post[-1]) / post.sum())}


def summarize2(res, pi_grid, ks_grid, kl_grid, eta_grid):
    fg, fl, ft = res["lattice"]
    c = res["c"]
    out = summarize_split(res["ft_long"], res["fg_long"],
                          [pi_grid, kl_grid, eta_grid, ft], res["fgfl"],
                          res["rho"], res["lnc"], c_grid=c,
                          f_lattice=res["lattice"])
    out["kappa_long"] = out.pop("kappa")
    short = summarize(res["ft_short"], [pi_grid, ks_grid, eta_grid, ft], c,
                      res["lnc"])
    out["kappa_short"] = short["kappa"]
    # joint kappa posterior and the difference
    lnJ = _lse(_lse(res["node4"], 3), 0)                    # (n_ks, n_kl)
    J = np.exp(lnJ - np.max(lnJ))
    J /= J.sum()
    dk = np.round(np.subtract.outer(ks_grid, kl_grid), 9)
    vals, inv = np.unique(dk.ravel(), return_inverse=True)
    pd_ = np.bincount(inv, weights=J.ravel(), minlength=vals.size)
    cdf = np.cumsum(pd_)

    def q(u):
        return float(vals[min(int(np.searchsorted(cdf, u)), vals.size - 1)])
    out["dkappa"] = {"p2_5": q(0.025), "p16": q(0.16), "p50": q(0.5),
                     "p84": q(0.84), "p97_5": q(0.975),
                     "mean": float(np.sum(vals * pd_))}
    tests = {"p_dkappa_le0": float(pd_[vals <= 0].sum()),
             "p_dkappa_lt0": float(pd_[vals < 0].sum())}
    # best node on the diagonal kappa_short = kappa_long (shared nodes)
    ia, ib = np.nonzero(np.abs(dk) < 1e-9)
    if ia.size:
        tests["two_dlnZ_best_vs_diagonal"] = float(
            2.0 * (np.max(lnJ) - np.max(lnJ[ia, ib])))
        tests["n_diagonal_nodes"] = int(ia.size)
        k = int(np.argmax(lnJ[ia, ib]))
        tests["kappa_best_on_diagonal"] = float(ks_grid[ia[k]])
    a, bb = np.unravel_index(int(np.argmax(lnJ)), lnJ.shape)
    tests["kappa_short_kappa_long_at_best_node"] = [float(ks_grid[a]),
                                                   float(kl_grid[bb])]
    out["tests"] = tests
    return out, lnJ, (vals, pd_)


def gate(summary, grids):
    """Edge-mass gate on every free axis (a single-node axis is fixed)."""
    bad, fixed = [], []
    for name, grid in grids.items():
        if len(grid) == 1:
            fixed.append(name)
            continue
        m = summary[name].get("edge_node_mass")
        if m is not None and m > EDGE_MASS_MAX:
            bad.append("%s: edge nodes hold %.2e" % (name, m))
    return bad, fixed


def run(injset, obs_csv, out_dir, pi_grid, ks_grid, kl_grid, eta_grid,
        p_q_days, p_env=5.0, q_min=None, n_workers=1, use_errors=False,
        split_f=None, below_shape=None, below_pcirc_ecc="raise",
        k1_floor=None, k1_unknown="raise", inj_cache=None, log=print):
    k1_floor = None if k1_floor is None else float(k1_floor)
    if inj_cache is None:
        data, stars, man = load_injection_set(injset)
        inj = InjectionSet(data, stars, man, p_env=p_env, k1_floor=k1_floor)
    else:
        inj = inj_cache
    table = pd.read_csv(obs_csv)
    cut = float(table["logP_cut"].iloc[0])
    logp_q = cut if p_q_days is None else float(np.log10(float(p_q_days)))
    obs = ObservedData(table, inj.stars, cut, inj.p_circ, p_env,
                       inj.bounds["e_max"], below_pcirc_ecc=below_pcirc_ecc,
                       k1_floor=k1_floor, k1_unknown=k1_unknown,
                       below_shape=below_shape)
    lat = split_f_lattice(split_f)
    fg, fl, ft = lat
    log("[2k] obs %s  cut=%.4f  logP_q=%.4f (%.3f d)  injections=%d"
        % (obs.summary(), cut, logp_q, 10 ** logp_q, inj.logP.size))
    res = evaluate(inj, obs, pi_grid, ks_grid, kl_grid, eta_grid, logp_q,
                   q_min=q_min, n_workers=n_workers, use_errors=use_errors,
                   log=log, f_lattice=lat)
    summary, lnJ, (dvals, dpost) = summarize2(res, pi_grid, ks_grid, kl_grid,
                                              eta_grid)
    bad, fixed = gate(summary, {"pi": pi_grid, "kappa_short": ks_grid,
                                "kappa_long": kl_grid, "eta": eta_grid,
                                "f_tot": ft, "f_gt": fg})
    os.makedirs(out_dir, exist_ok=True)
    common = dict(cube_schema_version=np.array(5),
                  variants=np.array([TAG_SPLIT2K]), pi_grid=pi_grid,
                  eta_grid=eta_grid, fg_grid=fg, fl_grid=fl, ft_grid=ft,
                  lnpost_fg_fl=res["fgfl"], rho_edges_log10=RHO_EDGES,
                  lnpost_rho=res["rho"], c_grid=res["c"], c2_grid=res["c"],
                  lnpost_c1c2=res["lnc"], kappa_short_grid=ks_grid,
                  kappa_long_grid=kl_grid, logp_q=np.array(logp_q),
                  lnpost_ks_kl=lnJ, dkappa_grid=dvals, dkappa_post=dpost)
    key = "v__%s__gmf_%s_cube" % (TAG_SPLIT2K, METRIC)
    np.savez(os.path.join(out_dir, "grid_cubes.npz"), kappa_grid=kl_grid,
             fbin_grid=ft, lnpost_node4=res["node4"],
             **{key: res["ft_long"]}, **common)
    np.savez(os.path.join(out_dir, "grid_cubes_kshort.npz"),
             kappa_grid=ks_grid, fbin_grid=ft, **{key: res["ft_short"]},
             **common)
    np.savez(os.path.join(out_dir, "grid_cubes_fgt.npz"), kappa_grid=kl_grid,
             fbin_grid=fg, **{key: res["fg_long"]}, **common)
    meta = {"injset": os.path.abspath(injset) if injset else None,
            "obs_csv": os.path.abspath(obs_csv), "model": "split2k",
            "tag": TAG_SPLIT2K, "metric": METRIC, "q_min": q_min,
            "k1_floor": k1_floor, "below_shape": below_shape,
            "below_pcirc_ecc": below_pcirc_ecc,
            "kappa_split_p_days": float(10 ** logp_q), "logp_q": logp_q,
            "split_f": split_f, "n_workers": int(n_workers),
            "p_env": float(p_env), "use_errors": bool(use_errors),
            "grids": {"pi": [float(pi_grid[0]), float(pi_grid[-1]),
                             int(len(pi_grid))],
                      "kappa_short": [float(ks_grid[0]), float(ks_grid[-1]),
                                      int(len(ks_grid))],
                      "kappa_long": [float(kl_grid[0]), float(kl_grid[-1]),
                                     int(len(kl_grid))],
                      "eta": [float(eta_grid[0]), float(eta_grid[-1]),
                              int(len(eta_grid))]},
            "fixed_axes": fixed, "gate": "REJECTED" if bad else "ACCEPTED",
            "gate_reasons": bad, "obs_summary": obs.summary(),
            "detect_method": inj.manifest.get("detect_method"),
            "written": _dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out_dir, "posterior_summary.yaml"), "w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)
    with open(os.path.join(out_dir, "run_posterior.yaml"), "w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    open(os.path.join(out_dir, "REJECTED" if bad else "ACCEPTED"),
         "w").write("\n".join(bad) + "\n")
    for name in ("pi", "kappa_short", "kappa_long", "dkappa", "eta", "f_tot",
                 "f_gt", "f_lt", "log10_rho", "c1", "c2"):
        v = summary[name]
        log("  %-12s p50 %+.3f  [p16 %+.3f, p84 %+.3f]%s"
            % (name, v["p50"], v["p16"], v["p84"],
               "  edge %.1e" % v["edge_node_mass"]
               if "edge_node_mass" in v else ""))
    log("  tests: %s" % summary["tests"])
    log("  gate: %s%s%s" % ("REJECTED " if bad else "ACCEPTED", "; ".join(bad),
                            ("  (fixed axes: %s)" % ", ".join(fixed))
                            if fixed else ""))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--injset", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kappa-split-p", type=float, default=None,
                    help="P_q in days (default: the catalogue's period cut)")
    ap.add_argument("--pi-grid", required=True, help="lo,hi,n")
    ap.add_argument("--eta-grid", required=True, help="lo,hi,n")
    ap.add_argument("--kappa-short-grid", required=True, help="lo,hi,n")
    ap.add_argument("--kappa-long-grid", required=True, help="lo,hi,n")
    ap.add_argument("--split-f", default=None, metavar="fg_lo,fg_hi,fl_hi,d")
    ap.add_argument("--q-min", type=float, default=None)
    ap.add_argument("--p-env", type=float, default=5.0)
    ap.add_argument("--use-errors", action="store_true")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--below-shape", choices=["k1", "k1e"], default=None)
    ap.add_argument("--below-pcirc-as-circular", action="store_true")
    ap.add_argument("--k1-floor", type=float, default=None)
    ap.add_argument("--k1-unknown-above-floor", action="store_true")
    ap.add_argument("--subset", default=None,
                    help="k,n: use block k of n disjoint per-star blocks of "
                         "the injection set (Monte-Carlo error runs)")
    a = ap.parse_args(argv)
    inj = None
    if a.subset:
        data, stars, man = load_injection_set(a.injset)
        inj = InjectionSet(data, stars, man, p_env=a.p_env,
                           subset=tuple(int(v) for v in a.subset.split(",")),
                           k1_floor=a.k1_floor)
        del data
    run(a.injset, a.obs, a.out_dir, parse_grid(a.pi_grid),
        parse_grid(a.kappa_short_grid), parse_grid(a.kappa_long_grid),
        parse_grid(a.eta_grid), a.kappa_split_p, p_env=a.p_env,
        q_min=a.q_min, n_workers=a.n_workers, use_errors=a.use_errors,
        split_f=a.split_f, below_shape=a.below_shape,
        below_pcirc_ecc=("as_circular" if a.below_pcirc_as_circular
                         else "raise"),
        k1_floor=a.k1_floor,
        k1_unknown=("above_floor" if a.k1_unknown_above_floor else "raise"),
        inj_cache=inj)
    return 0


if __name__ == "__main__":
    sys.exit(main())
