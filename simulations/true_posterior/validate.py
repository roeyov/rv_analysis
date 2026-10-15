"""
simulations.true_posterior.validate — bridge checks between the labelled
injection set and an existing bias-grid run.

``bridge`` compares the injection set, reweighted to a grid node's
(pi, kappa, eta), with that run's own det shards at the same node. The 16
f_bin nodes of a (pi, kappa, eta) are pooled: the binary draws and the
detection do not depend on f_bin. At each sampled node it compares

  * the detection fraction: reweighted star-mean of a_s (c1 = c2 = 0,
    which is the grid's model) against sum n_det / sum n_bin from the
    shards, with a z-score;
  * the detected-sample marginals of logP, e, K1 and q: a weighted KS
    distance between the reweighted detected injections and the shards'
    detected arrays. Both hold true injected values. The p-value uses the
    Kish effective size of the weights.

A pass means the |z| values scatter like N(0, 1) and the KS p-values show
no pile-up near 0. Only then do the new injection set and its reweighting
reproduce the simulation behind the adopted CvM result.

    python -m simulations.true_posterior.validate bridge \
        --injset <dir> --run-dir <bias_grid run dir> [--n-nodes 60] \
        [--q-min 0.1] --out <csv>

``ppc`` runs posterior-predictive checks of a finished posterior (see
``ppc`` below). The model settings of the run (``--q-min``, ``--k1-floor``)
must be repeated on the command line; a mismatch with the run's
``run_posterior.yaml`` is an error.

``reanalyze`` re-reads a finished coverage campaign (no likelihood
evaluation): every mock's stored cube is re-summarized with both quantile
estimators (cell-centred and piecewise-linear), and the per-truth signed
z, half-width / grid-step and end-node mass are tabulated. This separates
a quantile-estimator (phase) error from a genuine width or offset error.

    python -m simulations.true_posterior.validate reanalyze \
        --out-dir <coverage out dir> --campaign-dir <mock campaign dir> \
        --model split --out <dir>
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.special import kolmogorov

from simulations.true_posterior.injection_set import load_injection_set
from simulations.true_posterior.likelihood import (
    InjectionSet, ObservedData, node_table, node_table_split,
    shape_precompute, trapz_logweights)
from simulations.true_posterior.population import powerlaw_cdf


def _wecdf_ks(x_w, w, x_s):
    """KS distance between a weighted sample (x_w, w) and a plain sample."""
    o = np.argsort(x_w)
    xw, ww = x_w[o], w[o]
    cw = np.cumsum(ww) / ww.sum()
    xs = np.sort(x_s)
    grid = np.concatenate([xw, xs])
    Fw = np.interp(grid, xw, cw, left=0.0, right=1.0)
    # right-continuous ECDF of the plain sample
    Fs = np.searchsorted(xs, grid, side="right") / xs.size
    return float(np.max(np.abs(Fw - Fs)))


def _pick_nodes(npi, nka, net, n, seed):
    corners = [(i, k, l) for i in (0, npi - 1) for k in (0, nka - 1)
               for l in (0, net - 1)]
    centre = [(npi // 2, nka // 2, net // 2)]
    rng = np.random.default_rng(seed)
    extra = set()
    while len(extra) < max(0, n - len(corners) - 1):
        extra.add((int(rng.integers(npi)), int(rng.integers(nka)),
                   int(rng.integers(net))))
    return corners + centre + sorted(extra - set(corners) - set(centre))


def bridge(injset, run_dir, n_nodes=60, seed=0, q_min=None, log=print):
    data, stars, man = load_injection_set(injset)
    S = InjectionSet(data, stars, man)
    z = np.load(os.path.join(run_dir, "grid_cubes.npz"), allow_pickle=True)
    grids = [np.asarray(z[k], float) for k in ("pi_grid", "kappa_grid",
                                               "eta_grid")]
    idx = np.load(os.path.join(run_dir, "det_index.npz"))["step_to_ijkl"]
    shard_dir = os.path.join(run_dir, "det_shards")
    nodes = _pick_nodes(len(grids[0]), len(grids[1]), len(grids[2]),
                        n_nodes, seed)
    rows = []
    for (i, k, l) in nodes:
        steps = idx[(idx[:, 1] == i) & (idx[:, 2] == k) & (idx[:, 3] == l),
                    0]
        n_det = n_bin = 0
        det_arr = {"logP": [], "e": [], "K1": [], "q": []}
        for st in steps:
            with np.load(os.path.join(shard_dir, "step_%06d.npz" % st)) as sh:
                nd = sh["logP"].size
                n_det += nd
                n_bin += nd + sh["logP_nondet"].size
                for key in det_arr:
                    det_arr[key].append(sh[key])
        det_arr = {kk: np.concatenate(v) for kk, v in det_arr.items()}
        a_ext = n_det / n_bin
        se_ext = np.sqrt(a_ext * (1 - a_ext) / n_bin)

        pi, ka, et = grids[0][i], grids[1][k], grids[2][l]
        a_s = S.alpha(pi, ka, et, 0.0, 0.0, q_min=q_min)
        se_s = S.alpha_se(pi, ka, et, 0.0, 0.0, q_min=q_min)
        a_rw = float(a_s.mean())
        se_rw = float(np.sqrt(np.sum(se_s ** 2)) / a_s.size)
        zval = (a_rw - a_ext) / np.hypot(se_rw, se_ext)

        # weighted detected sample, each star normalized to equal injection
        w = S.weights(pi, ka, et, 0.0, 0.0, q_min=q_min)
        den = S._star_sum(w) * S.N
        w_norm = w / np.repeat(den, (S.seg_end - S.seg).astype(int))
        wd = w_norm * S.det
        keep = wd > 0
        n_eff = wd[keep].sum() ** 2 / np.sum(wd[keep] ** 2)
        rec = {"i": i, "k": k, "l": l, "pi": pi, "kappa": ka, "eta": et,
               "n_bin_ext": n_bin, "a_ext": a_ext, "a_rw": a_rw,
               "diff": a_rw - a_ext, "z": zval, "n_eff_det": n_eff}
        vals = {"logP": S.logP, "e": np.where(S.short, 0.0, S.e),
                "K1": S.K1.astype(float), "q": S.q}
        for key in ("logP", "e", "K1", "q"):
            dks = _wecdf_ks(vals[key][keep], wd[keep], det_arr[key])
            ne = n_eff * det_arr[key].size / (n_eff + det_arr[key].size)
            rec["ks_" + key] = dks
            rec["p_" + key] = float(kolmogorov(dks * np.sqrt(ne)))
        rows.append(rec)
        log("node (%2d,%2d,%2d) pi=%+.2f ka=%+.2f eta=%+.2f  a_ext=%.4f "
            "a_rw=%.4f z=%+.2f  KS p: logP %.2f e %.2f K1 %.2f q %.2f"
            % (i, k, l, pi, ka, et, a_ext, a_rw, zval, rec["p_logP"],
               rec["p_e"], rec["p_K1"], rec["p_q"]))
    df = pd.DataFrame(rows)
    zs = df["z"].to_numpy()
    summ = {"n_nodes": int(len(df)), "mean_z": float(zs.mean()),
            "sd_z": float(zs.std(ddof=1)), "max_abs_z": float(np.abs(zs).max()),
            "frac_abs_z_gt3": float(np.mean(np.abs(zs) > 3))}
    for key in ("logP", "e", "K1", "q"):
        p = df["p_" + key].to_numpy()
        summ["min_p_" + key] = float(p.min())
        summ["frac_p_lt_0.01_" + key] = float(np.mean(p < 0.01))
    return df, summ


# ---------------------------------------------------------------------------
# Posterior-predictive checks
# ---------------------------------------------------------------------------

def _ks_to_weighted(x_obs, x_pred, w_pred):
    """KS distance between a sample and a weighted predictive sample."""
    if len(x_obs) == 0 or w_pred.sum() <= 0:
        return np.nan
    return _wecdf_ks(x_pred, w_pred, np.asarray(x_obs, float))


PS_U_THRESHOLDS = (1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
PS_K1_RANGES = ((0.0, 10.0), (10.0, 25.0), (25.0, 50.0), (50.0, 100.0),
                (100.0, np.inf))


def _period_groups(logp, cut, p_groups):
    """Masks of the period groups used by the period-split statistics."""
    g = {"below_cut": logp < cut, "above_cut": logp >= cut}
    for pd_ in p_groups:
        lq = float(np.log10(pd_))
        tag = "%gd" % pd_
        g["lt_" + tag] = logp < lq
        g["ge_" + tag] = logp >= lq
        if lq > cut:
            g["cut_to_" + tag] = (logp >= cut) & (logp < lq)
    return g


def _period_split_stats(logp, k1, cut, p_groups, thresholds=PS_U_THRESHOLDS,
                        k1_ranges=PS_K1_RANGES):
    """Period-split statistics of one catalogue of detected orbits.

    u0 = log10 K1 + logP / 3 (no e term, so Lucy-Sweeney limits play no
    role). Returns a flat dict: ``n|<group>``, ``u|<group>|<t>`` (count with
    u0 < t), ``k|<group>|<lo>_<hi>`` (count with lo <= K1 < hi),
    ``spearman`` (rank correlation of u0 with logP over all orbits) and
    ``du|<P>d`` (mean u0 below P minus mean u0 at or above P)."""
    from scipy.stats import spearmanr
    logp = np.asarray(logp, float)
    k1 = np.asarray(k1, float)
    u0 = np.log10(np.maximum(k1, 1e-12)) + logp / 3.0
    out = {}
    groups = _period_groups(logp, cut, p_groups)
    for name, m in groups.items():
        out["n|" + name] = float(m.sum())
        for t in thresholds:
            out["u|%s|%.1f" % (name, t)] = float((u0[m] < t).sum())
        for lo, hi in k1_ranges:
            key = "%g_%s" % (lo, "inf" if not np.isfinite(hi) else "%g" % hi)
            out["k|%s|%s" % (name, key)] = float(
                ((k1[m] >= lo) & (k1[m] < hi)).sum())
    out["spearman"] = (float(spearmanr(u0, logp)[0]) if u0.size > 2
                       else np.nan)
    for pd_ in p_groups:
        a, b = groups["lt_%gd" % pd_], groups["ge_%gd" % pd_]
        out["du|%gd" % pd_] = (float(u0[a].mean() - u0[b].mean())
                               if a.any() and b.any() else np.nan)
    return out


def _period_split_summary(obs, reps):
    """Observed value, replicate summary and predictive p-values of every
    period-split statistic. Counts are compared as fractions of the
    group's size (the group sizes differ between replicates)."""
    def one(o, x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if not x.size or not np.isfinite(o):
            return {"obs": float(o) if np.isfinite(o) else None}
        return {"obs": float(o), "rep_median": float(np.median(x)),
                "rep_16_84": [float(np.percentile(x, 16)),
                              float(np.percentile(x, 84))],
                "p_le": float(np.mean(x <= o)), "p_ge": float(np.mean(x >= o)),
                "p": float(min(1.0, 2 * min(np.mean(x >= o),
                                            np.mean(x <= o)))),
                "n_rep": int(x.size)}
    res = {}
    for key, o in obs.items():
        kind = key.split("|")[0]
        x = np.array([r[key] for r in reps], float)
        if kind in ("u", "k"):
            grp = key.split("|")[1]
            n_o = obs["n|" + grp]
            n_r = np.array([r["n|" + grp] for r in reps], float)
            with np.errstate(invalid="ignore", divide="ignore"):
                fr = np.where(n_r > 0, x / n_r, np.nan)
            rec = one(o / n_o if n_o > 0 else np.nan, fr)
            rec.update(obs_count=int(o), obs_n=int(n_o),
                       expected_count=(float(n_o * np.nanmean(fr))
                                       if np.isfinite(fr).any() else None))
            res[key] = rec
        else:
            res[key] = one(o, x)
    return res


def _check_run_settings(post_dir, **given):
    """Refuse a predictive check whose model settings differ from the run.

    ``given`` maps a ``run_posterior.yaml`` key (q_min, k1_floor, p_env,
    below_shape) to the value the check was asked to use. A key absent from
    the yaml counts as None (runs written before the key existed).
    """
    path = os.path.join(post_dir, "run_posterior.yaml")
    if not os.path.exists(path):
        warnings.warn("no run_posterior.yaml in %s: the model settings of "
                      "the predictive check cannot be compared with the run"
                      % post_dir, stacklevel=2)
        return
    import yaml
    with open(path) as fh:
        meta = yaml.safe_load(fh) or {}

    def same(a, b):
        if a is None or b is None:
            return a is None and b is None
        if isinstance(a, str) or isinstance(b, str):
            return a == b
        return abs(float(a) - float(b)) <= 1e-12

    bad = ["%s: run %r, check %r" % (k, meta.get(k), v)
           for k, v in given.items() if not same(meta.get(k), v)]
    if bad:
        raise ValueError("predictive check settings differ from the run in "
                         "%s (%s); pass the run's values"
                         % (post_dir, "; ".join(bad)))
    if meta.get("use_errors"):
        warnings.warn("the run used measurement errors (use_errors); the "
                      "predictive check evaluates the model without them",
                      stacklevel=2)


def ppc(injset, obs_csv, post_dir, n_draws=200, seed=1, p_env=5.0,
        e_low=0.15, k1_low=25.0, log=print, below_pcirc_ecc="raise",
        q_min=None, k1_floor=None, k1_unknown="raise", below_shape=None,
        p_groups=(5.0, 10.0, 20.0), reps_per_draw=1, save_reps=None):
    """Posterior-predictive checks of the true-posterior model.

    Draws (pi, kappa, eta, f) from the grid posterior in ``post_dir`` and
    (c1, c2) from their conditional at that node. For each draw it
    simulates a replicate catalogue: per star, a binary with probability f
    whose orbit is resampled from the star's injections with weight
    w_j(theta, c), detected if that injection was detected; otherwise a
    single, falsely detected with rate beta_s. It then compares the data
    with the replicates:
      N_D         number of detected stars (all periods, incl. FPs)
      frac_below  fraction of detected orbits below logP* (obs: orbits of
                  DX + below-cut DO)
      frac_low_e  fraction of DX with LS-circular or e < e_low (a proxy:
                  replicates use true e, the data use LS flags)
      ks_logP, ks_K1, ks_e_hi   KS discrepancy of DX logP, K1 and
                  e >= e_low against the draw's predictive distribution,
                  data vs replicate (p = P(D_rep >= D_obs)).
      frac_lowK1  fraction of DX with K1 < k1_low km/s (above the cut: the
                  sample of the K1 shape term)
      frac_lowK1_all  the same over all detected orbits (DX plus the
                  below-cut DO rows), the sample of the K1 histogram panel
      low_e_long  the low-e fraction restricted to P >= p_env (the period
                  bin of c2), with the same proxy as frac_low_e. Besides
                  the replicate comparison it gives the model's expected
                  fraction per posterior draw (no replicate noise) and the
                  predictive distribution of the number of low-e systems
                  among the observed long-period orbits (a binomial with
                  the draw's fraction, averaged over the draws).
    Count statistics get two-sided predictive p-values.

    ``q_min``, ``k1_floor`` and ``below_shape`` are the run's model
    settings (q domain, amplitude-floor survey, below-cut orbit terms); they
    must equal the values recorded in ``post_dir/run_posterior.yaml``. The
    (f, c) draws at each node use the run's likelihood, so ``below_shape``
    changes them; the statistics themselves do not depend on it.

    Period-split statistics (``period_split`` in the result): for the period
    groups below / above the cut and below / at-or-above each of
    ``p_groups`` (days), the number of detected orbits with
    u0 = log10 K1 + logP/3 below fixed thresholds and with K1 in fixed
    ranges, the rank correlation of u0 with logP, and the mean-u0
    difference across each split; observed (DX plus below-cut DO rows with a
    K1) against the replicates. ``reps_per_draw`` > 1 adds replicates per
    posterior draw for these statistics only, from a separate random
    stream, so every other statistic is unchanged. ``save_reps`` (npz path)
    keeps the detected binaries of all replicates (replicate, star, logP,
    K1, e, q), the false positives and the parameters of each draw.
    """
    _check_run_settings(post_dir, q_min=q_min, k1_floor=k1_floor,
                        p_env=p_env, below_shape=below_shape)
    data, stars, man = load_injection_set(injset)
    S = InjectionSet(data, stars, man, p_env=p_env, k1_floor=k1_floor)
    table = pd.read_csv(obs_csv)
    cut = float(table["logP_cut"].iloc[0])
    O = ObservedData(table, S.stars, cut, S.p_circ, p_env,
                     S.bounds["e_max"], below_pcirc_ecc=below_pcirc_ecc,
                     k1_floor=k1_floor, k1_unknown=k1_unknown,
                     below_shape=below_shape)
    z = np.load(os.path.join(post_dir, "grid_cubes.npz"), allow_pickle=True)
    key = [k for k in z.files if k.endswith("__gmf_lik_cube")][0]
    split = "likmixsplit" in key
    cube = np.asarray(z[key], float)
    grids = [np.asarray(z[k], float) for k in ("pi_grid", "kappa_grid",
                                               "eta_grid", "fbin_grid")]
    c_grid = np.asarray(z["c_grid"], float)
    T = S.alpha_tensors(grids[0], grids[1], grids[2], cut, q_min=q_min,
                        groups=("den", "num_all", "num_cut", "den_cut"))
    if split:
        fg_grid = np.asarray(z["fg_grid"], float)
        fl_grid = np.asarray(z["fl_grid"], float)
        b = S.bounds
        F_gt = np.array([1.0 - float(powerlaw_cdf(cut, p_, b["log_p_min"],
                                                  b["log_p_max"]))
                         for p_ in grids[0]])
    sh = shape_precompute(O, S, grids[0], grids[1], grids[2], q_min=q_min)
    rng = np.random.default_rng(seed)
    p = np.exp(cube - cube.max()).ravel()
    draws = rng.choice(p.size, size=n_draws, p=p / p.sum())
    lw = trapz_logweights(c_grid)

    # observed statistics
    dx = O.table[O.table["outcome"] == "DX"]
    below = int(((O.table["outcome"] == "DO")
                 & (O.table["reason"] == "below_cut")).sum())
    obs_stats = {
        "N_D": int(O.is_D.sum()),
        "frac_below": below / (below + len(dx)),
        "frac_low_e": float(np.mean(dx["e_is_upper_limit"].to_numpy(bool)
                                    | (dx["e"].to_numpy(float) < e_low))),
    }
    e_hi_obs = dx.loc[~dx["e_is_upper_limit"].astype(bool)
                      & (dx["e"] >= e_low), "e"].to_numpy(float)
    logp_obs = dx["logP"].to_numpy(float)
    k1_obs = dx["K1"].to_numpy(float)
    k1_below_obs = O.table.loc[(O.table["outcome"] == "DO")
                               & (O.table["reason"] == "below_cut"),
                               "K1"].to_numpy(float)
    k1_all_obs = np.concatenate([k1_obs, k1_below_obs[np.isfinite(k1_below_obs)]])
    obs_stats["frac_lowK1"] = float(np.mean(k1_obs < k1_low))
    obs_stats["frac_lowK1_all"] = float(np.mean(k1_all_obs < k1_low))
    # long-period subset (P >= p_env), same low-e proxy as frac_low_e
    logp_env = float(np.log10(p_env))
    long_obs = logp_obs >= logp_env
    lowe_obs = (dx["e_is_upper_limit"].to_numpy(bool)
                | (dx["e"].to_numpy(float) < e_low))
    n_long_obs = int(long_obs.sum())
    k_long_obs = int((lowe_obs & long_obs).sum())
    obs_stats["frac_low_e_long"] = (k_long_obs / n_long_obs if n_long_obs
                                    else float("nan"))

    seg_n = (S.seg_end - S.seg).astype(int)
    e_eff = np.where(S.short | S.circ_comp, 0.0, S.e)
    # period-split statistics: observed orbits on both sides of the cut
    orb = O.table[((O.table["outcome"] == "DX")
                   | ((O.table["outcome"] == "DO")
                      & (O.table["reason"] == "below_cut")))
                  & np.isfinite(O.table["K1"].astype(float))
                  & np.isfinite(O.table["logP"].astype(float))]
    ps_obs = _period_split_stats(orb["logP"].to_numpy(float),
                                 orb["K1"].to_numpy(float), cut, p_groups)
    ps_rep = []
    rng2 = np.random.default_rng([int(seed), 7919])     # extra replicates
    sv = {k: [] for k in ("rep", "star", "inj", "fp_rep", "fp_star")}
    sv_draw = []
    n_rep_total = 0

    def keep_rep(det_idx, fp_stars):
        nonlocal n_rep_total
        det_idx = np.asarray(det_idx, dtype=np.int64)
        ps_rep.append(_period_split_stats(S.logP[det_idx], S.K1[det_idx],
                                          cut, p_groups))
        if save_reps:
            sv["rep"].append(np.full(det_idx.size, n_rep_total, np.int32))
            sv["star"].append(S.star[det_idx].astype(np.int16))
            sv["inj"].append(det_idx)
            sv["fp_rep"].append(np.full(len(fp_stars), n_rep_total, np.int32))
            sv["fp_star"].append(np.asarray(fp_stars, dtype=np.int16))
        n_rep_total += 1
    rep = {k: [] for k in ("N_D", "frac_below", "frac_low_e", "D_logP_obs",
                           "D_logP_rep", "D_K1_obs", "D_K1_rep",
                           "D_ehi_obs", "D_ehi_rep", "frac_lowK1",
                           "frac_lowK1_all", "frac_low_e_long",
                           "pred_low_e_long")}
    is_long = S.logP >= logp_env
    for d in draws:
        i, k, l, fi = np.unravel_index(d, cube.shape)
        if split:
            # draw (f_gt, f_lt, c1, c2) at this node, conditional on
            # f_gt + f_lt = f_tot (the drawn cube cell)
            tab = node_table_split(T, sh, O, S.beta, i, k, l, fg_grid,
                                   fl_grid, c_grid, c_grid, F_gt[i])
            gi, li = np.meshgrid(np.arange(fg_grid.size),
                                 np.arange(fl_grid.size), indexing="ij")
            on = (gi + li) == fi
            tw = tab + lw[None, None, :, None] + lw[None, None, None, :]
            tw = np.where(on[:, :, None, None], tw, -np.inf)
            pc = np.nan_to_num(np.exp(tw - np.max(tw)))
            ci = rng.choice(pc.size, p=(pc / pc.sum()).ravel())
            g_, l_, a_, b_ = np.unravel_index(ci, pc.shape)
            fgt, flt = fg_grid[g_], fl_grid[l_]
            c1, c2 = c_grid[a_], c_grid[b_]
            f = fgt + flt
        else:
            f = grids[3][fi]
            tot = node_table(T, sh, O, S.beta, i, k, l, [f], c_grid)[0]
            pc = np.exp(tot + lw[:, None] + lw[None, :] - np.max(tot))
            pc = np.nan_to_num(pc)
            ci = rng.choice(pc.size, p=(pc / pc.sum()).ravel())
            c1, c2 = c_grid[ci // c_grid.size], c_grid[ci % c_grid.size]
        w = S.weights(grids[0][i], grids[1][k], grids[2][l], c1, c2,
                      q_min=q_min)
        # predictive detected distribution above the cut (all stars)
        den = np.repeat(S._star_sum(w) * S.N, seg_n)
        wn = w / den
        pred = wn * S.det * (S.logP >= cut)
        # expected low-e fraction among detected long-period systems
        pl = pred * is_long
        pl_tot = float(pl.sum())
        rep["pred_low_e_long"].append(
            float(pl[e_eff < e_low].sum() / pl_tot) if pl_tot > 0
            else np.nan)
        # replicate catalogue
        n_d = n_below = n_above = 0
        rl, rk, re, rlow = [], [], [], 0
        rk_below = []
        n_long = n_long_low = 0
        det_j, fp_s = [], []
        for s in range(S.n_stars):
            u = rng.random()
            if u < f:
                sl = slice(S.seg[s], S.seg_end[s])
                ws = w[sl]
                if split:   # component chosen by f_gt : f_lt
                    side = S.logP[sl] >= cut
                    ws = ws * (side if u < fgt else ~side)
                j = sl.start + rng.choice(ws.size, p=ws / ws.sum())
                if S.det[j]:
                    det_j.append(j)
                    n_d += 1
                    if S.logP[j] >= cut:
                        n_above += 1
                        rl.append(S.logP[j]); rk.append(S.K1[j])
                        if e_eff[j] < e_low:
                            rlow += 1
                        else:
                            re.append(e_eff[j])
                        if is_long[j]:
                            n_long += 1
                            n_long_low += int(e_eff[j] < e_low)
                    else:
                        n_below += 1
                        rk_below.append(S.K1[j])
            elif rng.random() < S.beta[s]:
                n_d += 1
                fp_s.append(s)
        keep_rep(det_j, fp_s)
        sv_draw.append((grids[0][i], grids[1][k], grids[2][l], f,
                        fgt if split else np.nan, flt if split else np.nan,
                        c1, c2, int(reps_per_draw)))
        # extra replicates of this draw (separate stream; period-split
        # statistics and the saved catalogues only)
        for _ in range(int(reps_per_draw) - 1):
            det_x, fp_x = [], []
            for s in range(S.n_stars):
                u = rng2.random()
                if u < f:
                    sl = slice(S.seg[s], S.seg_end[s])
                    ws = w[sl]
                    if split:
                        side = S.logP[sl] >= cut
                        ws = ws * (side if u < fgt else ~side)
                    j = sl.start + rng2.choice(ws.size, p=ws / ws.sum())
                    if S.det[j]:
                        det_x.append(j)
                elif rng2.random() < S.beta[s]:
                    fp_x.append(s)
            keep_rep(det_x, fp_x)
        rep["N_D"].append(n_d)
        rep["frac_below"].append(n_below / max(n_below + n_above, 1))
        rep["frac_low_e"].append(rlow / max(n_above, 1))
        rep["frac_low_e_long"].append(n_long_low / n_long if n_long
                                      else np.nan)
        rk_arr = np.asarray(rk, float)
        rk_all = np.concatenate([rk_arr, np.asarray(rk_below, float)])
        rep["frac_lowK1"].append(float(np.mean(rk_arr < k1_low))
                                 if rk_arr.size else np.nan)
        rep["frac_lowK1_all"].append(float(np.mean(rk_all < k1_low))
                                     if rk_all.size else np.nan)
        pe = pred * (e_eff >= e_low)
        for name, xo, xr, xp, wp in (
                ("logP", logp_obs, rl, S.logP, pred),
                ("K1", k1_obs, rk, S.K1.astype(float), pred),
                ("ehi", e_hi_obs, re, e_eff, pe)):
            rep["D_%s_obs" % name].append(_ks_to_weighted(xo, xp, wp))
            rep["D_%s_rep" % name].append(_ks_to_weighted(xr, xp, wp))
    rep = {k: np.asarray(v, float) for k, v in rep.items()}

    def two_sided(x, o):
        return float(min(1.0, 2 * min(np.mean(x >= o), np.mean(x <= o))))
    res = {}
    for k in ("N_D", "frac_below", "frac_low_e", "frac_lowK1",
              "frac_lowK1_all"):
        x = rep[k][np.isfinite(rep[k])]
        res[k] = {"obs": obs_stats[k], "rep_median": float(np.median(x)),
                  "rep_16_84": [float(np.percentile(x, 16)),
                                float(np.percentile(x, 84))],
                  "p": two_sided(x, obs_stats[k])}
    res["frac_lowK1"]["k1_low"] = float(k1_low)
    # long-period low-e systems: replicate comparison, expected fraction,
    # and the predictive distribution of the count among the n_long_obs
    # observed orbits (binomial mixture over the posterior draws)
    x = rep["frac_low_e_long"][np.isfinite(rep["frac_low_e_long"])]
    pr = rep["pred_low_e_long"][np.isfinite(rep["pred_low_e_long"])]
    low = {"p_env": float(p_env), "e_low": float(e_low),
           "n_long_obs": n_long_obs, "k_low_obs": k_long_obs,
           "obs": obs_stats["frac_low_e_long"]}
    if x.size and pr.size and n_long_obs:
        from scipy.stats import binom
        kk = np.arange(n_long_obs + 1)
        pmf = binom.pmf(kk[None, :], n_long_obs, pr[:, None]).mean(axis=0)
        cdf = np.cumsum(pmf)

        def kq(u):
            return int(kk[min(int(np.searchsorted(cdf, u)), kk.size - 1)])
        low.update(
            rep_median=float(np.median(x)),
            rep_16_84=[float(np.percentile(x, 16)),
                       float(np.percentile(x, 84))],
            p=two_sided(x, obs_stats["frac_low_e_long"]),
            pred_frac_mean=float(pr.mean()),
            pred_frac_median=float(np.median(pr)),
            pred_frac_16_84=[float(np.percentile(pr, 16)),
                             float(np.percentile(pr, 84))],
            expected_count=float(n_long_obs * pr.mean()),
            count_median=kq(0.5), count_16_84=[kq(0.16), kq(0.84)],
            count_2_5_97_5=[kq(0.025), kq(0.975)],
            count_pmf=[float(v) for v in pmf])
    res["low_e_long"] = low
    for name in ("logP", "K1", "ehi"):
        do, dr = rep["D_%s_obs" % name], rep["D_%s_rep" % name]
        res["ks_" + name] = {"D_obs_median": float(np.nanmedian(do)),
                             "p": float(np.nanmean(dr >= do))}
    for k, v in res.items():
        log("[ppc] %-12s %s" % (k, v))
    ps = _period_split_summary(ps_obs, ps_rep)
    res["period_split"] = {"p_groups_days": [float(v) for v in p_groups],
                           "n_replicates": int(n_rep_total),
                           "reps_per_draw": int(reps_per_draw),
                           "stats": ps}
    for key in ["spearman"] + ["du|%gd" % v for v in p_groups]:
        log("[ppc] period_split %-10s %s" % (key, ps[key]))
    for pd_ in p_groups:
        for t in (1.6, 1.8):
            key = "u|lt_%gd|%.1f" % (pd_, t)
            log("[ppc] period_split %-14s %s" % (key, ps[key]))
    if save_reps:
        inj = np.concatenate(sv["inj"]) if sv["inj"] else np.zeros(0, int)
        tmp = save_reps + ".tmp.npz"
        np.savez_compressed(
            tmp, rep=np.concatenate(sv["rep"]),
            star=np.concatenate(sv["star"]),
            logP=S.logP[inj].astype(np.float32),
            K1=S.K1[inj].astype(np.float32),
            e=e_eff[inj].astype(np.float32), q=S.q[inj].astype(np.float32),
            fp_rep=np.concatenate(sv["fp_rep"]),
            fp_star=np.concatenate(sv["fp_star"]),
            draws=np.array(sv_draw, float),
            draws_columns=np.array(["pi", "kappa", "eta", "f", "f_gt",
                                    "f_lt", "c1", "c2", "reps"]),
            n_replicates=np.array(n_rep_total), cut=np.array(cut),
            star_ids=np.array([str(x) for x in S.stars["ID"]]),
            obs_star=orb["star_id"].astype(str).to_numpy(),
            obs_logP=orb["logP"].to_numpy(float),
            obs_K1=orb["K1"].to_numpy(float),
            obs_outcome=orb["outcome"].astype(str).to_numpy())
        os.replace(tmp, save_reps)
        log("[ppc] wrote %s (%d replicates, %d detected binaries)"
            % (save_reps, n_rep_total, inj.size))
    return res


# ---------------------------------------------------------------------------
# Re-analysis of a finished coverage campaign
# ---------------------------------------------------------------------------

def reanalyze(out_dir, campaign_dir, model="mix", methods=("cell", "linear"),
              log=print):
    """Per-mock rows for every estimator in ``methods`` plus summaries.

    Returns (rows, tables, hw) where ``rows`` holds one row per
    (mock, axis, method), ``tables`` maps method -> coverage table and
    ``hw`` summarizes hw68 / grid step per (truth, axis).
    """
    from simulations.closure_coverage import (
        coverage_rows_for_run, coverage_table)
    from simulations.true_posterior.run_posterior import (
        METRIC, TAG, TAG_SPLIT)
    tag = TAG_SPLIT if model == "split" else TAG
    grid_dirs = sorted(glob.glob(os.path.join(out_dir, "*", "seed_*",
                                              "grid_run")))
    if not grid_dirs:
        raise SystemExit("no <tag>/seed_*/grid_run under %s" % out_dir)
    parts = []
    for gd in grid_dirs:
        seed_dir = os.path.dirname(gd)
        truth_tag = os.path.basename(os.path.dirname(seed_dir))
        seed = int(os.path.basename(seed_dir).split("_")[1])
        truth_yaml = os.path.join(campaign_dir, truth_tag,
                                  "seed_%d" % seed, "truth_params.yaml")
        if not os.path.exists(truth_yaml):
            log("[reanalyze] missing %s — skipped" % truth_yaml)
            continue
        if not os.path.exists(os.path.join(gd, "grid_cubes.npz")):
            continue
        for m in methods:
            df = coverage_rows_for_run(gd, truth_yaml, truth_tag, seed,
                                       variant=tag, test=METRIC, method=m)
            df["method"] = m
            parts.append(df)
    rows = pd.concat(parts, ignore_index=True)
    tables = {}
    for m in methods:
        sub = rows[rows["method"] == m]
        t = coverage_table(sub)
        t.insert(0, "method", m)
        tables[m] = t
    # half-width in grid steps (estimator-independent to first order)
    base = rows[rows["method"] == methods[0]].copy()
    base["hw_over_step"] = base["hw68"] / base["grid_step"]
    hw = (base.groupby(["truth_tag", "axis"], sort=False)["hw_over_step"]
          .agg(["median", "min", "max",
                lambda v: float(np.mean(v < 0.8))])
          .rename(columns={"<lambda_0>": "frac_lt_0.8"}).reset_index())
    pooled = (base.groupby("axis", sort=False)["hw_over_step"]
              .agg(["median", "min", "max",
                    lambda v: float(np.mean(v < 0.8))])
              .rename(columns={"<lambda_0>": "frac_lt_0.8"}).reset_index())
    pooled.insert(0, "truth_tag", "POOLED")
    hw = pd.concat([hw, pooled], ignore_index=True)
    edge = (base.groupby(["truth_tag", "axis"], sort=False)["edge_mass"]
            .agg(["median", "max"]).reset_index())
    edge_pooled = (base.groupby("axis", sort=False)["edge_mass"]
                   .agg(["median", "max"]).reset_index())
    edge_pooled.insert(0, "truth_tag", "POOLED")
    edge = pd.concat([edge, edge_pooled], ignore_index=True).rename(
        columns={"median": "edge_mass_median", "max": "edge_mass_max"})
    hw = hw.merge(edge, on=["truth_tag", "axis"], how="left")
    return rows, tables, hw


# ---------------------------------------------------------------------------
# Flat-prior finite-N offset of a power-law index (toy)
# ---------------------------------------------------------------------------

def flatprior_toy(alpha, xmin, xmax, n_obs, n_rep=4000, seed=0, half=2.0,
                  step=0.0025):
    """Mean offset of the posterior median (and mean, mode) of a power-law
    index from its truth, for N iid draws x ~ x^alpha on [xmin, xmax] with
    a flat prior on alpha. Pure statistics: no selection, no grid effects
    (the index grid is fine and spans +- ``half`` around the truth)."""
    from simulations.bias_grid_lib.marginals import grid_quantiles
    from simulations.true_posterior.population import (
        powerlaw_logpdf, powerlaw_ppf)
    rng = np.random.default_rng(seed)
    grid = np.arange(alpha - half, alpha + half + 1e-9, step)
    med, mean, mode = [], [], []
    for _ in range(n_rep):
        x = powerlaw_ppf(rng.random(n_obs), alpha, xmin, xmax)
        lnl = powerlaw_logpdf(x[None, :], grid[:, None], xmin, xmax).sum(1)
        p = np.exp(lnl - lnl.max())
        med.append(grid_quantiles(p, grid, (0.5,))[0])
        mean.append(float(np.sum(grid * p) / p.sum()))
        mode.append(float(grid[np.argmax(p)]))
    med, mean, mode = (np.asarray(v) - alpha for v in (med, mean, mode))
    lam = alpha + 1.0
    return {"alpha": alpha, "xmin": xmin, "xmax": xmax, "n_obs": n_obs,
            "n_rep": n_rep,
            "median_offset": float(med.mean()),
            "median_offset_se": float(med.std(ddof=1) / np.sqrt(n_rep)),
            "mean_offset": float(mean.mean()),
            "mode_offset": float(mode.mean()),
            "sd_median": float(med.std(ddof=1)),
            "formula_5lam_3N": float(5.0 * lam / (3.0 * (n_obs - 1)))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    fp = sub.add_parser("flatprior",
                        help="flat-prior finite-N offset toy for power-law "
                             "indices")
    fp.add_argument("--case", action="append", required=True,
                    help="name:alpha:xmin:xmax:N (repeatable)")
    fp.add_argument("--n-rep", type=int, default=4000)
    fp.add_argument("--out", required=True, help="csv path")
    r = sub.add_parser("reanalyze")
    r.add_argument("--out-dir", required=True,
                   help="coverage output dir (<tag>/seed_*/grid_run)")
    r.add_argument("--campaign-dir", required=True,
                   help="mock campaign dir with truth_params.yaml files")
    r.add_argument("--model", choices=["mix", "split"], default="mix")
    r.add_argument("--out", required=True, help="output directory")
    b = sub.add_parser("bridge")
    b.add_argument("--injset", required=True)
    b.add_argument("--run-dir", required=True)
    b.add_argument("--n-nodes", type=int, default=60)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--q-min", type=float, default=None)
    b.add_argument("--out", required=True)
    p = sub.add_parser("ppc")
    p.add_argument("--injset", required=True)
    p.add_argument("--obs", required=True)
    p.add_argument("--post-dir", required=True)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--below-pcirc-as-circular", action="store_true",
                   help="treat eccentric orbits below p_circ as circular "
                        "(needed for cuts below p_circ, e.g. pstar_2d)")
    p.add_argument("--q-min", type=float, default=None,
                   help="q domain of the run (must match its "
                        "run_posterior.yaml)")
    p.add_argument("--k1-floor", type=float, default=None,
                   help="amplitude floor of the run in km/s (must match its "
                        "run_posterior.yaml)")
    p.add_argument("--k1-unknown-above-floor", action="store_true",
                   help="with --k1-floor: accept detected stars without a "
                        "K1, as the run did")
    p.add_argument("--below-shape", choices=["k1", "k1e"], default=None,
                   help="below-cut orbit terms of the run (must match its "
                        "run_posterior.yaml)")
    p.add_argument("--p-groups", default="5,10,20",
                   help="period splits in days of the period-split "
                        "statistics")
    p.add_argument("--reps-per-draw", type=int, default=1,
                   help="replicates per posterior draw for the period-split "
                        "statistics (extra ones use a separate random "
                        "stream; other statistics are unchanged)")
    p.add_argument("--save-reps", default=None,
                   help="npz path: keep the detected binaries of every "
                        "replicate")
    p.add_argument("--out", required=True, help="yaml output path")
    a = ap.parse_args(argv)
    if a.cmd == "flatprior":
        rows = []
        for spec in a.case:
            name, al, lo, hi, n = spec.split(":")
            rec = flatprior_toy(float(al), float(lo), float(hi), int(n),
                                n_rep=a.n_rep)
            rec = {"case": name, **rec}
            rows.append(rec)
            print("[flatprior] %-10s alpha %+.3f N %3d: median offset %+.4f "
                  "+- %.4f  (mean %+.4f, mode %+.4f; 5lam/3(N-1) = %+.4f; "
                  "sd of median %.3f)"
                  % (name, rec["alpha"], rec["n_obs"], rec["median_offset"],
                     rec["median_offset_se"], rec["mean_offset"],
                     rec["mode_offset"], rec["formula_5lam_3N"],
                     rec["sd_median"]), flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        return 0
    if a.cmd == "reanalyze":
        rows, tables, hw = reanalyze(a.out_dir, a.campaign_dir, a.model)
        os.makedirs(a.out, exist_ok=True)
        rows.to_csv(os.path.join(a.out, "reanalysis_rows.csv"), index=False)
        pd.concat(tables.values(), ignore_index=True).to_csv(
            os.path.join(a.out, "reanalysis_coverage.csv"), index=False)
        hw.to_csv(os.path.join(a.out, "reanalysis_hw.csv"), index=False)
        fmt = lambda v: "%.3f" % v   # noqa: E731
        with pd.option_context("display.width", 160,
                               "display.max_columns", 30):
            for m, t in tables.items():
                print("\n== coverage, quantile method = %s ==" % m)
                cols = ["truth_tag", "axis", "level", "n_mocks", "coverage",
                        "wilson_lo", "wilson_hi", "z_mean", "z_median",
                        "n_below", "n_above", "hw68_mean"]
                print(t[cols].to_string(index=False, float_format=fmt))
            print("\n== 68%% half-width / grid step (%s), end-node mass =="
                  % list(tables)[0])
            print(hw.to_string(index=False, float_format=fmt))
        print("\nwrote %s/reanalysis_{rows,coverage,hw}.csv" % a.out)
        return 0
    if a.cmd == "bridge":
        df, summ = bridge(a.injset, a.run_dir, n_nodes=a.n_nodes,
                          seed=a.seed, q_min=a.q_min)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        df.to_csv(a.out, index=False)
        print("SUMMARY", summ)
    else:
        import yaml
        res = ppc(a.injset, a.obs, a.post_dir, n_draws=a.n_draws,
                  seed=a.seed,
                  below_pcirc_ecc=("as_circular"
                                   if a.below_pcirc_as_circular else "raise"),
                  q_min=a.q_min, k1_floor=a.k1_floor,
                  k1_unknown=("above_floor" if a.k1_unknown_above_floor
                              else "raise"),
                  below_shape=a.below_shape,
                  p_groups=tuple(float(v) for v in a.p_groups.split(",")),
                  reps_per_draw=a.reps_per_draw, save_reps=a.save_reps)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            yaml.safe_dump(res, fh, sort_keys=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
