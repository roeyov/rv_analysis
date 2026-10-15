"""
simulations.true_posterior.pdet_bins — bin-wise detection probability of
the survey under a population fit, from the labelled injection set.

For each bin of logP, e, q and K1 (the bins of the paper's histogram figures),

    p_det(bin) = sum_{j in bin} w_j d_j / sum_{j in bin} w_j,

where j runs over all injections of all catalogue stars (equal injection
counts per star), w_j = p(z_j | theta, c) / g(z_j) is the importance weight
at the fit's parameters and d_j the detection flag. This is the detection
probability of a binary whose logP (or e, or q) falls in the bin, averaged
over the other orbital elements, the stars and the nuisances under that
fit. e is the true orbital eccentricity (0 for the circular component and
below p_circ). With ``--split-cut/--f-gt/--f-lt`` the period weights carry
the split model's two normalisations (so the e and q averages use its
period distribution).

Output: a CSV with columns axis, lo, hi, pdet, w_sum, n_inj, plus the
per-bin intrinsic mass fraction ``mass`` (sum of weights in the bin over
the axis total; injections outside the axis' edges are excluded from both,
so the cumulative ``mass`` is the intrinsic CDF normalised over the binned
range). Equal-weight per star is used because the observed samples are
drawn from the same 134 stars.

K1 is a derived quantity: each injection's K1 was drawn with its star's
M1 scatter and an isotropic inclination, so the reweighted K1 axes give
the intrinsic K1 distribution implied by the fitted power laws and the
assumed nuisance distributions. ``K1`` uses the histogram's 25 km/s bins
(for the bin-wise completeness correction), ``K1_fine`` 2 km/s bins (for
a smooth intrinsic CDF).

With ``--logp-window lo,hi`` the e, q and ``all`` averages are restricted
to that period window (the figures' (0, 3) dex), so the three axes share
one denominator and sum(mass * pdet) equals the ``all`` value on each.

``--q-min`` restricts the mass-ratio law to [q_min, 1] (the q domain of a
qmin01-type fit); injections below it get zero weight.

``--map-out <npz>`` also writes the detection probability on a 2-D
(logP, q) lattice over the full model range, as weighted sums per cell
(``w_sum``, ``wd_sum``, ``w2_sum``, ``w2d_sum``; ``pdet = wd_sum / w_sum``
with the delta-method standard error ``pdet_se`` and the Kish effective
size ``n_eff``). The map ignores ``--logp-window``. The file also carries
the overall detection probability over the full range and inside the
window, and the same for q < 0.1.

``--pu-out <npz>`` writes the same weighted sums on (logP bin, u),
(logP bin, u0) and (logP bin, K1) lattices over the full model range, with
u = log10 K1 + (1/3) logP + (1/2) log10(1 - e^2) (``k1_density.u_of``, true
e of the injection) and u0 the same without the e term. The first and last
u / K1 bins collect everything beyond the edges. From these tables the
expected fraction of detected binaries with u (or K1) in any range follows
for any grouping of the period bins: sum(wd_sum in range) / sum(wd_sum).

``--obs <catalogue> --orbit-out <csv>`` writes one row per observed orbit
(DX rows and below-cut DO rows with a K1): where the orbit sits in the
model's distribution of DETECTED simulated binaries of the same star with
|logP - logP_obs| <= ``--orbit-half-width``. ``pit_u0`` is the weighted
fraction of those with u0 below the observed u0 (both sides without the e
term, so Lucy-Sweeney limits play no role); ``pit_u`` uses the true u of the
injections and the likelihood's u of the orbit (e = 0 for LS rows and below
p_circ). Under the fitted population the ranks are uniform on (0, 1) and
unrelated to period. The table also gives the model probabilities of u0
below fixed thresholds and of K1 in fixed ranges for that star and period,
and ``pdet_at_u0``, the detection probability within 0.1 dex of the
observed u0. Injection K1 and u are true values.

    python -m simulations.true_posterior.pdet_bins --injset <pb set> \\
        --theta=-0.845,-0.638,-0.292,0.504,0.095 \\
        --split-cut 0.5266 --f-gt 0.544 --f-lt 0.116 \\
        --logp-window 0,3 --out <csv> [--map-out <npz>] \\
        [--pu-out <npz> --pu-logp-edges 0.046,...,3.5] \\
        [--obs <obs_catalog.csv> --orbit-out <csv>]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from simulations.true_posterior.injection_set import load_injection_set
from simulations.true_posterior.likelihood import InjectionSet
from simulations.true_posterior.population import powerlaw_cdf

DEFAULT_LOGP_EDGES = np.round(np.arange(0.0, 3.0001, 0.25), 6)     # 0.25 dex
DEFAULT_E_EDGES = np.round(np.linspace(0.0, 1.0, 11), 6)
DEFAULT_Q_EDGES = np.round(np.linspace(0.0, 1.0, 11), 6)
DEFAULT_K1_EDGES = np.round(np.linspace(0.0, 300.0, 13), 6)        # km/s, 25 wide
DEFAULT_K1_FINE_EDGES = np.round(np.linspace(0.0, 300.0, 151), 6)  # 2 km/s
U_THRESHOLDS = (1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
K1_RANGES = ((0.0, 10.0), (10.0, 25.0), (25.0, 50.0), (50.0, 100.0),
             (100.0, np.inf))


def _edges(spec, default):
    if spec is None:
        return default
    lo, hi, n = spec.split(",")
    return np.linspace(float(lo), float(hi), int(n) + 1)


def bin_pdet(x, w, d, edges):
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    n = len(edges) - 1
    w_sum = np.bincount(idx[ok], weights=w[ok], minlength=n)
    wd_sum = np.bincount(idx[ok], weights=(w * d)[ok], minlength=n)
    n_inj = np.bincount(idx[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdet = np.where(w_sum > 0, wd_sum / w_sum, np.nan)
    return pdet, w_sum, n_inj


def _cell_index(x, edges):
    """Bin index of x on ``edges``; the outer edges are inclusive (float32
    injections may sit a rounding error outside them). -1 / n outside."""
    edges = np.asarray(edges, float)
    n = edges.size - 1
    idx = np.searchsorted(edges, x, side="right") - 1
    tol = 1e-6 * (edges[-1] - edges[0])
    idx[(idx == n) & (x <= edges[-1] + tol)] = n - 1
    idx[(idx == -1) & (x >= edges[0] - tol)] = 0
    return idx


def pdet_map(logP, q, w, d, logp_edges, q_edges):
    """Weighted detection probability on a (logP, q) lattice.

    Returns a dict of (n_logP, n_q) arrays: the weighted sums, ``pdet``,
    its delta-method standard error for a self-normalized ratio,
    sqrt(sum w^2 (d - p)^2) / sum w, the Kish effective size and the raw
    injection counts; plus ``w_outside``, the weight outside the lattice.
    """
    logp_edges = np.asarray(logp_edges, float)
    q_edges = np.asarray(q_edges, float)
    n1, n2 = logp_edges.size - 1, q_edges.size - 1
    i = _cell_index(logP, logp_edges)
    j = _cell_index(q, q_edges)
    ok = (i >= 0) & (i < n1) & (j >= 0) & (j < n2)
    flat = i[ok] * n2 + j[ok]
    wk, dk = w[ok], d[ok]

    def tot(v=None):
        return np.bincount(flat, weights=v, minlength=n1 * n2).reshape(n1, n2)
    w_sum, wd_sum = tot(wk), tot(wk * dk)
    w2_sum, w2d_sum = tot(wk * wk), tot(wk * wk * dk)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdet = np.where(w_sum > 0, wd_sum / w_sum, np.nan)
        # sum w^2 (d - p)^2 with d in {0, 1}
        s2 = (1.0 - pdet) ** 2 * w2d_sum + pdet ** 2 * (w2_sum - w2d_sum)
        pdet_se = np.where(w_sum > 0, np.sqrt(np.maximum(s2, 0.0)) / w_sum,
                           np.nan)
        n_eff = np.where(w2_sum > 0, w_sum ** 2 / w2_sum, 0.0)
    return {"pdet": pdet, "pdet_se": pdet_se, "n_eff": n_eff,
            "w_sum": w_sum, "wd_sum": wd_sum, "w2_sum": w2_sum,
            "w2d_sum": w2d_sum, "n_inj": tot().astype(np.int64),
            "n_inj_det": tot(dk).astype(np.int64),
            "w_outside": float(np.sum(w[~ok]))}


def orbit_table(S, w, d, table, u_inj, u0_inj, half_width):
    """Per-orbit ranks in the model's detected distribution (module
    docstring). ``table`` is an obs_catalog; returns a DataFrame."""
    ids = [str(x) for x in S.stars["ID"]]
    pos = {sid: i for i, sid in enumerate(ids)}
    tab = table.copy()
    tab["star_id"] = tab["star_id"].astype(str)
    reason = tab["reason"].fillna("").astype(str) if "reason" in tab else ""
    use = ((tab["outcome"] == "DX")
           | ((tab["outcome"] == "DO") & (reason == "below_cut")))
    use &= np.isfinite(tab["K1"].astype(float)) & np.isfinite(
        tab["logP"].astype(float))
    wd = w * d
    lo_all, hi_all = S.bounds["log_p_min"], S.bounds["log_p_max"]
    rows = []
    for _, r in tab[use].iterrows():
        i = pos[r["star_id"]]
        sl = slice(int(S.seg[i]), int(S.seg_end[i]))
        lp, k1 = float(r["logP"]), float(r["K1"])
        ls = bool(r["e_is_upper_limit"])
        e_obs = float(r["e"]) if np.isfinite(r["e"]) else 0.0
        e_lik = 0.0 if (ls or 10.0 ** lp < S.p_circ) else e_obs
        u0 = np.log10(k1) + lp / 3.0
        u_lik = u0 + 0.5 * np.log10(1.0 - e_lik ** 2)
        a = max(lp - half_width, lo_all)
        b = min(lp + half_width, hi_all)
        m = (S.logP[sl] >= a) & (S.logP[sl] <= b)
        wm, wdm = w[sl][m], wd[sl][m]
        x0, x, kk = u0_inj[sl][m], u_inj[sl][m], S.K1[sl][m]
        W = float(wdm.sum())
        near = np.abs(x0 - u0) <= 0.1
        rec = {"star_id": r["star_id"], "outcome": r["outcome"],
               "reason": r.get("reason", ""), "logP": lp, "K1": k1,
               "e": e_obs, "ls": ls, "u0": u0, "u_lik": u_lik,
               "win_lo": a, "win_hi": b, "n_inj_win": int(m.sum()),
               "n_det_win": int((d[sl][m] > 0).sum()),
               "n_eff_det": (W * W / float((wdm * wdm).sum())
                             if W > 0 else 0.0),
               "pdet_win": (W / float(wm.sum()) if wm.sum() > 0
                            else np.nan),
               "pdet_at_u0": (float(wdm[near].sum() / wm[near].sum())
                              if wm[near].sum() > 0 else np.nan),
               "n_inj_at_u0": int(near.sum())}
        if W > 0:
            rec["pit_u0"] = float(wdm[x0 <= u0].sum() / W)
            rec["pit_u"] = float(wdm[x <= u_lik].sum() / W)
            for t in U_THRESHOLDS:
                rec["p_u0_lt_%.1f" % t] = float(wdm[x0 < t].sum() / W)
            for lo, hi in K1_RANGES:
                key = ("p_k1_ge%g" % lo if not np.isfinite(hi)
                       else "p_k1_%g_%g" % (lo, hi))
                rec[key] = float(wdm[(kk >= lo) & (kk < hi)].sum() / W)
        rows.append(rec)
    return pd.DataFrame(rows)


def _ratio(w, d, m):
    """(weighted detection probability, weight fraction) of mask m."""
    tot = float(np.sum(w))
    wm = float(np.sum(w[m]))
    return (float(np.sum(w[m] * d[m]) / wm) if wm > 0 else np.nan,
            wm / tot if tot > 0 else np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--injset", required=True)
    ap.add_argument("--theta", required=True,
                    help="pi,kappa,eta[,c1,c2] (c1 = c2 = 0 if omitted)")
    ap.add_argument("--split-cut", type=float, default=None)
    ap.add_argument("--f-gt", type=float, default=None)
    ap.add_argument("--f-lt", type=float, default=None)
    ap.add_argument("--logp-edges", default=None, help="lo,hi,n")
    ap.add_argument("--e-edges", default=None, help="lo,hi,n")
    ap.add_argument("--q-edges", default=None, help="lo,hi,n")
    ap.add_argument("--k1-edges", default=None,
                    help="lo,hi,n for the K1 axis in km/s (default 0,300,12)")
    ap.add_argument("--k1-fine-edges", default=None,
                    help="lo,hi,n for the fine K1 axis (default 0,300,150)")
    ap.add_argument("--logp-window", default=None,
                    help="lo,hi: restrict the e, q and all averages to this "
                         "logP window (e.g. 0,3)")
    ap.add_argument("--max-per-star", type=int, default=None)
    ap.add_argument("--q-min", type=float, default=None,
                    help="restrict the q domain of the fit (e.g. 0.1)")
    ap.add_argument("--map-out", default=None,
                    help="npz path for the 2-D (logP, q) detection map")
    ap.add_argument("--map-logp-edges", default="0.046,3.5,24",
                    help="lo,hi,n of the map's linear logP cells")
    ap.add_argument("--map-q-edges", default="0.01,1.0,20",
                    help="lo,hi,n of the map's logarithmic q cells")
    ap.add_argument("--pu-out", default=None,
                    help="npz path for the (logP bin, u / u0 / K1) weighted "
                         "sums")
    ap.add_argument("--pu-logp-edges", default=None,
                    help="comma list of the logP bin edges of --pu-out")
    ap.add_argument("--u-edges", default="0.0,3.4,170", help="lo,hi,n")
    ap.add_argument("--k1-map-edges", default="0,400,400", help="lo,hi,n")
    ap.add_argument("--obs", default=None,
                    help="obs_catalog.csv for --orbit-out")
    ap.add_argument("--orbit-out", default=None,
                    help="csv path for the per-orbit rank table")
    ap.add_argument("--orbit-half-width", type=float, default=0.125,
                    help="half-width in logP of the period window of "
                         "--orbit-out")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.pu_out and not a.pu_logp_edges:
        ap.error("--pu-out needs --pu-logp-edges")
    if bool(a.orbit_out) != bool(a.obs):
        ap.error("--orbit-out and --obs go together")

    th = [float(v) for v in a.theta.split(",")]
    pi, kappa, eta = th[:3]
    c1 = th[3] if len(th) > 3 else 0.0
    c2 = th[4] if len(th) > 4 else 0.0
    data, stars, man = load_injection_set(a.injset)
    S = InjectionSet(data, stars, man, max_per_star=a.max_per_star)
    del data
    w = S.weights(pi, kappa, eta, c1, c2, q_min=a.q_min)
    if a.split_cut is not None:
        b = S.bounds
        F_gt = 1.0 - float(powerlaw_cdf(a.split_cut, pi, b["log_p_min"],
                                        b["log_p_max"]))
        above = S.logP >= a.split_cut
        f_tot = a.f_gt + a.f_lt
        # period weights of the split model relative to the single power law
        w = w * np.where(above, (a.f_gt / f_tot) / F_gt,
                         (a.f_lt / f_tot) / (1.0 - F_gt))
    d = S.det.astype(float)
    e_true = np.where(S.short | S.circ_comp, 0.0, S.e)
    win = np.ones(S.logP.shape, dtype=bool)
    lo_w = hi_w = float("nan")
    if a.logp_window:
        lo, hi = (float(v) for v in a.logp_window.split(","))
        lo_w, hi_w = lo, hi
        win = (S.logP >= lo) & (S.logP < hi)
    rows = []
    for axis, x, edges in (("logP", S.logP, _edges(a.logp_edges,
                                                   DEFAULT_LOGP_EDGES)),
                           ("e", e_true, _edges(a.e_edges, DEFAULT_E_EDGES)),
                           ("q", S.q, _edges(a.q_edges, DEFAULT_Q_EDGES)),
                           ("K1", S.K1, _edges(a.k1_edges, DEFAULT_K1_EDGES)),
                           ("K1_fine", S.K1,
                            _edges(a.k1_fine_edges, DEFAULT_K1_FINE_EDGES))):
        pdet, w_sum, n_inj = bin_pdet(x[win], w[win], d[win], edges)
        tot = w_sum.sum()
        for i in range(len(edges) - 1):
            rows.append({"axis": axis, "lo": edges[i], "hi": edges[i + 1],
                         "pdet": pdet[i], "w_sum": w_sum[i],
                         "mass": w_sum[i] / tot, "n_inj": int(n_inj[i])})
        if len(edges) <= 16:
            print("%-4s " % axis + " ".join("[%.2f,%.2f):%.3f" % (edges[i], edges[i + 1], pdet[i])
                                            for i in range(len(edges) - 1)),
                  flush=True)
        else:
            print("%-7s %d bins on [%g, %g]; intrinsic mass inside the range "
                  "%.4f of the window total" % (axis, len(edges) - 1, edges[0],
                                                edges[-1], tot / np.sum(w[win])),
                  flush=True)
    overall = float(np.sum(w[win] * d[win]) / np.sum(w[win]))
    rows.append({"axis": "all", "lo": np.nan, "hi": np.nan, "pdet": overall,
                 "w_sum": float(np.sum(w[win])), "mass": 1.0,
                 "n_inj": int(win.sum())})
    print("overall p_det under the fit: %.4f" % overall)
    df = pd.DataFrame(rows)
    df.attrs = {}
    with open(a.out, "w") as fh:
        fh.write("# injset=%s theta=%s split_cut=%s f_gt=%s f_lt=%s "
                 "logp_window=%s%s\n"
                 % (a.injset, a.theta, a.split_cut, a.f_gt, a.f_lt,
                    a.logp_window,
                    "" if a.q_min is None else " q_min=%s" % a.q_min))
        df.to_csv(fh, index=False)
    print("wrote", a.out)
    if a.map_out:
        lo, hi, n = a.map_logp_edges.split(",")
        logp_edges = np.linspace(float(lo), float(hi), int(n) + 1)
        lo, hi, n = a.map_q_edges.split(",")
        q_edges = np.geomspace(float(lo), float(hi), int(n) + 1)
        m = pdet_map(S.logP, S.q, w, d, logp_edges, q_edges)
        everything = np.ones(w.shape, dtype=bool)
        lowq = S.q < 0.1
        nan = float("nan")
        # written under a temporary name and renamed, so that a killed
        # job cannot leave a truncated map behind
        tmp = a.map_out + ".tmp.npz"
        np.savez(tmp, logp_edges=logp_edges, q_edges=q_edges,
                 theta=np.array([pi, kappa, eta, c1, c2]),
                 split_cut=nan if a.split_cut is None else a.split_cut,
                 f_gt=nan if a.f_gt is None else a.f_gt,
                 f_lt=nan if a.f_lt is None else a.f_lt,
                 q_min=nan if a.q_min is None else a.q_min,
                 logp_window=(np.array([lo_w, hi_w]) if a.logp_window
                              else np.array([nan, nan])),
                 injset=np.array(str(a.injset)),
                 n_injections=np.array(w.size, dtype=np.int64),
                 w_total=float(np.sum(w)),
                 pdet_all_full=_ratio(w, d, everything)[0],
                 pdet_all_window=overall,
                 pdet_lowq_full=_ratio(w, d, lowq)[0],
                 mass_lowq_full=_ratio(w, d, lowq)[1],
                 pdet_lowq_window=_ratio(w[win], d[win], lowq[win])[0],
                 mass_lowq_window=_ratio(w[win], d[win], lowq[win])[1],
                 **m)
        os.replace(tmp, a.map_out)
        print("map: %d x %d cells, weight outside the lattice %.3g of %.6g; "
              "p_det full range %.4f, window %.4f; q < 0.1: %.4f (%.4f of "
              "the binaries, full range)"
              % (logp_edges.size - 1, q_edges.size - 1, m["w_outside"],
                 float(np.sum(w)), _ratio(w, d, everything)[0], overall,
                 _ratio(w, d, lowq)[0], _ratio(w, d, lowq)[1]))
        print("wrote", a.map_out)
    if a.pu_out or a.orbit_out:
        lk = np.log10(np.maximum(S.K1.astype(float), 1e-12))
        u0_inj = lk + S.logP / 3.0
        u_inj = u0_inj + 0.5 * np.log10(1.0 - e_true.astype(float) ** 2)
    if a.pu_out:
        lp_edges = np.array([float(v) for v in a.pu_logp_edges.split(",")])
        u_edges = _edges(a.u_edges, None)
        k_edges = _edges(a.k1_map_edges, None)
        tabs = {}
        for name, y, edges in (("u", u_inj, u_edges), ("u0", u0_inj, u_edges),
                               ("k1", S.K1.astype(float), k_edges)):
            m = pdet_map(S.logP, np.clip(y, edges[0], edges[-1]), w, d,
                         lp_edges, edges)
            for key in ("w_sum", "wd_sum", "w2_sum", "w2d_sum", "n_inj",
                        "n_inj_det"):
                tabs["%s_%s" % (name, key)] = m[key]
            tabs["%s_w_outside" % name] = m["w_outside"]
            tabs["%s_n_clipped" % name] = np.array(
                [int(np.sum(y < edges[0])), int(np.sum(y > edges[-1]))])
        nan = float("nan")
        tmp = a.pu_out + ".tmp.npz"
        np.savez(tmp, logp_edges=lp_edges, u_edges=u_edges, k1_edges=k_edges,
                 theta=np.array([pi, kappa, eta, c1, c2]),
                 split_cut=nan if a.split_cut is None else a.split_cut,
                 f_gt=nan if a.f_gt is None else a.f_gt,
                 f_lt=nan if a.f_lt is None else a.f_lt,
                 q_min=nan if a.q_min is None else a.q_min,
                 injset=np.array(str(a.injset)),
                 n_injections=np.array(w.size, dtype=np.int64),
                 n_stars=np.array(S.n_stars, dtype=np.int64),
                 w_total=float(np.sum(w)), wd_total=float(np.sum(w * d)),
                 **tabs)
        os.replace(tmp, a.pu_out)
        print("pu tables: %d logP bins x %d u bins / %d K1 bins; detected "
              "weight inside the logP edges %.6f of the total"
              % (lp_edges.size - 1, u_edges.size - 1, k_edges.size - 1,
                 tabs["u0_wd_sum"].sum() / float(np.sum(w * d))))
        print("wrote", a.pu_out)
    if a.orbit_out:
        ot = orbit_table(S, w, d, pd.read_csv(a.obs), u_inj, u0_inj,
                         a.orbit_half_width)
        with open(a.orbit_out, "w") as fh:
            fh.write("# injset=%s theta=%s split_cut=%s f_gt=%s f_lt=%s "
                     "obs=%s half_width=%s%s\n"
                     % (a.injset, a.theta, a.split_cut, a.f_gt, a.f_lt,
                        a.obs, a.orbit_half_width,
                        "" if a.q_min is None else " q_min=%s" % a.q_min))
            ot.to_csv(fh, index=False)
        print("orbit table: %d orbits; smallest detected-weight effective "
              "size %.0f; mean rank in u0 %.3f"
              % (len(ot), ot["n_eff_det"].min(), ot["pit_u0"].mean()))
        print("wrote", a.orbit_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
