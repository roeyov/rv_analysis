"""
simulations.true_posterior.likelihood — the per-star selection-corrected
likelihood.

Model. Each catalogue star s is a binary with probability f (f_bin). Its
orbit z = (logP, q, e) comes from the grid's power laws with p_circ
applied: e = 0 below p_circ. Above p_circ a fraction c(P) of binaries is
circular. c1 is the fraction on [p_circ, P_env) and c2 the fraction on
[P_env, inf), with P_env = 5 d by default; both are nuisance parameters.
The rest have e drawn from e^eta. Detection uses the grid's detector.
Singles are falsely detected with rate beta_s.

    ln L = sum_{s in ND} ln[1 - f a_s - (1-f) b_s]
         + sum_{s in D}  ln[f a_s + (1-f) b_s]
         + sum_{s in DX} [ln p_s(x_s | theta, c) - ln a_s^{>=cut}(theta, c)]

a_s(theta, c) is the probability that a binary of star s is detected
(a_s^{>=cut} counts only logP >= logP*). This module estimates it by
self-normalized importance reweighting of the labelled injection set:

    a_s = sum_j det_j w_j / sum_j w_j,
    w_j = wP_j(pi) * wq_j(kappa) * we_j(eta, c)

with wP = p_P/g_P and wq = p_q/g_q. The e factor we_j is
    1                                           if P_j < p_circ;
    c(P_j) / w_circ                             if the draw was circular;
    (1 - c(P_j)) p_e(e_j|eta) / ((1-w_circ) g_e(e_j))   otherwise.
The draws come from g and the per-dimension reference mixtures keep the
weights bounded. The self-normalized form is used because it is much
less noisy than N_s^-1 sum det w when a_s is close to 1. Both its
numerator and its denominator are linear in (c1, c2).

The observed-system density p_s(x | theta, c) is built by
``ObservedData`` and ``shape_terms`` below. P_det,s(x_s) cancels between
numerator and denominator: detection depends only on the RV curve, i.e.
on (P, e, K1, omega, T0, noise, cadence), never on q or M1 on their own.

Per detected star with a usable orbit above the cut (DX), the density of
x = (logP, e, log10 K1) is:
  eccentric orbit:   (1 - c_b) p_P(logP|pi) p_e(e|eta) p_s(u|kappa),
  LS-circular orbit: p_P(logP|pi) [c_b p_s(u0|kappa)
                                   + (1 - c_b) I_s(eta, kappa)],
  with I_s = int_0^{e_lim} p_e(e|eta) p_s(u(e)|kappa) de,
where c_b is c1 or c2 by period bin, set to 1 below p_circ (e is forced to
0 there). "LS-circular" means the orbit's e is reported as a
Lucy-Sweeney upper limit e_lim. The censored term assumes the detection
probability is flat over e in [0, e_lim] at the observed (P, K1).

Split model, ``below_shape`` (optional). A detected orbit below the cut is
normally scored f_lt a_lt (detected, below the cut): its K1 and e are not
used. With ``below_shape`` it is scored f_lt times the same (e, log10 K1)
density as above, without the period factor p_P:
  "k1"   f_lt p_s(u0|kappa)                     (amplitude only, u at e = 0)
  "k1e"  f_lt times the eccentric / LS-circular expression above with
         p_P := 1 and c_b by the orbit's period (1 below p_circ)
The period of such an orbit is only used to pick c_b; no density of it
enters, so pi is constrained exactly as without the option. P_det(x) again
drops out, and the selection stays in the ND terms through a_lt.
"""

import types
import warnings

import numpy as np

from simulations.true_posterior.population import (
    ReferenceDensity, powerlaw_cdf, powerlaw_logpdf)
from simulations.true_posterior.k1_density import M1_CLIP, UDensity, u_of


class InjectionSet:
    """Per-star view of a labelled injection set, ready for reweighting."""

    def __init__(self, data, stars, manifest, p_env=5.0, max_per_star=None,
                 subset=None, k1_floor=None):
        """``subset=(k, n)`` keeps block k of n contiguous, equal-size
        blocks of every star's records (and of its singles), so the n
        blocks are disjoint and each is an unbiased subsample (shards are
        iid chunks). ``max_per_star`` then keeps the first records of what
        remains.

        ``k1_floor`` (km/s) redefines the survey with an amplitude floor: a
        simulated binary counts as detected only if the detector flagged it
        and its true K1 is at least the floor. The observed table must be
        classified the same way, i.e. detected orbits below the floor
        become ND (``ObservedData`` checks this). A detected star without
        a measured K1 (period lower limit, no solution) is then scored as
        "flagged and K1 >= floor", which is an assumption about that star;
        ``ObservedData`` refuses such rows unless told to accept it. The
        floor acts on the true K1 of a simulated binary and on the fitted
        K1 of an observed one. The singles' false-positive rate beta_s is
        left as stored: no fitted amplitude exists for a false positive,
        so beta_s is an upper limit under a floor."""
        order = np.argsort(data["star"], kind="stable")
        st = data["star"][order]
        first = np.searchsorted(st, st, side="left")
        n_s = np.searchsorted(st, st, side="right") - first
        rank = np.arange(st.size) - first
        keep = np.ones(st.size, dtype=bool)
        if subset is not None:
            k, n = int(subset[0]), int(subset[1])
            if not (n >= 1 and 0 <= k < n):
                raise ValueError("subset must be (k, n) with 0 <= k < n")
            lo = (k * n_s) // n
            keep &= (rank >= lo) & (rank < ((k + 1) * n_s) // n)
            rank = rank - lo
        if max_per_star is not None:
            keep &= rank < int(max_per_star)
        order = order[keep]
        self.subset = None if subset is None else (int(subset[0]),
                                                   int(subset[1]))
        self.max_per_star = max_per_star
        self.star = data["star"][order].astype(np.int64)
        self.logP = data["logP"][order]
        self.q = data["q"][order]
        self.e = data["e_draw"][order]
        self.circ_comp = data["circ_comp"][order].astype(bool)
        self.det = data["det"][order].astype(bool)
        self.K1 = data["K1"][order]
        self.k1_floor = None if k1_floor is None else float(k1_floor)
        if self.k1_floor is not None:
            self.det = self.det & (self.K1 >= self.k1_floor)
        self.stars = stars.reset_index(drop=True)
        self.manifest = manifest
        self.n_stars = int(len(self.stars))
        idx = self.stars["star_index"].to_numpy()
        if not np.array_equal(idx, np.arange(self.n_stars)) and \
                not np.all(np.isin(np.unique(self.star), idx)):
            raise ValueError("injection records reference unknown stars")
        # Segment bounds per star (records are sorted by star index).
        self.star_ids = self.stars["star_index"].to_numpy()
        self.seg = np.searchsorted(self.star, self.star_ids)
        self.seg_end = np.searchsorted(self.star, self.star_ids, side="right")
        self.N = (self.seg_end - self.seg).astype(float)
        if np.any(self.N == 0):
            raise ValueError("stars without injections: %s"
                             % list(self.stars.loc[self.N == 0, "ID"]))

        b = manifest["bounds"]
        self.bounds = b
        self.p_circ = float(manifest["p_circ"])
        self.p_env = float(p_env)
        ref = manifest["reference"]
        spec = {"logP": {"alphas": ref["logP"]["alphas"],
                         "weights": ref["logP"]["weights"]},
                "q": {"alphas": ref["q"]["alphas"],
                      "weights": ref["q"]["weights"]},
                "e": {"alphas": ref["e"]["alphas"],
                      "weights": ref["e"]["weights"]},
                "w_circ": ref["w_circ"]}
        self.ref = ReferenceDensity(b, spec)
        self.w_circ = self.ref.w_circ

        P = 10.0 ** self.logP
        self.short = P < self.p_circ
        self.mid = (~self.short) & (P < self.p_env)
        self.long = P >= self.p_env
        self.lg_P = self.ref.g_logP.logpdf(self.logP)
        self.lg_q = self.ref.g_q.logpdf(self.q)
        ecc = ~self.circ_comp
        self.lg_e = np.full(self.e.shape, np.nan)
        self.lg_e[ecc] = self.ref.g_e.logpdf(self.e[ecc])

        # Singles: per-star false-positive rate.
        s_star = data["single_star"].astype(np.int64)
        s_det = data["single_det"].astype(bool)
        if subset is not None and s_star.size:
            # the same block of every star's singles, so the subsets stay
            # disjoint in beta_s as well
            o = np.argsort(s_star, kind="stable")
            ss = s_star[o]
            f0 = np.searchsorted(ss, ss, side="left")
            ns = np.searchsorted(ss, ss, side="right") - f0
            rk = np.arange(ss.size) - f0
            k, n = self.subset
            m = (rk >= (k * ns) // n) & (rk < ((k + 1) * ns) // n)
            s_star, s_det = ss[m], s_det[o][m]
        n_s = np.bincount(s_star, minlength=self.star_ids.max() + 1)
        n_fp = np.bincount(s_star, weights=s_det,
                           minlength=self.star_ids.max() + 1)
        n_s = n_s[self.star_ids].astype(float)
        n_fp = n_fp[self.star_ids].astype(float)
        self.n_single = n_s
        self.beta = np.where(n_s > 0, n_fp / np.maximum(n_s, 1), 0.0)

    # -- weight factors ----------------------------------------------------
    def wP(self, pi):
        return np.exp(powerlaw_logpdf(self.logP, pi, self.bounds["log_p_min"],
                                      self.bounds["log_p_max"]) - self.lg_P)

    def wq(self, kappa, q_min=None):
        """p_q/g_q; ``q_min`` > bounds restricts the model domain (qmin01)."""
        lo = self.bounds["q_min"] if q_min is None else float(q_min)
        return np.exp(powerlaw_logpdf(self.q, kappa, lo, self.bounds["q_max"])
                      - self.lg_q)

    def we_ecc(self, eta):
        """Eccentric-component e factor (0 for circular draws / short P)."""
        out = np.zeros(self.e.shape)
        m = (~self.circ_comp) & (~self.short)
        out[m] = np.exp(powerlaw_logpdf(self.e[m], eta, 1e-6,
                                        self.bounds["e_max"]) - self.lg_e[m]
                        ) / (1.0 - self.w_circ)
        return out

    def we_circ(self):
        """Circular-component e factor (0 for eccentric draws / short P)."""
        out = np.zeros(self.e.shape)
        m = self.circ_comp & (~self.short)
        out[m] = 1.0 / self.w_circ
        return out

    # -- per-star sums -------------------------------------------------------
    def _star_sum(self, v):
        """sum of v over each star's records, divided by N_s."""
        cs = np.concatenate([[0.0], np.cumsum(v)])
        return (cs[self.seg_end] - cs[self.seg]) / self.N

    _PARTS = ("short", "mid_c", "mid_e", "long_c", "long_e")

    def _parts(self, base, ec, ee):
        return {"short": self._star_sum(base * self.short),
                "mid_c": self._star_sum(base * self.mid * ec),
                "mid_e": self._star_sum(base * self.mid * ee),
                "long_c": self._star_sum(base * self.long * ec),
                "long_e": self._star_sum(base * self.long * ee)}

    def alpha_parts(self, pi, kappa, eta, cuts=(None,), q_min=None):
        """Per-star partial weight sums at one (pi, kappa, eta).

        Returns ``{"den": parts, "num": {cut: parts}}``. Each ``parts`` is a
        dict of arrays (n_stars,) keyed short, mid_c, mid_e, long_c, long_e.
        ``den`` sums all injections; ``num[cut]`` sums the detected ones,
        restricted to logP >= cut when cut is not None. ``combine`` turns
        numerator and denominator into a_s(c1, c2).
        """
        wpq = self.wP(pi) * self.wq(kappa, q_min=q_min)
        ec = self.we_circ()
        ee = self.we_ecc(eta)
        out = {"den": self._parts(wpq, ec, ee), "num": {}}
        for cut in cuts:
            base = wpq * self.det
            if cut is not None:
                base = base * (self.logP >= cut)
            out["num"][cut] = self._parts(base, ec, ee)
        return out

    @staticmethod
    def _lin(parts, c1, c2):
        return (parts["short"] + c1 * parts["mid_c"] + (1 - c1) * parts["mid_e"]
                + c2 * parts["long_c"] + (1 - c2) * parts["long_e"])

    @classmethod
    def combine(cls, parts, c1, c2, cut=None):
        """a_s(c1, c2) = numerator(c) / denominator(c) (self-normalized)."""
        den = cls._lin(parts["den"], c1, c2)
        num = cls._lin(parts["num"][cut], c1, c2)
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)

    def weights(self, pi, kappa, eta, c1, c2, q_min=None):
        """Full per-injection weights w_j (theta, c)."""
        ec = self.we_circ()
        ee = self.we_ecc(eta)
        we = (self.short + self.mid * (c1 * ec + (1 - c1) * ee)
              + self.long * (c2 * ec + (1 - c2) * ee))
        return self.wP(pi) * self.wq(kappa, q_min=q_min) * we

    def alpha(self, pi, kappa, eta, c1, c2, cut=None, q_min=None):
        """a_s(theta, c) for every star (optionally logP >= cut only)."""
        parts = self.alpha_parts(pi, kappa, eta, cuts=(cut,), q_min=q_min)
        return self.combine(parts, c1, c2, cut=cut)

    def alpha_se(self, pi, kappa, eta, c1, c2, cut=None, q_min=None):
        """Delta-method standard error of the self-normalized a_s."""
        w = self.weights(pi, kappa, eta, c1, c2, q_min=q_min)
        d = self.det if cut is None else self.det & (self.logP >= cut)
        a = self.alpha(pi, kappa, eta, c1, c2, cut=cut, q_min=q_min)
        a_rec = np.repeat(a, (self.seg_end - self.seg).astype(int))
        sw = self._star_sum(w) * self.N
        s2 = self._star_sum(w * w * (d - a_rec) ** 2) * self.N
        return np.sqrt(s2) / np.maximum(sw, 1e-300)

    def n_eff(self, pi, kappa, eta, c1=0.0, c2=0.0):
        """Kish effective sample size of the weights, per star."""
        w = self.weights(pi, kappa, eta, c1, c2)
        s1 = self._star_sum(w) * self.N
        s2 = self._star_sum(w * w) * self.N
        return np.where(s2 > 0, s1 * s1 / np.maximum(s2, 1e-300), 0.0)

    # -- grid tensors --------------------------------------------------------
    def alpha_tensors(self, pi_grid, kappa_grid, eta_grid, cut, q_min=None,
                      progress=None, groups=("den", "num_all", "num_cut"),
                      logp_range=None, lim_logp=None):
        """Per-star weight sums on a (pi, kappa, eta) grid, by tensor
        contraction (a few BLAS calls per star instead of a node loop).

        Returns ``{group: {part: array}}`` for group in ``den`` (all
        injections), ``num_all`` (detected) and ``num_cut`` (detected with
        logP >= cut). Parts short/mid_c/long_c have shape
        (n_stars, n_pi, n_kappa); mid_e/long_e add a trailing n_eta axis.
        a_s = combine(num)/combine(den) as in ``combine``.

        ``logp_range = (lo, hi)`` (either may be None) restricts every sum
        to the injections with lo <= logP < hi. The sums are additive over
        disjoint ranges, which is what a model whose q law differs between
        period ranges needs (``two_kappa``): one tensor set per range, each
        on its own kappa grid.

        Group ``num_lim`` (censored period lower limits) sums the detected
        injections with logP >= lim_logp[s], a per-star limit (array of
        n_stars, NaN where a star has none; those stars fall back on the
        cut, so for them num_lim equals num_cut).
        """
        if "num_lim" in groups and lim_logp is None:
            raise ValueError("group num_lim needs lim_logp")
        pis = np.asarray(pi_grid, dtype=float)
        kas = np.asarray(kappa_grid, dtype=float)
        ets = np.asarray(eta_grid, dtype=float)
        b = self.bounds
        q_lo = b["q_min"] if q_min is None else float(q_min)
        s3 = (self.n_stars, pis.size, kas.size)
        s4 = s3 + (ets.size,)
        out = {g: {"short": np.zeros(s3), "mid_c": np.zeros(s3),
                   "long_c": np.zeros(s3), "mid_e": np.zeros(s4),
                   "long_e": np.zeros(s4)}
               for g in groups}
        for s in range(self.n_stars):
            sl = slice(self.seg[s], self.seg_end[s])
            if logp_range is not None:
                lp_s = self.logP[sl]
                keep = np.ones(lp_s.shape, dtype=bool)
                if logp_range[0] is not None:
                    keep &= lp_s >= float(logp_range[0])
                if logp_range[1] is not None:
                    keep &= lp_s < float(logp_range[1])
                sl = np.arange(sl.start, sl.stop)[keep]
                n = sl.size
            else:
                n = sl.stop - sl.start
            A = np.exp(powerlaw_logpdf(self.logP[sl][None, :], pis[:, None],
                                       b["log_p_min"], b["log_p_max"])
                       - self.lg_P[sl][None, :])
            B = np.exp(powerlaw_logpdf(self.q[sl][None, :], kas[:, None],
                                       q_lo, b["q_max"])
                       - self.lg_q[sl][None, :])
            short = self.short[sl]
            ecc = (~self.circ_comp[sl]) & (~short)
            E = np.zeros((ets.size, n))
            if ecc.any():
                E[:, ecc] = np.exp(
                    powerlaw_logpdf(self.e[sl][ecc][None, :], ets[:, None],
                                    1e-6, b["e_max"])
                    - self.lg_e[sl][ecc][None, :]) / (1.0 - self.w_circ)
            C = np.where(self.circ_comp[sl] & ~short, 1.0 / self.w_circ, 0.0)
            mid, lng = self.mid[sl], self.long[sl]
            det = self.det[sl]
            above = (self.logP[sl] >= cut) if cut is not None else None
            masks = {"den": None, "num_all": det,
                     "den_cut": above,
                     "num_cut": (det & above) if above is not None else None}
            if "num_lim" in groups:
                lim = float(lim_logp[s])
                lim = cut if not np.isfinite(lim) else max(lim, cut)
                masks["num_lim"] = det & (self.logP[sl] >= lim)
            for g in groups:
                m = masks[g]
                Am = A if m is None else A * m
                o = out[g]
                o["short"][s] = (Am * short) @ B.T
                o["mid_c"][s] = (Am * (mid * C)) @ B.T
                o["long_c"][s] = (Am * (lng * C)) @ B.T
                Amid, Alng = Am * mid, Am * lng
                for li in range(ets.size):
                    o["mid_e"][s, :, :, li] = (Amid * E[li]) @ B.T
                    o["long_e"][s, :, :, li] = (Alng * E[li]) @ B.T
            if progress is not None:
                progress(s)
        return out


# ---------------------------------------------------------------------------
# Observed data
# ---------------------------------------------------------------------------

CAT_ND, CAT_DX, CAT_DOB, CAT_DOA, CAT_DOANY = 0, 1, 2, 3, 4


class ObservedData:
    """Per-star observed outcomes, aligned with the injection set's stars.

    ``table`` has the obs_catalog schema: star_id, outcome (ND/DX/DO),
    logP, e, e_is_upper_limit, K1. For LS rows, ``e`` holds the reported
    upper limit. Every injection-sample star must appear exactly once.
    Optional columns sigma_K1 and sigma_e give measurement errors.
    """

    def __init__(self, table, inj_stars, cut, p_circ, p_env, e_max,
                 below_pcirc_ecc="raise", k1_floor=None, k1_unknown="raise",
                 below_shape=None, censor_plim=False):
        """``k1_floor``: amplitude floor of the survey (see
        ``InjectionSet``). ``k1_unknown`` says what to do, under a floor,
        with detected stars that have no K1: ``"raise"`` or
        ``"above_floor"`` (score them as detections above the floor; their
        IDs are kept in ``k1_floor_unknown``).

        ``below_shape`` (split model): ``None`` scores below-cut orbits as
        plain detections; ``"k1"`` / ``"k1e"`` add their amplitude density /
        amplitude and eccentricity densities (module docstring). The rows
        concerned are kept in ``self.below`` with the ``dx_*`` attribute
        names, so ``shape_precompute`` and ``_dx_lnp`` apply to them."""
        ids = [str(x) for x in inj_stars["ID"]]
        pos = {sid: i for i, sid in enumerate(ids)}
        tab = table.copy()
        tab["star_id"] = tab["star_id"].astype(str)
        if tab["star_id"].duplicated().any():
            raise ValueError("duplicate star_id rows in the observed table")
        missing = sorted(set(ids) - set(tab["star_id"]))
        extra = sorted(set(tab["star_id"]) - set(ids))
        if missing or extra:
            raise ValueError("observed table vs injection sample mismatch: "
                             "missing %s, extra %s" % (missing, extra))
        tab["pos"] = tab["star_id"].map(pos)
        tab = tab.sort_values("pos").reset_index(drop=True)
        if not set(tab["outcome"]) <= {"ND", "DX", "DO"}:
            raise ValueError("unknown outcomes %s" % set(tab["outcome"]))
        self.n_stars = len(ids)
        self.is_D = (tab["outcome"] != "ND").to_numpy()
        self.k1_floor = None if k1_floor is None else float(k1_floor)
        self.k1_floor_unknown = []
        if k1_floor is not None:
            # amplitude-floor survey (InjectionSet(k1_floor=...)): a detected
            # orbit below the floor would be a detection the model cannot
            # produce; such stars must enter as ND
            if k1_unknown not in ("raise", "above_floor"):
                raise ValueError("k1_unknown must be 'raise' or "
                                 "'above_floor', got %r" % (k1_unknown,))
            k1_tab = np.asarray(tab["K1"], dtype=float)
            low = self.is_D & np.isfinite(k1_tab) & (k1_tab < float(k1_floor))
            if low.any():
                raise ValueError(
                    "detected orbits with K1 below the amplitude floor of "
                    "%g km/s: %s; reclassify them as ND"
                    % (float(k1_floor), list(tab.loc[low, "star_id"])))
            # a detected star without a K1 is scored with the floored
            # detection probability, i.e. as if its K1 were known to be
            # above the floor: that has to be asked for
            unk = self.is_D & ~np.isfinite(k1_tab)
            self.k1_floor_unknown = [str(x) for x in tab.loc[unk, "star_id"]]
            if unk.any() and k1_unknown != "above_floor":
                raise ValueError(
                    "detected stars without a K1 under the amplitude floor "
                    "of %g km/s: %s; they would be scored as detections "
                    "above the floor. Pass k1_unknown='above_floor' to "
                    "accept this, or enter them as ND"
                    % (float(k1_floor), self.k1_floor_unknown))
        # Per-star outcome categories for the split model (see
        # node_table_split): DX, DO below the cut, DO known to lie above the
        # cut (P lower limit), DO with no usable period (no solution,
        # excluded, false positive in mocks).
        out = tab["outcome"].to_numpy()
        rs = (tab["reason"].fillna("").astype(str).to_numpy()
              if "reason" in tab.columns else np.full(len(tab), ""))
        cat = np.full(len(tab), CAT_ND)
        cat[out == "DX"] = CAT_DX
        do = out == "DO"
        cat[do & (rs == "below_cut")] = CAT_DOB
        cat[do & (rs == "P_lower_limit")] = CAT_DOA
        cat[do & ~np.isin(rs, ["below_cut", "P_lower_limit"])] = CAT_DOANY
        self.cat = cat
        self.cut = float(cut)
        # censor_plim: a period-lower-limit star (CAT_DOA) contributes
        # f_gt P(det, logP >= log P_lim) instead of f_gt P(det, logP >= cut),
        # so its limit informs pi (TS5-015). lim_logp is NaN elsewhere.
        self.censor_plim = bool(censor_plim)
        self.lim_logp = np.full(len(tab), np.nan)
        if self.censor_plim:
            if "P_lower_limit" not in tab.columns:
                raise ValueError("censor_plim needs a P_lower_limit column")
            m = cat == CAT_DOA
            plim = np.asarray(tab["P_lower_limit"], dtype=float)[m]
            if not np.all(np.isfinite(plim) & (plim > 0)):
                raise ValueError("censor_plim: missing P_lower_limit for %s"
                                 % list(tab.loc[m, "star_id"]))
            lim = np.log10(plim)
            if np.any(lim < self.cut - 1e-12):
                raise ValueError("censor_plim: a period lower limit lies "
                                 "below the cut")
            self.lim_logp[m] = lim
        dx = tab[tab["outcome"] == "DX"]
        if (dx["logP"] < cut - 1e-12).any():
            raise ValueError("DX rows below the cut")
        self.dx_pos = dx["pos"].to_numpy(dtype=int)
        self.dx_ids = dx["star_id"].to_numpy()
        self.dx_logP = dx["logP"].to_numpy(dtype=float)
        P = 10.0 ** self.dx_logP
        self.dx_ls = dx["e_is_upper_limit"].astype(bool).to_numpy()
        e = dx["e"].to_numpy(dtype=float)
        self.dx_e = np.where(self.dx_ls, 0.0, e)
        self.dx_elim = np.where(self.dx_ls, e, 0.0)
        self.dx_K1 = dx["K1"].to_numpy(dtype=float)
        if np.any((~self.dx_ls) & ((self.dx_e <= 0) | (self.dx_e > e_max))):
            raise ValueError("eccentric DX rows with e outside (0, e_max]")
        # period bins: 0 = below p_circ (e forced 0), 1 = mid, 2 = long
        self.dx_bin = np.where(P < p_circ, 0, np.where(P < p_env, 1, 2))
        conflict = (self.dx_bin == 0) & ~self.dx_ls
        self.n_below_pcirc_ecc = int(conflict.sum())
        if conflict.any():
            if below_pcirc_ecc != "as_circular":
                raise ValueError(
                    "eccentric orbit(s) %s below p_circ are impossible in the "
                    "model (e is forced to 0 there); pass "
                    "below_pcirc_ecc='as_circular' to treat them as circular"
                    % list(dx.loc[conflict, "star_id"]))
            # treat as circular: e information dropped, u at e = 0
            self.dx_ls = self.dx_ls | conflict
            self.dx_e = np.where(conflict, 0.0, self.dx_e)
            self.dx_elim = np.where(conflict, 0.0, self.dx_elim)
        self.dx_u = u_of(self.dx_K1, P, self.dx_e)
        self.dx_u0 = u_of(self.dx_K1, P, 0.0)
        self.sigma_u = (dx["sigma_K1"].to_numpy(float)
                        / (self.dx_K1 * np.log(10))
                        if "sigma_K1" in dx.columns else None)
        self.sigma_e = (np.where(self.dx_ls, np.nan,
                                 dx["sigma_e"].to_numpy(float))
                        if "sigma_e" in dx.columns else None)
        if below_shape not in (None, "k1", "k1e"):
            raise ValueError("below_shape must be None, 'k1' or 'k1e', got %r"
                             % (below_shape,))
        self.below_shape = below_shape
        self.below = None
        if below_shape is not None:
            self.below = self._below_rows(tab, cat, p_circ, p_env, e_max,
                                          below_shape, below_pcirc_ecc)
        self.table = tab

    @staticmethod
    def _below_rows(tab, cat, p_circ, p_env, e_max, mode, below_pcirc_ecc):
        """Below-cut orbits with a K1, as a namespace with the ``dx_*``
        attributes of the orbits above the cut (plus ``used``, the per-star
        mask of these rows). ``mode`` "k1": every row is treated as circular
        in period class 0, so its term is p_s(u0 | kappa) alone. "k1e": e, the
        LS flag and the period class are those of the row, with the
        ``below_pcirc_ecc`` rule for significant e below p_circ."""
        k1 = np.asarray(tab["K1"], dtype=float)
        lp = np.asarray(tab["logP"], dtype=float)
        used = (cat == CAT_DOB) & np.isfinite(k1) & (k1 > 0) & np.isfinite(lp)
        db = tab[used]
        b = types.SimpleNamespace(used=used, mode=mode)
        b.dx_pos = db["pos"].to_numpy(dtype=int)
        b.dx_ids = db["star_id"].to_numpy()
        b.dx_logP = db["logP"].to_numpy(dtype=float)
        b.dx_K1 = db["K1"].to_numpy(dtype=float)
        P = 10.0 ** b.dx_logP
        n = b.dx_pos.size
        b.n_below_pcirc_ecc = 0
        if mode == "k1":
            b.dx_ls = np.ones(n, dtype=bool)
            b.dx_e = np.zeros(n)
            b.dx_elim = np.zeros(n)
            b.dx_bin = np.zeros(n, dtype=int)
            b.sigma_e = None
        else:
            if db["e"].isna().any() or db["e_is_upper_limit"].isna().any():
                raise ValueError(
                    "below_shape='k1e' needs e and e_is_upper_limit for the "
                    "below-cut orbits %s" % list(
                        db.loc[db["e"].isna()
                               | db["e_is_upper_limit"].isna(), "star_id"]))
            b.dx_ls = db["e_is_upper_limit"].astype(bool).to_numpy()
            e = db["e"].to_numpy(dtype=float)
            b.dx_e = np.where(b.dx_ls, 0.0, e)
            b.dx_elim = np.where(b.dx_ls, e, 0.0)
            if np.any((~b.dx_ls) & ((b.dx_e <= 0) | (b.dx_e > e_max))):
                raise ValueError("eccentric below-cut rows with e outside "
                                 "(0, e_max]")
            b.dx_bin = np.where(P < p_circ, 0, np.where(P < p_env, 1, 2))
            conflict = (b.dx_bin == 0) & ~b.dx_ls
            b.n_below_pcirc_ecc = int(conflict.sum())
            if conflict.any():
                if below_pcirc_ecc != "as_circular":
                    raise ValueError(
                        "eccentric below-cut orbit(s) %s below p_circ are "
                        "impossible in the model (e is forced to 0 there); "
                        "pass below_pcirc_ecc='as_circular' to treat them "
                        "as circular" % list(db.loc[conflict, "star_id"]))
                b.dx_ls = b.dx_ls | conflict
                b.dx_e = np.where(conflict, 0.0, b.dx_e)
                b.dx_elim = np.where(conflict, 0.0, b.dx_elim)
            b.sigma_e = (np.where(b.dx_ls, np.nan,
                                  db["sigma_e"].to_numpy(float))
                         if "sigma_e" in db.columns else None)
        b.dx_u = u_of(b.dx_K1, P, b.dx_e)
        b.dx_u0 = u_of(b.dx_K1, P, 0.0)
        b.sigma_u = (db["sigma_K1"].to_numpy(float) / (b.dx_K1 * np.log(10))
                     if "sigma_K1" in db.columns else None)
        return b

    def summary(self):
        out = {"n_stars": int(self.n_stars), "D": int(self.is_D.sum()),
               "ND": int((~self.is_D).sum()), "DX": int(self.dx_pos.size),
               "DX_LS": int(self.dx_ls.sum()),
               "DX_bins": np.bincount(self.dx_bin, minlength=3).tolist(),
               "below_pcirc_ecc_as_circular": self.n_below_pcirc_ecc}
        if self.k1_floor is not None:
            out["k1_unknown_scored_above_floor"] = list(self.k1_floor_unknown)
        if self.below is not None:
            b = self.below
            out["below_shape"] = {
                "mode": b.mode, "n": int(b.dx_pos.size),
                "LS": int(b.dx_ls.sum()),
                "bins": np.bincount(b.dx_bin, minlength=3).tolist(),
                "below_pcirc_ecc_as_circular": int(b.n_below_pcirc_ecc)}
        return out


def shape_precompute(obs, inj, pi_grid, kappa_grid, eta_grid, q_min=None,
                     n_ebins=400, use_errors=False):
    """Grid tables of the DX observed-density factors.

    Returns dict with
      lnpP  (n_pi, n_dx)          ln p_P(logP_d | pi)
      lnpe  (n_eta, n_dx)         ln p_e(e_d | eta)   (0 for LS rows)
      pu    (n_kappa, n_dx)       p_s(u_d | kappa)    (u0 for LS rows)
      I     (n_eta, n_kappa, n_dx) censored integral (0 for eccentric rows)
    """
    b = inj.bounds
    q_lo = b["q_min"] if q_min is None else float(q_min)
    pis = np.asarray(pi_grid, float)
    kas = np.asarray(kappa_grid, float)
    ets = np.asarray(eta_grid, float)
    m1 = inj.stars["Mspec"].to_numpy(float)[obs.dx_pos]
    U = UDensity(m1, kas, q_lo, b["q_max"])
    nd = obs.dx_pos.size
    lnpP = powerlaw_logpdf(obs.dx_logP[None, :], pis[:, None],
                           b["log_p_min"], b["log_p_max"])
    lnpe = np.zeros((ets.size, nd))
    ecc = ~obs.dx_ls
    lnpe[:, ecc] = powerlaw_logpdf(obs.dx_e[ecc][None, :], ets[:, None],
                                   1e-6, b["e_max"])
    if use_errors and obs.sigma_e is not None:
        # p_e convolved with a Gaussian e error: integral of p_e(e'|eta)
        # N(e_d; e', sigma_e) de', on CDF-weighted geometric e bins.
        edges = np.geomspace(1e-6, b["e_max"], 2001)
        mid = np.sqrt(edges[1:] * edges[:-1])
        dF = (powerlaw_cdf(edges[1:][None, :], ets[:, None], 1e-6,
                           b["e_max"])
              - powerlaw_cdf(edges[:-1][None, :], ets[:, None], 1e-6,
                             b["e_max"]))            # (n_eta, n_bins)
        for d in np.where(ecc)[0]:
            sig = obs.sigma_e[d]
            if not (np.isfinite(sig) and sig > 0):
                continue
            kern = (np.exp(-0.5 * ((obs.dx_e[d] - mid) / sig) ** 2)
                    / (np.sqrt(2 * np.pi) * sig))
            with np.errstate(divide="ignore"):
                lnpe[:, d] = np.log(dF @ kern)
    pu = np.zeros((kas.size, nd))
    I = np.zeros((ets.size, kas.size, nd))
    tail, outside = [], []
    for d in range(nd):
        u_d = obs.dx_u0[d] if obs.dx_ls[d] else obs.dx_u[d]
        for ki in range(kas.size):
            if use_errors and obs.sigma_u is not None:
                pu[ki, d] = U.smoothed_pdf(d, ki, u_d, obs.sigma_u[d])
            else:
                pu[ki, d] = U.pdf(d, ki, u_d)
        # a u beyond the model's support would zero the whole posterior;
        # one at the round-off floor carries no kappa information (see
        # k1_density.REL_FLOOR) and is reported
        if np.all(pu[:, d] <= 0):
            outside.append(str(obs.dx_ids[d]))
        elif all(U.at_floor(d, ki, u_d) for ki in range(kas.size)):
            tail.append(str(obs.dx_ids[d]))
        # censored integral of LS-circular rows: int_0^{e_lim} p_e p_u de
        if obs.dx_ls[d] and obs.dx_elim[d] > 1e-6:
            edges = np.geomspace(1e-6, obs.dx_elim[d], n_ebins + 1)
            mid = np.sqrt(edges[1:] * edges[:-1])
            dF = (powerlaw_cdf(edges[1:][None, :], ets[:, None], 1e-6,
                               b["e_max"])
                  - powerlaw_cdf(edges[:-1][None, :], ets[:, None], 1e-6,
                                 b["e_max"]))           # (n_eta, n_ebins)
            P_d = 10.0 ** obs.dx_logP[d]
            u_e = u_of(obs.dx_K1[d], P_d, mid)
            for ki in range(kas.size):
                pk = U.pdf(d, ki, u_e)                  # (n_ebins,)
                I[:, ki, d] = dF @ pk
    if outside:
        raise ValueError("observed K1 outside the model's support (u above "
                         "the M1 = %.0f Msun, q = 1, i = 90 deg limit) for %s"
                         % (M1_CLIP[1], outside))
    if tail:
        warnings.warn("u-density at the round-off floor for %s: these "
                      "systems' K1 lie in the far tail of the model "
                      "(M1 scatter) and contribute no kappa information"
                      % tail, stacklevel=2)
    n_ls = int(np.sum(obs.dx_ls & (obs.dx_elim > 1e-6)))
    if n_ls and not np.any(I > 0):
        raise RuntimeError("censored integrals are all zero for %d LS rows"
                           % n_ls)
    out = {"lnpP": lnpP, "lnpe": lnpe, "pu": pu, "I": I}
    below = getattr(obs, "below", None)
    if below is not None and below.dx_pos.size:
        # the same tables for the below-cut orbits (ObservedData.below);
        # their period density is not part of the likelihood, so lnpP := 0
        sb = shape_precompute(below, inj, pi_grid, kappa_grid, eta_grid,
                              q_min=q_min, n_ebins=n_ebins,
                              use_errors=use_errors)
        sb["lnpP"] = np.zeros_like(sb["lnpP"])
        out["below"] = sb
    return out


def _lin(parts, s, i, k, l, c1, c2):
    """combine() on tensor slices, broadcast over (c1, c2, stars)."""
    sh = parts["short"][:, i, k]
    return (sh[None, None, :]
            + c1[:, None, None] * parts["mid_c"][None, None, :, i, k]
            + (1 - c1)[:, None, None] * parts["mid_e"][None, None, :, i, k, l]
            + c2[None, :, None] * parts["long_c"][None, None, :, i, k]
            + (1 - c2)[None, :, None] * parts["long_e"][None, None, :, i, k, l])


def trapz_logweights(grid):
    g = np.asarray(grid, float)
    if g.size == 1:
        return np.zeros(1)
    w = np.zeros(g.size)
    d = np.diff(g)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return np.log(w / w.sum())


def node_table(T, shape, obs, beta, i, k, l, f_grid, c_grid,
               c2_grid=None, count=True):
    """Unmarginalized ln L(f, c1, c2) at grid node (pi_i, kappa_k, eta_l).

    ``c_grid`` holds the c1 nodes and ``c2_grid`` the c2 nodes (default:
    same as c1). ``count=False`` drops the per-star Bernoulli count terms,
    giving a shape-only likelihood (analogue of the CvM binom_none
    variant). Returns (n_f, n_c1, n_c2), without prior weights.
    """
    c = np.asarray(c_grid, float)
    c2 = c if c2_grid is None else np.asarray(c2_grid, float)
    D = _lin(T["den"], None, i, k, l, c, c2)            # (nc1, nc2, ns)
    A_all = _lin(T["num_all"], None, i, k, l, c, c2) / D
    A_cut = _lin(T["num_cut"], None, i, k, l, c, c2) / D
    f = np.asarray(f_grid, float)[:, None]
    isD = obs.is_D
    cnt = np.zeros((f.shape[0], c.size, c2.size))
    with np.errstate(divide="ignore", invalid="ignore"):
        if count:
            # one (c1, c2) pair at a time: cache-resident (n_f, n_star)
            # temporaries (see node_table_split)
            bD, bN = beta[isD][None, :], beta[~isD][None, :]
            for i1 in range(c.size):
                for i2 in range(c2.size):
                    a = A_all[i1, i2]
                    pD = f * a[isD][None, :] + (1.0 - f) * bD
                    pN = f * a[~isD][None, :] + (1.0 - f) * bN
                    cnt[:, i1, i2] = (np.log(pD).sum(axis=-1)
                                      + np.log1p(-pN).sum(axis=-1))
    # shape term (no f dependence)
    pos, bins = obs.dx_pos, obs.dx_bin
    cb = np.stack([np.ones((c.size, c2.size)),             # below p_circ
                   np.broadcast_to(c[:, None], (c.size, c2.size)),
                   np.broadcast_to(c2[None, :], (c.size, c2.size))])
    cd = cb[bins]                                          # (ndx, nc, nc)
    lnpP = shape["lnpP"][i]                                # (ndx,)
    pu = shape["pu"][k]
    with np.errstate(divide="ignore", invalid="ignore"):
        ecc_term = (np.log1p(-cd) + (lnpP + shape["lnpe"][l]
                                     + np.log(pu))[:, None, None])
        ls_term = lnpP[:, None, None] + np.log(
            cd * pu[:, None, None]
            + (1.0 - cd) * shape["I"][l, k][:, None, None])
        lnp = np.where(obs.dx_ls[:, None, None], ls_term, ecc_term)
        lnacut = np.log(np.moveaxis(A_cut[:, :, pos], -1, 0))
    shape_c = (lnp - lnacut).sum(axis=0)                  # (nc1, nc2)
    return cnt + shape_c[None]                            # (nf, nc1, nc2)


def node_loglike(T, shape, obs, beta, i, k, l, f_grid, c_grid,
                 return_c=False, c2_grid=None, count=True):
    """ln L(f) at grid node (pi_i, kappa_k, eta_l), c1 and c2 marginalized
    with uniform priors (trapezoid rule on ``c_grid``).

    Returns ``lnL_f`` (n_f,); with ``return_c`` also the (n_c, n_c) table
    ln sum_f exp(lnL(f, c1, c2)) for the c posterior.
    """
    c = np.asarray(c_grid, float)
    c2 = c if c2_grid is None else np.asarray(c2_grid, float)
    tot = node_table(T, shape, obs, beta, i, k, l, f_grid, c_grid,
                     c2_grid=c2_grid, count=count)
    lw = trapz_logweights(c)
    lw2 = trapz_logweights(c2)
    tot_w = tot + lw[None, :, None] + lw2[None, None, :]
    m = np.max(tot_w, axis=(1, 2), keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    lnL_f = (m + np.log(np.sum(np.exp(tot_w - m), axis=(1, 2),
                               keepdims=True)))[:, 0, 0]
    if not return_c:
        return lnL_f
    mc = np.max(tot, axis=0, keepdims=True)
    mc = np.where(np.isfinite(mc), mc, 0.0)
    # c1 = 1 or c2 = 1 cells are -inf (eccentric orbits impossible): log(0)
    with np.errstate(divide="ignore"):
        lnc = (mc + np.log(np.sum(np.exp(tot - mc), axis=0,
                                  keepdims=True)))[0]
    return lnL_f, lnc


# ---------------------------------------------------------------------------
# Split model: power law above logP*, free binary fraction below it
# ---------------------------------------------------------------------------
#
# Each star is, independently,
#   a binary with logP >= logP* with probability f_gt  (power law in logP
#     truncated to [logP*, log_p_max], index pi),
#   a binary with logP <  logP* with probability f_lt  (same functional
#     form truncated to [log_p_min, logP*]; only its normalization is free),
#   single otherwise (falsely detected with rate beta_s).
# q, e, the circular fractions and the detector are as in the full-range
# model. The pure power law is the special case f_gt : f_lt = F_gt : F_lt.
# The categorical per-star likelihood is
#   ND        1 - f_gt a_gt - f_lt a_lt - (1 - f_gt - f_lt) beta
#   DX        f_gt p_s(x | theta, c) / F_gt          (P_det(x) cancels)
#   DO below  f_lt a_lt
#             (with below_shape: f_lt p_s(e, u | theta, c) for the rows that
#              have a K1; no period density, P_det(x) cancels)
#   DO above  f_gt a_gt                              (P lower limits)
#             (with censor_plim: f_gt P(det, logP >= log P_lim)/F_gt, so
#              the limit itself informs pi)
#   DO any    f_gt a_gt + f_lt a_lt + (1 - f_gt - f_lt) beta
# where a_gt = P(det, logP >= cut)/F_gt and a_lt = P(det, logP < cut)/F_lt
# are the within-component detection probabilities of star s. The prior on
# (f_gt, f_lt) is flat on the simplex f_gt + f_lt <= 1.


def _dx_lnp(shape, obs, i, k, l, c1, c2):
    """ln p_s(x_d | theta, c) for every DX system, shape (ndx, n1, n2)."""
    n1, n2 = c1.size, c2.size
    cb = np.stack([np.ones((n1, n2)),
                   np.broadcast_to(c1[:, None], (n1, n2)),
                   np.broadcast_to(c2[None, :], (n1, n2))])
    cd = cb[obs.dx_bin]
    lnpP = shape["lnpP"][i]
    pu = shape["pu"][k]
    with np.errstate(divide="ignore", invalid="ignore"):
        ecc_term = (np.log1p(-cd) + (lnpP + shape["lnpe"][l]
                                     + np.log(pu))[:, None, None])
        ls_term = lnpP[:, None, None] + np.log(
            cd * pu[:, None, None]
            + (1.0 - cd) * shape["I"][l, k][:, None, None])
    return np.where(obs.dx_ls[:, None, None], ls_term, ecc_term)


def node_table_split(T, shape, obs, beta, i, k, l, fg, fl, c1, c2, F_gt):
    """ln L(f_gt, f_lt, c1, c2) at grid node (pi_i, kappa_k, eta_l).

    Returns (n_gt, n_lt, n_c1, n_c2); cells with f_gt + f_lt > 1 are -inf.
    """
    fg = np.asarray(fg, float)
    fl = np.asarray(fl, float)
    c1 = np.asarray(c1, float)
    c2 = np.asarray(c2, float)
    F_gt = float(F_gt)
    F_lt = 1.0 - F_gt
    # Within-component detection probabilities, self-normalized on each
    # side of the cut with the same weights (so both lie in [0, 1] exactly;
    # dividing by the analytic F_gt instead lets MC noise push them > 1).
    D = _lin(T["den"], None, i, k, l, c1, c2)              # (n1, n2, ns)
    Dg = _lin(T["den_cut"], None, i, k, l, c1, c2)
    Ng = _lin(T["num_cut"], None, i, k, l, c1, c2)
    Na = _lin(T["num_all"], None, i, k, l, c1, c2)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_gt = np.where(Dg > 0, Ng / np.where(Dg > 0, Dg, 1.0), 0.0)
        Dl = D - Dg
        a_lt = np.where(Dl > 0, np.clip(Na - Ng, 0.0, None)
                        / np.where(Dl > 0, Dl, 1.0), 0.0)
    G = fg[:, None, None]
    L = fl[None, :, None]
    cat = obs.cat
    shp = (fg.size, fl.size, c1.size, c2.size)
    tot = np.zeros(shp)
    m_nd, m_any = cat == CAT_ND, cat == CAT_DOANY
    with np.errstate(divide="ignore", invalid="ignore"):
        # Count terms, one (c1, c2) pair at a time: the (n_gt, n_lt, n_star)
        # temporaries stay cache-resident, which keeps the node loop from
        # being memory-bound when many workers run (same arithmetic and
        # summation order as the broadcast form, so results are identical).
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
        sb = shape.get("below")
        if sb is not None:
            # below_shape: these rows contribute f_lt p_s(e, u | theta, c)
            # (no period density, no a_lt); the others stay f_lt a_lt
            m = m & ~obs.below.used
            tot += (obs.below.dx_pos.size * np.log(fl)[None, :, None, None]
                    + _dx_lnp(sb, obs.below, i, k, l, c1, c2).sum(
                        axis=0)[None, None])
        if m.any():
            tot += (m.sum() * np.log(fl)[None, :, None, None]
                    + np.log(a_lt[..., m]).sum(axis=-1)[None, None])
        m = cat == CAT_DOA
        if m.any():
            a_doa = a_gt
            if getattr(obs, "censor_plim", False):
                # censored limit: P(det, logP >= log P_lim) within the
                # component above the cut (same normalisation as a_gt)
                Nl = _lin(T["num_lim"], None, i, k, l, c1, c2)
                a_doa = np.where(Dg > 0, Nl / np.where(Dg > 0, Dg, 1.0), 0.0)
            tot += (m.sum() * np.log(fg)[:, None, None, None]
                    + np.log(a_doa[..., m]).sum(axis=-1)[None, None])
        n_dx = obs.dx_pos.size
        if n_dx:
            tot += (n_dx * (np.log(fg)[:, None, None, None] - np.log(F_gt))
                    + _dx_lnp(shape, obs, i, k, l, c1, c2).sum(
                        axis=0)[None, None])
    simplex = (fg[:, None] + fl[None, :]) <= 1.0 + 1e-12
    return np.where(simplex[:, :, None, None], tot, -np.inf)


def _lse(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis,
                                            keepdims=True)), axis=axis)


def node_split_outputs(tot, fg, fl, c1, c2, n_ft, F_gt, rho_edges):
    """Marginals of one node's split table.

    Returns (ln_ft (n_ft,), ln_fg (n_gt,), ln_fgfl (n_gt, n_lt),
    ln_rho (len(rho_edges)+1,), ln_c (n1, n2)). f_tot lives on the lattice
    fg[0] + (g + l) * d (fg and fl share spacing d, fl[0] = 0). The rho
    array has an underflow bin at index 0 for rho = 0 (f_lt = 0) and
    ``rho_edges`` bins after it.
    """
    lw1 = trapz_logweights(c1)
    lw2 = trapz_logweights(c2)
    t = tot + lw1[None, None, :, None] + lw2[None, None, None, :]
    fgfl = _lse(_lse(t, 3), 2)                             # (ng, nl)
    ln_c = _lse(_lse(tot, 0), 0)                           # (n1, n2)
    idx = np.add.outer(np.arange(fg.size), np.arange(fl.size))
    ln_ft = np.full(n_ft, -np.inf)
    ok = idx < n_ft
    np.logaddexp.at(ln_ft, idx[ok], fgfl[ok])
    ln_fg = _lse(fgfl + trapz_logweights(fl)[None, :], 1)
    F_lt = 1.0 - F_gt
    with np.errstate(divide="ignore"):
        lrho = (np.log10(np.maximum(fl, 0.0))[None, :] - np.log10(F_lt)
                - np.log10(fg)[:, None] + np.log10(F_gt))
    b = np.where(np.isfinite(lrho),
                 np.searchsorted(rho_edges, lrho) , 0)
    b = np.clip(b, 0, len(rho_edges))
    ln_rho = np.full(len(rho_edges) + 1, -np.inf)
    np.logaddexp.at(ln_rho, b.ravel(), fgfl.ravel())
    return ln_ft, ln_fg, fgfl, ln_rho, ln_c
