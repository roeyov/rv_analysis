"""Per-grid-point scorer — pure, picklable, called from forward + resume.

``_compute_scores`` returns the dict of KS/AD/CvM p-values, the
Wasserstein log-GMF, the binomial pmf, and every scalar that ends up in
the per-row CSV. ``_make_scoring_ctx`` bundles the per-run constants
(obs arrays, clip ranges, MAD scales, scope flags) so every grid point
gets the same context object.
"""

import numpy as np
from scipy.stats import binom

from simulations.bias_grid_lib.constants import _E_SCORE_MODES
from simulations.bias_grid_lib.statistics import (
    _ALL_TESTS, _clip_to_range, _safe_distance, _safe_pvalue,
    _mad, _wasserstein_distance,
)


def _make_scoring_ctx(obs_logP, obs_e, obs_K1, clip_range,
                      e_score_mode, obs_e_cont,
                      n_obs_circ, n_obs_e_total,
                      N_det_obs, N_stars,
                      sim_logP_floor=0.0):
    """Bundle the per-run constants that every grid-point score needs.

    ``sim_logP_floor`` is the joint-mask threshold applied to the
    sim-detected arrays before CDF tests under scope="exclude". It is
    0 in every other scope, so the mask is a no-op there.
    """
    if e_score_mode not in _E_SCORE_MODES:
        raise ValueError(
            "e_score_mode must be one of %s, got %r" %
            (_E_SCORE_MODES, e_score_mode))
    obs_logP_arr = np.asarray(obs_logP)
    obs_e_arr = np.asarray(obs_e)
    obs_K1_arr = np.asarray(obs_K1)
    obs_e_cont_arr = (np.asarray(obs_e_cont) if obs_e_cont is not None
                      else None)
    # Per-channel scale for Wasserstein normalization (MAD of obs), so
    # the three dimensionful distances combine on a common scale. Match
    # the e-side choice to what _compute_scores compares against: in
    # split/eccentric_only the continuous (e>0) tail is what feeds the
    # distance, so its MAD is the right scale; in combined mode the full
    # obs_e is used. Stored in ctx and persisted to the cube npz so the
    # explorer can surface it.
    obs_e_for_wass = (obs_e_cont_arr
                      if e_score_mode in ("split", "eccentric_only")
                      and obs_e_cont_arr is not None
                      else obs_e_arr)
    wass_sigma = {
        "logP": _mad(obs_logP_arr),
        "e":    _mad(obs_e_for_wass),
        "K1":   _mad(obs_K1_arr),
    }
    return {
        "obs_logP": obs_logP_arr,
        "obs_e": obs_e_arr,
        "obs_K1": obs_K1_arr,
        "clip_range": clip_range,
        "e_score_mode": e_score_mode,
        "obs_e_cont": obs_e_cont_arr,
        "n_obs_circ": int(n_obs_circ) if n_obs_circ is not None else 0,
        "n_obs_e_total": int(n_obs_e_total) if n_obs_e_total is not None
                         else 0,
        "N_det_obs": int(N_det_obs),
        "N_stars": int(N_stars),
        "wass_sigma": wass_sigma,
        "sim_logP_floor": float(sim_logP_floor),
    }


def _compute_scores(res, ctx):
    """Compute KS/AD/CvM p-values + log-GMF for one grid-point result.

    Pure function: takes a worker `res` dict (with logP_det/e_det/K1_det,
    n_physical, p_det) and the per-run scoring context; returns a dict
    of every scalar we want in the cubes and the CSV row.

    p_det = n_detected / n_physical is the per-star detection rate (binaries
    detected divided by all injected realizations, binary + single). It
    already absorbs f_bin via the per-realization coin flip, so the binomial
    success probability is `p_det` directly — multiplying by f_bin again
    would double-count.

    No mutation, no I/O — safe to call from any process.
    """
    # Under scope="exclude" (schema v3), sim injection covers the full
    # [log_p_min, log_p_max] range — drop the log_p_min override that the
    # old "exclude" used. Restrict the CDF tests to logP>=cutoff via this
    # joint mask on the sim-detected arrays, matching the obs-side filter
    # the engine already applied. floor=0 in every other scope so the
    # mask is a no-op there. p_det / n_physical (used by the binomial)
    # stay over the FULL sim sample on purpose: the binomial speaks to
    # the full O-star population (catalog totals).
    floor = ctx.get("sim_logP_floor", 0.0)
    if floor > 0.0:
        sim_logP_arr = np.asarray(res["logP_det"])
        keep_sim = sim_logP_arr >= floor
        sim_logP_in = sim_logP_arr[keep_sim]
        sim_e_in = np.asarray(res["e_det"])[keep_sim]
        sim_K1_in = np.asarray(res["K1_det"])[keep_sim]
    else:
        sim_logP_in = res["logP_det"]
        sim_e_in = res["e_det"]
        sim_K1_in = res["K1_det"]

    sim_clipped = {
        "logP": _clip_to_range(sim_logP_in, *ctx["clip_range"]["logP"]),
        "e": _clip_to_range(sim_e_in, *ctx["clip_range"]["e"]),
        "K1": _clip_to_range(sim_K1_in, *ctx["clip_range"]["K1"]),
    }

    # The binomial uses the FULL detection count: p_det is the per-realization
    # detection probability across all logP, and N_det_obs is the full observed
    # binary count. The logP cutoff is scoped to the period CDF goodness-of-fit
    # only (see clip_range["logP"] and ctx["obs_logP"]) — it must not leak into
    # the count metric, otherwise f_bin gets inflated to compensate for the
    # below-cutoff binaries that were drawn but never compared.
    n_total_sim = res["n_physical"]
    p_det = res["p_det"]

    if n_total_sim > 0 and p_det > 0:
        p_binom = float(binom.pmf(ctx["N_det_obs"],
                                  ctx["N_stars"],
                                  p_det))
    else:
        p_binom = 0.0

    out = {"p_binom": p_binom}
    mode = ctx["e_score_mode"]
    for tname, tfn in _ALL_TESTS.items():
        p_logP = _safe_pvalue(tfn, ctx["obs_logP"], sim_clipped["logP"])
        p_K1 = _safe_pvalue(tfn, ctx["obs_K1"], sim_clipped["K1"],
                            min_samples=3)
        if mode == "combined":
            p_e = _safe_pvalue(tfn, ctx["obs_e"], sim_clipped["e"])
            p_e_circ = None
        else:  # "split" or "eccentric_only"
            sim_e_cont = sim_clipped["e"][sim_clipped["e"] > 0]
            p_e = _safe_pvalue(tfn, ctx["obs_e_cont"], sim_e_cont)
            if mode == "split":
                n_sim_e = len(sim_clipped["e"])
                n_sim_circ = int(np.sum(sim_clipped["e"] == 0))
                f_circ_sim = n_sim_circ / max(n_sim_e, 1)
                p_e_circ = float(binom.pmf(
                    ctx["n_obs_circ"], ctx["n_obs_e_total"], f_circ_sim
                )) if n_sim_e > 0 else 0.0
            else:
                p_e_circ = None

        gmf_pvals = [p_logP, p_e, p_K1, p_binom]
        if mode == "split":
            gmf_pvals.append(p_e_circ)
        log_gmf = 0.0
        for pv in gmf_pvals:
            if pv > 0:
                log_gmf += np.log(pv)
            else:
                log_gmf = -np.inf
                break

        out["%s_p_logP" % tname] = p_logP
        out["%s_p_e" % tname] = p_e
        out["%s_p_K1" % tname] = p_K1
        if mode == "split":
            out["%s_p_e_circ" % tname] = p_e_circ
        out["log_gmf_%s" % tname] = log_gmf

    # Wasserstein-1 distance branch. Distances are sign-flipped (so
    # argmax still selects the best fit) and normalized by per-channel
    # MAD (precomputed in ctx) before summing. p_binom and p_e_circ
    # remain real pmfs and enter on the log scale unchanged. p_e_circ
    # is independent of the test function, so we reuse the value left
    # by the last iteration of the p-value loop above.
    sigma = ctx["wass_sigma"]
    d_logP = _safe_distance(_wasserstein_distance,
                            ctx["obs_logP"], sim_clipped["logP"])
    d_K1 = _safe_distance(_wasserstein_distance,
                          ctx["obs_K1"], sim_clipped["K1"], min_samples=3)
    if mode == "combined":
        d_e = _safe_distance(_wasserstein_distance,
                             ctx["obs_e"], sim_clipped["e"])
    else:
        sim_e_cont = sim_clipped["e"][sim_clipped["e"] > 0]
        d_e = _safe_distance(_wasserstein_distance,
                             ctx["obs_e_cont"], sim_e_cont)

    out["wass_p_logP"] = d_logP   # stored in p_* slot so the explorer's
    out["wass_p_e"]    = d_e      # generic per-param loop picks them up;
    out["wass_p_K1"]   = d_K1     # values are distances, not probabilities

    if (np.isfinite(d_logP) and np.isfinite(d_e) and np.isfinite(d_K1)
            and p_binom > 0):
        log_gmf_wass = -(d_logP / sigma["logP"]
                         + d_e   / sigma["e"]
                         + d_K1  / sigma["K1"]) + np.log(p_binom)
        if mode == "split":
            if p_e_circ and p_e_circ > 0:
                log_gmf_wass += np.log(p_e_circ)
            else:
                log_gmf_wass = -np.inf
    else:
        log_gmf_wass = -np.inf

    if mode == "split":
        out["wass_p_e_circ"] = p_e_circ
    out["log_gmf_wass"] = log_gmf_wass

    out["log_gmf"] = out["log_gmf_ks"]
    return out
