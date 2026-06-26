"""Resolve eccentricity-scoring and logP-cutoff modes from cfg dicts.

The numerical-cutoff helper (``_numerical_logP_cutoff``) finds the
"elbow" of the empirical period CDF — used as a lower bound on logP for
the goodness-of-fit, below which the power-law model is contaminated by
short-period attrition.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from simulations.bias_grid_lib.constants import (
    _E_SCORE_MODES, _LOGP_CUTOFF_MODES, _LOGP_CUTOFF_SCOPES,
)
from simulations.bias_grid_lib.logging_utils import logger


def _resolve_logP_cutoff_mode(cfg):
    """Pick the logP cutoff mode from a config dict; default 'none'."""
    mode = cfg.get("logP_cutoff_mode", "none")
    if mode not in _LOGP_CUTOFF_MODES:
        raise ValueError("logP_cutoff_mode must be one of %s, got %r"
                         % (_LOGP_CUTOFF_MODES, mode))
    return mode


def _resolve_logP_cutoff_scope(cfg):
    """Pick the logP cutoff *scope* from a config dict; default 'period_only'.

    Orthogonal to ``logP_cutoff_mode``: the mode is *how* to find the
    cutoff value; the scope is *how* to apply it once found.
    """
    scope = cfg.get("logP_cutoff_scope", "period_only")
    if scope not in _LOGP_CUTOFF_SCOPES:
        raise ValueError("logP_cutoff_scope must be one of %s, got %r"
                         % (_LOGP_CUTOFF_SCOPES, scope))
    return scope


def _numerical_logP_cutoff(obs_logP, smooth_sigma=0.15, n_grid=2000):
    """First positive→negative zero-crossing of the smoothed CDF's 2nd derivative.

    Builds the empirical CDF of ``obs_logP``, interpolates it onto a
    uniform grid, applies a Gaussian smoother with width ``smooth_sigma``
    (in dex of logP), and returns the leftmost logP where
    ``d^2 CDF / d(logP)^2`` crosses from positive to negative. This is
    the "elbow" of the CDF — below it the model is contaminated by
    short-period attrition (mergers / common envelope) that the
    power-law model does not describe.

    Returns 0.0 if no positive→negative crossing is found, so callers
    can use the return value as a drop-in lower bound for ``clip_range``.
    """
    xs = np.sort(np.asarray(obs_logP, dtype=float))
    if len(xs) < 4:
        return 0.0
    ys = np.arange(1, len(xs) + 1) / len(xs)
    x_grid = np.linspace(xs.min(), xs.max(), n_grid)
    cdf_interp = np.interp(x_grid, xs, ys)
    dx = x_grid[1] - x_grid[0]
    sigma_pix = smooth_sigma / dx
    smoothed = gaussian_filter1d(cdf_interp, sigma=sigma_pix, mode="nearest")
    d1 = np.gradient(smoothed, dx)
    d2 = np.gradient(d1, dx)
    sign = np.sign(d2)
    transitions = np.where((sign[:-1] > 0) & (sign[1:] <= 0))[0]
    if len(transitions) == 0:
        return 0.0
    idx = transitions[0]
    a, b = d2[idx], d2[idx + 1]
    frac = a / (a - b) if (a - b) != 0 else 0.0
    return float(x_grid[idx] + frac * dx)


def _compute_logP_cutoff(obs_logP, mode, *, smooth_sigma=0.15,
                         manual_value=None):
    """Resolve the lower-bound logP cutoff for the CDF goodness-of-fit.

    Returns 0.0 for ``mode == 'none'`` (preserves backward-compat
    behavior since 0.0 is also the lower edge of the historical
    ``clip_range['logP']``). Negative or non-finite cutoffs fall back
    to 0.0 with a warning — useful when the elbow detector fails on a
    near-linear CDF.
    """
    if mode == "none":
        return 0.0
    if mode == "manual":
        if manual_value is None or not np.isfinite(manual_value):
            raise ValueError(
                "logP_cutoff_mode='manual' requires a finite "
                "logP_cutoff_value in cfg")
        return float(manual_value)
    if mode == "numerical":
        cutoff = _numerical_logP_cutoff(obs_logP, smooth_sigma=smooth_sigma)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            logger.warning(
                "logP_cutoff_mode='numerical' returned %r (no usable "
                "elbow at σ=%.2f); falling back to 0.0", cutoff, smooth_sigma)
            return 0.0
        return float(cutoff)
    raise ValueError("Unknown logP_cutoff_mode %r" % mode)


def _build_variant_inputs(e_score_mode, logP_cutoff_mode, apply_lucy_sweeny_e,
                          logP_cutoff_scope,
                          obs_logP, obs_e_value, obs_e_is_upper_limit, obs_K1,
                          n_stars_sample, n_catalog_total, n_catalog_nonsingle,
                          smooth_sigma=0.15, manual_value=None):
    """Resolve the obs-side scoring inputs for ONE scoring variant.

    A "variant" is one ``(e_score_mode, logP_cutoff_mode,
    apply_lucy_sweeny_e)`` combination. This is the pure, grid-independent
    bookkeeping the engine historically did inline once per run: apply the
    Lucy-Sweeney convention to the observed eccentricities, mask obs under
    the cutoff scope, size the binomial counts, build ``clip_range`` + the
    pre-split observed eccentricities + the ``sim_logP_floor`` joint mask.
    Factoring it out lets the engine build all 12 variant contexts up front
    and re-score one cached simulation against every one of them.

    ``obs_e_value`` / ``obs_e_is_upper_limit`` are the RAW observed
    eccentricities + upper-limit mask (as stored in the cube). Lucy is
    applied here per variant: upper-limit rows → 0 when True, else kept.
    This function never mutates its inputs; it returns masked *copies* per
    variant. Returns a dict ready to splat into ``_make_scoring_ctx`` plus
    the resolved ``logP_cutoff`` / ``N_stars`` / ``N_det_obs`` the cube
    persists per variant.
    """
    obs_logP = np.asarray(obs_logP, dtype=float)
    obs_K1 = np.asarray(obs_K1, dtype=float)
    # Lucy-Sweeney: collapse observed e upper-limits to 0 (True) or keep
    # the reported limit value (False). This is the only thing the lucy
    # axis changes (logP/K1/counts/sim are all lucy-independent).
    obs_e = np.asarray(obs_e_value, dtype=float).copy()
    if apply_lucy_sweeny_e:
        obs_e[np.asarray(obs_e_is_upper_limit, dtype=bool)] = 0.0

    logP_cutoff = _compute_logP_cutoff(
        obs_logP, logP_cutoff_mode,
        smooth_sigma=smooth_sigma, manual_value=manual_value)

    has_cut = logP_cutoff > 0.0
    if logP_cutoff_scope == "exclude":
        # Catalog-binomial semantics (schema v3): the binomial ALWAYS
        # speaks to the full O-star population (catalog totals), so the
        # recovered f_bin is a population binary fraction that does not
        # depend on the period cutoff — identical for logP_cutoff_mode in
        # {none, numerical}. A positive cutoff additionally masks the obs
        # P/e/K1 CDF panels (and the sim-det arrays, via sim_logP_floor)
        # to logP >= cutoff; with no cutoff (none / no-elbow) nothing is
        # masked but the binomial stays the catalog one.
        if n_catalog_total is None or n_catalog_nonsingle is None:
            raise ValueError(
                "logP_cutoff_scope=exclude requires n_catalog_total + "
                "n_catalog_nonsingle (load from ostar_catalog.csv); got None")
        if has_cut:
            keep = obs_logP >= logP_cutoff
            obs_logP_f, obs_e_f, obs_K1_f = (
                obs_logP[keep], obs_e[keep], obs_K1[keep])
        else:
            obs_logP_f, obs_e_f, obs_K1_f = obs_logP, obs_e, obs_K1
        obs_logP_above = obs_logP_f
        N_stars = int(n_catalog_total)
        N_det_obs = int(n_catalog_nonsingle)
        sim_logP_floor = float(logP_cutoff) if has_cut else 0.0
    else:
        # period_only: the cutoff (if any) scopes only the period CDF test;
        # obs e/K1, the binomial counts, and intrinsic draws stay full.
        obs_logP_f, obs_e_f, obs_K1_f = obs_logP, obs_e, obs_K1
        obs_logP_above = (obs_logP[obs_logP >= logP_cutoff]
                          if has_cut else obs_logP)
        N_stars = int(n_stars_sample)
        N_det_obs = len(obs_logP)
        sim_logP_floor = 0.0

    clip_range = {
        "logP": (logP_cutoff,
                 float(obs_logP_f.max()) if len(obs_logP_f) else logP_cutoff),
        "e": (0.0, 1.0),
        "K1": (0.0, float(obs_K1_f.max()) if len(obs_K1_f) else 0.0),
    }

    # Pre-split observed eccentricities for the circular/continuous modes.
    # Under scope=exclude these come from the period-filtered obs sample
    # (obs_e_f), matching the CDF panels.
    obs_e_cont = None
    n_obs_circ = None
    n_obs_e_total = None
    if e_score_mode != "combined":
        obs_e_cont = obs_e_f[obs_e_f > 0]
        if e_score_mode == "split":
            n_obs_circ = int(np.sum(obs_e_f == 0))
            n_obs_e_total = len(obs_e_f)
        elif e_score_mode == "eccentric_only" and len(obs_e_cont) > 0:
            # Tighten the simulated e-CDF window to the observed significant-e
            # range, analogous to the logP cutoff. obs_e_cont spans exactly
            # [min, max], so no observed system is dropped — only the simulated
            # array is clipped (via clip_range["e"] in _compute_scores).
            clip_range["e"] = (float(obs_e_cont.min()), float(obs_e_cont.max()))

    return {
        "e_score_mode": e_score_mode,
        "logP_cutoff_mode": logP_cutoff_mode,
        "apply_lucy_sweeny_e": bool(apply_lucy_sweeny_e),
        "logP_cutoff_scope": logP_cutoff_scope,
        "logP_cutoff": float(logP_cutoff),
        "obs_logP": obs_logP_above,
        "obs_e": obs_e_f,
        "obs_K1": obs_K1_f,
        "obs_e_cont": obs_e_cont,
        "n_obs_circ": n_obs_circ,
        "n_obs_e_total": n_obs_e_total,
        "N_stars": N_stars,
        "N_det_obs": N_det_obs,
        "clip_range": clip_range,
        "sim_logP_floor": sim_logP_floor,
    }


def _resolve_e_score_mode(cfg):
    """Pick the eccentricity scoring mode from a config dict.

    Accepts the new ``e_score_mode`` key (string enum) and falls back
    to the legacy boolean ``split_e_circular`` (True → "split",
    False → "combined") for old YAMLs and seed run-config snapshots.
    Defaults to "combined" if neither is present.
    """
    if "e_score_mode" in cfg:
        mode = cfg["e_score_mode"]
    elif "split_e_circular" in cfg:
        mode = "split" if cfg["split_e_circular"] else "combined"
    else:
        mode = "combined"
    if mode not in _E_SCORE_MODES:
        raise ValueError(
            "e_score_mode must be one of %s, got %r" % (_E_SCORE_MODES, mode))
    return mode
