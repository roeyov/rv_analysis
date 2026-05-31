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
