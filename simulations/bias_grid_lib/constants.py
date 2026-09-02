"""Physical constants, histogram-bin definitions, and schema/mode enums.

Centralizes the read-only constants that every other bias_grid_lib
sub-module references. Moving them here keeps the small modules
side-effect free and gives external scripts (e.g. bias_grid_explorer.py,
bias_grid_summary_png.py) a stable location for the histogram bin edges.
"""

import numpy as np


# numpy 2.x renamed trapz -> trapezoid
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------

G_CGS = 6.674e-8
MSUN = 1.989e33
RSUN = 6.957e10
DAY = 86400.0
KM = 1e5
TWOPI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# 2D detection-probability histogram bins (accumulated across all grid points)
# ---------------------------------------------------------------------------

_HIST_NBINS = 30
_HIST_BINS = {
    "logP": np.linspace(0.0, 4.0, _HIST_NBINS + 1),
    "e":    np.linspace(0.0, 0.95, _HIST_NBINS + 1),
    "K1":   np.linspace(0.0, 300.0, _HIST_NBINS + 1),
    "q":    np.linspace(0.1, 1.0, _HIST_NBINS + 1),
}
_HIST_PAIRS = [("logP", "e"), ("logP", "K1"), ("logP", "q"),
               ("e", "K1"), ("e", "q"), ("q", "K1")]


# ---------------------------------------------------------------------------
# Scoring-mode enums (used by cutoffs.py and the engine)
# ---------------------------------------------------------------------------

_E_SCORE_MODES = ("combined", "split", "eccentric_only")
_LOGP_CUTOFF_MODES = ("none", "numerical", "manual")
_LOGP_CUTOFF_SCOPES = ("period_only", "exclude")


# ---------------------------------------------------------------------------
# All-variants run (schema v4): one simulation pass is scored under every
# (e_score_mode, logP_cutoff_mode) combination so the explorer can compare
# fitting mechanisms without re-injecting populations. ``manual`` is not
# enumerated by default (it needs a hand-set value) but may be requested
# explicitly via variant_filter.logP_cutoff_mode together with a YAML
# bias_grid.logP_cutoff_value; scope / smooth_sigma stay run-level.
# ---------------------------------------------------------------------------

_RUN_E_SCORE_MODES = ("combined", "split", "eccentric_only")
_RUN_LOGP_CUTOFF_MODES = ("none", "numerical")
# Lucy-Sweeney axis: collapse observed e upper-limits to 0 (True) vs keep
# the reported limit value (False). False first → matches the historical
# default and the closure-tool default variant.
_RUN_LUCY_SWEENY = (False, True)


def _lucy_token(apply_lucy_sweeny_e):
    return "lucyT" if apply_lucy_sweeny_e else "lucyF"


def variant_tag(e_score_mode, logP_cutoff_mode, apply_lucy_sweeny_e):
    """Canonical npz/dict key for one scoring variant.

    e.g. ``variant_tag("split", "numerical", True)
    -> "split__numerical__lucyT"``. Namespaces per-variant cubes/metadata
    in the persisted npz and the in-memory cube dicts.
    """
    return "%s__%s__%s" % (e_score_mode, logP_cutoff_mode,
                           _lucy_token(apply_lucy_sweeny_e))


def split_variant_tag(tag):
    """Inverse of ``variant_tag``.

    Returns ``(e_score_mode, logP_cutoff_mode, apply_lucy_sweeny_e)``. For
    legacy 2-part tags (schema v4) the lucy element is ``None`` — the
    caller falls back to the cube's shared ``apply_lucy_sweeny_e`` scalar.
    """
    parts = tag.split("__")
    em, cm = parts[0], parts[1]
    lucy = (parts[2] == "lucyT") if len(parts) >= 3 else None
    return em, cm, lucy


def _as_filter_set(value):
    """Normalize a variant-filter value to a set, or None if unconstrained.

    Accepts a scalar (``"eccentric_only"`` / ``True``), an iterable of
    scalars, or None/absent. Booleans are coerced so ``True`` matches a Lucy
    axis value of ``True`` regardless of how the YAML expressed it.
    """
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return {value}
    return set(value)


def all_variants(restrict=None):
    """The (e_score_mode, logP_cutoff_mode, apply_lucy_sweeny_e) variants.

    Cartesian product of the 3 e-score modes, 2 cutoff modes, and 2 Lucy
    settings (12 variants). Order is stable so cube/checkpoint enumeration
    is deterministic.

    ``restrict`` optionally subsets the product. It is a dict with any of the
    keys ``e_score_mode`` / ``logP_cutoff_mode`` / ``apply_lucy_sweeny_e``;
    each value may be a scalar, an iterable, or omitted (⇒ unconstrained on
    that axis). ``restrict=None`` (the default) returns all 12, so every
    existing caller is unaffected. Raises ``ValueError`` if the filter selects
    no variants.

    ``logP_cutoff_mode: manual`` is opt-in only: it is valid *inside*
    ``restrict`` (the caller must also set ``logP_cutoff_value``) but never
    part of the default sweep — the enumerated axis stays {none, numerical}
    unless a manual mode is explicitly requested.
    """
    cm_axis = list(_RUN_LOGP_CUTOFF_MODES)
    if restrict:
        cm_req = _as_filter_set(restrict.get("logP_cutoff_mode"))
        if cm_req:
            bad = sorted(m for m in cm_req if m not in _LOGP_CUTOFF_MODES)
            if bad:
                raise ValueError(
                    "variant_filter.logP_cutoff_mode contains unknown "
                    "mode(s) %s; valid: %s" % (bad, _LOGP_CUTOFF_MODES))
            # Opt-in extension: explicitly-requested non-run modes (i.e.
            # "manual") join the axis, after the defaults for stable order.
            cm_axis += [m for m in _LOGP_CUTOFF_MODES
                        if m in cm_req and m not in cm_axis]
    full = [(em, cm, lucy)
            for em in _RUN_E_SCORE_MODES
            for cm in cm_axis
            for lucy in _RUN_LUCY_SWEENY]
    if not restrict:
        return full
    em_set = _as_filter_set(restrict.get("e_score_mode"))
    cm_set = _as_filter_set(restrict.get("logP_cutoff_mode"))
    lucy_set = _as_filter_set(restrict.get("apply_lucy_sweeny_e"))
    if lucy_set is not None:
        lucy_set = {bool(v) for v in lucy_set}
    out = [(em, cm, lucy) for (em, cm, lucy) in full
           if (em_set is None or em in em_set)
           and (cm_set is None or cm in cm_set)
           and (lucy_set is None or bool(lucy) in lucy_set)]
    if not out:
        raise ValueError(
            "variant_filter %r selected no variants (valid e_score_mode=%s, "
            "logP_cutoff_mode=%s, apply_lucy_sweeny_e={True, False})"
            % (restrict, _RUN_E_SCORE_MODES, _RUN_LOGP_CUTOFF_MODES))
    return out


# Bumped whenever the persisted-cube schema grows new required keys. The
# explorer hard-errors below schema v3 (the minimum that carries the obs
# arrays it renders); v3 cubes load as a single synthetic variant.
#   v2 (2026-05): adds obs_logP/obs_e_value/obs_e_is_upper_limit/obs_K1/
#                 obs_q_sb2/obs_n_sb1/obs_n_sb2, plus logP_cutoff_smooth_sigma,
#                 sb1_tex, sb2_tex.
#   v3 (2026-05): catalog-based binomial under scope=exclude. Adds
#                 obs_n_catalog_total, obs_n_catalog_nonsingle, ostar_catalog.
#                 Scoring semantics change under scope=exclude: log_p_min no
#                 longer overridden; sim-det e/K1 jointly masked by
#                 sim_det_logP>=cutoff at scoring time; binomial uses catalog
#                 counts (134 / 75 for the BLOeM O-star sample).
#   v4 (2026-06): ALL-VARIANTS run. One injection/detection pass (single
#                 pdet_cube + det_shards + global hists) is scored under all
#                 6 (e_score_mode, logP_cutoff_mode) variants. Goodness cubes
#                 are namespaced 'v__<tag>__gmf_<t>_cube' / 'v__<tag>__<t>_
#                 <par>_cube' (+ '_e_circ' for split) and stored float32; a
#                 'variants' key lists the tags and each variant carries its
#                 own logP_cutoff / N_stars / N_det_obs / wass_sigma. scope,
#                 smooth_sigma, value, lucy stay run-level (shared).
#   v5 (2026-06): adds apply_lucy_sweeny_e as a 3rd variant axis → 12
#                 variants; tags are now 3-part 'v__<em>__<cm>__lucyT/F__…'
#                 and each variant stores its own apply_lucy_sweeny_e. The
#                 shared apply_lucy_sweeny_e scalar is a legacy placeholder
#                 (v5 readers use the per-variant value). v4 (2-part tags,
#                 lucy from the shared scalar) and v3 (single-mode) cubes
#                 remain readable.
#   v6 (2026-07): adds obs_qmin_sb1 (SB1 minimum mass ratios from the
#                 q_min column of the SB1 table; NaN where unavailable).
#                 Display-only — scoring never uses q. The explorer reads
#                 it as an optional key, so v3–v5 cubes remain readable.
#   v7 (2026-07): det shards gain kill_flag_nondet (int8, aligned with the
#                 *_nondet arrays: which gate killed each nondetected
#                 injection — see KILL_FLAG_LABELS) and checkpoint_results.csv
#                 gains n_kill_rlof/bicc/fap/tech/drv columns (-1 where
#                 unknown). Display-only — scoring is unaffected. Optional
#                 keys on read, so v3–v6 runs remain readable.
CUBE_SCHEMA_VERSION = 7

# Minimum cube schema the explorer can still render (v3 single-mode cubes
# load as one synthetic variant; v4 carries the full 6-variant namespace).
EXPLORER_MIN_SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Per-injection kill flags (which gate killed a nondetection)
# ---------------------------------------------------------------------------
# Codes 0-3 must match fast_detect/detector.py (KILL_DETECTED/RLOF/BICC/FAP —
# duplicated there because fast_detect is deliberately standalone). Codes 4-5
# only occur outside fast_detect:
#   4 = technical failure in detect_method='pipeline' (period_search_failed /
#       no_candidates / ls_fap_nan / all_fits_failed / null_fit_failed);
#   5 = the single ΔRV gate of detect_method='rv_threshold'.
# NOTE detect_method='pipeline' applies ONLY the prob_bicc gate, so its
# nondetections can only carry codes 2 or 4 (never 1 or 3).
KILL_DETECTED = 0
KILL_RLOF = 1
KILL_BICC = 2
KILL_FAP = 3
KILL_TECH = 4
KILL_DRV = 5

KILL_FLAG_LABELS = {
    KILL_DETECTED: "detected",
    KILL_RLOF: "rocheLobe",
    KILL_BICC: "bicc",
    KILL_FAP: "fap",
    KILL_TECH: "technical",
    KILL_DRV: "drv",
}

# n_kill_* CSV columns / scalar keys, in code order 1..5.
KILL_COUNT_KEYS = ("n_kill_rlof", "n_kill_bicc", "n_kill_fap",
                   "n_kill_tech", "n_kill_drv")

# ---------------------------------------------------------------------------
# Shard-storage layout
# ---------------------------------------------------------------------------

_DET_SHARDS_DIR = "det_shards"
