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
# enumerated (it needs a hand-set value); scope / smooth_sigma / value are
# held fixed by bias_grid.main (scope=exclude, sigma=0.15, value=None).
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
    """
    full = [(em, cm, lucy)
            for em in _RUN_E_SCORE_MODES
            for cm in _RUN_LOGP_CUTOFF_MODES
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
CUBE_SCHEMA_VERSION = 5

# Minimum cube schema the explorer can still render (v3 single-mode cubes
# load as one synthetic variant; v4 carries the full 6-variant namespace).
EXPLORER_MIN_SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Shard-storage layout
# ---------------------------------------------------------------------------

_DET_SHARDS_DIR = "det_shards"
