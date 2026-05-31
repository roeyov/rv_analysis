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

# Bumped whenever the persisted-cube schema grows new required keys. The
# explorer hard-errors below this version (see bias_grid_explorer.py).
#   v2 (2026-05): adds obs_logP/obs_e_value/obs_e_is_upper_limit/obs_K1/
#                 obs_q_sb2/obs_n_sb1/obs_n_sb2, plus logP_cutoff_smooth_sigma,
#                 sb1_tex, sb2_tex.
#   v3 (2026-05): catalog-based binomial under scope=exclude. Adds
#                 obs_n_catalog_total, obs_n_catalog_nonsingle, ostar_catalog.
#                 Scoring semantics change under scope=exclude: log_p_min no
#                 longer overridden; sim-det e/K1 jointly masked by
#                 sim_det_logP>=cutoff at scoring time; binomial uses catalog
#                 counts (134 / 75 for the BLOeM O-star sample).
CUBE_SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Shard-storage layout
# ---------------------------------------------------------------------------

_DET_SHARDS_DIR = "det_shards"
