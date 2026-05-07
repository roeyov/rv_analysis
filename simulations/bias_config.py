"""
simulations.bias_config — Default configuration for the bias correction grid search.
"""

import numpy as np

DEFAULT_BIAS_CFG = {
    # Period range (log10 days)
    "log_p_min": 0.0,
    "log_p_max": 3.5,

    # Mass ratio range
    "q_min": 0.1,
    "q_max": 1.0,

    # Eccentricity
    "e_max": 0.95,
    "p_circ": 2.26,       # tidal circularization cutoff [days]
    "split_e_circular": True,  # split e scoring: binomial for circular fraction + test for e>0

    # Grid search
    "n_inject_per_star": 100,
    "per_star_mode": True,

    # Observed sample sizes
    "n_stars_sample": 134,   # full O-star sample (134 = 159 - 25 Oe)
    "n_det_obs": 71,         # 46 SB1 + 25 SB2 detected

    # Detection thresholds (must match pipeline)
    "prob_bicc_threshold": 0.5,
    "fap_threshold": 0.2,

    # Observed data paths
    "sb1_tex": "/Users/roeyovadia/Roey/Masters/Reasearch/Ostars_article/tables/sb1_solutions.tex",
    "sb2_tex": "/Users/roeyovadia/Roey/Masters/Reasearch/Ostars_article/tables/sb2_solutions.tex",
    "mass_file": "/Users/roeyovadia/Documents/Data/BLOeM_Data/mass_bloem.csv",
}

# =============================================================================
# Named grid presets
# =============================================================================
# Each preset bundles the 4D grid + a default n_inject_per_star.
# CLI flag --preset selects one; --n-inject overrides the default.
#
# Reference literature values (Sana+2012 Galactic): π=-0.55, κ=-0.10, η=-0.45,
# f_bin=0.69. Config D ranges are ~±3σ around these.
# =============================================================================

GRID_PRESETS = {
    "single": {  # single grid point — Sana+2012 values, for end-to-end testing
        "pi":    np.array([-0.55]),
        "kappa": np.array([-0.10]),
        "eta":   np.array([-0.45]),
        "fbin":  np.array([0.69]),
        "n_inject_per_star": 2,
    },
    "quick": {   # sanity check — 3×3×3×3 = 81 points
        "pi":    np.linspace(-1.5, 0.5, 3),
        "kappa": np.linspace(-2.0, 1.0, 3),
        "eta":   np.linspace(-1.0, 0.0, 3),
        "fbin":  np.array([0.4, 0.6, 0.8]),
        "n_inject_per_star": 2,
    },
    "G": {       # single-machine sanity preset — 4×4×3×3 = 144 points
        "pi":    np.linspace(-1.5, 0.5, 4),
        "kappa": np.linspace(-2.5, 1.0, 4),
        "eta":   np.linspace(-1.0, 0.0, 3),
        "fbin":  np.linspace(0.3, 1.0, 3),
        "n_inject_per_star": 30,
    },
    "D": {       # minimum scientifically defensible run — 8×8×6×6 = 2304 points
        "pi":    np.linspace(-1.5, 0.5, 8),
        "kappa": np.linspace(-2.5, 1.0, 8),
        "eta":   np.linspace(-1.0, 0.0, 6),
        "fbin":  np.linspace(0.3, 1.0, 6),
        "n_inject_per_star": 100,
    },
    "default": {  # wide-range reference grid — 15×15×10×8 = 18000 points
        "pi":    np.linspace(-2.0, 1.0, 15),
        "kappa": np.linspace(-3.0, 2.0, 15),
        "eta":   np.linspace(-1.5, 0.5, 10),
        "fbin":  np.linspace(0.3, 1.0, 8),
        "n_inject_per_star": 100,
    },
    "default2": {  # zoomed grid around D posterior peaks — 15×15×10×8 = 18000 points
        "pi":    np.linspace(-1.0, 0.0, 15),
        "kappa": np.linspace(-2.5, 0.5, 15),
        "eta":   np.linspace(-0.8, 0.3, 10),
        "fbin":  np.linspace(0.5, 1.0, 8),
        "n_inject_per_star": 100,
    },
    "default3": {  # zoomed grid around default2 posteriors — 15×15×10×8 = 18000 points
        "pi":    np.linspace(-0.7, -0.3, 15),
        "kappa": np.linspace(-2.0, 0.0, 15),
        "eta":   np.linspace(-0.6, 0.2, 10),
        "fbin":  np.linspace(0.7, 0.95, 8),
        "n_inject_per_star": 100,
    },
    "final": {  # publication run — 30×30×20×15 = 270,000 points, ~24h
        "pi":    np.linspace(-0.8, -0.25, 30),
        "kappa": np.linspace(-2.1, 0.2, 30),
        "eta":   np.linspace(-0.7, 0.3, 20),
        "fbin":  np.linspace(0.65, 1.0, 15),
        "n_inject_per_star": 300,
    },
    "final2": {  # publication run with wider eta — 30×30×20×15 = 270,000 points
        "pi":    np.linspace(-0.8, -0.25, 30),
        "kappa": np.linspace(-2.1, 0.2, 30),
        "eta":   np.linspace(-1.4, -0.4, 20),
        "fbin":  np.linspace(0.65, 1.0, 15),
        "n_inject_per_star": 300,
    },
    # --- final3 generation: ranges informed by final2_rv3 posteriors ---
    # pi upper edge hit → extend to -0.10; kappa well-constrained → narrow;
    # eta bimodal/edge-hitting → widen both sides; fbin very tight → narrow.
    "final3_1min": {  # quick validation — 10×10×6×5 = 3,000 pts, ~1 min
        "pi":    np.linspace(-0.80, -0.10, 10),
        "kappa": np.linspace(-2.10, 0.20, 10),
        "eta":   np.linspace(-0.80, 0.90, 6),
        "fbin":  np.linspace(0.70, 0.95, 5),
        "n_inject_per_star": 100,
    },
    "final3_10min": {  # development iteration — 17×17×11×9 = 28,611 pts, ~10 min
        "pi":    np.linspace(-0.80, -0.10, 17),
        "kappa": np.linspace(-2.10, 0.20, 17),
        "eta":   np.linspace(-0.80, 0.90, 11),
        "fbin":  np.linspace(0.70, 0.95, 9),
        "n_inject_per_star": 100,
    },
    "final3_3hr": {  # serious analysis — 30×30×20×15 = 270,000 pts, ~3 hr
        "pi":    np.linspace(-0.80, -0.10, 30),
        "kappa": np.linspace(-2.10, 0.20, 30),
        "eta":   np.linspace(-0.80, 0.90, 20),
        "fbin":  np.linspace(0.70, 0.95, 15),
        "n_inject_per_star": 200,
    },
    "final3_10hr": {  # publication quality — 40×40×27×20 = 864,000 pts, ~10 hr
        "pi":    np.linspace(-0.80, -0.10, 40),
        "kappa": np.linspace(-2.10, 0.20, 40),
        "eta":   np.linspace(-0.80, 0.90, 27),
        "fbin":  np.linspace(0.70, 0.95, 20),
        "n_inject_per_star": 200,
    },
}

# Backwards-compat alias (old code imported DEFAULT_GRIDS)
DEFAULT_GRIDS = {k: v for k, v in GRID_PRESETS["default"].items()
                 if k in ("pi", "kappa", "eta", "fbin")}
