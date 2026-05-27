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
    # e scoring mode (3-way):
    #   "combined"       — single 2-sample test on full e (zeros + nonzeros)
    #   "split"          — 2-sample test on e>0 + binomial on circular fraction
    #   "eccentric_only" — 2-sample test on e>0 only; circular fraction ignored
    "e_score_mode": "combined",

    # Lucy-Sweeney handling for observed eccentricity upper limits.
    # True  — rows reported as "\leq 0.0X" in the tex tables are treated
    #         as circular (e = 0). Matches the historical behavior.
    # False — rows reported as "\leq 0.0X" keep the limit value as e
    #         (e = 0.0X). Use for sensitivity tests against e_score_mode.
    "apply_lucy_sweeny_e": False,

    # logP cutoff mode (3-way). Drops short-period systems (expected
    # merger / common-envelope attrition) so the power-law model is fit
    # only to the surviving conditional distribution P >= 10^cutoff.
    #   "none"      — no cutoff; score on the full observed range
    #   "numerical" — locate the first elbow of the smoothed obs CDF via
    #                 a Gaussian-smoothed 2nd derivative
    #   "manual"    — use the explicit value in `logP_cutoff_value`
    "logP_cutoff_mode": "numerical",
    "logP_cutoff_value": None,        # used iff mode == "manual"
    "logP_cutoff_smooth_sigma": 0.15, # Gaussian σ (dex) for "numerical"

    # logP cutoff scope (2-way). Controls HOW the cutoff (found by
    # logP_cutoff_mode) is APPLIED.
    #   "period_only" — cutoff filters only the period CDF KS/AD/CvM
    #                   test. Obs e/K1, N_det_obs, N_stars, intrinsic
    #                   sim draws all unchanged. Recovered f_bin is
    #                   the TOTAL binary fraction over [log_p_min,
    #                   log_p_max].
    #   "exclude"     — below-cutoff systems are removed everywhere:
    #                   obs P/e/K1 CDFs (filter obs_logP/e/K1),
    #                   binomial numerator (N_det_obs ← 70 - n_dropped),
    #                   binomial denominator (N_stars ← original -
    #                   n_dropped), AND intrinsic sim draws (via
    #                   log_p_min override → [cutoff, log_p_max]).
    #                   Recovered f_bin becomes the ABOVE-CUTOFF
    #                   binary fraction.
    "logP_cutoff_scope": "exclude",

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
    "rv_dir": "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_neb_div_from_coadded/",
    "sb2_analysis_dir": "/Users/roeyovadia/Documents/Data/BLOeM_Data/BLOeM_DR5.0_Combined_perStar_sb2",
    "ostar_catalog": "/Users/roeyovadia/Roey/Masters/Reasearch/Ostars_article/tables/ostar_catalog.csv",
    "output_dir": None,           # None -> <base_dir>/bias_grid_results/<preset>/

    # Run shape (former CLI flags)
    "preset": "final3_5min",
    "n_inject": None,             # None -> preset default
    "seed": 42,
    # n_stars_sample (above) doubles as the override: when None at runtime
    # the loader sets it to len(star_df).
    "detect_method": "rv_threshold",  # pipeline | rv_threshold

    # Parallelism (former CLI flags)
    "n_workers": None,            # None -> cpu_count - 2
    "parallel_grid": True,
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
        "fbin":  np.linspace(0.60, 0.95, 5),
        "n_inject_per_star": 100,
    },
    "final3_10min": {  # development iteration — 17×17×11×9 = 28,611 pts, ~10 min
        "pi":    np.linspace(-0.80, -0.10, 17),
        "kappa": np.linspace(-2.10, 0.20, 17),
        "eta":   np.linspace(-0.80, 0.90, 11),
        "fbin":  np.linspace(0.60, 0.95, 9),
        "n_inject_per_star": 100,
    },
        "final3_10min2": {  # 17×33×11×11 = 67,881 pts, ~20 min (kappa extended to 0.70)
        "pi":    np.linspace(-0.60, -0.30, 17),
        "kappa": np.linspace(-0.30, 0.70, 33),
        "eta":   np.linspace(-0.60, 0.0, 11),
        "fbin":  np.linspace(0.5, 0.8, 11),
        "n_inject_per_star": 100,
    },
    # final3_10min2 posteriors hit the pi upper edge (-0.30) and fbin lower
    # edge (0.50). Extend both, bracket Sana+12 (pi=-0.55, kappa=-0.10,
    # eta=-0.45, fbin=0.69), and use a uniform 0.05 step across all axes.
    "final3_10min3": {  # 27×25×17×13 = 149,175 pts, ~10 min on astro3 post-refactor
        "pi":    np.linspace(-0.80, 0.50, 27),   # extend past hit upper edge; covers Sana -0.55
        "kappa": np.linspace(-0.50, 0.70, 25),   # slight widen vs 10min2; Sana -0.10 well inside
        "eta":   np.linspace(-0.60, 0.20, 17),   # narrow around constrained mode ~-0.24; Sana -0.45 inside
        "fbin":  np.linspace(0.40, 1.00, 13),    # extend past hit lower edge; brackets Sana 0.69
        "n_inject_per_star": 100,
    },
    "final3_1min3": {  # 16×14×10×7 = 15,680 pts, ~2 min on astro3 post-refactor
        "pi":    np.linspace(-0.80, 0.50, 16),   # extend past hit upper edge; covers Sana -0.55
        "kappa": np.linspace(-1.50, 0.70, 14),   # slight widen vs 10min2; Sana -0.10 well inside
        "eta":   np.linspace(-0.60, 0.20, 10),   # narrow around constrained mode ~-0.24; Sana -0.45 inside
        "fbin":  np.linspace(0.40, 1.00, 7),     # extend past hit lower edge; brackets Sana 0.69
        "n_inject_per_star": 100,
    },
    # Same axis bounds as final3_10min3 but uniform ~0.03 step and kappa
    # lower bound pushed to -1.50 (captures equal-mass-biased priors).
    "final3_2hr": {  # 44×74×28×21 = 1,914,528 pts, ~2 hr on astro3 post-refactor
        "pi":    np.linspace(-0.95, 0.20, 44),   # step 0.030
        "kappa": np.linspace(-2.50, 0.10, 74),   # step 0.030; lower edge widened from -0.50
        "eta":   np.linspace(-0.80, 0.00, 28),   # step 0.030
        "fbin":  np.linspace(0.40, 1.00, 21),    # step 0.030
        "n_inject_per_star": 100,
    },

    "final3_5min": {
        "pi":    np.linspace(-0.95, 0.20, 20),   # step 0.030
        "kappa": np.linspace(-2.50, 0.10, 33),   # step 0.030; lower edge widened from -0.50
        "eta":   np.linspace(-0.80, 0.00, 13),   # step 0.030
        "fbin":  np.linspace(0.40, 1.00, 10),    # step 0.030
        "n_inject_per_star": 100,
    },
    # Per-truth closure presets — each 21×21×13×13 = 74,529 pts, centered on
    # its truth with ≥6 cells of margin from every edge.
    "final3_5min_truthA": {  # Sana 2012 (pi=-0.55, kappa=-0.10, eta=-0.45, fbin=0.69)
        "pi":    np.linspace(-0.95,  0.20, 21),
        "kappa": np.linspace(-1.10,  0.90, 21),
        "eta":   np.linspace(-0.80,  0.00, 13),
        "fbin":  np.linspace( 0.40,  1.00, 13),
        "n_inject_per_star": 100,
    },
    "final3_5min_truthB": {  # novel (pi=+1.00, kappa=0.00, eta=-0.30, fbin=0.90)
        "pi":    np.linspace( 0.30,  1.70, 21),
        "kappa": np.linspace(-1.00,  1.00, 21),
        "eta":   np.linspace(-0.80,  0.20, 13),
        "fbin":  np.linspace( 0.50,  1.00, 13),
        "n_inject_per_star": 100,
    },
    "final3_5min_truthC": {  # novel (pi=-0.50, kappa=+0.50, eta=+0.50, fbin=0.40)
        "pi":    np.linspace(-0.95,  0.20, 21),
        "kappa": np.linspace(-0.50,  1.50, 21),
        "eta":   np.linspace(-0.30,  1.30, 13),
        "fbin":  np.linspace( 0.10,  0.70, 13),
        "n_inject_per_star": 100,
    },
    "final3_3hr": {  # serious analysis — 30×30×20×15 = 270,000 pts, ~3 hr
        "pi":    np.linspace(-0.80, -0.10, 30),
        "kappa": np.linspace(-2.10, 0.20, 30),
        "eta":   np.linspace(-0.80, 0.90, 20),
        "fbin":  np.linspace(0.60, 0.95, 15),
        "n_inject_per_star": 200,
    },
    "final3_10hr": {  # publication quality — 40×40×27×20 = 864,000 pts, ~10 hr
        "pi":    np.linspace(-0.80, -0.10, 40),
        "kappa": np.linspace(-2.10, 0.20, 40),
        "eta":   np.linspace(-0.80, 0.90, 27),
        "fbin":  np.linspace(0.60, 0.95, 20),
        "n_inject_per_star": 200,
    },
}

# Backwards-compat alias (old code imported DEFAULT_GRIDS)
DEFAULT_GRIDS = {k: v for k, v in GRID_PRESETS["default"].items()
                 if k in ("pi", "kappa", "eta", "fbin")}
