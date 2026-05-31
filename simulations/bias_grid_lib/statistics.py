"""Two-sample tests and distance metrics used by the scorer.

Wraps SciPy's KS / AD / CvM / Wasserstein primitives in safe-to-call
helpers that return sentinels (0.0 or +inf) on degenerate inputs, so the
grid scorer can treat every cell uniformly without try/except scaffolding.
"""

import warnings
import numpy as np
from scipy.stats import (ks_2samp, anderson_ksamp, cramervonmises_2samp,
                         wasserstein_distance)


# ---------------------------------------------------------------------------
# Statistical test helpers
# ---------------------------------------------------------------------------

def _clip_to_range(sim_arr, lo, hi):
    """Restrict simulated detected array to [lo, hi]."""
    return sim_arr[(sim_arr >= lo) & (sim_arr <= hi)]


def _safe_pvalue(test_fn, obs, sim, min_samples=5):
    """Compute a two-sample test p-value, returning 0.0 on failure."""
    if len(sim) < min_samples:
        return 0.0
    try:
        return float(test_fn(obs, sim))
    except Exception:
        return 0.0


def _ks_pvalue(obs, sim):
    return ks_2samp(obs, sim).pvalue


def _ad_pvalue(obs, sim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return anderson_ksamp([obs, sim]).pvalue


def _cvm_pvalue(obs, sim):
    return cramervonmises_2samp(obs, sim).pvalue


def _wasserstein_distance(obs, sim):
    return float(wasserstein_distance(obs, sim))


def _safe_distance(dist_fn, obs, sim, min_samples=5):
    """Compute a two-sample distance, returning np.inf on failure or
    insufficient samples. Mirror of `_safe_pvalue` but for distance
    metrics that should sort as "bigger = worse fit"."""
    if len(sim) < min_samples:
        return float("inf")
    try:
        return float(dist_fn(obs, sim))
    except Exception:
        return float("inf")


def _mad(x, floor=1e-6):
    """Median absolute deviation, with a lower floor to avoid div-by-zero
    on near-degenerate samples (single value, or all-equal observations)."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return floor
    return max(float(np.median(np.abs(x - np.median(x)))), floor)


# p-value tests: combine via Σ log p, best = argmax.
_ALL_TESTS = {"ks": _ks_pvalue, "ad": _ad_pvalue, "cvm": _cvm_pvalue}

# Distance tests: combine via -Σ d_i/σ_i + log p_binom (+ log p_e_circ),
# best = argmax (after sign flip). σ is set per-channel from obs MAD.
_DIST_TESTS = {"wass": _wasserstein_distance}

# Union for cube allocation, restoration, and save sites — order matters
# for any logic that derives a default-test alias from the first entry.
_SCORED_TESTS = tuple(list(_ALL_TESTS) + list(_DIST_TESTS))
