"""
simulations.true_posterior.k7_patch — import-time switch of the fast
detector's model comparison to SEVEN free parameters for the Keplerian model
(variant runs only; the adopted injection set does not use it).

``fast_detect.batched_fit.fit_batch`` profiles the jitter in closed form and
then scores the orbit with ``nvarys=6`` and the constant model with
``nvarys=2`` in the corrected BIC. The production fit and the paper count
seven free parameters for the orbit (P, K1, e, omega, T0, gamma and the
jitter) and two for the constant model. For 26 epochs the penalty of the
orbit is therefore 28.2 in the simulations against 34.9 in the real
analysis, and ``p_bin > 0.5`` of the fast detector corresponds to
``p_bin > 0.035`` of the paper's definition.

Importing this module replaces ``batched_fit._llh_bicc`` by a wrapper that
scores the orbit with seven parameters and leaves the constant model as it
is. No file of ``fast_detect`` is edited. The wrapper raises on any other
parameter count, so a change of the detector code cannot pass silently.

``set_orbit_count(6 | 7)`` switches between the coded and the paper's count
inside one process (used by the tools that compare the two).

Used through ``injection_set_k7`` (forkserver preload: the pool workers
inherit the switch).

Since 2026-10-10 ``fast_detect.batched_fit`` itself scores the orbit with
seven parameters (``N_PARAM_ORBIT``), so with the default count this module
changes nothing; ``set_orbit_count(6)`` reproduces the old detector for
comparisons.
"""

import numpy as np

from fast_detect import batched_fit

N_ORBIT_CODED = 6
N_ORBIT_PAPER = 7
N_NULL = 2

if not getattr(batched_fit, "_k7_patch_applied", False):
    batched_fit._k7_orig_llh_bicc = batched_fit._llh_bicc
    batched_fit._k7_orbit_count = N_ORBIT_PAPER

    def _llh_bicc_switch(SSR, err2, N, nvarys):
        orig = batched_fit._k7_orig_llh_bicc
        if nvarys == N_NULL:
            return orig(SSR, err2, N, nvarys)
        if nvarys not in (N_ORBIT_CODED, N_ORBIT_PAPER):
            raise RuntimeError("k7_patch: fast_detect.batched_fit scored a "
                               "model with nvarys=%r; expected %d or %d "
                               "(orbit) or %d (constant)"
                               % (nvarys, N_ORBIT_CODED, N_ORBIT_PAPER,
                                  N_NULL))
        k = batched_fit._k7_orbit_count
        if N - k - 2.0 <= 0:
            # BICc is undefined for so few epochs: the orbit is never
            # preferred (the coded count is used for the likelihood value)
            llh, _ = orig(SSR, err2, N, N_ORBIT_CODED)
            return llh, np.full_like(np.asarray(SSR, float), np.inf)
        return orig(SSR, err2, N, k)

    batched_fit._llh_bicc = _llh_bicc_switch
    batched_fit._k7_patch_applied = True


def set_orbit_count(k):
    """Score the orbit with ``k`` free parameters (6 = as coded, 7 = paper)."""
    if k not in (N_ORBIT_CODED, N_ORBIT_PAPER):
        raise ValueError("orbit count must be %d or %d" % (N_ORBIT_CODED,
                                                           N_ORBIT_PAPER))
    batched_fit._k7_orbit_count = int(k)


def orbit_count():
    return int(batched_fit._k7_orbit_count)
