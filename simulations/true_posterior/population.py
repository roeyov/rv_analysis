"""
simulations.true_posterior.population — power-law densities for the true
posterior, and the reference density of the labelled injection set.

The grid draws each of x = logP, q and e from p(x) ∝ x^alpha on
[xmin, xmax] with ``physics.powerlaw_draw``. That sampler clamps xmin to
>= 1e-10 and switches to log-uniform when |alpha + 1| < 1e-8. The
functions here reproduce those conventions exactly, so the likelihood's
intrinsic density is the density the simulation sampled from.

The injection set draws from a reference density g instead of from any
single theta. Per dimension, g is an equal-weight mixture of power laws
whose exponents span the prior range. Any exponent inside the span gives
x^a <= x^a_lo + x^a_hi, so the importance weights p(x|theta) / g(x) are
bounded. For e there is also a point mass at e = 0 (an explicit
circularised component, used above p_circ) with probability ``w_circ``.
"""

import numpy as np

XMIN_FLOOR = 1e-10      # physics.powerlaw_draw: xmin = max(xmin, 1e-10)
LOGUNIF_TOL = 1e-8      # physics.powerlaw_draw: |alpha + 1| < 1e-8


def _bounds(xmin, xmax):
    lo = max(float(xmin), XMIN_FLOOR)
    hi = max(float(xmax), lo + XMIN_FLOOR)
    return lo, hi


def powerlaw_lognorm(alpha, xmin, xmax):
    """ln Z with Z = integral of x^alpha over [xmin, xmax] (vectorized).

    Uses expm1 so the result is continuous through alpha = -1, where Z
    becomes ln(xmax / xmin).
    """
    lo, hi = _bounds(xmin, xmax)
    a = np.asarray(alpha, dtype=float) + 1.0
    lnlo, lnhi = np.log(lo), np.log(hi)
    small = np.abs(a) < LOGUNIF_TOL
    a_safe = np.where(small, 1.0, a)
    # (hi^a - lo^a) / a, written with expm1 for precision near a = 0.
    num = np.expm1(a_safe * lnhi) - np.expm1(a_safe * lnlo)
    lnz = np.log(np.abs(num)) - np.log(np.abs(a_safe))
    return np.where(small, np.log(lnhi - lnlo), lnz)


def powerlaw_logpdf(x, alpha, xmin, xmax):
    """ln p(x) for p(x) ∝ x^alpha on [xmin, xmax]; -inf outside.

    ``x`` and ``alpha`` broadcast against each other.
    """
    lo, hi = _bounds(xmin, xmax)
    x = np.asarray(x, dtype=float)
    a = np.asarray(alpha, dtype=float)
    # Inside the log-uniform branch the sampler draws exactly x^-1, so the
    # density must use exponent -1 there, not alpha.
    a_eff = np.where(np.abs(a + 1.0) < LOGUNIF_TOL, -1.0, a)
    inside = (x >= lo) & (x <= hi)
    xs = np.where(inside, x, 1.0)
    out = a_eff * np.log(xs) - powerlaw_lognorm(a, lo, hi)
    return np.where(inside, out, -np.inf)


def powerlaw_cdf(x, alpha, xmin, xmax):
    """P(X <= x) for the truncated power law (vectorized)."""
    lo, hi = _bounds(xmin, xmax)
    x = np.clip(np.asarray(x, dtype=float), lo, hi)
    a = np.asarray(alpha, dtype=float) + 1.0
    small = np.abs(a) < LOGUNIF_TOL
    a_safe = np.where(small, 1.0, a)
    num = np.expm1(a_safe * np.log(x)) - np.expm1(a_safe * np.log(lo))
    den = np.expm1(a_safe * np.log(hi)) - np.expm1(a_safe * np.log(lo))
    logu = np.log(x / lo) / np.log(hi / lo)
    return np.where(small, logu, num / den)


def powerlaw_ppf(u, alpha, xmin, xmax):
    """Inverse CDF. Same formula as physics.powerlaw_draw."""
    lo, hi = _bounds(xmin, xmax)
    u = np.asarray(u, dtype=float)
    a = np.asarray(alpha, dtype=float) + 1.0
    small = np.abs(a) < LOGUNIF_TOL
    a_safe = np.where(small, 1.0, a)
    base = lo ** a_safe + u * (hi ** a_safe - lo ** a_safe)
    base = np.maximum(base, 1e-300)
    x_pl = np.exp(np.log(base) / a_safe)
    x_lu = lo * (hi / lo) ** u
    return np.where(small, x_lu, x_pl)


class PowerLawMixture:
    """Mixture of truncated power laws on a common [xmin, xmax]."""

    def __init__(self, alphas, xmin, xmax, weights=None):
        self.alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
        self.xmin, self.xmax = _bounds(xmin, xmax)
        k = self.alphas.size
        w = np.full(k, 1.0 / k) if weights is None else np.asarray(
            weights, dtype=float)
        if w.shape != (k,) or np.any(w <= 0):
            raise ValueError("weights must be positive, one per component")
        self.weights = w / w.sum()

    def sample(self, n, rng):
        """Draw n values: a random component, then an inverse-CDF draw."""
        comp = rng.choice(self.alphas.size, size=n, p=self.weights)
        u = rng.uniform(0.0, 1.0, n)
        return powerlaw_ppf(u, self.alphas[comp], self.xmin, self.xmax)

    def logpdf(self, x):
        """ln g(x) = logsumexp_k [ln w_k + ln p(x | alpha_k)]."""
        x = np.asarray(x, dtype=float)
        comps = np.stack([np.log(w) + powerlaw_logpdf(x, a, self.xmin,
                                                      self.xmax)
                          for a, w in zip(self.alphas, self.weights)])
        m = np.max(comps, axis=0)
        m_safe = np.where(np.isfinite(m), m, 0.0)
        out = m_safe + np.log(np.sum(np.exp(comps - m_safe), axis=0))
        return np.where(np.isfinite(m), out, -np.inf)

    def to_dict(self):
        return {"alphas": self.alphas.tolist(), "xmin": self.xmin,
                "xmax": self.xmax, "weights": self.weights.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["alphas"], d["xmin"], d["xmax"], d["weights"])


# Default reference exponents. They span the ext2 prior box plus margin
# (pi [-1.55, 0.10], kappa [-1.30, 1.80] and [-2.44, 1.80] on q >= 0.1,
# eta [-1.25, 0.55]), so the grid and moderate prior extensions reweight
# with bounded weights.
DEFAULT_REFERENCE = {
    "logP": {"alphas": [-2.0, -0.75, 0.5]},
    "q": {"alphas": [-2.5, -0.35, 1.8]},
    "e": {"alphas": [-1.25, -0.35, 1.0]},
    "w_circ": 0.15,
}


class ReferenceDensity:
    """Product reference density g(logP) g(q) g(e) of the injection set.

    For e: with probability ``w_circ`` the draw is the circular component
    (e_draw = 0 exactly); otherwise e_draw comes from the power-law mixture.
    The p_circ rule (e forced to 0 below p_circ) is applied afterwards by
    ``injection._star_binary_rvs``, exactly as in the grid.
    """

    def __init__(self, bounds, spec=None):
        spec = DEFAULT_REFERENCE if spec is None else spec
        self.spec = {k: (dict(v) if isinstance(v, dict) else v)
                     for k, v in spec.items()}
        self.w_circ = float(spec["w_circ"])
        if not 0.0 <= self.w_circ < 1.0:
            raise ValueError("w_circ must be in [0, 1)")
        self.g_logP = PowerLawMixture(spec["logP"]["alphas"],
                                      bounds["log_p_min"], bounds["log_p_max"],
                                      spec["logP"].get("weights"))
        self.g_q = PowerLawMixture(spec["q"]["alphas"],
                                   bounds["q_min"], bounds["q_max"],
                                   spec["q"].get("weights"))
        self.g_e = PowerLawMixture(spec["e"]["alphas"], 1e-6,
                                   bounds["e_max"], spec["e"].get("weights"))

    def sample(self, n, rng):
        """(logP, q, e_draw, circ_comp) for n injections, in a fixed order."""
        logP = self.g_logP.sample(n, rng)
        q = self.g_q.sample(n, rng)
        circ = rng.uniform(0.0, 1.0, n) < self.w_circ
        e = self.g_e.sample(n, rng)
        e[circ] = 0.0
        return logP, q, e, circ

    def to_dict(self):
        return {"logP": self.g_logP.to_dict(), "q": self.g_q.to_dict(),
                "e": self.g_e.to_dict(), "w_circ": self.w_circ}
