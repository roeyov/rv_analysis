"""
simulations.true_posterior.k1_density — exact density of K1 given (P, e)
for one star, under the grid's model.

The grid computes (``physics.compute_K1_batch``)

    K1 = (2 pi G / P)^(1/3) M1^(1/3) q (1+q)^(-2/3) sin i / sqrt(1 - e^2)

with M1 ~ clip(N(M1_s, 0.3 M1_s), 7, 80) Msun, q ~ q^kappa on
[q_min, q_max], and cos i ~ U(0, 1). Define

    u = log10 K1 + (1/3) log10 P[d] + (1/2) log10(1 - e^2)
      = C0 + a + b + c,    with
      a = (1/3) log10 M1,  b = log10(q (1+q)^(-2/3)),  c = log10 sin i.

u does not depend on P or e. Its density is the convolution of the three
independent terms. The density of log10 K1 at fixed (P, e) is that of u,
shifted by -(1/3) log10 P - (1/2) log10(1 - e^2) (unit Jacobian).

Each term is discretised on a common grid of spacing h. Bin masses come
from the term's exact CDF, which handles the point masses at the M1 clip
edges and the integrable sin-i singularity at i = 90 deg. The discrete
masses are convolved by FFT. With h = 5e-4 dex the discretisation
broadening (variance 3 h^2 / 12) is negligible next to K1 errors.

Far tails. The FFT convolution has a round-off floor of about 1e-16 of
the peak density. A system whose observed u lies that far out (the real
catalogue has one: a K1 that needs M1 several sigma above the star's
clipped-normal scatter) would otherwise contribute round-off noise that
changes by O(1) in ln p between neighbouring kappa nodes. The lattice
density is therefore floored at ``REL_FLOOR`` times its peak, so such a
system contributes a kappa-independent constant (no information on
kappa), and ``shape_precompute`` warns about it. REL_FLOOR = 1e-12 is far
above the noise and far below any informative tail (a 7-sigma M1
excursion).
"""

import numpy as np
from scipy.special import ndtr

from simulations.bias_grid_lib.constants import DAY, G_CGS, KM, MSUN
from simulations.true_posterior.population import powerlaw_cdf

C0 = (np.log10(2.0 * np.pi * G_CGS * MSUN / DAY) / 3.0) - np.log10(KM)
M1_CLIP = (7.0, 80.0)          # injection._star_binary_rvs
M1_SCATTER = 0.3
H_DEFAULT = 5e-4
C_MIN = -6.0                   # P(log10 sin i < -6) ~ 5e-13
REL_FLOOR = 1e-12              # lattice density floor, relative to the peak


def _grid(lo, hi, h):
    """Bin centres covering [lo, hi] on the lattice k*h."""
    k0 = int(np.floor(lo / h)) - 1
    k1 = int(np.ceil(hi / h)) + 1
    return np.arange(k0, k1 + 1) * h, k0


def _masses(cdf, centres, h):
    m = cdf(centres + 0.5 * h) - cdf(centres - 0.5 * h)
    return np.clip(m, 0.0, None)


def a_masses(m1_star, h=H_DEFAULT):
    """Bin masses of a = (1/3) log10 M1 for one star (clipped normal)."""
    mu, sig = float(m1_star), M1_SCATTER * float(m1_star)
    lo, hi = np.log10(M1_CLIP[0]) / 3.0, np.log10(M1_CLIP[1]) / 3.0

    def cdf(a):
        m = 10.0 ** (3.0 * np.asarray(a))
        v = ndtr((m - mu) / sig)
        return np.where(a < lo, 0.0, np.where(a >= hi, 1.0, v))
    x, k0 = _grid(lo, hi, h)
    return _masses(cdf, x, h), k0


def _b_of_q(q):
    return np.log10(q) - (2.0 / 3.0) * np.log10(1.0 + q)


def b_masses(kappa, q_min, q_max, h=H_DEFAULT):
    """Bin masses of b = log10(q (1+q)^(-2/3)) for q ~ q^kappa."""
    qq = np.geomspace(q_min, q_max, 20001)
    bb = _b_of_q(qq)                         # strictly increasing in q
    lo, hi = bb[0], bb[-1]

    def cdf(b):
        b = np.asarray(b, dtype=float)
        qv = np.exp(np.interp(b, bb, np.log(qq)))
        v = powerlaw_cdf(qv, kappa, q_min, q_max)
        return np.where(b < lo, 0.0, np.where(b >= hi, 1.0, v))
    x, k0 = _grid(lo, hi, h)
    return _masses(cdf, x, h), k0


def c_masses(h=H_DEFAULT):
    """Bin masses of c = log10 sin i for isotropic inclinations."""
    def cdf(c):
        c = np.asarray(c, dtype=float)
        s2 = 10.0 ** (2.0 * np.minimum(c, 0.0))
        v = 1.0 - np.sqrt(np.clip(1.0 - s2, 0.0, None))
        return np.where(c >= 0.0, 1.0, v)
    x, k0 = _grid(C_MIN, 0.0, h)
    return _masses(cdf, x, h), k0


def _conv(*arrs):
    n = sum(len(a) for a in arrs) - len(arrs) + 1
    nfft = 1 << int(np.ceil(np.log2(n)))
    out = np.ones(nfft // 2 + 1, dtype=complex)
    for a in arrs:
        out *= np.fft.rfft(a, nfft)
    return np.clip(np.fft.irfft(out, nfft)[:n], 0.0, None)


class UDensity:
    """p_s(u | kappa) on the lattice, for a set of stars and kappa values.

    ``logpdf(star_pos, kappa_pos, u)`` interpolates ln p on the lattice.
    Build once per (stars, kappa grid); the stars' M1 are the injection
    sample's catalogue masses (``Mspec``).
    """

    def __init__(self, m1_stars, kappas, q_min, q_max, h=H_DEFAULT,
                 rel_floor=REL_FLOOR):
        self.h = float(h)
        self.rel_floor = float(rel_floor)
        self.kappas = np.atleast_1d(np.asarray(kappas, dtype=float))
        mc, kc = c_masses(h)
        self._dens = {}
        self._k0 = {}
        self._floor = {}
        b_list = [b_masses(k, q_min, q_max, h) for k in self.kappas]
        for si, m1 in enumerate(np.atleast_1d(m1_stars)):
            ma, ka = a_masses(m1, h)
            dens = [_conv(ma, mb, mc) / self.h for mb, _kb in b_list]
            # one floor per star, independent of kappa, so a far-tail
            # system contributes a kappa-independent constant
            floor = self.rel_floor * max(float(m.max()) for m in dens)
            for ki, (m, (_mb, kb)) in enumerate(zip(dens, b_list)):
                self._dens[(si, ki)] = np.maximum(m, floor)
                self._floor[(si, ki)] = floor
                self._k0[(si, ki)] = ka + kb + kc

    def pdf(self, si, ki, u):
        """Density of u for star si and kappa index ki (0 off-support)."""
        d = self._dens[(si, ki)]
        k0 = self._k0[(si, ki)]
        pos = (np.asarray(u, dtype=float) - C0) / self.h - k0
        return np.interp(pos, np.arange(d.size), d, left=0.0, right=0.0)

    def at_floor(self, si, ki, u):
        """True where pdf(u) is at the round-off floor (an uninformative
        far-tail value) or off the support."""
        return np.asarray(self.pdf(si, ki, u)) <= self._floor[(si, ki)]

    def logpdf(self, si, ki, u):
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(si, ki, u))

    def smoothed_pdf(self, si, ki, u, sigma_u):
        """Density of u convolved with N(0, sigma_u) (measurement error)."""
        if not sigma_u > 0:
            return self.pdf(si, ki, u)
        d = self._dens[(si, ki)]
        k0 = self._k0[(si, ki)]
        grid = C0 + (np.arange(d.size) + k0) * self.h
        w = np.exp(-0.5 * ((float(u) - grid) / sigma_u) ** 2)
        return float(np.sum(d * w) * self.h / (np.sqrt(2 * np.pi) * sigma_u))


def u_of(K1, P, e):
    """Observed u from K1 [km/s], P [d] and e."""
    return (np.log10(K1) + np.log10(P) / 3.0
            + 0.5 * np.log10(1.0 - np.asarray(e, dtype=float) ** 2))
