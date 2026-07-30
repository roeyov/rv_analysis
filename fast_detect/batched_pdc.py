"""Batched Phase-Distance-Correlation (PDC) periodogram.

Reproduces ``period_search.periodogram.pdc_opt`` (the Numba
``PDC.pdc_func.calc_pdc_optimized`` kernel) for a whole batch of RV curves that
share one field's MJDs.

PDC power at frequency f is a distance correlation
``P(f) = <A, B(f)>_F / sqrt(||A||^2 * ||B(f)||^2)`` where

  * A       = U-centered VALUE-distance matrix (Zucker error-aware distance) —
              depends on the RV curve, so **per injection**;
  * B(f)    = U-centered PHASE-distance matrix from the signed time differences
              modulo 1/f — depends only on (times, f), so **shared** across the
              whole batch.

So, exactly like the batched Lomb-Scargle, the frequency scan is a single GEMM
``B_flat @ A_flat^T`` (n_freq x N^2)·(N^2 x n_batch), chunked over frequency
with a running peak. The permutation FAP shuffles the RV values (-> new A) and
takes the batched PDC max-power; iter_fap is the fraction of shuffles above the
candidate peak — matching ``period_search.permutation.pdc_permutation_max_powers``.

Value-distance A uses scipy.special.erf and is built once per injection on the
host (float64); only the frequency GEMM uses the pluggable backend.
"""

import numpy as np
from numba import njit
from scipy import special

from period_search.periodogram import make_pdc_freq_grid


@njit(cache=True)
def _fused_pdc_peak(num, denB, denA, tiny, start, best_pow, best_idx):
    """Fused PDC power + running peak for one frequency block (numpy path).

    Mirrors the generic path's ``where(denom > 0, num / maximum(denom, tiny),
    0.0)`` + strict-> running max/argmax element-for-element, without
    materializing the (n_block, n_batch) outer/denom/power temporaries.
    Updates best_pow/best_idx (n_batch,) in place.
    """
    n_block, n_batch = num.shape
    for f in range(n_block):
        for b in range(n_batch):
            denom = np.sqrt(denB[f] * denA[b])
            if denom > 0.0:
                d = denom if denom > tiny else tiny
                p = num[f, b] / d
            else:
                p = 0.0
            if p > best_pow[b]:
                best_pow[b] = p
                best_idx[b] = start + f


def _u_center(a):
    """U-centering matching PDC.pdc_func.unbiased_u_centering_fast (batched).

    a : (..., n, n). Returns U-centered matrix with zero diagonal.
    """
    n = a.shape[-1]
    row = a.sum(axis=-1, keepdims=True) / (n - 2)        # (...,n,1)
    col = a.sum(axis=-2, keepdims=True) / (n - 2)        # (...,1,n)
    tot = a.sum(axis=(-2, -1), keepdims=True) / ((n - 1) * (n - 2))
    out = a - row - col + tot
    idx = np.arange(n)
    out[..., idx, idx] = 0.0
    return out


def _value_distance_uc(Y, err):
    """U-centered value-distance matrices for a batch (host, float64).

    Y   : (n_epoch, n_batch)   err : (n_batch,) uniform per-star RV error.
    Returns A_uc (n_batch, n_epoch, n_epoch) and its flattened Frobenius norm².
    """
    Yt = np.asarray(Y, float).T                          # (B, E)
    err = np.asarray(err, float)
    diff = Yt[:, :, None] - Yt[:, None, :]               # (B,E,E) signed
    e2 = 2.0 * err[:, None, None] ** 2                   # e1²+e2² with e1=e2=err
    den = np.sqrt(2.0 * e2)
    x = diff / den
    yv = (2.0 * err[:, None, None]) / den                # (e1+e2)/den
    e = np.sqrt(e2) * (np.exp(-x * x) + x * special.erf(x) - yv)
    a = np.sqrt(np.maximum(e, 0.0))                      # value distance
    idx = np.arange(Yt.shape[1])
    a[:, idx, idx] = 0.0
    A = _u_center(a)                                     # (B,E,E)
    return A


class BatchedPDC:
    def __init__(self, times, pmin, pmax, samples_per_peak=10, xp=None,
                 freq_chunk=1024):
        from fast_detect.backends import get_xp
        self.xp = xp if xp is not None else get_xp("numpy")
        self.t = np.asarray(times, float)
        self.n = self.t.size
        T = float(np.ptp(self.t))
        self.freqs = make_pdc_freq_grid(pmin, pmax, T, samples_per_peak)
        self.n_freq = self.freqs.size
        self.freq_chunk = int(freq_chunk)
        self.tdiff = self.t[:, None] - self.t[None, :]    # signed (E,E)

    def _B_flat_chunk(self, f_block):
        """U-centered phase-distance matrices for a freq block -> (n_block, E*E),
        plus their Frobenius norm² (n_block,). Host numpy (cheap)."""
        inv_f = (1.0 / f_block)[:, None, None]            # (F,1,1)
        mpd = np.remainder(self.tdiff[None, :, :], inv_f)  # (F,E,E)
        b = mpd * (inv_f - mpd)
        B = _u_center(b)                                  # (F,E,E)
        denB = (B * B).sum(axis=(-2, -1))                 # (F,)
        return B.reshape(B.shape[0], -1), denB

    def _scan(self, Y, err):
        """Chunked freq scan. Returns best_idx, best_pow (host), A (B,E,E), denA.

        The per-chunk GEMM, denominator, and the running peak+argmax over
        frequency all stay ON THE BACKEND; only the final per-injection peak
        and its index are transferred to the host once, after the loop.
        """
        xp = self.xp
        A = _value_distance_uc(Y, err)                    # (B,E,E)
        n_batch = A.shape[0]
        A_flat = A.reshape(n_batch, -1)
        denA = (A_flat * A_flat).sum(axis=1)              # host float64 (returned)
        if xp.name == "numpy":
            # Fast path: BLAS GEMM + fused numba peak kernel (no
            # outer/denom/power temporaries). Same operands and evaluation
            # order as the generic path -> bit-identical peak power/index.
            best_pow_h = np.full(n_batch, -1e30)
            best_idx_h = np.zeros(n_batch, dtype=np.int64)
            A_T = A_flat.T
            for s in range(0, self.n_freq, self.freq_chunk):
                e = min(s + self.freq_chunk, self.n_freq)
                B_flat, denB = self._B_flat_chunk(self.freqs[s:e])
                num = np.matmul(B_flat, A_T)              # (n_block, B)
                _fused_pdc_peak(num, denB, denA, 1e-30, s,
                                best_pow_h, best_idx_h)
            return best_idx_h.astype(int), best_pow_h, A, denA
        A_x = xp.asarray(A_flat.T)                        # uploaded once
        denA_x = xp.asarray(denA)                         # uploaded once (B,)
        best_pow = xp.full((n_batch,), -1e30)
        best_idx = xp.zeros((n_batch,), dtype=xp.index_dtype)
        tiny = 1e-30
        for s in range(0, self.n_freq, self.freq_chunk):
            e = min(s + self.freq_chunk, self.n_freq)
            B_flat, denB = self._B_flat_chunk(self.freqs[s:e])
            num = xp.matmul(xp.asarray(B_flat), A_x)      # (n_block, B) on backend
            denomB = xp.asarray(denB)                     # (n_block,)
            outer = denomB[:, None] * denA_x[None, :]     # (n_block, B)
            denom = xp.sqrt(outer)
            # match host np.where(denom > 0, num/denom, 0.0): both num and denom
            # are 0 exactly where the phase-distance/value-distance norm vanishes.
            P = xp.where(denom > 0, num / xp.maximum(denom, tiny), 0.0)
            block_max = xp.max(P, axis=0)                 # (B,)
            block_arg = xp.astype(xp.argmax(P, axis=0), xp.index_dtype)
            upd = block_max > best_pow
            best_pow = xp.where(upd, block_max, best_pow)
            best_idx = xp.where(upd, s + block_arg, best_idx)
        best_idx = xp.to_numpy(best_idx).astype(int)      # ONE transfer, indexes host self.freqs
        best_pow = xp.to_numpy(best_pow)
        return best_idx, best_pow, A, denA

    def peak(self, Y, err):
        """Best period + peak PDC power per curve. Y:(n_epoch,n_batch)."""
        best_idx, best_pow, _, _ = self._scan(Y, err)
        return 1.0 / self.freqs[best_idx], best_pow

    def max_power(self, Y, err):
        _, p, _, _ = self._scan(Y, err)
        return p

    def _local_power(self, A, denA, f_local):
        """PDC power at per-curve local frequencies (host). A:(B,E,E),
        f_local:(B,L) -> (B,L)."""
        inv_f = (1.0 / f_local)[:, :, None, None]         # (B,L,1,1)
        mpd = np.remainder(self.tdiff[None, None, :, :], inv_f)  # (B,L,E,E)
        b = mpd * (inv_f - mpd)
        B = _u_center(b)                                  # (B,L,E,E)
        denB = (B * B).sum(axis=(-2, -1))                 # (B,L)
        num = (A[:, None, :, :] * B).sum(axis=(-2, -1))   # (B,L)
        denom = np.sqrt(denB * denA[:, None])
        return np.where(denom > 0, num / denom, 0.0)

    def peak_with_fwhm(self, Y, err, half_window=80):
        """Best period + peak power + FWHM period bounds (like BatchedLS)."""
        best_idx, best_pow, A, denA = self._scan(Y, err)
        n_batch = A.shape[0]
        W = int(half_window); offs = np.arange(-W, W + 1)
        loc = np.clip(best_idx[:, None] + offs[None, :], 0, self.n_freq - 1)
        f_local = self.freqs[loc]                          # (B,L) (freqs descending)
        P = self._local_power(A, denA, f_local)            # (B,L)
        c = W; half = (best_pow / 2.0)[:, None]
        below = P <= half; cols = np.arange(2 * W + 1)
        left_below = np.where(below & (cols[None, :] < c), cols[None, :], -1).max(axis=1)
        right_below = np.where(below & (cols[None, :] > c), cols[None, :], 2 * W + 1).min(axis=1)
        brow = np.arange(n_batch)
        il = np.clip(left_below, 0, 2 * W)
        p1 = P[brow, il]; p2 = P[brow, np.clip(il + 1, 0, 2 * W)]
        x1 = f_local[brow, il]; x2 = f_local[brow, np.clip(il + 1, 0, 2 * W)]
        tL = np.where(p2 != p1, (half[:, 0] - p1) / (p2 - p1), 0.0)
        fL = np.where(left_below >= 0, x1 + tL * (x2 - x1), f_local[:, 0])
        ir = np.clip(right_below, 0, 2 * W)
        p1r = P[brow, np.clip(ir - 1, 0, 2 * W)]; p2r = P[brow, ir]
        x1r = f_local[brow, np.clip(ir - 1, 0, 2 * W)]; x2r = f_local[brow, ir]
        tR = np.where(p2r != p1r, (half[:, 0] - p1r) / (p2r - p1r), 0.0)
        fR = np.where(right_below <= 2 * W, x1r + tR * (x2r - x1r), f_local[:, -1])
        period = 1.0 / self.freqs[best_idx]
        # f_local is descending, so fL (near left/higher-freq side) > fR
        flo = 1.0 / np.maximum(fL, fR); fhi = 1.0 / np.minimum(fL, fR)
        return period, best_pow, flo, fhi

    def power_spectrum(self, y, err):
        """Full PDC spectrum for a single curve (validation vs pdc_opt)."""
        A = _value_distance_uc(np.asarray(y)[:, None], np.atleast_1d(err))
        A_flat = A.reshape(1, -1)
        denA = float((A_flat * A_flat).sum())
        out = np.empty(self.n_freq)
        for s in range(0, self.n_freq, self.freq_chunk):
            e = min(s + self.freq_chunk, self.n_freq)
            B_flat, denB = self._B_flat_chunk(self.freqs[s:e])
            num = (B_flat @ A_flat.T)[:, 0]
            denom = np.sqrt(denB * denA)
            out[s:e] = np.where(denom > 0, num / denom, 0.0)
        return self.freqs, out
