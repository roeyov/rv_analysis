"""Batched floating-mean Lomb-Scargle (normalization='model').

Reproduces ``period_search.periodogram.ls`` (astropy LombScargle with
``fit_mean=True, center_data=True, normalization='model'``) but evaluated for a
whole batch of RV curves that share the SAME time sampling — the bias-grid case,
where all injections of a field share one MJD array.

Why this is correct and batchable
---------------------------------
The model at frequency f is ``y ≈ a·cos(ωt) + b·sin(ωt) + c`` (floating mean,
ω=2πf). With per-star UNIFORM errors the 'model' power
``P = (χ²₀ − χ²)/χ²`` is independent of the error value, so a single
error-free least-squares solve per (freq, curve) gives the exact power for every
star. The 3×3 normal matrix depends only on (times, f) — shared across all
curves — so the per-frequency fit reduces to a closed-form quadratic form in the
three projections (Σ c·y, Σ s·y, Σ y), which are batched matmuls. No per-curve
linear-algebra and nothing backend-specific in the hot path.

Memory: the full (n_freq × n_batch) spectrum is never materialized; we scan the
frequency grid in chunks and keep only the running peak power + its frequency,
which is all detection needs (candidate period + peak power, and shuffle
max-powers for the permutation FAP).
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _fused_ls_peak(cY, sY, oY, SYY, i00, i11, i22, i01, i02, i12,
                   ff, start, best_pow, best_idx):
    """Fused power assembly + running peak for one frequency block.

    Replaces the ~12 full (n_block, n_batch) float64 temporaries of the
    unfused numpy expression in ``_power_chunk`` (DRAM-bandwidth bound) with
    one streaming pass. The arithmetic mirrors ``_power_chunk`` term-for-term
    in the same evaluation order (numba default: no FMA contraction, no
    reassociation), and ties resolve exactly like the chunked
    max/argmax + strict-> running update -> bit-identical peak power/index.
    Updates best_pow/best_idx (n_batch,) in place.
    """
    n_block, n_batch = cY.shape
    for f in range(n_block):
        for b in range(n_batch):
            cYf = cY[f, b]
            sYf = sY[f, b]
            oYb = oY[b]
            SYYb = SYY[b]
            Q = (i00[f] * cYf * cYf + i11[f] * sYf * sYf + i22[f] * oYb * oYb
                 + 2.0 * i01[f] * cYf * sYf + 2.0 * i02[f] * cYf * oYb
                 + 2.0 * i12[f] * sYf * oYb)
            chi2_0 = SYYb - oYb * oYb / ff
            chi2 = SYYb - Q
            p = (chi2_0 - chi2) / chi2
            if p > best_pow[b]:
                best_pow[b] = p
                best_idx[b] = start + f


def autopower_grid(times, pmin, pmax, samples_per_peak=50):
    """Frequency grid matching astropy LombScargle.autopower(method='fast').

    f in [1/pmax, 1/pmin] with spacing df = 1/(samples_per_peak·baseline).
    """
    t = np.asarray(times, dtype=np.float64)
    baseline = float(t.max() - t.min())
    df = 1.0 / (samples_per_peak * baseline)
    fmin, fmax = 1.0 / pmax, 1.0 / pmin
    n = int(np.ceil((fmax - fmin) / df)) + 1
    return fmin + df * np.arange(n, dtype=np.float64)


class BatchedLS:
    """Batched floating-mean Lomb-Scargle peak finder over a fixed time grid.

    Parameters
    ----------
    times : (n_epoch,) array          shared MJDs for the batch (one field).
    pmin, pmax, samples_per_peak      frequency grid (match the pipeline config).
    xp                                backend (fast_detect.backends.get_xp(...)).
    freq_chunk                        # frequencies processed per block.
    """

    def __init__(self, times, pmin, pmax, samples_per_peak=50, xp=None,
                 freq_chunk=4096):
        from fast_detect.backends import get_xp
        self.xp = xp if xp is not None else get_xp("numpy")
        self.t = np.asarray(times, dtype=np.float64)
        self.n_epoch = self.t.size
        self.freqs = autopower_grid(self.t, pmin, pmax, samples_per_peak)
        self.n_freq = self.freqs.size
        self.freq_chunk = int(freq_chunk)
        self._periods_xp = self.xp.asarray(1.0 / self.freqs)
        self._design_inv_cache = None

    def _design_inv(self):
        """Data-independent 3x3 normal-matrix inverse entries per frequency.

        Depends only on (t, freqs) so it is computed once per instance (the
        instances are cached per field). Computed with the SAME chunking and
        reduction order as ``_power_chunk`` -> bit-identical values.
        Returns (i00, i11, i22, i01, i02, i12), each (n_freq,) float64.
        """
        if self._design_inv_cache is None:
            n = self.n_freq
            out = tuple(np.empty(n) for _ in range(6))
            t = self.t
            ff = float(self.n_epoch)
            for start in range(0, n, self.freq_chunk):
                stop = min(start + self.freq_chunk, n)
                w = 2.0 * np.pi * self.freqs[start:stop]
                ang = w[:, None] * t[None, :]
                C = np.cos(ang)
                S = np.sin(ang)
                a = np.sum(C * C, axis=1)
                d = np.sum(S * S, axis=1)
                bb = np.sum(C * S, axis=1)
                cc = np.sum(C, axis=1)
                ee = np.sum(S, axis=1)
                det = (a * (d * ff - ee * ee)
                       - bb * (bb * ff - ee * cc)
                       + cc * (bb * ee - d * cc))
                inv = 1.0 / det
                out[0][start:stop] = (d * ff - ee * ee) * inv
                out[1][start:stop] = (a * ff - cc * cc) * inv
                out[2][start:stop] = (a * d - bb * bb) * inv
                out[3][start:stop] = (cc * ee - bb * ff) * inv
                out[4][start:stop] = (bb * ee - cc * d) * inv
                out[5][start:stop] = (bb * cc - a * ee) * inv
            self._design_inv_cache = out
        return self._design_inv_cache

    # -- core ---------------------------------------------------------------
    def _power_chunk(self, f_block, Y, SYY, oY):
        """Power for a block of frequencies. Y:(n_epoch,n_batch) backend array.

        Returns power (n_block, n_batch) backend array via the closed-form
        floating-mean quadratic form.
        """
        xp = self.xp
        t = xp.asarray(self.t)                       # (n_epoch,)
        w = xp.asarray(2.0 * np.pi * f_block)        # (n_block,)
        ang = w[:, None] * t[None, :]                # (n_block, n_epoch)
        C = xp.cos(ang)                              # (n_block, n_epoch)
        S = xp.sin(ang)
        # normal-matrix entries per frequency (sums over epochs)
        a = xp.sum(C * C, axis=1)                    # Σc²   (n_block,)
        d = xp.sum(S * S, axis=1)                    # Σs²
        bb = xp.sum(C * S, axis=1)                   # Σcs
        cc = xp.sum(C, axis=1)                       # Σc
        ee = xp.sum(S, axis=1)                       # Σs
        ff = float(self.n_epoch)                     # N
        # symmetric 3x3 inverse (closed form); M=[[a,bb,cc],[bb,d,ee],[cc,ee,ff]]
        det = (a * (d * ff - ee * ee)
               - bb * (bb * ff - ee * cc)
               + cc * (bb * ee - d * cc))
        inv = 1.0 / det
        i00 = (d * ff - ee * ee) * inv
        i11 = (a * ff - cc * cc) * inv
        i22 = (a * d - bb * bb) * inv
        i01 = (cc * ee - bb * ff) * inv
        i02 = (bb * ee - cc * d) * inv
        i12 = (bb * cc - a * ee) * inv
        # projections b = [Σc·y, Σs·y, Σy]
        cY = xp.matmul(C, Y)                         # (n_block, n_batch)
        sY = xp.matmul(S, Y)
        oY = oY[None, :]                             # (1, n_batch) broadcast
        # quadratic form Q = bᵀ M⁻¹ b
        Q = (i00[:, None] * cY * cY
             + i11[:, None] * sY * sY
             + i22[:, None] * oY * oY
             + 2.0 * i01[:, None] * cY * sY
             + 2.0 * i02[:, None] * cY * oY
             + 2.0 * i12[:, None] * sY * oY)
        chi2_0 = SYY[None, :] - oY * oY / ff         # constant-model residual
        chi2 = SYY[None, :] - Q                      # full-fit residual
        return (chi2_0 - chi2) / chi2                # 'model' normalization

    def _scan(self, Y):
        """Chunked frequency scan. Returns (best_idx, best_pow, SYY, oY) with
        Y kept as a backend array. best_idx/best_pow are host numpy."""
        xp = self.xp
        if xp.name == "numpy":
            # Fast path: BLAS projections + the fused numba peak kernel — the
            # (n_block, n_batch) power matrix is never materialized. The C/S
            # trig + GEMMs are computed per chunk exactly as _power_chunk does,
            # so results are bit-identical to the generic path below.
            Yh = np.asarray(Y, dtype=np.float64)
            SYY = np.sum(Yh * Yh, axis=0)
            oY = np.sum(Yh, axis=0)
            n_batch = Yh.shape[1]
            i00, i11, i22, i01, i02, i12 = self._design_inv()
            best_pow = np.full(n_batch, -1e30)
            best_idx = np.zeros(n_batch, dtype=np.int64)
            t = self.t
            ff = float(self.n_epoch)
            for start in range(0, self.n_freq, self.freq_chunk):
                stop = min(start + self.freq_chunk, self.n_freq)
                w = 2.0 * np.pi * self.freqs[start:stop]
                ang = w[:, None] * t[None, :]
                C = np.cos(ang)
                S = np.sin(ang)
                cY = np.matmul(C, Yh)
                sY = np.matmul(S, Yh)
                _fused_ls_peak(cY, sY, oY, SYY,
                               i00[start:stop], i11[start:stop],
                               i22[start:stop], i01[start:stop],
                               i02[start:stop], i12[start:stop],
                               ff, start, best_pow, best_idx)
            return best_idx.astype(int), best_pow, Yh, SYY, oY
        Yx = xp.asarray(Y)
        SYY = xp.sum(Yx * Yx, axis=0)                # (n_batch,)
        oY = xp.sum(Yx, axis=0)
        n_batch = Yx.shape[1]
        # On-device running peak+argmax over frequency chunks: keep the
        # reduction on the backend and transfer only the final per-curve
        # peak power + index ONCE after the loop (avoids per-chunk GPU->host
        # copies that transfer-throttle the GPU).
        best_pow = xp.full((n_batch,), -1e30)
        best_idx = xp.zeros((n_batch,), dtype=xp.index_dtype)
        for start in range(0, self.n_freq, self.freq_chunk):
            stop = min(start + self.freq_chunk, self.n_freq)
            P = self._power_chunk(self.freqs[start:stop], Yx, SYY, oY)
            block_max = xp.max(P, axis=0)                     # (n_batch,)
            block_arg = xp.astype(xp.argmax(P, axis=0), xp.index_dtype)
            upd = block_max > best_pow
            best_pow = xp.where(upd, block_max, best_pow)
            best_idx = xp.where(upd, start + block_arg, best_idx)
        # ONE transfer at the very end (downstream indexes host self.freqs).
        best_idx = xp.to_numpy(best_idx).astype(int)
        best_pow = xp.to_numpy(best_pow)
        return best_idx, best_pow, Yx, SYY, oY

    def peak(self, Y):
        """Best period + peak power for each curve.

        Y : (n_epoch, n_batch) — host numpy or backend array.
        Returns (period[n_batch], peak_power[n_batch]) host numpy arrays.
        """
        best_idx, best_pow, _, _, _ = self._scan(Y)
        return 1.0 / self.freqs[best_idx], best_pow

    def _local_power(self, y_cols, f_local):
        """Power of each curve at its OWN local frequency window (host numpy).

        y_cols  : (n_epoch, n_batch) host numpy (curve b is column b).
        f_local : (n_batch, L) frequencies at which to evaluate curve b.
        Returns (n_batch, L) power via the same floating-mean 'model' form.
        """
        t = self.t                                   # (n_epoch,)
        ang = 2.0 * np.pi * f_local[:, :, None] * t[None, None, :]   # (B,L,E)
        C = np.cos(ang); S = np.sin(ang)
        a = np.sum(C * C, axis=2); d = np.sum(S * S, axis=2)
        bb = np.sum(C * S, axis=2); cc = np.sum(C, axis=2); ee = np.sum(S, axis=2)
        ff = float(self.n_epoch)
        det = (a * (d * ff - ee * ee) - bb * (bb * ff - ee * cc)
               + cc * (bb * ee - d * cc))
        inv = 1.0 / det
        i00 = (d * ff - ee * ee) * inv; i11 = (a * ff - cc * cc) * inv
        i22 = (a * d - bb * bb) * inv;  i01 = (cc * ee - bb * ff) * inv
        i02 = (bb * ee - cc * d) * inv; i12 = (bb * cc - a * ee) * inv
        Yb = y_cols.T[:, None, :]                     # (B,1,E)
        cY = np.sum(C * Yb, axis=2); sY = np.sum(S * Yb, axis=2)
        oY = np.sum(y_cols, axis=0)[:, None]          # (B,1)
        SYY = np.sum(y_cols * y_cols, axis=0)[:, None]
        Q = (i00 * cY * cY + i11 * sY * sY + i22 * oY * oY
             + 2 * i01 * cY * sY + 2 * i02 * cY * oY + 2 * i12 * sY * oY)
        chi2_0 = SYY - oY * oY / ff
        chi2 = SYY - Q
        return (chi2_0 - chi2) / chi2

    def peak_with_fwhm(self, Y, half_window=80):
        """Best period + peak power + the peak's FWHM PERIOD bounds per curve.

        Replicates period_search.candidates._fwhm_bounds (half-power crossings,
        linearly interpolated) on a local window around each curve's peak, then
        returns (period, peak_power, fwhm_per_low, fwhm_per_high). The fit then
        lets Period vary in [fwhm_per_low, fwhm_per_high] (use_fwhm=True).
        """
        Yh = np.asarray(Y, dtype=np.float64)
        best_idx, best_pow, _, _, _ = self._scan(Yh)
        n_batch = Yh.shape[1]
        W = int(half_window)
        offs = np.arange(-W, W + 1)
        loc_idx = np.clip(best_idx[:, None] + offs[None, :], 0, self.n_freq - 1)  # (B,L)
        f_local = self.freqs[loc_idx]                                            # (B,L)
        P = self._local_power(Yh, f_local)                                       # (B,L)
        c = W                                          # center column = the peak
        half = (best_pow / 2.0)[:, None]
        below = P <= half                              # (B,L)
        cols = np.arange(2 * W + 1)
        # left crossing: largest below-index strictly left of center
        left_below = np.where(below & (cols[None, :] < c), cols[None, :], -1).max(axis=1)
        # right crossing: smallest below-index strictly right of center
        rmask = np.where(below & (cols[None, :] > c), cols[None, :], 2 * W + 1)
        right_below = rmask.min(axis=1)
        b = np.arange(n_batch)
        # interpolate fL between (left_below, left_below+1); fallback = window edge
        has_l = left_below >= 0
        il = np.clip(left_below, 0, 2 * W)
        p1 = P[b, il]; p2 = P[b, np.clip(il + 1, 0, 2 * W)]
        x1 = f_local[b, il]; x2 = f_local[b, np.clip(il + 1, 0, 2 * W)]
        tL = np.where(p2 != p1, (half[:, 0] - p1) / (p2 - p1), 0.0)
        fL = np.where(has_l, x1 + tL * (x2 - x1), f_local[:, 0])
        has_r = right_below <= 2 * W
        ir = np.clip(right_below, 0, 2 * W)
        p1r = P[b, np.clip(ir - 1, 0, 2 * W)]; p2r = P[b, ir]
        x1r = f_local[b, np.clip(ir - 1, 0, 2 * W)]; x2r = f_local[b, ir]
        tR = np.where(p2r != p1r, (half[:, 0] - p1r) / (p2r - p1r), 0.0)
        fR = np.where(has_r, x1r + tR * (x2r - x1r), f_local[:, -1])
        period = 1.0 / self.freqs[best_idx]
        fwhm_per_low = 1.0 / fR                        # higher freq -> lower period
        fwhm_per_high = 1.0 / fL
        return period, best_pow, fwhm_per_low, fwhm_per_high

    def max_power(self, Y):
        """Peak power only (for permutation nulls). Y:(n_epoch,n_batch)."""
        _, p = self.peak(Y)
        return p

    def power_spectrum(self, y):
        """Full power spectrum for a SINGLE curve (validation against astropy).

        y : (n_epoch,) -> (freqs, power) host numpy.
        """
        Y = np.asarray(y, dtype=np.float64)[:, None]
        xp = self.xp
        Yx = xp.asarray(Y)
        SYY = xp.sum(Yx * Yx, axis=0)
        oY = xp.sum(Yx, axis=0)
        out = np.empty(self.n_freq)
        for start in range(0, self.n_freq, self.freq_chunk):
            stop = min(start + self.freq_chunk, self.n_freq)
            P = self._power_chunk(self.freqs[start:stop], Yx, SYY, oY)
            out[start:stop] = self.xp.to_numpy(P)[:, 0]
        return self.freqs, out
