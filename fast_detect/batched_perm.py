"""Batched permutation false-alarm probability (iter_fap).

The pipeline's empirical FAP (``period_search.candidates`` /
``period_search.permutation.ls_permutation_max_powers_mp``) shuffles the RV
values across epochs, recomputes the Lomb-Scargle max-power for each shuffle, and
sets

    iter_fap = (# shuffles with max-power >= candidate peak power) / n_perm

The production code does this with a ``ProcessPoolExecutor`` PER detection — which
inside the bias grid means nested multiprocessing across 94 workers. Here it is a
single batched Lomb-Scargle over an ``(n_epoch × (n_batch · n_perm))`` matrix of
shuffled curves: the most GPU/BLAS-friendly part of the whole detector.

Parity note: matching the production iter_fap *exactly* (same RNG draws) is neither
needed nor sensible — iter_fap is a Monte-Carlo statistic and the bias grid only
consumes the per-cell detection fraction. We reproduce the same null *distribution*
(same shuffling, same LS), which is what governs the iter_fap < 0.2 gate.
"""

import numpy as np


def _make_shuffles(Y, n_perm, rng):
    """Build the shuffled-curve matrix.

    Y : (n_epoch, n_batch). For each curve and each of n_perm permutations,
    independently permute the epoch values (times are held fixed).
    Returns (n_epoch, n_batch * n_perm) with column order
    [curve0·perm0, curve0·perm1, ..., curve1·perm0, ...].
    """
    n_epoch, n_batch = Y.shape
    # (n_batch, n_perm, n_epoch) independent permutations of epoch indices
    idx = np.argsort(rng.random((n_batch, n_perm, n_epoch)), axis=2)
    # gather: out[b,p,:] = Y[idx[b,p,:], b]
    Yt = Y.T  # (n_batch, n_epoch)
    shuffled = np.take_along_axis(Yt[:, None, :], idx, axis=2)  # (n_batch,n_perm,n_epoch)
    out = shuffled.reshape(n_batch * n_perm, n_epoch).T          # (n_epoch, n_batch*n_perm)
    return np.ascontiguousarray(out)


def iter_fap_fn(maxpow_fn, Y, peak_power, n_perm, rng, perm_chunk=20000,
                col_arg=None):
    """Empirical permutation FAP per curve, for any periodogram.

    maxpow_fn : callable (n_epoch, m) -> (m,) max-power over the freq grid for
                each shuffled curve (e.g. BatchedLS.max_power). If ``col_arg`` is
                given (length n_batch*n_perm, e.g. per-column errors for PDC),
                maxpow_fn is called as maxpow_fn(chunk, col_arg_chunk).
    Returns (n_batch,) iter_fap in [0, 1].
    """
    Y = np.asarray(Y, dtype=np.float64)
    n_epoch, n_batch = Y.shape
    Yp = _make_shuffles(Y, n_perm, rng)             # (n_epoch, n_batch*n_perm)
    total = Yp.shape[1]
    ca = None if col_arg is None else np.asarray(col_arg)
    maxp = np.empty(total)
    for s in range(0, total, perm_chunk):
        e = min(s + perm_chunk, total)
        maxp[s:e] = (maxpow_fn(Yp[:, s:e]) if ca is None
                     else maxpow_fn(Yp[:, s:e], ca[s:e]))
    maxp = maxp.reshape(n_batch, n_perm)            # (n_batch, n_perm)
    return (maxp >= peak_power[:, None]).sum(axis=1) / float(n_perm)


def iter_fap(bls, Y, peak_power, n_perm, rng, perm_chunk=20000):
    """LS convenience wrapper (see iter_fap_fn)."""
    return iter_fap_fn(bls.max_power, Y, peak_power, n_perm, rng, perm_chunk)
