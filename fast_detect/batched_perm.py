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
                col_arg=None, active=None, fail_thresh=None):
    """Empirical permutation FAP per curve, for any periodogram.

    maxpow_fn : callable (n_epoch, m) -> (m,) max-power over the freq grid for
                each shuffled curve (e.g. BatchedLS.max_power). If ``col_arg`` is
                given (length n_batch*n_perm, e.g. per-column errors for PDC),
                maxpow_fn is called as maxpow_fn(chunk, col_arg_chunk).
    active :    optional bool mask (n_batch,). When given, the expensive
                ``maxpow_fn`` is run ONLY for the shuffles of the active
                (short-circuit-surviving) curves; inactive curves return
                iter_fap = 1.0 (a definite fail — they are already excluded by
                the cheap prob_bicc / mass_flag gates upstream). The full-batch
                shuffle draw is kept UNCHANGED, so every active curve gets the
                exact same iter_fap it would in a full-batch call
                (bit-identical), while the periodogram work scales with the
                number of survivors, not n_batch.
    fail_thresh : the downstream detection gate (iter_fap < fail_thresh).
                When given, shuffles are scored in stages and a curve stops as
                soon as its exceedance count reaches
                ``ceil(fail_thresh * n_perm)`` — at that point the gate outcome
                is already a definite FAIL, so the remaining shuffles cannot
                change the detected mask. Early-stopped curves report the
                CENSORED count/n_perm (a lower bound, guaranteed >=
                fail_thresh); curves that pass the gate always score all
                n_perm shuffles and report the exact value. None = score
                everything (historical behaviour).
    Returns (n_batch,) iter_fap in [0, 1].
    """
    Y = np.asarray(Y, dtype=np.float64)
    n_epoch, n_batch = Y.shape
    Yp = _make_shuffles(Y, n_perm, rng)             # (n_epoch, n_batch*n_perm) — full draw
    ca = None if col_arg is None else np.asarray(col_arg)
    if active is None:
        act_inj = np.arange(n_batch)
    else:
        act_inj = np.flatnonzero(np.asarray(active, dtype=bool))
    ifap = np.ones(n_batch)                          # inactive -> 1.0 (fails iter_fap<thresh)
    if act_inj.size == 0:
        return ifap

    def _score_cols(cols):
        """maxpow over the given flattened shuffle columns, perm_chunk-ed."""
        Ya = Yp[:, cols]
        caa = None if ca is None else ca[cols]
        total = Ya.shape[1]
        maxp = np.empty(total)
        for s in range(0, total, perm_chunk):
            e = min(s + perm_chunk, total)
            maxp[s:e] = (maxpow_fn(Ya[:, s:e]) if caa is None
                         else maxpow_fn(Ya[:, s:e], caa[s:e]))
        return maxp

    # flattened shuffle columns for curve i are i*n_perm .. i*n_perm+n_perm
    if fail_thresh is None:
        act_cols = (act_inj[:, None] * n_perm
                    + np.arange(n_perm)[None, :]).ravel()
        maxp = _score_cols(act_cols).reshape(act_inj.size, n_perm)
        ifap[act_inj] = (maxp >= peak_power[act_inj, None]).sum(axis=1) \
            / float(n_perm)
        return ifap

    # Staged exact-safe early-stop: two probe stages of fail_k shuffles, then
    # the remainder. Per-column max-powers are identical to the one-shot path
    # (columns are scored independently), so surviving curves' iter_fap stays
    # bit-identical; only definite-fails skip their leftover shuffles.
    fail_k = max(int(np.ceil(fail_thresh * n_perm)), 1)
    edges = sorted({0, min(fail_k, n_perm), min(2 * fail_k, n_perm), n_perm})
    peaks = peak_power[act_inj]
    counts = np.zeros(act_inj.size, dtype=np.int64)
    alive = np.ones(act_inj.size, dtype=bool)        # not yet a definite fail
    for s0, s1 in zip(edges[:-1], edges[1:]):
        rows = np.flatnonzero(alive)
        if rows.size == 0:
            break
        cols = (act_inj[rows][:, None] * n_perm
                + np.arange(s0, s1)[None, :]).ravel()
        m = _score_cols(cols).reshape(rows.size, s1 - s0)
        counts[rows] += (m >= peaks[rows, None]).sum(axis=1)
        if s1 < n_perm:
            newly = counts[rows] >= fail_k
            if newly.any():
                dead = rows[newly]
                ifap[act_inj[dead]] = counts[dead] / float(n_perm)
                alive[dead] = False
    rows = np.flatnonzero(alive)                     # scored all n_perm
    ifap[act_inj[rows]] = counts[rows] / float(n_perm)
    return ifap


def iter_fap(bls, Y, peak_power, n_perm, rng, perm_chunk=20000):
    """LS convenience wrapper (see iter_fap_fn)."""
    return iter_fap_fn(bls.max_power, Y, peak_power, n_perm, rng, perm_chunk)
