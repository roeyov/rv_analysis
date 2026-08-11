"""
Permutation-based false-alarm probability estimation for periodograms.

Supports both Lomb-Scargle (multiprocessing) and PDC (sequential Numba-accelerated,
or batched GEMM — see pdc_permutation_max_powers_batched).
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import tqdm
from scipy import special

from period_search.periodogram import ls, pdc_opt, make_pdc_freq_grid
# These globals are initialized once per worker process (fast, avoids repeated pickling)
_WORK_MJDS = None
_WORK_ERR = None
_WORK_PMIN = None
_WORK_PMAX = None
_WORK_NORM = None
_WORK_LS_METHOD = None
_WORK_FA_METHOD = None
_WORK_CENTER_DATA = None
_WORK_RANDOM_STATE = None


def _init_worker(
    mjds,
    err_vs,
    pmin,
    pmax,
    norm,
    ls_method,
    fa_method,
    center_data,
    random_state,
):
    global _WORK_MJDS, _WORK_ERR, _WORK_PMIN, _WORK_PMAX
    global _WORK_NORM, _WORK_LS_METHOD, _WORK_FA_METHOD, _WORK_CENTER_DATA, _WORK_RANDOM_STATE

    _WORK_MJDS = np.asarray(mjds)
    _WORK_ERR = np.asarray(err_vs)
    _WORK_PMIN = float(pmin)
    _WORK_PMAX = float(pmax)
    _WORK_NORM = norm
    _WORK_LS_METHOD = ls_method
    _WORK_FA_METHOD = fa_method
    _WORK_CENTER_DATA = bool(center_data)
    _WORK_RANDOM_STATE = random_state


def _one_perm(seed_and_rvs):
    seed, rvs = seed_and_rvs
    rng = np.random.default_rng(seed)
    mixed_rv = rng.permutation(rvs)

    # Keep your original unpacking: (_, iter_ls_max_power, _, _, _, _, _)
    _, iter_ls_max_power, _, _, _, _, _ = ls(
        _WORK_MJDS,
        mixed_rv,
        data_err=_WORK_ERR,
        pmin=_WORK_PMIN,
        pmax=_WORK_PMAX,
        norm=_WORK_NORM,
        ls_method=_WORK_LS_METHOD,
        fa_method=_WORK_FA_METHOD,
        center_data=_WORK_CENTER_DATA,
        random_state=_WORK_RANDOM_STATE,
    )
    return float(iter_ls_max_power)




# from your_module import pdc_opt  # <--- Import your optimized Numba function

def pdc_permutation_max_powers(
        *,
        rvs,
        mjds,
        err_vs,
        n_iter,
        pmin,
        pmax,
        probabilities=(0.5, 0.01, 0.001),
        random_state=12345,
        show_progress=True,
        samples_per_peak=10,
):
    """
    Run PDC on random permutations using a standard loop.
    Since 'pdc_opt' is already parallelized internally via Numba,
    this sequential loop is the most efficient approach on M4 Max.
    """
    rvs = np.asarray(rvs)
    mjds = np.asarray(mjds)
    # Handle empty errors safely
    if err_vs is None or len(err_vs) == 0:
        err_vs = np.array([])
    else:
        err_vs = np.asarray(err_vs)

    n_iter = int(n_iter)
    if n_iter <= 0:
        return np.asarray([], dtype=float)

    # 1. Setup Random Seeds (Reproducibility)
    base_seed = 12345 if random_state is None else int(random_state)
    ss = np.random.SeedSequence(base_seed)
    # Generate a stream of distinct seeds for each iteration
    child_seeds = ss.spawn(n_iter)

    results = np.zeros(n_iter, dtype=float)

    # 2. The Loop
    # We create the iterator first to optionally wrap it in tqdm
    iterator = range(n_iter)
    if show_progress:
        iterator = tqdm.tqdm(iterator, total=n_iter, desc="PDC Permutations (M4 Optimized)")

    for i in iterator:
        # Create specific RNG for this iteration
        rng_seed = child_seeds[i]
        rng = np.random.default_rng(rng_seed)

        # Shuffle RVS
        mixed_rv = rng.permutation(rvs)

        # Run Optimized PDC
        # pdc_opt uses all cores, so we wait for it to finish
        _, max_pow, _, _, _, _, _ = pdc_opt(
            mjds,
            mixed_rv,
            data_err=err_vs,
            pmin=pmin,
            pmax=pmax,
            probabilities=probabilities,
            samples_per_peak=samples_per_peak,
        )

        results[i] = max_pow

    return results
def ls_permutation_max_powers_mp(
    *,
    rvs,
    mjds,
    err_vs,
    n_iter,
    pmin,
    pmax,
    norm,
    ls_method,
    fa_method,
    center_data=True,
    random_state=12345,
    n_workers=None,
    show_progress=False,
):
    """
    Run LS on random permutations of rvs (keeping mjds fixed) in multiprocessing,
    returning an array of iter_ls_max_power values (length = n_iter).

    Parameters
    ----------
    show_progress : bool
        If True, uses tqdm for progress. Pass tqdm.tqdm if you want it.
    """

    rvs = np.asarray(rvs)
    mjds = np.asarray(mjds)
    err_vs = np.asarray(err_vs)

    n_iter = int(n_iter)
    if n_iter <= 0:
        return np.asarray([], dtype=float)

    if n_workers is None:
        n_workers = os.cpu_count()-1 or 1
    n_workers = int(n_workers)

    # Inside a bias-grid subprocess, force sequential to avoid
    # nested multiprocessing deadlock on macOS fork.
    if os.environ.get("_BIAS_GRID_SUBPROCESS"):
        n_workers = 1

    # reproducible per-iteration seeds
    # if random_state is None -> still create deterministic default sequence
    base_seed = 12345 if random_state is None else int(random_state)
    ss = np.random.SeedSequence(base_seed)
    child = ss.spawn(n_iter)
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in child]

    # Run tasks
    results = np.empty(n_iter, dtype=float)

    if n_workers == 1:
        # Sequential — avoids forking (safe inside subprocess workers).
        _init_worker(mjds, err_vs, pmin, pmax, norm,
                     ls_method, fa_method, center_data, random_state)
        it = range(n_iter)
        if show_progress:
            it = tqdm.tqdm(it, total=n_iter, desc="LS permutations")
        for k in it:
            results[k] = _one_perm((seeds[k], rvs))
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(
                mjds,
                err_vs,
                pmin,
                pmax,
                norm,
                ls_method,
                fa_method,
                center_data,
                random_state,
            ),
        ) as ex:
            futures = [ex.submit(_one_perm, (seeds[i], rvs)) for i in range(n_iter)]

            it = as_completed(futures)
            if show_progress:
                it = tqdm.tqdm(it, total=len(futures), desc="LS permutations")

            k = 0
            for fut in it:
                results[k] = fut.result()
                k += 1

    return results


# ---------------------------------------------------------------------------
# Batched PDC permutation null
#
# `pdc_permutation_max_powers` calls the Numba `pdc_opt` kernel once per
# shuffle, which dominates the per-star runtime (~250 s at n_iter=10000).
# Every shuffle shares the same MJDs and the same per-epoch errors (only the
# RV values are permuted), so the phase-distance matrices B(f) — the expensive,
# frequency-dependent part — are common to all shuffles and the whole frequency
# scan collapses into one GEMM per frequency chunk:
#
#     power(f, k) = <A_k, B(f)> / sqrt(||A_k||^2 * ||B(f)||^2)
#
# with A_k the U-centered value-distance matrix of shuffle k. This is the
# design used by fast_detect.batched_pdc for the bias grid, generalised here
# from one scalar error per curve to the real per-epoch error vector.
#
# The shuffles themselves are replayed from the production seed stream, so this
# path scores the *same* permutations as the sequential one.
# ---------------------------------------------------------------------------

def _replay_pdc_perm_indices(random_state, n_iter, n_epoch):
    """
    Reproduce the shuffles drawn by `pdc_permutation_max_powers`.

    That function seeds one Generator per iteration from
    ``SeedSequence(random_state).spawn(n_iter)`` and calls ``rng.permutation(rvs)``.
    Generator.permutation shuffles positions, so drawing ``permutation(n_epoch)``
    from an identically seeded Generator yields the index array `p` with
    ``rvs[p] == rng.permutation(rvs)``.

    Returns (n_iter, n_epoch) int array.
    """
    base_seed = 12345 if random_state is None else int(random_state)
    ss = np.random.SeedSequence(base_seed)
    child_seeds = ss.spawn(int(n_iter))

    idx = np.empty((int(n_iter), int(n_epoch)), dtype=np.int64)
    for i, child in enumerate(child_seeds):
        idx[i] = np.random.default_rng(child).permutation(int(n_epoch))
    return idx


def _u_center_batch(a):
    """
    U-centering of a batch of distance matrices, matching
    PDC.pdc_func.unbiased_u_centering_fast (which zeroes the diagonal).

    a : (..., n, n) -> same shape, U-centered.
    """
    n = a.shape[-1]
    row = a.sum(axis=-1, keepdims=True) / (n - 2)
    col = a.sum(axis=-2, keepdims=True) / (n - 2)
    tot = a.sum(axis=(-2, -1), keepdims=True) / ((n - 1) * (n - 2))
    out = a - row - col + tot
    d = np.arange(n)
    out[..., d, d] = 0.0
    return out


def _value_distance_uc_perepoch(Yb, err_vs):
    """
    U-centered value-distance matrices for a batch of RV curves, with
    **per-epoch** errors — the batched form of
    PDC.pdc_func.calc_pdc_distance_matrix_fast.

    Yb     : (n_batch, n_epoch) RV values (one row per shuffle)
    err_vs : (n_epoch,) per-epoch RV errors, or None/empty for the
             error-free branch (a = |v_i - v_j|).

    Returns A_uc (n_batch, n_epoch, n_epoch).
    """
    Yb = np.asarray(Yb, dtype=float)
    diff = Yb[:, :, None] - Yb[:, None, :]                 # (B,E,E) signed

    if err_vs is None or len(err_vs) == 0:
        a = np.abs(diff)
    else:
        e = np.asarray(err_vs, dtype=float)
        # Fixed across shuffles: depends only on the (unpermuted) errors.
        e_sq_sum = e[:, None] ** 2 + e[None, :] ** 2        # (E,E)
        den = np.sqrt(2.0 * e_sq_sum)                       # (E,E)
        yv = (e[:, None] + e[None, :]) / den                # (E,E)
        sqrt_e_sq_sum = np.sqrt(e_sq_sum)                   # (E,E)

        x = diff / den[None, :, :]                          # (B,E,E)
        ee = sqrt_e_sq_sum[None, :, :] * (
            np.exp(-x * x) + x * special.erf(x) - yv[None, :, :]
        )
        a = np.sqrt(np.maximum(ee, 0.0))

    d = np.arange(Yb.shape[1])
    a[:, d, d] = 0.0
    return _u_center_batch(a)


def pdc_permutation_max_powers_batched(
        *,
        rvs,
        mjds,
        err_vs,
        n_iter,
        pmin,
        pmax,
        random_state=12345,
        samples_per_peak=10,
        perm_chunk=2500,
        freq_chunk=128,
        show_progress=False,
):
    """
    Batched equivalent of `pdc_permutation_max_powers`.

    Scores the identical seeded shuffles, but evaluates the frequency scan as a
    GEMM over all permutations at once instead of one `pdc_opt` call each.
    Returns the (n_iter,) array of maximum PDC powers, drop-in compatible with
    the sequential implementation.
    """
    rvs = np.asarray(rvs, dtype=float)
    mjds = np.asarray(mjds, dtype=float)
    if err_vs is None or len(err_vs) == 0:
        err_vs = np.array([])
    else:
        err_vs = np.asarray(err_vs, dtype=float)

    n_iter = int(n_iter)
    if n_iter <= 0:
        return np.asarray([], dtype=float)

    n_epoch = rvs.size
    # Frequency grid identical to pdc_opt's.
    freqs = make_pdc_freq_grid(pmin, pmax, float(np.ptp(mjds)), samples_per_peak)
    n_freq = freqs.size
    tdiff = mjds[:, None] - mjds[None, :]                   # signed (E,E)

    # --- A side: one U-centered value-distance matrix per shuffle ---
    perm_idx = _replay_pdc_perm_indices(random_state, n_iter, n_epoch)
    A_flat = np.empty((n_iter, n_epoch * n_epoch), dtype=float)
    perm_chunk = max(1, int(perm_chunk))
    for s in range(0, n_iter, perm_chunk):
        e = min(s + perm_chunk, n_iter)
        A = _value_distance_uc_perepoch(rvs[perm_idx[s:e]], err_vs)
        A_flat[s:e] = A.reshape(e - s, -1)
    denA = np.einsum("ij,ij->i", A_flat, A_flat)            # ||A_k||^2
    A_T = A_flat.T                                          # (E*E, n_iter)

    # --- B side: shared phase-distance matrices, chunked over frequency ---
    best_pow = np.full(n_iter, -np.inf, dtype=float)
    freq_chunk = max(1, int(freq_chunk))
    blocks = range(0, n_freq, freq_chunk)
    if show_progress:
        blocks = tqdm.tqdm(blocks, total=(n_freq + freq_chunk - 1) // freq_chunk,
                           desc="PDC permutations (batched)")

    for s in blocks:
        e = min(s + freq_chunk, n_freq)
        inv_f = (1.0 / freqs[s:e])[:, None, None]           # (F,1,1)
        mpd = np.remainder(tdiff[None, :, :], inv_f)
        b = mpd * (inv_f - mpd)
        B = _u_center_batch(b)                              # (F,E,E)
        B_flat = B.reshape(B.shape[0], -1)
        denB = np.einsum("ij,ij->i", B_flat, B_flat)        # (F,)

        num = B_flat @ A_T                                  # (F, n_iter)
        denom = np.sqrt(denB[:, None] * denA[None, :])
        # Production sets the power to 0 only for den == 0 and otherwise
        # divides, so a non-finite denominator propagates rather than being
        # silently zeroed; mirror that instead of treating NaN as "not > 0".
        safe = np.where(denom == 0, 1.0, denom)
        power = np.where(denom == 0, 0.0, num / safe)
        np.maximum(best_pow, power.max(axis=0), out=best_pow)

    return best_pow
