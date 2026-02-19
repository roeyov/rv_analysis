"""
Permutation-based false-alarm probability estimation for periodograms.

Supports both Lomb-Scargle (multiprocessing) and PDC (sequential, Numba-accelerated).
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import tqdm

from period_search.periodogram import ls, pdc_opt
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
            probabilities=probabilities
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

    # reproducible per-iteration seeds
    # if random_state is None -> still create deterministic default sequence
    base_seed = 12345 if random_state is None else int(random_state)
    ss = np.random.SeedSequence(base_seed)
    child = ss.spawn(n_iter)
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in child]

    # Run tasks
    results = np.empty(n_iter, dtype=float)

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

        # Order doesn’t matter for your usage; store sequentially
        k = 0
        for fut in it:
            results[k] = fut.result()
            k += 1

    return results
