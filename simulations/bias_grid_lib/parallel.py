"""Multiprocessing entry points + shared-state initializers.

The two pool-initializer functions stash heavy per-worker data
(``field_mjds``, ``rv_err_arr``, the scoring context …) in
module-level dicts. Forked workers inherit these dicts and never pay
the per-task IPC cost of pickling them again.

All workers + initializers live at module scope so ``forkserver`` can
locate them by qualified name across process boundaries.
"""

import os
import numpy as np

from simulations.bias_grid_lib.constants import _HIST_NBINS, _HIST_PAIRS
from simulations.bias_grid_lib.checkpointing import (
    _load_det_shard, _save_det_shard,
)
from simulations.bias_grid_lib.injection import (
    _worker_star_injections, _worker_star_injections_vectorized,
)
from simulations.bias_grid_lib.scoring import _compute_scores


# ---------------------------------------------------------------------------
# Shared state for parallel grid workers (set via pool initializer to avoid
# pickling large arrays into every task tuple — fixes BrokenPipeError).
# ---------------------------------------------------------------------------
_shared = {}
_resume_shared = {}


def _init_grid_worker(field_mjds, field_arr, rv_err_arr, M1_arr, R1_arr,
                      gamma_arr, cfg, args_dict, detect_method,
                      checkpoint_dir=None, scoring_ctx=None):
    """Pool initializer: stash shared data in module-level dict."""
    _shared["field_mjds"] = field_mjds
    _shared["field_arr"] = field_arr
    _shared["rv_err_arr"] = rv_err_arr
    _shared["M1_arr"] = M1_arr
    _shared["R1_arr"] = R1_arr
    _shared["gamma_arr"] = gamma_arr
    _shared["cfg"] = cfg
    _shared["args_dict"] = args_dict
    _shared["detect_method"] = detect_method
    _shared["checkpoint_dir"] = checkpoint_dir
    _shared["scoring_ctx"] = scoring_ctx


def _init_resume_worker(checkpoint_dir, scoring_ctx):
    """Pool initializer for parallel resume re-scoring.

    Stash the scoring context once per worker so per-task IPC stays small.
    """
    _resume_shared["checkpoint_dir"] = checkpoint_dir
    _resume_shared["scoring_ctx"] = scoring_ctx


def _resume_score_worker(task):
    """Worker for parallel resume re-scoring.

    Loads one shard from disk, reconstructs the per-shard histograms,
    runs all KS/AD/CvM tests via _compute_scores, and returns a small
    dict the main process can fold into cubes/hists/CSV. No mutation
    of any shared state happens in the worker.

    Returns None if the shard file vanished between discovery and load
    (race with another process / manual deletion).
    """
    from simulations.bias_grid_lib.checkpointing import _hists_from_shard
    step, i, j, k, l, pi, kappa, eta, fbin = task
    checkpoint_dir = _resume_shared["checkpoint_dir"]
    ctx = _resume_shared["scoring_ctx"]
    shard = _load_det_shard(checkpoint_dir, step)
    if shard is None:
        return None
    hist_total, hist_det = _hists_from_shard(shard)
    # Prefer the persisted scalar counts; fall back to len(arrays) for
    # shards written before those scalars were saved (the fallback
    # under-counts because RLOF binaries and single stars are absent
    # from logP_det/logP_nondet but present in the original n_physical).
    n_det_arr = len(shard.get("logP", []))
    n_nondet_arr = len(shard.get("logP_nondet", []))
    n_physical = int(shard.get("n_physical", n_det_arr + n_nondet_arr))
    n_detected = int(shard.get("n_detected", n_det_arr))
    n_rlof = int(shard.get("n_rlof", 0))
    n_false_positive = int(shard.get("n_false_positive", 0))
    p_det = n_detected / max(n_physical, 1)
    res_for_scoring = {
        "p_det": p_det,
        "n_physical": n_physical,
        "logP_det": shard.get("logP", np.array([])),
        "e_det": shard.get("e", np.array([])),
        "K1_det": shard.get("K1", np.array([])),
    }
    scores = _compute_scores(res_for_scoring, ctx)
    return {
        "step": step, "i": i, "j": j, "k": k, "l": l,
        "pi": pi, "kappa": kappa, "eta": eta, "fbin": fbin,
        "p_det": p_det,
        "n_detected": n_detected,
        "n_physical": n_physical,
        "n_rlof": n_rlof,
        "n_false_positive": n_false_positive,
        "hist_total": hist_total,
        "hist_det": hist_det,
        "scores": scores,
    }


def _worker_grid_point(task):
    """Process one grid point (top-level for multiprocessing).

    Runs all stars sequentially within the grid point, so each CPU
    handles one grid point at a time.

    Returns (step, i, j, k, l, result_dict).
    """
    (step, i, j, k, l, pi, kappa, eta, f_bin,
     seed, n_inject) = task

    # Retrieve shared data set by _init_grid_worker.
    field_mjds = _shared["field_mjds"]
    field_arr = _shared["field_arr"]
    rv_err_arr = _shared["rv_err_arr"]
    M1_arr = _shared["M1_arr"]
    R1_arr = _shared["R1_arr"]
    gamma_arr = _shared["gamma_arr"]
    cfg = _shared["cfg"]
    args_dict = _shared["args_dict"]
    detect_method = _shared["detect_method"]

    os.environ["_BIAS_GRID_SUBPROCESS"] = "1"

    rng = np.random.default_rng(seed + step * 137)

    # Build one task per star (same logic as _run_one_grid_point).
    star_tasks = []
    for fld in sorted(field_mjds.keys()):
        fld_mask = (field_arr == fld)
        if not fld_mask.any():
            continue
        MJDs = field_mjds[fld]
        for star_idx in np.where(fld_mask)[0]:
            star_seed = int(rng.integers(0, 2**63))
            star_tasks.append((
                star_seed, MJDs,
                float(rv_err_arr[star_idx]),
                float(M1_arr[star_idx]),
                float(R1_arr[star_idx]),
                float(gamma_arr[star_idx]),
                n_inject, f_bin, pi, kappa, eta,
                cfg, args_dict, detect_method,
            ))

    # Run stars sequentially (parallelism is at the grid-point level).
    worker_fn = (_worker_star_injections_vectorized
                 if detect_method == "rv_threshold"
                 else _worker_star_injections)
    results = [worker_fn(t) for t in star_tasks]

    # Aggregate across stars.
    all_logP_det = []
    all_e_det = []
    all_K1_det = []
    all_q_det = []
    all_logP_nondet = []
    all_e_nondet = []
    all_K1_nondet = []
    all_q_nondet = []
    all_logP_rlof = []
    all_q_rlof = []
    all_e_rlof = []
    all_M1_rlof = []
    all_R1_rlof = []
    n_physical = 0
    n_detected = 0
    n_rlof = 0
    n_false_positive = 0
    agg_hist_total = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                      for pair in _HIST_PAIRS}
    agg_hist_det = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                    for pair in _HIST_PAIRS}

    for r in results:
        n_physical += r["n_physical"]
        n_detected += r["n_detected"]
        n_rlof += r["n_rlof"]
        n_false_positive += r["n_false_positive"]
        all_logP_det.extend(r["logP_det"])
        all_e_det.extend(r["e_det"])
        all_K1_det.extend(r["K1_det"])
        all_q_det.extend(r["q_det"])
        all_logP_nondet.extend(r["logP_nondet"])
        all_e_nondet.extend(r["e_nondet"])
        all_K1_nondet.extend(r["K1_nondet"])
        all_q_nondet.extend(r["q_nondet"])
        all_logP_rlof.extend(r["logP_rlof"])
        all_q_rlof.extend(r["q_rlof"])
        all_e_rlof.extend(r["e_rlof"])
        all_M1_rlof.extend(r["M1_rlof"])
        all_R1_rlof.extend(r["R1_rlof"])
        for pair in _HIST_PAIRS:
            agg_hist_total[pair] += r["hist_total"][pair]
            agg_hist_det[pair] += r["hist_det"][pair]

    p_det = n_detected / max(n_physical, 1)

    res = {
        "p_det": p_det,
        "n_detected": n_detected,
        "n_physical": n_physical,
        "n_rlof": n_rlof,
        "n_false_positive": n_false_positive,
        "logP_det": np.array(all_logP_det),
        "e_det": np.array(all_e_det),
        "K1_det": np.array(all_K1_det),
        "q_det": np.array(all_q_det),
        "logP_nondet": np.array(all_logP_nondet),
        "e_nondet": np.array(all_e_nondet),
        "K1_nondet": np.array(all_K1_nondet),
        "q_nondet": np.array(all_q_nondet),
        "logP_rlof": np.array(all_logP_rlof),
        "q_rlof": np.array(all_q_rlof),
        "e_rlof": np.array(all_e_rlof),
        "M1_rlof": np.array(all_M1_rlof),
        "R1_rlof": np.array(all_R1_rlof),
        "hist_total": agg_hist_total,
        "hist_det": agg_hist_det,
    }

    # When running inside a parallel-grid worker, save the shard here
    # and strip heavy arrays so only lightweight data travels through
    # the IPC pipe (prevents OOM in the main process).
    checkpoint_dir = _shared.get("checkpoint_dir")
    if checkpoint_dir:
        _save_det_shard(checkpoint_dir, step, i, j, k, l, res)

    # Score in-worker so the KS/AD/CvM work parallelizes across cores
    # instead of bottlenecking the main process. Mirrors the pattern
    # used by _resume_score_worker.
    scoring_ctx = _shared.get("scoring_ctx")
    if scoring_ctx is not None:
        res["scores"] = _compute_scores(res, scoring_ctx)

    if checkpoint_dir:
        # Drop everything _score_and_accumulate doesn't need when
        # `scores` is supplied: the det arrays are already in the shard
        # and main only reads p_det, n_*, and the small histograms.
        for _key in ("logP_det", "e_det", "K1_det", "q_det",
                     "logP_nondet", "e_nondet", "K1_nondet", "q_nondet",
                     "logP_rlof", "q_rlof", "e_rlof",
                     "M1_rlof", "R1_rlof"):
            res.pop(_key, None)

    return (step, i, j, k, l, res)
