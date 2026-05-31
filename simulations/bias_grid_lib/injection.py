"""Synthetic-binary injection + per-star multiprocessing workers.

Two worker shapes live here:

- ``_worker_star_injections`` — Python-level per-injection loop. Required
  by the ``pipeline`` detection backend (full periodogram + lmfit) where
  the inner detection call cannot be vectorized.
- ``_worker_star_injections_vectorized`` — NumPy/Numba batched worker for
  the constant-rv_err ``rv_threshold`` detector. Identical return-dict
  schema so the engine can swap backends transparently.

``inject_and_detect`` and ``detect_single_star`` are kept around as the
two-arg primitives used by external scripts (e.g. closure-test
generation).
"""

import os
import numpy as np

from simulations.bias_grid_lib.constants import (
    TWOPI, _HIST_BINS, _HIST_NBINS, _HIST_PAIRS,
)
from simulations.bias_grid_lib.detection import (
    DETECTION_METHODS, _detect_full_pipeline,
)
from simulations.bias_grid_lib.physics import (
    compute_K1, compute_K1_batch, kepler_E_batch, powerlaw_draw,
    rv_model_jit,
)


# ---------------------------------------------------------------------------
# Injection & detection
# ---------------------------------------------------------------------------

def inject_and_detect(MJDs, rv_err, P_days, e, q, M1, R1, incl, omega, T0,
                      gamma, rng, args_dict,
                      detect_fn=None, K1=None):
    """
    Inject a synthetic binary signal and run the detection function.

    Returns (detected, info_dict).
    """
    if detect_fn is None:
        detect_fn = _detect_full_pipeline

    # Use pre-computed K1 if provided, otherwise compute it
    if K1 is None:
        K1 = compute_K1(P_days, e, q, M1, incl)

    # Synthesize noise-free RV
    rv_true = rv_model_jit(MJDs, P_days, T0, omega, e, K1, gamma)

    # Add noise
    noise = rng.normal(0.0, rv_err)
    rv_obs = rv_true + noise

    # Run detection
    detected, info = detect_fn(MJDs, rv_obs, rv_err, args_dict)
    return detected, info


def detect_single_star(MJDs, rv_err, gamma, rng, args_dict,
                       detect_fn=None):
    """
    Generate noise-only RV data (single star) and check for false positive.
    """
    if detect_fn is None:
        detect_fn = _detect_full_pipeline

    rv_obs = gamma + rng.normal(0.0, rv_err)

    detected, info = detect_fn(MJDs, rv_obs, rv_err, args_dict)
    return detected, info


def _worker_star_injections(task):
    """Process all injections for one star (top-level for multiprocessing).

    Each worker gets a deterministic seed so results are reproducible
    regardless of n_workers.

    Returns dict with per-star detection counts, detected param arrays,
    and 2D detection-probability histograms.
    """
    (star_seed, MJDs_raw, rv_err_val, M1, R1, gamma,
     n_inject, f_bin, pi, kappa, eta, cfg, args_dict,
     detect_method) = task

    MJDs = np.asarray(MJDs_raw, dtype=np.float64)

    # Prevent nested multiprocessing: the pipeline's permutation code
    # uses ProcessPoolExecutor, which deadlocks inside a forked worker.
    # Setting this env var is checked by period_search/permutation.py
    # to force n_workers=1.
    os.environ["_BIAS_GRID_SUBPROCESS"] = "1"

    detect_fn = DETECTION_METHODS[detect_method]
    rng = np.random.default_rng(star_seed)
    rv_err = np.full(len(MJDs), rv_err_val)

    # Precompute effective threshold for constant-rv_err max-min detection
    if detect_method == "rv_threshold":
        drv_thresh = args_dict.get("drv_threshold", 20.0)
        sign_thresh = args_dict.get("significance_threshold", 4.0)
        eff_thresh = max(drv_thresh, sign_thresh * rv_err_val * np.sqrt(2.0))
        args_dict = {**args_dict, "_rv_eff_threshold": eff_thresh}

    logP_det = []
    e_det = []
    K1_det = []
    q_det = []
    logP_nondet = []
    e_nondet = []
    K1_nondet = []
    q_nondet = []
    logP_rlof = []
    q_rlof = []
    e_rlof = []
    M1_rlof = []
    R1_rlof = []
    n_physical = 0
    n_detected = 0
    n_rlof = 0
    n_false_positive = 0

    # 2D histogram accumulators for detection probability maps
    hist_total = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                  for pair in _HIST_PAIRS}
    hist_det = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                for pair in _HIST_PAIRS}

    for _ in range(n_inject):
        is_binary = rng.random() < f_bin

        if is_binary:
            logP = powerlaw_draw(1, pi, cfg["log_p_min"],
                                 cfg["log_p_max"], rng)[0]
            P = 10.0 ** logP
            q = powerlaw_draw(1, kappa, cfg["q_min"],
                              cfg["q_max"], rng)[0]
            e = powerlaw_draw(1, eta, 1e-6, cfg["e_max"], rng)[0]
            if P < cfg["p_circ"]:
                e = 0.0

            cos_i = rng.uniform(0, 1)
            incl = np.arccos(cos_i)
            omega = rng.uniform(0, TWOPI)
            T0 = float(rng.uniform(MJDs.min() - P, MJDs.min()))

            M1_j = max(7.0, min(80.0, rng.normal(M1, 0.3 * M1)))
            R1_j = max(4.0, rng.normal(R1, 0.3 * R1))

            # Compute K1 before detection so it's available for histograms
            K1_val = compute_K1(P, e, q, M1_j, incl)

            det, info = inject_and_detect(
                MJDs, rv_err, P, e, q, M1_j, R1_j,
                incl, omega, T0, gamma, rng, args_dict,
                detect_fn=detect_fn, K1=K1_val,
            )

            if info.get("reason") == "RLOF":
                n_rlof += 1
                n_physical += 1
                n_detected += 1
                # RLOF binaries would be trivially detected but don't
                # have clean orbital solutions — exclude from CDF arrays
                # (logP_det etc.) so KS/AD/CvM tests are unaffected.
                logP_rlof.append(logP)
                q_rlof.append(q)
                e_rlof.append(e)
                M1_rlof.append(M1_j)
                R1_rlof.append(R1_j)
            else:
                n_physical += 1

                # Bin into 2D histograms (all physical binaries)
                vals = {"logP": logP, "e": e, "K1": K1_val, "q": q}
                for (a, b) in _HIST_PAIRS:
                    ia = max(0, min(np.searchsorted(_HIST_BINS[a], vals[a]) - 1,
                                   _HIST_NBINS - 1))
                    ib = max(0, min(np.searchsorted(_HIST_BINS[b], vals[b]) - 1,
                                   _HIST_NBINS - 1))
                    hist_total[(a, b)][ia, ib] += 1
                    if det:
                        hist_det[(a, b)][ia, ib] += 1

                if det:
                    n_detected += 1
                    logP_det.append(logP)
                    e_det.append(e)
                    K1_det.append(K1_val)
                    q_det.append(q)
                else:
                    logP_nondet.append(logP)
                    e_nondet.append(e)
                    K1_nondet.append(K1_val)
                    q_nondet.append(q)
        else:
            det, info = detect_single_star(
                MJDs, rv_err, gamma, rng, args_dict,
                detect_fn=detect_fn,
            )
            n_physical += 1
            if det:
                n_false_positive += 1

    return {
        "n_physical": n_physical,
        "n_detected": n_detected,
        "n_rlof": n_rlof,
        "n_false_positive": n_false_positive,
        "logP_det": logP_det,
        "e_det": e_det,
        "K1_det": K1_det,
        "q_det": q_det,
        "logP_nondet": logP_nondet,
        "e_nondet": e_nondet,
        "K1_nondet": K1_nondet,
        "q_nondet": q_nondet,
        "logP_rlof": logP_rlof,
        "q_rlof": q_rlof,
        "e_rlof": e_rlof,
        "M1_rlof": M1_rlof,
        "R1_rlof": R1_rlof,
        "hist_total": hist_total,
        "hist_det": hist_det,
    }


def _worker_star_injections_vectorized(task):
    """Vectorized: process all injections for one star in batch.

    Replaces the Python-level per-injection loop with NumPy/Numba
    batch operations.  Only valid for detect_method='rv_threshold'.

    Returns the same dict schema as _worker_star_injections.
    """
    (star_seed, MJDs_raw, rv_err_val, M1, R1, gamma,
     n_inject, f_bin, pi, kappa, eta, cfg, args_dict,
     detect_method) = task

    MJDs = np.asarray(MJDs_raw, dtype=np.float64)
    rng = np.random.default_rng(star_seed)
    n_epochs = len(MJDs)

    # Precompute detection threshold (constant rv_err -> scalar sigma)
    drv_thresh = args_dict.get("drv_threshold", 20.0)
    sign_thresh = args_dict.get("significance_threshold", 4.0)
    eff_thresh = max(drv_thresh, sign_thresh * rv_err_val * np.sqrt(2.0))

    # --- Draw all binary/single decisions at once ---
    is_binary = rng.random(n_inject) < f_bin
    n_bin = int(is_binary.sum())
    n_single = n_inject - n_bin

    # --- Binary branch (all at once) ---
    if n_bin > 0:
        logP = powerlaw_draw(n_bin, pi, cfg["log_p_min"],
                             cfg["log_p_max"], rng)
        P = 10.0 ** logP
        q_arr = powerlaw_draw(n_bin, kappa, cfg["q_min"],
                              cfg["q_max"], rng)
        e_arr = powerlaw_draw(n_bin, eta, 1e-6, cfg["e_max"], rng)
        e_arr[P < cfg["p_circ"]] = 0.0

        cos_i = rng.uniform(0, 1, n_bin)
        incl = np.arccos(cos_i)
        omega = rng.uniform(0, TWOPI, n_bin)
        T0_arr = np.array([rng.uniform(MJDs.min() - P[j], MJDs.min())
                           for j in range(n_bin)])

        M1_j = np.clip(rng.normal(M1, 0.3 * M1, n_bin), 7.0, 80.0)
        R1_j = np.maximum(4.0, rng.normal(R1, 0.3 * R1, n_bin))

        # Batch K1
        K1_arr = compute_K1_batch(P, e_arr, q_arr, M1_j, incl)

        # Batch RV model: phase -> mean anomaly -> Kepler -> true anomaly -> RV
        phase = ((MJDs[None, :] - T0_arr[:, None]) / P[:, None]) % 1.0
        M_anom = TWOPI * phase  # (n_bin, n_epochs)

        E = kepler_E_batch(M_anom, e_arr)  # (n_bin, n_epochs)
        nu = 2.0 * np.arctan2(
            np.sqrt(1.0 + e_arr[:, None]) * np.sin(E / 2.0),
            np.sqrt(1.0 - e_arr[:, None]) * np.cos(E / 2.0))

        # Handle circular orbits: for e < 1e-8, E = M and nu = M
        circ = e_arr < 1e-8
        if circ.any():
            nu[circ] = M_anom[circ]

        rv_true = gamma + K1_arr[:, None] * (
            np.cos(nu + omega[:, None]) +
            e_arr[:, None] * np.cos(omega[:, None]))

        # Batch noise + detection
        noise = rng.normal(0.0, rv_err_val, (n_bin, n_epochs))
        rv_obs = rv_true + noise
        detected_bin = (rv_obs.max(axis=1) - rv_obs.min(axis=1)) > eff_thresh

        # Classify
        n_physical = n_bin
        n_detected = int(detected_bin.sum())
        det_mask = detected_bin

        logP_det = logP[det_mask].tolist()
        e_det = e_arr[det_mask].tolist()
        K1_det = K1_arr[det_mask].tolist()
        q_det = q_arr[det_mask].tolist()

        logP_nondet = logP[~det_mask].tolist()
        e_nondet = e_arr[~det_mask].tolist()
        K1_nondet = K1_arr[~det_mask].tolist()
        q_nondet = q_arr[~det_mask].tolist()

        # Batch histogram binning
        hist_total = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                      for pair in _HIST_PAIRS}
        hist_det = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                    for pair in _HIST_PAIRS}

        vals = {"logP": logP, "e": e_arr, "K1": K1_arr, "q": q_arr}
        for (a, b) in _HIST_PAIRS:
            ia = np.clip(np.searchsorted(_HIST_BINS[a], vals[a]) - 1,
                         0, _HIST_NBINS - 1)
            ib = np.clip(np.searchsorted(_HIST_BINS[b], vals[b]) - 1,
                         0, _HIST_NBINS - 1)
            np.add.at(hist_total[(a, b)], (ia, ib), 1)
            np.add.at(hist_det[(a, b)], (ia[det_mask], ib[det_mask]), 1)
    else:
        n_physical = 0
        n_detected = 0
        logP_det, e_det, K1_det, q_det = [], [], [], []
        logP_nondet, e_nondet, K1_nondet, q_nondet = [], [], [], []
        hist_total = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                      for pair in _HIST_PAIRS}
        hist_det = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                    for pair in _HIST_PAIRS}

    # --- Single star branch (false positives) ---
    n_false_positive = 0
    if n_single > 0:
        n_physical += n_single
        noise_single = rng.normal(0.0, rv_err_val, (n_single, n_epochs))
        rv_obs_single = gamma + noise_single
        detected_single = (rv_obs_single.max(axis=1) -
                           rv_obs_single.min(axis=1)) > eff_thresh
        n_false_positive = int(detected_single.sum())

    return {
        "n_physical": n_physical,
        "n_detected": n_detected,
        "n_rlof": 0,
        "n_false_positive": n_false_positive,
        "logP_det": logP_det,
        "e_det": e_det,
        "K1_det": K1_det,
        "q_det": q_det,
        "logP_nondet": logP_nondet,
        "e_nondet": e_nondet,
        "K1_nondet": K1_nondet,
        "q_nondet": q_nondet,
        "logP_rlof": [],
        "q_rlof": [],
        "e_rlof": [],
        "M1_rlof": [],
        "R1_rlof": [],
        "hist_total": hist_total,
        "hist_det": hist_det,
    }
