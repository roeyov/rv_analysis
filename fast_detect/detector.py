"""detect_batch — the full faithful detection gate, batched over a population.

Reproduces the bias-grid detection's *true* criterion (the production
``filter_expression``, configs/params.yaml): a system is detected iff its
candidate solution satisfies

    prob_bicc > 0.5  AND  mass_flag_peri == 0  AND  iter_fap < 0.2

evaluated for a whole batch of injected RV curves that share one field's MJDs.
(The current ``detect_method: pipeline`` applies only prob_bicc > 0.5; this is
the faithful gate the fast detector implements.)

The Roche-lobe term is switchable: ``cfg["apply_rlof_gate"] = False`` drops the
``mass_flag_peri`` condition, so detection reduces to
``prob_bicc > 0.5 AND iter_fap < fap_thresh`` and KILL_RLOF never appears in
the kill flags. Default True = the faithful gate above.

v1 reproduces the **LS** candidate branch. The PDC branch
(``PDC_iter_fap < 0.2``) is a follow-up (batched PDC periodogram); LS is the
cheaper, dominant branch and the one validated first.
"""

import numpy as np

from fast_detect.backends import get_xp
from fast_detect.batched_ls import BatchedLS
from fast_detect.batched_pdc import BatchedPDC
from fast_detect import batched_fit, batched_perm, rlof

# Per-injection kill-flag codes: which gate killed the detection. Must stay in
# sync with simulations/bias_grid_lib/constants.py KILL_FLAG_LABELS (fast_detect
# is deliberately standalone, so the codes are duplicated, not imported).
KILL_DETECTED = 0   # passed all gates
KILL_RLOF = 1       # Roche-lobe overflow of the fitted solution (mass_flag_peri)
KILL_BICC = 2       # prob_bicc <= 0.5
KILL_FAP = 3        # permutation iter_fap >= fap_thresh


def _peri_params(cfg):
    p = cfg.get("periodogram_params", {})
    return (float(p.get("min_period", 0.4)),
            float(p.get("max_period", 10000.0)),
            int(p.get("ls_sample_per_peak", 50)),
            int(p.get("window_iterations", 10)),
            int(p.get("pdc_samples_per_peak", 10)))


# The LS/PDC periodogram objects depend only on (MJDs, bounds, sampling) — NOT on
# the RV data — so they are identical for every star in a field and every grid
# cell. Cache them (there are only ~8 distinct BLOeM MJD arrays) so a per-star
# batched detect call doesn't rebuild the 96k/19k-frequency grids each time.
_PGRAM_CACHE = {}


def _get_pgrams(mjds, pmin, pmax, spp, pdc_spp, backend):
    mjds = np.asarray(mjds, dtype=np.float64)
    key = (mjds.tobytes(), pmin, pmax, spp, pdc_spp, backend)
    hit = _PGRAM_CACHE.get(key)
    if hit is None:
        xp = get_xp(backend)
        hit = (BatchedLS(mjds, pmin, pmax, samples_per_peak=spp, xp=xp),
               BatchedPDC(mjds, pmin, pmax, samples_per_peak=pdc_spp, xp=xp))
        _PGRAM_CACHE[key] = hit
    return hit


def _branch(pgram, Y, errs, mass_rows, cfg, n_perm, rng, fap_thresh,
            fit_kwargs, has_err, iter_seed, exclude=None, apply_rlof=True):
    """One candidate branch (LS or PDC): fit at the peak (P free in its FWHM),
    then apply prob_bicc>0.5 & mass_flag_peri==0 & iter_fap<fap_thresh
    (the mass_flag_peri condition is skipped when ``apply_rlof`` is False).

    Short-circuit: the two CHEAP gates (prob_bicc, then the Roche-lobe cut) are
    evaluated first, and the EXPENSIVE permutation iter_fap (~80% of the cost) is
    run ONLY for the survivors. Since the gate is an AND, this leaves the detected
    mask unchanged (survivors' iter_fap is bit-identical — the full-batch shuffle
    draw is preserved). ``exclude`` (bool mask) additionally drops injections
    already detected by an earlier branch (branches are OR'd), skipping their
    permutations too.
    Returns (detected[bool], prob_bicc, iter_fap, P_fit, e_fit, K1_fit,
    mass_flag[bool])."""
    N = Y.shape[1]
    excl = None
    if exclude is not None:
        excl = np.asarray(exclude, dtype=bool)
        if not excl.any():
            excl = None
        elif excl.all():
            # Every column already detected by the earlier branch: nothing
            # left to decide — pure placeholders, no scan/fit/permutation
            # (the periodograms cannot take an empty batch).
            zeros = np.zeros(N)
            return (np.zeros(N, dtype=bool), zeros, np.ones(N),
                    np.ones(N), zeros, zeros, np.zeros(N, dtype=bool))
    if excl is None:
        period, peak_pow, flo, fhi = (pgram.peak_with_fwhm(Y, errs) if has_err
                                      else pgram.peak_with_fwhm(Y))
        fit = batched_fit.fit_batch(Y, pgram.t, errs, period, flo, fhi,
                                    track_best=apply_rlof,
                                    **(fit_kwargs or {}))
    else:
        # Branches are OR'd, so a column already detected by an earlier branch
        # cannot change the outcome — the candidate scan + fit run on the
        # remaining columns only, and placeholders (prob=0 -> fails the cheap
        # gate, exactly like a sub-threshold fit) are scattered back. The
        # permutation below still sees the FULL Y: its full-batch shuffle draw
        # + `active` mask keep every surviving column's iter_fap bit-identical
        # to a no-exclude call.
        keep = np.flatnonzero(~excl)
        Yk = Y[:, keep]
        ek = errs[keep]
        period_k, peak_k, flo_k, fhi_k = (pgram.peak_with_fwhm(Yk, ek)
                                          if has_err
                                          else pgram.peak_with_fwhm(Yk))
        fit_k = batched_fit.fit_batch(Yk, pgram.t, ek, period_k, flo_k, fhi_k,
                                      track_best=apply_rlof,
                                      **(fit_kwargs or {}))
        peak_pow = np.zeros(N)
        peak_pow[keep] = peak_k
        fit = {}
        for key, fill in (("prob_bicc", 0.0), ("P", 1.0), ("e", 0.0),
                          ("K1", 0.0)):
            full = np.full(N, fill)
            full[keep] = fit_k[key]
            fit[key] = full
    # cheap gate 1: prob_bicc > 0.5
    prob = fit["prob_bicc"] > 0.5
    # cheap gate 2: Roche-lobe overflow, evaluated only for prob-passers
    mass_flag = np.zeros(N, dtype=bool)
    if apply_rlof:
        sp = np.where(prob)[0]
        if sp.size:
            mass_flag[sp] = rlof.mass_flag_peri_batch(
                [mass_rows[i] for i in sp], fit["K1"][sp], fit["e"][sp], fit["P"][sp])
    prelim = prob & (~mass_flag)
    if exclude is not None:
        prelim = prelim & (~np.asarray(exclude, dtype=bool))
    # expensive gate: permutation iter_fap, only for the survivors
    maxpow_fn = pgram.max_power
    col_arg = np.repeat(errs, n_perm) if has_err else None
    ifap = batched_perm.iter_fap_fn(maxpow_fn, Y, peak_pow, n_perm,
                                    np.random.default_rng(iter_seed),
                                    col_arg=col_arg, active=prelim,
                                    fail_thresh=fap_thresh)
    detected = prelim & (ifap < fap_thresh)
    return detected, fit["prob_bicc"], ifap, fit["P"], fit["e"], fit["K1"], mass_flag


def detect_batch(mjds, rv_matrix, errs, mass_rows, cfg,
                 backend="numpy", rng=None, fap_thresh=0.2, fit_kwargs=None):
    """Detected/not-detected for each injected curve.

    mjds       : (n_epoch,) shared MJDs of the field.
    rv_matrix  : (n_epoch, n_batch) noisy injected RV curves.
    errs       : (n_batch,) per-injection (per-star, uniform) RV error.
    mass_rows  : per-injection star rows for the Roche-lobe check.
    cfg        : pipeline config dict (periodogram_params used;
                 apply_rlof_gate switches the Roche-lobe gate, default True).
    Returns (detected[bool, n_batch], info dict of arrays).
    """
    rng = rng or np.random.default_rng()
    apply_rlof = bool(cfg.get("apply_rlof_gate", True))
    pmin, pmax, spp, n_perm, pdc_spp = _peri_params(cfg)
    rv_matrix = np.asarray(rv_matrix, dtype=np.float64)
    errs = np.asarray(errs, dtype=np.float64)
    seed = int(rng.integers(0, 2 ** 31))

    bls, bpdc = _get_pgrams(mjds, pmin, pmax, spp, pdc_spp, backend)

    # filter_expression = (LS candidate passes) OR (PDC candidate passes)
    ls_det, ls_pb, ls_fap, ls_P, ls_e, ls_K, ls_mf = _branch(
        bls, rv_matrix, errs, mass_rows, cfg, n_perm, rng, fap_thresh,
        fit_kwargs, has_err=False, iter_seed=seed, apply_rlof=apply_rlof)
    # PDC only needs to decide injections LS did NOT already detect (branches are
    # OR'd) -> pass ls_det as `exclude` so PDC skips their permutations too.
    # periodogram_params.run_pdc=false drops the PDC branch entirely (detection
    # = LS only); placeholders keep the info schema and make stage_pdc=0 so the
    # kill flag reflects the LS stage alone.
    run_pdc = bool(cfg.get("periodogram_params", {}).get("run_pdc", True))
    if run_pdc:
        pdc_det, pdc_pb, pdc_fap, pdc_P, pdc_e, pdc_K, pdc_mf = _branch(
            bpdc, rv_matrix, errs, mass_rows, cfg, n_perm, rng, fap_thresh,
            fit_kwargs, has_err=True, iter_seed=seed + 1, exclude=ls_det,
            apply_rlof=apply_rlof)
    else:
        n_b = rv_matrix.shape[1]
        pdc_det = np.zeros(n_b, dtype=bool)
        pdc_pb = np.zeros(n_b)
        pdc_fap = np.ones(n_b)
        pdc_P = np.ones(n_b)
        pdc_e = np.zeros(n_b)
        pdc_K = np.zeros(n_b)
        pdc_mf = np.zeros(n_b, dtype=bool)

    detected = ls_det | pdc_det
    # Per-injection kill flag: the gate that killed detection, taken from the
    # branch that progressed FURTHEST through the AND chain
    # (bicc -> rocheLobe -> fap). Stage per branch: 0 = failed prob_bicc,
    # 1 = failed Roche lobe, 2 = failed permutation FAP. Meaningful only where
    # ~detected; detected entries are forced to KILL_DETECTED. (For entries the
    # PDC branch skipped via `exclude`, the PDC stage is 2 with synthetic
    # ifap=1.0 — harmless, those entries are detected overall.)
    stage_ls = np.where(ls_pb <= 0.5, 0, np.where(ls_mf, 1, 2))
    stage_pdc = np.where(pdc_pb <= 0.5, 0, np.where(pdc_mf, 1, 2))
    stage = np.maximum(stage_ls, stage_pdc)
    kill_flag = np.array([KILL_BICC, KILL_RLOF, KILL_FAP],
                         dtype=np.int8)[stage]
    kill_flag[detected] = KILL_DETECTED

    info = {
        "detected_ls": ls_det, "detected_pdc": pdc_det,
        "prob_bicc_ls": ls_pb, "LS_iter_fap": ls_fap,
        "prob_bicc_pdc": pdc_pb, "PDC_iter_fap": pdc_fap,
        "P_fit_ls": ls_P, "P_fit_pdc": pdc_P,
        "mass_flag_ls": ls_mf, "mass_flag_pdc": pdc_mf,
        "kill_flag": kill_flag,
    }
    return detected, info
