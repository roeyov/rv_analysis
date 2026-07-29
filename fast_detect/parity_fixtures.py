"""Bit-parity fixtures for the fast_detect speedup work (Tier A gates).

``generate`` runs the CURRENT code on fixed-seed synthetic populations and
saves every detection output to ``$SCRIPTS_OUT/bias_speedup/fixtures/``;
``check`` re-runs the same inputs and compares against the saved files.
Run BOTH with pinned threads so BLAS reduction order is reproducible::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python -m fast_detect.parity_fixtures generate
    python -m fast_detect.parity_fixtures check              # exact (bitwise)
    python -m fast_detect.parity_fixtures check --rtol 1e-10 # fused-kernel mode

Three fixture levels:

kernel   raw ``BatchedLS`` / ``BatchedPDC`` ``peak_with_fwhm`` + ``max_power``
         outputs — the rtol gate for the fused-power-kernel change (masks and
         argmax indices stay exact; float powers may drift within --rtol).
detect   full ``detect_batch`` over a mixed binary/noise population, gate ON
         and gate OFF.
worker   ``_worker_field_injections_batched`` end-to-end result dicts —
         catches injection-level changes (singles skip, T0 vectorize).

Comparison contract (what downstream code actually consumes):
  * ``detected`` / ``kill_flag`` / worker result dicts: always exact.
  * PDC-branch per-column outputs are compared only where ``~ls_det`` —
    branches are OR'd, so PDC values on LS-detected columns are dead.
  * Fit params (P_fit, mass_flag) are compared only in the gate-ON fixture —
    with the RLOF gate off they feed nothing.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

from simulations.common import BLOEM_MJD_ARRAYS
from simulations.bias_config import DEFAULT_BIAS_CFG
from simulations.bias_grid_lib.injection import (
    _draw_star_binaries, _worker_field_injections_batched, _zero_err_mass_row,
)
from fast_detect.detector import detect_batch, _get_pgrams
from fast_detect import batched_perm

_SCRIPTS_OUT = os.environ.get(
    "SCRIPTS_OUT", "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut")
FIXTURE_DIR = os.path.join(_SCRIPTS_OUT, "bias_speedup", "fixtures")

_FIELDS = [0, 4, 7]          # 3 of the ~8 BLOeM MJD grids
_N_DETECT = 1500             # curves per detect-level fixture
_N_KERNEL = 800              # curves per kernel-level fixture
_SEED0 = 20260729

# Sampling cfg for _draw_star_binaries (matches the ext run's bounds).
_DRAW_CFG = {"log_p_min": 0.046, "log_p_max": 3.5,
             "q_min": 0.01, "q_max": 1.0, "e_max": 0.95, "p_circ": 2.26}
_STAR = {"M1": 25.0, "R1": 9.0, "gamma": 150.0, "rv_err": 6.0}


def _mixed_population(mjds, n_curves, seed, f_bin=0.7):
    """Fixed-seed binaries + noise singles sharing one field's MJDs.

    Returns (Y [n_epoch, n], errs [n], mass_rows [n], n_bin).
    """
    rng = np.random.default_rng(seed)
    n_epochs = mjds.size
    n_bin = int(round(f_bin * n_curves))
    n_single = n_curves - n_bin
    (_, _, _, _, rv_obs, M1_j, R1_j) = _draw_star_binaries(
        mjds, n_epochs, _STAR["rv_err"], _STAR["M1"], _STAR["R1"],
        _STAR["gamma"], n_bin, -0.82, -1.0, -0.5, _DRAW_CFG, rng)
    rv_single = _STAR["gamma"] + rng.normal(
        0.0, _STAR["rv_err"], (n_single, n_epochs))
    Y = np.concatenate([rv_obs, rv_single], axis=0).T   # (n_epoch, n)
    errs = np.full(n_curves, _STAR["rv_err"])
    mass_rows = ([_zero_err_mass_row(M1_j[j], R1_j[j]) for j in range(n_bin)]
                 + [_zero_err_mass_row(_STAR["M1"], _STAR["R1"])] * n_single)
    return Y, errs, mass_rows, n_bin


def _worker_tasks():
    """Two gate-off field tasks + one gate-on repeat of the first."""
    cfg_off = copy.deepcopy(DEFAULT_BIAS_CFG)
    cfg_off.update(_DRAW_CFG)
    cfg_off["apply_rlof_gate"] = False
    cfg_on = copy.deepcopy(cfg_off)
    cfg_on["apply_rlof_gate"] = True
    args_dict = {"fast_detect_backend": "numpy"}

    def task(field, seeds, rv_err, M1, R1, gamma, n_inject, f_bin,
             pi, kappa, eta, cfg):
        return (np.array(BLOEM_MJD_ARRAYS[field], float),
                np.array(seeds, dtype=np.int64),
                np.array(rv_err, float), np.array(M1, float),
                np.array(R1, float), np.array(gamma, float),
                n_inject, f_bin, pi, kappa, eta,
                cfg, args_dict, "pipeline_batched")

    return {
        "worker_t0_gateoff": task(0, [1111111, 2222222, 3333333],
                                  [4.0, 6.0, 9.0], [20.0, 30.0, 40.0],
                                  [8.0, 10.0, 12.0], [140.0, 160.0, 180.0],
                                  400, 0.65, -0.82, -1.0, -0.5, cfg_off),
        "worker_t1_gateoff": task(5, [4444444, 5555555, 6666666],
                                  [3.0, 7.0, 12.0], [18.0, 28.0, 55.0],
                                  [7.0, 11.0, 15.0], [130.0, 155.0, 175.0],
                                  400, 0.45, -1.70, 1.80, -1.25, cfg_off),
        "worker_t0_gateon": task(0, [1111111, 2222222, 3333333],
                                 [4.0, 6.0, 9.0], [20.0, 30.0, 40.0],
                                 [8.0, 10.0, 12.0], [140.0, 160.0, 180.0],
                                 400, 0.65, -0.82, -1.0, -0.5, cfg_on),
    }


# ---------------------------------------------------------------------------
# Fixture computation (shared by generate and check)
# ---------------------------------------------------------------------------

def _compute_kernel(field):
    mjds = np.array(BLOEM_MJD_ARRAYS[field], float)
    Y, errs, _, _ = _mixed_population(mjds, _N_KERNEL, _SEED0 + field)
    # Production grid = the _peri_params defaults (cfg carries no
    # periodogram_params key on the bias-grid path).
    bls, bpdc = _get_pgrams(mjds, 0.4, 10000.0, 50, 10, "numpy")
    ls_period, ls_pow, ls_flo, ls_fhi = bls.peak_with_fwhm(Y)
    pdc_period, pdc_pow, pdc_flo, pdc_fhi = bpdc.peak_with_fwhm(Y, errs)
    n_shuf = 100
    shuf = batched_perm._make_shuffles(
        Y[:, :n_shuf], 10, np.random.default_rng(_SEED0 + 900 + field))
    ls_maxp = bls.max_power(shuf)
    pdc_maxp = bpdc.max_power(shuf, np.repeat(errs[:n_shuf], 10))
    return {
        "ls_period": ls_period, "ls_pow": ls_pow,
        "ls_flo": ls_flo, "ls_fhi": ls_fhi,
        "pdc_period": pdc_period, "pdc_pow": pdc_pow,
        "pdc_flo": pdc_flo, "pdc_fhi": pdc_fhi,
        "ls_maxp": ls_maxp, "pdc_maxp": pdc_maxp,
    }


def _compute_detect(field, apply_rlof):
    mjds = np.array(BLOEM_MJD_ARRAYS[field], float)
    Y, errs, mass_rows, _ = _mixed_population(mjds, _N_DETECT, _SEED0 + field)
    cfg = {"apply_rlof_gate": bool(apply_rlof)}
    det, info = detect_batch(mjds, Y, errs, mass_rows, cfg, backend="numpy",
                             rng=np.random.default_rng(_SEED0 + 500 + field))
    return {
        "detected": det, "kill_flag": info["kill_flag"],
        "detected_ls": info["detected_ls"], "detected_pdc": info["detected_pdc"],
        "prob_bicc_ls": info["prob_bicc_ls"], "LS_iter_fap": info["LS_iter_fap"],
        "prob_bicc_pdc": info["prob_bicc_pdc"], "PDC_iter_fap": info["PDC_iter_fap"],
        "P_fit_ls": info["P_fit_ls"], "P_fit_pdc": info["P_fit_pdc"],
        "mass_flag_ls": info["mass_flag_ls"], "mass_flag_pdc": info["mass_flag_pdc"],
    }


def _compute_worker(task):
    res = _worker_field_injections_batched(task)
    out = {k: np.asarray(res[k]) for k in (
        "n_physical", "n_detected", "n_rlof", "n_false_positive",
        "logP_det", "e_det", "K1_det", "q_det",
        "logP_nondet", "e_nondet", "K1_nondet", "q_nondet",
        "kill_flag_nondet")}
    for pair, h in res["hist_total"].items():
        out["hist_total_%s_%s" % pair] = h
    for pair, h in res["hist_det"].items():
        out["hist_det_%s_%s" % pair] = h
    return out


def _all_fixtures():
    """{name: thunk} for every fixture case."""
    cases = {}
    for f in _FIELDS:
        cases["kernel_f%d" % f] = (lambda f=f: _compute_kernel(f))
        cases["detect_f%d_gateoff" % f] = (lambda f=f: _compute_detect(f, False))
        cases["detect_f%d_gateon" % f] = (lambda f=f: _compute_detect(f, True))
    for name, task in _worker_tasks().items():
        cases[name] = (lambda t=task: _compute_worker(t))
    return cases


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

# Keys that stay bitwise-exact even in --rtol mode (decisions + integers).
_ALWAYS_EXACT = {
    "detected", "kill_flag", "detected_ls", "detected_pdc",
    "mass_flag_ls", "mass_flag_pdc",
    "n_physical", "n_detected", "n_rlof", "n_false_positive",
    "kill_flag_nondet",
}
# PDC per-column outputs: dead where LS already detected -> compare on ~ls_det.
_PDC_MASKED = {"prob_bicc_pdc", "PDC_iter_fap", "P_fit_pdc", "mass_flag_pdc"}
# Fit params: dead when the RLOF gate is off -> only compared in gateon cases.
_FIT_KEYS = {"P_fit_ls", "P_fit_pdc"}
# Permutation FAPs: the <0.2 gate decision must stay exact, and the value must
# stay exact for PASSING curves. For definite-fail curves the exact-safe
# early-stop stops shuffling once >= ceil(0.2*n_perm) exceedances are seen, so
# their recorded value is a censored lower bound (still >= 0.2) — dead
# downstream (only detected/kill_flag are consumed).
_FAP_KEYS = {"LS_iter_fap", "PDC_iter_fap"}
_FAP_THRESH = 0.2


def _compare_case(name, saved, fresh, rtol):
    """Returns list of mismatch strings (empty = pass)."""
    errors = []
    gateoff = name.endswith("gateoff")
    ls_det = saved.get("detected_ls")
    for key in saved.files if hasattr(saved, "files") else saved.keys():
        a, b = np.asarray(saved[key]), np.asarray(fresh[key])
        if gateoff and key in _FIT_KEYS:
            continue
        if key in _PDC_MASKED and ls_det is not None:
            keep = ~np.asarray(ls_det, dtype=bool)
            a, b = a[keep], b[keep]
        if a.shape != b.shape:
            errors.append("%s: shape %s != %s" % (key, a.shape, b.shape))
            continue
        if key in _FAP_KEYS:
            gate_a, gate_b = a < _FAP_THRESH, b < _FAP_THRESH
            if not np.array_equal(gate_a, gate_b):
                errors.append("%s: %d gate decisions flipped"
                              % (key, int(np.sum(gate_a != gate_b))))
            elif not np.array_equal(a[gate_a], b[gate_a]):
                errors.append("%s: passing-curve values differ" % key)
            elif np.any(b[~gate_b] < _FAP_THRESH):
                errors.append("%s: censored fail value below thresh" % key)
            continue
        exact = (rtol == 0.0) or (key in _ALWAYS_EXACT) \
            or not np.issubdtype(a.dtype, np.floating)
        if exact:
            if not np.array_equal(a, b):
                bad = np.sum(a != b)
                errors.append("%s: %d/%d values differ (exact)" %
                              (key, int(bad), a.size))
        else:
            ok = np.allclose(a, b, rtol=rtol, atol=0.0, equal_nan=True)
            if not ok:
                rel = np.abs(b - a) / np.maximum(np.abs(a), 1e-300)
                errors.append("%s: max rel diff %.3e > rtol %.1e" %
                              (key, float(np.nanmax(rel)), rtol))
    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["generate", "check"])
    ap.add_argument("--rtol", type=float, default=0.0,
                    help="float tolerance for non-decision arrays "
                         "(0 = bitwise exact; use 1e-10 for the fused kernel)")
    ap.add_argument("--only", default=None,
                    help="substring filter on fixture names")
    args = ap.parse_args()

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    cases = _all_fixtures()
    if args.only:
        cases = {k: v for k, v in cases.items() if args.only in k}

    timings = {}
    failed = []
    for name, thunk in cases.items():
        t0 = time.perf_counter()
        fresh = thunk()
        timings[name] = time.perf_counter() - t0
        path = os.path.join(FIXTURE_DIR, name + ".npz")
        if args.mode == "generate":
            np.savez(path, **fresh)
            print("wrote  %-22s (%.1fs, %d arrays)"
                  % (name, timings[name], len(fresh)))
        else:
            if not os.path.exists(path):
                failed.append(name)
                print("MISSING %-22s (no baseline; run generate first)" % name)
                continue
            saved = np.load(path, allow_pickle=False)
            errors = _compare_case(name, saved, fresh, args.rtol)
            if errors:
                failed.append(name)
                print("FAIL   %-22s (%.1fs)" % (name, timings[name]))
                for e in errors:
                    print("         " + e)
            else:
                print("PASS   %-22s (%.1fs)" % (name, timings[name]))

    tpath = os.path.join(
        FIXTURE_DIR, "timings_%s_%d.json" % (args.mode, int(time.time())))
    with open(tpath, "w") as fh:
        json.dump(timings, fh, indent=2)
    print("total %.1fs  (timings -> %s)" % (sum(timings.values()), tpath))
    if failed:
        print("FAILED: %s" % ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
