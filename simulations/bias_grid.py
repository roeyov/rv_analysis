"""
simulations.bias_grid — Sana+2012/2013 forward-modeling bias correction grid search.

4D grid search over (π, κ, η, f_bin) to determine intrinsic binary fraction
and power-law shape parameters by comparing simulated detected populations
to the observed SB1+SB2 sample.

Usage:
    python -m simulations.bias_grid --config params_bias.yaml
    python -m simulations.bias_grid --config params_bias.yaml --quick   # 3x3x3x3 test
"""

import os
import re
import copy
import time
import logging
import argparse
import itertools
import yaml
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, anderson_ksamp, cramervonmises_2samp, binom

from numba import njit

logger = logging.getLogger("bias_grid")


def setup_logging(output_dir, level=logging.DEBUG):
    """Configure file + stdout logging. Call once in main()."""
    os.makedirs(output_dir, exist_ok=True)
    logger.setLevel(level)
    # File handler (DEBUG level — everything)
    fh = logging.FileHandler(
        os.path.join(output_dir, "bias_grid.log"), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    # Stdout handler (INFO level)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(sh)

from simulations.common import BLOEM_MJD_ARRAYS, SIGMA_SHAPE, SIGMA_LOC, SIGMA_SCALE
from simulations.common import simulate_system_refined
from simulations.bias_config import DEFAULT_BIAS_CFG, GRID_PRESETS
from pipeline.config import load_args


# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------

G_CGS = 6.674e-8
MSUN = 1.989e33
RSUN = 6.957e10
DAY = 86400.0
KM = 1e5
TWOPI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# 2D detection-probability histogram bins (accumulated across all grid points)
# ---------------------------------------------------------------------------

_HIST_NBINS = 30
_HIST_BINS = {
    "logP": np.linspace(0.0, 4.0, _HIST_NBINS + 1),
    "e":    np.linspace(0.0, 0.95, _HIST_NBINS + 1),
    "K1":   np.linspace(0.0, 300.0, _HIST_NBINS + 1),
    "q":    np.linspace(0.1, 1.0, _HIST_NBINS + 1),
}
_HIST_PAIRS = [("logP", "e"), ("logP", "K1"), ("logP", "q"),
               ("e", "K1"), ("e", "q"), ("q", "K1")]


# ---------------------------------------------------------------------------
# Pluggable detection functions
# ---------------------------------------------------------------------------

def _detect_full_pipeline(MJDs, rv_obs, rv_err, args_dict):
    """Original: full periodogram + lmfit + BICc pipeline."""
    from pipeline.evaluator import detect_from_arrays
    return detect_from_arrays(MJDs, rv_obs, rv_err, args_dict, use_fwhm=True)


def _detect_rv_threshold(MJDs, rv_obs, rv_err, args_dict):
    """Fast: pairwise delta-RV significance threshold.

    When rv_err is constant (uniform errors), the pairwise sigma matrix
    reduces to a scalar: rv_err_val * sqrt(2).  The two conditions
    (|drv| > drv_thresh  AND  significance > sign_thresh) collapse to
    a single threshold on max(rv) - min(rv).
    """
    drv_thresh = args_dict.get("drv_threshold", 20.0)
    sign_thresh = args_dict.get("significance_threshold", 4.0)

    # Pre-computed threshold may be cached in args_dict by the caller
    eff_thresh = args_dict.get("_rv_eff_threshold")
    if eff_thresh is not None:
        detected = bool((rv_obs.max() - rv_obs.min()) > eff_thresh)
    else:
        # Fallback: compute from rv_err (handles non-uniform case)
        from orbital.statistics import binary_rv_threshold
        detected = bool(binary_rv_threshold(rv_obs, rv_err,
                                            drv_tresh=drv_thresh,
                                            sign_threshold=sign_thresh))

    info = {"reason": "", "detection_method": "rv_threshold"}
    return detected, info


DETECTION_METHODS = {
    "pipeline": _detect_full_pipeline,
    "rv_threshold": _detect_rv_threshold,
}


# ---------------------------------------------------------------------------
# Statistical test helpers
# ---------------------------------------------------------------------------

def _clip_to_range(sim_arr, lo, hi):
    """Restrict simulated detected array to [lo, hi]."""
    return sim_arr[(sim_arr >= lo) & (sim_arr <= hi)]


def _safe_pvalue(test_fn, obs, sim, min_samples=5):
    """Compute a two-sample test p-value, returning 0.0 on failure."""
    if len(sim) < min_samples:
        return 0.0
    try:
        return float(test_fn(obs, sim))
    except Exception:
        return 0.0


def _ks_pvalue(obs, sim):
    return ks_2samp(obs, sim).pvalue


def _ad_pvalue(obs, sim):
    return anderson_ksamp([obs, sim]).pvalue


def _cvm_pvalue(obs, sim):
    return cramervonmises_2samp(obs, sim).pvalue


_ALL_TESTS = {"ks": _ks_pvalue, "ad": _ad_pvalue, "cvm": _cvm_pvalue}


# ---------------------------------------------------------------------------
# Pure scoring (used by both forward and resume paths, picklable for workers)
# ---------------------------------------------------------------------------

def _make_scoring_ctx(obs_logP, obs_e, obs_K1, clip_range,
                      split_e_circular, obs_e_cont,
                      n_obs_circ, n_obs_e_total,
                      N_det_obs, N_stars):
    """Bundle the per-run constants that every grid-point score needs."""
    return {
        "obs_logP": np.asarray(obs_logP),
        "obs_e": np.asarray(obs_e),
        "obs_K1": np.asarray(obs_K1),
        "clip_range": clip_range,
        "split_e_circular": bool(split_e_circular),
        "obs_e_cont": np.asarray(obs_e_cont) if obs_e_cont is not None
                      else None,
        "n_obs_circ": int(n_obs_circ) if n_obs_circ is not None else 0,
        "n_obs_e_total": int(n_obs_e_total) if n_obs_e_total is not None
                         else 0,
        "N_det_obs": int(N_det_obs),
        "N_stars": int(N_stars),
    }


def _compute_scores(res, fbin, ctx):
    """Compute KS/AD/CvM p-values + log-GMF for one grid-point result.

    Pure function: takes a worker `res` dict (with logP_det/e_det/K1_det,
    n_physical, p_det) and the per-run scoring context; returns a dict
    of every scalar we want in the cubes and the CSV row.

    No mutation, no I/O — safe to call from any process.
    """
    p_det = res["p_det"]
    sim_clipped = {
        "logP": _clip_to_range(res["logP_det"], *ctx["clip_range"]["logP"]),
        "e": _clip_to_range(res["e_det"], *ctx["clip_range"]["e"]),
        "K1": _clip_to_range(res["K1_det"], *ctx["clip_range"]["K1"]),
    }

    n_total_sim = res["n_physical"]
    if n_total_sim > 0 and p_det > 0:
        p_binom = float(binom.pmf(ctx["N_det_obs"],
                                  ctx["N_stars"],
                                  fbin * p_det))
    else:
        p_binom = 0.0

    out = {"p_binom": p_binom}
    split = ctx["split_e_circular"]
    for tname, tfn in _ALL_TESTS.items():
        p_logP = _safe_pvalue(tfn, ctx["obs_logP"], sim_clipped["logP"])
        p_K1 = _safe_pvalue(tfn, ctx["obs_K1"], sim_clipped["K1"],
                            min_samples=3)
        if split:
            sim_e_cont = sim_clipped["e"][sim_clipped["e"] > 0]
            p_e = _safe_pvalue(tfn, ctx["obs_e_cont"], sim_e_cont)
            n_sim_e = len(sim_clipped["e"])
            n_sim_circ = int(np.sum(sim_clipped["e"] == 0))
            f_circ_sim = n_sim_circ / max(n_sim_e, 1)
            p_e_circ = float(binom.pmf(
                ctx["n_obs_circ"], ctx["n_obs_e_total"], f_circ_sim
            )) if n_sim_e > 0 else 0.0
        else:
            p_e = _safe_pvalue(tfn, ctx["obs_e"], sim_clipped["e"])
            p_e_circ = None

        gmf_pvals = [p_logP, p_e, p_K1, p_binom]
        if split:
            gmf_pvals.append(p_e_circ)
        log_gmf = 0.0
        for pv in gmf_pvals:
            if pv > 0:
                log_gmf += np.log(pv)
            else:
                log_gmf = -np.inf
                break

        out["%s_p_logP" % tname] = p_logP
        out["%s_p_e" % tname] = p_e
        out["%s_p_K1" % tname] = p_K1
        if split:
            out["%s_p_e_circ" % tname] = p_e_circ
        out["log_gmf_%s" % tname] = log_gmf

    out["log_gmf"] = out["log_gmf_ks"]
    return out


# ---------------------------------------------------------------------------
# Numba-accelerated Kepler solver & RV model
# ---------------------------------------------------------------------------

@njit(cache=True)
def kepler_E(M, e, tol=1e-10, maxiter=100):
    """Solve Kepler's equation M = E - e sin(E) via Newton-Raphson."""
    E = np.copy(M)
    for i in range(len(M)):
        for _ in range(maxiter):
            dE = -(E[i] - e * np.sin(E[i]) - M[i]) / (1.0 - e * np.cos(E[i]))
            E[i] += dE
            if np.abs(dE) < tol:
                break
    return E


@njit(cache=True)
def rv_model_jit(t, P, T0, omega, e, K1, gamma):
    """Compute Keplerian RV curve (numba-accelerated)."""
    phase = ((t - T0) / P) % 1.0
    M = TWOPI * phase
    if e < 1e-8:
        return gamma + K1 * np.cos(M + omega)
    E = kepler_E(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                           np.sqrt(1.0 - e) * np.cos(E / 2.0))
    return gamma + K1 * (np.cos(nu + omega) + e * np.cos(omega))


@njit(cache=True)
def kepler_E_batch(M_2d, e_1d, tol=1e-10, maxiter=100):
    """Solve Kepler's equation for a batch of systems.

    Parameters
    ----------
    M_2d : ndarray, shape (n_sys, n_epochs)
        Mean anomalies.
    e_1d : ndarray, shape (n_sys,)
        Eccentricities (one per system).

    Returns
    -------
    E : ndarray, shape (n_sys, n_epochs)
        Eccentric anomalies.
    """
    n_sys, n_ep = M_2d.shape
    E = np.copy(M_2d)
    for s in range(n_sys):
        ecc = e_1d[s]
        if ecc < 1e-8:
            continue  # circular: E = M (already copied)
        for i in range(n_ep):
            for _ in range(maxiter):
                dE = -(E[s, i] - ecc * np.sin(E[s, i]) - M_2d[s, i]) / \
                     (1.0 - ecc * np.cos(E[s, i]))
                E[s, i] += dE
                if abs(dE) < tol:
                    break
    return E


def compute_K1_batch(P_days, e, q, M1_Msun, incl):
    """Vectorized K1 [km/s] computation. All inputs shape (n,)."""
    P_s = P_days * DAY
    M1 = M1_Msun * MSUN
    M2 = q * M1
    a = (G_CGS * (M1 + M2) * P_s ** 2 / (4 * np.pi ** 2)) ** (1.0 / 3.0)
    a1 = a * M2 / (M1 + M2)
    return TWOPI * a1 * np.sin(incl) / (P_s * np.sqrt(1 - e ** 2)) / KM


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def compute_K1(P_days, e, q, M1_Msun, incl):
    """Compute K1 [km/s] from physical parameters."""
    P_s = P_days * DAY
    M1 = M1_Msun * MSUN
    M2 = q * M1
    a = (G_CGS * (M1 + M2) * P_s ** 2 / (4 * np.pi ** 2)) ** (1.0 / 3.0)
    a1 = a * M2 / (M1 + M2)
    K1 = TWOPI * a1 * np.sin(incl) / (P_s * np.sqrt(1 - e ** 2)) / KM
    logger.debug("compute_K1: P=%.2f q=%.3f M1=%.1f incl=%.2f → K1=%.2f",
                 P_days, q, M1_Msun, incl, K1)
    return K1


def roche_lobe_check(P_days, e, q, M1_Msun, R1_Rsun):
    """Return True if the primary fits inside its Roche lobe at periastron."""
    if q <= 0 or P_days <= 0:
        return False
    P_s = P_days * DAY
    M1 = M1_Msun * MSUN
    M2 = q * M1_Msun * MSUN
    a = (G_CGS * (M1 + M2) * P_s ** 2 / (4 * np.pi ** 2)) ** (1.0 / 3.0)
    r_peri = a * (1.0 - e)
    q_don = 1.0 / q
    q23 = q_don ** (2.0 / 3.0)
    rl = 0.49 * q23 / (0.6 * q23 + np.log(1.0 + q_don ** (1.0 / 3.0)))
    result = (R1_Rsun * RSUN) < rl * r_peri
    logger.debug("roche_lobe_check: P=%.2f e=%.3f q=%.3f M1=%.1f R1=%.1f → %s",
                 P_days, e, q, M1_Msun, R1_Rsun, result)
    return result


def powerlaw_draw(n, alpha, xmin, xmax, rng):
    """Draw n samples from p(x) ∝ x^alpha on [xmin, xmax]."""
    a = alpha + 1.0
    u = rng.uniform(0, 1, n)
    xmin = max(float(xmin), 1e-10)
    xmax = max(float(xmax), xmin + 1e-10)
    if abs(a) < 1e-8:
        return np.asarray(xmin * (xmax / xmin) ** u, dtype=np.float64)
    base = xmin ** a + u * (xmax ** a - xmin ** a)
    base = np.maximum(np.real(base), 1e-300)
    samples = np.real(np.exp(np.log(base) / a)).astype(np.float64)
    logger.debug("powerlaw_draw: n=%d alpha=%.2f [%.3f,%.3f] → [%.3f,%.3f]",
                 n, alpha, xmin, xmax, samples.min(), samples.max())
    return samples


# ---------------------------------------------------------------------------
# LaTeX table parser
# ---------------------------------------------------------------------------

def _parse_val_with_errors(s):
    """Parse '1.23^{+0.01}_{-0.01}' → 1.23, or handle \\leq / \\geq."""
    s = s.strip().replace("$", "")
    if r"\dots" in s or s == r"\dots":
        return np.nan
    if r"\leq" in s:
        # Upper limit on eccentricity → treat as the limit value
        val = re.sub(r"\\leq\s*", "", s)
        return float(val)
    if r"\geq" in s:
        # Lower limit on period → return negative to flag exclusion
        val = re.sub(r"\\geq\s*", "", s)
        return -float(val)
    # Standard: value^{+err}_{-err}
    m = re.match(r"([0-9.eE+-]+)", s)
    if m:
        return float(m.group(1))
    return np.nan


def load_observed_from_tex(sb1_path, sb2_path):
    """
    Parse SB1 and SB2 LaTeX solution tables and return observed distributions.

    Returns
    -------
    obs : dict with keys:
        'logP'  : array of log10(P/days)
        'e'     : array of eccentricities
        'K1'    : array of K1 [km/s]
        'q_sb2' : array of mass ratios (SB2 only)
        'n_sb1' : number of usable SB1 systems
        'n_sb2' : number of SB2 systems
    """
    all_logP = []
    all_e = []
    all_K1 = []
    sb2_q = []

    # --- Parse SB1 table ---
    with open(sb1_path, "r") as f:
        sb1_lines = f.readlines()

    for line in sb1_lines:
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("\\"):
            continue
        if "&" not in line:
            continue

        cols = [c.strip() for c in line.replace("\\\\", "").split("&")]
        if len(cols) < 8:
            continue

        # Skip header/footer lines
        try:
            int(cols[0])
        except ValueError:
            continue

        P_val = _parse_val_with_errors(cols[2])
        e_val = _parse_val_with_errors(cols[5])
        K1_val = _parse_val_with_errors(cols[6])

        # Skip systems with P >= baseline (flagged as negative by parser)
        if P_val < 0 or np.isnan(P_val):
            continue
        if np.isnan(K1_val):
            continue

        # Skip the "outer" row of 4-043 double-Kepler
        if "(outer)" in cols[1]:
            continue

        all_logP.append(np.log10(P_val))
        # Circular orbit (Lucy-Sweeney upper limit) → e = 0
        e_is_upper_limit = r"\leq" in cols[5]
        all_e.append(0.0 if (np.isnan(e_val) or e_is_upper_limit) else e_val)
        all_K1.append(K1_val)

    n_sb1 = len(all_logP)

    # --- Parse SB2 table ---
    with open(sb2_path, "r") as f:
        sb2_lines = f.readlines()

    # SB2 rows come in pairs: primary (has all params) + secondary (only K and M sin^3 i)
    for line in sb2_lines:
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("\\"):
            continue
        if "&" not in line:
            continue

        cols = [c.strip() for c in line.replace("\\\\", "").split("&")]
        if len(cols) < 10:
            continue

        # Primary row has the system number in col[0]
        try:
            int(cols[0])
        except ValueError:
            continue

        # This is a primary row
        P_val = _parse_val_with_errors(cols[2])
        e_val = _parse_val_with_errors(cols[5])
        q_val = _parse_val_with_errors(cols[6])
        K1_val = _parse_val_with_errors(cols[8])

        if np.isnan(P_val) or P_val < 0:
            continue
        if np.isnan(K1_val):
            continue

        all_logP.append(np.log10(P_val))
        # Circular orbit (Lucy-Sweeney upper limit) → e = 0
        e_is_upper_limit = r"\leq" in cols[5]
        all_e.append(0.0 if (np.isnan(e_val) or e_is_upper_limit) else e_val)
        all_K1.append(K1_val)
        sb2_q.append(q_val if not np.isnan(q_val) else np.nan)

    n_sb2 = len(all_logP) - n_sb1

    obs = {
        "logP": np.array(all_logP),
        "e": np.array(all_e),
        "K1": np.array(all_K1),
        "q_sb2": np.array(sb2_q),
        "n_sb1": n_sb1,
        "n_sb2": n_sb2,
    }
    logger.info("load_observed_from_tex: SB1=%d SB2=%d total=%d",
                n_sb1, n_sb2, len(all_logP))
    return obs


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


# ---------------------------------------------------------------------------
# Shared state for parallel grid workers (set via pool initializer to avoid
# pickling large arrays into every task tuple — fixes BrokenPipeError).
# ---------------------------------------------------------------------------
_shared = {}
_resume_shared = {}


def _init_grid_worker(field_mjds, field_arr, rv_err_arr, M1_arr, R1_arr,
                      gamma_arr, cfg, args_dict, detect_method,
                      checkpoint_dir=None):
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
    scores = _compute_scores(res_for_scoring, fbin, ctx)
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
        # Keep only what _score_and_accumulate needs: det arrays for
        # scoring (logP/e/K1), histograms, and scalar counts.
        for _key in ("q_det",
                     "logP_nondet", "e_nondet", "K1_nondet", "q_nondet",
                     "logP_rlof", "q_rlof", "e_rlof",
                     "M1_rlof", "R1_rlof"):
            res.pop(_key, None)

    return (step, i, j, k, l, res)


# ---------------------------------------------------------------------------
# Checkpointing — shard-based detected array storage
# ---------------------------------------------------------------------------

_DET_SHARDS_DIR = "det_shards"


def _det_shard_path(checkpoint_dir, step):
    """Return path for one grid-point's detected-array shard."""
    return os.path.join(checkpoint_dir, _DET_SHARDS_DIR,
                        "step_%06d.npz" % step)


def _save_det_shard(checkpoint_dir, step, i, j, k, l, res):
    """Flush one grid point's detected arrays to a small .npz shard file.

    Also persists the scalar physical-population counts (n_physical,
    n_detected, n_rlof, n_false_positive). These cannot be reconstructed
    from logP_det/logP_nondet alone because RLOF binaries and single
    stars are part of n_physical/n_detected but absent from those
    arrays — without the scalars, resume scoring computes a wrong
    p_det and corrupts pdet_cube / log_gmf cells.
    """
    shard_dir = os.path.join(checkpoint_dir, _DET_SHARDS_DIR)
    os.makedirs(shard_dir, exist_ok=True)
    np.savez(
        _det_shard_path(checkpoint_dir, step),
        logP=res["logP_det"],
        e=res["e_det"],
        K1=res["K1_det"],
        q=res["q_det"],
        logP_nondet=res["logP_nondet"],
        e_nondet=res["e_nondet"],
        K1_nondet=res["K1_nondet"],
        q_nondet=res["q_nondet"],
        n_physical=np.int64(res.get("n_physical", 0)),
        n_detected=np.int64(res.get("n_detected", 0)),
        n_rlof=np.int64(res.get("n_rlof", 0)),
        n_false_positive=np.int64(res.get("n_false_positive", 0)),
        ijkl=np.array([i, j, k, l]),
    )


def _load_det_shard(checkpoint_dir, step):
    """Load one grid point's detected arrays from its shard file.

    Returns the array contents plus, if present, the scalar
    population counts (n_physical / n_detected / n_rlof /
    n_false_positive). For shards written before the counts were
    persisted, these keys are simply absent and the caller must fall
    back to the (lossy) len-of-arrays approximation.
    """
    path = _det_shard_path(checkpoint_dir, step)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    result = {
        "logP": d["logP"],
        "e": d["e"],
        "K1": d["K1"],
        "q": d["q"],
    }
    # Backward compat: older shards may not have nondet arrays
    if "logP_nondet" in d:
        result["logP_nondet"] = d["logP_nondet"]
        result["e_nondet"] = d["e_nondet"]
        result["K1_nondet"] = d["K1_nondet"]
        result["q_nondet"] = d["q_nondet"]
    # Scalar population counts (added later — fall back to
    # len(arrays) if absent).
    for key in ("n_physical", "n_detected", "n_rlof", "n_false_positive"):
        if key in d.files:
            result[key] = int(d[key])
    return result


def _hists_from_shard(shard):
    """Reconstruct per-grid-point hist_total / hist_det from a shard.

    Mirrors the worker's binning (np.searchsorted against _HIST_BINS,
    over physical = detected + nondetected; RLOF binaries are excluded
    from both n_physical and the shard, so the reconstruction matches
    the original computation exactly.
    """
    vals_det = {"logP": shard["logP"], "e": shard["e"],
                "K1": shard["K1"], "q": shard["q"]}
    vals_all = {
        k: np.concatenate([vals_det[k],
                           shard.get(k + "_nondet", np.array([]))])
        for k in ("logP", "e", "K1", "q")
    }
    hist_total = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                  for pair in _HIST_PAIRS}
    hist_det = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                for pair in _HIST_PAIRS}
    for (a, b) in _HIST_PAIRS:
        if len(vals_all[a]):
            ia_all = np.clip(np.searchsorted(_HIST_BINS[a], vals_all[a]) - 1,
                             0, _HIST_NBINS - 1)
            ib_all = np.clip(np.searchsorted(_HIST_BINS[b], vals_all[b]) - 1,
                             0, _HIST_NBINS - 1)
            np.add.at(hist_total[(a, b)], (ia_all, ib_all), 1)
        if len(vals_det[a]):
            ia_det = np.clip(np.searchsorted(_HIST_BINS[a], vals_det[a]) - 1,
                             0, _HIST_NBINS - 1)
            ib_det = np.clip(np.searchsorted(_HIST_BINS[b], vals_det[b]) - 1,
                             0, _HIST_NBINS - 1)
            np.add.at(hist_det[(a, b)], (ia_det, ib_det), 1)
    return hist_total, hist_det


def _save_det_index(checkpoint_dir, step_to_ijkl):
    """Save a lightweight index mapping steps → (i,j,k,l) grid indices."""
    np.savez(
        os.path.join(checkpoint_dir, "det_index.npz"),
        step_to_ijkl=np.array(step_to_ijkl),
    )



def _save_checkpoint(checkpoint_dir, completed_steps,
                     gmf_cubes, pdet_cube, test_cubes,
                     pi_grid, kappa_grid, eta_grid, fbin_grid,
                     all_results,
                     n_inject_per_star, seed, preset_name,
                     global_hists=None):
    """Save intermediate results so a killed run can be resumed.

    Detected arrays are saved per-grid-point as shard files by the
    caller (_score_and_accumulate), so this function only persists the
    cubes, scalar CSV, and global histograms.
    """
    logger.debug("_save_checkpoint: step %d", completed_steps)
    save_kw = dict(
        pdet_cube=pdet_cube,
        pi_grid=pi_grid,
        kappa_grid=kappa_grid,
        eta_grid=eta_grid,
        fbin_grid=fbin_grid,
        completed_steps=np.array(completed_steps),
        n_inject_per_star=np.array(n_inject_per_star),
        seed=np.array(seed),
        preset_name=np.array(preset_name),
    )
    # All tests
    for tname in _ALL_TESTS:
        save_kw["gmf_%s_cube" % tname] = gmf_cubes[tname]
        for par in ("logP", "e", "K1"):
            save_kw["%s_%s_cube" % (tname, par)] = test_cubes[tname][par]
        if "e_circ" in test_cubes[tname]:
            save_kw["%s_e_circ_cube" % tname] = test_cubes[tname]["e_circ"]
    np.savez(os.path.join(checkpoint_dir, "checkpoint_cubes.npz"), **save_kw)
    # Save results list as CSV (arrays already stripped by caller).
    # Skip if empty — parallel mode writes CSV incrementally.
    if all_results:
        rows = []
        for r in all_results:
            rows.append({k: v for k, v in r.items()
                         if not isinstance(v, (np.ndarray, dict))})
        pd.DataFrame(rows).to_csv(
            os.path.join(checkpoint_dir, "checkpoint_results.csv"),
            index=False,
        )
    # Save global 2D histograms (fixed-size, small)
    if global_hists is not None:
        save_kw = {}
        for k, v in _HIST_BINS.items():
            save_kw["bins_%s" % k] = v
        for (a, b) in _HIST_PAIRS:
            save_kw["hist_total_%s_%s" % (a, b)] = global_hists["total"][(a, b)]
            save_kw["hist_det_%s_%s" % (a, b)] = global_hists["det"][(a, b)]
        np.savez(
            os.path.join(checkpoint_dir, "checkpoint_hists.npz"),
            **save_kw,
        )


# ---------------------------------------------------------------------------
# Grid Search Engine
# ---------------------------------------------------------------------------

class GridSearchEngine:
    """
    4D grid search over (π, κ, η, f_bin) using the real BLOeM pipeline.

    For each grid point:
      1. For each star, decide binary (prob=f_bin) vs single
      2. If binary: draw P, q, e from power laws → inject & detect
      3. If single: generate noise-only data → check false positives
      4. KS test detected distributions vs observed
      5. Binomial probability for N_det
      6. GMF = P_KS(logP) × P_KS(e) × P_KS(K1) × P_binom(N_det)
    """

    def __init__(self, field_mjds, star_ids, M1_arr, R1_arr, field_arr,
                 rv_err_arr, gamma_arr, args_dict, cfg=None):
        """
        Parameters
        ----------
        field_mjds : dict
            {field_number: array of MJDs}
        star_ids : array-like
            Star identifiers.
        M1_arr, R1_arr : array-like
            Primary masses [M_sun] and radii [R_sun] per star.
        field_arr : array-like
            Field number per star.
        rv_err_arr : array-like
            Mean RV error per star [km/s].
        gamma_arr : array-like
            Systemic velocity per star [km/s].
        args_dict : dict
            Pipeline config (from params_bias.yaml).
        cfg : dict or None
            Grid search config. Defaults to DEFAULT_BIAS_CFG.
        """
        self.field_mjds = field_mjds
        self.star_ids = np.asarray(star_ids)
        self.M1_arr = np.asarray(M1_arr, dtype=float)
        self.R1_arr = np.asarray(R1_arr, dtype=float)
        self.field_arr = np.asarray(field_arr)
        self.rv_err_arr = np.asarray(rv_err_arr, dtype=float)
        self.gamma_arr = np.asarray(gamma_arr, dtype=float)
        self.args_dict = args_dict
        self.cfg = cfg or DEFAULT_BIAS_CFG
        self.detect_method = "pipeline"

    def _run_one_grid_point(self, pi, kappa, eta, f_bin, n_inject, rng,
                            n_workers=1):
        """
        Run injection-recovery for one (π, κ, η, f_bin) grid point.

        For each star:
        - With probability f_bin → draw binary params → inject & detect
        - With probability 1-f_bin → single star → detect false positive

        When n_workers > 1, stars are processed in parallel via
        multiprocessing.Pool.  Each star gets a deterministic seed
        derived from rng so results are reproducible regardless of
        n_workers.

        Returns dict with detection counts and detected parameter arrays.
        """
        cfg = self.cfg

        # Build one task per star.  Each task gets a unique deterministic
        # seed drawn from `rng` so the random sequence is fully determined
        # by the parent seed, independent of parallelism.
        tasks = []
        for fld in sorted(self.field_mjds.keys()):
            fld_mask = (self.field_arr == fld)
            if not fld_mask.any():
                continue
            MJDs = self.field_mjds[fld]
            for star_idx in np.where(fld_mask)[0]:
                star_seed = int(rng.integers(0, 2**63))
                tasks.append((
                    star_seed,
                    MJDs,
                    float(self.rv_err_arr[star_idx]),
                    float(self.M1_arr[star_idx]),
                    float(self.R1_arr[star_idx]),
                    float(self.gamma_arr[star_idx]),
                    n_inject,
                    f_bin,
                    pi, kappa, eta,
                    cfg,
                    self.args_dict,
                    self.detect_method,
                ))

        # Run tasks — parallel or sequential.
        worker_fn = (_worker_star_injections_vectorized
                     if self.detect_method == "rv_threshold"
                     else _worker_star_injections)
        if n_workers > 1 and len(tasks) > 1:
            import multiprocessing as _mp
            ctx = _mp.get_context("forkserver")
            with ctx.Pool(processes=min(n_workers, len(tasks))) as pool:
                results = pool.map(worker_fn, tasks)
        else:
            results = [worker_fn(t) for t in tasks]

        # Aggregate results across all stars.
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

        logger.info("_run_one_grid_point: π=%.2f κ=%.2f η=%.2f f=%.2f "
                     "→ n_det=%d n_phys=%d n_rlof=%d n_fp=%d p_det=%.3f",
                     pi, kappa, eta, f_bin,
                     n_detected, n_physical, n_rlof, n_false_positive, p_det)

        return {
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

    def run(self, pi_grid, kappa_grid, eta_grid, fbin_grid,
            obs_logP, obs_e, obs_K1,
            n_inject_per_star=None, seed=42,
            progress_callback=None, result_callback=None,
            checkpoint_dir=None, preset_name="custom",
            n_workers=1,
            grid_start=0, grid_end=None,
            parallel_grid=False):
        """
        Run the full 4D grid search.

        Parameters
        ----------
        pi_grid, kappa_grid, eta_grid, fbin_grid : array-like
            Grid values for each parameter.
        obs_logP, obs_e, obs_K1 : array-like
            Observed orbital parameters for KS comparison.
        n_inject_per_star : int or None
            Injections per star per grid point. Default from cfg.
        seed : int
            Base random seed.
        progress_callback : callable or None
            Called with (step, total, pi, kappa, eta, fbin, p_det).
        result_callback : callable or None
            Called after each grid point with (step, total, result_dict).
        checkpoint_dir : str or None
            If set, save intermediate results after each grid point.
        preset_name : str
            Name of the preset being run (stored in checkpoint for
            mismatch detection on resume).
        n_workers : int
            Number of parallel workers. In default mode, parallelizes
            stars within each grid point. With parallel_grid=True,
            parallelizes grid points themselves (stars run sequentially).
        grid_start : int
            First step index to process (inclusive). For SLURM tasks.
        grid_end : int or None
            Last step index to process (exclusive). None = all.
        parallel_grid : bool
            If True, run grid points in parallel (one per CPU) with
            stars sequential within each. Default False.

        Returns
        -------
        dict with 'gmf_cube', 'pdet_cube', 'best_fit', 'results', grids.
        """
        if n_inject_per_star is None:
            n_inject_per_star = self.cfg.get("n_inject_per_star", 100)

        split_e_circular = self.cfg.get("split_e_circular", False)

        N_det_obs = len(obs_logP)
        N_stars = self.cfg.get("n_stars_sample", len(self.M1_arr))

        n_pi = len(pi_grid)
        n_kappa = len(kappa_grid)
        n_eta = len(eta_grid)
        n_fbin = len(fbin_grid)
        total = n_pi * n_kappa * n_eta * n_fbin

        shape = (n_pi, n_kappa, n_eta, n_fbin)
        pdet_cube = np.zeros(shape)

        # Per-test p-value cubes and GMF cubes
        test_cubes = {}
        gmf_cubes = {}
        for tname in _ALL_TESTS:
            tc = {
                "logP": np.zeros(shape),
                "e": np.zeros(shape),
                "K1": np.zeros(shape),
            }
            if split_e_circular:
                tc["e_circ"] = np.zeros(shape)
            test_cubes[tname] = tc
            gmf_cubes[tname] = np.full(shape, -np.inf)

        # Backward-compat aliases (KS is the default)
        gmf_cube = gmf_cubes["ks"]
        ks_logP_cube = test_cubes["ks"]["logP"]
        ks_e_cube = test_cubes["ks"]["e"]
        ks_K1_cube = test_cubes["ks"]["K1"]

        # Survey-sensitivity bounds for clipping simulated detected arrays.
        # Based on the highest observed detection (not the survey baseline),
        # so we compare CDFs only where we have constraining power.
        clip_range = {
            "logP": (0.0, obs_logP.max()),
            "e": (0.0, 1.0),
            "K1": (0.0, obs_K1.max()),
        }
        logger.info("CDF clip ranges: logP=[%.2f,%.2f] e=[%.3f,%.3f] "
                     "K1=[%.1f,%.1f]",
                     *clip_range["logP"], *clip_range["e"],
                     *clip_range["K1"])

        # Pre-split observed eccentricities for the circular/continuous
        # scoring mode (avoids recomputing inside the inner loop).
        if split_e_circular:
            obs_e_cont = obs_e[obs_e > 0]
            n_obs_circ = int(np.sum(obs_e == 0))
            n_obs_e_total = len(obs_e)
            logger.info("split_e_circular: %d circular (e=0) + %d eccentric "
                         "out of %d observed",
                         n_obs_circ, len(obs_e_cont), n_obs_e_total)

        # Lightweight index: which steps have been processed and their
        # grid indices.  The actual detected arrays live on disk as
        # individual shard files (det_shards/step_NNNNNN.npz) to avoid
        # unbounded memory growth.
        step_to_ijkl = []

        # Global 2D detection-probability histograms
        global_hist_total = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                             for pair in _HIST_PAIRS}
        global_hist_det = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                           for pair in _HIST_PAIRS}

        # --- Resume from checkpoint if available ---
        all_results = []
        # Set of step indices already present in checkpoint_results.csv;
        # gates incremental CSV appends to avoid duplicate rows on resume.
        m_csv = set()
        start_step = 0
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_npz = os.path.join(checkpoint_dir, "checkpoint_cubes.npz")
            ckpt_csv = os.path.join(checkpoint_dir,
                                    "checkpoint_results.csv")
            if os.path.exists(ckpt_npz) and os.path.exists(ckpt_csv):
                ckpt = np.load(ckpt_npz, allow_pickle=True)

                # Full validation — grid values, n_inject, seed, preset name.
                # Refuse to resume on any mismatch rather than silently
                # overwriting the checkpoint on the next save.
                mismatches = []

                ckpt_gmf_key = ("gmf_ks_cube" if "gmf_ks_cube" in ckpt.files
                                else "gmf_cube")
                if ckpt[ckpt_gmf_key].shape != gmf_cube.shape:
                    mismatches.append(
                        "cube shape: stored=%s current=%s" % (
                            ckpt[ckpt_gmf_key].shape, gmf_cube.shape))

                for name, cur in (("pi_grid", pi_grid),
                                  ("kappa_grid", kappa_grid),
                                  ("eta_grid", eta_grid),
                                  ("fbin_grid", fbin_grid)):
                    stored = ckpt[name]
                    if (stored.shape != np.asarray(cur).shape
                            or not np.allclose(stored, cur)):
                        mismatches.append(
                            "%s: stored=%s current=%s" % (
                                name, stored, np.asarray(cur)))

                if "n_inject_per_star" in ckpt.files:
                    stored_n = int(ckpt["n_inject_per_star"])
                    if stored_n != int(n_inject_per_star):
                        mismatches.append(
                            "n_inject_per_star: stored=%d current=%d" % (
                                stored_n, int(n_inject_per_star)))

                if "seed" in ckpt.files:
                    stored_seed = int(ckpt["seed"])
                    if stored_seed != int(seed):
                        mismatches.append(
                            "seed: stored=%d current=%d" % (
                                stored_seed, int(seed)))

                if "preset_name" in ckpt.files:
                    stored_preset = str(ckpt["preset_name"])
                    if stored_preset != str(preset_name):
                        mismatches.append(
                            "preset_name: stored=%s current=%s" % (
                                stored_preset, preset_name))

                if mismatches:
                    raise RuntimeError(
                        "Checkpoint in %s is incompatible with the current "
                        "run:\n  - %s\n"
                        "Refusing to overwrite. Either pass --output-dir "
                        "<new_dir> or delete the checkpoint_* files in "
                        "that directory." % (
                            checkpoint_dir, "\n  - ".join(mismatches)))

                pdet_cube = ckpt["pdet_cube"]
                # Restore per-test cubes (with backward compat for
                # old checkpoints that only have KS)
                for tname in _ALL_TESTS:
                    gmf_key = "gmf_%s_cube" % tname
                    if gmf_key in ckpt.files:
                        gmf_cubes[tname][:] = ckpt[gmf_key]
                    elif tname == "ks" and "gmf_cube" in ckpt.files:
                        gmf_cubes["ks"][:] = ckpt["gmf_cube"]
                    for par in ("logP", "e", "K1"):
                        tc_key = "%s_%s_cube" % (tname, par)
                        if tc_key in ckpt.files:
                            test_cubes[tname][par][:] = ckpt[tc_key]
                        elif tname == "ks":
                            # Backward compat: old key names
                            old_key = "ks_%s_cube" % par
                            if old_key in ckpt.files:
                                test_cubes["ks"][par][:] = ckpt[old_key]
                # Update aliases
                gmf_cube = gmf_cubes["ks"]
                ks_logP_cube = test_cubes["ks"]["logP"]
                ks_e_cube = test_cubes["ks"]["e"]
                ks_K1_cube = test_cubes["ks"]["K1"]
                start_step = int(ckpt["completed_steps"])
                prev_df = pd.read_csv(ckpt_csv)
                all_results = prev_df.to_dict("records")
                if "step" in prev_df.columns:
                    m_csv = set(int(s) for s in prev_df["step"].tolist())
                # Restore global histograms from checkpoint
                hists_ckpt = os.path.join(checkpoint_dir,
                                          "checkpoint_hists.npz")
                if os.path.exists(hists_ckpt):
                    hdata = np.load(hists_ckpt)
                    for (a, b) in _HIST_PAIRS:
                        key_t = "hist_total_%s_%s" % (a, b)
                        key_d = "hist_det_%s_%s" % (a, b)
                        if key_t in hdata.files:
                            global_hist_total[(a, b)] = hdata[key_t]
                            global_hist_det[(a, b)] = hdata[key_d]
                # Restore step_to_ijkl from det_index
                det_idx_path = os.path.join(checkpoint_dir,
                                            "det_index.npz")
                if os.path.exists(det_idx_path):
                    idx_data = np.load(det_idx_path)
                    step_to_ijkl = [
                        tuple(row) for row in idx_data["step_to_ijkl"]
                    ]
                logger.info("Resumed from checkpoint at step %d/%d",
                            start_step, total)

        if grid_end is None:
            grid_end = total

        grid_points = list(itertools.product(
            enumerate(pi_grid), enumerate(kappa_grid),
            enumerate(eta_grid), enumerate(fbin_grid),
        ))

        # Build the per-run scoring context once. Both the local
        # _score_and_accumulate (forward path) and the resume re-score
        # workers consume this through _compute_scores.
        scoring_ctx = _make_scoring_ctx(
            obs_logP=obs_logP, obs_e=obs_e, obs_K1=obs_K1,
            clip_range=clip_range,
            split_e_circular=split_e_circular,
            obs_e_cont=obs_e_cont if split_e_circular else None,
            n_obs_circ=n_obs_circ if split_e_circular else None,
            n_obs_e_total=n_obs_e_total if split_e_circular else None,
            N_det_obs=N_det_obs, N_stars=N_stars,
        )

        # Buffer for per-grid-point CSV rows in parallel mode. Flushed
        # at every cube checkpoint and once at end-of-run (not per-row,
        # which previously cost a file open+stat per result).
        _pending_csv_rows = []

        def _flush_csv_rows():
            """Append any buffered rows to checkpoint_results.csv."""
            if not _pending_csv_rows or not checkpoint_dir:
                return
            _csv = os.path.join(checkpoint_dir, "checkpoint_results.csv")
            header = not os.path.exists(_csv)
            pd.DataFrame(_pending_csv_rows).to_csv(
                _csv, mode="a", index=False, header=header,
            )
            _pending_csv_rows.clear()

        # Helper: write one grid-point result into cubes / hists / CSV.
        # If `scores` is given (parallel resume path), skip the KS/AD/CvM
        # work — the worker already did it. Otherwise compute scores
        # inline (forward path, single-threaded but bounded by the
        # inflight semaphore around imap_unordered).
        def _score_and_accumulate(step, i, j, k, l, pi, kappa, eta, fbin,
                                  res, scores=None):
            p_det = res["p_det"]
            pdet_cube[i, j, k, l] = p_det

            if scores is None:
                scores = _compute_scores(res, fbin, scoring_ctx)

            for tname in _ALL_TESTS:
                test_cubes[tname]["logP"][i, j, k, l] = \
                    scores["%s_p_logP" % tname]
                test_cubes[tname]["e"][i, j, k, l] = \
                    scores["%s_p_e" % tname]
                test_cubes[tname]["K1"][i, j, k, l] = \
                    scores["%s_p_K1" % tname]
                if split_e_circular:
                    test_cubes[tname]["e_circ"][i, j, k, l] = \
                        scores["%s_p_e_circ" % tname]
                gmf_cubes[tname][i, j, k, l] = scores["log_gmf_%s" % tname]

            # Accumulate 2D histograms (fixed-size, negligible memory).
            for pair in _HIST_PAIRS:
                global_hist_total[pair] += res["hist_total"][pair]
                global_hist_det[pair] += res["hist_det"][pair]

            # Flush per-grid-point detected arrays to a shard file on
            # disk in serial mode. In parallel modes the shard was already
            # written (forward: by _worker_grid_point; resume: by the
            # original run that produced the shard).
            if checkpoint_dir and not parallel_grid:
                _save_det_shard(checkpoint_dir, step, i, j, k, l, res)
            step_to_ijkl.append((step, i, j, k, l))

            # Build the CSV row from res scalars + scores. Drop any
            # heavy arrays/dicts that may have been on res.
            row = {
                "step": step,
                "pi": pi, "kappa": kappa, "eta": eta, "fbin": fbin,
                "p_det": p_det,
                "n_detected": int(res.get("n_detected", 0)),
                "n_physical": int(res.get("n_physical", 0)),
                "n_rlof": int(res.get("n_rlof", 0)),
                "n_false_positive": int(res.get("n_false_positive", 0)),
            }
            row.update({
                k: v for k, v in scores.items()
                if not isinstance(v, (np.ndarray, dict))
            })

            if not parallel_grid:
                all_results.append(row)
            elif step not in m_csv:
                _pending_csv_rows.append(row)
                m_csv.add(step)

            return p_det, scores["log_gmf_ks"]

        t0 = time.time()
        steps_done = 0
        # Auto-scale checkpoint cadence: ~100 checkpoints per run regardless
        # of grid size. Works for tiny SLURM tasks and 270k-point grids alike.
        checkpoint_every = max(1, (grid_end - grid_start) // 100)

        # --- Parallel grid mode: one grid point per CPU ---
        if parallel_grid and n_workers > 1:
            import multiprocessing as _mp
            import threading

            # Pre-create shard directory before workers start.
            if checkpoint_dir:
                os.makedirs(os.path.join(checkpoint_dir, _DET_SHARDS_DIR),
                            exist_ok=True)

            # Build set of already-completed steps from existing shards.
            existing_shards = set()
            if checkpoint_dir:
                shard_dir = os.path.join(checkpoint_dir, _DET_SHARDS_DIR)
                for fname in os.listdir(shard_dir):
                    if fname.startswith("step_") and fname.endswith(".npz"):
                        try:
                            existing_shards.add(
                                int(fname[5:-4]))  # step_NNNNNN.npz
                        except ValueError:
                            pass
                if existing_shards:
                    logger.info("Found %d existing shard files — "
                                "re-scoring and skipping",
                                len(existing_shards))

            # ---- Resume gap re-score (parallelized) ------------------
            #
            # m_cube: steps already reflected in the loaded cube/hist
            # checkpoint (step_to_ijkl is persisted alongside the cubes).
            # gap_steps: shards on disk whose stats never made it into
            # the cubes — typically the worker→main backlog at kill time.
            # We rebuild p-values + log-GMF for each gap shard. With the
            # forward inflight throttle below, this gap is small in
            # future runs, but for legacy checkpoints it can be
            # 100k+ and warrants a worker pool.
            from tqdm import tqdm as _tqdm
            m_cube = {row[0] for row in step_to_ijkl}
            gap_steps = existing_shards - m_cube
            _shard_items = [
                (step, i, j, k, l, pi, kappa, eta, fbin)
                for step, ((i, pi), (j, kappa), (k, eta), (l, fbin))
                in enumerate(grid_points)
                if step in gap_steps
            ]

            ctx_mp = _mp.get_context("forkserver")
            if _shard_items:
                logger.info("Re-scoring %d gap shards (of %d total on disk) "
                            "across %d workers",
                            len(_shard_items), len(existing_shards),
                            n_workers)
                n_resume_pool = min(n_workers, len(_shard_items))
                resume_inflight = threading.Semaphore(n_resume_pool * 2)

                def _throttled_resume(items):
                    for t in items:
                        resume_inflight.acquire()
                        yield t

                with ctx_mp.Pool(
                        processes=n_resume_pool,
                        initializer=_init_resume_worker,
                        initargs=(checkpoint_dir, scoring_ctx),
                        maxtasksperchild=2000,
                ) as resume_pool:
                    rescored = 0
                    for r in _tqdm(
                            resume_pool.imap_unordered(
                                _resume_score_worker,
                                _throttled_resume(_shard_items)),
                            total=len(_shard_items),
                            desc="Re-scoring shards",
                            unit="pt", mininterval=1.0):
                        try:
                            if r is None:
                                continue
                            # `scores` already computed by the worker.
                            res = {
                                "p_det": r["p_det"],
                                "n_detected": r["n_detected"],
                                "n_physical": r["n_physical"],
                                "n_rlof": r["n_rlof"],
                                "n_false_positive": r["n_false_positive"],
                                "hist_total": r["hist_total"],
                                "hist_det": r["hist_det"],
                            }
                            _score_and_accumulate(
                                r["step"], r["i"], r["j"], r["k"], r["l"],
                                r["pi"], r["kappa"], r["eta"], r["fbin"],
                                res, scores=r["scores"])
                            rescored += 1
                            # Periodic CSV flush so re-score progress
                            # survives a kill mid-resume.
                            if (checkpoint_dir
                                    and rescored % checkpoint_every == 0):
                                _flush_csv_rows()
                        finally:
                            # Release AFTER consumption so the inflight
                            # cap bounds (queued + in-worker + waiting
                            # for main) — not just queued tasks.
                            resume_inflight.release()

            # Flush any remaining re-scored rows + one cube checkpoint
            # before starting the forward pass — makes the forward
            # checkpoint state consistent with what's on disk.
            if checkpoint_dir and _shard_items:
                _flush_csv_rows()
                _save_checkpoint(
                    checkpoint_dir, start_step,
                    gmf_cubes, pdet_cube, test_cubes,
                    pi_grid, kappa_grid, eta_grid, fbin_grid,
                    [],
                    n_inject_per_star=n_inject_per_star,
                    seed=seed,
                    preset_name=preset_name,
                    global_hists={
                        "total": global_hist_total,
                        "det": global_hist_det,
                    },
                )
                _save_det_index(checkpoint_dir, step_to_ijkl)

            # ---- Forward pass: run new tasks ------------------------
            pending_tasks = []
            for step, ((i, pi), (j, kappa), (k, eta), (l, fbin)) in \
                    enumerate(grid_points):
                if step < max(start_step, grid_start):
                    continue
                if step >= grid_end:
                    break
                if step in existing_shards:
                    continue
                pending_tasks.append((
                    step, i, j, k, l, pi, kappa, eta, fbin,
                    seed, n_inject_per_star,
                ))

            logger.info("Parallel grid mode: %d grid points to compute "
                        "(%d already done) across %d workers",
                        len(pending_tasks), len(existing_shards),
                        n_workers)

            if not pending_tasks:
                logger.info("All grid points already completed.")

            n_pool = min(n_workers, max(1, len(pending_tasks)))
            # Inflight throttle: caps the number of (in-worker + queued
            # for main) tasks. Workers naturally idle when main falls
            # behind, so a kill leaves ≤ 2·n_pool orphan shards instead
            # of unbounded backlog. n_pool*2 keeps every worker busy
            # plus a small buffer for IPC slack.
            inflight = threading.Semaphore(n_pool * 2)

            def _throttled(tasks):
                for t in tasks:
                    inflight.acquire()
                    yield t

            with ctx_mp.Pool(
                    processes=n_pool,
                    initializer=_init_grid_worker,
                    initargs=(self.field_mjds, self.field_arr,
                              self.rv_err_arr, self.M1_arr,
                              self.R1_arr, self.gamma_arr,
                              self.cfg, self.args_dict,
                              self.detect_method,
                              checkpoint_dir),
                    maxtasksperchild=500,
            ) as pool:
                for task_result in pool.imap_unordered(
                        _worker_grid_point, _throttled(pending_tasks)):
                    try:
                        step, i, j, k, l, res = task_result
                        pi = pi_grid[i]
                        kappa = kappa_grid[j]
                        eta = eta_grid[k]
                        fbin = fbin_grid[l]

                        p_det, log_gmf = _score_and_accumulate(
                            step, i, j, k, l, pi, kappa, eta, fbin, res)

                        steps_done += 1
                        elapsed = time.time() - t0
                        rate = steps_done / elapsed if elapsed > 0 else 0
                        remaining = len(pending_tasks) - steps_done
                        eta_s = remaining / rate if rate > 0 else 0
                        if steps_done % 500 == 0 or steps_done == len(pending_tasks):
                            logger.info(
                                "[step %d | %d/%d] π=%.2f κ=%.2f η=%.2f f=%.2f "
                                "p_det=%.3f logGMF=%.2f ETA %.0fmin",
                                step, steps_done, len(pending_tasks),
                                pi, kappa, eta, fbin,
                                p_det, log_gmf, eta_s / 60)

                        if progress_callback:
                            progress_callback(step, total, pi, kappa,
                                              eta, fbin, p_det)
                        if result_callback:
                            result_callback(step, total, res)

                        # Periodic checkpoint at ~1% intervals so a
                        # crash never loses more than ~1% of work.
                        if (checkpoint_dir and steps_done > 0
                                and steps_done % checkpoint_every == 0):
                            _flush_csv_rows()
                            _save_checkpoint(
                                checkpoint_dir, start_step + steps_done,
                                gmf_cubes, pdet_cube, test_cubes,
                                pi_grid, kappa_grid, eta_grid, fbin_grid,
                                [],
                                n_inject_per_star=n_inject_per_star,
                                seed=seed,
                                preset_name=preset_name,
                                global_hists={
                                    "total": global_hist_total,
                                    "det": global_hist_det,
                                },
                            )
                            _save_det_index(checkpoint_dir, step_to_ijkl)
                            logger.info("Checkpoint saved at %d steps done",
                                        steps_done)
                    finally:
                        # Release AFTER processing so the cap bounds
                        # (queued + in-worker + waiting for main) — not
                        # just queued tasks. This is what stops workers
                        # from running ahead and orphaning shards.
                        inflight.release()

            # Save one final checkpoint after all grid points complete.
            if checkpoint_dir:
                _flush_csv_rows()
                _save_checkpoint(
                    checkpoint_dir, grid_end,
                    gmf_cubes, pdet_cube, test_cubes,
                    pi_grid, kappa_grid, eta_grid, fbin_grid,
                    [],  # CSV already flushed above
                    n_inject_per_star=n_inject_per_star,
                    seed=seed,
                    preset_name=preset_name,
                    global_hists={
                        "total": global_hist_total,
                        "det": global_hist_det,
                    },
                )
                _save_det_index(checkpoint_dir, step_to_ijkl)

        else:
            # --- Sequential grid loop (original behaviour) ---
            for step, ((i, pi), (j, kappa), (k, eta), (l, fbin)) in \
                    enumerate(grid_points):
                if step < max(start_step, grid_start):
                    continue
                if step >= grid_end:
                    break

                rng = np.random.default_rng(seed + step * 137)

                res = self._run_one_grid_point(
                    pi, kappa, eta, fbin,
                    n_inject_per_star, rng,
                    n_workers=n_workers,
                )

                p_det, log_gmf = _score_and_accumulate(
                    step, i, j, k, l, pi, kappa, eta, fbin, res)

                steps_done += 1
                elapsed = time.time() - t0
                rate = steps_done / elapsed if elapsed > 0 else 0
                remaining = (grid_end - step - 1)
                eta_s = remaining / rate if rate > 0 else 0
                logger.info(
                    "[step %d | %d/%d in range] π=%.2f κ=%.2f η=%.2f "
                    "f=%.2f p_det=%.3f logGMF=%.2f ETA %.0fmin",
                    step, steps_done, grid_end - grid_start,
                    pi, kappa, eta, fbin,
                    p_det, log_gmf, eta_s / 60)

                if progress_callback:
                    progress_callback(step, total, pi, kappa,
                                      eta, fbin, p_det)

                if result_callback:
                    result_callback(step, total, res)

                # --- Checkpoint: save after every grid point ---
                if checkpoint_dir:
                    _save_checkpoint(
                        checkpoint_dir, step + 1,
                        gmf_cubes, pdet_cube, test_cubes,
                        pi_grid, kappa_grid, eta_grid, fbin_grid,
                        all_results,
                        n_inject_per_star=n_inject_per_star,
                        seed=seed,
                        preset_name=preset_name,
                        global_hists={
                            "total": global_hist_total,
                            "det": global_hist_det,
                        },
                    )
                    _save_det_index(checkpoint_dir, step_to_ijkl)

        # Best fit (per test)
        best_fits = {}
        for tname in _ALL_TESTS:
            gc = gmf_cubes[tname]
            idx = np.unravel_index(np.nanargmax(gc), gc.shape)
            best_fits[tname] = (float(pi_grid[idx[0]]),
                                float(kappa_grid[idx[1]]),
                                float(eta_grid[idx[2]]),
                                float(fbin_grid[idx[3]]))

        # KS best fit for backward compat / logging
        best_pi, best_kappa, best_eta, best_fbin = best_fits["ks"]
        best_idx = np.unravel_index(
            np.nanargmax(gmf_cubes["ks"]), gmf_cubes["ks"].shape)

        logger.info("Grid search complete in %.1f min", (time.time()-t0)/60)
        for tname in _ALL_TESTS:
            bf = best_fits[tname]
            logger.info("Best fit (%s): π=%.2f, κ=%.2f, η=%.2f, f_bin=%.2f",
                         tname.upper(), *bf)

        return {
            "pi_grid": pi_grid,
            "kappa_grid": kappa_grid,
            "eta_grid": eta_grid,
            "fbin_grid": fbin_grid,
            "results": all_results,
            "best_fit": (best_pi, best_kappa, best_eta, best_fbin),
            "best_fits": best_fits,
            "best_idx": best_idx,
            "gmf_cube": gmf_cube,
            "gmf_cubes": gmf_cubes,
            "pdet_cube": pdet_cube,
            "test_cubes": test_cubes,
            "ks_logP_cube": ks_logP_cube,
            "ks_e_cube": ks_e_cube,
            "ks_K1_cube": ks_K1_cube,
            "N_stars": self.cfg.get("n_stars_sample", len(self.M1_arr)),
            "N_det_obs": len(obs_logP),
            "step_to_ijkl": step_to_ijkl,
            "checkpoint_dir": checkpoint_dir,
            "global_hists": {
                "total": global_hist_total,
                "det": global_hist_det,
            },
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid_results(results, output_dir=None, obs_logP=None, obs_e=None,
                      obs_K1=None):
    """
    Plot grid search results: 2D marginalized GMF heatmaps,
    1D marginalized posteriors, best-fit CDF comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pi_grid = results["pi_grid"]
    kappa_grid = results["kappa_grid"]
    eta_grid = results["eta_grid"]
    fbin_grid = results["fbin_grid"]
    gmf = results["gmf_cube"]
    best = results["best_fit"]

    # Convert log GMF to probability
    gmf_shifted = gmf - np.nanmax(gmf)
    prob = np.exp(gmf_shifted)
    prob = np.nan_to_num(prob, nan=0.0)

    param_names = ["π", "κ", "η", "f_bin"]
    grids = [pi_grid, kappa_grid, eta_grid, fbin_grid]
    # Reference values (Sana+2012 Galactic)
    ref_vals = [-0.55, -0.10, -0.45, 0.69]

    # ====== 1D Marginalized Posteriors ======
    fig_1d, axes_1d = plt.subplots(1, 4, figsize=(18, 4))
    colors = ["#4393c3", "#d6604d", "#5aae61", "#9970ab"]

    for ax, axis_idx, label, color, ref in zip(
            axes_1d, range(4), param_names, colors, ref_vals):
        grid = grids[axis_idx]
        axes_to_sum = tuple(i for i in range(4) if i != axis_idx)
        post_1d = np.nansum(prob, axis=axes_to_sum)

        # Normalize
        if np.sum(post_1d) > 0 and len(grid) > 1:
            post_1d /= np.trapz(post_1d, grid)

        ax.fill_between(grid, post_1d, alpha=0.3, color=color)
        ax.plot(grid, post_1d, color=color, lw=2)

        # Mode
        mode_idx = np.argmax(post_1d)
        mode = grid[mode_idx]
        ax.axvline(mode, color="crimson", lw=1.5, ls="--",
                   label=f"Mode = {mode:.2f}")

        # 68% CI
        cdf = np.cumsum(post_1d)
        if cdf[-1] > 0:
            cdf /= cdf[-1]
            lo = np.interp(0.16, cdf, grid)
            hi = np.interp(0.84, cdf, grid)
            ax.axvspan(lo, hi, alpha=0.12, color="crimson",
                       label=f"68% CI [{lo:.2f}, {hi:.2f}]")

        ax.axvline(ref, color="grey", lw=1, ls=":",
                   label=f"Sana+12: {ref}")
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Posterior density", fontsize=11)
        ax.legend(fontsize=7)

    plt.suptitle(
        f"Best fit: π={best[0]:.2f}, κ={best[1]:.2f}, "
        f"η={best[2]:.2f}, f_bin={best[3]:.2f}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_1d.savefig(os.path.join(output_dir, "grid_1d_posteriors.pdf"),
                       bbox_inches="tight")
        logger.info("Saved 1D posteriors to %s/grid_1d_posteriors.pdf",
                     output_dir)

    # ====== 2D Marginalized GMF Heatmaps ======
    pairs = [
        (0, 1, "π", "κ"),
        (0, 2, "π", "η"),
        (0, 3, "π", "f_bin"),
        (1, 2, "κ", "η"),
        (1, 3, "κ", "f_bin"),
        (2, 3, "η", "f_bin"),
    ]

    fig_2d, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (i1, i2, l1, l2) in zip(axes, pairs):
        axes_to_sum = tuple(i for i in range(4) if i not in (i1, i2))
        marg = np.nansum(prob, axis=axes_to_sum)

        g1 = grids[i1]
        g2 = grids[i2]

        im = ax.imshow(marg.T, origin="lower", aspect="auto",
                       extent=[g1[0], g1[-1], g2[0], g2[-1]],
                       cmap="RdYlBu_r", interpolation="bilinear")
        ax.set_xlabel(l1, fontsize=12)
        ax.set_ylabel(l2, fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Marginal prob")
        ax.plot(best[i1], best[i2], "w*", markersize=12,
                markeredgecolor="k", markeredgewidth=1)
        ax.set_title(f"{l1} vs {l2}", fontsize=11)

    plt.suptitle(
        f"Best fit: π={best[0]:.2f}, κ={best[1]:.2f}, "
        f"η={best[2]:.2f}, f_bin={best[3]:.2f}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if output_dir:
        fig_2d.savefig(os.path.join(output_dir, "grid_2d_gmf.pdf"),
                       bbox_inches="tight")
        logger.info("Saved 2D heatmaps to %s/grid_2d_gmf.pdf", output_dir)

    # ====== CDF comparison: best-fit detected vs observed ======
    fig_cdf = None
    best_res = None
    if obs_logP is not None:
        # Find the best-fit result entry (the one with highest log_gmf
        # that still has the detected arrays).
        best_pi, best_kappa, best_eta, best_fbin = best
        for r in results["results"]:
            if (np.isclose(r["pi"], best_pi)
                    and np.isclose(r["kappa"], best_kappa)
                    and np.isclose(r["eta"], best_eta)
                    and np.isclose(r["fbin"], best_fbin)):
                # Check that detected arrays survived (not stripped by
                # checkpoint round-trip).
                if isinstance(r.get("logP_det"), np.ndarray):
                    best_res = r
                break

    if best_res is not None and len(best_res["logP_det"]) >= 2:
        fig_cdf, axes_cdf = plt.subplots(1, 3, figsize=(16, 5))

        param_pairs = [
            ("logP_det", obs_logP, r"$\log_{10}(P/\mathrm{d})$", "#4393c3"),
            ("e_det", obs_e, "$e$", "#d6604d"),
            ("K1_det", obs_K1, "$K_1$ [km/s]", "#5aae61"),
        ]

        for ax, (key, obs_arr, xlabel, color) in zip(axes_cdf, param_pairs):
            sim = np.sort(best_res[key])
            obs_s = np.sort(obs_arr)

            # Empirical CDF
            sim_cdf = np.arange(1, len(sim) + 1) / len(sim)
            obs_cdf = np.arange(1, len(obs_s) + 1) / len(obs_s)

            ax.step(obs_s, obs_cdf, where="post", color="k", lw=2,
                    label=f"Observed (n={len(obs_s)})")
            ax.step(sim, sim_cdf, where="post", color=color, lw=2,
                    ls="--",
                    label=f"Simulated (n={len(sim)})")

            # KS p-value annotation
            ks_key = {"logP_det": "ks_p_logP", "e_det": "ks_p_e",
                      "K1_det": "ks_p_K1"}[key]
            pval = best_res.get(ks_key, np.nan)
            ax.text(0.05, 0.95, f"KS p = {pval:.3f}",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="wheat", alpha=0.5))

            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("CDF", fontsize=12)
            ax.legend(fontsize=9)
            ax.set_ylim(0, 1.05)

        plt.suptitle(
            f"Best fit CDF: π={best[0]:.2f}, κ={best[1]:.2f}, "
            f"η={best[2]:.2f}, f_bin={best[3]:.2f}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        if output_dir:
            fig_cdf.savefig(os.path.join(output_dir, "grid_cdf_bestfit.pdf"),
                            bbox_inches="tight")
            logger.info("Saved CDF comparison to %s/grid_cdf_bestfit.pdf",
                        output_dir)

    return {"fig_1d": fig_1d, "fig_2d": fig_2d, "fig_cdf": fig_cdf}


def format_grid_summary(results):
    """Create a summary DataFrame of all grid points, sorted by GMF."""
    rows = []
    for r in results["results"]:
        row = {
            "π": f"{r['pi']:.2f}",
            "κ": f"{r['kappa']:.2f}",
            "η": f"{r['eta']:.2f}",
            "f_bin": f"{r['fbin']:.2f}",
            "p_det": f"{r['p_det']:.3f}",
            "P_binom": f"{r['p_binom']:.4f}",
            "N_det": r["n_detected"],
        }
        for tname in _ALL_TESTS:
            prefix = tname.upper()
            row["%s_logP" % prefix] = f"{r.get('%s_p_logP' % tname, 0):.3f}"
            row["%s_e" % prefix] = f"{r.get('%s_p_e' % tname, 0):.3f}"
            row["%s_K1" % prefix] = f"{r.get('%s_p_K1' % tname, 0):.3f}"
            lgmf = r.get("log_gmf_%s" % tname, r.get("log_gmf", -np.inf))
            row["log_GMF_%s" % prefix] = (
                f"{lgmf:.2f}" if np.isfinite(lgmf) else "-inf")
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_sort"] = [r.get("log_gmf_ks", r.get("log_gmf", -np.inf))
                   for r in results["results"]]
    df = df.sort_values("_sort", ascending=False).drop("_sort", axis=1)
    logger.info("format_grid_summary: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# SLURM aggregation
# ---------------------------------------------------------------------------

def aggregate_tasks(base_dir, output_dir=None):
    """
    Merge partial results from SLURM array tasks into final output.

    Discovers all task_*/ subdirectories under base_dir, loads their
    checkpoint_cubes.npz and checkpoint_results.csv, merges the cubes,
    and produces the final grid_cubes.npz and PDF plots.

    Parameters
    ----------
    base_dir : str
        Directory containing task_*/ subdirectories.
    output_dir : str or None
        Where to write merged output. Defaults to base_dir.
    """
    import glob as _glob

    if output_dir is None:
        output_dir = base_dir

    task_dirs = sorted(_glob.glob(os.path.join(base_dir, "task_*")))
    if not task_dirs:
        raise RuntimeError(
            "No task_*/ subdirectories found in %s" % base_dir)

    logger.info("aggregate_tasks: found %d task dirs in %s",
                len(task_dirs), base_dir)

    # Load first task to get grid shapes and metadata.
    first_npz = np.load(
        os.path.join(task_dirs[0], "checkpoint_cubes.npz"),
        allow_pickle=True)
    pi_grid = first_npz["pi_grid"]
    kappa_grid = first_npz["kappa_grid"]
    eta_grid = first_npz["eta_grid"]
    fbin_grid = first_npz["fbin_grid"]
    _gmf_key = "gmf_ks_cube" if "gmf_ks_cube" in first_npz.files else "gmf_cube"
    shape = first_npz[_gmf_key].shape

    # Initialise merged cubes.
    pdet_cube = np.zeros(shape)
    gmf_cubes_agg = {}
    test_cubes_agg = {}
    for tname in _ALL_TESTS:
        gmf_cubes_agg[tname] = np.full(shape, -np.inf)
        test_cubes_agg[tname] = {
            "logP": np.zeros(shape),
            "e": np.zeros(shape),
            "K1": np.zeros(shape),
        }
    all_results_dfs = []

    # Detected-array merging accumulators
    all_det_logP = []
    all_det_e = []
    all_det_K1 = []
    all_det_q = []
    all_step_to_ijkl = []
    merged_hist_total = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                         for pair in _HIST_PAIRS}
    merged_hist_det = {pair: np.zeros((_HIST_NBINS, _HIST_NBINS))
                       for pair in _HIST_PAIRS}
    has_detected = False

    for td in task_dirs:
        npz_path = os.path.join(td, "checkpoint_cubes.npz")
        csv_path = os.path.join(td, "checkpoint_results.csv")
        if not os.path.exists(npz_path) or not os.path.exists(csv_path):
            logger.warning("  Skipping incomplete task dir: %s", td)
            continue

        ckpt = np.load(npz_path, allow_pickle=True)

        # Merge cubes: each task only fills its own cells.
        pdet_cube += ckpt["pdet_cube"]
        # Per-test cubes (with backward compat)
        for tname in _ALL_TESTS:
            gmf_key = "gmf_%s_cube" % tname
            if gmf_key in ckpt.files:
                gmf_cubes_agg[tname] = np.maximum(
                    gmf_cubes_agg[tname], ckpt[gmf_key])
            elif tname == "ks" and "gmf_cube" in ckpt.files:
                gmf_cubes_agg["ks"] = np.maximum(
                    gmf_cubes_agg["ks"], ckpt["gmf_cube"])
            for par in ("logP", "e", "K1"):
                tc_key = "%s_%s_cube" % (tname, par)
                if tc_key in ckpt.files:
                    test_cubes_agg[tname][par] += ckpt[tc_key]
                elif tname == "ks":
                    old_key = "ks_%s_cube" % par
                    if old_key in ckpt.files:
                        test_cubes_agg["ks"][par] += ckpt[old_key]
            # e_circ cube (split_e_circular mode)
            ec_key = "%s_e_circ_cube" % tname
            if ec_key in ckpt.files:
                if "e_circ" not in test_cubes_agg[tname]:
                    test_cubes_agg[tname]["e_circ"] = np.zeros(shape)
                test_cubes_agg[tname]["e_circ"] += ckpt[ec_key]

        # Merge detected arrays and histograms — shard-based or legacy
        shard_dir = os.path.join(td, _DET_SHARDS_DIR)
        det_index_path = os.path.join(td, "det_index.npz")
        det_legacy_path = os.path.join(td, "checkpoint_detected.npz")

        if os.path.isdir(shard_dir) and os.path.exists(det_index_path):
            # New shard-based format
            has_detected = True
            idx_data = np.load(det_index_path)
            task_ijkl = idx_data["step_to_ijkl"]
            all_step_to_ijkl.extend(task_ijkl.tolist())
            # Note: shards are on disk; we copy them to the merged
            # output shard dir below.

            # Merge histograms from checkpoint_hists.npz
            hists_path = os.path.join(td, "checkpoint_hists.npz")
            if os.path.exists(hists_path):
                hdata = np.load(hists_path)
                for pair in _HIST_PAIRS:
                    a, b = pair
                    key_t = "hist_total_%s_%s" % (a, b)
                    key_d = "hist_det_%s_%s" % (a, b)
                    if key_t in hdata.files:
                        merged_hist_total[pair] += hdata[key_t]
                        merged_hist_det[pair] += hdata[key_d]

        elif os.path.exists(det_legacy_path):
            # Legacy concatenated format
            has_detected = True
            det = np.load(det_legacy_path, allow_pickle=True)
            task_offsets = det["offsets"]
            task_ijkl = det["step_to_ijkl"]

            for idx in range(len(task_ijkl)):
                lo = int(task_offsets[idx])
                hi = int(task_offsets[idx + 1])
                all_det_logP.append(det["all_logP_det"][lo:hi])
                all_det_e.append(det["all_e_det"][lo:hi])
                all_det_K1.append(det["all_K1_det"][lo:hi])
                all_det_q.append(det["all_q_det"][lo:hi])
            all_step_to_ijkl.extend(task_ijkl.tolist())

            for pair in _HIST_PAIRS:
                a, b = pair
                key_t = "hist_total_%s_%s" % (a, b)
                key_d = "hist_det_%s_%s" % (a, b)
                if key_t in det.files:
                    merged_hist_total[pair] += det[key_t]
                    merged_hist_det[pair] += det[key_d]

        df = pd.read_csv(csv_path)
        all_results_dfs.append(df)
        logger.info("  Loaded %s: %d rows", td, len(df))

    if not all_results_dfs:
        raise RuntimeError("No valid task results found in %s" % base_dir)

    merged_df = pd.concat(all_results_dfs, ignore_index=True)
    total_expected = len(pi_grid) * len(kappa_grid) * len(eta_grid) * \
        len(fbin_grid)
    if len(merged_df) < total_expected:
        logger.warning(
            "Merged %d rows but expected %d — some tasks may be incomplete",
            len(merged_df), total_expected)

    # Best fit from merged cubes (per test).
    best_fits = {}
    for tname in _ALL_TESTS:
        gc = gmf_cubes_agg[tname]
        idx = np.unravel_index(np.nanargmax(gc), gc.shape)
        best_fits[tname] = (float(pi_grid[idx[0]]),
                            float(kappa_grid[idx[1]]),
                            float(eta_grid[idx[2]]),
                            float(fbin_grid[idx[3]]))
        logger.info("Best fit (%s): π=%.2f, κ=%.2f, η=%.2f, f_bin=%.2f",
                     tname.upper(), *best_fits[tname])

    best_fit = best_fits["ks"]
    best_idx = np.unravel_index(
        np.nanargmax(gmf_cubes_agg["ks"]), gmf_cubes_agg["ks"].shape)

    # Build results dict for plotting / summary.
    all_results = merged_df.to_dict("records")
    results = {
        "pi_grid": pi_grid,
        "kappa_grid": kappa_grid,
        "eta_grid": eta_grid,
        "fbin_grid": fbin_grid,
        "results": all_results,
        "best_fit": best_fit,
        "best_fits": best_fits,
        "best_idx": best_idx,
        "gmf_cube": gmf_cubes_agg["ks"],
        "gmf_cubes": gmf_cubes_agg,
        "pdet_cube": pdet_cube,
        "test_cubes": test_cubes_agg,
        "ks_logP_cube": test_cubes_agg["ks"]["logP"],
        "ks_e_cube": test_cubes_agg["ks"]["e"],
        "ks_K1_cube": test_cubes_agg["ks"]["K1"],
    }

    # Save merged output.
    os.makedirs(output_dir, exist_ok=True)

    save_kw_cubes = dict(
        pdet_cube=pdet_cube,
        pi_grid=pi_grid,
        kappa_grid=kappa_grid,
        eta_grid=eta_grid,
        fbin_grid=fbin_grid,
    )
    for tname in _ALL_TESTS:
        save_kw_cubes["gmf_%s_cube" % tname] = gmf_cubes_agg[tname]
        for par in ("logP", "e", "K1"):
            save_kw_cubes["%s_%s_cube" % (tname, par)] = \
                test_cubes_agg[tname][par]
        if "e_circ" in test_cubes_agg[tname]:
            save_kw_cubes["%s_e_circ_cube" % tname] = \
                test_cubes_agg[tname]["e_circ"]
    np.savez(os.path.join(output_dir, "grid_cubes.npz"), **save_kw_cubes)
    logger.info("Saved merged cubes to %s/grid_cubes.npz", output_dir)

    # Save merged histograms and detected-array index
    if has_detected:
        # Copy shard files from task dirs to merged output
        import shutil
        merged_shard_dir = os.path.join(output_dir, _DET_SHARDS_DIR)
        os.makedirs(merged_shard_dir, exist_ok=True)
        for td in task_dirs:
            src_shard_dir = os.path.join(td, _DET_SHARDS_DIR)
            if os.path.isdir(src_shard_dir):
                for fname in os.listdir(src_shard_dir):
                    shutil.copy2(os.path.join(src_shard_dir, fname),
                                 os.path.join(merged_shard_dir, fname))
        _save_det_index(output_dir, all_step_to_ijkl)

        # Save merged histograms
        save_kw = {}
        for k, v in _HIST_BINS.items():
            save_kw["bins_%s" % k] = v
        for (a, b) in _HIST_PAIRS:
            save_kw["hist_total_%s_%s" % (a, b)] = merged_hist_total[(a, b)]
            save_kw["hist_det_%s_%s" % (a, b)] = merged_hist_det[(a, b)]
        np.savez(os.path.join(output_dir, "grid_hists.npz"), **save_kw)
        logger.info("Saved merged histograms + %d shard files to %s",
                     len(all_step_to_ijkl), output_dir)

    plot_grid_results(results, output_dir)

    # Clean up task directories now that everything is merged.
    import shutil as _shutil
    for td in task_dirs:
        _shutil.rmtree(td)
    logger.info("Aggregation complete. Deleted %d task directories.",
                len(task_dirs))

    return results


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _bloem_id_root(star_id):
    """Strip multi-component SB labels (e.g. ' Aa,Ab', ' B') so the star_id
    matches the underlying BLOeM star in mass_bloem.csv. Returns the
    cleaned root id, e.g. '4-080 Aa,Ab' -> '4-080'."""
    s = re.sub(r"\$\^\{[^}]*\}\$", "", str(star_id))  # drop $^{(a)}$
    s = re.sub(r"\s+(Aa,Ab|Aa|Ab|A|B|C).*$", "", s)
    return s.strip()


def load_observed_star_properties(mass_file, rv_dir, sb1_tex, sb2_tex):
    """Build the bias-grid star sample from the SB1+SB2 LaTeX tables.

    Returns one row per detected binary in the LaTeX tables (~71 entries),
    with M1/R1 from mass_bloem.csv and rv_err/gamma/MJDs from per-star
    CCF_RVs CSVs in rv_dir. Uses sample-median fallbacks when an entry is
    missing — guarantees the full LaTeX sample size, mirroring how
    horvitz_thompson.py builds its star list.

    Returns
    -------
    DataFrame with columns: ID, Mspec, R_star, field, rv_err, gamma.
    """
    from simulations.horvitz_thompson import load_observed_binaries

    obs_binaries = load_observed_binaries(sb1_tex, sb2_tex)

    # Pre-load mass file (full BLOeM catalog) for M1/R1 lookup, with
    # cleaned key column.
    massdf = pd.read_csv(mass_file)
    massdf["_key"] = massdf["ID"].astype(str).str.replace(
        "BLOeM_", "", regex=False)
    M1_med = float(massdf["Mspec"].median())
    R1_med = float(massdf["R_star"].median())

    # Index CSV files in rv_dir by cleaned star id.
    csv_by_id = {}
    if rv_dir and os.path.isdir(rv_dir):
        import glob
        for f in glob.glob(os.path.join(rv_dir, "*_CCF_RVs.csv")):
            base = os.path.basename(f).replace("_CCF_RVs.csv", "")
            csv_by_id[base.replace("BLOeM_", "")] = f

    rows = []
    n_csv_hit = 0
    n_mass_hit = 0
    for _, b in obs_binaries.iterrows():
        sid_raw = str(b["star_id"])
        sid = _bloem_id_root(sid_raw)

        # M1, R1 from mass file (median fallback).
        mrow = massdf[massdf["_key"] == sid]
        if len(mrow) > 0:
            M1 = float(mrow.iloc[0]["Mspec"])
            R1 = float(mrow.iloc[0]["R_star"])
            n_mass_hit += 1
        else:
            M1 = M1_med
            R1 = R1_med

        # rv_err / gamma / field from CSV (defaults if missing).
        csv_path = csv_by_id.get(sid)
        if csv_path is not None:
            try:
                rv_df = pd.read_csv(csv_path)
                rv_err = float(rv_df["Mean RVsig"].median())
                gamma_csv = float(rv_df["Mean RV"].median())
                star_mjds = rv_df["MJD"].values
                best_field = 0
                best_overlap = 0
                for fi, fld_mjds in enumerate(BLOEM_MJD_ARRAYS):
                    overlap = sum(1 for m in star_mjds
                                  if any(abs(m - fm) < 0.5
                                         for fm in fld_mjds))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_field = fi
                n_csv_hit += 1
            except Exception:
                rv_err = 2.0
                gamma_csv = float(b.get("gamma", 168.0))
                best_field = 0
        else:
            rv_err = 2.0
            gamma_csv = float(b.get("gamma", 168.0))
            best_field = 0

        # Prefer the orbital-solution gamma from the LaTeX table when
        # available; fall back to the per-CSV median.
        gamma = float(b["gamma"]) if pd.notna(b.get("gamma")) else gamma_csv

        rows.append({
            "ID": sid_raw,        # keep the raw label so 4-080 Aa,Ab and
                                  # 4-080 B remain distinct rows
            "_root": sid,
            "Mspec": M1,
            "R_star": R1,
            "field": best_field,
            "rv_err": rv_err,
            "gamma": gamma,
        })

    df = pd.DataFrame(rows)
    logger.info("load_observed_star_properties: %d stars built "
                "(mass-file hits: %d, csv hits: %d)",
                len(df), n_mass_hit, n_csv_hit)
    return df


def load_star_properties(mass_file, rv_dir=None):
    """
    Load star properties from mass_bloem.csv.

    Returns DataFrame with columns: ID, Mspec, R_star, field, rv_err, gamma.
    """
    massdf = pd.read_csv(mass_file)

    # If rv_dir is provided, load per-star RV errors and gamma from CSVs
    if rv_dir and os.path.isdir(rv_dir):
        import glob
        rv_files = glob.glob(os.path.join(rv_dir, "*_CCF_RVs.csv"))
        rv_info = {}
        rv_files_by_key = {}  # remember source path for each key
        for f in rv_files:
            base = os.path.basename(f).replace("_CCF_RVs.csv", "")
            # Normalize: strip BLOeM_ prefix so the key matches the
            # mass-file ID after its own BLOeM_ strip below.
            star = base.replace("BLOeM_", "")
            try:
                df = pd.read_csv(f)
                rv_info[star] = {
                    "rv_err": df["Mean RVsig"].median(),
                    "gamma": df["Mean RV"].median(),
                    "field": None,  # will be assigned by MJD matching
                }
                rv_files_by_key[star] = f
            except Exception:
                continue

        # Match stars to fields by closest MJD array
        for star, info in rv_info.items():
            try:
                df = pd.read_csv(rv_files_by_key[star])
                star_mjds = df["MJD"].values
                best_field = 0
                best_overlap = 0
                for fi, fld_mjds in enumerate(BLOEM_MJD_ARRAYS):
                    overlap = sum(1 for m in star_mjds
                                 if any(abs(m - fm) < 0.5
                                        for fm in fld_mjds))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_field = fi
                info["field"] = best_field
            except Exception:
                info["field"] = 0

        # Merge
        rows = []
        for _, mrow in massdf.iterrows():
            sid = mrow["ID"]
            # Try matching by BLOeM ID (e.g. "1-023")
            star_key = sid.replace("BLOeM_", "") if "BLOeM_" in sid else sid
            if star_key in rv_info:
                M1_val = mrow["Mspec"]
                R1_val = mrow["R_star"]
                if M1_val < 5 or M1_val > 120 or R1_val < 2 or R1_val > 30:
                    logger.debug("load_star_properties: skipping %s "
                                 "(M1=%.1f R1=%.1f outside O-star range)",
                                 star_key, M1_val, R1_val)
                    continue
                ri = rv_info[star_key]
                rows.append({
                    "ID": star_key,
                    "Mspec": M1_val,
                    "R_star": R1_val,
                    "field": ri["field"],
                    "rv_err": ri["rv_err"],
                    "gamma": ri["gamma"],
                })

        if not rows:
            raise RuntimeError(
                "No stars matched between %s and CSV files in %s. "
                "Check that rv_dir actually contains *_CCF_RVs.csv files "
                "(e.g. /Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/"
                "CCF/dr5_neb_div_from_coadded/)." % (mass_file, rv_dir))
        result = pd.DataFrame(rows)
        logger.info("load_star_properties: %d stars loaded (with RV info)",
                     len(result))
        return result

    raise RuntimeError(
        "rv_dir is required so per-star MJDs/rv_err/gamma are loaded "
        "from real CCF outputs. Pass --rv-dir <path>.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bias correction grid search (Sana+2012 style).",
    )
    parser.add_argument(
        "--config", default="params_bias.yaml",
        help="Path to pipeline config YAML. Default: params_bias.yaml",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for results and plots. If omitted, "
             "defaults to <base_dir>/bias_grid_results/<preset>/.",
    )
    parser.add_argument(
        "--preset", choices=sorted(GRID_PRESETS.keys()), default="D",
        help="Named grid preset from bias_config.GRID_PRESETS. "
             "Default: D (minimum scientifically defensible run).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Deprecated alias for --preset quick.",
    )
    parser.add_argument(
        "--n-inject", type=int, default=None,
        help="Override n_inject_per_star for the selected preset.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--sb1-tex", default=None,
        help="Path to sb1_solutions.tex.",
    )
    parser.add_argument(
        "--sb2-tex", default=None,
        help="Path to sb2_solutions.tex.",
    )
    parser.add_argument(
        "--mass-file", default=None,
        help="Path to mass_bloem.csv.",
    )
    parser.add_argument(
        "--rv-dir", default=None,
        help="Directory with *_CCF_RVs.csv files for per-star RV errors.",
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="Parallel workers for per-star inner loop. "
             "Default: cpu_count - 2.",
    )
    # --- SLURM array job support ---
    parser.add_argument(
        "--grid-start", type=int, default=None,
        help="First grid-point step index (inclusive) for this SLURM task.",
    )
    parser.add_argument(
        "--grid-end", type=int, default=None,
        help="Last grid-point step index (exclusive) for this SLURM task.",
    )
    parser.add_argument(
        "--aggregate", type=str, default=None,
        help="Path to directory containing task_*/ subdirs to aggregate. "
             "Skips computation; merges partial results and plots.",
    )
    parser.add_argument(
        "--detect-method", choices=sorted(DETECTION_METHODS.keys()),
        default="pipeline",
        help="Detection function to use. 'pipeline' = full periodogram + "
             "lmfit + BICc (slow). 'rv_threshold' = pairwise delta-RV "
             "significance (fast). Default: pipeline.",
    )
    parser.add_argument(
        "--parallel-grid", action="store_true",
        help="Parallelize over grid points (one per CPU) instead of "
             "over stars within each grid point. Useful when each grid "
             "point is fast (e.g. rv_threshold detection).",
    )
    cli = parser.parse_args()

    # --quick is a deprecated alias for --preset quick
    if cli.quick:
        cli.preset = "quick"

    # --- Aggregate mode: merge partial SLURM task results and exit ---
    if cli.aggregate:
        agg_dir = cli.aggregate
        setup_logging(agg_dir)
        logger.info("Aggregating SLURM task results from %s", agg_dir)
        aggregate_tasks(agg_dir)
        return

    # Load pipeline config
    args_dict = load_args(cli.config)

    # Load bias config
    cfg = copy.deepcopy(DEFAULT_BIAS_CFG)

    # Paths
    sb1_tex = cli.sb1_tex or cfg["sb1_tex"]
    sb2_tex = cli.sb2_tex or cfg["sb2_tex"]
    mass_file = cli.mass_file or cfg["mass_file"]
    output_dir = cli.output_dir or os.path.join(
        args_dict.get("base_dir", "."), "bias_grid_results", cli.preset)

    # SLURM task mode: per-task subdirectory
    is_task_mode = cli.grid_start is not None
    if is_task_mode:
        grid_end = cli.grid_end if cli.grid_end is not None else 0
        output_dir = os.path.join(
            output_dir, "task_%d_%d" % (cli.grid_start, grid_end))

    # Setup logging (before any work)
    setup_logging(output_dir)

    # 1) Load observed distributions
    logger.info("Loading observed distributions from LaTeX tables...")
    obs = load_observed_from_tex(sb1_tex, sb2_tex)
    logger.info("  SB1: %d, SB2: %d, Total: %d",
                obs['n_sb1'], obs['n_sb2'], len(obs['logP']))
    cfg["n_det_obs"] = len(obs["logP"])

    # 2) Load star properties — use the full O-star catalog (134 stars)
    # so the injection sample represents the entire population, not just
    # the detected binaries.
    logger.info("Loading full star sample from mass catalog...")
    star_df = load_star_properties(mass_file, cli.rv_dir)
    logger.info("  %d stars in injection sample", len(star_df))

    field_arr = star_df["field"].values
    rv_err_arr = star_df["rv_err"].values
    gamma_arr = star_df["gamma"].values
    M1_arr = star_df["Mspec"].values
    R1_arr = star_df["R_star"].values
    star_ids = star_df["ID"].values

    # Build field_mjds dict
    field_mjds = {}
    for fi, mjds in enumerate(BLOEM_MJD_ARRAYS):
        field_mjds[fi] = np.array(mjds)

    # n_stars_sample for the binomial must be the full survey population
    # (134), NOT len(star_df) which may be smaller due to missing CSVs.
    # The default from bias_config.py (134) is correct; don't override it.
    logger.info("  n_stars_sample for binomial: %d (injection sample: %d)",
                cfg["n_stars_sample"], len(star_df))

    # 3) Setup grids from the selected preset
    preset = GRID_PRESETS[cli.preset]
    pi_grid = np.asarray(preset["pi"])
    kappa_grid = np.asarray(preset["kappa"])
    eta_grid = np.asarray(preset["eta"])
    fbin_grid = np.asarray(preset["fbin"])
    n_inject = cli.n_inject or preset["n_inject_per_star"]

    total = len(pi_grid) * len(kappa_grid) * len(eta_grid) * len(fbin_grid)

    logger.info("=" * 60)
    logger.info("  Bias Correction Grid Search")
    logger.info("  Preset: %s", cli.preset)
    logger.info("  Grid: %d×%d×%d×%d = %d points",
                len(pi_grid), len(kappa_grid), len(eta_grid),
                len(fbin_grid), total)
    logger.info("  Injections per star: %d", n_inject)
    logger.info("  Stars: %d", len(star_df))
    logger.info("  Observed detections: %d", len(obs['logP']))
    logger.info("  Output: %s", output_dir)
    logger.info("  Detection method: %s", cli.detect_method)
    logger.info("=" * 60)

    # 3b) Dump the resolved run config to YAML for reproducibility.
    # Includes the chosen preset (resolved n_inject + grid arrays) and the
    # full bias_cfg as actually used (defaults + any CLI overrides).
    os.makedirs(output_dir, exist_ok=True)
    run_cfg_yaml = {
        "preset_name": cli.preset,
        "preset": {
            "n_inject_per_star": int(n_inject),
            "pi": pi_grid.tolist(),
            "kappa": kappa_grid.tolist(),
            "eta": eta_grid.tolist(),
            "fbin": fbin_grid.tolist(),
        },
        "bias_cfg": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in cfg.items()
        },
        "cli_overrides": {
            "config": cli.config,
            "seed": cli.seed,
            "sb1_tex": sb1_tex,
            "sb2_tex": sb2_tex,
            "mass_file": mass_file,
            "rv_dir": cli.rv_dir,
            "detect_method": cli.detect_method,
            "parallel_grid": bool(cli.parallel_grid),
            "n_workers": cli.n_workers,
            "n_inject": cli.n_inject,
            "grid_start": cli.grid_start,
            "grid_end": cli.grid_end,
        },
    }
    run_cfg_path = os.path.join(output_dir, "run_config.yaml")
    with open(run_cfg_path, "w") as fh:
        yaml.safe_dump(run_cfg_yaml, fh, sort_keys=False, default_flow_style=False)
    logger.info("Saved run config to %s", run_cfg_path)

    # 4) Run grid search
    engine = GridSearchEngine(
        field_mjds=field_mjds,
        star_ids=star_ids,
        M1_arr=M1_arr,
        R1_arr=R1_arr,
        field_arr=field_arr,
        rv_err_arr=rv_err_arr,
        gamma_arr=gamma_arr,
        args_dict=args_dict,
        cfg=cfg,
    )
    engine.detect_method = cli.detect_method

    if cli.parallel_grid:
        # In parallel-grid mode, default to cpu_count - 2 workers
        # (each worker handles one grid point).
        n_workers = cli.n_workers or max(1, os.cpu_count() - 2)
    elif cli.detect_method != "pipeline" and cli.n_workers is None:
        n_workers = 1
    else:
        n_workers = cli.n_workers or max(1, os.cpu_count() - 2)
    logger.info("  Workers: %d (parallel_grid=%s)",
                n_workers, cli.parallel_grid)

    grid_start = cli.grid_start or 0
    grid_end_val = cli.grid_end  # None means all

    if is_task_mode:
        logger.info("  SLURM task mode: steps [%d, %d)", grid_start,
                     grid_end_val if grid_end_val is not None else total)

    results = engine.run(
        pi_grid, kappa_grid, eta_grid, fbin_grid,
        obs["logP"], obs["e"], obs["K1"],
        n_inject_per_star=n_inject,
        seed=cli.seed,
        checkpoint_dir=output_dir,
        preset_name=cli.preset,
        n_workers=n_workers,
        grid_start=grid_start,
        grid_end=grid_end_val,
        parallel_grid=cli.parallel_grid,
    )

    if is_task_mode:
        # In SLURM task mode, checkpoint files are the output.
        # Aggregation is done separately via --aggregate.
        logger.info("Task complete. Results in %s", output_dir)
        return

    # 5) Save results (single-machine mode only)
    os.makedirs(output_dir, exist_ok=True)

    # Save cubes
    save_kw_cubes = dict(
        pdet_cube=results["pdet_cube"],
        pi_grid=pi_grid,
        kappa_grid=kappa_grid,
        eta_grid=eta_grid,
        fbin_grid=fbin_grid,
    )
    # All tests
    if "gmf_cubes" in results:
        for tname in _ALL_TESTS:
            save_kw_cubes["gmf_%s_cube" % tname] = results["gmf_cubes"][tname]
            for par in ("logP", "e", "K1"):
                save_kw_cubes["%s_%s_cube" % (tname, par)] = \
                    results["test_cubes"][tname][par]
            if "e_circ" in results["test_cubes"][tname]:
                save_kw_cubes["%s_e_circ_cube" % tname] = \
                    results["test_cubes"][tname]["e_circ"]
    np.savez(os.path.join(output_dir, "grid_cubes.npz"), **save_kw_cubes)
    logger.info("Saved grid cubes to %s/grid_cubes.npz", output_dir)

    # Save global histograms (detected arrays are already on disk as
    # individual shard files in det_shards/).
    if "global_hists" in results:
        save_kw = {}
        for k, v in _HIST_BINS.items():
            save_kw["bins_%s" % k] = v
        for (a, b) in _HIST_PAIRS:
            save_kw["hist_total_%s_%s" % (a, b)] = \
                results["global_hists"]["total"][(a, b)]
            save_kw["hist_det_%s_%s" % (a, b)] = \
                results["global_hists"]["det"][(a, b)]
        np.savez(os.path.join(output_dir, "grid_hists.npz"), **save_kw)
        logger.info("Saved global histograms to %s/grid_hists.npz",
                     output_dir)
    if results.get("step_to_ijkl"):
        _save_det_index(output_dir, results["step_to_ijkl"])
        logger.info("Saved det index + %d shard files in %s/det_shards/",
                     len(results["step_to_ijkl"]), output_dir)

    # Plots
    plot_grid_results(results, output_dir,
                      obs_logP=obs["logP"], obs_e=obs["e"],
                      obs_K1=obs["K1"])

    logger.info("Done.")


if __name__ == "__main__":
    main()
