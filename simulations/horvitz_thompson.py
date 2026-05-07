"""
simulations.horvitz_thompson — Horvitz-Thompson estimator for the intrinsic
binary fraction, using targeted injection-recovery to estimate per-binary
detection probabilities.

For each detected binary with observed (P, e, K1, gamma):
  1. Fix (P, e, K1) from the orbital solution
  2. Randomize nuisance parameters: omega, T0, noise realization
  3. Inject into the star's actual MJD array with realistic RV errors
  4. Run detect_from_arrays → detected or not
  5. p_det_i = n_detected / n_trials

Then:  f_bin = (1/N_stars) × Σ (1/p_det_i)

Usage:
    python -m simulations.horvitz_thompson --config params_bias.yaml
    python -m simulations.horvitz_thompson --config params_bias.yaml --n-trials 10 --subset 5
"""

import argparse
import copy
import logging
import os
import re
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from simulations.bias_grid import rv_model_jit, TWOPI
from simulations.bias_config import DEFAULT_BIAS_CFG
from simulations.common import BLOEM_MJD_ARRAYS
from pipeline.config import load_args

logger = logging.getLogger("horvitz_thompson")


# ---------------------------------------------------------------------------
# Parse observed binaries from LaTeX tables (extended from bias_grid)
# ---------------------------------------------------------------------------

def _parse_val_with_errors(s):
    """Parse '1.23^{+0.01}_{-0.01}' → 1.23, or handle \\leq / \\geq."""
    s = s.strip().replace("$", "")
    if r"\dots" in s or s == r"\dots":
        return np.nan
    if r"\leq" in s:
        val = re.sub(r"\\leq\s*", "", s)
        return float(val)
    if r"\geq" in s:
        val = re.sub(r"\\geq\s*", "", s)
        return -float(val)  # negative flags exclusion
    m = re.match(r"([0-9.eE+-]+)", s)
    if m:
        return float(m.group(1))
    return np.nan


def load_observed_binaries(sb1_path, sb2_path):
    """Parse SB1 + SB2 tables, returning a DataFrame with per-binary info.

    Returns DataFrame with columns: star_id, P, e, K1, gamma, source (SB1/SB2).
    """
    rows = []

    # --- SB1 ---
    with open(sb1_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("\\"):
                continue
            if "&" not in line:
                continue
            cols = [c.strip() for c in line.replace("\\\\", "").split("&")]
            if len(cols) < 8:
                continue
            try:
                int(cols[0])
            except ValueError:
                continue

            star_id = cols[1].strip()
            if "(outer)" in star_id:
                continue

            P_val = _parse_val_with_errors(cols[2])
            e_val = _parse_val_with_errors(cols[5])
            K1_val = _parse_val_with_errors(cols[6])
            gamma_val = _parse_val_with_errors(cols[7])

            if P_val < 0 or np.isnan(P_val):
                continue
            if np.isnan(K1_val):
                continue

            # Clean star_id: remove annotations like "(inner)"
            star_id = re.sub(r"\s*\(.*?\)", "", star_id).strip()

            rows.append({
                "star_id": star_id,
                "P": P_val,
                "e": e_val if not np.isnan(e_val) else 0.0,
                "K1": K1_val,
                "gamma": gamma_val if not np.isnan(gamma_val) else 168.0,
                "source": "SB1",
            })

    n_sb1 = len(rows)

    # --- SB2 ---
    with open(sb2_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("\\"):
                continue
            if "&" not in line:
                continue
            cols = [c.strip() for c in line.replace("\\\\", "").split("&")]
            if len(cols) < 10:
                continue
            try:
                int(cols[0])
            except ValueError:
                continue

            star_id = cols[1].strip()
            P_val = _parse_val_with_errors(cols[2])
            e_val = _parse_val_with_errors(cols[5])
            K1_val = _parse_val_with_errors(cols[8])
            gamma_val = _parse_val_with_errors(cols[7]) if len(cols) > 7 else np.nan

            if np.isnan(P_val) or P_val < 0:
                continue
            if np.isnan(K1_val):
                continue

            star_id = re.sub(r"\s*\(.*?\)", "", star_id).strip()

            rows.append({
                "star_id": star_id,
                "P": P_val,
                "e": e_val if not np.isnan(e_val) else 0.0,
                "K1": K1_val,
                "gamma": gamma_val if not np.isnan(gamma_val) else 168.0,
                "source": "SB2",
            })

    df = pd.DataFrame(rows)
    logger.info("Loaded %d observed binaries (SB1=%d, SB2=%d)",
                len(df), n_sb1, len(df) - n_sb1)
    return df


# ---------------------------------------------------------------------------
# Match stars to MJD arrays and RV errors
# ---------------------------------------------------------------------------

def match_star_to_field(star_id, rv_dir):
    """Load a star's RV CSV and match to the best-fitting BLOeM MJD array.

    Returns (MJDs, rv_err_median, field_idx) or None if CSV not found.
    """
    csv_name = f"BLOeM_{star_id}_CCF_RVs.csv"
    csv_path = os.path.join(rv_dir, csv_name)

    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    mjds = df["MJD"].values
    rv_err = df["Mean RVsig"].median()

    best_field = 0
    best_overlap = 0
    for fi, fld_mjds in enumerate(BLOEM_MJD_ARRAYS):
        fld_arr = np.array(fld_mjds)
        overlap = sum(1 for m in mjds if np.min(np.abs(m - fld_arr)) < 0.5)
        if overlap > best_overlap:
            best_overlap = overlap
            best_field = fi

    return np.array(BLOEM_MJD_ARRAYS[best_field]), rv_err, best_field


# ---------------------------------------------------------------------------
# Targeted injection-recovery for one binary
# ---------------------------------------------------------------------------

def estimate_pdet_single(P, e, K1, gamma, MJDs, rv_err, args_dict,
                         n_trials=100, seed=42):
    """Estimate detection probability for one binary via injection-recovery.

    Fixes (P, e, K1, gamma) and randomizes (omega, T0, noise) over n_trials.
    Returns (p_det, n_detected, n_trials).
    """
    from pipeline.evaluator import detect_from_arrays

    rng = np.random.default_rng(seed)
    MJDs = np.asarray(MJDs, dtype=np.float64)
    n_det = 0

    for trial in range(n_trials):
        omega = rng.uniform(0, TWOPI)
        T0 = float(rng.uniform(MJDs.min() - P, MJDs.min()))
        rv_errs = np.full(len(MJDs), rv_err)

        rv_true = rv_model_jit(MJDs, P, T0, omega, e, K1, gamma)
        noise = rng.normal(0.0, rv_errs)
        rv_obs = rv_true + noise

        try:
            detected, _ = detect_from_arrays(MJDs, rv_obs, rv_errs, args_dict,
                                             use_fwhm=True)
        except Exception:
            detected = False

        if detected:
            n_det += 1

    # Floor at 1/n_trials to avoid infinite weights
    p_det = max(n_det / n_trials, 1.0 / n_trials)
    return p_det, n_det, n_trials


def _worker_pdet(binary_row, MJDs, rv_err, args_dict, n_trials, base_seed):
    """Multiprocessing worker for estimate_pdet_single."""
    idx = binary_row["_idx"]
    seed = base_seed + idx * 137

    p_det, n_det, n_t = estimate_pdet_single(
        binary_row["P"], binary_row["e"], binary_row["K1"],
        binary_row["gamma"], MJDs, rv_err, args_dict,
        n_trials=n_trials, seed=seed,
    )
    return {
        "star_id": binary_row["star_id"],
        "P": binary_row["P"],
        "e": binary_row["e"],
        "K1": binary_row["K1"],
        "gamma": binary_row["gamma"],
        "source": binary_row["source"],
        "p_det": p_det,
        "n_det": n_det,
        "n_trials": n_t,
        "weight": 1.0 / p_det,
    }


# ---------------------------------------------------------------------------
# Horvitz-Thompson estimator
# ---------------------------------------------------------------------------

def horvitz_thompson_fbin(p_det_arr, N_stars, n_bootstrap=10000, seed=42):
    """Compute intrinsic binary fraction via Horvitz-Thompson.

    Parameters
    ----------
    p_det_arr : array-like
        Detection probabilities for each detected binary.
    N_stars : int
        Total number of stars in the sample.
    n_bootstrap : int
        Number of bootstrap resamples for uncertainty.

    Returns
    -------
    dict with f_bin, sigma, CI_68, CI_95.
    """
    p_det = np.asarray(p_det_arr, dtype=float)
    weights = 1.0 / p_det
    f_bin = np.sum(weights) / N_stars

    # Bootstrap uncertainty
    rng = np.random.default_rng(seed)
    N_det = len(p_det)
    boot_fbins = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(N_det, size=N_det, replace=True)
        boot_fbins[b] = np.sum(1.0 / p_det[idx]) / N_stars

    sigma = np.std(boot_fbins)
    ci_68 = np.percentile(boot_fbins, [16, 84])
    ci_95 = np.percentile(boot_fbins, [2.5, 97.5])

    return {
        "f_bin": f_bin,
        "f_obs": N_det / N_stars,
        "N_det": N_det,
        "N_stars": N_stars,
        "sigma": sigma,
        "CI_68": ci_68,
        "CI_95": ci_95,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Horvitz-Thompson intrinsic binary fraction estimator.")
    parser.add_argument("--config", default="params_bias.yaml",
                        help="Pipeline config YAML.")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Injection-recovery trials per binary (default: 100).")
    parser.add_argument("--n-stars", type=int, default=None,
                        help="Total sample size (default: from bias config).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for results.")
    parser.add_argument("--sb1-tex", default=None)
    parser.add_argument("--sb2-tex", default=None)
    parser.add_argument("--rv-dir", default=None,
                        help="Directory with BLOeM_*_CCF_RVs.csv files.")
    parser.add_argument("--subset", type=int, default=None,
                        help="Process only the first N binaries (for testing).")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Number of parallel workers (default: 1).")
    args = parser.parse_args()

    # Load pipeline config
    args_dict = load_args(args.config)
    cfg = copy.deepcopy(DEFAULT_BIAS_CFG)

    sb1_tex = args.sb1_tex or cfg["sb1_tex"]
    sb2_tex = args.sb2_tex or cfg["sb2_tex"]
    N_stars = args.n_stars or cfg["n_stars_sample"]
    output_dir = args.output_dir or os.path.join(
        args_dict.get("base_dir", "."), "horvitz_thompson_results")
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "ht.log"), mode="a"),
            logging.StreamHandler(),
        ],
    )

    # RV data directory (parent of per-star output dirs)
    rv_dir = args.rv_dir or os.path.dirname(
        args_dict.get("base_dir", ".").rstrip("/"))

    # 1) Load observed binaries
    logger.info("Loading observed binaries from LaTeX tables...")
    obs_df = load_observed_binaries(sb1_tex, sb2_tex)
    logger.info("  %d binaries loaded", len(obs_df))

    if args.subset:
        obs_df = obs_df.head(args.subset)
        logger.info("  (subset: using first %d)", len(obs_df))

    # 2) Match each binary to its MJD array and RV errors
    logger.info("Matching stars to MJD arrays (rv_dir=%s)...", rv_dir)
    matched = []
    for _, row in obs_df.iterrows():
        result = match_star_to_field(row["star_id"], rv_dir)
        if result is None:
            logger.warning("  %s: RV CSV not found, using default field 0",
                           row["star_id"])
            MJDs = np.array(BLOEM_MJD_ARRAYS[0])
            rv_err = 2.0  # default
        else:
            MJDs, rv_err, field_idx = result
        matched.append({"MJDs": MJDs, "rv_err": rv_err})

    # 3) Run targeted injection-recovery
    logger.info("=" * 60)
    logger.info("  Horvitz-Thompson: Targeted Injection-Recovery")
    logger.info("  Binaries: %d | Trials/binary: %d | Total runs: %d",
                len(obs_df), args.n_trials, len(obs_df) * args.n_trials)
    logger.info("  N_stars (sample): %d", N_stars)
    logger.info("=" * 60)

    results = []
    t0 = time.time()

    for i, (_, row) in enumerate(obs_df.iterrows()):
        star_id = row["star_id"]
        m = matched[i]

        logger.info("[%d/%d] %s: P=%.2f e=%.3f K1=%.1f ...",
                    i + 1, len(obs_df), star_id, row["P"], row["e"], row["K1"])

        p_det, n_det, n_t = estimate_pdet_single(
            row["P"], row["e"], row["K1"], row["gamma"],
            m["MJDs"], m["rv_err"], args_dict,
            n_trials=args.n_trials, seed=args.seed + i * 137,
        )

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(obs_df) - i - 1)

        logger.info("  → p_det=%.3f (%d/%d) | elapsed=%.0fs ETA=%.0fs",
                    p_det, n_det, n_t, elapsed, eta)

        results.append({
            "star_id": star_id,
            "P": row["P"],
            "e": row["e"],
            "K1": row["K1"],
            "gamma": row["gamma"],
            "source": row["source"],
            "rv_err": m["rv_err"],
            "p_det": p_det,
            "n_det": n_det,
            "n_trials": n_t,
            "weight": 1.0 / p_det,
        })

    results_df = pd.DataFrame(results)

    # 4) Compute HT estimate
    ht = horvitz_thompson_fbin(results_df["p_det"].values, N_stars,
                               seed=args.seed)

    # 5) Output
    csv_path = os.path.join(output_dir, "horvitz_thompson_results.csv")
    results_df.to_csv(csv_path, index=False)

    summary = (
        f"{'=' * 60}\n"
        f"  Horvitz-Thompson Binary Fraction Estimate\n"
        f"{'=' * 60}\n"
        f"  N_stars (sample):       {ht['N_stars']}\n"
        f"  N_det (detected):       {ht['N_det']}\n"
        f"  f_obs (observed):       {ht['f_obs']:.3f}\n"
        f"  f_bin (HT corrected):   {ht['f_bin']:.3f} ± {ht['sigma']:.3f}\n"
        f"  68% CI:                 [{ht['CI_68'][0]:.3f}, {ht['CI_68'][1]:.3f}]\n"
        f"  95% CI:                 [{ht['CI_95'][0]:.3f}, {ht['CI_95'][1]:.3f}]\n"
        f"  Correction factor:      {ht['f_bin'] / ht['f_obs']:.2f}×\n"
        f"{'=' * 60}\n"
        f"  Per-binary p_det range: [{results_df['p_det'].min():.3f}, "
        f"{results_df['p_det'].max():.3f}]\n"
        f"  Median p_det:           {results_df['p_det'].median():.3f}\n"
        f"  Trials per binary:      {args.n_trials}\n"
        f"  Total pipeline runs:    {len(results_df) * args.n_trials}\n"
        f"  Elapsed time:           {time.time() - t0:.0f}s\n"
        f"{'=' * 60}\n"
    )

    print(summary)
    logger.info(summary)

    summary_path = os.path.join(output_dir, "horvitz_thompson_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    logger.info("Results: %s", csv_path)
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
