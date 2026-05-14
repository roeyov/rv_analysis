"""
simulations.rescore_subsampled — Rescore an existing bias grid run using
subsampled two-sample tests to eliminate sample-size asymmetry bias.

Problem: The original scoring compares ~10 000 simulated detections against
N_obs=71 observed systems. KS/AD/CvM tests gain extreme statistical power at
these unequal sample sizes, rejecting even visually-good fits (e.g. e p<0.005
despite good CDF overlap).

Fix: For each grid point, randomly draw N_obs systems from the simulated
detections (without replacement), compute the test statistic, repeat M times,
and take the median log-p-value. This calibrates the test to the same effective
sample size as the observations.

This approach is standard in massive-star binary surveys:
  - Banyard et al. 2022 (NGC 6231, A&A 658, A69): drew 10 000 artificial
    populations of N_obs binaries, averaged detection fractions.
  - Sana et al. 2012 (Science 337, 444): synthetic populations match the
    observational sample size and cadence.

Usage:
    python -m simulations.rescore_subsampled \\
        --input-dir /path/to/final2_rv \\
        --output-dir /path/to/final2_rv_subsampled \\
        --n-draws 200 --seed 42 --n-workers 8
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
from scipy.stats import binom
from multiprocessing import Pool

logger = logging.getLogger("rescore_subsampled")

# Ensure project root is importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from simulations.bias_grid import (
    load_observed_from_tex, _load_det_shard,
    _ALL_TESTS, _safe_pvalue, _clip_to_range,
)
from simulations.bias_config import DEFAULT_BIAS_CFG


# ---------------------------------------------------------------------------
# Core subsampling function
# ---------------------------------------------------------------------------

def _subsample_pvalues(test_fn, obs, sim_clipped, n_obs, n_draws, rng,
                       min_samples=5):
    """Draw n_obs from sim_clipped, compute p-value, repeat n_draws times.

    Returns the median of log(p) across draws — robust to both tails.
    Falls back to full comparison if n_sim < n_obs.
    """
    n_sim = len(sim_clipped)
    if n_sim < min_samples:
        return -np.inf
    if n_sim < n_obs:
        # Fallback: full comparison (test has low power anyway)
        p = _safe_pvalue(test_fn, obs, sim_clipped, min_samples=min_samples)
        return np.log(p) if p > 0 else -np.inf

    log_ps = np.empty(n_draws)
    for m in range(n_draws):
        idx = rng.choice(n_sim, size=n_obs, replace=False)
        sub = sim_clipped[idx]
        p = _safe_pvalue(test_fn, obs, sub, min_samples=min_samples)
        log_ps[m] = np.log(p) if p > 0 else -np.inf

    return np.median(log_ps)


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

def _process_one_point(args):
    """Score a single grid point with subsampled tests."""
    (step, i, j, k, l, input_dir, obs_logP, obs_e, obs_K1,
     clip_range, p_det, N_det_obs, N_stars,
     n_draws, child_seed) = args

    shard = _load_det_shard(input_dir, step)
    if shard is None or len(shard["logP"]) < 5:
        return (i, j, k, l, None)

    # Clip to observed range
    sim_clipped = {
        "logP": _clip_to_range(shard["logP"], *clip_range["logP"]),
        "e": _clip_to_range(shard["e"], *clip_range["e"]),
        "K1": _clip_to_range(shard["K1"], *clip_range["K1"]),
    }

    # Binomial term (shared across tests, not resampled). p_det already
    # absorbs f_bin via the per-realization coin flip in the simulator,
    # so it is the binomial success probability directly.
    p_binom = binom.pmf(N_det_obs, N_stars, p_det) \
        if p_det > 0 else 0.0

    rng = np.random.default_rng(child_seed)

    result = {}
    for tname, tfn in _ALL_TESTS.items():
        min_K1 = 3  # match existing convention
        log_p_logP = _subsample_pvalues(
            tfn, obs_logP, sim_clipped["logP"],
            N_det_obs, n_draws, rng)
        log_p_e = _subsample_pvalues(
            tfn, obs_e, sim_clipped["e"],
            N_det_obs, n_draws, rng)
        log_p_K1 = _subsample_pvalues(
            tfn, obs_K1, sim_clipped["K1"],
            N_det_obs, n_draws, rng, min_samples=min_K1)

        # GMF: sum of median-log-p across parameters + binomial
        log_gmf = log_p_logP + log_p_e + log_p_K1
        if p_binom > 0:
            log_gmf += np.log(p_binom)
        else:
            log_gmf = -np.inf

        result[tname] = {
            "p_logP": np.exp(log_p_logP) if np.isfinite(log_p_logP) else 0.0,
            "p_e": np.exp(log_p_e) if np.isfinite(log_p_e) else 0.0,
            "p_K1": np.exp(log_p_K1) if np.isfinite(log_p_K1) else 0.0,
            "log_gmf": log_gmf,
        }

    return (i, j, k, l, result)


# ---------------------------------------------------------------------------
# Main rescoring engine
# ---------------------------------------------------------------------------

def rescore_subsampled(input_dir, output_dir, n_draws=200, seed=42,
                       n_workers=1, sb1_tex=None, sb2_tex=None):
    """Rescore all grid points using subsampled two-sample tests."""

    # Load original cubes for grid axes and metadata
    cubes_path = os.path.join(input_dir, "grid_cubes.npz")
    if not os.path.exists(cubes_path):
        cubes_path = os.path.join(input_dir, "checkpoint_cubes.npz")
    cubes = np.load(cubes_path, allow_pickle=True)

    pi_grid = cubes["pi_grid"]
    kappa_grid = cubes["kappa_grid"]
    eta_grid = cubes["eta_grid"]
    fbin_grid = cubes["fbin_grid"]
    pdet_cube = cubes["pdet_cube"]

    shape = (len(pi_grid), len(kappa_grid), len(eta_grid), len(fbin_grid))

    # Load shard index
    idx_path = os.path.join(input_dir, "det_index.npz")
    step_to_ijkl = np.load(idx_path)["step_to_ijkl"]
    n_steps = len(step_to_ijkl)

    # Load observed distributions
    sb1_tex = sb1_tex or DEFAULT_BIAS_CFG["sb1_tex"]
    sb2_tex = sb2_tex or DEFAULT_BIAS_CFG["sb2_tex"]
    obs = load_observed_from_tex(sb1_tex, sb2_tex)
    obs_logP = obs["logP"]
    obs_e = obs["e"]
    obs_K1 = obs["K1"]
    N_det_obs = len(obs_logP)
    N_stars = DEFAULT_BIAS_CFG["n_stars_sample"]

    # Clip ranges based on observed extremes
    clip_range = {
        "logP": (obs_logP.min(), obs_logP.max()),
        "e": (obs_e.min(), obs_e.max()),
        "K1": (obs_K1.min(), obs_K1.max()),
    }

    logger.info("Subsampled rescoring: %d grid points, %d draws/point, "
                "seed=%d, workers=%d",
                n_steps, n_draws, seed, n_workers)
    logger.info("Observed: %d systems (SB1=%d, SB2=%d)",
                N_det_obs, obs["n_sb1"], obs["n_sb2"])
    logger.info("Clip ranges: logP=[%.2f,%.2f] e=[%.3f,%.3f] K1=[%.1f,%.1f]",
                *clip_range["logP"], *clip_range["e"], *clip_range["K1"])

    # Allocate cubes
    gmf_cubes = {}
    test_cubes = {}
    for tname in _ALL_TESTS:
        gmf_cubes[tname] = np.full(shape, -np.inf)
        test_cubes[tname] = {
            "logP": np.zeros(shape),
            "e": np.zeros(shape),
            "K1": np.zeros(shape),
        }

    # Pre-generate deterministic child seeds for each grid point
    master_rng = np.random.default_rng(seed)
    child_seeds = master_rng.integers(0, 2**32, size=n_steps)

    # Build task list
    tasks = []
    for idx_row, row in enumerate(step_to_ijkl):
        step = int(row[0])
        i, j, k, l = int(row[1]), int(row[2]), int(row[3]), int(row[4])
        p_det = float(pdet_cube[i, j, k, l])
        tasks.append((
            step, i, j, k, l, input_dir,
            obs_logP, obs_e, obs_K1,
            clip_range, p_det, N_det_obs, N_stars,
            n_draws, int(child_seeds[idx_row]),
        ))

    # Histograms path (for symlinking)
    hists_path = os.path.join(input_dir, "grid_hists.npz")
    if not os.path.exists(hists_path):
        hists_path = os.path.join(input_dir, "checkpoint_hists.npz")

    t0 = time.time()
    n_done = 0
    n_empty = 0

    def _accumulate(result):
        nonlocal n_done, n_empty
        i, j, k, l, data = result
        n_done += 1
        if data is None:
            n_empty += 1
            return
        for tname in _ALL_TESTS:
            test_cubes[tname]["logP"][i, j, k, l] = data[tname]["p_logP"]
            test_cubes[tname]["e"][i, j, k, l] = data[tname]["p_e"]
            test_cubes[tname]["K1"][i, j, k, l] = data[tname]["p_K1"]
            gmf_cubes[tname][i, j, k, l] = data[tname]["log_gmf"]

    if n_workers > 1:
        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(_process_one_point, tasks,
                                              chunksize=100):
                _accumulate(result)
                if n_done % 10000 == 0:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    eta_s = (n_steps - n_done) / rate if rate > 0 else 0
                    logger.info("[%d/%d] %.1f pts/s, ETA %.0fs",
                                n_done, n_steps, rate, eta_s)
    else:
        for task in tasks:
            result = _process_one_point(task)
            _accumulate(result)
            if n_done % 10000 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta_s = (n_steps - n_done) / rate if rate > 0 else 0
                logger.info("[%d/%d] %.1f pts/s, ETA %.0fs",
                            n_done, n_steps, rate, eta_s)

    elapsed = time.time() - t0
    logger.info("Done: %d points in %.1fs (%.1f pts/s, %d empty)",
                n_done, elapsed, n_done / max(elapsed, 1), n_empty)

    # Best fits per test
    best_fits = {}
    for tname in _ALL_TESTS:
        gc = gmf_cubes[tname]
        idx = np.unravel_index(np.nanargmax(gc), gc.shape)
        best_fits[tname] = np.array([
            float(pi_grid[idx[0]]),
            float(kappa_grid[idx[1]]),
            float(eta_grid[idx[2]]),
            float(fbin_grid[idx[3]]),
        ])
        logger.info("Best fit (%s): pi=%.3f kappa=%.3f eta=%.3f fbin=%.3f "
                     "(logGMF=%.2f)",
                     tname.upper(), *best_fits[tname], gc[idx])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_kw = dict(
        pdet_cube=pdet_cube,
        pi_grid=pi_grid,
        kappa_grid=kappa_grid,
        eta_grid=eta_grid,
        fbin_grid=fbin_grid,
        # Metadata
        n_draws=np.array(n_draws),
        seed=np.array(seed),
        method=np.array("subsampled"),
        # Backward-compat (KS as default)
        gmf_cube=gmf_cubes["ks"],
        best_fit=best_fits["ks"],
        ks_logP_cube=test_cubes["ks"]["logP"],
        ks_e_cube=test_cubes["ks"]["e"],
        ks_K1_cube=test_cubes["ks"]["K1"],
    )
    # All tests
    for tname in _ALL_TESTS:
        save_kw["gmf_%s_cube" % tname] = gmf_cubes[tname]
        save_kw["best_fit_%s" % tname] = best_fits[tname]
        for par in ("logP", "e", "K1"):
            save_kw["%s_%s_cube" % (tname, par)] = test_cubes[tname][par]

    np.savez(os.path.join(output_dir, "grid_cubes.npz"), **save_kw)

    # Symlink det_shards, det_index, and histograms from original run
    for name in ["det_shards", "det_index.npz"]:
        src = os.path.join(input_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    if os.path.exists(hists_path):
        dst = os.path.join(output_dir, "grid_hists.npz")
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(hists_path), dst)

    logger.info("Saved subsampled rescored cubes to %s", output_dir)
    logger.info("Symlinked det_shards + det_index + hists from %s", input_dir)

    return gmf_cubes, best_fits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rescore a bias grid run with subsampled two-sample tests "
                    "(KS, AD, CvM). Draws N_obs from simulated detections "
                    "M times and takes median p-value.",
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Path to the original bias grid output (with det_shards/).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Default: <input-dir>_subsampled/.",
    )
    parser.add_argument(
        "--n-draws", type=int, default=200,
        help="Number of subsample draws per grid point (default: 200).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel workers (default: 1).",
    )
    parser.add_argument(
        "--sb1-tex", default=None,
        help="Path to sb1_solutions.tex.",
    )
    parser.add_argument(
        "--sb2-tex", default=None,
        help="Path to sb2_solutions.tex.",
    )
    cli = parser.parse_args()

    output_dir = cli.output_dir or (cli.input_dir.rstrip("/") + "_subsampled")

    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, "rescore_subsampled.log"),
                             mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(sh)

    rescore_subsampled(
        cli.input_dir, output_dir,
        n_draws=cli.n_draws, seed=cli.seed, n_workers=cli.n_workers,
        sb1_tex=cli.sb1_tex, sb2_tex=cli.sb2_tex,
    )


if __name__ == "__main__":
    main()
