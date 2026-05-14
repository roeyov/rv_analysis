"""
simulations.rescore_grid — Rescore an existing bias grid run with all three
statistical tests (KS, Anderson-Darling, Cramér-von Mises), applying
observed-range clipping. Does NOT re-run the injection-detection loop.

Reads detected parameter arrays from det_shards/, recomputes the GMF cubes,
and saves results compatible with bias_grid_explorer.

Usage:
    python -m simulations.rescore_grid \
        --input-dir /path/to/fina_rv \
        --output-dir /path/to/fina_rv_rescored
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
from scipy.stats import binom

logger = logging.getLogger("rescore_grid")

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
# Rescoring engine
# ---------------------------------------------------------------------------

def rescore(input_dir, output_dir, sb1_tex=None, sb2_tex=None,
            n_stars_sample=None):
    """Rescore all grid points using all three statistical tests."""

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
    if n_stars_sample is not None:
        N_stars = int(n_stars_sample)
    else:
        # Auto-detect from the original run's saved config; fall back to
        # the global default only if no run_config.yaml is present.
        run_cfg_path = os.path.join(input_dir, "run_config.yaml")
        if os.path.exists(run_cfg_path):
            import yaml
            with open(run_cfg_path) as fh:
                run_cfg = yaml.safe_load(fh)
            N_stars = int(run_cfg.get("bias_cfg", {}).get(
                "n_stars_sample", DEFAULT_BIAS_CFG["n_stars_sample"]))
            logger.info("N_stars=%d (from %s)", N_stars, run_cfg_path)
        else:
            N_stars = int(DEFAULT_BIAS_CFG["n_stars_sample"])
            logger.info("N_stars=%d (default; no run_config.yaml)", N_stars)

    # Clip ranges based on observed extremes
    clip_range = {
        "logP": (obs_logP.min(), obs_logP.max()),
        "e": (obs_e.min(), obs_e.max()),
        "K1": (obs_K1.min(), obs_K1.max()),
    }

    logger.info("Rescoring %d grid points with all tests (KS, AD, CvM)",
                n_steps)
    logger.info("Observed: %d systems (SB1=%d, SB2=%d)",
                N_det_obs, obs["n_sb1"], obs["n_sb2"])
    logger.info("Clip ranges: logP=[%.2f,%.2f] e=[%.3f,%.3f] K1=[%.1f,%.1f]",
                *clip_range["logP"], *clip_range["e"], *clip_range["K1"])

    # Allocate cubes for all tests
    gmf_cubes = {}
    test_cubes = {}
    for tname in _ALL_TESTS:
        gmf_cubes[tname] = np.full(shape, -np.inf)
        test_cubes[tname] = {
            "logP": np.zeros(shape),
            "e": np.zeros(shape),
            "K1": np.zeros(shape),
        }

    # Histograms path (for symlinking)
    hists_path = os.path.join(input_dir, "grid_hists.npz")
    if not os.path.exists(hists_path):
        hists_path = os.path.join(input_dir, "checkpoint_hists.npz")

    t0 = time.time()
    n_done = 0
    n_empty = 0

    for row in step_to_ijkl:
        step = int(row[0])
        i, j, k, l = int(row[1]), int(row[2]), int(row[3]), int(row[4])

        shard = _load_det_shard(input_dir, step)
        if shard is None or len(shard["logP"]) < 5:
            n_empty += 1
            n_done += 1
            continue

        # Clip to observed range
        sim_clipped = {
            "logP": _clip_to_range(shard["logP"], *clip_range["logP"]),
            "e": _clip_to_range(shard["e"], *clip_range["e"]),
            "K1": _clip_to_range(shard["K1"], *clip_range["K1"]),
        }

        # Binomial term (shared across tests). p_det = n_detected/n_physical
        # already absorbs f_bin via the per-realization coin flip, so it is
        # the binomial success probability directly — no extra fbin factor.
        p_det = pdet_cube[i, j, k, l]
        p_binom = binom.pmf(N_det_obs, N_stars, p_det) \
            if p_det > 0 else 0.0

        # Compute p-values and GMF for each test
        for tname, tfn in _ALL_TESTS.items():
            p_logP = _safe_pvalue(tfn, obs_logP, sim_clipped["logP"])
            p_e = _safe_pvalue(tfn, obs_e, sim_clipped["e"])
            p_K1 = _safe_pvalue(tfn, obs_K1, sim_clipped["K1"],
                                min_samples=3)

            test_cubes[tname]["logP"][i, j, k, l] = p_logP
            test_cubes[tname]["e"][i, j, k, l] = p_e
            test_cubes[tname]["K1"][i, j, k, l] = p_K1

            log_gmf = 0.0
            for pval in [p_logP, p_e, p_K1, p_binom]:
                if pval > 0:
                    log_gmf += np.log(pval)
                else:
                    log_gmf = -np.inf
                    break
            gmf_cubes[tname][i, j, k, l] = log_gmf

        n_done += 1
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

    logger.info("Saved rescored cubes to %s", output_dir)
    logger.info("Symlinked det_shards + det_index + hists from %s", input_dir)

    return gmf_cubes, best_fits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rescore a bias grid run with all three statistical tests "
                    "(KS, Anderson-Darling, Cramér-von Mises) and "
                    "observed-range clipping.",
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Path to the original bias grid output (with det_shards/).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Default: <input-dir>_rescored/.",
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
        "--n-stars-sample", type=int, default=None,
        help="Override binomial denominator (closure tests should use the "
             "synthetic catalog's n_stars from truth_params.yaml). "
             "If unset, uses DEFAULT_BIAS_CFG['n_stars_sample'] (134).",
    )
    cli = parser.parse_args()

    output_dir = cli.output_dir or (cli.input_dir.rstrip("/") + "_rescored")

    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, "rescore.log"), mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(sh)

    rescore(cli.input_dir, output_dir,
            sb1_tex=cli.sb1_tex, sb2_tex=cli.sb2_tex,
            n_stars_sample=cli.n_stars_sample)


if __name__ == "__main__":
    main()
