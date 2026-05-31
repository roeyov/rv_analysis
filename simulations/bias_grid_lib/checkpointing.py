"""Shard-based persistence: per-grid-point detected arrays + cube checkpoints.

The shard format keeps each grid point's detected/non-detected arrays in
its own ``det_shards/step_NNNNNN.npz`` so the engine never holds the
full per-cell array list in memory. ``_save_checkpoint`` writes the
fixed-size cubes, per-CSV row table, and 2D detection-probability
histograms in three companion files.
"""

import os
import numpy as np
import pandas as pd

from simulations.bias_grid_lib.constants import (
    CUBE_SCHEMA_VERSION, _DET_SHARDS_DIR, _HIST_BINS, _HIST_NBINS, _HIST_PAIRS,
)
from simulations.bias_grid_lib.logging_utils import logger
from simulations.bias_grid_lib.statistics import _SCORED_TESTS


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
                     e_score_mode,
                     logP_cutoff_mode="none",
                     logP_cutoff=0.0,
                     logP_cutoff_scope="period_only",
                     apply_lucy_sweeny_e=True,
                     logP_cutoff_smooth_sigma=0.15,
                     sb1_tex="",
                     sb2_tex="",
                     obs=None,
                     n_catalog_total=None,
                     n_catalog_nonsingle=None,
                     ostar_catalog="",
                     wass_sigma=None,
                     global_hists=None):
    """Save intermediate results so a killed run can be resumed.

    Detected arrays are saved per-grid-point as shard files by the
    caller (_score_and_accumulate), so this function only persists the
    cubes, scalar CSV, and global histograms.
    """
    logger.debug("_save_checkpoint: step %d", completed_steps)
    save_kw = dict(
        cube_schema_version=np.array(CUBE_SCHEMA_VERSION),
        pdet_cube=pdet_cube,
        pi_grid=pi_grid,
        kappa_grid=kappa_grid,
        eta_grid=eta_grid,
        fbin_grid=fbin_grid,
        completed_steps=np.array(completed_steps),
        n_inject_per_star=np.array(n_inject_per_star),
        seed=np.array(seed),
        preset_name=np.array(preset_name),
        e_score_mode=np.array(e_score_mode),
        logP_cutoff_mode=np.array(logP_cutoff_mode),
        logP_cutoff=np.array(logP_cutoff),
        logP_cutoff_scope=np.array(logP_cutoff_scope),
        apply_lucy_sweeny_e=np.array(bool(apply_lucy_sweeny_e)),
        logP_cutoff_smooth_sigma=np.array(float(logP_cutoff_smooth_sigma)),
        sb1_tex=np.array(str(sb1_tex)),
        sb2_tex=np.array(str(sb2_tex)),
    )
    if obs is not None:
        save_kw.update(
            obs_logP=np.asarray(obs["logP"]),
            obs_e_value=np.asarray(obs["e_value"]),
            obs_e_is_upper_limit=np.asarray(obs["e_is_upper_limit"], dtype=bool),
            obs_K1=np.asarray(obs["K1"]),
            obs_q_sb2=np.asarray(obs["q_sb2"]),
            obs_n_sb1=np.array(int(obs["n_sb1"])),
            obs_n_sb2=np.array(int(obs["n_sb2"])),
        )
    if wass_sigma is not None:
        save_kw.update(
            wass_sigma_logP=np.array(float(wass_sigma["logP"])),
            wass_sigma_e=np.array(float(wass_sigma["e"])),
            wass_sigma_K1=np.array(float(wass_sigma["K1"])),
        )
    if n_catalog_total is not None:
        save_kw["obs_n_catalog_total"] = np.array(int(n_catalog_total))
    if n_catalog_nonsingle is not None:
        save_kw["obs_n_catalog_nonsingle"] = np.array(int(n_catalog_nonsingle))
    if ostar_catalog:
        save_kw["ostar_catalog"] = np.array(str(ostar_catalog))
    # All tests
    for tname in _SCORED_TESTS:
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
