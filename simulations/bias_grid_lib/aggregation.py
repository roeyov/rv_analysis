"""SLURM-task result merger.

``aggregate_tasks(base_dir)`` walks every ``task_*/`` subdirectory of
``base_dir``, loads each task's checkpoint cubes / CSV / shards /
histograms, merges them into final ``grid_cubes.npz`` + ``grid_hists.npz``
files at the top level, copies all shards into a single
``det_shards/`` directory, runs the end-of-run plots, and removes the
per-task subdirectories.
"""

import os
import numpy as np
import pandas as pd

from simulations.bias_grid_lib.constants import (
    CUBE_SCHEMA_VERSION, _DET_SHARDS_DIR, _HIST_BINS, _HIST_NBINS, _HIST_PAIRS,
)
from simulations.bias_grid_lib.checkpointing import _save_det_index
from simulations.bias_grid_lib.logging_utils import logger
from simulations.bias_grid_lib.plotting import plot_grid_results
from simulations.bias_grid_lib.statistics import _ALL_TESTS, _SCORED_TESTS


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
    for tname in _SCORED_TESTS:
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
    merged_e_score_mode = None
    merged_logP_cutoff_mode = None
    merged_logP_cutoff = None
    merged_logP_cutoff_scope = None
    merged_apply_lucy_sweeny_e = None

    for td in task_dirs:
        npz_path = os.path.join(td, "checkpoint_cubes.npz")
        csv_path = os.path.join(td, "checkpoint_results.csv")
        if not os.path.exists(npz_path) or not os.path.exists(csv_path):
            logger.warning("  Skipping incomplete task dir: %s", td)
            continue

        ckpt = np.load(npz_path, allow_pickle=True)

        # Track scoring mode across tasks — flag if SLURM tasks were
        # somehow run with different settings.
        if "e_score_mode" in ckpt.files:
            this_mode = str(ckpt["e_score_mode"])
        else:
            has_e_circ = any(("%s_e_circ_cube" % t) in ckpt.files
                             for t in _ALL_TESTS)
            this_mode = "split" if has_e_circ else "combined"
        if merged_e_score_mode is None:
            merged_e_score_mode = this_mode
        elif this_mode != merged_e_score_mode:
            logger.warning(
                "Task %s e_score_mode=%s differs from %s; merge may be "
                "inconsistent", td, this_mode, merged_e_score_mode)

        # Same check for the logP cutoff. Pre-feature checkpoints lack
        # the keys → treat as "none" / 0.0 / "period_only".
        this_logP_mode = (str(ckpt["logP_cutoff_mode"])
                          if "logP_cutoff_mode" in ckpt.files else "none")
        this_logP_cutoff = (float(ckpt["logP_cutoff"])
                            if "logP_cutoff" in ckpt.files else 0.0)
        this_logP_scope = (str(ckpt["logP_cutoff_scope"])
                           if "logP_cutoff_scope" in ckpt.files
                           else "period_only")
        if merged_logP_cutoff_mode is None:
            merged_logP_cutoff_mode = this_logP_mode
            merged_logP_cutoff = this_logP_cutoff
            merged_logP_cutoff_scope = this_logP_scope
        elif (this_logP_mode != merged_logP_cutoff_mode
              or not np.isclose(this_logP_cutoff, merged_logP_cutoff,
                                atol=1e-6)
              or this_logP_scope != merged_logP_cutoff_scope):
            logger.warning(
                "Task %s logP_cutoff_mode=%s, cutoff=%.4f, scope=%s differs "
                "from %s, %.4f, %s; merge may be inconsistent",
                td, this_logP_mode, this_logP_cutoff, this_logP_scope,
                merged_logP_cutoff_mode, merged_logP_cutoff,
                merged_logP_cutoff_scope)

        # Lucy-Sweeney handling — same cross-task consistency check.
        this_lucy = (bool(ckpt["apply_lucy_sweeny_e"])
                     if "apply_lucy_sweeny_e" in ckpt.files else True)
        if merged_apply_lucy_sweeny_e is None:
            merged_apply_lucy_sweeny_e = this_lucy
        elif this_lucy != merged_apply_lucy_sweeny_e:
            logger.warning(
                "Task %s apply_lucy_sweeny_e=%s differs from %s; merge "
                "may be inconsistent", td, this_lucy,
                merged_apply_lucy_sweeny_e)

        # Merge cubes: each task only fills its own cells.
        pdet_cube += ckpt["pdet_cube"]
        # Per-test cubes (with backward compat)
        for tname in _SCORED_TESTS:
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
            # e_circ cube (only present when e_score_mode == "split")
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
    for tname in _SCORED_TESTS:
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
        "e_score_mode": merged_e_score_mode or "combined",
        "logP_cutoff_mode": merged_logP_cutoff_mode or "none",
        "logP_cutoff": merged_logP_cutoff if merged_logP_cutoff is not None else 0.0,
        "logP_cutoff_scope": merged_logP_cutoff_scope or "period_only",
    }

    # Save merged output.
    os.makedirs(output_dir, exist_ok=True)

    save_kw_cubes = dict(
        cube_schema_version=np.array(CUBE_SCHEMA_VERSION),
        pdet_cube=pdet_cube,
        pi_grid=pi_grid,
        kappa_grid=kappa_grid,
        eta_grid=eta_grid,
        fbin_grid=fbin_grid,
        e_score_mode=np.array(merged_e_score_mode or "combined"),
        logP_cutoff_mode=np.array(merged_logP_cutoff_mode or "none"),
        logP_cutoff=np.array(merged_logP_cutoff
                             if merged_logP_cutoff is not None else 0.0),
        logP_cutoff_scope=np.array(merged_logP_cutoff_scope or "period_only"),
        apply_lucy_sweeny_e=np.array(
            bool(merged_apply_lucy_sweeny_e)
            if merged_apply_lucy_sweeny_e is not None else True),
    )
    # Forward the obs arrays + extra metadata from the first task's
    # checkpoint (they're identical across tasks — same obs is loaded
    # in every worker). The keys are required by the explorer; aggregate
    # runs that consumed pre-schema-v2 task checkpoints will lack them.
    for k in ("logP_cutoff_smooth_sigma", "sb1_tex", "sb2_tex",
              "obs_logP", "obs_e_value", "obs_e_is_upper_limit",
              "obs_K1", "obs_q_sb2", "obs_n_sb1", "obs_n_sb2",
              "obs_n_catalog_total", "obs_n_catalog_nonsingle",
              "ostar_catalog"):
        if k in first_npz.files:
            save_kw_cubes[k] = first_npz[k]
        else:
            logger.warning(
                "aggregate_tasks: task checkpoints lack %s — merged cube "
                "will fail the explorer's schema check. Re-run tasks with "
                "the updated bias_grid.py.", k)
    # Wasserstein per-channel σ — optional (only present on cubes built
    # after the Wasserstein metric was added). Silently skip on older
    # task checkpoints; the explorer just falls back to plain labels.
    for k in ("wass_sigma_logP", "wass_sigma_e", "wass_sigma_K1"):
        if k in first_npz.files:
            save_kw_cubes[k] = first_npz[k]
    for tname in _SCORED_TESTS:
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
