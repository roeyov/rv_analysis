"""SLURM-task result merger (all-variants schema v4).

``aggregate_tasks(base_dir)`` walks every ``task_*/`` subdirectory of
``base_dir``, loads each task's checkpoint cubes / CSV / shards /
histograms, merges them into final ``grid_cubes.npz`` + ``grid_hists.npz``
files at the top level, copies all shards into a single
``det_shards/`` directory, and removes the per-task subdirectories.

Each task scores the SAME 6 (e_score_mode, logP_cutoff_mode) variants, so
the merge loops over the variant namespace ('v__<tag>__…') and combines
each variant's goodness cubes independently. The shared ``pdet_cube`` /
global histograms / det_shards are variant-independent and merge once.
Static plotting is intentionally not run — model comparison lives in the
interactive explorer.
"""

import os
import numpy as np
import pandas as pd

from simulations.bias_grid_lib.constants import (
    CUBE_SCHEMA_VERSION, _DET_SHARDS_DIR, _HIST_BINS, _HIST_NBINS, _HIST_PAIRS,
)
from simulations.bias_grid_lib.checkpointing import (
    _save_det_index, _save_variant_cube_kw,
)
from simulations.bias_grid_lib.logging_utils import logger
from simulations.bias_grid_lib.statistics import _SCORED_TESTS


def _variant_has_e_circ(npz, tag):
    """True if `tag` carries an e_circ cube in `npz` (split variants only)."""
    return any("v__%s__%s_e_circ_cube" % (tag, t) in npz.files
               for t in _SCORED_TESTS)


def _reconstruct_variant_meta(npz, tags):
    """Rebuild the per-variant metadata dict from a task checkpoint npz.

    The scalar metadata (resolved cutoff, N_stars, N_det_obs, wass_sigma,
    modes) is identical across SLURM tasks, so the first task's values
    carry through to the merged cube unchanged.
    """
    variant_meta = {}
    for tag in tags:
        lucy_key = "v__%s__apply_lucy_sweeny_e" % tag
        meta = {
            "e_score_mode": str(npz["v__%s__e_score_mode" % tag]),
            "logP_cutoff_mode": str(npz["v__%s__logP_cutoff_mode" % tag]),
            "apply_lucy_sweeny_e": (bool(npz[lucy_key])
                                    if lucy_key in npz.files else False),
            "logP_cutoff": float(npz["v__%s__logP_cutoff" % tag]),
            "N_stars": int(npz["v__%s__N_stars" % tag]),
            "N_det_obs": int(npz["v__%s__N_det_obs" % tag]),
            "has_e_circ": _variant_has_e_circ(npz, tag),
        }
        if "v__%s__wass_sigma_logP" % tag in npz.files:
            meta["wass_sigma"] = {
                ch: float(npz["v__%s__wass_sigma_%s" % (tag, ch)])
                for ch in ("logP", "e", "K1")
            }
        variant_meta[tag] = meta
    return variant_meta


def aggregate_tasks(base_dir, output_dir=None):
    """
    Merge partial results from SLURM array tasks into final output.

    Discovers all task_*/ subdirectories under base_dir, loads their
    checkpoint_cubes.npz and checkpoint_results.csv, merges the cubes
    (per variant), and produces the final grid_cubes.npz.

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

    # Load first task to get grid shapes, the variant set, and the shared
    # metadata block.
    first_npz = np.load(
        os.path.join(task_dirs[0], "checkpoint_cubes.npz"),
        allow_pickle=True)
    pi_grid = first_npz["pi_grid"]
    kappa_grid = first_npz["kappa_grid"]
    eta_grid = first_npz["eta_grid"]
    fbin_grid = first_npz["fbin_grid"]
    if "variants" not in first_npz.files:
        raise RuntimeError(
            "Task checkpoint %s predates the all-variants schema (no "
            "'variants' key). Re-run the tasks with the updated bias_grid."
            % task_dirs[0])
    tags = [str(t) for t in first_npz["variants"]]
    # Metrics actually present in the cube — single-metric runs persist only
    # their scored set. Probe tags[0]; default test keeps "ks" when scored.
    scored_tests = [t for t in _SCORED_TESTS
                    if "v__%s__gmf_%s_cube" % (tags[0], t) in first_npz.files]
    if not scored_tests:
        raise RuntimeError(
            "No gmf_<test>_cube keys for variant %s in %s — cannot merge."
            % (tags[0], task_dirs[0]))
    default_test = "ks" if "ks" in scored_tests else scored_tests[0]
    shape = first_npz["v__%s__gmf_%s_cube" % (tags[0], default_test)].shape
    has_e_circ = {tag: _variant_has_e_circ(first_npz, tag) for tag in tags}

    # Initialise merged per-variant cubes.
    pdet_cube = np.zeros(shape)
    gmf_cubes_v = {}
    test_cubes_v = {}
    for tag in tags:
        gmf_cubes_v[tag] = {
            t: np.full(shape, -np.inf, dtype=np.float32) for t in scored_tests
        }
        tcv = {}
        for t in scored_tests:
            d = {
                "logP": np.zeros(shape, dtype=np.float32),
                "e": np.zeros(shape, dtype=np.float32),
                "K1": np.zeros(shape, dtype=np.float32),
            }
            if has_e_circ[tag]:
                d["e_circ"] = np.zeros(shape, dtype=np.float32)
            tcv[t] = d
        test_cubes_v[tag] = tcv

    all_results_dfs = []
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

        # Variant-set consistency: every task must score the same 6
        # variants (else the merge would combine mismatched cubes).
        this_tags = ([str(t) for t in ckpt["variants"]]
                     if "variants" in ckpt.files else None)
        if this_tags is None or set(this_tags) != set(tags):
            logger.warning(
                "  Task %s variant set %s differs from %s — skipping to "
                "avoid an inconsistent merge", td, this_tags, tags)
            continue

        # Merge cubes: each task only fills its own (disjoint) cells.
        pdet_cube += ckpt["pdet_cube"]
        for tag in tags:
            for tname in scored_tests:
                gk = "v__%s__gmf_%s_cube" % (tag, tname)
                if gk in ckpt.files:
                    gmf_cubes_v[tag][tname] = np.maximum(
                        gmf_cubes_v[tag][tname], ckpt[gk])
                for par in ("logP", "e", "K1"):
                    ck = "v__%s__%s_%s_cube" % (tag, tname, par)
                    if ck in ckpt.files:
                        test_cubes_v[tag][tname][par] += ckpt[ck]
                if has_e_circ[tag]:
                    ek = "v__%s__%s_e_circ_cube" % (tag, tname)
                    if ek in ckpt.files:
                        test_cubes_v[tag][tname]["e_circ"] += ckpt[ek]

        # Merge detected-array index + histograms (shard-based format).
        shard_dir = os.path.join(td, _DET_SHARDS_DIR)
        det_index_path = os.path.join(td, "det_index.npz")
        if os.path.isdir(shard_dir) and os.path.exists(det_index_path):
            has_detected = True
            idx_data = np.load(det_index_path)
            all_step_to_ijkl.extend(idx_data["step_to_ijkl"].tolist())
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

    # Best fit per variant (KS) for logging.
    for tag in tags:
        gc = gmf_cubes_v[tag]["ks"]
        idx = np.unravel_index(np.nanargmax(gc), gc.shape)
        logger.info("Best fit [%s] (KS): π=%.2f, κ=%.2f, η=%.2f, f_bin=%.2f",
                    tag, float(pi_grid[idx[0]]), float(kappa_grid[idx[1]]),
                    float(eta_grid[idx[2]]), float(fbin_grid[idx[3]]))

    variant_meta = _reconstruct_variant_meta(first_npz, tags)

    # Save merged output.
    os.makedirs(output_dir, exist_ok=True)
    save_kw_cubes = dict(
        cube_schema_version=np.array(CUBE_SCHEMA_VERSION),
        pdet_cube=pdet_cube,
        pi_grid=pi_grid,
        kappa_grid=kappa_grid,
        eta_grid=eta_grid,
        fbin_grid=fbin_grid,
    )
    # Forward shared run-level metadata + obs block from the first task
    # (identical across tasks — same obs loaded in every worker).
    for k in ("apply_lucy_sweeny_e", "logP_cutoff_scope",
              "logP_cutoff_smooth_sigma", "sb1_tex", "sb2_tex",
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
    # Adaptive injection-budget provenance (optional; absent in legacy runs).
    for k in ("adaptive_n_inject", "adaptive_n_above_target",
              "adaptive_n_inject_max", "adaptive_log_p_min",
              "adaptive_log_p_max", "adaptive_cutoff",
              "adaptive_n_inject_grid"):
        if k in first_npz.files:
            save_kw_cubes[k] = first_npz[k]
    _save_variant_cube_kw(save_kw_cubes, gmf_cubes_v, test_cubes_v,
                          variant_meta, tags)

    merged_cubes_path = os.path.join(output_dir, "grid_cubes.npz")
    np.savez(merged_cubes_path, **save_kw_cubes)
    logger.info("Saved merged cubes (%d variants) to %s",
                len(tags), merged_cubes_path)

    # Save merged histograms and detected-array index.
    if has_detected:
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

        save_kw = {}
        for k, v in _HIST_BINS.items():
            save_kw["bins_%s" % k] = v
        for (a, b) in _HIST_PAIRS:
            save_kw["hist_total_%s_%s" % (a, b)] = merged_hist_total[(a, b)]
            save_kw["hist_det_%s_%s" % (a, b)] = merged_hist_det[(a, b)]
        np.savez(os.path.join(output_dir, "grid_hists.npz"), **save_kw)
        logger.info("Saved merged histograms + %d shard files to %s",
                     len(all_step_to_ijkl), output_dir)

    # Validate the merged cube loads before destroying the per-task dirs —
    # a merge bug must never delete the only copy of the data.
    try:
        _check = np.load(merged_cubes_path, allow_pickle=True)
        for tag in tags:
            assert "v__%s__gmf_ks_cube" % tag in _check.files
        assert "variants" in _check.files
    except Exception as exc:  # noqa: BLE001 — guard before irreversible delete
        logger.error(
            "Merged cube validation failed (%s). Leaving task_*/ dirs in "
            "place so no data is lost.", exc)
        return {"variants": tags, "output_dir": output_dir}

    import shutil as _shutil
    for td in task_dirs:
        _shutil.rmtree(td)
    logger.info("Aggregation complete. Deleted %d task directories.",
                len(task_dirs))

    return {"variants": tags, "output_dir": output_dir}
