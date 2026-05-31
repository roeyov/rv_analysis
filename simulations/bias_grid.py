"""
simulations.bias_grid — Sana+2012/2013 forward-modeling bias correction grid search.

4D grid search over (π, κ, η, f_bin) to determine intrinsic binary fraction
and power-law shape parameters by comparing simulated detected populations
to the observed SB1+SB2 sample.

Usage:
    python -m simulations.bias_grid --config configs/params_bias.yaml
    python -m simulations.bias_grid --config configs/params_bias.yaml --quick   # 3x3x3x3 test

Module structure
----------------
This file is the CLI entry point and a facade over ``bias_grid_lib``.
Implementation lives in ``simulations/bias_grid_lib/``. The
``from simulations.bias_grid import …`` contract is preserved — every
symbol previously defined here is re-exported below.
"""

import argparse
import copy
import os

import numpy as np
import yaml

# --- Facade re-exports: preserve `from simulations.bias_grid import X` ---
from simulations.bias_grid_lib.aggregation import aggregate_tasks
from simulations.bias_grid_lib.checkpointing import (
    _det_shard_path, _hists_from_shard,
    _load_det_shard, _save_checkpoint, _save_det_index, _save_det_shard,
)
from simulations.bias_grid_lib.constants import (
    CUBE_SCHEMA_VERSION,
    G_CGS, MSUN, RSUN, DAY, KM, TWOPI,
    _DET_SHARDS_DIR,
    _E_SCORE_MODES, _LOGP_CUTOFF_MODES, _LOGP_CUTOFF_SCOPES,
    _HIST_BINS, _HIST_NBINS, _HIST_PAIRS,
    _trapz,
)
from simulations.bias_grid_lib.cutoffs import (
    _compute_logP_cutoff, _numerical_logP_cutoff,
    _resolve_e_score_mode,
    _resolve_logP_cutoff_mode, _resolve_logP_cutoff_scope,
)
from simulations.bias_grid_lib.detection import (
    DETECTION_METHODS, _detect_full_pipeline, _detect_rv_threshold,
)
from simulations.bias_grid_lib.engine import GridSearchEngine
from simulations.bias_grid_lib.injection import (
    _worker_star_injections, _worker_star_injections_vectorized,
    detect_single_star, inject_and_detect,
)
from simulations.bias_grid_lib.logging_utils import logger, setup_logging
from simulations.bias_grid_lib.obs_io import (
    _load_catalog_counts, _parse_val_with_errors, load_observed_from_tex,
)
from simulations.bias_grid_lib.parallel import (
    _init_grid_worker, _init_resume_worker,
    _resume_score_worker, _worker_grid_point,
)
from simulations.bias_grid_lib.physics import (
    compute_K1, compute_K1_batch, kepler_E, kepler_E_batch,
    powerlaw_draw, roche_lobe_check, rv_model_jit,
)
from simulations.bias_grid_lib.plotting import (
    format_grid_summary, plot_grid_results,
)
from simulations.bias_grid_lib.scoring import (
    _compute_scores, _make_scoring_ctx,
)
from simulations.bias_grid_lib.star_loading import (
    load_observed_star_properties, load_star_properties,
)
from simulations.bias_grid_lib.statistics import (
    _ALL_TESTS, _DIST_TESTS, _SCORED_TESTS,
    _ad_pvalue, _clip_to_range, _cvm_pvalue, _ks_pvalue,
    _mad, _safe_distance, _safe_pvalue, _wasserstein_distance,
)

from simulations.bias_config import DEFAULT_BIAS_CFG, GRID_PRESETS
from simulations.common import BLOEM_MJD_ARRAYS
from pipeline.config import load_args


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bias correction grid search (Sana+2012 style). "
                    "All bias-grid input parameters are read from the "
                    "'bias_grid' section of the YAML config; only "
                    "execution / SLURM-task control remains as CLI.",
    )
    parser.add_argument(
        "--config", default="configs/params_bias.yaml",
        help="Path to pipeline config YAML. Default: configs/params_bias.yaml",
    )
    # --- SLURM array job support (execution control, not config) ---
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
    cli = parser.parse_args()

    # --- Aggregate mode: merge partial SLURM task results and exit ---
    if cli.aggregate:
        agg_dir = cli.aggregate
        setup_logging(agg_dir)
        logger.info("Aggregating SLURM task results from %s", agg_dir)
        aggregate_tasks(agg_dir)
        return

    # Load pipeline config + bias-grid block
    args_dict = load_args(cli.config)
    bg_cfg = args_dict.get("bias_grid", {}) or {}

    def _bg(key):
        """Read a bias-grid setting from YAML, falling back to defaults."""
        if key in bg_cfg and bg_cfg[key] is not None:
            return bg_cfg[key]
        return DEFAULT_BIAS_CFG.get(key)

    # Validate preset early so we can fail fast on a bad YAML value.
    preset_name = _bg("preset")
    if preset_name not in GRID_PRESETS:
        raise ValueError(
            "Unknown preset %r in bias_grid.preset; valid: %s" %
            (preset_name, sorted(GRID_PRESETS.keys())))

    detect_method = _bg("detect_method")
    if detect_method not in DETECTION_METHODS:
        raise ValueError(
            "Unknown detect_method %r in bias_grid; valid: %s" %
            (detect_method, sorted(DETECTION_METHODS.keys())))

    e_score_mode_cfg = _bg("e_score_mode")
    if e_score_mode_cfg not in ("combined", "split", "eccentric_only"):
        raise ValueError(
            "Unknown e_score_mode %r in bias_grid; valid: "
            "combined | split | eccentric_only" % (e_score_mode_cfg,))

    logP_cutoff_mode_cfg = _bg("logP_cutoff_mode")
    if logP_cutoff_mode_cfg not in _LOGP_CUTOFF_MODES:
        raise ValueError(
            "Unknown logP_cutoff_mode %r in bias_grid; valid: %s"
            % (logP_cutoff_mode_cfg, " | ".join(_LOGP_CUTOFF_MODES)))
    if logP_cutoff_mode_cfg == "manual" and _bg("logP_cutoff_value") is None:
        raise ValueError(
            "logP_cutoff_mode='manual' requires bias_grid.logP_cutoff_value "
            "to be set in the YAML config")

    logP_cutoff_scope_cfg = _bg("logP_cutoff_scope")
    if logP_cutoff_scope_cfg not in _LOGP_CUTOFF_SCOPES:
        raise ValueError(
            "Unknown logP_cutoff_scope %r in bias_grid; valid: %s"
            % (logP_cutoff_scope_cfg, " | ".join(_LOGP_CUTOFF_SCOPES)))

    # Load bias config (defaults + YAML overrides for the scoring-related keys).
    cfg = copy.deepcopy(DEFAULT_BIAS_CFG)
    cfg["e_score_mode"] = e_score_mode_cfg
    cfg["logP_cutoff_mode"] = logP_cutoff_mode_cfg
    cfg["logP_cutoff_value"] = _bg("logP_cutoff_value")
    cfg["logP_cutoff_smooth_sigma"] = _bg("logP_cutoff_smooth_sigma")
    cfg["logP_cutoff_scope"] = logP_cutoff_scope_cfg
    # Drop the legacy boolean so _resolve_e_score_mode uses the new
    # explicit key without ambiguity.
    cfg.pop("split_e_circular", None)

    # Paths
    sb1_tex = _bg("sb1_tex")
    sb2_tex = _bg("sb2_tex")
    mass_file = _bg("mass_file")
    rv_dir = _bg("rv_dir")
    sb2_analysis_dir = _bg("sb2_analysis_dir")
    ostar_catalog = _bg("ostar_catalog")
    output_dir = _bg("output_dir") or os.path.join(
        args_dict.get("base_dir", "."), "bias_grid_results", preset_name)

    # SLURM task mode: per-task subdirectory
    is_task_mode = cli.grid_start is not None
    if is_task_mode:
        grid_end = cli.grid_end if cli.grid_end is not None else 0
        output_dir = os.path.join(
            output_dir, "task_%d_%d" % (cli.grid_start, grid_end))

    # Setup logging (before any work)
    setup_logging(output_dir)

    # 1) Load observed distributions
    apply_lucy_sweeny_e = bool(_bg("apply_lucy_sweeny_e"))
    cfg["apply_lucy_sweeny_e"] = apply_lucy_sweeny_e
    logger.info("Loading observed distributions from LaTeX tables "
                "(apply_lucy_sweeny_e=%s)...", apply_lucy_sweeny_e)
    obs = load_observed_from_tex(sb1_tex, sb2_tex,
                                 apply_lucy_sweeny_e=apply_lucy_sweeny_e)
    # Catalog counts feed the binomial under scope="exclude" (full O-star
    # population, independent of the period cutoff). Loaded eagerly so a
    # bad path fails fast before the grid starts.
    n_catalog_total, n_catalog_nonsingle = _load_catalog_counts(ostar_catalog)
    logger.info("  SB1: %d, SB2: %d, Total: %d",
                obs['n_sb1'], obs['n_sb2'], len(obs['logP']))
    cfg["n_det_obs"] = len(obs["logP"])

    # 2) Load star properties — use the full O-star catalog (134 stars)
    # so the injection sample represents the entire population, not just
    # the detected binaries. SB2 stars (per ostar_catalog) pull from
    # rv_final_for_mcmc.csv; everyone else uses *_CCF_RVs.csv.
    logger.info("Loading full star sample from mass catalog...")
    star_df = load_star_properties(
        mass_file, rv_dir,
        sb2_analysis_dir=sb2_analysis_dir,
        ostar_catalog=ostar_catalog,
    )
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

    # n_stars_sample for the binomial denominator. Defaults to the actual
    # number of stars realized in the injection sample (so the binomial is
    # self-consistent with the population the simulator runs over). YAML
    # bias_grid.n_stars_sample overrides for closure tests.
    n_stars_sample_cfg = bg_cfg.get("n_stars_sample")
    if n_stars_sample_cfg is not None:
        cfg["n_stars_sample"] = int(n_stars_sample_cfg)
    else:
        cfg["n_stars_sample"] = len(star_df)
    logger.info("  n_stars_sample for binomial: %d (injection sample: %d)",
                cfg["n_stars_sample"], len(star_df))
    logger.info("  e_score_mode: %s", cfg["e_score_mode"])
    logger.info("  logP_cutoff_mode: %s (value=%s, smooth_sigma=%.3f)",
                cfg["logP_cutoff_mode"],
                cfg.get("logP_cutoff_value"),
                cfg.get("logP_cutoff_smooth_sigma", 0.15))
    logger.info("  logP_cutoff_scope: %s", cfg["logP_cutoff_scope"])

    # 3) Setup grids from the selected preset
    preset = GRID_PRESETS[preset_name]
    pi_grid = np.asarray(preset["pi"])
    kappa_grid = np.asarray(preset["kappa"])
    eta_grid = np.asarray(preset["eta"])
    fbin_grid = np.asarray(preset["fbin"])
    n_inject = _bg("n_inject") or preset["n_inject_per_star"]
    seed = _bg("seed")
    parallel_grid = bool(_bg("parallel_grid"))
    n_workers_cfg = _bg("n_workers")

    total = len(pi_grid) * len(kappa_grid) * len(eta_grid) * len(fbin_grid)

    logger.info("=" * 60)
    logger.info("  Bias Correction Grid Search")
    logger.info("  Preset: %s", preset_name)
    logger.info("  Grid: %d×%d×%d×%d = %d points",
                len(pi_grid), len(kappa_grid), len(eta_grid),
                len(fbin_grid), total)
    logger.info("  Injections per star: %d", n_inject)
    logger.info("  Stars: %d", len(star_df))
    logger.info("  Observed detections: %d", len(obs['logP']))
    logger.info("  Output: %s", output_dir)
    logger.info("  Detection method: %s", detect_method)
    logger.info("=" * 60)

    # 3b) Dump the resolved run config to YAML for reproducibility.
    # Captures the bias_grid block as actually resolved (YAML + defaults)
    # plus the selected preset arrays.
    os.makedirs(output_dir, exist_ok=True)
    resolved_bg = {
        "sb1_tex": sb1_tex,
        "sb2_tex": sb2_tex,
        "mass_file": mass_file,
        "rv_dir": rv_dir,
        "sb2_analysis_dir": sb2_analysis_dir,
        "ostar_catalog": ostar_catalog,
        "output_dir": output_dir,
        "preset": preset_name,
        "n_inject": int(n_inject),
        "seed": seed,
        "n_stars_sample": cfg["n_stars_sample"],
        "e_score_mode": cfg["e_score_mode"],
        "apply_lucy_sweeny_e": apply_lucy_sweeny_e,
        "detect_method": detect_method,
        "n_workers": n_workers_cfg,
        "parallel_grid": parallel_grid,
    }
    run_cfg_yaml = {
        "config_path": cli.config,
        "preset_name": preset_name,
        "preset": {
            "n_inject_per_star": int(n_inject),
            "pi": pi_grid.tolist(),
            "kappa": kappa_grid.tolist(),
            "eta": eta_grid.tolist(),
            "fbin": fbin_grid.tolist(),
        },
        "bias_grid": resolved_bg,
        "bias_cfg": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in cfg.items()
        },
        "slurm": {
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
    engine.detect_method = detect_method

    if parallel_grid:
        # In parallel-grid mode, default to cpu_count - 2 workers
        # (each worker handles one grid point).
        n_workers = n_workers_cfg or max(1, os.cpu_count() - 2)
    elif detect_method != "pipeline" and n_workers_cfg is None:
        n_workers = 1
    else:
        n_workers = n_workers_cfg or max(1, os.cpu_count() - 2)
    logger.info("  Workers: %d (parallel_grid=%s)",
                n_workers, parallel_grid)

    grid_start = cli.grid_start or 0
    grid_end_val = cli.grid_end  # None means all

    if is_task_mode:
        logger.info("  SLURM task mode: steps [%d, %d)", grid_start,
                     grid_end_val if grid_end_val is not None else total)

    results = engine.run(
        pi_grid, kappa_grid, eta_grid, fbin_grid,
        obs["logP"], obs["e"], obs["K1"],
        n_inject_per_star=n_inject,
        seed=seed,
        checkpoint_dir=output_dir,
        preset_name=preset_name,
        n_workers=n_workers,
        grid_start=grid_start,
        grid_end=grid_end_val,
        parallel_grid=parallel_grid,
        obs=obs,
        sb1_tex=sb1_tex,
        sb2_tex=sb2_tex,
        n_catalog_total=n_catalog_total,
        n_catalog_nonsingle=n_catalog_nonsingle,
        ostar_catalog=ostar_catalog,
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
        cube_schema_version=np.array(CUBE_SCHEMA_VERSION),
        pdet_cube=results["pdet_cube"],
        pi_grid=pi_grid,
        kappa_grid=kappa_grid,
        eta_grid=eta_grid,
        fbin_grid=fbin_grid,
        apply_lucy_sweeny_e=np.array(bool(apply_lucy_sweeny_e)),
        logP_cutoff_smooth_sigma=np.array(
            float(cfg.get("logP_cutoff_smooth_sigma", 0.15))),
        sb1_tex=np.array(str(sb1_tex)),
        sb2_tex=np.array(str(sb2_tex)),
        obs_logP=np.asarray(obs["logP"]),
        obs_e_value=np.asarray(obs["e_value"]),
        obs_e_is_upper_limit=np.asarray(obs["e_is_upper_limit"], dtype=bool),
        obs_K1=np.asarray(obs["K1"]),
        obs_q_sb2=np.asarray(obs["q_sb2"]),
        obs_n_sb1=np.array(int(obs["n_sb1"])),
        obs_n_sb2=np.array(int(obs["n_sb2"])),
        obs_n_catalog_total=np.array(int(n_catalog_total)),
        obs_n_catalog_nonsingle=np.array(int(n_catalog_nonsingle)),
        ostar_catalog=np.array(str(ostar_catalog)),
    )
    if "e_score_mode" in results:
        save_kw_cubes["e_score_mode"] = np.array(results["e_score_mode"])
    if "logP_cutoff_mode" in results:
        save_kw_cubes["logP_cutoff_mode"] = np.array(
            results["logP_cutoff_mode"])
        save_kw_cubes["logP_cutoff"] = np.array(results["logP_cutoff"])
        save_kw_cubes["logP_cutoff_scope"] = np.array(
            results.get("logP_cutoff_scope", "period_only"))
    if "wass_sigma" in results:
        save_kw_cubes["wass_sigma_logP"] = np.array(
            float(results["wass_sigma"]["logP"]))
        save_kw_cubes["wass_sigma_e"] = np.array(
            float(results["wass_sigma"]["e"]))
        save_kw_cubes["wass_sigma_K1"] = np.array(
            float(results["wass_sigma"]["K1"]))
    # All tests
    if "gmf_cubes" in results:
        for tname in _SCORED_TESTS:
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
