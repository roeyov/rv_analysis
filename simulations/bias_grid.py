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
    _save_variant_cube_kw,
)
from simulations.bias_grid_lib.constants import (
    CUBE_SCHEMA_VERSION, EXPLORER_MIN_SCHEMA_VERSION,
    G_CGS, MSUN, RSUN, DAY, KM, TWOPI,
    _DET_SHARDS_DIR,
    _E_SCORE_MODES, _LOGP_CUTOFF_MODES, _LOGP_CUTOFF_SCOPES,
    _HIST_BINS, _HIST_NBINS, _HIST_PAIRS,
    _trapz, all_variants, variant_tag, split_variant_tag,
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

    # All-variants run (schema v4): one simulation pass is scored under
    # every (e_score_mode, logP_cutoff_mode) variant, so the per-variant
    # e_score_mode / logP_cutoff_mode YAML keys are no longer read here —
    # they are enumerated internally (constants.all_variants). Only the
    # run-level knobs (cutoff scope, smoothing σ, Lucy) remain configurable.
    logP_cutoff_scope_cfg = _bg("logP_cutoff_scope")
    if logP_cutoff_scope_cfg not in _LOGP_CUTOFF_SCOPES:
        raise ValueError(
            "Unknown logP_cutoff_scope %r in bias_grid; valid: %s"
            % (logP_cutoff_scope_cfg, " | ".join(_LOGP_CUTOFF_SCOPES)))
    if logP_cutoff_scope_cfg != "exclude":
        logger.warning(
            "logP_cutoff_scope=%r — the canonical all-variants matrix "
            "assumes 'exclude'. Proceeding with the configured scope.",
            logP_cutoff_scope_cfg)

    # Load bias config (defaults + run-level YAML overrides).
    cfg = copy.deepcopy(DEFAULT_BIAS_CFG)
    cfg["logP_cutoff_scope"] = logP_cutoff_scope_cfg
    cfg["logP_cutoff_smooth_sigma"] = _bg("logP_cutoff_smooth_sigma")
    # The variant matrix only spans logP_cutoff_mode in {none, numerical};
    # there is no 'manual' variant, so no manual cutoff value is used.
    cfg["logP_cutoff_value"] = None
    # Drop single-mode keys so nothing downstream reads a stale scalar.
    cfg.pop("e_score_mode", None)
    cfg.pop("logP_cutoff_mode", None)
    cfg.pop("split_e_circular", None)

    # Sampling bounds for the log-P power law (now YAML-configurable; the
    # defaults reproduce historical runs). log_p_min is also the lower bound
    # used by the adaptive-budget F_>(π) — set it to a physical value
    # (e.g. the observed minimum logP) to avoid the x_min→0 normalization
    # blow-up for π < -1.
    cfg["log_p_min"] = float(_bg("log_p_min"))
    cfg["log_p_max"] = float(_bg("log_p_max"))

    # Adaptive injection budget (opt-in; default OFF). See DEFAULT_BIAS_CFG
    # and engine.run for the n_inject(π, f_bin) scaling.
    cfg["adaptive_n_inject"] = bool(_bg("adaptive_n_inject"))
    cfg["n_above_cutoff_target"] = int(_bg("n_above_cutoff_target"))
    cfg["n_inject_max"] = int(_bg("n_inject_max"))
    _adaptive_cutoff = _bg("adaptive_n_cutoff")
    cfg["adaptive_n_cutoff"] = (None if _adaptive_cutoff is None
                                else float(_adaptive_cutoff))

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

    # 1) Load observed distributions.
    # apply_lucy_sweeny_e is now a SWEPT variant axis (schema v5), not a
    # single run knob — the engine derives both conventions per variant
    # from the raw obs e_value + e_is_upper_limit. The value below is only
    # a legacy placeholder for the cube's shared scalar (v3/v4 readers);
    # the obs["e"] it produces is unused by the all-variants scorer.
    apply_lucy_sweeny_e = bool(_bg("apply_lucy_sweeny_e"))
    cfg["apply_lucy_sweeny_e"] = apply_lucy_sweeny_e
    logger.info("Loading observed distributions from LaTeX tables "
                "(apply_lucy_sweeny_e is swept: {False, True})...")
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
    _variant_tags = [variant_tag(em, cm, lucy)
                     for (em, cm, lucy) in all_variants()]
    logger.info("  All-variants run: %d variants = %s",
                len(_variant_tags), ", ".join(_variant_tags))
    logger.info("  logP_cutoff_scope: %s (fixed across variants); "
                "smooth_sigma=%.3f",
                cfg["logP_cutoff_scope"],
                cfg.get("logP_cutoff_smooth_sigma", 0.15))

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
    if cfg["adaptive_n_inject"]:
        logger.info("  Injections per star: ADAPTIVE (baseline %d, "
                    "target N>cutoff=%d, cap=%d, log_p_min=%.3f)",
                    n_inject, cfg["n_above_cutoff_target"],
                    cfg["n_inject_max"], cfg["log_p_min"])
    else:
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
        "variants": [variant_tag(em, cm, lucy)
                     for (em, cm, lucy) in all_variants()],
        "logP_cutoff_scope": cfg["logP_cutoff_scope"],
        "logP_cutoff_smooth_sigma": cfg.get("logP_cutoff_smooth_sigma", 0.15),
        "apply_lucy_sweeny_e": apply_lucy_sweeny_e,
        "log_p_min": cfg["log_p_min"],
        "log_p_max": cfg["log_p_max"],
        "adaptive_n_inject": cfg["adaptive_n_inject"],
        "n_above_cutoff_target": cfg["n_above_cutoff_target"],
        "n_inject_max": cfg["n_inject_max"],
        "adaptive_n_cutoff": cfg["adaptive_n_cutoff"],
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

    # Save cubes (schema v4: one shared pdet_cube + obs block, plus
    # namespaced per-variant goodness cubes via _save_variant_cube_kw).
    save_kw_cubes = dict(
        cube_schema_version=np.array(CUBE_SCHEMA_VERSION),
        pdet_cube=results["pdet_cube"],
        pi_grid=pi_grid,
        kappa_grid=kappa_grid,
        eta_grid=eta_grid,
        fbin_grid=fbin_grid,
        apply_lucy_sweeny_e=np.array(bool(apply_lucy_sweeny_e)),
        logP_cutoff_scope=np.array(
            str(results.get("logP_cutoff_scope", "exclude"))),
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
    _save_variant_cube_kw(
        save_kw_cubes,
        results["gmf_cubes_by_variant"],
        results["test_cubes_by_variant"],
        results["variant_meta"],
        results["variants"],
    )
    np.savez(os.path.join(output_dir, "grid_cubes.npz"), **save_kw_cubes)
    logger.info("Saved grid cubes (%d variants) to %s/grid_cubes.npz",
                len(results["variants"]), output_dir)

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

    # Static plotting is intentionally not run here: with 6 variants the
    # comparison lives in the interactive explorer (bias_grid_explorer.py),
    # which switches between fitting mechanisms on one loaded run.
    logger.info("Done. Explore with: streamlit run "
                "simulations/bias_grid_explorer.py -- --output-dir %s",
                output_dir)


if __name__ == "__main__":
    main()
