"""GridSearchEngine — the 4D (π, κ, η, f_bin) bias-grid driver.

Holds the per-star observable arrays and orchestrates:

1. For each grid point, run injection-recovery across every star.
2. Score the resulting (logP, e, K1) sim-detected distributions against
   the observed binary sample (KS / AD / CvM / Wasserstein) + the binomial
   on the detection count.
3. Aggregate per-grid-point scores into the 4D cubes used downstream.

Parallel modes:

- ``parallel_grid=False`` (default): stars run in parallel within a single
  grid point, grid points sequentially.
- ``parallel_grid=True``: grid points run in parallel (one per CPU), with
  stars sequential inside each grid point. Used for large grids on
  many-core hosts (e.g. astro3).

The ``run()`` method is intentionally one long method — extracting helpers
risks behavioural drift around the resume/forward/sequential branches and
the per-row CSV buffering. See plan §Risks.
"""

import itertools
import os
import threading
import time

import numpy as np
import pandas as pd

from simulations.bias_config import DEFAULT_BIAS_CFG
from simulations.bias_grid_lib.checkpointing import (
    _DET_SHARDS_DIR, _save_checkpoint, _save_det_index, _save_det_shard,
)
from simulations.bias_grid_lib.constants import (
    _HIST_NBINS, _HIST_PAIRS, all_variants, variant_tag,
)
from simulations.bias_grid_lib.cutoffs import (
    _build_variant_inputs, _resolve_logP_cutoff_scope,
)
from simulations.bias_grid_lib.injection import (
    _worker_star_injections, _worker_star_injections_vectorized,
)
from simulations.bias_grid_lib.logging_utils import logger
from simulations.bias_grid_lib.physics import n_inject_for_budget
from simulations.bias_grid_lib.parallel import (
    _init_grid_worker, _init_resume_worker, _resume_score_worker,
    _worker_grid_point,
)
from simulations.bias_grid_lib.scoring import (
    _compute_scores, _make_scoring_ctx,
)
from simulations.bias_grid_lib.statistics import _ALL_TESTS, _SCORED_TESTS


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
            parallel_grid=False,
            obs=None, sb1_tex="", sb2_tex="",
            n_catalog_total=None, n_catalog_nonsingle=None,
            ostar_catalog="",
            variants=None, scored_tests=None):
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

        # Legacy shared scalar (schema v5 sweeps lucy per variant; this is
        # only a backward-compat placeholder written to the cube/checkpoint
        # for v3/v4 readers).
        apply_lucy_sweeny_e = bool(self.cfg.get("apply_lucy_sweeny_e", False))

        # --- All-variants scoring (schema v5) ---------------------------
        # One injection/detection pass is scored under every
        # (e_score_mode, logP_cutoff_mode, apply_lucy_sweeny_e) variant
        # (12). The cutoff *scope*, smoothing σ, and manual value are
        # run-level (shared); bias_grid.main fixes scope='exclude', σ=0.15,
        # value=None. The expensive simulation is invariant to all of these
        # — only the cheap scoring context differs per variant (see
        # cutoffs._build_variant_inputs + scoring._compute_scores).
        logP_cutoff_scope = _resolve_logP_cutoff_scope(self.cfg)
        smooth_sigma = float(self.cfg.get("logP_cutoff_smooth_sigma", 0.15))
        manual_value = self.cfg.get("logP_cutoff_value")
        n_stars_sample = self.cfg.get("n_stars_sample", len(self.M1_arr))

        # Keep the observed arrays pristine — every variant masks its own
        # copy. logP/K1 are lucy-independent; the eccentricities are kept
        # RAW (e_value + upper-limit mask) so each variant applies its own
        # Lucy-Sweeney convention inside _build_variant_inputs.
        obs_logP_raw = np.asarray(obs_logP, dtype=float).copy()
        obs_K1_raw = np.asarray(obs_K1, dtype=float).copy()
        if obs is not None and "e_value" in obs:
            obs_e_value = np.asarray(obs["e_value"], dtype=float)
            obs_e_is_upper_limit = np.asarray(obs["e_is_upper_limit"], dtype=bool)
        else:
            # Fallback: treat the passed obs_e as already-resolved values
            # with no upper limits (lucy axis degenerates to those values).
            obs_e_value = np.asarray(obs_e, dtype=float)
            obs_e_is_upper_limit = np.zeros(len(obs_e_value), dtype=bool)

        # Variants/metrics default to the full sweep (all 12 / all 4) so any
        # caller that omits them reproduces the historical behaviour; the
        # orchestrator passes restricted lists for single-variant / single-
        # metric runs.
        if variants is None:
            variants = all_variants()            # up to 12 (em, cm, lucy) tuples
        scored_tests = list(scored_tests) if scored_tests else list(_SCORED_TESTS)
        # Default-test alias: keep "ks" when it is scored (⇒ identical legacy
        # aliases / return dict), else fall back to the first active metric.
        default_test = "ks" if "ks" in scored_tests else scored_tests[0]
        # Per-cell p-value tests + Wasserstein toggle, threaded into each
        # scoring ctx (pickled to workers) so the scorer only runs the
        # requested metrics.
        active_pvalue_tests = {k: _ALL_TESTS[k]
                               for k in scored_tests if k in _ALL_TESTS}
        score_wass = "wass" in scored_tests

        tags = [variant_tag(em, cm, lucy) for (em, cm, lucy) in variants]
        default_tag = variant_tag("combined", "numerical", False)
        if default_tag not in tags:
            default_tag = tags[0]

        scoring_ctxs = {}
        variant_meta = {}
        for (em, cm, lucy) in variants:
            tag = variant_tag(em, cm, lucy)
            vi = _build_variant_inputs(
                em, cm, lucy, logP_cutoff_scope,
                obs_logP_raw, obs_e_value, obs_e_is_upper_limit, obs_K1_raw,
                n_stars_sample, n_catalog_total, n_catalog_nonsingle,
                smooth_sigma=smooth_sigma, manual_value=manual_value)
            ctx = _make_scoring_ctx(
                obs_logP=vi["obs_logP"], obs_e=vi["obs_e"], obs_K1=vi["obs_K1"],
                clip_range=vi["clip_range"], e_score_mode=em,
                obs_e_cont=vi["obs_e_cont"],
                n_obs_circ=vi["n_obs_circ"] or 0,
                n_obs_e_total=vi["n_obs_e_total"] or 0,
                N_det_obs=vi["N_det_obs"], N_stars=vi["N_stars"],
                sim_logP_floor=vi["sim_logP_floor"])
            # Restrict which metrics the per-cell scorer computes (default:
            # all). Travels with the ctx into the parallel workers.
            ctx["pvalue_tests"] = active_pvalue_tests
            ctx["score_wass"] = score_wass
            scoring_ctxs[tag] = ctx
            variant_meta[tag] = {
                "e_score_mode": em,
                "logP_cutoff_mode": cm,
                "apply_lucy_sweeny_e": bool(lucy),
                "logP_cutoff": vi["logP_cutoff"],
                "N_stars": vi["N_stars"],
                "N_det_obs": vi["N_det_obs"],
                "wass_sigma": ctx["wass_sigma"],
                "has_e_circ": (em == "split"),
            }
            logger.info(
                "  variant %-34s logP_cutoff=%.3f N_det_obs=%d N_stars=%d",
                tag, vi["logP_cutoff"], vi["N_det_obs"], vi["N_stars"])

        # Backward-compat single-mode scalars used by logging / the resume
        # validator / the return dict. Sourced from the default variant.
        e_score_mode = variant_meta[default_tag]["e_score_mode"]
        logP_cutoff_mode = variant_meta[default_tag]["logP_cutoff_mode"]
        logP_cutoff = variant_meta[default_tag]["logP_cutoff"]
        N_stars = variant_meta[default_tag]["N_stars"]
        N_det_obs = variant_meta[default_tag]["N_det_obs"]

        # Bundle the extra scoring-context kwargs once so every
        # _save_checkpoint call (4 sites: pre-resume, periodic, parallel
        # final, sequential) records them consistently.
        _ckpt_extra = dict(
            logP_cutoff_smooth_sigma=float(
                self.cfg.get("logP_cutoff_smooth_sigma", 0.15)),
            sb1_tex=sb1_tex,
            sb2_tex=sb2_tex,
            obs=obs,
            n_catalog_total=n_catalog_total,
            n_catalog_nonsingle=n_catalog_nonsingle,
            ostar_catalog=ostar_catalog,
        )

        n_pi = len(pi_grid)
        n_kappa = len(kappa_grid)
        n_eta = len(eta_grid)
        n_fbin = len(fbin_grid)
        total = n_pi * n_kappa * n_eta * n_fbin

        # --- Adaptive injection budget (opt-in) -------------------------
        # When enabled, n_inject is scaled per (π, f_bin) cell so the
        # expected number of injected binary periods above the cutoff is a
        # constant target, compensating for the surviving-sample collapse
        # at steep π under scope="exclude". Disabled → every cell uses the
        # flat baseline (behaviour identical to pre-feature runs). The
        # budget is a function of (i, l) only (κ, η do not affect logP).
        adaptive_n_inject = bool(self.cfg.get("adaptive_n_inject", False))
        if adaptive_n_inject:
            budget_cutoff = self.cfg.get("adaptive_n_cutoff")
            budget_cutoff = (float(logP_cutoff) if budget_cutoff is None
                             else float(budget_cutoff))
            n_above_target = int(self.cfg.get("n_above_cutoff_target", 200))
            n_inject_max = int(self.cfg.get("n_inject_max", 50000))
            log_p_min = float(self.cfg.get("log_p_min", 0.0))
            log_p_max = float(self.cfg.get("log_p_max", 3.5))
            # The injection loops over every star in M1_arr (one task per
            # star), so the pooled above-cutoff count scales with that count
            # — NOT n_stars_sample, which is the catalog total feeding the
            # binomial only.
            n_stars_budget = int(len(self.M1_arr))
            _budget_cache = {}

            def _n_inject_for_cell(i, l):
                key = (i, l)
                cached = _budget_cache.get(key)
                if cached is None:
                    cached = n_inject_for_budget(
                        pi_grid[i], n_stars_budget, fbin_grid[l],
                        n_above_target, log_p_min, log_p_max, budget_cutoff,
                        n_inject_max, n_inject_per_star)
                    _budget_cache[key] = cached
                return cached

            # Precompute the full (i, l) budget table once for logging and so
            # the checkpoint can persist it.
            n_inject_grid = np.array(
                [[_n_inject_for_cell(i, l) for l in range(n_fbin)]
                 for i in range(n_pi)], dtype=np.int64)
            n_capped = int(np.sum(n_inject_grid >= n_inject_max))
            logger.info(
                "Adaptive n_inject: cutoff=%.3f target=%d log_p_min=%.3f → "
                "n_inject min=%d median=%d max=%d (%d/%d (π,f_bin) cells "
                "capped at %d)",
                budget_cutoff, n_above_target, log_p_min,
                int(n_inject_grid.min()), int(np.median(n_inject_grid)),
                int(n_inject_grid.max()), n_capped, n_inject_grid.size,
                n_inject_max)
            _ckpt_extra["adaptive_meta"] = dict(
                adaptive_n_inject=True,
                n_above_cutoff_target=n_above_target,
                n_inject_max=n_inject_max,
                log_p_min=log_p_min,
                log_p_max=log_p_max,
                budget_cutoff=budget_cutoff,
                n_inject_grid=n_inject_grid,
            )
        else:
            n_inject_grid = None
            _ckpt_extra["adaptive_meta"] = dict(adaptive_n_inject=False)

            def _n_inject_for_cell(i, l):
                return n_inject_per_star

        shape = (n_pi, n_kappa, n_eta, n_fbin)
        pdet_cube = np.zeros(shape)        # shared across all variants

        # Per-variant goodness cubes. The expensive pdet_cube + det_shards
        # + global hists are shared; only these cheap goodness cubes
        # multiply by variant. Stored float32 (precision is non-critical
        # for argmax/display) to keep the 6× footprint modest.
        gmf_cubes_v = {}
        test_cubes_v = {}
        for tag in tags:
            has_e_circ = variant_meta[tag]["has_e_circ"]
            gmf_cubes_v[tag] = {
                t: np.full(shape, -np.inf, dtype=np.float32)
                for t in scored_tests
            }
            tcv = {}
            for t in scored_tests:
                d = {
                    "logP": np.zeros(shape, dtype=np.float32),
                    "e": np.zeros(shape, dtype=np.float32),
                    "K1": np.zeros(shape, dtype=np.float32),
                }
                if has_e_circ:
                    d["e_circ"] = np.zeros(shape, dtype=np.float32)
                tcv[t] = d
            test_cubes_v[tag] = tcv

        # Backward-compat aliases (default variant, default metric) for
        # logging / the return dict; downstream consumers read variant-keyed
        # cubes. default_test is "ks" whenever ks is scored, else the first
        # active metric — so these aliases hold real cubes either way.
        gmf_cubes = gmf_cubes_v[default_tag]
        test_cubes = test_cubes_v[default_tag]
        gmf_cube = gmf_cubes[default_test]
        ks_logP_cube = test_cubes[default_test]["logP"]
        ks_e_cube = test_cubes[default_test]["e"]
        ks_K1_cube = test_cubes[default_test]["K1"]

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

                ckpt_gmf_key = "v__%s__gmf_%s_cube" % (default_tag, default_test)
                if (ckpt_gmf_key in ckpt.files
                        and ckpt[ckpt_gmf_key].shape != gmf_cube.shape):
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

                # The all-variants schema (v4) persists the full variant
                # set; the checkpoint metadata is no longer a single
                # scalar mode. Resume requires the SAME variant set and
                # the same (shared) cutoff scope. A single-mode v3
                # checkpoint lacks the 'variants' key → refuse to resume.
                stored_variants = ([str(v) for v in ckpt["variants"]]
                                   if "variants" in ckpt.files else None)
                if stored_variants is None:
                    mismatches.append(
                        "variants: checkpoint predates the all-variants "
                        "schema (no 'variants' key); cannot resume")
                elif set(stored_variants) != set(tags):
                    mismatches.append(
                        "variants: stored=%s current=%s" % (
                            sorted(stored_variants), sorted(tags)))

                stored_logP_scope = (str(ckpt["logP_cutoff_scope"])
                                     if "logP_cutoff_scope" in ckpt.files
                                     else "period_only")
                if stored_logP_scope != logP_cutoff_scope:
                    mismatches.append(
                        "logP_cutoff_scope: stored=%s current=%s" % (
                            stored_logP_scope, logP_cutoff_scope))

                # Adaptive injection budget must match: a different target /
                # cutoff / sampling bound changes n_inject per cell and would
                # silently mix incompatible shards on resume.
                stored_adaptive = (bool(ckpt["adaptive_n_inject"])
                                   if "adaptive_n_inject" in ckpt.files
                                   else False)
                if stored_adaptive != adaptive_n_inject:
                    mismatches.append(
                        "adaptive_n_inject: stored=%s current=%s" % (
                            stored_adaptive, adaptive_n_inject))
                elif adaptive_n_inject:
                    for key, cur, fmt in (
                            ("adaptive_n_above_target", n_above_target, "%d"),
                            ("adaptive_n_inject_max", n_inject_max, "%d"),
                            ("adaptive_cutoff", budget_cutoff, "%.4f"),
                            ("adaptive_log_p_min", log_p_min, "%.4f"),
                            ("adaptive_log_p_max", log_p_max, "%.4f")):
                        if key not in ckpt.files:
                            continue
                        stored_v = ckpt[key]
                        if not np.isclose(float(stored_v), float(cur)):
                            mismatches.append(
                                ("%s: stored=" + fmt + " current=" + fmt) % (
                                    key, float(stored_v), float(cur)))

                if mismatches:
                    raise RuntimeError(
                        "Checkpoint in %s is incompatible with the current "
                        "run:\n  - %s\n"
                        "Refusing to overwrite. Either pass --output-dir "
                        "<new_dir> or delete the checkpoint_* files in "
                        "that directory." % (
                            checkpoint_dir, "\n  - ".join(mismatches)))

                pdet_cube = ckpt["pdet_cube"]
                # Restore namespaced per-variant goodness cubes.
                for tag in tags:
                    has_e_circ = variant_meta[tag]["has_e_circ"]
                    for tname in scored_tests:
                        gmf_key = "v__%s__gmf_%s_cube" % (tag, tname)
                        if gmf_key in ckpt.files:
                            gmf_cubes_v[tag][tname][:] = ckpt[gmf_key]
                        for par in ("logP", "e", "K1"):
                            tc_key = "v__%s__%s_%s_cube" % (tag, tname, par)
                            if tc_key in ckpt.files:
                                test_cubes_v[tag][tname][par][:] = ckpt[tc_key]
                        if has_e_circ:
                            ec_key = "v__%s__%s_e_circ_cube" % (tag, tname)
                            if ec_key in ckpt.files:
                                test_cubes_v[tag][tname]["e_circ"][:] = \
                                    ckpt[ec_key]
                # Refresh default-variant aliases
                gmf_cubes = gmf_cubes_v[default_tag]
                test_cubes = test_cubes_v[default_tag]
                gmf_cube = gmf_cubes[default_test]
                ks_logP_cube = test_cubes[default_test]["logP"]
                ks_e_cube = test_cubes[default_test]["e"]
                ks_K1_cube = test_cubes[default_test]["K1"]
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

        # Per-variant scoring contexts (scoring_ctxs) were built once in the
        # preamble. Both _score_and_accumulate (forward path) and the
        # parallel workers consume them through _compute_scores — one
        # _compute_scores call per (grid point, variant). The simulated
        # population (res) is identical across variants; only the cheap
        # scoring differs.

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

        # Helper: write one grid-point result into cubes / hists / CSV,
        # scoring it under EVERY variant. If `scores_by_variant` is given
        # (parallel paths), reuse the worker-computed scores; otherwise
        # score inline here (one _compute_scores per variant).
        def _score_and_accumulate(step, i, j, k, l, pi, kappa, eta, fbin,
                                  res, scores_by_variant=None):
            p_det = res["p_det"]
            pdet_cube[i, j, k, l] = p_det

            if scores_by_variant is None:
                scores_by_variant = {
                    tag: _compute_scores(res, scoring_ctxs[tag])
                    for tag in tags
                }

            for tag in tags:
                scores = scores_by_variant[tag]
                has_e_circ = variant_meta[tag]["has_e_circ"]
                gc = gmf_cubes_v[tag]
                tc = test_cubes_v[tag]
                for tname in scored_tests:
                    tc[tname]["logP"][i, j, k, l] = scores["%s_p_logP" % tname]
                    tc[tname]["e"][i, j, k, l] = scores["%s_p_e" % tname]
                    tc[tname]["K1"][i, j, k, l] = scores["%s_p_K1" % tname]
                    if has_e_circ:
                        tc[tname]["e_circ"][i, j, k, l] = \
                            scores["%s_p_e_circ" % tname]
                    gc[tname][i, j, k, l] = scores["log_gmf_%s" % tname]

            # Accumulate 2D histograms (fixed-size, negligible memory).
            # Variant-independent — done once.
            for pair in _HIST_PAIRS:
                global_hist_total[pair] += res["hist_total"][pair]
                global_hist_det[pair] += res["hist_det"][pair]

            # Flush per-grid-point detected arrays to a shard file on
            # disk in serial mode. In parallel modes the shard was already
            # written (forward: by _worker_grid_point; resume: by the
            # original run that produced the shard). The shard is
            # variant-independent (raw detected populations).
            if checkpoint_dir and not parallel_grid:
                _save_det_shard(checkpoint_dir, step, i, j, k, l, res)
            step_to_ijkl.append((step, i, j, k, l))

            # Build the CSV row: shared scalars + each variant's log-GMF
            # per test (flat columns log_gmf_<tname>__<tag>). The cubes are
            # authoritative; the CSV is a convenience/sanity table + resume
            # bookkeeping (one row per step keeps m_csv step-gating valid).
            row = {
                "step": step,
                "pi": pi, "kappa": kappa, "eta": eta, "fbin": fbin,
                "p_det": p_det,
                "n_detected": int(res.get("n_detected", 0)),
                "n_physical": int(res.get("n_physical", 0)),
                "n_rlof": int(res.get("n_rlof", 0)),
                "n_false_positive": int(res.get("n_false_positive", 0)),
            }
            for tag in tags:
                for tname in scored_tests:
                    row["log_gmf_%s__%s" % (tname, tag)] = \
                        scores_by_variant[tag]["log_gmf_%s" % tname]

            if not parallel_grid:
                all_results.append(row)
            elif step not in m_csv:
                _pending_csv_rows.append(row)
                m_csv.add(step)

            return p_det, scores_by_variant[default_tag]["log_gmf_%s" % default_test]

        t0 = time.time()
        steps_done = 0
        # Auto-scale checkpoint cadence: ~100 checkpoints per run regardless
        # of grid size. Works for tiny SLURM tasks and 270k-point grids alike.
        checkpoint_every = max(1, (grid_end - grid_start) // 100)

        # --- Parallel grid mode: one grid point per CPU ---
        if parallel_grid and n_workers > 1:
            import multiprocessing as _mp

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
                        initargs=(checkpoint_dir, scoring_ctxs),
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
                            # Per-variant scores already computed by worker.
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
                                res,
                                scores_by_variant=r["scores_by_variant"])
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
                    gmf_cubes_v, pdet_cube, test_cubes_v,
                    pi_grid, kappa_grid, eta_grid, fbin_grid,
                    [],
                    n_inject_per_star=n_inject_per_star,
                    seed=seed,
                    preset_name=preset_name,
                    variant_meta=variant_meta,
                    tags=tags,
                    logP_cutoff_scope=logP_cutoff_scope,
                    apply_lucy_sweeny_e=apply_lucy_sweeny_e,
                    global_hists={
                        "total": global_hist_total,
                        "det": global_hist_det,
                    },
                    **_ckpt_extra,
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
                    seed, _n_inject_for_cell(i, l),
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
                              checkpoint_dir,
                              scoring_ctxs),
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

                        scores_by_variant = res.pop("scores_by_variant", None)
                        p_det, log_gmf = _score_and_accumulate(
                            step, i, j, k, l, pi, kappa, eta, fbin, res,
                            scores_by_variant=scores_by_variant)

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
                                gmf_cubes_v, pdet_cube, test_cubes_v,
                                pi_grid, kappa_grid, eta_grid, fbin_grid,
                                [],
                                n_inject_per_star=n_inject_per_star,
                                seed=seed,
                                preset_name=preset_name,
                                variant_meta=variant_meta,
                                tags=tags,
                                logP_cutoff_scope=logP_cutoff_scope,
                                apply_lucy_sweeny_e=apply_lucy_sweeny_e,
                                global_hists={
                                    "total": global_hist_total,
                                    "det": global_hist_det,
                                },
                                **_ckpt_extra,
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
                    gmf_cubes_v, pdet_cube, test_cubes_v,
                    pi_grid, kappa_grid, eta_grid, fbin_grid,
                    [],  # CSV already flushed above
                    n_inject_per_star=n_inject_per_star,
                    seed=seed,
                    preset_name=preset_name,
                    variant_meta=variant_meta,
                    tags=tags,
                    logP_cutoff_scope=logP_cutoff_scope,
                    apply_lucy_sweeny_e=apply_lucy_sweeny_e,
                    global_hists={
                        "total": global_hist_total,
                        "det": global_hist_det,
                    },
                    **_ckpt_extra,
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
                    _n_inject_for_cell(i, l), rng,
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
                        gmf_cubes_v, pdet_cube, test_cubes_v,
                        pi_grid, kappa_grid, eta_grid, fbin_grid,
                        all_results,
                        n_inject_per_star=n_inject_per_star,
                        seed=seed,
                        preset_name=preset_name,
                        variant_meta=variant_meta,
                        tags=tags,
                        logP_cutoff_scope=logP_cutoff_scope,
                        apply_lucy_sweeny_e=apply_lucy_sweeny_e,
                        global_hists={
                            "total": global_hist_total,
                            "det": global_hist_det,
                        },
                        **_ckpt_extra,
                    )
                    _save_det_index(checkpoint_dir, step_to_ijkl)

        # Best fit per (variant, test) — recomputed from the cubes.
        best_fits_by_variant = {}
        for tag in tags:
            bf_v = {}
            for tname in scored_tests:
                gc = gmf_cubes_v[tag][tname]
                idx = np.unravel_index(np.nanargmax(gc), gc.shape)
                bf_v[tname] = (float(pi_grid[idx[0]]),
                               float(kappa_grid[idx[1]]),
                               float(eta_grid[idx[2]]),
                               float(fbin_grid[idx[3]]))
            best_fits_by_variant[tag] = bf_v

        # Default-variant aliases for logging / backward compat.
        best_fits = best_fits_by_variant[default_tag]
        best_pi, best_kappa, best_eta, best_fbin = best_fits[default_test]
        best_idx = np.unravel_index(
            np.nanargmax(gmf_cubes_v[default_tag][default_test]),
            gmf_cubes_v[default_tag][default_test].shape)

        logger.info("Grid search complete in %.1f min", (time.time()-t0)/60)
        for tag in tags:
            bf = best_fits_by_variant[tag][default_test]
            logger.info("Best fit [%s] (%s): π=%.2f, κ=%.2f, η=%.2f, "
                        "f_bin=%.2f", tag, default_test.upper(), *bf)

        return {
            "pi_grid": pi_grid,
            "kappa_grid": kappa_grid,
            "eta_grid": eta_grid,
            "fbin_grid": fbin_grid,
            "results": all_results,
            # All-variants structures (schema v4)
            "variants": tags,
            "variant_meta": variant_meta,
            "gmf_cubes_by_variant": gmf_cubes_v,
            "test_cubes_by_variant": test_cubes_v,
            "best_fits_by_variant": best_fits_by_variant,
            "pdet_cube": pdet_cube,
            "logP_cutoff_scope": logP_cutoff_scope,
            "step_to_ijkl": step_to_ijkl,
            "checkpoint_dir": checkpoint_dir,
            "global_hists": {
                "total": global_hist_total,
                "det": global_hist_det,
            },
            # Default-variant aliases (backward compat)
            "best_fit": (best_pi, best_kappa, best_eta, best_fbin),
            "best_fits": best_fits,
            "best_idx": best_idx,
            "gmf_cube": gmf_cubes_v[default_tag][default_test],
            "gmf_cubes": gmf_cubes_v[default_tag],
            "test_cubes": test_cubes_v[default_tag],
            "ks_logP_cube": test_cubes_v[default_tag][default_test]["logP"],
            "ks_e_cube": test_cubes_v[default_tag][default_test]["e"],
            "ks_K1_cube": test_cubes_v[default_tag][default_test]["K1"],
            "N_stars": variant_meta[default_tag]["N_stars"],
            "N_det_obs": variant_meta[default_tag]["N_det_obs"],
            "e_score_mode": variant_meta[default_tag]["e_score_mode"],
            "logP_cutoff_mode": variant_meta[default_tag]["logP_cutoff_mode"],
            "logP_cutoff": variant_meta[default_tag]["logP_cutoff"],
            "wass_sigma": variant_meta[default_tag]["wass_sigma"],
        }
