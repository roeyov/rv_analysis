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
from simulations.bias_grid_lib.constants import _HIST_NBINS, _HIST_PAIRS
from simulations.bias_grid_lib.cutoffs import (
    _compute_logP_cutoff, _resolve_e_score_mode,
    _resolve_logP_cutoff_mode, _resolve_logP_cutoff_scope,
)
from simulations.bias_grid_lib.injection import (
    _worker_star_injections, _worker_star_injections_vectorized,
)
from simulations.bias_grid_lib.logging_utils import logger
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
            ostar_catalog=""):
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

        e_score_mode = _resolve_e_score_mode(self.cfg)
        apply_lucy_sweeny_e = bool(self.cfg.get("apply_lucy_sweeny_e", True))

        # Resolve the low-period cutoff before sizing N_det_obs. The cutoff
        # truncates BOTH observed and simulated period arrays to a common
        # window so the KS/AD/CvM tests compare conditional CDFs given
        # P >= 10^cutoff (the regime where the power-law model is valid;
        # below it the obs sample is contaminated by short-period attrition
        # that the model does not describe).
        logP_cutoff_mode = _resolve_logP_cutoff_mode(self.cfg)
        logP_cutoff_scope = _resolve_logP_cutoff_scope(self.cfg)
        logP_cutoff = _compute_logP_cutoff(
            obs_logP, logP_cutoff_mode,
            smooth_sigma=self.cfg.get("logP_cutoff_smooth_sigma", 0.15),
            manual_value=self.cfg.get("logP_cutoff_value"),
        )

        original_N_stars = self.cfg.get(
            "n_stars_sample", len(self.M1_arr))
        n_dropped_obs = (int(np.sum(obs_logP < logP_cutoff))
                         if logP_cutoff > 0.0 else 0)

        if logP_cutoff_scope == "exclude" and logP_cutoff > 0.0:
            # Scope=exclude (catalog-binomial semantics, schema v3):
            #   - obs P/e/K1 CDFs jointly masked by obs_logP >= cutoff
            #     (same as the historical exclude obs-side filter).
            #   - sim injection covers the FULL [log_p_min, log_p_max]
            #     range (no log_p_min override). The sim-side joint mask
            #     against sim_logP >= cutoff is applied later inside the
            #     scoring context.
            #   - binomial uses the O-star catalog counts (total +
            #     non-single) — independent of the period cutoff, so
            #     f_bin reads as the FULL-population binary fraction.
            #   - p_det is the full-sample sim detection rate
            #     (det/inject across the full range), feeding the
            #     binomial against the catalog totals.
            keep = obs_logP >= logP_cutoff
            obs_logP = obs_logP[keep]
            obs_e = obs_e[keep]
            obs_K1 = obs_K1[keep]
            obs_logP_above = obs_logP                       # already filtered
            if n_catalog_total is None or n_catalog_nonsingle is None:
                raise ValueError(
                    "scope=exclude requires n_catalog_total + "
                    "n_catalog_nonsingle (load from ostar_catalog.csv); "
                    "got None")
            N_stars_eff = int(n_catalog_total)
            N_det_obs_override = int(n_catalog_nonsingle)
            logger.info(
                "logP_cutoff_scope=exclude, mode=%s, cutoff=%.3f (P=%.2f d): "
                "obs CDF panels filtered to logP>=cutoff (%d/%d kept); "
                "sim injection covers full [%.3f, %.3f]; sim-det jointly "
                "masked by sim_logP>=cutoff at scoring time; binomial "
                "uses catalog totals (%d non-single / %d total)",
                logP_cutoff_mode, logP_cutoff, 10 ** logP_cutoff,
                len(obs_logP), len(obs_logP) + n_dropped_obs,
                self.cfg.get("log_p_min", 0.0), self.cfg["log_p_max"],
                N_det_obs_override, N_stars_eff)
        elif logP_cutoff > 0.0:
            # period_only (default): cutoff scoped to the logP CDF KS test
            # only. obs e/K1 stay full, binomial stays full, intrinsic
            # draws stay full.
            obs_logP_above = obs_logP[obs_logP >= logP_cutoff]
            N_stars_eff = original_N_stars
            N_det_obs_override = None
            logger.info(
                "logP_cutoff_scope=period_only, mode=%s, cutoff=%.3f "
                "(P=%.2f d); logP CDF test uses %d/%d obs (binomial + "
                "e/K1 tests use all %d)",
                logP_cutoff_mode, logP_cutoff, 10 ** logP_cutoff,
                len(obs_logP_above), len(obs_logP), len(obs_logP))
        else:
            obs_logP_above = obs_logP
            N_stars_eff = original_N_stars
            N_det_obs_override = None
            logger.info(
                "logP_cutoff_mode=%s, scope=%s, cutoff=0.0 (no truncation)",
                logP_cutoff_mode, logP_cutoff_scope)

        N_det_obs = (N_det_obs_override
                     if N_det_obs_override is not None
                     else len(obs_logP))
        N_stars = N_stars_eff

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

        shape = (n_pi, n_kappa, n_eta, n_fbin)
        pdet_cube = np.zeros(shape)

        # Per-test p-value cubes and GMF cubes
        test_cubes = {}
        gmf_cubes = {}
        for tname in _SCORED_TESTS:
            tc = {
                "logP": np.zeros(shape),
                "e": np.zeros(shape),
                "K1": np.zeros(shape),
            }
            if e_score_mode == "split":
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
        # so we compare CDFs only where we have constraining power. The
        # logP lower bound is the inflection cutoff resolved above (0.0
        # when logP_cutoff_mode='none').
        clip_range = {
            "logP": (logP_cutoff, obs_logP.max()),
            "e": (0.0, 1.0),
            "K1": (0.0, obs_K1.max()),
        }
        logger.info("CDF clip ranges: logP=[%.2f,%.2f] e=[%.3f,%.3f] "
                     "K1=[%.1f,%.1f]",
                     *clip_range["logP"], *clip_range["e"],
                     *clip_range["K1"])

        # Pre-split observed eccentricities for the circular/continuous
        # scoring modes (avoids recomputing inside the inner loop).
        # "split" needs all three; "eccentric_only" only needs obs_e_cont.
        obs_e_cont = None
        n_obs_circ = None
        n_obs_e_total = None
        if e_score_mode != "combined":
            obs_e_cont = obs_e[obs_e > 0]
            if e_score_mode == "split":
                n_obs_circ = int(np.sum(obs_e == 0))
                n_obs_e_total = len(obs_e)
                logger.info(
                    "e_score_mode=split: %d circular (e=0) + %d eccentric "
                    "out of %d observed",
                    n_obs_circ, len(obs_e_cont), n_obs_e_total)
            else:
                logger.info(
                    "e_score_mode=eccentric_only: %d eccentric (e>0) of %d "
                    "observed; circular fraction ignored",
                    len(obs_e_cont), len(obs_e))
        else:
            logger.info("e_score_mode=combined: full e distribution tested")

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

                # Recover the stored eccentricity-scoring mode. Checkpoints
                # from before the 3-way switch lack this key — infer it
                # from the legacy boolean if present, else from the
                # presence of *_e_circ_cube entries.
                if "e_score_mode" in ckpt.files:
                    stored_mode = str(ckpt["e_score_mode"])
                elif "split_e_circular" in ckpt.files:
                    stored_mode = ("split"
                                   if bool(ckpt["split_e_circular"])
                                   else "combined")
                else:
                    has_e_circ = any(("%s_e_circ_cube" % t) in ckpt.files
                                     for t in _ALL_TESTS)
                    stored_mode = "split" if has_e_circ else "combined"
                if stored_mode != e_score_mode:
                    mismatches.append(
                        "e_score_mode: stored=%s current=%s" % (
                            stored_mode, e_score_mode))

                # Recover the stored logP cutoff — checkpoints written
                # before this feature lack both keys; treat that as
                # "none / 0.0" so old runs remain resumable.
                stored_logP_mode = (str(ckpt["logP_cutoff_mode"])
                                    if "logP_cutoff_mode" in ckpt.files
                                    else "none")
                stored_logP_cutoff = (float(ckpt["logP_cutoff"])
                                      if "logP_cutoff" in ckpt.files
                                      else 0.0)
                stored_logP_scope = (str(ckpt["logP_cutoff_scope"])
                                     if "logP_cutoff_scope" in ckpt.files
                                     else "period_only")
                if stored_logP_mode != logP_cutoff_mode:
                    mismatches.append(
                        "logP_cutoff_mode: stored=%s current=%s" % (
                            stored_logP_mode, logP_cutoff_mode))
                elif not np.isclose(stored_logP_cutoff, logP_cutoff,
                                    atol=1e-6):
                    mismatches.append(
                        "logP_cutoff: stored=%.6f current=%.6f" % (
                            stored_logP_cutoff, logP_cutoff))
                if stored_logP_scope != logP_cutoff_scope:
                    mismatches.append(
                        "logP_cutoff_scope: stored=%s current=%s" % (
                            stored_logP_scope, logP_cutoff_scope))

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
                for tname in _SCORED_TESTS:
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
                    if e_score_mode == "split":
                        ec_key = "%s_e_circ_cube" % tname
                        if ec_key in ckpt.files:
                            test_cubes[tname]["e_circ"][:] = ckpt[ec_key]
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
        # obs_logP_above is the above-cutoff view; the period KS test
        # consumes it via ctx["obs_logP"]. obs_e and obs_K1 are the full
        # observed arrays so their KS tests and the binomial use the
        # complete 70-system sample.
        # sim_logP_floor: under scope="exclude" we drop the log_p_min
        # injection override (so sim covers the full range) and instead
        # apply the cutoff as a joint mask on sim-detected arrays at
        # CDF-test time. Floor=0 in every other scope.
        sim_logP_floor = (float(logP_cutoff)
                          if logP_cutoff_scope == "exclude"
                             and logP_cutoff > 0.0
                          else 0.0)
        scoring_ctx = _make_scoring_ctx(
            obs_logP=obs_logP_above, obs_e=obs_e, obs_K1=obs_K1,
            clip_range=clip_range,
            e_score_mode=e_score_mode,
            obs_e_cont=obs_e_cont,
            n_obs_circ=n_obs_circ,
            n_obs_e_total=n_obs_e_total,
            N_det_obs=N_det_obs, N_stars=N_stars,
            sim_logP_floor=sim_logP_floor,
        )
        # Now that the per-channel Wasserstein normalization scales are
        # known, propagate them through every _save_checkpoint call so
        # the explorer can surface them and aggregate runs can preserve
        # them across task checkpoints.
        _ckpt_extra["wass_sigma"] = scoring_ctx["wass_sigma"]

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
                scores = _compute_scores(res, scoring_ctx)

            for tname in _SCORED_TESTS:
                test_cubes[tname]["logP"][i, j, k, l] = \
                    scores["%s_p_logP" % tname]
                test_cubes[tname]["e"][i, j, k, l] = \
                    scores["%s_p_e" % tname]
                test_cubes[tname]["K1"][i, j, k, l] = \
                    scores["%s_p_K1" % tname]
                if e_score_mode == "split":
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
                    e_score_mode=e_score_mode,
                    logP_cutoff_mode=logP_cutoff_mode,
                    logP_cutoff=logP_cutoff,
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
                              checkpoint_dir,
                              scoring_ctx),
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

                        scores = res.pop("scores", None)
                        p_det, log_gmf = _score_and_accumulate(
                            step, i, j, k, l, pi, kappa, eta, fbin, res,
                            scores=scores)

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
                                e_score_mode=e_score_mode,
                                logP_cutoff_mode=logP_cutoff_mode,
                                logP_cutoff=logP_cutoff,
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
                    gmf_cubes, pdet_cube, test_cubes,
                    pi_grid, kappa_grid, eta_grid, fbin_grid,
                    [],  # CSV already flushed above
                    n_inject_per_star=n_inject_per_star,
                    seed=seed,
                    preset_name=preset_name,
                    e_score_mode=e_score_mode,
                    logP_cutoff_mode=logP_cutoff_mode,
                    logP_cutoff=logP_cutoff,
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
                        e_score_mode=e_score_mode,
                        logP_cutoff_mode=logP_cutoff_mode,
                        logP_cutoff=logP_cutoff,
                        logP_cutoff_scope=logP_cutoff_scope,
                        apply_lucy_sweeny_e=apply_lucy_sweeny_e,
                        global_hists={
                            "total": global_hist_total,
                            "det": global_hist_det,
                        },
                        **_ckpt_extra,
                    )
                    _save_det_index(checkpoint_dir, step_to_ijkl)

        # Best fit (per test)
        best_fits = {}
        for tname in _SCORED_TESTS:
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
        for tname in _SCORED_TESTS:
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
            "N_stars": N_stars,
            "N_det_obs": len(obs_logP),
            "step_to_ijkl": step_to_ijkl,
            "checkpoint_dir": checkpoint_dir,
            "e_score_mode": e_score_mode,
            "logP_cutoff_mode": logP_cutoff_mode,
            "logP_cutoff": logP_cutoff,
            "logP_cutoff_scope": logP_cutoff_scope,
            "wass_sigma": scoring_ctx["wass_sigma"],
            "global_hists": {
                "total": global_hist_total,
                "det": global_hist_det,
            },
        }
