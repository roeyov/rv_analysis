"""
sb2_pipeline.batch — SB2 fixed-RVs batch orchestrator.

Usage:
    python -m sb2_pipeline.batch --config configs/params_sb2_fixed_rvs.yaml
    python -m sb2_pipeline.batch --config ... --corrections-only

Per star (serial by design — no Pool; kaleido runs only in the periodogram
stage, in the main process, where it has been stable historically):
  corrections -> |rv1-rv2| periodogram (LS+PDC, analytic FAP) -> joint
  9-parameter lmfit over candidates + two-constant null -> best BICc row ->
  eccentric MCMC -> Lucy-Sweeney -> circular MCMC fallback.

Resume: a star is complete only when lmfit_summary.csv AND mcmc/star_done.json
(written after the whole MCMC stage, incl. the circular fallback) exist; with
only the summary present, the MCMC stage reruns. The cross-star CSVs are
merged with any previous run's rows, never clobbered.

Outputs under sb2_io.output_dir (never inside the archived data tree):
  sb2_corrections_audit.csv, sb2_lmfit_log.csv, sb2_mcmc_params.csv,
  choose_param.json, and per star: data/, periodogram/, lmfit_solutions/,
  lmfit_summary.csv, mcmc/[circular/].
"""

import matplotlib
matplotlib.use("Agg")  # noqa: E402  — before any pyplot import (headless batch)

import argparse
import copy
import json
import os
import traceback

import numpy as np
import pandas as pd

from utils.constants import (
    PLOT, MANUAL_CANDIDATES, PERIODOGRAM_PARAMS,
    PERI_MIN_PERIOD, PERI_MAX_PERIOD,
    LMFIT_PARAMS, MCMC_PARAMS, GAMMA,
    PERIOD, K1_STR, K2_STR, OMEGA, ECC, T,
    LN_SIGMA_JITTER, LN_SIGMA_JITTER_1, LN_SIGMA_JITTER_2,
)
from pipeline.config import load_args, change_search_region_default, \
    _inject_manual_candidates
from pipeline.evaluator import _ensure_star_out_dir
from period_search.candidates import find_periods
from orbital.statistics import calculate_binary_probability
from mcmc.selector_app import get_best_row
from mcmc.batch import save_choose_params_file

from sb2_pipeline.constants import (
    SB2_IO, SB2_ROOT, PROVENANCE_CSV, CORRECTIONS_CSV, OUTPUT_DIR,
    OBJECT_LIST, MJD_MATCH_TOLERANCE, RV_FILE_PRIORITY, USE_FWHM,
    FILTER_EXPRESSION, FIELD_TO_CHECK, TAKE_MIN,
    SB2_RV1, SB2_ERR1, SB2_RV2, SB2_ERR2,
)
from sb2_pipeline.io import (
    STAR_RE, CorrectionMatchError,
    load_provenance, load_corrections, load_sb2_star, apply_corrections,
    write_corrected_csv, make_data_dict, paired_drv,
)
from sb2_pipeline.fitting import fit_sb2_candidates, fit_sb2_null
from sb2_pipeline.statistics import apply_k_swap_to_data
from sb2_pipeline.mcmc_run import run_mcmc_sb2_star


_PREFERRED_ORDER = [
    "star_name", "candidate_method", "candidate_period", "solution_id",
    "is_best", "LS_power", "LS_fap", "PDC_power", "PDC_fap",
    "fwhm_per_low", "fwhm_per_high",
    "method", "nfev", "ndata", "nvarys", "chisqr", "redchi",
    "llh", "aic", "bic", "bicc", "ev_bic", "prob_bicc", "k_swapped", "phs",
]

_BEST_PARAM_COLS = [PERIOD, T, ECC, OMEGA, GAMMA, K1_STR, K2_STR,
                    LN_SIGMA_JITTER_1, LN_SIGMA_JITTER_2]


# ---------------------------------------------------------------------------
# Star list
# ---------------------------------------------------------------------------

def build_star_list(cfg, provenance):
    """Provenance star_ids ∪ data-root dirs with sb2_analysis runs."""
    sb2_root = cfg[SB2_IO][SB2_ROOT]
    stars = set(provenance.index)
    for d in os.listdir(sb2_root):
        if STAR_RE.match(d) and os.path.isdir(
                os.path.join(sb2_root, d, "sb2_analysis")):
            stars.add(d)

    object_list = cfg[SB2_IO].get(OBJECT_LIST) or []
    if object_list:
        stars = {s for s in stars
                 if any(pat in s for pat in object_list)}
    return sorted(stars)


# ---------------------------------------------------------------------------
# lmfit summary assembly
# ---------------------------------------------------------------------------

def _write_lmfit_summary(rows, out_csv):
    df = pd.DataFrame(rows)
    ordered = [c for c in _PREFERRED_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    df = df[ordered + rest]
    df.to_csv(out_csv, index=False)
    return df


def _mark_best(df, best, out_csv):
    df["is_best"] = False
    df.loc[df.index == best.name, "is_best"] = True
    df.to_csv(out_csv, index=False)


# ---------------------------------------------------------------------------
# Per-star processing
# ---------------------------------------------------------------------------

def process_star(star_id, cfg, corr_df, provenance, corrections_only=False,
                 prev_log_row=None):
    """
    Full per-star flow. Never raises. Returns (log_row, mcmc_row_or_None,
    audit_rows). prev_log_row (dict or None) supplies periodogram fields on
    the MCMC-only resume path, where find_periods is not rerun.
    """
    sb2_cfg = cfg[SB2_IO]
    out_root = sb2_cfg[OUTPUT_DIR]
    use_fwhm = bool(sb2_cfg.get(USE_FWHM, True))

    log = {
        "star_name": star_id, "status": "ok",
        "adopted_timestamp": "", "timestamp_is_latest": np.nan,
        "rv_source_file": "", "n_epochs_raw": 0, "n_epochs_used": 0,
        "n_dropped_nonfinite": 0, "has_rv3": False,
        "mean_rv_correction": np.nan,
        "ls_best_period": np.nan, "ls_fap": np.nan,
        "pdc_best_period": np.nan, "pdc_fap": np.nan,
        "n_candidates": 0, "best_solution_id": -1,
        "best_candidate_method": "", "best_period": np.nan,
        "best_K1": np.nan, "best_K2": np.nan, "best_gamma": np.nan,
        "best_ecc": np.nan, "best_omega": np.nan,
        "best_ln_sigmaJ1": np.nan, "best_ln_sigmaJ2": np.nan,
        "best_bicc": np.nan, "null_bicc": np.nan, "prob_bicc": np.nan,
        "p_bicc_kep_vs_circ": np.nan, "k_swapped": False, "notes": "",
    }
    mcmc_row = None
    audit_rows = []

    # ---- load + correct ----
    try:
        star = load_sb2_star(sb2_cfg[SB2_ROOT], star_id, provenance,
                             sb2_cfg.get(RV_FILE_PRIORITY))
    except FileNotFoundError as e:
        log["status"] = "no_data"
        log["notes"] = str(e)
        return log, None, []
    except Exception as e:
        log["status"] = "load_failed"
        log["notes"] = str(e)
        return log, None, []

    log.update({
        "adopted_timestamp": star.timestamp,
        "timestamp_is_latest": star.timestamp_is_latest,
        "rv_source_file": star.rv_source_file,
        "n_epochs_raw": star.n_raw,
        "has_rv3": star.has_rv3,
    })
    has_adopted = (star_id in provenance.index
                   and isinstance(provenance.loc[star_id, "adopted_timestamp"], str)
                   and provenance.loc[star_id, "adopted_timestamp"].strip() != "")
    if not has_adopted:
        log["status"] = "no_adopted_solution_ok"
    if star.has_rv3:
        log["notes"] = "quadruple 2+2; rv3 corrected, not fitted. " + log["notes"]

    try:
        apply_corrections(star, corr_df,
                          tolerance=float(sb2_cfg.get(MJD_MATCH_TOLERANCE, 5e-3)))
    except CorrectionMatchError as e:
        log["status"] = "correction_match_failed"
        log["notes"] = str(e)
        return log, None, []
    audit_rows = star.audit
    log["n_epochs_used"] = len(star.df)
    log["n_dropped_nonfinite"] = star.n_dropped_nonfinite
    log["mean_rv_correction"] = float(
        np.mean([a["rv_correction"] for a in audit_rows]))

    star_out = _ensure_star_out_dir(out_root, star_id, plot=True)
    write_corrected_csv(star, os.path.join(star_out, "data",
                                           "rv_sb2_corrected.csv"))
    if corrections_only:
        log["status"] = "corrections_only"
        return log, None, audit_rows

    # Cleaned per-epoch table (sb2_pipeline.cleaning): swapped values and
    # use1/use2 masks, in the as-loaded component labelling. When present it
    # supersedes the plain corrected table for everything downstream.
    cleaned_csv = os.path.join(star_out, "data", "rv_sb2_cleaned.csv")
    df_fit = star.df
    if os.path.exists(cleaned_csv):
        df_fit = pd.read_csv(cleaned_csv, comment="#")
        n_sw = (int(df_fit["swap_applied"].sum())
                if "swap_applied" in df_fit.columns else 0)
        n_m1 = (int((~df_fit["use1"].astype(bool)).sum())
                if "use1" in df_fit.columns else 0)
        n_m2 = (int((~df_fit["use2"].astype(bool)).sum())
                if "use2" in df_fit.columns else 0)
        log["notes"] = (f"cleaned: {n_sw} swapped, {n_m1}+{n_m2} masked. "
                        + log["notes"])
        print(f"  {star_id}: using cleaned RVs "
              f"({n_sw} swapped, {n_m1}/{n_m2} masked)")

    data = make_data_dict(df_fit)
    summary_csv = os.path.join(star_out, "lmfit_summary.csv")
    # Completion sentinel: written as the LAST artifact of the MCMC stage
    # (after the Lucy-Sweeney gate and any circular fallback), so an
    # interrupted star is never mistaken for a finished one.
    done_sentinel = os.path.join(star_out, "mcmc", "star_done.json")

    mp = cfg[MCMC_PARAMS]
    filter_expr = mp.get(FILTER_EXPRESSION, None)
    field_to_check = mp.get(FIELD_TO_CHECK, "bicc")
    take_min = mp.get(TAKE_MIN, True)

    # ---- resume logic ----
    if os.path.exists(summary_csv) and os.path.exists(done_sentinel):
        log["status"] = "skipped_existing"
        return log, None, audit_rows

    if os.path.exists(summary_csv):
        print(f"  {star_id}: lmfit_summary.csv exists -> MCMC only")
        df = pd.read_csv(summary_csv)
        if prev_log_row:
            for key in ("ls_best_period", "ls_fap", "pdc_best_period",
                        "pdc_fap", "n_candidates"):
                if key in prev_log_row and pd.notna(prev_log_row[key]):
                    log[key] = prev_log_row[key]
    else:
        # ---- periodogram on |rv1 - rv2| (epochs where both are used) ----
        drv_mjd, drv, derr = paired_drv(df_fit)
        try:
            (ls_p, ls_fap, _ls_fal, _ls_mp, pdc_p, pdc_fap, _pdc_mp,
             candidates_df) = find_periods(
                drv, drv_mjd, derr, cfg, star_id,
                out_dir=os.path.join(star_out, "periodogram"),
                use_fwhm=use_fwhm)
        except Exception as e:
            log["status"] = "fit_failed"
            log["notes"] = f"periodogram failed: {e}"
            return log, None, audit_rows
        log.update({"ls_best_period": float(ls_p), "ls_fap": float(ls_fap),
                    "pdc_best_period": float(pdc_p),
                    "pdc_fap": float(pdc_fap)})

        peri = cfg[PERIODOGRAM_PARAMS]
        # full_period_range=False anchors each manual candidate to a +-10%
        # window (the default full-range bounds let DE wander to an
        # arbitrary period, defeating the point of a targeted candidate).
        candidates_df = _inject_manual_candidates(
            candidates_df, cfg.get(MANUAL_CANDIDATES, []) or [],
            pmin=peri[PERI_MIN_PERIOD], pmax=peri[PERI_MAX_PERIOD],
            full_period_range=False)
        log["n_candidates"] = len(candidates_df)

        # ---- joint lmfit over candidates + null ----
        local_cfg = copy.deepcopy(cfg)
        all_rv = np.concatenate([data[SB2_RV1], data[SB2_RV2]])
        all_err = np.concatenate([data[SB2_ERR1], data[SB2_ERR2]])
        w = 1.0 / all_err ** 2
        gamma0 = float(np.sum(all_rv * w) / np.sum(w))
        change_search_region_default(
            local_cfg, GAMMA, gamma0,
            float(np.min(all_rv)) - 20.0, float(np.max(all_rv)) + 20.0, True)

        try:
            rows = fit_sb2_candidates(local_cfg, data, candidates_df, star_id,
                                      star_out, use_fwhm=use_fwhm,
                                      plot=cfg.get(PLOT, True))
            null_row = fit_sb2_null(local_cfg, data, star_id,
                                    solution_id=len(rows))
        except Exception as e:
            traceback.print_exc()
            log["status"] = "fit_failed"
            log["notes"] = f"lmfit failed: {e}"
            return log, None, audit_rows

        if not rows:
            log["status"] = "fit_failed"
            log["notes"] = "no candidate fit converged"
            return log, None, audit_rows

        for row in rows:
            row["prob_bicc"] = calculate_binary_probability(
                row["bicc"], null_row["bicc"])
        null_row["prob_bicc"] = np.nan
        df = _write_lmfit_summary(rows + [null_row], summary_csv)

    # ---- best-row selection ----
    null_sel = df[df["candidate_method"].str.contains("null_hyp_jitter",
                                                      na=False)]
    if not null_sel.empty:
        log["null_bicc"] = float(null_sel.iloc[0]["bicc"])

    best = get_best_row(df, filter_expr, field_to_check, take_min)
    if best is None:
        log["status"] = "no_acceptable_row"
        return log, None, audit_rows
    _mark_best(df, best, summary_csv)

    log.update({
        "best_solution_id": int(best.get("solution_id", -1)),
        "best_candidate_method": str(best.get("candidate_method", "")),
        "best_period": float(best.get(f"{PERIOD}_value", np.nan)),
        "best_K1": float(best.get(f"{K1_STR}_value", np.nan)),
        "best_K2": float(best.get(f"{K2_STR}_value", np.nan)),
        "best_gamma": float(best.get(f"{GAMMA}_value", np.nan)),
        "best_ecc": float(best.get(f"{ECC}_value", np.nan)),
        "best_omega": float(best.get(f"{OMEGA}_value", np.nan)),
        "best_ln_sigmaJ1": float(best.get(f"{LN_SIGMA_JITTER_1}_value", np.nan)),
        "best_ln_sigmaJ2": float(best.get(f"{LN_SIGMA_JITTER_2}_value", np.nan)),
        "best_ln_sigmaJ": float(best.get(f"{LN_SIGMA_JITTER}_value", np.nan)),
        "best_bicc": float(best.get("bicc", np.nan)),
        "prob_bicc": float(best.get("prob_bicc", np.nan)),
        "p_bicc_kep_vs_circ": float(best.get("p_bicc_kep_vs_circ", np.nan)),
        "k_swapped": bool(best.get("k_swapped", False)),
    })

    # ---- data for MCMC (K1<K2 canonical labelling) ----
    mcmc_df = df_fit
    if bool(best.get("k_swapped", False)):
        mcmc_df = apply_k_swap_to_data(df_fit)
    swapped_star = copy.copy(star)
    swapped_star.df = mcmc_df
    write_corrected_csv(swapped_star, os.path.join(star_out, "data",
                                                   "rv_sb2_for_mcmc.csv"))
    data_mcmc = make_data_dict(mcmc_df)

    # ---- MCMC ----
    try:
        mcmc_row = run_mcmc_sb2_star(
            cfg, best, data_mcmc, os.path.join(star_out, "mcmc"), star_id)
        mcmc_row["star_name"] = star_id
        mcmc_row["best_solution_id"] = log["best_solution_id"]
        mcmc_row["k_swapped"] = log["k_swapped"]
        with open(done_sentinel, "w") as f:
            json.dump({"star_name": star_id,
                       "adopted_model": mcmc_row.get("adopted_model", ""),
                       "best_solution_id": log["best_solution_id"]}, f,
                      indent=2)
            f.write("\n")
    except Exception as e:
        traceback.print_exc()
        log["status"] = "mcmc_failed"
        log["notes"] = f"MCMC failed: {e}. " + log["notes"]

    return log, mcmc_row, audit_rows


# ---------------------------------------------------------------------------
# Cross-star CSV helpers (shard-aware)
# ---------------------------------------------------------------------------

_CROSS_STAR_BASES = ("sb2_lmfit_log", "sb2_corrections_audit",
                     "sb2_mcmc_params")


def _rows_by_star(path):
    """CSV -> {star_name: [row dicts]} ({} when the file is absent)."""
    grouped = {}
    if os.path.exists(path):
        for rec in pd.read_csv(path).to_dict("records"):
            grouped.setdefault(rec["star_name"], []).append(rec)
    return grouped


def aggregate_cross_star(out_root):
    """Merge shard CSVs into the canonical cross-star CSVs (shard rows win)."""
    import glob
    for base in _CROSS_STAR_BASES:
        canonical = os.path.join(out_root, f"{base}.csv")
        merged = _rows_by_star(canonical)
        shard_files = sorted(glob.glob(
            os.path.join(out_root, f"{base}.shard*.csv")))
        for sf in shard_files:
            merged.update(_rows_by_star(sf))
        rows = [r for star in sorted(merged) for r in merged[star]]
        if rows:
            pd.DataFrame(rows).to_csv(canonical, index=False)
        print(f"aggregated {base}.csv: {len(merged)} stars, {len(rows)} rows "
              f"({len(shard_files)} shard files merged)")
        for sf in shard_files:
            os.remove(sf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SB2 fixed-RVs batch: corrections + |dRV| periodogram + "
                    "joint Keplerian lmfit + MCMC + Lucy-Sweeney.")
    parser.add_argument("--config", required=True,
                        help="Path to the SB2 params YAML.")
    parser.add_argument("--corrections-only", action="store_true",
                        help="Only apply/audit the RV corrections, then stop.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total parallel shard processes (each launched "
                             "separately; per-star seeds keep results "
                             "identical to a serial run).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="This process's shard index in [0, num_shards).")
    parser.add_argument("--aggregate", action="store_true",
                        help="Merge shard cross-star CSVs into the canonical "
                             "ones and exit (run once after all shards).")
    cli = parser.parse_args()

    cfg = load_args(cli.config)
    sb2_cfg = cfg[SB2_IO]
    out_root = sb2_cfg[OUTPUT_DIR]
    os.makedirs(out_root, exist_ok=True)

    if cli.aggregate:
        aggregate_cross_star(out_root)
        return

    provenance = load_provenance(sb2_cfg[PROVENANCE_CSV])
    corr_df = load_corrections(sb2_cfg[CORRECTIONS_CSV])
    stars = build_star_list(cfg, provenance)
    if cli.num_shards > 1:
        if not (0 <= cli.shard_index < cli.num_shards):
            raise SystemExit("--shard-index must be in [0, num_shards)")
        stars = stars[cli.shard_index::cli.num_shards]

    mp = cfg[MCMC_PARAMS]
    print("=" * 50)
    print("  SB2 fixed-RVs batch")
    print(f"  Config:        {cli.config}")
    print(f"  SB2 data root: {sb2_cfg[SB2_ROOT]}")
    print(f"  Corrections:   {sb2_cfg[CORRECTIONS_CSV]}")
    print(f"  Output dir:    {out_root}")
    if cli.num_shards > 1:
        print(f"  Shard:         {cli.shard_index + 1}/{cli.num_shards}")
    print(f"  Stars:         {len(stars)}")
    print(f"  Mode:          "
          f"{'corrections-only' if cli.corrections_only else 'full'}")
    print("=" * 50)

    save_choose_params_file(out_root, {
        "filter_expression": mp.get(FILTER_EXPRESSION),
        "field_to_check": mp.get(FIELD_TO_CHECK, "bicc"),
        "take_min": mp.get(TAKE_MIN, True),
        "lucy_sweeney_threshold": mp.get("lucy_sweeney_threshold", 2.49),
        "lucy_sweeney_estimator": mp.get("lucy_sweeney_estimator", "median"),
    })

    # Previous runs' cross-star rows, keyed by star, so a resumed run merges
    # instead of clobbering (skipped stars keep their earlier full rows).
    # Sharded processes write to shard-suffixed files holding ONLY their own
    # stars (canonical rows are still consulted for resume); a final
    # --aggregate pass folds the shard files back into the canonical CSVs.
    suffix = f".shard{cli.shard_index}" if cli.num_shards > 1 else ""
    log_csv = os.path.join(out_root, f"sb2_lmfit_log{suffix}.csv")
    audit_csv = os.path.join(out_root, f"sb2_corrections_audit{suffix}.csv")
    mcmc_csv = os.path.join(out_root, f"sb2_mcmc_params{suffix}.csv")

    log_map = _rows_by_star(os.path.join(out_root, "sb2_lmfit_log.csv"))
    audit_map = _rows_by_star(
        os.path.join(out_root, "sb2_corrections_audit.csv"))
    mcmc_map = _rows_by_star(os.path.join(out_root, "sb2_mcmc_params.csv"))
    if suffix:
        log_map.update(_rows_by_star(log_csv))
        audit_map.update(_rows_by_star(audit_csv))
        mcmc_map.update(_rows_by_star(mcmc_csv))
        shard_set = set(stars)
        log_map = {k: v for k, v in log_map.items() if k in shard_set}
        audit_map = {k: v for k, v in audit_map.items() if k in shard_set}
        mcmc_map = {k: v for k, v in mcmc_map.items() if k in shard_set}

    def _flush():
        def _dump(mapping, path):
            rows = [r for star in sorted(mapping) for r in mapping[star]]
            if rows:
                pd.DataFrame(rows).to_csv(path, index=False)
        _dump(log_map, log_csv)
        _dump(audit_map, audit_csv)
        _dump(mcmc_map, mcmc_csv)

    statuses = []
    for i, star_id in enumerate(stars, 1):
        print(f"\n=== [{i}/{len(stars)}] {star_id} ===")
        prev_log = (log_map.get(star_id) or [None])[0]
        try:
            log, mcmc_row, audit = process_star(
                star_id, cfg, corr_df, provenance,
                corrections_only=cli.corrections_only,
                prev_log_row=prev_log)
        except Exception as e:  # belt and braces: process_star shouldn't raise
            traceback.print_exc()
            log = {"star_name": star_id, "status": "crashed", "notes": str(e)}
            mcmc_row, audit = None, []

        if audit:  # fresh corrections audit is always authoritative
            audit_map[star_id] = audit
        keep_prev = (prev_log is not None
                     and log["status"] in ("skipped_existing",
                                           "corrections_only"))
        if not keep_prev:
            log_map[star_id] = [log]
            if mcmc_row is not None:
                mcmc_map[star_id] = [mcmc_row]
            elif log["status"] in ("no_acceptable_row", "fit_failed",
                                   "mcmc_failed", "correction_match_failed"):
                # A fresh (re)fit superseded whatever MCMC row existed before.
                mcmc_map.pop(star_id, None)

        _flush()
        statuses.append(log["status"])
        print(f"  [{star_id}] status: {log['status']}")

    print("\n=== SB2 batch completed ===")
    print(pd.Series(statuses).value_counts().to_string())
    if cli.num_shards > 1:
        print(f"(shard {cli.shard_index + 1}/{cli.num_shards} done — run "
              f"--aggregate once all shards finish)")


if __name__ == "__main__":
    main()
