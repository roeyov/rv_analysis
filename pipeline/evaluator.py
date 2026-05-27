"""
pipeline.evaluator — Main orchestration for the binary-star detection pipeline.

Entry points:
    main_single()   Process a single star CSV
    main_multiple()  Batch-process a directory of CSVs
    manual_append_dispatcher()  Append manual candidate solutions
"""

import os
import glob
import copy
import multiprocessing

import numpy as np
import pandas as pd
from types import SimpleNamespace
from lmfit.minimizer import AbortFitException

from utils.constants import *  # includes SNR_PPL

from orbital.plotting import print_lmfit_result, print_lmfit_result_null
from orbital.fitting import lmfit_on_sample, chisqr_with_jitter, get_rv_weighted_mean
from orbital.statistics import (
    calculate_statistical_flags, summarize_result,
    calculate_phase_criterias, coverage_qc,
    calculate_binary_probability, lr_test_prob,
    f_test, rv_significance, binary_rv_threshold,
    DecsionFlags,
)
from utils.roche_lobe import compute_min_period_row
from period_search.candidates import find_periods

from pipeline.data_loading import (
    load_final_data_from_ccf_out, get_bloem_object_name,
    _load_and_clean_csv, _rename_to_internal_cols,
    _prepare_inputs_for_period_search,
)
from pipeline.config import (
    PARAM_FILE, JSON_PARAM_FILE, load_args, _load_json_args,
    change_search_region_default, _inject_manual_candidates,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_star_out_dir(path_to_out, star_name, plot=True):
    """
    Create per-star output dir if requested and return its path.

    New behaviour is handled in main_single:
      * If lmfit_summary.csv does NOT exist  -> run full pipeline.
      * If it exists but MCMC output is missing -> reload lmfit_summary and run only MCMC.
      * If both lmfit_summary and MCMC outputs exist -> everything is skipped.
    """
    if not path_to_out:
        return None

    star_out_path = os.path.join(path_to_out, star_name)
    os.makedirs(star_out_path, exist_ok=True)
    if plot:
        for sub_dir in ["periodogram", "lmfit_solutions", "mcmc"]:
            os.makedirs(os.path.join(star_out_path, sub_dir), exist_ok=True)

    return star_out_path


def _find_candidates(rvs, mjds, err_vs, args_dict, star_name, out_dir, use_fwhm):
    """
    Wrapper around find_periods. Returns:
      (ls_p, ls_fap, ls_fal, ls_mp, pdc_p, pdc_fap, pdc_mp, candidates_df)
    """
    return find_periods(rvs, mjds, err_vs, args_dict, star_name, out_dir=out_dir, use_fwhm=use_fwhm)


def debug_compare(mini_results, data):
    # 1. Get the value lmfit stored
    lmfit_val = mini_results.residual

    # 2. Re-run YOUR minimizer function exactly as lmfit did
    #    (Passing the exact same params and keywords)
    import collections
    # Ensure params are the generic lmfit Parameters object
    p = mini_results.params

    # Reconstruct kws exactly as passed to minimize
    kws = {
        TIME_STAMPS: data[TIME_STAMPS],
        RADIAL_VELS: data[RADIAL_VELS],
        ERRORS: data[ERRORS]
    }

    # Call the exact function used in minimization
    recalc_val = chisqr_with_jitter(p, **kws)

    # 3. Call your new manual function
    manual_stats = calculate_statistical_flags(data, mini_results, is_null=False)
    manual_val = manual_stats['llh']

    print(f"1. LMFIT Stored Result: {lmfit_val}")
    print(f"2. Recalculated (Orig): {recalc_val}")
    print(f"3. Manual 'llh':        {manual_val}")

    diff = abs(lmfit_val - manual_val)
    print(f"Difference: {diff}")

    if diff > 1e-5:
        print(">> DISCREPANCY DETECTED. Check _safe_errors or Parameter Keys.")


# ---------------------------------------------------------------------------
# Candidate fitting (single worker + serial/parallel wrappers)
# ---------------------------------------------------------------------------

def _fit_single_candidate_worker(packed_args):
    """
    Worker function to process a single candidate in a separate process.
    Unpacks arguments, performs a deep copy of mutable resources, and runs the fit.
    """
    (cand, args_dict_template, data, star_name, sid, out_dir,
     use_fwhm, ls_fap, median_snr_obs, median_snr_calc) = packed_args

    # 1. Deep copy shared resources to ensure process isolation
    # The original code modifies args_dict in place via change_search_region_default
    local_args_dict = copy.deepcopy(args_dict_template)

    chosen_period = float(cand.period)

    # 2. Setup Search Regions (Logic from original loop)
    if not use_fwhm:
        change_search_region_default(
            local_args_dict, PERIOD, chosen_period,
            chosen_period * 0.9, chosen_period * 1.1, False
        )
    else:
        change_search_region_default(
            local_args_dict, PERIOD, chosen_period,
            float(cand.fwhm_per_low), float(cand.fwhm_per_high), True
        )

    change_search_region_default(
        local_args_dict, T, min(data[TIME_STAMPS]),
        min(data[TIME_STAMPS]) - chosen_period, min(data[TIME_STAMPS]) + chosen_period, True
    )

    # 3. Run Fitting
    try:
        mini_results = lmfit_on_sample(local_args_dict, data, use_jitter=cand.jitter)
    except AbortFitException:
        return None  # Signal failure/skip
    except Exception as e:
        print(f"Error in process for period {chosen_period}: {e}")
        return None

    # 4. Generate Outputs and Summaries
    # Note: print_lmfit_result generates plots. In parallel, ensure your backend
    # is thread-safe or headless (e.g., Agg) to avoid GUI conflicts.
    # If out_dir is None, print_lmfit_result will skip saving plots/reports
    phs_path = print_lmfit_result(data, local_args_dict, star_name, mini_results, sid, out_dir=out_dir)

    row = summarize_result(mini_results, star_name)
    row.update({
        "candidate_method": getattr(cand, "method", None) + ("_jitter" if cand.jitter else ""),
        "candidate_period": chosen_period,
        "LS_power": float(getattr(cand, "LS_power", np.nan)),
        "LS_fap": float(getattr(cand, "LS_fap", np.nan)),
        "PDC_power": float(getattr(cand, "PDC_power", np.nan)),
        "PDC_fap": float(getattr(cand, "PDC_fap", np.nan)),
        "PDC_iter_fap": float(getattr(cand, "PDC_iter_fap", np.nan)),
        "LS_iter_fap": float(getattr(cand, "LS_iter_fap", np.nan)),
        "phs": phs_path,
        "p_val": np.nan, "prob_full": np.nan,
        "prob_aic": np.nan, "prob_bic": np.nan,
        "is_best": False,
        SNR_PPL: median_snr_obs,
        "SNR": median_snr_calc,
        "solution_id": sid,
        "temp_lmfit": mini_results,
        RADIAL_VELS: data[RADIAL_VELS].values,
        ERRORS: data[ERRORS].values,
        MJD: data[TIME_STAMPS].values,
    })

    if cand.jitter:
        row.update(calculate_statistical_flags(data, mini_results, is_null=False))

    # 5. Quality Control Metrics
    phase_cv = calculate_phase_criterias(data, mini_results)
    row.update(phase_cv)
    row.update(coverage_qc(phase_cv, data[RADIAL_VELS]))

    return (row, mini_results, cand)


def _fit_candidates_over_periods_parallel(candidates_df, args_dict, data, star_name,
                                          init_sid_counter, out_dir, use_fwhm, ls_fap,
                                          median_snr_obs, median_snr_calc, n_cores=None):
    """
    Parallel version of _fit_candidates_over_periods.
    Spawns a process for each candidate in candidates_df.
    """
    rows = []
    best_result, best_cand = None, None
    cur_red_chi = 1e9

    if candidates_df is None or candidates_df.empty:
        return rows, best_result, best_cand

    if np.isnan(ls_fap):
        return rows, best_result, best_cand

    # 1. Prepare Task List
    tasks = []
    sid_counter = init_sid_counter

    # Iterate over the dataframe
    for cand_row in candidates_df.itertuples(index=False):
        # FIX: Convert the Pandas named tuple to a standard dict, then SimpleNamespace.
        # This removes the un-pickleable "pandas.core.frame.Pandas" type.
        cand_safe = SimpleNamespace(**cand_row._asdict())

        task_args = (
            cand_safe,
            args_dict,  # Passed as template, worker will deepcopy
            data,
            star_name,
            sid_counter,
            out_dir,
            use_fwhm,
            ls_fap,
            median_snr_obs,
            median_snr_calc
        )
        tasks.append(task_args)
        sid_counter += 1

    # 2. Run Parallel Processing
    print(f"Starting parallel fit for {len(tasks)} candidates on {n_cores if n_cores else 'all'} cores...")

    # Use 'spawn' or 'fork' context safely if needed, but default Pool usually works with standard types
    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(_fit_single_candidate_worker, tasks)

    # 3. Aggregate Results
    for res in results:
        if res is None:
            continue

        row, mini_results, cand = res
        rows.append(row)

        if abs(mini_results.redchi - 1) < abs(cur_red_chi - 1):
            cur_red_chi = mini_results.redchi
            best_result = mini_results
            best_cand = cand

    return rows, best_result, best_cand

def _fit_candidates_over_periods(candidates_df, args_dict, data, star_name,init_sid_counter, out_dir,
                                 use_fwhm, ls_fap,
                                 median_snr_obs, median_snr_calc):
    """
    Iterate over candidate periods, run lmfit, summarize each result, compute phase coverage,
    and track best fit by proximity of redchi to 1. Returns (rows, best_result, best_cand).
    """
    rows = []
    best_result, best_cand = None, None
    cur_red_chi = 1e9

    if candidates_df is None or candidates_df.empty:
        return rows, best_result, best_cand
    sid_counter = init_sid_counter
    for cand in candidates_df.itertuples(index=False):
        chosen_period = float(cand.period)

        # early guard as in your code
        if np.isnan(ls_fap):
            break

        # set search region for PERIOD
        if not use_fwhm:
            change_search_region_default(
                args_dict, PERIOD, chosen_period,
                chosen_period * 0.9, chosen_period * 1.1, False
            )
        else:
            change_search_region_default(
                args_dict, PERIOD, chosen_period,
                float(cand.fwhm_per_low), float(cand.fwhm_per_high), True
            )
        change_search_region_default(
            args_dict, T, min(data[TIME_STAMPS]),
            min(data[TIME_STAMPS]) - chosen_period, min(data[TIME_STAMPS]) + chosen_period, True
        )
        try:
            mini_results = lmfit_on_sample(args_dict, data, use_jitter=cand.jitter)
        except AbortFitException:
            print("AbortFitException")
            continue
        # except RuntimeError:
        #     print("RuntimeError")
        #     continue

        # summarize + annotate
        phs_path = print_lmfit_result(data, args_dict, star_name, mini_results,sid_counter, out_dir=out_dir)
        row = summarize_result(mini_results, star_name)
        row.update({
            "candidate_method": getattr(cand, "method", None)+("_jitter" if cand.jitter else ""),
            "candidate_period": chosen_period,
            "LS_power": float(getattr(cand, "LS_power", np.nan)),
            "LS_fap": float(getattr(cand, "LS_fap", np.nan)),
            "PDC_power": float(getattr(cand, "PDC_power", np.nan)),
            "PDC_fap": float(getattr(cand, "PDC_fap", np.nan)),
            "phs": phs_path,
            "p_val": np.nan, "prob_full": np.nan,
            "prob_aic": np.nan, "prob_bic": np.nan,
            "is_best": False,
            SNR_PPL: median_snr_obs,
            "SNR": median_snr_calc,
            "solution_id": sid_counter,
            "temp_lmfit": mini_results,  # used later for F-test
            RADIAL_VELS: data[RADIAL_VELS].values,
            ERRORS: data[ERRORS].values,
            MJD: data[TIME_STAMPS].values,
        })
        if cand.jitter:
            # debug_compare(mini_results, data)
            row.update(calculate_statistical_flags(data,mini_results,is_null=False))
        # add phase/coverage quality metrics

        phase_cv = calculate_phase_criterias(data, mini_results)
        row.update(phase_cv)
        row.update(coverage_qc(phase_cv, data[RADIAL_VELS]))
        rows.append(row)

        # track best by closeness of redchi to 1
        if abs(mini_results.redchi - 1) < abs(cur_red_chi - 1):
            cur_red_chi = mini_results.redchi
            best_result = mini_results
            best_cand = cand
        sid_counter += 1
    return rows, best_result, best_cand


# ---------------------------------------------------------------------------
# Null hypothesis, statistics, finalization
# ---------------------------------------------------------------------------

def _run_null_hypothesis(args_dict, data, star_name, use_jitter=False,solution_id=0,out_dir=None):
    """Run the null (constant RV) model and return its lmfit result + a pre-built row."""
    # lock PERIOD and K1 near zero
    change_search_region_default(args_dict, PERIOD, 0, -0.1, 0.1, False)
    change_search_region_default(args_dict, K1_STR, 0, -0.1, 0.1, False)
    null_res = lmfit_on_sample(args_dict, data,
                               null_hyp=True,use_jitter=use_jitter)

    null_row = {
        'star_name': star_name,
        'method':    'null_hyp_jitter' if use_jitter else 'null_hyp',
        'nfev':      getattr(null_res, "nfev", None),
        'ndata':     len(data[TIME_STAMPS]),
        'nvarys':    getattr(null_res, "nvarys", 1),
        'chisqr':    getattr(null_res, "chisqr", None),
        'redchi':    null_res.redchi,
        'aic':       null_res.aic,
        'bic':       null_res.bic,
        f'{GAMMA}_init': null_res.init_values.get(GAMMA, None) if use_jitter else None,
        f'{GAMMA}_value': null_res.params[GAMMA].value if use_jitter else getattr(null_res, GAMMA, None),
        f'{GAMMA}_vary':  null_res.params[GAMMA].vary if use_jitter else None,
        f'{GAMMA}_stderr': null_res.params[GAMMA].stderr if use_jitter else None,
        f'{LN_SIGMA_JITTER}_init': null_res.init_values.get(LN_SIGMA_JITTER, None) if use_jitter else None,
        f'{LN_SIGMA_JITTER}_value': null_res.params[LN_SIGMA_JITTER].value if use_jitter else None,
        f'{LN_SIGMA_JITTER}_vary': null_res.params[LN_SIGMA_JITTER].vary if use_jitter else None,
        f'{LN_SIGMA_JITTER}_stderr': null_res.params[LN_SIGMA_JITTER].stderr if use_jitter else None,
        'phs': None,
        'p_val': np.nan,
        'prob_full': np.nan,
        'prob_aic': np.nan,
        'prob_bic': np.nan,
        'candidate_method': 'null_hyp_jitter' if use_jitter else 'null_hyp',
        'candidate_period': 0.0,
        'LS_power': np.nan,
        'LS_fap': np.nan,
        'PDC_power': np.nan,
        'PDC_fap': np.nan,
        'is_best': False,
        'solution_id': solution_id,
    }
    if use_jitter:
        null_row.update(calculate_statistical_flags(data,null_res,is_null=True))
    # --- generate plots + report for the null model ---
    print_lmfit_result_null(
        data,
        args_dict,
        star_name,
        null_res,
        solution_id=solution_id,
        out_dir=out_dir,
        use_jitter=use_jitter,
    )

    return null_res, null_row


def _attach_probs_constraints(rows, null_row, data, rvs, massdf, path_to_csv,use_jitter=False):
    """Fill LR/AIC/BIC probabilities, F-test, and optional Roche-lobe constraints."""
    null_chisqr = null_row["redchi"] * (len(data[TIME_STAMPS]) - 1)
    null_aic, null_bic, null_bicc, null_ev_bic = null_row["aic"], null_row["bic"], null_row["bicc"], null_row["ev_bic"]
    null_aic = float(null_aic)
    null_bic = float(null_bic)
    null_bicc = float(null_bicc)
    null_ev_bic = float(null_ev_bic)

    key = f"{get_bloem_object_name(path_to_csv)}"
    try:
        mass_row = massdf.loc[massdf["ID"] == key].iloc[0] if massdf is not None else None
    except IndexError:
        mass_row = None

    for row in rows:
        if 'null_hyp' in row.get('candidate_method'):
            continue
        if use_jitter != ('jitter' in row.get('candidate_method')):
            continue

        p_val, prob_full = lr_test_prob(row['chisqr'], null_chisqr, 5, 1, len(rvs))
        p_aic = calculate_binary_probability(row['aic'], null_aic)
        p_bic = calculate_binary_probability(row['bic'], null_bic)
        p_bicc = calculate_binary_probability(row['bicc'], null_bicc)
        p_ev_bic = calculate_binary_probability(row['ev_bic'], null_ev_bic)
        row.update(dict(p_val=p_val, prob_full=prob_full, prob_aic=p_aic, prob_bic=p_bic, prob_bicc=p_bicc, prob_ev_bic=p_ev_bic))

        # F test vs null
        f_dict = f_test(data, row, null_row)
        row.update(f_dict)

        # Physical constraints (optional)
        try:
            P_fit = float(row["candidate_period"])
            K1_fit = float(row.get("K1_value", np.nan))
            e_fit = float(row.get("Eccentricity_value", np.nan))
            peri_dict = compute_min_period_row(
                mass_row, P_fit=P_fit, K1=K1_fit, e=e_fit,
                alpha_peri=1.2, alpha_apa=1.0, i_deg=90.0
            )
            row.update(peri_dict)
        except Exception as _:
            print("problem finding Roche-lobe constraint")


def _mark_best_row(df, key="redchi"):
    # compute diffs vectorized
    s = df[key].astype(float)
    diffs = s

    # treat non-finite redchis as infinitely bad
    diffs[~np.isfinite(s)] = np.inf

    if np.isinf(diffs).all():
        return

    best_idx = diffs.idxmin()

    # ensure column exists
    if "is_best" not in df.columns:
        df["is_best"] = False

    # this avoids SettingWithCopyWarning
    df.loc[best_idx, "is_best"] = True


def _apply_detection_flags(rows, rvs, err_vs, null_res, data, fap_thresh=1e-3, use_jitter=False):
    """Apply FAP/ΔRV decision flags and attach null stats per row."""
    is_binary_th = binary_rv_threshold(rvs, err_vs, drv_tresh=20, sign_threshold=4)
    null_chisqr = null_res.redchi * (len(data[TIME_STAMPS]) - 1)

    for row in rows:
        if use_jitter != ('jitter' in row.get('candidate_method')):
            continue
        row_ls_fap = row.get("LS_fap")
        row_pdc_fap = row.get("PDC_fap")

        is_ls = bool(np.isfinite(row_ls_fap) and (row_ls_fap < fap_thresh))
        is_pdc = bool(np.isfinite(row_pdc_fap) and (row_pdc_fap < fap_thresh))

        if 'null_hyp' in row.get("method"):  # null row has no FAP
            is_ls = False
            is_pdc = False

        bin_flag = (is_pdc << DecsionFlags.PDC_BIN.value) | \
                   (is_ls << DecsionFlags.LOMB_SCARGLE_BIN.value) | \
                   (is_binary_th << DecsionFlags.DELTA_RV_BIN.value)

        row["ls_fap_dec"] = is_ls
        row["pdc_fap_dec"] = is_pdc
        row["drv_dec"] = is_binary_th
        row["bin_flag"] = bin_flag

        # Attach null stats per row for convenience
        row["null_chisqr"] = null_chisqr
        row["null_redchi"] = null_res.redchi
        row["null_gamma"] = getattr(null_res, "gamma", None)
        row["null_aic"] = null_res.aic
        row["null_bic"] = null_res.bic


def _finalize_and_save(rows, star_out_path):
    """Build results DataFrame, order columns, save CSV, return (df, features_vector)."""
    results_df = pd.DataFrame(rows)
    mask = (
            # (results_df["bin_flag"] <= 0) | \
           (results_df["bin_flag_Ftest"] == 0) |
           (results_df["mass_flag_peri"] == 1) ) & \
           (results_df["candidate_method"].str.contains('null_hyp', na=False))

    # mask = results_df["candidate_method"].str.contains('null_hyp', na=False)
    results_df = results_df[~mask].copy()

    preferred_order = [
        'star_name', 'candidate_method', 'candidate_period', 'LS_power', 'LS_fap', 'PDC_power', 'PDC_fap',
        'is_best', 'method', 'nfev', 'ndata', 'nvarys', 'chisqr', 'redchi', 'aic', 'bic',
        'gamma_init', 'gamma_value', 'gamma_vary', 'gamma_stderr', 'phs',
        'p_val', 'prob_full', 'prob_aic', 'prob_bic',
        'null_chisqr', 'null_redchi', 'null_gamma', 'null_aic', 'null_bic',
        'drv_dec', 'ls_fap_dec', 'pdc_fap_dec', 'bin_flag'
    ]
    cols = [c for c in preferred_order if c in results_df.columns] + \
           [c for c in results_df.columns if c not in preferred_order]
    results_df = results_df[cols]
    results_df.sort_values(by="redchi", inplace=True)

    return results_df


# ---------------------------------------------------------------------------
# MCMC integration
# ---------------------------------------------------------------------------

def run_mcmc(args_dict, results_df, MJDs, rv_obs, rv_sigmas, out_dir , star_name = "",from_csv=False):
    """Run MCMC."""
    from mcmc.analysis import lucy_sweeney_significant, summarise_chain
    from mcmc.runner import run_mcmc_ecc, run_mcmc_circ
    from mcmc.mcmc_plotting import make_corner, plot_orbit_with_band_phase
    add_jitter = True
    try:
        best_row = results_df[(results_df.is_best == True) & (results_df.bin_flag > 0)].iloc[0]
    except IndexError:
        best_row = results_df[(results_df.is_best == True)].iloc[0]
    print("Best row solution ID is {}".format(best_row.solution_id))
    if from_csv:
        initial_ecc = np.array([
            best_row[PERIOD+"_value"],
            best_row[T+"_value"],
            best_row[OMEGA+"_value"],
            best_row[ECC+"_value"],
            best_row[K1_STR+"_value"],
            best_row[GAMMA+"_value"],
        ])

        e_lm = best_row[ECC+"_value"]
        sigma_e_lm =  best_row[ECC+"_stderr"]
    else:
        lmfit_result = best_row.temp_lmfit
        initial_ecc = np.array([
            lmfit_result.params[PERIOD].value,
            lmfit_result.params[T].value,
            lmfit_result.params[OMEGA].value,
            lmfit_result.params[ECC].value,
            lmfit_result.params[K1_STR].value,
            lmfit_result.params[GAMMA].value,
        ])

        e_lm       = lmfit_result.params[ECC].value
        sigma_e_lm = lmfit_result.params[ECC].stderr
    ecc_significant = lucy_sweeney_significant(e_lm, sigma_e_lm,
                                               threshold=2.45, add_jitter=add_jitter)
    print(f"Lucy–Sweeney says eccentricity significant? {ecc_significant}")

    P_center  = initial_ecc[0]
    T0_center = initial_ecc[1]

    # --------------------------
    # Eccentric MCMC
    # --------------------------
    flat_ecc, logp_ecc, sampler_ecc = run_mcmc_ecc(
        MJDs, rv_obs, rv_sigmas,\
        initial_ecc,
        P_center=P_center,
        T0_center=T0_center,
        dT0_days=P_center/2.0,
        dP_frac=0.01,
        nwalkers=args_dict[MCMC_PARAMS][WALKERS],
        nsteps=args_dict[MCMC_PARAMS][STEPS],
        nburn=args_dict[MCMC_PARAMS][BURN],
        thin=args_dict[MCMC_PARAMS][THIN],
        progress=True,
        add_jitter=add_jitter
    )

    print("Posterior samples (ecc):", flat_ecc.shape)

    # ----------------------------------------------------
    # Recenter T0 samples so they are near the true value
    # (within ~±P/2 of T0_true) for plotting purposes.
    # ----------------------------------------------------
    truths_ecc = None
    truths_phase_ecc = None

    labels_ecc = [r"$P$ [d]", r"$T_0$ [MJD]", r"$\omega$ [deg]",
                  r"$e$", r"$K_1$ [km/s]", r"$\gamma$ [km/s]"]
    if add_jitter:
        labels_ecc.append(r"log σ_jit")

    make_corner(flat_ecc, logp_ecc,
                truths_ecc, labels_ecc,
                tag="ecc", omega_in_col2=True, out_dir=out_dir, star_name=star_name)

    chain_names= ["P", "T0", "omega", "e", "K1", "gamma"]
    if add_jitter:
        chain_names.append("log_sj")
    summarise_chain(flat_ecc,
                    chain_names,
                    tag="ecc", out_dir=out_dir, add_jitter=add_jitter)

    plot_orbit_with_band_phase(
        MJDs, rv_obs, rv_sigmas,
        flat_ecc, logp_ecc,
        truths=truths_phase_ecc,
        tag="ecc",
        circular=False,
        out_dir = out_dir,
        add_jitter=add_jitter,
        star_name = star_name
    )

    # --------------------------
    # Circular MCMC (if e NOT significant)
    # --------------------------
    if not ecc_significant:
        initial_circ = np.array([
            best_row[PERIOD+"_value"],
            best_row[T+"_value"],
            best_row[K1_STR+"_value"],
            best_row[GAMMA+"_value"],
        ])

        flat_circ, logp_circ, sampler_circ = run_mcmc_circ(
            MJDs, rv_obs, rv_sigmas,
            initial_circ,
            P_center=P_center,
            T0_center=T0_center,
            dT0_days=P_center/2.0,
            dP_frac=0.01,
            nwalkers=args_dict[MCMC_PARAMS][WALKERS],
            nsteps=args_dict[MCMC_PARAMS][STEPS],
            nburn=args_dict[MCMC_PARAMS][BURN],
            thin=args_dict[MCMC_PARAMS][THIN],
            progress=True,
            add_jitter=add_jitter
        )

        print("Posterior samples (circ):", flat_circ.shape)

        # Recenter T0 samples near T0_true for plotting
        truths_circ = None
        truths_phase_circ = None

        labels_circ = [r"$P$ [d]", r"$T_0$ [MJD]",
                       r"$K_1$ [km/s]", r"$\gamma$ [km/s]"]
        if add_jitter:
            labels_circ.append(r"log σ_jit")
        make_corner(flat_circ, logp_circ,
                    truths_circ, labels_circ,
                    tag="circ", omega_in_col2=False, out_dir=out_dir, star_name=star_name)
        chain_names = ["P", "T0", "K1", "gamma"]
        if add_jitter:
            chain_names.append("log_sj")

        summarise_chain(flat_circ,
                        ["P", "T0", "K1", "gamma"],
                        tag="circ", out_dir=out_dir, add_jitter=add_jitter)

        plot_orbit_with_band_phase(
            MJDs, rv_obs, rv_sigmas,
            flat_circ, logp_circ,
            truths=truths_phase_circ,
            tag="circ",
            circular=True,
            out_dir = out_dir,
            add_jitter=add_jitter,
            star_name=star_name
        )
    else:
        print("Skipping circular MCMC: Lucy–Sweeney says eccentricity is significant.")


# ---------------------------------------------------------------------------
# Lightweight detection for bias-correction injection-recovery
# ---------------------------------------------------------------------------

def detect_from_arrays(mjds, rvs, errs, args_dict, use_fwhm=True):
    """
    Run the detection pipeline on in-memory arrays (no CSV I/O, no plots).

    Returns
    -------
    detected : bool
        True if a significant orbital solution was found (prob_bicc > 0.5).
    info : dict
        Fitted parameters and diagnostics. Keys include:
        'period', 'ecc', 'K1', 'gamma', 'prob_bicc', 'bicc', 'redchi',
        'LS_fap', 'PDC_fap', 'LS_iter_fap', 'PDC_iter_fap'.
    """
    data = {
        TIME_STAMPS: np.asarray(mjds, dtype=float),
        RADIAL_VELS: np.asarray(rvs, dtype=float),
        ERRORS: np.asarray(errs, dtype=float),
    }

    # Set gamma search region from data
    local_args = copy.deepcopy(args_dict)
    gamma0 = get_rv_weighted_mean(data)
    change_search_region_default(
        local_args, GAMMA, gamma0, min(rvs), max(rvs), True
    )

    info = {
        'detected': False, 'period': np.nan, 'ecc': np.nan, 'K1': np.nan,
        'gamma': np.nan, 'prob_bicc': np.nan, 'bicc': np.nan,
        'redchi': np.nan, 'LS_fap': np.nan, 'PDC_fap': np.nan,
        'LS_iter_fap': np.nan, 'PDC_iter_fap': np.nan, 'reason': '',
    }

    # 1) Period search (no plotting)
    try:
        ls_p, ls_fap, ls_fal, ls_mp, pdc_p, pdc_fap, pdc_mp, candidates_df = (
            _find_candidates(rvs, mjds, errs, local_args,
                             star_name="inject", out_dir=None,
                             use_fwhm=use_fwhm)
        )
    except Exception as e:
        info['reason'] = f'period_search_failed: {e}'
        return False, info

    if candidates_df is None or candidates_df.empty:
        info['reason'] = 'no_candidates'
        return False, info

    if np.isnan(ls_fap):
        info['reason'] = 'ls_fap_nan'
        return False, info

    # 2) Fit top candidate(s) with jitter
    cand_jitter = candidates_df.copy()
    cand_jitter["jitter"] = True

    best_row_dict = None
    best_mini = None
    best_redchi_diff = 1e9

    for cand in cand_jitter.itertuples(index=False):
        chosen_period = float(cand.period)
        fit_args = copy.deepcopy(local_args)

        if use_fwhm:
            change_search_region_default(
                fit_args, PERIOD, chosen_period,
                float(cand.fwhm_per_low), float(cand.fwhm_per_high), True
            )
        else:
            change_search_region_default(
                fit_args, PERIOD, chosen_period,
                chosen_period * 0.9, chosen_period * 1.1, False
            )
        change_search_region_default(
            fit_args, T, min(mjds),
            min(mjds) - chosen_period, min(mjds) + chosen_period, True
        )

        try:
            mini_results = lmfit_on_sample(fit_args, data, use_jitter=True)
        except Exception:
            continue

        row = summarize_result(mini_results, "inject")
        row.update(calculate_statistical_flags(data, mini_results, is_null=False))
        row['LS_fap'] = float(getattr(cand, 'LS_fap', np.nan))
        row['PDC_fap'] = float(getattr(cand, 'PDC_fap', np.nan))
        row['LS_iter_fap'] = float(getattr(cand, 'LS_iter_fap', np.nan))
        row['PDC_iter_fap'] = float(getattr(cand, 'PDC_iter_fap', np.nan))
        row['candidate_period'] = chosen_period

        diff = abs(mini_results.redchi - 1)
        if diff < best_redchi_diff:
            best_redchi_diff = diff
            best_row_dict = row
            best_mini = mini_results

    if best_row_dict is None:
        info['reason'] = 'all_fits_failed'
        return False, info

    # 3) Null hypothesis (constant RV + jitter)
    null_args = copy.deepcopy(local_args)
    change_search_region_default(null_args, PERIOD, 0, -0.1, 0.1, False)
    change_search_region_default(null_args, K1_STR, 0, -0.1, 0.1, False)
    try:
        null_res = lmfit_on_sample(null_args, data, null_hyp=True, use_jitter=True)
    except Exception:
        info['reason'] = 'null_fit_failed'
        return False, info

    null_stats = calculate_statistical_flags(data, null_res, is_null=True)

    # 4) BICc comparison
    null_bicc = float(null_stats.get('bicc', np.inf))
    orb_bicc = float(best_row_dict.get('bicc', np.inf))
    prob_bicc = calculate_binary_probability(orb_bicc, null_bicc)

    # 5) Populate info
    info.update({
        'period': float(best_row_dict.get('candidate_period', np.nan)),
        'ecc': float(best_row_dict.get(f'{ECC}_value', np.nan)),
        'K1': float(best_row_dict.get(f'{K1_STR}_value', np.nan)),
        'gamma': float(best_row_dict.get(f'{GAMMA}_value', np.nan)),
        'prob_bicc': prob_bicc,
        'bicc': orb_bicc,
        'redchi': float(best_row_dict.get('redchi', np.nan)),
        'LS_fap': float(best_row_dict.get('LS_fap', np.nan)),
        'PDC_fap': float(best_row_dict.get('PDC_fap', np.nan)),
        'LS_iter_fap': float(best_row_dict.get('LS_iter_fap', np.nan)),
        'PDC_iter_fap': float(best_row_dict.get('PDC_iter_fap', np.nan)),
    })

    detected = prob_bicc > 0.5
    info['detected'] = detected
    info['reason'] = 'detected' if detected else f'prob_bicc={prob_bicc:.3f}'

    return detected, info


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main_single(path_to_csv, path_to_out=None, massdf=None, use_fwhm=False,
                json_param_file=PARAM_FILE):

    args_dict = load_args(json_param_file)
    star_name = get_bloem_object_name(path_to_csv)
    # per-star output dir
    star_out_path = _ensure_star_out_dir(path_to_out, star_name, args_dict[PLOT])
    # 1) load & clean
    data = _load_and_clean_csv(path_to_csv)
    _rename_to_internal_cols(data)

    # 2) prepare arrays & SNR stats
    rvs, mjds, err_vs, median_snr_calc, median_snr_obs = _prepare_inputs_for_period_search(data)

    # 3) load JSON args & set gamma search region (needed also for "MCMC only" mode)
    change_search_region_default(
        args_dict, GAMMA, get_rv_weighted_mean(data),
        min(rvs), max(rvs), True
    )

    # Get the "plot" parameter to control whether to save plots and reports
    save_plots = args_dict.get(PLOT, True)

    # --- Early-exit logic based on existing outputs ---
    lmfit_csv = os.path.join(star_out_path, "lmfit_summary.csv") if star_out_path else None
    mcmc_dir  = os.path.join(star_out_path, "mcmc") if star_out_path else None

    has_lmfit = bool(lmfit_csv and os.path.exists(lmfit_csv))
    has_mcmc_output = bool(mcmc_dir and os.path.isdir(mcmc_dir) and any(os.scandir(mcmc_dir)))

    # Case 3: lmfit_summary exists AND mcmc outputs exist -> skip everything
    if has_lmfit and has_mcmc_output:
        print(f"{star_name}: lmfit_summary.csv and MCMC outputs exist; skipping.")
        res_df = pd.read_csv(lmfit_csv)
        return res_df

    # Case 2: lmfit_summary exists but no mcmc output -> load df and run only MCMC
    if has_lmfit and not has_mcmc_output:
        print(f"{star_name}: lmfit_summary.csv exists but no MCMC outputs; running MCMC only.")
        res_df = pd.read_csv(lmfit_csv)
        if len(res_df) <= 1:
            print(f"No significant solutions for {star_name}. Omiting MCMC execution")
            return res_df
        # run_mcmc(args_dict, res_df, mjds, rvs, err_vs, mcmc_dir, star_name,from_csv=True)
        return res_df

    # ----------------------------------------------------------------
    # Case 1: no lmfit_summary yet – run full pipeline (your original flow)
    # ----------------------------------------------------------------
    # 4) period search - conditionally set out_dir based on save_plots
    periodogram_out_dir = os.path.join(star_out_path, "periodogram") if (star_out_path and save_plots) else None
    ls_p, ls_fap, ls_fal, ls_mp, pdc_p, pdc_fap, pdc_mp, candidates_df =(
        _find_candidates( rvs, mjds, err_vs, args_dict, star_name,
                          out_dir=periodogram_out_dir,
                          use_fwhm=use_fwhm, ))

    peri_params = args_dict[PERIODOGRAM_PARAMS]
    pmin = peri_params[PERI_MIN_PERIOD]
    pmax = peri_params[PERI_MAX_PERIOD]

    # OPTIONAL: manual candidates to always include in next run
    manual_periods = args_dict.get(MANUAL_CANDIDATES, [30])  # <--- new optional JSON key
    candidates_df = _inject_manual_candidates(
        candidates_df,
        manual_periods,
        pmin=pmin, pmax=pmax,
        jitter=True,
        full_period_range=True,
    )

    cand_jitter = candidates_df.copy()
    cand_jitter["jitter"] = True
    # Combine into a single DataFrame for processing
    candidates_combined = cand_jitter

    # 5) fit all candidates - conditionally set out_dir based on save_plots
    solutions_out_dir = os.path.join(star_out_path, "lmfit_solutions") if (star_out_path and save_plots) else None
    rows, best_result, best_cand = _fit_candidates_over_periods_parallel(
        candidates_combined, args_dict, data, star_name, 0,
        solutions_out_dir if solutions_out_dir else (os.path.dirname(path_to_csv) if save_plots else None),
        use_fwhm, ls_fap,
        median_snr_obs=median_snr_obs, median_snr_calc=median_snr_calc,
    )

    # null_res, null_row = _run_null_hypothesis(
    #     args_dict, data, star_name,
    #     use_jitter=False,
    #     solution_id=len(rows),  # <---- pass the id
    #     out_dir = solutions_out_dir if solutions_out_dir else (os.path.dirname(path_to_csv) if save_plots else None),
    # )
    # rows.append(null_row)

    null_res_j, null_row_j = _run_null_hypothesis(
        args_dict, data, star_name,
        use_jitter=True,
        solution_id=len(rows),
        out_dir = solutions_out_dir if solutions_out_dir else (os.path.dirname(path_to_csv) if save_plots else None),
    )
    rows.append(null_row_j)
    # 7) probabilities, F-test, and physical constraints
    # _attach_probs_constraints(rows, null_row, data, rvs, massdf, path_to_csv,use_jitter=False)
    _attach_probs_constraints(rows, null_row_j, data, rvs, massdf, path_to_csv,use_jitter=True)


    # 9) decisions & attach null stats
    # _apply_detection_flags(rows, rvs, err_vs, null_res, data, fap_thresh=1e-3)
    # 9) decisions & attach null stats
    _apply_detection_flags(rows, rvs, err_vs, null_res_j, data, fap_thresh=1e-3, use_jitter=True)

    # 10) finalize & save
    final_df = _finalize_and_save(rows, star_out_path)
    _mark_best_row(final_df, "redchi")

    if star_out_path:
        out_csv = f"{star_out_path}/lmfit_summary.csv"
        final_df.to_csv(out_csv, index=False)
        print(f"Results saved to {out_csv}")

    if len(final_df) <= 1:
        print(f"No significant solutions for {star_name}. Omiting MCMC execution")
        return final_df

    # run_mcmc(args_dict, final_df, mjds, rvs, err_vs, os.path.join(star_out_path, "mcmc"), star_name)

    return final_df


def manual_append_dispatcher(
    path_to_input: str,
    path_to_out: str,
    manual_period: float,
    *,
    obj_list=None,
    massdf=None,
    use_jitter: bool = True,
    full_period_range: bool = True,
    json_param_file: str = PARAM_FILE,
):
    """
    Find the exact *_CCF_RVs.csv files like main_multiple does, and append ONE manual
    candidate solution for each matching star, without re-running the full pipeline.

    - path_to_input can be:
        * a directory that contains many *_CCF_RVs.csv
        * a single *_CCF_RVs.csv file
    - obj_list filters by substring match (same as main_multiple)
    - skip_existing_same_period avoids duplicating a MANUAL row for the same period
    """
    # Collect files
    if os.path.isdir(path_to_input):
        list_of_csvs = glob.glob(os.path.join(path_to_input, '*_CCF_RVs.csv'))
    elif os.path.isfile(path_to_input) and path_to_input.endswith('_CCF_RVs.csv'):
        list_of_csvs = [path_to_input]
    else:
        raise ValueError(f"Invalid input: {path_to_input} (need dir or *_CCF_RVs.csv file)")

    # Filter by obj_list (same style as main_multiple)
    if obj_list:
        filtered = []
        for obj in obj_list:
            for p in list_of_csvs:
                if obj in p:
                    filtered.append(p)
        list_of_csvs = filtered

    for path_to_csv in sorted(list_of_csvs):
        star_name = get_bloem_object_name(path_to_csv)
        star_out_path = os.path.join(path_to_out, star_name)
        print(f"{star_name}: appending MANUAL period {manual_period} (jitter={use_jitter}, full_range={full_period_range})")
        append_manual_candidate_solution(
            path_to_csv=path_to_csv,
            path_to_out=path_to_out,
            manual_period=manual_period,
            use_jitter=use_jitter,
            full_period_range=full_period_range,
            json_param_file=json_param_file,
            massdf=massdf,
        )

def main_multiple(path_to_ccf_out_dir, out_dir=None, obj_list=None, massdf=None):
    list_of_csvs = glob.glob(os.path.join(path_to_ccf_out_dir, '*_CCF_RVs.csv'))
    if obj_list:
        filtered_list_of_csvs = []
        for i in obj_list:
            for j in list_of_csvs:
                if i in j:
                    filtered_list_of_csvs.append(j)
        list_of_csvs = filtered_list_of_csvs
    for csv in sorted(list_of_csvs):
        if "6-107" in csv:
            continue
        print(csv)
        path_to_csv = os.path.join(path_to_ccf_out_dir, csv)
        res_df = main_single(path_to_csv, out_dir,massdf=massdf, use_fwhm=True)

def _get_next_solution_id(existing_df: pd.DataFrame) -> int:
    if existing_df is None or existing_df.empty or "solution_id" not in existing_df.columns:
        return 0
    s = pd.to_numeric(existing_df["solution_id"], errors="coerce")
    s = s[np.isfinite(s)]
    return int(s.max() + 1) if len(s) else 0


def _extract_null_row_from_csv(existing_df: pd.DataFrame, use_jitter: bool) -> dict | None:
    """
    Pull the null row from the existing lmfit_summary.csv for comparison.
    Returns a dict with the keys your code expects in _attach_probs_constraints/_apply_detection_flags.
    """
    if existing_df is None or existing_df.empty:
        return None

    # candidate_method is what you set; fallback to method if needed
    col = "candidate_method" if "candidate_method" in existing_df.columns else "method"
    if col not in existing_df.columns:
        return None

    null_tag = "null_hyp_jitter" if use_jitter else "null_hyp"
    mask = existing_df[col].astype(str).str.contains(null_tag, na=False)
    if not mask.any():
        return None

    # pick the last one (or the best, doesn't matter much as long as it's the null)
    row = existing_df.loc[mask].iloc[-1].to_dict()

    # Make sure numeric fields are numeric where needed
    for k in ["redchi", "aic", "bic", "bicc", "ev_bic", "chisqr"]:
        if k in row:
            try:
                row[k] = float(row[k])
            except Exception:
                pass

    return row


def append_manual_candidate_solution(
    path_to_csv: str,
    path_to_out: str,
    manual_period: float,
    *,
    use_jitter: bool = True,
    full_period_range: bool = True,
    json_param_file: str = PARAM_FILE,
    massdf: pd.DataFrame | None = None,
):
    """
    One-shot append:
      1) loads existing lmfit_summary.csv from star output dir
      2) finds next solution_id
      3) pulls null statistical flags from csv (if missing -> runs null once)
      4) runs _fit_single_candidate_worker once with full or narrow period bounds
      5) appends row to csv and resaves (keeping all prior solutions)
    """
    args_dict = load_args(json_param_file)

    star_name = get_bloem_object_name(path_to_csv)

    star_out_path = _ensure_star_out_dir(path_to_out, star_name, args_dict[PLOT])

    # load & prep data
    data = _load_and_clean_csv(path_to_csv)
    _rename_to_internal_cols(data)
    rvs, mjds, err_vs, median_snr_calc, median_snr_obs = _prepare_inputs_for_period_search(data)

    # args
    change_search_region_default(
        args_dict, GAMMA, get_rv_weighted_mean(data),
        min(rvs), max(rvs), True
    )

    # Get the "plot" parameter to control whether to save plots and reports
    save_plots = args_dict.get(PLOT, True)

    peri_params = args_dict[PERIODOGRAM_PARAMS]
    pmin = peri_params[PERI_MIN_PERIOD]
    pmax = peri_params[PERI_MAX_PERIOD]

    # existing csv
    lmfit_csv = os.path.join(star_out_path, "lmfit_summary.csv")
    if os.path.exists(lmfit_csv):
        existing_df = pd.read_csv(lmfit_csv)
    else:
        existing_df = pd.DataFrame()

    next_sid = _get_next_solution_id(existing_df)

    # pull null row from csv (preferred)
    null_row = _extract_null_row_from_csv(existing_df, use_jitter=use_jitter)

    # if null row missing (or lacks needed keys), run null once (cheap) and append it too
    needs_null = (
            null_row is None
            or any(k not in null_row for k in ["redchi", "aic", "bic", "bicc", "ev_bic"])
    )

    # Conditionally set out_dir based on save_plots
    solutions_out_dir = os.path.join(star_out_path, "lmfit_solutions") if save_plots else None

    if needs_null:
        null_res, null_row_new = _run_null_hypothesis(
            args_dict, data, star_name,
            use_jitter=use_jitter,
            solution_id=next_sid,
            out_dir=solutions_out_dir,
        )
        # ensure it also lands in the CSV (so next manual add doesn't need to rerun null)
        null_row = null_row_new

        # save null row first
        null_row_savable = {k: v for k, v in null_row.items()
                            if k not in ["temp_lmfit", RADIAL_VELS, ERRORS, MJD]}
        existing_df = pd.concat([existing_df, pd.DataFrame([null_row_savable])], ignore_index=True)
        next_sid += 1

        # also need null_res for _apply_detection_flags
        null_res_for_flags = null_res
    else:
        # we still need null_res-like values for _apply_detection_flags;
        # create a tiny shim object with the required attributes.
        null_res_for_flags = SimpleNamespace(
            redchi=float(null_row.get("redchi", np.nan)),
            aic=float(null_row.get("aic", np.nan)),
            bic=float(null_row.get("bic", np.nan)),
            gamma=null_row.get("null_gamma", None),
        )

    # build a "candidate" for the worker
    cand = SimpleNamespace(
        period=float(manual_period),
        method="MANUAL",
        jitter=bool(use_jitter),
        LS_power=np.nan, LS_fap=np.nan, LS_iter_fap=np.nan,
        PDC_power=np.nan, PDC_fap=np.nan, PDC_iter_fap=np.nan,
    )

    # Full range vs narrow around period
    if full_period_range:
        use_fwhm_for_worker = True
        cand.fwhm_per_low = float(pmin)
        cand.fwhm_per_high = float(pmax)
    else:
        # your default non-fwhm window behavior (±10%)
        use_fwhm_for_worker = False
        cand.fwhm_per_low = float(manual_period * 0.9)
        cand.fwhm_per_high = float(manual_period * 1.1)

    packed = (
        cand,
        args_dict,   # template (worker deepcopies)
        data,
        star_name,
        int(next_sid),
        solutions_out_dir,
        use_fwhm_for_worker,
        np.nan,  # ls_fap (not used in worker currently)
        median_snr_obs,
        median_snr_calc,
    )

    res = _fit_single_candidate_worker(packed)
    if res is None:
        print(f"{star_name}: manual candidate fit failed/aborted for P={manual_period}")
        return existing_df

    row, mini_results, _ = res

    # attach probs/constraints using the null row (from csv or newly created)
    _attach_probs_constraints([row], null_row, data, rvs, massdf, path_to_csv, use_jitter=use_jitter)

    # decision flags and null comparisons
    _apply_detection_flags([row], rvs, err_vs, null_res_for_flags, data, fap_thresh=1e-3, use_jitter=use_jitter)

    # make row CSV-safe
    row_savable = {k: v for k, v in row.items()
                   if k not in ["temp_lmfit", RADIAL_VELS, ERRORS, MJD]}

    # append + re-mark best
    new_df = pd.concat([existing_df, pd.DataFrame([row_savable])], ignore_index=True)
    # (optional) sort like you do
    if "redchi" in new_df.columns:
        try:
            new_df["redchi"] = pd.to_numeric(new_df["redchi"], errors="coerce")
            new_df.sort_values(by="redchi", inplace=True)
        except Exception:
            pass

    _mark_best_row(new_df, "redchi")

    new_df.to_csv(lmfit_csv, index=False)
    print(f"{star_name}: appended MANUAL solution (P={manual_period}) -> {lmfit_csv}")

    return new_df


# ---------------------------------------------------------------------------
# CLI entry point:  python -m pipeline.evaluator --config configs/params.yaml
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Binary-star detection pipeline (period search + orbital fitting).",
    )
    parser.add_argument(
        "--config", default=PARAM_FILE,
        help="Path to params YAML (or .json). Default: %(default)s",
    )
    cli = parser.parse_args()

    args_dict = load_args(cli.config)

    pio = args_dict.get("pipeline_io", {})
    rv_input_dir = pio.get("rv_input_dir", "")
    output_dir   = pio.get("output_dir", "")
    obj_list     = pio.get("object_list", [])
    mass_file    = pio.get("mass_file", "")
    use_fwhm     = pio.get("use_fwhm", True)

    massdf = None
    if mass_file:
        massdf = pd.read_csv(mass_file)

    print("=" * 50)
    print("  Pipeline evaluator")
    print(f"  Config:       {cli.config}")
    print(f"  RV input dir: {rv_input_dir}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Object list:  {obj_list or '(all)'}")
    print(f"  use_fwhm:     {use_fwhm}")
    print("=" * 50)

    if not obj_list:
        # process all stars
        main_multiple(rv_input_dir, out_dir=output_dir, massdf=massdf)
    else:
        main_multiple(rv_input_dir, out_dir=output_dir,
                      obj_list=obj_list, massdf=massdf)
