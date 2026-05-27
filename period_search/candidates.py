"""
period_search.candidates — Peak selection and candidate period identification.

Functions for identifying significant peaks in periodograms, computing FWHM
bounds, assembling candidate period tables, and running the full period-search
pipeline (find_periods).
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from utils.constants import (
    PERIODOGRAM_PARAMS, PERI_MIN_PERIOD, PERI_MAX_PERIOD,
    PERI_LS_NORM, PERI_LS_METHOD, PERI_LS_FA_METHOD,
    PERI_RANDOM_STATE, N_SIG_PERIODS, MIN_SEP, WINDOW_ITERATIONS,
    PERI_PDC_SAMPLES_PER_PEAK, PERI_RUN_PERMUTATIONS_LS,PERI_RUN_PERMUTATIONS_PDC,
    PERI_SHOW_PERM_PROGRESS, PLOT_STYLE,
)
from period_search.periodogram import ls, pdc_opt, plot_periodogram_plotly
from period_search.permutation import ls_permutation_max_powers_mp, pdc_permutation_max_powers
from PDC.pdc_func import pdc_window
from utils.periodogramFAPAnalysis import analyze_permutation_convergence


# ---------------------------------------------------------------------------
# FWHM bounds
# ---------------------------------------------------------------------------

def _fwhm_bounds(freq, power, idx):
    pk = power[idx];
    if not np.isfinite(pk) or pk <= 0: return (np.nan, np.nan)
    half = pk/2.0
    # left
    i = idx
    while i>0 and power[i-1] > half: i -= 1
    if i==0: fL = freq[0]
    else:
        x1,x2,p1,p2 = freq[i-1],freq[i],power[i-1],power[i]
        t = (half - p1)/(p2 - p1) if p2!=p1 else 0.0
        fL = x1 + t*(x2 - x1)
    # right
    j = idx
    while j < len(power)-1 and power[j+1] > half: j += 1
    if j==len(power)-1: fR = freq[-1]
    else:
        x1,x2,p1,p2 = freq[j],freq[j+1],power[j],power[j+1]
        t = (half - p1)/(p2 - p1) if p2!=p1 else 0.0
        fR = x1 + t*(x2 - x1)
    return (fL, fR)


# ---------------------------------------------------------------------------
# append_period_results
# ---------------------------------------------------------------------------

def append_period_results(results,
                          method,
                          per,
                          periodogram_res,
                          pmin,
                          pmax,
                          include_harmonics=True,
                          include_fwhm=False):
    """
    Append periodogram diagnostics for a given period (and optionally its
    half/double harmonics) into `results`.

    Parameters
    ----------
    results : list
        List of dicts that will be extended in-place.
    method : str
        The "primary" method name (e.g. "ls", "pdc").
    per : float
        Candidate period (days).
    periodogram_res : dict
        Mapping method_name -> dict with keys:
            "freq", "pow", "fap_vec", "fap_by_iter"
    pmin, pmax : float
        Valid period range; if per not in (pmin, pmax), nothing is done.
    include_harmonics : bool
        If True, also append results at P/2 and 2P.
    include_fwhm : bool
        If True, compute FWHM bounds for the base method.
    """
    if not (pmin < per < pmax):
        return

    # ---------- helpers ----------

    def _freq_arr(method_name):
        return periodogram_res[method_name]["freq"]

    def _idx_for_freq(method_name, freq_target):
        freqs = _freq_arr(method_name)
        return int(np.abs(freqs - freq_target).argmin())

    def _metrics_at_idx(method_name, idx):
        d = periodogram_res[method_name]
        power = d["pow"][idx]
        fap = d["fap_vec"][idx]
        fap_by_iter = d.get("fap_by_iter", None)

        if fap_by_iter is not None and len(fap_by_iter) > 0:
            iter_fap = np.sum(fap_by_iter >= power) / len(fap_by_iter)
        else:
            iter_fap = np.nan

        return power, fap, iter_fap

    def _maybe_add_fwhm(rec, method_name, idx):
        if not include_fwhm:
            return
        fL, fR = _fwhm_bounds(
            _freq_arr(method_name),
            periodogram_res[method_name]["pow"],
            idx
        )
        rec.update({
            "fwhm_freq_low": fL,
            "fwhm_freq_high": fR,
            "fwhm_per_low": 1.0 / fR if np.isfinite(fR) and fR > 0 else np.nan,
            "fwhm_per_high": 1.0 / fL if np.isfinite(fL) and fL > 0 else np.nan,
        })

    def _add_other_methods(rec, freq_target, base_method):
        for other_method in periodogram_res.keys():
            if other_method == base_method:
                continue
            idx_o = _idx_for_freq(other_method, freq_target)
            pow_o, fap_o, iter_fap_o = _metrics_at_idx(other_method, idx_o)
            rec[other_method + "_power"]    = pow_o
            rec[other_method + "_fap"]      = fap_o
            rec[other_method + "_iter_fap"] = iter_fap_o

    def _append_entry(label, period_value, freq_target, base_method):
        idx = _idx_for_freq(base_method, freq_target)
        pow_b, fap_b, iter_fap_b = _metrics_at_idx(base_method, idx)

        rec = {
            "method": label,
            "period": period_value,
            base_method + "_power":    pow_b,
            base_method + "_fap":      fap_b,
            base_method + "_iter_fap": iter_fap_b,
        }

        _maybe_add_fwhm(rec, base_method, idx)
        _add_other_methods(rec, freq_target, base_method)
        results.append(rec)

    # ---------- base period ----------
    base_freq = 1.0 / per
    _append_entry(method, per, base_freq, method)

    if not include_harmonics:
        return

    # ---------- half period P/2 ----------
    half_per = per / 2.0
    if pmin < half_per < pmax:
        half_freq = 1.0 / half_per
        _append_entry(f"{method}_half", half_per, half_freq, method)

    # ---------- double period 2P ----------
    double_per = per * 2.0
    if pmin < double_per < pmax:
        double_freq = 1.0 / double_per
        _append_entry(f"{method}_double", double_per, double_freq, method)


# ---------------------------------------------------------------------------
# significant_periods
# ---------------------------------------------------------------------------

def significant_periods(periods, powers,
                        max_periods=1,
                        min_separation=1.1):
    """
    Identify up to `max_periods` significant periods from a power spectrum,
    ensuring selected periods differ by at least a factor `min_separation`.

    Separation check is done in log-space:
        |ln P_i - ln P_j| >= ln(min_separation)

    Parameters
    ----------
    periods : array_like, shape (N,)
        Array of period values (need not be evenly spaced).
    powers : array_like, shape (N,)
        Corresponding power values.
    max_periods : int, optional
        Maximum number of periods to return (default: 5).
    min_separation : float, optional
        Minimum multiplicative separation between any two returned periods
        (default: 1.2 means at least ±20% apart).

    Returns
    -------
    selected_periods : list of float
        The most significant periods, sorted by descending power, pruned
        to enforce the minimum log-space separation criterion.
    """
    periods = np.asarray(periods)
    powers = np.asarray(powers)

    # 1. Find all local peaks
    peak_idx, _ = find_peaks(powers)
    pk_periods = periods[peak_idx]
    pk_powers = powers[peak_idx]

    # 2. Sort peaks by descending power
    order = np.argsort(pk_powers)[::-1]

    # 3. Precompute logs and threshold
    log_periods = np.log(pk_periods)
    log_thresh = np.log(min_separation)

    # 4. Greedily pick peaks, enforcing log-space separation
    selected = []
    selected_logs = []
    for i in order:
        lp = log_periods[i]
        # check against all already-selected peaks
        if all(abs(lp - sl) >= log_thresh for sl in selected_logs):
            selected.append(pk_periods[i])
            selected_logs.append(lp)
            if len(selected) >= max_periods:
                break

    return selected


# ---------------------------------------------------------------------------
# find_periods  —  full period-search pipeline
# ---------------------------------------------------------------------------

def find_periods(rvs, mjds, err_vs, args_dict, star_name, out_dir=None, use_fwhm=True):
    peri_params = args_dict[PERIODOGRAM_PARAMS]
    pmin = peri_params[PERI_MIN_PERIOD]
    pmax = peri_params[PERI_MAX_PERIOD]
    pdc_spp = peri_params.get(PERI_PDC_SAMPLES_PER_PEAK, 10)
    plot_style = args_dict.get(PLOT_STYLE, "interactive")

    # --- Lomb-Scargle ---
    period, _, fap, fal, freq_ls, pow, ls_fap_vec = ls(mjds,
                                                    rvs,
                                                    data_err=err_vs,
                                                    pmin=pmin, pmax=pmax,
                                                    norm=peri_params[PERI_LS_NORM], ls_method=peri_params[PERI_LS_METHOD],
                                                    fa_method=peri_params[PERI_LS_FA_METHOD],
                                                    center_data=True, random_state=peri_params[PERI_RANDOM_STATE])
    ls_sig_periods = significant_periods(1 / freq_ls, pow,max_periods=peri_params[N_SIG_PERIODS],min_separation=peri_params[MIN_SEP])

    freq_w, pdc_w = pdc_window(mjds, pmin=pmin, pmax=pmax, samples_per_peak=pdc_spp)
    plot_periodogram_plotly(freq_w, pdc_w, fal=None, pmin=pmin, pmax=pmax,
                            star_id=star_name + '_WA', out_dir=out_dir, periodogram_kind='PDC', style=plot_style)

    best_period1, _, fap1, fal1, freq_pdc, pdc_power_reg, pdc_fap_vec = pdc_opt(
        mjds, rvs, data_err=err_vs, pmin=pmin, pmax=pmax, samples_per_peak=pdc_spp
    )

    pdc_sig_periods = significant_periods(1 / freq_pdc, pdc_power_reg,max_periods=peri_params[N_SIG_PERIODS],min_separation=peri_params[MIN_SEP])

    # --- Collect results into a DataFrame ---
    n_iter = peri_params[WINDOW_ITERATIONS]
    # n_iter = 100_000  # hardcoded for FAP convergence investigation

    show_perm_progress = peri_params.get(PERI_SHOW_PERM_PROGRESS, True)
    run_perms = peri_params.get(PERI_RUN_PERMUTATIONS_LS, False)
    if run_perms:
        ls_iterations = ls_permutation_max_powers_mp(
            rvs=rvs,
            mjds=mjds,
            err_vs=err_vs,
            n_iter=n_iter,
            pmin=pmin,
            pmax=pmax,
            norm=peri_params[PERI_LS_NORM],
            ls_method=peri_params[PERI_LS_METHOD],
            fa_method=peri_params[PERI_LS_FA_METHOD],
            center_data=True,
            random_state=peri_params[PERI_RANDOM_STATE],
            show_progress=show_perm_progress,
        )
    else:
        ls_iterations = []
    run_perms = peri_params.get(PERI_RUN_PERMUTATIONS_PDC, False)

    if run_perms:
        pdc_iterations = pdc_permutation_max_powers(
            rvs=rvs,
            mjds=mjds,
            err_vs=err_vs,
            n_iter=n_iter,
            pmin=pmin,
            pmax=pmax,
            probabilities=(0.5, 0.01, 0.001),
            random_state=peri_params[PERI_RANDOM_STATE],
            show_progress=show_perm_progress,
            samples_per_peak=pdc_spp,
        )
    else:
        pdc_iterations = []

    # --- Plot periodograms (use empirical FAL from permutations when available) ---
    empirical_percentiles = [50, 80, 99]
    empirical_fal_labels = [1 - p / 100 for p in empirical_percentiles]
    analytical_fal_labels = [0.5, 0.01, 0.001]

    if len(ls_iterations) > 0:
        ls_fal_plot = np.percentile(ls_iterations, empirical_percentiles)
        ls_fal_labels = empirical_fal_labels
    else:
        ls_fal_plot = fal
        ls_fal_labels = analytical_fal_labels
    plot_periodogram_plotly(freq_ls, pow, ls_fal_plot, pmin=pmin, pmax=pmax,
                            star_id=star_name, out_dir=out_dir, periodogram_kind='LS',
                            style=plot_style, fal_labels=ls_fal_labels)

    if len(pdc_iterations) > 0:
        pdc_fal_plot = np.percentile(pdc_iterations, empirical_percentiles)
        pdc_fal_labels = empirical_fal_labels
    else:
        pdc_fal_plot = fal1
        pdc_fal_labels = analytical_fal_labels
    plot_periodogram_plotly(freq_pdc, pdc_power_reg, fal=pdc_fal_plot, pmin=pmin, pmax=pmax,
                            star_id=star_name, out_dir=out_dir, periodogram_kind='PDC_opt',
                            style=plot_style, fal_labels=pdc_fal_labels)

    # # --- FAP convergence analysis ---
    # ls_convergence = analyze_permutation_convergence(
    #     Z=np.array(ls_iterations),
    #     Z_obs=max(pow),
    #     out_dir=out_dir,
    #     tag="LS",
    #     star_name=star_name,
    # )
    # pdc_convergence = analyze_permutation_convergence(
    #     Z=np.array(pdc_iterations),
    #     Z_obs=max(pdc_power_reg),
    #     out_dir=out_dir,
    #     tag="PDC",
    #     star_name=star_name,
    # )
    # print(f"\n{'='*60}")
    # print(f"FAP convergence summary for {star_name}")
    # print(f"{'='*60}")
    # print(f"LS  convergence: {ls_convergence}")
    # print(f"PDC convergence: {pdc_convergence}")
    # print(f"{'='*60}\n")

    # if out_dir:
    #     summary_path = os.path.join(out_dir, f"{star_name}_fap_convergence_summary.json")
    #     with open(summary_path, "w") as f:
    #         json.dump({"LS": ls_convergence, "PDC": pdc_convergence}, f, indent=2, default=str)

    results = []
    periodogram_res = {
        "LS":{
            "freq": freq_ls,
            "pow": pow,
            "fap_vec": ls_fap_vec,
            "fap_by_iter" : np.array(ls_iterations)
        },
        "PDC":{
            "freq": freq_pdc,
            "pow": pdc_power_reg,
            "fap_vec": pdc_fap_vec,
            "fap_by_iter": np.array(pdc_iterations)
        }
        }
    for per in ls_sig_periods:
        append_period_results(results, "LS", per, periodogram_res, pmin, pmax,include_harmonics=False,include_fwhm=use_fwhm)
    for per in pdc_sig_periods:
        append_period_results(results, "PDC", per, periodogram_res, pmin, pmax,include_harmonics=False, include_fwhm=use_fwhm)

    results_df = pd.DataFrame(results)

    return period, fap, fal,max(pow), best_period1, fap1,max(pdc_power_reg), results_df
