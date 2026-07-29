"""
de_restart_test_8031.py — Seeded multi-restart DE test on BLOeM 8-031, sid-17.

Background: in the third_harmonics run, the single unseeded
differential_evolution fit of the direct PDC candidate at P~254.3 d (sid 17)
converged onto the near-parabolic ridge (P=261.1 d, K1=67 km/s, e=0.993,
mass_flag_peri=1), while harmonic-seeded rows at the same period found a
physical minimum with better BICc (sid 22: e=0.73, bicc 123.3). The ridge is
a large-volume attractor for an unseeded DE population; finding the physical
basin is currently luck.

This script reproduces sid-17's exact fit setup (same FWHM period window,
same objective, same bounds) and compares:
  A. N unseeded single runs        -> what the pipeline does today (scatter)
  B. N seeded restarts             -> deterministic; pick best physical row
  C. e<=0.9 capped seeded restarts -> fallback if B yields nothing physical

Proposed pipeline rule: per candidate, run the seeded restarts, keep the
lowest-BICc solution with mass_flag_peri == 0 (fall back to the e-capped
pass if none). If this reliably lands in the sid-22 basin, it can replace
the single unseeded run in _fit_single_candidate_worker.

Run:  cd Scripts && conda run -n tau_binary python de_restart_test_8031.py
"""

import copy
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import lmfit

from utils.constants import *
from orbital.fitting import (
    define_search_param, chisqr_with_jitter,
    get_rv_weighted_mean, lmfit_on_sample,
)
from orbital.statistics import (
    calculate_statistical_flags, calculate_binary_probability,
)
from utils.roche_lobe import compute_min_period_row
from period_search.candidates import find_periods
from pipeline.config import load_args, change_search_region_default
from pipeline.data_loading import (
    _load_and_clean_csv, _rename_to_internal_cols,
    _prepare_inputs_for_period_search,
)

CONFIG = "configs/params_dr5_third_harmonics.yaml"
STAR = "BLOeM_8-031"
STAR_CSV = ("/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/"
            "dr5_neb_div_from_coadded/BLOeM_8-031_CCF_RVs.csv")
OUT_DIR = ("/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/"
           "dr5_neb_div_from_coadded/third_harmonics/BLOeM_8-031")

TARGET_P = 254.3051            # sid-17 candidate (direct PDC peak)
N_UNSEEDED = 5                 # experiment A
SEEDS = (562, 8031, 17, 42, 123)   # experiment B (deterministic)
ECC_CAP = 0.9
CAP_SEEDS = (1, 2, 3)          # experiment C (fallback)


def _fit_one(packed):
    """One DE fit of the sid-17 candidate window; optionally seeded/e-capped."""
    label, seed, ecc_max, fit_args, data, null_bicc, mass_row = packed
    lp = fit_args[LMFIT_PARAMS]
    params = lmfit.Parameters()
    for field in (PERIOD, GAMMA, K1_STR, OMEGA, ECC, T, LN_SIGMA_JITTER):
        define_search_param(params, lp[SEARCH_REGION], field)
    if ecc_max is not None:
        params[ECC].max = ecc_max
        params[ECC].value = min(params[ECC].value, ecc_max)

    fcn_kws = {TIME_STAMPS: np.array(data[TIME_STAMPS]),
               RADIAL_VELS: np.array(data[RADIAL_VELS]),
               ERRORS: np.abs(np.array(data[ERRORS]))}
    mini = lmfit.Minimizer(chisqr_with_jitter, params, fcn_kws=fcn_kws)
    max_nfev = int(lp.get(MAX_NFEV, 200000))
    t0 = time.time()
    try:
        res = mini.minimize(method="differential_evolution",
                            max_nfev=max_nfev, seed=seed)
    except TypeError:
        # older lmfit: no kwarg passthrough -> seed the global RNG instead
        np.random.seed(seed)
        res = mini.minimize(method="differential_evolution", max_nfev=max_nfev)

    stats = calculate_statistical_flags(data, res, is_null=False)
    P = res.params[PERIOD].value
    K1 = res.params[K1_STR].value
    e = res.params[ECC].value
    row = dict(label=label, seed=seed, ecc_cap=ecc_max,
               Period=P, K1=K1, ecc=e,
               omega=res.params[OMEGA].value,
               gamma=res.params[GAMMA].value,
               bicc=float(stats["bicc"]),
               prob_bicc=calculate_binary_probability(float(stats["bicc"]),
                                                      null_bicc),
               nfev=res.nfev, sec=round(time.time() - t0, 1))
    try:
        peri_dict = compute_min_period_row(mass_row, P_fit=P, K1=K1, e=e,
                                           alpha_peri=1.2, alpha_apa=1.0,
                                           i_deg=90.0)
        row["mass_flag_peri"] = peri_dict.get("mass_flag_peri", np.nan)
    except Exception:
        row["mass_flag_peri"] = np.nan
    return row


def main():
    args_dict = load_args(CONFIG)
    data = _load_and_clean_csv(STAR_CSV)
    _rename_to_internal_cols(data)
    rvs, mjds, err_vs, _, _ = _prepare_inputs_for_period_search(data)

    change_search_region_default(args_dict, GAMMA, get_rv_weighted_mean(data),
                                 min(rvs), max(rvs), True)

    # --- re-derive the sid-17 PDC candidate window (permutations off: FWHM
    #     bounds depend only on the periodogram itself) ---
    peri = args_dict[PERIODOGRAM_PARAMS]
    peri["run_permutations_ls"] = False
    peri["run_permutations_pdc"] = False
    *_, candidates_df = find_periods(rvs, mjds, err_vs, args_dict, STAR,
                                     out_dir=None, use_fwhm=True)
    pdc = candidates_df[candidates_df["method"] == "PDC"]
    cand = pdc.loc[(pdc["period"] - TARGET_P).abs().idxmin()]
    print(f"candidate: P={cand.period:.4f} d, FWHM window = "
          f"[{cand.fwhm_per_low:.4f}, {cand.fwhm_per_high:.4f}] d")

    fit_args = copy.deepcopy(args_dict)
    change_search_region_default(fit_args, PERIOD, float(cand.period),
                                 float(cand.fwhm_per_low),
                                 float(cand.fwhm_per_high), True)
    change_search_region_default(fit_args, T, min(mjds),
                                 min(mjds) - float(cand.period),
                                 min(mjds) + float(cand.period), True)

    # --- null model (constant + jitter) for prob_bicc ---
    null_res = lmfit_on_sample(fit_args, data, null_hyp=True, use_jitter=True)
    null_bicc = float(calculate_statistical_flags(data, null_res,
                                                  is_null=True)["bicc"])
    print(f"null bicc = {null_bicc:.2f}")

    mass_file = args_dict.get("pipeline_io", {}).get("mass_file", "")
    mass_row = None
    if mass_file and os.path.exists(mass_file):
        massdf = pd.read_csv(mass_file)
        sel = massdf.loc[massdf["ID"] == STAR]
        mass_row = sel.iloc[0] if len(sel) else None

    tasks = (
        [(f"unseeded_{i}", None, None, fit_args, data, null_bicc, mass_row)
         for i in range(N_UNSEEDED)]
        + [(f"seeded_{s}", s, None, fit_args, data, null_bicc, mass_row)
           for s in SEEDS]
        + [(f"ecap_{s}", s, ECC_CAP, fit_args, data, null_bicc, mass_row)
           for s in CAP_SEEDS]
    )
    print(f"running {len(tasks)} DE fits in parallel...")
    with Pool(min(len(tasks), 7)) as pool:
        rows = pool.map(_fit_one, tasks)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = ["label", "seed", "ecc_cap", "Period", "K1", "ecc", "bicc",
            "prob_bicc", "mass_flag_peri", "nfev", "sec"]
    print()
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # --- current behaviour vs proposed selection rule ---
    current = out[out.label == "unseeded_0"].iloc[0]
    seeded = out[out.label.str.startswith("seeded")]
    phys = seeded[seeded.mass_flag_peri == 0]
    used_cap = False
    if phys.empty:
        cap = out[out.label.str.startswith("ecap")]
        phys = cap[cap.mass_flag_peri == 0]
        used_cap = True
    proposed = phys.loc[phys.bicc.idxmin()] if not phys.empty else None

    print("\n--- pipeline reference rows (third_harmonics lmfit_summary) ---")
    ref = pd.read_csv(os.path.join(OUT_DIR, "lmfit_summary.csv"))
    refc = ref[ref.solution_id.isin([13, 17, 22])
               & ~ref.candidate_method.astype(str).str.contains("null")]
    print(refc[["solution_id", "candidate_method", "Period_value", "K1_value",
                "Eccentricity_value", "bicc", "prob_bicc", "mass_flag_peri"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print(f"\ncurrent pipeline (single unseeded run): "
          f"P={current.Period:.2f}  K1={current.K1:.1f}  e={current.ecc:.3f}  "
          f"bicc={current.bicc:.2f}  mass_flag_peri={current.mass_flag_peri}")
    if proposed is not None:
        print(f"proposed pick ({proposed.label}"
              f"{', via e-cap fallback' if used_cap else ''}): "
              f"P={proposed.Period:.2f}  K1={proposed.K1:.1f}  "
              f"e={proposed.ecc:.3f}  bicc={proposed.bicc:.2f}  "
              f"mass_flag_peri={proposed.mass_flag_peri}")
    else:
        print("proposed strategy found no physical solution "
              "(would keep the current row + flag)")

    csv_path = os.path.join(OUT_DIR, "de_restart_test.csv")
    out.to_csv(csv_path, index=False)
    print(f"\nall runs saved to {csv_path}")


if __name__ == "__main__":
    main()
