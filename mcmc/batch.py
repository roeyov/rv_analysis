"""
mcmc.batch — Batch MCMC runner over a directory of lmfit results.

Usage:
    python -m mcmc.batch --config params.yaml
"""

import os
import glob
import json

import numpy as np
import pandas as pd
from pathlib import Path

from utils.constants import *
from mcmc.selector_app import get_best_row
from pipeline.config import PARAM_FILE, load_args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_param(row, par_name, kind="value", default=np.nan):
    """
    Read *par_name*_*kind* from a pandas Series, falling back to *par_name*.
    """
    col1 = f"{par_name}_{kind}"
    if col1 in row.index:
        return row[col1]
    if par_name in row.index:
        return row[par_name]
    return default


def load_rv_csv(rv_path, mjd_col="MJD", rv_col="Mean RV",
                rverr_col="Mean RVsig"):
    """Load a single RV CSV and return MJDs, RVs, RV errors."""
    df = pd.read_csv(rv_path)
    df = df.dropna(subset=[mjd_col, rv_col, rverr_col])
    return (
        df[mjd_col].values.astype(float),
        df[rv_col].values.astype(float),
        df[rverr_col].values.astype(float),
    )


def save_choose_params_file(out_dir, choose_params):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    choose_param_path = out_path / "choose_param.json"
    with open(choose_param_path, "w", encoding="utf-8") as f:
        json.dump(choose_params, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Saved chooser params to: {choose_param_path}")


# ---------------------------------------------------------------------------
# Single-star MCMC drivers
# ---------------------------------------------------------------------------

def run_mcmc_null_full(args_dict, best_row, MJDs, rv_obs, rv_sigmas,
                       out_dir, star_name=""):
    """Run null-model MCMC for a single lmfit row."""
    from mcmc.mcmc_plotting import make_corner
    from mcmc.analysis import summarise_chain
    from mcmc.runner import run_mcmc_null

    add_jitter = True
    gamma_val = float(get_param(best_row, GAMMA, kind="value"))
    initial = np.array([gamma_val])

    flat_null, logp_null, _ = run_mcmc_null(
        MJDs, rv_obs, rv_sigmas, initial,
        nwalkers=args_dict[MCMC_PARAMS][WALKERS],
        nsteps=args_dict[MCMC_PARAMS][STEPS],
        nburn=args_dict[MCMC_PARAMS][BURN],
        thin=args_dict[MCMC_PARAMS][THIN],
        progress=True, add_jitter=add_jitter,
    )
    print("Posterior samples (null):", flat_null.shape)

    labels = [r"$\gamma$ [km/s]"]
    if add_jitter:
        labels.append(r"log σ_jit")

    make_corner(flat_null, logp_null, None, labels,
                tag="null", omega_in_col2=True,
                out_dir=out_dir, star_name=star_name)

    chain_names = ["gamma"]
    if add_jitter:
        chain_names.append("log_sj")

    return summarise_chain(flat_null, chain_names, tag="null",
                           out_dir=out_dir, add_jitter=add_jitter)


def run_mcmc_single_star(args_dict, best_row, MJDs, rv_obs, rv_sigmas,
                         out_dir, star_name=""):
    """Run eccentric (and optionally circular) MCMC for a single lmfit row."""
    from mcmc.analysis import lucy_sweeney_significant, summarise_chain
    from mcmc.runner import run_mcmc_ecc, run_mcmc_circ
    from mcmc.mcmc_plotting import make_corner, plot_orbit_with_band_phase

    add_jitter = True

    P_val     = float(get_param(best_row, PERIOD, kind="value"))
    T0_val    = float(get_param(best_row, T,      kind="value"))
    omega_val = float(get_param(best_row, OMEGA,  kind="value"))
    ecc_val   = float(get_param(best_row, ECC,    kind="value"))
    K1_val    = float(get_param(best_row, K1_STR, kind="value"))
    gamma_val = float(get_param(best_row, GAMMA,  kind="value"))

    initial_ecc = np.array([P_val, T0_val, omega_val, ecc_val, K1_val, gamma_val])
    P_center = initial_ecc[0]
    T0_center = initial_ecc[1]

    # -- Eccentric MCMC --
    flat_ecc, logp_ecc, _ = run_mcmc_ecc(
        MJDs, rv_obs, rv_sigmas, initial_ecc,
        P_center=P_center, T0_center=T0_center,
        dT0_days=P_center / 2.0, dP_frac=0.01,
        nwalkers=args_dict[MCMC_PARAMS][WALKERS],
        nsteps=args_dict[MCMC_PARAMS][STEPS],
        nburn=args_dict[MCMC_PARAMS][BURN],
        thin=args_dict[MCMC_PARAMS][THIN],
        progress=True, add_jitter=add_jitter,
    )
    print("Posterior samples (ecc):", flat_ecc.shape)

    labels_ecc = [r"$P$ [d]", r"$T_0$ [MJD]", r"$\omega$ [deg]",
                  r"$e$", r"$K_1$ [km/s]", r"$\gamma$ [km/s]"]
    if add_jitter:
        labels_ecc.append(r"log σ_jit")

    make_corner(flat_ecc, logp_ecc, None, labels_ecc, tag="ecc",
                omega_in_col2=True, out_dir=out_dir, star_name=star_name)

    chain_names = ["P", "T0", "omega", "e", "K1", "gamma"]
    if add_jitter:
        chain_names.append("log_sj")

    dict_row = summarise_chain(flat_ecc, chain_names, tag="ecc",
                               out_dir=out_dir, add_jitter=add_jitter)

    plot_orbit_with_band_phase(
        MJDs, rv_obs, rv_sigmas, flat_ecc, logp_ecc,
        truths=None, tag="ecc", circular=False,
        out_dir=out_dir, add_jitter=add_jitter, star_name=star_name,
    )

    # -- Lucy-Sweeney --
    e_lm = dict_row["ecc_e_mode"]
    sigma_e_lm = dict_row["ecc_e_errm"]
    ecc_significant = lucy_sweeney_significant(e_lm, sigma_e_lm,
                                                threshold=2.45,
                                                add_jitter=add_jitter)
    print(f"Lucy–Sweeney says eccentricity significant? {ecc_significant}")

    # -- Circular MCMC (if e NOT significant) --
    if not ecc_significant:
        initial_circ = np.array([P_val, T0_val, K1_val, gamma_val])
        flat_circ, logp_circ, _ = run_mcmc_circ(
            MJDs, rv_obs, rv_sigmas, initial_circ,
            P_center=P_center, T0_center=T0_center,
            dT0_days=P_center / 2.0, dP_frac=0.01,
            nwalkers=args_dict[MCMC_PARAMS][WALKERS],
            nsteps=args_dict[MCMC_PARAMS][STEPS],
            nburn=args_dict[MCMC_PARAMS][BURN],
            thin=args_dict[MCMC_PARAMS][THIN],
            progress=True, add_jitter=add_jitter,
        )
        print("Posterior samples (circ):", flat_circ.shape)

        labels_circ = [r"$P$ [d]", r"$T_0$ [MJD]",
                       r"$K_1$ [km/s]", r"$\gamma$ [km/s]"]
        if add_jitter:
            labels_circ.append(r"log σ_jit")

        make_corner(flat_circ, logp_circ, None, labels_circ, tag="circ",
                    omega_in_col2=False, out_dir=out_dir, star_name=star_name)

        circ_names = ["P", "T0", "K1", "gamma"]
        if add_jitter:
            circ_names.append("log_sj")
        circ_row = summarise_chain(flat_circ, circ_names, tag="circ",
                                   out_dir=out_dir, add_jitter=add_jitter)
        dict_row.update(circ_row)

        plot_orbit_with_band_phase(
            MJDs, rv_obs, rv_sigmas, flat_circ, logp_circ,
            truths=None, tag="circ", circular=True,
            out_dir=out_dir, add_jitter=add_jitter, star_name=star_name,
        )
    else:
        print("Skipping circular MCMC: eccentricity is significant.")

    return dict_row


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_mcmc_batch(args_dict, rv_dir, lmfit_dir, out_dir,
                   wanted_periods=None,
                   filter_expression=None,
                   field_to_check="bic",
                   take_min=True):
    """
    Loop over RV files ``{rv_dir}/{star}_CCF_RVs.csv`` and corresponding
    lmfit summaries ``{lmfit_dir}/{star}/lmfit_summary.csv``, running MCMC
    for each.
    """
    rv_pattern = os.path.join(rv_dir, "*_CCF_RVs.csv")
    rv_files = sorted(glob.glob(rv_pattern))
    print(f"Found {len(rv_files)} RV files to scan in {rv_dir}")

    if filter_expression is None:
        filter_expression = (
            "~candidate_method.str.contains('MANUAL')"
            "& ~candidate_method.str.contains('null')"
            "& candidate_method.str.contains('jitter')"
            "& (prob_bicc > 0.5)"
            "& (mass_flag_peri == 0)"
        )

    choose_params = {
        "filter_expression": filter_expression,
        "field_to_check": field_to_check,
        "take_min": take_min,
    }
    save_choose_params_file(out_dir, choose_params)

    rows = []
    for rv_path in rv_files:
        star_name = os.path.basename(rv_path).replace("_CCF_RVs.csv", "")
        if wanted_periods and star_name not in wanted_periods:
            continue
        print(f"\n=== Processing star: {star_name} ===")

        lmfit_path = os.path.join(lmfit_dir, star_name, "lmfit_summary.csv")
        if not os.path.exists(lmfit_path):
            print(f"  No lmfit_summary.csv found at {lmfit_path}, skipping.")
            continue

        try:
            results_df = pd.read_csv(lmfit_path)
        except Exception as e:
            print(f"  Failed to read {lmfit_path}: {e}, skipping.")
            continue

        if results_df.empty:
            print(f"  lmfit_summary.csv for {star_name} is empty, skipping.")
            continue

        try:
            MJDs, rv_obs, rv_sigmas = load_rv_csv(rv_path)
        except Exception as e:
            print(f"  Failed to load RVs from {rv_path}: {e}, skipping.")
            continue

        best_row = get_best_row(results_df, filter_expression,
                                field_to_check, take_min)
        star_out_dir = os.path.join(out_dir, star_name)
        Path(star_out_dir).mkdir(parents=True, exist_ok=True)

        if best_row is None:
            null_row = results_df[
                results_df.candidate_method.str.contains('null_hyp_jitter')
            ].iloc[0]
            row_res = run_mcmc_null_full(
                args_dict, null_row, MJDs, rv_obs, rv_sigmas,
                star_out_dir, star_name,
            )
            row_res["star_name"] = star_name
            rows.append(row_res)
            print(f"  Done. Results saved under {star_out_dir}")
            continue

        print(f"  Running MCMC for {star_name} "
              f"(period={best_row.get(PERIOD + '_value', np.nan)}).")
        try:
            row_res = run_mcmc_single_star(
                args_dict, best_row, MJDs, rv_obs, rv_sigmas,
                star_out_dir, star_name,
            )
            row_res["star_name"] = star_name
            rows.append(row_res)
            print(f"  Done. Results saved under {star_out_dir}")
        except Exception as e:
            print(f"  MCMC failed for {star_name}: {e}")

    return rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch MCMC posterior sampling.",
    )
    parser.add_argument(
        "--config", default=PARAM_FILE,
        help="Path to params.yaml (or .json). Default: %(default)s",
    )
    cli = parser.parse_args()

    args_dict = load_args(cli.config)

    mcmc_io = args_dict.get(MCMC_PARAMS, {})
    rv_dir    = mcmc_io.get("rv_input_dir", "")
    lmfit_dir = mcmc_io.get("lmfit_input_dir", "")
    out_dir   = mcmc_io.get("output_dir", "")

    filter_expression = mcmc_io.get("filter_expression", None)
    field_to_check    = mcmc_io.get("field_to_check", "bic")
    take_min_val      = mcmc_io.get("take_min", True)

    print("=" * 50)
    print("  Batch MCMC runner")
    print(f"  Config:          {cli.config}")
    print(f"  RV input dir:    {rv_dir}")
    print(f"  LMFIT input dir: {lmfit_dir}")
    print(f"  Output dir:      {out_dir}")
    print(f"  Filter:          {filter_expression}")
    print(f"  Field:           {field_to_check}  (min={take_min_val})")
    print("=" * 50)

    all_res = run_mcmc_batch(
        args_dict, rv_dir=rv_dir, lmfit_dir=lmfit_dir, out_dir=out_dir,
        filter_expression=filter_expression,
        field_to_check=field_to_check,
        take_min=take_min_val,
    )
    df = pd.DataFrame(all_res)
    df.to_csv(os.path.join(out_dir, "mcmc_params.csv"), index=False)
    print("\n=== Batch MCMC Completed ===")
