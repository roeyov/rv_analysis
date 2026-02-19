from utils.constants import *
import numpy as np
from compare_best_row_selectors2 import get_best_row
import json
import os
from pathlib import Path


# Small helper: read param_value or fall back to plain param column
def get_param(row, par_name, kind="value", default=np.nan):
    """
    par_name: e.g. PERIOD, ECC, ...
    kind: 'value' or 'stderr'
    Looks first for f'{par_name}_{kind}', then for par_name.
    """
    col1 = f"{par_name}_{kind}"
    if col1 in row.index:
        return row[col1]
    if par_name in row.index:
        return row[par_name]
    return default

def run_mcmc_null_full(args_dict, best_row, MJDs, rv_obs, rv_sigmas, out_dir, star_name=""):
    """Run MCMC for a single lmfit solution row (best_row is a pandas.Series)."""
    from MCMC.MCMC4 import (
        make_corner,
        summarise_chain,
        run_mcmc_null
    )

    add_jitter = True

    # ------------------------------------------------------------------
    # Build initial vectors DIRECTLY from best_row columns
    # ------------------------------------------------------------------
    gamma_val  = float(get_param(best_row, GAMMA,  kind="value"))

    initial_ecc = np.array([
        gamma_val,
    ])

    # --------------------------
    # Eccentric MCMC
    # --------------------------
    flat_null, logp_null, sampler_null = run_mcmc_null(
        MJDs, rv_obs, rv_sigmas,
        initial_ecc,
        nwalkers=args_dict[MCMC_PARAMS][WALKERS],
        nsteps=args_dict[MCMC_PARAMS][STEPS],
        nburn=args_dict[MCMC_PARAMS][BURN],
        thin=args_dict[MCMC_PARAMS][THIN],
        progress=True,
        add_jitter=add_jitter,
    )

    print("Posterior samples (null):", flat_null.shape)

    truths_null = None
    truths_phase_null = None

    labels_ecc = [r"$\gamma$ [km/s]"]
    if add_jitter:
        labels_ecc.append(r"log σ_jit")

    make_corner(
        flat_null, logp_null,
        truths_null, labels_ecc,
        tag="null", omega_in_col2=True,
        out_dir=out_dir, star_name=star_name
    )

    chain_names = ["gamma"]
    if add_jitter:
        chain_names.append("log_sj")

    dict_row = summarise_chain(
        flat_null,
        chain_names,
        tag="null", out_dir=out_dir,
        add_jitter=add_jitter,
    )
    return dict_row



def run_mcmc(args_dict, best_row, MJDs, rv_obs, rv_sigmas, out_dir, star_name=""):
    """Run MCMC for a single lmfit solution row (best_row is a pandas.Series)."""
    from MCMC.MCMC4 import (
        lucy_sweeney_significant,
        run_mcmc_ecc,
        make_corner,
        summarise_chain,
        plot_orbit_with_band_phase,
        run_mcmc_circ,

    )
    add_jitter = True

    # ------------------------------------------------------------------
    # Build initial vectors DIRECTLY from best_row columns
    # ------------------------------------------------------------------
    P_val      = float(get_param(best_row, PERIOD, kind="value"))
    T0_val     = float(get_param(best_row, T,      kind="value"))
    omega_val  = float(get_param(best_row, OMEGA,  kind="value"))
    ecc_val    = float(get_param(best_row, ECC,    kind="value"))
    K1_val     = float(get_param(best_row, K1_STR, kind="value"))
    gamma_val  = float(get_param(best_row, GAMMA,  kind="value"))

    initial_ecc = np.array([
        P_val,
        T0_val,
        omega_val,
        ecc_val,
        K1_val,
        gamma_val,
    ])

    P_center  = initial_ecc[0]
    T0_center = initial_ecc[1]

    # --------------------------
    # Eccentric MCMC
    # --------------------------
    flat_ecc, logp_ecc, sampler_ecc = run_mcmc_ecc(
        MJDs, rv_obs, rv_sigmas,
        initial_ecc,
        P_center=P_center,
        T0_center=T0_center,
        dT0_days=P_center / 2.0,
        dP_frac=0.01,
        nwalkers=args_dict[MCMC_PARAMS][WALKERS],
        nsteps=args_dict[MCMC_PARAMS][STEPS],
        nburn=args_dict[MCMC_PARAMS][BURN],
        thin=args_dict[MCMC_PARAMS][THIN],
        progress=True,
        add_jitter=add_jitter,
    )

    print("Posterior samples (ecc):", flat_ecc.shape)

    truths_ecc = None
    truths_phase_ecc = None

    labels_ecc = [r"$P$ [d]", r"$T_0$ [MJD]", r"$\omega$ [deg]",
                  r"$e$", r"$K_1$ [km/s]", r"$\gamma$ [km/s]"]
    if add_jitter:
        labels_ecc.append(r"log σ_jit")

    make_corner(
        flat_ecc, logp_ecc,
        truths_ecc, labels_ecc,
        tag="ecc", omega_in_col2=True,
        out_dir=out_dir, star_name=star_name
    )

    chain_names = ["P", "T0", "omega", "e", "K1", "gamma"]
    if add_jitter:
        chain_names.append("log_sj")

    dict_row = summarise_chain(
        flat_ecc,
        chain_names,
        tag="ecc", out_dir=out_dir,
        add_jitter=add_jitter,
    )

    plot_orbit_with_band_phase(
        MJDs, rv_obs, rv_sigmas,
        flat_ecc, logp_ecc,
        truths=truths_phase_ecc,
        tag="ecc",
        circular=False,
        out_dir=out_dir,
        add_jitter=add_jitter,
        star_name=star_name,
    )
    # Lucy–Sweeney significance test using e and σ_e from the row
    e_lm       = dict_row["ecc_e_mode"]
    sigma_e_lm = dict_row["ecc_e_errm"]

    ecc_significant = lucy_sweeney_significant(
        e_lm, sigma_e_lm,
        threshold=2.45,
        add_jitter=add_jitter
    )
    print(f"Lucy–Sweeney says eccentricity significant? {ecc_significant}")
    # --------------------------
    # Circular MCMC (if e NOT significant)
    # --------------------------
    if not ecc_significant:
        initial_circ = np.array([
            P_val,
            T0_val,
            K1_val,
            gamma_val,
        ])

        flat_circ, logp_circ, sampler_circ = run_mcmc_circ(
            MJDs, rv_obs, rv_sigmas,
            initial_circ,
            P_center=P_center,
            T0_center=T0_center,
            dT0_days=P_center / 2.0,
            dP_frac=0.01,
            nwalkers=args_dict[MCMC_PARAMS][WALKERS],
            nsteps=args_dict[MCMC_PARAMS][STEPS],
            nburn=args_dict[MCMC_PARAMS][BURN],
            thin=args_dict[MCMC_PARAMS][THIN],
            progress=True,
            add_jitter=add_jitter,
        )

        print("Posterior samples (circ):", flat_circ.shape)

        truths_circ = None
        truths_phase_circ = None

        labels_circ = [r"$P$ [d]", r"$T_0$ [MJD]",
                       r"$K_1$ [km/s]", r"$\gamma$ [km/s]"]
        if add_jitter:
            labels_circ.append(r"log σ_jit")

        make_corner(
            flat_circ, logp_circ,
            truths_circ, labels_circ,
            tag="circ", omega_in_col2=False,
            out_dir=out_dir, star_name=star_name
        )

        chain_names = ["P", "T0", "K1", "gamma"]
        if add_jitter:
            chain_names.append("log_sj")

        circ_row = summarise_chain(
            flat_circ,
            ["P", "T0", "K1", "gamma"],
            tag="circ", out_dir=out_dir,
            add_jitter=add_jitter,
        )
        dict_row.update(circ_row)
        plot_orbit_with_band_phase(
            MJDs, rv_obs, rv_sigmas,
            flat_circ, logp_circ,
            truths=truths_phase_circ,
            tag="circ",
            circular=True,
            out_dir=out_dir,
            add_jitter=add_jitter,
            star_name=star_name,
        )
    else:
        print("Skipping circular MCMC: Lucy–Sweeney says eccentricity is significant.")
    return dict_row

import os
import glob
import numpy as np
import pandas as pd

# Adjust these to your actual paths
INPUT_RV_DIR      = "/input/dir"
INPUT_LMFIT_DIR   = "/another/input/dir"
OUTPUT_MCMC_DIR   = "/out/dir"

# Adjust these if your RV CSV uses different column names
MJD_COL    = "MJD"
RV_COL     = 'Mean RV'
RVERR_COL  =  'Mean RVsig'

def load_rv_csv(rv_path):
    """Load a single RV CSV and return MJDs, RVs, RV errors."""
    df = pd.read_csv(rv_path)
    df = df.dropna(subset=[MJD_COL, RV_COL, RVERR_COL])
    MJDs      = df[MJD_COL].values.astype(float)
    rv_obs    = df[RV_COL].values.astype(float)
    rv_sigmas = df[RVERR_COL].values.astype(float)
    return MJDs, rv_obs, rv_sigmas

def save_choose_params_file(out_dir, choose_params):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    choose_param_path = out_path / "choose_param.json"
    with open(choose_param_path, "w", encoding="utf-8") as f:
        json.dump(choose_params, f, indent=2, ensure_ascii=False)
        f.write("\n")  # nice POSIX newline at EOF

    print(f"Saved chooser params to: {choose_param_path}")


def run_mcmc_batch(args_dict,
                   rv_dir=INPUT_RV_DIR,
                   lmfit_dir=INPUT_LMFIT_DIR,
                   out_dir=OUTPUT_MCMC_DIR,
                   wanted_periods=None):
    """
    Loop over RV files of the form:
        rv_dir/{star_name}_CCF_RVs.csv
    and lmfit summaries of the form:
        lmfit_dir/{star_name}/lmfit_summary.csv

    If wanted_periods is None:
        - If a valid row exists in the lmfit CSV (has_valid_row),
          choose best_row = get_best_row_no_1day_aliasing(results_df)
          and run MCMC once per star.

    If wanted_periods is a dict {star_name: [P1, P2, ...]}:
        - Only stars that appear as keys are processed.
        - For each requested period P_target, pick the row whose `period`
          is closest to P_target and run MCMC for that row.
    """
    from pathlib import Path

    rv_pattern = os.path.join(rv_dir, "*_CCF_RVs.csv")
    rv_files = sorted(glob.glob(rv_pattern))

    print(f"Found {len(rv_files)} RV files to scan in {rv_dir}")

    filter_expression = (
            "~candidate_method.str.contains('MANUAL')"
            "& ~candidate_method.str.contains('null')"
            "& candidate_method.str.contains('jitter')"
            "& (prob_bicc > 0.5)"
            "& (mass_flag_peri == 0)"
    )

    field_to_check = "bic"

    take_min = True

    # filter_expression = (
    #     "candidate_method.str.contains('jitter')"
    #     "& (bin_flag>0)"
    #     "& (mass_flag_peri == 0)"
    # )
    # field_to_check = "bic"
    # take_min = True


    choose_params = {
        "filter_expression": filter_expression,
        "field_to_check": field_to_check,
        "take_min": take_min,
    }
    save_choose_params_file(out_dir, choose_params)
    rows = []
    for rv_path in rv_files:
        star_name = os.path.basename(rv_path).replace("_CCF_RVs.csv", "")
        #for debuuging:
        # if star_name not in ['BLOeM_8-020'] :
        #     continue
        if wanted_periods and star_name not in wanted_periods.keys():
            continue
        print(f"\n=== Processing star: {star_name} ===")

        # If we have a wanted_periods dict, skip stars not in it
        if wanted_periods and star_name not in wanted_periods:
            print(f"  Star not in wanted_periods dict, skipping.")
            continue

        # lmfit summary path
        lmfit_path = os.path.join(lmfit_dir, star_name, "lmfit_summary.csv")
        if not os.path.exists(lmfit_path):
            print(f"  No lmfit_summary.csv found at {lmfit_path}, skipping.")
            continue

        # Load lmfit summary
        try:
            results_df = pd.read_csv(lmfit_path)
        except Exception as e:
            print(f"  Failed to read {lmfit_path}: {e}, skipping.")
            continue

        if results_df.empty:
            print(f"  lmfit_summary.csv for {star_name} is empty, skipping.")
            continue

        # Load RV data
        try:
            MJDs, rv_obs, rv_sigmas = load_rv_csv(rv_path)
        except Exception as e:
            print(f"  Failed to load RVs from {rv_path}: {e}, skipping.")
            continue

        # choose best row here (instead of inside run_mcmc)
        best_row = get_best_row(results_df,filter_expression,field_to_check,take_min)
        # Output dir per star
        star_out_dir = os.path.join(out_dir, star_name)
        Path(star_out_dir).mkdir(parents=True, exist_ok=True)
        if best_row is None:
            null_row = results_df[results_df.candidate_method.str.contains('null_hyp_jitter')].iloc[0]
            row_res = run_mcmc_null_full(
                args_dict=args_dict,
                best_row=null_row,
                MJDs=MJDs,
                rv_obs=rv_obs,
                rv_sigmas=rv_sigmas,
                out_dir=star_out_dir,
                star_name=star_name,
            )
            row_res["star_name"] = star_name
            rows.append(row_res)
            print(f"  Done. Results saved under {star_out_dir}")
            continue
        print(f"  Running MCMC for {star_name} using best row (period={best_row.get(PERIOD+"_value", np.nan)}).")
        try:
            row_res = run_mcmc(
                args_dict=args_dict,
                best_row=best_row,
                MJDs=MJDs,
                rv_obs=rv_obs,
                rv_sigmas=rv_sigmas,
                out_dir=star_out_dir,
                star_name=star_name,
            )
            row_res["star_name"] = star_name
            rows.append(row_res)
            print(f"  Done. Results saved under {star_out_dir}")
        except Exception as e:
            print(f"  MCMC failed for {star_name}: {e}")
        continue  # move to next star
    return rows



if __name__ == "__main__":

    # --------------------------------------------------------
    # Explicit paths (EDIT THESE)
    # --------------------------------------------------------
    json_param_file = '/Users/roeyovadia/Roey/Masters/Reasearch/Scripts/params.json'   # contains ONLY args_dict
    RV_INPUT_DIR = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_neb_div_from_coadded/"            # contains {star}_CCF_RVs.csv
    LMFIT_INPUT_DIR = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_neb_div_from_coadded/second/" # contains {star}/lmfit_summary.csv
    OUTPUT_DIR = LMFIT_INPUT_DIR+ "/mcmc_min_withNull/"                # output goes here

    # --------------------------------------------------------
    # Optional: map star_name -> list of desired periods
    # If None or empty, behavior = old behavior (use best row for all stars)
    # If not None, only these stars are executed, and for each period in the list
    # the row with the closest period in the CSV is used.
    # --------------------------------------------------------
    WANTED_PERIODS = {
        # examples:
        # "BLOeM_4-076": [106.0],
        # "BLOeM_6-067": [5.0],
    }
    # If you want to completely disable this behavior, set:
    # WANTED_PERIODS = None

    # --------------------------------------------------------
    # Load args_dict directly from JSON
    # --------------------------------------------------------
    if not os.path.exists(json_param_file):
        raise FileNotFoundError(f"Parameter JSON file not found: {json_param_file}")

    with open(json_param_file, "r") as f:
        args_dict = json.load(f)

    print("=== Loaded args_dict from JSON ===")
    print(args_dict)
    print()

    print("=== Explicit I/O configuration ===")
    print(f"  RV input dir:      {RV_INPUT_DIR}")
    print(f"  LMFIT input dir:   {LMFIT_INPUT_DIR}")
    print(f"  Output dir:        {OUTPUT_DIR}")
    print()

    # --------------------------------------------------------
    # Run batch MCMC
    # --------------------------------------------------------
    all_res = run_mcmc_batch(
        args_dict=args_dict,
        rv_dir=RV_INPUT_DIR,
        lmfit_dir=LMFIT_INPUT_DIR,
        out_dir=OUTPUT_DIR,
        wanted_periods=WANTED_PERIODS,   # <--- new argument
    )
    df = pd.DataFrame(all_res)
    df.to_csv(OUTPUT_DIR + "mcmc_params.csv", index=False)
    print("\n=== Batch MCMC Completed ===")
