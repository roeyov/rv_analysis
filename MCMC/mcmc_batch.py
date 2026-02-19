#!/usr/bin/env python3
"""
Batch wrapper for MCMC4.py

Usage examples:
    python run_MCMC4_batch.py /path/to/csv_dir

    # With custom pattern
    python run_MCMC4_batch.py /path/to/csv_dir --pattern "*_CCF_RVs.csv"

    # With a root output folder
    python run_MCMC4_batch.py /path/to/csv_dir --output-root all_outputs

    # Using pre-computed lmfit solutions from a CSV
    # (only stars present in solutions CSV will be run,
    #  and the code will start at the MCMC stage)
    python run_MCMC4_batch.py /path/to/csv_dir \
        --solutions-csv solutions.csv
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import MCMC4 as m4
from utils.constants import *


def extract_star_name_from_path(path: str) -> str:
    """
    Extract star_name from a path like ".../BLOeM_6-032_CCF_RVs.csv" → "BLOeM_6-032".

    If the filename does not end with "_CCF_RVs.csv", fall back to the stem.
    """
    base = os.path.basename(path)
    suffix = "_CCF_RVs.csv"
    if base.endswith(suffix):
        return base[:-len(suffix)]
    return os.path.splitext(base)[0]


def build_initial_from_solution_row(row: pd.Series, mode: str = "ecc") -> np.ndarray:
    """
    Build an initial parameter vector from a solutions CSV row.

    Expects columns:
        P_value, T0_value, K1_value, gamma_value
    And for eccentric mode also:
        omega_value, e_value

    mode = "ecc":  initial = [P, T0, omega, e, K1, gamma]
    mode = "circ": initial = [P, T0, K1, gamma]
    """
    mode = mode.lower()
    if mode not in ("ecc", "circ"):
        raise ValueError("mode must be 'ecc' or 'circ'")

    if mode == "ecc":
        param_order = [PERIOD,T,OMEGA, ECC,K1_STR,GAMMA]
    else:
        param_order = [PERIOD,T,K1_STR,GAMMA]

    vals = []
    for p in param_order:
        col = f"{p}_value"
        if col not in row.index:
            raise KeyError(f"Missing column '{col}' in solutions CSV for star {row.get('star_name', '?')}")
        vals.append(float(row[col]))

    return np.array(vals, dtype=float)


def run_single_file(
    csv_path: str,
    base_output_dir: str | None = None,
    make_periodogram_plots: bool = False,
    solution_row: pd.Series | None = None,
):
    """
    Run the full MCMC4 pipeline on a single CSV file.

    If solution_row is None:
        - Follow the original pipeline: LS period search + lmfit + MCMC.
    If solution_row is given:
        - Only run stars that have a row in solutions CSV.
        - Skip LS & lmfit, and start directly from MCMC using the provided solution.
    """
    star_name = extract_star_name_from_path(csv_path)

    print("\n" + "=" * 70)
    print(f"Processing star: {star_name}")
    print(f"File: {csv_path}")
    if solution_row is not None:
        print("Using pre-computed lmfit solution from solutions CSV (starting at MCMC stage).")
    else:
        print("Deriving solution from RVs (LS + lmfit + MCMC).")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Configure MCMC4 module-level globals for this run
    # ------------------------------------------------------------------
    m4.MODE = "file"
    m4.star_name = star_name
    m4.INPUT_CSV = csv_path

    if base_output_dir is None:
        out_dir = f"output_{star_name}"
    else:
        out_dir = os.path.join(base_output_dir, f"output_{star_name}")

    m4.OUTPUT_DIR = out_dir
    os.makedirs(m4.OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load RV data
    # ------------------------------------------------------------------
    MJDs, rv_obs, rv_sigmas = m4.load_rvs_from_csv(csv_path)
    have_truth = False  # we're in "file" mode here

    # ------------------------------------------------------------------
    # Get initial solution: either from RVs (LS + lmfit) or from solutions CSV
    # ------------------------------------------------------------------
    if solution_row is None:
        # ---- Original path: LS + lmfit ----
        P_guess, _, _ = m4.ls_best_period(
            MJDs,
            rv_obs,
            rv_sigmas,
            Pmin=m4.Pmin,
            Pmax=m4.Pmax,
            plot=make_periodogram_plots,
        )

        initial_ecc, lmfit_result = m4.lmfit_initial_guess(
            MJDs,
            rv_obs,
            rv_sigmas,
            P_guess,
            mode="ecc",
            P_window_frac=0.05,
        )

        e_lm       = lmfit_result.params["e"].value
        sigma_e_lm = lmfit_result.params["e"].stderr
    else:
        # ---- New path: use precomputed lmfit solution ----
        # Build initial_ecc directly from the solutions row
        initial_ecc = build_initial_from_solution_row(solution_row, mode="ecc")
        # Extract e and its uncertainty from the same row
        if f"{ECC}_value" not in solution_row.index or f"{ECC}_stderr" not in solution_row.index:
            raise KeyError(f"Solutions CSV must contain '{ECC}_value' and '{ECC}_stderr' for star {star_name}")
        e_lm       = float(solution_row[f"{ECC}_value"])
        sigma_e_lm = float(solution_row[f"{ECC}_stderr"])

    ecc_significant = m4.lucy_sweeney_significant(e_lm, sigma_e_lm, threshold=2.45)
    print(f"Lucy–Sweeney says eccentricity significant? {ecc_significant}")

    P_center  = initial_ecc[0]
    T0_center = initial_ecc[1]

    # ------------------------------------------------------------------
    # Eccentric MCMC
    # ------------------------------------------------------------------
    flat_ecc, logp_ecc, sampler_ecc = m4.run_mcmc_ecc(
        MJDs,
        rv_obs,
        rv_sigmas,
        initial_ecc,
        P_center=P_center,
        T0_center=T0_center,
        dT0_days=P_center / 2.0,
        dP_frac=0.01,
        nwalkers=32,
        nsteps=m4.Nsteps_MCMC,
        nburn=m4.Nburn_MCMC,
        thin=10,
        progress=True,
    )

    print("Posterior samples (ecc):", flat_ecc.shape)

    truths_ecc = None  # no true orbit in file mode
    labels_ecc = [
        r"$P$ [d]",
        r"$T_0$ [MJD]",
        r"$\omega$ [deg]",
        r"$e$",
        r"$K_1$ [km/s]",
        r"$\gamma$ [km/s]",
    ]

    m4.make_corner(
        flat_ecc,
        logp_ecc,
        truths_ecc,
        labels_ecc,
        tag="ecc",
        omega_in_col2=True,
    )
    m4.summarise_chain(
        flat_ecc,
        [PERIOD,T,OMEGA, ECC,K1_STR,GAMMA],
        tag="ecc",
    )

    m4.plot_orbit_with_band_phase(
        MJDs,
        rv_obs,
        rv_sigmas,
        flat_ecc,
        logp_ecc,
        truths=truths_ecc,
        tag="ecc",
        circular=False,
    )

    # ------------------------------------------------------------------
    # Circular MCMC (if e is NOT significant)
    # ------------------------------------------------------------------
    if not ecc_significant:
        print("Eccentricity not significant; running circular MCMC.")
        print("Using circular parameters (e=0, ω=90°).")

        if solution_row is None:
            # Original path: derive a circular lmfit solution from data
            initial_circ, lmfit_result_circ = m4.lmfit_initial_guess(
                MJDs,
                rv_obs,
                rv_sigmas,
                P_guess=P_center,
                mode="circ",
                P_window_frac=0.05,
            )
        else:
            # New path: build circular initial guess directly from solutions row
            initial_circ = build_initial_from_solution_row(solution_row, mode="circ")

        P_center_circ  = initial_circ[0]
        T0_center_circ = initial_circ[1]

        flat_circ, logp_circ, sampler_circ = m4.run_mcmc_circ(
            MJDs,
            rv_obs,
            rv_sigmas,
            initial_circ,
            P_center=P_center_circ,
            T0_center=T0_center_circ,
            dT0_days=P_center_circ / 2.0,
            dP_frac=0.01,
            nwalkers=32,
            nsteps=m4.Nsteps_MCMC_circ,
            nburn=m4.Nburn_MCMC_circ,
            thin=10,
            progress=True,
        )

        print("Posterior samples (circ):", flat_circ.shape)

        truths_circ = None
        labels_circ = [
            r"$P$ [d]",
            r"$T_0$ [MJD]",
            r"$K_1$ [km/s]",
            r"$\gamma$ [km/s]",
        ]

        m4.make_corner(
            flat_circ,
            logp_circ,
            truths_circ,
            labels_circ,
            tag="circ",
            omega_in_col2=False,
        )
        m4.summarise_chain(
            flat_circ,
            [PERIOD,T,K1_STR,GAMMA],
            tag="circ",
        )

        m4.plot_orbit_with_band_phase(
            MJDs,
            rv_obs,
            rv_sigmas,
            flat_circ,
            logp_circ,
            truths=truths_circ,
            tag="circ",
            circular=True,
        )
    else:
        print("Skipping circular MCMC: Lucy–Sweeney says eccentricity is significant.")

    print(f"Finished star: {star_name}")
    return flat_ecc, logp_ecc


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for MCMC4.py over *_CCF_RVs.csv files."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing *_CCF_RVs.csv files.",
    )
    parser.add_argument(
        "--pattern",
        default="*_CCF_RVs.csv",
        help="Glob pattern for input CSV files (default: '*_CCF_RVs.csv').",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional root output directory. If given, per-star outputs "
             "will be under OUTPUT_ROOT/output_{star_name}.",
    )
    parser.add_argument(
        "--no-periodogram",
        action="store_true",
        help="Disable periodogram plot (only relevant when NOT using solutions CSV).",
    )
    parser.add_argument(
        "--solutions-csv",
        default=None,
        help=(
            "Path to a CSV containing pre-computed lmfit solutions. "
            "Must have a 'star_name' column and columns like P_value, P_stderr, "
            "T0_value, T0_stderr, omega_value, omega_stderr, e_value, e_stderr, "
            "K1_value, K1_stderr, gamma_value, gamma_stderr. "
            "If given, only stars present in this CSV will be processed and "
            "the pipeline will start at the MCMC stage using these solutions."
        ),
    )

    args = parser.parse_args()

    pattern_path = os.path.join(args.input_dir, args.pattern)
    csv_files = sorted(glob.glob(pattern_path))

    if not csv_files:
        print(f"No files found matching: {pattern_path}")
        return

    solutions_df = None
    if args.solutions_csv is not None:
        if not os.path.isfile(args.solutions_csv):
            print(f"Solutions CSV not found: {args.solutions_csv}")
            return
        solutions_df = pd.read_csv(args.solutions_csv)
        if "star_name" not in solutions_df.columns:
            raise KeyError("Solutions CSV must contain a 'star_name' column.")
        # Index by star_name for quick lookup
        solutions_df["star_name"] = solutions_df["star_name"].astype(str).str.strip()
        solutions_df = solutions_df.set_index("star_name")

    print(f"Found {len(csv_files)} files.")

    for csv_path in csv_files:
        star_name = extract_star_name_from_path(csv_path)

        sol_row = None
        if solutions_df is not None:
            if star_name not in solutions_df.index:
                print(f"Skipping {star_name}: no row in solutions CSV.")
                continue
            sol_row = solutions_df.loc[star_name]

        run_single_file(
            csv_path,
            base_output_dir=args.output_root,
            make_periodogram_plots=(solutions_df is None and not args.no_periodogram),
            solution_row=sol_row,
        )


if __name__ == "__main__":
    main()
