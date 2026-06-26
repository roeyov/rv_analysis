"""
run_mcmc_non_rl.py — MCMC for "non-RL" systems (mass-flag-excluded solutions).

Runs the MCMC stage ONLY (no CCF, no pipeline) for the subset of stars that:
  * have NO solution passing the STRICT filter (includes mass_flag_peri == 0), but
  * DO have a solution passing the RELAXED filter (same minus mass_flag_peri == 0).

Outputs land under .../second_pdc/mcmc_out_non_rl/ . The RELAXED filter drives the
best-row selection that feeds MCMC.

Run from the Scripts/ dir (so `pipeline`/`mcmc` import) in the `tau_binary` env:
    python run_mcmc_non_rl.py
"""

import os
import glob

import pandas as pd

from pipeline.config import load_args
from mcmc.batch import run_mcmc_batch
from mcmc.selector_app import get_best_row

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_neb_div_from_coadded"
LMFIT = os.path.join(BASE, "second_pdc")
OUT = os.path.join(LMFIT, "mcmc_out_non_rl")

# ---------------------------------------------------------------------------
# Filter expressions
# ---------------------------------------------------------------------------
STRICT = (
    "~candidate_method.str.contains('MANUAL')"
    " & ~candidate_method.str.contains('null')"
    " & candidate_method.str.contains('jitter')"
    " & (prob_bicc > 0.5)"
    " & (mass_flag_peri == 0)"
    " & ((candidate_method.str.contains('LS') & (LS_iter_fap < 0.2))"
    " | (candidate_method.str.contains('PDC') & (PDC_iter_fap < 0.2)))"
)

RELAXED = (
    "~candidate_method.str.contains('MANUAL')"
    " & ~candidate_method.str.contains('null')"
    " & candidate_method.str.contains('jitter')"
    " & (prob_bicc > 0.5)"
    " & ((candidate_method.str.contains('LS') & (LS_iter_fap < 0.2))"
    " | (candidate_method.str.contains('PDC') & (PDC_iter_fap < 0.2)))"
)

FIELD = "bicc"
TAKE_MIN = True


def main():
    args_dict = load_args("configs/params.yaml")  # walker/step settings only

    # ---- Build the subset: relaxed-pass but strict-fail ----
    subset = []
    lmfit_csvs = sorted(glob.glob(os.path.join(LMFIT, "BLOeM_*", "lmfit_summary.csv")))
    print(f"Scanning {len(lmfit_csvs)} lmfit_summary.csv files in {LMFIT}\n")

    for csv_path in lmfit_csvs:
        star = os.path.basename(os.path.dirname(csv_path))
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [skip] {star}: failed to read csv ({e})")
            continue
        if df.empty:
            continue

        strict = get_best_row(df, STRICT, FIELD, TAKE_MIN)
        relaxed = get_best_row(df, RELAXED, FIELD, TAKE_MIN)

        if strict is None and relaxed is not None:
            subset.append(star)
            print(f"  [non-RL] {star}  (relaxed best bicc={relaxed.get('bicc'):.2f},"
                  f" mass_flag_peri={relaxed.get('mass_flag_peri')})")

    print(f"\n=== {len(subset)} non-RL systems selected ===")
    for s in subset:
        print(f"   {s}")

    if not subset:
        print("\nNo systems matched. Nothing to run.")
        return

    # ---- Run MCMC for the subset (RELAXED filter drives best-row choice) ----
    print(f"\nLaunching MCMC -> {OUT}\n")
    all_res = run_mcmc_batch(
        args_dict,
        rv_dir=BASE,
        lmfit_dir=LMFIT,
        out_dir=OUT,
        wanted_periods=subset,
        filter_expression=RELAXED,
        field_to_check=FIELD,
        take_min=TAKE_MIN,
    )

    df_res = pd.DataFrame(all_res)
    out_csv = os.path.join(OUT, "mcmc_params.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nDone. Summary -> {out_csv}")


if __name__ == "__main__":
    main()
