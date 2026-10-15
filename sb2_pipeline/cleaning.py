"""
sb2_pipeline.cleaning — post-fit per-epoch quality pass.

Compares each star's corrected RVs against its ADOPTED posterior-median model
(err-only normalisation — jitter would hide exactly the outliers we hunt) and
decides per epoch:
    keep            max(z1, z2) <= z_out
    swap            exchanging the two measured values brings both residuals
                    under z_ok (component mis-assignment, gamma-crossing type)
    mask comp i     z_i > z_out and the swap does not fix it (use_i -> False)

With --apply, writes data/rv_sb2_cleaned.csv (as-loaded labelling: values
swapped where the swap action fired, use1/use2 flags) and wipes the star's
fit/MCMC outputs so the next batch run refits it from the cleaned table.
Stars where more than --max-flag-frac of epochs would change are left
untouched and flagged for manual review.

Usage:
    python -m sb2_pipeline.cleaning --config configs/params_sb2_fixed_rvs.yaml
        [--z-out 5] [--z-ok 3] [--apply] [--stars 1-006 2-019 ...]
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from pipeline.config import load_args
from mcmc.chain_io import load_chain

from sb2_pipeline.constants import (
    SB2_IO, OUTPUT_DIR, MJD_COL, RV1, ERR1, RV2, ERR2,
    USE1, USE2, SWAP_APPLIED,
)
from sb2_pipeline.mcmc_model import sb2_rv_model_mcmc


def _adopted_model_curves(star_out, t):
    """
    (m1, m2) posterior-median model at times t, in CANONICAL labelling,
    plus the adopted tag. Uses the circular chain when it exists (it is only
    run when adopted), else the eccentric chain.
    """
    circ_npz = os.path.join(star_out, "mcmc", "circular", "chains_circ.npz")
    ecc_npz = os.path.join(star_out, "mcmc", "chains_ecc.npz")
    if os.path.exists(circ_npz):
        d = load_chain(circ_npz)
        med = np.median(d["flat_samples_pruned"], axis=0)
        P, T0, K1, K2, gamma = med[:5]
        m1, m2 = sb2_rv_model_mcmc(t, P, T0, np.pi / 2, 0.0, K1, K2, gamma)
        return m1, m2, "circ"
    d = load_chain(ecc_npz)
    med = np.median(d["flat_samples_pruned"], axis=0)
    P, T0, omega, e, K1, K2, gamma = med[:7]
    m1, m2 = sb2_rv_model_mcmc(t, P, T0, omega, e, K1, K2, gamma)
    return m1, m2, "ecc"


def detect_star(star_out, star_id, z_out=5.0, z_ok=3.0):
    """
    Per-epoch decisions for one completed star.

    Returns (decisions_df, cleaned_df) in the AS-LOADED labelling of
    data/rv_sb2_corrected.csv; cleaned_df is None when nothing changes.
    """
    corr_csv = os.path.join(star_out, "data", "rv_sb2_corrected.csv")
    summary_csv = os.path.join(star_out, "lmfit_summary.csv")
    df = pd.read_csv(corr_csv, comment="#")
    summary = pd.read_csv(summary_csv)
    best = summary[summary["is_best"]].iloc[0]
    k_swapped = bool(best.get("k_swapped", False))

    t = df[MJD_COL].to_numpy(float)
    m1c, m2c, tag = _adopted_model_curves(star_out, t)
    # Map canonical model curves back to the as-loaded component labelling.
    m1, m2 = (m2c, m1c) if k_swapped else (m1c, m2c)

    rv1 = df[RV1].to_numpy(float)
    err1 = df[ERR1].to_numpy(float)
    rv2 = df[RV2].to_numpy(float)
    err2 = df[ERR2].to_numpy(float)

    z1 = np.abs(rv1 - m1) / err1
    z2 = np.abs(rv2 - m2) / err2
    z1s = np.abs(rv1 - m2) / err1     # rv1 measured value against model 2
    z2s = np.abs(rv2 - m1) / err2     # rv2 measured value against model 1

    decisions = []
    cleaned = df.copy()
    cleaned[USE1] = True
    cleaned[USE2] = True
    cleaned[SWAP_APPLIED] = False
    changed = False
    for k in range(len(df)):
        zmax = max(z1[k], z2[k])
        action = "keep"
        if zmax > z_out:
            if max(z1s[k], z2s[k]) < min(z_ok, zmax):
                action = "swap"
                cleaned.loc[k, [RV1, RV2]] = [rv2[k], rv1[k]]
                cleaned.loc[k, [ERR1, ERR2]] = [err2[k], err1[k]]
                cleaned.loc[k, SWAP_APPLIED] = True
                changed = True
            else:
                bits = []
                if z1[k] > z_out:
                    cleaned.loc[k, USE1] = False
                    bits.append("mask1")
                if z2[k] > z_out:
                    cleaned.loc[k, USE2] = False
                    bits.append("mask2")
                action = "+".join(bits)
                changed = True
        decisions.append({
            "star_name": star_id, "epoch": df["epoch"].iloc[k],
            "mjd": t[k], "action": action, "model": tag,
            "z1": z1[k], "z2": z2[k], "z1_swap": z1s[k], "z2_swap": z2s[k],
        })
    return pd.DataFrame(decisions), (cleaned if changed else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", required=True)
    ap.add_argument("--z-out", type=float, default=5.0)
    ap.add_argument("--z-ok", type=float, default=3.0)
    ap.add_argument("--max-flag-frac", type=float, default=0.2,
                    help="Refuse to auto-clean a star when more than this "
                         "fraction of epochs would change.")
    ap.add_argument("--apply", action="store_true",
                    help="Write cleaned CSVs and wipe fit/MCMC outputs of "
                         "changed stars (next batch run refits them).")
    ap.add_argument("--redetect", action="store_true",
                    help="Re-evaluate stars that already have a cleaned CSV "
                         "against the CURRENT adopted model (iterative "
                         "cleaning: a first-pass model contaminated by the "
                         "very outliers/swaps it should flag mis-classifies "
                         "them; the post-refit model does not). Stars whose "
                         "decisions are unchanged are left untouched.")
    ap.add_argument("--stars", nargs="*", default=None,
                    help="Optional star id substrings to restrict to.")
    cli = ap.parse_args()

    cfg = load_args(cli.config)
    out_root = cfg[SB2_IO][OUTPUT_DIR]

    all_dec, to_refit, review = [], [], []
    star_dirs = sorted(
        d for d in os.listdir(out_root)
        if os.path.exists(os.path.join(out_root, d, "mcmc", "star_done.json"))
    )
    if cli.stars:
        star_dirs = [d for d in star_dirs
                     if any(p in d for p in cli.stars)]

    for star_id in star_dirs:
        star_out = os.path.join(out_root, star_id)
        cleaned_csv = os.path.join(star_out, "data", "rv_sb2_cleaned.csv")
        prev_cleaned = None
        if os.path.exists(cleaned_csv):
            if not cli.redetect:
                print(f"{star_id}: already cleaned — skipping detection "
                      f"(use --redetect to iterate)")
                continue
            prev_cleaned = pd.read_csv(cleaned_csv, comment="#")
        try:
            dec, cleaned = detect_star(star_out, star_id,
                                       z_out=cli.z_out, z_ok=cli.z_ok)
        except Exception as e:
            print(f"{star_id}: detection failed: {e}")
            continue
        all_dec.append(dec)
        flagged = dec[dec.action != "keep"]
        if prev_cleaned is not None:
            # Iteration convergence check: unchanged decisions -> stable.
            new_state = (cleaned if cleaned is not None
                         else prev_cleaned.assign(**{USE1: True, USE2: True,
                                                     SWAP_APPLIED: False}))
            same = (
                prev_cleaned[USE1].astype(bool).equals(new_state[USE1].astype(bool))
                and prev_cleaned[USE2].astype(bool).equals(new_state[USE2].astype(bool))
                and prev_cleaned[SWAP_APPLIED].astype(bool).equals(
                    new_state[SWAP_APPLIED].astype(bool))
            )
            if same:
                print(f"{star_id}: decisions stable under current model")
                continue
            print(f"{star_id}: decisions CHANGED under current model")
            if cleaned is None:
                # Everything reverts to keep: drop the cleaned CSV entirely.
                if cli.apply:
                    os.remove(cleaned_csv)
                    for sub in ("lmfit_summary.csv",):
                        p = os.path.join(star_out, sub)
                        if os.path.exists(p):
                            os.remove(p)
                    for sub in ("mcmc", "periodogram", "lmfit_solutions"):
                        p = os.path.join(star_out, sub)
                        if os.path.isdir(p):
                            shutil.rmtree(p)
                    print(f"{star_id}: cleaning reverted, outputs wiped")
                to_refit.append(star_id)
                continue
        if flagged.empty:
            continue
        frac = len(flagged) / len(dec)
        print(f"\n{star_id}: {len(flagged)}/{len(dec)} epochs flagged "
              f"(frac {frac:.2f})")
        print(flagged[["epoch", "mjd", "action", "z1", "z2",
                       "z1_swap", "z2_swap"]].round(2).to_string(index=False))
        if frac > cli.max_flag_frac:
            print(f"{star_id}: exceeds --max-flag-frac "
                  f"({cli.max_flag_frac}) -> MANUAL REVIEW, not cleaned")
            review.append(star_id)
            continue
        to_refit.append(star_id)
        if cli.apply and cleaned is not None:
            with open(cleaned_csv, "w") as f:
                f.write("# SB2 cleaned per-epoch RVs (as-loaded labelling)\n")
                f.write(f"# Star: {star_id}\n")
                f.write(f"# Actions: {dict(flagged.action.value_counts())} "
                        f"(z_out={cli.z_out}, z_ok={cli.z_ok})\n")
                f.write("#\n")
                cleaned.to_csv(f, index=False)
            for sub in ("lmfit_summary.csv",):
                p = os.path.join(star_out, sub)
                if os.path.exists(p):
                    os.remove(p)
            for sub in ("mcmc", "periodogram", "lmfit_solutions"):
                p = os.path.join(star_out, sub)
                if os.path.isdir(p):
                    shutil.rmtree(p)
            print(f"{star_id}: cleaned CSV written, outputs wiped for refit")

    if all_dec:
        report = pd.concat(all_dec, ignore_index=True)
        report_path = os.path.join(out_root, "sb2_cleaning_report.csv")
        report.to_csv(report_path, index=False)
        print(f"\nSaved {report_path}")
    print(f"\nstars to refit: {to_refit}")
    print(f"manual review (too many flags): {review}")


if __name__ == "__main__":
    main()
