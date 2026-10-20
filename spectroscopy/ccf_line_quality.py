"""
spectroscopy.ccf_line_quality — per-line quality of a CCF run, and epochs to
check by eye before fitting.

Reads every ``*_CCF_RVs.csv`` in a directory (normally CCF pass 2) and
reports, per line, the fraction of epochs with each ``mean_rv`` flag

    0 used, 1 sigma-clipped, 2 missing value, 3 EW consistent with zero
    (EW - 4 EWsig <= 0), 4 epoch S/N < 5

plus the median RV error, the RMS of (line RV - Mean RV) over used epochs,
and the slope of line RV vs Mean RV for stars whose Mean RV varies by more
than ``--min-range`` km/s (a line that follows the orbit has slope ~1).

It then lists epochs that still have a Mean RV but where fewer than two lines
were used, or S/N < 10: such epochs can carry a wrong Mean RV (e.g. a single
line locked onto noise) and should be checked by eye before fitting.

Usage:
    python -m spectroscopy.ccf_line_quality <pass2_dir> [--min-range 30]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def _lines(df):
    return [c[:-3] for c in df.columns
            if c.endswith(" RV") and f"{c[:-3]} flag" in df.columns]


def per_star_table(files, min_range):
    rows, epochs = [], []
    for f in files:
        star = os.path.basename(f).replace("_CCF_RVs.csv", "")
        df = pd.read_csv(f)
        lines = _lines(df)
        mean = df["Mean RV"]
        rng = np.nanmax(mean) - np.nanmin(mean) if mean.notna().any() else np.nan
        for ln in lines:
            flag = df[f"{ln} flag"]
            ok = flag == 0
            resid = (df[f"{ln} RV"] - mean)[ok & mean.notna()]
            slope = np.nan
            if rng > min_range and ok.sum() >= 5:
                slope = np.polyfit(mean[ok], df[f"{ln} RV"][ok], 1)[0]
            rows.append({
                "star": star, "line": ln,
                **{f"f{k}": (flag == k).mean() for k in range(5)},
                "med_RVsig": df[f"{ln} RVsig"].median(),
                "rms_resid": np.sqrt(np.mean(resid ** 2)) if len(resid) else np.nan,
                "med_EW": df[f"{ln} EW"].median(),
                "slope": slope, "mean_rv_range": rng,
            })
        n_used = sum((df[f"{ln} flag"] == 0).astype(int) for ln in lines)
        med = mean.median()
        for i, r in df.iterrows():
            if np.isfinite(r["Mean RV"]) and (n_used[i] < 2 or r["SNR"] < 10):
                epochs.append({
                    "star": star, "MJD": r["MJD"], "SNR": r["SNR"],
                    "n_lines_used": int(n_used[i]), "Mean RV": r["Mean RV"],
                    "Mean RVsig": r["Mean RVsig"],
                    "dev_from_median_sigma": abs(r["Mean RV"] - med) / r["Mean RVsig"],
                })
    return pd.DataFrame(rows), pd.DataFrame(epochs)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-line quality of a CCF run.")
    ap.add_argument("ccf_dir", help="directory with *_CCF_RVs.csv (CCF pass 2)")
    ap.add_argument("--min-range", type=float, default=30.0,
                    help="Mean-RV range (km/s) above which the slope is computed")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.ccf_dir, "*_CCF_RVs.csv")))
    if not files:
        raise SystemExit(f"no *_CCF_RVs.csv in {args.ccf_dir}")
    ps, ep = per_star_table(files, args.min_range)

    pd.set_option("display.width", 200)
    n_var = int((ps.groupby("star").mean_rv_range.first() > args.min_range).sum())
    print(f"{len(files)} stars; slope from the {n_var} stars with Mean-RV range "
          f"> {args.min_range:g} km/s\n")
    summ = ps.groupby("line").agg(
        used=("f0", "mean"), clipped=("f1", "mean"), missing=("f2", "mean"),
        ew_zero=("f3", "mean"), low_snr=("f4", "mean"),
        med_RVsig=("med_RVsig", "median"), med_rms_resid=("rms_resid", "median"),
        med_EW=("med_EW", "median"), med_slope=("slope", "median"))
    print(summ.round(3).to_string())

    print("\nstars where a line is used in < 50% of epochs:")
    bad = ps[ps.f0 < 0.5].sort_values(["line", "f0"])
    print(bad[["star", "line", "f0", "f1", "f2", "f3", "f4", "med_EW"]]
          .round(2).to_string(index=False) if len(bad) else "  none")

    print("\nepochs to check by eye (Mean RV kept, but < 2 lines used or S/N < 10):")
    print(ep.sort_values("dev_from_median_sigma", ascending=False).round(2)
          .to_string(index=False) if len(ep) else "  none")

    out = os.path.join(args.ccf_dir, "line_quality_per_star.csv")
    ps.to_csv(out, index=False)
    if len(ep):
        ep.to_csv(os.path.join(args.ccf_dir, "epochs_to_check.csv"), index=False)
    print(f"\ntables: {out}" + (" + epochs_to_check.csv" if len(ep) else ""))


if __name__ == "__main__":
    main()
