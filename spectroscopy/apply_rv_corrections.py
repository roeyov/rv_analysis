"""
spectroscopy.apply_rv_corrections — apply per-field, per-epoch RV zero-point
corrections to *_CCF_RVs.csv files.

A corrections CSV (e.g. RVcorrectionsDR5.csv) with columns
    config, MJD, rv_correction, Error
where config = F<field>_E<epoch>, is joined per field onto each star's CSV by
nearest MJD (the correction MJDs are the CCF MJDs rounded to ~4 decimals, so
exact matching is impossible; the default tolerance of 5e-3 d is far above the
rounding error and below half the closest epoch pair, 0.0185 d in field 8).
Two columns are appended to each CSV:

    fixed Mean RV    = Mean RV + sign * rv_correction
    fixed Mean RVsig = sqrt(Mean RVsig**2 + Error**2)

NaNs in the input RV columns propagate to the fixed columns. The script runs
in two phases — every file is validated before any file is written — so a
failure leaves the directory untouched. Re-running overwrites the fixed
columns in place (idempotent).

Usage:
    python -m spectroscopy.apply_rv_corrections \
        --corrections <dir>/RVcorrectionsDR5.csv --input-dir <dir> [--dry-run]
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# Epochs per field in the DR5 corrections table (warn-only sanity check)
EXPECTED_EPOCHS = {1: 25, 2: 26, 3: 26, 4: 27, 5: 26, 6: 26, 7: 26, 8: 28}

CONFIG_RE = re.compile(r"^F(\d+)_E\d+$")
STAR_RE = re.compile(r"BLOeM_(\d+)-\d{3}")


def load_corrections(path, tolerance):
    """Load the corrections CSV and split it into per-field tables."""
    corr = pd.read_csv(path)
    required = ["config", "MJD", "rv_correction", "Error"]
    missing = [c for c in required if c not in corr.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")

    fields = {}
    bad = [c for c in corr["config"] if not CONFIG_RE.match(str(c))]
    if bad:
        raise SystemExit(f"{path}: unparsable config entries: {bad}")
    corr["field"] = corr["config"].str.extract(CONFIG_RE).astype(int)

    for field, sub in corr.groupby("field"):
        sub = sub.sort_values("MJD").reset_index(drop=True)
        expected = EXPECTED_EPOCHS.get(field)
        if expected is not None and len(sub) != expected:
            print(f"WARNING: field {field} has {len(sub)} correction epochs "
                  f"(expected {expected})")
        gaps = np.diff(sub["MJD"].to_numpy())
        if len(gaps) and gaps.min() < 2 * tolerance:
            raise SystemExit(
                f"field {field}: correction epochs only {gaps.min():.5f} d apart "
                f"— ambiguous at tolerance {tolerance}; lower --tolerance")
        fields[field] = sub
    return fields


def match_star(path, fields, tolerance, rv_col, rverr_col):
    """Match every CSV row to a correction epoch. Returns (df, sub, idx) or raises ValueError."""
    name = os.path.basename(path)
    m = STAR_RE.search(name)
    if not m:
        raise ValueError(f"{name}: cannot parse field from filename")
    field = int(m.group(1))
    if field not in fields:
        raise ValueError(f"{name}: no corrections for field {field}")

    df = pd.read_csv(path)
    missing = [c for c in ("MJD", rv_col, rverr_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if len(df) == 0:
        raise ValueError(f"{name}: no data rows")
    if df["MJD"].isna().any():
        raise ValueError(f"{name}: NaN MJD at rows "
                         f"{df.index[df['MJD'].isna()].tolist()}")
    for col in (rv_col, rverr_col):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{name}: column '{col}' is not numeric")

    sub = fields[field]
    corr_mjds = sub["MJD"].to_numpy()
    idx, dists = [], []
    for mjd in df["MJD"].to_numpy():
        d = np.abs(corr_mjds - mjd)
        i = int(d.argmin())
        idx.append(i)
        dists.append(d[i])

    bad = [(k, df['MJD'].iloc[k], corr_mjds[i], d)
           for k, (i, d) in enumerate(zip(idx, dists)) if d > tolerance]
    if bad:
        lines = [f"    row {k}: MJD {mjd} nearest corr {cm} (|d|={d:.4f} d)"
                 for k, mjd, cm, d in bad]
        raise ValueError(f"{name}: {len(bad)} rows beyond tolerance {tolerance}:\n"
                         + "\n".join(lines))

    if len(set(idx)) != len(idx):
        dup = [sub["config"].iloc[i] for i in set(i for i in idx if idx.count(i) > 1)]
        raise ValueError(f"{name}: multiple rows matched the same correction "
                         f"epoch(s) {dup}")

    return df, sub, idx


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Apply per-field per-epoch RV zero-point corrections to "
                    "*_CCF_RVs.csv files (adds 'fixed' RV/RVsig columns).")
    ap.add_argument("--corrections", required=True,
                    help="Corrections CSV (config,MJD,rv_correction,Error)")
    ap.add_argument("--input-dir", required=True,
                    help="Directory holding *_CCF_RVs.csv files")
    ap.add_argument("--rv-col", default="Mean RV")
    ap.add_argument("--rverr-col", default="Mean RVsig")
    ap.add_argument("--out-rv-col", default="fixed Mean RV")
    ap.add_argument("--out-rverr-col", default="fixed Mean RVsig")
    ap.add_argument("--sign", type=float, default=1.0,
                    help="fixed = rv + sign*correction (default: +1)")
    ap.add_argument("--tolerance", type=float, default=5e-3,
                    help="max |MJD_csv - MJD_corr| in days (default: 5e-3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and report only; write nothing")
    args = ap.parse_args(argv)

    fields = load_corrections(args.corrections, args.tolerance)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*_CCF_RVs.csv")))
    if not files:
        raise SystemExit(f"no *_CCF_RVs.csv files in {args.input_dir}")

    # ---- phase 1: validate every file before writing anything ----
    prepared, errors = [], []
    for path in files:
        try:
            prepared.append((path, *match_star(path, fields, args.tolerance,
                                               args.rv_col, args.rverr_col)))
        except ValueError as e:
            errors.append(str(e))

    if errors:
        print(f"\n{len(errors)} file(s) FAILED validation — nothing written:")
        for e in errors:
            print("  " + e)
        sys.exit(1)

    # ---- phase 2: compute + write ----
    total_rows = total_nan = 0
    for path, df, sub, idx in prepared:
        applied = sub["rv_correction"].to_numpy()[idx]
        corr_err = sub["Error"].to_numpy()[idx]

        df = df.drop(columns=[args.out_rv_col, args.out_rverr_col],
                     errors="ignore")
        df[args.out_rv_col] = df[args.rv_col] + args.sign * applied
        df[args.out_rverr_col] = np.sqrt(df[args.rverr_col] ** 2 + corr_err ** 2)

        n_nan = int(df[args.rv_col].isna().sum())
        total_rows += len(df)
        total_nan += n_nan

        if not args.dry_run:
            tmp = path + ".tmp"
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)

        print(f"{os.path.basename(path):32s} rows={len(df):3d} "
              f"matched={len(set(idx))}/{len(sub)} nan={n_nan:2d} "
              f"corr=[{applied.min():+.3f},{applied.max():+.3f}]")

    action = "validated (dry run)" if args.dry_run else "written"
    print(f"\n{len(prepared)} files {action}: {total_rows} rows "
          f"({total_nan} with NaN {args.rv_col}), sign={args.sign:+g}, "
          f"tolerance={args.tolerance:g} d")


if __name__ == "__main__":
    main()
