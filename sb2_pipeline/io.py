"""
sb2_pipeline.io — SB2 per-epoch RV loading and zero-point corrections.

Data layout (read-only, never written to):
    <sb2_root>/BLOeM_<f>-<nnn>/sb2_analysis/<YYYYMMDD_HHMMSS>/data/rv_*.csv

The run directory per star is the provenance table's adopted_timestamp when
present (it pins e.g. BLOeM_4-080 to a non-latest run), otherwise the latest
timestamp. Within it the most-processed RV file wins (rv_final_for_mcmc ->
rv_final -> rv_corrected -> rv_extracted); "rv_corrected" refers to the
gamma-crossing component-assignment fix, unrelated to zero-points.

Correction recipe (identical to spectroscopy.apply_rv_corrections, sign -1):
    fixed_rv  = rv - rv_correction
    fixed_err = sqrt(rv_err**2 + Error**2)
applied to rv1 and rv2 (and rv3 when present) with the SAME per-epoch value —
the zero-point shift is common-mode, so rv1 - rv2 is invariant.

Epoch matching is nearest-MJD within tolerance per field, cross-checked
against the stored integer epoch column; both must agree for every row.
"""

import os
import re
from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

from sb2_pipeline.constants import (
    EPOCH, MJD_COL, RV1, ERR1, RV2, ERR2, RV3, ERR3,
    CORR_CONFIG, CORR_MJD, CORR_VALUE, CORR_ERROR,
    DEFAULT_RV_FILE_PRIORITY,
)

STAR_RE = re.compile(r"BLOeM_(\d+)-\d{3}")
TIMESTAMP_RE = re.compile(r"\d{8}_\d{6}")


class CorrectionMatchError(RuntimeError):
    """Raised when an epoch cannot be matched unambiguously to a correction."""


@dataclass
class SB2Star:
    star_id: str
    timestamp: str
    timestamp_is_latest: bool
    rv_path: str
    rv_source_file: str
    df: pd.DataFrame
    has_rv3: bool = False
    n_raw: int = 0
    audit: list = dc_field(default_factory=list)
    n_dropped_nonfinite: int = 0


# ---------------------------------------------------------------------------
# Provenance / run-dir resolution
# ---------------------------------------------------------------------------

def load_provenance(provenance_csv):
    """Provenance table indexed by star_id (adopted_timestamp, table_model...)."""
    df = pd.read_csv(provenance_csv)
    return df.set_index("star_id")


def resolve_run_dir(sb2_root, star_id, provenance=None):
    """
    Return (timestamp, is_latest) for a star.

    The provenance adopted_timestamp wins when present; otherwise the latest
    directory matching YYYYMMDD_HHMMSS. Raises FileNotFoundError when the
    star has no sb2_analysis runs at all.
    """
    analysis_dir = os.path.join(sb2_root, star_id, "sb2_analysis")
    if not os.path.isdir(analysis_dir):
        raise FileNotFoundError(f"{star_id}: no sb2_analysis directory")

    ts_dirs = sorted(
        d for d in os.listdir(analysis_dir)
        if TIMESTAMP_RE.fullmatch(d) and os.path.isdir(os.path.join(analysis_dir, d))
    )
    if not ts_dirs:
        raise FileNotFoundError(f"{star_id}: no timestamped runs under {analysis_dir}")
    latest = ts_dirs[-1]

    adopted = None
    if provenance is not None and star_id in provenance.index:
        val = provenance.loc[star_id, "adopted_timestamp"]
        if isinstance(val, str) and val.strip():
            adopted = val.strip()

    if adopted is not None:
        if adopted not in ts_dirs:
            raise FileNotFoundError(
                f"{star_id}: adopted_timestamp {adopted} not found in {ts_dirs}")
        return adopted, adopted == latest
    return latest, True


def pick_rv_file(data_dir, priority=None):
    """First existing RV file per the priority chain -> (path, basename)."""
    for name in (priority or DEFAULT_RV_FILE_PRIORITY):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path, name
    raise FileNotFoundError(f"No RV file among {priority} in {data_dir}")


def load_sb2_star(sb2_root, star_id, provenance=None, priority=None):
    """Load one star's per-epoch SB2 RVs into an SB2Star (uncorrected)."""
    timestamp, is_latest = resolve_run_dir(sb2_root, star_id, provenance)
    data_dir = os.path.join(sb2_root, star_id, "sb2_analysis", timestamp, "data")
    rv_path, rv_file = pick_rv_file(data_dir, priority)

    df = pd.read_csv(rv_path, comment="#")
    df = df.drop(columns=[c for c in df.columns if c.endswith("_original")],
                 errors="ignore")

    required = {MJD_COL, RV1, ERR1, RV2, ERR2}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{star_id}: {rv_file} missing columns {sorted(missing)}")

    return SB2Star(
        star_id=star_id,
        timestamp=timestamp,
        timestamp_is_latest=is_latest,
        rv_path=rv_path,
        rv_source_file=rv_file,
        df=df.reset_index(drop=True),
        has_rv3=RV3 in df.columns,
        n_raw=len(df),
    )


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

def load_corrections(path):
    """RVcorrectionsDR5.csv with the F<field>_E<epoch> config parsed out."""
    df = pd.read_csv(path)
    extracted = df[CORR_CONFIG].str.extract(r"F(\d+)_E(\d+)")
    if extracted.isna().any().any():
        bad = df.loc[extracted.isna().any(axis=1), CORR_CONFIG].tolist()
        raise ValueError(f"Unparseable correction configs: {bad}")
    df["field"] = extracted[0].astype(int)
    df["epoch_num"] = extracted[1].astype(int)
    return df


def apply_corrections(star, corr_df, tolerance=5e-3):
    """
    Apply the per-epoch zero-point corrections to an SB2Star in place.

    Matching is nearest-MJD within *tolerance* against the star's field, with
    two hard cross-checks (any failure raises CorrectionMatchError):
      * no two epochs may match the same correction row,
      * the matched E<NN> must equal the stored integer epoch column.

    Fills star.audit with one dict per raw epoch (including dropped ones) and
    replaces star.df with the corrected, finite-only table.
    """
    m = STAR_RE.match(star.star_id)
    if not m:
        raise CorrectionMatchError(f"Cannot parse field from {star.star_id!r}")
    star_field = int(m.group(1))

    sub = corr_df[corr_df["field"] == star_field].reset_index(drop=True)
    if sub.empty:
        raise CorrectionMatchError(f"No corrections for field {star_field}")

    corr_mjds = sub[CORR_MJD].to_numpy(float)
    df = star.df.copy()
    mjds = df[MJD_COL].to_numpy(float)

    idx, dists = [], []
    for mjd in mjds:
        d = np.abs(corr_mjds - mjd)
        i = int(d.argmin())
        if not np.isfinite(d[i]) or d[i] > tolerance:
            raise CorrectionMatchError(
                f"{star.star_id}: MJD {mjd} has no correction within "
                f"{tolerance} d (closest {d[i]:.4g} d)")
        idx.append(i)
        dists.append(float(d[i]))
    if len(set(idx)) != len(idx):
        raise CorrectionMatchError(
            f"{star.star_id}: two epochs matched the same correction row")

    if EPOCH in df.columns:
        stored = df[EPOCH].to_numpy(int)
        matched = sub.loc[idx, "epoch_num"].to_numpy(int)
        if not np.array_equal(stored, matched):
            bad = np.nonzero(stored != matched)[0].tolist()
            raise CorrectionMatchError(
                f"{star.star_id}: epoch-column cross-check failed at rows {bad} "
                f"(stored {stored[bad]}, matched {matched[bad]})")

    corr_val = sub.loc[idx, CORR_VALUE].to_numpy(float)
    corr_err = sub.loc[idx, CORR_ERROR].to_numpy(float)
    configs = sub.loc[idx, CORR_CONFIG].to_numpy(str)

    components = [(RV1, ERR1), (RV2, ERR2)]
    if star.has_rv3:
        components.append((RV3, ERR3))

    raw = {c: df[c].to_numpy(float) for pair in components for c in pair}
    for rv_col, err_col in components:
        df[rv_col] = raw[rv_col] - corr_val
        df[err_col] = np.sqrt(raw[err_col] ** 2 + corr_err ** 2)

    # Common-mode invariance: (rv1-c)-(rv2-c) == rv1-rv2 up to float rounding.
    if not np.allclose(np.abs(df[RV1] - df[RV2]),
                       np.abs(raw[RV1] - raw[RV2]),
                       rtol=0.0, atol=1e-8, equal_nan=True):
        raise CorrectionMatchError(
            f"{star.star_id}: |rv1-rv2| changed under a common-mode shift")

    finite = np.ones(len(df), dtype=bool)
    for col in (MJD_COL, RV1, ERR1, RV2, ERR2):
        finite &= np.isfinite(df[col].to_numpy(float))

    audit = []
    for k in range(len(df)):
        rec = {
            "star_name": star.star_id,
            "epoch": int(df[EPOCH].iloc[k]) if EPOCH in df.columns else -1,
            "mjd": mjds[k],
            "config": configs[k],
            "delta_mjd": dists[k],
            "rv_correction": corr_val[k],
            "corr_error": corr_err[k],
            "rv_source_file": star.rv_source_file,
            "timestamp": star.timestamp,
            "dropped": bool(~finite[k]),
            "drop_reason": "" if finite[k] else "nonfinite rv/err/mjd",
        }
        for rv_col, err_col in components:
            rec[f"{rv_col}_raw"] = raw[rv_col][k]
            rec[f"{rv_col}_corr"] = float(df[rv_col].iloc[k])
            rec[f"{err_col}_raw"] = raw[err_col][k]
            rec[f"{err_col}_corr"] = float(df[err_col].iloc[k])
        for fb in ("rv1_err_fallback", "rv2_err_fallback"):
            rec[fb] = bool(df[fb].iloc[k]) if fb in df.columns else np.nan
        audit.append(rec)

    star.audit = audit
    star.n_dropped_nonfinite = int((~finite).sum())
    star.df = df[finite].reset_index(drop=True)
    return star


def write_corrected_csv(star, out_path):
    """Write the corrected per-epoch table with a provenance comment header."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# SB2 per-epoch zero-point-corrected RVs\n")
        f.write(f"# Star: {star.star_id}\n")
        f.write(f"# Source: {star.rv_path}\n")
        f.write(f"# Source timestamp: {star.timestamp} "
                f"(is_latest={star.timestamp_is_latest})\n")
        f.write("# Correction: fixed = rv - rv_correction; "
                "err = sqrt(rv_err^2 + Error^2)  [RVcorrectionsDR5]\n")
        f.write("#\n")
        star.df.to_csv(f, index=False)


def make_data_dict(df):
    """
    Corrected (or cleaned) table -> joint-fit data dict.

    Each component gets its own time axis so that use1/use2 epoch flags
    (from sb2_pipeline.cleaning) can mask single-component points.
    """
    from sb2_pipeline.constants import (
        SB2_TS1, SB2_TS2, SB2_RV1, SB2_ERR1, SB2_RV2, SB2_ERR2, USE1, USE2,
    )
    n = len(df)
    use1 = (df[USE1].to_numpy(bool) if USE1 in df.columns
            else np.ones(n, dtype=bool))
    use2 = (df[USE2].to_numpy(bool) if USE2 in df.columns
            else np.ones(n, dtype=bool))
    mjd = df[MJD_COL].to_numpy(float)
    data = {
        SB2_TS1: mjd[use1],
        SB2_RV1: df[RV1].to_numpy(float)[use1],
        SB2_ERR1: df[ERR1].to_numpy(float)[use1],
        SB2_TS2: mjd[use2],
        SB2_RV2: df[RV2].to_numpy(float)[use2],
        SB2_ERR2: df[ERR2].to_numpy(float)[use2],
    }
    # Masked points ride along for display only (light symbols in the RV
    # panel); the fit/MCMC layers read exclusively the six keys above.
    data["masked"] = {
        SB2_TS1: mjd[~use1],
        SB2_RV1: df[RV1].to_numpy(float)[~use1],
        SB2_ERR1: df[ERR1].to_numpy(float)[~use1],
        SB2_TS2: mjd[~use2],
        SB2_RV2: df[RV2].to_numpy(float)[~use2],
        SB2_ERR2: df[ERR2].to_numpy(float)[~use2],
    }
    return data


def paired_drv(df):
    """(mjd, |rv1-rv2|, sqrt(err1^2+err2^2)) over epochs where BOTH are used."""
    from sb2_pipeline.constants import USE1, USE2
    n = len(df)
    both = np.ones(n, dtype=bool)
    if USE1 in df.columns:
        both &= df[USE1].to_numpy(bool)
    if USE2 in df.columns:
        both &= df[USE2].to_numpy(bool)
    sub = df[both]
    drv = np.abs(sub[RV1].to_numpy(float) - sub[RV2].to_numpy(float))
    derr = np.sqrt(sub[ERR1].to_numpy(float) ** 2
                   + sub[ERR2].to_numpy(float) ** 2)
    return sub[MJD_COL].to_numpy(float), drv, derr
