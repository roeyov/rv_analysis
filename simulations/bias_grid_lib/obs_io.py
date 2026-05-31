"""LaTeX SB1/SB2 table parsing + O-star catalog count loader.

These functions translate the human-edited LaTeX solution tables and
``ostar_catalog.csv`` into the obs arrays the scorer consumes.
"""

import os
import re
import numpy as np
import pandas as pd

from simulations.bias_grid_lib.logging_utils import logger


# ---------------------------------------------------------------------------
# LaTeX table parser
# ---------------------------------------------------------------------------

def _parse_val_with_errors(s):
    """Parse '1.23^{+0.01}_{-0.01}' → 1.23, or handle \\leq / \\geq."""
    s = s.strip().replace("$", "")
    if r"\dots" in s or s == r"\dots":
        return np.nan
    if r"\leq" in s:
        # Upper limit on eccentricity → treat as the limit value
        val = re.sub(r"\\leq\s*", "", s)
        return float(val)
    if r"\geq" in s:
        # Lower limit on period → return negative to flag exclusion
        val = re.sub(r"\\geq\s*", "", s)
        return -float(val)
    # Standard: value^{+err}_{-err}
    m = re.match(r"([0-9.eE+-]+)", s)
    if m:
        return float(m.group(1))
    return np.nan


def load_observed_from_tex(sb1_path, sb2_path, apply_lucy_sweeny_e=True):
    """
    Parse SB1 and SB2 LaTeX solution tables and return observed distributions.

    Parameters
    ----------
    apply_lucy_sweeny_e : bool, default True
        Controls how rows whose eccentricity is reported as a Lucy-Sweeney
        upper limit ("\\leq 0.0X") are folded into ``obs['e']``.
        True  — collapse to e = 0 (circular).
        False — keep the limit value itself as e.

    Returns
    -------
    obs : dict with keys:
        'logP'             : array of log10(P/days)
        'e'                : array of eccentricities, post-Lucy-Sweeney
        'e_value'          : raw eccentricity per row (NaN→0 only); upper
                             limits retain their reported value here
        'e_is_upper_limit' : boolean mask, True for Lucy-Sweeney rows
        'K1'               : array of K1 [km/s]
        'q_sb2'            : array of mass ratios (SB2 only)
        'n_sb1'            : number of usable SB1 systems
        'n_sb2'            : number of SB2 systems
    """
    all_logP = []
    all_e = []
    all_e_value = []
    all_e_upper = []
    all_K1 = []
    sb2_q = []

    # --- Parse SB1 table ---
    with open(sb1_path, "r") as f:
        sb1_lines = f.readlines()

    for line in sb1_lines:
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("\\"):
            continue
        if "&" not in line:
            continue

        cols = [c.strip() for c in line.replace("\\\\", "").split("&")]
        if len(cols) < 8:
            continue

        # Skip header/footer lines
        try:
            int(cols[0])
        except ValueError:
            continue

        P_val = _parse_val_with_errors(cols[2])
        e_val = _parse_val_with_errors(cols[5])
        K1_val = _parse_val_with_errors(cols[6])

        # Skip systems with P >= baseline (flagged as negative by parser)
        if P_val < 0 or np.isnan(P_val):
            continue
        if np.isnan(K1_val):
            continue

        # Skip the "outer" row of 4-043 double-Kepler
        if "(outer)" in cols[1]:
            continue

        all_logP.append(np.log10(P_val))
        e_is_upper_limit = r"\leq" in cols[5]
        if np.isnan(e_val):
            e_to_append = 0.0
            e_raw = 0.0
        elif e_is_upper_limit:
            e_to_append = 0.0 if apply_lucy_sweeny_e else e_val
            e_raw = e_val
        else:
            e_to_append = e_val
            e_raw = e_val
        all_e.append(e_to_append)
        all_e_value.append(e_raw)
        all_e_upper.append(bool(e_is_upper_limit and not np.isnan(e_val)))
        all_K1.append(K1_val)

    n_sb1 = len(all_logP)

    # --- Parse SB2 table ---
    with open(sb2_path, "r") as f:
        sb2_lines = f.readlines()

    # SB2 rows come in pairs: primary (has all params) + secondary (only K and M sin^3 i)
    for line in sb2_lines:
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("\\"):
            continue
        if "&" not in line:
            continue

        cols = [c.strip() for c in line.replace("\\\\", "").split("&")]
        if len(cols) < 10:
            continue

        # Primary row has the system number in col[0]
        try:
            int(cols[0])
        except ValueError:
            continue

        # This is a primary row
        P_val = _parse_val_with_errors(cols[2])
        e_val = _parse_val_with_errors(cols[5])
        q_val = _parse_val_with_errors(cols[6])
        K1_val = _parse_val_with_errors(cols[8])

        if np.isnan(P_val) or P_val < 0:
            continue
        if np.isnan(K1_val):
            continue

        all_logP.append(np.log10(P_val))
        e_is_upper_limit = r"\leq" in cols[5]
        if np.isnan(e_val):
            e_to_append = 0.0
            e_raw = 0.0
        elif e_is_upper_limit:
            e_to_append = 0.0 if apply_lucy_sweeny_e else e_val
            e_raw = e_val
        else:
            e_to_append = e_val
            e_raw = e_val
        all_e.append(e_to_append)
        all_e_value.append(e_raw)
        all_e_upper.append(bool(e_is_upper_limit and not np.isnan(e_val)))
        all_K1.append(K1_val)
        sb2_q.append(q_val if not np.isnan(q_val) else np.nan)

    n_sb2 = len(all_logP) - n_sb1

    obs = {
        "logP": np.array(all_logP),
        "e": np.array(all_e),
        "e_value": np.array(all_e_value),
        "e_is_upper_limit": np.array(all_e_upper, dtype=bool),
        "K1": np.array(all_K1),
        "q_sb2": np.array(sb2_q),
        "n_sb1": n_sb1,
        "n_sb2": n_sb2,
    }
    logger.info("load_observed_from_tex: SB1=%d SB2=%d total=%d "
                "(apply_lucy_sweeny_e=%s)",
                n_sb1, n_sb2, len(all_logP), apply_lucy_sweeny_e)
    return obs


def _load_catalog_counts(path):
    """Read ostar_catalog.csv and return (n_total, n_nonsingle).

    n_total is the number of catalog rows; n_nonsingle counts rows whose
    "Binary status" column is anything except "Apparently single". Used
    as the binomial denominator and numerator under scope=exclude (the
    catalog represents the full O-star population for the binomial,
    independent of period cutoffs).
    """
    df = pd.read_csv(path)
    if "Binary status" not in df.columns:
        raise KeyError(
            "%s lacks 'Binary status' column (got: %s)"
            % (path, list(df.columns)))
    n_total = int(len(df))
    n_nonsingle = int((df["Binary status"] != "Apparently single").sum())
    logger.info("_load_catalog_counts: %s → total=%d, non-single=%d",
                os.path.basename(path), n_total, n_nonsingle)
    return n_total, n_nonsingle
