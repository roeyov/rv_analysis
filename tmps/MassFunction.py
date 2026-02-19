# By Zehava Katabi — refactored with constants, single-append flow, and extended output columns

import sympy as sp
import pandas as pd
import re
from pathlib import Path

# =========================
# Constants (paths & files)
# =========================
MASS_FILE = Path("/Users/roeyovadia/Documents/Data/mass_bloem.csv")
ROOT_DIR = Path("/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/repeat_with_coadded/results2")
OUT_PATH = Path("/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/repeat_with_coadded/results2/binary_masses.csv")

LMFIT_SUMMARY_CSV = "lmfit_summary.csv"
REPORT_NAME_FMT = "{star}_sid-{solution_id}_report.txt"

# =========================
# Constants (CSV columns)
# =========================
# lmfit_summary.csv columns
BEST_FLAG_COL = "is_best"
ECC_VALUE_COL = "Eccentricity_value"   # change if your CSV uses "ecc"
PER_VALUE_COL = "Period_value"         # change if your CSV uses "p"
SOL_ID_COL = 'solution_id'

# mass_bloem.csv columns
MASS_ID_COL = "ID"
MSPEC_COL = "Mspec"
MSPEC_ERR_PLUS_COL = "Mspec_er_plus"
MSPEC_ERR_MINUS_COL = "Mspec_er_minus"

# =========================
# Constants (result columns)
# =========================
STAR_NAME_COL = "star_name"
M2_COL = "M2_Msun"
M2_PLUS_COL = "M2_plus_Msun"
M2_MINUS_COL = "M2_minus_Msun"

MSPEC_PLUS_OUT = "Mspec_plus"
MSPEC_MINUS_OUT = "Mspec_minus"
SPT_OUT = "SpT"
# =========================
# Constants (report keys)
# =========================
REPORT_KEY_PERIOD = "Period"
REPORT_KEY_K1 = "K1"
REPORT_KEY_ECC = "Eccentricity"
RESULT_COLS = [
    STAR_NAME_COL, SPT_OUT, REPORT_KEY_PERIOD, REPORT_KEY_K1,
    REPORT_KEY_ECC, MSPEC_COL, MSPEC_PLUS_OUT, MSPEC_MINUS_OUT,
    M2_COL, M2_PLUS_COL, M2_MINUS_COL
]



# =========================
# Helpers
# =========================
def get_best_ecc_p(csv_path: Path):
    df = pd.read_csv(csv_path)
    if df[BEST_FLAG_COL].dtype == object:
        df[BEST_FLAG_COL] = df[BEST_FLAG_COL].astype(str).str.lower().eq("true")
    best = df.loc[df[BEST_FLAG_COL] == True]
    if best.empty:
        return None, None
    return best.iloc[0][ECC_VALUE_COL], best.iloc[0][PER_VALUE_COL], best.iloc[0][SOL_ID_COL]


def extract_param(lines: pd.Series, key: str, cast=float):
    pat = rf'^\s*{re.escape(key)}\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)'
    m = lines.str.extract(pat, expand=False).dropna()
    if m.empty:
        raise KeyError(f"'{key}' not found in report")
    return cast(m.iloc[0])


# =========================
# Main
# =========================
masdf = pd.read_csv(MASS_FILE, header=0)
results = []

for sub in ROOT_DIR.iterdir():
    if not sub.is_dir():
        continue

    star = sub.name
    print(f"--- Processing {star} ---")

    # Default row
    row = {
        STAR_NAME_COL: star,
        M2_COL: "",
        M2_PLUS_COL: "0",
        M2_MINUS_COL: "0",
        MSPEC_COL: "",
        MSPEC_PLUS_OUT: "",
        MSPEC_MINUS_OUT: "",
        REPORT_KEY_PERIOD: "",
        REPORT_KEY_K1: "",
        REPORT_KEY_ECC: "",
        SPT_OUT: ""  # empty SpT column
    }

    try:
        ecc, p, sid = get_best_ecc_p(sub / LMFIT_SUMMARY_CSV)
        print(f"ecc = {ecc}, p = {p}")

        if ecc is None or p is None:
            row[M2_COL] = f"ecc = {ecc} p = {p} not found."
            continue

        report_path = sub / REPORT_NAME_FMT.format(star=star, solution_id=sid)
        if not report_path.exists():
            row[M2_COL] = f"Report not found for {star}"
            print(row[M2_COL])
            continue

        with open(report_path, "r") as f:
            lines = pd.Series([line.strip() for line in f.readlines()])

        P = extract_param(lines, REPORT_KEY_PERIOD)
        K1 = extract_param(lines, REPORT_KEY_K1)
        E = extract_param(lines, REPORT_KEY_ECC)

        match = masdf.loc[masdf[MASS_ID_COL] == star]
        if match.empty:
            raise KeyError(f"'{star}' not found in mass file")

        mspec = float(match.iloc[0][MSPEC_COL])
        mspec_err_plus = float(match.iloc[0][MSPEC_ERR_PLUS_COL])
        mspec_err_minus = float(match.iloc[0][MSPEC_ERR_MINUS_COL])
        print(f"Mspec({star}) = {mspec:.5f}")

        f_m = (P * (K1 ** 3) * (1 - E ** 2) ** (3 / 2)) * 1.036149e-7

        M2 = sp.symbols("M2", positive=True)
        eq_nom = f_m * (mspec + M2) ** 2 - M2 ** 3
        eq_plus = f_m * ((mspec + mspec_err_plus) + M2) ** 2 - M2 ** 3
        eq_minus = f_m * ((mspec + mspec_err_minus) + M2) ** 2 - M2 ** 3

        M2_val = sp.nsolve(eq_nom, mspec)
        M2_plus_val = sp.nsolve(eq_plus, float(M2_val))
        M2_minus_val = sp.nsolve(eq_minus, float(M2_val))

        print(f"{M2_COL}({star}) = {float(M2_val):.5f}")
        print(f"{M2_PLUS_COL}({star}) = {float(M2_plus_val):.5f}")
        print(f"{M2_MINUS_COL}({star}) = {float(M2_minus_val):.5f}")

        row[M2_COL] = float(M2_val)
        row[M2_PLUS_COL] = float(M2_plus_val)
        row[M2_MINUS_COL] = float(M2_minus_val)

        # Populate new outputs
        row[MSPEC_COL] = mspec
        row[MSPEC_PLUS_OUT] = mspec + mspec_err_plus
        row[MSPEC_MINUS_OUT] = mspec + mspec_err_minus
        row[REPORT_KEY_PERIOD] = P
        row[REPORT_KEY_K1] = K1
        row[REPORT_KEY_ECC] = E

    except (KeyError, IndexError) as e:
        row[M2_COL] = f"Could not process {star}. Error: {e}"

    except ValueError as e:
        row[M2_COL] = f"Numerical solver failed for {star}. Error: {e}"

    finally:
        results.append(row)

pd.DataFrame(results, columns=RESULT_COLS).to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
