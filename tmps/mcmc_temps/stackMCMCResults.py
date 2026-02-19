import os
import re
import glob
import pandas as pd
import numpy as np

ROOT = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/only_He_from_coAdded2/dbg_newT"

# ---- helpers ----
LINE_RE = re.compile(
    r"^\s*(?P<param>[A-Za-z0-9_]+):\s*mode=(?P<mode>[-+0-9.eE]+),\s*median=(?P<median>[-+0-9.eE]+),\s*-err=(?P<merr>[-+0-9.eE]+),\s*\+err=(?P<perr>[-+0-9.eE]+)"
)

def parse_results_txt(txt_path: str, suffix: str) -> dict:
    """
    Parse lines like:
      {param}: mode=160.242, median=159.998, -err=0.533811, +err=0.507629
    and return columns:
      {param}_mode_{suffix}, {param}_median_{suffix}, {param}_merr_{suffix}, {param}_perr_{suffix}
    """
    out = {}
    if not os.path.isfile(txt_path):
        return out
    with open(txt_path, "r") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            d = m.groupdict()
            param = d["param"]
            out[f"{param}_mode_{suffix}"]   = float(d["mode"])
            out[f"{param}_median_{suffix}"] = float(d["median"])
            out[f"{param}_merr_{suffix}"]   = float(d["merr"])
            out[f"{param}_perr_{suffix}"]   = float(d["perr"])
    return out

def read_bin_flag(csv_path: str) -> float:
    """
    Read lmfit_summary.csv, pick row where is_best==True; if none, take first row.
    Return bin_flag (NaN if missing).
    """
    df = pd.read_csv(csv_path)
    sel = df[df.get("is_best", False) == True]  # noqa: E712
    if sel.empty:
        sel = df.iloc[:1]
    # safely fetch bin_flag
    try:
        return float(sel.iloc[0]["bin_flag"])
    except Exception:
        return np.nan

# ---- main sweep ----
rows = []
for csv_path in glob.glob(os.path.join(ROOT, "*", "lmfit_summary.csv")):
    star_dir = os.path.dirname(csv_path)
    star_name = os.path.basename(star_dir)

    # bin_flag from CSV
    bin_flag = read_bin_flag(csv_path)

    # parse ecc
    ecc_path  = os.path.join(star_dir, "mcmc", "results_ecc.txt")
    if not os.path.isfile(ecc_path):
        continue  # skip stars with no eccentric solution
    ecc_dict  = parse_results_txt(ecc_path, "ecc")

    # parse circ (optional)
    circ_path = os.path.join(star_dir, "mcmc", "results_circ.txt")
    circ_exists = os.path.isfile(circ_path)
    circ_dict = parse_results_txt(circ_path, "circ") if circ_exists else {}

    row = {
        "star_name": star_name,
        "bin_flag": bin_flag,
        "is_circ_exists": bool(circ_exists),
    }
    row.update(ecc_dict)
    row.update(circ_dict)
    rows.append(row)

final_df = pd.DataFrame(rows)

# (optional) sort columns: put id/flags first, then the rest alphabetically
front = ["star_name", "bin_flag", "is_circ_exists"]
final_df = final_df.reindex(columns=front + sorted([c for c in final_df.columns if c not in front]))

# (optional) save
out_csv = os.path.join(ROOT, "stacked_results.csv")
final_df.to_csv(out_csv, index=False)

print(f"Built stacked DataFrame with {len(final_df)} stars and {final_df.shape[1]} columns.")
final_df.head()
