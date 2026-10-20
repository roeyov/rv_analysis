"""
pipeline.detection_summary — one row per star: is it detected, and with which
solution (selected exactly as mcmc.batch does: ``mcmc_params.filter_expression``,
best ``field_to_check``), plus the MCMC posterior medians when they exist.

Writes ``<base_dir>/<lmfit_subdir>/detections_summary.csv`` and prints a table.

Usage:
    python -m pipeline.detection_summary --config configs/params_bdwarfs.yaml
"""

import argparse
import contextlib
import glob
import io
import os

import numpy as np
import pandas as pd

from mcmc.selector_app import get_best_row
from pipeline.config import PARAM_FILE, load_args

BEST_COLS = ("candidate_method", "Period_value", "Period_stderr", "K1_value",
             "K1_stderr", "Eccentricity_value", "prob_bicc", "bicc",
             "LS_iter_fap", "PDC_iter_fap", "mass_flag_peri", "p_bicc_kep_vs_circ")


def summarize(cfg):
    lmfit_dir = cfg["pipeline_io"]["output_dir"]
    mp = cfg["mcmc_params"]
    expr = " ".join((mp.get("filter_expression") or "").split())
    field, take_min = mp.get("field_to_check", "bicc"), mp.get("take_min", True)
    rows = []
    for f in sorted(glob.glob(os.path.join(lmfit_dir, "*", "lmfit_summary.csv"))):
        star = os.path.basename(os.path.dirname(f))
        df = pd.read_csv(f)
        with contextlib.redirect_stdout(io.StringIO()):
            best = get_best_row(df, expr, field, take_min)
        r = {"star": star, "detected": best is not None}
        if best is not None:
            r.update({k: best.get(k, np.nan) for k in BEST_COLS})
        else:
            # strongest non-null, non-manual candidate, for context only
            cand = df[~df.candidate_method.str.contains("null|MANUAL")]
            if len(cand):
                c = cand.sort_values(field, ascending=take_min).iloc[0]
                r.update({k: c.get(k, np.nan) for k in BEST_COLS})
                r["candidate_method"] = "(fails filter) " + str(c.candidate_method)
        rows.append(r)
    t = pd.DataFrame(rows)

    mcmc_csv = os.path.join(cfg["mcmc_params"]["output_dir"], "mcmc_params.csv")
    if len(t) and os.path.exists(mcmc_csv):
        m = pd.read_csv(mcmc_csv)
        keep = ["star_name"] + [c for c in m.columns
                                if c.endswith(("_median", "_errm", "_errp"))
                                and c.split("_")[0] in ("ecc", "circ")
                                and c.split("_")[1] in ("P", "K1", "e")]
        t = t.merge(m[keep].rename(columns={"star_name": "star"}), on="star", how="left")
    return t, lmfit_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="Detections + best solutions per star.")
    ap.add_argument("--config", default=PARAM_FILE)
    args = ap.parse_args(argv)

    t, lmfit_dir = summarize(load_args(args.config))
    if t.empty:
        raise SystemExit(f"no lmfit_summary.csv under {lmfit_dir}")
    out = os.path.join(lmfit_dir, "detections_summary.csv")
    t.to_csv(out, index=False)

    pd.set_option("display.width", 220)
    cols = [c for c in ("star", "detected", "candidate_method", "Period_value", "K1_value",
                        "Eccentricity_value", "prob_bicc", "LS_iter_fap", "PDC_iter_fap",
                        "mass_flag_peri", "ecc_P_median", "ecc_K1_median") if c in t]
    print(f"{int(t.detected.sum())} / {len(t)} detected "
          f"(filter = mcmc_params.filter_expression)\n")
    print(t.sort_values(["detected", "star"], ascending=[False, True])[cols]
          .round(4).to_string(index=False))
    print(f"\n{out}")


if __name__ == "__main__":
    main()
