"""
simulations.true_posterior.obs_catalog — per-star observed outcomes.

The true-posterior likelihood is a product over the catalogue O stars (134
for BLOeM). Each star has exactly one outcome:

  ND  not detected: "Apparently single" in ostar_catalog.csv.
  DX  detected, and its orbit enters the shape term: a usable orbital
      solution with logP >= logP*.
  DO  detected, but no orbit in the shape term. The ``reason`` column says
      why:
        below_cut      usable orbit with logP < logP*
        P_lower_limit  only a lower limit on P (P >= baseline row)
        no_solution    row present but no converged solution (\\ldots)
        no_row         non-single in the catalogue but absent from both
                       tables

"Detected" means catalogue non-single (77 stars for ext3). This is the
same count the CvM binomial uses.

The CvM scorer pools the table rows. The likelihood needs one orbit per
star. A star with several usable rows (4-080: "Aa,Ab" in the SB2 table and
"B" in the SB1 table) must be resolved explicitly with ``--choose``; the
builder refuses to guess.

Usage (local paths from the smoke config; the tables are the paper's):
    python -m simulations.true_posterior.obs_catalog \
        --config configs/params_bias_local_tier4_smoke.yaml \
        --choose "4-080=4-080 B" \
        --out $SCRIPTS_OUT/simulation_pipeline/true_posterior/obs_catalog_ext3.csv
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

from simulations.bias_grid_lib.cutoffs import _compute_logP_cutoff
from simulations.bias_grid_lib.obs_io import load_observed_from_tex

_STAR_RE = re.compile(r"\b(\d-\d{3})\b")
SINGLE_STATUS = "Apparently single"


def star_of(label):
    """Catalogue star ID ('F-NNN') inside a table row label."""
    m = _STAR_RE.search(str(label))
    if not m:
        raise ValueError("no F-NNN star ID in row label %r" % (label,))
    return m.group(1)


def scan_unusable_rows(tex_path):
    """Rows the obs parser drops, keyed by star, with the reason.

    Returns ``{star_id: (reason, label, P_lower_limit_or_nan)}`` for rows
    whose P column is a lower limit (``\\geq``) or which carry no
    solution (``\\dots``/``\\ldots`` in P). Rows below an RLOF separator
    get reason ``rlof_section``. '(outer)' rows are skipped: the star is
    represented by its inner orbit.
    """
    out = {}
    in_rlof = False
    with open(tex_path) as fh:
        for line in fh:
            s = line.strip()
            if "RLOF" in s or "Roche-lobe overflow" in s:
                in_rlof = True
                continue
            if not s or s.startswith("%") or "&" not in s:
                continue
            cols = [c.strip() for c in s.replace("\\\\", "").split("&")]
            if len(cols) < 3:
                continue
            try:
                int(cols[0])
            except ValueError:
                continue
            m = _STAR_RE.search(cols[1])
            if not m:
                continue
            sid = m.group(1)
            label = re.sub(r"\^\{?\(?[a-z]\)?\}?", "",
                           cols[1].replace("$", "")).strip()
            if in_rlof:
                out.setdefault(sid, ("rlof_section", label, np.nan))
                continue
            if "(outer)" in cols[1]:
                continue
            pcol = cols[2]
            if r"\geq" in pcol:
                val = float(re.sub(r"[^0-9.eE+-]", "",
                                   pcol.split(r"\geq")[1]))
                out[sid] = ("P_lower_limit", label, val)
            elif r"\dots" in pcol or r"\ldots" in pcol:
                out[sid] = ("no_solution", label, np.nan)
    return out


def _load_errors(csv_paths):
    """sigma_K1 and sigma_e per row label from the machine-readable
    solution tables (mean of the minus and plus errors). LS rows have no
    e error; their p95 limit already encodes the uncertainty."""
    frames = []
    for p in csv_paths:
        d = pd.read_csv(p, dtype=str)
        lab = d["BLOeM ID"].astype(str).str.strip()

        def num(col):
            return pd.to_numeric(d[col], errors="coerce") if col in d \
                else pd.Series(np.nan, index=d.index)
        frames.append(pd.DataFrame({
            "row_label": lab,
            "sigma_K1": 0.5 * (num("K1_errm") + num("K1_errp")),
            "sigma_e": 0.5 * (num("e_errm") + num("e_errp")),
        }))
    err = pd.concat(frames, ignore_index=True)
    return err.drop_duplicates("row_label", keep="first")


def build_star_table(catalog_csv, sb1_tex, sb2_tex, logP_cut,
                     star_props=None, choose=None, exclude_ids=None,
                     errors_csvs=None):
    """One row per catalogue star with its outcome and (if any) orbit.

    Parameters
    ----------
    logP_cut : float
        logP* of the shape term (same value as the CvM run's cutoff).
    star_props : DataFrame, optional
        Output of ``star_loading.load_star_properties``; its ID, Mspec,
        R_star, field, rv_err columns are merged in.
    choose : dict, optional
        {star_id: row_label} for stars with more than one usable row.
    exclude_ids : iterable of str, optional
        Row labels dropped from the usable rows (robustness variants; same
        semantics as bias_grid's obs_exclude_ids). The star's outcome then
        falls back to DO/'excluded'.
    """
    choose = dict(choose or {})
    cat = pd.read_csv(catalog_csv)
    for col in ("BLOeM ID", "Binary status"):
        if col not in cat.columns:
            raise KeyError("%s lacks column %r" % (catalog_csv, col))
    cat_ids = [str(x).strip() for x in cat["BLOeM ID"]]
    if len(set(cat_ids)) != len(cat_ids):
        raise ValueError("duplicate star IDs in %s" % catalog_csv)
    status = dict(zip(cat_ids, cat["Binary status"].astype(str).str.strip()))

    obs = load_observed_from_tex(sb1_tex, sb2_tex, apply_lucy_sweeny_e=False)
    rows = pd.DataFrame({
        "row_label": obs["id"], "table": obs["table"],
        "logP": obs["logP"], "e": obs["e_value"],
        "e_is_upper_limit": obs["e_is_upper_limit"], "K1": obs["K1"],
    })
    rows["star_id"] = [star_of(r) for r in rows["row_label"]]
    excluded = set(str(x).strip() for x in (exclude_ids or ()))
    if excluded:
        missing = excluded - set(rows["row_label"])
        if missing:
            raise ValueError("exclude_ids %s match no usable row"
                             % sorted(missing))
        rows = rows[~rows["row_label"].isin(excluded)]
    stray = sorted(set(rows["star_id"]) - set(cat_ids))
    if stray:
        raise ValueError("table rows for stars not in the catalogue: %s"
                         % stray)
    singles_with_rows = sorted(s for s in set(rows["star_id"])
                               if status[s] == SINGLE_STATUS)
    if singles_with_rows:
        raise ValueError("usable orbit rows for catalogue singles: %s"
                         % singles_with_rows)

    unusable = {}
    for path in (sb1_tex, sb2_tex):
        for sid, info in scan_unusable_rows(path).items():
            unusable.setdefault(sid, info)
    rlof_nonsingle = sorted(s for s, (r, _l, _p) in unusable.items()
                            if r == "rlof_section"
                            and status.get(s) != SINGLE_STATUS)
    if rlof_nonsingle:
        raise ValueError("RLOF-section rows for non-single stars (check "
                         "the census): %s" % rlof_nonsingle)

    out = []
    for sid in cat_ids:
        st = status[sid]
        rec = {"star_id": sid, "binary_status": st, "outcome": None,
               "reason": "", "row_label": "", "table": "", "n_rows": 0,
               "logP": np.nan, "e": np.nan, "e_is_upper_limit": False,
               "K1": np.nan, "P_lower_limit": np.nan}
        mine = rows[rows["star_id"] == sid]
        rec["n_rows"] = int(len(mine))
        if st == SINGLE_STATUS:
            rec["outcome"] = "ND"
            out.append(rec)
            continue
        if len(mine) == 0:
            reason, label, plim = unusable.get(sid, ("no_row", "", np.nan))
            if sid in {star_of(x) for x in excluded}:
                reason = "excluded"
            rec.update(outcome="DO", reason=reason, row_label=label,
                       P_lower_limit=plim)
            out.append(rec)
            continue
        if len(mine) > 1:
            if sid not in choose:
                raise ValueError(
                    "star %s has %d usable rows %s; pass choose={%r: <label>}"
                    % (sid, len(mine), list(mine["row_label"]), sid))
            pick = mine[mine["row_label"] == choose[sid]]
            if len(pick) != 1:
                raise ValueError("choose[%r]=%r is not one of %s"
                                 % (sid, choose[sid], list(mine["row_label"])))
            r = pick.iloc[0]
        else:
            r = mine.iloc[0]
        rec.update(row_label=r["row_label"], table=r["table"],
                   logP=float(r["logP"]), e=float(r["e"]),
                   e_is_upper_limit=bool(r["e_is_upper_limit"]),
                   K1=float(r["K1"]))
        if rec["logP"] >= logP_cut:
            rec["outcome"] = "DX"
        else:
            rec.update(outcome="DO", reason="below_cut")
        out.append(rec)

    unused_choices = set(choose) - set(rows["star_id"])
    if unused_choices:
        raise ValueError("choose entries for stars without usable rows: %s"
                         % sorted(unused_choices))

    df = pd.DataFrame(out)
    if errors_csvs:
        err = _load_errors(errors_csvs)
        df = df.merge(err, on="row_label", how="left")
        dx = df["outcome"] == "DX"
        miss = df.loc[dx & df["sigma_K1"].isna(), "row_label"].tolist()
        if miss:
            raise ValueError("no K1 error for DX rows %s in %s"
                             % (miss, errors_csvs))
    if star_props is not None:
        sp = star_props.rename(columns={"ID": "star_id"})[
            ["star_id", "Mspec", "R_star", "field", "rv_err"]]
        missing = sorted(set(df["star_id"]) - set(sp["star_id"]))
        if missing:
            raise ValueError("stars missing from the injection sample: %s"
                             % missing)
        df = df.merge(sp, on="star_id", how="left", validate="one_to_one")
    df["logP_cut"] = float(logP_cut)          # constant; travels with the CSV
    df.attrs["logP_cut"] = float(logP_cut)
    return df


def summarize(df):
    """Outcome counts, as a dict."""
    c = df["outcome"].value_counts().to_dict()
    do_reasons = df.loc[df["outcome"] == "DO", "reason"].value_counts()
    dx = df[df["outcome"] == "DX"]
    return {
        "n_stars": int(len(df)),
        "ND": int(c.get("ND", 0)),
        "D": int(c.get("DX", 0) + c.get("DO", 0)),
        "DX": int(c.get("DX", 0)),
        "DO": int(c.get("DO", 0)),
        "DO_reasons": {str(k): int(v) for k, v in do_reasons.items()},
        "DX_LS_circular": int(dx["e_is_upper_limit"].sum()),
        "DX_eccentric": int((~dx["e_is_upper_limit"]).sum()),
        "logP_cut": df.attrs.get("logP_cut"),
    }


def _bias_paths(config_path):
    """Table/catalogue/star-data paths from a bias YAML (bias_grid block)."""
    import yaml
    from simulations.bias_config import DEFAULT_BIAS_CFG
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    bg = cfg.get("bias_grid") or {}

    def get(k):
        v = bg.get(k)
        return DEFAULT_BIAS_CFG.get(k) if v is None else v
    return {k: get(k) for k in ("sb1_tex", "sb2_tex", "ostar_catalog",
                                "mass_file", "rv_dir", "sb2_analysis_dir",
                                "logP_cutoff_smooth_sigma")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", required=True,
                    help="bias YAML providing table/catalogue/star paths")
    ap.add_argument("--sb1-tex", default=None, help="override sb1_tex")
    ap.add_argument("--sb2-tex", default=None, help="override sb2_tex")
    ap.add_argument("--cut", type=float, default=None,
                    help="logP* (default: numerical elbow of the table "
                         "logP, as bias_grid computes it)")
    ap.add_argument("--choose", action="append", default=[],
                    help="STAR=ROW_LABEL for stars with several usable rows")
    ap.add_argument("--exclude", action="append", default=[],
                    help="row label to drop (robustness variants)")
    ap.add_argument("--errors-csv", action="append", default=[],
                    help="machine-readable solution table(s) providing "
                         "K1/e errors (sb1_solutions.csv, sb2_solutions.csv)")
    ap.add_argument("--solutions-only", action="store_true",
                    help="count only stars with an orbital solution as "
                         "detected (binom_n72 analogue): no_solution / "
                         "P_lower_limit / no_row stars become ND")
    ap.add_argument("--no-star-props", action="store_true",
                    help="skip load_star_properties (no M1/field columns)")
    ap.add_argument("--out", default=None, help="CSV output path")
    args = ap.parse_args(argv)

    paths = _bias_paths(args.config)
    sb1 = args.sb1_tex or paths["sb1_tex"]
    sb2 = args.sb2_tex or paths["sb2_tex"]
    if args.cut is None:
        obs = load_observed_from_tex(sb1, sb2)
        cut = _compute_logP_cutoff(
            obs["logP"], "numerical",
            smooth_sigma=float(paths["logP_cutoff_smooth_sigma"] or 0.15))
    else:
        cut = float(args.cut)
    choose = {}
    for c in args.choose:
        k, _, v = c.partition("=")
        choose[k.strip()] = v.strip()

    props = None
    if not args.no_star_props:
        from simulations.bias_grid_lib.star_loading import (
            load_star_properties)
        props = load_star_properties(
            paths["mass_file"], paths["rv_dir"],
            sb2_analysis_dir=paths["sb2_analysis_dir"],
            ostar_catalog=paths["ostar_catalog"])

    df = build_star_table(paths["ostar_catalog"], sb1, sb2, cut,
                          star_props=props, choose=choose,
                          exclude_ids=args.exclude,
                          errors_csvs=args.errors_csv or None)
    if args.solutions_only:
        m = (df["outcome"] == "DO") & df["reason"].isin(
            ["no_solution", "P_lower_limit", "no_row"])
        df.loc[m, "outcome"] = "ND"
        df.loc[m, "reason"] = "solutions_only"
    summ = summarize(df)
    print("logP* = %.6f" % cut)
    for k, v in summ.items():
        print("  %-16s %s" % (k, v))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df.to_csv(args.out, index=False)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
