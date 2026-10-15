"""
simulations.true_posterior.paper_catalogs — observed catalogues for the
paper's check runs, each derived from the frozen catalogue of the adopted
fit (``obs_catalog.csv`` of the split_5sig run).

The frozen catalogue was parsed from the paper's LaTeX tables, whose cells
are rounded for display. Every catalogue written here keeps the frozen
catalogue's stars, schema and logP* and changes only what its name says:

  fullprec   every orbit re-read from the machine-readable solution tables
             (sb1_solutions.csv, sb2_solutions.csv) as they stand now: e,
             K1 and their errors at the tables' precision, the eccentricity
             limit of a Lucy-Sweeney circular orbit from ``e_p95``, and the
             table a star currently sits in. The period is re-read too,
             unless the frozen value agrees with the table at the table's
             four decimals: the LaTeX cell then carried more digits and is
             kept. Outcomes are re-derived from logP against the frozen
             logP*.
  k1floor    amplitude-floor survey: every detected orbit with K1 below the
             floor becomes ND. Use with ``run_posterior --k1-floor``; the
             likelihood refuses the floor with any other table. Detected
             stars without a K1 stay detected, i.e. they are taken to lie
             above the floor (``--k1-unknown-above-floor``).
  k1floor_strict   the same, with the period-lower-limit stars (no measured
             K1) entered as ND as well. Together the two bracket the
             unknown amplitudes of those stars.
  doubtful   the apparently single stars most likely to be misclassified
             (the classical RV variables and the solutions rejected by the
             Roche-lobe test) enter as detections without a usable period
             (DO, reason ``doubtful_single``).

``k1floor`` and ``doubtful`` start from the frozen catalogue, so their runs
differ from the adopted fit by one change only.

The report also gives the inflection point of the smoothed period CDF
(``bias_grid_lib.cutoffs._numerical_logP_cutoff``) for the frozen and the
re-read period lists. The catalogues keep the frozen logP* either way.

    python -m simulations.true_posterior.paper_catalogs \\
        --frozen <split_5sig obs_catalog.csv> --tables <paper tables dir> \\
        --out-dir <dir> [--k1-floor 10]
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

from simulations.bias_grid_lib.cutoffs import _numerical_logP_cutoff
from simulations.true_posterior.obs_catalog import SINGLE_STATUS, star_of

COLUMNS = ["star_id", "binary_status", "outcome", "reason", "row_label",
           "table", "n_rows", "logP", "e", "e_is_upper_limit", "K1",
           "P_lower_limit", "sigma_K1", "sigma_e", "logP_cut"]
ORBIT_COLS = ["logP", "e", "K1", "sigma_K1", "sigma_e"]

# Apparently single stars that are significant RV variables under the
# classical criterion ("Classical RV variable" = yes in ostar_catalog.csv).
RV_VARIABLE_SINGLES = ["1-027", "2-016", "2-020", "3-094", "4-040", "6-019",
                       "6-032", "8-024"]
# Apparently single stars whose only solution fails the Roche-lobe test.
ROCHE_VETOED_SINGLES = ["1-025", "2-007", "6-005", "8-053"]
# Detected orbits below 10 km/s in the frozen catalogue and in the tables.
EXPECTED_BELOW_10 = ["1-068", "2-035", "2-087", "3-078", "4-074", "6-091",
                     "6-098", "7-044", "8-028"]
FROZEN_COUNTS = {"ND": 57, "DX": 57, "DO": 20, "DX_LS": 19,
                 "below_cut": 14, "P_lower_limit": 3, "no_solution": 3}
# Usable table rows that are not a star's catalogue row. The period cut was
# derived from the pooled table rows, so these enter the inflection point;
# the value is the period of the LaTeX cell the original parser read.
POOLED_SECOND_ROWS = {"4-080 Aa,Ab": 3.31464}


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def counts(df):
    dx = df[df["outcome"] == "DX"]
    out = {k: int((df["outcome"] == k).sum()) for k in ("ND", "DX", "DO")}
    out["DX_LS"] = int(dx["e_is_upper_limit"].astype(bool).sum())
    reasons = df.loc[df["outcome"] == "DO", "reason"].value_counts()
    out.update({str(k): int(v) for k, v in reasons.items()})
    return out


def check_counts(df, expected, name):
    got = counts(df)
    if got != expected:
        raise AssertionError("%s: outcome counts %s, expected %s"
                             % (name, got, expected))
    if len(df) != 134 or df["star_id"].duplicated().any():
        raise AssertionError("%s: not 134 unique stars" % name)


def load_frozen(path):
    # exact float parsing, so that unchanged rows are written back verbatim
    df = pd.read_csv(path, float_precision="round_trip")
    if list(df.columns) != COLUMNS:
        raise ValueError("unexpected columns in %s: %s"
                         % (path, list(df.columns)))
    df["star_id"] = df["star_id"].astype(str)
    check_counts(df, FROZEN_COUNTS, "frozen catalogue")
    if df["logP_cut"].nunique() != 1:
        raise ValueError("logP_cut is not constant in %s" % path)
    return df


def load_solution_rows(tables_dir):
    """One record per row of the two solution tables.

    kind is ``orbit`` (numeric P), ``P_lower_limit`` or ``no_solution``.
    ``(outer)`` rows are skipped: a star is represented by its inner orbit.
    """
    recs = []
    for table, fname in (("SB1", "sb1_solutions.csv"),
                         ("SB2", "sb2_solutions.csv")):
        d = pd.read_csv(os.path.join(tables_dir, fname), dtype=str)
        for _, r in d.iterrows():
            label = str(r["BLOeM ID"]).strip()
            if "(outer)" in label:
                continue
            p = str(r["P [d]"]).strip()
            rec = {"row_label": label, "star_id": star_of(label),
                   "table": table, "kind": "orbit", "P_lower_limit": np.nan}
            if p.startswith(">="):
                rec.update(kind="P_lower_limit", P_lower_limit=float(p[2:]))
            elif p in ("---", "nan", ""):
                rec["kind"] = "no_solution"
            else:
                e_txt = str(r["e"]).strip()
                ls = e_txt.startswith("<=")

                def num(col):
                    return float(r[col])
                rec.update(
                    logP=float(np.log10(float(p))),
                    e_is_upper_limit=ls,
                    # LS rows: the 95th percentile behind the "<=" cell
                    e=num("e_p95") if ls else float(e_txt),
                    e_display=float(e_txt[2:]) if ls else float(e_txt),
                    K1=num("K1 [km/s]"),
                    sigma_K1=0.5 * (num("K1_errm") + num("K1_errp")),
                    sigma_e=(np.nan if ls
                             else 0.5 * (num("e_errm") + num("e_errp"))))
                if not all(np.isfinite([rec["logP"], rec["e"], rec["K1"],
                                        rec["sigma_K1"]])):
                    raise ValueError("non-finite orbit values in row %r of "
                                     "%s" % (label, fname))
                if ls and abs(rec["e"] - rec["e_display"]) > 0.0051:
                    raise ValueError("row %r: e_p95 %.4f does not round to "
                                     "the displayed limit %.2f"
                                     % (label, rec["e"], rec["e_display"]))
            recs.append(rec)
    rows = pd.DataFrame(recs)
    if rows["row_label"].duplicated().any():
        raise ValueError("row labels shared between the two tables: %s"
                         % sorted(rows.loc[rows["row_label"].duplicated(),
                                           "row_label"]))
    return rows


def rebuild_fullprec(frozen, rows, catalog_csv):
    """The frozen catalogue with every orbit re-read from the CSV tables."""
    cat = pd.read_csv(catalog_csv, dtype=str)
    status = dict(zip(cat["BLOeM ID"].astype(str).str.strip(),
                      cat["Binary status"].astype(str).str.strip()))
    cut = float(frozen["logP_cut"].iloc[0])
    by_label = rows.set_index("row_label")
    usable_per_star = rows[rows["kind"] == "orbit"].groupby("star_id").size()
    out, changes = [], []
    for _, f in frozen.iterrows():
        rec = f.to_dict()
        sid = rec["star_id"]
        if status.get(sid) != rec["binary_status"]:
            raise AssertionError("%s: status %r in the catalogue, %r frozen"
                                 % (sid, status.get(sid),
                                    rec["binary_status"]))
        mine = rows[rows["star_id"] == sid]
        if rec["outcome"] == "ND":
            if rec["binary_status"] != SINGLE_STATUS or len(mine):
                raise AssertionError("%s: ND star with table rows or a "
                                     "non-single status" % sid)
            out.append(rec)
            continue
        label = rec["row_label"]
        if label not in by_label.index:
            raise AssertionError("%s: row %r is in neither table"
                                 % (sid, label))
        r = by_label.loc[label]
        if rec["outcome"] == "DO" and rec["reason"] in ("no_solution",
                                                        "P_lower_limit"):
            if r["kind"] != rec["reason"]:
                raise AssertionError("%s: frozen %s, the table now has %s"
                                     % (sid, rec["reason"], r["kind"]))
            if (r["kind"] == "P_lower_limit"
                    and r["P_lower_limit"] != rec["P_lower_limit"]):
                raise AssertionError("%s: period lower limit changed" % sid)
            out.append(rec)
            continue
        if r["kind"] != "orbit":
            raise AssertionError("%s: frozen orbit, the table now has %s"
                                 % (sid, r["kind"]))
        new = dict(rec)
        # the CSV holds P to four decimals; a frozen (LaTeX) period that
        # rounds to it carries at least as many digits and is kept
        p_tab, p_frozen = 10.0 ** r["logP"], 10.0 ** rec["logP"]
        keep_p = abs(round(p_frozen, 4) - round(p_tab, 4)) < 1e-9
        new.update(table=r["table"], n_rows=int(usable_per_star.get(sid, 0)),
                   logP=rec["logP"] if keep_p else r["logP"], e=r["e"],
                   e_is_upper_limit=bool(r["e_is_upper_limit"]),
                   K1=r["K1"], sigma_K1=r["sigma_K1"], sigma_e=r["sigma_e"])
        if new["logP"] >= cut:
            new.update(outcome="DX", reason=np.nan)
        else:
            new.update(outcome="DO", reason="below_cut")
        changes.append({
            "star_id": sid, "row_label": label,
            "table_old": rec["table"], "table_new": new["table"],
            "outcome_old": rec["outcome"], "outcome_new": new["outcome"],
            "ls_old": bool(rec["e_is_upper_limit"]),
            "ls_new": new["e_is_upper_limit"],
            "P_old": 10.0 ** rec["logP"], "P_new": 10.0 ** new["logP"],
            "P_table": p_tab, "P_source": "frozen" if keep_p else "table",
            "e_old": rec["e"], "e_new": new["e"],
            "K1_old": rec["K1"], "K1_new": new["K1"],
            "sigma_K1_old": rec["sigma_K1"], "sigma_K1_new": new["sigma_K1"],
            "dlogP": new["logP"] - rec["logP"], "de": new["e"] - rec["e"],
            "dlogK1": float(np.log10(new["K1"] / rec["K1"]))})
        out.append(new)
    # every usable table row must have been consumed, except the pooled
    # second rows, and every table star must be a catalogue star
    used = set(frozen["row_label"].dropna())
    spare = sorted(set(rows.loc[rows["kind"] == "orbit", "row_label"]) - used)
    if spare != sorted(POOLED_SECOND_ROWS):
        raise AssertionError("usable table rows that are no star's "
                             "catalogue row: %s, expected %s"
                             % (spare, sorted(POOLED_SECOND_ROWS)))
    stray = sorted(set(rows["star_id"]) - set(frozen["star_id"]))
    if stray:
        raise AssertionError("table rows for stars outside the catalogue: "
                             "%s" % stray)
    df = pd.DataFrame(out)[COLUMNS]
    df["n_rows"] = df["n_rows"].astype(int)
    df["e_is_upper_limit"] = df["e_is_upper_limit"].astype(bool)
    return df, pd.DataFrame(changes)


def apply_k1_floor(frozen, floor, strict=False):
    """Detected orbits with K1 < floor become ND (orbit fields blanked).

    ``strict`` also turns the period-lower-limit stars into ND: they have
    no measured K1, so whether they lie above the floor is not known.
    Returns (catalogue, ids below the floor, ids turned ND for unknown K1).
    """
    df = frozen.copy()
    low = (df["outcome"] != "ND") & (df["K1"] < float(floor))
    ids = sorted(df.loc[low, "star_id"])
    df.loc[low, ["outcome", "reason"]] = ["ND", "below_k1_floor"]
    df.loc[low, ORBIT_COLS] = np.nan
    df.loc[low, "e_is_upper_limit"] = False
    unknown = []
    if strict:
        m = (df["outcome"] == "DO") & (df["reason"] == "P_lower_limit")
        unknown = sorted(df.loc[m, "star_id"])
        df.loc[m, ["outcome", "reason"]] = ["ND", "k1_unknown_under_floor"]
        df.loc[m, "P_lower_limit"] = np.nan
    return df, ids, unknown


def add_doubtful(frozen, ids):
    """ND stars in ``ids`` become detections without a usable period."""
    df = frozen.copy()
    m = df["star_id"].isin(ids)
    if int(m.sum()) != len(set(ids)) or not (df.loc[m, "outcome"]
                                             == "ND").all():
        raise AssertionError("doubtful singles must all be ND stars of the "
                             "catalogue: %s" % sorted(ids))
    df.loc[m, ["outcome", "reason"]] = ["DO", "doubtful_single"]
    return df


# Outcome counts expected when the period cut is moved to 10 d and 20 d
# (kappa / period check, 2026-10-13): the orbits between the adopted cut and
# the new one become below-cut detections.
CUT_COUNTS = {
    10.0: {"ND": 57, "DX": 40, "DO": 37, "DX_LS": 9,
           "below_cut": 31, "P_lower_limit": 3, "no_solution": 3},
    20.0: {"ND": 57, "DX": 33, "DO": 44, "DX_LS": 7,
           "below_cut": 38, "P_lower_limit": 3, "no_solution": 3},
}


def apply_cut(frozen, p_cut_days):
    """The frozen catalogue with the period cut moved up to ``p_cut_days``.

    Orbits between the frozen cut and the new one change from DX to
    DO/below_cut; their orbit columns are kept (the split likelihood does
    not use them unless ``below_shape`` is set). ``logP_cut`` is replaced in
    every row. The period-lower-limit stars must stay above the new cut.
    Returns (catalogue, ids moved below the cut)."""
    new_cut = float(np.log10(float(p_cut_days)))
    old_cut = float(frozen["logP_cut"].iloc[0])
    if new_cut <= old_cut:
        raise ValueError("apply_cut only moves the cut up (frozen %.4f, "
                         "asked %.4f)" % (old_cut, new_cut))
    df = frozen.copy()
    move = (df["outcome"] == "DX") & (df["logP"] < new_cut)
    ids = sorted(df.loc[move, "star_id"])
    df.loc[move, ["outcome", "reason"]] = ["DO", "below_cut"]
    lim = df.loc[(df["outcome"] == "DO") & (df["reason"] == "P_lower_limit"),
                 "P_lower_limit"].astype(float)
    if (lim < float(p_cut_days)).any():
        raise AssertionError("period lower limits below the new cut of %g d: "
                             "%s" % (p_cut_days, sorted(lim[lim < p_cut_days])))
    df["logP_cut"] = new_cut
    return df, ids


def write(df, out_dir, name):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "obs_catalog.csv")
    df[COLUMNS].to_csv(path, index=False)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--frozen", required=True,
                    help="obs_catalog.csv of the adopted fit")
    ap.add_argument("--tables", required=True,
                    help="paper tables dir (sb1/sb2_solutions.csv, "
                         "ostar_catalog.csv)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k1-floor", type=float, default=10.0)
    ap.add_argument("--smooth-sigma", type=float, default=0.15)
    ap.add_argument("--cuts", default=None,
                    help="comma list of period cuts in days: write only the "
                         "catalogues cut<P>d/ with the cut moved there "
                         "(apply_cut) and cut_report.txt, nothing else")
    a = ap.parse_args(argv)

    frozen = load_frozen(a.frozen)
    cut = float(frozen["logP_cut"].iloc[0])
    if a.cuts:
        os.makedirs(a.out_dir, exist_ok=True)
        lines = ["frozen catalogue: %s (md5 %s)" % (os.path.abspath(a.frozen),
                                                    md5(a.frozen)),
                 "frozen logP* = %r (P* = %.4f d)" % (cut, 10.0 ** cut)]
        for pc in (float(v) for v in a.cuts.split(",")):
            df, ids = apply_cut(frozen, pc)
            name = "cut%gd" % pc
            if pc in CUT_COUNTS:
                check_counts(df, CUT_COUNTS[pc], name)
            elif len(df) != 134 or df["star_id"].duplicated().any():
                raise AssertionError("%s: not 134 unique stars" % name)
            path = write(df, a.out_dir, name)
            lines += ["", "== %s: %s (md5 %s)" % (name, path, md5(path)),
                      "logP_cut = %r" % float(df["logP_cut"].iloc[0]),
                      "counts %s" % counts(df),
                      "DX -> DO/below_cut (%d): %s" % (len(ids), ids)]
        report = os.path.join(a.out_dir, "cut_report.txt")
        with open(report, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print("\nwrote %s" % report)
        return 0
    rows = load_solution_rows(a.tables)
    catalog_csv = os.path.join(a.tables, "ostar_catalog.csv")
    os.makedirs(a.out_dir, exist_ok=True)
    lines = ["frozen catalogue: %s (md5 %s)" % (os.path.abspath(a.frozen),
                                                md5(a.frozen)),
             "tables: %s" % os.path.abspath(a.tables)]
    for f in ("sb1_solutions.csv", "sb2_solutions.csv", "ostar_catalog.csv"):
        lines.append("  %s md5 %s" % (f, md5(os.path.join(a.tables, f))))
    lines.append("logP* (frozen, kept in every catalogue) = %r  (P* = %.4f d)"
                 % (cut, 10.0 ** cut))

    # the frozen catalogue must survive a read/write round trip unchanged,
    # so that the derived catalogues differ from it in the edited rows only
    rt = write(frozen, a.out_dir, "_roundtrip")
    same = md5(rt) == md5(a.frozen)
    lines.append("round trip of the frozen catalogue byte-identical: %s"
                 % same)
    if not same:
        raise AssertionError("the frozen catalogue does not survive a "
                             "read/write round trip")

    # ---- fullprec -------------------------------------------------------
    full, ch = rebuild_fullprec(frozen, rows, catalog_csv)
    check_counts(full, FROZEN_COUNTS, "fullprec")
    moved = ch[ch["outcome_old"] != ch["outcome_new"]]
    flag = ch[ch["ls_old"] != ch["ls_new"]]
    if len(moved) or len(flag):
        raise AssertionError("fullprec changes an outcome or a Lucy-Sweeney "
                             "flag: %s" % sorted(set(moved["star_id"])
                                                 | set(flag["star_id"])))
    p_full = write(full, a.out_dir, "fullprec")
    ch.to_csv(os.path.join(a.out_dir, "fullprec", "changes_vs_frozen.csv"),
              index=False)
    lines += ["", "== fullprec: %s (md5 %s)" % (p_full, md5(p_full)),
              "counts %s" % counts(full),
              "rows re-read: %d; table changed: %s"
              % (len(ch), ch.loc[ch["table_old"] != ch["table_new"],
                                 ["star_id", "table_old",
                                  "table_new"]].values.tolist()),
              "period kept from the frozen catalogue (agrees with the table "
              "at four decimals): %d rows; re-read from the table: %d rows"
              % (int((ch["P_source"] == "frozen").sum()),
                 int((ch["P_source"] == "table").sum())),
              "largest shifts (new minus frozen):"]
    with pd.option_context("display.width", 200):
        for col, n in (("dlogK1", 8), ("de", 8), ("dlogP", 5)):
            top = ch.reindex(ch[col].abs().sort_values(
                ascending=False).index).head(n)
            lines.append("  by |%s|:" % col)
            lines += ["    " + ln for ln in top[
                ["star_id", "P_old", "P_new", "e_old", "e_new", "K1_old",
                 "K1_new"]].to_string(index=False).splitlines()]

    # inflection point of the smoothed period CDF, frozen and re-read lists.
    # The frozen list is the frozen catalogue's orbits plus the second
    # orbit of every star that has two usable rows (the CvM scorer pooled
    # the table rows), read here from the tables.
    orb = rows[rows["kind"] == "orbit"]
    second = orb[orb["row_label"].isin(list(POOLED_SECOND_ROWS))]
    froz_orb = frozen[frozen["logP"].notna()]
    logp_frozen = np.concatenate([
        froz_orb["logP"].to_numpy(float),
        np.log10([POOLED_SECOND_ROWS[k] for k in second["row_label"]])])
    # re-read list: the fullprec orbits plus the pooled second rows
    logp_full = np.concatenate([full.loc[full["logP"].notna(),
                                         "logP"].to_numpy(float),
                                second["logP"].to_numpy(float)])
    c_frozen = _numerical_logP_cutoff(logp_frozen, smooth_sigma=a.smooth_sigma)
    c_full = _numerical_logP_cutoff(logp_full, smooth_sigma=a.smooth_sigma)
    if abs(c_frozen - cut) > 1e-12:
        raise AssertionError("the frozen period list does not reproduce the "
                             "stored logP*: %.12f vs %.12f" % (c_frozen, cut))
    lines += ["", "== inflection point of the smoothed period CDF "
              "(sigma %.2f dex)" % a.smooth_sigma,
              "frozen list (%d orbits, incl. the pooled second rows %s): "
              "logP* = %.10f (P* = %.4f d), reproduces the stored value "
              "(difference %.1e)"
              % (logp_frozen.size, list(second["row_label"]), c_frozen,
                 10.0 ** c_frozen, c_frozen - cut),
              "re-read list (%d orbits): logP* = %.10f (P* = %.4f d); "
              "shift from the stored value %+.2e dex"
              % (logp_full.size, c_full, 10.0 ** c_full, c_full - cut),
              "stars between the two cuts: %s"
              % sorted(full.loc[full["logP"].between(
                  min(cut, c_full), max(cut, c_full)), "star_id"])]

    # ---- k1floor --------------------------------------------------------
    flo, ids, _ = apply_k1_floor(frozen, a.k1_floor)
    ids_tab = sorted(full.loc[(full["outcome"] != "ND")
                              & (full["K1"] < a.k1_floor), "star_id"])
    if ids != ids_tab:
        raise AssertionError("stars below the floor differ between the "
                             "frozen catalogue %s and the tables %s"
                             % (ids, ids_tab))
    if a.k1_floor == 10.0 and ids != EXPECTED_BELOW_10:
        raise AssertionError("stars below 10 km/s: %s, expected %s"
                             % (ids, EXPECTED_BELOW_10))
    n_ls = int(frozen.loc[frozen["star_id"].isin(ids)
                          & (frozen["outcome"] == "DX"),
                          "e_is_upper_limit"].sum())
    was = frozen.loc[frozen["star_id"].isin(ids), "outcome"].value_counts()
    exp = dict(FROZEN_COUNTS)
    exp["ND"] += len(ids)
    exp["DX"] -= int(was.get("DX", 0))
    exp["DO"] -= int(was.get("DO", 0))
    exp["DX_LS"] -= n_ls
    exp["below_cut"] -= int(was.get("DO", 0))
    check_counts(flo, exp, "k1floor")
    name = "k1floor%g" % a.k1_floor
    p_flo = write(flo, a.out_dir, name)
    lines += ["", "== %s: %s (md5 %s)" % (name, p_flo, md5(p_flo)),
              "counts %s" % counts(flo),
              "became ND (K1 < %g km/s): %s; of these %d were Lucy-Sweeney "
              "circular; frozen outcomes %s"
              % (a.k1_floor, ids, n_ls, was.to_dict()),
              "frozen K1 of these: %s"
              % frozen.loc[frozen["star_id"].isin(ids),
                           ["star_id", "K1"]].values.tolist(),
              "detected stars without a K1, kept as detections and scored "
              "as lying above the floor (run with --k1-unknown-above-floor)"
              ": period lower limits %s, no solution %s"
              % (sorted(flo.loc[(flo["outcome"] == "DO")
                                & (flo["reason"] == "P_lower_limit"),
                                "star_id"]),
                 sorted(flo.loc[(flo["outcome"] == "DO")
                                & (flo["reason"] == "no_solution"),
                                "star_id"]))]
    # strict variant: the period-lower-limit stars as ND too
    strict, ids_s, unknown = apply_k1_floor(frozen, a.k1_floor, strict=True)
    if ids_s != ids:
        raise AssertionError("strict floor changed the below-floor set")
    exp_s = dict(exp)
    exp_s["ND"] += len(unknown)
    exp_s["DO"] -= len(unknown)
    exp_s.pop("P_lower_limit")
    if len(unknown) != FROZEN_COUNTS["P_lower_limit"]:
        raise AssertionError("period-lower-limit stars: %s" % unknown)
    check_counts(strict, exp_s, "k1floor_strict")
    name_s = name + "_strict"
    p_strict = write(strict, a.out_dir, name_s)
    lines += ["", "== %s: %s (md5 %s)" % (name_s, p_strict, md5(p_strict)),
              "counts %s" % counts(strict),
              "as %s, with the period-lower-limit stars %s entered as ND "
              "(no measured K1); the no-solution stars %s stay detected"
              % (name, unknown,
                 sorted(strict.loc[(strict["outcome"] == "DO")
                                   & (strict["reason"] == "no_solution"),
                                   "star_id"]))]

    # ---- doubtful -------------------------------------------------------
    doubt_ids = RV_VARIABLE_SINGLES + ROCHE_VETOED_SINGLES
    cat = pd.read_csv(catalog_csv, dtype=str)
    single = cat["Binary status"].astype(str).str.strip() == SINGLE_STATUS
    rv_yes = sorted(cat.loc[single & (cat["Classical RV variable"].astype(str)
                                      .str.strip().str.lower() == "yes"),
                            "BLOeM ID"].astype(str).str.strip())
    if rv_yes != sorted(RV_VARIABLE_SINGLES):
        raise AssertionError("classical RV variables among the singles in "
                             "the catalogue: %s, expected %s"
                             % (rv_yes, sorted(RV_VARIABLE_SINGLES)))
    dbt = add_doubtful(frozen, doubt_ids)
    exp = dict(FROZEN_COUNTS)
    exp["ND"] -= len(doubt_ids)
    exp["DO"] += len(doubt_ids)
    exp["doubtful_single"] = len(doubt_ids)
    check_counts(dbt, exp, "doubtful")
    p_dbt = write(dbt, a.out_dir, "doubtful")
    lines += ["", "== doubtful: %s (md5 %s)" % (p_dbt, md5(p_dbt)),
              "counts %s" % counts(dbt),
              "became DO/doubtful_single: %s (classical RV variables, "
              "checked against the catalogue) + %s (Roche-lobe vetoed)"
              % (RV_VARIABLE_SINGLES, ROCHE_VETOED_SINGLES)]

    report = os.path.join(a.out_dir, "catalog_report.txt")
    with open(report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote %s" % report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
