"""
pipeline.compare_runs_report — Per-star PDF comparison of two evaluator runs.

Compares the best orbital solution (standard STRICT filter on BICc), the
null-model fit (gamma, ln_sigmaJ, BICc) and the MCMC posteriors of two runs
of the pipeline over the same sample, embedding each run's own phase-folded
fit figure per star. Stars whose results changed in an interesting way
(detection flip, non-alias period change, >3-sigma posterior shift, or a
null-model gamma shift >2 km/s) are flagged, and every star page carries a
two-sentence rule-generated analysis.

Usage (defaults compare second_pdc vs fourth_fixed_rvs):
    python -m pipeline.compare_runs_report
    python -m pipeline.compare_runs_report --stars BLOeM_2-079,BLOeM_4-074 --output /tmp/smoke.pdf
"""

import argparse
import contextlib
import io
import os
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from mcmc.selector_app import get_best_row
from simulations.ht_report import AA_RC

# cmr10 lacks glyphs for '_', '>', '|' (star names, model notes, captions);
# DejaVu Serif keeps the serif look with full coverage.
RC = dict(AA_RC, **{"font.serif": ["DejaVu Serif"],
                    "axes.formatter.use_mathtext": False})

BASE = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_neb_div_from_coadded"

# keep in sync with run_mcmc_non_rl.py (copied verbatim; do not import that module)
STRICT = (
    "~candidate_method.str.contains('MANUAL')"
    " & ~candidate_method.str.contains('null')"
    " & candidate_method.str.contains('jitter')"
    " & (prob_bicc > 0.5)"
    " & (mass_flag_peri == 0)"
    " & ((candidate_method.str.contains('LS') & (LS_iter_fap < 0.2))"
    " | (candidate_method.str.contains('PDC') & (PDC_iter_fap < 0.2)))"
)

PAGE = (11.0, 8.5)  # US Letter landscape

SCALARS = [
    "Period_value", "GAMMA_value", "K1_value", "OMEGA_rad_value",
    "Eccentricity_value", "T0_value", "ln_sigmaJ_value",
    "bicc", "prob_bicc", "redchi",
    "candidate_method", "candidate_period", "solution_id",
]

# (table label, lmfit column, format, MCMC name or None)
PARAM_ROWS = [
    ("P [d]",       "Period_value",       "{:.6g}", "P"),
    ("K1 [km/s]",   "K1_value",           "{:.2f}", "K1"),
    ("e",           "Eccentricity_value", "{:.3f}", "e"),
    ("omega [rad]", "OMEGA_rad_value",    "{:.3f}", None),
    ("gamma [km/s]", "GAMMA_value",       "{:.2f}", "gamma"),
    ("T0 [MJD]",    "T0_value",           "{:.3f}", None),
    ("ln sigma_J",  "ln_sigmaJ_value",    "{:.2f}", None),
    ("BICc",        "bicc",               "{:.1f}", None),
    ("prob_bicc",   "prob_bicc",          "{:.3f}", None),
    ("redchi",      "redchi",             "{:.2f}", None),
]

# MCMC column stem per comparable parameter ({b} = shared block)
MCMC_PAR = {"P": "{b}_P", "K1": "{b}_K1", "e": "ecc_e", "gamma": "{b}_gamma"}

STATUS_COLORS = {
    "detected-both": "#2e7d32", "gained": "#1565c0", "lost": "#ef6c00",
    "none": "#757575", "OLD-only": "#616161", "NEW-only": "#616161",
}

EXPECTED_GAINED = {"BLOeM_4-074", "BLOeM_6-091", "BLOeM_7-044"}
EXPECTED_LOST = {"BLOeM_2-079"}


def _fmt(v, spec="{:.3g}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def discover_stars(run_dir):
    return sorted(
        d for d in os.listdir(run_dir)
        if os.path.isfile(os.path.join(run_dir, d, "lmfit_summary.csv"))
    )


def load_run(run_dir):
    out = {}
    for star in discover_stars(run_dir):
        df = pd.read_csv(os.path.join(run_dir, star, "lmfit_summary.csv"))
        nulls = df[df["candidate_method"] == "null_hyp_jitter"]
        with contextlib.redirect_stdout(io.StringIO()):
            best = get_best_row(df, STRICT, "bicc", take_min=True)
        out[star] = {"best": best,
                     "null": nulls.iloc[0] if len(nulls) else None}
    return out


def run_frame(recs, suffix):
    rows = []
    for star, d in recs.items():
        r = {"star": star, f"in{suffix}": True,
             f"detected{suffix}": d["best"] is not None}
        if d["best"] is not None:
            for c in SCALARS:
                r[f"{c}{suffix}"] = d["best"].get(c, np.nan)
        if d["null"] is not None:
            r[f"null_GAMMA{suffix}"] = d["null"]["GAMMA_value"]
            r[f"null_lnsJ{suffix}"] = d["null"]["ln_sigmaJ_value"]
            r[f"null_bicc{suffix}"] = d["null"]["bicc"]
            r[f"null_sid{suffix}"] = d["null"]["solution_id"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("star")


def load_mcmc(csv_path, suffix):
    m = pd.read_csv(csv_path)
    m["star_name"] = m["star_name"].astype(str).str.strip()
    m = m.set_index("star_name")

    def _block(row):
        if pd.notna(row.get("ecc_P_median")):
            return "ecc"
        if pd.notna(row.get("circ_P_median")):
            return "circ"
        if pd.notna(row.get("null_gamma_median")):
            return "null"
        return None

    m["block"] = m.apply(_block, axis=1)
    return m.add_suffix(suffix)


def star_status(rec):
    if rec["in_old"] and not rec["in_new"]:
        return "OLD-only"
    if rec["in_new"] and not rec["in_old"]:
        return "NEW-only"
    if rec["detected_old"] and rec["detected_new"]:
        return "detected-both"
    if rec["detected_new"]:
        return "gained"
    if rec["detected_old"]:
        return "lost"
    return "none"


def _sigma_comb(rec, col):
    so = 0.5 * (abs(rec.get(f"{col}_errm_mold", np.nan))
                + abs(rec.get(f"{col}_errp_mold", np.nan)))
    sn = 0.5 * (abs(rec.get(f"{col}_errm_mnew", np.nan))
                + abs(rec.get(f"{col}_errp_mnew", np.nan)))
    return np.sqrt(so ** 2 + sn ** 2)


def _mcmc_nsig(rec, name):
    """Delta(median)/sigma_comb for a parameter, or NaN when not comparable."""
    bo, bn = rec.get("block_mold"), rec.get("block_mnew")
    if bo != bn or bo not in ("ecc", "circ"):
        return np.nan
    if name == "e" and bo != "ecc":
        return np.nan
    col = MCMC_PAR[name].format(b=bo)
    mo = rec.get(f"{col}_median_mold", np.nan)
    mn = rec.get(f"{col}_median_mnew", np.nan)
    sig = _sigma_comb(rec, col)
    if not (np.isfinite(mo) and np.isfinite(mn)) or not np.isfinite(sig) or sig <= 0:
        return np.nan
    return (mn - mo) / sig


def evaluate_interest(rec):
    """Composite trigger: (is_interesting, reasons, alias_note, model_note)."""
    reasons, alias_note, model_note = [], None, None
    st = rec["status"]
    if st == "gained":
        reasons.append("detection gained")
    if st == "lost":
        reasons.append("detection lost")

    if st == "detected-both":
        ratio = rec["Period_value_new"] / rec["Period_value_old"]
        rel = {r: abs(ratio - r) / r for r in (1.0, 0.5, 2.0)}
        if rel[1.0] > 0.05:
            if rel[0.5] <= 0.05:
                alias_note = f"1:2 period alias (Pnew/Pold={ratio:.3f})"
            elif rel[2.0] <= 0.05:
                alias_note = f"2:1 period alias (Pnew/Pold={ratio:.3f})"
            else:
                reasons.append(
                    f"period changed {100 * abs(ratio - 1):.0f}% (no 1:2/2:1 alias)")

    bo, bn = rec.get("block_mold"), rec.get("block_mnew")
    if isinstance(bo, str) and isinstance(bn, str) and bo != bn:
        model_note = f"MCMC model changed: {bo} to {bn}"
    for name in MCMC_PAR:
        nsig = _mcmc_nsig(rec, name)
        if np.isfinite(nsig) and abs(nsig) > 3:
            reasons.append(f"{name} shifted {abs(nsig):.1f} sigma (MCMC)")

    dg = rec.get("null_GAMMA_new", np.nan) - rec.get("null_GAMMA_old", np.nan)
    if np.isfinite(dg) and abs(dg) > 2.0:
        reasons.append(f"null gamma shift {dg:+.2f} km/s")

    return len(reasons) > 0, reasons, alias_note, model_note


def build_analysis(rec, label_a, label_b):
    """Two-sentence rule-generated analysis for one star."""
    st = rec["status"]

    if st == "OLD-only":
        s1 = f"This target is present only in {label_a} and has no counterpart in the new run."
    elif st == "NEW-only":
        s1 = f"This target is present only in {label_b} and has no counterpart in the old run."
    elif st == "gained":
        s1 = (f"With the zero-point-corrected RVs the star now passes the detection filter "
              f"at P = {_fmt(rec.get('Period_value_new'), '{:.4g}')} d "
              f"(K1 = {_fmt(rec.get('K1_value_new'), '{:.1f}')} km/s), "
              f"whereas the original RVs yielded no accepted orbit.")
    elif st == "lost":
        s1 = (f"The P = {_fmt(rec.get('Period_value_old'), '{:.4g}')} d solution accepted "
              f"with the original RVs no longer passes the detection filter after the "
              f"zero-point correction.")
    elif st == "none":
        s1 = "Neither run yields an orbital solution that passes the detection filter."
    else:  # detected-both
        po, pn = rec["Period_value_old"], rec["Period_value_new"]
        if rec.get("alias_note"):
            ppart = (f"the best period moves from {po:.4g} to {pn:.4g} d "
                     f"({rec['alias_note']})")
        elif abs(pn / po - 1) > 0.05:
            ppart = (f"the best period moves from {po:.4g} to {pn:.4g} d "
                     f"with no simple alias relation")
        else:
            ppart = f"the best period is essentially unchanged ({po:.4g} to {pn:.4g} d)"
        shifts = []
        for name in ("K1", "e", "gamma"):
            nsig = _mcmc_nsig(rec, name)
            if np.isfinite(nsig) and abs(nsig) > 1:
                shifts.append(f"{name} shifts by {abs(nsig):.1f} sigma")
        extra = ""
        if shifts:
            extra = ", while " + " and ".join(shifts)
        elif rec.get("model_note"):
            extra = f", and the preferred model changed ({rec['model_note'].split(': ')[-1]})"
        s1 = f"Both runs detect the star: {ppart}{extra}."

    dg = rec.get("null_GAMMA_new", np.nan) - rec.get("null_GAMMA_old", np.nan)
    dj = rec.get("null_lnsJ_new", np.nan) - rec.get("null_lnsJ_old", np.nan)
    if np.isfinite(dg) and np.isfinite(dj):
        if abs(dg) <= 2.0 and abs(dj) <= 0.5:
            s2 = (f"The null-model gamma shifts by {dg:+.2f} km/s and ln sigma_J by "
                  f"{dj:+.2f}, consistent with a pure zero-point correction.")
        else:
            s2 = (f"The null-model gamma shifts by {dg:+.2f} km/s and ln sigma_J by "
                  f"{dj:+.2f}, a substantive change that merits a closer look.")
    else:
        s2 = "A null-model comparison is not available for this star."
    return f"{s1} {s2}"


def build_merged(dir_a, dir_b, mcmc_a, mcmc_b, label_a, label_b):
    merged = run_frame(load_run(dir_a), "_old").join(
        run_frame(load_run(dir_b), "_new"), how="outer")
    if os.path.exists(mcmc_a):
        merged = merged.join(load_mcmc(mcmc_a, "_mold"), how="left")
    if os.path.exists(mcmc_b):
        merged = merged.join(load_mcmc(mcmc_b, "_mnew"), how="left")
    merged = merged.sort_index()
    for c in ("in_old", "in_new", "detected_old", "detected_new"):
        if c not in merged.columns:
            merged[c] = False
        merged[c] = merged[c].where(merged[c].notna(), False).astype(bool)
    both = int((merged["in_old"] & merged["in_new"]).sum())
    assert both >= min(100, len(merged) - 2), \
        f"only {both} stars common to both runs — star-name mismatch?"

    merged["status"] = merged.apply(star_status, axis=1)
    interest = merged.apply(evaluate_interest, axis=1)
    merged["is_interesting"] = [t[0] for t in interest]
    merged["interest_reasons"] = [t[1] for t in interest]
    merged["alias_note"] = [t[2] for t in interest]
    merged["model_note"] = [t[3] for t in interest]
    merged["analysis"] = merged.apply(
        lambda r: build_analysis(r, label_a, label_b), axis=1)
    return merged


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_path(run_dir, star, sid, kind):
    return os.path.join(run_dir, star, "lmfit_solutions",
                        f"{star}_sid-{int(sid)}_{kind}_residuals.png")


def pick_figure(run_dir, star, rec, side, label):
    """Return (png path or None, caption) for one run's panel."""
    detected = rec[f"detected_{side}"]
    if not rec[f"in_{side}"]:
        return None, f"{label}: star not present in this run"
    if detected:
        sid = rec[f"solution_id_{side}"]
        method = rec.get(f"candidate_method_{side}", "")
        pcand = rec.get(f"candidate_period_{side}", np.nan)
        cap = (f"{label}  |  sid={int(sid)}  |  {method}  |  "
               f"P_cand={_fmt(pcand, '{:.4g}')} d")
        for kind in ("phase", "time"):
            p = fig_path(run_dir, star, sid, kind)
            if os.path.exists(p):
                return p, cap
        return None, cap
    sid = rec.get(f"null_sid_{side}", np.nan)
    cap = f"{label}  |  null model (non-detection) - time residuals"
    if np.isfinite(sid):
        p = fig_path(run_dir, star, sid, "time")
        if os.path.exists(p):
            return p, cap
    return None, cap


def load_thumb(path, thumb_px):
    try:
        im = Image.open(path)
        im.thumbnail((thumb_px, 900), Image.LANCZOS)
        return np.asarray(im)
    except Exception:
        return None


def _draw_panel(ax, path, caption, thumb_px):
    ax.axis("off")
    ax.set_title(caption, fontsize=8)
    arr = load_thumb(path, thumb_px) if path else None
    if arr is None:
        name = os.path.basename(path) if path else "no figure for this run"
        if path:
            print(f"[missing fig] {path}")
        ax.text(0.5, 0.5, f"figure not found:\n{name}", ha="center",
                va="center", fontsize=8, color="0.4",
                bbox=dict(fc="#eeeeee", ec="none"))
        return
    ax.imshow(arr)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _chip(ax, x, y, text, color, fontsize=9):
    ax.text(x, y, text, fontsize=fontsize, color="white", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none"))
    return x + 0.016 * len(text) + 0.05


def page_summary(pdf, merged, meta):
    mask = merged["in_old"] & merged["in_new"]
    gained = sorted(merged.index[(merged["status"] == "gained") & mask])
    lost = sorted(merged.index[(merged["status"] == "lost") & mask])
    n_det_old = int((merged["detected_old"] & mask).sum())
    n_det_new = int((merged["detected_new"] & mask).sum())

    left = [
        "Run comparison report",
        "",
        f"OLD (A): {meta['label_a']}",
        f"         {meta['dir_a']}",
        f"NEW (B): {meta['label_b']}",
        f"         {meta['dir_b']}",
        "",
        f"Star dirs: OLD {int(merged['in_old'].sum())}, "
        f"NEW {int(merged['in_new'].sum())}, common {int(mask.sum())}",
        f"Detections (common stars, STRICT filter on BICc): "
        f"OLD {n_det_old}, NEW {n_det_new}",
        f"Gained in NEW: {', '.join(s.replace('BLOeM_', '') for s in gained) or 'none'}",
        f"Lost in NEW:   {', '.join(s.replace('BLOeM_', '') for s in lost) or 'none'}",
        "",
        "Interesting = detection flip, period change >5% (non-alias),",
        ">3-sigma MCMC posterior shift (P/K1/e/gamma, same model block),",
        "or null-model gamma shift >2 km/s.",
        "Stars present in only one run are shown but excluded from counts.",
    ]

    listing = ["Interesting stars:", ""]
    for star in merged.index[merged["is_interesting"]]:
        listing.append(f"{star.replace('BLOeM_', ''):10s} [{merged.at[star, 'status']}]")
        for r in merged.at[star, "interest_reasons"]:
            listing.append(f"    - {r}")
    if len(listing) == 2:
        listing.append("(none)")

    # metadata on top of page 1; interesting list flows below it in two
    # columns, continuing on extra pages if needed
    first_lines, cont_lines = 22, 44
    cols_first = [listing[:first_lines], listing[first_lines:2 * first_lines]]
    rest = listing[2 * first_lines:]
    cont_pages = [rest[i:i + 2 * cont_lines] for i in
                  range(0, len(rest), 2 * cont_lines)]
    n_pages = 1 + len(cont_pages)

    fig = plt.figure(figsize=PAGE)
    fig.text(0.05, 0.97, f"Summary (1/{n_pages})", fontsize=13, fontweight="bold")
    fig.text(0.05, 0.92, "\n".join(left), fontsize=9,
             fontfamily="monospace", va="top")
    for k, chunk in enumerate(cols_first):
        if chunk:
            fig.text(0.05 + 0.47 * k, 0.52, "\n".join(chunk), fontsize=8,
                     fontfamily="monospace", va="top")
    pdf.savefig(fig)
    plt.close(fig)

    for i, page in enumerate(cont_pages, start=2):
        fig = plt.figure(figsize=PAGE)
        fig.text(0.05, 0.97, f"Summary ({i}/{n_pages})",
                 fontsize=13, fontweight="bold")
        for k in range(2):
            chunk = page[k * cont_lines:(k + 1) * cont_lines]
            if chunk:
                fig.text(0.05 + 0.47 * k, 0.92, "\n".join(chunk), fontsize=8,
                         fontfamily="monospace", va="top")
        pdf.savefig(fig)
        plt.close(fig)


def page_overview(pdf, merged):
    mask = merged["in_old"] & merged["in_new"]
    det = merged[mask & merged["detected_old"] & merged["detected_new"]]
    common = merged[mask]

    fig = plt.figure(figsize=PAGE)
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.90,
                          bottom=0.08, hspace=0.32, wspace=0.28)
    fig.text(0.05, 0.95, "Global overview (stars detected in both runs, "
             "unless noted)", fontsize=12, fontweight="bold")

    def scatter(ax, xcol, ycol, title, log=False):
        x, y = det[xcol], det[ycol]
        colors = ["#c62828" if det.at[s, "is_interesting"]
                  else ("#f9a825" if det.at[s, "alias_note"] else "0.3")
                  for s in det.index]
        ax.scatter(x, y, s=14, c=colors, alpha=0.8, lw=0)
        lo = np.nanmin([x.min(), y.min()])
        hi = np.nanmax([x.max(), y.max()])
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8)
        if log:
            ax.plot([lo, hi], [2 * lo, 2 * hi], color="0.8", lw=0.7, ls="--")
            ax.plot([lo, hi], [0.5 * lo, 0.5 * hi], color="0.8", lw=0.7, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
        for s in det.index:
            if det.at[s, "is_interesting"]:
                ax.annotate(s.replace("BLOeM_", ""),
                            (det.at[s, xcol], det.at[s, ycol]),
                            fontsize=6, color="#c62828",
                            xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("OLD")
        ax.set_ylabel("NEW")
        ax.set_title(title, fontsize=10)

    scatter(fig.add_subplot(gs[0, 0]), "Period_value_old", "Period_value_new",
            "P [d] (guides: 1:1, 1:2, 2:1)", log=True)
    scatter(fig.add_subplot(gs[0, 1]), "K1_value_old", "K1_value_new", "K1 [km/s]")
    scatter(fig.add_subplot(gs[0, 2]), "GAMMA_value_old", "GAMMA_value_new",
            "gamma [km/s]")
    scatter(fig.add_subplot(gs[1, 0]), "Eccentricity_value_old",
            "Eccentricity_value_new", "e (lmfit)")

    ax = fig.add_subplot(gs[1, 1])
    dg = (common["null_GAMMA_new"] - common["null_GAMMA_old"]).dropna()
    ax.hist(dg, bins=30, color="0.55")
    for v in (-2, 2):
        ax.axvline(v, color="crimson", ls="--", lw=0.9)
    ax.set_xlabel("null gamma NEW - OLD [km/s]")
    ax.set_title("null-model gamma shift (all common stars)", fontsize=10)

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(common["null_lnsJ_old"], common["null_lnsJ_new"], s=12,
               c="0.3", alpha=0.7, lw=0)
    lo = np.nanmin(common[["null_lnsJ_old", "null_lnsJ_new"]].values)
    hi = np.nanmax(common[["null_lnsJ_old", "null_lnsJ_new"]].values)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8)
    ax.set_xlabel("OLD")
    ax.set_ylabel("NEW")
    ax.set_title("null-model ln sigma_J", fontsize=10)

    pdf.savefig(fig)
    plt.close(fig)


def _style_table(tbl, n_cols, red_cells=(), amber_rows=()):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_text_props(fontweight="bold", fontsize=8)
        cell.set_facecolor("#d9e2f3")
    for (i, j) in red_cells:
        tbl[i, j].set_facecolor("#fce4e4")
    for i in amber_rows:
        for j in range(n_cols):
            tbl[i, j].set_facecolor("#fff3cd")
    for _, cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")


def _param_table(ax, rec):
    ax.axis("off")
    cols = ["Param", "OLD", "NEW", "Delta (N-O)", "Delta/sigma (MCMC)"]
    cells, red_cells, amber_rows = [], [], []
    for i, (label, col, fmt, mname) in enumerate(PARAM_ROWS, start=1):
        vo = rec.get(f"{col}_old", np.nan)
        vn = rec.get(f"{col}_new", np.nan)
        delta = (vn - vo) if (np.isfinite(vo) and np.isfinite(vn)) else np.nan
        nsig = _mcmc_nsig(rec, mname) if mname else np.nan
        cells.append([label, _fmt(vo, fmt), _fmt(vn, fmt),
                      _fmt(delta, "{:+.4g}"), _fmt(nsig, "{:+.1f}")])
        if np.isfinite(nsig) and abs(nsig) > 3:
            red_cells.append((i, 4))
        if label.startswith("P ") and rec.get("alias_note"):
            amber_rows.append(i)
    tbl = ax.table(cellText=cells, colLabels=cols, loc="center",
                   cellLoc="center",
                   colWidths=[0.22, 0.21, 0.21, 0.19, 0.17])
    _style_table(tbl, len(cols), red_cells, amber_rows)


def _null_table(ax, rec):
    ax.axis("off")
    rows = [("gamma [km/s]", "null_GAMMA", "{:.2f}"),
            ("ln sigma_J", "null_lnsJ", "{:.2f}"),
            ("BICc", "null_bicc", "{:.1f}")]
    cells, red_cells = [], []
    for i, (label, stem, fmt) in enumerate(rows, start=1):
        vo = rec.get(f"{stem}_old", np.nan)
        vn = rec.get(f"{stem}_new", np.nan)
        delta = (vn - vo) if (np.isfinite(vo) and np.isfinite(vn)) else np.nan
        cells.append([label, _fmt(vo, fmt), _fmt(vn, fmt), _fmt(delta, "{:+.3g}")])
        if stem == "null_GAMMA" and np.isfinite(delta) and abs(delta) > 2:
            red_cells.extend((i, j) for j in range(4))
    tbl = ax.table(cellText=cells, colLabels=["Null model", "OLD", "NEW", "Delta"],
                   loc="center", cellLoc="center",
                   colWidths=[0.34, 0.23, 0.23, 0.20])
    _style_table(tbl, 4, red_cells)

    notes = [n for n in (rec.get("model_note"),) if n]
    if not isinstance(rec.get("block_mold"), str):
        notes.append("no MCMC row (OLD)")
    if not isinstance(rec.get("block_mnew"), str):
        notes.append("no MCMC row (NEW)")
    if notes:
        ax.text(0.0, -0.02, "; ".join(notes), fontsize=7.5,
                style="italic", color="0.35", transform=ax.transAxes)


def page_star(pdf, star, rec, dir_a, dir_b, label_a, label_b, thumb_px):
    fig = plt.figure(figsize=PAGE)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.17, 0.31, 0.52],
                          left=0.045, right=0.965, top=0.97,
                          bottom=0.02, hspace=0.14)
    ax_head = fig.add_subplot(gs[0])
    ax_head.axis("off")
    ax_head.set_xlim(0, 1)
    ax_head.set_ylim(0, 1)

    ax_head.text(0.0, 0.78, star, fontsize=15, fontweight="bold", va="center")
    x = 0.24
    x = _chip(ax_head, x, 0.78, rec["status"],
              STATUS_COLORS.get(rec["status"], "0.4"))
    if rec.get("alias_note"):
        x = _chip(ax_head, x, 0.78, rec["alias_note"], "#f9a825", fontsize=8)
    if rec.get("model_note"):
        x = _chip(ax_head, x, 0.78, rec["model_note"], "#f9a825", fontsize=8)
    if rec["is_interesting"]:
        _chip(ax_head, max(x, 0.86), 0.78, "INTERESTING", "#c62828")

    wrapped = textwrap.fill(rec["analysis"], width=150)
    ax_head.text(0.0, 0.42, wrapped, fontsize=9, style="italic", va="top")

    gs_mid = gs[1].subgridspec(1, 2, width_ratios=[0.60, 0.40], wspace=0.10)
    _param_table(fig.add_subplot(gs_mid[0]), rec)
    _null_table(fig.add_subplot(gs_mid[1]), rec)

    gs_bot = gs[2].subgridspec(1, 2, wspace=0.03)
    pa, ca = pick_figure(dir_a, star, rec, "old", label_a)
    pb, cb = pick_figure(dir_b, star, rec, "new", label_b)
    _draw_panel(fig.add_subplot(gs_bot[0]), pa, ca, thumb_px)
    _draw_panel(fig.add_subplot(gs_bot[1]), pb, cb, thumb_px)

    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dir-a", default=f"{BASE}/second_pdc")
    ap.add_argument("--label-a", default="second_pdc (orig RVs)")
    ap.add_argument("--dir-b", default=f"{BASE}/fourth_fixed_rvs")
    ap.add_argument("--label-b", default="fourth_fixed_rvs (ZP-corrected)")
    ap.add_argument("--mcmc-a", default=None,
                    help="default: {dir-a}/mcmc_out/mcmc_params.csv")
    ap.add_argument("--mcmc-b", default=None)
    ap.add_argument("--output", default=None,
                    help="default: {dir-b}/comparison_<a>_vs_<b>.pdf")
    ap.add_argument("--stars", default=None,
                    help="comma-separated star-name substrings (smoke tests)")
    ap.add_argument("--thumb-px", type=int, default=1400)
    args = ap.parse_args()

    mcmc_a = args.mcmc_a or os.path.join(args.dir_a, "mcmc_out", "mcmc_params.csv")
    mcmc_b = args.mcmc_b or os.path.join(args.dir_b, "mcmc_out", "mcmc_params.csv")
    output = args.output or os.path.join(
        args.dir_b,
        f"comparison_{os.path.basename(args.dir_a)}_vs_{os.path.basename(args.dir_b)}.pdf")

    merged = build_merged(args.dir_a, args.dir_b, mcmc_a, mcmc_b,
                          args.label_a, args.label_b)
    full_run = args.stars is None
    if not full_run:
        toks = [t.strip() for t in args.stars.split(",") if t.strip()]
        merged = merged[[any(t in s for t in toks) for s in merged.index]]

    mask = merged["in_old"] & merged["in_new"]
    gained = set(merged.index[(merged["status"] == "gained") & mask])
    lost = set(merged.index[(merged["status"] == "lost") & mask])
    print(f"Common stars: {int(mask.sum())} | detections OLD "
          f"{int((merged['detected_old'] & mask).sum())}, NEW "
          f"{int((merged['detected_new'] & mask).sum())}")
    print(f"Gained: {sorted(gained)}\nLost:   {sorted(lost)}")
    print(f"Interesting ({int(merged['is_interesting'].sum())}): "
          f"{sorted(merged.index[merged['is_interesting']])}")
    if full_run and (gained != EXPECTED_GAINED or lost != EXPECTED_LOST):
        print("WARNING: gained/lost sets differ from the expected "
              f"{sorted(EXPECTED_GAINED)} / {sorted(EXPECTED_LOST)}")

    meta = dict(dir_a=args.dir_a, dir_b=args.dir_b,
                label_a=args.label_a, label_b=args.label_b)
    with plt.rc_context(RC):
        with PdfPages(output) as pdf:
            page_summary(pdf, merged, meta)
            page_overview(pdf, merged)
            page_no = 3
            for star in merged.index:
                page_no += 1
                rec = merged.loc[star]
                flags = [rec["status"]] + (["INTERESTING"] if rec["is_interesting"] else [])
                print(f"page {page_no}: {star} [{', '.join(flags)}]")
                page_star(pdf, star, rec, args.dir_a, args.dir_b,
                          args.label_a, args.label_b, args.thumb_px)

    print(f"\nReport written to {output} "
          f"({os.path.getsize(output) / 1e6:.1f} MB, {page_no} pages approx.)")


if __name__ == "__main__":
    main()
