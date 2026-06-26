"""
simulations.ht_report — Generate a publication-quality PDF report summarising
the Horvitz–Thompson bias-corrected binary fraction for the BLOeM O-star sample.

Usage:
    python -m simulations.ht_report
    python -m simulations.ht_report --csv /path/to/horvitz_thompson_results.csv
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── A&A journal style (from mcmc/mcmc_plotting.py) ──────────────────
AA_RC = {
    "font.family":              "serif",
    "font.serif":               ["cmr10", "Computer Modern Roman",
                                 "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":         "cm",
    "axes.formatter.use_mathtext": True,
    "font.size":                10,
    "axes.labelsize":           11,
    "axes.titlesize":           12,
    "xtick.labelsize":          9,
    "ytick.labelsize":          9,
    "legend.fontsize":          9,
    "xtick.direction":          "in",
    "ytick.direction":          "in",
    "xtick.top":                True,
    "ytick.right":              True,
    "xtick.minor.visible":      True,
    "ytick.minor.visible":      True,
    "axes.linewidth":           0.7,
    "lines.linewidth":          1.4,
    "errorbar.capsize":         2,
    "savefig.dpi":              300,
}

PAGE = (8.5, 11)          # US Letter

# ── Sample & literature constants ────────────────────────────────────
N_STARS = 134              # total O-star sample (134 = 159 - 25 Oe)
N_DET   = 75               # detected binaries (43 SB1 + 27 SB2 + 5 multi)
N_HT    = 71               # binaries with orbital solutions in the CSV
N_EXTRA = N_DET - N_HT     # higher-order multiples assumed p_det = 1

REFINED = {                # 200-trial re-run results
    "8-031": {"p_det": 0.44,  "n_det": 88,  "n_trials": 200},
    "1-068": {"p_det": 0.705, "n_det": 141, "n_trials": 200},
    "2-079": {"p_det": 0.78,  "n_det": 156, "n_trials": 200},
    "3-078": {"p_det": 0.97,  "n_det": 194, "n_trials": 200},
    "8-028": {"p_det": 1.00,  "n_det": 200, "n_trials": 200},
}

# Literature reference values
SANA12 = {"f": 0.69, "ef": 0.09, "label": "Sana+2012\n(Galactic)"}
SANA13 = {"f": 0.51, "ef": 0.04, "label": "Sana+2013\n(LMC 30 Dor)"}


# =====================================================================
# Data loading & HT computation
# =====================================================================

def _clean_star_id(s):
    """Strip LaTeX artefacts from star IDs."""
    return re.sub(r"\$.*?\$", "", s).strip().rstrip("\\")


def load_and_patch(csv_path):
    """Load 30-trial CSV and patch with 200-trial refined values."""
    df = pd.read_csv(csv_path)
    df["star_id"] = df["star_id"].apply(_clean_star_id)

    # Store original 30-trial p_det before patching
    df["p_det_30"] = df["p_det"].copy()

    for sid, vals in REFINED.items():
        mask = df["star_id"] == sid
        if mask.any():
            df.loc[mask, "p_det"]    = vals["p_det"]
            df.loc[mask, "n_det"]    = vals["n_det"]
            df.loc[mask, "n_trials"] = vals["n_trials"]
            df.loc[mask, "weight"]   = 1.0 / vals["p_det"]
    return df


def compute_ht(df, n_bootstrap=50_000, seed=42):
    """Horvitz–Thompson estimator with bootstrap uncertainty."""
    p_det = np.concatenate([
        df["p_det"].values,
        np.ones(N_EXTRA),       # 4 extra binaries assumed p_det = 1
    ])
    weights = 1.0 / p_det
    f_bin = np.sum(weights) / N_STARS
    f_obs = N_DET / N_STARS

    rng = np.random.default_rng(seed)
    N = len(p_det)
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        boot[b] = np.sum(1.0 / p_det[idx]) / N_STARS

    return {
        "f_bin":  f_bin,
        "f_obs":  f_obs,
        "sigma":  np.std(boot),
        "CI_68":  np.percentile(boot, [16, 84]),
        "CI_95":  np.percentile(boot, [2.5, 97.5]),
        "boot":   boot,
    }


# =====================================================================
# Text-page helpers
# =====================================================================

def _text_page(pdf, blocks, title=None):
    """Render a text page from a list of (y_position, fontsize, text) tuples."""
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    if title:
        fig.text(0.5, 0.95, title, fontsize=16, fontweight="bold",
                 ha="center", va="top", fontfamily="serif")
    for y, fs, txt in blocks:
        fig.text(0.08, y, txt, fontsize=fs, va="top", ha="left",
                 fontfamily="serif", linespacing=1.5,
                 transform=fig.transFigure)
    pdf.savefig(fig)
    plt.close(fig)


# =====================================================================
# Page generators
# =====================================================================

def page_title(pdf, ht):
    blocks = [
        (0.88, 14, "Horvitz-Thompson Bias-Corrected Spectroscopic Binary Fraction"),
        (0.84, 12, "BLOeM O-Star Sample -- Small Magellanic Cloud"),
        (0.78, 10, "Internal Technical Report"),
        (0.70, 11, "Summary"),
        (0.66, 10,
         f"Sample size (N*):                 {N_STARS}\n"
         f"Detected binaries (Ndet):         {N_DET}  (43 SB1 + 27 SB2 + 5 higher-order)\n"
         f"Binaries with orbital solutions:  {N_HT}  (46 SB1 + 25 SB2)\n"
         f"Observed binary fraction (fobs):  {ht['f_obs']:.3f}\n"
         f"HT-corrected fraction (fbin):     {ht['f_bin']:.3f} +/- {ht['sigma']:.3f}\n"
         f"68% confidence interval:          [{ht['CI_68'][0]:.3f},  {ht['CI_68'][1]:.3f}]\n"
         f"95% confidence interval:          [{ht['CI_95'][0]:.3f},  {ht['CI_95'][1]:.3f}]\n"
         f"Correction factor:                {ht['f_bin']/ht['f_obs']:.2f}x"),
        (0.38, 11, "Computational Cost"),
        (0.34, 10,
         "Initial injection-recovery run:   71 binaries x 30 trials  =  2,130 pipeline evaluations\n"
         "Refined run (5 low-pdet stars):   5  binaries x 200 trials =  1,000 pipeline evaluations\n"
         "Total pipeline evaluations:       3,130"),
    ]
    _text_page(pdf, blocks)


def page_methodology(pdf):
    blocks = [
        (0.90, 13, "1.  Horvitz-Thompson Estimator"),
        (0.87, 9,
         "The Horvitz-Thompson (HT) estimator (Horvitz & Thompson 1952) corrects for\n"
         "non-uniform detection probabilities by weighting each detected unit by the\n"
         "inverse of its inclusion probability.  Applied to the spectroscopic binary\n"
         "fraction:"),
        (0.77, 14,
         r"$\hat{f}_{\rm bin}"
         r" \;=\; \frac{1}{N_\star}"
         r" \sum_{i=1}^{N_{\rm det}}"
         r" \frac{1}{\hat{p}_{{\rm det},i}}$"),
        (0.72, 9,
         "where N* = 134 is the total O-type star sample, Ndet = 75 is the number of\n"
         "detected spectroscopic binaries, and pdet,i is the detection probability of\n"
         "the i-th binary estimated via injection-recovery simulations.\n"
         "\n"
         "If pdet,i = 1 for all binaries, the estimator reduces to the observed fraction\n"
         "fobs = Ndet / N*.  Systems with pdet,i < 1 receive weight > 1, effectively\n"
         "accounting for similar binaries that went undetected."),
        (0.55, 13, "2.  Injection-Recovery Protocol"),
        (0.52, 9,
         "For each of the 71 binaries with constrained orbital solutions (46 SB1 + 25 SB2\n"
         "from Tables A.1 and A.2), we estimate pdet,i as follows:\n"
         "\n"
         r"  (a) Fix the orbital parameters (P, e, K1, $\gamma$) at their best-fit values." "\n"
         r"  (b) Draw random nuisance parameters:  $\omega$ in [0, 2$\pi$),  T0 in" "\n"
         r"      [MJDmin - P, MJDmin], and a Gaussian noise realisation." "\n"
         "  (c) Inject the synthetic Keplerian RV signal into the star's actual BLOeM\n"
         "      MJD cadence (~25 epochs over ~2 years).\n"
         "  (d) Run the full detection pipeline:  Lomb-Scargle + PDC periodogram,\n"
         "      lmfit orbital fitting, BICc model comparison, detection if Pbin > 0.5.\n"
         "  (e) Repeat for Ntrials realisations.  The detection probability is:\n"
         "      pdet,i  =  Ndetected / Ntrials .\n"
         "\n"
         "Initial run:  30 trials per star for all 71 binaries (2,130 pipeline evaluations).\n"
         "Five systems with pdet < 1.0 were then re-run with 200 trials each (1,000\n"
         "additional evaluations) to tighten the estimates that dominate the HT correction.\n"
         "The remaining 4 detected binaries (higher-order multiples without standard\n"
         "orbital solutions) are assigned pdet = 1."),
        (0.16, 13, "3.  Bootstrap Uncertainty"),
        (0.13, 9,
         "Confidence intervals are obtained by resampling the Ndet = 75 detection\n"
         "probabilities with replacement (50,000 bootstrap iterations) and recomputing\n"
         "fbin for each resample.  The 16th/84th and 2.5th/97.5th percentiles of the\n"
         "resulting distribution define the 68% and 95% confidence intervals."),
    ]
    _text_page(pdf, blocks, title="Methodology")


def page_fig_pdet_period(pdf, df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sb1 = df[df["source"] == "SB1"]
    sb2 = df[df["source"] == "SB2"]

    sc1 = ax.scatter(np.log10(sb1["P"]), sb1["p_det"],
                     c=sb1["K1"], cmap="viridis", vmin=0, vmax=200,
                     s=50, marker="o", edgecolors="k", linewidths=0.4,
                     zorder=3, label="SB1")
    ax.scatter(np.log10(sb2["P"]), sb2["p_det"],
               c=sb2["K1"], cmap="viridis", vmin=0, vmax=200,
               s=50, marker="s", edgecolors="k", linewidths=0.4,
               zorder=3, label="SB2")
    cb = fig.colorbar(sc1, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$K_1$ [km s$^{-1}$]")

    ax.axhline(1.0, color="grey", ls="--", lw=0.8, alpha=0.6)

    # Annotate refined systems
    for _, row in df[df["p_det"] < 1.0].iterrows():
        ax.annotate(row["star_id"],
                    (np.log10(row["P"]), row["p_det"]),
                    textcoords="offset points", xytext=(6, -4),
                    fontsize=7, fontstyle="italic")

    ax.set_xlabel(r"$\log_{10}(P\,/\,{\rm d})$")
    ax.set_ylabel(r"$\hat{p}_{\rm det}$")
    ax.set_ylim(0.25, 1.08)
    ax.set_title(r"Detection Probability vs. Orbital Period")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_fig_pdet_K1(pdf, df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sb1 = df[df["source"] == "SB1"]
    sb2 = df[df["source"] == "SB2"]

    sc1 = ax.scatter(sb1["K1"], sb1["p_det"],
                     c=sb1["e"], cmap="coolwarm", vmin=0, vmax=1,
                     s=50, marker="o", edgecolors="k", linewidths=0.4,
                     zorder=3, label="SB1")
    ax.scatter(sb2["K1"], sb2["p_det"],
               c=sb2["e"], cmap="coolwarm", vmin=0, vmax=1,
               s=50, marker="s", edgecolors="k", linewidths=0.4,
               zorder=3, label="SB2")
    cb = fig.colorbar(sc1, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Eccentricity")

    ax.axhline(1.0, color="grey", ls="--", lw=0.8, alpha=0.6)

    for _, row in df[df["p_det"] < 1.0].iterrows():
        ax.annotate(row["star_id"],
                    (row["K1"], row["p_det"]),
                    textcoords="offset points", xytext=(6, -4),
                    fontsize=7, fontstyle="italic")

    ax.set_xlabel(r"$K_1$ [km s$^{-1}$]")
    ax.set_ylabel(r"$\hat{p}_{\rm det}$")
    ax.set_ylim(0.25, 1.08)
    ax.set_xscale("log")
    ax.set_title(r"Detection Probability vs. RV Semi-Amplitude")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_fig_comparison(pdf, ht):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [
        r"$f_{\rm obs}$" + "\n(this work)",
        r"$\hat{f}_{\rm bin}$ (HT)" + "\n(this work)",
        SANA12["label"],
        SANA13["label"],
    ]
    vals  = [ht["f_obs"], ht["f_bin"], SANA12["f"], SANA13["f"]]
    errs  = [0, ht["sigma"], SANA12["ef"], SANA13["ef"]]
    cols  = ["#4393c3", "#d6604d", "#a6a6a6", "#bfbfbf"]

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, yerr=errs, color=cols, edgecolor="k",
                  linewidth=0.5, width=0.55, capsize=4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Spectroscopic Binary Fraction")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("Comparison with Literature")

    # Annotate values
    for xi, v, e in zip(x, vals, errs):
        txt = f"{v:.3f}" if e == 0 else rf"${v:.3f}\pm{e:.3f}$"
        ax.text(xi, v + e + 0.03, txt, ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_fig_bootstrap(pdf, ht):
    boot = ht["boot"]
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(boot, bins=80, color="#4393c3", edgecolor="white",
            linewidth=0.3, alpha=0.8, density=True, zorder=2)

    # 95% CI shading
    ax.axvspan(ht["CI_95"][0], ht["CI_95"][1], color="#d6604d",
               alpha=0.12, label="95% CI", zorder=1)
    # 68% CI shading
    ax.axvspan(ht["CI_68"][0], ht["CI_68"][1], color="#d6604d",
               alpha=0.25, label="68% CI", zorder=1)
    # Point estimate
    ax.axvline(ht["f_bin"], color="k", lw=1.5, ls="-",
               label=rf"$\hat{{f}}_{{\rm bin}}$ = {ht['f_bin']:.3f}", zorder=4)

    ax.set_xlabel(r"$\hat{f}_{\rm bin}$")
    ax.set_ylabel("Probability Density")
    ax.set_title(r"Bootstrap Distribution ($N_{\rm boot}$ = 50,000)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── Tables ───────────────────────────────────────────────────────────

def _make_table_page(pdf, cell_text, col_labels, title, highlight_rows=None):
    """Render one table page."""
    fig, ax = plt.subplots(figsize=PAGE)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    n_rows = len(cell_text)
    n_cols = len(col_labels)

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.15)

    # Style header
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_text_props(fontweight="bold", fontsize=8)
        cell.set_facecolor("#d9e2f3")
        cell.set_edgecolor("grey")

    # Highlight rows with p_det < 1
    if highlight_rows:
        for i in highlight_rows:
            for j in range(n_cols):
                tbl[i + 1, j].set_facecolor("#fce4e4")

    # Set edge colours
    for key, cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")

    if title:
        fig.text(0.5, 0.97, title, fontsize=12, fontweight="bold",
                 ha="center", va="top", fontfamily="serif")

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def pages_full_table(pdf, df):
    """Two-page table of all 71 binaries."""
    col_labels = ["#", "Star ID", "Type", "P [d]", "e", "K1 [km/s]",
                  "sigma_RV [km/s]", "p_det", "n_det / n_trials", "Weight"]

    rows = []
    highlight = []
    for i, (_, r) in enumerate(df.iterrows()):
        rows.append([
            str(i + 1),
            r["star_id"],
            r["source"],
            f"{r['P']:.2f}" if r["P"] < 100 else f"{r['P']:.1f}",
            f"{r['e']:.3f}",
            f"{r['K1']:.1f}",
            f"{r['rv_err']:.2f}",
            f"{r['p_det']:.3f}",
            f"{int(r['n_det'])} / {int(r['n_trials'])}",
            f"{r['weight']:.3f}",
        ])
        if r["p_det"] < 1.0:
            highlight.append(i)

    mid = len(rows) // 2 + 1  # split roughly in half
    hl_p1 = [h for h in highlight if h < mid]
    hl_p2 = [h - mid for h in highlight if h >= mid]

    _make_table_page(pdf, rows[:mid], col_labels,
                     "Table 1 -- Per-Binary Detection Probabilities (1/2)",
                     highlight_rows=hl_p1)
    _make_table_page(pdf, rows[mid:], col_labels,
                     "Table 1 -- Per-Binary Detection Probabilities (2/2)",
                     highlight_rows=hl_p2)


def page_refined_table(pdf, df):
    """Page with 5-star refined table and discussion text."""
    ref_df = df[df["star_id"].isin(REFINED.keys())].sort_values("p_det")

    col_labels = ["Star ID", "P [d]", "e", "K1 [km/s]",
                  "p_det (30 trials)", "p_det (200 trials)", "Weight"]
    rows = []
    for _, r in ref_df.iterrows():
        rows.append([
            r["star_id"],
            f"{r['P']:.1f}",
            f"{r['e']:.3f}",
            f"{r['K1']:.1f}",
            f"{r['p_det_30']:.3f}",
            f"{r['p_det']:.3f}",
            f"{r['weight']:.2f}",
        ])

    fig, ax = plt.subplots(figsize=PAGE)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.96,
             "Table 2 -- Refined Detection Probabilities (200 Trials)",
             fontsize=12, fontweight="bold", ha="center", va="top",
             fontfamily="serif")

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="upper center", cellLoc="center",
                   bbox=[0.05, 0.70, 0.9, 0.23])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    for j in range(len(col_labels)):
        tbl[0, j].set_text_props(fontweight="bold", fontsize=9)
        tbl[0, j].set_facecolor("#d9e2f3")
    for key, cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")

    discussion = (
        "Physical Drivers of Incomplete Detection\n"
        "\n"
        "Only 5 of 71 binaries (7%) exhibit detection probabilities below unity,\n"
        "indicating that the BLOeM detection pipeline is highly complete for the\n"
        "observed binary population.  The systems with reduced detectability share\n"
        "common orbital characteristics:\n"
        "\n"
        "- BLOeM 8-031 (pdet = 0.44):  The most extreme case.  With e = 0.98, the\n"
        "  radial velocity signal is confined to a narrow periastron passage lasting\n"
        "  only a few days within a 259-day orbit.  Whether the ~25 BLOeM epochs\n"
        "  sample this brief window is highly orientation-dependent, leading to a\n"
        "  detection rate of only 44%.\n"
        "\n"
        "- BLOeM 1-068 (pdet = 0.71) and BLOeM 2-079 (pdet = 0.78):  Both systems\n"
        "  have very low RV semi-amplitudes (K1 = 3.0 and 2.4 km/s, respectively),\n"
        "  comparable to the measurement uncertainties.  Combined with moderate\n"
        "  eccentricity (e = 0.6 and 0.5), the signal is marginal and easily masked\n"
        "  by noise.\n"
        "\n"
        "- BLOeM 3-078 (pdet = 0.97):  A long-period system (P = 269 d) with low\n"
        "  K1 = 5.0 km/s.  The combination of long period and weak signal leads\n"
        "  to occasional non-detections, though the rate is high.\n"
        "\n"
        "- BLOeM 8-028 (pdet = 1.00):  Initially showed pdet = 0.97 with 30 trials,\n"
        "  but converged to perfect detection with 200 trials, consistent with\n"
        "  stochastic fluctuation in the initial run.\n"
        "\n"
        "In summary, the detection incompleteness is driven by the interplay of three\n"
        "factors: (i) extreme eccentricity concentrating the RV signal into narrow\n"
        "orbital phases, (ii) low K1 approaching the noise floor, and (iii) long\n"
        "orbital periods reducing the number of observed cycles within the ~2-year\n"
        "baseline."
    )
    fig.text(0.08, 0.66, discussion, fontsize=9, va="top", ha="left",
             fontfamily="serif", linespacing=1.45)

    pdf.savefig(fig)
    plt.close(fig)


def page_discussion(pdf, ht):
    blocks = [
        (0.88, 13, "Discussion"),
        (0.84, 10,
         "Magnitude of the Correction\n"
         "\n"
         f"The Horvitz-Thompson correction increases the observed binary fraction from\n"
         f"fobs = {ht['f_obs']:.3f} to fbin = {ht['f_bin']:.3f} +/- {ht['sigma']:.3f}, "
         "a relative correction\n"
         f"of only {(ht['f_bin']/ht['f_obs'] - 1)*100:.1f}%.  "
         "This modest adjustment reflects the high completeness of\n"
         "the BLOeM survey:  with ~25 epochs spanning ~2 years, the pipeline recovers\n"
         r"the vast majority of spectroscopic binaries with orbital periods $P < 10^3$ d" "\n"
         r"and semi-amplitudes $K_1 > 5$ km/s."),
        (0.64, 10,
         "Comparison with Literature\n"
         "\n"
         f"The bias-corrected SMC binary fraction of {ht['f_bin']:.3f} +/- {ht['sigma']:.3f} "
         "can be compared\n"
         "with previous spectroscopic surveys of massive stars:\n"
         "\n"
         f"  - Galactic O-stars (Sana et al. 2012):  fbin = {SANA12['f']} +/- {SANA12['ef']}\n"
         f"  - LMC 30 Doradus (Sana et al. 2013):    fbin = {SANA13['f']} +/- {SANA13['ef']}\n"
         f"  - SMC / BLOeM (this work):               fbin = {ht['f_bin']:.3f} +/- {ht['sigma']:.3f}\n"
         "\n"
         "The Galactic value from Sana et al. (2012) represents a different quantity:\n"
         "the intrinsic binary fraction after forward-modelling the full 4D parameter\n"
         "space (period, mass-ratio, eccentricity power-law exponents plus fbin),\n"
         "whereas the HT estimator corrects only for detection bias at the observed\n"
         "orbital parameters.  The two approaches are complementary."),
        (0.34, 10,
         "Caveats\n"
         "\n"
         "The Horvitz-Thompson estimator corrects for detection bias conditional on\n"
         "the observed orbital parameters.  It does not account for:\n"
         "\n"
         r"  (i)   Binaries in entirely unexplored parameter space (e.g., very long" "\n"
         r"         periods $P \gg 2$ yr where fewer than one orbit is covered)." "\n"
         "\n"
         r"  (ii)  Face-on systems with $\sin\,i \to 0$ and $K_1 \to 0$ that produce no" "\n"
         "         measurable RV variation at any orbital phase.\n"
         "\n"
         r"  (iii) Systems with $K_1$ below the detection threshold (~2 km/s) that" "\n"
         "         are fundamentally undetectable with the BLOeM RV precision.\n"
         "\n"
         "The HT-corrected fraction should therefore be interpreted as a lower bound\n"
         "on the true intrinsic binary fraction.  A full forward-modelling approach\n"
         "(Sana et al. 2012, 2013) that samples the intrinsic parameter distributions\n"
         "and accounts for geometric inclination effects is needed to constrain\n"
         "the complete multiplicity properties."),
    ]
    _text_page(pdf, blocks)


def page_references(pdf):
    blocks = [
        (0.88, 13, "References"),
        (0.83, 10,
         'Horvitz, D. G. & Thompson, D. J. 1952,\n'
         '  "A Generalization of Sampling Without Replacement From a Finite Universe",\n'
         '  Journal of the American Statistical Association, 47, 663\n'
         '\n'
         'Sana, H., de Mink, S. E., de Koter, A., et al. 2012,\n'
         '  "Binary Interaction Dominates the Evolution of Massive Stars",\n'
         '  Science, 337, 444\n'
         '\n'
         'Sana, H., de Koter, A., de Mink, S. E., et al. 2013,\n'
         '  "The VLT-FLAMES Tarantula Survey. VIII. Multiplicity Properties of the\n'
         '  O-type Star Population",\n'
         '  Astronomy & Astrophysics, 550, A107\n'
         '\n'
         'Sana, H., Shenar, T., Bodensteiner, J., et al. 2025,\n'
         '  "A High Fraction of Close Massive Binary Stars at Low Metallicity",\n'
         '  Nature Astronomy, 9, 1337'),
    ]
    _text_page(pdf, blocks)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Horvitz–Thompson results PDF report.")
    parser.add_argument(
        "--csv",
        default=os.path.join(
            "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/"
            "dr5_neb_div_from_coadded/horvitz_thompson_results",
            "horvitz_thompson_results.csv"),
        help="Path to horvitz_thompson_results.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.csv)
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "horvitz_thompson_report.pdf")

    # Load & compute
    df = load_and_patch(args.csv)
    ht = compute_ht(df)

    print(f"f_obs = {ht['f_obs']:.3f}")
    print(f"f_bin = {ht['f_bin']:.3f} ± {ht['sigma']:.3f}")
    print(f"68% CI: [{ht['CI_68'][0]:.3f}, {ht['CI_68'][1]:.3f}]")
    print(f"95% CI: [{ht['CI_95'][0]:.3f}, {ht['CI_95'][1]:.3f}]")

    # Generate PDF
    with plt.rc_context(AA_RC):
        with PdfPages(pdf_path) as pdf:
            page_title(pdf, ht)
            page_methodology(pdf)
            page_fig_pdet_period(pdf, df)
            page_fig_pdet_K1(pdf, df)
            page_fig_comparison(pdf, ht)
            page_fig_bootstrap(pdf, ht)
            pages_full_table(pdf, df)
            page_refined_table(pdf, df)
            page_discussion(pdf, ht)
            page_references(pdf)

    print(f"\nPDF saved to: {pdf_path}")


if __name__ == "__main__":
    main()
