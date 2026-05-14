"""
simulations.bias_grid_closure_generate — synthesize a fake "observed" detected
catalog at a known (pi, kappa, eta, fbin) truth and write it as LaTeX SB1/SB2
tables that simulations/bias_grid.py can ingest unchanged.

The synthetic detections are produced by the *same* per-star injection-recovery
machinery the grid uses internally (_worker_star_injections_vectorized) at the
truth point, with n_inject=1 (one binary/single coin flip per real BLOeM
star). The resulting detected catalog is split into SB1/SB2 by a mass-ratio
threshold, then formatted into LaTeX tables matching the columns the bias_grid
parser reads (see load_observed_from_tex).

Usage (single truth point):
    python -m simulations.bias_grid_closure_generate \
        --truth-pi -0.55 --truth-kappa -0.10 --truth-eta -0.45 \
        --truth-fbin 0.69 --seed 42 \
        --config params_bias.yaml \
        --output-dir $SCRIPTS_OUT/simulation_pipeline/bias_grid_closure/sana2012/

Real-sample paths (mass_file, rv_dir, sb2_analysis_dir, ostar_catalog) are
read from the bias_grid: section of the YAML, matching bias_grid.py and
audit_sample.py. Then run bias_grid against the synthesized tables:

    python -m simulations.bias_grid --config params_bias.yaml
"""

import argparse
import csv
import os
import sys

import numpy as np
import yaml

from simulations.bias_grid import (
    _worker_star_injections_vectorized,
    load_star_properties,
)
from simulations.bias_config import DEFAULT_BIAS_CFG
from simulations.common import BLOEM_MJD_ARRAYS
from pipeline.config import load_args


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------

def _fmt_period(P_days):
    """LaTeX-formatted period with placeholder errors."""
    if P_days < 1.0:
        return "$%.4f^{+0.0001}_{-0.0001}$" % P_days
    if P_days < 10.0:
        return "$%.3f^{+0.001}_{-0.001}$" % P_days
    if P_days < 1000.0:
        return "$%.2f^{+0.01}_{-0.01}$" % P_days
    return "$%.1f^{+0.1}_{-0.1}$" % P_days


def _fmt_eccentricity(e_val, P_days, p_circ):
    """LaTeX-formatted eccentricity. Circular orbits → '\\leq 0'.

    bias_grid's load_observed_from_tex treats '\\leq ...' as a circular-orbit
    flag (e=0). The truth sampler forces e=0 when P < p_circ, so we match.
    """
    if P_days < p_circ or e_val < 1e-6:
        return r"$\leq 0$"
    return "$%.3f^{+0.01}_{-0.01}$" % e_val


def _fmt_K1(K1_val):
    return "$%.1f^{+0.5}_{-0.5}$" % K1_val


def _fmt_q(q_val):
    return "$%.3f^{+0.01}_{-0.01}$" % q_val


SB1_PREAMBLE = r"""\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.3}
\begin{longtable}{llllllllllll}
\caption{SYNTHETIC SB1 catalog (closure test).}
\label{tab:sb1_solutions_synth} \\
\hline\hline
\# & BLOeM ID & \shortstack[l]{$P$ \\ {[d]}} & \shortstack[l]{$T_0$ \\ {[MJD]}} & \shortstack[l]{$\omega$ \\ {[$^\circ$]}} & $e$ & \shortstack[l]{$K_1$ \\ {[km\,s$^{-1}$]}} & \shortstack[l]{$\gamma$ \\ {[km\,s$^{-1}$]}} & \shortstack[l]{$f(M)$ \\ {[$M_\odot$]}} & \shortstack[l]{$M_{2,\min}$ \\ {[$M_\odot$]}} & $q_{\min}$ & Notes \\
\hline
\endfirsthead
\multicolumn{12}{c}{\tablename\ \thetable{} -- continued} \\
\hline\hline
\# & BLOeM ID & \shortstack[l]{$P$ \\ {[d]}} & \shortstack[l]{$T_0$ \\ {[MJD]}} & \shortstack[l]{$\omega$ \\ {[$^\circ$]}} & $e$ & \shortstack[l]{$K_1$ \\ {[km\,s$^{-1}$]}} & \shortstack[l]{$\gamma$ \\ {[km\,s$^{-1}$]}} & \shortstack[l]{$f(M)$ \\ {[$M_\odot$]}} & \shortstack[l]{$M_{2,\min}$ \\ {[$M_\odot$]}} & $q_{\min}$ & Notes \\
\hline
\endhead
\hline
\endfoot
\hline
\endlastfoot
"""

SB1_FOOTER = r"""\end{longtable}
"""

SB2_PREAMBLE = r"""\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.3}
\begin{longtable}{lllllllllll}
\caption{SYNTHETIC SB2 catalog (closure test).}
\label{tab:sb2_solutions_synth} \\
\hline\hline
\# & BLOeM ID & \shortstack[l]{$P$ \\ {[d]}} & \shortstack[l]{$T_0$ \\ {[MJD]}} & \shortstack[l]{$\omega$ \\ {[$^\circ$]}} & $e$ & $q$ & \shortstack[l]{$\gamma$ \\ {[km\,s$^{-1}$]}} & \shortstack[l]{$K_i$ \\ {[km\,s$^{-1}$]}} & \shortstack[l]{$M_i\sin^3 i$ \\ {[$M_\odot$]}} & \shortstack[l]{$f(M)$ \\ {[$M_\odot$]}} \\
\hline
\endfirsthead
\multicolumn{11}{c}{\tablename\ \thetable{} -- continued} \\
\hline\hline
\# & BLOeM ID & \shortstack[l]{$P$ \\ {[d]}} & \shortstack[l]{$T_0$ \\ {[MJD]}} & \shortstack[l]{$\omega$ \\ {[$^\circ$]}} & $e$ & $q$ & \shortstack[l]{$\gamma$ \\ {[km\,s$^{-1}$]}} & \shortstack[l]{$K_i$ \\ {[km\,s$^{-1}$]}} & \shortstack[l]{$M_i\sin^3 i$ \\ {[$M_\odot$]}} & \shortstack[l]{$f(M)$ \\ {[$M_\odot$]}} \\
\hline
\endhead
\hline
\endfoot
\hline
\endlastfoot
"""

SB2_FOOTER = r"""\end{longtable}
"""


def _write_sb1_table(path, rows, p_circ):
    """rows: list of dicts with star_id, P, e, K1."""
    with open(path, "w") as f:
        f.write(SB1_PREAMBLE)
        for i, r in enumerate(rows, start=1):
            cols = [
                str(i),
                r["star_id"],
                _fmt_period(r["P"]),
                r"\dots",
                r"\dots",
                _fmt_eccentricity(r["e"], r["P"], p_circ),
                _fmt_K1(r["K1"]),
                r"\dots",
                r"\dots",
                r"\dots",
                r"\dots",
                r"",
            ]
            f.write(" & ".join(cols) + r" \\" + "\n")
        f.write(SB1_FOOTER)


def _write_sb2_table(path, rows, p_circ):
    """rows: list of dicts with star_id, P, e, K1, q."""
    with open(path, "w") as f:
        f.write(SB2_PREAMBLE)
        for i, r in enumerate(rows, start=1):
            primary = [
                str(i),
                r["star_id"],
                _fmt_period(r["P"]),
                r"\dots",
                r"\dots",
                _fmt_eccentricity(r["e"], r["P"], p_circ),
                _fmt_q(r["q"]),
                r"\dots",
                _fmt_K1(r["K1"]),
                r"\dots",
                r"\dots",
            ]
            f.write(" & ".join(primary) + r" \\" + "\n")
            # Secondary continuation: empty cols[0] → parser skips on int() fail
            secondary = ["", "", "", "", "", "", "", "",
                         _fmt_K1(r["K1"] / max(r["q"], 1e-3)),
                         r"\dots", ""]
            f.write(" & ".join(secondary) + r" \\" + "\n")
        f.write(SB2_FOOTER)


# ---------------------------------------------------------------------------
# Synthetic-catalog generation
# ---------------------------------------------------------------------------

def generate_synthetic_detections(star_df, args_dict, cfg,
                                  truth_pi, truth_kappa, truth_eta, truth_fbin,
                                  seed, n_inject=1):
    """Run rv_threshold injection-recovery at truth, one shot per star.

    Returns
    -------
    detections : list of dict
        One row per detected binary. Keys: star_id, P, e, K1, q.
    summary : dict
        n_physical, n_detected, n_false_positive (across all stars).
    """
    rng = np.random.default_rng(seed)
    detections = []
    n_physical = 0
    n_detected = 0
    n_false_positive = 0

    star_ids = star_df["ID"].values
    M1_arr = star_df["Mspec"].values.astype(float)
    R1_arr = star_df["R_star"].values.astype(float)
    field_arr = star_df["field"].values
    rv_err_arr = star_df["rv_err"].values.astype(float)
    gamma_arr = star_df["gamma"].values.astype(float)

    for i, sid in enumerate(star_ids):
        fld = int(field_arr[i])
        MJDs = np.asarray(BLOEM_MJD_ARRAYS[fld], dtype=float)
        star_seed = int(rng.integers(0, 2**63))

        task = (
            star_seed,
            MJDs,
            float(rv_err_arr[i]),
            float(M1_arr[i]),
            float(R1_arr[i]),
            float(gamma_arr[i]),
            n_inject,
            float(truth_fbin),
            float(truth_pi),
            float(truth_kappa),
            float(truth_eta),
            cfg,
            args_dict,
            "rv_threshold",
        )
        result = _worker_star_injections_vectorized(task)

        n_physical += result["n_physical"]
        n_detected += result["n_detected"]
        n_false_positive += result["n_false_positive"]

        for j in range(len(result["logP_det"])):
            detections.append({
                "star_id": str(sid),
                "logP": float(result["logP_det"][j]),
                "P": float(10.0 ** result["logP_det"][j]),
                "e": float(result["e_det"][j]),
                "K1": float(result["K1_det"][j]),
                "q": float(result["q_det"][j]),
            })

    summary = {
        "n_stars": int(len(star_ids)),
        "n_physical": int(n_physical),
        "n_detected": int(n_detected),
        "n_false_positive": int(n_false_positive),
    }
    return detections, summary


def split_sb1_sb2(detections, q_threshold):
    """SB2 if q >= q_threshold; SB1 otherwise."""
    sb1 = [d for d in detections if d["q"] < q_threshold]
    sb2 = [d for d in detections if d["q"] >= q_threshold]
    return sb1, sb2


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_truth_yaml(path, payload):
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_obs_csv(path, detections, sb2_q_threshold):
    fields = ["star_id", "role", "P", "logP", "e", "K1", "q"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in detections:
            row = dict(d)
            row["role"] = "SB2" if d["q"] >= sb2_q_threshold else "SB1"
            w.writerow({k: row[k] for k in fields})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    ap = argparse.ArgumentParser(
        description=(
            "Generate synthetic SB1/SB2 LaTeX tables at a known truth point "
            "for closure-testing simulations.bias_grid."))
    ap.add_argument("--truth-pi", type=float, required=True,
                    help="Truth value for pi (period distribution slope).")
    ap.add_argument("--truth-kappa", type=float, required=True,
                    help="Truth value for kappa (mass-ratio slope).")
    ap.add_argument("--truth-eta", type=float, required=True,
                    help="Truth value for eta (eccentricity slope).")
    ap.add_argument("--truth-fbin", type=float, required=True,
                    help="Truth value for f_bin (intrinsic binary fraction).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base RNG seed.")
    ap.add_argument("--config", type=str, default="params_bias.yaml",
                    help="Pipeline config YAML. Sample paths are read from "
                         "its bias_grid: section.")
    ap.add_argument("--output-dir", type=str, required=True,
                    help="Where to write sb1_solutions.tex / sb2_solutions.tex / "
                         "truth_params.yaml / synthetic_obs.csv.")
    ap.add_argument("--sb2-q-threshold", type=float, default=0.5,
                    help="q >= this -> SB2 row, else SB1.")
    ap.add_argument("--n-inject", type=int, default=1,
                    help="Injections per star (default 1 = one realization).")
    return ap


def main(argv=None):
    args = _build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    args_dict = load_args(args.config)
    cfg = dict(DEFAULT_BIAS_CFG)
    p_circ = float(cfg.get("p_circ", 2.26))

    bg_cfg = args_dict.get("bias_grid", {}) or {}

    def _bg(key):
        if key in bg_cfg and bg_cfg[key] is not None:
            return bg_cfg[key]
        return DEFAULT_BIAS_CFG.get(key)

    star_df = load_star_properties(
        _bg("mass_file"),
        _bg("rv_dir"),
        sb2_analysis_dir=_bg("sb2_analysis_dir"),
        ostar_catalog=_bg("ostar_catalog"),
    )
    print("[closure-gen] loaded %d real BLOeM stars" % len(star_df))

    detections, summary = generate_synthetic_detections(
        star_df, args_dict, cfg,
        truth_pi=args.truth_pi,
        truth_kappa=args.truth_kappa,
        truth_eta=args.truth_eta,
        truth_fbin=args.truth_fbin,
        seed=args.seed,
        n_inject=args.n_inject,
    )
    print("[closure-gen] %d detected of %d physical (%d false positives)" %
          (summary["n_detected"], summary["n_physical"],
           summary["n_false_positive"]))

    sb1_rows, sb2_rows = split_sb1_sb2(detections, args.sb2_q_threshold)
    print("[closure-gen] SB1=%d SB2=%d (split at q>=%.2f)" %
          (len(sb1_rows), len(sb2_rows), args.sb2_q_threshold))

    sb1_path = os.path.join(args.output_dir, "sb1_solutions.tex")
    sb2_path = os.path.join(args.output_dir, "sb2_solutions.tex")
    _write_sb1_table(sb1_path, sb1_rows, p_circ)
    _write_sb2_table(sb2_path, sb2_rows, p_circ)

    obs_csv = os.path.join(args.output_dir, "synthetic_obs.csv")
    _write_obs_csv(obs_csv, detections, args.sb2_q_threshold)

    truth_yaml = os.path.join(args.output_dir, "truth_params.yaml")
    _write_truth_yaml(truth_yaml, {
        "truth": {
            "pi": float(args.truth_pi),
            "kappa": float(args.truth_kappa),
            "eta": float(args.truth_eta),
            "fbin": float(args.truth_fbin),
        },
        "seed": int(args.seed),
        "n_inject_per_star": int(args.n_inject),
        "sb2_q_threshold": float(args.sb2_q_threshold),
        "summary": summary,
        "n_sb1": int(len(sb1_rows)),
        "n_sb2": int(len(sb2_rows)),
        "config": args.config,
        "mass_file": _bg("mass_file"),
        "rv_dir": _bg("rv_dir"),
        "sb2_analysis_dir": _bg("sb2_analysis_dir"),
        "ostar_catalog": _bg("ostar_catalog"),
        "p_circ": p_circ,
    })

    print("[closure-gen] wrote:")
    print("  %s" % sb1_path)
    print("  %s" % sb2_path)
    print("  %s" % obs_csv)
    print("  %s" % truth_yaml)


if __name__ == "__main__":
    sys.exit(main())
