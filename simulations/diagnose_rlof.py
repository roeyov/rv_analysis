"""
Diagnostic script: run a single grid point (Sana+2012 values) with high
n_inject and analyze the properties of RLOF-rejected systems.

Usage:
    python -m simulations.diagnose_rlof [--n-inject 5000]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from simulations.bias_grid import (
    _worker_star_injections, roche_lobe_check, DETECTION_METHODS,
)
from simulations.bias_config import DEFAULT_BIAS_CFG
from simulations.common import BLOEM_MJD_ARRAYS, SIGMA_SHAPE, SIGMA_LOC, SIGMA_SCALE
import pandas as pd
from scipy.stats import lognorm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-inject", type=int, default=5000,
                        help="Injections per star (default 5000)")
    parser.add_argument("--pi", type=float, default=-0.55)
    parser.add_argument("--kappa", type=float, default=-0.10)
    parser.add_argument("--eta", type=float, default=-0.45)
    parser.add_argument("--fbin", type=float, default=0.69)
    parser.add_argument("--output", type=str, default="rlof_diagnostic.png")
    args = parser.parse_args()

    cfg = DEFAULT_BIAS_CFG.copy()
    n_inject = args.n_inject

    # Use representative star properties (median O-star)
    # Load mass file if available, else use typical values
    mass_file = cfg.get("mass_file", "")
    try:
        df_mass = pd.read_csv(mass_file, encoding="utf-8-sig")
        M1_vals = df_mass["Mspec"].values
        R1_vals = df_mass["R_star"].values
        print(f"Loaded {len(M1_vals)} stars from {mass_file}")
        print(f"  M1: median={np.median(M1_vals):.1f}, range=[{M1_vals.min():.1f}, {M1_vals.max():.1f}]")
        print(f"  R1: median={np.median(R1_vals):.1f}, range=[{R1_vals.min():.1f}, {R1_vals.max():.1f}]")
    except Exception as e:
        print(f"Could not load mass file ({e}), using typical O-star values")
        M1_vals = np.array([20.0, 30.0, 40.0, 50.0])
        R1_vals = np.array([8.0, 10.0, 13.0, 16.0])

    # Pick a few representative stars
    n_stars = min(10, len(M1_vals))
    indices = np.linspace(0, len(M1_vals) - 1, n_stars, dtype=int)

    # Use first field MJDs
    MJDs = np.array(BLOEM_MJD_ARRAYS[0])

    # Collect RLOF info across stars
    all_logP_rlof = []
    all_q_rlof = []
    all_e_rlof = []
    all_M1_rlof = []
    all_R1_rlof = []
    all_logP_phys = []
    all_q_phys = []
    total_rlof = 0
    total_phys = 0

    args_dict = {"drv_threshold": 20.0, "significance_threshold": 4.0}

    print(f"\nRunning {n_inject} injections x {n_stars} stars...")
    print(f"  pi={args.pi}, kappa={args.kappa}, eta={args.eta}, fbin={args.fbin}")

    for idx in indices:
        M1 = float(M1_vals[idx])
        R1 = float(R1_vals[idx])
        rv_err = 1.5  # typical

        task = (
            42 + idx,  # seed
            MJDs,
            rv_err,
            M1, R1,
            0.0,  # gamma
            n_inject,
            args.fbin,
            args.pi, args.kappa, args.eta,
            cfg, args_dict,
            "rv_threshold",
        )
        result = _worker_star_injections(task)

        total_rlof += result["n_rlof"]
        total_phys += result["n_physical"]
        all_logP_rlof.extend(result["logP_rlof"])
        all_q_rlof.extend(result["q_rlof"])
        all_e_rlof.extend(result["e_rlof"])
        all_M1_rlof.extend(result["M1_rlof"])
        all_R1_rlof.extend(result["R1_rlof"])
        all_logP_phys.extend(result["logP_det"])  # detected = subset of physical

        print(f"  Star idx={idx}: M1={M1:.1f} R1={R1:.1f} → "
              f"RLOF={result['n_rlof']}, physical={result['n_physical']}, "
              f"detected={result['n_detected']}")

    all_logP_rlof = np.array(all_logP_rlof)
    all_q_rlof = np.array(all_q_rlof)
    all_e_rlof = np.array(all_e_rlof)
    all_M1_rlof = np.array(all_M1_rlof)
    all_R1_rlof = np.array(all_R1_rlof)

    total_drawn = total_rlof + total_phys
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_rlof} RLOF / {total_drawn} drawn = {total_rlof/total_drawn:.1%} rejected")
    print(f"{'='*60}")

    if len(all_logP_rlof) == 0:
        print("No RLOF systems found — nothing to plot.")
        return

    # --- Plots ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"RLOF-rejected systems diagnostic\n"
        f"pi={args.pi}, kappa={args.kappa}, eta={args.eta}, fbin={args.fbin} | "
        f"RLOF fraction = {total_rlof/total_drawn:.1%}",
        fontsize=12,
    )

    # 1. logP distribution: RLOF vs physical
    ax = axes[0, 0]
    bins_logP = np.linspace(0, 4, 40)
    ax.hist(all_logP_rlof, bins=bins_logP, alpha=0.6, label="RLOF-rejected",
            density=True, color="red")
    # For comparison, draw from the same power law to show the intrinsic
    rng = np.random.default_rng(123)
    from simulations.bias_grid import powerlaw_draw
    intrinsic_logP = powerlaw_draw(100000, args.pi, cfg["log_p_min"], cfg["log_p_max"], rng)
    ax.hist(intrinsic_logP, bins=bins_logP, alpha=0.4, label="Intrinsic draw",
            density=True, color="gray", histtype="step", linewidth=2)
    ax.set_xlabel("log P [days]")
    ax.set_ylabel("Density")
    ax.set_title("Period distribution")
    ax.legend()
    ax.axvline(np.log10(2), color="k", ls="--", alpha=0.5, label="p_circ=2d")

    # 2. RLOF fraction vs logP
    ax = axes[0, 1]
    bins = np.linspace(0, 4, 30)
    h_rlof, _ = np.histogram(all_logP_rlof, bins=bins)
    # Need all drawn logP (rlof + physical) — approximate from intrinsic
    # Actually recompute: total binaries drawn per bin
    # Use the RLOF + detected (physical detected) as proxy
    # Better: compute analytically
    h_total = h_rlof.copy()
    # We don't have all physical logP, but we can estimate the fraction
    # by binning both populations
    ax.bar(bins[:-1], h_rlof, width=np.diff(bins), alpha=0.7, color="red",
           align="edge", label="RLOF count")
    ax.set_xlabel("log P [days]")
    ax.set_ylabel("Count")
    ax.set_title("RLOF rejections by period")
    ax.legend()

    # 3. q distribution of RLOF systems
    ax = axes[0, 2]
    ax.hist(all_q_rlof, bins=30, alpha=0.7, color="red")
    ax.set_xlabel("q = M2/M1")
    ax.set_ylabel("Count")
    ax.set_title("Mass ratio of RLOF systems")

    # 4. logP vs q scatter (RLOF)
    ax = axes[1, 0]
    ax.scatter(all_logP_rlof, all_q_rlof, s=1, alpha=0.3, color="red")
    ax.set_xlabel("log P [days]")
    ax.set_ylabel("q")
    ax.set_title("RLOF systems in P-q space")

    # 5. logP vs M1 (RLOF)
    ax = axes[1, 1]
    ax.scatter(all_logP_rlof, all_M1_rlof, s=1, alpha=0.3, color="red")
    ax.set_xlabel("log P [days]")
    ax.set_ylabel("M1 [Msun]")
    ax.set_title("RLOF systems: period vs primary mass")

    # 6. Minimum non-RLOF period vs R1 (analytical)
    ax = axes[1, 2]
    R1_range = np.linspace(4, 25, 50)
    for q_val in [0.2, 0.5, 0.8, 1.0]:
        P_min = []
        for R1_v in R1_range:
            # Find minimum P where Roche lobe is NOT filled (e=0, M1=30)
            for logP_try in np.linspace(0, 3, 500):
                if roche_lobe_check(10**logP_try, 0.0, q_val, 30.0, R1_v):
                    P_min.append(logP_try)
                    break
            else:
                P_min.append(3.0)
        ax.plot(R1_range, P_min, label=f"q={q_val}")
    ax.set_xlabel("R1 [Rsun]")
    ax.set_ylabel("Min log P (no RLOF, e=0, M1=30)")
    ax.set_title("Analytical RLOF period floor")
    ax.legend()
    ax.axhline(np.log10(2), color="k", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {args.output}")


if __name__ == "__main__":
    main()
