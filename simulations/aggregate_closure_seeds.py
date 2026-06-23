"""
simulations.aggregate_closure_seeds — Aggregate per-seed closure_summary.csv
files from a closure-seed sweep produced by run_closure_seeds.sh, plus a
joint best-fit over the summed log_gmf cubes.

Usage:
    python -m simulations.aggregate_closure_seeds \
        --base-dir $SCRIPTS_OUT/simulation_pipeline/bias_grid_closure/sana2012_10seeds
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import yaml

from simulations.bias_grid_lib.cube_io import read_gmf_cube


def _load_truth(seed_dir):
    with open(os.path.join(seed_dir, "truth_params.yaml")) as fh:
        truth = yaml.safe_load(fh)
    return truth


def _load_per_seed_summaries(base_dir):
    rows = []
    for seed_dir in sorted(glob.glob(os.path.join(base_dir, "seed_*"))):
        seed = int(os.path.basename(seed_dir).split("_")[1])
        summary_csv = os.path.join(seed_dir, "grid_run",
                                   "closure_report", "closure_summary.csv")
        if not os.path.exists(summary_csv):
            print(f"[seed {seed}] missing closure_summary.csv -- skipping")
            continue
        df = pd.read_csv(summary_csv)
        ks_row = df[df["test"] == "ks"].iloc[0]
        truth = _load_truth(seed_dir)
        rows.append({
            "seed": seed,
            "n_stars": truth["summary"]["n_stars"],
            "n_detected": truth["summary"]["n_detected"],
            "n_false_positive": truth["summary"]["n_false_positive"],
            "recovered_pi": ks_row["recovered_pi"],
            "recovered_kappa": ks_row["recovered_kappa"],
            "recovered_eta": ks_row["recovered_eta"],
            "recovered_fbin": ks_row["recovered_fbin"],
            "log_gmf_max": ks_row["log_gmf_max"],
            "log_gmf_at_truth": ks_row["log_gmf_at_truth"],
            "log_gmf_gap": ks_row["log_gmf_gap"],
        })
    return pd.DataFrame(rows)


def _joint_best_fit(base_dir, variant=None):
    """Sum log_gmf cubes across seeds and report the joint mode.

    Each per-seed cube is the KS log-GMF for the chosen scoring `variant`
    from grid_run/grid_cubes.npz (or checkpoint_cubes.npz). All-variants
    (schema v4) cubes are read via the namespaced 'v__<tag>__gmf_ks_cube'
    key (default variant `eccentric_only__numerical`); legacy single-mode
    cubes fall back to the bare key. Cubes share grid axes by construction.
    """
    cubes_paths = []
    for seed_dir in sorted(glob.glob(os.path.join(base_dir, "seed_*"))):
        for name in ("grid_cubes.npz", "checkpoint_cubes.npz"):
            p = os.path.join(seed_dir, "grid_run", name)
            if os.path.exists(p):
                cubes_paths.append(p)
                break
    if not cubes_paths:
        return None

    sum_log_gmf = None
    pi_g = kappa_g = eta_g = fbin_g = None
    resolved_tag = None
    for p in cubes_paths:
        z = np.load(p, allow_pickle=True)
        try:
            cube, resolved_tag = read_gmf_cube(z, test="ks", tag=variant)
            cube = cube.astype(np.float64)
        except KeyError:
            print(f"  no log_gmf cube in {p} -- skipping")
            continue
        if sum_log_gmf is None:
            sum_log_gmf = np.zeros_like(cube)
            pi_g = z["pi_grid"]
            kappa_g = z["kappa_grid"]
            eta_g = z["eta_grid"]
            fbin_g = z["fbin_grid"]
        sum_log_gmf += cube

    if sum_log_gmf is None:
        return None

    finite = np.isfinite(sum_log_gmf)
    if not finite.any():
        return None
    flat_idx = np.argmax(np.where(finite, sum_log_gmf, -np.inf))
    i, j, k, l = np.unravel_index(flat_idx, sum_log_gmf.shape)
    return {
        "n_seeds_summed": len(cubes_paths),
        "variant": resolved_tag,
        "joint_pi": float(pi_g[i]),
        "joint_kappa": float(kappa_g[j]),
        "joint_eta": float(eta_g[k]),
        "joint_fbin": float(fbin_g[l]),
        "joint_log_gmf_sum": float(sum_log_gmf[i, j, k, l]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True,
                    help="Directory containing seed_NN/ subdirs.")
    ap.add_argument("--out", default=None,
                    help="Optional CSV path to write per-seed table.")
    ap.add_argument("--variant", default=None,
                    help="Scoring variant tag '<e_score_mode>__<logP_cutoff_"
                         "mode>' to read from all-variants (v4) cubes. "
                         "Default: eccentric_only__numerical.")
    args = ap.parse_args()

    df = _load_per_seed_summaries(args.base_dir)
    if df.empty:
        print("No completed seeds found.")
        return

    truth = _load_truth(sorted(glob.glob(
        os.path.join(args.base_dir, "seed_*")))[0])["truth"]

    print("=" * 78)
    print(f"Per-seed closure recovery ({len(df)} seeds)")
    print("=" * 78)
    print(df.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    print()
    print("=" * 78)
    print("Mean +/- std across seeds (KS)")
    print("=" * 78)
    for col, t in (("recovered_pi", truth["pi"]),
                   ("recovered_kappa", truth["kappa"]),
                   ("recovered_eta", truth["eta"]),
                   ("recovered_fbin", truth["fbin"])):
        m, s = df[col].mean(), df[col].std()
        delta = m - t
        print(f"  {col:>16s}: {m:7.3f} +/- {s:6.3f}   "
              f"(truth={t:6.3f}, delta={delta:+.3f}, "
              f"|delta|/std={abs(delta)/max(s,1e-9):.2f})")

    print(f"  {'log_gmf_gap':>16s}: {df['log_gmf_gap'].mean():7.3f} "
          f"+/- {df['log_gmf_gap'].std():6.3f}")

    joint = _joint_best_fit(args.base_dir, variant=args.variant)
    if joint:
        print()
        print("=" * 78)
        print(f"Joint best fit (sum log_gmf over {joint['n_seeds_summed']} "
              f"seeds, variant={joint['variant']})")
        print("=" * 78)
        print(f"  joint_pi    = {joint['joint_pi']:7.3f}   "
              f"(truth={truth['pi']:6.3f}, delta={joint['joint_pi']-truth['pi']:+.3f})")
        print(f"  joint_kappa = {joint['joint_kappa']:7.3f}   "
              f"(truth={truth['kappa']:6.3f}, delta={joint['joint_kappa']-truth['kappa']:+.3f})")
        print(f"  joint_eta   = {joint['joint_eta']:7.3f}   "
              f"(truth={truth['eta']:6.3f}, delta={joint['joint_eta']-truth['eta']:+.3f})")
        print(f"  joint_fbin  = {joint['joint_fbin']:7.3f}   "
              f"(truth={truth['fbin']:6.3f}, delta={joint['joint_fbin']-truth['fbin']:+.3f})")
        print(f"  log_gmf_sum = {joint['joint_log_gmf_sum']:8.2f}")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWrote per-seed table to {args.out}")


if __name__ == "__main__":
    main()
