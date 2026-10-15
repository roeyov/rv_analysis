"""
sb2_pipeline.replot — regenerate MCMC figures from saved chains (no
re-sampling): phase-band plots and walker traces for every completed star.

Usage:
    python -m sb2_pipeline.replot --config configs/params_sb2_fixed_rvs.yaml
        [--stars 1-006 ...]
"""

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import argparse
import os

import pandas as pd

from pipeline.config import load_args
from mcmc.chain_io import load_chain

from sb2_pipeline.constants import SB2_IO, OUTPUT_DIR
from sb2_pipeline.io import make_data_dict
from sb2_pipeline.plotting import (
    plot_sb2_orbit_with_band_phase, plot_chain_traces,
)


def replot_star(star_out, star_id):
    mcmc_csv = os.path.join(star_out, "data", "rv_sb2_for_mcmc.csv")
    data = make_data_dict(pd.read_csv(mcmc_csv, comment="#"))

    ecc_npz = os.path.join(star_out, "mcmc", "chains_ecc.npz")
    d = load_chain(ecc_npz)
    plot_sb2_orbit_with_band_phase(
        data, d["flat_samples_pruned"], d["flat_logprob_pruned"],
        tag="ecc", circular=False,
        out_dir=os.path.join(star_out, "mcmc"), star_name=star_id)
    plot_chain_traces(ecc_npz,
                      os.path.join(star_out, "mcmc", "traces_ecc.png"),
                      star_name=star_id)

    circ_npz = os.path.join(star_out, "mcmc", "circular", "chains_circ.npz")
    if os.path.exists(circ_npz):
        d = load_chain(circ_npz)
        plot_sb2_orbit_with_band_phase(
            data, d["flat_samples_pruned"], d["flat_logprob_pruned"],
            tag="circ", circular=True,
            out_dir=os.path.join(star_out, "mcmc", "circular"),
            star_name=star_id)
        plot_chain_traces(
            circ_npz,
            os.path.join(star_out, "mcmc", "circular", "traces_circ.png"),
            star_name=star_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stars", nargs="*", default=None)
    cli = ap.parse_args()

    out_root = load_args(cli.config)[SB2_IO][OUTPUT_DIR]
    star_dirs = sorted(
        d for d in os.listdir(out_root)
        if os.path.exists(os.path.join(out_root, d, "mcmc", "star_done.json"))
    )
    if cli.stars:
        star_dirs = [d for d in star_dirs if any(p in d for p in cli.stars)]

    for star_id in star_dirs:
        print(f"replotting {star_id}")
        try:
            replot_star(os.path.join(out_root, star_id), star_id)
        except Exception as e:
            print(f"  {star_id}: replot failed: {e}")


if __name__ == "__main__":
    main()
