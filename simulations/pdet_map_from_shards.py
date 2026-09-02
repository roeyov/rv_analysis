"""p_det(P, q) detection-probability map at a chosen posterior point.

Tier 3.1 of the 2026-09 variant campaign — the definitive answer to
LP-010 (does the survey have any sensitivity below q = 0.1?) and JV-018.

Unlike the explorer's run-global (logP, q) map (summed over ALL grid
nodes and q-binned only down to 0.1 by ``_HIST_BINS``), this bins the
RAW per-injection arrays of the det shards at the node(s) around a
chosen (pi, kappa, eta, f_bin) point, with custom q bins down to the
injection floor (0.01). p_det(P, q) = detected binaries / injected
binaries per cell, marginalised over e, inclination, omega, T0, noise
draws, and the per-star cadences/errors of the full 134-star sample —
exactly the conditional detection probability at the adopted posterior
point.

Emits a compact npz (the A&A figure is rendered separately in the paper
repo); optional pooling over a +/-N-node neighborhood trades locality
for injection statistics.

Usage (astro3):
  python -m simulations.pdet_map_from_shards \
      --run-dir /jonathan_storage/roeyovadia/bias_grid_results/ed_hoc_rlof_check_ext3 \
      --center map --neighborhood 1 \
      --out /tmp/ext3_pdet_map.npz
"""

import argparse
import os
import sys

import numpy as np

GRID_KEYS = ("pi_grid", "kappa_grid", "eta_grid", "fbin_grid")


def _load(run_dir, variant, metric):
    path = None
    for name in ("grid_cubes.npz", "checkpoint_cubes.npz"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError("no cube npz in %s" % run_dir)
    z = np.load(path, allow_pickle=True)
    grids = [np.asarray(z[g], dtype=float) for g in GRID_KEYS]
    tags = [str(t) for t in np.atleast_1d(z["variants"])]
    if variant is None:
        variant = tags[0]
    gmf = np.asarray(z["v__%s__gmf_%s_cube" % (variant, metric)],
                     dtype=np.float64)
    return z, grids, gmf, variant


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--metric", default="cvm")
    ap.add_argument("--center", default="map",
                    help="'map' (joint argmax) or 'pi,kappa,eta,fbin'")
    ap.add_argument("--neighborhood", type=int, default=1,
                    help="pool shards within +/-N nodes on every axis "
                         "(N=1 -> up to 3^4 = 81 nodes)")
    ap.add_argument("--n-logp-bins", type=int, default=24)
    ap.add_argument("--n-q-bins", type=int, default=20)
    ap.add_argument("--logp-range", default="0.046,3.5")
    ap.add_argument("--q-range", default="0.01,1.0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z, grids, gmf, variant = _load(args.run_dir, args.variant, args.metric)

    if args.center == "map":
        center = np.unravel_index(np.nanargmax(gmf), gmf.shape)
    else:
        vals = [float(x) for x in args.center.split(",")]
        center = tuple(int(np.argmin(np.abs(g - v)))
                       for g, v in zip(grids, vals))
    center_vals = [float(g[i]) for g, i in zip(grids, center)]
    print("center node %s -> (pi=%.3f kappa=%.3f eta=%.3f fbin=%.3f)"
          % (center, *center_vals))

    # Select steps whose ijkl fall inside the neighborhood box.
    idx_path = os.path.join(args.run_dir, "det_index.npz")
    if not os.path.exists(idx_path):
        raise FileNotFoundError("%s missing — cannot map steps to nodes"
                                % idx_path)
    step_to_ijkl = np.asarray(np.load(idx_path)["step_to_ijkl"])
    ijkl = step_to_ijkl[:, 1:5]
    N = args.neighborhood
    inside = np.all(np.abs(ijkl - np.asarray(center)) <= N, axis=1)
    steps = step_to_ijkl[inside, 0]
    print("pooling %d shard(s) within +/-%d nodes" % (len(steps), N))
    if len(steps) == 0:
        raise RuntimeError("no shards inside the neighborhood")

    lp_lo, lp_hi = (float(x) for x in args.logp_range.split(","))
    q_lo, q_hi = (float(x) for x in args.q_range.split(","))
    logp_edges = np.linspace(lp_lo, lp_hi, args.n_logp_bins + 1)
    # log-spaced q bins resolve the q < 0.1 decade LP-010 asks about.
    q_edges = np.logspace(np.log10(q_lo), np.log10(q_hi),
                          args.n_q_bins + 1)

    h_det = np.zeros((args.n_logp_bins, args.n_q_bins), dtype=np.int64)
    h_tot = np.zeros_like(h_det)
    shard_dir = os.path.join(args.run_dir, "det_shards")
    n_det_inj = 0
    n_tot_inj = 0
    for step in steps:
        with np.load(os.path.join(shard_dir,
                                  "step_%06d.npz" % int(step))) as d:
            for arrs, hist in (((d["logP"], d["q"]), h_det),
                               ((d["logP_nondet"], d["q_nondet"]), None)):
                lp, q = (np.asarray(a, dtype=float) for a in arrs)
                hh, _, _ = np.histogram2d(lp, q,
                                          bins=(logp_edges, q_edges))
                if hist is h_det:
                    h_det += hh.astype(np.int64)
                    n_det_inj += len(lp)
                h_tot += hh.astype(np.int64)
                if hist is None:
                    n_tot_inj += len(lp)
    n_tot_inj += n_det_inj
    print("binned %d detected / %d injected binaries"
          % (n_det_inj, n_tot_inj))

    with np.errstate(invalid="ignore", divide="ignore"):
        pdet = np.where(h_tot > 0, h_det / h_tot, np.nan)

    # Sensitivity headline for the LP-010 text: detections below q=0.1.
    q_centers = 0.5 * (q_edges[1:] + q_edges[:-1])
    sub01 = q_centers < 0.1
    print("q<0.1 injected: %d, detected: %d (p_det=%.3f)"
          % (h_tot[:, sub01].sum(), h_det[:, sub01].sum(),
             h_det[:, sub01].sum() / max(h_tot[:, sub01].sum(), 1)))

    np.savez_compressed(
        args.out,
        pdet=pdet.astype(np.float32),
        n_det=h_det, n_tot=h_tot,
        logp_edges=logp_edges, q_edges=q_edges,
        center_ijkl=np.asarray(center),
        center_values=np.asarray(center_vals),
        neighborhood=np.int64(N),
        n_shards=np.int64(len(steps)),
        variant=np.str_(variant), metric=np.str_(args.metric),
        run_dir=np.str_(os.path.abspath(args.run_dir)),
    )
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
