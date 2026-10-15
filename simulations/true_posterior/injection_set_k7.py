"""
simulations.true_posterior.injection_set_k7 — build an injection set whose
detector scores the Keplerian model with seven free parameters in the
corrected BIC (see ``k7_patch``). Same command line as ``injection_set``.

With the seed, chunk size and number of noise-only curves per chunk of an
existing set, the records are the same simulated binaries and noise curves,
classified with the other parameter count: a paired comparison.

The switch reaches the pool workers through the forkserver preload (Linux).
A failed preload is silent, so the build is checked afterwards: the
noise-only flag rate of the set must lie below ``--max-fp`` (the coded count
gives 2.2 per cent, seven parameters about 0.05 per cent). On other
platforms keep ``--n-workers 1``.

    python -u -m simulations.true_posterior.injection_set_k7 \
        --config configs/params_bias_astro3_ed_hoc_rlof_check_ext3.yaml \
        --out-dir <new dir> --n-per-star 20000 --chunk 250 \
        --n-single-per-star 2000 --n-workers 80
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys

import numpy as np
import yaml

PRELOAD = "simulations.true_posterior.k7_patch"


def flag_rates(out_dir):
    """(detected fraction of the binaries, flag rate of the noise curves)."""
    n_bin = n_det = n_sng = n_fp = 0
    for f in sorted(glob.glob(os.path.join(out_dir, "shards", "*.npz"))):
        with np.load(f) as z:
            n_bin += z["det"].size
            n_det += int(np.sum(z["det"]))
            if "single_det" in z.files:
                n_sng += z["single_det"].size
                n_fp += int(np.sum(z["single_det"]))
    return n_det / max(n_bin, 1), n_fp / max(n_sng, 1), n_bin, n_sng


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--max-fp", type=float, default=0.005)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-workers", type=int, default=1)
    a, rest = ap.parse_known_args(argv)
    if a.n_workers > 1 and not sys.platform.startswith("linux"):
        raise SystemExit("injection_set_k7: the switch reaches pool workers "
                         "only through the Linux forkserver; use "
                         "--n-workers 1 here")
    mp.set_forkserver_preload([PRELOAD])
    from simulations.true_posterior import k7_patch
    from simulations.true_posterior import injection_set
    assert k7_patch.orbit_count() == k7_patch.N_ORBIT_PAPER
    rc = injection_set.main(rest + ["--out-dir", a.out_dir,
                                    "--n-workers", str(a.n_workers)])
    det, fp, n_bin, n_sng = flag_rates(a.out_dir)
    rec = {"module": PRELOAD, "bicc_nvarys_orbit": k7_patch.N_ORBIT_PAPER,
           "bicc_nvarys_constant": k7_patch.N_NULL,
           "detected_fraction": float(det), "noise_flag_rate": float(fp),
           "n_binaries": int(n_bin), "n_noise_curves": int(n_sng),
           "max_fp": float(a.max_fp)}
    with open(os.path.join(a.out_dir, "detector_patch.yaml"), "w") as fh:
        yaml.safe_dump(rec, fh, sort_keys=False)
    print("[injset_k7] detected fraction %.4f, noise-only flag rate %.5f "
          "(%d curves)" % (det, fp, n_sng), flush=True)
    if n_sng and not (fp <= a.max_fp):
        print("[injset_k7] FAILED: the noise-only flag rate is above %.4f; "
              "the seven-parameter switch did not reach the workers"
              % a.max_fp, flush=True)
        open(os.path.join(a.out_dir, "PATCH_NOT_APPLIED"), "w").close()
        return 3
    return rc


if __name__ == "__main__":
    sys.exit(main())
