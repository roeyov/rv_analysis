"""Paired A/B of the Tier-B detection-config levers (Gate B).

For each corner cell of the ed_hoc_rlof_check_ext grid, draw ONE fixed binary
population (fixed seeds, shared across variants) and run ``detect_batch``
under the base periodogram config and each candidate lever:

  base   min_period 0.4, spp 50           (production defaults)
  B1     min_period 1.0                   (injected floor is P = 1.11 d)
  B2_25 / B2_20 / B2_15                   ls_sample_per_peak variants
  B1B2   min_period 1.0 + spp 20          (recommended combo)
  B3     LS branch only — evaluated from the BASE run's detected_ls
         (branches are OR'd, so no separate detection pass is needed)

The comparison is fully paired: identical injected curves and identical
detect-rng seed per (cell, field), so per-curve detection flips are
attributable to the config alone. Per (cell, variant) the report gives
p_det, Δp_det vs base, 0→1/1→0 flip counts, an exact McNemar p-value, and
wall time. Output CSV: $SCRIPTS_OUT/bias_speedup/ab_levers.csv.

Run pinned single-thread for reproducible timing:
    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    python -m fast_detect.ab_levers [--n-per-field 1700] [--fields 0,4,7]
"""

import argparse
import csv
import os
import time

import numpy as np
from scipy import stats

from simulations.common import BLOEM_MJD_ARRAYS
from simulations.bias_grid_lib.injection import _draw_star_binaries
from fast_detect.detector import detect_batch

_SCRIPTS_OUT = os.environ.get(
    "SCRIPTS_OUT", "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut")
_OUT_CSV = os.path.join(_SCRIPTS_OUT, "bias_speedup", "ab_levers.csv")

_DRAW_CFG = {"log_p_min": 0.046, "log_p_max": 3.5,
             "q_min": 0.01, "q_max": 1.0, "e_max": 0.95, "p_circ": 2.26}
_SEED0 = 30260730

# (pi, kappa) corners of the ext grid x mid eta; f_bin is irrelevant here
# (pure-binary populations - p_det is per-binary).
_CELLS = [
    ("pi-1.70_ka-3.05", -1.70, -3.05, -0.35),
    ("pi-1.70_ka+1.80", -1.70, 1.80, -0.35),
    ("pi-0.82_ka-3.05", -0.82, -3.05, -0.35),
    ("pi-0.82_ka+1.80", -0.82, 1.80, -0.35),
    ("pi+0.20_ka-3.05", 0.20, -3.05, -0.35),
    ("pi+0.20_ka+1.80", 0.20, 1.80, -0.35),
]

# Star parameter sets cycled per curve quarter (M1, R1, gamma, rv_err).
_STARS = [(20.0, 8.0, 140.0, 3.0), (25.0, 9.0, 150.0, 6.0),
          (30.0, 10.0, 160.0, 9.0), (45.0, 13.0, 175.0, 12.0)]


def _peri(pmin=0.4, spp=50, run_pdc=True):
    return {"min_period": pmin, "max_period": 10000.0,
            "ls_sample_per_peak": spp, "window_iterations": 10,
            "pdc_samples_per_peak": 10, "run_pdc": run_pdc}


_VARIANTS = [
    ("base", _peri()),
    ("B1_pmin1.0", _peri(pmin=1.0)),
    ("B2_spp25", _peri(spp=25)),
    ("B2_spp20", _peri(spp=20)),
    ("B2_spp15", _peri(spp=15)),
    ("B1B2_pmin1.0_spp20", _peri(pmin=1.0, spp=20)),
    # run_pdc=False must reproduce the base run's detected_ls exactly
    # (checked below) — this variant exists to measure B3's wall time.
    ("B3_run_pdc_false", _peri(run_pdc=False)),
    ("B1B2B3_full_stack", _peri(pmin=1.0, spp=20, run_pdc=False)),
]


def _cell_population(cell_seed, fields, n_per_field, pi, kappa, eta):
    """Fixed binary population per cell: {field: (Y, errs)}."""
    pops = {}
    for field in fields:
        mjds = np.array(BLOEM_MJD_ARRAYS[field], float)
        rng = np.random.default_rng(cell_seed + field)
        blocks, errs = [], []
        n_star = n_per_field // len(_STARS)
        for (M1, R1, gamma, rv_err) in _STARS:
            (_, _, _, _, rv_obs, _, _) = _draw_star_binaries(
                mjds, mjds.size, rv_err, M1, R1, gamma,
                n_star, pi, kappa, eta, _DRAW_CFG, rng)
            blocks.append(rv_obs)
            errs.append(np.full(n_star, rv_err))
        pops[field] = (np.concatenate(blocks, axis=0).T, np.concatenate(errs))
    return pops


def _mcnemar_p(flips01, flips10):
    n = flips01 + flips10
    if n == 0:
        return 1.0
    return stats.binomtest(min(flips01, flips10), n, 0.5,
                           alternative="two-sided").pvalue


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-per-field", type=int, default=1700)
    ap.add_argument("--fields", default="0,4,7")
    ap.add_argument("--out", default=_OUT_CSV)
    args = ap.parse_args()
    fields = [int(f) for f in args.fields.split(",")]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for ci, (cname, pi, kappa, eta) in enumerate(_CELLS):
        pops = _cell_population(_SEED0 + 1000 * ci, fields, args.n_per_field,
                                pi, kappa, eta)
        det_by_variant = {}
        ls_base = []
        for vname, peri in _VARIANTS:
            cfg = {"apply_rlof_gate": False, "periodogram_params": peri}
            det_all = []
            t0 = time.perf_counter()
            for field in fields:
                Y, errs = pops[field]
                det, info = detect_batch(
                    np.array(BLOEM_MJD_ARRAYS[field], float), Y, errs, None,
                    cfg, backend="numpy",
                    rng=np.random.default_rng(_SEED0 + 77 * ci + field))
                det_all.append(det)
                if vname == "base":
                    ls_base.append(info["detected_ls"])
            det_by_variant[vname] = np.concatenate(det_all)
            wall = time.perf_counter() - t0
            base = det_by_variant["base"]
            det = det_by_variant[vname]
            f01 = int(np.sum(~base & det))
            f10 = int(np.sum(base & ~det))
            rows.append({
                "cell": cname, "pi": pi, "kappa": kappa, "eta": eta,
                "variant": vname, "n": det.size,
                "p_det": det.mean(), "p_det_base": base.mean(),
                "delta": det.mean() - base.mean(),
                "flips_0to1": f01, "flips_1to0": f10,
                "mcnemar_p": _mcnemar_p(f01, f10), "wall_s": wall,
            })
            print("%-18s %-20s p_det=%.4f d=%+.4f flips=+%d/-%d p=%.3g %.1fs"
                  % (cname, vname, det.mean(), det.mean() - base.mean(),
                     f01, f10, rows[-1]["mcnemar_p"], wall))
        # Toggle validation: run_pdc=False must equal the base run's
        # detected_ls exactly (same seeds, LS branch untouched).
        ls = np.concatenate(ls_base)
        if not np.array_equal(det_by_variant["B3_run_pdc_false"], ls):
            raise AssertionError(
                "%s: run_pdc=False detected mask != base detected_ls" % cname)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
