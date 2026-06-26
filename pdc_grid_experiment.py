"""
PDC Frequency Grid Resolution Experiment
=========================================

Compares different frequency grid strategies for the PDC periodogram:
  1. Current log-spaced grid (step=0.0005)
  2. Finer log-spaced grid (step=0.0001)  — reference "ground truth"
  3. Linear grid with samples_per_peak = 5, 10, 20

For each grid, measures:
  - Number of frequency points
  - Wall-clock time
  - Top-N peak periods and powers
  - Peak location deviation from ground truth

Usage:
    python pdc_grid_experiment.py [path_to_csv]

    Defaults to BLOeM_4-098_CCF_RVs.csv if no argument given.
"""

import sys
import time
import numpy as np
import pandas as pd

from PDC.pdc_func import calc_pdc_optimized
from period_search.periodogram import get_closest_b_vals, make_pdc_freq_grid
from period_search.candidates import significant_periods
from pipeline.data_loading import _load_and_clean_csv, _rename_to_internal_cols
from utils.constants import TIME_STAMPS, RADIAL_VELS, ERRORS

DEFAULT_CSV = (
    "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/"
    "CCF/for_zehava_from_coadded/BLOeM_4-098_CCF_RVs.csv"
)

PMIN = 0.4
PMAX = 10_000.0
N_TOP_PEAKS = 5
MIN_SEP = 1.1


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

def _log_grid(pmin, pmax, step):
    """Original log-period grid (descending freq)."""
    log_p = np.arange(np.log(pmin), np.log(pmax), step)
    p_range = np.exp(log_p)
    return 1.0 / p_range  # descending freq


def _linear_grid(pmin, pmax, T_baseline, samples_per_peak):
    """Linear freq grid via make_pdc_freq_grid."""
    return make_pdc_freq_grid(pmin, pmax, T_baseline, samples_per_peak)


# ---------------------------------------------------------------------------
# Run PDC on a given freq grid
# ---------------------------------------------------------------------------

def run_pdc_on_grid(freq, times, data, data_err):
    """Run PDC kernel, return (pdc_power, wall_seconds)."""
    # Pre-compute distance matrices (same as calc_pdc_optimized internals)
    from PDC.pdc_func import calc_pdc_distance_matrix_fast, calculate_pdc_loop
    A, _ = calc_pdc_distance_matrix_fast(data, data_err)
    times_arr = np.asarray(times, dtype=float)
    time_diff = times_arr[:, None] - times_arr[None, :]

    # Warm up numba (first call compiles)
    if len(freq) > 10:
        _ = calculate_pdc_loop(freq[:10], time_diff, A)

    t0 = time.perf_counter()
    pdc_power = calculate_pdc_loop(freq, time_diff, A)
    wall = time.perf_counter() - t0
    return pdc_power, wall


def extract_peaks(freq, power, n_peaks, min_sep):
    """Extract top peaks using the pipeline's significant_periods."""
    periods = 1.0 / freq
    sig = significant_periods(periods, power, max_periods=n_peaks, min_separation=min_sep)
    # Get power at each significant period
    peak_info = []
    for p in sig:
        idx = int(np.abs(periods - p).argmin())
        peak_info.append((p, power[idx]))
    return peak_info


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    print(f"Loading data from: {csv_path}")

    data = _load_and_clean_csv(csv_path)
    _rename_to_internal_cols(data)

    times = np.asarray(data[TIME_STAMPS], dtype=float)
    rvs = np.asarray(data[RADIAL_VELS], dtype=float)
    errs = np.asarray(data[ERRORS], dtype=float)

    T_baseline = float(np.ptp(times))
    n_data = len(times)
    print(f"Star: N={n_data} epochs, T_baseline={T_baseline:.1f} days")
    print(f"Period range: [{PMIN}, {PMAX}] days")
    print(f"Peak width (1/T): {1.0/T_baseline:.6f} /d")
    print()

    # Define grid configurations
    configs = [
        ("Log step=0.0001 (reference)", lambda: _log_grid(PMIN, PMAX, 0.0001)),
        ("Log step=0.0005 (current)",   lambda: _log_grid(PMIN, PMAX, 0.0005)),
        ("Linear spp=5",                lambda: _linear_grid(PMIN, PMAX, T_baseline, 5)),
        ("Linear spp=10",               lambda: _linear_grid(PMIN, PMAX, T_baseline, 10)),
        ("Linear spp=20",               lambda: _linear_grid(PMIN, PMAX, T_baseline, 20)),
    ]

    results = []
    ref_peaks = None

    for name, grid_fn in configs:
        freq = grid_fn()
        n_pts = len(freq)
        print(f"--- {name}: {n_pts:,} points ---")

        power, wall_s = run_pdc_on_grid(freq, times, rvs, errs)
        peaks = extract_peaks(freq, power, N_TOP_PEAKS, MIN_SEP)

        if ref_peaks is None:
            ref_peaks = peaks  # first config is reference

        # Compute deviation from reference — match by nearest period, not rank
        deviations = []
        for i, (p, pw) in enumerate(peaks):
            if ref_peaks:
                # Find the reference peak closest in log-period space
                best_ref_idx = min(range(len(ref_peaks)),
                                   key=lambda j: abs(np.log(p) - np.log(ref_peaks[j][0])))
                ref_p, ref_pw = ref_peaks[best_ref_idx]
                dp = abs(p - ref_p) / ref_p * 100
                dpw = abs(pw - ref_pw) / ref_pw * 100 if ref_pw > 0 else float('nan')
                deviations.append((dp, dpw))
            else:
                deviations.append((float('nan'), float('nan')))

        row = {
            "Grid": name,
            "N_points": n_pts,
            "Wall_s": wall_s,
        }
        for i, (p, pw) in enumerate(peaks):
            row[f"Peak{i+1}_P"] = f"{p:.4f}"
            row[f"Peak{i+1}_pow"] = f"{pw:.6f}"
        for i, (dp, dpw) in enumerate(deviations):
            row[f"Peak{i+1}_dP%"] = f"{dp:.3f}" if np.isfinite(dp) else "-"
            row[f"Peak{i+1}_dpow%"] = f"{dpw:.2f}" if np.isfinite(dpw) else "-"

        results.append(row)

        # Print peaks
        for i, (p, pw) in enumerate(peaks):
            dev_str = ""
            if i < len(deviations) and np.isfinite(deviations[i][0]):
                dev_str = f"  (ΔP={deviations[i][0]:.3f}%, Δpow={deviations[i][1]:.2f}%)"
            print(f"  Peak {i+1}: P={p:.4f} d, power={pw:.6f}{dev_str}")
        print(f"  Wall time: {wall_s:.3f}s")
        print()

    # Summary table
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    header = f"{'Grid':<30s} {'N_pts':>8s} {'Wall_s':>8s}"
    for i in range(min(3, N_TOP_PEAKS)):
        header += f" {'P'+str(i+1):>10s} {'pow'+str(i+1):>10s} {'dP%':>7s}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['Grid']:<30s} {r['N_points']:>8,d} {r['Wall_s']:>8.3f}"
        for i in range(min(3, N_TOP_PEAKS)):
            p_key = f"Peak{i+1}_P"
            pw_key = f"Peak{i+1}_pow"
            dp_key = f"Peak{i+1}_dP%"
            line += f" {r.get(p_key, '-'):>10s} {r.get(pw_key, '-'):>10s} {r.get(dp_key, '-'):>7s}"
        print(line)

    print()
    print("Reference grid: Log step=0.0001 (finest)")
    print("ΔP% = period deviation from reference peak location")


if __name__ == "__main__":
    main()
