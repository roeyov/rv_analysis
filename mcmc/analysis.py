"""
mcmc.analysis — Lucy-Sweeney eccentricity test and chain summary statistics.
"""

import os
import numpy as np


def lucy_sweeney_significant(e, sigma_e, threshold=2.45, add_jitter=False):
    """
    Lucy–Sweeney test: eccentricity is significant if e / sigma_e >= threshold.
    """
    if sigma_e is None or not np.isfinite(sigma_e) or sigma_e <= 0:
        print("Lucy–Sweeney: sigma_e invalid; treating e as NOT significant.")
        return False

    ratio = e / sigma_e
    print(f"Lucy–Sweeney: e = {e:.4f}, sigma_e = {sigma_e:.4f}, e/sigma_e = {ratio:.2f}")
    return ratio >= threshold


def summarise_chain(flat_samples, names, tag, out_dir=".", add_jitter=False):
    """
    Compute and save summary statistics (mode, median, errors) for each
    parameter in the chain.

    Returns a dict of {f'{tag}_{name}_{stat}': value} entries.
    """
    outfile = os.path.join(out_dir, f"results_{tag}.txt")
    res_dict = {}
    with open(outfile, "w") as f:
        for i, name in enumerate(names):
            vals = flat_samples[:, i]
            q16, q50, q84 = np.percentile(vals, [16, 50, 84])
            errm, errp = q50 - q16, q84 - q50

            hist, bins = np.histogram(vals, bins=50)
            imax = np.argmax(hist)
            mode = 0.5 * (bins[imax] + bins[imax+1])

            f.write(
                f"{name}: mode={mode:.6g}, "
                f"median={q50:.6g}, "
                f"-err={errm:.6g}, +err={errp:.6g}\n"
            )
            res_dict[f'{tag}_{name}_mode']= mode
            res_dict[f'{tag}_{name}_median']= q50
            res_dict[f'{tag}_{name}_errm']= errm
            res_dict[f'{tag}_{name}_errp']= errp
    print(f"Saved {outfile}")
    return res_dict
