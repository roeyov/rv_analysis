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


def prune_stuck_walkers(chain, log_prob, delta_mean=10.0,
                        delta_competitive=5.0):
    """
    Identify stuck walkers in a walker-resolved post-burn chain.

    A walker is flagged as stuck when its MEAN log-posterior over the
    retained steps sits more than *delta_mean* below the best walker's mean
    (its typical samples are ~e^delta_mean less probable than where the
    ensemble lives). Mode-competitiveness guard: a flagged walker whose
    single BEST sample comes within *delta_competitive* of the global
    maximum is kept — it is sampling a competitive secondary mode, not
    stuck, and pruning it would hide genuine multi-modality.

    Same recipe as the 2026-07 adopted clean reruns
    (Ostars_article/proposed_phase_figures/scripts/rerun_circ_mcmc_*.py).

    Parameters
    ----------
    chain : (n_steps, n_walkers, ndim) array
    log_prob : (n_steps, n_walkers) array

    Returns
    -------
    keep : (n_walkers,) bool array
    info : dict with n_dropped, n_competitive_kept, dropped walker indices
    """
    walker_mean = log_prob.mean(axis=0)
    walker_max = log_prob.max(axis=0)
    keep = walker_mean >= walker_mean.max() - delta_mean
    competitive = (log_prob.max() - walker_max) < delta_competitive
    n_competitive = int((~keep & competitive).sum())
    keep = keep | competitive
    info = {
        "n_dropped": int((~keep).sum()),
        "n_competitive_kept": n_competitive,
        "dropped": np.where(~keep)[0].tolist(),
        "delta_mean": delta_mean,
        "delta_competitive": delta_competitive,
    }
    if n_competitive:
        print(f"  prune guard: kept {n_competitive} low-mean walker(s) whose "
              f"best sample is within {delta_competitive} of the global "
              "maximum (competitive mode)")
    if info["n_dropped"]:
        print(f"  pruned {info['n_dropped']}/{len(keep)} stuck walker(s): "
              f"{info['dropped']}")
    return keep, info


def summarise_chain(flat_samples, names, tag, out_dir=".", add_jitter=False,
                    percentiles=(2.5, 5, 16, 50, 75, 84, 95, 97.5)):
    """
    Compute and save summary statistics (mode, median, errors, percentiles)
    for each parameter in the chain.

    Writes results_{tag}.txt (one "name: k=v, k=v" line per parameter — no
    other line shapes; the paper-side parser depends on that) and a
    percentiles_{tag}.csv sidecar with the full percentile ladder.

    Returns a dict of {f'{tag}_{name}_{stat}': value} entries, including
    p75/p95 for every parameter.
    """
    percentiles = tuple(sorted(set(percentiles) | {16, 50, 75, 84, 95}))
    outfile = os.path.join(out_dir, f"results_{tag}.txt")
    pctfile = os.path.join(out_dir, f"percentiles_{tag}.csv")
    res_dict = {}
    with open(outfile, "w") as f, open(pctfile, "w") as pf:
        pf.write("param," + ",".join(f"p{p:g}" for p in percentiles) + "\n")
        for i, name in enumerate(names):
            vals = flat_samples[:, i]
            pvals = np.percentile(vals, percentiles)
            pmap = dict(zip(percentiles, pvals))
            q16, q50, q84 = pmap[16], pmap[50], pmap[84]
            errm, errp = q50 - q16, q84 - q50

            hist, bins = np.histogram(vals, bins=50)
            imax = np.argmax(hist)
            mode = 0.5 * (bins[imax] + bins[imax+1])

            f.write(
                f"{name}: mode={mode:.6g}, "
                f"median={q50:.6g}, "
                f"-err={errm:.6g}, +err={errp:.6g}, "
                + ", ".join(f"p{p:g}={pmap[p]:.6g}" for p in percentiles)
                + "\n"
            )
            pf.write(name + "," + ",".join(f"{v:.6g}" for v in pvals) + "\n")

            res_dict[f'{tag}_{name}_mode']= mode
            res_dict[f'{tag}_{name}_median']= q50
            res_dict[f'{tag}_{name}_errm']= errm
            res_dict[f'{tag}_{name}_errp']= errp
            res_dict[f'{tag}_{name}_p75'] = pmap[75]
            res_dict[f'{tag}_{name}_p95'] = pmap[95]
    print(f"Saved {outfile}")
    print(f"Saved {pctfile}")
    return res_dict
