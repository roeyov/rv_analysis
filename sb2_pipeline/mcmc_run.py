"""
sb2_pipeline.mcmc_run — emcee runners for the joint SB2 model, the
Lucy-Sweeney gate, and the per-star MCMC driver (eccentric -> gate ->
circular fallback). Delegates sampling to mcmc.runner.run_mcmc_generic.
"""

import os

import numpy as np

from utils.constants import (
    LMFIT_PARAMS, MCMC_PARAMS, WALKERS, STEPS, BURN, THIN,
    DP_FRAC, SIGMA_P_FRAC, PRUNE_STUCK_WALKERS,
    PERIOD, GAMMA, K1_STR, K2_STR, OMEGA, ECC, T,
    LN_SIGMA_JITTER_1, LN_SIGMA_JITTER_2,
)
from mcmc.runner import run_mcmc_generic
from mcmc.chain_io import derive_seed
from mcmc.analysis import lucy_sweeney_significant, summarise_chain
from mcmc.batch import get_param

from sb2_pipeline.constants import (
    ECC_PARAM_NAMES, CIRC_PARAM_NAMES, ECC_LABELS, CIRC_LABELS,
    ECC_PARAM_NAMES_SHARED, CIRC_PARAM_NAMES_SHARED,
    ECC_LABELS_SHARED, CIRC_LABELS_SHARED,
    LUCY_SWEENEY_THRESHOLD, LUCY_SWEENEY_ESTIMATOR,
)
from sb2_pipeline.mcmc_model import (
    log_probability_sb2, log_probability_sb2_circ,
    log_probability_sb2_sharedJ, log_probability_sb2_circ_sharedJ,
)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_mcmc_sb2_ecc(t1, rv1, err1, t2, rv2, err2, initial, P_center,
                     T0_center, dT0_days, dP_frac=0.01, sigma_P_frac=0.002,
                     nwalkers=32, nsteps=4000, nburn=100, thin=10,
                     progress=True, seed=None, save_path=None, meta=None,
                     prune=False, shared_jitter=False):
    """Eccentric joint MCMC. initial = [P, T0, omega, e, K1, K2, gamma]."""
    initial = np.asarray(initial, float)
    if shared_jitter:
        initial = np.append(initial,
                            np.log(np.median(np.concatenate([err1, err2]))))
    else:
        initial = np.append(initial, [np.log(np.median(err1)),
                                      np.log(np.median(err2))])

    jitter = np.array([
        1e-3 * max(initial[0], 1e-3),      # P
        0.1,                               # T0
        0.05,                              # omega
        0.02,                              # e
        0.05 * max(initial[4], 1e-2),      # K1
        0.05 * max(initial[5], 1e-2),      # K2
        0.1,                               # gamma
    ] + ([0.1] if shared_jitter else [0.1, 0.1]))

    logprob_args = (t1, rv1, err1, t2, rv2, err2, P_center, T0_center,
                    initial[4], initial[5], initial[2],
                    dP_frac, dT0_days, sigma_P_frac)

    return run_mcmc_generic(
        initial,
        log_probability_sb2_sharedJ if shared_jitter else log_probability_sb2,
        logprob_args,
        jitter, nwalkers, nsteps, nburn, thin, progress,
        seed=seed, save_path=save_path,
        param_names=(ECC_PARAM_NAMES_SHARED if shared_jitter
                     else ECC_PARAM_NAMES),
        meta=meta, prune=prune,
    )


def run_mcmc_sb2_circ(t1, rv1, err1, t2, rv2, err2, initial, P_center,
                      T0_center, dT0_days, dP_frac=0.01, sigma_P_frac=0.002,
                      nwalkers=32, nsteps=4000, nburn=100, thin=10,
                      progress=True, seed=None, save_path=None, meta=None,
                      prune=False, shared_jitter=False):
    """Circular joint MCMC. initial = [P, T0, K1, K2, gamma]."""
    initial = np.asarray(initial, float)
    if shared_jitter:
        initial = np.append(initial,
                            np.log(np.median(np.concatenate([err1, err2]))))
    else:
        initial = np.append(initial, [np.log(np.median(err1)),
                                      np.log(np.median(err2))])

    jitter = np.array([
        1e-3 * max(initial[0], 1e-3),      # P
        0.1,                               # T0
        0.05 * max(initial[2], 1e-2),      # K1
        0.05 * max(initial[3], 1e-2),      # K2
        0.1,                               # gamma
    ] + ([0.1] if shared_jitter else [0.1, 0.1]))

    logprob_args = (t1, rv1, err1, t2, rv2, err2, P_center, T0_center,
                    initial[2], initial[3], dP_frac, dT0_days, sigma_P_frac)

    return run_mcmc_generic(
        initial,
        (log_probability_sb2_circ_sharedJ if shared_jitter
         else log_probability_sb2_circ),
        logprob_args,
        jitter, nwalkers, nsteps, nburn, thin, progress,
        seed=seed, save_path=save_path,
        param_names=(CIRC_PARAM_NAMES_SHARED if shared_jitter
                     else CIRC_PARAM_NAMES),
        meta=meta, prune=prune,
    )


# ---------------------------------------------------------------------------
# Lucy-Sweeney gate
# ---------------------------------------------------------------------------

def sb2_lucy_sweeney(ecc_summary, estimator="median", threshold=2.49):
    """
    Single switch point for the eccentricity-significance test.

    estimator "median" (SB2/paper convention: e_median / (p50-p16)) or
    "mode" (SB1 batch convention). sigma is always the minus-side error.
    """
    if estimator not in ("median", "mode"):
        raise ValueError(f"Unknown Lucy-Sweeney estimator {estimator!r}")
    e_est = ecc_summary[f"ecc_e_{estimator}"]
    sigma_minus = ecc_summary["ecc_e_errm"]
    significant = lucy_sweeney_significant(e_est, sigma_minus,
                                           threshold=threshold)
    ratio = e_est / sigma_minus if sigma_minus and sigma_minus > 0 else np.inf
    return {
        "e_est": e_est,
        "e_sigma_minus": sigma_minus,
        "e_over_sigma": ratio,
        "lucy_sweeney_estimator": estimator,
        "lucy_sweeney_threshold": threshold,
        "lucy_sweeney_significant": bool(significant),
    }


# ---------------------------------------------------------------------------
# Per-star driver
# ---------------------------------------------------------------------------

def run_mcmc_sb2_star(args_dict, best_row, data, out_dir, star_name=""):
    """
    Eccentric MCMC -> corner/phase/traces -> Lucy-Sweeney gate -> circular
    MCMC into out_dir/circular when eccentricity is not significant.
    data: the make_data_dict() dict (canonical K1<K2 labelling).
    Returns the cross-star summary-row dict.
    """
    from sb2_pipeline.constants import (
        SB2_TS1, SB2_TS2, SB2_RV1, SB2_ERR1, SB2_RV2, SB2_ERR2,
    )
    from sb2_pipeline.plotting import (
        make_corner_sb2, plot_sb2_orbit_with_band_phase, plot_chain_traces,
    )

    t1 = data[SB2_TS1]
    rv1 = data[SB2_RV1]
    err1 = data[SB2_ERR1]
    t2 = data[SB2_TS2]
    rv2 = data[SB2_RV2]
    err2 = data[SB2_ERR2]

    from sb2_pipeline.constants import JITTER_MODE

    mp = args_dict[MCMC_PARAMS]
    dP_frac = float(mp.get(DP_FRAC, 0.01))
    sigma_P_frac = float(mp.get(SIGMA_P_FRAC, 0.002))
    prune = mp.get(PRUNE_STUCK_WALKERS, False)
    base_seed = args_dict.get("random_seed")
    shared = (args_dict[LMFIT_PARAMS].get(JITTER_MODE, "separate")
              == "shared")
    ecc_names = ECC_PARAM_NAMES_SHARED if shared else ECC_PARAM_NAMES
    circ_names = CIRC_PARAM_NAMES_SHARED if shared else CIRC_PARAM_NAMES
    ecc_labels = ECC_LABELS_SHARED if shared else ECC_LABELS
    circ_labels = CIRC_LABELS_SHARED if shared else CIRC_LABELS
    n_jit = 1 if shared else 2

    P_val = float(get_param(best_row, PERIOD, kind="value"))
    T0_val = float(get_param(best_row, T, kind="value"))
    omega_val = float(get_param(best_row, OMEGA, kind="value"))
    ecc_val = float(get_param(best_row, ECC, kind="value"))
    K1_val = float(get_param(best_row, K1_STR, kind="value"))
    K2_val = float(get_param(best_row, K2_STR, kind="value"))
    gamma_val = float(get_param(best_row, GAMMA, kind="value"))

    initial_ecc = np.array([P_val, T0_val, omega_val, ecc_val,
                            K1_val, K2_val, gamma_val])
    dT0_days = P_val / 2.0

    # -- Eccentric MCMC --
    seed_ecc = (derive_seed(base_seed, star_name, "ecc")
                if base_seed is not None else None)
    save_ecc = os.path.join(out_dir, "chains_ecc.npz")
    flat_ecc, logp_ecc, _ = run_mcmc_sb2_ecc(
        t1, rv1, err1, t2, rv2, err2, initial_ecc,
        P_center=P_val, T0_center=T0_val, dT0_days=dT0_days,
        dP_frac=dP_frac, sigma_P_frac=sigma_P_frac,
        nwalkers=mp[WALKERS], nsteps=mp[STEPS], nburn=mp[BURN], thin=mp[THIN],
        progress=True, seed=seed_ecc, save_path=save_ecc,
        meta={"star_name": star_name, "tag": "ecc"}, prune=prune,
        shared_jitter=shared,
    )
    print("Posterior samples (ecc):", flat_ecc.shape)

    dict_row = summarise_chain(flat_ecc, ecc_names, tag="ecc",
                               out_dir=out_dir)

    make_corner_sb2(flat_ecc, logp_ecc, ecc_labels, tag="ecc",
                    out_dir=out_dir, star_name=star_name,
                    omega_col=2, n_trailing_drop=n_jit)
    plot_sb2_orbit_with_band_phase(
        data, flat_ecc, logp_ecc,
        tag="ecc", circular=False, out_dir=out_dir, star_name=star_name)
    plot_chain_traces(save_ecc, os.path.join(out_dir, "traces_ecc.png"),
                      star_name=star_name)

    # Label-swap diagnostic: fraction of posterior with K1 > K2.
    dict_row["ecc_swap_fraction"] = float(np.mean(flat_ecc[:, 4] > flat_ecc[:, 5]))

    # -- Lucy-Sweeney gate --
    ls = sb2_lucy_sweeney(
        dict_row,
        estimator=mp.get(LUCY_SWEENEY_ESTIMATOR, "median"),
        threshold=float(mp.get(LUCY_SWEENEY_THRESHOLD, 2.49)),
    )
    dict_row.update(ls)
    print(f"Lucy-Sweeney ({ls['lucy_sweeney_estimator']}, "
          f">{ls['lucy_sweeney_threshold']}): significant? "
          f"{ls['lucy_sweeney_significant']}")

    # -- Circular MCMC (when e NOT significant) --
    if not ls["lucy_sweeney_significant"]:
        circ_dir = os.path.join(out_dir, "circular")
        os.makedirs(circ_dir, exist_ok=True)
        initial_circ = np.array([P_val, T0_val, K1_val, K2_val, gamma_val])
        seed_circ = (derive_seed(base_seed, star_name, "circ")
                     if base_seed is not None else None)
        save_circ = os.path.join(circ_dir, "chains_circ.npz")
        flat_circ, logp_circ, _ = run_mcmc_sb2_circ(
            t1, rv1, err1, t2, rv2, err2, initial_circ,
            P_center=P_val, T0_center=T0_val, dT0_days=dT0_days,
            dP_frac=dP_frac, sigma_P_frac=sigma_P_frac,
            nwalkers=mp[WALKERS], nsteps=mp[STEPS], nburn=mp[BURN],
            thin=mp[THIN], progress=True, seed=seed_circ, save_path=save_circ,
            meta={"star_name": star_name, "tag": "circ"}, prune=prune,
            shared_jitter=shared,
        )
        print("Posterior samples (circ):", flat_circ.shape)

        dict_row.update(summarise_chain(flat_circ, circ_names,
                                        tag="circ", out_dir=circ_dir))
        make_corner_sb2(flat_circ, logp_circ, circ_labels, tag="circ",
                        out_dir=circ_dir, star_name=star_name,
                        omega_col=None, n_trailing_drop=n_jit)
        plot_sb2_orbit_with_band_phase(
            data, flat_circ, logp_circ,
            tag="circ", circular=True, out_dir=circ_dir, star_name=star_name)
        plot_chain_traces(save_circ, os.path.join(circ_dir, "traces_circ.png"),
                          star_name=star_name)
        dict_row["adopted_model"] = "circ"
    else:
        print("Skipping circular MCMC: eccentricity is significant.")
        dict_row["adopted_model"] = "ecc"

    return dict_row
