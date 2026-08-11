"""
mcmc.runner — emcee MCMC wrappers for eccentric, circular, and null models.
"""

import numpy as np
import emcee

from mcmc.models import (
    log_probability,
    log_probability_circ,
    log_probability_null,
    log_probability_double_kepler,
)
from mcmc.chain_io import save_chain


def run_mcmc_generic(initial, log_prob_fn, logprob_args,
                     jitter, nwalkers=1000, nsteps=4000,
                     nburn=100, thin=10, progress=True,
                     seed=None, save_path=None, param_names=None, meta=None,
                     prune=False):
    """Generic MCMC runner using emcee.

    seed: reseeds the global legacy RandomState, which emcee's
    EnsembleSampler clones at construction, so the whole run (walker ball +
    sampling) is deterministic. Callers relying on ambient global RNG state
    after this returns would see a reseeded stream.

    prune: apply stuck-walker pruning (mcmc.analysis.prune_stuck_walkers —
    walker-mean delta-lnP > 10 drop with a delta-lnP < 5 competitive-mode
    guard) to the RETURNED flat samples/logprob. Sampling itself is
    unaffected, and the saved .npz always keeps the full raw walker-resolved
    chain plus the keep_walkers mask, so pruning is auditable and reversible.
    """
    if seed is not None:
        np.random.seed(seed)
    initial = np.array(initial)
    ndim = initial.size

    pos = initial + jitter * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob_fn, args=logprob_args
    )
    sampler.run_mcmc(pos, nsteps, progress=progress)

    chain = sampler.get_chain(discard=nburn, thin=thin)
    log_prob = sampler.get_log_prob(discard=nburn, thin=thin)
    if prune:
        from mcmc.analysis import prune_stuck_walkers
        keep, prune_info = prune_stuck_walkers(chain, log_prob)
    else:
        keep = np.ones(nwalkers, dtype=bool)
        prune_info = None
    # step-major/walker-minor reshape == emcee's flat=True ordering; with an
    # all-True mask this is bit-identical to get_chain(flat=True).
    flat_samples = chain[:, keep, :].reshape(-1, ndim)
    flat_logprob = log_prob[:, keep].reshape(-1)
    if save_path:
        if param_names is None:
            param_names = [f"p{i}" for i in range(ndim)]
        save_chain(sampler, save_path, param_names, nburn, thin,
                   seed=seed, initial=initial, meta=meta,
                   keep_walkers=keep, prune_info=prune_info)
    return flat_samples, flat_logprob, sampler


def run_mcmc_ecc(t, rv, rv_err, initial, P_center, T0_center,
                 dT0_days, dP_frac=0.01,
                 nwalkers=32, nsteps=4000, nburn=100, thin=10,
                 progress=True, add_jitter=False,
                 seed=None, save_path=None, meta=None,
                 sigma_P_frac=0.002, prune=False):
    """Run eccentric orbit MCMC."""
    if add_jitter:
        initial = np.append(initial, np.log(np.median(rv_err)))

    jitter = np.array([
        1e-3 * max(initial[0], 1e-3),
        0.1, 0.05, 0.02,
        0.05 * max(initial[4], 1e-2),
        0.1
    ])
    if add_jitter:
        jitter = np.append(jitter, 0.1)

    param_names = ["P", "T0", "omega", "e", "K1", "gamma"]
    if add_jitter:
        param_names.append("log_sj")

    logprob_args = (t, rv, rv_err, P_center, T0_center, initial[4],initial[2],
                    dP_frac, dT0_days, add_jitter, sigma_P_frac)

    flat_samples, flat_logprob, sampler = run_mcmc_generic(
        initial, log_probability, logprob_args,
        jitter, nwalkers, nsteps, nburn, thin, progress,
        seed=seed, save_path=save_path, param_names=param_names, meta=meta,
        prune=prune
    )
    return flat_samples, flat_logprob, sampler


def run_mcmc_circ(t, rv, rv_err, initial, P_center, T0_center,
                  dT0_days, dP_frac=0.01,
                  nwalkers=32, nsteps=4000, nburn=100, thin=10, progress=True, add_jitter=False,
                  seed=None, save_path=None, meta=None,
                  sigma_P_frac=0.002, prune=False):
    """Run circular orbit MCMC."""
    if add_jitter:
        initial = np.append(initial, np.log(np.median(rv_err)))

    jitter = np.array([
        1e-3 * max(initial[0], 1e-3),
        0.1,
        0.05 * max(initial[2], 1e-2),
        0.1,
    ])
    if add_jitter:
        jitter = np.append(jitter, 0.1)

    param_names = ["P", "T0", "K1", "gamma"]
    if add_jitter:
        param_names.append("log_sj")

    logprob_args = (t, rv, rv_err, P_center, T0_center,
                    initial[2], dP_frac, dT0_days, add_jitter, sigma_P_frac)

    samples, logp, sampler = run_mcmc_generic(
        initial, log_probability_circ, logprob_args,
        jitter, nwalkers, nsteps, nburn, thin, progress,
        seed=seed, save_path=save_path, param_names=param_names, meta=meta,
        prune=prune
    )
    return samples, logp, sampler

def run_mcmc_null(t, rv, rv_err, initial,
                 nwalkers=32, nsteps=4000, nburn=100, thin=10, progress=True, add_jitter=False,
                 seed=None, save_path=None, meta=None, prune=False):
    """Run null (constant RV) model MCMC."""
    if add_jitter:
        initial = np.append(initial, np.log(np.median(rv_err)))

    jitter = np.array([
        0.1,
    ])
    if add_jitter:
        jitter = np.append(jitter, 0.1)

    param_names = ["gamma"]
    if add_jitter:
        param_names.append("log_sj")

    logprob_args = (t, rv, rv_err, add_jitter)

    samples, logp, sampler = run_mcmc_generic(
        initial, log_probability_null, logprob_args,
        jitter, nwalkers, nsteps, nburn, thin, progress,
        seed=seed, save_path=save_path, param_names=param_names, meta=meta,
        prune=prune
    )
    return samples, logp, sampler


def run_mcmc_double_kepler(t, rv, rv_err, initial,
                           P_in_center, T0_in_center, K1_in_center,
                           P_out_center, T0_out_center, K1_out_center,
                           dP_in_frac=0.01, dT0_in_days=2.0,
                           dP_out_frac=0.3, dT0_out_days=500.0,
                           nwalkers=64, nsteps=5000, nburn=500, thin=5,
                           progress=True,
                           seed=None, save_path=None, meta=None,
                           prune=False):
    """Run double-Keplerian (hierarchical triple) MCMC.

    initial = [P_in, T0_in, omega_in, e_in, K1_in,
               P_out, T0_out, omega_out, e_out, K1_out,
               gamma, log_sj]
    """
    initial = np.array(initial, dtype=float)

    jitter = np.array([
        1e-3 * max(initial[0], 1e-3),   # P_in
        0.1,                              # T0_in
        0.05,                             # omega_in
        0.02,                             # e_in
        0.05 * max(initial[4], 1e-2),    # K1_in
        1.0,                              # P_out
        1.0,                              # T0_out
        0.05,                             # omega_out
        0.02,                             # e_out
        0.5,                              # K1_out
        0.1,                              # gamma
        0.1,                              # log_sj
    ])

    param_names = ["P_in", "T0_in", "omega_in", "e_in", "K1_in",
                   "P_out", "T0_out", "omega_out", "e_out", "K1_out",
                   "gamma", "log_sj"]

    logprob_args = (t, rv, rv_err,
                    P_in_center, T0_in_center, K1_in_center,
                    P_out_center, T0_out_center, K1_out_center,
                    dP_in_frac, dT0_in_days,
                    dP_out_frac, dT0_out_days)

    samples, logp, sampler = run_mcmc_generic(
        initial, log_probability_double_kepler, logprob_args,
        jitter, nwalkers, nsteps, nburn, thin, progress,
        seed=seed, save_path=save_path, param_names=param_names, meta=meta,
        prune=prune
    )
    return samples, logp, sampler
