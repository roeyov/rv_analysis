"""
mcmc.runner — emcee MCMC wrappers for eccentric, circular, and null models.
"""

import numpy as np
import emcee

from mcmc.models import (
    log_probability,
    log_probability_circ,
    log_probability_null,
)


def run_mcmc_generic(initial, log_prob_fn, logprob_args,
                     jitter, nwalkers=1000, nsteps=4000,
                     nburn=100, thin=10, progress=True):
    """Generic MCMC runner using emcee."""
    initial = np.array(initial)
    ndim = initial.size

    pos = initial + jitter * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob_fn, args=logprob_args
    )
    sampler.run_mcmc(pos, nsteps, progress=progress)

    flat_samples = sampler.get_chain(discard=nburn, thin=thin, flat=True)
    flat_logprob = sampler.get_log_prob(discard=nburn, thin=thin, flat=True)
    return flat_samples, flat_logprob, sampler


def run_mcmc_ecc(t, rv, rv_err, initial, P_center, T0_center,
                 dT0_days, dP_frac=0.01,
                 nwalkers=32, nsteps=4000, nburn=100, thin=10,
                 progress=True, add_jitter=False):
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

    logprob_args = (t, rv, rv_err, P_center, T0_center, initial[4],initial[2],
                    dP_frac, dT0_days, add_jitter)

    flat_samples, flat_logprob, sampler = run_mcmc_generic(
        initial, log_probability, logprob_args,
        jitter, nwalkers, nsteps, nburn, thin, progress
    )
    return flat_samples, flat_logprob, sampler


def run_mcmc_circ(t, rv, rv_err, initial, P_center, T0_center,
                  dT0_days, dP_frac=0.01,
                  nwalkers=32, nsteps=4000, nburn=100, thin=10, progress=True, add_jitter=False):
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
    logprob_args = (t, rv, rv_err, P_center, T0_center,
                    initial[2], dP_frac, dT0_days, add_jitter)

    samples, logp, sampler = run_mcmc_generic(
        initial, log_probability_circ, logprob_args,
        jitter, nwalkers, nsteps, nburn, thin, progress
    )
    return samples, logp, sampler

def run_mcmc_null(t, rv, rv_err, initial,
                 nwalkers=32, nsteps=4000, nburn=100, thin=10, progress=True, add_jitter=False):
    """Run null (constant RV) model MCMC."""
    if add_jitter:
        initial = np.append(initial, np.log(np.median(rv_err)))

    jitter = np.array([
        0.1,
    ])
    if add_jitter:
        jitter = np.append(jitter, 0.1)
    logprob_args = (t, rv, rv_err, add_jitter)

    samples, logp, sampler = run_mcmc_generic(
        initial, log_probability_null, logprob_args,
        jitter, nwalkers, nsteps, nburn, thin, progress
    )
    return samples, logp, sampler
