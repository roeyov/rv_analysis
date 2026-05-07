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


def run_mcmc_double_kepler(t, rv, rv_err, initial,
                           P_in_center, T0_in_center, K1_in_center,
                           P_out_center, T0_out_center, K1_out_center,
                           dP_in_frac=0.01, dT0_in_days=2.0,
                           dP_out_frac=0.3, dT0_out_days=500.0,
                           nwalkers=64, nsteps=5000, nburn=500, thin=5,
                           progress=True):
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

    logprob_args = (t, rv, rv_err,
                    P_in_center, T0_in_center, K1_in_center,
                    P_out_center, T0_out_center, K1_out_center,
                    dP_in_frac, dT0_in_days,
                    dP_out_frac, dT0_out_days)

    samples, logp, sampler = run_mcmc_generic(
        initial, log_probability_double_kepler, logprob_args,
        jitter, nwalkers, nsteps, nburn, thin, progress
    )
    return samples, logp, sampler
