"""
mcmc.chain_io — persistence and seeding helpers for emcee runs.

Chains are saved per (star, model-tag) as compressed .npz with the post-burn,
thinned, walker-resolved chain plus the diagnostics that cannot be recomputed
from the saved samples alone (acceptance fractions, autocorrelation times
measured on the live sampler).
"""

import json
import zlib

import numpy as np


def derive_seed(base_seed, star_name, tag):
    """
    Derive a deterministic, order-invariant seed for one (star, model) run.

    Each (star, tag) pair gets its own seed from the base seed, so a star's
    result does not depend on batch iteration order or on which other stars
    ran before it; a one-off rerun of a single star reproduces the batch
    result bit-for-bit.
    """
    return int(np.random.SeedSequence([
        int(base_seed),
        zlib.crc32(star_name.encode()),
        zlib.crc32(tag.encode()),
    ]).generate_state(1)[0])


def save_chain(sampler, path, param_names, nburn, thin,
               seed=None, initial=None, meta=None,
               keep_walkers=None, prune_info=None):
    """
    Save a sampler's post-burn thinned chain and diagnostics to *path* (.npz).

    Stored fields:
      chain               (n_kept, nwalkers, ndim)  get_chain(discard, thin)
                          — always the FULL raw chain, never pruned
      log_prob            (n_kept, nwalkers)
      param_names         (ndim,) str
      acceptance_fraction (nwalkers,)
      tau                 (ndim,)  autocorr time of the post-burn chain (tol=0)
      tau_full            (ndim,)  autocorr time of the full chain (tol=0),
                                   the convention used by mcmc_diagnostics
      initial             (ndim,)  start vector (empty array if not given)
      keep_walkers        (nwalkers,) bool — stuck-walker pruning mask used
                          for the published summaries (all True if no
                          pruning was applied); prune parameters/droppees in
                          meta_json under "prune"
      nwalkers, nsteps, nburn, thin, seed   scalars (seed = -1 if unseeded)
      meta_json           JSON string of extra metadata ("{}" if none)
    """
    chain = sampler.get_chain(discard=nburn, thin=thin)
    log_prob = sampler.get_log_prob(discard=nburn, thin=thin)
    if keep_walkers is None:
        keep_walkers = np.ones(chain.shape[1], dtype=bool)
    if prune_info is not None:
        meta = {**(meta or {}), "prune": prune_info}

    def _tau(**kwargs):
        try:
            return np.asarray(sampler.get_autocorr_time(tol=0, **kwargs))
        except Exception as exc:  # pathological chains: record NaNs, keep going
            print(f"  autocorr time failed ({exc}); storing NaNs")
            return np.full(chain.shape[2], np.nan)

    np.savez_compressed(
        path,
        chain=chain,
        log_prob=log_prob,
        param_names=np.array(list(param_names), dtype=str),
        acceptance_fraction=np.asarray(sampler.acceptance_fraction),
        tau=_tau(discard=nburn),
        tau_full=_tau(),
        initial=(np.asarray(initial, dtype=float)
                 if initial is not None else np.array([])),
        keep_walkers=np.asarray(keep_walkers, dtype=bool),
        nwalkers=sampler.nwalkers,
        nsteps=sampler.iteration,
        nburn=nburn,
        thin=thin,
        seed=(seed if seed is not None else -1),
        meta_json=json.dumps(meta or {}),
    )
    print(f"Saved chain to {path}")


def load_chain(path):
    """
    Load a chain saved by save_chain.

    Returns a dict of the stored fields plus flat views identical to emcee's
    get_chain/get_log_prob with flat=True (step-major, walker-minor):
      flat_samples (n_kept*nwalkers, ndim), flat_logprob (n_kept*nwalkers,).
    """
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files}
    out["param_names"] = [str(name) for name in out["param_names"]]
    out["meta"] = json.loads(str(out.pop("meta_json")))
    ndim = out["chain"].shape[2]
    if "keep_walkers" not in out:  # pre-pruning generations (<= mcmc_out4)
        out["keep_walkers"] = np.ones(out["chain"].shape[1], dtype=bool)
    out["flat_samples"] = out["chain"].reshape(-1, ndim)
    out["flat_logprob"] = out["log_prob"].reshape(-1)
    keep = out["keep_walkers"]
    out["flat_samples_pruned"] = out["chain"][:, keep, :].reshape(-1, ndim)
    out["flat_logprob_pruned"] = out["log_prob"][:, keep].reshape(-1)
    return out
