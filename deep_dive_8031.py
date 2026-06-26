"""
Deep-dive analysis for BLOeM 8-031.

Diagnoses the high-eccentricity artifact (e≈0.994 at P≈259d) and finds
robust orbital parameters through six complementary analyses.

Usage:
    conda activate tau_binary
    cd /Users/roeyovadia/Roey/Masters/Reasearch/Scripts
    python deep_dive_8031.py
"""

import sys
import os
import copy
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital.fitting import chisqr_with_jitter
from orbital.kepler import nus1, v1mod
from mcmc.models import rv_model as mcmc_rv_model, log_likelihood, Max_e
from mcmc.runner import run_mcmc_generic
from mcmc.mcmc_plotting import make_corner
from mcmc.analysis import summarise_chain
from mcmc.batch import load_rv_csv
from period_search.periodogram import ls as ls_periodogram
from utils.constants import (
    TIME_STAMPS, RADIAL_VELS, ERRORS,
    PERIOD, GAMMA, K1_STR, OMEGA, ECC, T, LN_SIGMA_JITTER,
)

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_neb_div_from_coadded"
RV_PATH = os.path.join(BASE_DIR, "BLOeM_8-031_CCF_RVs.csv")
LMFIT_CSV = os.path.join(BASE_DIR, "second", "BLOeM_8-031", "lmfit_summary.csv")
OUT_DIR = os.path.join(BASE_DIR, "second", "BLOeM_8-031", "deep_dive")

# Reference pipeline solution (P≈259d, solution_id=5)
P_REF = 259.266256
CANDIDATE_PERIODS = [259.27, 131.47, 88.94, 1.005]


# ── Helpers ────────────────────────────────────────────────────────────

def make_params(P_val, e_val, fix_P=True, fix_e=True,
                gamma_bounds=(100, 250), K1_bounds=(0, 400),
                omega_bounds=(0, 2*np.pi), T0_bounds=None,
                jitter_bounds=(-2, 4.5),
                gamma_init=163., K1_init=10., omega_init=np.pi,
                T0_init=60500., jitter_init=0.5):
    """Build lmfit.Parameters for chisqr_with_jitter."""
    p = lmfit.Parameters()
    p.add(PERIOD, value=P_val, vary=not fix_P,
          min=P_val*0.9 if not fix_P else None,
          max=P_val*1.1 if not fix_P else None)
    p.add(GAMMA, value=gamma_init, min=gamma_bounds[0], max=gamma_bounds[1])
    p.add(K1_STR, value=K1_init, min=K1_bounds[0], max=K1_bounds[1])
    p.add(OMEGA, value=omega_init, min=omega_bounds[0], max=omega_bounds[1])
    p.add(ECC, value=e_val, vary=not fix_e, min=0.0, max=0.9999)
    if T0_bounds is None:
        T0_bounds = (T0_init - P_val, T0_init + P_val)
    p.add(T, value=T0_init, min=T0_bounds[0], max=T0_bounds[1])
    p.add(LN_SIGMA_JITTER, value=jitter_init, min=jitter_bounds[0], max=jitter_bounds[1])
    return p


def fit_kws(MJDs, rv_obs, rv_sigmas):
    return {TIME_STAMPS: MJDs, RADIAL_VELS: rv_obs, ERRORS: rv_sigmas}


def do_fit(params, MJDs, rv_obs, rv_sigmas, method='differential_evolution'):
    """Run lmfit minimisation and return result."""
    kws = fit_kws(MJDs, rv_obs, rv_sigmas)
    mini = lmfit.Minimizer(chisqr_with_jitter, params, fcn_kws=kws)
    return mini.minimize(method=method)


def get_cost(result):
    """Extract scalar cost from lmfit result."""
    r = result.residual
    return float(r.item()) if hasattr(r, 'item') else float(r)


def bicc(cost, nvarys, ndata):
    """Corrected Bayesian Information Criterion."""
    if ndata - nvarys - 2 <= 0:
        return np.inf
    return nvarys * np.log(ndata) * (ndata / (ndata - nvarys - 2)) + cost


def extract_fit_vals(result):
    """Extract key values from an lmfit result."""
    p = result.params
    return {
        'P': p[PERIOD].value,
        'e': p[ECC].value,
        'K1': p[K1_STR].value,
        'gamma': p[GAMMA].value,
        'omega': p[OMEGA].value,
        'T0': p[T].value,
        'ln_sj': p[LN_SIGMA_JITTER].value,
        'cost': get_cost(result),
        'nvarys': result.nvarys,
    }


# ======================================================================
# Analysis 1: Eccentricity Landscape
# ======================================================================

def analysis_1_ecc_landscape(MJDs, rv_obs, rv_sigmas, out_dir):
    """Fix P, scan e grid, fit remaining params."""
    print("\n" + "="*60)
    print("Analysis 1: Eccentricity Landscape (P={:.1f}d)".format(P_REF))
    print("="*60)

    N = len(MJDs)
    e_grid = np.arange(0.0, 0.96, 0.05)
    results = []

    for e_val in e_grid:
        params = make_params(P_REF, e_val, fix_P=True, fix_e=True,
                             T0_init=np.median(MJDs))
        res = do_fit(params, MJDs, rv_obs, rv_sigmas)
        cost = get_cost(res)
        bic_val = bicc(cost, res.nvarys, N)
        K1_fit = res.params[K1_STR].value
        gamma_fit = res.params[GAMMA].value
        results.append({'e': e_val, 'cost': cost, 'bicc': bic_val,
                        'K1': K1_fit, 'gamma': gamma_fit, 'nvarys': res.nvarys})
        print(f"  e={e_val:.2f}  cost={cost:.1f}  BICc={bic_val:.1f}  "
              f"K1={K1_fit:.2f}  gamma={gamma_fit:.2f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "ecc_landscape.csv"), index=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(df['e'], df['cost'], 'o-', color='C0')
    axes[0].set_xlabel('Eccentricity'); axes[0].set_ylabel('Cost (neg-2logL)')
    axes[0].set_title('Cost vs Eccentricity')

    axes[1].plot(df['e'], df['bicc'], 's-', color='C1')
    axes[1].set_xlabel('Eccentricity'); axes[1].set_ylabel('BICc')
    axes[1].set_title('BICc vs Eccentricity')
    i_best = df['bicc'].idxmin()
    axes[1].axvline(df.loc[i_best, 'e'], color='red', ls='--', alpha=0.7,
                    label=f"best e={df.loc[i_best, 'e']:.2f}")
    axes[1].legend()

    axes[2].plot(df['e'], df['K1'], '^-', color='C2')
    axes[2].set_xlabel('Eccentricity'); axes[2].set_ylabel('K1 [km/s]')
    axes[2].set_title('Fitted K1 vs Eccentricity')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ecc_landscape.png"), dpi=150)
    plt.close(fig)

    best = df.loc[i_best]
    print(f"\n  Best BICc at e={best['e']:.2f}: BICc={best['bicc']:.1f}, "
          f"K1={best['K1']:.2f} km/s")
    return df


# ======================================================================
# Analysis 2: Circular Fits at Multiple Periods
# ======================================================================

def analysis_2_circular_fits(MJDs, rv_obs, rv_sigmas, out_dir):
    """Force circular orbit (e=0) at each candidate period."""
    print("\n" + "="*60)
    print("Analysis 2: Circular Fit Comparison")
    print("="*60)

    N = len(MJDs)
    results = []

    for P_cand in CANDIDATE_PERIODS:
        params = make_params(P_cand, 0.0, fix_P=False, fix_e=True,
                             T0_init=np.median(MJDs))
        # Fix omega to pi/2 for circular
        params[OMEGA].set(value=np.pi/2, vary=False)
        res = do_fit(params, MJDs, rv_obs, rv_sigmas)
        cost = get_cost(res)
        bic_val = bicc(cost, res.nvarys, N)
        vals = {
            'P_cand': P_cand,
            'P_fit': res.params[PERIOD].value,
            'K1': res.params[K1_STR].value,
            'gamma': res.params[GAMMA].value,
            'T0': res.params[T].value,
            'ln_sj': res.params[LN_SIGMA_JITTER].value,
            'cost': cost,
            'bicc': bic_val,
        }
        results.append(vals)
        print(f"  P_cand={P_cand:.1f}d -> P_fit={vals['P_fit']:.2f}d  "
              f"K1={vals['K1']:.2f}  BICc={bic_val:.1f}  "
              f"jitter={np.exp(vals['ln_sj']):.2f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "circular_fits.csv"), index=False)

    # Phase-folded plot for best circular fit
    best = df.loc[df['bicc'].idxmin()]
    P_best = best['P_fit']
    phase = ((MJDs - best['T0']) / P_best) % 1.0
    phase_grid = np.linspace(0, 1, 500)
    t_grid = best['T0'] + phase_grid * P_best
    rv_model_grid = mcmc_rv_model(t_grid, P_best, best['T0'],
                                   np.pi/2, 0.0, best['K1'], best['gamma'])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(phase, rv_obs, yerr=rv_sigmas, fmt='o', color='k',
                ecolor='gray', capsize=2, label='Data')
    ax.plot(phase_grid, rv_model_grid, 'r-', lw=2,
            label=f'Circular P={P_best:.1f}d, K1={best["K1"]:.1f} km/s')
    ax.set_xlabel('Phase'); ax.set_ylabel('RV [km/s]')
    ax.set_title(f'Best Circular Fit (BICc={best["bicc"]:.1f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "circular_best_phase.png"), dpi=150)
    plt.close(fig)

    print(f"\n  Best circular: P={P_best:.2f}d, K1={best['K1']:.2f}, "
          f"BICc={best['bicc']:.1f}")
    return df


# ======================================================================
# Analysis 3: Bootstrap
# ======================================================================

def _bootstrap_worker(seed, MJDs, rv_obs, rv_sigmas, P_ref):
    """Single bootstrap iteration: perturb RVs, refit."""
    rng = np.random.RandomState(seed)
    rv_pert = rv_obs + rng.randn(len(rv_obs)) * rv_sigmas

    params = make_params(P_ref, 0.5, fix_P=True, fix_e=False,
                         T0_init=np.median(MJDs))
    kws = {TIME_STAMPS: MJDs, RADIAL_VELS: rv_pert, ERRORS: rv_sigmas}
    mini = lmfit.Minimizer(chisqr_with_jitter, params, fcn_kws=kws)
    try:
        res = mini.minimize(method='differential_evolution')
        cost = get_cost(res)
        return {
            'e': res.params[ECC].value,
            'K1': res.params[K1_STR].value,
            'gamma': res.params[GAMMA].value,
            'omega': res.params[OMEGA].value,
            'cost': cost,
        }
    except Exception:
        return None


def analysis_3_bootstrap(MJDs, rv_obs, rv_sigmas, out_dir, n_boot=200):
    """Perturb RVs within errors, refit, get distribution of e and K1."""
    print("\n" + "="*60)
    print(f"Analysis 3: RV Bootstrap ({n_boot} realizations)")
    print("="*60)

    n_workers = max(1, mp.cpu_count() - 1)
    worker_fn = partial(_bootstrap_worker,
                        MJDs=MJDs, rv_obs=rv_obs, rv_sigmas=rv_sigmas,
                        P_ref=P_REF)

    print(f"  Running {n_boot} bootstrap fits with {n_workers} workers...")
    with mp.Pool(n_workers) as pool:
        raw = pool.map(worker_fn, range(n_boot))

    results = [r for r in raw if r is not None]
    print(f"  {len(results)}/{n_boot} fits converged")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "bootstrap_results.csv"), index=False)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df['e'], bins=30, edgecolor='k', alpha=0.7)
    axes[0].axvline(0.994, color='red', ls='--', label='Pipeline e=0.994')
    axes[0].set_xlabel('Eccentricity'); axes[0].set_ylabel('Count')
    axes[0].set_title('Bootstrap Eccentricity Distribution')
    axes[0].legend()

    sc = axes[1].scatter(df['e'], df['K1'], c=df['cost'], cmap='viridis_r',
                         s=15, alpha=0.7)
    plt.colorbar(sc, ax=axes[1], label='Cost')
    axes[1].set_xlabel('Eccentricity'); axes[1].set_ylabel('K1 [km/s]')
    axes[1].set_title('Bootstrap K1 vs e')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bootstrap.png"), dpi=150)
    plt.close(fig)

    e_med = np.median(df['e'])
    e_lo, e_hi = np.percentile(df['e'], [16, 84])
    print(f"  e: median={e_med:.3f}, 16-84%=[{e_lo:.3f}, {e_hi:.3f}]")
    print(f"  K1: median={np.median(df['K1']):.2f} km/s")
    return df


# ======================================================================
# Analysis 4: Jackknife
# ======================================================================

def analysis_4_jackknife(MJDs, rv_obs, rv_sigmas, out_dir):
    """Leave-one-out: remove each epoch, refit, track e sensitivity."""
    print("\n" + "="*60)
    print("Analysis 4: Leave-One-Out Jackknife (28 fits)")
    print("="*60)

    N = len(MJDs)
    results = []

    for i in range(N):
        mask = np.ones(N, bool)
        mask[i] = False
        params = make_params(P_REF, 0.5, fix_P=True, fix_e=False,
                             T0_init=np.median(MJDs))
        res = do_fit(params, MJDs[mask], rv_obs[mask], rv_sigmas[mask])
        cost = get_cost(res)
        results.append({
            'removed_idx': i,
            'removed_MJD': MJDs[i],
            'removed_RV': rv_obs[i],
            'e': res.params[ECC].value,
            'K1': res.params[K1_STR].value,
            'cost': cost,
        })
        print(f"  Remove epoch {i:2d} (MJD={MJDs[i]:.1f}, RV={rv_obs[i]:.1f}): "
              f"e={res.params[ECC].value:.4f}  K1={res.params[K1_STR].value:.2f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "jackknife_results.csv"), index=False)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(range(N), df['e'], color='steelblue', edgecolor='k', height=0.7)
    axes[0].axvline(0.994, color='red', ls='--', lw=1.5, label='Pipeline e=0.994')
    axes[0].set_ylabel('Removed epoch index')
    axes[0].set_xlabel('Fitted Eccentricity')
    axes[0].set_title('Jackknife: e per removed epoch')
    axes[0].legend()

    axes[1].scatter(df['removed_RV'], df['e'], c='steelblue', edgecolors='k', s=50)
    axes[1].set_xlabel('Removed RV [km/s]')
    axes[1].set_ylabel('Fitted e (without that point)')
    axes[1].set_title('Influence of each RV on eccentricity')
    axes[1].axhline(0.994, color='red', ls='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "jackknife.png"), dpi=150)
    plt.close(fig)

    most_influential = df.loc[(df['e'] - 0.994).abs().idxmax()]
    print(f"\n  Most influential: epoch {int(most_influential['removed_idx'])} "
          f"(MJD={most_influential['removed_MJD']:.1f}, "
          f"RV={most_influential['removed_RV']:.1f}), "
          f"e drops to {most_influential['e']:.4f}")
    return df


# ======================================================================
# Analysis 5: Residual Periodogram
# ======================================================================

def analysis_5_residual_periodogram(MJDs, rv_obs, rv_sigmas, circ_df, out_dir):
    """LS periodogram on residuals from best circular fit."""
    print("\n" + "="*60)
    print("Analysis 5: Residual Periodogram")
    print("="*60)

    best_circ = circ_df.loc[circ_df['bicc'].idxmin()]
    P_c, T0_c = best_circ['P_fit'], best_circ['T0']
    K1_c, gamma_c = best_circ['K1'], best_circ['gamma']

    rv_model_circ = mcmc_rv_model(MJDs, P_c, T0_c, np.pi/2, 0.0, K1_c, gamma_c)
    resid_circ = rv_obs - rv_model_circ

    # Also compute residuals from pipeline eccentric solution
    lmfit_df = pd.read_csv(LMFIT_CSV)
    ecc_row = lmfit_df[lmfit_df['solution_id'] == 5].iloc[0]  # P≈259d PDC_jitter
    rv_model_ecc = mcmc_rv_model(MJDs,
                                  ecc_row['Period_value'], ecc_row['T0_value'],
                                  ecc_row['OMEGA_rad_value'],
                                  ecc_row['Eccentricity_value'],
                                  ecc_row['K1_value'], ecc_row['GAMMA_value'])
    resid_ecc = rv_obs - rv_model_ecc

    # LS periodograms
    pmin, pmax = 0.4, 5000.
    bp_c, _, fap_c, fal_c, freq_c, pow_c, fapvec_c = ls_periodogram(
        MJDs, resid_circ, data_err=rv_sigmas, pmin=pmin, pmax=pmax)
    bp_e, _, fap_e, fal_e, freq_e, pow_e, fapvec_e = ls_periodogram(
        MJDs, resid_ecc, data_err=rv_sigmas, pmin=pmin, pmax=pmax)

    print(f"  Circular residuals: best_period={bp_c:.2f}d, FAP={fap_c:.4f}")
    print(f"  Eccentric residuals: best_period={bp_e:.2f}d, FAP={fap_e:.4f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    per_c = 1.0 / freq_c
    axes[0].plot(per_c, pow_c, 'k-', lw=0.5)
    axes[0].set_ylabel('LS Power')
    axes[0].set_title(f'Residuals from Circular Fit (P={P_c:.1f}d)')
    if fal_c is not None and len(fal_c) >= 3:
        for lvl, ls_style, lbl in zip(fal_c, [':', '--', '-'],
                                       ['50%', '1%', '0.1%']):
            axes[0].axhline(lvl, color='red', ls=ls_style, alpha=0.5, label=lbl)
        axes[0].legend(fontsize=8)
    axes[0].axvline(bp_c, color='blue', ls='--', alpha=0.4,
                    label=f'Peak {bp_c:.1f}d')

    per_e = 1.0 / freq_e
    axes[1].plot(per_e, pow_e, 'k-', lw=0.5)
    axes[1].set_ylabel('LS Power')
    axes[1].set_xlabel('Period [days]')
    axes[1].set_title(f'Residuals from Eccentric Fit (P={ecc_row["Period_value"]:.1f}d, e={ecc_row["Eccentricity_value"]:.3f})')
    axes[1].set_xscale('log')
    axes[1].axvline(bp_e, color='blue', ls='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "residual_periodograms.png"), dpi=150)
    plt.close(fig)

    return {'circ_best_period': bp_c, 'circ_fap': fap_c,
            'ecc_best_period': bp_e, 'ecc_fap': fap_e}


# ======================================================================
# Analysis 6: MCMC with Beta Prior on Eccentricity
# ======================================================================

def log_prior_beta_ecc(theta, P_center, T0_center, dT0_days, K1_center,
                       dP_frac=0.01, sigma_P_frac=0.002,
                       beta_a=0.867, beta_b=3.03):
    """
    Eccentric prior with Beta(a,b) on eccentricity (Kipping 2013).
    theta = [P, T0, omega, e, K1, gamma, log_sj]
    """
    P, T0, omega, e, K1, gamma, log_sj = theta

    Pmin = P_center * (1 - dP_frac)
    Pmax = P_center * (1 + dP_frac)
    if not (Pmin < P < Pmax):
        return -np.inf
    sigma_P = sigma_P_frac * P_center
    lp = -0.5 * ((P - P_center) / sigma_P)**2

    if not (T0_center - dT0_days < T0 < T0_center + dT0_days):
        return -np.inf
    if not (0.001 < e < 0.99):
        return -np.inf
    if not (0 < K1 < K1_center * 2):
        return -np.inf
    if not (0 <= omega <= 2 * np.pi):
        return -np.inf

    # Beta prior on eccentricity
    lp += sp_stats.beta.logpdf(e, beta_a, beta_b)

    # Weak prior on jitter
    lp += -0.5 * (log_sj / 3.0)**2

    return lp


def log_prob_beta(theta, t, rv, rv_err,
                  P_center, T0_center, K1_center,
                  dP_frac, dT0_days, beta_a, beta_b):
    """Log-probability with Beta eccentricity prior."""
    lp = log_prior_beta_ecc(theta, P_center, T0_center, dT0_days, K1_center,
                            dP_frac=dP_frac, beta_a=beta_a, beta_b=beta_b)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, rv, rv_err, add_jitter=True)


def analysis_6_mcmc_beta(MJDs, rv_obs, rv_sigmas, ecc_landscape_df, out_dir):
    """MCMC with Beta prior on eccentricity."""
    print("\n" + "="*60)
    print("Analysis 6: MCMC with Beta(0.867, 3.03) Eccentricity Prior")
    print("="*60)

    # Initialize from eccentricity landscape minimum
    best_row = ecc_landscape_df.loc[ecc_landscape_df['bicc'].idxmin()]
    e_init = max(best_row['e'], 0.05)  # avoid e=0 boundary

    # Do a quick lmfit at the landscape-best e to get good T0/omega
    params = make_params(P_REF, e_init, fix_P=True, fix_e=True,
                         T0_init=np.median(MJDs))
    res = do_fit(params, MJDs, rv_obs, rv_sigmas)
    p = res.params

    initial = np.array([
        P_REF,
        p[T].value,
        p[OMEGA].value,
        e_init,
        p[K1_STR].value,
        p[GAMMA].value,
        np.log(np.median(rv_sigmas)),  # log_sj
    ])

    K1_center = max(p[K1_STR].value, 5.0)
    T0_center = p[T].value
    dT0_days = P_REF / 2.0

    jitter_arr = np.array([
        1e-3 * P_REF,   # P
        0.5,             # T0
        0.1,             # omega
        0.05,            # e
        0.5,             # K1
        0.2,             # gamma
        0.1,             # log_sj
    ])

    logprob_args = (MJDs, rv_obs, rv_sigmas,
                    P_REF, T0_center, K1_center,
                    0.01, dT0_days, 0.867, 3.03)

    nwalkers, nsteps, nburn, thin = 32, 5000, 1000, 5
    print(f"  Initial: P={initial[0]:.1f}, T0={initial[1]:.1f}, "
          f"omega={initial[2]:.2f}, e={initial[3]:.3f}, "
          f"K1={initial[4]:.2f}, gamma={initial[5]:.2f}")
    print(f"  Running emcee: {nwalkers} walkers, {nsteps} steps...")

    flat_samples, flat_logprob, sampler = run_mcmc_generic(
        initial, log_prob_beta, logprob_args,
        jitter_arr, nwalkers=nwalkers, nsteps=nsteps,
        nburn=nburn, thin=thin, progress=True,
    )
    print(f"  Posterior samples: {flat_samples.shape}")

    # Corner plot
    labels = [r"$P$ [d]", r"$T_0$", r"$\omega$ [rad]",
              r"$e$", r"$K_1$ [km/s]", r"$\gamma$ [km/s]", r"log $\sigma_J$"]
    make_corner(flat_samples, flat_logprob, None, labels, tag="beta_prior",
                omega_in_col2=True, out_dir=out_dir, star_name="BLOeM_8-031")

    chain_names = ["P", "T0", "omega", "e", "K1", "gamma", "log_sj"]
    dict_row = summarise_chain(flat_samples, chain_names, tag="beta_prior",
                               out_dir=out_dir, add_jitter=True)

    # Save chain for later reuse
    np.savez(os.path.join(out_dir, "mcmc_beta_chain.npz"),
             flat_samples=flat_samples, flat_logprob=flat_logprob)

    # Phase-folded orbit with 68% credible band
    from mcmc.mcmc_plotting import plot_orbit_with_band_phase
    plot_orbit_with_band_phase(
        MJDs, rv_obs, rv_sigmas, flat_samples, flat_logprob,
        truths=None, tag="beta_prior", circular=False,
        out_dir=out_dir, add_jitter=True, star_name="BLOeM_8-031",
    )

    # Eccentricity posterior plot with prior overlay
    e_samples = flat_samples[:, 3]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(e_samples, bins=50, density=True, alpha=0.7, color='steelblue',
            edgecolor='k', label='Posterior')
    e_plot = np.linspace(0.001, 0.99, 200)
    ax.plot(e_plot, sp_stats.beta.pdf(e_plot, 0.867, 3.03), 'r-', lw=2,
            label='Beta(0.867, 3.03) prior')
    ax.set_xlabel('Eccentricity')
    ax.set_ylabel('Density')
    ax.set_title('Eccentricity Posterior (Beta Prior)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mcmc_ecc_posterior.png"), dpi=150)
    plt.close(fig)

    e_mode = dict_row.get("beta_prior_e_mode", np.nan)
    e_med = np.median(e_samples)
    e_lo, e_hi = np.percentile(e_samples, [16, 84])
    K1_med = np.median(flat_samples[:, 4])
    K1_lo, K1_hi = np.percentile(flat_samples[:, 4], [16, 84])

    print(f"  e: mode={e_mode:.3f}, median={e_med:.3f}, "
          f"16-84%=[{e_lo:.3f}, {e_hi:.3f}]")
    print(f"  K1: median={K1_med:.2f}, 16-84%=[{K1_lo:.2f}, {K1_hi:.2f}]")

    return dict_row, flat_samples


# ======================================================================
# Summary
# ======================================================================

def print_summary(ecc_df, circ_df, boot_df, jack_df, resid_info, mcmc_row,
                  mcmc_samples, out_dir):
    """Print and save final comparison table."""
    print("\n" + "="*60)
    print("SUMMARY: BLOeM 8-031 Deep Dive")
    print("="*60)

    lines = []
    def p(s):
        print(s)
        lines.append(s)

    # Pipeline
    p(f"Pipeline best (P≈259d):  e=0.994, K1=92.3 km/s, BICc=123.8")

    # Ecc landscape
    best_ecc = ecc_df.loc[ecc_df['bicc'].idxmin()]
    p(f"Ecc landscape minimum:   e={best_ecc['e']:.2f}, "
      f"K1={best_ecc['K1']:.2f} km/s, BICc={best_ecc['bicc']:.1f}")

    # Circular
    best_circ = circ_df.loc[circ_df['bicc'].idxmin()]
    p(f"Best circular fit:       P={best_circ['P_fit']:.1f}d, "
      f"K1={best_circ['K1']:.2f} km/s, BICc={best_circ['bicc']:.1f}")

    # Bootstrap
    e_boot = boot_df['e']
    p(f"Bootstrap e:             median={e_boot.median():.3f}, "
      f"16-84%=[{e_boot.quantile(0.16):.3f}, {e_boot.quantile(0.84):.3f}]")

    # Jackknife
    e_range = jack_df['e']
    p(f"Jackknife e range:       [{e_range.min():.3f}, {e_range.max():.3f}]")
    most_inf = jack_df.loc[(jack_df['e'] - 0.994).abs().idxmax()]
    p(f"  Most influential:      epoch {int(most_inf['removed_idx'])} "
      f"(MJD={most_inf['removed_MJD']:.1f})")

    # Residual periodogram
    p(f"Residual periodogram:    circ peak={resid_info['circ_best_period']:.1f}d "
      f"(FAP={resid_info['circ_fap']:.4f})")

    # MCMC
    e_samp = mcmc_samples[:, 3]
    K1_samp = mcmc_samples[:, 4]
    p(f"MCMC (Beta prior):       e={np.median(e_samp):.3f} "
      f"[{np.percentile(e_samp, 16):.3f}, {np.percentile(e_samp, 84):.3f}], "
      f"K1={np.median(K1_samp):.2f} "
      f"[{np.percentile(K1_samp, 16):.2f}, {np.percentile(K1_samp, 84):.2f}]")

    with open(os.path.join(out_dir, "summary.txt"), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSummary saved to {out_dir}/summary.txt")


# ======================================================================
# Main
# ======================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    MJDs, rv_obs, rv_sigmas = load_rv_csv(RV_PATH)
    print(f"  N={len(MJDs)}, MJD range: {MJDs.min():.1f}-{MJDs.max():.1f} "
          f"({MJDs.max()-MJDs.min():.0f} days)")
    print(f"  RV range: {rv_obs.min():.2f}-{rv_obs.max():.2f} km/s "
          f"(Delta={rv_obs.max()-rv_obs.min():.2f})")

    ecc_df = analysis_1_ecc_landscape(MJDs, rv_obs, rv_sigmas, OUT_DIR)
    circ_df = analysis_2_circular_fits(MJDs, rv_obs, rv_sigmas, OUT_DIR)
    boot_df = analysis_3_bootstrap(MJDs, rv_obs, rv_sigmas, OUT_DIR, n_boot=200)
    jack_df = analysis_4_jackknife(MJDs, rv_obs, rv_sigmas, OUT_DIR)
    resid_info = analysis_5_residual_periodogram(MJDs, rv_obs, rv_sigmas,
                                                  circ_df, OUT_DIR)
    mcmc_row, mcmc_samples = analysis_6_mcmc_beta(MJDs, rv_obs, rv_sigmas,
                                                   ecc_df, OUT_DIR)

    print_summary(ecc_df, circ_df, boot_df, jack_df, resid_info,
                  mcmc_row, mcmc_samples, OUT_DIR)

    print("\nDone! All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
