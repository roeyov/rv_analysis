"""
Double-Keplerian analysis for BLOeM 4-043.

1. Fit double-Keplerian model (differential_evolution)
2. Compute BICc, compare to null and single-Kepler
3. Plot time series with both orbital components
4. Run MCMC posterior sampling on 12 parameters
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from orbital.fitting import lmfit_double_kepler
from orbital.kepler import rv_double_kepler_from_times, rv_model_from_times
from mcmc.runner import run_mcmc_double_kepler
from mcmc.mcmc_plotting import make_corner, _AA_RC, _FULL_WIDTH, _sanitize_name
from mcmc.analysis import summarise_chain
from mcmc.models import rv_model_double_kepler
from utils.constants import TIME_STAMPS, RADIAL_VELS, ERRORS

# ── Data ──────────────────────────────────────────────────────────────
csv_path = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_reruns_from_coadded/BLOeM_4-043_CCF_RVs.csv"
df = pd.read_csv(csv_path)
mjds = df["MJD"].values
rvs = df["merged RV"].values
errs = df["merged RVsig"].values

data = {TIME_STAMPS: mjds, RADIAL_VELS: rvs, ERRORS: errs}
star_name = "BLOeM_4-043"
out_dir = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/dr5_reruns_from_coadded/second_pdc/BLOeM_4-043/double_kepler"
os.makedirs(out_dir, exist_ok=True)

n = len(rvs)

# ── Reference values: null model (recomputed from new data) ──
gamma_null = np.mean(rvs)
# Fit jitter analytically: minimise llh = sum log(sig2) + sum(res^2/sig2)
from scipy.optimize import minimize_scalar
def _null_llh(ln_sj):
    s2 = errs**2 + np.exp(ln_sj)**2
    return np.sum(np.log(s2)) + np.sum((rvs - gamma_null)**2 / s2)
opt = minimize_scalar(_null_llh, bounds=(-2, 5), method='bounded')
null_ln_sj = opt.x
null_s2 = errs**2 + np.exp(null_ln_sj)**2
null_llh = np.sum(np.log(null_s2)) + np.sum((rvs - gamma_null)**2 / null_s2)
null_nvarys = 2
null_aic = 2 * null_nvarys + null_llh
null_bic = null_nvarys * np.log(n) + null_llh
null_bicc = null_nvarys * np.log(n) * (n / (n - null_nvarys - 2)) + null_llh

# TODO: update single-Kepler values from new lmfit run on this dataset
single_aic = np.nan
single_bic = np.nan
single_nvarys = 7
single_llh_val = np.nan
single_bicc = np.nan

print("=" * 70)
print(f"  DOUBLE-KEPLERIAN ANALYSIS: {star_name}")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Fit double-Keplerian
# ══════════════════════════════════════════════════════════════════════
print("\n[1/4] Fitting double-Keplerian model...")

inner_params = {
    'Period': {'value': 4.4449, 'min': 3.5, 'max': 5.5, 'vary': True},
    'K1':     {'value': 83.87, 'min': 40, 'max': 150, 'vary': True},
    'omega':  {'value': 0.171, 'min': 0.0, 'max': 6.4, 'vary': True},
    'ecc':    {'value': 0.123, 'min': 0.0, 'max': 0.95, 'vary': True},
    'T0':     {'value': 60244.078, 'min': 60242, 'max': 60247, 'vary': True},
    'gamma':  {'value': 168, 'min': 50, 'max': 300, 'vary': True},
    'ln_sigmaJ': {'value': 1.0, 'min': -2, 'max': 4.5, 'vary': True},
}
outer_params = {
    'Period': {'value': 480, 'min': 100, 'max': 1500, 'vary': True},
    'K1':     {'value': 30, 'min': 1, 'max': 100, 'vary': True},
    'omega':  {'value': 3.14, 'min': 0.0, 'max': 6.4, 'vary': True},
    'ecc':    {'value': 0.3, 'min': 0.0, 'max': 0.95, 'vary': True},
    'T0':     {'value': 60400, 'min': 60242, 'max': 61742, 'vary': True},
}

result = lmfit_double_kepler(data, inner_params, outer_params)

# Extract fitted values
p = result.params
P_in  = p['Period_in'].value
T0_in = p['T0_in'].value
w_in  = p['OMEGA_in'].value
e_in  = p['Ecc_in'].value
K_in  = p['K1_in'].value
P_out  = p['Period_out'].value
T0_out = p['T0_out'].value
w_out  = p['OMEGA_out'].value
e_out  = p['Ecc_out'].value
K_out  = p['K1_out'].value
gamma  = p['GAMMA'].value
ln_sj  = p['ln_sigmaJ'].value
sigmaJ = np.exp(ln_sj)

# Compute residuals & stats
rv_mod = rv_double_kepler_from_times(mjds, P_in, T0_in, w_in, e_in, K_in,
                                      P_out, T0_out, w_out, e_out, K_out, gamma)
residuals = rvs - rv_mod
sig2 = errs**2 + sigmaJ**2
chisqr = np.sum(residuals**2 / sig2)
k = result.nvarys  # 12
redchi = chisqr / (n - k)
llh = chisqr + np.sum(np.log(sig2))

aic_dk = 2 * k + llh
bic_dk = k * np.log(n) + llh
bicc_dk = k * np.log(n) * (n / (n - k - 2)) + llh
ev_bic_dk = k * np.log(n) + llh * (1 - 1/n)

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Log results and BICc comparison
# ══════════════════════════════════════════════════════════════════════
print("\n[2/4] Results & model comparison")

print("\n--- Fitted Parameters ---")
for name, par in result.params.items():
    stderr_str = f"+/- {par.stderr:.5f}" if par.stderr is not None else "no stderr"
    print(f"  {name:15s} = {par.value:12.5f}  {stderr_str}")

print(f"\n  sigma_J = {sigmaJ:.4f} km/s")
print(f"  Residual std = {np.std(residuals):.4f} km/s")
print(f"  chi2 = {chisqr:.4f}, red_chi2 = {redchi:.4f}")
print(f"  llh = {llh:.4f}")
print(f"  AIC = {aic_dk:.2f}, BIC = {bic_dk:.2f}, BICc = {bicc_dk:.2f}")

print("\n--- Model Comparison ---")
print(f"{'Model':<20s} {'AIC':>10s} {'BIC':>10s} {'BICc':>10s} {'sigma_J':>10s}")
print("-" * 62)
print(f"{'Null (jitter)':<20s} {null_aic:10.2f} {null_bic:10.2f} {null_bicc:10.2f} {np.exp(null_ln_sj):10.3f}")
print(f"{'Single-Kepler':<20s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s}  # TODO: rerun single-Kepler")
print(f"{'Double-Kepler':<20s} {aic_dk:10.2f} {bic_dk:10.2f} {bicc_dk:10.2f} {sigmaJ:10.3f}")
print()
print(f"  Delta BICc (double vs null)   = {bicc_dk - null_bicc:.2f}")
print(f"  Delta BICc (double vs single) = {bicc_dk - single_bicc:.2f}")

# Probability from BICc
def prob_from_ic(ic_model, ic_ref):
    delta = ic_ref - ic_model
    return 1 / (1 + np.exp(-0.5 * delta))

prob_vs_null = prob_from_ic(bicc_dk, null_bicc)
prob_vs_single = prob_from_ic(bicc_dk, single_bicc)
print(f"  P(double > null | BICc)   = {prob_vs_null:.6f}")
print(f"  P(double > single | BICc) = {prob_vs_single:.6f}")

# Save results to file
results_file = os.path.join(out_dir, "double_kepler_results.txt")
with open(results_file, "w") as f:
    f.write(f"Double-Keplerian fit results for {star_name}\n")
    f.write("=" * 60 + "\n\n")
    f.write("Fitted Parameters:\n")
    for name, par in result.params.items():
        stderr_str = f"+/- {par.stderr:.6f}" if par.stderr is not None else "no stderr"
        f.write(f"  {name:15s} = {par.value:14.6f}  {stderr_str}\n")
    f.write(f"\nsigma_J = {sigmaJ:.4f} km/s\n")
    f.write(f"chi2 = {chisqr:.4f}, red_chi2 = {redchi:.4f}\n")
    f.write(f"AIC = {aic_dk:.2f}, BIC = {bic_dk:.2f}, BICc = {bicc_dk:.2f}\n")
    f.write(f"\nDelta BICc vs null   = {bicc_dk - null_bicc:.2f}\n")
    f.write(f"Delta BICc vs single = {bicc_dk - single_bicc:.2f}\n")
    f.write(f"P(double > null | BICc)   = {prob_vs_null:.6f}\n")
    f.write(f"P(double > single | BICc) = {prob_vs_single:.6f}\n")
print(f"\nSaved results to {results_file}")

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Time-series plot with both periods
# ══════════════════════════════════════════════════════════════════════
print("\n[3/4] Plotting time series...")

with plt.rc_context(_AA_RC):
    fig = plt.figure(figsize=(_FULL_WIDTH, _FULL_WIDTH * 0.55))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1], hspace=0.0)
    ax_main = fig.add_subplot(gs[0])
    ax_outer = fig.add_subplot(gs[1], sharex=ax_main)
    ax_resid = fig.add_subplot(gs[2], sharex=ax_main)

    # Dense time grid
    t_grid = np.linspace(mjds.min() - 10, mjds.max() + 10, 2000)

    # Full model
    rv_full = rv_double_kepler_from_times(t_grid, P_in, T0_in, w_in, e_in, K_in,
                                           P_out, T0_out, w_out, e_out, K_out, gamma)
    # Inner-only component (gamma_eff varies with outer orbit)
    rv_outer_only = rv_model_from_times(t_grid, P_out, T0_out, w_out, e_out, K_out, gamma)
    rv_inner_only = rv_model_from_times(t_grid, P_in, T0_in, w_in, e_in, K_in, gamma)

    # ── Top panel: data + full model + outer envelope ──
    ax_main.errorbar(mjds, rvs, yerr=errs, fmt='o', ms=4.5, color='C0',
                     ecolor='C0', elinewidth=0.7, capsize=2, capthick=0.7,
                     zorder=5, label='Data')
    ax_main.plot(t_grid, rv_full, color='C3', lw=1.4, label='Double-Kepler model', zorder=3)
    ax_main.plot(t_grid, rv_outer_only, color='C2', lw=1.2, ls='--',
                 label=f'Outer orbit ($P_{{out}}$={P_out:.0f} d)', zorder=2)
    ax_main.axhline(gamma, color='0.70', ls=':', lw=0.7, zorder=0)
    ax_main.set_ylabel(r'RV (km$\,$s$^{-1}$)')
    ax_main.tick_params(labelbottom=False)
    ax_main.legend(fontsize=7.5, loc='upper right', frameon=True, fancybox=False,
                   edgecolor='black', prop={'family': 'serif'})
    ax_main.set_title(f'{_sanitize_name(star_name)} — Double-Keplerian fit '
                      f'($P_{{in}}$={P_in:.3f} d, $P_{{out}}$={P_out:.0f} d)',
                      fontsize=11, weight='bold', loc='left')

    # ── Middle panel: outer orbit (data - inner model) ──
    rv_inner_at_data = rv_model_from_times(mjds, P_in, T0_in, w_in, e_in, K_in, gamma)
    rv_outer_data = rvs - rv_inner_at_data + gamma  # subtract inner, keep gamma
    rv_outer_grid = rv_model_from_times(t_grid, P_out, T0_out, w_out, e_out, K_out, gamma)

    ax_outer.errorbar(mjds, rv_outer_data, yerr=errs, fmt='o', ms=4, color='C2',
                      ecolor='C2', elinewidth=0.7, capsize=2, capthick=0.7, zorder=5)
    ax_outer.plot(t_grid, rv_outer_grid, color='C2', lw=1.4, zorder=3)
    ax_outer.axhline(gamma, color='0.70', ls=':', lw=0.7)
    ax_outer.set_ylabel(r'RV$-$inner (km$\,$s$^{-1}$)', fontsize=9)
    ax_outer.tick_params(labelbottom=False)

    # ── Bottom panel: residuals ──
    ax_resid.errorbar(mjds, residuals, yerr=errs, fmt='o', ms=4, color='C0',
                      ecolor='C0', elinewidth=0.7, capsize=2, capthick=0.7, zorder=5)
    ax_resid.axhline(0, color='0.50', ls='-', lw=0.7)
    ax_resid.set_xlabel('MJD')
    ax_resid.set_ylabel(r'O$-$C (km$\,$s$^{-1}$)', fontsize=9)

    fig.align_ylabels([ax_main, ax_outer, ax_resid])

    outname = os.path.join(out_dir, "timeseries_double_kepler.png")
    fig.savefig(outname, dpi=300, bbox_inches='tight')
    print(f"Saved {outname}")
    plt.close(fig)

    # ── Phase-folded on inner period ──
    fig2, (ax_ph, ax_ph_r) = plt.subplots(2, 1, figsize=(_FULL_WIDTH, _FULL_WIDTH * 0.38),
                                           gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0},
                                           sharex=True)

    # Subtract outer orbit from data
    rv_outer_at_data = rv_model_from_times(mjds, P_out, T0_out, w_out, e_out, K_out, 0.0)
    rvs_inner = rvs - rv_outer_at_data  # data with outer orbit removed

    phase_in = ((mjds - T0_in) / P_in) % 1.0
    phase_grid = np.linspace(0, 1, 500)
    t_phase = T0_in + phase_grid * P_in
    rv_phase_inner = rv_model_from_times(t_phase, P_in, T0_in, w_in, e_in, K_in, gamma)

    order = np.argsort(phase_in)
    ax_ph.errorbar(phase_in[order], rvs_inner[order], yerr=errs[order],
                   fmt='o', ms=4.5, color='C0', ecolor='C0',
                   elinewidth=0.7, capsize=2, capthick=0.7, zorder=5, label='Data (outer subtracted)')
    ax_ph.plot(phase_grid, rv_phase_inner, color='C3', lw=1.4, label=f'Inner orbit ($P$={P_in:.3f} d)')
    ax_ph.axhline(gamma, color='0.70', ls=':', lw=0.7)
    ax_ph.set_ylabel(r'RV (km$\,$s$^{-1}$)')
    ax_ph.tick_params(labelbottom=False)
    ax_ph.legend(fontsize=8, loc='upper right', prop={'family': 'serif'})
    ax_ph.set_title(f'{_sanitize_name(star_name)} — Phase-folded on inner period',
                    fontsize=11, weight='bold', loc='left')

    resid_inner = rvs_inner - rv_model_from_times(mjds, P_in, T0_in, w_in, e_in, K_in, gamma)
    ax_ph_r.errorbar(phase_in[order], resid_inner[order], yerr=errs[order],
                     fmt='o', ms=4, color='C0', ecolor='C0',
                     elinewidth=0.7, capsize=2, capthick=0.7, zorder=5)
    ax_ph_r.axhline(0, color='0.50', ls='-', lw=0.7)
    ax_ph_r.set_xlabel('Orbital phase')
    ax_ph_r.set_ylabel(r'O$-$C')
    ax_ph_r.set_xlim(0, 1)
    fig2.align_ylabels([ax_ph, ax_ph_r])

    outname2 = os.path.join(out_dir, "phase_folded_inner.png")
    fig2.savefig(outname2, dpi=300, bbox_inches='tight')
    print(f"Saved {outname2}")
    plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════
# STEP 4: MCMC
# ══════════════════════════════════════════════════════════════════════
print("\n[4/4] Running MCMC (12 parameters, this will take a while)...")

initial = np.array([P_in, T0_in, w_in, e_in, K_in,
                    P_out, T0_out, w_out, e_out, K_out,
                    gamma, ln_sj])

flat_samples, flat_logprob, sampler = run_mcmc_double_kepler(
    mjds, rvs, errs, initial,
    P_in_center=P_in, T0_in_center=T0_in, K1_in_center=K_in,
    P_out_center=P_out, T0_out_center=T0_out, K1_out_center=K_out,
    dP_in_frac=0.01, dT0_in_days=P_in / 2.0,
    dP_out_frac=0.3, dT0_out_days=P_out / 2.0,
    nwalkers=64, nsteps=5000, nburn=500, thin=5,
    progress=True
)

print(f"Posterior samples: {flat_samples.shape}")

# Corner plot
labels_dk = [r"$P_{in}$ [d]", r"$T_{0,in}$ [MJD]", r"$\omega_{in}$ [rad]",
             r"$e_{in}$", r"$K_{1,in}$ [km/s]",
             r"$P_{out}$ [d]", r"$T_{0,out}$ [MJD]", r"$\omega_{out}$ [rad]",
             r"$e_{out}$", r"$K_{1,out}$ [km/s]",
             r"$\gamma$ [km/s]", r"log $\sigma_J$"]

make_corner(flat_samples, flat_logprob,
            truths=None, labels=labels_dk,
            tag="double_kepler", omega_in_col2=False,
            out_dir=out_dir, add_jitter=False,
            star_name=star_name)

chain_names = ["P_in", "T0_in", "omega_in", "e_in", "K1_in",
               "P_out", "T0_out", "omega_out", "e_out", "K1_out",
               "gamma", "log_sj"]
summarise_chain(flat_samples, chain_names,
                tag="double_kepler", out_dir=out_dir, add_jitter=False)

# Phase-folded orbit plot with MCMC posterior band (inner orbit)
print("\nPlotting MCMC posterior orbit band...")

with plt.rc_context(_AA_RC):
    imax = np.argmax(flat_logprob)
    theta_map = flat_samples[imax]

    P_in_map, T0_in_map = theta_map[0], theta_map[1]
    phase_data = ((mjds - T0_in_map) / P_in_map) % 1.0
    phase_grid = np.linspace(0.0, 1.0, 1000, endpoint=False)

    # Subtract outer orbit from data using MAP values
    rv_outer_map = rv_model_from_times(mjds, theta_map[5], theta_map[6],
                                       theta_map[7], theta_map[8], theta_map[9], 0.0)
    rvs_inner_map = rvs - rv_outer_map

    # Posterior envelope for inner orbit
    rng = np.random.default_rng(42)
    ndraw = min(300, len(flat_samples))
    idx = rng.choice(len(flat_samples), size=ndraw, replace=False)

    models_inner = np.empty((ndraw, phase_grid.size))
    for i, k in enumerate(idx):
        s = flat_samples[k]
        t_ph = s[1] + phase_grid * s[0]  # T0_in + phase * P_in
        models_inner[i] = rv_model_from_times(t_ph, s[0], s[1], s[2], s[3], s[4], s[10])

    lo = np.percentile(models_inner, 16, axis=0)
    hi = np.percentile(models_inner, 84, axis=0)

    # MAP inner orbit curve
    t_ph_map = T0_in_map + phase_grid * P_in_map
    rv_map_inner = rv_model_from_times(t_ph_map, theta_map[0], theta_map[1],
                                        theta_map[2], theta_map[3], theta_map[4],
                                        theta_map[10])
    rv_map_data = rv_model_from_times(mjds, theta_map[0], theta_map[1],
                                       theta_map[2], theta_map[3], theta_map[4],
                                       theta_map[10])
    resid_map = rvs_inner_map - rv_map_data

    order = np.argsort(phase_data)

    fig = plt.figure(figsize=(_FULL_WIDTH, _FULL_WIDTH * 0.38))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
    ax = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax)

    ax.axhline(theta_map[10], color='0.70', ls=':', lw=0.7, zorder=0)
    ax.fill_between(phase_grid, lo, hi, color='C3', alpha=0.20, label='68% credible band')
    ax.plot(phase_grid, rv_map_inner, color='C3', lw=1.6, label='MAP orbit')
    ax.errorbar(phase_data[order], rvs_inner_map[order], yerr=errs[order],
                fmt='o', ms=4.5, color='C0', ecolor='C0',
                elinewidth=0.7, capsize=2, capthick=0.7, zorder=5, label='Data (outer subtracted)')

    ax.set_title(f'{_sanitize_name(star_name)} — Inner orbit (MCMC)',
                 fontsize=12, weight='bold', loc='left')
    ax.set_ylabel(r'RV (km$\,$s$^{-1}$)')
    ax.tick_params(labelbottom=False)

    ax_r.axhline(0, color='0.50', ls='-', lw=0.7)
    ax_r.errorbar(phase_data[order], resid_map[order], yerr=errs[order],
                  fmt='o', ms=4.5, color='C0', ecolor='C0',
                  elinewidth=0.7, capsize=2, capthick=0.7, zorder=5)
    ax_r.set_xlabel('Orbital phase')
    ax_r.set_ylabel(r'O$-$C (km$\,$s$^{-1}$)')
    ax_r.set_xlim(0, 1)

    fig.align_ylabels([ax, ax_r])
    handles, labels_leg = ax.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', ncol=len(labels_leg),
               frameon=True, fancybox=False, edgecolor='black', fontsize=8.5,
               bbox_to_anchor=(0.5, -0.15), prop={'family': 'serif'})

    outname = os.path.join(out_dir, "phase_double_kepler_mcmc.png")
    fig.savefig(outname, dpi=300, bbox_inches='tight')
    print(f"Saved {outname}")
    plt.close(fig)

print("\nDone!")
