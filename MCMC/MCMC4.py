import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import sys
import os
from astropy.timeseries import LombScargle
from lmfit import Parameters, Minimizer
import pandas as pd

# ==============================
#   GLOBAL SETTINGS / MODES
# ==============================

MODE = "file"  # "simulate" or "file"
#MODE = "simulate"  # "simulate" or "file"


# Input RV file (if "file" mode)

#starname = 'BLOeM_6-067'
starname = 'BLOeM_6-032'

#INPUT_CSV = "Lee_Test/" + starname + "_CCF_RVs.csv"

INPUT_CSV = "Test_Roey/" + starname + "_CCF_RVs.csv"


# Output directory

OUTPUT_DIR = "output_" + starname
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
#  Column configuration for RV file
# ==============================

HAS_HEADER = True   # False if no header row

# Can be *either* column names (str) or indices (int)
TIME_COL  = "MJD"
RV_COL    = "merged RV"
RVERR_COL = "merged RVsig"

#TIME_COL  = 0
#RV_COL    = 21
#RVERR_COL = 22

# -----------------------------
# Binary parameters (simulation only)
# -----------------------------

P_true = 138.84          # days
T0_true = 728.4     # MJD
omega_true_deg = 182.0 # deg
e_true = 0.08
K1_true = 11.0        # km/s
gamma_true = 10.2     # km/s

# Observational parameters (simulation only)
N_obs = 25
baseline_years = 2.0
mean_sigma = 2.0
sigma_of_sigma = 0.5
sig_min = 0.1         # minimum RV error per epoch (avoid negatives)

# Period search range (in days):
Pmin, Pmax =2., 1000.0

# Plotting options
PlotTrueOrbit = False

# MCMC parameters
Nsteps_MCMC      = 2000
Nburn_MCMC       = 100
Nsteps_MCMC_circ = 2000
Nburn_MCMC_circ  = 100

# Maximum eccentricity to probe
Max_e = 0.99

# ======================================================================
# 1. Keplerian RV model helpers
# ======================================================================

def kepler_E(M, e, tol=1e-10, maxiter=100):
    M = np.asarray(M)
    E = M.copy()
    for _ in range(maxiter):
        f = E - e * np.sin(E) - M
        fprime = 1 - e * np.cos(E)
        dE = -f / fprime
        E = E + dE
        if np.all(np.abs(dE) < tol):
            break
    return E

def true_anomaly(t, P, T0, e):
    M = 2.0 * np.pi * ((t - T0) / P % 1.0)
    E = kepler_E(M, e)
    fac = np.sqrt((1 + e) / (1 - e))
    tan_nu_over2 = fac * np.tan(E / 2.0)
    nu = 2.0 * np.arctan(tan_nu_over2)
    nu = (nu + 2*np.pi) % (2*np.pi)
    return nu

def rv_model(t, P, T0, omega, e, K1, gamma):
    nu = true_anomaly(t, P, T0, e)
    return gamma + K1 * (np.cos(nu + omega) + e * np.cos(omega))


# ======================================================================
# 2. Lomb–Scargle period search
# ======================================================================

def ls_best_period(t_mjd,
                   rv_obs,
                   rv_err=None,
                   Pmin=0.5,
                   Pmax=1000.0,
                   oversampling=100,
                   plot=False):
    fmin = 1.0 / Pmax
    fmax = 1.0 / Pmin

    baseline = t_mjd.max() - t_mjd.min()
    df = 1.0 / (oversampling * baseline)
    Nf = int((fmax - fmin) / df)
    freq = fmin + df * np.arange(Nf)

    ls = LombScargle(t_mjd, rv_obs, rv_err)
    power = ls.power(freq)

    best_idx = np.argmax(power)
    best_freq = freq[best_idx]
    best_period = 1.0 / best_freq

    if plot:
        plt.figure()
        periods = 1.0 / freq
        plt.plot(periods, power, "k-")
        plt.axvline(best_period, color="C1", ls="--",
                    label=f"Best P = {best_period:.4f} d")
        plt.xscale("log")
        plt.xlabel("Period [days]")
        plt.ylabel("Lomb–Scargle power")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "periodogram.png"), dpi=200)
        plt.show()

    print("Best period found at ", best_period)
    return best_period, freq, power


# ======================================================================
# 3. lmfit initial guess (ecc + circ, unified)
# ======================================================================

def _print_lmfit_result(prefix, result, names):
    print(prefix)
    for name in names:
        par = result.params[name]
        val = par.value
        err = par.stderr
        if err is None:
            print(f"  {name} = {val:.6g}  (+/- ?)")
        else:
            print(f"  {name} = {val:.6g} +/- {err:.3g}")

def lmfit_initial_guess(t, rv, rv_err, P_guess,
                        mode="ecc", P_window_frac=0.05):
    """
    mode = "ecc":  (P, T0, omega, e, K1, gamma)
    mode = "circ": (P, T0, K1, gamma) with e=0, omega=pi/2
    """
    if mode not in ("ecc", "circ"):
        raise ValueError("mode must be 'ecc' or 'circ'")

    params = Parameters()
    Pmin = P_guess * (1.0 - P_window_frac)
    Pmax = P_guess * (1.0 + P_window_frac)

    params.add('P',  value=P_guess, min=Pmin, max=Pmax, vary=False)
    params.add('T0', value=t.min(), min=t.min() - P_guess/2.0,
               max=t.max() + P_guess/2.0, vary=True)
    params.add('K1',    value=np.std(rv)*2.0, min=0.1, max=300.0, vary=True)
    params.add('gamma', value=np.mean(rv),    min=-1e3,max=1e3,   vary=True)
    if mode == "ecc":
        params.add('omega', value=0.0, min=0.0, max=2*np.pi, vary=True)
        params.add('e',     value=0.1, min=0.0, max=0.9,   vary=True)
        def residual(params, t, rv, rv_err):
            P = params['P'].value
            T0 = params['T0'].value
            omega = params['omega'].value
            e = params['e'].value
            K1 = params['K1'].value
            gamma = params['gamma'].value
            model = rv_model(t, P, T0, omega, e, K1, gamma)
            return (rv - model) / rv_err
        names = ['P', 'T0', 'omega', 'e', 'K1', 'gamma']
    else:
        def residual(params, t, rv, rv_err):
            P = params['P'].value
            T0 = params['T0'].value
            K1 = params['K1'].value
            gamma = params['gamma'].value
            model = rv_model(t, P, T0, np.pi/2.0, 0.0, K1, gamma)
            return (rv - model) / rv_err
        names = ['P', 'T0', 'K1', 'gamma']
    minner = Minimizer(residual, params, fcn_args=(t, rv, rv_err))
    result_de = minner.minimize(method='differential_evolution')
    result    = minner.minimize(method='leastsq', params=result_de.params)

    _print_lmfit_result(
        "lmfit DE + leastsq (eccentric)" if mode == "ecc"
        else "lmfit circular (e=0, ω=90°)",
        result, names
    )

    pbest = result.params

    if mode == "ecc":
        initial = np.array([
            pbest['P'].value,
            pbest['T0'].value,
            pbest['omega'].value,
            pbest['e'].value,
            pbest['K1'].value,
            pbest['gamma'].value,
        ])
    else:
        initial = np.array([
            pbest['P'].value,
            pbest['T0'].value,
            pbest['K1'].value,
            pbest['gamma'].value,
        ])

    return initial, result


# ======================================================================
# 4. Lucy–Sweeney test
# ======================================================================

def lucy_sweeney_significant(e, sigma_e, threshold=2.45, add_jitter=False):
    if sigma_e is None or not np.isfinite(sigma_e) or sigma_e <= 0:
        print("Lucy–Sweeney: sigma_e invalid; treating e as NOT significant.")
        return False

    ratio = e / sigma_e
    print(f"Lucy–Sweeney: e = {e:.4f}, sigma_e = {sigma_e:.4f}, e/sigma_e = {ratio:.2f}")
    return ratio >= threshold


# ======================================================================
# 5. Generate mock RVs (simulation)
# ======================================================================

def generate_mock_rvs(
    N,
    P, T0, omega, e, K1, gamma,
    baseline_years,
    mean_sigma,
    sigma_of_sigma,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    MJDs = T0 + baseline_years * 365.0 * np.random.random(N)
    MJDs = np.sort(MJDs)
    rv_true = rv_model(MJDs, P, T0, omega, e, K1, gamma)

    #Sample RV sigmas from a Gaussian distribution, clipping them at sig_min
    rv_sigmas = np.random.normal(mean_sigma, sigma_of_sigma, N)
    rv_sigmas = np.clip(rv_sigmas, sig_min, None)

    noise = np.random.normal(loc=0.0, scale=rv_sigmas)
    rv_obs = rv_true + noise

    return MJDs, rv_obs, rv_true, rv_sigmas


# ======================================================================
# 6. Load RVs from CSV (data mode)
# ======================================================================

#def load_rvs_from_csv(filename):
    #"""
    #Load MJD, merged RV, merged RVsig from the BLOeM CSV.
    #Columns (0-based): 0=MJD, 21=merged RV, 22=merged RVsig
    #"""
    #data = np.loadtxt(filename, delimiter=",", skiprows=1,
                      #usecols=(0, 21, 22))
    #mjd, rv, rv_sig = data.T

    #mask = np.isfinite(rv) & np.isfinite(rv_sig)
    #mjd, rv, rv_sig = mjd[mask], rv[mask], rv_sig[mask]

    #print(f"Loaded {mjd.size} RV points from {filename}")
    #return mjd, rv, rv_sig



def load_rvs_from_csv(filename,
                      time_col=TIME_COL,
                      rv_col=RV_COL,
                      rv_err_col=RVERR_COL):
    """
    Load time, RV, RVerr from CSV.

    time_col / rv_col / rv_err_col:
      - int  → column index
      - str  → column name (if has_header=True)
    """
    header = 0 if not isinstance(time_col, int) else None
    df = pd.read_csv(filename, header=header)

    def get_col(col):
        return df[col].to_numpy() if isinstance(col, str) else df.iloc[:, col].to_numpy()

    t   = get_col(time_col)
    rv  = get_col(rv_col)
    err = get_col(rv_err_col)

    mask = np.isfinite(t) & np.isfinite(rv) & np.isfinite(err)
    t, rv, err = t[mask], rv[mask], err[mask]

    print(f"Loaded {t.size} RV points from {filename}")
    return t, rv, err


# ======================================================================
# 7. Log-likelihood & priors (eccentric + circular)
# ======================================================================

# ---------------------------
# Priors (eccentric)
# ---------------------------
def log_prior(theta, P_center, T0_center, dT0_days, K1_center, omega_center=np.pi,
              dP_frac=0.01, sigma_P_frac=0.002, add_jitter=False):
    """
    If add_jitter=False:
        theta = [P, T0, omega, e, K1, gamma]
    If add_jitter=True:
        theta = [P, T0, omega, e, K1, gamma, log_sj]
        with a weak prior on log_sj.
    """
    if add_jitter:
        P, T0, omega, e, K1, gamma, log_sj = theta
    else:
        P, T0, omega, e, K1, gamma = theta

    # Period narrow prior (Gaussian around P_center)
    Pmin = P_center * (1 - dP_frac)
    Pmax = P_center * (1 + dP_frac)
    if not (Pmin < P < Pmax):
        return -np.inf
    sigma_P = sigma_P_frac * P_center
    lp = -0.5 * ((P - P_center) / sigma_P)**2

    # Simple box priors for others
    if not (T0_center - dT0_days < T0 < T0_center + dT0_days):
        return -np.inf
    if not (0.0 <= e < Max_e):
        return -np.inf
    if not (0 < K1 <  K1_center*2):
        return -np.inf
    if not (omega_center-np.pi <= omega <= omega_center + np.pi):
        return -np.inf

    # Weak prior on log sigma_jit if enabled (Normal(0, 3))
    if add_jitter:
        lp += -0.5 * (log_sj / 3.0)**2

    return lp


def log_likelihood(theta, t, rv, rv_err, add_jitter=False):
    if add_jitter:
        P, T0, omega, e, K1, gamma, log_sj = theta
        s_jit = np.exp(log_sj)
        var = rv_err**2 + s_jit**2
    else:
        P, T0, omega, e, K1, gamma = theta
        var = rv_err**2

    model_rv = rv_model(t, P, T0, omega, e, K1, gamma)
    return -0.5 * np.sum((rv - model_rv)**2 / var + np.log(2*np.pi*var))


def log_probability(theta, t, rv, rv_err,
                    P_center, T0_center, K1_center, omega_center=np.pi,
                    dP_frac=0.1, dT0_days=2.0, add_jitter=False):
    """
    Wrapper that combines prior and likelihood and passes add_jitter through.
    """
    lp = log_prior(theta, P_center, T0_center, dT0_days, K1_center,omega_center=omega_center,
                   dP_frac=dP_frac, add_jitter=add_jitter)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, rv, rv_err, add_jitter=add_jitter)


# ---------------------------
# Priors (circular)
# ---------------------------
def log_prior_circ(theta, P_center, T0_center, dT0_days, K1_center,
                   dP_frac=0.01, sigma_P_frac=0.002, add_jitter=False):
    """
    If add_jitter=False:
        theta = [P, T0, K1, gamma]
    If add_jitter=True:
        theta = [P, T0, K1, gamma, log_sj]
    """
    if add_jitter:
        P, T0, K1, gamma, log_sj = theta
    else:
        P, T0, K1, gamma = theta

    Pmin = P_center * (1 - dP_frac)
    Pmax = P_center * (1 + dP_frac)
    if not (Pmin < P < Pmax):
        return -np.inf
    sigma_P = sigma_P_frac * P_center
    lp = -0.5 * ((P - P_center) / sigma_P)**2

    if not (T0_center - dT0_days < T0 < T0_center + dT0_days):
        return -np.inf
    if not (0 < K1 <  K1_center*2):
        return -np.inf

    if add_jitter:
        lp += -0.5 * (log_sj / 3.0)**2

    return lp

def log_prior_null(theta, add_jitter=False):
    """
    If add_jitter=False:
        theta = [P, T0, K1, gamma]
    If add_jitter=True:
        theta = [P, T0, K1, gamma, log_sj]
    """
    if add_jitter:
        gamma, log_sj = theta
    else:
        gamma = theta
    lp = 0
    if add_jitter:
        lp += -0.5 * (log_sj / 3.0)**2
    return lp

def log_likelihood_circ(theta, t, rv, rv_err, add_jitter=False):
    if add_jitter:
        P, T0, K1, gamma, log_sj = theta
        s_jit = np.exp(log_sj)
        var = rv_err**2 + s_jit**2
    else:
        P, T0, K1, gamma = theta
        var = rv_err**2

    model_rv = rv_model(t, P, T0, np.pi/2, 0.0, K1, gamma)
    return -0.5 * np.sum((rv - model_rv)**2 / var + np.log(2*np.pi*var))

def log_likelihood_null(theta, t, rv, rv_err, add_jitter=False):
    if add_jitter:
        gamma, log_sj = theta
        s_jit = np.exp(log_sj)
        var = rv_err**2 + s_jit**2
    else:
        gamma = theta
        var = rv_err**2

    model_rv = np.ones(rv.shape)*gamma
    return -0.5 * np.sum((rv - model_rv)**2 / var + np.log(2*np.pi*var))


def log_probability_circ(theta, t, rv, rv_err,
                         P_center, T0_center, K1_center,
                         dP_frac=0.01, dT0_days=2.0, add_jitter=False):
    lp = log_prior_circ(theta, P_center, T0_center, dT0_days, K1_center,
                        dP_frac=dP_frac, add_jitter=add_jitter)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_circ(theta, t, rv, rv_err, add_jitter=add_jitter)


def log_probability_null(theta, t, rv, rv_err,  add_jitter=False):
    lp = log_prior_null(theta, add_jitter=add_jitter)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_null(theta, t, rv, rv_err, add_jitter=add_jitter)

# ======================================================================
# 8. Generic MCMC runner
# ======================================================================

def run_mcmc_generic(initial, log_prob_fn, logprob_args,
                     jitter, nwalkers=1000, nsteps=4000,
                     nburn=100, thin=10, progress=True):
    initial = np.array(initial)
    ndim = initial.size

    pos = initial + jitter * np.random.randn(nwalkers, ndim)
    # cov = np.cov(pos.T)
    # print("cond =", np.linalg.cond(cov))

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
    if add_jitter:
        initial = np.append(initial, np.log(np.median(rv_err)))

    jitter = np.array([
        1e-3 * max(initial[0], 1e-3),
        0.1, 0.05, 0.02,
        0.05 * max(initial[4], 1e-2),
        0.1
    ])
    # jitter = np.array([
    #     1e-4 * max(initial[0], 1e-3),
    #     0.001, 0.005, 0.002,
    #     0.005 * max(initial[4], 1e-2),
    #     0.001
    # ])
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

# ======================================================================
# 9. Utility: summarise chain
# ======================================================================

def summarise_chain(flat_samples, names, tag, out_dir=OUTPUT_DIR, add_jitter=False):
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



# ======================================================================
# 10. Corner plot helper (with logprob filtering)
# ======================================================================

def make_corner(flat_samples, flat_logprob,
                truths, labels, tag, omega_in_col2=True,
                dlogp_clip=15.0, out_dir=OUTPUT_DIR, add_jitter=False, star_name="", show=False):
    if flat_logprob is not None:
        max_lp = np.max(flat_logprob)
        mask = flat_logprob > (max_lp - dlogp_clip)
        samples_plot = flat_samples[mask] if np.any(mask) else flat_samples.copy()
    else:
        samples_plot = flat_samples.copy()

    ndim = samples_plot.shape[1]

    truths_plot = truths
    if omega_in_col2 and ndim >= 3:
        samples_plot[:, 2] = np.degrees(samples_plot[:, 2])
        if truths is not None:
            truths_plot = list(truths)
            truths_plot[2] = np.degrees(truths_plot[2])

    fig = corner.corner(
        samples_plot,
        labels=labels,
        truths=truths_plot,
        truth_color="C0",
        show_titles=False,
        max_n_ticks=4,
        bins=30,
        smooth=1.0,
        quantiles=[0.16, 0.5, 0.84],
    )
    orbit_type = "Circular" if "circ" in tag.lower() else "Eccentric"
    fig.suptitle(f"{star_name} — {orbit_type} Orbit ({tag})", fontsize=14, weight="bold")

    axes = np.array(fig.axes).reshape(ndim, ndim)

    for i, label in enumerate(labels):
        ax = axes[i, i]
        vals = samples_plot[:, i]
        q16, q50, q84 = np.percentile(vals, [16, 50, 84])
        errm, errp = q50 - q16, q84 - q50
        err = max(errm, errp)

        if err > 0 and np.isfinite(err):
            exp = int(np.floor(np.log10(err)))
            decimals = max(0, -exp + 1)  # 2 sig figs in error
        else:
            decimals = 3

        fmt = f"{{:.{decimals}f}}"
        title = (
            f"{fmt.format(q50)}"
            f"$^{{+{fmt.format(errp)}}}_{{-{fmt.format(errm)}}}$"
        )
        ax.set_title(title, fontsize=10)

    fig.subplots_adjust(top=0.93, hspace=0.05, wspace=0.05)
    outname = os.path.join(out_dir, f"corner_{tag}.png")
    fig.savefig(outname, dpi=200, bbox_inches="tight")
    print(f"Saved {outname}")
    if show:
        plt.show()
    plt.close()


# ======================================================================
# 11. Phase plot (ecc + circ unified)
# ======================================================================

def plot_orbit_with_band_phase(t, rv, rv_err,
                               flat_samples, flat_logprob,
                               truths, tag, circular=False, out_dir=OUTPUT_DIR
                               ,add_jitter=False, star_name="", show=False):
    imax = np.argmax(flat_logprob)
    theta_map = flat_samples[imax]
    plt.figure(figsize=(10, 6))  # (width, height) in inches

    if not circular:
        P_map, T0_map = theta_map[0], theta_map[1]
    else:
        P_map, T0_map = theta_map[0], theta_map[1]

    phase_data = ((t - T0_map) / P_map) % 1.0
    phase_grid = np.linspace(0.0, 1.0, 1000, endpoint=False)

    theta_med = np.median(flat_samples, axis=0)

    if not circular:
        def model_in_phase(theta):
            if add_jitter:
                P, T0, omega, e, K1, gamma,_ = theta
            else:
                P, T0, omega, e, K1, gamma = theta
            t_grid = T0 + phase_grid * P
            return rv_model(t_grid, P, T0, omega, e, K1, gamma)
    else:
        def model_in_phase(theta):
            if add_jitter:
                P, T0, K1, gamma,_ = theta
            else:
                P, T0, K1, gamma = theta
            t_grid = T0 + phase_grid * P
            return rv_model(t_grid, P, T0, np.pi/2, 0.0, K1, gamma)

    rv_map = model_in_phase(theta_map)
    rv_med = model_in_phase(theta_med)

    rng = np.random.default_rng(123)
    ndraw = min(300, len(flat_samples))
    idx = rng.choice(len(flat_samples), size=ndraw, replace=False)

    models = np.empty((ndraw, phase_grid.size))
    for i, k in enumerate(idx):
        models[i] = model_in_phase(flat_samples[k])

    lo = np.percentile(models, 16, axis=0)
    hi = np.percentile(models, 84, axis=0)

    order = np.argsort(phase_data)

    plt.errorbar(phase_data[order], rv[order], yerr=rv_err[order],
                 fmt="o", label="data")
    plt.fill_between(phase_grid, lo, hi, alpha=0.25,
                     label="68% credible band")
    plt.plot(phase_grid, rv_map, lw=2, label="MAP orbit")
    plt.plot(phase_grid, rv_med, lw=1, ls="--", label="median orbit")

    if truths is not None:
        if not circular:
            P_true_eff, T0_true_eff, omega_true, e_true, K1_true, gamma_true = truths
            t_true_grid = T0_true_eff + phase_grid * P_true_eff
            rv_true_grid = rv_model(t_true_grid, P_true_eff, T0_true_eff,
                                    omega_true, e_true, K1_true, gamma_true)
        else:
            P_true_eff, T0_true_eff, K1_true, gamma_true = truths
            t_true_grid = T0_true_eff + phase_grid * P_true_eff
            rv_true_grid = rv_model(t_true_grid, P_true_eff, T0_true_eff,
                                    np.pi/2, 0.0, K1_true, gamma_true)
        plt.plot(phase_grid, rv_true_grid, lw=1, ls=":",
                 label="true orbit")

    orbit_type = "Circular" if circular else "Eccentric"
    plt.xlabel("Phase")
    plt.ylabel("RV [km/s]")
    plt.title(f"{star_name} — {orbit_type} Orbit ({tag})", fontsize=14, weight="bold")
    plt.legend()
    plt.tight_layout()
    outname = os.path.join(out_dir, f"phase_{tag}.png")
    plt.savefig(outname, dpi=200)
    print(f"Saved {outname}")
    if show:
        plt.show()
    plt.close()


# ======================================================================
# 12. MAIN
# ======================================================================

if __name__ == "__main__":

    if sig_min <= 0.0:
        print("sig_min must be strictly positive. Exiting....")
        sys.exit()

    have_truth = (MODE == "simulate")

    # --------------------------
    # Data: simulate or load
    # --------------------------
    if MODE == "simulate":
        if e_true == 0.0:
            print("Changing omega_true to 90 degrees (circular orbit)")
            omega_true_deg = 90.0
        omega_true = np.deg2rad(omega_true_deg)

        MJDs, rv_obs, rv_true_vals, rv_sigmas = generate_mock_rvs(
            N_obs,
            P_true, T0_true, omega_true, e_true, K1_true, gamma_true,
            baseline_years,
            mean_sigma,
            sigma_of_sigma,
        )

        if PlotTrueOrbit:
            # time domain
            plt.figure(figsize=(12, 4))
            plt.errorbar(MJDs, rv_obs, yerr=rv_sigmas, fmt="o",
                         label='sampled points')
            DenseMJDs = np.arange(MJDs[0]-1, MJDs[-1] + 1, 0.01)
            plt.plot(DenseMJDs, rv_model(DenseMJDs, P_true, T0_true,
                                         omega_true, e_true, K1_true, gamma_true),
                     color='black', label='true solution')
            plt.xlabel("Time [MJD]")
            plt.ylabel("RV [km/s]")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "true_orbit_time.png"),
                        dpi=200)
            plt.show()

            # phase domain
            plt.figure(figsize=(6, 6))
            DensePhis = np.linspace(0.0, 1.0, 1000, endpoint=False)
            PhisData = ((MJDs - T0_true) / P_true) % 1.0
            plt.errorbar(PhisData, rv_obs, yerr=rv_sigmas, fmt="o",
                         label='sampled points')
            DenseTimes = T0_true + DensePhis * P_true
            plt.plot(DensePhis,
                     rv_model(DenseTimes, P_true, T0_true,
                              omega_true, e_true, K1_true, gamma_true),
                     color='black', label='true solution')
            plt.xlabel("Phase")
            plt.ylabel("RV [km/s]")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "true_orbit_phase.png"),
                        dpi=200)
            plt.show()

    elif MODE == "file":
        print(f"Reading RVs from {INPUT_CSV}")
        MJDs, rv_obs, rv_sigmas = load_rvs_from_csv(INPUT_CSV)
        omega_true = None
    else:
        raise ValueError("MODE must be 'simulate' or 'file'")

    # --------------------------
    # LS period search
    # --------------------------
    P_guess, _, _ = ls_best_period(
        MJDs, rv_obs, rv_sigmas, Pmin=Pmin, Pmax=Pmax, plot=True
    )

    # --------------------------
    # lmfit initial guess (eccentric)
    # --------------------------
    initial_ecc, lmfit_result = lmfit_initial_guess(
        MJDs, rv_obs, rv_sigmas,
        P_guess,
        mode="ecc",
        P_window_frac=0.05,
    )

    e_lm       = lmfit_result.params['e'].value
    sigma_e_lm = lmfit_result.params['e'].stderr
    ecc_significant = lucy_sweeney_significant(e_lm, sigma_e_lm,
                                               threshold=2.45)
    print(f"Lucy–Sweeney says eccentricity significant? {ecc_significant}")

    P_center  = initial_ecc[0]
    T0_center = initial_ecc[1]

    # --------------------------
    # Eccentric MCMC
    # --------------------------
    flat_ecc, logp_ecc, sampler_ecc = run_mcmc_ecc(
        MJDs, rv_obs, rv_sigmas,
        initial_ecc,
        P_center=P_center,
        T0_center=T0_center,
        dT0_days=P_center/2.0,
        dP_frac=0.01,
        nwalkers=32,
        nsteps=Nsteps_MCMC,
        nburn=Nburn_MCMC,
        thin=10,
        progress=True,
    )

    print("Posterior samples (ecc):", flat_ecc.shape)

    # ----------------------------------------------------
    # Recenter T0 samples so they are near the true value
    # (within ~±P/2 of T0_true) for plotting purposes.
    # ----------------------------------------------------
    if have_truth:
        P_chain  = flat_ecc[:, 0]
        T0_chain = flat_ecc[:, 1]
        n_cycles_chain = np.round((T0_chain - T0_true) / P_chain)
        flat_ecc[:, 1] = T0_chain - n_cycles_chain * P_chain

        # Now the median T0 will already be close to T0_true, so
        # T0_truth_eff ≈ T0_true, but we keep your existing logic:
        P_med_ecc  = np.median(flat_ecc[:, 0])
        T0_med_ecc = np.median(flat_ecc[:, 1])
        n_cycles   = np.round((T0_med_ecc - T0_true) / P_med_ecc)
        T0_truth_eff = T0_true + n_cycles * P_med_ecc

        truths_ecc       = [P_true, T0_truth_eff,
                            omega_true, e_true, K1_true, gamma_true]
        truths_phase_ecc = truths_ecc
    else:
        truths_ecc = None
        truths_phase_ecc = None

    labels_ecc = [r"$P$ [d]", r"$T_0$ [MJD]", r"$\omega$ [deg]",
                  r"$e$", r"$K_1$ [km/s]", r"$\gamma$ [km/s]"]

    make_corner(flat_ecc, logp_ecc,
                truths_ecc, labels_ecc,
                tag="ecc", omega_in_col2=True)
    summarise_chain(flat_ecc,
                    ["P", "T0", "omega", "e", "K1", "gamma"],
                    tag="ecc")

    plot_orbit_with_band_phase(
        MJDs, rv_obs, rv_sigmas,
        flat_ecc, logp_ecc,
        truths=truths_phase_ecc,
        tag="ecc",
        circular=False,
    )

    # --------------------------
    # Circular MCMC (if e NOT significant)
    # --------------------------
    if not ecc_significant:
        print("Eccentricity not significant; running circular MCMC.")
        print("Using circular lmfit (e=0, ω=90°) for good T0.")

        initial_circ, lmfit_result_circ = lmfit_initial_guess(
            MJDs, rv_obs, rv_sigmas,
            P_guess=P_center,
            mode="circ",
            P_window_frac=0.05,
        )

        P_center_circ  = initial_circ[0]
        T0_center_circ = initial_circ[1]

        flat_circ, logp_circ, sampler_circ = run_mcmc_circ(
            MJDs, rv_obs, rv_sigmas,
            initial_circ,
            P_center=P_center_circ,
            T0_center=T0_center_circ,
            dT0_days=P_center_circ/2.0,
            dP_frac=0.01,
            nwalkers=32,
            nsteps=Nsteps_MCMC_circ,
            nburn=Nburn_MCMC_circ,
            thin=10,
            progress=True,
        )

        print("Posterior samples (circ):", flat_circ.shape)

        # Recenter T0 samples near T0_true for plotting
        if have_truth:
            P_chain_c  = flat_circ[:, 0]
            T0_chain_c = flat_circ[:, 1]
            n_cycles_chain_c = np.round((T0_chain_c - T0_true) / P_chain_c)
            flat_circ[:, 1] = T0_chain_c - n_cycles_chain_c * P_chain_c

            P_med_c  = np.median(flat_circ[:, 0])
            T0_med_c = np.median(flat_circ[:, 1])
            n_cycles_c   = np.round((T0_med_c - T0_true) / P_med_c)
            T0_truth_eff_c = T0_true + n_cycles_c * P_med_c

            truths_circ = [P_true, T0_truth_eff_c, K1_true, gamma_true]
            truths_phase_circ = truths_circ
        else:
            truths_circ = None
            truths_phase_circ = None

        labels_circ = [r"$P$ [d]", r"$T_0$ [MJD]",
                       r"$K_1$ [km/s]", r"$\gamma$ [km/s]"]

        make_corner(flat_circ, logp_circ,
                    truths_circ, labels_circ,
                    tag="circ", omega_in_col2=False)
        summarise_chain(flat_circ,
                        ["P", "T0", "K1", "gamma"],
                        tag="circ")

        plot_orbit_with_band_phase(
            MJDs, rv_obs, rv_sigmas,
            flat_circ, logp_circ,
            truths=truths_phase_circ,
            tag="circ",
            circular=True,
        )
    else:
        print("Skipping circular MCMC: Lucy–Sweeney says eccentricity is significant.")
