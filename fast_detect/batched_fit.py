"""Batched orbital fit -> BICc -> prob_bicc (detection gate #1).

Reproduces the bias-grid detection's BICc model comparison
([pipeline/evaluator.py:805-820](), [orbital/statistics.py:186]()): an orbital
fit (Period free within the periodogram peak's FWHM window) vs a constant+jitter
null, decided by ``prob_bicc = sigmoid(0.5*(null_bicc - orb_bicc))``.

Key simplifications that make it a *batched* problem (exploited, not approximated):

* The model ``v = gamma + K*(cos(nu+omega) + e*cos(omega))`` is **linear in
  (gamma, K)** for fixed (P, e, omega, T0) -> a 2x2 closed-form least-squares
  giving the residual sum of squares SSR.
* With per-star **uniform errors**, the jitter term profiles out in closed form:
  the optimal ``sig2 = err^2 + sigmaJ^2`` is ``clip(SSR/N, sig2_lo, sig2_hi)``
  (sigmaJ bounded as in the search region), so the negative-log-likelihood ``llh``
  is an explicit function of SSR. Minimising llh over (P,e,omega,T0) therefore
  reduces to **minimising SSR** -> a Keplerian template-match periodogram.

So the eccentric fit is a batched scan over a shared (e, omega, phase) template
grid (Kepler solved once per (e, phase), reused across omega), with Period taken
across the per-injection FWHM window. Reuses the numba ``kepler_E_batch`` solver.
"""

import numpy as np
from simulations.bias_grid_lib.physics import kepler_E_batch

# sigmaJ search-region bounds (ln_sigmaJ in [-2, 4.5]; see lmfit_params.search_region)
_SIGJ_LO = float(np.exp(-2.0))
_SIGJ_HI = float(np.exp(4.5))


def _ssr_linear(Y, s, SYY, oY, N):
    """Residual sum of squares of the linear fit [1, s] -> y, per injection.

    Y,s : (n_epoch, n_batch). SYY,oY : (n_batch,). Returns SSR (n_batch,).
    """
    Ss = s.sum(axis=0)
    Sss = (s * s).sum(axis=0)
    Ssy = (s * Y).sum(axis=0)
    Sy = oY
    det = N * Sss - Ss * Ss
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    # SSR = SYY - bᵀ M⁻¹ b, b=[Sy,Ssy], M=[[N,Ss],[Ss,Sss]]
    Q = (Sss * Sy * Sy - 2.0 * Ss * Sy * Ssy + N * Ssy * Ssy) / det
    return SYY - Q


# Free parameters counted in the corrected BIC, as in the production lmfit
# fit and Eq. (bicc) of the paper: the orbit has P, K1, e, omega, T0, gamma
# and the jitter (profiled here, fitted there); the constant model has gamma
# and the jitter. Until 2026-10-10 the orbit was scored with 6, which made
# the fast detector more permissive than the production classification.
N_PARAM_ORBIT = 7
N_PARAM_CONSTANT = 2


def _llh_bicc(SSR, err2, N, nvarys):
    """Profiled-jitter negative-log-likelihood and BICc, per injection.

    SSR : (n_batch,) residual sum of squares. err2 : (n_batch,) per-star err².
    With N <= nvarys + 2 the BICc is undefined and the model is never
    preferred (BICc = +inf).
    """
    sig2 = np.clip(SSR / N, err2 + _SIGJ_LO ** 2, err2 + _SIGJ_HI ** 2)
    llh = N * np.log(sig2) + SSR / sig2
    if N - nvarys - 2.0 <= 0:
        return llh, np.full_like(np.asarray(llh, dtype=float), np.inf)
    bicc = nvarys * np.log(N) * (N / (N - nvarys - 2.0)) + llh
    return llh, bicc


def fit_batch(Y, t, err, period, fwhm_lo, fwhm_hi,
              n_e=10, n_omega=12, n_phase=12, n_period=3, e_max=0.95,
              track_best=True):
    """Batched orbital fit + null -> detection-fit quantities.

    Y        : (n_epoch, n_batch) observed RV curves (host numpy, one field).
    t        : (n_epoch,) shared MJDs.
    err      : (n_batch,) per-injection RV error (scalar per star -> uniform).
    period   : (n_batch,) candidate period (periodogram peak).
    fwhm_lo/hi : (n_batch,) FWHM period bounds (Period varies within these).
    n_e,n_omega,n_phase,n_period : template-grid resolution.
    track_best : record WHICH template won (best P/e/omega + the template
               itself for the K,gamma solve). Only the Roche-lobe gate
               consumes those; prob_bicc depends on best_ssr alone, so
               callers with the gate off pass False and skip the per-template
               bookkeeping + the K,gamma solve (P/e/omega/K1/gamma become
               placeholders). best_ssr — and therefore prob_bicc — is
               bit-identical either way (masked assign vs elementwise min).

    Returns dict with prob_bicc, orb_bicc, null_bicc, and best-fit P,e,omega,K,
    gamma (for the downstream Roche-lobe / mass_flag_peri check).
    """
    Y = np.asarray(Y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n_epoch, n_batch = Y.shape
    N = float(n_epoch)
    err2 = np.asarray(err, dtype=np.float64) ** 2
    SYY = (Y * Y).sum(axis=0)
    oY = Y.sum(axis=0)

    # Period grid: n_period values spanning each injection's FWHM window.
    if n_period <= 1:
        P_levels = [period]
    else:
        fr = np.linspace(0.0, 1.0, n_period)
        P_levels = [fwhm_lo + f * (fwhm_hi - fwhm_lo) for f in fr]
    e_grid = np.linspace(0.0, e_max, n_e)
    phase_grid = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    omega_grid = np.linspace(0.0, 2.0 * np.pi, n_omega, endpoint=False)

    best_ssr = np.full(n_batch, np.inf)
    best_e = np.zeros(n_batch)
    best_w = np.zeros(n_batch)
    best_P = period.copy()
    if track_best:
        best_s = np.zeros((n_epoch, n_batch))  # template at the best combo (for K,gamma)

    for P in P_levels:
        Pc = np.clip(P, 1e-6, None)
        for e in e_grid:
            sqp, sqm = np.sqrt(1.0 + e), np.sqrt(1.0 - e)
            ecosw_grid = e * np.cos(omega_grid)
            for phi in phase_grid:
                T0 = phi * Pc                                  # (n_batch,)
                frac = ((t[None, :] - T0[:, None]) / Pc[:, None]) % 1.0  # (B,E)
                M = 2.0 * np.pi * frac
                E = kepler_E_batch(np.ascontiguousarray(M), np.full(n_batch, e))
                nu = 2.0 * np.arctan2(sqp * np.sin(E / 2.0), sqm * np.cos(E / 2.0))  # (B,E)
                for iw, w in enumerate(omega_grid):
                    s = (np.cos(nu + w) + ecosw_grid[iw]).T    # (E,B)
                    ssr = _ssr_linear(Y, s, SYY, oY, N)
                    if not track_best:
                        np.minimum(best_ssr, ssr, out=best_ssr)
                        continue
                    upd = ssr < best_ssr
                    if upd.any():
                        best_ssr[upd] = ssr[upd]
                        best_e[upd] = e
                        best_w[upd] = w
                        best_P[upd] = Pc[upd]
                        best_s[:, upd] = s[:, upd]

    if track_best:
        # K, gamma at the best template (linear solve) -> for RLOF / mass check.
        Ss = best_s.sum(axis=0); Sss = (best_s * best_s).sum(axis=0)
        Ssy = (best_s * Y).sum(axis=0)
        det = np.where(np.abs(N * Sss - Ss * Ss) < 1e-12, 1e-12, N * Sss - Ss * Ss)
        K_fit = (N * Ssy - Ss * oY) / det      # slope on s == K1
        gamma_fit = (oY - Ss * K_fit) / N
    else:
        K_fit = np.zeros(n_batch)
        gamma_fit = np.zeros(n_batch)

    ssr_null = SYY - oY * oY / N
    llh_orb, orb_bicc = _llh_bicc(best_ssr, err2, N, nvarys=N_PARAM_ORBIT)
    llh_null, null_bicc = _llh_bicc(ssr_null, err2, N, nvarys=N_PARAM_CONSTANT)
    prob_bicc = 1.0 / (1.0 + np.exp(-0.5 * (null_bicc - orb_bicc)))

    return {
        "prob_bicc": prob_bicc,
        "orb_bicc": orb_bicc,
        "null_bicc": null_bicc,
        "P": best_P,
        "e": best_e,
        "omega": best_w,
        "K1": np.abs(K_fit),
        "gamma": gamma_fit,
        "ssr_orb": best_ssr,
        "ssr_null": ssr_null,
    }
