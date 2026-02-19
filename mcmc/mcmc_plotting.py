"""
mcmc.mcmc_plotting — Corner plots and phase-folded orbit bands.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import corner

from mcmc.models import rv_model


def make_corner(flat_samples, flat_logprob,
                truths, labels, tag, omega_in_col2=True,
                dlogp_clip=15.0, out_dir=".", add_jitter=False, star_name="", show=False):
    """
    Corner plot with optional logprob clipping, omega conversion to degrees,
    and formatted parameter titles.
    """
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


def plot_orbit_with_band_phase(t, rv, rv_err,
                               flat_samples, flat_logprob,
                               truths, tag, circular=False, out_dir=".",
                               add_jitter=False, star_name="", show=False):
    """Phase-folded orbit plot with 68% credible band."""
    imax = np.argmax(flat_logprob)
    theta_map = flat_samples[imax]
    plt.figure(figsize=(10, 6))

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
