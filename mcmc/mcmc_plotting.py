"""
mcmc.mcmc_plotting — Corner plots and phase-folded orbit bands.

Styled for Astronomy & Astrophysics full-width figures.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import corner

from mcmc.models import rv_model

# ── A&A journal house style ─────────────────────────────────────────
_AA_RC = {
    "font.family":          "serif",
    "font.serif":           ["cmr10", "Computer Modern Roman",
                             "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":     "cm",
    "axes.formatter.use_mathtext": True,
    "font.size":            16,
    "axes.labelsize":       18,
    "axes.titlesize":       18,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "legend.fontsize":      14,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "axes.linewidth":       0.7,
    "lines.linewidth":      1.4,
    "errorbar.capsize":     2,
    "savefig.dpi":          300,
}

_AA_RC_RV = {**_AA_RC,
    "font.size":            10,
    "axes.labelsize":       11,
    "axes.titlesize":       11,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      9,
}

_FULL_WIDTH = 7.09          # inches ≈ 180 mm (A&A two-column)


def _sanitize_name(name):
    """Escape chars that cmr10 cannot render (underscore, tilde)."""
    return name.replace("~", r"$\sim$").replace("_", r"$\_$")


# ─────────────────────────────────────────────────────────────────────
def make_corner(flat_samples, flat_logprob,
                truths, labels, tag, omega_in_col2=True,
                dlogp_clip=15.0, out_dir=".", add_jitter=False,
                star_name="", show=False):
    """
    Corner plot with optional logprob clipping, omega conversion to degrees,
    and formatted parameter titles.
    """
    if not out_dir and not show:
        return
    with plt.rc_context(_AA_RC):
        if flat_logprob is not None:
            max_lp = np.max(flat_logprob)
            mask = flat_logprob > (max_lp - dlogp_clip)
            samples_plot = (flat_samples[mask] if np.any(mask)
                            else flat_samples.copy())
        else:
            samples_plot = flat_samples.copy()

        # Drop jitter column (last) from corner plot
        labels = list(labels)
        if add_jitter or (labels and "jit" in labels[-1].lower()):
            samples_plot = samples_plot[:, :-1]
            if truths is not None:
                truths = list(truths)[:-1]
            labels = labels[:-1]

        ndim = samples_plot.shape[1]

        truths_plot = truths
        if omega_in_col2 and ndim >= 3:
            samples_plot[:, 2] = np.degrees(samples_plot[:, 2])
            if truths is not None:
                truths_plot = list(truths)
                truths_plot[2] = np.degrees(truths_plot[2])

        _font_kw = {"family": "serif", "fontsize": 22}
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
            label_kwargs=_font_kw,
            title_kwargs={"family": "serif", "fontsize": 16},
        )
        if "circ" in tag.lower():
            orbit_type = "Circular"
        elif "null" in tag.lower():
            orbit_type = "Null"
        else:
            orbit_type = "Eccentric"
        fig.suptitle(f"{_sanitize_name(star_name)} -- {orbit_type} Orbit",
                     fontsize=20, weight="bold", y=1.02)

        axes = np.array(fig.axes).reshape(ndim, ndim)

        for i, label in enumerate(labels):
            ax = axes[i, i]
            vals = samples_plot[:, i]
            q16, q50, q84 = np.percentile(vals, [16, 50, 84])
            errm, errp = q50 - q16, q84 - q50
            err = max(errm, errp)

            if err > 0 and np.isfinite(err):
                exp = int(np.floor(np.log10(err)))
                decimals = max(0, -exp + 1)
            else:
                decimals = 3

            fmt = f"{{:.{decimals}f}}"
            title = (
                f"{fmt.format(q50)}"
                f"$^{{+{fmt.format(errp)}}}_{{-{fmt.format(errm)}}}$"
            )
            ax.set_title(title, fontsize=20)

        fig.subplots_adjust(top=0.93, hspace=0.05, wspace=0.05)
        outname = os.path.join(out_dir, f"corner_{tag}.png")
        fig.savefig(outname, dpi=300, bbox_inches="tight")
        print(f"Saved {outname}")
        if show:
            plt.show()
        plt.close()


# ─────────────────────────────────────────────────────────────────────
def plot_orbit_with_band_phase(t, rv, rv_err,
                               flat_samples, flat_logprob,
                               truths, tag, circular=False, out_dir=".",
                               add_jitter=False, star_name="", show=False):
    """Phase-folded orbit plot with 68 % credible band and O−C residuals."""
    if not out_dir and not show:
        return
    with plt.rc_context(_AA_RC_RV):
        imax = np.argmax(flat_logprob)
        theta_map = flat_samples[imax]

        P_map, T0_map = theta_map[0], theta_map[1]
        phase_data = ((t - T0_map) / P_map) % 1.0
        phase_grid = np.linspace(0.0, 1.0, 1000, endpoint=False)

        theta_med = np.median(flat_samples, axis=0)

        # ── model helpers ────────────────────────────────────────────
        if not circular:
            def _unpack(theta):
                if add_jitter:
                    P, T0, omega, e, K1, gamma, _ = theta
                else:
                    P, T0, omega, e, K1, gamma = theta
                return P, T0, omega, e, K1, gamma

            def model_on_grid(theta):
                P, T0, omega, e, K1, gamma = _unpack(theta)
                return rv_model(T0 + phase_grid * P,
                                P, T0, omega, e, K1, gamma)

            def model_at_data(theta):
                P, T0, omega, e, K1, gamma = _unpack(theta)
                return rv_model(t, P, T0, omega, e, K1, gamma)
        else:
            def _unpack(theta):
                if add_jitter:
                    P, T0, K1, gamma, _ = theta
                else:
                    P, T0, K1, gamma = theta
                return P, T0, K1, gamma

            def model_on_grid(theta):
                P, T0, K1, gamma = _unpack(theta)
                return rv_model(T0 + phase_grid * P,
                                P, T0, np.pi / 2, 0.0, K1, gamma)

            def model_at_data(theta):
                P, T0, K1, gamma = _unpack(theta)
                return rv_model(t, P, T0, np.pi / 2, 0.0, K1, gamma)

        rv_map_grid = model_on_grid(theta_map)
        rv_med_grid = model_on_grid(theta_med)
        rv_map_data = model_at_data(theta_map)

        # ── posterior draw envelope ──────────────────────────────────
        rng = np.random.default_rng(123)
        ndraw = min(300, len(flat_samples))
        idx = rng.choice(len(flat_samples), size=ndraw, replace=False)

        models = np.empty((ndraw, phase_grid.size))
        for i, k in enumerate(idx):
            models[i] = model_on_grid(flat_samples[k])

        lo = np.percentile(models, 16, axis=0)
        hi = np.percentile(models, 84, axis=0)

        residuals = rv - rv_map_data
        order = np.argsort(phase_data)

        gamma_map = _unpack(theta_map)[-1]

        # ── figure: main + residuals (3:1) ───────────────────────────
        fig = plt.figure(figsize=(_FULL_WIDTH, _FULL_WIDTH * 0.38))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
        ax = fig.add_subplot(gs[0])
        ax_r = fig.add_subplot(gs[1], sharex=ax)

        # ── top panel: orbit ─────────────────────────────────────────
        ax.axhline(gamma_map, color="0.70", ls=":", lw=0.7, zorder=0)

        ax.fill_between(phase_grid, lo, hi, color="C3", alpha=0.20,
                        label=r"68% credible band")
        ax.plot(phase_grid, rv_map_grid, color="C3", lw=1.6,
                label="MAP orbit")
        ax.plot(phase_grid, rv_med_grid, color="C3", lw=1.0, ls="--",
                label="Median orbit")

        ax.errorbar(phase_data[order], rv[order], yerr=rv_err[order],
                    fmt="o", ms=4.5, color="C0", ecolor="C0",
                    elinewidth=0.7, capsize=2, capthick=0.7,
                    zorder=5, label="Data")

        if truths is not None:
            if not circular:
                Pt, T0t, wt, et, Kt, gt = truths
                rv_true = rv_model(T0t + phase_grid * Pt,
                                   Pt, T0t, wt, et, Kt, gt)
            else:
                Pt, T0t, Kt, gt = truths
                rv_true = rv_model(T0t + phase_grid * Pt,
                                   Pt, T0t, np.pi / 2, 0.0, Kt, gt)
            ax.plot(phase_grid, rv_true, color="C2", lw=1.2, ls=":",
                    label="True orbit")

        ax.set_title(_sanitize_name(star_name), fontsize=10, weight="bold", loc="left")
        ax.set_ylabel(r"RV (km$\,$s$^{-1}$)")
        ax.tick_params(labelbottom=False)

        # ── bottom panel: O−C ────────────────────────────────────────
        ax_r.axhline(0, color="0.50", ls="-", lw=0.7)
        ax_r.errorbar(phase_data[order], residuals[order],
                      yerr=rv_err[order],
                      fmt="o", ms=4.5, color="C0", ecolor="C0",
                      elinewidth=0.7, capsize=2, capthick=0.7, zorder=5)

        ax_r.set_xlabel("Orbital phase")
        ax_r.set_ylabel(r"O$-$C (km$\,$s$^{-1}$)")
        ax_r.set_xlim(0, 1)

        fig.align_ylabels([ax, ax_r])

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(labels), frameon=True, fancybox=False,
                   edgecolor="black", fontsize=7,
                   bbox_to_anchor=(0.5, -0.15),
                   prop={"family": "serif"})

        outname = os.path.join(out_dir, f"phase_{tag}.png")
        fig.savefig(outname, dpi=300, bbox_inches="tight")
        print(f"Saved {outname}")
        if show:
            plt.show()
        plt.close(fig)
