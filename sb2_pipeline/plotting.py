"""
sb2_pipeline.plotting — SB2 figures: per-candidate lmfit phase plot, corner,
two-component phase-folded credible band, and walker trace plots.

All matplotlib (Agg-safe, A&A rcParams) — no kaleido anywhere in the SB2
candidate loop.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import corner

from utils.constants import (
    PERIOD, GAMMA, K1_STR, K2_STR, OMEGA, ECC, T,
    LN_SIGMA_JITTER, LN_SIGMA_JITTER_1, LN_SIGMA_JITTER_2,
)
from mcmc.mcmc_plotting import _AA_RC, _AA_RC_RV, _FULL_WIDTH, _sanitize_name
from mcmc.chain_io import load_chain

from sb2_pipeline.constants import (
    SB2_TS1, SB2_TS2, SB2_RV1, SB2_ERR1, SB2_RV2, SB2_ERR2,
)
from sb2_pipeline.model import sb2_rv_model

# Phase-figure height/width ratio of the FIGURE SPEC (cf. mcmc_plotting
# _PHASE_ASPECT): the saved PNG comes out taller than this because the
# below-axes legend is included in the crop. 0.30 lands the cropped PNG
# near 0.37, so three SB2 phase figures stack per A&A page in the paper's
# orbit atlas (was 0.42 -> PNG 0.51-0.58, two per page; 2026-09-23).
_SB2_PHASE_ASPECT = 0.30
from sb2_pipeline.mcmc_model import sb2_rv_model_mcmc

_C1, _C2 = "C0", "C3"          # component 1 (blue) / component 2 (red)


# ---------------------------------------------------------------------------
# lmfit per-candidate figure
# ---------------------------------------------------------------------------

def plot_sb2_lmfit_solution(data, result, args_dict, star_name, solution_id,
                            out_dir):
    """
    Phase-folded plot of both components with the joint lmfit model and a
    shared residual panel. Also writes the lmfit fit report. Returns the
    figure path ("" when out_dir is falsy).
    """
    if not out_dir:
        return ""
    sol_dir = os.path.join(out_dir, "lmfit_solutions")
    os.makedirs(sol_dir, exist_ok=True)

    p = result.params
    P = float(p[PERIOD].value)
    T0 = float(p[T].value)
    omega = float(p[OMEGA].value)
    ecc = float(p[ECC].value)
    K1 = float(p[K1_STR].value)
    K2 = float(p[K2_STR].value)
    gamma = float(p[GAMMA].value)
    if LN_SIGMA_JITTER in p:            # shared-jitter mode
        sj1 = sj2 = float(np.exp(p[LN_SIGMA_JITTER].value))
    else:
        sj1 = float(np.exp(p[LN_SIGMA_JITTER_1].value))
        sj2 = float(np.exp(p[LN_SIGMA_JITTER_2].value))

    t1 = np.asarray(data[SB2_TS1], float)
    rv1 = np.asarray(data[SB2_RV1], float)
    err1 = np.asarray(data[SB2_ERR1], float)
    t2 = np.asarray(data[SB2_TS2], float)
    rv2 = np.asarray(data[SB2_RV2], float)
    err2 = np.asarray(data[SB2_ERR2], float)

    phase1 = ((t1 - T0) / P) % 1.0
    phase2 = ((t2 - T0) / P) % 1.0
    grid = np.linspace(0.0, 1.0, 1000, endpoint=False)
    g1, g2 = sb2_rv_model(T0 + grid * P, P, T0, omega, ecc, K1, K2, gamma)
    m1, _ = sb2_rv_model(t1, P, T0, omega, ecc, K1, K2, gamma)
    _, m2 = sb2_rv_model(t2, P, T0, omega, ecc, K1, K2, gamma)
    err1_tot = np.sqrt(err1 ** 2 + sj1 ** 2)
    err2_tot = np.sqrt(err2 ** 2 + sj2 ** 2)

    with plt.rc_context(_AA_RC_RV):
        fig = plt.figure(figsize=(_FULL_WIDTH, _FULL_WIDTH * 0.45))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
        ax = fig.add_subplot(gs[0])
        ax_r = fig.add_subplot(gs[1], sharex=ax)

        ax.axhline(gamma, color="0.70", ls=":", lw=0.7, zorder=0)
        ax.plot(grid, g1, color=_C1, lw=1.4, label="Model 1")
        ax.plot(grid, g2, color=_C2, lw=1.4, label="Model 2")
        ax.errorbar(phase1, rv1, yerr=err1_tot,
                    fmt="none", ecolor="0.65", elinewidth=0.6, zorder=3)
        ax.errorbar(phase1, rv1, yerr=err1, fmt="o", ms=4, color=_C1,
                    elinewidth=0.7, zorder=4, label="RV$_1$")
        ax.errorbar(phase2, rv2, yerr=err2_tot,
                    fmt="none", ecolor="0.65", elinewidth=0.6, zorder=3)
        ax.errorbar(phase2, rv2, yerr=err2, fmt="s", ms=4, color=_C2,
                    elinewidth=0.7, zorder=4, label="RV$_2$")

        masked = data.get("masked") or {}
        mlabel = "masked"
        for tsk, rvk, errk, mk in ((SB2_TS1, SB2_RV1, SB2_ERR1, "o"),
                                   (SB2_TS2, SB2_RV2, SB2_ERR2, "s")):
            mts = np.asarray(masked.get(tsk, []), float)
            if not mts.size:
                continue
            mph = ((mts - T0) / P) % 1.0
            color = _C1 if mk == "o" else _C2
            ax.errorbar(mph, np.asarray(masked[rvk], float),
                        yerr=np.asarray(masked[errk], float),
                        fmt=mk, ms=4, color=color, mfc="white", alpha=0.35,
                        elinewidth=0.6, zorder=2, label=mlabel)
            mlabel = None

        ax.set_title(
            f"{_sanitize_name(star_name)}  sol {solution_id}: "
            rf"$P$={P:.6g} d, $e$={ecc:.3f}, $K_1$={K1:.1f}, $K_2$={K2:.1f}, "
            rf"$\gamma$={gamma:.1f} km/s",
            fontsize=8, loc="left")
        ax.set_ylabel(r"RV (km$\,$s$^{-1}$)")
        ax.tick_params(labelbottom=False)
        ax.legend(fontsize=6, ncol=4, loc="best", frameon=False)

        ax_r.axhline(0, color="0.50", lw=0.7)
        ax_r.errorbar(phase1, rv1 - m1, yerr=err1_tot, fmt="none",
                      ecolor="0.65", elinewidth=0.6, zorder=3)
        ax_r.errorbar(phase1, rv1 - m1, yerr=err1, fmt="o", ms=3.5, color=_C1,
                      elinewidth=0.6, zorder=4)
        ax_r.errorbar(phase2, rv2 - m2, yerr=err2_tot, fmt="none",
                      ecolor="0.65", elinewidth=0.6, zorder=3)
        ax_r.errorbar(phase2, rv2 - m2, yerr=err2, fmt="s", ms=3.5, color=_C2,
                      elinewidth=0.6, zorder=4)
        ax_r.set_xlabel("Orbital phase")
        ax_r.set_ylabel(r"O$-$C")
        ax_r.set_xlim(0, 1)
        fig.align_ylabels([ax, ax_r])

        out_path = os.path.join(sol_dir, f"sb2_fit_{solution_id}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    import lmfit
    with open(os.path.join(sol_dir, f"fit_report_{solution_id}.txt"), "w") as f:
        f.write(f"{star_name} solution {solution_id}\n")
        f.write(lmfit.fit_report(result))
        f.write("\n")
    return out_path


# ---------------------------------------------------------------------------
# Corner plot (SB2 variant of mcmc.mcmc_plotting.make_corner: parameterized
# omega column and number of trailing jitter columns)
# ---------------------------------------------------------------------------

def make_corner_sb2(flat_samples, flat_logprob, labels, tag, out_dir,
                    star_name="", omega_col=2, n_trailing_drop=2,
                    dlogp_clip=15.0, show=False):
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

        labels = list(labels)
        if n_trailing_drop:
            samples_plot = samples_plot[:, :-n_trailing_drop]
            labels = labels[:-n_trailing_drop]

        ndim = samples_plot.shape[1]
        if omega_col is not None and ndim > omega_col:
            samples_plot[:, omega_col] = np.degrees(samples_plot[:, omega_col])

        _font_kw = {"family": "serif", "fontsize": 22}
        fig = corner.corner(
            samples_plot,
            labels=labels,
            show_titles=False,
            max_n_ticks=4,
            bins=30,
            smooth=1.0,
            quantiles=[0.16, 0.5, 0.84],
            label_kwargs=_font_kw,
            title_kwargs={"family": "serif", "fontsize": 16},
        )
        orbit_type = "Circular" if "circ" in tag.lower() else "Eccentric"
        fig.suptitle(f"{_sanitize_name(star_name)} -- {orbit_type} SB2 Orbit",
                     fontsize=20, weight="bold", y=1.02)

        axes = np.array(fig.axes).reshape(ndim, ndim)
        for i in range(ndim):
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
            ax.set_title(f"{fmt.format(q50)}"
                         f"$^{{+{fmt.format(errp)}}}_{{-{fmt.format(errm)}}}$",
                         fontsize=20)

        fig.subplots_adjust(top=0.93, hspace=0.05, wspace=0.05)
        outname = os.path.join(out_dir, f"corner_{tag}.png")
        fig.savefig(outname, dpi=300, bbox_inches="tight")
        print(f"Saved {outname}")
        if show:
            plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# Phase-folded credible band (two components)
# ---------------------------------------------------------------------------

def plot_sb2_orbit_with_band_phase(data, flat_samples, flat_logprob,
                                   tag, circular=False, out_dir=".",
                                   star_name="", show=False):
    """Two-component phase plot: MAP + median curves, 16-84% bands, O-C panel."""
    if not out_dir and not show:
        return
    t1 = np.asarray(data[SB2_TS1], float)
    rv1 = np.asarray(data[SB2_RV1], float)
    err1 = np.asarray(data[SB2_ERR1], float)
    t2 = np.asarray(data[SB2_TS2], float)
    rv2 = np.asarray(data[SB2_RV2], float)
    err2 = np.asarray(data[SB2_ERR2], float)

    with plt.rc_context(_AA_RC_RV):
        imax = np.argmax(flat_logprob)
        theta_map = flat_samples[imax]
        theta_med = np.median(flat_samples, axis=0)

        P_map, T0_map = theta_map[0], theta_map[1]
        phase1 = ((t1 - T0_map) / P_map) % 1.0
        phase2 = ((t2 - T0_map) / P_map) % 1.0
        phase_grid = np.linspace(0.0, 1.0, 1000, endpoint=False)

        if not circular:
            def _unpack(theta):
                P, T0, omega, e, K1, K2, gamma = theta[:7]
                return P, T0, omega, e, K1, K2, gamma
        else:
            def _unpack(theta):
                P, T0, K1, K2, gamma = theta[:5]
                return P, T0, np.pi / 2, 0.0, K1, K2, gamma

        def model_on_grid(theta):
            # evaluate at the same absolute times the data fold uses
            # (MAP ephemeris): re-anchoring each draw at its own T0 cancels
            # the T0 scatter but leaves the correlated omega scatter as a
            # horizontal phase smear, inflating the band for loose-omega fits
            P, T0, omega, e, K1, K2, gamma = _unpack(theta)
            return sb2_rv_model_mcmc(T0_map + phase_grid * P_map,
                                     P, T0, omega, e, K1, K2, gamma)

        def model_at_data(theta):
            P, T0, omega, e, K1, K2, gamma = _unpack(theta)
            m1, _ = sb2_rv_model_mcmc(t1, P, T0, omega, e, K1, K2, gamma)
            _, m2 = sb2_rv_model_mcmc(t2, P, T0, omega, e, K1, K2, gamma)
            return m1, m2

        map1_grid, map2_grid = model_on_grid(theta_map)
        med1_grid, med2_grid = model_on_grid(theta_med)
        map1_data, map2_data = model_at_data(theta_map)

        rng = np.random.default_rng(123)
        ndraw = min(300, len(flat_samples))
        idx = rng.choice(len(flat_samples), size=ndraw, replace=False)
        draws1 = np.empty((ndraw, phase_grid.size))
        draws2 = np.empty((ndraw, phase_grid.size))
        for i, k in enumerate(idx):
            draws1[i], draws2[i] = model_on_grid(flat_samples[k])
        lo1, hi1 = np.percentile(draws1, [16, 84], axis=0)
        lo2, hi2 = np.percentile(draws2, [16, 84], axis=0)

        gamma_map = _unpack(theta_map)[-1]
        # trailing jitter columns: theta has 7 (ecc) / 5 (circ) orbital
        # slots, the rest are 1 (shared) or 2 (separate) log-jitters
        n_orb = 5 if circular else 7
        if flat_samples.shape[1] - n_orb == 1:
            s_jit1 = s_jit2 = float(np.exp(theta_map[-1]))
        else:
            s_jit1 = float(np.exp(theta_map[-2]))
            s_jit2 = float(np.exp(theta_map[-1]))
        err1_tot = np.sqrt(err1 ** 2 + s_jit1 ** 2)
        err2_tot = np.sqrt(err2 ** 2 + s_jit2 ** 2)

        fig = plt.figure(figsize=(_FULL_WIDTH, _FULL_WIDTH * _SB2_PHASE_ASPECT))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
        ax = fig.add_subplot(gs[0])
        ax_r = fig.add_subplot(gs[1], sharex=ax)

        ax.axhline(gamma_map, color="0.70", ls=":", lw=0.7, zorder=0)

        ax.fill_between(phase_grid, lo1, hi1, color=_C1, alpha=0.18,
                        label=r"68% band")
        ax.fill_between(phase_grid, lo2, hi2, color=_C2, alpha=0.18)
        ax.plot(phase_grid, map1_grid, color=_C1, lw=1.6, label="MAP orbit 1")
        ax.plot(phase_grid, map2_grid, color=_C2, lw=1.6, label="MAP orbit 2")
        ax.plot(phase_grid, med1_grid, color=_C1, lw=1.0, ls="--")
        ax.plot(phase_grid, med2_grid, color=_C2, lw=1.0, ls="--")

        ax.errorbar(phase1, rv1, yerr=err1_tot,
                    fmt="none", ecolor="0.60", elinewidth=0.6, capsize=2,
                    capthick=0.6, zorder=4,
                    label=r"$\sigma \oplus \sigma_\mathrm{jit}$")
        ax.errorbar(phase1, rv1, yerr=err1,
                    fmt="o", ms=4.5, color=_C1, ecolor=_C1, elinewidth=0.7,
                    capsize=2, capthick=0.7, zorder=5,
                    label=r"RV$_1$")
        ax.errorbar(phase2, rv2, yerr=err2_tot,
                    fmt="none", ecolor="0.60", elinewidth=0.6, capsize=2,
                    capthick=0.6, zorder=4)
        ax.errorbar(phase2, rv2, yerr=err2,
                    fmt="s", ms=4.5, color=_C2, ecolor=_C2, elinewidth=0.7,
                    capsize=2, capthick=0.7, zorder=5,
                    label=r"RV$_2$")

        # Masked (excluded) points: light open symbols in the RV panel only.
        masked = data.get("masked") or {}
        for tsk, rvk, errk, mk, mlabel in (
                (SB2_TS1, SB2_RV1, SB2_ERR1, "o", r"masked$_1$"),
                (SB2_TS2, SB2_RV2, SB2_ERR2, "s", r"masked$_2$")):
            mts = np.asarray(masked.get(tsk, []), float)
            if not mts.size:
                continue
            mph = ((mts - T0_map) / P_map) % 1.0
            color = _C1 if mk == "o" else _C2
            ax.errorbar(mph, np.asarray(masked[rvk], float),
                        yerr=np.asarray(masked[errk], float),
                        fmt=mk, ms=4.5, color=color, mfc="white",
                        alpha=0.35, elinewidth=0.7, capsize=2,
                        capthick=0.7, zorder=3, label=mlabel)

        ax.set_title(_sanitize_name(star_name), fontsize=10, weight="bold",
                     loc="left")
        ax.set_ylabel(r"RV (km$\,$s$^{-1}$)")
        ax.tick_params(labelbottom=False)

        ax_r.axhline(0, color="0.50", ls="-", lw=0.7)
        ax_r.errorbar(phase1, rv1 - map1_data, yerr=err1_tot,
                      fmt="none", ecolor="0.60", elinewidth=0.6, capsize=2,
                      capthick=0.6, zorder=4)
        ax_r.errorbar(phase1, rv1 - map1_data,
                      yerr=err1, fmt="o", ms=4, color=_C1,
                      elinewidth=0.6, capsize=2, capthick=0.6, zorder=5)
        ax_r.errorbar(phase2, rv2 - map2_data, yerr=err2_tot,
                      fmt="none", ecolor="0.60", elinewidth=0.6, capsize=2,
                      capthick=0.6, zorder=4)
        ax_r.errorbar(phase2, rv2 - map2_data,
                      yerr=err2, fmt="s", ms=4, color=_C2,
                      elinewidth=0.6, capsize=2, capthick=0.6, zorder=5)
        ax_r.set_xlabel("Orbital phase")
        # Units match the RV panel; short label as in the SB1 figures
        ax_r.set_ylabel(r"O$-$C")
        ax_r.set_xlim(0, 1)
        fig.align_ylabels([ax, ax_r])

        # masked1/masked2 stay out of the legend: open symbols are defined
        # by the paper's SB2 example-figure caption (fig:sb2_phase_example),
        # and a uniform 6-entry single-row legend keeps every phase PNG at
        # the same 3-per-page atlas height (the old 2-row masked legend made
        # those figures ~90 px taller).
        handles, leg_labels = ax.get_legend_handles_labels()
        keep = [(h, l) for h, l in zip(handles, leg_labels)
                if not l.startswith("masked")]
        handles, leg_labels = [h for h, _ in keep], [l for _, l in keep]
        fig.legend(handles, leg_labels, loc="lower center",
                   ncol=len(leg_labels), frameon=True, fancybox=False,
                   edgecolor="black", fontsize=7,
                   bbox_to_anchor=(0.5, -0.19), prop={"family": "serif"})

        outname = os.path.join(out_dir, f"phase_{tag}.png")
        # Tight-crop only vertically: the legend is narrower than the axes,
        # so a plain tight bbox shrinks the width and the figure prints
        # oversized at \textwidth. Spanning the full figure width matches
        # the SB1 pipeline PNG geometry (2127 px wide, aspect ~0.39).
        from matplotlib.transforms import Bbox
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        fig.savefig(outname, dpi=300,
                    bbox_inches=Bbox([[0.0, tb.y0 - 0.1],
                                      [_FULL_WIDTH, tb.y1 + 0.1]]))
        print(f"Saved {outname}")
        if show:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Walker trace plots
# ---------------------------------------------------------------------------

def plot_chain_traces(npz_path, out_path, star_name=""):
    """
    Per-parameter walker traces from a chains_*.npz (post-burn, thinned).
    Kept walkers grey; pruned walkers (keep_walkers False) red.
    """
    d = load_chain(npz_path)
    chain = d["chain"]                    # (n_kept, nwalkers, ndim)
    keep = d["keep_walkers"]
    names = d["param_names"]
    n_kept, nwalkers, ndim = chain.shape
    steps = np.arange(n_kept)

    with plt.rc_context(_AA_RC_RV):
        fig, axes = plt.subplots(ndim, 1, figsize=(_FULL_WIDTH, 1.1 * ndim),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        for i in range(ndim):
            ax = axes[i]
            for w in range(nwalkers):
                color = "0.4" if keep[w] else "crimson"
                alpha = 0.30 if keep[w] else 0.55
                ax.plot(steps, chain[:, w, i], color=color, lw=0.4,
                        alpha=alpha)
            ax.set_ylabel(names[i], fontsize=8)
            ax.tick_params(labelsize=7)
        axes[-1].set_xlabel(f"step (post-burn, thin={int(d['thin'])})",
                            fontsize=8)
        n_pruned = int((~keep).sum())
        title = f"{_sanitize_name(star_name)} -- {os.path.basename(npz_path)}"
        if n_pruned:
            title += f"  ({n_pruned}/{nwalkers} walkers pruned, red)"
        fig.suptitle(title, fontsize=9)
        fig.subplots_adjust(hspace=0.15, top=0.96)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close(fig)
