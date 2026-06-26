"""
Interactive GUI for visualising the three Sana+2012 power-law distributions.

Sliders control π (period), κ (mass-ratio), and η (eccentricity).
Reference values from Sana+2012 are shown as dashed lines.

Usage:
    python -m simulations.powerlaw_gui
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ── Reference values (Sana+2012, Galactic O-stars) ──────────────────────────
REF_PI = -0.55
REF_KAPPA = -1.0
REF_ETA = -0.45

# ── Domain bounds ────────────────────────────────────────────────────────────
LOGP_MIN, LOGP_MAX = 0.15, 3.5   # log10(P/days)
Q_MIN, Q_MAX = 0.1, 1.0
E_MIN, E_MAX = 1e-4, 0.9

N_PTS = 500


def powerlaw_pdf(x, alpha, xmin, xmax):
    """Normalised p(x) ∝ x^alpha on [xmin, xmax]."""
    a = alpha + 1.0
    if abs(a) < 1e-8:
        # α = -1  →  p(x) = 1 / (x ln(xmax/xmin))
        pdf = 1.0 / (x * np.log(xmax / xmin))
    else:
        norm = (xmax ** a - xmin ** a) / a
        pdf = x ** alpha / norm
    return pdf


def powerlaw_cdf(x, alpha, xmin, xmax):
    """CDF of p(x) ∝ x^alpha on [xmin, xmax]; runs 0 → 1."""
    a = alpha + 1.0
    if abs(a) < 1e-8:
        return np.log(x / xmin) / np.log(xmax / xmin)
    return (x ** a - xmin ** a) / (xmax ** a - xmin ** a)


def build_gui():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(bottom=0.30, wspace=0.35, top=0.92)
    fig.canvas.manager.set_window_title("Power-law distributions  (π, κ, η)")

    # ── x-grids ──────────────────────────────────────────────────────────
    logP = np.linspace(LOGP_MIN, LOGP_MAX, N_PTS)
    q = np.linspace(Q_MIN, Q_MAX, N_PTS)
    e = np.linspace(E_MIN, E_MAX, N_PTS)

    # ── Initial curves ───────────────────────────────────────────────────
    line_p, = axes[0].plot(logP, powerlaw_pdf(logP, REF_PI, LOGP_MIN, LOGP_MAX),
                           color="C0", lw=2)
    ref_p, = axes[0].plot(logP, powerlaw_pdf(logP, REF_PI, LOGP_MIN, LOGP_MAX),
                          color="C0", lw=1, ls="--", alpha=0.4, label="Sana+2012")

    line_k, = axes[1].plot(q, powerlaw_pdf(q, REF_KAPPA, Q_MIN, Q_MAX),
                           color="C1", lw=2)
    ref_k, = axes[1].plot(q, powerlaw_pdf(q, REF_KAPPA, Q_MIN, Q_MAX),
                          color="C1", lw=1, ls="--", alpha=0.4, label="Sana+2012")

    line_e, = axes[2].plot(e, powerlaw_pdf(e, REF_ETA, E_MIN, E_MAX),
                           color="C2", lw=2)
    ref_e, = axes[2].plot(e, powerlaw_pdf(e, REF_ETA, E_MIN, E_MAX),
                          color="C2", lw=1, ls="--", alpha=0.4, label="Sana+2012")

    # ── Twin axes for CDFs (fixed 0–1 scale) ─────────────────────────────
    axes_cdf = [ax.twinx() for ax in axes]
    for axc in axes_cdf:
        axc.set_ylim(0.0, 1.05)
        axc.set_ylabel("CDF", fontsize=12)
        axc.tick_params(axis="y", labelsize=11)

    cdf_p, = axes_cdf[0].plot(logP, powerlaw_cdf(logP, REF_PI, LOGP_MIN, LOGP_MAX),
                              color="C0", lw=1.5, ls=":", label="CDF")
    cdf_k, = axes_cdf[1].plot(q, powerlaw_cdf(q, REF_KAPPA, Q_MIN, Q_MAX),
                              color="C1", lw=1.5, ls=":", label="CDF")
    cdf_e, = axes_cdf[2].plot(e, powerlaw_cdf(e, REF_ETA, E_MIN, E_MAX),
                              color="C2", lw=1.5, ls=":", label="CDF")

    # ── Axis labels ──────────────────────────────────────────────────────
    axes[0].set(xlabel=r"$\log_{10}\,P\;[\mathrm{days}]$",
                ylabel=r"$f(\log P)$",
                title=r"Period  ($\pi$)")
    axes[1].set(xlabel=r"$q = M_2/M_1$",
                ylabel=r"$f(q)$",
                title=r"Mass ratio  ($\kappa$)")
    axes[2].set(xlabel=r"$e$",
                ylabel=r"$f(e)$",
                title=r"Eccentricity  ($\eta$)")

    # Merge PDF + CDF handles into one legend per panel
    pdf_lines = [line_p, line_k, line_e]
    cdf_lines = [cdf_p, cdf_k, cdf_e]
    ref_lines = [ref_p, ref_k, ref_e]
    for ax, axc, pl, cl, rl in zip(axes, axes_cdf, pdf_lines, cdf_lines, ref_lines):
        ax.set_ylim(bottom=0)
        pl.set_label("PDF")
        handles = [pl, rl, cl]
        ax.legend(handles, [h.get_label() for h in handles], fontsize=12, loc="lower right")

    # ── Annotation showing current formula ───────────────────────────────
    txt_p = axes[0].text(0.97, 0.95, "", transform=axes[0].transAxes,
                         ha="right", va="top", fontsize=12,
                         bbox=dict(fc="white", ec="grey", alpha=0.8))
    txt_k = axes[1].text(0.97, 0.95, "", transform=axes[1].transAxes,
                         ha="right", va="top", fontsize=12,
                         bbox=dict(fc="white", ec="grey", alpha=0.8))
    txt_e = axes[2].text(0.97, 0.95, "", transform=axes[2].transAxes,
                         ha="right", va="top", fontsize=12,
                         bbox=dict(fc="white", ec="grey", alpha=0.8))

    def _fmt(name, var, val):
        return rf"$f({name}) \propto {name}^{{{var}}}$" + f"\n{var} = {val:+.2f}"

    def _update_text(val=None):
        txt_p.set_text(_fmt(r"\log P", r"\pi", sl_pi.val))
        txt_k.set_text(_fmt("q", r"\kappa", sl_kappa.val))
        txt_e.set_text(_fmt("e", r"\eta", sl_eta.val))

    # ── Sliders ──────────────────────────────────────────────────────────
    ax_pi = fig.add_axes([0.12, 0.14, 0.76, 0.03])
    ax_ka = fig.add_axes([0.12, 0.08, 0.76, 0.03])
    ax_et = fig.add_axes([0.12, 0.02, 0.76, 0.03])

    sl_pi = Slider(ax_pi, r"$\pi$", -3.0, 2.0, valinit=REF_PI, color="C0")
    sl_kappa = Slider(ax_ka, r"$\kappa$", -3.0, 2.0, valinit=REF_KAPPA, color="C1")
    sl_eta = Slider(ax_et, r"$\eta$", -1.5, 2.0, valinit=REF_ETA, color="C2")

    def update(val):
        y_p = powerlaw_pdf(logP, sl_pi.val, LOGP_MIN, LOGP_MAX)
        y_k = powerlaw_pdf(q, sl_kappa.val, Q_MIN, Q_MAX)
        y_e = powerlaw_pdf(e, sl_eta.val, E_MIN, E_MAX)

        line_p.set_ydata(y_p)
        line_k.set_ydata(y_k)
        line_e.set_ydata(y_e)

        cdf_p.set_ydata(powerlaw_cdf(logP, sl_pi.val, LOGP_MIN, LOGP_MAX))
        cdf_k.set_ydata(powerlaw_cdf(q, sl_kappa.val, Q_MIN, Q_MAX))
        cdf_e.set_ydata(powerlaw_cdf(e, sl_eta.val, E_MIN, E_MAX))

        for ax, y in zip(axes, [y_p, y_k, y_e]):
            ymax = np.nanmax(y)
            ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)

        _update_text()
        fig.canvas.draw_idle()

    sl_pi.on_changed(update)
    sl_kappa.on_changed(update)
    sl_eta.on_changed(update)

    # Initial text
    _update_text()
    update(None)

    plt.show()


if __name__ == "__main__":
    build_gui()
