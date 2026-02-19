import os.path

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pylab as pylab
from PDC.pdc_func import calc_pdc, pdc_window, calc_pdc_optimized
from scipy.stats.distributions import chi2
from astropy.timeseries import LombScargle

params = {'legend.fontsize': 'large',
          'figure.figsize': (3, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

pmin = 1.
pmax = 2000.


def get_closest_b_vals(a_vec, b_vec, a_values):
    """
    For each target in a_values, find the element in a_vec that is closest to it
    and return the corresponding value from b_vec.

    Parameters
    ----------
    a_vec : array-like
        Array of x-values (independent variable).
    b_vec : array-like
        Array of y-values (dependent variable, same length as a_vec).
    a_values : float or array-like
        One or more target a-values to match.

    Returns
    -------
    b_closest : ndarray or float
        b_vec value(s) corresponding to the closest a_vec element(s).
    a_matched : ndarray or float
        The actual a_vec element(s) that were closest.
    """
    a_vec = np.asarray(a_vec)
    b_vec = np.asarray(b_vec)
    a_values_1 = np.atleast_1d(a_values)

    if len(a_vec) != len(b_vec):
        raise ValueError("a_vec and b_vec must have the same length")

    idxs = np.array([np.argmin(np.abs(a_vec - a)) for a in a_values_1])
    b_closest = b_vec[idxs]
    a_matched = a_vec[idxs]

    # Return scalar if input was scalar
    if np.isscalar(a_values) or a_values_1.shape == ():
        return b_closest.item(), a_matched.item()
    return b_closest, a_matched

def pdc(time, data, data_err=[], pmin=1., pmax=1000 , probabilities=(0.5, 0.01, 0.001)):
    # freq = np.arange(1/pmax, 1/pmin, 0.00001)
    log_p = np.arange(np.log(pmin), np.log(pmax),0.0005)
    p_range = np.exp(log_p)
    freq = 1/p_range
    A, a, pdc_power_reg = calc_pdc(freq, time, data, data_err)
    fap1 = chi2.sf(np.max(pdc_power_reg) * len(time) + 1, 1)
    fap_vec = chi2.sf(pdc_power_reg * len(time) + 1, 1)
    maxpow1 = np.max(pdc_power_reg)
    max_freq1 = freq[np.argmax(pdc_power_reg)]
    best_period1 = 1/max_freq1
    fal1, fap_to_fal1 = get_closest_b_vals(fap_vec, pdc_power_reg,probabilities)
    return best_period1, maxpow1, fap1, fal1, freq, pdc_power_reg, fap_vec

# def pdc_opt(time, data, data_err=[], pmin=1., pmax=1000 , probabilities=(0.5, 0.01, 0.001)):
#     # freq = np.arange(1/pmax, 1/pmin, 0.00001)
#     log_p = np.arange(np.log(pmin), np.log(pmax),0.0005)
#     p_range = np.exp(log_p)
#     freq = 1/p_range
#     A, a, pdc_power_reg = calc_pdc(freq, time, data, data_err)
#     fap1 = chi2.sf(np.max(pdc_power_reg) * len(time) + 1, 1)
#     fap_vec = chi2.sf(pdc_power_reg * len(time) + 1, 1)
#     maxpow1 = np.max(pdc_power_reg)
#     max_freq1 = freq[np.argmax(pdc_power_reg)]
#     best_period1 = 1/max_freq1
#     fal1, fap_to_fal1 = get_closest_b_vals(fap_vec, pdc_power_reg,probabilities)
#     return best_period1, maxpow1, fap1, fal1, freq, pdc_power_reg, fap_vec


def pdc_opt(times, data, data_err=[], pmin=1., pmax=1000, probabilities=(0.5, 0.01, 0.001)):
    """
    Drop-in replacement for 'pdc' that uses the M4-optimized kernel.
    """
    # 1. Frequency Grid Generation (Identical to original)
    log_p = np.arange(np.log(pmin), np.log(pmax), 0.0005)
    p_range = np.exp(log_p)
    freq = 1 / p_range

    # 2. Optimized Calculation
    # This calls the Numba-compiled function
    A, a, pdc_power_reg = calc_pdc_optimized(freq, times, data, data_err)

    # 3. Statistics Calculation (Identical to original)
    # We use numpy functions where possible for speed
    n_times = len(times)
    max_pow = np.max(pdc_power_reg)

    fap1 = chi2.sf(max_pow * n_times + 1, 1)
    fap_vec = chi2.sf(pdc_power_reg * n_times + 1, 1)

    max_idx = np.argmax(pdc_power_reg)
    max_freq1 = freq[max_idx]
    best_period1 = 1.0 / max_freq1

    # Assuming 'get_closest_b_vals' is defined in your scope
    fal1, fap_to_fal1 = get_closest_b_vals(fap_vec, pdc_power_reg, probabilities)

    return best_period1, max_pow, fap1, fal1, freq, pdc_power_reg, fap_vec

def ls(time, data, data_err=None, probabilities=(0.5, 0.01, 0.001),
       pmin=1., pmax=1000., norm='model',
       ls_method='fast', fa_method='baluev',
       samples_per_peak=50, nterms=1, center_data=True,
       n_bootstraps=1000, random_state=None):
    """
    Returns:
        best_period1, fap_peak, fal1, frequency1, power1, fap_vec
        where fap_vec[i] is the FAP corresponding to power1[i]
    """
    if data_err is not None and len(data_err) > 0:
        ls1 = LombScargle(time, data, data_err, normalization=norm, nterms=nterms, center_data=center_data)
    else:
        ls1 = LombScargle(time, data, normalization=norm, nterms=nterms, center_data=center_data)

    fmin, fmax = 1.0/pmax, 1.0/pmin
    frequency1, power1 = ls1.autopower(method=ls_method,
                                       minimum_frequency=fmin,
                                       maximum_frequency=fmax,
                                       samples_per_peak=samples_per_peak)

    if nterms > 1:
        fal1 = [np.nan] * len(probabilities)
    else:
        fal1 = ls1.false_alarm_level(probabilities, method=fa_method,
                                     minimum_frequency=fmin, maximum_frequency=fmax)

    if fa_method == 'bootstrap':
        fap_vec = ls1.false_alarm_probability(power1, method='bootstrap',
                                              minimum_frequency=fmin, maximum_frequency=fmax)
    else:
        fap_vec = ls1.false_alarm_probability(power1, method=fa_method,
                                              minimum_frequency=fmin, maximum_frequency=fmax)

    maxpow1 = np.max(power1)
    max_idx = np.argmax(power1)
    max_freq1 = frequency1[max_idx]
    best_period1 = 1.0 / max_freq1
    fap_peak = fap_vec[max_idx]
    return best_period1,maxpow1, fap_peak, fal1, frequency1, power1, fap_vec




def plotls(frequency, power, fal, bins = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], star_id='', pmin=1., pmax=1000., out_dir=None):
    '''
    Function to create publication-ready plot of the periodogram obtained with "ls"
​
    Parameters:
            frequency: The list of frequencies of the periodogram returned by "ls"
            power: The list of powers of the periodogram returned by "ls"
            fal: The false alarm levels returned by "ls"
            bins: list of ticks and tick labels for the orbital period axis
            star_id: Star name or identifyer
​
    Returns:
            Periodogram computed by "ls"
    '''

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(1/frequency, power, 'k-', alpha=0.5)
    ax.yaxis.set_label_coords(-0.09, 0.5)
    ax.set(xlim=(0.3, 2), ylim=(-0.03*power.max(), power.max()+0.1*power.max()),
        xlabel='Period (days)',
        ylabel='Power')
    fig.suptitle("Primary best Period : {0:.3f} days".format(1/frequency[np.argmax(power)]))
    plt.xscale('log')
    tickLabels = map(str, bins)
    ax.set_xticks(bins)
    ax.set_xticklabels(tickLabels)
    if len(fal) > 0:
        ax.plot( (0.5, 800), (fal[0], fal[0]), '-r', lw=1.2)
        ax.plot( (0.5, 800), (fal[1], fal[1]), '--y', lw=1.2)
        ax.plot( (0.5, 800), (fal[2], fal[2]), ':g', lw=1.2)
    ax.set_xlim(pmin, pmax)
    if power.max()+0.1*power.max() >= 10:
        ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.0f}'))
    else:
        ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.1f}'))
    if star_id:
        plt.title(star_id + ' periodogram')
    if out_dir:
        plt.savefig(out_dir)
    plt.show()
    plt.close()

import os
import numpy as np
import plotly.graph_objects as go

def plot_periodogram_plotly(
    frequency,
    power,
    fal,
    bins = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
    star_id='',
    pmin=1.0,
    pmax=1000.0,
    out_dir=None,
    periodogram_kind="LS",   # "LS" for Lomb–Scargle, "PDC" for Phase–Distance Correlation
):
    """
    Plot a periodogram with Plotly and save as HTML/PNG.

    periodogram_kind:
        "LS"  → Lomb–Scargle periodogram (default)
        "PDC" → Phase–distance correlation periodogram

    Args:
        frequency (array-like): Frequencies (1/days).
        power     (array-like): Periodogram statistic (e.g., LS power or PDC value).
        fal       (array-like): False-alarm levels in the same units as `power`
                                (typically for p ≈ 0.1, 0.01, 0.001).
        bins      (list[float]): Tick locations for period axis (days).
        star_id   (str): Star name / identifier (for title and filenames).
        pmin, pmax (float): Minimum and maximum period to display (days).
        out_dir   (str|Path|None): If provided, saves HTML + PNG there and returns
                                   path to HTML. If None, returns HTML string.
        periodogram_kind (str): "LS" or "PDC" – controls human-readable labels.

    Returns:
        str: Path to saved HTML (if out_dir is provided) or HTML string (if out_dir is None).
    """
    # Map kind → nice labels
    kind_key = (periodogram_kind or "LS").upper()
    if kind_key == "PDC" or  kind_key == "PDC_OPT":
        method_label = "PDC"
        y_label      = "PDC statistic"
        trace_name   = "PDC statistic"
    else:
        method_label = "LS"
        y_label      = "LS power"
        trace_name   = "LS power"

    frequency = np.asarray(frequency, dtype=float)
    power     = np.asarray(power, dtype=float)
    if bins is None:
        bins = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    # Guard: avoid division by zero/infs
    mask = np.isfinite(frequency) & (frequency > 0) & np.isfinite(power)
    freq = frequency[mask]
    powv = power[mask]
    period = 1.0 / freq

    # y-range
    if powv.size:
        ymax = float(np.nanmax(powv))
    else:
        ymax = 1.0
    yr_lo = -0.03 * ymax
    yr_hi = ymax * 1.10

    # Best period
    if powv.size:
        best_idx = int(np.nanargmax(powv))
        best_period = float(period[best_idx])
        best_power  = float(powv[best_idx])
    else:
        best_period, best_power = np.nan, np.nan

    # Build figure
    fig = go.Figure()

    # Main line
    fig.add_trace(
        go.Scatter(
            x=period,
            y=powv,
            mode="lines",
            line=dict(width=2),
            name=trace_name,
            hovertemplate=(
                "Orbital period = %{x:.5g} days<br>"
                f"{y_label} = %{{y:.5g}}"
                "<extra></extra>"
            ),
        )
    )

    # False-alarm levels (FAL): show up to 3 lines.
    fal_colors = ["red", "goldenrod", "green"]
    fal_styles = ["solid", "dash", "dot"]
    approx_ps  = [0.5, 0.01, 0.001]

    if fal is not None:
        fal_flat = np.asarray(fal).ravel()
        for i, val in enumerate(fal_flat[:3]):
            if np.isfinite(val):
                p_label = f"p ≈ {approx_ps[i]}" if i < len(approx_ps) else ""
                fig.add_hline(
                    y=float(val),
                    line_width=1.5,
                    line_dash=fal_styles[i % len(fal_styles)],
                    line_color=fal_colors[i % len(fal_colors)],
                    annotation_text=f"False-alarm level ({p_label})",
                    annotation_position="top left",
                    opacity=0.9
                )

    # Mark best period
    if np.isfinite(best_period):
        fig.add_vline(
            x=best_period,
            line_width=1.5,
            line_dash="dash",
            line_color="dodgerblue",
            annotation_text=f"Best period ≈ {best_period:.3f} d",
            annotation_position="top right"
        )

    # Titles
    if np.isfinite(best_period):
        title_main = f"Best orbital period ≈ {best_period:.3f} days"
    else:
        title_main = f"{method_label} periodogram"

    if star_id:
        title_full = f"{star_id} — {method_label} periodogram ({title_main})"
    else:
        title_full = f"{method_label} periodogram — {title_main}"

    # y tick format
    y_format = ".0f" if yr_hi >= 10 else ".1f"

    # Axis ranges (log period)
    log_pmin = np.log10(max(pmin, 1e-6))
    log_pmax = np.log10(max(pmax, pmin * 1.001))

    fig.update_layout(
        title=title_full,
        xaxis_title="Orbital period (days)",
        yaxis_title=y_label,
        template="plotly_white",
        margin=dict(l=80, r=20, t=80, b=80),
        font=dict(size=18),
        title_font=dict(size=22),
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=16),
        ),
        xaxis=dict(
            type="log",
            range=[log_pmin, log_pmax],
            tickmode="array",
            tickvals=bins,
            ticktext=[str(b) for b in bins],
            title_font=dict(size=20),
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            range=[yr_lo, yr_hi],
            tickformat=y_format,
            title_font=dict(size=20),
            tickfont=dict(size=16),
        ),
    )

    fig.update_xaxes(range=[log_pmin, log_pmax])

    # Save or return
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        suffix = "ls" if kind_key == "LS" else "pdc" if kind_key == "PDC" else "pdc_opt"
        base_html = f"{star_id}_{suffix}_periodogram.html" if star_id else f"{suffix}_periodogram.html"
        base_png  = f"{star_id}_{suffix}_periodogram.png"  if star_id else f"{suffix}_periodogram.png"

        out_html = os.path.join(out_dir, base_html)
        out_png  = os.path.join(out_dir, base_png)

        fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
        fig.write_image(
            out_png,
            width=900,  # wide enough for long titles
            height=550,  # tall enough so subtitles aren't cropped
            scale=4  # multiplies resolution → very crisp
        )
        return out_html
    else:
        return fig.to_html(include_plotlyjs="cdn", full_html=True)



if __name__ == '__main__':
    path_to_csv = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/stripped_star_pot_wCOadded/BLOeM_5-104_CCF_RVs.csv"
    data = pd.read_csv(path_to_csv, sep=' ')
    v1s = data['Mean RV']
    hjds1 = data['MJD']
    ervv1s = data['Mean RVsig']
    # v1s, hjds1, errs_v1 = out_single_and_plot(t0=0 ,p = 50, ecc = 0.4, omega = 10, k1=40, k2= 20, gamma=30, nrv=25, sig_rv=3.0, plot=True)
    # ervv1s = [3.] * 25

    period, fap, fal, freq, pow = ls(hjds1, v1s,data_err=ervv1s, pmin=1, pmax=1000)
    plotls(freq, pow, fal, pmin=pmin, pmax=pmax)

    best_period1, fap1, fap_vec, freq, pdc_power_reg = pdc(hjds1, v1s, data_err=ervv1s, pmin=1., pmax=1000)
    plotls(freq, pdc_power_reg, fal=[] , pmin=pmin, pmax=pmax)
    # probs = np.arange(-1, 1, 0.0001)
    # fap1 = chi2.sf(probs * 25 , 1)  # Survival function of chi-square distribution
    #
    # # Create scatter plot
    # plt.figure(figsize=(8, 6))
    # plt.scatter(probs, fap1, color='blue', alpha=0.6, edgecolors='black', label='FAP1')
    #
    # # Add labels and title
    # plt.xlabel('Probability (probs)', fontsize=14)
    # plt.ylabel('FAP1', fontsize=14)
    # plt.title('Scatter Plot of FAP1 vs. Probability', fontsize=16)
    #
    # # Add a grid for better readability
    # plt.grid(True, linestyle='--', alpha=0.7)
    #
    # # Add legend
    # plt.legend()
    #
    # # Show the plot
    # plt.tight_layout()
    # plt.show()

# print(chi2.sf(0.05 * 25 , 1) ) # Survival function of chi-square distribution


