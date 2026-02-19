"""
spectroscopy.ccf_core — Cross-correlation functions for radial velocity extraction.

All functions accept configuration explicitly — no module-level globals.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from spectroscopy.constants import (
    WAVELENGTH, SCI_NORM, S2N, SNR_PPL, EPOCH_ID,
)
from spectroscopy.broadening import rotational_broadening, C_LIGHT
from spectroscopy.equivalent_width import calculate_equivalent_width2
from spectroscopy.mean_rv import dict_to_df


# ---------- low-level CCF ----------

def CCF(f1, f2, n):
    """Normalised cross-correlation coefficient."""
    return np.sum(f1 * f2) / np.std(f1) / np.std(f2) / n


def calc_s2n(flux, wl, s2n_range):
    """
    Signal-to-noise from the inverse std of flux in *s2n_range*.

    Parameters
    ----------
    flux : array
    wl : array
    s2n_range : (float, float)
        Wavelength window used for the noise estimate.
    """
    mask = (wl > s2n_range[0]) & (wl < s2n_range[1])
    return 1.0 / np.std(flux[mask])


# ---------- single-observation CCF ----------

def cross_cor_real(flux, temp, wgl, cci, sr, vel_range, n_res,
                   fit_rif=0.95, plot_first=False, plot_all=False):
    """
    Cross-correlate *flux* against *temp* and return (RV, sigma_RV).

    Parameters
    ----------
    flux, temp : 1-D arrays (mean-subtracted copies)
    wgl : wavelength grid
    cci : cross-correlation index array
    sr : shift-range index array
    vel_range : velocity grid (km/s)
    n_res : effective number of independent pixels
    fit_rif : fraction of CCF peak for parabola fit range
    plot_first : bool — show diagnostic plots for this call
    plot_all : bool — always show diagnostic plots

    Returns
    -------
    np.ndarray of shape (2,) : [RV, sigma_RV]  (NaN on failure)
    """
    CCFarr = np.array([
        CCF(np.copy(flux), (np.roll(temp, s))[cci], n_res) for s in sr
    ])
    IndMax = np.argmax(CCFarr)
    CCFMAX1 = CCFarr[IndMax]
    LeftEdgeArr = np.abs(fit_rif * CCFMAX1 - CCFarr[:IndMax])
    RightEdgeArr = np.abs(fit_rif * CCFMAX1 - CCFarr[IndMax + 1:])

    if len(LeftEdgeArr) == 0 or len(RightEdgeArr) == 0:
        if plot_all or plot_first:
            fig1, ax1 = plt.subplots()
            ax1.plot(vel_range, CCFarr, color='C0')
            ax1.set_xlabel('Radial velocity [km/s]')
            ax1.set_ylabel('Normalized CCF')
            plt.show(block=True)
        print("Can't find local maximum in CCF")
        return np.array([np.nan, np.nan])

    IndFit1 = np.argmin(np.abs(fit_rif * CCFMAX1 - CCFarr[:IndMax]))
    IndFit2 = (
        np.argmin(np.abs(fit_rif * CCFMAX1 - CCFarr[IndMax + 1:])) + IndMax + 1
    )

    a, b, c = np.polyfit(
        vel_range[IndFit1:IndFit2 + 1], CCFarr[IndFit1:IndFit2 + 1], 2
    )
    vmax = -b / (2 * a)
    CCFAtMax = min(1 - 1e-20, c - (b ** 2) / (4 * a))

    if plot_first or plot_all:
        FineVeloGrid = np.arange(vel_range[IndFit1], vel_range[IndFit2], 0.1)
        parable = a * FineVeloGrid ** 2 + b * FineVeloGrid + c
        fig1, ax1 = plt.subplots()
        ax1.plot(vel_range, CCFarr, color='C0')
        ax1.plot(FineVeloGrid, parable, color='C1', linewidth=1.5)
        ax1.set_xlabel('Radial velocity [km/s]')
        ax1.set_ylabel('Normalized CCF')
        plt.show(block=True)

        fig2, ax2 = plt.subplots()
        wgl_plot = wgl if len(cci) == len(wgl) else wgl[cci]
        ax2.plot(wgl_plot, flux, color='k', label='observation', alpha=0.8)
        ax2.plot(wgl_plot, temp[cci], color='orchid',
                 label='template, unshifted', alpha=0.9)
        ax2.plot(wgl_plot * (1 + vmax / C_LIGHT), temp[cci],
                 color='turquoise', label=f'shifted by {vmax:.2f}', alpha=0.9)
        ax2.set_xlabel(r'Wavelength [$\AA$]')
        ax2.set_ylabel('Normalized flux')
        ax2.legend(loc='best')
        plt.show(block=True)

    if CCFAtMax > 1:
        print("Failed to cross-correlate: template probably sucks!")
        print("Check cross-correlation function + parable fit.")
        return np.nan, np.nan

    CFFdvdvAtMax = 2 * a
    return np.array([
        vmax,
        np.sqrt(-1.0 / (n_res * CFFdvdvAtMax * CCFAtMax / (1 - CCFAtMax ** 2))),
    ])


# ---------- full observation-set CCF ----------

def cross_cor(data, star_name, temp, lines_to_ranges, cfg,
              meta_data=None, seperate_speed=False):
    """
    Run CCF on all observations in *data* against *temp*.

    Parameters
    ----------
    data : dict
        {mjd: {WAVELENGTH: ..., SCI_NORM: ..., ...}}
    star_name : str
    temp : dict or None
        Template spectrum (same dict layout) or None to use first obs.
    lines_to_ranges : dict
        {line_name: (wl_min, wl_max)}
    cfg : pd.Series or SimpleNamespace
        Configuration with attributes: velocity_range, intr_kind,
        fit_range_fraction, plot_first, plot_all, path_to_meta_data_csv,
        S2Nrange.
    meta_data : pd.DataFrame or None
        Metadata with columns ID, vsini (needed when rotational broadening
        is applied).
    seperate_speed : bool
        Whether to measure RVs for each spectral line separately.

    Returns
    -------
    pd.DataFrame
        CCF output with RV, RVsig, EW columns per line, S2N, SNR_PPL, EPOCH_ID.
    """
    # Compute S/N for each observation
    for mjd, spectrum in data.items():
        data[mjd][S2N] = calc_s2n(spectrum[SCI_NORM], spectrum[WAVELENGTH],
                                  cfg.S2Nrange)

    ranges_to_lines = {value: key for key, value in lines_to_ranges.items()}
    first_mjd = next(iter(data.keys()))

    delta_wl = data[first_mjd][WAVELENGTH][1] - data[first_mjd][WAVELENGTH][0]
    resolution = data[first_mjd][WAVELENGTH][1] / delta_wl
    vbin = C_LIGHT / resolution
    s_range = np.arange(
        int(cfg.velocity_range[0] / vbin),
        int(cfg.velocity_range[1] / vbin) + 1, 1,
    )
    velo_range = vbin * s_range

    # Build logarithmic wavelength grid
    ranges = sorted(lines_to_ranges.values())
    wl_regions = (
        np.array(ranges)
        * np.array([
            1.0 - 1.1 * cfg.velocity_range[1] / C_LIGHT,
            1 - 1.1 * cfg.velocity_range[0] / C_LIGHT,
        ])
    )
    min_wl = min(wl_regions[:, 0])
    max_wl = max(wl_regions[:, 1])
    n_waves = int(np.log(max_wl / min_wl) / np.log(1.0 + vbin / C_LIGHT))
    wave_grid_log = min_wl * (1.0 + vbin / C_LIGHT) ** np.arange(n_waves)

    int_is = np.array([
        np.argmin(np.abs(wave_grid_log - r[0])) for r in ranges
    ])
    int_fs = np.array([
        np.argmin(np.abs(wave_grid_log - r[1])) for r in ranges
    ])
    n_res = np.sum(int_fs - int_is)
    cross_corr_index = np.concatenate([
        np.arange(int_is[i], int_fs[i]) for i in range(len(int_fs))
    ])

    # Prepare template
    chosen_temp = data[first_mjd] if temp is None else temp
    if getattr(cfg, 'path_to_meta_data_csv', '') != '' and temp is not None and meta_data is not None:
        vsini_id = star_name.split('_')[1] if '_' in star_name else star_name
        rot_vel = meta_data[meta_data.ID.str.contains(vsini_id)].vsini.values[0]
        broadened = rotational_broadening(
            chosen_temp[WAVELENGTH], chosen_temp[SCI_NORM], rot_vel,
        )
        template = interp1d(
            chosen_temp[WAVELENGTH], broadened,
            bounds_error=False, fill_value=1.0, kind=cfg.intr_kind,
        )(wave_grid_log)
    else:
        template = interp1d(
            chosen_temp[WAVELENGTH], chosen_temp[SCI_NORM],
            bounds_error=False, fill_value=1.0, kind=cfg.intr_kind,
        )(wave_grid_log)

    # Track whether first-plot flag should be consumed
    do_plot_first = getattr(cfg, 'plot_first', False)
    do_plot_all = getattr(cfg, 'plot_all', False)

    out_dict = {}
    for mjd, spectrum in sorted(data.items()):
        out_dict[mjd] = {}

        if seperate_speed:
            for i in range(len(int_fs)):
                idx_range = np.arange(int_is[i], int_fs[i])
                fluxes = interp1d(
                    spectrum[WAVELENGTH],
                    np.nan_to_num(spectrum[SCI_NORM]),
                    bounds_error=False, fill_value=1.0, kind=cfg.intr_kind,
                )(wave_grid_log[idx_range])

                ew, ew_err = calculate_equivalent_width2(
                    wave_grid_log[idx_range], fluxes, data[mjd][S2N],
                )

                CCFeval = cross_cor_real(
                    np.copy(fluxes - np.mean(fluxes)),
                    np.copy(template - np.mean(template)),
                    wave_grid_log[idx_range], idx_range, s_range, velo_range,
                    np.argmin(np.abs(wave_grid_log - ranges[i][1]))
                    - np.argmin(np.abs(wave_grid_log - ranges[i][0])),
                    cfg.fit_range_fraction,
                    plot_first=do_plot_first,
                    plot_all=do_plot_all,
                )
                line_name = ranges_to_lines[ranges[i]]
                out_dict[mjd][f"{line_name} RV"] = CCFeval[0]
                out_dict[mjd][f"{line_name} RVsig"] = CCFeval[1]
                out_dict[mjd][f"{line_name} EW"] = ew
                out_dict[mjd][f"{line_name} EWsig"] = ew_err

        fluxes = interp1d(
            spectrum[WAVELENGTH],
            np.nan_to_num(spectrum[SCI_NORM]),
            bounds_error=False, fill_value=1.0, kind=cfg.intr_kind,
        )(wave_grid_log[cross_corr_index])

        CCFeval = cross_cor_real(
            np.copy(fluxes - np.mean(fluxes)),
            np.copy(template - np.mean(template)),
            wave_grid_log, cross_corr_index, s_range, velo_range,
            n_res, cfg.fit_range_fraction,
            plot_first=do_plot_first,
            plot_all=do_plot_all,
        )
        out_dict[mjd]["merged RV"] = CCFeval[0]
        out_dict[mjd]["merged RVsig"] = CCFeval[1]
        out_dict[mjd][S2N] = data[mjd][S2N]
        out_dict[mjd][SNR_PPL] = data[mjd][SNR_PPL]
        out_dict[mjd][EPOCH_ID] = data[mjd][EPOCH_ID]

        # Consume plot_first after first observation
        do_plot_first = False

    return dict_to_df(out_dict)
