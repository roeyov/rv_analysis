import argparse
import ast
import os
import yaml

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
from Spectroscopy.MeanRVCalculater import calculate_weighted_rv_with_flags
from itertools import cycle
import plotly.graph_objects as go

from spectrasDrawer import load_all_spectra, load_templates, load_template
from FilesCollector import find_files_with_strings, load_elements_list
from MeanRVCalculater import calculate_weighted_rv_with_flags, dict_to_df
from plot_extrema_spectra import save_minmax_overlay_plots
from constants import *
from ccf_animation import animate_ccf_process

input_df = marker_dict = meta_data = None
C_LIGHT = 299792.458

def plot_rv_vs_mjd_combined(cc_result_df, mean_calc_df, name, out):
    """
    Make one interactive HTML with a button to switch between:
      - All_RVS:          cc_result_df (all RV columns, no filtering)
      - Final_RVS:        mean_calc_df (only 'Mean RV', filter=True)
      - Significant_RVS:  mean_calc_df (all RV columns, filter=True)

    Returns: path to the saved HTML.
    """
    os.makedirs(os.path.join(out, name), exist_ok=True)

    def _select_rv_columns(df, plot_only=None):
        if plot_only is None:
            return [c for c in df.columns if ("RV" in c) and ("RVsig" not in c) and ("flag" not in c) and (c != "MJD")]
        return [key + " RV" for key in plot_only]

    def _apply_filter(df, rv_col, do_filter):
        flag_col = rv_col.replace("RV", "flag")
        if do_filter and flag_col in df.columns:
            return df[df[flag_col] == 0]
        return df

    # map a few mpl markers to plotly symbols (extend if needed)
    mpl_to_plotly_symbol = {
        'o': 'circle', 's': 'square', '^': 'triangle-up', 'v': 'triangle-down',
        '<': 'triangle-left', '>': 'triangle-right', 'D': 'diamond',
        '*': 'star', 'x': 'x', '+': 'cross', '.': 'circle-open'
    }

    fig = go.Figure()
    trace_sections = []  # list of (section_name, [trace_indices])

    # ---- Section 1: All_RVS (cc_result_df, no filtering) ----
    all_indices = []
    rv_cols_all = _select_rv_columns(cc_result_df, plot_only=None)
    for rv_col in rv_cols_all:
        marker, color = marker_dict[rv_col]
        symbol = mpl_to_plotly_symbol.get(marker, 'circle')
        dfp = _apply_filter(cc_result_df, rv_col, False)
        x = dfp["MJD"].to_numpy()
        y = dfp[rv_col].to_numpy()
        unc = rv_col.replace("RV", "RVsig")
        if unc in dfp.columns:
            yerr = dfp[unc].to_numpy()
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name=rv_col,
                marker=dict(symbol=symbol, color=color, opacity=0.9, size=8),
                error_y=dict(type="data", array=yerr, visible=True),
                hovertemplate=f"MJD=%{{x}}<br>RV=%{{y}} km/s<extra>{rv_col}</extra>",
                visible=True  # default section
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name=rv_col,
                marker=dict(symbol=symbol, color=color, opacity=0.9, size=8),
                hovertemplate=f"MJD=%{{x}}<br>RV=%{{y}} km/s<extra>{rv_col}</extra>",
                visible=True
            ))
        all_indices.append(len(fig.data) - 1)
    trace_sections.append(("All_RVS", all_indices))

    # ---- Section 2: Final_RVS (mean_calc_df, only 'Mean RV', filter=True) ----
    final_indices = []
    rv_cols_final = _select_rv_columns(mean_calc_df, plot_only=["Mean"])
    for rv_col in rv_cols_final:
        marker, color = marker_dict[rv_col]
        symbol = mpl_to_plotly_symbol.get(marker, 'circle')
        dfp = _apply_filter(mean_calc_df, rv_col, True)
        x = dfp["MJD"].to_numpy()
        y = dfp[rv_col].to_numpy()
        unc = rv_col.replace("RV", "RVsig")
        if unc in dfp.columns:
            yerr = dfp[unc].to_numpy()
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name=f"{rv_col} (final)",
                marker=dict(symbol=symbol, color=color, opacity=0.95, size=10),
                error_y=dict(type="data", array=yerr, visible=True),
                hovertemplate=f"MJD=%{{x}}<br>RV=%{{y}} km/s<extra>{rv_col}</extra>",
                visible=False
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name=f"{rv_col} (final)",
                marker=dict(symbol=symbol, color=color, opacity=0.95, size=10),
                hovertemplate=f"MJD=%{{x}}<br>RV=%{{y}} km/s<extra>{rv_col}</extra>",
                visible=False
            ))
        nan_mask = mean_calc_df["Mean RV"].isna()
        nan_mjds = mean_calc_df.loc[nan_mask, "MJD"]

        for mjd in nan_mjds:
            fig.add_vline(
                x=mjd,
                line=dict(color="red", dash="dash", width=1),
                annotation_text=f"MJD={mjd:.2f}",
                annotation_position="top",
                annotation=dict(font=dict(color="red"))
            )

        # 2. Add header with count of valid vs total
        n_total = len(mean_calc_df)
        n_valid = mean_calc_df["Mean RV"].notna().sum()
        footer_text = f"{n_valid}/{n_total} epochs with valid Mean RV"

        fig.update_layout(
            annotations=[
                dict(
                    text=footer_text,
                    x=0.5,  # centered
                    xref="paper",
                    y=-0.10,  # push it below x-axis
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="black")
                )
            ]
        )

        final_indices.append(len(fig.data) - 1)
    trace_sections.append(("Final_RVS", final_indices))

    # ---- Section 3: Significant_RVS (mean_calc_df, all RVs, filter=True) ----
    sig_indices = []
    rv_cols_sig = _select_rv_columns(mean_calc_df, plot_only=None)
    for rv_col in rv_cols_sig:
        marker, color = marker_dict.get(rv_col, ('o', '#1f77b4'))
        symbol = mpl_to_plotly_symbol.get(marker, 'circle')
        dfp = _apply_filter(mean_calc_df, rv_col, True)
        if dfp.empty:
            continue
        x = dfp["MJD"].to_numpy()
        y = dfp[rv_col].to_numpy()
        unc = rv_col.replace("RV", "RVsig")
        if unc in dfp.columns:
            yerr = dfp[unc].to_numpy()
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name=f"{rv_col} (signif.)",
                marker=dict(symbol=symbol, color=color, opacity=0.9, size=8),
                error_y=dict(type="data", array=yerr, visible=True),
                hovertemplate=f"MJD=%{{x}}<br>RV=%{{y}} km/s<extra>{rv_col}</extra>",
                visible=False
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name=f"{rv_col} (signif.)",
                marker=dict(symbol=symbol, color=color, opacity=0.9, size=8),
                hovertemplate=f"MJD=%{{x}}<br>RV=%{{y}} km/s<extra>{rv_col}</extra>",
                visible=False
            ))
        sig_indices.append(len(fig.data) - 1)
    trace_sections.append(("Significant_RVS", sig_indices))

    # ---- Buttons to toggle sections ----
    def vis_mask(active_name):
        mask = [False] * len(fig.data)
        for sect_name, idxs in trace_sections:
            if sect_name == active_name:
                for i in idxs:
                    mask[i] = True
        return mask

    buttons = []
    for sect_name, _ in trace_sections:
        buttons.append(dict(
            label=sect_name,
            method="update",
            args=[{"visible": vis_mask(sect_name)},
                  {"title": f"RV vs MJD {name} | {sect_name}"}]
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5, xanchor="center",
            y=1.15, yanchor="top",
            buttons=buttons,
            showactive=True,
            pad={"r": 10, "t": 5}
        )],
        title=f"RV vs MJD {name} | All_RVS",
        xaxis_title="MJD [d]",
        yaxis_title="RV [km/s]",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", yanchor="top"),
        margin=dict(l=60, r=20, t=100, b=100),
    )

    # filename
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in str(name)).strip().replace(" ", "_")
    html_path = os.path.join(out, name, f"{safe_name}_rv_vs_mjd_combined.html")
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    return html_path

def plot_rv_vs_mjd(df, name, plot_only=None, filter=False, out=None, out_suffix=''):
    """
    If `out` is None -> show with Matplotlib (no saving).
    If `out` is a path  -> save interactive Plotly HTML at that path (returns filepath).
    """
    _plot_rv_vs_mjd_mpl(df, name, plot_only=plot_only, filter=filter)
    if out is not None:
        os.makedirs(out, exist_ok=True)
        return _plot_rv_vs_mjd_plotly_save(df, name, plot_only=plot_only, filter=filter,
                                           out=out, out_suffix=out_suffix)
    return None


def _select_rv_columns(df, plot_only):
    if plot_only is None:
        rv_columns = [
            col for col in df.columns
            if ("RV" in col) and ("RVsig" not in col) and ("flag" not in col) and (col != "MJD")
        ]
    else:
        rv_columns = [key + " RV" for key in plot_only]
    return rv_columns


def _apply_filter(df, rv_col, do_filter):
    flag_col = rv_col.replace("RV", "flag")
    if do_filter and flag_col in df.columns:
        return df[df[flag_col] == 0]
    return df


# -------- Matplotlib (show only) --------
def _plot_rv_vs_mjd_mpl(df, name, plot_only=None, filter=False):
    rv_columns = _select_rv_columns(df, plot_only)

    plt.figure(figsize=(8, 6))
    for rv_col in rv_columns:
        marker, color = marker_dict[rv_col]  # assumes your global marker_dict
        uncertainty_col = rv_col.replace("RV", "RVsig")
        df_to_plot = _apply_filter(df, rv_col, filter)

        if uncertainty_col in df.columns:
            plt.errorbar(df_to_plot["MJD"], df_to_plot[rv_col], yerr=df_to_plot[uncertainty_col],
                         fmt=marker, linestyle='', color=color,
                         label=rv_col, capsize=3, alpha=0.8)
        else:
            plt.plot(df_to_plot["MJD"], df_to_plot[rv_col], marker=marker, linestyle='',
                     color=color, label=rv_col, alpha=0.8)

    plt.xlabel("MJD [d]")
    plt.ylabel("RV [km/s]")
    plt.title(f"RV vs MJD {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------- Plotly (save HTML) --------
def _plot_rv_vs_mjd_plotly_save(df, name, plot_only=None, filter=False, out='.', out_suffix=''):
    rv_columns = _select_rv_columns(df, plot_only)

    # Map common Matplotlib markers to Plotly symbols
    mpl_to_plotly_symbol = {
        'o': 'circle',
        's': 'square',
        '^': 'triangle-up',
        'v': 'triangle-down',
        '<': 'triangle-left',
        '>': 'triangle-right',
        'D': 'diamond',
        '*': 'star',
        'x': 'x',
        '+': 'cross',
        '.': 'circle-open'
    }

    fig = go.Figure()

    for rv_col in rv_columns:
        marker, color = marker_dict[rv_col]  # assumes your global marker_dict
        symbol = mpl_to_plotly_symbol.get(marker, 'circle')

        uncertainty_col = rv_col.replace("RV", "RVsig")
        df_to_plot = _apply_filter(df, rv_col, filter)

        x = df_to_plot["MJD"].to_numpy()
        y = df_to_plot[rv_col].to_numpy()

        if uncertainty_col in df.columns:
            yerr = df_to_plot[uncertainty_col].to_numpy()
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="markers",
                name=rv_col,
                marker=dict(symbol=symbol, color=color, opacity=0.9, size=8),
                error_y=dict(type="data", array=yerr, visible=True),
                hovertemplate=f"MJD=%{{x}}<br>RV=%{{y}} km/s<extra>{rv_col}</extra>"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="markers",
                name=rv_col,
                marker=dict(symbol=symbol, color=color, opacity=0.9, size=8),
                hovertemplate=f"MJD=%{{x}}<br>RV=%{{y}} km/s<extra>{rv_col}</extra>"
            ))

    fig.update_layout(
        title=f"RV vs MJD {name}",
        xaxis_title="MJD [d]",
        yaxis_title="RV [km/s]",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", yanchor="top"),
        margin=dict(l=60, r=20, t=60, b=100),
    )

    # filename
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in str(name)).strip().replace(" ", "_")
    suffix = (f"_{out_suffix}" if out_suffix else "")
    out_dir = os.path.join(out, name)
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"{safe_name}_rv_vs_mjd{suffix}.html")
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    return html_path


def calculate_equivalent_width2(wavelength,
                               normalized_flux,
                               snr=None,
                               sigma_threshold=3.0):
    """
    Calculates the Equivalent Width (EW) from pre-normalized flux.

    This function uses an SNR-based thresholding method ("Method 2")
    to algorithmically determine the line boundaries for integration.

    - If 'snr' is provided, it finds the boundaries where the line
      depth drops below 'sigma_threshold * noise_std' and calculates
      both EW and the EW error.
    - If 'snr' is None, it falls back to integrating the *entire*
      provided array and returns 'np.nan' for the error.

    Args:
        wavelength (np.ndarray): 1D array of wavelength values. Assumed
                                 to be pre-sliced to the search region.
        normalized_flux (np.ndarray): 1D array of *continuum-normalized*
                                      flux values.
        snr (float, optional): The signal-to-noise ratio *per pixel*
                               in the continuum. Required for boundary
                               finding and error calculation.
        sigma_threshold (float, optional): The significance level for
                                           boundary finding. Defaults to 3.0.

    Returns:
        tuple: (ew, ew_error)
               - ew (float): The calculated Equivalent Width.
               - ew_error (float): The calculated error, or np.nan if
                                   SNR was not provided.
    """

    # --- 0. Handle Edge Cases ---
    n_total_pix = len(wavelength)

    if n_total_pix == 0:
        return 0.0, np.nan  # No data, no EW

    # Calculate the *global* pixel step. We need this as a fallback
    # if the final line region is only 1 pixel wide.
    if n_total_pix > 1:
        delta_lambda_global = np.mean(np.diff(wavelength))
    else:
        # We can't calculate a width for a single point.
        # But we can estimate the EW if we assume a delta_lambda.
        # This is ambiguous. Let's return 0.
        return 0.0, np.nan

    blue_idx = 0
    red_idx = n_total_pix - 1

    # --- 1. Find Boundaries (Method 2) ---
    if snr is not None and snr > 0:
        noise_std = 1.0 / snr
        depth_threshold = sigma_threshold * noise_std

        # Find the line core
        try:
            core_idx = np.argmin(normalized_flux)
        except ValueError:
            return 0.0, np.nan  # Should be caught by n_total_pix==0

        core_depth = 1.0 - normalized_flux[core_idx]

        # If the line core isn't significant, the EW is zero.
        if core_depth < depth_threshold:
            return 0.0, 0.0  # EW is 0, error on a null measurement is 0

        # "Walk" left (blue-ward) from the core
        blue_idx = core_idx
        for i in range(core_idx - 1, -1, -1):
            depth = 1.0 - normalized_flux[i]
            if depth < depth_threshold:
                break  # We've exited the line
            blue_idx = i  # This pixel is still part of the line

        # "Walk" right (red-ward) from the core
        red_idx = core_idx
        for i in range(core_idx + 1, n_total_pix):
            depth = 1.0 - normalized_flux[i]
            if depth < depth_threshold:
                break  # We've exited the line
            red_idx = i  # This pixel is still part of the line

    # --- 2. Slice to the integration region ---
    # `blue_idx` and `red_idx` are now the *inclusive* indices
    line_indices = slice(blue_idx, red_idx + 1)
    line_wave = wavelength[line_indices]
    line_norm_flux = normalized_flux[line_indices]

    n_line_pix = len(line_wave)

    if n_line_pix == 0:
        return 0.0, 0.0  # Should be impossible if core was found, but safe

    # Determine the pixel step to use for integration
    if n_line_pix == 1:
        # Only one pixel in region, use the global step
        delta_lambda = delta_lambda_global
    else:
        # Use the average step *within* the line region
        delta_lambda = np.mean(np.diff(line_wave))

    # --- 3. Calculate EW ---
    # Integrand is (1 - F_norm).
    integrand = 1.0 - line_norm_flux
    ew = np.sum(integrand) * delta_lambda

    # --- 4. Calculate Error ---
    if snr is not None and snr > 0:
        ew_error = (np.sqrt(n_line_pix) * delta_lambda) / snr
    else:
        ew_error = np.nan

    return ew, ew_error

def calculate_equivalent_width(wavelength, normalized_flux, snr=None):
    """
    Calculate the equivalent width (EW) and its uncertainty.

    Parameters:
      wavelength : 1D numpy array of wavelength values (e.g., in Angstroms)
      normalized_flux       : 1D numpy array of flux values corresponding to the spectral line
      flux_err   : (Optional) 1D numpy array of uncertainties in flux; if provided,
                   the error in EW is computed.

    Returns:
      ew    : The equivalent width (in same wavelength units)
      ew_err: The propagated error on EW (or None if flux_err is not provided)
    """
    # Calculate the integrand: (1 - flux/continuum)
    integrand = 1 - normalized_flux

    # Use trapezoidal integration to compute the equivalent width.
    ew = np.trapz(integrand, wavelength)
    n_pix = wavelength.size
    # If no error is provided, return None for ew_err.
    if snr is None:
        return ew, None

    # For error propagation, we need the effective pixel width.
    # If the wavelength array is non-uniform, we approximate by taking the median spacing.
    delta_lambda = np.median(np.diff(wavelength))

    # Propagate the error: each bin contributes (delta_lambda/continuum * flux_err_i)
    ew_err = (np.sqrt(n_pix)*delta_lambda)/snr
    # --- 4. Plotting (Optional) ---
    if True:
        plt.figure(figsize=(10, 6))
        # Plot the normalized spectrum
        plt.plot(wavelength, normalized_flux, label='Normalized Data',
                 color='gray', alpha=0.8, drawstyle='steps-mid')

        # Plot the continuum line at y=1
        plt.axhline(1.0, linestyle='--', color='red', label='Continuum (y=1.0)')

        # Highlight the line region
        plt.axvspan(wavelength[0], wavelength[-1], color='blue',
                    alpha=0.1, label='Line Region')

        # Fill the area of the EW
        plt.fill_between(wavelength, normalized_flux, 1.0,
                         color='blue', alpha=0.3, label='Equivalent Width Area')

        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Normalized Flux')
        plt.title(f'Equivalent Width (EW) Calculation (Normalized)\n'
                   f'EW = {ew:.4f} $\pm$ {ew_err:.4f} Angstroms')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        # Set y-limits to make sense for normalized data
        plot_min = np.min(wavelength) - 0.1 if n_pix > 0 else 0.8
        plot_max = np.max(normalized_flux) + 0.1 if n_pix > 0 else 1.2
        plt.ylim(min(plot_min, 0.8), max(plot_max, 1.2))

        plt.show()
    return ew, ew_err

def rotational_broadening(ModWaves, ModFlux, v_rot, epsilon=0.6):
    """
    Apply rotational broadening to a spectrum in linear wavelength scale.
    Parameters:
        ModWaves (np.ndarray): Array of wavelengths in Angstroms.
        ModFlux (np.ndarray): Array of normalized flux values.
        v_rot (float): Rotational velocity in km/s.
        epsilon (float): Limb darkening coefficient (default is 0.6).
    Returns:
        np.ndarray: Rotationally broadened flux.
    """
    if v_rot == 0:
        return ModFlux
    # Convert wavelength to logarithmic scale
    log_wavelengths = np.log(ModWaves)
    # Interpolate flux on a uniform logarithmic wavelength grid
    delta_log_lambda = np.mean(np.diff(log_wavelengths))
    log_wavelengths_uniform = np.arange(log_wavelengths.min(), log_wavelengths.max(), delta_log_lambda)
    flux_uniform = np.interp(log_wavelengths_uniform, log_wavelengths, ModFlux)
    # Convert v_rot to the equivalent delta_log_lambda
    delta_log_lambda_vrot = v_rot / C_LIGHT
    # Calculate the number of points needed to cover the broadening kernel
    n_points = int(2 * delta_log_lambda_vrot / delta_log_lambda) + 1
    x = np.linspace(-v_rot, v_rot, n_points)
    kernel = np.zeros_like(x)
    mask = np.abs(x) <= v_rot
    kernel[mask] = (2 * (1 - epsilon) * np.sqrt(v_rot**2 - x[mask]**2) +
                    epsilon * (v_rot**2 - x[mask]**2)) / v_rot**2
    kernel /= np.sum(kernel)  # Normalize kernel
    # Convolve the flux with the broadening kernel
    broadened_flux_uniform = convolve1d(flux_uniform, kernel, mode='reflect')
    # Interpolate back to the original wavelength grid
    broadened_flux = np.interp(log_wavelengths, log_wavelengths_uniform, broadened_flux_uniform)
    return broadened_flux


def list_absolute_paths(directory):
    """
    List absolute paths of all files and directories in the given directory.

    Parameters:
        directory (str): The directory path.

    Returns:
        list: A list of absolute paths.
    """
    try:
        return [os.path.join(os.path.abspath(directory), item) for item in os.listdir(directory) if input_df.str_identifier in item]
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def parse_list_of_lists(arg):
    try:
        # Safely evaluate the string as a Python literal
        val = ast.literal_eval(arg)
        # Optionally, validate that it has the structure you expect:
        # e.g., that it's a list of lists of numbers.
        if (not isinstance(val, list) or
            not all(isinstance(sublist, list) for sublist in val)):
            raise ValueError
        return val
    except (SyntaxError, ValueError):
        raise argparse.ArgumentTypeError(
            "Argument must be in the form [[a1,a2],[b1,b2],...] (numbers only)."
        )

def calc_s2n(flux,wl):
    s2n_indices = (wl > input_df.S2Nrange[0]) * (wl < input_df.S2Nrange[1])
    return 1/np.std(flux[s2n_indices])

def CCF(f1, f2, n):
    f1 = f1
    f2 = f2
    return np.sum(f1 * f2) / np.std(f1) / np.std(f2) / n

# Returns RV and error following Zucker+ 2003
def cross_cor_real(flux, temp, wgl, cci, sr, vel_range, n_res, fit_rif=0.95):
    global input_df
    CCFarr = np.array([CCF(np.copy(flux),
                           (np.roll(temp, s))[cci] , n_res) for s in sr])
    IndMax = np.argmax(CCFarr)
    CCFMAX1 = CCFarr[IndMax]
    LeftEdgeArr = np.abs(fit_rif * CCFMAX1 - CCFarr[:IndMax])
    RightEdgeArr = np.abs(fit_rif * CCFMAX1 - CCFarr[IndMax+1:])

    if len(LeftEdgeArr) == 0 or len(RightEdgeArr) == 0:
        if  input_df.plot_all or input_df.original_plot_first or True:
            fig1, ax1 = plt.subplots()
            ax1.plot(vel_range, CCFarr, color='C0')
            ax1.set_xlabel('Radial velocity [km/s]')
            ax1.set_ylabel('Normalized CCF')
            # if not os.path.isdir(os.path.join(input_df.path_to_output,'CCFParabolas')):
            #     os.mkdir(os.path.join(input_df.path_to_output,'CCFParabolas'))
            # fig1.savefig('CCFParabolas/CCF_parabola_' + cutname + '.pdf')
            plt.show(block=True)
        print("Can't find local maximum in CCF")
        return np.array([np.nan, np.nan])

    IndFit1 = np.argmin(np.abs(fit_rif * CCFMAX1 -
                               CCFarr[:IndMax]))
    IndFit2 = np.argmin(np.abs(fit_rif * CCFMAX1 -
                               CCFarr[IndMax+1:])) + IndMax + 1

    a, b, c = np.polyfit(vel_range[IndFit1:IndFit2+1],
                         CCFarr[IndFit1:IndFit2+1], 2)
    vmax = -b/(2*a)
    CCFAtMax = min(1-1E-20, c - (b**2)/(4*a))
    # print(IndFit1, IndFit2, a, b, c, CCFAtMax, vmax, IndMax)

    if input_df.plot_first or input_df.plot_all:
        # plot the ccf
        FineVeloGrid = np.arange(vel_range[IndFit1], vel_range[IndFit2], .1)
        parable = (a * FineVeloGrid ** 2 + b * FineVeloGrid + c)
        fig1, ax1 = plt.subplots()
        ax1.plot(vel_range, CCFarr, color='C0')
        ax1.plot(FineVeloGrid, parable, color='C1', linewidth=1.5)
        ax1.set_xlabel('Radial velocity [km/s]')
        ax1.set_ylabel('Normalized CCF')
        # if not os.path.isdir(os.path.join(input_df.path_to_output,'CCFParabolas')):
        #     os.mkdir(os.path.join(input_df.path_to_output,'CCFParabolas'))
        #fig1.savefig('CCFParabolas/CCF_parabola_' + cutname + '.pdf')
        plt.show(block=True)
        # plot the spectrum and the template
        fig2, ax2 = plt.subplots()
        ax2.plot(wgl if len(cci)==len(wgl) else wgl[cci], flux, color='k',
                 label='observation', alpha=0.8)
        ax2.plot(wgl if len(cci)==len(wgl) else wgl[cci], temp[cci], color='orchid',
                 label='template, unshifted', alpha=0.9)
        ax2.plot(((wgl if len(cci)==len(wgl) else wgl[cci]) *(1+vmax/C_LIGHT)), temp[cci],
                 color='turquoise', label='shifted by {:.2f}'.format(vmax), alpha=0.9)
        ax2.set_xlabel(r'Wavelength [$\AA$]')
        ax2.set_ylabel('Normalized flux')
        ax2.legend(loc='best')
        plt.show(block=True)
        # # 1. Ensure we only send matching valid pixels to the animation
        # # Note: Check if 'flux' is already cut or if it needs [cci] as well.
        # # Based on your previous code, 'flux' might be full length or pre-cut.
        # # Safest bet is to ensure wgl and temp match.
        #
        # # Check if wgl covers the full range or just the cci range
        # wgl_for_anim = wgl[cci] if len(wgl) > len(cci) else wgl
        # flux_for_anim = flux  # flux is usually already cut in your pipeline
        #
        # # We need a template chunk that matches wgl_for_anim length
        # # Your code assumes temp is indexable by cci.
        # temp_for_anim = temp[cci]
        #
        # # 2. Call the animation with MATCHED arrays
        # animate_ccf_process(wgl_for_anim,
        #                     flux_for_anim,
        #                     temp_for_anim,
        #                     vel_range,
        #                     output_filename='/Users/roeyovadia/Roey/Masters/Reasearch/presentaions/Seminar/ccf_debug.mp4', fit_rif=fit_rif)
        input_df.plot_first = False

    if CCFAtMax > 1:
        print("Failed to cross-correlate: template probably sucks!")
        print("Check cross-correlation function + parable fit.")
        return np.nan, np.nan
    CFFdvdvAtMax = 2*a
    return np.array([vmax, np.sqrt(-1./(n_res * CFFdvdvAtMax *
                                        CCFAtMax / (1 - CCFAtMax**2)))])



def  cross_cor(data, star_name, temp, lines_to_ranges, seperate_speed=False):

    for mjd, spectrum in data.items():
        data[mjd][S2N] = calc_s2n(spectrum[SCI_NORM], spectrum[WAVELENGTH])
        # data[mjd][S2N] = data[mjd][SNR_PPL]

    ranges_to_lines = {value:key for key, value in lines_to_ranges.items()}
    first_mjd = next(iter(data.keys()))

    delta_wl = data[first_mjd][WAVELENGTH][1] - data[first_mjd][WAVELENGTH][0]
    resolution = data[first_mjd][WAVELENGTH][1] / delta_wl
    vbin = C_LIGHT / resolution
    s_range = np.arange(int(input_df.velocity_range[0] / vbin), int(input_df.velocity_range[1] / vbin) + 1, 1)
    velo_range = vbin * s_range

    # For N in error formula (NRes):
    ranges = sorted(lines_to_ranges.values())
    wl_regions = ranges * \
                      np.array([1. - 1.1 * input_df.velocity_range[1] / C_LIGHT, 1 - 1.1 * input_df.velocity_range[0] / C_LIGHT])
    min_wl = min(wl_regions[:,0])
    max_wl = max(wl_regions[:, 1])
    n_waves = int(np.log(max_wl / min_wl) / np.log(1. + vbin / C_LIGHT))
    wave_grid_log = min_wl * (1. + vbin / C_LIGHT) ** np.arange(n_waves)

    int_is = np.array([np.argmin(np.abs(wave_grid_log - ranges[i][0]))
                      for i in np.arange(len(ranges))])
    int_fs = np.array([np.argmin(np.abs(wave_grid_log - ranges[i][1]))
                      for i in np.arange(len(ranges))])
    n_res = np.sum(int_fs - int_is)
    cross_corr_index = np.concatenate(([np.arange(int_is[i], int_fs[i])
                                    for i in np.arange(len(int_fs))]))

    chosen_temp = data[first_mjd] if temp is None else temp
    if input_df.path_to_meta_data_csv != "" and temp is not None:
        vsini_id = star_name.split('_')[1] if ('_'  in star_name) else star_name
        rot_vel = meta_data[meta_data.ID.str.contains(vsini_id)].vsini.values[0]
        broadened_template = rotational_broadening(chosen_temp[WAVELENGTH], chosen_temp[SCI_NORM], rot_vel)
        template = interp1d(chosen_temp[WAVELENGTH], broadened_template, bounds_error=False,
            fill_value=1., kind=input_df.intr_kind)(wave_grid_log)
    else:
        template = interp1d(chosen_temp[WAVELENGTH], chosen_temp[SCI_NORM], bounds_error=False,
                            fill_value=1., kind=input_df.intr_kind)(wave_grid_log)

    out_dict = {}
    # Perform CCF for each observation
    for mjd, spectrum in sorted(data.items()):
        out_dict[mjd]  = {}
        if seperate_speed:
            for i in range(len(int_fs)):
                fluxes = interp1d(spectrum[WAVELENGTH], np.nan_to_num(spectrum[SCI_NORM]),
                                  bounds_error=False, fill_value=1., kind=input_df.intr_kind)(wave_grid_log[np.arange(int_is[i], int_fs[i])])
                # ew, ew_err = calculate_equivalent_width(wave_grid_log[np.arange(int_is[i], int_fs[i])],fluxes, data[mjd][S2N])
                ew, ew_err = calculate_equivalent_width2(wave_grid_log[np.arange(int_is[i], int_fs[i])],fluxes, data[mjd][S2N])

                CCFeval = cross_cor_real(np.copy(fluxes - np.mean(fluxes)),
                                         np.copy(template - np.mean(template)),
                                         wave_grid_log[np.arange(int_is[i], int_fs[i])],
                                         np.arange(int_is[i], int_fs[i]),
                                         s_range,
                                         velo_range,
                                         np.argmin(np.abs(wave_grid_log - ranges[i][1])) - np.argmin(np.abs(wave_grid_log - ranges[i][0])),
                                         input_df.fit_range_fraction)
                if np.isnan(CCFeval[0]) or np.isnan(CCFeval[1]):
                    pass
                out_dict[mjd]["{} RV".format(ranges_to_lines[ranges[i]])] = CCFeval[0]
                out_dict[mjd]["{} RVsig".format(ranges_to_lines[ranges[i]])] = CCFeval[1]
                out_dict[mjd]["{} EW".format(ranges_to_lines[ranges[i]])] = ew
                out_dict[mjd]["{} EWsig".format(ranges_to_lines[ranges[i]])] = ew_err

        fluxes = interp1d(spectrum[WAVELENGTH], np.nan_to_num(spectrum[SCI_NORM]),
                          bounds_error=False, fill_value=1., kind=input_df.intr_kind)(wave_grid_log[cross_corr_index])
        CCFeval = cross_cor_real(np.copy(fluxes - np.mean(fluxes)),
                                 np.copy(template - np.mean(template)),
                                 wave_grid_log,
                                 cross_corr_index,
                                 s_range,
                                 velo_range,
                                 n_res, input_df.fit_range_fraction)
        if np.isnan(CCFeval[0]) or np.isnan(CCFeval[1]):
            pass
        out_dict[mjd]["merged RV"] = CCFeval[0]
        out_dict[mjd]["merged RVsig"] = CCFeval[1]
        out_dict[mjd][S2N]=data[mjd][S2N]
        out_dict[mjd][SNR_PPL]= data[mjd][SNR_PPL]
        out_dict[mjd][EPOCH_ID]= data[mjd][EPOCH_ID]

    out_df = dict_to_df(out_dict)
    return out_df

def create_coadded_spectra(data,rv_out_df, rv_name="merged RV"):
    first_mjd = next(iter(data.keys()))

    # Initialize data storage
    t = []
    # Extract RV values for each MJD
    s2n_sqrd_sum = np.sum([data[mjd][S2N] ** 2 for mjd in data.keys()])
    coadd_spec = 0
    wave_grid = data[first_mjd][WAVELENGTH]

    for mjd in sorted(rv_out_df.MJD.values):
        try:
            cur_v = rv_out_df[rv_out_df.MJD == mjd][rv_name].values[0]
            weight_s2n = data[mjd][S2N] ** 2 / s2n_sqrd_sum
            if np.isnan(cur_v) or np.isnan(weight_s2n): continue
            shift_spec = interp1d(data[mjd][WAVELENGTH] * (1. - cur_v / C_LIGHT),
                                  np.nan_to_num(data[mjd][SCI_NORM]),
                                  bounds_error=False, fill_value=1.,
                                  kind=input_df.intr_kind)(wave_grid)
            coadd_spec += weight_s2n * shift_spec
            t.append(mjd)
        except KeyError:
            continue

    coadd_spec_dict = {WAVELENGTH:wave_grid, SCI_NORM:coadd_spec}
    return  coadd_spec_dict

def load_input_from_yaml(yaml_file):
    """
    Reads a YAML file and converts it to a Pandas Series object where
    all fields in the YAML file are stored as key-value pairs.

    Parameters:
        yaml_file (str): Path to the YAML file.

    Returns:
        pd.Series: A Pandas Series object containing all YAML fields.
    """
    try:
        with open(yaml_file, 'r') as file:
            # Load YAML data as a dictionary
            yaml_data = yaml.safe_load(file)
        # Convert dictionary to Pandas Series
        series = pd.Series(yaml_data)
        return series
    except FileNotFoundError:
        print(f"Error: File not found: {yaml_file}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def make_marker_dict():
    global marker_dict
    line_names = ['H_Gamma', 'H_Delta', 'H_Epsilon', 'HeI_4471', 'HeI_4388', 'HeI+HeII_4026', 'HeII_4542', 'HeII_4200',
                  'Median', 'Mean', 'NII_4447', 'NII_4440']
    # Define marker and color cycles
    markers = cycle(['o', 's', 'v', '^', 'D', '*', 'p', 'h'])
    colors = cycle(['blue', 'green', 'red', 'purple', 'orange', 'brown', 'cyan', 'magenta'])

    marker_dict = {}
    for window, marker,color in zip(line_names+["merged", "Mean"], markers, colors):
        marker_dict["{} RV".format(window)] = marker, color

def calc_final_rv():
    pass

def main():
    global input_df, marker_dict, meta_data
    TEMPLATE_INPUT_FILE = "./ccf_input.yaml"

    parser = argparse.ArgumentParser(description="A simple example of argparse")
    # Add arguments
    parser.add_argument('--input_file', type=str, help='a differnet file from the one in this directory,'
                                                            'if necessery.\n if not given the file in this directory will run {}'
                                                            '.'.format(TEMPLATE_INPUT_FILE))
    args = parser.parse_args()

    fp_yaml = args.input_file if args.input_file else TEMPLATE_INPUT_FILE
    input_df = load_input_from_yaml(fp_yaml)

    # Convert the dictionary keys back into numerical ranges
    lines_to_windows = input_df.cross_cor_ranges
    lines_to_windows = {key: tuple(map(float, value.strip("()").split(", "))) for key, value in lines_to_windows.items()}

    if input_df.path_to_list_of_objects:
        elements = load_elements_list(input_df.path_to_list_of_objects)
        all_files = find_files_with_strings(elements, input_df.path_to_observations, input_df.str_identifier)
    elif input_df.list_of_objects:
        elements = input_df.list_of_objects
        all_files = find_files_with_strings(elements, input_df.path_to_observations, input_df.str_identifier)
    else:
        elements = ["object"]
        all_files = {"object" : list_absolute_paths(input_df.path_to_observations)}


    if input_df.template_path == '':
        template =  {star : None for star in elements}
    elif os.path.isfile(input_df.template_path):
        fixed_template = load_template(input_df.template_path, WAVELENGTH, SCI_NORM)
        template =  {star : fixed_template for star in elements }
    elif os.path.isdir(input_df.template_path):
        template = load_templates(input_df.template_path, elements, WAVELENGTH, SCI_NORM)
    else:
        template = {"object" : None}

    if input_df.path_to_output != '':
        os.makedirs(input_df.path_to_output, exist_ok=True)

    if input_df.path_to_meta_data_csv != '':
        meta_data = pd.read_csv(input_df.path_to_meta_data_csv, usecols=input_df.columns_to_load)

    make_marker_dict()
    visuals_path = os.path.join(input_df.path_to_output, 'visuals')
    input_df["original_plot_first"] = input_df.plot_first
    for star in sorted(elements):
        print('Star {}'.format(star))
        if star not in template.keys(): continue
        a = load_all_spectra(all_files[star], MJD_MID, WAVELENGTH, SCI_NORM)

        cc_result = cross_cor(a,star, template[star], lines_to_windows,seperate_speed=input_df.seperate_speed)
        # Plot measured RVs
        plot_rv_vs_mjd(cc_result, star)
        mean_calc = calculate_weighted_rv_with_flags(cc_result)
        plot_rv_vs_mjd(mean_calc, star, plot_only=["Mean"], filter=True)
        plot_rv_vs_mjd(mean_calc, star, filter=True)
        plot_path = plot_rv_vs_mjd_combined(
            cc_result_df=cc_result,
            mean_calc_df=mean_calc,
            name=star,
            out=visuals_path,
        )
        coadd_spec = create_coadded_spectra(a, cc_result , rv_name="Mean RV")
        x = mean_calc
        x[S2N] = cc_result[S2N]
        y = pd.DataFrame(coadd_spec, columns=[WAVELENGTH, SCI_NORM])
        plt.close('all')  # close all open figures

        if input_df.path_to_output != '':
            y.to_csv(os.path.join(input_df.path_to_output, star + "_CoAdded.csv"), index=False, sep=',')
            x.to_csv(os.path.join(input_df.path_to_output, star + "_CCF_RVs.csv"), index=False, sep=',')

            save_minmax_overlay_plots(
                a,
                mean_calc,
                lines_to_windows,
                star,
                out_root=visuals_path,    # common parent folder
                mjd_col=MJD_MID,                     # match your constants
                wl_col=WAVELENGTH,
                flux_col=SCI_NORM
            )
        input_df.plot_first = input_df["original_plot_first"]

if __name__ == '__main__':
    main()


