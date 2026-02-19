"""
spectroscopy.equivalent_width — Equivalent width calculations.
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_equivalent_width2(wavelength, normalized_flux, snr=None,
                                sigma_threshold=3.0):
    """
    Calculate Equivalent Width using SNR-based boundary finding (Method 2).

    Uses SNR-based thresholding to determine line boundaries for integration.

    Parameters
    ----------
    wavelength : np.ndarray
        1D array of wavelength values (pre-sliced to the search region).
    normalized_flux : np.ndarray
        1D array of continuum-normalized flux values.
    snr : float, optional
        Signal-to-noise ratio per pixel in the continuum.
    sigma_threshold : float
        Significance level for boundary finding (default 3.0).

    Returns
    -------
    tuple
        (ew, ew_error)
    """
    n_total_pix = len(wavelength)

    if n_total_pix == 0:
        return 0.0, np.nan

    if n_total_pix > 1:
        delta_lambda_global = np.mean(np.diff(wavelength))
    else:
        return 0.0, np.nan

    blue_idx = 0
    red_idx = n_total_pix - 1

    # Find boundaries
    if snr is not None and snr > 0:
        noise_std = 1.0 / snr
        depth_threshold = sigma_threshold * noise_std

        try:
            core_idx = np.argmin(normalized_flux)
        except ValueError:
            return 0.0, np.nan

        core_depth = 1.0 - normalized_flux[core_idx]

        if core_depth < depth_threshold:
            return 0.0, 0.0

        # Walk blue-ward from the core
        blue_idx = core_idx
        for i in range(core_idx - 1, -1, -1):
            depth = 1.0 - normalized_flux[i]
            if depth < depth_threshold:
                break
            blue_idx = i

        # Walk red-ward from the core
        red_idx = core_idx
        for i in range(core_idx + 1, n_total_pix):
            depth = 1.0 - normalized_flux[i]
            if depth < depth_threshold:
                break
            red_idx = i

    # Slice to integration region
    line_indices = slice(blue_idx, red_idx + 1)
    line_wave = wavelength[line_indices]
    line_norm_flux = normalized_flux[line_indices]
    n_line_pix = len(line_wave)

    if n_line_pix == 0:
        return 0.0, 0.0

    if n_line_pix == 1:
        delta_lambda = delta_lambda_global
    else:
        delta_lambda = np.mean(np.diff(line_wave))

    # Calculate EW
    integrand = 1.0 - line_norm_flux
    ew = np.sum(integrand) * delta_lambda

    # Calculate Error
    if snr is not None and snr > 0:
        ew_error = (np.sqrt(n_line_pix) * delta_lambda) / snr
    else:
        ew_error = np.nan

    return ew, ew_error


def calculate_equivalent_width(wavelength, normalized_flux, snr=None):
    """
    Calculate the equivalent width (EW) and its uncertainty using trapezoidal integration.

    Parameters
    ----------
    wavelength : np.ndarray
        1D array of wavelength values.
    normalized_flux : np.ndarray
        1D array of flux values for the spectral line.
    snr : float, optional
        Signal-to-noise ratio. If provided, error is computed.

    Returns
    -------
    tuple
        (ew, ew_err)
    """
    integrand = 1 - normalized_flux
    ew = np.trapz(integrand, wavelength)
    n_pix = wavelength.size

    if snr is None:
        return ew, None

    delta_lambda = np.median(np.diff(wavelength))
    ew_err = (np.sqrt(n_pix) * delta_lambda) / snr

    # Diagnostic plot
    if True:
        plt.figure(figsize=(10, 6))
        plt.plot(wavelength, normalized_flux, label='Normalized Data',
                 color='gray', alpha=0.8, drawstyle='steps-mid')
        plt.axhline(1.0, linestyle='--', color='red', label='Continuum (y=1.0)')
        plt.axvspan(wavelength[0], wavelength[-1], color='blue',
                    alpha=0.1, label='Line Region')
        plt.fill_between(wavelength, normalized_flux, 1.0,
                         color='blue', alpha=0.3, label='Equivalent Width Area')
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Normalized Flux')
        plt.title(f'Equivalent Width (EW) Calculation (Normalized)\n'
                  f'EW = {ew:.4f} $\\pm$ {ew_err:.4f} Angstroms')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plot_min = np.min(wavelength) - 0.1 if n_pix > 0 else 0.8
        plot_max = np.max(normalized_flux) + 0.1 if n_pix > 0 else 1.2
        plt.ylim(min(plot_min, 0.8), max(plot_max, 1.2))
        plt.show()

    return ew, ew_err
