"""
spectroscopy.broadening — Rotational broadening kernel for spectra.
"""

import numpy as np
from scipy.ndimage import convolve1d

C_LIGHT = 299792.458  # km/s


def rotational_broadening(ModWaves, ModFlux, v_rot, epsilon=0.6):
    """
    Apply rotational broadening to a spectrum in linear wavelength scale.

    Parameters
    ----------
    ModWaves : np.ndarray
        Array of wavelengths in Angstroms.
    ModFlux : np.ndarray
        Array of normalized flux values.
    v_rot : float
        Rotational velocity in km/s.
    epsilon : float
        Limb darkening coefficient (default is 0.6).

    Returns
    -------
    np.ndarray
        Rotationally broadened flux.
    """
    if v_rot == 0:
        return ModFlux

    # Convert wavelength to logarithmic scale
    log_wavelengths = np.log(ModWaves)
    delta_log_lambda = np.mean(np.diff(log_wavelengths))
    log_wavelengths_uniform = np.arange(
        log_wavelengths.min(), log_wavelengths.max(), delta_log_lambda
    )
    flux_uniform = np.interp(log_wavelengths_uniform, log_wavelengths, ModFlux)

    # Convert v_rot to the equivalent delta_log_lambda
    delta_log_lambda_vrot = v_rot / C_LIGHT

    # Broadening kernel
    n_points = int(2 * delta_log_lambda_vrot / delta_log_lambda) + 1
    x = np.linspace(-v_rot, v_rot, n_points)
    kernel = np.zeros_like(x)
    mask = np.abs(x) <= v_rot
    kernel[mask] = (
        2 * (1 - epsilon) * np.sqrt(v_rot**2 - x[mask]**2)
        + epsilon * (v_rot**2 - x[mask]**2)
    ) / v_rot**2
    kernel /= np.sum(kernel)

    broadened_flux_uniform = convolve1d(flux_uniform, kernel, mode='reflect')

    # Interpolate back to the original wavelength grid
    broadened_flux = np.interp(
        log_wavelengths, log_wavelengths_uniform, broadened_flux_uniform
    )
    return broadened_flux
