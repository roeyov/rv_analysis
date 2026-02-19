"""
spectroscopy.coaddition — S/N-weighted spectral coaddition.
"""

import numpy as np
from scipy.interpolate import interp1d

from spectroscopy.constants import WAVELENGTH, SCI_NORM, S2N

C_LIGHT = 299792.458  # km/s


def create_coadded_spectra(data, rv_out_df, rv_name="merged RV",
                           intr_kind="linear"):
    """
    Create an S/N²-weighted co-added spectrum from individual observations.

    Parameters
    ----------
    data : dict
        {mjd: {WAVELENGTH: ..., SCI_NORM: ..., S2N: ...}} observation data.
    rv_out_df : pd.DataFrame
        DataFrame with columns 'MJD' and *rv_name* containing measured radial
        velocities for each epoch.
    rv_name : str
        Column name in *rv_out_df* that contains the RV to shift by.
    intr_kind : str
        Interpolation kind passed to scipy.interpolate.interp1d.

    Returns
    -------
    dict
        {WAVELENGTH: wave_grid, SCI_NORM: coadd_spec}
    """
    first_mjd = next(iter(data.keys()))
    wave_grid = data[first_mjd][WAVELENGTH]
    s2n_sqrd_sum = np.sum([data[mjd][S2N] ** 2 for mjd in data.keys()])

    coadd_spec = 0
    for mjd in sorted(rv_out_df.MJD.values):
        try:
            cur_v = rv_out_df[rv_out_df.MJD == mjd][rv_name].values[0]
            weight_s2n = data[mjd][S2N] ** 2 / s2n_sqrd_sum
            if np.isnan(cur_v) or np.isnan(weight_s2n):
                continue
            shift_spec = interp1d(
                data[mjd][WAVELENGTH] * (1.0 - cur_v / C_LIGHT),
                np.nan_to_num(data[mjd][SCI_NORM]),
                bounds_error=False, fill_value=1.0,
                kind=intr_kind,
            )(wave_grid)
            coadd_spec += weight_s2n * shift_spec
        except KeyError:
            continue

    return {WAVELENGTH: wave_grid, SCI_NORM: coadd_spec}
