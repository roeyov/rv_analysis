"""
spectroscopy.coaddition — S/N-weighted spectral coaddition.
"""

from collections import Counter

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
    # Epochs that actually contribute: present in *data*, finite RV, finite
    # S/N. The weight normalisation must run over exactly this set — summing
    # over all epochs and then skipping some scales the coadd continuum by
    # sum(w_used)/sum(w_all) < 1.
    used_mjds = [
        mjd for mjd in sorted(rv_out_df.MJD.values)
        if mjd in data
        and not np.isnan(rv_out_df[rv_out_df.MJD == mjd][rv_name].values[0])
        and not np.isnan(data[mjd][S2N])
    ]
    if not used_mjds:
        raise ValueError("no usable epochs for co-addition")

    # Reference grid: the most common grid length among the used epochs
    # (earliest such epoch on ties) — dict order is filesystem-dependent and
    # can hand the output a single odd-sampling epoch's grid.
    length_counts = Counter(len(data[mjd][WAVELENGTH]) for mjd in used_mjds)
    ref_len = length_counts.most_common(1)[0][0]
    ref_mjd = next(m for m in used_mjds if len(data[m][WAVELENGTH]) == ref_len)
    wave_grid = data[ref_mjd][WAVELENGTH]

    s2n_sqrd_sum = np.sum([data[mjd][S2N] ** 2 for mjd in used_mjds])

    coadd_spec = 0
    for mjd in used_mjds:
        cur_v = rv_out_df[rv_out_df.MJD == mjd][rv_name].values[0]
        weight_s2n = data[mjd][S2N] ** 2 / s2n_sqrd_sum
        shift_spec = interp1d(
            data[mjd][WAVELENGTH] * (1.0 - cur_v / C_LIGHT),
            np.nan_to_num(data[mjd][SCI_NORM]),
            bounds_error=False, fill_value=1.0,
            kind=intr_kind,
        )(wave_grid)
        coadd_spec += weight_s2n * shift_spec

    return {WAVELENGTH: wave_grid, SCI_NORM: coadd_spec}
