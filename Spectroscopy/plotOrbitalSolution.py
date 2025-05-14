import os
import glob
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import imageio.v2 as imageio

from constants import *
from spectrasDrawer import find_files_with_strings, load_elements_list, load_all_spectra
from BinaryPrediction.ModelEvaluatorSingle import get_bloem_object_name, load_final_data_from_ccf_out


def read_spectrum(fits_path):
    """
    Read wavelength and flux arrays from a FITS spectrum.
    """
    hdul = fits.open(fits_path)
    try:
        flux = hdul[0].data.astype(float)
        hdr = hdul[0].header
        wave = hdr['CRVAL1'] + np.arange(flux.size) * hdr['CDELT1']
    finally:
        hdul.close()
    return wave, flux


def solve_kepler(M, e, n_iter=8):
    """
    Solve Kepler's equation E - e*sin(E) = M for eccentric anomaly E.
    """
    E = M.copy()
    for _ in range(n_iter):
        E += (M - (E - e * np.sin(E))) / (1 - e * np.cos(E))
    return E


def compute_rv_model(phases, K, e, omega_deg, gamma):
    """
    Compute the radial velocity model for given orbital parameters.
    """
    M = 2 * np.pi * phases
    E = solve_kepler(M, e)
    # true anomaly
    f = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )
    omega = np.deg2rad(omega_deg)
    return gamma + K * (np.cos(f + omega) + e * np.cos(omega))


def create_frame(wave, flux, obj_name, mjd, phase,
                 phase_grid, rv_curve, data_phases, data_rvs, orbit_params):
    """
    Build a single image frame: top spectrum, bottom RV curve + points.
    """
    K, e, omega, gamma = orbit_params
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [2, 1]}
    )
    # spectrum
    center = 4471.0
    delta = 15.0
    # Option A: filter before plotting
    mask = (wave >= center - delta) & (wave <= center + delta)
    ax1.plot(wave[mask], flux[mask])

    # Option B: plot everything but then zoom in
    ax1.plot(wave, flux)
    ax1.set_xlim(center - delta, center + delta)

    # in either case:
    ax1.set_xlabel("Wavelength (Å)")
    ax1.set_ylabel("Flux")
    ax1.set_title(f"{obj_name} — MJD {mjd:.3f}, phase={phase:.3f}")
    # RV orbit
    ax2.plot(phase_grid, rv_curve, lw=1)
    ax2.scatter(data_phases, data_rvs, s=10, label="RV data")
    ax2.scatter(
        [phase],
        compute_rv_model(np.array([phase]), K, e, omega, gamma),
        c='red', s=50, label="current phase"
    )
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("RV (km/s)")
    ax2.legend(loc='best', fontsize='small')

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def make_spectra_orbit_gif(
    spec_files,
    object_name,
    rv_csv_path,
    orbit_results,
    output_path,
    duration=0.5
):
    """
    Generate an animated GIF showing spectra and orbital RV solution.

    Parameters
    ----------
    spec_files : list of str
        Paths to FITS spectra for this object, ordered by MJD.
    object_name : str
        Identifier for the object.
    rv_csv_path : str
        CSV file with columns ['MJD', 'RV'] for the object.
    orbit_results : pandas.DataFrame
        Table of orbital-fit rows with columns ['MJD', 'phis', 'K', 'e',
        'omega', 'gamma', 'T0', 'period'] (one row per spectrum).
    output_path : str
        File path for the output GIF (including .gif).
    duration : float
        Seconds per frame in the GIF.
    """
    # load RV data
    rv_data = pd.read_csv(rv_csv_path, sep=' ')

    # extract orbital parameters
    K = orbit_results['K1_value'].iloc[0]
    e = orbit_results['Eccentricity_value'].iloc[0]
    omega = orbit_results['OMEGA_rad_value'].iloc[0]
    gamma = orbit_results['GAMMA_value'].iloc[0]
    T0 = orbit_results['T0_value'].iloc[0]
    period = orbit_results['Period_value'].iloc[0]
    orbit_params = (K, e, omega, gamma)

    # prepare model curve
    phase_grid = np.linspace(0, 1, 200)
    rv_curve = compute_rv_model(phase_grid, *orbit_params)

    # compute data phases for scatter
    phis =  orbit_results['phs']
    data_rvs = rv_data['Mean RV']
    mjd = rv_data['MJD'].values


    # apply to both
    phis = np.array([float(x) for x in phis.values[0].strip('[]').split()])
    idx = np.argsort(phis)  # → array([1, 3, 0, 2])

    phis_sorted = phis[idx]  # → array([1.5, 2.1, 3.2, 4.8])
    mjd_sorted = mjd[idx]
    rvs_sorted = data_rvs[idx]
    # write frames
    writer = imageio.get_writer(output_path, mode='I', duration=duration)
    for idx in range(len(mjd_sorted)):
        cur_mjd=  mjd_sorted[idx]
        cur_phase = phis_sorted[idx]

        # read matching spectrum
        spec_match = [f for f in spec_files if f"{cur_mjd:.3f}" in str(f)]
        if not spec_match:
            continue
        wave, flux = spec_files[spec_match[0]][WAVELENGTH], spec_files[spec_match[0]][SCI_NORM]

        # create frame & append
        frame = create_frame(
            wave, flux,
            object_name, cur_mjd, cur_phase,
            phase_grid, rv_curve,
            phis_sorted, rvs_sorted,
            orbit_params
        )
        writer.append_data(frame)
    writer.close()

    print(f"Saved GIF for {object_name} at {output_path}")

ccf_out_dir = "/Users/roeyovadia/Roey/Masters//Reasearch/scriptsOut/CCF/ostars_sb1_from_coAdded/"
lmfit_res = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/OrbitalFitting/sb1_ostars_coAdded/lmfit_summary.csv"
lmfit_file = pd.read_csv(lmfit_res)
elements = ["7-069"]
fits_suf = FITS_SUF_COMBINED

all_files = find_files_with_strings(elements, DATA_RELEASE_4_PATH, fits_suf)
for star in elements:
    a = load_all_spectra(all_files[star], MJD_MID, WAVELENGTH, SCI_NORM)
    list_of_csvs = glob.glob(os.path.join(ccf_out_dir, f'*{star}_CCF_RVs.csv'))
    rv_csv_path = list_of_csvs[0]
    orbit_results = lmfit_file[lmfit_file['star_name'] == star]
    make_spectra_orbit_gif(a,star,rv_csv_path,orbit_results,'./example.gif',duration=3)