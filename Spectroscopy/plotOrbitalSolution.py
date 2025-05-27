import os
import glob
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import imageio.v2 as imageio

# import modular functions from your RV utilities
from emceeOmLMFITexe import (
    extract_observations,
    compute_rv_curve_phase,
)

from constants import *
from spectrasDrawer import find_files_with_strings, load_elements_list, load_all_spectra
from BinaryPrediction.ModelEvaluatorSingle import get_bloem_object_name, load_final_data_from_ccf_out


def read_spectrum(fits_path):
    """
    Read wavelength and flux arrays from a FITS spectrum.
    """
    with fits.open(fits_path) as hdul:
        flux = hdul[0].data.astype(float)
        hdr = hdul[0].header
        wave = hdr['CRVAL1'] + np.arange(flux.size) * hdr['CDELT1']
    return wave, flux


def create_frame(wave, flux, obj_name, mjd, phase,
                 phase_grid, rv_curve, data_phases, data_rvs, orbit_params,
                 wl_center=4471.0, wl_delta=15.0,
                 flux_ylim=None):
    """
    Build a single image frame: top spectrum around wl_center,
    bottom RV curve + points.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [2, 1]}
    )
    # plot spectrum around center wavelength
    mask = (wave >= wl_center - wl_delta) & (wave <= wl_center + wl_delta)
    ax1.plot(wave[mask], flux[mask], color='black')
    ax1.set_xlim(wl_center - wl_delta, wl_center + wl_delta)
    # fix Y-axis limits for all frames if provided
    if flux_ylim is not None:
        ax1.set_ylim(flux_ylim)
    ax1.set_xlabel("Wavelength (Å)")
    ax1.set_ylabel("Flux")
    ax1.set_title(f"{obj_name} — MJD {mjd:.3f}, phase={phase:.3f}")

    # plot RV orbit
    ax2.plot(phase_grid, rv_curve, lw=1, color='blue', label='Model')
    ax2.scatter(data_phases, data_rvs, s=10, color='gray', label="Data")
    ax2.scatter([phase], [np.interp(phase, phase_grid, rv_curve)],
                c='red', s=50, label="Current phase")
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
    Generate an animated GIF showing spectra and phase-folded RV model.

    Uses compute_rv_curve_phase from your RV utilities.
    """
    # load RV data
    rv_data = pd.read_csv(rv_csv_path, sep=' ')
    hjds = rv_data['MJD'].values
    data_rvs = rv_data['Mean RV'].values

    # extract orbital parameters
    K = orbit_results['K1_value'].iloc[0]
    e = orbit_results['Eccentricity_value'].iloc[0]
    omega = orbit_results['OMEGA_rad_value'].iloc[0]
    gamma = orbit_results['GAMMA_value'].iloc[0]
    T0 = orbit_results['T0_value'].iloc[0]
    period = orbit_results['Period_value'].iloc[0]

    # compute phase grid & model
    phase_grid, rv_curve = compute_rv_curve_phase(
        period, T0, e, gamma, K, omega
    )
    if np.isnan(rv_curve.all()):
        print("RV curve kepler solution did not converge")
        return
    # parse and sort observed phases
    phis_str = orbit_results['phs'].iloc[0]
    if type(phis_str) == float:
        print("?")
        return 

    phis = np.fromstring(phis_str.strip('[]'), sep=' ')
    idx = np.argsort(phis)
    phis_sorted = phis[idx]
    mjd_sorted = hjds[idx]
    rvs_sorted = data_rvs[idx]
    # build GIF
    writer = imageio.get_writer(output_path, mode='I', duration=duration)
    for mjd, phase, rv_obs in zip(mjd_sorted, phis_sorted, rvs_sorted):
        # find spectrum
        matches = [f for f in spec_files if f"{mjd:.3f}" in str(f)]
        if not matches:
            continue
        spec_path = matches[0]
        wave, flux = spec_files[spec_path][WAVELENGTH], spec_files[spec_path][SCI_NORM]

        frame = create_frame(
            wave, flux,
            object_name, mjd, phase,
            phase_grid, rv_curve,
            phis_sorted, rvs_sorted,
            (K, e, omega, gamma), flux_ylim=(0.7,1.1),wl_delta=10
        )
        writer.append_data(frame)
    writer.close()

    print(f"Saved GIF for {object_name} at {output_path}")


# Example usage
ccf_out_dir = "/Users/roeyovadia/Roey/Masters//Reasearch/scriptsOut/CCF/ostars_sb1_from_coAdded/"
lmfit_res = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/OrbitalFitting/sb1_ostars_coAdded/lmfit_summary.csv"
lmfit_file = pd.read_csv(lmfit_res)
elements = load_elements_list("/Users/roeyovadia/Documents/Data/lists/ostars_sb1.txt")

all_files = find_files_with_strings(elements, DATA_RELEASE_4_PATH, FITS_SUF_COMBINED)
for star in elements:
    star_split = star.split("_")
    star_number = star_split[0] if len(star_split) == 1 else star_split[1]
    spec_dict = load_all_spectra(all_files[star], MJD_MID, WAVELENGTH, SCI_NORM)
    rv_csvs = glob.glob(os.path.join(ccf_out_dir, f'*{star_number}_CCF_RVs.csv'))
    if len(rv_csvs) == 0:
        print("no data for ", star)
        continue
    rv_csv = rv_csvs[0]
    orbit_df = lmfit_file[lmfit_file['star_name'] == star_number]
    make_spectra_orbit_gif(
        spec_dict, star, rv_csv, orbit_df,
        f'/Users/roeyovadia/Roey/Masters//Reasearch/scriptsOut/orbitalGifs/{star}_spectra.gif', duration=10
    )
