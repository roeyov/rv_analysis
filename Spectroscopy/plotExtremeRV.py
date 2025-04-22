import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import matplotlib
matplotlib.use('TkAgg')
from constants import *
from spectrasDrawer import find_files_with_strings, load_elements_list, load_all_spectra
from BinaryPrediction.ModelEvaluatorSingle import get_bloem_object_name, load_final_data_from_ccf_out


def plot_spectra(max_spectra, max_wl, min_spectra, min_wl, wav_min=None, wav_max=None, star_name=''):
    """
    Plot two spectra on top of each other with an optional wavelength range.

    Parameters:
        max_spectra (np.ndarray): Array of flux values for the maximum spectrum.
        max_wl (np.ndarray): Array of wavelengths corresponding to the max spectrum.
        min_spectra (np.ndarray): Array of flux values for the minimum spectrum.
        min_wl (np.ndarray): Array of wavelengths corresponding to the min spectrum.
        wav_min (float, optional): Minimum wavelength for the plot. If None, uses full range.
        wav_max (float, optional): Maximum wavelength for the plot. If None, uses full range.
    """
    # If a wavelength range is provided, filter the data accordingly.
    if wav_min is not None and wav_max is not None:
        mask_max = (max_wl >= wav_min) & (max_wl <= wav_max)
        mask_min = (min_wl >= wav_min) & (min_wl <= wav_max)
        max_wl_filtered = max_wl[mask_max]
        max_spectra_filtered = max_spectra[mask_max]
        min_wl_filtered = min_wl[mask_min]
        min_spectra_filtered = min_spectra[mask_min]
    else:
        max_wl_filtered = max_wl
        max_spectra_filtered = max_spectra
        min_wl_filtered = min_wl
        min_spectra_filtered = min_spectra

    # Plot the spectra
    plt.figure(figsize=(10, 6))
    plt.plot(max_wl_filtered, max_spectra_filtered, label='Max Spectrum')
    plt.plot(min_wl_filtered, min_spectra_filtered, label='Min Spectrum')
    plt.xlabel("Wavelength")
    plt.ylabel("Normalized Flux")
    plt.title(f"{star_name} Spectra Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

ccf_out_dir = "/Users/roeyovadia/Roey/Masters//Reasearch/scriptsOut/CCF/ostars_sb1_from_coAdded/"
elements = load_elements_list("/Users/roeyovadia/Documents/Data/lists/ostars_sb1.txt")

# elements = ["BLOeM_5-104"]
fits_suf = FITS_SUF_COMBINED

all_files = find_files_with_strings(elements, DATA_RELEASE_4_PATH, fits_suf)

for star in elements:
    a = load_all_spectra(all_files[star], MJD_MID, WAVELENGTH, SCI_NORM)

    list_of_csvs = glob.glob(os.path.join(ccf_out_dir, f'*{star}_CCF_RVs.csv'))
    assert len(list_of_csvs) == 1
    path_to_csv = list_of_csvs[0]
    data = pd.read_csv(path_to_csv, sep=' ')
    star_name = get_bloem_object_name(path_to_csv)
    rvs, mjds, err_vs = load_final_data_from_ccf_out(data)
    min_rv_index, max_rv_index = np.argmin(rvs), np.argmax(rvs)


    max_spectra, max_wl = a[mjds[max_rv_index]][SCI_NORM], a[mjds[max_rv_index]][WAVELENGTH]
    min_spectra, min_wl = a[mjds[min_rv_index]][SCI_NORM], a[mjds[min_rv_index]][WAVELENGTH]

    plot_spectra(max_spectra, max_wl, min_spectra, min_wl, wav_min=None, wav_max=None, star_name=star)