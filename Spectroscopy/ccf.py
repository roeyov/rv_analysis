import argparse
import ast
import os
import yaml

import pandas as pd
import itertools
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')

from spectrasDrawer import load_all_spectra, load_templates, load_template
from FilesCollector import find_files_with_strings, load_elements_list
from MeanRVCalculater import process_df, dict_to_df
from constants import *

input_df = marker_dict = meta_data = None
S2N_REGION = (4500,4550)
VELOCITY_REGION = (-400,400)
S2N = "SNR"
C_LIGHT = 299792.458

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

    if input_df.plot_first is True or input_df.plot_all is True:
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
    if input_df.path_to_meta_data_csv != "":
        rot_vel = meta_data[meta_data.ID == star_name].vsini.values[0]
        broadened_template = rotational_broadening(chosen_temp[WAVELENGTH], chosen_temp[SCI_NORM], rot_vel)
        template = interp1d(chosen_temp[WAVELENGTH], broadened_template, bounds_error=False,
            fill_value=1., kind=input_df.intr_kind)(wave_grid_log)
        # template = interp1d(chosen_temp[WAVELENGTH], chosen_temp[SCI_NORM], bounds_error=False,
        #     fill_value=1., kind=input_df.intr_kind)(wave_grid_log)
        # template = rotational_broadening(wave_grid_log, template, rot_vel)

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
        # out_dict[mjd]["merged RV"] = CCFeval[0]
        # out_dict[mjd]["merged RVsig"] = CCFeval[1]

    df = dict_to_df(out_dict)
    result_df = process_df(df, sigma_clip=3, verbose=True)
    # print(out_dict)
    # Initialize data storage
    t = []
    v_dict = {}
    sig_v_dict = {}
    t_dict = {}
    # Extract RV values for each MJD
    s2n_sqrd_sum = np.sum([data[mjd][S2N] ** 2 for mjd in data.keys()])
    coadd_spec = 0
    wave_grid = data[first_mjd][WAVELENGTH]

    for mjd in sorted(out_dict.keys()):
        try:
            for rv_key in out_dict[mjd]:
                if "RV" in rv_key and "sig" not in rv_key:
                    if out_dict[mjd].get(rv_key + "sig", 0)  < 0 : continue
                    v_dict.setdefault(rv_key, []).append(out_dict[mjd][rv_key])
                    sig_v_dict.setdefault(rv_key, []).append(out_dict[mjd].get(rv_key + "sig", 0))
                    t_dict.setdefault(rv_key, []).append(mjd)

            cur_v = out_dict[mjd]["merged RV"]
            weight_s2n = data[mjd][S2N] ** 2 / s2n_sqrd_sum
            shift_spec = interp1d(data[mjd][WAVELENGTH] * (1. - cur_v / C_LIGHT),
                                  np.nan_to_num(data[mjd][SCI_NORM]),
                                  bounds_error=False, fill_value=1.,
                                  kind=input_df.intr_kind)(wave_grid)
            coadd_spec += weight_s2n * shift_spec

            t.append(mjd)
        except KeyError:
            continue

    # Plot measured RVs
    if True:
        fig, ax = plt.subplots()
        for rv_key in sorted(v_dict.keys()):
            ax.errorbar(t_dict[rv_key], v_dict[rv_key] , yerr=sig_v_dict[rv_key], fmt=marker_dict[rv_key][0],
                        label=rv_key,c=marker_dict[rv_key][1], alpha=0.51)

        ax.set_xlabel('MJD [d]')
        ax.set_ylabel(r'$\Delta$RV [km/s]')
        ax.set_title(r'{} '.format(star_name))
        ax.legend()
        plt.show(block=True)
    coadd_spec_dict = {WAVELENGTH:wave_grid, SCI_NORM:coadd_spec}
    return out_dict,coadd_spec_dict

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


def make_marker_dict(windows):
    global marker_dict
    marker_styles = itertools.cycle(['o', 's', '^', 'D', 'v', '<', '>'])
    color_styles = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'k'])

    marker_dict = {}
    for window, marker,color in zip(windows+["merged"], marker_styles, color_styles):
        marker_dict["{} RV".format(window)] = marker, color


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

    make_marker_dict(sorted(lines_to_windows.keys()))

    original_plot_first = input_df.plot_first
    final_dict = {}
    for star in sorted(elements):
        print('Star {}'.format(star))
        a = load_all_spectra(all_files[star], MJD_MID, WAVELENGTH, SCI_NORM)

        cc_result, coadd_spec = cross_cor(a,star, template[star], lines_to_windows,seperate_speed=input_df.seperate_speed)

        final_dict[star] = cc_result

        x = pd.DataFrame(cc_result)
        y = pd.DataFrame(coadd_spec, columns=[WAVELENGTH, SCI_NORM])
        if input_df.path_to_output != '':
            y.to_csv(os.path.join(input_df.path_to_output, star + "_CoAdded.csv"), index=False, sep=' ')
            x.to_csv(os.path.join(input_df.path_to_output, star + "_CCF_RVs.csv"), index=False, sep=' ')
        input_df.plot_first = original_plot_first

if __name__ == '__main__':
    main()


