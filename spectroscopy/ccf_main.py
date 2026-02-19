"""
spectroscopy.ccf_main — Entry point for the CCF radial-velocity pipeline.

Usage:
    python -m spectroscopy.ccf_main [--input_file path/to/ccf_input.yaml]
"""

import argparse
import ast
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from itertools import cycle

from spectroscopy.constants import (
    WAVELENGTH, SCI_NORM, S2N, MJD_MID, SNR_PPL, EPOCH_ID,
)
from spectroscopy.ccf_core import cross_cor
from spectroscopy.coaddition import create_coadded_spectra
from spectroscopy.plotting import (
    plot_rv_vs_mjd, plot_rv_vs_mjd_combined,
)
from spectroscopy.mean_rv import calculate_weighted_rv_with_flags
from spectroscopy.spectra_drawer import load_all_spectra, load_templates, load_template
from spectroscopy.files_collector import find_files_with_strings, load_elements_list
from spectroscopy.plot_extrema_spectra import save_minmax_overlay_plots


def parse_list_of_lists(arg):
    """Parse a string like '[[a1,a2],[b1,b2]]' into a Python list of lists."""
    try:
        val = ast.literal_eval(arg)
        if not isinstance(val, list) or not all(isinstance(s, list) for s in val):
            raise ValueError
        return val
    except (SyntaxError, ValueError):
        raise argparse.ArgumentTypeError(
            "Argument must be in the form [[a1,a2],[b1,b2],...] (numbers only)."
        )


def load_input_from_yaml(yaml_file):
    """Read a YAML config file into a pd.Series."""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        return pd.Series(data)
    except FileNotFoundError:
        print(f"Error: File not found: {yaml_file}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def make_marker_dict():
    """Return a dict mapping RV column names to (marker, color) tuples."""
    line_names = [
        'H_Gamma', 'H_Delta', 'H_Epsilon', 'HeI_4471', 'HeI_4388',
        'HeI+HeII_4026', 'HeII_4542', 'HeII_4200', 'Median', 'Mean',
        'NII_4447', 'NII_4440',
    ]
    markers = cycle(['o', 's', 'v', '^', 'D', '*', 'p', 'h'])
    colors = cycle(['blue', 'green', 'red', 'purple', 'orange', 'brown',
                    'cyan', 'magenta'])

    md = {}
    for window, marker, color in zip(line_names + ["merged", "Mean"],
                                      markers, colors):
        md[f"{window} RV"] = marker, color
    return md


def _list_absolute_paths(directory, str_identifier):
    """List absolute paths of files matching *str_identifier* in *directory*."""
    try:
        return [
            os.path.join(os.path.abspath(directory), item)
            for item in os.listdir(directory)
            if str_identifier in item
        ]
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def main():
    TEMPLATE_INPUT_FILE = "./ccf_input.yaml"

    parser = argparse.ArgumentParser(description="CCF radial-velocity pipeline")
    parser.add_argument(
        '--input_file', type=str,
        help=f'YAML config (default: {TEMPLATE_INPUT_FILE})',
    )
    args = parser.parse_args()

    fp_yaml = args.input_file if args.input_file else TEMPLATE_INPUT_FILE
    cfg = load_input_from_yaml(fp_yaml)

    # Convert cross-correlation ranges
    lines_to_windows = cfg.cross_cor_ranges
    lines_to_windows = {
        key: tuple(map(float, value.strip("()").split(", ")))
        for key, value in lines_to_windows.items()
    }

    # Discover objects
    if cfg.path_to_list_of_objects:
        elements = load_elements_list(cfg.path_to_list_of_objects)
        all_files = find_files_with_strings(
            elements, cfg.path_to_observations, cfg.str_identifier,
        )
    elif cfg.list_of_objects:
        elements = cfg.list_of_objects
        all_files = find_files_with_strings(
            elements, cfg.path_to_observations, cfg.str_identifier,
        )
    else:
        elements = ["object"]
        all_files = {"object": _list_absolute_paths(
            cfg.path_to_observations, cfg.str_identifier,
        )}

    # Load templates
    if cfg.template_path == '':
        template = {star: None for star in elements}
    elif os.path.isfile(cfg.template_path):
        fixed = load_template(cfg.template_path, WAVELENGTH, SCI_NORM)
        template = {star: fixed for star in elements}
    elif os.path.isdir(cfg.template_path):
        template = load_templates(cfg.template_path, elements, WAVELENGTH, SCI_NORM)
    else:
        template = {"object": None}

    if cfg.path_to_output != '':
        os.makedirs(cfg.path_to_output, exist_ok=True)

    meta_data = None
    if cfg.path_to_meta_data_csv != '':
        meta_data = pd.read_csv(cfg.path_to_meta_data_csv,
                                usecols=cfg.columns_to_load)

    marker_dict = make_marker_dict()
    visuals_path = os.path.join(cfg.path_to_output, 'visuals')
    cfg["original_plot_first"] = cfg.plot_first

    for star in sorted(elements):
        print(f'Star {star}')
        if star not in template:
            continue
        a = load_all_spectra(all_files[star], MJD_MID, WAVELENGTH, SCI_NORM)

        cc_result = cross_cor(
            a, star, template[star], lines_to_windows, cfg,
            meta_data=meta_data, seperate_speed=cfg.seperate_speed,
        )
        plot_rv_vs_mjd(cc_result, star, marker_dict=marker_dict)

        mean_calc = calculate_weighted_rv_with_flags(cc_result)
        plot_rv_vs_mjd(mean_calc, star, plot_only=["Mean"], filter=True,
                       marker_dict=marker_dict)
        plot_rv_vs_mjd(mean_calc, star, filter=True, marker_dict=marker_dict)
        plot_rv_vs_mjd_combined(
            cc_result_df=cc_result, mean_calc_df=mean_calc,
            name=star, out=visuals_path, marker_dict=marker_dict,
        )

        coadd_spec = create_coadded_spectra(
            a, cc_result, rv_name="Mean RV", intr_kind=cfg.intr_kind,
        )
        x = mean_calc
        x[S2N] = cc_result[S2N]
        y = pd.DataFrame(coadd_spec, columns=[WAVELENGTH, SCI_NORM])
        plt.close('all')

        if cfg.path_to_output != '':
            y.to_csv(os.path.join(cfg.path_to_output, star + "_CoAdded.csv"),
                     index=False, sep=',')
            x.to_csv(os.path.join(cfg.path_to_output, star + "_CCF_RVs.csv"),
                     index=False, sep=',')
            save_minmax_overlay_plots(
                a, mean_calc, lines_to_windows, star,
                out_root=visuals_path,
                mjd_col=MJD_MID, wl_col=WAVELENGTH, flux_col=SCI_NORM,
            )
        cfg.plot_first = cfg["original_plot_first"]


if __name__ == '__main__':
    main()
