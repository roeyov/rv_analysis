import os
from datetime import datetime

from FilesCollector import find_files_with_strings, load_json_elements, load_elements_list
import numpy as np
from pandas import read_csv
from constants import *
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def draw_spectra(data, header_name):
    for time, spectra in data.items():
        draw_single_spectra(spectra[WAVELENGTH], spectra[SCI_NORM] ,time)
    plt.title(header_name)
    plt.show()

def animate_data(data, filename="animation", interval_scaling=100, outdir='', WAVELENGTH_REGION=None):
    """
    Animates a sequence of plots based on the given data structure.
    The time between frames is proportional to the time differences between samples.

    Parameters:
        data (dict): A dictionary where keys are time points and values are
                     dictionaries with keys 'X' and 'Y' representing data for each time.
        filename (str): The base filename for the output animation file.
        interval_scaling (float): A scaling factor for frame durations.
        outdir (str): Output directory for saving the animation.
        WAVELENGTH_REGION (list of tuples): A list of (start, end) wavelength ranges
                                            to create subplots for each region.
    """

    if WAVELENGTH_REGION is None:
        # Default to a single region if not provided
        WAVELENGTH_REGION = [(min(data[next(iter(data))][WAVELENGTH]),
                               max(data[next(iter(data))][WAVELENGTH]))]

    # Sort data by time to ensure the animation plays in the correct order
    sorted_times = sorted(data.keys())
    intervals = [
        interval_scaling * (sorted_times[i + 1] - sorted_times[i])
        for i in range(len(sorted_times) - 1)
    ]
    if intervals:
        intervals.append(intervals[-1])  # Repeat the last interval for the final frame
    else:
        # In case there is only one time point, set a default interval
        intervals = [interval_scaling]

    # Extract data series in time order
    data_series = [data[t] for t in sorted_times]

    # Create one subplot per wavelength region
    n_regions = len(WAVELENGTH_REGION)
    fig, axes = plt.subplots(n_regions, 1, figsize=(6, 4*n_regions), squeeze=False)
    axes = axes.flatten()  # Flatten to a list for easy iteration

    def init():
        # Initialize all subplots
        for ax, (start_wv, end_wv) in zip(axes, WAVELENGTH_REGION):
            ax.clear()
            ax.set_xlim(start_wv, end_wv)
            ax.set_ylim(0.4, 1.1)
            ax.set_xlabel("Wavelength")
            ax.set_ylabel("Normalized Intensity")
        return []

    def update(frame):
        # Update each subplot for the given frame
        current_data = data_series[frame]
        wv = current_data[WAVELENGTH]
        sci = current_data[SCI_NORM]
        fig.suptitle(f"{filename} - Time: {sorted_times[frame]:.1f}")
        for ax, (start_wv, end_wv) in zip(axes, WAVELENGTH_REGION):
            ax.clear()
            # ax.set_title(f"Time: {sorted_times[frame]}")
            ax.set_xlim(start_wv, end_wv)
            ax.set_ylim(0.4, 1.1)
            # Filter data points within the region if needed
            mask = (wv >= start_wv) & (wv <= end_wv)
            ax.plot(wv[mask], sci[mask], '-')

        return []

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(data_series), init_func=init, blit=False
    )

    # Dynamically set the interval between frames
    def dynamic_interval():
        for i in range(len(data_series)):
            ani.event_source.interval = intervals[i] if i < len(intervals) else interval_scaling
            yield i
    ani.frame_seq = dynamic_interval()

    # Save the animation
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        save_path = f"{os.path.join(outdir, filename)}.gif"
        ani.save(save_path, writer='ffmpeg', fps=8)  # Adjust fps as needed
        print(f"Animation saved as {save_path}")
    plt.close(fig)


def draw_single_spectra(w,p,t):
    plt.plot(w,p, label=t)

def load_template(filename, x_name, y_name):
    try:
        data = np.loadtxt(filename, delimiter=' ')
    except ValueError:
        data = np.loadtxt(filename, skiprows=1, delimiter=' ')
    x = data[:, 0]
    y = data[:, 1]
    return {x_name: x, y_name: y}

def load_templates(template_dir,object_list, x_name, y_name):
    c = 0
    ret_temps = {}
    template_list = os.listdir(template_dir)
    for element in object_list:
        chosen_template_fp = ''
        for template in template_list:
            if element in template and "_CCF_RVs.csv" not in template:
                chosen_template_fp = os.path.join(template_dir, template)
                break
        if len(chosen_template_fp)>0:
            ret_temps[element] = load_template(chosen_template_fp, x_name, y_name)
        else:
            ret_temps[element] = None
            print(f"No template found for {element}. using first MJD sample as template")
    print(f"Found {c} templates in {template_dir}")
    return ret_temps

def load_all_spectra(files, time_name, x_name, y_name):

    if isinstance(files, str):
        files = [files]

    ret_spectra = {}
    for fp in files:
        if fp.endswith(".fits"):
            with fits.open(fp, ignore_missing_simple=True) as hdul:
                x = hdul[1].data[x_name]
                y = hdul[1].data[y_name]
                time = hdul[0].header[time_name]
                ret_spectra[time] = {x_name: x , y_name: y}
        else:
            parent_dir = os.path.dirname(fp)
            df = read_csv(os.path.join(parent_dir,'ObsDat.txt'), delimiter=' ')
            time = df[df.obsname == fp].MJD.values[0]
            data = np.loadtxt(fp)
            x = data[:, 0]
            y = data[:, 1]
            ret_spectra[time] = {x_name: x, y_name: y}

    return ret_spectra


if __name__ == '__main__':

    # Get the current date
    current_date = datetime.now()

    # Format the date as dd_mm_yy
    formatted_date = current_date.strftime("%d_%m_%y")
    # INTERESTING_WAVELENGTH = [4340,4471,4542,4101,4388,4026,3970,4200]
    INTERESTING_WAVELENGTH = [4340,4101,4471,4026,4388]
    WL_RADIUS = 10
    WAVELENGTH_REGION = [(a-WL_RADIUS,a+WL_RADIUS) for a in INTERESTING_WAVELENGTH]

    json_file_key = 'Sample O + 10 early BVs'  # Update to the directory you want to search
    elements = load_elements_list("/Users/roeyovadia/Documents/Data/lists/All_ostars.txt")
    elements = ["BLOeM_7-069" ]
    fits_suf = FITS_SUF_COMBINED

    all_files = find_files_with_strings(elements, DATA_RELEASE_4_PATH, fits_suf)

    for star in elements:
        a = load_all_spectra(all_files[star], MJD_MID, WAVELENGTH,SCI_NORM)
        # Run the animation
        animate_data(a, filename=star, interval_scaling=200, outdir=r"/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/spectrasDrawer/tomer_{}".format(formatted_date), WAVELENGTH_REGION=WAVELENGTH_REGION)

