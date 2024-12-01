from FilesCollector import find_files_with_strings, load_json_elements
from constants import *
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def draw_spectra(data, header_name):
    for time, spectra in data.items():
        draw_single_spectra(spectra[WAVELENGTH], spectra[SCI_NORM] ,time)
    plt.title(header_name)
    plt.show()

def animate_data(data, filename="animation", interval_scaling=100):
    """
    Animates a sequence of plots based on the given data structure.
    The time between frames is proportional to the time differences between samples.

    Parameters:
        data (dict): A dictionary where keys are time points and values are
                     dictionaries with keys 'X' and 'Y' representing data for each time.
        interval_scaling (float): A scaling factor for frame duration, default is 100.
    """
    # Sort data by time to ensure the animation plays in the correct order
    sorted_times = sorted(data.keys())
    intervals = [
        interval_scaling * (sorted_times[i + 1] - sorted_times[i])
        for i in range(len(sorted_times) - 1)
    ]
    intervals.append(intervals[-1])  # Repeat the last interval for the final frame

    # Extract X and Y data for plotting
    data_series = [data[t] for t in sorted_times]

    # Set up the figure and axis
    fig, ax = plt.subplots()

    def init():
        ax.clear()
        # Set consistent limits across all plots
        all_x = [item[WAVELENGTH] for item in data_series]
        ax.set_xlim(WAVELENGTH_REGION[0], WAVELENGTH_REGION[1])
        ax.set_ylim(0,1.1)
        return []

    def update(frame):
        ax.clear()
        current_data = data_series[frame]
        ax.plot(current_data[WAVELENGTH], current_data[SCI_NORM], '-')  # Plot the current frame data
        ax.set_ylim(0,1.1)
        ax.set_xlim(4300, 4400)

        ax.set_title(f"Time: {sorted_times[frame]}")
        return []

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(data_series), init_func=init, blit=False
    )

    # Dynamically set the interval between frames
    def dynamic_interval():
        for i in range(len(data_series)):
            ani.event_source.interval = intervals[i]
            yield i

    ani.frame_seq = dynamic_interval()
    save_path = f"{filename}.gif"
    ani.save(save_path, writer='ffmpeg', fps=10)  # Adjust fps as needed
    print(f"Animation saved as {save_path}")
    # plt.show()

def draw_single_spectra(w,p,t):
    plt.plot(w,p, label=t)


def load_all_spectra(files, time_name, x_name, y_name):
    ret_spectra = {}
    for fp in files:
        with fits.open(fp, ignore_missing_simple=True) as hdul:
            x = hdul[1].data[x_name]
            y = hdul[1].data[y_name]
            time = hdul[0].header[time_name]
            ret_spectra[time] = {x_name: x , y_name: y}

    return ret_spectra


WAVELENGTH_REGION = (4300,4400)

json_file_key = 'Sample O + 10 early BVs'  # Update to the directory you want to search
elements = load_json_elements(OSTARS_IDS_JSON, json_file_key)
fits_suf = FITS_SUF_COMBINED

all_files = find_files_with_strings(elements, DATA_RELEASE_3_PATH, fits_suf)

for star in elements:
    a = load_all_spectra(all_files[star], MJD_MID, WAVELENGTH,SCI_NORM)
    # Run the animation
    animate_data(a, star, interval_scaling=200, outdir = )