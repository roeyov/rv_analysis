import os
from datetime import datetime

from spectroscopy.files_collector import find_files_with_strings, load_json_elements, load_elements_list
import numpy as np
from pandas import read_csv
from spectroscopy.constants import *
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def draw_spectra(data, header_name):
    for time, spectra in data.items():
        draw_single_spectra(spectra[WAVELENGTH], spectra[SCI_NORM] ,time)
    plt.title(header_name)
    plt.show()

def animate_data(
    data,
    filename="animation",
    outdir="",
    WAVELENGTH_REGION=None,
    interval_ms=800,   # time between frames in the interactive animation (ms)
    fps=1,             # frames per second in the saved GIF (lower = slower)
    ylim=(0.55, 1.1),
):
    """
    Uniform-timing animation (all frames same spacing).

    - Interactive playback speed is controlled by `interval_ms`.
    - Saved GIF speed is controlled by `fps` (writer uses constant fps).

    The animation shows:
    - Top title: object name, mid-exposure time, and (if available) epoch & S/N.
    - Axes labels: wavelength and normalized flux with units.
    """

    if not data:
        return

    if WAVELENGTH_REGION is None:
        first = data[next(iter(data))]
        WAVELENGTH_REGION = [
            (float(np.min(first[WAVELENGTH])), float(np.max(first[WAVELENGTH])))
        ]

    # Sort data by time to ensure consistent ordering
    sorted_times = sorted(data.keys())
    data_series = [data[t] for t in sorted_times]

    n_regions = len(WAVELENGTH_REGION)
    fig, axes = plt.subplots(
        n_regions, 1, figsize=(8, 6 * n_regions), squeeze=False
    )
    axes = axes.flatten()

    # --- helper for setting axes style ---
    def style_axis(ax, start_wv, end_wv):
        ax.set_xlim(start_wv, end_wv)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Wavelength [Å]", fontsize=14)
        ax.set_ylabel("Normalized flux", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)

    def init():
        for ax, (start_wv, end_wv) in zip(axes, WAVELENGTH_REGION):
            ax.clear()
            style_axis(ax, start_wv, end_wv)
        # A generic title; will be updated per frame
        fig.suptitle(filename, fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return []

    def update(frame):
        current_data = data_series[frame]
        wv = current_data[WAVELENGTH]
        sci = current_data[SCI_NORM]

        time_val = sorted_times[frame]

        # Optional extra info if present in the dict:
        snr   = current_data.get(SNR_PPL, None)
        epoch = current_data.get(EPOCH_ID, None)

        extra_bits = []
        if epoch is not None:
            extra_bits.append(f"epoch {epoch}")
        if snr is not None:
            try:
                extra_bits.append(f"S/N ≈ {snr:.0f}")
            except Exception:
                pass

        extra_str = ""
        if extra_bits:
            extra_str = " (" + ", ".join(extra_bits) + ")"

        fig.suptitle(
            f"{filename} — MJD = {time_val:.5f}{extra_str}",
            fontsize=18
        )

        for ax, (start_wv, end_wv) in zip(axes, WAVELENGTH_REGION):
            ax.clear()
            style_axis(ax, start_wv, end_wv)
            mask = (wv >= start_wv) & (wv <= end_wv)
            ax.plot(wv[mask], sci[mask], "-", linewidth=1.5)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=len(data_series),
        init_func=init,
        blit=False,
        interval=interval_ms,
        repeat=True
    )

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        save_path = os.path.join(outdir, f"{filename}.gif")

        from matplotlib.animation import PillowWriter
        # lower fps => slower animation in the GIF
        ani.save(save_path, writer=PillowWriter(fps=fps))
        print(f"Animation saved as {save_path}")

    plt.close(fig)





def draw_single_spectra(w,p,t):
    plt.plot(w,p, label=t)

def load_template(filename, x_name, y_name):
    try:
        data = np.loadtxt(filename, delimiter=',')
    except ValueError:
        data = np.loadtxt(filename, skiprows=1, delimiter=',')
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
            c += 1
        else:
            ret_temps[element] = None
            print(f"No template found for {element}. using first MJD sample as template")
    print(f"Found {c} templates in {template_dir}")
    return ret_temps

def get_epoch_id(fp):
    return int(fp.split("_")[-2])

def get_key_from_header(fits_path, key):
    """Return header value for *key* from the first HDU that contains it.
    Casts to float when possible; returns None if not found or not numeric.
    """
    try:
        with fits.open(fits_path) as hdul:
            for hdu in hdul:
                hdr = getattr(hdu, "header", None)
                if hdr is None:
                    continue
                if key in hdr:
                    val = hdr[key]
                    # Try to coerce to float
                    try:
                        fval = float(val)
                        # Guard against NaN or inf
                        if np.isfinite(fval):
                            return fval
                        return None
                    except Exception:
                        return None
    except Exception as e:
        # Could be a corrupted FITS, unreadable file, etc.
        return None
    return None

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
                ret_spectra[time] = {x_name: x , y_name: y, SNR_PPL: get_key_from_header(fp, SNR_PPL), EPOCH_ID: get_epoch_id(fp) }
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
    INTERESTING_WAVELENGTH = [4471]
    WL_RADIUS = 10
    WAVELENGTH_REGION = [(a-WL_RADIUS,a+WL_RADIUS) for a in INTERESTING_WAVELENGTH]

    json_file_key = 'Sample O + 10 early BVs'  # Update to the directory you want to search
    elements = load_elements_list("/Users/roeyovadia/Documents/Data/BLOeM_Data/lists/All_ostars.txt.rtf")
    elements = ["BLOeM_2-024","BLOeM_1-078", "BLOeM_2-085"]
    fits_suf = FITS_SUF_COMBINED

    all_files = find_files_with_strings(elements, DATA_RELEASE_4_PATH, fits_suf)

    for star in elements:
        a = load_all_spectra(all_files[star], MJD_MID, WAVELENGTH,SCI_NORM)
        # Run the animation
        animate_data(
            a,
            filename=star,
            interval_ms=1200,  # 1.2 seconds per frame (interactive)
            fps=5,  # 1 frame per second in the GIF
            outdir=r"/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/spectrasDrawer/seminar_{}".format(formatted_date),
            WAVELENGTH_REGION=WAVELENGTH_REGION
        )



