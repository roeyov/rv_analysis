import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

# ==========================================
# Constants and Mock Data Generation
# (You can replace this section with your actual data loading)
# ==========================================
C_LIGHT = 299792.458  # km/s


def create_mock_spectrum(wavelengths, centers, depths, sigma=1.0):
    flux = np.ones_like(wavelengths)
    for c, d in zip(centers, depths):
        # Gaussian absorption lines
        flux -= d * np.exp(-0.5 * ((wavelengths - c) / sigma) ** 2)
    return flux


# 1. Setup Wavelength Grid
wgl = np.linspace(5000, 5100, 2000)  # Angstroms

# 2. Create "Observed" Flux (Target) - Static
# Let's pretend the star is moving away at roughly +30 km/s
true_rv_shift = 30.0  # km/s
obs_centers_rest = [5020, 5050, 5080]
obs_centers_shifted = [c * (1 + true_rv_shift / C_LIGHT) for c in obs_centers_rest]
obs_flux = create_mock_spectrum(wgl, obs_centers_shifted, [0.4, 0.6, 0.3], sigma=0.8)
# Add some noise for realism
obs_flux += np.random.normal(0, 0.02, size=len(wgl))

# 3. Create "Template" Flux - Moving
# Template is at rest (0 km/s)
temp_flux = create_mock_spectrum(wgl, obs_centers_rest, [0.5, 0.8, 0.4], sigma=0.5)

# 4. Define Velocity Search Range
vel_range = np.linspace(-100, 150, 251)  # km/s scan range


# ==========================================
# THE ANIMATION FUNCTION
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d


def animate_ccf_process(wgl, obs_flux, temp_flux, vel_range, output_filename=None, fit_rif=0.8):
    # --- 1. Pre-Processing & Pre-Calculation ---
    # Normalize inputs once
    if np.abs(np.mean(obs_flux)) > 1e-5:
        obs_flux = obs_flux - np.mean(obs_flux)
    static_norm_temp = temp_flux - np.mean(temp_flux)

    n_points = len(obs_flux)
    std_obs = np.std(obs_flux)

    # Setup Interpolator
    template_interpolator = interp1d(wgl, static_norm_temp,
                                     kind='cubic',
                                     bounds_error=False,
                                     fill_value=0.0)

    # Pre-calculate the full CCF array (Math)
    print("Pre-calculating CCF for fit...")
    pre_ccf_values = []
    for v in vel_range:
        wgl_rest = wgl / (1 + v / 299792.458)
        shifted_temp = template_interpolator(wgl_rest)

        numerator = np.sum(obs_flux * shifted_temp)
        denominator = std_obs * np.std(shifted_temp) * n_points
        val = 0 if denominator == 0 else numerator / denominator
        pre_ccf_values.append(val)

    pre_ccf_values = np.array(pre_ccf_values)

    # --- 2. Parabola Logic (fit_rif) ---
    ind_max = np.argmax(pre_ccf_values)
    ccf_max = pre_ccf_values[ind_max]
    threshold = fit_rif * ccf_max

    try:
        # Find Left Edge
        left_candidates = np.where(pre_ccf_values[:ind_max] < threshold)[0]
        ind_fit_1 = left_candidates[-1] + 1 if len(left_candidates) > 0 else 0

        # Find Right Edge
        right_candidates = np.where(pre_ccf_values[ind_max:] < threshold)[0]
        ind_fit_2 = (right_candidates[0] + ind_max) - 1 if len(right_candidates) > 0 else len(pre_ccf_values) - 1

        # Safety check for minimum points
        if ind_fit_2 - ind_fit_1 < 2:
            ind_fit_1 = max(0, ind_max - 2)
            ind_fit_2 = min(len(pre_ccf_values) - 1, ind_max + 2)

        x_fit_data = vel_range[ind_fit_1: ind_fit_2 + 1]
        y_fit_data = pre_ccf_values[ind_fit_1: ind_fit_2 + 1]

        # Fit parabola
        a, b, c = np.polyfit(x_fit_data, y_fit_data, 2)
        v_max_fit = -b / (2 * a)

        # Generate smooth parabola curve for plotting
        fine_v_grid = np.linspace(x_fit_data[0], x_fit_data[-1], 50)
        fine_parabola = a * fine_v_grid ** 2 + b * fine_v_grid + c

    except Exception as e:
        print(f"Fit failed: {e}")
        v_max_fit = np.nan
        fine_v_grid = []
        fine_parabola = []
        x_fit_data = [vel_range[-1]]  # Dummy to prevent crash

    # --- 3. Setup Figure ---
    fig, (ax_spec, ax_ccf) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(hspace=0.3)

    # Top Plot
    ax_spec.set_title(r"Spectral Alignment - BLOeM 1-078 - HeI 4026 $\AA$", fontsize=14)
    ax_spec.plot(wgl, obs_flux, 'k-', alpha=0.5, label='Observed', linewidth=1)
    line_template, = ax_spec.plot([], [], 'r-', label='Template', linewidth=1.5, alpha=0.8)
    ax_spec.set_xlim(wgl[0], wgl[-1])
    y_range = np.max(obs_flux) - np.min(obs_flux)
    ax_spec.set_ylim(np.min(obs_flux) - 0.2 * y_range, np.max(obs_flux) + 0.2 * y_range)
    ax_spec.legend(loc='upper right')

    # Bottom Plot
    ax_ccf.set_title("Cross-Correlation Function", fontsize=14)
    ax_ccf.set_xlabel('Radial Velocity [km/s]')
    ax_ccf.set_ylabel('CCF')
    ax_ccf.set_xlim(vel_range[0], vel_range[-1])
    y_min, y_max = np.min(pre_ccf_values), np.max(pre_ccf_values)
    ax_ccf.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    # 1. The History Line (Solid Blue, No Markers)
    line_ccf_history, = ax_ccf.plot([], [], 'b-', linewidth=1.5)

    # 2. The Current Head (Single Red Dot)
    point_ccf_current, = ax_ccf.plot([], [], 'ro', markersize=6)

    # 3. The Fit Elements
    line_parabola, = ax_ccf.plot([], [], 'orange', linewidth=2.5, label='Parabola Fit', alpha=0.8)
    line_vertical = ax_ccf.axvline(x=v_max_fit, color='green', linestyle='--', alpha=0.0)
    text_vel = ax_ccf.text(0.05, 0.9, '', transform=ax_ccf.transAxes, color='green', fontweight='bold')

    def init():
        line_template.set_data([], [])
        line_ccf_history.set_data([], [])
        point_ccf_current.set_data([], [])
        line_parabola.set_data([], [])
        line_vertical.set_alpha(0.0)
        text_vel.set_text('')
        return line_template, line_ccf_history, point_ccf_current, line_parabola, line_vertical, text_vel

    def update(frame_idx):
        current_vel = vel_range[frame_idx]

        # Update Spectra
        wgl_rest = wgl / (1 + current_vel / 299792.458)
        shifted_temp = template_interpolator(wgl_rest)
        line_template.set_data(wgl, shifted_temp)

        # Update CCF Line
        # FIX: Slice the array instead of appending. This prevents the "loop" line.
        x_data = vel_range[:frame_idx + 1]
        y_data = pre_ccf_values[:frame_idx + 1]

        line_ccf_history.set_data(x_data, y_data)

        # Update Current Dot
        point_ccf_current.set_data([current_vel], [pre_ccf_values[frame_idx]])

        # Reveal Fit
        if len(fine_v_grid) > 0 and current_vel >= x_fit_data[-1]:
            line_parabola.set_data(fine_v_grid, fine_parabola)
            line_vertical.set_alpha(1.0)
            text_vel.set_text(f"RV: {v_max_fit:.2f} km/s")

        return line_template, line_ccf_history, point_ccf_current, line_parabola, line_vertical, text_vel

    print("Generating animation...")
    # repeat=False helps prevent confusion, but slicing method above fixes it even if True
    ani = FuncAnimation(fig, update, frames=len(vel_range),
                        init_func=init, blit=True, interval=50, repeat=False)

    if output_filename:
        try:
            ani.save(output_filename, writer='ffmpeg', fps=30)
        except:
            ani.save(output_filename.replace('.mp4', '.gif'), writer='pillow', fps=20)
    else:
        plt.show()



# ==========================================
# Main execution block
# ==========================================
if __name__ == "__main__":
    # Run the animation with mock data created at the top
    # Set output_filename to e.g., 'ccf_animation.gif' or 'ccf_animation.mp4' to save file
    # Set to None to just view on screen.
    animate_ccf_process(wgl, obs_flux, temp_flux, vel_range, output_filename=None)