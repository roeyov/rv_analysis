"""
spectroscopy — CCF radial-velocity pipeline.

Submodules:
    constants          Column names and file paths
    ccf_core           CCF algorithm (no globals)
    equivalent_width   Equivalent width calculations
    broadening         Rotational broadening kernel
    coaddition         S/N-weighted spectral coaddition
    plotting           RV vs MJD plots (mpl + plotly)
    ccf_main           Entry point + YAML config loader
    mean_rv            Weighted RV + quality flags
    files_collector    File discovery utilities
    spectra_drawer     Spectra loading
    plot_extrema_spectra  Min/max overlay plots
    plot_orbital_solution Orbital solution visualization
    plot_extreme_rv    Min/max RV overlay plotting
    ccf_animation      CCF process animation
"""
