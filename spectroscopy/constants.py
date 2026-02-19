"""
Spectroscopy.constants — Column names and paths for the CCF / spectroscopy pipeline.

Only constants that are actually referenced in the codebase are kept here.
ESO FITS header keywords (~530 lines) were removed — they were never used in code.
"""

# --------------- Data paths ---------------
OSTARS_IDS_JSON = r"/Users/roeyovadia/Documents/Data/BLOeM Project Overview.json"
DATA_RELEASE_3_PATH = r"/Users/roeyovadia/Documents/Data/BLOeM_DR3.0"
DATA_RELEASE_4_PATH = r"/Users/roeyovadia/Documents/Data/BLOeM_Data/BLOeM_DR4.0_Combined"

# --------------- FITS file suffixes ---------------
FITS_SUF_COMBINED = 'Combined.fits'

# --------------- FITS column / header names ---------------
S2N = "SNR"
WAVELENGTH = 'WAVELENGTH'
SCI_NORM = 'SCI_NORM'
SCI_NORM_ERR = 'SCI_NORM_ERR'
MJD_MID = 'MJD_MID'
BARYCORR = 'BARYCORR'
SNR_PPL = 'SNR_PPL'
EPOCH_ID = 'EPOCH_ID'
FIELD_ID = 'FIELD_ID'
