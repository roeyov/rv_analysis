"""
sb2_pipeline.constants — SB2-local column names and YAML keys.

Shared orbital-parameter names (Period, GAMMA, K1, K2, OMEGA_rad,
Eccentricity, T0, ln_sigmaJ1, ln_sigmaJ2) live in utils.constants; only
names that exist nowhere else in the repo are defined here.
"""

# ---- per-epoch RV table columns (as written by the SB2 analysis tool) ----
EPOCH = "epoch"
MJD_COL = "mjd"
RV1 = "rv1"
ERR1 = "rv1_err"
RV2 = "rv2"
ERR2 = "rv2_err"
RV3 = "rv3"
ERR3 = "rv3_err"
FALLBACK1 = "rv1_err_fallback"
FALLBACK2 = "rv2_err_fallback"

# ---- internal data-dict keys for the joint fit ----
# Each component carries its own time axis so single-component points can be
# masked out (use1/use2 flags in the cleaned per-epoch CSV).
SB2_TS = "ts"        # legacy shared-axis key (kept for the periodogram pair)
SB2_TS1 = "ts1"
SB2_TS2 = "ts2"
SB2_RV1 = "rv1"
SB2_ERR1 = "err1"
SB2_RV2 = "rv2"
SB2_ERR2 = "err2"

# ---- cleaned-CSV epoch-flag columns (sb2_pipeline.cleaning) ----
USE1 = "use1"
USE2 = "use2"
SWAP_APPLIED = "swap_applied"

# ---- null-model parameter names (two independent constants) ----
GAMMA1 = "GAMMA1"
GAMMA2 = "GAMMA2"

# ---- YAML section: sb2_io ----
SB2_IO = "sb2_io"
SB2_ROOT = "sb2_root"
PROVENANCE_CSV = "provenance_csv"
CORRECTIONS_CSV = "corrections_csv"
OUTPUT_DIR = "output_dir"
OBJECT_LIST = "object_list"
MJD_MATCH_TOLERANCE = "mjd_match_tolerance"
RV_FILE_PRIORITY = "rv_file_priority"
USE_FWHM = "use_fwhm"

# ---- YAML keys: mcmc_params extras ----
LUCY_SWEENEY_THRESHOLD = "lucy_sweeney_threshold"
LUCY_SWEENEY_ESTIMATOR = "lucy_sweeney_estimator"
FILTER_EXPRESSION = "filter_expression"
FIELD_TO_CHECK = "field_to_check"
TAKE_MIN = "take_min"

# ---- corrections table columns ----
CORR_CONFIG = "config"
CORR_MJD = "MJD"
CORR_VALUE = "rv_correction"
CORR_ERROR = "Error"

# ---- jitter mode: one nuisance term per component, or one shared ----
JITTER_MODE = "jitter_mode"          # lmfit_params key: "separate" | "shared"

# ---- MCMC theta orderings ----
ECC_PARAM_NAMES = ["P", "T0", "omega", "e", "K1", "K2", "gamma",
                   "log_sj1", "log_sj2"]
CIRC_PARAM_NAMES = ["P", "T0", "K1", "K2", "gamma", "log_sj1", "log_sj2"]
ECC_PARAM_NAMES_SHARED = ["P", "T0", "omega", "e", "K1", "K2", "gamma",
                          "log_sj"]
CIRC_PARAM_NAMES_SHARED = ["P", "T0", "K1", "K2", "gamma", "log_sj"]

ECC_LABELS = [r"$P$ [d]", r"$T_0$ [MJD]", r"$\omega$ [deg]", r"$e$",
              r"$K_1$ [km/s]", r"$K_2$ [km/s]", r"$\gamma$ [km/s]",
              r"log σ$_{\rm jit,1}$", r"log σ$_{\rm jit,2}$"]
CIRC_LABELS = [r"$P$ [d]", r"$T_0$ [MJD]",
               r"$K_1$ [km/s]", r"$K_2$ [km/s]", r"$\gamma$ [km/s]",
               r"log σ$_{\rm jit,1}$", r"log σ$_{\rm jit,2}$"]
ECC_LABELS_SHARED = ECC_LABELS[:7] + [r"log σ$_{\rm jit}$"]
CIRC_LABELS_SHARED = CIRC_LABELS[:5] + [r"log σ$_{\rm jit}$"]

DEFAULT_RV_FILE_PRIORITY = [
    "rv_final_for_mcmc.csv",
    "rv_final.csv",
    "rv_corrected.csv",
    "rv_extracted.csv",
]
