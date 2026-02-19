
from __future__ import annotations
import os, re, json, glob, io
from pathlib import Path
from dataclasses import dataclass
import copy

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.ticker as mticker

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import brentq
from typing import List, Tuple, Dict, Optional
from scipy.special import gammainccinv
from datetime import datetime
import lmfit
import zipfile

import matplotlib.pyplot as plt
SETTINGS_FILE = Path.home() / ".sb1_gui_settings.json"

def _load_settings():
    try:
        return json.loads(Path(SETTINGS_FILE).read_text())
    except Exception:
        return {}

def _save_settings_from_state():
    payload = {
        "root_folder":  st.session_state.get("root_folder",  ""),
        "output_root":  st.session_state.get("output_root",  ""),
        "mass_csv":     st.session_state.get("mass_csv",     ""),
        "json_params":  st.session_state.get("json_params",  ""),
        "model_root": st.session_state.get("model_root", ""),
        "last_star":    st.session_state.get("last_star",    ""),
    }
    try:
        Path(SETTINGS_FILE).write_text(json.dumps(payload, indent=2))
    except Exception:
        pass

SETTINGS = _load_settings()

# Optional dependencies
try:
    import astropy.io.fits as fits
    HAVE_ASTROPY = True
except Exception as e:
    HAVE_ASTROPY = False
    ASTROPY_ERR = str(e)

try:
    import corner
    import emcee
    HAVE_CORNER = True
except ImportError:
    HAVE_CORNER = False
    corner = None


# Optional: Plotly for interactive zoom
try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

try:
    from streamlit_plotly_events import plotly_events
    HAVE_PLOTLY_EVENTS = True
except Exception:
    HAVE_PLOTLY_EVENTS = False

try:
    from matplotlib.animation import FuncAnimation
    import matplotlib
    # Use non-interactive backend for animations in Streamlit
    matplotlib.use('Agg')
    HAVE_ANIMATION = True
except ImportError:
    HAVE_ANIMATION = False
    FuncAnimation = None

HAVE_ORBIT = True
ORBIT_IMPORT_ERR = ""

try:
    sys.path.append(str(Path(__file__).parent))
    from utils.periodagram import ls, pdc, plotls
    PRIMARY_PERIODAGRAM = "utils.periodagram"
except Exception as e1:
    try:
        from PDC import ls, pdc, plotls  # type: ignore
        PRIMARY_PERIODAGRAM = "PDC"
    except Exception as e2:
        ls = pdc = plotls = None  # type: ignore
        HAVE_ORBIT = False
        ORBIT_IMPORT_ERR = f"{e1}\n{e2}"

try:
    from utils.constants import *
    from emceeOmLMFITexe11res111 import (
        lmfit_on_sample, print_lmfit_result,
        summarize_result, get_rv_weighted_mean, corner_plot2
    )
except Exception as e3:
    HAVE_ORBIT = False
    ORBIT_IMPORT_ERR = (ORBIT_IMPORT_ERR + "\n" + str(e3)).strip()

# Provide a minimal plotls if the imported module didn't define one
if "plotls" not in globals() or plotls is None:
    import matplotlib.pyplot as _plt
    def plotls(freq, power, fal=None, pmin=1.0, pmax=5000.0, star_id="periodogram", out_dir=None):  # type: ignore
        fig, ax = _plt.subplots(figsize=(6.5, 3.5))
        ax.plot(freq, power, lw=1)
        ax.set_xlabel("Frequency [1/day]")
        ax.set_ylabel("Power")
        ax.set_title(star_id)
        if out_dir:
            fig.savefig(out_dir, dpi=200, bbox_inches="tight")
            _plt.close(fig)
# ---------- end orbit-fitting imports ----------

st.set_page_config(page_title="SB1 Orbit GUI", layout="wide")

# -------------------------------
# Domain constants & defaults
# -------------------------------
CLIGHT = 2.997_924_58e5  # km/s
P_SHORT_MAX = 6.0
E_MAX_SHORT = 0.9
DEFAULT_HE_LINES = [4026.191, 4387.929, 4471.479, 4143.760]
DEFAULT_H_LINES  = [3970.072, 4101.734, 4340.462]

# Velocity search window for per‑line CCF (km/s)
V_MIN = -400.0
V_MAX =  400.0

# Helpers
ID_RE = re.compile(r"\b\d-\d{3}\b")
RSUN2AU = 0.00465047
C1 = 9651121.605
C2 = 365.2563356

# -------------------------------
# UI — Sidebar: inputs & switches
st.sidebar.header("Inputs & Settings")
root_folder = st.sidebar.text_input(
    "Root folder with per-star subfolders (spectra)",
    value=SETTINGS.get("root_folder", ""),
    key="root_folder",
    on_change=_save_settings_from_state
)
output_root = st.sidebar.text_input(
    "Output folder (per-star outputs go here)",
    value=SETTINGS.get("output_root", ""),
    key="output_root",
    on_change=_save_settings_from_state
)
mass_csv = st.sidebar.text_input(
    "Mass table CSV (mass_bloem.csv)",
    value=SETTINGS.get("mass_csv", ""),
    key="mass_csv",
    on_change=_save_settings_from_state
)
json_params = st.sidebar.text_input(
    "LMFIT params JSON (used by orbit fit)",
    value=SETTINGS.get("json_params", ""),
    key="json_params",
    on_change=_save_settings_from_state
)

st.sidebar.markdown("---")
use_helium = st.sidebar.checkbox("Use He lines", True)
use_balmer = st.sidebar.checkbox("Use Balmer (H) lines", False)
user_he = st.sidebar.text_input("He lines (Å, comma‑sep)", ", ".join(str(x) for x in DEFAULT_HE_LINES))
user_h  = st.sidebar.text_input("H lines (Å, comma‑sep)",  ", ".join(str(x) for x in DEFAULT_H_LINES))

line_tol = st.sidebar.number_input("Line match tolerance (Å)", 0.1, 10.0, 4.0, step=0.1)
win_He   = st.sidebar.number_input("He window width (Å)", 5, 200, 50, step=1)
win_H    = st.sidebar.number_input("H window width (Å)", 5, 200, 50, step=1)

s2n_cut  = st.sidebar.number_input("SNR floor (skip FITS with SNR ≤)", 0, 200, 10, step=1)
fit_range = st.sidebar.slider("Parabola fit range (fraction of peak)", 0.50, 0.99, 0.95, step=0.01)
n_sig_out = st.sidebar.number_input("Outlier threshold n_sig (|ΔRV| ≤ n_sig·σ)", 0.5, 10.0, 5.0, step=0.1)
error_model = st.sidebar.selectbox(
    "RV error model",
    ["Tonry–Davis (robust)", "Curvature + Nres (legacy script)"],
    index=1,
    help="Choose how per-line 1σ RV is computed."
)
final_error_type = st.sidebar.selectbox(
    "Final CCF error type",
    ["Weighted only", "Weighted + Statistical"],
    index=0,
    key="final_error_type",
    help="Selects the error for the final CCF file. 'Weighted + Statistical' adds the standard error of the mean of the lines in quadrature."
)
ccf_lambda_min = st.sidebar.number_input("Global λ min (Å)", 3500.0, 7000.0, 3985.0, step=5.0)
ccf_lambda_max = st.sidebar.number_input("Global λ max (Å)", 3500.0, 7000.0, 4570.0, step=5.0)
two_pass = st.sidebar.checkbox("Two-pass CCF (model → coadd)", True)
model_root = st.sidebar.text_input(
    "Model templates folder (optional)",
    value=SETTINGS.get("model_root", ""),
    key="model_root",
    on_change=_save_settings_from_state,
)
if st.sidebar.button("Reset saved settings"):
    try:
        SETTINGS_FILE.unlink()
        st.success("Cleared saved settings.")
    except Exception:
        pass

st.sidebar.markdown("---")
st.sidebar.subheader("Co-add Weighting")
snr_mode = st.sidebar.radio(
    "SNR source for co-add weights",
    ["Whole Spectrum (Original)", "FITS Header (SNR_PPL)", "Custom Ranges"],
    index=0,
    key="coadd_snr_mode",
    help="Determines how spectra are weighted when creating the co-added template."
)
# This text input only matters if "Custom Ranges" is selected
snr_custom_ranges_text = st.sidebar.text_input(
    "Custom SNR ranges (Å)",
    value="4045-4060; 4220-4230; 4495-4550",
    help="Wavelength windows for SNR calculation, e.g., '4000-4100; 4500-4550'",
    key="coadd_snr_ranges"
)
st.sidebar.markdown("---")

st.sidebar.markdown("---")
recache = st.sidebar.button("Clear all caches")
if recache:
    st.cache_data.clear()
    st.cache_resource.clear()

st.sidebar.markdown("---")
st.sidebar.subheader("Classifier (SB1 / SB2)")
clf_tol = st.sidebar.number_input("Line match tolerance (Å)", 0.05, 10.0, 1.5, step=0.05)

# controls for line detection strictness (only used in SNR picker and auto-detect)
detect_prom = st.sidebar.number_input("Line-detect min prominence", 0.0, 0.2, 0.04, step=0.005, help="Higher ⇒ fewer lines")
detect_dist = st.sidebar.number_input("Line-detect min pixel distance", 1, 100, 15, step=1, help="Higher ⇒ fewer lines")
keep_only_known = st.sidebar.checkbox("Restrict to known H/He lines", True, help="Match to the known_sets list only")
max_lines_keep = st.sidebar.number_input("Cap: keep strongest N detected", 1, 200, 12, step=1, help="Applied after filters")

use_classifier = st.sidebar.checkbox("Run SB1/SB2 classifier", False)

# Template for RVs (your note: use first obs for CCF)
template_mode = st.sidebar.selectbox(
    "Template for RVs",
    ["first observation", "combined mask FITS (*Combined*.fits)", "model template (model_root)"],
    index=0, key="clf_template"
)

# CCF settings for classifier (independent of CCF section)
clf_lam_txt = st.sidebar.text_input("Cross-correlation λ windows (Å)",
    value="4000-4500", help="Semicolon-separated windows, e.g. 4000-4160; 4400-4570", key="clf_lams")
clf_vmin = st.sidebar.number_input("Classifier v_min (km/s)", -2000.0, 0.0, -400.0, step=10.0, key="clf_vmin")
clf_vmax = st.sidebar.number_input("Classifier v_max (km/s)", 0.0, 2000.0,  400.0, step=10.0,  key="clf_vmax")
clf_fitf = st.sidebar.slider("Parabola fit range (fraction of peak, classifier)", 0.50, 0.99, 0.95, step=0.01, key="clf_fitf")
clf_overs = st.sidebar.number_input("Oversample factor", 1.0, 8.0, 1.0, step=0.5, key="clf_overs")


st.sidebar.caption("Line selection")
clf_line_mode = st.sidebar.radio("How to select lines", ["Auto-detect on coadd", "Manual list"], index=0, key="clf_linemode")
clf_detect_span = st.sidebar.text_input("Auto-detect range (Å)", value="4000-4570", key="clf_detect_span")
clf_manual_lines = st.sidebar.text_input("Manual lines (Å, comma-sep)", value="", key="clf_manual_list")
clf_tol   = st.sidebar.number_input("Line match tolerance (Å)", 0.1, 10.0, 4.0, step=0.1, key="clf_tol")
clf_winH  = st.sidebar.number_input("Window H (Å)", 5, 200, 50, step=1, key="clf_winH")
clf_winHe = st.sidebar.number_input("Window He/HeII (Å)", 5, 200, 50, step=1, key="clf_winHe")
clf_P     = st.sidebar.number_input("Significance P for χ² threshold", 0.001, 0.5, 0.05, step=0.001, key="clf_P")

write_class_to_summary = st.sidebar.checkbox("After classifying, add class to orbit_summary.csv", True, key="clf_write")
st.sidebar.markdown("---")


# -------------------------------
# Utility: parsing user line lists
# -------------------------------
def parse_lines(txt: str) -> List[float]:
    out = []
    for tok in re.split(r"[,\s]+", txt.strip()):
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            pass
    return out

HE_LINES = parse_lines(user_he)
H_LINES  = parse_lines(user_h)
ALL_LINES = []
if use_balmer: ALL_LINES += H_LINES
if use_helium: ALL_LINES += HE_LINES

# -------------------------------
# CCF / Spectra helpers
# -------------------------------
@dataclass
class Obs:
    mjd: float
    name: str
    snr: Optional[float]
    spectrum: np.ndarray  # (N,2) [wavelength, flux]


def _norm_ranges(ranges):
    # ensure list of (lo<=hi) with simple merge
    if not ranges: return []
    rr = [(float(min(a,b)), float(max(a,b))) for (a,b) in ranges]
    rr.sort()
    out = [rr[0]]
    for a,b in rr[1:]:
        lo,hi = out[-1]
        if a <= hi:
            out[-1] = (lo, max(hi,b))
        else:
            out.append((a,b))
    return out

def create_zip_from_plots(plots: Dict[str, bytes]) -> bytes:
    """Creates a zip file in memory from a dictionary of plot data."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, data in plots.items():
            zf.writestr(filename, data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def _windows_from_selected_points(selected):
    if not selected: return None
    xs = [float(p["x"]) for p in selected if "x" in p]
    if not xs: return None
    return (min(xs), max(xs))


def read_ascii(path: Path, col0=0, col1=1) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None, sep=r"\s+", comment="#")
    return df.iloc[:, col0].to_numpy(float), df.iloc[:, col1].to_numpy(float)


def read_fits(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not HAVE_ASTROPY:
        raise RuntimeError(f"astropy not available: {ASTROPY_ERR}")
    with fits.open(path) as hdul:
        data = hdul[1].data
        wave = np.asarray(data['WAVELENGTH'], dtype=float)
        flux = np.asarray(data['SCI_NORM'], dtype=float)
    return wave, flux


def read_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ext = path.suffix.lower()
    if ext in (".fits", ".fit", ".tfits", ".hfits"):
        return read_fits(path)
    else:
        return read_ascii(path)


def fits_header_key(path: Path, key: str) -> Optional[str]:
    if not HAVE_ASTROPY:
        return None
    try:
        with fits.open(path) as hdul:
            for hdu in hdul:
                if key in hdu.header:
                    return hdu.header.get(key)
    except Exception:
        return None
    return None


def find_absorption_lines(
    wave: np.ndarray,
    flux: np.ndarray,
    *,
    prominence: float = 0.02,
    distance: int = 12,
    rel_height: float = 0.70,
) -> List[Dict[str, float]]:
    """
    Peak-find on 1 - flux (absorption). Returns centers, width, prominence, depth.
    Make 'prominence' a bit larger (e.g. 0.03–0.06) and 'distance' larger to reduce clutter.
    """
    inv = 1.0 - np.asarray(flux)
    smoothed = savgol_filter(inv, window_length=11, polyorder=2, mode='interp')
    pks, props = find_peaks(smoothed, prominence=prominence, distance=distance)
    widths, _, left_ips, right_ips = peak_widths(smoothed, pks, rel_height=rel_height)
    idx = np.arange(len(wave))
    def x_at(ip): return np.interp(ip, idx, wave)
    lines = []
    for i, pk in enumerate(pks):
        center = float(wave[pk])
        w = float(x_at(right_ips[i]) - x_at(left_ips[i]))
        prom = float(props["prominences"][i]) if "prominences" in props else np.nan
        depth = float(smoothed[pk])
        lines.append({"center": center, "width": w, "prominence": prom, "depth": depth})
    return lines


def pick_num(row: Dict[str, object], *names) -> float:
    for n in names:
        if n in row:
            try:
                v = float(row[n])
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return np.nan


def equivalent_width(center: float, wave: np.ndarray, flux: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    w = wave[mask]
    f = flux[mask]
    if w.size < 2:
        return np.nan, np.nan
    EW = np.trapz(1 - f, x=w)
    dlam = np.mean(np.diff(w)) if w.size > 1 else 0.0
    # crude noise estimate around line
    std = np.std(flux[(wave >= center+20) & (wave <= center+25)]) if np.any((wave >= center+20) & (wave <= center+25)) else np.std(f)
    err = (w[-1]-w[0]) * dlam * std * np.sqrt(2)
    return float(EW), float(err)


def ccf_scalar(obs: np.ndarray, mask: np.ndarray) -> float:
    n = len(obs)
    num = float(np.sum(obs * mask))
    den = float(np.std(obs) * np.std(mask) * n)
    if den == 0:
        return 0.0
    return num / den


def crosscor_line(obs_flux: np.ndarray, mod_flux: np.ndarray, wave_log: np.ndarray,
                   fit_fraction: float, vmin: float = V_MIN, vmax: float = V_MAX,
                   error_model: str = "Tonry–Davis (robust)") -> Tuple[float,float,float,float]:

    # Build velocity grid from log sampling
    # Assume constant dv implied by log grid spacing (approx ok locally)
    # Use simple integer shifts on the log grid
    # Prepare shifts
    # Convert log step to dv ≈ c * dlnλ
    dln = np.mean(np.diff(np.log(wave_log)))
    vbin = CLIGHT * dln
    s_range = np.arange(int(vmin/vbin), int(vmax/vbin), 1)
    vel = vbin * s_range

    # roll model and compute CCF
    def rolled(k):
        return np.roll(mod_flux, k)
    ccf = np.array([ccf_scalar(obs_flux, rolled(k)) for k in s_range])
    i_max = int(np.argmax(ccf))
    ccf_max = float(ccf[i_max])
    if i_max == 0 or i_max == len(ccf)-1:
        return np.nan, np.nan, ccf_max, np.nan

    # FWHM estimate
    half = 0.5 * ccf_max
    inds = np.nonzero(ccf >= half)[0]
    fwhm = float(vel[inds[-1]] - vel[inds[0]]) if inds.size > 1 else np.nan

    # Parabola about the peak within fit_fraction
    # Parabola about the peak within fit_fraction
    left = np.argmin(np.abs(fit_fraction*ccf_max - ccf[:i_max]))
    right = np.argmin(np.abs(fit_fraction*ccf_max - ccf[i_max+1:])) + i_max + 1
    if right <= left:
        return np.nan, np.nan, ccf_max, fwhm
    a, b, c = np.polyfit(vel[left:right+1], ccf[left:right+1], 2)
    v_peak = -b/(2*a)
    ccf_at_max = c - (b*b)/(4*a)

    # --- Compute both sigmas ---
    # 1) Tonry–Davis (1979) σ_v ≈ (3/8) * FWHM / (1 + R)
    mask_far = np.ones_like(ccf, dtype=bool)
    mask_far[max(0, i_max-20): min(len(ccf), i_max+20)] = False
    noise_rms = np.std(ccf[mask_far]) if np.any(mask_far) else np.nan
    if np.isfinite(noise_rms) and np.isfinite(fwhm) and fwhm > 0 and ccf_max > 0:
        R = ccf_max / (np.sqrt(2.0) * max(noise_rms, 1e-12))
        sigma_td = (3.0/8.0) * fwhm / (1.0 + R)
    else:
        sigma_td = np.abs(fwhm)/6.0 if np.isfinite(fwhm) else np.nan

    # 2) Curvature + Nres (your legacy script)
    Nres = max(2, len(wave_log))                  # points contributing
    cmax = min(1.0 - 1e-12, float(ccf_at_max))    # clamp like your code
    denom = Nres * abs(2.0*a) * max(cmax, 1e-12)  # |d^2CCF/dv^2| = |2a|
    sigma_curv = np.sqrt(max(0.0, (1.0 - cmax*cmax) / max(denom, 1e-30)))

    # Pick model
    sigma = sigma_td if error_model.startswith("Tonry") else sigma_curv
    return float(v_peak), float(sigma), float(ccf_max), fwhm


def _clip_to_data(x: np.ndarray, y: np.ndarray,
                  lo_req: float, hi_req: float) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
    """Return (x_sel, y_sel, lo_used, hi_used, had_overlap).
    If no points fall in [lo_req, hi_req], fall back to full data span."""
    lo_data, hi_data = float(np.nanmin(x)), float(np.nanmax(x))
    lo_used = max(lo_req, lo_data)
    hi_used = min(hi_req, hi_data)
    sel = (x >= lo_used) & (x <= hi_used)
    if np.count_nonzero(sel) >= 2:
        return x[sel], y[sel], lo_used, hi_used, True
    # no overlap → show full data so the figure is never empty
    return x, y, lo_data, hi_data, False

def phase_from_epoch(T_epoch: float, t_ref: float, P: float) -> float:
    # map to [-0.5, 0.5)
    return (( (T_epoch - t_ref)/P + 0.5 ) % 1.0) - 0.5

def epoch_from_phase(phi: float, t_ref: float, P: float) -> float:
    # keep phase in [-0.5, 0.5)
    phi = ((phi + 0.5) % 1.0) - 0.5
    return t_ref + phi * P
# -------------------------------
# Star discovery & reading
# -------------------------------
@st.cache_data(show_spinner=False)
def discover_stars(root: str) -> List[Path]:
    p = Path(root)
    if not p.is_dir():
        return []
    subs = [q for q in p.iterdir() if q.is_dir()]
    # keep only those with some spectra‑looking files
    keep = []
    for s in subs:
        files = list(s.glob("*.fits")) + list(s.glob("*.fit")) + list(s.glob("*.txt")) + list(s.glob("*.dat")) + list(s.glob("*.ascii"))
        if files:
            keep.append(s)
    return sorted(keep, key=lambda x: x.name)


@st.cache_data(show_spinner=False)
def load_observations(star_folder: Path, s2n_floor: float) -> List[Obs]:
    obs = []
    for f in sorted(star_folder.glob("*.fits")):
        # Optional contamination skip
        if HAVE_ASTROPY:
            left = fits_header_key(f, 'FPDLEFT') or ''
            right = fits_header_key(f, 'FPDRIGHT') or ''
            if (left == 'CALSIM') or (right == 'CALSIM'):
                continue
        try:
            w, fl = read_file(f)
        except Exception:
            continue
        # SNR estimate (simple)
        snr = float(np.nanmean(fl)/max(1e-12, np.nanstd(fl)))
        if HAVE_ASTROPY:
            snr_hdr = fits_header_key(f, 'SNR_PPL')
            snr = float(snr_hdr) if snr_hdr is not None else snr
        if snr <= s2n_floor:
            continue
        # MJD
        mjd = None
        if HAVE_ASTROPY:
            mjd = fits_header_key(f, 'MJD_MID')
        if mjd is None:
            mjd = len(obs)  # fallback index
        obs.append(Obs(mjd=float(mjd), name=f.name, snr=snr, spectrum=np.column_stack([w, fl])))
    # also allow ASCII
    for f in sorted(star_folder.glob("*.txt")) + sorted(star_folder.glob("*.dat")) + sorted(star_folder.glob("*.ascii")):
        try:
            w, fl = read_file(f)
        except Exception:
            continue
        mjd = len(obs)
        snr = float(np.nanmean(fl)/max(1e-12, np.nanstd(fl)))
        obs.append(Obs(mjd=float(mjd), name=f.name, snr=snr, spectrum=np.column_stack([w, fl])))
    # sort by mjd
    return sorted(obs, key=lambda r: r.mjd)


# -------------------------------
# Core pipeline: per‑star CCF → RV table
# -------------------------------
@dataclass
class RVResult:
    rv_df: pd.DataFrame
    mean_df: pd.DataFrame  # MJD, Mean RV, Mean RVsig
    detected_lines: List[float]
    plots: Dict[str, bytes]
    coadd_wave: Optional[np.ndarray] = None          # template wave actually used
    coadd_flux: Optional[np.ndarray] = None          # template flux actually used
    coadd0_wave: Optional[np.ndarray] = None         # unshifted coadd (if available)
    coadd0_flux: Optional[np.ndarray] = None
def load_model_template(star_folder: Path, model_root: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (wave, flux) for the model template of this star, or None."""
    if not model_root:
        return None
    star = star_folder.name
    for pat in (f"model_roation_BLOeM_{star}.txt", f"model_BLOeM_{star}.txt"):
        p = Path(model_root) / pat
        if p.is_file():
            w, f = read_ascii(p)   # same reader you already have
            return np.asarray(w, float), np.asarray(f, float)
    return None

def lmfit_params_dataframe(result):
    """Turn an lmfit MinimizerResult into a nice parameters DataFrame."""
    if result is None or not hasattr(result, "params") or result.params is None:
        return pd.DataFrame()
    rows = []
    for name, p in result.params.items():
        rows.append({
            "Parameter": name,
            "Value": float(getattr(p, "value", np.nan)),
            "Stderr (1σ)": (
                float(p.stderr) if getattr(p, "stderr", None) is not None else np.nan
            ),
            "Min": (None if np.isneginf(getattr(p, "min", -np.inf)) else float(p.min)),
            "Max": (None if np.isposinf(getattr(p, "max",  np.inf)) else float(p.max)),
            "Vary": bool(getattr(p, "vary", True)),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("Parameter").reset_index(drop=True)

def zip_files(paths):
    """Return bytes of a ZIP file containing the given paths (arcname=basename)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            p = Path(p)
            if p.exists() and p.is_file():
                zf.write(p, arcname=p.name)
    buf.seek(0)
    return buf.getvalue()

# --- binary decision: pairwise ΔRV significance ---
def binary_rv_threshold(rvs, err_vs, drv_tresh=20.0, sign_threshold=4.0) -> bool:
    r = np.asarray(rvs, float).ravel()
    e = np.asarray(err_vs, float).ravel()
    if r.size == 0 or e.size == 0:
        return False
    diff = np.abs(r[:, None] - r[None, :])
    sig  = np.sqrt(e[:, None]**2 + e[None, :]**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = diff / np.where(sig > 0, sig, np.nan)
    z = np.where(np.isfinite(z), z, 0.0)
    return np.nanmax(np.where(diff > drv_tresh, z, 0.0)) > sign_threshold


def compute_ccf_for_star(
        obs_list: List[Obs], star_name: str, all_lines: List[float], tol: float,
        w_he: float, w_h: float, lam_min: float, lam_max: float,
        fit_fraction: float, two_pass: bool = True,
        n_sig_out: float = 3.0,
        model_template: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        final_error_type: str = "Weighted only"
) -> RVResult:
    """
    Pass 1:
      - If a model template is provided, use it as the template.
      - Else build an unshifted coadd and use that as the template.
    Pass 2 (if two_pass):
      - Shift each observation by its pass-1 weighted-mean RV, rebuild a coadd,
        optionally re-detect lines, and re-measure RVs.
    """
    if not obs_list:
        return RVResult(pd.DataFrame(), pd.DataFrame(), [], {})

    base = obs_list[0].spectrum
    wave = base[:, 0]
    gmask = (wave >= lam_min) & (wave <= lam_max)
    wave_lim = wave[gmask]

    # S/N per obs and weights (used for coadds), now with flexible source
    snr_mode_choice = st.session_state.get("coadd_snr_mode", "Whole Spectrum (Original)")

    if snr_mode_choice == "FITS Header (SNR_PPL)":
        # The load_observations function already prioritizes SNR_PPL and stores it in o.snr
        st.info("Using SNR from FITS header (SNR_PPL) for co-add weighting.")
        s2n = np.array([o.snr if o.snr is not None else 0.0 for o in obs_list])

    elif snr_mode_choice == "Custom Ranges":
        # Use the helper functions you already have to parse text and calculate SNR
        ranges_text = st.session_state.get("coadd_snr_ranges", "")
        custom_ranges = parse_ranges_text(ranges_text)
        if custom_ranges:
            st.info(f"Using custom ranges {custom_ranges} for co-add weighting.")
            s2n = np.array([
                snr_from_windows(o.spectrum[:, 0], o.spectrum[:, 1], custom_ranges)
                for o in obs_list
            ])
        else:
            st.warning(
                "Custom SNR ranges were selected but the input is empty/invalid. Falling back to whole spectrum.")
            # Fallback to original method
            s2n = np.array(
                [float(np.nanmean(o.spectrum[:, 1]) / max(1e-12, np.nanstd(o.spectrum[:, 1]))) for o in obs_list])

    else:  # "Whole Spectrum (Original)"
        st.info("Using whole spectrum SNR for co-add weighting.")
        s2n = np.array(
            [float(np.nanmean(o.spectrum[:, 1]) / max(1e-12, np.nanstd(o.spectrum[:, 1]))) for o in obs_list])

    # This part remains the same - it correctly uses the s2n array calculated above
    wts = np.clip(s2n, 1e-6, np.inf) ** 2

    def _choose_lines(wave_ref: np.ndarray, flux_ref: np.ndarray) -> List[Dict[str, float]]:
        """
        wave_ref and flux_ref MUST be the same length and already limited
        to the desired wavelength span before calling this function.
        """
        if len(wave_ref) != len(flux_ref):
            raise ValueError(f"_choose_lines got mismatched shapes: {len(wave_ref)} vs {len(flux_ref)}")

        det = find_absorption_lines(wave_ref, flux_ref)

        chosen = []
        for L in all_lines:
            best = None
            best_d = 1e9
            for d in det:
                dd = abs(d['center'] - L)
                if dd <= tol and dd < best_d:
                    best = d
                    best_d = dd
            if best is not None:
                chosen.append(best)
        return chosen

    def _measure_with_template(
            tw: np.ndarray, tf: np.ndarray, chosen: List[Dict[str, float]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[np.ndarray]]:
        """
        Returns:
          rv_df_raw, rv_df_clean, mean_df_raw, mean_df_clean, keep_masks
        Outlier rule (per observation, across lines): keep line j if
            |RV_j - w_mean| <= n_sig_out * err_j
        where w_mean is the weight-1/σ^2 mean over all valid lines.
        """
        rows = {"Line": [c['center'] for c in chosen]}
        mean_raw, sig_raw = [], []
        mean_cln, sig_cln = [], []
        keep_masks: List[np.ndarray] = []

        for o in obs_list:
            o_wave = o.spectrum[:, 0]
            o_flux = np.nan_to_num(o.spectrum[:, 1])
            rvs, errs = [], []
            for c in chosen:
                L = c['center']
                win = w_he if any(abs(L - h) <= tol for h in DEFAULT_HE_LINES + HE_LINES) else w_h
                msk = (o_wave >= L - win / 2) & (o_wave <= L + win / 2)
                if np.count_nonzero(msk) < 4:
                    rvs.append(np.nan)
                    errs.append(np.nan)
                    continue
                wl, fl = o_wave[msk], o_flux[msk]
                if wl[-1] <= wl[0]:
                    rvs.append(np.nan)
                    errs.append(np.nan)
                    continue
                dln = np.log(wl[-1] / wl[0]) / max(2, len(wl) - 1)
                wlog = wl[0] * np.exp(dln * np.arange(len(wl)))
                obs_log = interp1d(wl, fl, bounds_error=False, fill_value=1.0)(wlog)
                tmask = (tw >= wl[0]) & (tw <= wl[-1])
                if np.count_nonzero(tmask) < 2:
                    rvs.append(np.nan)
                    errs.append(np.nan)
                    continue
                tmpl_log = interp1d(tw[tmask], tf[tmask], bounds_error=False, fill_value=1.0)(wlog)

                vpk, sig, cpk, fwhm = crosscor_line(
                    obs_log - np.nanmean(obs_log),
                    tmpl_log - np.nanmean(tmpl_log),
                    wlog, fit_fraction,
                    error_model=error_model
                )
                EWmask = (o_wave >= L - win / 2 + 13) & (o_wave <= L + win / 2 - 9)
                EW, eEW = equivalent_width(L, o_wave, o_flux, EWmask)
                if not np.isfinite(EW) or not np.isfinite(eEW) or (EW <= 0.11) or (abs(EW - eEW) < 0.01):
                    rvs.append(np.nan)
                    errs.append(np.nan)
                    continue
                rvs.append(vpk)
                errs.append(sig)

            rows[f"MJD_{o.mjd:.2f}_RV"] = rvs
            rows[f"MJD_{o.mjd:.2f}_err"] = errs

            r, e = np.array(rvs, float), np.array(errs, float)
            ok = np.isfinite(r) & np.isfinite(e) & (e > 0)

            if ok.sum() == 0:
                mean_raw.append(np.nan)
                sig_raw.append(np.nan)
                mean_cln.append(np.nan)
                sig_cln.append(np.nan)
                keep_masks.append(np.zeros_like(ok, dtype=bool))
            else:
                W = 1.0 / (e[ok] ** 2)
                wmean = float(np.sum(W * r[ok]) / np.sum(W))
                wsig = float(np.sqrt(1.0 / np.sum(W)))
                mean_raw.append(wmean)
                sig_raw.append(wsig)

                diff = np.abs(r - wmean)
                thr = n_sig_out * e
                keep = ok & (diff <= thr)
                if keep.sum() == 0:
                    keep = ok
                keep_masks.append(keep)

                Wc = 1.0 / (e[keep] ** 2)
                wmean_c = float(np.sum(Wc * r[keep]) / np.sum(Wc))
                wsig_c = float(np.sqrt(1.0 / np.sum(Wc)))

                # Choose the final error based on the user's selection
                final_sig = wsig_c
                if final_error_type == "Weighted + Statistical":
                    num_lines = keep.sum()
                    if num_lines > 1:
                        # Statistical error = stdev.s / sqrt(n)
                        stati_err = np.std(r[keep], ddof=1) / np.sqrt(num_lines)
                        if np.isfinite(stati_err):
                            # Final error = sqrt(weighted^2 + statistical^2)
                            final_sig = np.sqrt(wsig_c ** 2 + stati_err ** 2)

                mean_cln.append(wmean_c)
                sig_cln.append(final_sig)
        rv_df_raw = pd.DataFrame(rows)
        rv_df_clean = rv_df_raw.copy()
        for o, keep in zip(obs_list, keep_masks):
            rv_col, er_col = f"MJD_{o.mjd:.2f}_RV", f"MJD_{o.mjd:.2f}_err"
            if rv_col in rv_df_clean.columns:
                rv_df_clean.loc[~keep, rv_col] = np.nan
            if er_col in rv_df_clean.columns:
                rv_df_clean.loc[~keep, er_col] = np.nan

        mean_df_raw = pd.DataFrame({"MJD": [o.mjd for o in obs_list], "Mean RV": mean_raw, "Mean RVsig": sig_raw})
        mean_df_clean = pd.DataFrame({"MJD": [o.mjd for o in obs_list], "Mean RV": mean_cln, "Mean RVsig": sig_cln})
        return rv_df_raw, rv_df_clean, mean_df_raw, mean_df_clean, keep_masks

    if model_template is not None:
        t_wave, t_flux = model_template
        t_flux_on_lim = np.interp(wave_lim, t_wave, t_flux, left=1.0, right=1.0)
        chosen = _choose_lines(wave_lim, t_flux_on_lim)
    else:
        coadd = np.zeros_like(wave)
        tot = 0.0
        for o, wt in zip(obs_list, wts):
            coadd += wt * np.nan_to_num(o.spectrum[:, 1])
            tot += wt
        coadd /= max(1e-12, tot)
        t_wave, t_flux = wave, coadd
        chosen = _choose_lines(wave_lim, coadd[gmask])

    rv_df1_raw, rv_df1_clean, mean_df1_raw, mean_df1_clean, keep_masks1 = _measure_with_template(t_wave, t_flux, chosen)

    figs = {}

    def _fig_bytes(fig):
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        b.seek(0)
        return b.read()

    def _plot_lines(rv_df, chosen_lines, title, keep_masks_list=None):
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for i, L in enumerate(chosen_lines):
            xs = [o.mjd for o in obs_list]
            ys = [rv_df.get(f"MJD_{o.mjd:.2f}_RV", pd.Series([np.nan] * len(rv_df))).iloc[i] for o in obs_list]
            es = [rv_df.get(f"MJD_{o.mjd:.2f}_err", pd.Series([np.nan] * len(rv_df))).iloc[i] for o in obs_list]

            if keep_masks_list is None:
                ax.errorbar(xs, ys, yerr=es, fmt='o', capsize=2, label=f"{L:.1f} Å")
            else:
                ys_kept, es_kept = np.full_like(ys, np.nan, float), np.full_like(es, np.nan, float)
                ys_outlier = np.full_like(ys, np.nan, float)
                for j in range(len(obs_list)):
                    if j < len(keep_masks_list) and i < len(keep_masks_list[j]):
                        if keep_masks_list[j][i]:
                            ys_kept[j], es_kept[j] = ys[j], es[j]
                        elif np.isfinite(ys[j]):
                            ys_outlier[j] = ys[j]

                line, _, _ = ax.errorbar(xs, ys_kept, yerr=es_kept, fmt='o', capsize=2, label=f"{L:.1f} Å")
                ax.plot(xs, ys_outlier, 'x', color='red', markersize=7, markeredgewidth=1.5, label='_nolegend_')

        ax.set_xlabel("MJD")
        ax.set_ylabel("Line RV (km/s)")
        ax.legend(ncol=3, fontsize=9)
        ax.set_title(f"{star_name} — {title}")
        return _fig_bytes(fig)

    if not two_pass:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.errorbar(mean_df1_raw["MJD"], mean_df1_raw["Mean RV"], yerr=mean_df1_raw["Mean RVsig"], fmt='o', capsize=3)
        ax.set_xlabel("MJD");
        ax.set_ylabel("Weighted mean RV (km/s)");
        ax.set_title(f"{star_name} — Weighted means (before clean)")
        figs["p1_weighted_before.png"] = _fig_bytes(fig)
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.errorbar(mean_df1_clean["MJD"], mean_df1_clean["Mean RV"], yerr=mean_df1_clean["Mean RVsig"], fmt='o',
                    capsize=3)
        ax.set_xlabel("MJD");
        ax.set_ylabel("Weighted mean RV (km/s)");
        ax.set_title(f"{star_name} — Weighted means (after clean)")
        figs["p1_weighted_after.png"] = _fig_bytes(fig)

        figs["p1_lines_before.png"] = _plot_lines(rv_df1_raw, [c['center'] for c in chosen], "Line RVs (before)",
                                                  keep_masks_list=keep_masks1)
        figs["p1_lines_after.png"] = _plot_lines(rv_df1_clean, [c['center'] for c in chosen], "Line RVs (after)")

        figs["rv_vs_time.png"] = figs["p1_weighted_after.png"]
        figs["lines_rv_vs_time.png"] = figs["p1_lines_after.png"]

        return RVResult(rv_df1_clean, mean_df1_clean, [c['center'] for c in chosen], figs, coadd_wave=t_wave,
                        coadd_flux=t_flux, coadd0_wave=t_wave, coadd0_flux=t_flux)

    v_pass1 = mean_df1_clean["Mean RV"].to_numpy(float)
    coadd2 = np.zeros_like(wave)
    tot2 = 0.0
    for o, wt, v in zip(obs_list, wts, v_pass1):
        v = 0.0 if not np.isfinite(v) else v
        shifted = interp1d(o.spectrum[:, 0] * (1 - v / CLIGHT), np.nan_to_num(o.spectrum[:, 1]), bounds_error=False,
                           fill_value=1.0)(wave)
        coadd2 += wt * shifted
        tot2 += wt
    coadd2 /= max(1e-12, tot2)
    chosen2 = _choose_lines(wave_lim, coadd2[gmask])
    if not chosen2:
        chosen2 = chosen

    rv_df2_raw, rv_df2_clean, mean_df2_raw, mean_df2_clean, keep_masks2 = _measure_with_template(wave, coadd2, chosen2)

    fig, ax = plt.subplots(figsize=(6.5, 3.8));
    ax.errorbar(mean_df1_raw["MJD"], mean_df1_raw["Mean RV"], yerr=mean_df1_raw["Mean RVsig"], fmt='o', capsize=3);
    ax.set_xlabel("MJD");
    ax.set_ylabel("Weighted mean RV (km/s)");
    ax.set_title(f"{star_name} — Pass 1 (before clean)");
    figs["p1_weighted_before.png"] = _fig_bytes(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.8));
    ax.errorbar(mean_df1_clean["MJD"], mean_df1_clean["Mean RV"], yerr=mean_df1_clean["Mean RVsig"], fmt='o',
                capsize=3);
    ax.set_xlabel("MJD");
    ax.set_ylabel("Weighted mean RV (km/s)");
    ax.set_title(f"{star_name} — Pass 1 (after clean)");
    figs["p1_weighted_after.png"] = _fig_bytes(fig)
    figs["p1_lines_before.png"] = _plot_lines(rv_df1_raw, [c['center'] for c in chosen], "Pass 1 — Line RVs (before)",
                                              keep_masks_list=keep_masks1)
    figs["p1_lines_after.png"] = _plot_lines(rv_df1_clean, [c['center'] for c in chosen], "Pass 1 — Line RVs (after)")

    fig, ax = plt.subplots(figsize=(6.5, 3.8));
    ax.errorbar(mean_df2_raw["MJD"], mean_df2_raw["Mean RV"], yerr=mean_df2_raw["Mean RVsig"], fmt='o', capsize=3);
    ax.set_xlabel("MJD");
    ax.set_ylabel("Weighted mean RV (km/s)");
    ax.set_title(f"{star_name} — Pass 2 (before clean)");
    figs["p2_weighted_before.png"] = _fig_bytes(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.8));
    ax.errorbar(mean_df2_clean["MJD"], mean_df2_clean["Mean RV"], yerr=mean_df2_clean["Mean RVsig"], fmt='o',
                capsize=3);
    ax.set_xlabel("MJD");
    ax.set_ylabel("Weighted mean RV (km/s)");
    ax.set_title(f"{star_name} — Pass 2 (after clean)");
    figs["p2_weighted_after.png"] = _fig_bytes(fig)
    figs["p2_lines_before.png"] = _plot_lines(rv_df2_raw, [c['center'] for c in chosen2], "Pass 2 — Line RVs (before)",
                                              keep_masks_list=keep_masks2)
    figs["p2_lines_after.png"] = _plot_lines(rv_df2_clean, [c['center'] for c in chosen2], "Pass 2 — Line RVs (after)")

    figs["rv_vs_time.png"] = figs["p2_weighted_after.png"]
    figs["lines_rv_vs_time.png"] = figs["p2_lines_after.png"]

    return RVResult(rv_df2_clean, mean_df2_clean, [c['center'] for c in chosen2], figs, wave, coadd2, wave,
                    locals().get("coadd", None))

# -------------------------------
# Orbit fit (reusing your functions when available)
# -------------------------------
# -------------------------------
# Orbit fit (reusing your functions when available)
# -------------------------------
from typing import Any

@dataclass
class FitCandidate:
    period: float
    summary: Dict[str, Any]
    args: Dict[str, Any]
    bounds: Tuple[float, float]
    result: Any = None              # <-- add this
    images: List[Path] = None
    saved_dir: Optional[Path] = None


@dataclass
class OrbitOutcome:
    summary_row: Dict[str, object]
    best_period: Optional[float]
    ls_path: Optional[Path]
    pdc_path: Optional[Path]
    report_dir: Optional[Path]
    extra_images: List[Path] = None
    candidates: List[FitCandidate] = None
    best_index: Optional[int] = None
    # all non-periodogram PNGs


# ---- helper: pick top-N periods from an LS power spectrum ----
def _top_periods_from_ls(freq: np.ndarray, power: np.ndarray, k: int,
                         min_rel_sep: float = 0.01) -> List[float]:
    """
    Return up to k strongest LS period candidates (in days), enforcing a
    minimum relative separation in period space to avoid near-duplicates.
    """
    from scipy.signal import find_peaks
    if freq is None or power is None or len(freq) != len(power) or len(freq) == 0:
        return []
    P = 1.0 / np.maximum(freq, 1e-12)        # days
    peaks, _ = find_peaks(power, distance=max(1, len(power)//200))
    if peaks.size == 0:
        # fallback: just take global max if any
        j = int(np.argmax(power))
        return [float(P[j])]
    # sort peaks by power desc
    order = np.argsort(power[peaks])[::-1]
    picked = []
    for j in order:
        p = float(P[peaks[j]])
        if not np.isfinite(p) or p <= 0:
            continue
        if all(abs(p - q)/q >= min_rel_sep for q in picked):
            picked.append(p)
        if len(picked) >= k:
            break
    return picked

# ---- Period-candidate helpers (ported from your script) ----
from scipy.signal import find_peaks, peak_widths

def _period_errors_from_peak_fwhm(freq: np.ndarray, power: np.ndarray) -> Dict[float, float]:
    """Map {P_at_peak: sigma_P} using FWHM in frequency and dP = P^2 df."""
    peaks, _ = find_peaks(power)
    if len(peaks) == 0:
        return {}
    widths, _, left_ips, right_ips = peak_widths(power, peaks, rel_height=0.5)
    idx = np.arange(len(freq))
    f_at = lambda x: np.interp(x, idx, freq)
    perr: Dict[float, float] = {}
    for pk, li, ri in zip(peaks, left_ips, right_ips):
        f0 = float(freq[pk])
        if f0 <= 0:
            continue
        fwhm = abs(f_at(ri) - f_at(li))
        sigma_f = fwhm / 2.355 if fwhm > 0 else 0.0
        P = 1.0 / f0
        sigma_P = (P * P) * sigma_f if sigma_f > 0 else 0.0
        perr[float(P)] = float(sigma_P)
    return perr

def significant_periods(periods: np.ndarray, powers: np.ndarray,
                        max_periods: int = 15, min_separation: float = 1.2) -> List[float]:
    """Take power peaks, sort by power, keep up to max_periods, enforcing log-space separation ≥ ln(min_separation)."""
    periods = np.asarray(periods, float)
    powers  = np.asarray(powers,  float)
    pk_idx, _ = find_peaks(powers)
    if pk_idx.size == 0:
        return []
    pkP = periods[pk_idx]; pkPow = powers[pk_idx]
    order = np.argsort(pkPow)[::-1]
    log_thresh = np.log(min_separation)
    picked, picked_log = [], []
    for j in order:
        lp = float(np.log(pkP[j]))
        if all(abs(lp - s) >= log_thresh for s in picked_log):
            picked.append(float(pkP[j]))
            picked_log.append(lp)
            if len(picked) >= max_periods:
                break
    return picked

def _sigma_from_map(P: float, err_map: Dict[float, float], rel_tol: float = 0.01) -> Optional[float]:
    """Nearest σP by log-distance ≤ rel_tol; else None."""
    if not err_map:
        return None
    keys = np.array(list(err_map.keys()), float)
    i = int(np.argmin(np.abs(np.log(P) - np.log(keys))))
    Q = float(keys[i])
    if abs(np.log(P) - np.log(Q)) > rel_tol:
        return None
    return float(err_map[Q])

def change_search_region_default(args_dict, field_name, init_val, min_val, max_val, vary):
    args_dict[LMFIT_PARAMS][SEARCH_REGION][field_name][INIT_VAL] = init_val
    args_dict[LMFIT_PARAMS][SEARCH_REGION][field_name][MIN_VAL] = min_val
    args_dict[LMFIT_PARAMS][SEARCH_REGION][field_name][MAX_VAL] = max_val
    args_dict[LMFIT_PARAMS][SEARCH_REGION][field_name][VARY]    = vary

def is_unphysical(P: float, e: float, p_min_days=P_SHORT_MAX, e_max_short=E_MAX_SHORT) -> bool:
    return (P < p_min_days) and (e > e_max_short)


def run_orbit_fit(mean_df: pd.DataFrame, star_name: str, out_dir: Path,
                  json_param_file: Optional[str],
                  top_k: int = 5, n_sigma: float = 2.0, min_sep: float = 1.2) -> OrbitOutcome:
    if not HAVE_ORBIT:
        st.warning("Orbit-fit stack not importable. Skipping fit.\n\nImport error: " + ORBIT_IMPORT_ERR)
        return OrbitOutcome({}, None, None, None, None, [], [], None)

    # --- keep everything in-memory (no temp CSV roundtrip) ---
    data = mean_df.rename(columns={'MJD': TIME_STAMPS, 'Mean RV': RADIAL_VELS, 'Mean RVsig': ERRORS}).copy()
    rvs   = data[RADIAL_VELS].to_numpy(float)
    mjds  = data[TIME_STAMPS].to_numpy(float)
    errvs = data[ERRORS].to_numpy(float)

    # --- periodograms (use GUI settings) ---
    pmin = float(st.session_state.get("periodogram_pmin", 1.0))
    pmax = float(st.session_state.get("periodogram_pmax", 15000.0))
    ls_p, ls_fap, ls_fal, freq_ls, pow_ls = ls(mjds, rvs, data_err=errvs, pmin=pmin, pmax=pmax)
    pdc_p, pdc_fap, freq_pdc, pow_pdc = pdc(mjds, rvs, data_err=errvs, pmin=pmin, pmax=pmax)

    # <<< ADDED >>>
    # Calculate period uncertainties from the FWHM of periodogram peaks, just like the standalone script.
    ls_err_map = _period_errors_from_peak_fwhm(freq_ls, pow_ls)
    pdc_err_map = _period_errors_from_peak_fwhm(freq_pdc, pow_pdc)

    star_dir = out_dir / star_name
    star_dir.mkdir(parents=True, exist_ok=True)
    plotls(freq_ls,  pow_ls,  ls_fal, pmin=pmin, pmax=pmax, star_id=star_name+"_LS",
           out_dir=str(star_dir / f"{star_name}_ls_periodogram.png"))
    plotls(freq_pdc, pow_pdc, [],     pmin=pmin, pmax=pmax, star_id=star_name+"_PDC",
           out_dir=str(star_dir / f"{star_name}_pdc_periodogram.png"))

    # --- candidates (ranked by peak power; keep only top_k) ---
    ls_list  = significant_periods(1.0/np.maximum(freq_ls,1e-12),  pow_ls,  max_periods=15, min_separation=min_sep)
    pdc_list = significant_periods(1.0/np.maximum(freq_pdc,1e-12), pow_pdc, max_periods=15, min_separation=min_sep)

    # seeds around the strongest peaks
    jitter = 0.0005 if top_k <= 3 else 0.0005
    seeds = []
    if np.isfinite(ls_p):  seeds += [("ls",  float(ls_p))]
    if np.isfinite(pdc_p): seeds += [("pdc", float(pdc_p))]

    raw = []
    for src, P0 in seeds:
        # The 'None' here is a placeholder for sigma_P, which we will look up next.
        raw += [(P0, None, src), (P0*(1.0-jitter), None, src), (P0*(1.0+jitter), None, src)]
    for P in (ls_list[:top_k] + pdc_list[:top_k]):
        raw.append((P, None, "ls/pdc"))

    def _power_at(P, freq, power):
        f = 1.0/max(P, 1e-12); j = int(np.argmin(np.abs(freq - f)))
        return float(power[j]) if 0 <= j < len(power) else 0.0

    # <<< MODIFIED >>>
    # This block now correctly looks up sigma_P for each candidate period.
    scored = []
    seen = []
    # Note: sP from `raw` is ignored (it was a placeholder), we look it up fresh.
    for P, _, src_in in raw:
        if not (pmin < P < pmax):
            continue
        if any(abs(np.log(P) - np.log(q)) <= 0.01 for q in seen):
            continue

        # Look up sigma from the map corresponding to the stronger periodogram peak
        power_ls = _power_at(P, freq_ls, pow_ls)
        power_pdc = _power_at(P, freq_pdc, pow_pdc)
        score = max(power_ls, power_pdc)

        # Prefer LS map if powers are equal or LS is stronger
        if power_ls >= power_pdc:
            sP = _sigma_from_map(P, ls_err_map)
            src = 'ls'
        else:
            sP = _sigma_from_map(P, pdc_err_map)
            src = 'pdc'

        scored.append((score, P, sP, src))
        seen.append(P)

    scored.sort(reverse=True)
    picked = scored[:max(1, int(top_k))]
    # The final `candidates` list now contains tuples of (Period, sigma_Period, source)
    candidates = [(P, sP, src) for (_sc, P, sP, src) in picked]

    # --- args template ---
    if json_param_file and Path(json_param_file).is_file():
        with open(json_param_file, 'r') as jf:
            args_template = json.load(jf)
    else:
        args_template = {LMFIT_PARAMS: {SEARCH_REGION: {}}}
    sr = args_template.setdefault(LMFIT_PARAMS, {}).setdefault(SEARCH_REGION, {})
    for fld in (PERIOD, ECC, K1_STR, OMEGA, T, GAMMA):
        sr.setdefault(fld, {INIT_VAL: 0.0, MIN_VAL: -0.1, MAX_VAL: 0.1, VARY: True})
    change_search_region_default(args_template, GAMMA, float(np.nanmean(rvs)),
                                 float(np.nanmin(rvs)), float(np.nanmax(rvs)), True)

    is_binary_th  = binary_rv_threshold(rvs, errvs, drv_tresh=20.0, sign_threshold=4.0)

    import copy as _copy, tempfile
    fit_list: List[FitCandidate] = []

    # --- compute the null model ONCE (F-test) ---
    with tempfile.TemporaryDirectory() as td_all:
        args_null = _copy.deepcopy(args_template)
        srn = args_null[LMFIT_PARAMS][SEARCH_REGION]
        srn[PERIOD].update({INIT_VAL: 0.0, MIN_VAL: -0.1, MAX_VAL: 0.1, VARY: False})
        srn[K1_STR].update({INIT_VAL: 0.0, MIN_VAL: -0.1, MAX_VAL: 0.1, VARY: False})
        srn[ECC].update({INIT_VAL: 0.0, MIN_VAL: 0.0,  MAX_VAL: 0.0,  VARY: False})
        srn[OMEGA].update({INIT_VAL: 0.0, MIN_VAL: 0.0,  MAX_VAL: 0.0,  VARY: False})
        srn[T].update({INIT_VAL: 0.0, MIN_VAL: 0.0,  MAX_VAL: 0.0,  VARY: False})
        srn[GAMMA].update({VARY: True})
        res_null = lmfit_on_sample(args_null, td_all, data, star_name, null_hyp=True)
        p_null = int(getattr(res_null, 'nvarys', 1))
        if hasattr(res_null, 'chisqr'):
            chi2_null = float(res_null.chisqr)
        else:
            chi2_null = float(getattr(res_null, 'redchi', np.nan)) * (len(data[TIME_STAMPS]) - p_null)

        # --- fit only the chosen candidates ---
        for P0, sP, _src in candidates:
            args = _copy.deepcopy(args_template)

            if sP and sP > 0:
                lower = max(1.0,  P0 - n_sigma*sP)
                upper = min(1.0e4, P0 + n_sigma*sP)
                st.info(f"Testing P={P0:.6f} d (σ={sP:.6f} d). Search range: [{lower:.6f}, {upper:.6f}] d")
            else:
                lower = P0*(1.0-0.0005)
                upper = P0*(1.0+0.0005)
                st.info(f"Testing P={P0:.6f} d (σ not found). Using narrow search range: [{lower:.6f}, {upper:.6f}] d")

            change_search_region_default(args, PERIOD, P0, lower, upper, True)
            if not (lower < upper):
                pad = max(5.0*(sP or 0.0), 1e-3*P0, 1e-6)
                lower = max(1.0,  P0 - pad)
                upper = min(1.0e4, P0 + pad)
            pmin_i, pmax_i = (lower, upper) if lower < upper else (upper, lower)
            ls_p_i,  ls_fap_i, _ls_fal_i, _, _ = ls(mjds, rvs, data_err=errvs, pmin=pmin_i, pmax=pmax_i)
            pdc_p_i, pdc_fap_i,            _, _ = pdc(mjds, rvs, data_err=errvs, pmin=pmin_i, pmax=pmax_i)

            fap_thr = float(st.session_state.get("fap_percent", 0.1)) / 100.0
            is_binary_ls_i = bool(np.isfinite(ls_fap_i) and (ls_fap_i < fap_thr))
            is_binary_pdc_i = bool(np.isfinite(pdc_fap_i) and (pdc_fap_i < fap_thr))

            bin_flag_i = (is_binary_pdc_i << 2) | (is_binary_ls_i << 1) | (is_binary_th << 0)

            if P0 < P_SHORT_MAX:
                eblk = args[LMFIT_PARAMS][SEARCH_REGION][ECC]
                change_search_region_default(args, ECC, eblk[INIT_VAL], eblk[MIN_VAL], E_MAX_SHORT, True)

            res = lmfit_on_sample(args, td_all, data, star_name)

            row = summarize_result(res, star_name) or {}
            # backfill if summarize_result omitted fields
            try:
                if hasattr(res, "params"):
                    if PERIOD in res.params and not np.isfinite(pick_num(row, 'period_value', 'Period', 'period', 'P')):
                        row['period_value'] = float(res.params[PERIOD].value)
                    if K1_STR in res.params and not np.isfinite(pick_num(row, 'k1_value', 'K1', 'k1')):
                        row['k1_value'] = float(res.params[K1_STR].value)
                    # Always take e from the fitted params to avoid stale values from summarize_result
                    if ECC in res.params:
                        row['ecc_value'] = float(res.params[ECC].value)
                    if GAMMA in res.params and not np.isfinite(pick_num(row, 'gamma_value', 'gamma', 'Gamma', 'V0')):
                        row['gamma_value'] = float(res.params[GAMMA].value)
                    if OMEGA in res.params and not np.isfinite(
                            pick_num(row, 'omega_value', 'omega', 'Omega', 'w', 'argperi', 'arg_peri')):
                        row['omega_value'] = float(res.params[OMEGA].value)
                    if PHASE0 in res.params and PERIOD in res.params:
                        phi0_fit = float(res.params[PHASE0].value)
                        p_fit = float(res.params[PERIOD].value)
                        t_ref = float(np.median(data[TIME_STAMPS].to_numpy(float)))

                        # Calculate absolute T0 for reference, just as before
                        t0_absolute = t_ref + phi0_fit * p_fit
                        row['t_value_mjd'] = t0_absolute

                        time_diff = t0_absolute - t_ref
                        t0_offset = (time_diff / p_fit - np.round(time_diff / p_fit)) * p_fit
                        row['t_value'] = t0_offset
            except Exception:
                pass
            for _key, _attr in [('chisqr','chisqr'),('redchi','redchi'),('aic','aic'),('bic','bic'),
                                ('nvarys','nvarys'),('ndata','ndata')]:
                if _key not in row and hasattr(res, _attr):
                    try: row[_key] = float(getattr(res, _attr))
                    except Exception: row[_key] = getattr(res, _attr)

            final_P = pick_num(row, 'period_value','Period','period','P') or P0
            ecc_fit = pick_num(row, 'ecc_value','Eccentricity','e')
            if not np.isfinite(ecc_fit): ecc_fit = 0.0
            if is_unphysical(final_P, ecc_fit):
                continue

            # F-test using the one null model
            from scipy.stats import f as _f
            N      = int(len(data[TIME_STAMPS]))
            p_orb  = int(row.get('nvarys', getattr(res, 'nvarys', 0)))
            chi2_o = float(row.get('chisqr', getattr(res, 'chisqr', np.nan)))
            F_stat = F_pvalue = F_crit_99 = np.nan
            if np.isfinite(chi2_o) and np.isfinite(chi2_null) and (N > p_orb) and (p_orb > p_null) and (chi2_null > chi2_o):
                dfn = int(p_orb - p_null)
                dfd = int(N - p_orb)
                F_stat = ((chi2_null - chi2_o) / dfn) / (chi2_o / dfd)
                F_pvalue = float(1.0 - _f.cdf(F_stat, dfn=dfn, dfd=dfd))
                F_crit_99 = float(_f.ppf(0.99, dfn=dfn, dfd=dfd))
            row.update({
                'ls_best_period': float(ls_p_i) if np.isfinite(ls_p_i) else np.nan,
                'pdc_best_period': float(pdc_p_i) if np.isfinite(pdc_p_i) else np.nan,
                'ls_fap': float(ls_fap_i), 'pdc_fap': float(pdc_fap_i),
                'F_stat': float(F_stat), 'F_pvalue': float(F_pvalue), 'F_crit_99': float(F_crit_99),
                'drv_dec': bool(is_binary_th), 'ls_fap_dec': bool(is_binary_ls_i), 'pdc_fap_dec': bool(is_binary_pdc_i),
                'bin_flag': int(bin_flag_i)
            })


            fit_list.append(FitCandidate(
                period=float(final_P), summary=row, args=args, bounds=(lower, upper), result=res
            ))

    if not fit_list:
        return OrbitOutcome({}, None,
                            star_dir / f"{star_name}_ls_periodogram.png",
                            star_dir / f"{star_name}_pdc_periodogram.png",
                            star_dir, [], [], None)

    # choose best by |redchi-1| then chisqr
    def _score(fc: FitCandidate):
        r = fc.summary
        rc = float(r.get('redchi', np.inf)); cs = float(r.get('chisqr', np.inf))
        return (abs(rc-1.0) if np.isfinite(rc) else np.inf, cs)

    fit_list_sorted = sorted(fit_list, key=_score)
    best = fit_list_sorted[0]


    return OrbitOutcome(
        summary_row=best.summary,
        best_period=best.summary.get('period_value', best.period),
        ls_path=star_dir / f"{star_name}_ls_periodogram.png",
        pdc_path=star_dir / f"{star_name}_pdc_periodogram.png",
        report_dir=star_dir,
        extra_images=[],  # Return an empty list as no plots are saved automatically
        candidates=fit_list_sorted[:top_k],
        best_index=0
    )

import sympy as sp

M2 = sp.symbols('M2', positive=True)


def companion_mass_min(P_day: float, K1_kms: float, e: float, M1: float,
                        M1_plus: float, M1_minus: float) -> Dict[str, float]:
    # mass function in Msun (P in d, K in km/s)
    f_m = (P_day * K1_kms**3 * (1 - e**2)**(1.5)) * 1.036149e-7

    def solve(M1i: float) -> float:
        expr = f_m * (M1i + M2)**2 - M2**3
        for g in [M1i, max(0.1, 0.3*M1i), 1.0, 3.0, 10.0]:
            try:
                val = float(sp.nsolve(expr, g, tol=1e-14, maxsteps=200))
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                continue
        return float('nan')

    return {
        "M2_min": solve(M1),
        "M2_min_upper": solve(M1 + max(0.0, M1_plus)),
        "M2_min_lower": solve(max(0.1, M1 + M1_minus)),  # M1_minus may already be negative
        "f_m": f_m,
    }


def roche_f(q: float) -> float:
    return 0.49*q**(2/3) / (0.6*q**(2/3) + np.log1p(q**(1/3)))


def P_massfunc(M1: float, q: float, K1: float, e: float, sini3: float) -> float:
    return C1 * (M1 * sini3) / (q * (1 + q)**2) * K1**-3 * (1 - e**2)**(-1.5)


def P_roche(M1: float, R_AU: float, q: float, e_eff: float, alpha: float) -> float:
    f_q = roche_f(q)
    return C2 * R_AU**1.5 / ((alpha * f_q)**1.5 * e_eff**1.5 * np.sqrt(M1 * (1 + 1/q)))

# ==== New helpers to find all roots for P_min calculation ====
Q_MIN, Q_MAX = 1e-6, 1e4
Q_GRID = np.logspace(np.log10(Q_MIN), np.log10(Q_MAX), 4000)


def find_all_roots(residual, q_grid=Q_GRID, tol_rel=1e-10):
    """
    Scans a grid for sign changes in the residual function, brackets every
    potential root, and then refines each one using brentq.

    Returns a sorted list of all distinct roots found (e.g., [], [q1], or [q1, q2]).
    """
    try:
        y = np.array([residual(q) for q in q_grid])
        # Filter out any non-finite results from the residual function
        mask = np.isfinite(y)
        qg, yg = q_grid[mask], y[mask]

        # Find indices where the sign changes between adjacent points
        idx = np.where(np.sign(yg[:-1]) * np.sign(yg[1:]) < 0)[0]

        roots = []
        for i in idx:
            q1, q2 = qg[i], qg[i + 1]
            try:
                # Refine the root within the bracketed interval [q1, q2]
                r = brentq(residual, q1, q2)
                roots.append(float(r))
            except (ValueError, RuntimeError):
                # brentq can fail if the bracket is not valid despite the sign change
                pass

        roots.sort()

        # De-duplicate roots that are very close to each other
        dedup = []
        if roots:
            dedup.append(roots[0])
            for r in roots[1:]:
                if abs(r - dedup[-1]) > tol_rel * dedup[-1]:
                    dedup.append(r)
        return dedup
    except (ValueError, ZeroDivisionError):
        # Return empty list if the residual function fails globally
        return []


def Pmin_peri_apa(M1: float, R_Rsun: float, K1: float, e: float, alpha_peri: float = 1.2, alpha_apa: float = 1.0,
                  i_deg: float = 90.0) -> Dict[str, object]:
    R_AU = R_Rsun * RSUN2AU
    sini3 = np.sin(np.radians(i_deg)) ** 3

    # --- Periastron Solutions ---
    res_peri = lambda q: P_roche(M1, R_AU, q, 1 - e, alpha_peri) - P_massfunc(M1, q, K1, e, sini3)
    q_roots_peri = find_all_roots(res_peri)
    peri_sols = [{"q": q, "Pmin": float(P_massfunc(M1, q, K1, e, sini3))} for q in q_roots_peri]
    has_two_peri_solutions = (len(peri_sols) == 2)

    # --- Apastron Solutions ---
    res_apa = lambda q: P_roche(M1, R_AU, q, 1 + e, alpha_apa) - P_massfunc(M1, q, K1, e, sini3)
    q_roots_apa = find_all_roots(res_apa)
    apa_sols = [{"q": q, "Pmin": float(P_massfunc(M1, q, K1, e, sini3))} for q in q_roots_apa]
    has_two_apa_solutions = (len(apa_sols) == 2)

    # --- Determine the primary solution ---
    # If two solutions exist, choose the one with the lowest Pmin (most restrictive constraint).
    # The corresponding 'q' will be returned with it.
    best_peri_sol = min(peri_sols, key=lambda s: s["Pmin"]) if peri_sols else None
    P_peri = best_peri_sol["Pmin"] if best_peri_sol else np.nan
    q_peri = best_peri_sol["q"] if best_peri_sol else np.nan

    best_apa_sol = min(apa_sols, key=lambda s: s["Pmin"]) if apa_sols else None
    P_apa = best_apa_sol["Pmin"] if best_apa_sol else np.nan
    q_apa = best_apa_sol["q"] if best_apa_sol else np.nan

    # Best-case periastron (as q -> infinity)
    f_inf = 0.49 / 0.6
    P_peri_best = C2 * R_AU ** 1.5 / ((alpha_peri * f_inf) ** 1.5 * (1 - e) ** 1.5 * np.sqrt(M1))

    return {
        "Pmin_peri": P_peri,
        "q_at_min_peri": q_peri,
        "Pmin_apa": P_apa,
        "q_at_min_apa": q_apa,
        "Pmin_peri_best": P_peri_best,
        "has_two_peri_solutions": has_two_peri_solutions,
        "has_two_apa_solutions": has_two_apa_solutions,
    }

def _nearest_key(x: float, d: Dict[float, List[Tuple[float,float]]], tol: float = 1.0) -> Optional[float]:
    """Return the key in d whose value is closest to x within ±tol Å, else None."""
    if not d:
        return None
    ks = np.array([float(k) for k in d.keys()], float)
    j = int(np.argmin(np.abs(ks - float(x))))
    return float(ks[j]) if abs(ks[j] - x) <= tol else None

# ---- Classifier helpers ------------------------------------------------------

def parse_ranges_text(txt: str) -> List[Tuple[float, float]]:
    """
    Parse 'a-b; c-d; ...' into [(a,b), (c,d), ...]. Ignores empties.
    """
    out = []
    for chunk in re.split(r"[;]+", txt.strip()):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"\s*([+-]?\d+(?:\.\d+)?)\s*[-–]\s*([+-]?\d+(?:\.\d+)?)\s*$", chunk)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if b < a: a, b = b, a
            out.append((a, b))
    return out

# --- SNR window helpers -------------------------------------------------------
from typing import Union, DefaultDict
SNRWindows = Union[List[Tuple[float, float]], Dict[float, List[Tuple[float, float]]]]

def merge_ranges(ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping [lo,hi] ranges; returns a simplified list."""
    if not ranges:
        return []
    segs = sorted([(min(a, b), max(a, b)) for a, b in ranges], key=lambda x: x[0])
    out = [segs[0]]
    for a, b in segs[1:]:
        lo, hi = out[-1]
        if a <= hi:
            out[-1] = (lo, max(hi, b))
        else:
            out.append((a, b))
    return out

def flatten_windows(win: SNRWindows) -> List[Tuple[float, float]]:
    """Turn dict-of-per-line windows into a merged global list (for weighting)."""
    if isinstance(win, dict):
        allr = [r for lst in win.values() for r in lst]
        return merge_ranges(allr)
    return merge_ranges(win or [])

def snr_from_windows_multi(wave: np.ndarray, flux: np.ndarray, windows: SNRWindows) -> float:
    """Global SNR using union of windows (dict → union)."""
    return snr_from_windows(wave, flux, flatten_windows(windows))

def add_vrects_to_plotly(fig, windows: SNRWindows, y0=0.0, y1=1.0):
    """Overlay shaded rectangles for chosen SNR windows (works for dict or list)."""
    rngs = flatten_windows(windows)
    for (lo, hi) in rngs:
        fig.add_vrect(x0=float(lo), x1=float(hi), opacity=0.2, line_width=0)

def snr_from_windows(wave: np.ndarray, flux: np.ndarray,
                     windows: List[Tuple[float, float]]) -> float:
    """
    SNR ~ mean / std computed over concatenated user windows.
    Falls back to whole spectrum if windows are empty / invalid.
    """
    if windows:
        mask = np.zeros_like(wave, dtype=bool)
        for lo, hi in windows:
            mask |= (wave >= lo) & (wave <= hi)
        if mask.sum() >= 10:
            f = np.nan_to_num(flux[mask])
            mu, sd = float(np.nanmean(f)), float(np.nanstd(f))
            return (mu / max(sd, 1e-12)) if np.isfinite(mu) and np.isfinite(sd) else 0.0
    # fallback
    f = np.nan_to_num(flux)
    mu, sd = float(np.nanmean(f)), float(np.nanstd(f))
    return (mu / max(sd, 1e-12)) if np.isfinite(mu) and np.isfinite(sd) else 0.0


def _build_log_grid(obs0_wave: np.ndarray, lam_ranges: List[Tuple[float,float]],
                    oversample: float = 1.0):
    """
    Make a single logarithmic grid that spans min(lam_ranges)→max(lam_ranges)
    at the native resolution of obs0 (optionally oversampled).
    Returns (wavegridlog, vbin, idx_blocks) where idx_blocks are (i_start, i_stop).
    """
    lam_lo = min(lo for lo, _ in lam_ranges)
    lam_hi = max(hi for _, hi in lam_ranges)
    lam_lo = max(lam_lo, float(np.nanmin(obs0_wave)))
    lam_hi = min(lam_hi, float(np.nanmax(obs0_wave)))

    dlam = float(np.nanmedian(np.diff(obs0_wave)))
    R = float(obs0_wave[1] / dlam) if dlam > 0 else 50_000.0  # crude
    if oversample and oversample > 1.0:
        R = R * oversample

    # constant log spacing → Δlnλ = 1/R
    n_waves = int(np.log(lam_hi/lam_lo) / np.log(1.0 + 1.0/R))
    wavegridlog = lam_lo * (1.0 + 1.0/R) ** np.arange(max(2, n_waves))

    # vbin ~ c * Δlnλ
    dln = float(np.log(wavegridlog[1]/wavegridlog[0]))
    vbin = CLIGHT * dln

    # blocks for each user window
    idx = np.arange(len(wavegridlog))
    blocks = []
    for lo, hi in lam_ranges:
        i0 = int(np.argmin(np.abs(wavegridlog - max(lo, lam_lo))))
        i1 = int(np.argmin(np.abs(wavegridlog - min(hi, lam_hi))))
        if i1 <= i0: continue
        blocks.append((i0, i1))
    return wavegridlog, vbin, blocks


def _xcorr_on_blocks(obs_flux_log: np.ndarray, tmpl_flux_log: np.ndarray,
                     blocks: List[Tuple[int,int]], vbin: float,
                     vmin: float, vmax: float, fit_fraction: float):
    """
    Cross-correlate over the union of log-grid blocks (like your crosscorreal).
    Returns vmax, sigma (curvature model), r2, peak, fwhm.
    """
    # indices to keep
    keep = np.concatenate([np.arange(i0, i1) for (i0, i1) in blocks]) if blocks else np.arange(len(obs_flux_log))
    # normalize / mean-subtract
    a = obs_flux_log[keep]   - np.nanmean(obs_flux_log[keep])
    b_all = tmpl_flux_log    - np.nanmean(tmpl_flux_log)

    s_range = np.arange(int(vmin/vbin), int(vmax/vbin), 1)
    velo = s_range * vbin
    # CCF curve by rolling the template and correlating on the masked region
    ccf = np.array([ccf_scalar(a, np.roll(b_all, s)[keep]) for s in s_range])

    i_max = int(np.argmax(ccf))
    cmax  = float(ccf[i_max])
    if i_max == 0 or i_max == len(ccf)-1:
        return np.nan, np.nan, np.nan, cmax, np.nan

    # FWHM (like your script)
    half = 0.5*cmax
    ii = np.nonzero(ccf >= half)[0]
    fwhm = float(velo[ii[-1]] - velo[ii[0]]) if ii.size > 1 else np.nan

    # Parabola fit around peak within fit_fraction of max
    left  = np.argmin(np.abs(fit_fraction*cmax - ccf[:i_max]))
    right = np.argmin(np.abs(fit_fraction*cmax - ccf[i_max+1:])) + i_max + 1
    left = max(0, left); right = min(right, len(ccf)-1)
    if right <= left: return np.nan, np.nan, np.nan, cmax, fwhm
    a2, b2, c2 = np.polyfit(velo[left:right+1], ccf[left:right+1], 2)
    v_peak = -b2/(2*a2)
    c_at   = min(1.0 - 1e-12, c2 - (b2*b2)/(4*a2))

    # "R²" goodness around the fit window (same metrics you printed)
    fit_vals = a2*velo[left:right+1]**2 + b2*velo[left:right+1] + c2
    resid    = ccf[left:right+1] - fit_vals
    ss_res   = float(np.sum(resid**2))
    ss_tot   = float(np.sum((ccf[left:right+1] - np.mean(ccf[left:right+1]))**2))
    r2 = 1.0 - (ss_res/max(ss_tot, 1e-30))

    # Curvature-based sigma (like your legacy formula)
    Nres = int(len(keep))
    CCFdvdvAtMax = 2.0*a2
    sigma = np.sqrt(max(0.0, (1.0 - c_at*c_at) / max(Nres * abs(CCFdvdvAtMax) * c_at, 1e-30)))

    return float(v_peak), float(sigma), float(r2), float(cmax), float(fwhm)


def _snr_continuum_default(line_center: float) -> Optional[Tuple[float,float]]:
    # Your original continuum windows
    if 4000 <= line_center < 4160:   return 4045, 4060
    if 4160 <= line_center < 4400:   return 4220, 4230
    if 4400 <= line_center <= 4570:  return 4495, 4550
    return None

def _classify_lines_for_star(
    star_folder: Path,
    obs_list: List[Obs],
    lam_ranges: List[Tuple[float,float]],
    vmin: float, vmax: float, fit_fraction: float,
    oversample: float,
    snr_windows: Optional[Union[List[Tuple[float,float]], Dict[float, List[Tuple[float,float]]]]],
    auto_detect: bool,
    manual_lines: List[float],
    detect_span: Tuple[float,float],
    tolerance: float,
    window_H: float, window_He: float,
    P_level: float,
    known_sets: Dict[str, List[float]],
    template_mode: str,
    model_root: str = ""
):

    """
    Returns: (classification_str, per_line_results_df, coadd_wave, coadd_flux)
    """
    if not obs_list:
        return "No data", pd.DataFrame(), None, None

    # ---------- Choose template ----------
    if template_mode == "mask":
        mask_files = sorted(star_folder.glob("*Combined*.fits"))
        if not mask_files:
            st.warning("No *Combined*.fits found; falling back to first observation.")
            template_mode = "first"
        else:
            tw, tf = read_fits(mask_files[0])
    if template_mode == "model":
        tpl = load_model_template(star_folder, model_root)
        if tpl is None:
            st.warning("No model template found; falling back to first observation.")
            template_mode = "first"
        else:
            tw, tf = tpl
    if template_mode == "first":
        first = obs_list[0].spectrum
        tw, tf = first[:,0], np.nan_to_num(first[:,1])

    # ---------- Build log-grid once ----------
    obs0 = obs_list[0].spectrum
    wavegridlog, vbin, blocks = _build_log_grid(obs0[:,0], lam_ranges, oversample=max(1.0, oversample))
    # interpolate template to log-grid
    tmpl_log = interp1d(tw, np.nan_to_num(tf), bounds_error=False, fill_value=1.0)(wavegridlog)

    # ---------- RVs by xcorr on full windows ----------
    Vs, sigs = [], []
    for o in obs_list:
        wl = o.spectrum[:,0]; fl = np.nan_to_num(o.spectrum[:,1])
        obs_log = interp1d(wl, fl, bounds_error=False, fill_value=1.0)(wavegridlog)
        v, s, r2, peak, fwhm = _xcorr_on_blocks(obs_log, tmpl_log, blocks, vbin, vmin, vmax, fit_fraction)
        Vs.append(0.0 if not np.isfinite(v) else float(v))
        sigs.append(np.nan if not np.isfinite(s) else float(s))

    # ---------- SNRs (user windows or union of CCF λ-windows) ----------
    snrs = []
    if snr_windows is None:
        # union of lam_ranges → default behavior
        default_windows = lam_ranges
        for o in obs_list:
            wl = o.spectrum[:, 0];
            fl = np.nan_to_num(o.spectrum[:, 1])
            snrs.append(snr_from_windows(wl, fl, default_windows))
    else:
        # list: use as-is; dict (per-line): union across lines for weighting
        for o in obs_list:
            wl = o.spectrum[:, 0];
            fl = np.nan_to_num(o.spectrum[:, 1])
            snrs.append(snr_from_windows_multi(wl, fl, snr_windows))

    # ---------- SNR^2-weighted coadd on the *linear* base grid ----------
    base_wave = obs0[:,0]
    coadd = np.zeros_like(base_wave)
    totw  = 0.0
    for o, S, v in zip(obs_list, snrs, Vs):
        w = max(1e-6, float(S))**2
        shifted = interp1d(o.spectrum[:,0]*(1 - v/CLIGHT),
                           np.nan_to_num(o.spectrum[:,1]),
                           bounds_error=False, fill_value=1.0)(base_wave)
        coadd += w * shifted
        totw  += w
    coadd = coadd / max(1e-12, totw)

    # ---------- line list ----------
    detect_lo, detect_hi = detect_span
    mask_det = (base_wave >= detect_lo) & (base_wave <= detect_hi)
    det_wave = base_wave[mask_det]; det_flux = coadd[mask_det]

    detected_lines: List[Dict[str,float]] = []
    if auto_detect:
        detected_lines = find_absorption_lines(det_wave, det_flux)
        # match to user-provided known set (if any) using tolerance
        known_lines = []
        for name, arr in known_sets.items():
            if arr: known_lines += arr
        if known_lines:
            matched, used = [], set()
            for center in known_lines:
                best, best_d, best_i = None, 1e9, None
                for i, d in enumerate(detected_lines):
                    if i in used: continue
                    dd = abs(float(d['center']) - center)
                    if dd <= tolerance and dd < best_d:
                        best, best_d, best_i = d, dd, i
                if best is not None:
                    matched.append(best); used.add(best_i)
            detected_lines = matched
    else:
        # manual: accept the user list as "detected" centers with dummy widths
        detected_lines = [{"center": float(L), "width": 20.0} for L in manual_lines]

    if not detected_lines:
        return "Inconclusive - no lines", pd.DataFrame(), base_wave, coadd

    # ---------- per-line χ²/dof vs coadd (shift coadd to each obs) ----------
    H_lines_default   = [3970.072, 4101.734, 4340.462]
    He_lines_default  = [4026.191, 4387.929, 4471.479]
    #He_ext_default    = [4143.760, 4120.0]
    He_ext_default = [4143.760]
    def _window_for(center):
        if any(abs(center - h) <= tolerance for h in H_lines_default):  return window_H
        if any(abs(center - e) <= tolerance for e in He_ext_default):   return window_He
        if any(abs(center - he)<= tolerance for he in He_lines_default):return window_He
        return 30.0

    chi_per_line = {k: [] for k in range(len(detected_lines))}
    dof_per_line = {k: [] for k in range(len(detected_lines))}

    for oi, o in enumerate(obs_list):
        w = o.spectrum[:,0]; f = np.nan_to_num(o.spectrum[:,1])
        # shift coadd into obs frame
        shifted_coadd = interp1d(base_wave*(1 + Vs[oi]/CLIGHT),
                                 coadd, bounds_error=False, fill_value=1.0)(w)

        for idx, line in enumerate(detected_lines):
            center = float(line["center"])
            win = _window_for(center)
            x_lo, x_hi = center - win/2, center + win/2
            mask_line = (w >= x_lo) & (w <= x_hi)
            if mask_line.sum() < 3:
                continue
            # Uncertainty from continuum:
            # Try per-line dict (nearest key within tolerance), else global list, else defaults.
            cont_mask = np.zeros_like(w, dtype=bool)
            if isinstance(snr_windows, dict) and snr_windows:
                k = _nearest_key(center, snr_windows, tol=tolerance)  # <— use nearest center
                winlist = snr_windows.get(k, []) if k is not None else []
                for lo, hi in winlist:
                    cont_mask |= (w >= float(lo)) & (w <= float(hi))

            # If dict had no usable windows (or not a dict), try global list
            if not np.any(cont_mask) and isinstance(snr_windows, list) and snr_windows:
                for lo, hi in snr_windows:
                    cont_mask |= (w >= float(lo)) & (w <= float(hi))

            # Final fallback: your built-in default band near this line
            if not np.any(cont_mask):
                rng = _snr_continuum_default(center)
                if rng:
                    cont_mask = (w >= float(rng[0])) & (w <= float(rng[1]))

            if np.any(cont_mask):
                s_obs = float(np.nanstd(f[cont_mask]))
                s_coa = float(np.nanstd(shifted_coadd[cont_mask]))
            else:
                s_obs = float(np.nanstd(f))
                s_coa = float(np.nanstd(shifted_coadd))
            sig = np.sqrt(max(1e-30, s_obs*s_obs + s_coa*s_coa))

            diff = f[mask_line] - shifted_coadd[mask_line]
            chi2 = float(np.sum((diff / sig)**2))
            dof  = int(mask_line.sum() - 1)

            chi_per_line[idx].append(chi2)
            dof_per_line[idx].append(dof)

    line_rows = []
    decisions = []  # True → SB2, False → SB1
    for idx, line in enumerate(detected_lines):
        chis = [c for c in chi_per_line[idx] if np.isfinite(c)]
        dofs = [d for d in dof_per_line[idx] if (d is not None and d > 0)]
        if chis and dofs:
            chi_tot = float(np.sum(chis))
            dof_tot = int(np.sum(dofs))
            red = chi_tot / max(dof_tot, 1)  # <-- canonical aggregate reduced χ²
            chi_thr = (2.0 * gammainccinv(dof_tot / 2.0, P_level)) / max(dof_tot, 1)
        else:
            red, dof_tot, chi_thr = np.nan, 0, np.nan

        is_sb2 = (np.isfinite(red) and np.isfinite(chi_thr) and (red > chi_thr))
        decisions.append(is_sb2)
        line_rows.append({
            "Star": star_folder.name,
            "Line Center (A)": float(line["center"]),
            "dof_total": dof_tot,
            "avg_reduced_chi": red,
            "chi2_threshold": chi_thr,
            "Line Classification": "SB2" if is_sb2 else "SB1",
        })

    # final classification (same rules)
    n = len(decisions)
    n_sb2 = int(sum(decisions))
    n_sb1 = n - n_sb2
    if n == 0:
        final = "Inconclusive - no lines"
    elif n_sb2 == n:
        final = "SB2"
    elif n_sb1 == n:
        final = "SB1"
    elif n_sb2 == n-1:
        diffs = [r for r, d in zip(line_rows, decisions) if not d]
        info  = ", ".join([f"{r['Line Center (A)']:.2f}, {r['avg_reduced_chi']:.2f}, {r['chi2_threshold']:.2f}" for r in diffs])
        final = f"All lines except [{info}] agreed and classified as SB2"
    elif n_sb1 == n-1:
        diffs = [r for r, d in zip(line_rows, decisions) if d]
        info  = ", ".join([f"{r['Line Center (A)']:.2f}, {r['avg_reduced_chi']:.2f}, {r['chi2_threshold']:.2f}" for r in diffs])
        final = f"All lines except [{info}] agreed and classified as SB1"
    elif (n_sb2 == n-2) or (n_sb1 == n-2):
        typ  = "SB2" if (n_sb2 == n-2) else "SB1"
        diffs = [r for r, d in zip(line_rows, decisions) if (d if typ=="SB1" else not d)]
        info  = ", ".join([f"{r['Line Center (A)']:.2f}, {r['avg_reduced_chi']:.4f}, {r['chi2_threshold']:.4f}" for r in diffs])
        final = f"Two lines [{info}] do not agree, classified as {typ}"
    else:
        final = "Inconclusive - Lines disagree"

    df_lines = pd.DataFrame(line_rows)
    return final, df_lines, base_wave, coadd


def animate_spectra_data(
    obs_list: List[Obs],
    star_name: str,
    output_dir: Path,
    wavelength_regions: List[Tuple[float, float]] = None,
    fps: float = 6.0,
    fmt: str = "mp4",           # "mp4" (preferred) or "gif"
    max_points: int = 1200,     # per panel
    dpi: int = 90,
    normalize_flux: bool = True
) -> Optional[Path]:
    """
    Fast, flicker-free animation of spectra over time.
    - Reuses Line2D artists (no ax.clear()).
    - Fixed y-lims per panel (computed from all frames).
    - Uses blit=True.
    - Writes MP4 if ffmpeg is available; else GIF.
    """
    if not HAVE_ANIMATION:
        st.error("Animation requires matplotlib.animation.")
        return None
    if not obs_list:
        st.error("No spectra data available for animation")
        return None

    try:
        # sort by time
        sorted_obs = sorted(obs_list, key=lambda o: o.mjd)

        # default regions
        if wavelength_regions is None:
            lines = [4026.0, 4101.7, 4340.5, 4387.9, 4471.5]
            radius = 15.0
            wavelength_regions = [(l - radius, l + radius) for l in lines]

        # --- build a common linear grid per region and precompute all frames on those grids
        def _grid(lo, hi, n=max_points):
            n = max(50, min(n, 20000))
            return np.linspace(lo, hi, n, dtype=float)

        # choose a “reference” wavelength array per frame (just the file’s)
        frames = []
        for o in sorted_obs:
            w = o.spectrum[:, 0].astype(float)
            f = np.nan_to_num(o.spectrum[:, 1].astype(float))
            if normalize_flux:
                med = np.nanmedian(f)
                if np.isfinite(med) and med > 0:
                    f = f / med
            frames.append((o.mjd, o.snr, w, f))

        # precompute y for each (frame, region) on a shared X grid
        Xs = []
        Ys = []  # shape: [n_frames][n_regions] each an array
        ymins = []
        ymaxs = []
        for (lo, hi) in wavelength_regions:
            Xs.append(_grid(lo, hi, max_points))

        for (mjd, snr, w, f) in frames:
            row = []
            for X in Xs:
                # interpolate onto the shared grid
                y = np.interp(X, w, f, left=np.nan, right=np.nan)
                # trim nans at the edges
                if np.isnan(y).any():
                    good = np.isfinite(y)
                    if good.sum() >= 10:
                        y = y[good]
                        x_trim = X[good]
                        # re-expand to match X length with a light rolling mean to avoid ragged edges
                        y = np.interp(X, x_trim, y, left=np.nanmedian(y), right=np.nanmedian(y))
                row.append(y)
            Ys.append(row)

        # compute panel-wise y-lims across all frames (robust)
        for j in range(len(wavelength_regions)):
            all_y = np.concatenate([row[j] for row in Ys])
            lo = np.nanpercentile(all_y, 1)
            hi = np.nanpercentile(all_y, 99)
            pad = 0.08 * (hi - lo if np.isfinite(hi - lo) else 1.0)
            ymins.append(lo - pad)
            ymaxs.append(hi + pad)

        # --- figure & artists (one line per panel)
        n_regions = len(wavelength_regions)
        fig, axes = plt.subplots(n_regions, 1, figsize=(9, 2.6 * n_regions), squeeze=False)
        axes = axes.ravel().tolist()

        lines = []
        vlines = []
        for ax, (X, (lo, hi), ymin, ymax) in zip(axes, zip(Xs, wavelength_regions, ymins, ymaxs)):
            (ln,) = ax.plot(X, Ys[0][j], lw=1.1) # temporary y; will reset in init
            lines.append(ln)
            ax.set_xlim(lo, hi)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("Wavelength (Å)")
            ax.set_ylabel("Normalized Flux")
            ax.grid(True, alpha=0.3)
            mid = 0.5 * (lo + hi)
            vlines.append(ax.axvline(mid, ls="--", alpha=0.35, lw=0.8))

        title = fig.suptitle("", fontsize=14, y=0.98)

        # --- init & update (blit)
        def init():
            # first frame
            mjd0, snr0, *_ = frames[0]
            title.set_text(f"{star_name} • MJD {mjd0:.2f} • SNR {snr0:.1f} • {len(frames)} epochs")
            for j, ln in enumerate(lines):
                ln.set_ydata(Ys[0][j])
            artists = [title] + lines + vlines
            return artists

        def update(i):
            mjd, snr, *_ = frames[i]
            title.set_text(f"{star_name} • MJD {mjd:.2f} • SNR {snr:.1f} • frame {i+1}/{len(frames)}")
            for j, ln in enumerate(lines):
                ln.set_ydata(Ys[i][j])
            artists = [title] + lines  # vlines are static
            return artists

        # ---- MJD-paced frames (paste right before creating FuncAnimation) ----
        mjds = np.array([m for (m, *_) in frames], float)
        dt = np.diff(mjds)

        if dt.size == 0:
            intervals = np.array([1000.0 / max(1.0, fps)], float)
        else:
            # scale around median cadence so long gaps don't create super-long frames
            base = 1000.0 / max(1.0, fps)
            intervals = (dt / np.median(dt)) * base
            intervals = np.append(intervals, intervals[-1])  # one interval per frame

        def frame_gen():
            for i, ms in enumerate(intervals):
                ani.event_source.interval = float(ms)
                yield i

        # ----------------------------------------------------------------------

        ani = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=frame_gen(),  # <— use the generator
            blit=False,
            interval=float(intervals[0]),
            repeat=True,
        )

        # --- writer selection
        fmt = fmt.lower()
        if fmt == "mp4" and matplotlib.animation.writers.is_available("ffmpeg"):
            output_path = (output_dir / f"{star_name}_spectra_animation.mp4").resolve()
            ani.save(str(output_path), writer="ffmpeg", dpi=dpi, fps=fps, bitrate=1800)
        else:
            # GIF fallback (bigger + 256 colors)
            from matplotlib.animation import PillowWriter
            output_path = (output_dir / f"{star_name}_spectra_animation.gif").resolve()
            ani.save(str(output_path), writer=PillowWriter(fps=fps), dpi=dpi)

        plt.close(fig)
        return output_path

    except Exception as e:
        plt.close("all")
        st.error(f"Animation creation failed: {e}")
        return None


def create_phase_folded_animation(rv_df, orbital_params, star_name, output_dir,
                                  fps=6.0, fmt="mp4", dpi=90):
    if not HAVE_ANIMATION:
        return None
    try:
        # --- params from fit (may be NaN) ---
        P  = float(orbital_params.get('P', np.nan))
        K1 = float(orbital_params.get('K1', np.nan))
        e  = float(orbital_params.get('e', np.nan))
        om = float(orbital_params.get('omega', np.nan))
        T0 = float(orbital_params.get('T', np.nan))
        ga = float(orbital_params.get('gamma', np.nan))

        if not np.isfinite(P) or P <= 0:
            raise ValueError("Invalid period P")

        # --- use weighted-mean RVs (and drop NaNs) ---
        if {"MJD","Mean RV","Mean RVsig"}.issubset(rv_df.columns):
            times  = pd.to_numeric(rv_df["MJD"],      errors="coerce").to_numpy(float)
            rvs    = pd.to_numeric(rv_df["Mean RV"],  errors="coerce").to_numpy(float)
            errors = pd.to_numeric(rv_df["Mean RVsig"], errors="coerce").to_numpy(float)
        else:
            raise ValueError("rv_df must contain columns: MJD, Mean RV, Mean RVsig")

        m = np.isfinite(times) & np.isfinite(rvs)
        times, rvs = times[m], rvs[m]
        if times.size == 0:
            raise ValueError("No finite RV points to plot")

        # --- sane fallbacks ---
        if not np.isfinite(K1):
            K1 = 0.5*(np.nanpercentile(rvs,95)-np.nanpercentile(rvs,5))
        if not np.isfinite(ga):
            ga = float(np.nanmean(rvs))
        if not np.isfinite(T0):
            T0 = float(np.nanmin(times))

        if not np.isfinite(e):
            e = 0.0
        if np.isfinite(om) and abs(om) > 2 * np.pi + 1e-6:
            om = np.deg2rad(om)
        om = float(0.0 if not np.isfinite(om) else om)
        # --- require a complete set of orbital params ---
        for name, val in dict(P=P, K1=K1, e=e, omega=om, T=T0, gamma=ga).items():
            if not np.isfinite(val):
                raise ValueError(f"Missing/invalid parameter from fit: {name}")

        # keep degree→radian conversion if the caller gave degrees
        if abs(om) > 2 * np.pi + 1e-6:
            om = np.deg2rad(om)

        # e hygiene (mirror negatives, clamp)
        if e < 0:
            e = -e
            om = (om + np.pi) % (2 * np.pi)
        e = min(e, 0.99)

        T_ref = T0
        phases = np.mod((times - T_ref) / P, 1.0)

        # --- model curve over one cycle ---
        ph_model = np.linspace(0,1,400)
        M = 2*np.pi*ph_model
        E = M.copy()
        for _ in range(30):
            E -= (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
        nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
        rv_curve = ga + K1*(np.cos(nu + om) + e*np.cos(om))

        # --- figure & animation ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5))

        (line_time,)  = ax1.plot([], [], 'o', ms=10)
        ax1.set_xlabel('MJD'); ax1.set_ylabel('RV (km/s)'); ax1.grid(True, alpha=0.3)

        (line_model,) = ax2.plot(ph_model, rv_curve, '-', lw=2)
        (scat_phase,) = ax2.plot([], [], 'o', ms=10)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Orbital Phase'); ax2.set_ylabel('RV (km/s)'); ax2.grid(True, alpha=0.3)

        ylo, yhi = np.nanpercentile(rvs, 2), np.nanpercentile(rvs, 98)
        ypad = 0.08 * (yhi - ylo if np.isfinite(yhi - ylo) else 1.0)
        ax1.set_ylim(ylo - ypad, yhi + ypad)
        ax2.set_ylim(ylo - ypad, yhi + ypad)

        xlo, xhi = float(np.min(times)), float(np.max(times))
        xpad = 0.05 * max(1.0, xhi - xlo)
        ax1.set_xlim(xlo - xpad, xhi + xpad)

        title = fig.suptitle("", fontsize=14, y=0.98)

        def init():
            title.set_text(f"{star_name} • P={P:.3f} d, K1={K1:.1f} km/s, e={e:.2f}")
            line_time.set_data([], [])
            scat_phase.set_data([], [])
            return [title, line_time, scat_phase, line_model]

        def update(i):
            line_time.set_data(times[:i + 1], rvs[:i + 1])
            scat_phase.set_data(phases[:i + 1], rvs[:i + 1])
            title.set_text(f"{star_name} • MJD {times[i]:.2f} • Point {i + 1}/{len(rvs)}")
            return [title, line_time, scat_phase, line_model]
        ani = FuncAnimation(fig, update, init_func=init, frames=len(rvs),
                            blit=False, interval=1000.0/max(1.0, fps), repeat=True)

        fmt_l = fmt.lower()
        if fmt_l == "mp4" and matplotlib.animation.writers.is_available("ffmpeg"):
            out = (output_dir / f"{star_name}_phase_animation.mp4").resolve()
            ani.save(str(out), writer="ffmpeg", fps=fps, dpi=dpi, bitrate=1800)
        else:
            from matplotlib.animation import PillowWriter
            out = (output_dir / f"{star_name}_phase_animation.gif").resolve()
            ani.save(str(out), writer=PillowWriter(fps=fps), dpi=dpi)

        plt.close(fig)
        return out

    except Exception as e:
        plt.close('all')
        st.error(f"Phase animation creation failed: {e}")
        return None




def _lookup_mass_row(mass_csv_path: str, star_base: str):
    """Return M1, M1_plus, M1_minus, R_star, R_plus, R_minus for the given star (NaNs if missing)."""
    mspec = ms_plus = ms_minus = rstar = r_plus = r_minus = np.nan
    if mass_csv_path and Path(mass_csv_path).is_file():
        try:
            dfm = pd.read_csv(mass_csv_path)
            key = f"BLOeM_{star_base}"
            rowm = dfm.loc[dfm['ID'] == key]
            if rowm.empty:
                m = ID_RE.search(star_base or "")
                if m:
                    key = f"BLOeM_{m.group(0)}"
                    rowm = dfm.loc[dfm['ID'] == key]
            if not rowm.empty:
                mspec   = float(rowm['Mspec'].iloc[0])
                ms_plus = float(rowm['Mspec_er_plus'].iloc[0])
                ms_minus= float(rowm['Mspec_er_minus'].iloc[0])
                rstar   = float(rowm.get('R_star', pd.Series([np.nan])).iloc[0])
                r_plus  = float(rowm.get('R_star_er_plus',  pd.Series([np.nan])).iloc[0])
                r_minus = float(rowm.get('R_star_er_minus', pd.Series([np.nan])).iloc[0])
        except Exception:
            pass
    return mspec, ms_plus, ms_minus, rstar, r_plus, r_minus


def build_summary_row(star_name: str, summ: Dict[str, object],
                      mspec: float, ms_plus: float, ms_minus: float,
                      rstar: float, r_plus: float = np.nan, r_minus: float = np.nan,
                      alpha_peri: float = 1.2, alpha_apa: float = 1.0) -> Dict[str, object]:
    """
    Collects orbit params and derives M2,min and Pmin bounds, including a 3x3 grid
    calculation for the theoretical best-case Pmin.
    """
    # --- Step 1: Extract orbital parameters from the fit summary ---
    P = pick_num(summ, 'period_value', 'Period_value', 'Period', 'period', 'P')
    K1 = pick_num(summ, 'k1_value', 'K1_value', 'K1', 'k1')
    e = pick_num(summ, 'ecc_value', 'Eccentricity', 'e')
    gam = pick_num(summ, 'gamma_value', 'gamma', 'Gamma', 'V0')
    omega = pick_num(summ, 'omega_value', 'omega', 'Omega', 'w', 'argperi', 'arg_peri')
    t0 = pick_num(summ, 't_value', 'T', 't0', 'T0', 'T_peri', 'Tperi', 'T_periastron')
    chisqr = pick_num(summ, 'chisqr', 'chi2', 'chisq')
    redchi = pick_num(summ, 'redchi', 'chi2_red', 'reduced_chi2')
    aic = pick_num(summ, 'aic', 'AIC')
    bic = pick_num(summ, 'bic', 'BIC')
    nvarys = summ.get('nvarys', np.nan)
    ndata = summ.get('ndata', np.nan)
    ls_fap = pick_num(summ, 'ls_fap')
    pdc_fap = pick_num(summ, 'pdc_fap')
    F_stat = pick_num(summ, 'F_stat')
    F_crit_99 = pick_num(summ, 'F_crit_99')
    F_pvalue = pick_num(summ, 'F_pvalue')
    bin_flag = int(summ.get('bin_flag', 0))
    ftest_pass = (np.isfinite(F_stat) and np.isfinite(F_crit_99) and (F_stat > F_crit_99))

    # --- Step 2: Calculate Pmin grids (3x3) over M1 and R* uncertainties ---
    P_peri_c = P_peri_lo = P_peri_hi = np.nan
    P_apa_c = P_apa_lo = P_apa_hi = np.nan
    P_peri_best_c = P_peri_best_lo = P_peri_best_hi = np.nan

    if all(np.isfinite(x) for x in [mspec, K1, e, rstar]):
        M1_array = np.clip(np.array([
            mspec + ms_minus, mspec, mspec + ms_plus
        ]), 0.1, None)
        R_array = np.clip(np.array([
            rstar + r_minus, rstar, rstar + r_plus
        ]), 0.1, None)

        P_peri_grid = np.full((3, 3), np.nan, float)
        P_apa_grid = np.full((3, 3), np.nan, float)
        P_peri_best_grid = np.full((3, 3), np.nan, float)

        # CORRECTED: Define f_inf before the loop
        f_inf = 0.49 / 0.6

        for i, M1i in enumerate(M1_array):
            for j, Ri in enumerate(R_array):
                # This call is correct and uses the right function
                res = Pmin_peri_apa(M1i, Ri, K1, e, alpha_peri, alpha_apa)
                P_peri_grid[i, j] = float(res.get('Pmin_peri', np.nan))
                P_apa_grid[i, j] = float(res.get('Pmin_apa', np.nan))

                # This calculation now works because f_inf is defined
                R_AU_j = Ri * RSUN2AU
                P_peri_best_grid[i, j] = C2 * R_AU_j ** 1.5 / (
                            (alpha_peri * f_inf) ** 1.5 * (1 - e) ** 1.5 * np.sqrt(M1i))

        # Extract central, lower, and upper bounds for standard Pmin
        P_peri_c = float(P_peri_grid[1, 1])
        P_peri_lo = float(np.nanmin(P_peri_grid))
        P_peri_hi = float(np.nanmax(P_peri_grid))
        P_apa_c = float(P_apa_grid[1, 1])
        P_apa_lo = float(np.nanmin(P_apa_grid))
        P_apa_hi = float(np.nanmax(P_apa_grid))

        # Extract central, lower, and upper bounds for best-case Pmin
        P_peri_best_c = float(P_peri_best_grid[1, 1])
        P_peri_best_lo = float(np.nanmin(P_peri_best_grid))
        P_peri_best_hi = float(np.nanmax(P_peri_best_grid))

    # --- Step 3: Calculate M2_min ---
    M2_min = M2_min_lower = M2_min_upper = np.nan
    if np.isfinite(mspec) and np.isfinite(P) and np.isfinite(K1) and np.isfinite(e):
        r = companion_mass_min(P, K1, e, mspec, ms_plus, ms_minus)
        M2_min = float(r.get('M2_min', np.nan))
        M2_min_lower = float(r.get('M2_min_lower', np.nan))
        M2_min_upper = float(r.get('M2_min_upper', np.nan))

    # --- Step 4: Final binary classification ---
    use_lower = P_peri_lo if np.isfinite(P_peri_lo) else P_peri_c
    is_binary = int((bin_flag > 0) and ftest_pass and np.isfinite(P) and np.isfinite(use_lower) and (P > use_lower))
    is_black_hole = int(M2_min_lower > 3)

    # --- Step 5: Assemble and return the final dictionary ---
    return {
        "Star": star_name, "P": P, "K1": K1, "e": e, "gamma": gam, "omega": omega, "T0": t0,
        "chisqr": chisqr, "redchi": redchi, "aic": aic, "bic": bic, "nvarys": nvarys, "ndata": ndata,
        "ls_fap": ls_fap, "pdc_fap": pdc_fap, "F_stat": F_stat, "F_crit_99": F_crit_99, "F_pvalue": F_pvalue,
        "bin_flag": bin_flag,
        "Pmin_peri_central": P_peri_c,
        "Pmin_peri_lower": P_peri_lo,
        "Pmin_peri_upper": P_peri_hi,
        "Pmin_apa_central": P_apa_c,
        "Pmin_apa_lower": P_apa_lo,
        "Pmin_apa_upper": P_apa_hi,
        "Pmin_peri": P_peri_c,
        "Pmin_peri_best_central": P_peri_best_c,
        "Pmin_peri_best_lower": P_peri_best_lo,
        "Pmin_peri_best_upper": P_peri_best_hi,
        "M2_min": M2_min, "M2_min_lower": M2_min_lower, "M2_min_upper": M2_min_upper,
        "bh cand": is_black_hole, "M1": mspec, "R_star": rstar, "is_binary": is_binary,
    }

def append_summary_csv(row: Dict[str, object], csv_path: Path):
    """Append/update a single row in the summary CSV (dedupe by Star)."""
    df_row = pd.DataFrame([row])
    if csv_path.exists():
        try:
            old = pd.read_csv(csv_path)
            new = pd.concat([old, df_row], ignore_index=True)
            if "Star" in new.columns:
                new = new.drop_duplicates(subset=["Star"], keep="last")
        except Exception:
            new = df_row
    else:
        new = df_row
    new.to_csv(csv_path, index=False)


def save_errors_to_summary(uncertainties: dict, star_name: str, csv_path: Path):
    """Reads the summary CSV, adds/updates error columns for a star, and saves it."""
    if not csv_path.is_file():
        st.error(f"{csv_path.name} not found. Mark a fit as BEST first to create the file.")
        return

    # Map normalized internal names to the final column base names in the CSV
    param_map = {
        'period': 'P', 'k1': 'K1', 'eccentricity': 'e', 'e': 'e',
        'gamma': 'gamma', 'omega': 'omega',
         't_periastron': 'T0', 't_peri_mjd': 'T0', 't0_mjd': 'T0', 't_mjd': 'T0', 't_peri_epoch': 'T0'

    }

    errors_to_save = {}
    for internal_name, details in uncertainties.items():
        # Normalize: lowercase, remove _rad, map lmfit names to simpler ones
        norm_name = internal_name.lower().replace('_rad', '')
        if norm_name in param_map:
            csv_name = param_map[norm_name]
            errors_to_save[f"{csv_name}_err_plus"] = details.get('err_plus')
            errors_to_save[f"{csv_name}_err_minus"] = details.get('err_minus')

    if not errors_to_save:
        st.warning("No mappable errors found in the MCMC results.")
        return

    try:
        df = pd.read_csv(csv_path)
        if "Star" not in df.columns:
            st.error("Summary CSV is missing the required 'Star' column.")
            return

        star_mask = df['Star'] == star_name
        if not star_mask.any():
            st.warning(f"Star '{star_name}' not found in summary. Adding a new row with these errors.")
            new_row = {"Star": star_name, **errors_to_save}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Add or update each error column for the matching star
            for col, val in errors_to_save.items():
                df.loc[star_mask, col] = val

        df.to_csv(csv_path, index=False)
        st.success(f"Saved MCMC errors for {star_name} to {csv_path.name}")

    except Exception as e:
        st.error(f"Failed to update summary CSV: {e}")

def _install_streamlit_tqdm():
    # Safe no-op if tqdm or streamlit are unavailable.
    try:
        import tqdm, streamlit as st  # noqa: F401
    except Exception:
        return
    import tqdm as _tqdm
    if getattr(_tqdm, "_streamlit_patched", False):
        return
    _orig = _tqdm.tqdm
    class _STqdm(_orig):
        def __init__(self, *a, **k):
            self._slot = st.empty()
            super().__init__(*a, **k)
        def display(self, *a, **k):
            if self.total:
                pct = int(100 * self.n / self.total)
                self._slot.progress(pct, text=f"{self.desc or 'MCMC'}: {self.n}/{self.total}")
    _tqdm.tqdm = _STqdm
    _tqdm._streamlit_patched = True

def _read_note_from_csv(csv_path: Path, star: str) -> str:
    try:
        if csv_path.is_file():
            df = pd.read_csv(csv_path)
            if "Star" in df.columns:
                col = "note" if "note" in df.columns else ("Note" if "Note" in df.columns else None)
                if col:
                    vals = df.loc[df["Star"] == star, col].astype(str)
                    if not vals.empty:
                        return vals.iloc[0]
    except Exception:
        pass
    return ""

def update_note_in_csv(star: str, csv_path: Path, note: str, column: str = "note"):
    note = (note or "").strip()
    df_row = pd.DataFrame([{"Star": star, column: note}])
    if csv_path.exists():
        try:
            old = pd.read_csv(csv_path)
            if "Star" in old.columns:
                if column not in old.columns:
                    old[column] = ""
                mask = (old["Star"] == star)
                if mask.any():
                    old.loc[mask, column] = note
                    new = old
                else:
                    new = pd.concat([old, df_row], ignore_index=True)
            else:
                new = pd.concat([old, df_row], ignore_index=True)
        except Exception:
            new = df_row
    else:
        new = df_row
    new.to_csv(csv_path, index=False)

def append_or_update(out_csv: Path, row_dict: dict, key: str = "Star"):
    df_row = pd.DataFrame([row_dict])
    if out_csv.exists():
        try:
            old = pd.read_csv(out_csv)
            if key in old.columns and key in df_row.columns:
                mask = old[key].astype(str) == str(df_row.iloc[0][key])
                if mask.any():
                    for c in df_row.columns:
                        old.loc[mask, c] = df_row.iloc[0][c]
                    new = old
                else:
                    new = pd.concat([old, df_row], ignore_index=True)
            else:
                new = pd.concat([old, df_row], ignore_index=True)
        except Exception:
            new = df_row
    else:
        new = df_row
    new.to_csv(out_csv, index=False)

def extract_mcmc_uncertainties(mcmc_result, data=None, period_guess=None):
    if not hasattr(mcmc_result, 'flatchain') or mcmc_result.flatchain is None:
        return {}

    chain = mcmc_result.flatchain
    var_names = list(getattr(mcmc_result, "var_names", getattr(chain, "columns", [])))
    uncertainties = {}

    for i, param_name in enumerate(var_names):
        if param_name in ('ln_sigma', 'ln_sigma_extra'):
            continue

        # Get the MCMC chain for the current parameter from the results object
        if hasattr(chain, "columns") and param_name in chain.columns:
            param_chain = chain[param_name].to_numpy()
        elif hasattr(chain, "iloc"):
            param_chain = chain.iloc[:, i].to_numpy()
        else:
            arr = np.asarray(chain)
            param_chain = arr[:, i]

        if not np.isfinite(param_chain).any():
            continue

        # --- Special handling for the 'omega' parameter ---
        if param_name.lower().startswith("omega"):
            param_chain = np.asarray(param_chain)

            # 1. Calculate the circular mean of the angle chain. This is robust
            #    against periodicity and gives a result in [-pi, pi].
            circ_mean = np.arctan2(np.mean(np.sin(param_chain)), np.mean(np.cos(param_chain)))

            # 2. "Unwrap" the distribution by shifting it so the mean is at 0.
            #    This is crucial for calculating correct percentiles without boundary effects.
            unwrapped_chain = ((param_chain - circ_mean + np.pi) % (2 * np.pi)) - np.pi

            # 3. Now, safely calculate percentiles on the corrected, unwrapped chain.
            p16, med_shifted, p84 = np.nanpercentile(unwrapped_chain, [16, 50, 84])

            # 4. The errors are the distances from the median of the shifted distribution.
            lower_error = med_shifted - p16
            upper_error = p84 - med_shifted

            # 5. The final "best" value is the circular mean, mapped to the [0, 2*pi) range.
            final_median_omega = circ_mean % (2 * np.pi)

            # 6. Set the bounds for the refined fit around this positive median.
            lo = final_median_omega - 3* lower_error
            hi = final_median_omega + 3 * upper_error

            lo = max(0.0, lo)
            hi = min(2 * np.pi, hi)
            # --- END OF FIX ---

            uncertainties[param_name] = {
                'value': final_median_omega, 'min': lo, 'max': hi,
                'err_plus': upper_error, 'err_minus': lower_error
            }

            # Skip the generic logic below and move to the next parameter.
            continue

        # --- Generic logic for all other parameters (Period, K1, e, etc.) ---

        p16, med, p84 = np.nanpercentile(param_chain, [16, 50, 84])
        lower_error = med - p16
        upper_error = p84 - med

        # Define search bounds for the refined fit based on a wider ~1.5-sigma range
        lo = med - 3 * lower_error
        hi = med +3 * upper_error

        # Apply the e-cap to bounds IF the parameter is eccentricity and period is short
        if param_name in ("Eccentricity", "e") and period_guess is not None and np.isfinite(period_guess):
            try:
                if period_guess < P_SHORT_MAX:
                    hi = min(hi, float(E_MAX_SHORT))
                    med = min(med, float(E_MAX_SHORT))
            except NameError:
                pass

        if param_name.lower() in ("eccentricity", "e"):
            lo = max(0.0, lo)
            hi = min(0.99, hi) # A hard cap just below 1 is good practice

        # For Period (P) and K1, bounds must be positive.
        if param_name.lower() in ("period", "p", "k1"):
            lo = max(1e-6, lo) # Ensure it's a small positive number, not zero
        # --- END OF FIX ---

        uncertainties[param_name] = {
            'value': med, 'min': lo, 'max': hi,
            'err_plus': upper_error, 'err_minus': lower_error
        }


    return uncertainties

def run_refined_orbit_fit(base_candidate, mcmc_uncertainties, data, star_name, output_dir):
    """
    Run refined orbital fit using MCMC-derived parameter bounds and ensure plots are generated.
    """
    try:
        # Create refined parameter bounds from MCMC uncertainties
        refined_args = copy.deepcopy(base_candidate.args)

        for param_name, bounds in mcmc_uncertainties.items():
            # Find the corresponding parameter name in the lmfit arguments dict
            # This handles cases like 'omega' vs 'OMEGA_rad'
            key_to_update = None
            for key in refined_args[LMFIT_PARAMS][SEARCH_REGION]:
                if key.lower().startswith(param_name.lower().replace('_rad', '')):
                    key_to_update = key
                    break

            if key_to_update:
                param_dict = refined_args[LMFIT_PARAMS][SEARCH_REGION][key_to_update]
                param_dict[INIT_VAL] = bounds['value']
                param_dict[MIN_VAL] = bounds['min']
                param_dict[MAX_VAL] = bounds['max']
                param_dict[VARY] = True

        # Run the refined fit and generate plots
        with st.spinner("Running refined orbital fit and generating plots..."):
            refined_result = lmfit_on_sample(refined_args, str(output_dir), data, star_name)
            if refined_result and refined_result.success:
                print_lmfit_result(
                    data, refined_args, star_name, refined_result, out_dir=str(output_dir)
                )

        # Create summary from the NEW result
        refined_summary = summarize_result(refined_result, star_name) or {}

        # --- Correctly back-fill ALL parameters from the refined result ---
        if hasattr(refined_result, "params"):
            p = refined_result.params
            # Use pick_num to be robust against naming differences
            refined_summary['period_value'] = pick_num({k: v.value for k, v in p.items()}, PERIOD)
            refined_summary['k1_value'] = pick_num({k: v.value for k, v in p.items()}, K1_STR)
            refined_summary['ecc_value'] = pick_num({k: v.value for k, v in p.items()}, ECC)
            refined_summary['gamma_value'] = pick_num({k: v.value for k, v in p.items()}, GAMMA)
            refined_summary['omega_value'] = pick_num({k: v.value for k, v in p.items()}, OMEGA)

            # Correctly calculate and store t_value offset from the refined PHASE0
            phi0_fit = pick_num({k: v.value for k, v in p.items()}, PHASE0)
            p_fit = refined_summary['period_value']

            if np.isfinite(phi0_fit) and np.isfinite(p_fit):
                t_ref = float(np.median(data[TIME_STAMPS].to_numpy(float)))
                t0_abs = t_ref + phi0_fit * p_fit
                time_diff = t0_abs - t_ref
                t0_offset = (time_diff / p_fit - np.round(time_diff / p_fit)) * p_fit
                refined_summary['t_value'] = t0_offset

        # Add comparison metrics
        original_chi2 = base_candidate.summary.get('chisqr', np.nan)
        refined_chi2 = getattr(refined_result, 'chisqr', np.nan)
        refined_summary.update({
            'original_chisqr': original_chi2,
            'refined_chisqr': refined_chi2,
            'chi2_improvement': original_chi2 - refined_chi2
        })

        return FitCandidate(
            period=refined_summary.get('period_value', base_candidate.period),
            summary=refined_summary,
            args=refined_args,
            bounds=(0, 0),
            result=refined_result
        )

    except Exception as e:
        st.error(f"Refined fit failed: {e}")
        return None
# -------------------------------
# UI Body
# -------------------------------
st.title("SB1 Interactive GUI — CCF → Orbit Fit → Masses")

if root_folder and output_root:
    stars = discover_stars(root_folder)
    if not stars:
        st.info("No star subfolders with spectra found yet.")
    else:
        left, right = st.columns([1, 2])
        with left:
            star_names = [s.name for s in stars]


            # --- Define callback functions to safely modify the state ---
            def go_to_next_star():
                """Callback to select the next star in the list."""
                try:
                    current_index = star_names.index(st.session_state.sel_star_name)
                    next_index = min(len(star_names) - 1, current_index + 1)
                    st.session_state.sel_star_name = star_names[next_index]
                except (ValueError, IndexError):
                    if star_names:
                        st.session_state.sel_star_name = star_names[0]


            def go_to_previous_star():
                """Callback to select the previous star in the list."""
                try:
                    current_index = star_names.index(st.session_state.sel_star_name)
                    prev_index = max(0, current_index - 1)
                    st.session_state.sel_star_name = star_names[prev_index]
                except (ValueError, IndexError):
                    if star_names:
                        st.session_state.sel_star_name = star_names[0]


            # Set the initial star from saved settings if the state isn't set yet
            if "sel_star_name" not in st.session_state:
                initial_star = SETTINGS.get("last_star")
                if initial_star and initial_star in star_names:
                    st.session_state.sel_star_name = initial_star
                elif star_names:
                    st.session_state.sel_star_name = star_names[0]

            # The selectbox for random access. It also modifies the same session state key.
            st.selectbox(
                "Choose a star folder",
                options=star_names,
                key="sel_star_name"
            )

            # --- Buttons now use the on_click callbacks ---
            nav_cols = st.columns(2)
            with nav_cols[0]:
                st.button("◄ Previous Star", on_click=go_to_previous_star)

            with nav_cols[1]:
                st.button("Next Star ►", on_click=go_to_next_star)

            # The rest of your logic remains the same. After a button click,
            # the callback runs, the state is updated, and then the script
            # automatically reruns from the top, reflecting the change.
            sel_star_name = st.session_state.sel_star_name
            sel_star = stars[star_names.index(sel_star_name)]
            st.caption(f"Selected path: {sel_star}")

            # Persist the choice for the next time the app is opened
            st.session_state["last_star"] = sel_star_name
            _save_settings_from_state()

            go_ccf = st.button("Run CCF → RVs for this star")
        with right:
            if not HAVE_ASTROPY:
                st.warning("astropy not available — FITS reading disabled. ASCII still works.\n" + ASTROPY_ERR)

            if not HAVE_ORBIT:
                st.info("Orbit‑fit stack not importable; you can still do CCF + masses.\n\nImport error: " + ORBIT_IMPORT_ERR)
        if go_ccf:
            obs_list = load_observations(sel_star, s2n_cut)
            if not obs_list:
                st.error("No usable observations after SNR/contamination filtering.")
            else:
                with st.spinner("Computing CCFs and RVs…"):
                    model_tpl = load_model_template(sel_star, model_root)
                    rvres = compute_ccf_for_star(
                        obs_list, sel_star.name, ALL_LINES, line_tol, win_He, win_H,
                        ccf_lambda_min, ccf_lambda_max, fit_range,
                        two_pass=two_pass, n_sig_out=float(n_sig_out),
                        model_template=model_tpl,
                        final_error_type=st.session_state.final_error_type
                    )

                if rvres.rv_df.empty:
                    st.error("No lines matched within tolerance; tweak settings.")
                else:
                    # Save outputs
                    out_dir = Path(output_root)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    tag = sel_star.name
                    rv_csv = out_dir / f"{tag}_SB1_line_RVs.csv"
                    mean_csv = out_dir / f"{tag}_CCF_RVs.csv"
                    rvres.rv_df.to_csv(rv_csv, index=False)
                    rvres.mean_df.to_csv(mean_csv, index=False)
                    st.success(f"Saved: {rv_csv.name} and {mean_csv.name}")

                    # Download buttons
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.download_button("Download per‑line RV CSV", rvres.rv_df.to_csv(index=False).encode('utf-8'),
                                           file_name=rv_csv.name)
                    with c2:
                        st.download_button("Download weighted‑mean RV CSV",
                                           rvres.mean_df.to_csv(index=False).encode('utf-8'), file_name=mean_csv.name)
                    with c3:
                        zip_data = create_zip_from_plots(rvres.plots)
                        st.download_button(
                            label="Download All RV Plots (.zip)",
                            data=zip_data,
                            file_name=f"{sel_star.name}_rv_plots.zip",
                            mime="application/zip",
                        )

                    st.session_state['rv_mean_df'] = rvres.mean_df.copy()
                    st.session_state['last_rv_df'] = rvres.rv_df.copy()
                    st.session_state['last_star'] = sel_star.name
                    st.session_state['out_root'] = output_root
                    st.session_state['coadd_wave'] = rvres.coadd_wave
                    st.session_state['coadd_flux'] = rvres.coadd_flux
                    st.session_state['coadd0_wave'] = rvres.coadd0_wave
                    st.session_state['coadd0_flux'] = rvres.coadd0_flux

                    # Show plots: pass 1 and pass 2, before/after
                    st.caption(f"Outlier rule: keep line if |RV - w_mean| ≤ n_sig·σ  (n_sig = {float(n_sig_out):.2f})")

                    st.subheader("Pass 1 (template → coadd)")
                    c11, c12 = st.columns(2)
                    with c11:
                        st.image(rvres.plots["p1_weighted_before.png"],
                                 caption="Pass 1 — Weighted means (before clean)", use_container_width=True)
                    with c12:
                        st.image(rvres.plots["p1_weighted_after.png"], caption="Pass 1 — Weighted means (after clean)",
                                 use_container_width=True)

                    c13, c14 = st.columns(2)
                    with c13:
                        st.image(rvres.plots["p1_lines_before.png"],
                                 caption="Pass 1 — Line RVs (before clean, outliers marked 'x')",
                                 use_container_width=True)
                    with c14:
                        st.image(rvres.plots["p1_lines_after.png"], caption="Pass 1 — Line RVs (after clean)",
                                 use_container_width=True)

                    if two_pass:
                        st.subheader("Pass 2 (shifted coadd → remeasure)")
                        c21, c22 = st.columns(2)
                        with c21:
                            st.image(rvres.plots["p2_weighted_before.png"],
                                     caption="Pass 2 — Weighted means (before clean)", use_container_width=True)
                        with c22:
                            st.image(rvres.plots["p2_weighted_after.png"],
                                     caption="Pass 2 — Weighted means (after clean)", use_container_width=True)

                        c23, c24 = st.columns(2)
                        with c23:
                            st.image(rvres.plots["p2_lines_before.png"],
                                     caption="Pass 2 — Line RVs (before clean, outliers marked 'x')",
                                     use_container_width=True)
                        with c24:
                            st.image(rvres.plots["p2_lines_after.png"], caption="Pass 2 — Line RVs (after clean)",
                                     use_container_width=True)
        # --- Spectra viewer ----------------------------------------------------
        st.markdown("---")
        st.markdown("### Spectra viewer")

        # Load observations for the selected star (cached)
        try:
            obs_for_view = load_observations(sel_star, s2n_cut)
        except Exception as _e:
            obs_for_view = []
            st.error(f"Could not load spectra: {_e}")

        if not obs_for_view:
            st.info("No spectra found for this star.")
        else:
            # Wavelength limits from data (clipped by the global λ settings)
            lam_min_all = float(np.nanmin([np.nanmin(o.spectrum[:, 0]) for o in obs_for_view]))
            lam_max_all = float(np.nanmax([np.nanmax(o.spectrum[:, 0]) for o in obs_for_view]))
            lam_lo = max(ccf_lambda_min, lam_min_all)
            lam_hi = min(ccf_lambda_max, lam_max_all)

            c1, c2, c3 = st.columns([2, 2, 3])
            with c1:
                lam_min = st.number_input("λ min (Å)", 3000.0, 10000.0, value=float(lam_lo), step=5.0, key="spec_lmin")
            with c2:
                lam_max = st.number_input("λ max (Å)", 3000.0, 10000.0, value=float(lam_hi), step=5.0, key="spec_lmax")
            with c3:
                mode = st.radio(
                    "Epochs to plot",
                    ["All epochs", "Specific epochs", "Min & Max RV"],
                    index=0, horizontal=True, key="spec_mode"
                )

            # Build MJD label mapping for UI
            labels = [f"{o.mjd:.2f} — {o.name}" for o in obs_for_view]
            lab2mjd = {lab: o.mjd for lab, o in zip(labels, obs_for_view)}
            all_mjds = [o.mjd for o in obs_for_view]

            # Determine which epochs to plot
            mjds_to_plot: List[float] = []
            if mode == "All epochs":
                mjds_to_plot = all_mjds

            elif mode == "Specific epochs":
                picks = st.multiselect("Pick epochs (MJD — filename)", labels, default=labels[:min(3, len(labels))],
                                       key="spec_pick")
                mjds_to_plot = [lab2mjd[p] for p in picks]

            else:  # "Min & Max RV"
                rv_mean_df = st.session_state.get('rv_mean_df', None)
                if rv_mean_df is None or rv_mean_df.empty:
                    st.warning("No weighted-mean RV table available yet — run the CCF step first.")
                    mjds_to_plot = all_mjds  # fallback: show all
                else:
                    r = rv_mean_df["Mean RV"].to_numpy(float)
                    m = rv_mean_df["MJD"].to_numpy(float)
                    i_min = int(np.nanargmin(r))
                    i_max = int(np.nanargmax(r))
                    mjds_to_plot = [float(m[i_min]), float(m[i_max])]
                    st.caption(f"Min/Max RV epochs: {m[i_min]:.2f}, {m[i_max]:.2f}")

            # Display controls
            colA, colB, colC, colD = st.columns([2, 2, 2, 2])
            with colA:
                normalize = st.checkbox("Normalize to median", True, key="spec_norm")
            with colB:
                offset = st.number_input("Vertical offset per spectrum", 0.0, 10.0, 0.0, step=0.1, key="spec_offset")
            with colC:
                smooth_win = st.slider("Savgol window (odd, 0=off)", 0, 101, 0, step=2, key="spec_smooth")
            with colD:
                mark_lines = st.checkbox("Mark He/H lines", True, key="spec_mark_lines")

            # Plot (interactive option)
            row2 = st.columns([2, 2, 2, 2])
            with row2[0]:
                use_plotly = st.checkbox("Interactive zoom (Plotly)", True, key="spec_use_plotly")
            with row2[1]:
                show_hover = st.checkbox("Show hover readout", False, key="spec_show_hover")
            with row2[2]:
                overlay_coadd = st.checkbox("Overlay coadded template", True, key="spec_overlay_coadd")
            with row2[3]:
                cw = st.session_state.get('coadd_wave', None)
                cf = st.session_state.get('coadd_flux', None)
                if isinstance(cw, np.ndarray) and isinstance(cf, np.ndarray) and cw.size and cf.size:
                    coadd_df = pd.DataFrame({"wavelength": cw, "flux": cf})
                    st.download_button(
                        "Download coadd CSV",
                        coadd_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{sel_star.name}_coadd_template.csv"
                    )

            # Build processed series once
            series = []
            for idx, o in enumerate(obs_for_view):
                if o.mjd not in mjds_to_plot:
                    continue
                w = o.spectrum[:, 0]
                f = np.nan_to_num(o.spectrum[:, 1])
                sel = (w >= lam_min) & (w <= lam_max)
                if np.count_nonzero(sel) < 3:
                    continue
                w, f = w[sel], f[sel]
                if normalize:
                    med = np.nanmedian(f)
                    if np.isfinite(med) and med != 0:
                        f = f / med
                if smooth_win and smooth_win >= 3 and (smooth_win % 2 == 1):
                    try:
                        f = savgol_filter(f, smooth_win, polyorder=2, mode="interp")
                    except Exception:
                        pass
                y = f + (len(series) * offset)
                series.append({"w": w, "y": y, "label": f"MJD {o.mjd:.2f}"})

            if not series:
                st.info("No epochs matched the current selection and wavelength range.")
            else:
                if use_plotly and HAVE_PLOTLY:
                    fig = go.Figure()

                    # add one trace per selected epoch
                    for s in series:
                        fig.add_trace(go.Scatter(
                            x=s["w"].tolist(),
                            y=s["y"].tolist(),
                            mode="lines",
                            name=s["label"],
                            line=dict(width=1.6)
                        ))

                    fig.update_traces(connectgaps=True)
                    fig.update_yaxes(autorange=True, rangemode="tozero")

                    # overlay coadd if requested
                    cw = st.session_state.get('coadd_wave', None)
                    cf = st.session_state.get('coadd_flux', None)
                    if overlay_coadd and isinstance(cw, np.ndarray) and isinstance(cf,
                                                                                   np.ndarray) and cw.size and cf.size:
                        selc = (cw >= lam_min) & (cw <= lam_max)
                        yf = np.nan_to_num(cf[selc])
                        if normalize:
                            med = np.nanmedian(yf)
                            if np.isfinite(med) and med != 0:
                                yf = yf / med
                        fig.add_trace(go.Scatter(
                            x=cw[selc].tolist(),
                            y=yf.tolist(),
                            mode="lines",
                            name="Coadd (template)",
                            line=dict(width=3, dash="dot")
                        ))

                    if mark_lines:
                        for L in (HE_LINES + (H_LINES if use_balmer else [])):
                            if lam_min <= L <= lam_max:
                                fig.add_vline(x=float(L), line_dash="dash", opacity=0.25)

                    fig.update_layout(
                        xaxis_title="Wavelength (Å)",
                        yaxis_title="Flux" + (" + offset" if offset else ""),
                        hovermode=("x unified" if show_hover else False),
                        dragmode="zoom",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        margin=dict(l=40, r=10, t=10, b=40),
                    )
                    fig.update_xaxes(rangeslider_visible=True, fixedrange=False)
                    fig.update_yaxes(fixedrange=False)
                    if not show_hover:
                        fig.update_traces(hoverinfo="skip", hovertemplate=None)

                    st.plotly_chart(
                        fig, use_container_width=True,
                        config={"displaylogo": False, "displayModeBar": True, "scrollZoom": True}
                    )


                else:
                    # Static fallback (matplotlib)
                    fig, ax = plt.subplots(figsize=(8.0, 4.0))
                    for s in series:
                        ax.plot(s["w"], s["y"], lw=0.9, label=s["label"])
                    cw = st.session_state.get('coadd_wave', None)
                    cf = st.session_state.get('coadd_flux', None)
                    if overlay_coadd and isinstance(cw, np.ndarray) and isinstance(cf,
                                                                                   np.ndarray) and cw.size and cf.size:
                        selc = (cw >= lam_min) & (cw <= lam_max)
                        yf = np.nan_to_num(cf[selc])
                        if normalize:
                            med = np.nanmedian(yf)
                            if np.isfinite(med) and med != 0:
                                yf = yf / med
                        ax.plot(cw[selc], yf, lw=2.5, ls="--", label="Coadd (template)")
                    ax.set_xlabel("Wavelength (Å)")
                    ax.set_ylabel("Flux" + (" + offset" if offset else ""))
                    if mark_lines:
                        for L in (HE_LINES + (H_LINES if use_balmer else [])):
                            if lam_min <= L <= lam_max:
                                ax.axvline(L, ls="--", alpha=0.25)
                    ax.legend(ncol=3, fontsize=8)
                    st.pyplot(fig, clear_figure=True)

        st.markdown("---")
        st.markdown("### Spectral Animation Generator")

        if not HAVE_ANIMATION:
            st.warning("Animation features require matplotlib.animation. Try: pip install --upgrade matplotlib")
        else:
            # Animation controls
            anim_cols = st.columns([2, 2, 2])

            with anim_cols[0]:
                anim_speed = st.slider("Animation speed", 0.1, 5.0, 1.0, 0.1,
                                       help="Higher values = faster transitions")

            with anim_cols[1]:
                custom_regions = st.checkbox("Custom wavelength regions", False)

            with anim_cols[2]:
                include_phase_anim = st.checkbox("Include phase-folded RV animation", False)

            # Wavelength region selection
            if custom_regions:
                st.markdown("#### Define wavelength regions (Å)")
                region_text = st.text_area(
                    "Enter wavelength ranges (one per line: min,max)",
                    value="4020,4035\n4095,4110\n4330,4350\n4380,4395\n4465,4480",
                    height=100
                )

                try:
                    regions = []
                    for line in region_text.strip().split('\n'):
                        if ',' in line:
                            wl_min, wl_max = map(float, line.split(','))
                            regions.append((wl_min, wl_max))
                    wavelength_regions = regions if regions else None
                except:
                    st.error("Invalid wavelength region format. Use: min,max")
                    wavelength_regions = None
            else:
                # Default regions around key spectral lines
                default_lines = [4026, 4101, 4340, 4388, 4471]  # He and H lines
                radius = 15
                wavelength_regions = [(line - radius, line + radius) for line in default_lines]

                st.caption(f"Using default regions around: {', '.join(f'{l}±{radius}' for l in default_lines)} Å")

            # Animation generation buttons
            anim_button_cols = st.columns([2, 2])

            with anim_button_cols[0]:
                if st.button("Create Spectra Animation GIF", key="btn_create_spectra_gif"):
                    if not obs_for_view:
                        st.error("No spectral data loaded for this star")
                    else:
                        output_dir = Path(output_root) / sel_star.name
                        output_dir.mkdir(parents=True, exist_ok=True)

                        gif_path = animate_spectra_data(
                            obs_for_view,
                            sel_star.name,
                            output_dir,
                            fps=6.0 * anim_speed,  # use your slider to scale FPS
                            fmt="gif"
                        )

                        if gif_path and gif_path.exists():
                            st.success(f"Animation saved: {gif_path.name}")

                            # Display the GIF
                            with open(gif_path, 'rb') as f:
                                gif_data = f.read()
                            st.image(gif_data, caption=f"Spectral Evolution - {sel_star.name}")

                            # Download button
                            st.download_button(
                                "Download Spectra Animation",
                                gif_data,
                                file_name=gif_path.name,
                                mime="image/gif"
                            )

            with anim_button_cols[1]:
                rv_mean_df = st.session_state.get('rv_mean_df', None)
                fit_row = st.session_state.get('last_fit_row', {})

                can_make_phase_anim = (rv_mean_df is not None and not rv_mean_df.empty and
                                       fit_row and include_phase_anim)

                if st.button("Create Phase-Folded RV Animation",
                             disabled=not can_make_phase_anim, key="btn_create_phase_gif"):

                    if can_make_phase_anim:
                        output_dir = Path(output_root) / sel_star.name
                        output_dir.mkdir(parents=True, exist_ok=True)
                        # Prefer the actual lmfit result from the chosen candidate
                        res = st.session_state.get('last_fit_result', None)
                        orbital_params = None
                        if res is not None and hasattr(res, "params"):
                            p = res.params
                            try:
                                orbital_params = {
                                    'P': float(p[PERIOD].value),
                                    'K1': float(p[K1_STR].value),
                                    'e': float(p[ECC].value),
                                    'omega': float(p[OMEGA].value),  # usually radians
                                    'T': float(p[T].value),
                                    'gamma': float(p[GAMMA].value),
                                }
                            except Exception:
                                orbital_params = None

                        if orbital_params is None:
                            # Fallback to the summary dict (keeps your existing behavior below)
                            def _pick(row, *names):
                                for n in names:
                                    if n in row:
                                        try:
                                            v = float(row[n])
                                            if np.isfinite(v): return v
                                        except Exception:
                                            pass
                                return np.nan


                            P_val = _pick(fit_row, 'period_value', 'Period', 'period', 'P')
                            K1_val = _pick(fit_row, 'k1_value', 'K1', 'k1')
                            e_val = _pick(fit_row, 'ecc_value', 'Eccentricity', 'e')
                            T_val = _pick(fit_row, 't_value', 'T', 't0', 'T0', 'T_peri', 'Tperi', 'T_periastron')
                            omega_v = _pick(fit_row, 'omega_value', 'omega', 'Omega', 'w', 'argperi', 'arg_peri',
                                            'omega_deg')
                            if np.isfinite(omega_v) and abs(omega_v) > 2 * np.pi + 1e-6:
                                omega_v = np.deg2rad(omega_v)
                            gamma_v = _pick(fit_row, 'gamma_value', 'gamma', 'Gamma', 'V0')

                            orbital_params = {'P': P_val, 'K1': K1_val, 'e': e_val,
                                              'omega': omega_v, 'T': T_val, 'gamma': gamma_v}


                        def _pick(row, *names):
                            for n in names:
                                if n in row:
                                    try:
                                        v = float(row[n])
                                        if np.isfinite(v):
                                            return v
                                    except Exception:
                                        pass
                            return np.nan


                        # robust pulls
                        P_val = _pick(fit_row, 'period_value', 'Period', 'period', 'P')
                        K1_val = _pick(fit_row, 'k1_value', 'K1', 'k1')
                        e_val = _pick(fit_row, 'ecc_value', 'Eccentricity', 'e')
                        T_val = _pick(fit_row, 't_value', 'T', 't0', 'T0', 'T_peri', 'Tperi', 'T_periastron')
                        omega_v = _pick(fit_row, 'omega_value', 'omega', 'Omega', 'w', 'argperi', 'arg_peri',
                                        'omega_deg')
                        # convert to radians if it looks like degrees
                        if np.isfinite(omega_v) and abs(omega_v) > 2 * np.pi + 1e-6:
                            omega_v = np.deg2rad(omega_v)
                        gamma_v = _pick(fit_row, 'gamma_value', 'gamma', 'Gamma', 'V0')

                        orbital_params = {'P': P_val, 'K1': K1_val, 'e': e_val,
                                          'omega': omega_v, 'T': T_val, 'gamma': gamma_v}

                        # Use stored RV data
                        rv_df = st.session_state.get('rv_mean_df', pd.DataFrame())  # <-- use mean values
                        if rv_df.empty and 'rv_mean_df' in st.session_state:
                            # Fallback: create simple RV data from mean values
                            rv_data = []
                            for _, row in st.session_state['rv_mean_df'].iterrows():
                                rv_data.append({
                                    f'MJD_{row["MJD"]:.2f}_RV': [row['Mean RV']],
                                    f'MJD_{row["MJD"]:.2f}_err': [row['Mean RVsig']]
                                })
                            rv_df = pd.DataFrame(rv_data) if rv_data else pd.DataFrame()

                        if not rv_df.empty:
                            phase_gif_path = create_phase_folded_animation(rv_df, orbital_params, sel_star.name,
                                                                           output_dir)

                            if phase_gif_path and phase_gif_path.exists():
                                st.success(f"Phase animation saved: {phase_gif_path.name}")

                                with open(phase_gif_path, 'rb') as f:
                                    phase_gif_data = f.read()
                                st.image(phase_gif_data, caption=f"Phase-Folded RV - {sel_star.name}")

                                st.download_button(
                                    "Download Phase Animation",
                                    phase_gif_data,
                                    file_name=phase_gif_path.name,
                                    mime="image/gif"
                                )
                    else:
                        st.error("Need RV data and orbital fit to create phase animation")


        # -------------------------------
        # Classifier section (optional)
        # -------------------------------
        if use_classifier and root_folder and output_root:
            st.markdown("### Classifier (SB1 / SB2)")

            lam_ranges = parse_ranges_text(st.session_state.get("clf_lams", "4000-4500"))
            detect_rng = parse_ranges_text(st.session_state.get("clf_detect_span", "4000-4570")) or [(4000.0, 4570.0)]
            detect_span = detect_rng[0]

            if clf_line_mode == "Manual list":
                manual_lines = parse_lines(st.session_state.get("clf_manual_list", ""))
                auto_detect = False
            else:
                manual_lines = []
                auto_detect = True

            # default "known" sets to help the matcher when auto-detect is on
            known_sets = {
                "H": [3970.072, 4101.734, 4340.462],
                "He": [4026.191, 4387.929, 4471.479],
                "HeII": [4199.83],
                "He_ext": [4143.760],
            }

            # choose template mode keyword
            tmpl_mode = {"first observation": "first",
                         "combined mask FITS (*Combined*.fits)": "mask",
                         "model template (model_root)": "model"}[st.session_state["clf_template"]]

            # use the same observations we loaded for the viewer (fast)
            if not obs_for_view:
                st.info("No spectra available for this star.")
            else:
                # --- NEW: SNR selection mode ---
                chosen_snr = None
                st.info("Reached SNR selection block")  # temporary sanity check
                st.markdown("#### SNR selection")
                snr_mode = st.radio(
                    "How to choose SNR continuum ranges?",
                    ["Presets (default bands)", "Global list (text)", "Global (pick with plot)",
                     "Per-line (pick on plot)"],
                    index=0, horizontal=True, key="clf_snr_mode_body"
                )

                chosen_snr = None
                # bounds for sliders
                lam_lo_all = min(lo for lo, _ in lam_ranges)

                lam_hi_all = max(hi for _, hi in lam_ranges)

                # state holders
                st.session_state.setdefault("snr_global_ranges", [])  # List[Tuple]
                st.session_state.setdefault("snr_per_line", {})  # Dict[float, List[Tuple]]

                if snr_mode == "Presets (default bands)":
                    chosen_snr = None
                    st.caption(
                        "Using built-in continuum bands around 4045–4060, 4220–4230, 4495–4550 **per line** as appropriate.")
                elif snr_mode == "Global list (text)":
                    txt = st.text_input(
                        "Enter SNR ranges (Å):",
                        value="4045-4060; 4220-4230; 4495-4550",
                        help="Format: a-b; c-d; ...",
                        key="snr_txt_global"
                    )
                    chosen_snr = parse_ranges_text(txt) or None
                    if chosen_snr:
                        st.write("Using global SNR ranges:", chosen_snr)

                elif snr_mode == "Global (pick with plot)":
                    st.session_state.setdefault("snr_global_ranges", [])

                    # choose spectrum to display: coadd if available, else first epoch
                    cw = st.session_state.get("coadd_wave");
                    cf = st.session_state.get("coadd_flux")
                    if isinstance(cw, np.ndarray) and isinstance(cf, np.ndarray) and cw.size and cf.size:
                        base_w, base_f = cw.astype(float), np.nan_to_num(cf).astype(float)
                    else:
                        base = obs_for_view[0].spectrum
                        base_w, base_f = base[:, 0].astype(float), np.nan_to_num(base[:, 1]).astype(float)

                    lam_lo_req = min(lo for lo, _ in lam_ranges) if lam_ranges else float(np.nanmin(base_w))
                    lam_hi_req = max(hi for _, hi in lam_ranges) if lam_ranges else float(np.nanmax(base_w))
                    x, y, lo_used, hi_used, ok = _clip_to_data(base_w, base_f, lam_lo_req, lam_hi_req)

                    fig = go.Figure()

                    # Use non-GL and add tiny markers so selection picks up points
                    if x.size:
                        fig.add_trace(go.Scatter(
                            x=x.tolist(),
                            y=y.tolist(),
                            mode="lines+markers",
                            line=dict(width=1.6),
                            marker=dict(size=2, opacity=0.15),  # nearly invisible, but selectable
                            name="spectrum"
                        ))

                    # Shade already-saved windows + current pending selection
                    add_vrects_to_plotly(fig, st.session_state["snr_global_ranges"])
                    if st.session_state.get("snr_pending_global"):
                        lo_p, hi_p = st.session_state["snr_pending_global"]
                        fig.add_vrect(x0=lo_p, x1=hi_p, opacity=0.25, line_width=1, line_dash="dot")

                    fig.update_layout(
                        height=340,
                        margin=dict(l=20, r=10, t=10, b=30),
                        dragmode="select",
                        showlegend=False,
                        xaxis_title="λ (Å)", yaxis_title="Flux"
                    )
                    fig.update_xaxes(range=[lo_used, hi_used])

                    if not ok:
                        st.warning(f"No points inside [{lam_lo_req:.1f}, {lam_hi_req:.1f}] Å. "
                                   f"Showing full data [{lo_used:.1f}, {hi_used:.1f}] Å instead.")
                    # capture selection (render + remember)
                    if HAVE_PLOTLY_EVENTS:
                        selected = plotly_events(fig, select_event=True, override_height=340, key="snr_global_plot")
                    else:
                        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                        selected = []

                    # remember the latest box across reruns
                    st.session_state.setdefault("snr_pending_global", None)
                    rng = _windows_from_selected_points(selected)
                    if rng:
                        st.session_state["snr_pending_global"] = tuple(sorted(map(float, rng)))

                    # show what's pending right now
                    pending = st.session_state.get("snr_pending_global")
                    if pending:
                        st.caption(f"Pending selection: [{pending[0]:.1f}, {pending[1]:.1f}] Å")

                    c_add, c_undo, c_clear = st.columns(3)
                    if c_add.button("Add selection", key="snr_global_add"):
                        if pending:
                            st.session_state["snr_global_ranges"].append(pending)
                            st.session_state["snr_pending_global"] = None  # clear after adding
                    if c_undo.button("Undo last", key="snr_global_undo"):
                        if st.session_state["snr_global_ranges"]:
                            st.session_state["snr_global_ranges"].pop()
                    if c_clear.button("Clear all", key="snr_global_clear"):
                        st.session_state["snr_global_ranges"] = []
                        st.session_state["snr_pending_global"] = None

                    chosen_snr = st.session_state["snr_global_ranges"] or None
                    if chosen_snr:
                        st.write("Global SNR ranges:", chosen_snr)

                    # preview (also shade the pending box if any)
                    fig_preview = go.Figure()
                    if x.size:
                        fig_preview.add_trace(go.Scatter(x=x, y=y, mode="lines"))
                    add_vrects_to_plotly(fig_preview, chosen_snr)
                    if pending:
                        fig_preview.add_vrect(x0=pending[0], x1=pending[1], opacity=0.25, line_width=1, line_dash="dot")
                    fig_preview.update_layout(margin=dict(l=20, r=10, t=10, b=30), height=280, showlegend=False,
                                              xaxis_title="λ (Å)", yaxis_title="Flux")
                    st.plotly_chart(fig_preview, use_container_width=True, config={"displaylogo": False})

                elif snr_mode == "Per-line (pick on plot)":
                    st.caption("Draw a small box around a continuum near a line, then click **Add to nearest line**.")

                    # spectrum (coadd preferred)
                    cw = st.session_state.get("coadd_wave");
                    cf = st.session_state.get("coadd_flux")
                    if isinstance(cw, np.ndarray) and isinstance(cf, np.ndarray) and cw.size and cf.size:
                        base_w, base_f = cw.astype(float), np.nan_to_num(cf).astype(float)
                    else:
                        base = obs_for_view[0].spectrum
                        base_w, base_f = base[:, 0].astype(float), np.nan_to_num(base[:, 1]).astype(float)

                    lam_lo_req = min(lo for lo, _ in lam_ranges)
                    lam_hi_req = max(hi for _, hi in lam_ranges)
                    lo = max(lam_lo_req, float(np.nanmin(base_w)))
                    hi = min(lam_hi_req, float(np.nanmax(base_w)))
                    sel = (base_w >= lo) & (base_w <= hi)
                    x = base_w[sel];
                    y = base_f[sel]

                    # seed line list
                    centers_seed = st.session_state.get("snr_detected_lines", []) or manual_lines
                    if not centers_seed:
                        centers_seed = (
                                    known_sets.get("H", []) + known_sets.get("He", []) + known_sets.get("He_ext", []))
                    centers = [float(c) for c in centers_seed if lo <= float(c) <= hi]

                    st.session_state.setdefault("snr_per_line", {})  # Dict[center] -> List[(lo,hi)]
                    st.session_state.setdefault("snr_last_added", None)  # (center, (lo,hi))
                    fig = go.Figure()

                    # Use non-GL and add tiny markers so selection picks up points
                    if x.size:
                        fig.add_trace(go.Scatter(
                            x=x.tolist(),
                            y=y.tolist(),
                            mode="lines+markers",
                            line=dict(width=1.6),
                            marker=dict(size=2, opacity=0.15),  # nearly invisible, but selectable
                            name="spectrum"
                        ))

                    # Shade already-saved windows + current pending selection
                    add_vrects_to_plotly(fig, st.session_state["snr_global_ranges"])
                    if st.session_state.get("snr_pending_global"):
                        lo_p, hi_p = st.session_state["snr_pending_global"]
                        fig.add_vrect(x0=lo_p, x1=hi_p, opacity=0.25, line_width=1, line_dash="dot")

                    fig.update_layout(
                        height=340,
                        margin=dict(l=20, r=10, t=10, b=30),
                        dragmode="select",
                        showlegend=False,
                        xaxis_title="λ (Å)", yaxis_title="Flux"
                    )
                    fig.update_xaxes(range=[lo, hi])

                    for L in centers:
                        fig.add_vline(x=float(L), line_dash="dot", opacity=0.35)

                    # add previously saved and pending windows right on the main plot
                    add_vrects_to_plotly(fig, st.session_state["snr_per_line"])
                    if st.session_state.get("snr_pending_perline"):
                        lo_p, hi_p = st.session_state["snr_pending_perline"]
                        fig.add_vrect(x0=lo_p, x1=hi_p, opacity=0.25, line_width=1, line_dash="dot")

                    if HAVE_PLOTLY_EVENTS:
                        selected = plotly_events(fig, select_event=True, override_height=340, key="snr_perline_plot")
                    else:
                        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                        selected = []

                    # remember pending window across reruns
                    st.session_state.setdefault("snr_pending_perline", None)
                    rng = _windows_from_selected_points(selected)
                    if rng:
                        st.session_state["snr_pending_perline"] = tuple(sorted(map(float, rng)))

                    pending = st.session_state.get("snr_pending_perline")
                    if pending:
                        st.caption(f"Pending selection: [{pending[0]:.1f}, {pending[1]:.1f}] Å")

                    c_add, c_undo, c_clear = st.columns(3)
                    if c_add.button("Add to nearest line", key="snr_perline_add"):
                        if pending and centers:
                            lo_win, hi_win = pending
                            mid = 0.5 * (lo_win + hi_win)
                            nearest = float(min(centers, key=lambda c: abs(c - mid)))
                            st.session_state["snr_per_line"].setdefault(nearest, []).append((lo_win, hi_win))
                            st.session_state["snr_last_added"] = (nearest, (lo_win, hi_win))
                            st.session_state["snr_pending_perline"] = None
                            st.success(f"Added [{lo_win:.1f}, {hi_win:.1f}] Å → line {nearest:.2f} Å")

                    if c_undo.button("Undo last", key="snr_perline_undo"):
                        last = st.session_state.get("snr_last_added")
                        if last:
                            c, win = last
                            arr = st.session_state["snr_per_line"].get(c, [])
                            try:
                                arr.remove(win)
                            except ValueError:
                                pass
                            st.session_state["snr_last_added"] = None

                    if c_clear.button("Clear all", key="snr_perline_clear"):
                        st.session_state["snr_per_line"] = {}
                        st.session_state["snr_last_added"] = None
                        st.session_state["snr_pending_perline"] = None

                    chosen_snr = st.session_state["snr_per_line"] or None

                    # preview (shade saved windows + dashed pending)
                    if x.size:
                        fig_preview = go.Figure()
                        fig_preview.add_trace(go.Scatter(x=x, y=y, mode="lines"))
                        add_vrects_to_plotly(fig_preview, chosen_snr)
                        if pending:
                            fig_preview.add_vrect(x0=pending[0], x1=pending[1], opacity=0.25, line_width=1,
                                                  line_dash="dot")
                        fig_preview.update_layout(margin=dict(l=20, r=10, t=10, b=30), height=280, showlegend=False,
                                                  xaxis_title="λ (Å)", yaxis_title="Flux")
                        st.plotly_chart(fig_preview, use_container_width=True)
                st.caption(f"Using SNR windows → {('defaults' if chosen_snr is None else chosen_snr)}")

                # --- Run classifier ---

                st.session_state["chosen_snr"] = chosen_snr  # None | list[(lo,hi)] | dict{center: [(lo,hi), ...]}
                run_clf = st.button("Run classifier for this star", key="btn_run_classifier")
                if run_clf:
                    # get whatever the picker saved
                    snr_choice = st.session_state.get("chosen_snr", None)

                    with st.spinner("Classifying…"):
                        final_class, df_lines, coa_w, coa_f = _classify_lines_for_star(
                            sel_star, obs_for_view, lam_ranges,
                            float(st.session_state["clf_vmin"]),
                            float(st.session_state["clf_vmax"]),
                            float(st.session_state["clf_fitf"]),
                            float(st.session_state["clf_overs"]),
                            snr_choice,  # <-- the chosen SNR windows go in here
                            auto_detect, manual_lines, (float(detect_span[0]), float(detect_span[1])),
                            float(st.session_state["clf_tol"]),
                            float(st.session_state["clf_winH"]),
                            float(st.session_state["clf_winHe"]),
                            float(st.session_state["clf_P"]),
                            known_sets, tmpl_mode,
                            model_root=st.session_state.get("model_root", "")
                        )

                    st.subheader(f"Final classification: {final_class}")
                    st.session_state["last_classification"] = final_class

                    if coa_w is not None and coa_f is not None:
                        st.line_chart(pd.DataFrame({"λ (Å)": coa_w, "Coadd flux": coa_f}).set_index("λ (Å)"))

                    if not df_lines.empty:
                        st.dataframe(df_lines, use_container_width=True)

                        # save Excel
                        star_dir = Path(output_root) / sel_star.name
                        star_dir.mkdir(parents=True, exist_ok=True)
                        tstamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        xlsx_path = star_dir / f"classification.xlsx"
                        with pd.ExcelWriter(xlsx_path) as xw:
                            pd.DataFrame([{"Star": sel_star.name, "Classification": final_class}]).to_excel(xw,
                                                                                                            sheet_name="summary",
                                                                                                            index=False)
                            df_lines.to_excel(xw, sheet_name="per_line", index=False)
                        st.success(f"Saved classification results → {xlsx_path.name}")
                        st.download_button("Download classification Excel",
                                           data=Path(xlsx_path).read_bytes(),
                                           file_name=xlsx_path.name)

                        # optionally add to orbit_summary.csv
                        if write_class_to_summary:
                            try:
                                summ_path = Path(root_folder) / "orbit_summary.csv"
                                if summ_path.exists():
                                    df = pd.read_csv(summ_path)
                                else:
                                    df = pd.DataFrame(columns=["Star"])
                                # upsert by Star
                                if "Star" not in df.columns:
                                    df["Star"] = ""
                                mask = (df["Star"] == sel_star.name)
                                if mask.any():
                                    df.loc[mask, "Classification"] = final_class
                                else:
                                    df = pd.concat(
                                        [df, pd.DataFrame([{"Star": sel_star.name, "Classification": final_class}])],
                                        ignore_index=True)
                                df.to_csv(summ_path, index=False)
                                st.info(f"Added/updated classification in {summ_path.name}")
                            except Exception as e:
                                st.warning(f"Could not update orbit_summary.csv: {e}")

        st.markdown("### Orbit fit")

        rv_mean_df = st.session_state.get('rv_mean_df', None)
        if rv_mean_df is None or rv_mean_df.empty:
            st.info("Run the CCF step above first to produce weighted-mean RVs.")
        else:
            c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
            with c1:
                top_k = st.number_input("How many period candidates to try", 1, 30, 5, step=1, key="fit_topk")
            with c2:
                n_sigma = st.slider("± N·σ(P) window", 0.0, 5.0, 2.0, step=0.001, key="fit_nsigma")
            with c3:
                min_sep = st.slider("Peak separation factor", 1.00, 2.00, 1.20, step=0.05, key="fit_minsep",
                                    help="Minimum multiplicative separation between picked peaks (log-space).")
            with c4:
                go_multi = st.button("Search & fit top-N periods", key="btn_fit_multi")
            # NEW — periodogram search settings
            st.markdown("**Periodogram search settings**")
            pp_cols = st.columns([1, 1, 1])
            with pp_cols[0]:
                st.number_input(
                    "P_min [days]", min_value=0.1, max_value=1e6, step=0.1,
                    value=float(st.session_state.get("periodogram_pmin", 1.0)),
                    key="periodogram_pmin"
                )
            with pp_cols[1]:
                st.number_input(
                    "P_max [days]", min_value=0.2, max_value=1e6, step=1.0,
                    value=float(st.session_state.get("periodogram_pmax", 15000.0)),
                    key="periodogram_pmax"
                )
            with pp_cols[2]:
                st.number_input(
                    "FAP threshold (%)", min_value=0.0001, max_value=100.0, step=0.01, format="%.5f",
                    value=float(st.session_state.get("fap_percent", 0.1)),
                    key="fap_percent"
                )

            if go_multi:
                with st.spinner("Running period search and fitting candidates…"):
                    outcome = run_orbit_fit(
                        rv_mean_df,
                        st.session_state.get('last_star', 'star'),
                        Path(st.session_state.get('out_root', '.')),
                        st.session_state.get('json_params') or None,
                        top_k=int(top_k), n_sigma=float(n_sigma), min_sep=float(min_sep)
                    )
                # store for later sections
                st.session_state['fit_outcome'] = outcome
                st.session_state['fit_candidates'] = outcome.candidates or []
                st.session_state['fit_selected'] = outcome.best_index

            # If we have candidates, show the list and picker
            cands: List[FitCandidate] = st.session_state.get('fit_candidates', [])
            if cands:
                # table
                table_rows = []
                for i, c in enumerate(cands):
                    r = c.summary
                    table_rows.append(dict(
                        index=i,
                        period_value=pick_num(r, 'period_value', 'Period', 'period', 'P'),
                        k1_value=pick_num(r, 'k1_value', 'K1', 'k1'),
                        ecc_value=pick_num(r, 'ecc_value', 'Eccentricity', 'e'),
                        gamma_value=pick_num(r, 'gamma_value', 'gamma', 'Gamma', 'V0'),
                        omega_value=pick_num(r, 'omega_value', 'omega', 'Omega', 'w', 'argperi', 'arg_peri'),
                        t_value=pick_num(r, 't_value', 'T', 't0', 'T0', 'T_peri', 'Tperi', 'T_periastron'),
                        chisqr=pick_num(r, 'chisqr', 'chi2', 'chisq'),
                        redchi=pick_num(r, 'redchi', 'chi2_red', 'reduced_chi2'),
                        aic=pick_num(r, 'aic', 'AIC'),
                        bic=pick_num(r, 'bic', 'BIC'),
                        nvarys=int(r.get('nvarys', np.nan)) if str(r.get('nvarys', '')).isdigit() else r.get('nvarys',
                                                                                                             np.nan),
                        ndata=int(r.get('ndata', np.nan)) if str(r.get('ndata', '')).isdigit() else r.get('ndata',
                                                                                                          np.nan),
                        F_stat=pick_num(r, 'F_stat'),
                        F_pvalue=pick_num(r, 'F_pvalue'),
                        bin_flag=int(r.get('bin_flag', 0)),
                    ))
                df_cands = pd.DataFrame(table_rows)
                df_cands['redchi_score'] = (df_cands['redchi'] - 1.0).abs()
                df_cands = df_cands.sort_values(by=['redchi_score', 'chisqr'], na_position='last')
                st.dataframe(df_cands, use_container_width=True)

                # picker
                idx_default = st.session_state.get('fit_selected', 0) or 0
                idx_pick = st.selectbox(
                    "Choose a candidate to view",
                    options=list(df_cands['index'].to_numpy(int)),
                    index=min(idx_default, len(df_cands) - 1),
                    format_func=lambda
                        i: f"P={cands[i].summary.get('period_value', cands[i].period):.6f} d  (AIC={cands[i].summary.get('aic', np.nan):.1f})",
                    key="fit_pick_idx"
                )
                st.session_state['fit_selected'] = int(idx_pick)

                chosen = cands[int(idx_pick)]
                # Make this the 'current' fit for calculators
                st.session_state['last_fit_row'] = chosen.summary
                st.session_state['last_fit_result'] = chosen.result
                # convenience: we need the original data too
                data_path = Path(st.session_state.get('out_root', '.'), st.session_state.get('last_star', ''),
                                 f"{st.session_state.get('last_star', '')}_CCF_RVs.csv")
                data_for_fit = st.session_state['rv_mean_df'].rename(
                    columns={'MJD': TIME_STAMPS, 'Mean RV': RADIAL_VELS, 'Mean RVsig': ERRORS})

                col_actions = st.columns(3)
                if col_actions[0].button("Preview selected (no files)"):
                    import tempfile

                    with tempfile.TemporaryDirectory() as td:
                        # Use the ALREADY COMPUTED result stored in the candidate object
                        res = chosen.result

                        # Generates plots from the ORIGINAL result
                        print_lmfit_result(data_for_fit, chosen.args, st.session_state.get('last_star', 'star'), res,
                                           out_dir=td)

                        pngs = sorted(Path(td).glob("*.png"))
                        cols = st.columns(2)
                        for i, p in enumerate(pngs):
                            cols[i % 2].image(str(p), caption=p.name, use_container_width=True)
                if col_actions[1].button("Save selected fit to output folder"):
                    star = st.session_state.get('last_star', 'star')
                    target = Path(st.session_state.get('out_root', '.'), star,
                                  f"fit_P_{pick_num(chosen.summary, 'period_value', 'Period', 'P'):.6f}d")
                    target.mkdir(parents=True, exist_ok=True)
                    res = lmfit_on_sample(chosen.args, str(target), data_for_fit, star)
                    print_lmfit_result(data_for_fit, chosen.args, star, res, out_dir=str(target))
                    chosen.saved_dir = target
                    chosen.images = list(sorted(target.glob("*.png")))
                    st.success(f"Saved to {target}")

                if col_actions[2].button("Mark selected as BEST"):
                    star = st.session_state.get('last_star', 'star')
                    best_dir = Path(st.session_state.get('out_root', '.'), star, "best")
                    best_dir.mkdir(parents=True, exist_ok=True)

                    res = chosen.result

                    print_lmfit_result(data_for_fit, chosen.args, star, res, out_dir=str(best_dir))

                    with open(best_dir / "best_summary.json", "w") as jf:
                        json.dump(chosen.summary, jf, indent=2)

                    df_params_best = lmfit_params_dataframe(res)

                    # 2. Save detailed parameters to a separate CSV file
                    if not df_params_best.empty:
                        params_csv_path = best_dir / "best_parameters.csv"
                        df_params_best.to_csv(params_csv_path, index=False)

                    # 3. Save summary + params to an Excel file
                    try:
                        df_summary_best = pd.DataFrame([chosen.summary])
                        xlsx_path = best_dir / "best_summary.xlsx"
                        with pd.ExcelWriter(xlsx_path) as xw:
                            df_summary_best.to_excel(xw, sheet_name="summary", index=False)
                            if not df_params_best.empty:
                                df_params_best.to_excel(xw, sheet_name="params", index=False)
                        st.success(f"Saved all results (plots, params, summary) to the '{best_dir.name}' folder.")
                    except Exception as _e:
                        st.warning(f"Could not write best_summary.xlsx: {_e}")
                    # --- END OF ENHANCED SAVE SECTION ---

                    # --- Also build + save a one-row summary CSV in the *input* root ---
                    star = st.session_state.get('last_star', 'star')
                    # Pull stellar M and R from the mass table for this star
                    ms1, ms_plus, ms_minus, rstar, r_plus, r_minus = _lookup_mass_row(mass_csv, star)

                    summary_row = build_summary_row(
                        star_name=star,
                        summ=chosen.summary,
                        mspec=ms1, ms_plus=ms_plus, ms_minus=ms_minus,
                        rstar=rstar, r_plus=r_plus, r_minus=r_minus,
                        alpha_peri=1.2,  # tweak if desired
                        alpha_apa=1.0,
                    )

                    # Save to the INPUT root folder as requested
                    summary_csv_path = Path(root_folder) / "orbit_summary.csv"
                    append_summary_csv(summary_row, summary_csv_path)
                    st.info(f"Summary saved/updated: {summary_csv_path}")
                    st.session_state['last_fit_row'] = summary_row
                    st.session_state['last_fit_result'] = res
                # periodograms
                outcome = st.session_state.get('fit_outcome', None)
                if outcome and outcome.ls_path and outcome.ls_path.exists():
                    st.image(str(outcome.ls_path), caption="LS Periodogram", use_container_width=True)
                if outcome and outcome.pdc_path and outcome.pdc_path.exists():
                    st.image(str(outcome.pdc_path), caption="PDC Periodogram", use_container_width=True)
                if HAVE_CORNER:
                    st.markdown("#### MCMC Analysis")

                    mcmc_cols = st.columns([2, 2, 2])
                    with mcmc_cols[0]:
                        mcmc_steps = st.number_input("MCMC steps", 500, 10000, 2000, step=500, key="mcmc_steps")
                    with mcmc_cols[1]:
                        burn_steps = st.number_input("Burn-in steps", 100, 2000, 500, step=100, key="burn_steps")
                    with mcmc_cols[2]:
                        thin_steps = st.number_input("Thinning", 1, 50, 10, step=1, key="thin_steps")

                    # --- MCMC Actions ---
                    mcmc_action_cols = st.columns(2)
                    with mcmc_action_cols[0]:
                        if st.button("Generate Corner Plot (MCMC)", key="btn_corner"):
                            if 'last_fit_result' not in st.session_state or st.session_state['last_fit_result'] is None:
                                st.error("Please select a candidate fit from the table above first.")
                            else:
                                star_name = st.session_state.get('last_star', 'star')
                                output_path = Path(st.session_state.get('out_root', '.')) / star_name
                                output_path.mkdir(parents=True, exist_ok=True)
                                # 1. Create a deep copy to avoid changing the original arguments
                                import copy

                                args_for_mcmc = copy.deepcopy(chosen.args)

                                # 2. Ensure the 'CORNER_PARAMS' dictionary exists
                                corner_params = args_for_mcmc.setdefault('CORNER_PARAMS', {})

                                # 3. Update it with the values from your Streamlit UI controls
                                #    (The keys 'STEPS', 'BURN', 'THIN' must match what corner_plot2 expects)
                                steps_v = int(mcmc_steps)
                                burn_v = int(burn_steps)
                                thin_v = int(thin_steps)

                                # canonical (what we tried before)
                                corner_params['STEPS'] = steps_v
                                corner_params['BURN'] = burn_v
                                corner_params['THIN'] = thin_v

                                # common aliases so the MCMC code will always find them
                                corner_params['steps'] = steps_v
                                corner_params['nsteps'] = steps_v
                                corner_params['N_STEPS'] = steps_v

                                corner_params['burn'] = burn_v
                                corner_params['nburn'] = burn_v
                                corner_params['burnin'] = burn_v
                                corner_params['N_BURN'] = burn_v

                                corner_params['thin'] = thin_v
                                corner_params['nthin'] = thin_v

                                # Also ensure default values exist for other expected keys if they're missing
                                corner_params.setdefault('NAN_POLICY', 'omit')
                                corner_params.setdefault('CONRER_METHOD', 'emcee')
                                corner_params.setdefault('LN_SIGMA',
                                                         {'INIT_VAL': 1.0, 'MIN_VAL': 0.01, 'MAX_VAL': 100.0})

                                # 4. Call the function with the updated arguments dictionary
                                result = corner_plot2(
                                    args_for_mcmc,
                                    data_for_fit,
                                    chosen.result,
                                    star_name,
                                    str(output_path)  # Ensure output_path is a string
                                )

                                if result:
                                    corner_path, mcmc_res = result
                                    corner_path = Path(corner_path)
                                    st.success(f"Corner plot saved: {corner_path.name}")
                                    st.session_state['last_mcmc_result'] = mcmc_res
                                    st.session_state['corner_plot_path'] = str(corner_path)

                                    # --- Save an MCMC-derived summary row (medians ±1σ) and recomputed derived quantities ---
                                    try:
                                        chain = mcmc_res.flatchain
                                        cols = list(getattr(chain, "columns", []))


                                        def q(a):
                                            a = np.asarray(a, float)
                                            a = a[np.isfinite(a)]
                                            if a.size == 0:
                                                return (np.nan, np.nan, np.nan)
                                            p16, p50, p84 = np.nanpercentile(a, [16, 50, 84])
                                            return p50, p50 - p16, p84 - p50


                                        def grab(*names):
                                            for nm in names:
                                                if nm in cols:
                                                    return chain[nm].to_numpy()
                                            low = {str(c).lower(): str(c) for c in cols}
                                            for nm in names:
                                                if nm.lower() in low:
                                                    return chain[low[nm.lower()]].to_numpy()
                                            return None


                                        P = grab('Period', 'period', 'P')
                                        K1 = grab('K1', 'k1')
                                        e = grab('eccentricity', 'Eccentricity', 'e')
                                        g = grab('gamma', 'Gamma', 'V0', 'v0')
                                        w = grab('omega', 'Omega', 'w', 'argperi', 'arg_peri', 'omega_rad')
                                        T0 = grab('T0', 't0', 'T', 't')

                                        row = {"Star": st.session_state.get('last_star', 'star')}
                                        if P is not None: Pm, Pm_, Pp_ = q(P);   row.update(
                                            {"P": Pm, "P_err_minus": Pm_, "P_err_plus": Pp_})
                                        if K1 is not None: Km, Km_, Kp_ = q(K1);  row.update(
                                            {"K1": Km, "K1_err_minus": Km_, "K1_err_plus": Kp_})
                                        if e is not None: em, em_, ep_ = q(e);   row.update(
                                            {"e": em, "e_err_minus": em_, "e_err_plus": ep_})
                                        if g is not None: gm, gm_, gp_ = q(g);   row.update(
                                            {"gamma": gm, "gamma_err_minus": gm_, "gamma_err_plus": gp_})
                                        if w is not None: wm, wm_, wp_ = q(w);   row.update(
                                            {"omega": wm, "omega_err_minus": wm_, "omega_err_plus": wp_})
                                        PHASE0 = grab('PHASE0', 'phase0')
                                        if P is not None and PHASE0 is not None and 'data_for_fit' in locals():
                                            t_ref = float(np.median(data_for_fit[TIME_STAMPS].to_numpy(float)))

                                            # Replicate the logic from summarize_result across the entire MCMC chain
                                            T0_abs_chain = t_ref + PHASE0 * P
                                            time_diff_chain = T0_abs_chain - t_ref
                                            # This calculates the offset T0, wrapped to be close to the reference time
                                            T0_offset_chain = (time_diff_chain / P - np.round(time_diff_chain / P)) * P

                                            # Get statistics (median, -1sigma, +1sigma) on this derived offset chain
                                            tm, tm_, tp_ = q(T0_offset_chain)
                                            row.update({"T0": tm, "T0_err_minus": tm_, "T0_err_plus": tp_})

                                        # Recompute derived M2_min and Pmin bounds across posterior draws (subsample for speed)
                                        try:
                                            ms1, ms_plus, ms_minus, rstar, r_plus, r_minus = _lookup_mass_row(mass_csv,
                                                                                                              row[
                                                                                                                  "Star"])
                                            rng = np.random.default_rng(12345)
                                            nrows = len(chain)
                                            take = min(5000, nrows)
                                            idx = rng.choice(nrows, size=take,
                                                             replace=False) if take < nrows else np.arange(
                                                nrows)

                                            P_s = P[idx] if P is not None else np.full(take, np.nan)
                                            K1_s = K1[idx] if K1 is not None else np.full(take, np.nan)
                                            e_s = e[idx] if e is not None else np.full(take, np.nan)

                                            sigM = float(ms_plus if np.isfinite(ms_plus) else 0.0)
                                            if np.isfinite(ms_minus): sigM = 0.5 * (sigM + float(ms_minus))
                                            sigR = float(r_plus if np.isfinite(r_plus) else 0.0)
                                            if np.isfinite(r_minus):  sigR = 0.5 * (sigR + float(r_minus))
                                            M1_s = rng.normal(ms1, sigM if sigM > 0 else 0.0, size=take) if np.isfinite(
                                                ms1) else np.full(take, np.nan)
                                            R_s = rng.normal(rstar, sigR if sigR > 0 else 0.0,
                                                             size=take) if np.isfinite(
                                                rstar) else np.full(take, np.nan)

                                            M2_vals, Pperi_vals, Papa_vals = [], [], []
                                            for i in range(take):
                                                Pi, Ki, ei, Mi, Ri = P_s[i], K1_s[i], e_s[i], M1_s[i], R_s[i]
                                                if not all(np.isfinite([Pi, Ki, ei, Mi, Ri])):
                                                    continue
                                                try:
                                                    res_m2 = companion_mass_min(Pi, Ki, ei, Mi, 0.0, 0.0)
                                                    M2_vals.append(float(res_m2.get("M2_min", np.nan)))
                                                    ap = Pmin_peri_apa(
                                                        Mi, Ri, Ki, ei,
                                                        float(st.session_state.get('alpha_peri_after_fit', 1.2)),
                                                        float(st.session_state.get('alpha_apa_after_fit', 1.0)),
                                                    )
                                                    Pperi_vals.append(float(ap.get("Pmin_peri", np.nan)))
                                                    Papa_vals.append(float(ap.get("Pmin_apa", np.nan)))
                                                except Exception:
                                                    continue


                                            def q3(vals):
                                                vals = np.asarray(vals, float)
                                                vals = vals[np.isfinite(vals)]
                                                if vals.size == 0:
                                                    return (np.nan, np.nan, np.nan)
                                                p16, p50, p84 = np.nanpercentile(vals, [16, 50, 84])
                                                return p50, p50 - p16, p84 - p50


                                            m2_m, m2_lo, m2_hi = q3(M2_vals)
                                            pp_m, pp_lo, pp_hi = q3(Pperi_vals)
                                            pa_m, pa_lo, pa_hi = q3(Papa_vals)

                                            row.update({
                                                "M2_min": m2_m,
                                                "M2_min_err_minus": m2_lo,
                                                "M2_min_err_plus": m2_hi,
                                                "Pmin_peri": pp_m,
                                                "Pmin_peri_err_minus": pp_lo,
                                                "Pmin_peri_err_plus": pp_hi,
                                                "Pmin_apa": pa_m,
                                                "Pmin_apa_err_minus": pa_lo,
                                                "Pmin_apa_err_plus": pa_hi,
                                                "Pmin_peri_central": pp_m,
                                                "Pmin_peri_lower": pp_m - pp_lo,
                                                "Pmin_peri_upper": pp_m + pp_hi,

                                                "Pmin_apa_central": pa_m,
                                                "Pmin_apa_lower": pa_m - pa_lo,
                                                "Pmin_apa_upper": pa_m + pa_hi,

                                            })
                                        except Exception:
                                            pass

                                        row["MCMC_STEPS"] = int(mcmc_steps)
                                        row["MCMC_BURN"] = int(burn_steps)
                                        row["MCMC_THIN"] = int(thin_steps)

                                        out_csv = Path(root_folder) / "orbit_summary_mcmc.csv"
                                        append_or_update(out_csv, row)
                                        st.success(f"MCMC summary saved/updated: {out_csv.name}")
                                    except Exception:
                                        st.warning("Could not build/save MCMC summary row.")
                    import gc

                    with mcmc_action_cols[1]:
                        mcmc_result = st.session_state.get('last_mcmc_result', None)
                        uncertainties = extract_mcmc_uncertainties(mcmc_result, data_for_fit,
                                                                   period_guess=pick_num(chosen.summary, 'period_value',
                                                                                         'P'))
                        if mcmc_result and st.button("Run Refined Fit (MCMC bounds)", key="btn_refined"):
                            del mcmc_result
                            gc.collect()
                            if uncertainties:
                                star_name = st.session_state.get('last_star', 'star')
                                output_path = Path(st.session_state.get('out_root', '.')) / star_name / "refined_fit"
                                output_path.mkdir(parents=True, exist_ok=True)
                                refined_candidate = run_refined_orbit_fit(chosen, uncertainties, data_for_fit,
                                                                          star_name,
                                                                          output_path)
                                if refined_candidate:
                                    st.session_state['refined_candidate'] = refined_candidate
                                    st.success("Refined fit completed!")
                                    st.rerun()
                            else:
                                st.error("Could not extract uncertainties from MCMC result")

                    # --- Display MCMC and Refined Fit Results ---

                    if st.session_state.get('corner_plot_path'):
                        st.image(st.session_state['corner_plot_path'], caption="MCMC Corner Plot",
                                 use_container_width=True)

                    if st.session_state.get('refined_candidate'):
                        st.markdown("##### Refined Fit Results")
                        refined_candidate = st.session_state['refined_candidate']
                        comparison = pd.DataFrame([
                            {'Fit Type': 'Original', 'χ²': chosen.summary.get('chisqr', np.nan),
                             'Reduced χ²': chosen.summary.get('redchi', np.nan),
                             'AIC': chosen.summary.get('aic', np.nan)},
                            {'Fit Type': 'Refined (MCMC bounds)', 'χ²': refined_candidate.summary.get('chisqr', np.nan),
                             'Reduced χ²': refined_candidate.summary.get('redchi', np.nan),
                             'AIC': refined_candidate.summary.get('aic', np.nan)}
                        ])
                        st.dataframe(comparison, use_container_width=True)

                        # --- THIS IS THE RESTORED SECTION ---
                        st.markdown("##### Refined Fit Parameters")
                        df_params_ref = lmfit_params_dataframe(refined_candidate.result)
                        if not df_params_ref.empty:
                            st.dataframe(df_params_ref, use_container_width=True)
                        else:
                            st.info("No detailed parameter table available for this result.")
                        # --- END OF RESTORED SECTION ---

                        if st.button("Mark Refined Fit as BEST", key="btn_mark_refined_best"):
                            star = st.session_state.get('last_star', 'star')
                            ms1, ms_plus, ms_minus, rstar, r_plus, r_minus = _lookup_mass_row(mass_csv, star)
                            summary_row = build_summary_row(star_name=star, summ=refined_candidate.summary, mspec=ms1,
                                                            ms_plus=ms_plus, ms_minus=ms_minus, rstar=rstar,
                                                            r_plus=r_plus,
                                                            r_minus=r_minus)
                            summary_csv_path = Path(root_folder) / "orbit_summary.csv"
                            summary_row["Star"] = str(summary_row["Star"]).strip()

                            # Preserve selected diagnostics from the existing row (if present)
                            try:
                                import pandas as pd

                                if summary_csv_path.exists():
                                    df = pd.read_csv(summary_csv_path)
                                    star_norm = str(summary_row["Star"]).strip()
                                    prev = df[df["Star"].astype(str).str.strip() == star_norm]
                                    if not prev.empty:
                                        prev_row = prev.iloc[0]
                                        preserve_cols = ["ls_fap", "pdc_fap", "F_stat", "F_crit_99", "F_pvalue"]
                                        # map case-insensitively to whatever exists in the CSV
                                        lower_map = {c.lower(): c for c in df.columns}
                                        for want in preserve_cols:
                                            c = lower_map.get(want.lower())
                                            if c is not None and c in prev_row:
                                                summary_row[c] = prev_row[c]
                            except Exception:
                                pass

                            append_or_update(summary_csv_path, summary_row, key="Star")
                            st.success(f"Refined fit summary saved/updated in: {summary_csv_path}")

                            st.session_state['last_fit_row'] = summary_row
                            st.session_state['last_fit_result'] = refined_candidate.result

                        output_path = Path(st.session_state.get('out_root', '.')) / st.session_state.get('last_star',
                                                                                                         'star') / "refined_fit"
                        refined_plots = sorted(output_path.glob("*.png"))

                        # Display the plots if they were successfully generated.
                        if refined_plots:
                            grid_cols = st.columns(2)
                            for i, plot_path in enumerate(refined_plots):
                                grid_cols[i % 2].image(str(plot_path), caption=plot_path.name, use_container_width=True)

                            # --- NEW: Enhanced Download Section ---
                            st.markdown("---")  # Visual separator
                            d_cols = st.columns(3)
                            star_name = st.session_state.get('last_star', 'star')
                            # Button 1: Download Parameters CSV
                            with d_cols[0]:
                                params_csv = df_params_ref.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Parameters (.csv)",
                                    data=params_csv,
                                    file_name=f"{star_name}_refined_parameters.csv",
                                    mime="text/csv"
                                )

                            # Find specific plots for individual download
                            folded_plot_path = None
                            orbital_plot_path = None
                            for p in refined_plots:
                                if "folded" in p.name:
                                    folded_plot_path = p
                                elif "with_residuals" in p.name:  # Catches the non-folded one
                                    orbital_plot_path = p

                            # Button 2: Download Orbital Fit Plot
                            with d_cols[1]:
                                if orbital_plot_path and orbital_plot_path.exists():
                                    with open(orbital_plot_path, "rb") as f:
                                        plot_data = f.read()
                                    st.download_button(
                                        label="Download Orbital Plot",
                                        data=plot_data,
                                        file_name=orbital_plot_path.name,
                                        mime="image/png"
                                    )

                            # Button 3: Download Folded Plot
                            with d_cols[2]:
                                if folded_plot_path and folded_plot_path.exists():
                                    with open(folded_plot_path, "rb") as f:
                                        plot_data = f.read()
                                    st.download_button(
                                        label="Download Folded Plot",
                                        data=plot_data,
                                        file_name=folded_plot_path.name,
                                        mime="image/png"
                                    )
                        else:
                            st.warning("Refined fit plots could not be found or generated.")
                    mcmc_result = st.session_state.get('last_mcmc_result', None)
                    if mcmc_result:
                        st.markdown("---")
                        if st.button("Save MCMC Errors to Summary File", key="btn_save_mcmc_errors",
                                     help="Calculates asymmetric 1-sigma errors from the MCMC chains and adds them to orbit_summary.csv"):
                            uncertainties = extract_mcmc_uncertainties(mcmc_result, data=data_for_fit)
                            if uncertainties:
                                star_name = st.session_state.get('last_star', 'star')
                                summary_csv_path = Path(root_folder) / "orbit_summary.csv"
                                save_errors_to_summary(uncertainties, star_name, summary_csv_path)
                            else:
                                st.error("Could not extract uncertainties from MCMC result.")
                else:
                    st.info(
                        "Install 'corner' and 'emcee' packages to enable MCMC corner plots: pip install corner emcee")
                # fit figures for the chosen candidate
                st.markdown("#### Candidate fit figures")
                cols = st.columns(2)
                shown = 0
                for p in (chosen.images or []):
                    # try to show orbit, folded, residuals – but just show all PNGs we have
                    cols[shown % 2].image(str(p), caption=p.name, use_container_width=True)
                    shown += 1

                r = chosen.summary
                P_show = pick_num(r, 'period_value', 'Period', 'period', 'P')
                K1_show = pick_num(r, 'k1_value', 'K1', 'k1')
                e_show = pick_num(r, 'ecc_value', 'Eccentricity', 'e')
                g_show = pick_num(r, 'gamma_value', 'gamma', 'Gamma', 'V0')

                ft_cols = st.columns(4)
                ft_cols[0].metric("P (days)", f"{P_show:.6f}")
                ft_cols[1].metric("K1 (km/s)", f"{K1_show:.2f}")
                ft_cols[2].metric("e", f"{e_show:.3f}")
                ft_cols[3].metric("γ (km/s)", f"{g_show:.2f}")

                ft2 = st.columns(3)
                ft2[0].metric("F statistic", f"{pick_num(r, 'F_stat'):.3f}")
                ft2[1].metric("p-value", f"{pick_num(r, 'F_pvalue'):.3g}")
                ft2[2].metric("bin flag", str(int(r.get('bin_flag', 0))))

                st.caption("Tip: the chosen candidate’s parameters are now feeding the mass & P_min calculators below.")
            else:
                st.info("Click **Search & fit top-N periods** to see candidate fits.")

        st.markdown("---")
        st.markdown("### Companion mass & P_min calculators")

        # Pull last fit row if present (was saved in step #1)
        fit_row = st.session_state.get('last_fit_row', {}) or {}


        def _pick(row, *names):
            for n in names:
                if n in row:
                    try:
                        v = float(row[n])
                        if np.isfinite(v):
                            return v
                    except Exception:
                        pass
            return np.nan


        P_from_fit = _pick(fit_row, 'period_value', 'Period', 'period', 'P')
        K1_from_fit = _pick(fit_row, 'k1_value', 'K1', 'k1')
        e_from_fit = _pick(fit_row, 'ecc_value', 'Eccentricity', 'e')

        param_source = st.selectbox(
            "Parameter source",
            ["Latest orbit fit (above)", "From orbit fit report in output folder", "Manual entry"],
            index=0 if np.isfinite(P_from_fit) else 1,
            key="param_source_after_fit"
        )

        # Resolve P, K1, e according to the chosen source
        if param_source == "Latest orbit fit (above)":
            P_val, K1_val, e_val = P_from_fit, K1_from_fit, e_from_fit

        elif param_source == "From orbit fit report in output folder":
            star_name = st.session_state.get('last_star', '')
            rep_dir = Path(st.session_state.get('out_root', '.')) / star_name
            cand = None
            if rep_dir.is_dir():
                # *_report.txt preferred
                for p in sorted(rep_dir.glob("*_report.txt")):
                    cand = p;
                    break
                if cand is None:
                    for p in sorted(rep_dir.glob("*.txt")):
                        cand = p;
                        break
            if cand and cand.exists():
                txt = cand.read_text(encoding='utf-8', errors='ignore')


                def grab(key):
                    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", txt,
                                  flags=re.MULTILINE)
                    return float(m.group(1)) if m else np.nan


                P_val = grab('Period')
                K1_val = grab('K1')
                e_val = grab('Eccentricity')
                st.caption(f"Using report: {cand.name}")
            else:
                st.warning("No orbit fit report found; switch to Manual entry.")
                P_val = K1_val = e_val = np.nan

        else:  # Manual entry
            P_val = st.number_input("Period P (days)", 0.1, 1e5,
                                    value=float(P_from_fit if np.isfinite(P_from_fit) else 10.0),
                                    step=0.1, key="P_in_after_fit")
            K1_val = st.number_input("K1 (km/s)", 0.01, 1e4,
                                     value=float(K1_from_fit if np.isfinite(K1_from_fit) else 30.0),
                                     step=0.1, key="K1_in_after_fit")
            e_val = st.number_input("Eccentricity e", 0.0, 0.99,
                                    value=float(e_from_fit if np.isfinite(e_from_fit) else 0.1),
                                    step=0.01, key="e_in_after_fit")

        # ---- Pull stellar M and R from the mass table ----
        mspec = ms_plus = ms_minus = rstar = r_plus = r_minus = np.nan
        if mass_csv and Path(mass_csv).is_file():
            try:
                dfm = pd.read_csv(mass_csv)
                base = st.session_state.get('last_star', '')
                key = f"BLOeM_{base}"
                rowm = dfm.loc[dfm['ID'] == key]
                if rowm.empty:
                    m = ID_RE.search(base or "")
                    if m:
                        key = f"BLOeM_{m.group(0)}"
                        rowm = dfm.loc[dfm['ID'] == key]
                if not rowm.empty:
                    mspec = float(rowm['Mspec'].iloc[0])
                    ms_plus = float(rowm['Mspec_er_plus'].iloc[0])
                    ms_minus = float(rowm['Mspec_er_minus'].iloc[0])
                    rstar = float(rowm['R_star'].iloc[0]) if 'R_star' in rowm.columns else np.nan
                    r_plus = float(rowm.get('R_star_er_plus', pd.Series([np.nan])).iloc[0])
                    r_minus = float(rowm.get('R_star_er_minus', pd.Series([np.nan])).iloc[0])
                else:
                    st.warning(f"Mass row not found for {key} in {Path(mass_csv).name}")
            except Exception as _e:
                st.error(f"Failed reading mass table: {_e}")
        else:
            st.info("Provide a valid mass table CSV to enable M2 and P_min calculators.")

        col1, col2 = st.columns(2)

        with col1:
            if np.isfinite(mspec):
                st.metric("M1 (Msun)", f"{mspec:.2f}")
            if st.button("Compute M2,min from mass function", key="btn_m2min_after_fit"):
                if not all(np.isfinite(x) for x in [P_val, K1_val, e_val, mspec]):
                    st.error("Need P, K1, e and M1 to compute M2,min.")
                else:
                    out = companion_mass_min(P_val, K1_val, e_val, mspec, ms_plus, ms_minus)
                    st.json({k: (None if not np.isfinite(v) else float(v)) for k, v in out.items()})

        with col2:
            if np.isfinite(mspec) and np.isfinite(rstar):
                alpha_peri = st.number_input("α (periastron fill-factor)", 0.8, 2.0, 1.2, step=0.05,
                                             key="alpha_peri_after_fit")
                alpha_apa = st.number_input("α (apastron)", 0.8, 2.0, 1.0, step=0.05, key="alpha_apa_after_fit")
                if st.button("Compute P_min at periastron/apastron", key="btn_pmin_after_fit"):
                    if not all(np.isfinite(x) for x in [K1_val, e_val, mspec, rstar]):
                        st.error("Need K1, e, M1 and R* (with errors for bounds) to compute P_min.")
                    else:
                        # Run the full 3x3 grid calculation for all values
                        full_results = build_summary_row(
                            star_name=st.session_state.get('last_star', ''),
                            summ={'period_value': P_val, 'k1_value': K1_val, 'ecc_value': e_val},  # Dummy summary
                            mspec=mspec, ms_plus=ms_plus, ms_minus=ms_minus,
                            rstar=rstar, r_plus=r_plus, r_minus=r_minus,
                            alpha_peri=alpha_peri, alpha_apa=alpha_apa
                        )

                        # Get the central solution for q values and flags
                        res_c = Pmin_peri_apa(mspec, rstar, K1_val, e_val, alpha_peri, alpha_apa)

                        # Display a clean, organized dictionary of the results
                        out = {
                            "Pmin_peri_central": full_results.get("Pmin_peri_central"),
                            "Pmin_peri_lower": full_results.get("Pmin_peri_lower"),
                            "Pmin_peri_upper": full_results.get("Pmin_peri_upper"),
                            "Pmin_apa_central": full_results.get("Pmin_apa_central"),
                            "Pmin_apa_lower": full_results.get("Pmin_apa_lower"),
                            "Pmin_apa_upper": full_results.get("Pmin_apa_upper"),
                            "q_at_min_peri": res_c.get("q_at_min_peri"),
                            "q_at_min_apa": res_c.get("q_at_min_apa"),
                            "has_two_peri_solutions": res_c.get("has_two_peri_solutions"),
                            "has_two_apa_solutions": res_c.get("has_two_apa_solutions"),
                            "--- THEORETICAL LIMIT (q -> inf) ---": "---",
                            "Pmin_peri_best_central": full_results.get("Pmin_peri_best_central"),
                            "Pmin_peri_best_lower (min possible)": full_results.get("Pmin_peri_best_lower"),
                            "Pmin_peri_best_upper": full_results.get("Pmin_peri_best_upper"),
                        }
                        # Clean up for JSON display (handles booleans and NaNs)
                        display_dict = {}
                        for k, v in out.items():
                            if isinstance(v, str):
                                display_dict[k] = v
                            elif isinstance(v, bool):
                                display_dict[k] = v
                            else:
                                # Check if v is a number (and not None) before calling np.isfinite
                                is_numeric = isinstance(v, (int, float))
                                display_dict[k] = float(v) if is_numeric and np.isfinite(v) else None

                        st.json(display_dict)

            else:
                st.info("R_star required in the mass table to compute P_min.")


else:
    st.info("Enter the root spectra folder and an output folder in the sidebar to begin.")
# -------------------------------------------------------------
# Population histograms (across stars saved to orbit_summary.csv)
# -------------------------------------------------------------
st.markdown("---")
st.markdown("#### Notes")
try:
    star_for_note = st.session_state.get('last_star', '')
    summary_csv_path = Path(root_folder) / "orbit_summary.csv"
    existing_note = _read_note_from_csv(summary_csv_path, star_for_note) if star_for_note else ""
    note_text = st.text_area(
        "Write a note for this system (saved to orbit_summary.csv → 'note' column)",
        value=existing_note, height=120, key="note_text_area"
    )
    also_mcmc = st.checkbox("Also update note in orbit_summary_mcmc.csv", value=True, key="cb_note_mcmc")
    if st.button("Add Note", key="btn_add_note"):
        if not star_for_note:
            st.error("No star selected.")
        else:
            update_note_in_csv(star_for_note, summary_csv_path, note_text, column="note")
            if also_mcmc:
                try:
                    mcmc_csv_path = Path(root_folder) / "orbit_summary_mcmc.csv"
                    update_note_in_csv(star_for_note, mcmc_csv_path, note_text, column="note")
                except Exception:
                    pass
            st.success("Note saved.")
except Exception as _e:
    st.warning(f"Could not update note: {_e}")
st.markdown("### Population histograms and scatter plots")

# UI to select the data source
plot_source = st.selectbox(
    "Choose data for population plots",
    ("Orbital Summary", "Orbital Summary MCMC"),
    key="pop_plot_source"
)

# Determine file name based on selection
summary_filename = "orbit_summary_mcmc.csv" if "MCMC" in plot_source else "orbit_summary.csv"
summary_path = Path(root_folder) / summary_filename

if not root_folder:
    st.info("Enter a root folder (left sidebar) to enable population plots.")
elif not summary_path.exists():
    st.info(f"The selected file ({summary_path.name}) was not found. Try generating it first.")
else:
    try:
        df_pop = pd.read_csv(summary_path)
    except Exception as _e:
        st.error(f"Could not read {summary_path.name}: {_e}")
        df_pop = pd.DataFrame()

    if not df_pop.empty:
        # 1) Scope: only binary systems?
        only_bin = st.checkbox("Filter to is_binary == 1", True, key="pop_only_bin")
        if only_bin and "is_binary" in df_pop.columns:
            df_plot = df_pop[df_pop["is_binary"] == 1].copy()
        else:
            df_plot = df_pop.copy()

        # 2) Which bin_flag(s) to include
        flags_avail = sorted(int(x) for x in pd.to_numeric(df_plot.get("bin_flag", pd.Series(dtype=float)),
                                                           errors="coerce").dropna().unique())
        if flags_avail:
            use_all_flags = st.checkbox("Use all bin_flags (1–7)", True, key="pop_all_flags")
            if use_all_flags:
                chosen_flags = flags_avail
            else:
                chosen_flags = st.multiselect("Choose bin_flag values to include", flags_avail,
                                              default=flags_avail, key="pop_flags")
            if "bin_flag" in df_plot.columns and chosen_flags:
                df_plot = df_plot[df_plot["bin_flag"].isin(chosen_flags)]
        else:
            chosen_flags = []

        # 3) Pick parameters to histogram
        col_map = {
            "P": "P (days)",
            "e": "eccentricity",
            "K1": "K1 (km/s)",
            "omega": "ω (rad)",
            "T0": "T0 (MJD)",
            "M1": "M1 (Msun)",
            "gamma": "gamma (km/s)",
            "Pmin_peri": "Pmin_peri (central)",
            "Pmin_peri_lower": "Pmin_peri (lower)",
            "Pmin_peri_upper": "Pmin_peri (upper)",
            "bin_flag": "bin_flag",
            "ls_fap": "LS FAP",
            "pdc_fap": "PDC FAP",
            "redchi":"redchi",
            "q min (M2_min/M1)":"q min (M2 min/M1)"

        }
        numeric_cols = [c for c in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[c])]
        options = [c for c in col_map if c in numeric_cols] or numeric_cols
        default_params = [c for c in ["P", "e", "K1", "M1"] if c in options] or options[:3]

        params = st.multiselect("Parameters to histogram", options=options,
                                default=default_params, format_func=lambda c: col_map.get(c, c),
                                key="pop_params")

        # --- Histogram Controls (Single, corrected block) ---
        left, right = st.columns([1, 1])
        with left:
            nbins = st.number_input("Number of bins", 5, 200, 30, step=1, key="pop_bins")
        with right:
            by_flag = st.checkbox("Plot each bin_flag separately", False, key="pop_group")

        p_tick_density = st.selectbox(
            "Log scale tick density (for P, etc.)",
            ["Decades only", "1-2-5 per decade", "1-9 per decade"],
            index=1,
            key="pop_p_tick_density"
        )

        # NEW: Let user choose which parameters get a log scale
        log_scale_params = st.multiselect(
            "Use log scale for X-axis",
            options=params,
            default=[p for p in ["P"] if p in params],
            format_func=lambda c: col_map.get(c, c),
            key="pop_log_params"
        )

        # --- Plotting Loop ---
        for p in params:
            fig, ax = plt.subplots(figsize=(7.0, 3.8))

            col = pd.to_numeric(df_plot[p], errors="coerce")
            use_log_x = p in log_scale_params

            if use_log_x:
                bad = (~col.notna()) | (col <= 0)
                if bad.any():
                    st.warning(
                        f"For '{col_map.get(p, p)}', skipping {bad.sum()} non-positive or NaN values for log scale.")
                col = col[col.notna() & (col > 0)]
            else:
                col = col.dropna()

            if col.empty:
                st.info(f"No valid data to plot for '{col_map.get(p, p)}' with current settings.")
                plt.close(fig)
                continue

            data_all = col.to_numpy()

            if use_log_x:
                xmin, xmax = float(np.min(data_all)), float(np.max(data_all))
                pad = 0.02
                xmin_padded = xmin * (1 - pad) if xmin > 0 else 1e-3
                xmax_padded = xmax * (1 + pad)
                bins = np.logspace(np.log10(xmin_padded), np.log10(xmax_padded), int(nbins) + 1)
            else:
                bins = int(nbins)

            if by_flag and "bin_flag" in df_plot.columns and chosen_flags:
                for b in chosen_flags:
                    arr = pd.to_numeric(df_plot.loc[df_plot["bin_flag"] == b, p], errors="coerce")
                    arr = arr[arr.notna() & (arr > 0)] if use_log_x else arr.dropna()
                    if not arr.empty:
                        ax.hist(arr, bins=bins, histtype="step", linewidth=1.5, label=f"flag {b}", alpha=0.9)
                ax.legend(title="bin_flag")
            else:
                ax.hist(data_all, bins=bins, histtype="bar")

            if use_log_x:
                ax.set_xscale("log")
                ax.set_xlim(left=max(1e-9, data_all.min() * 0.9), right=data_all.max() * 1.1)

                if p_tick_density == "Decades only":
                    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
                    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
                    ax.xaxis.set_major_formatter(mticker.LogFormatter())
                    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
                elif p_tick_density == "1-2-5 per decade":
                    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
                    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
                    ax.xaxis.set_minor_locator(
                        mticker.LogLocator(base=10.0, subs=np.setdiff1d(np.arange(1, 10), [1, 2, 5]) * 0.1))
                    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
                else:
                    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=np.arange(1, 10)))
                    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
                    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

                #ax.grid(True, axis="x", which="both", alpha=0.25)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            ax.set_xlabel(col_map.get(p, p))
            ax.set_ylabel("Count")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

            ax.set_title(f"Histogram: {col_map.get(p, p)}")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

st.markdown("#### Scatter explorer")

# Only show after the histogram section has built df_plot/options/col_map
if ('df_plot' not in locals()) or ('options' not in locals()) or ('col_map' not in locals()) or df_plot.empty:
    st.info("Load a valid summary file (and/or adjust filters above) to enable scatter plots.")
else:
    # Pick axes and optional color encoding
    sc_cols = st.columns([2, 2, 2, 2])
    with sc_cols[0]:
        x_param = st.selectbox("X parameter", options=options,
                               index=(options.index("P") if "P" in options else 0),
                               format_func=lambda c: col_map.get(c, c),
                               key="sc_x")
    with sc_cols[1]:
        y_param = st.selectbox("Y parameter", options=options,
                               index=(options.index("e") if "e" in options else min(1, len(options)-1)),
                               format_func=lambda c: col_map.get(c, c),
                               key="sc_y")
    with sc_cols[2]:
        color_param = st.selectbox("Color by (optional)",
                                   options=["(none)"] + options,
                                   index=(options.index("K1")+1 if "K1" in options else 0),
                                   format_func=lambda c: "(none)" if c == "(none)" else col_map.get(c, c),
                                   key="sc_c")
    with sc_cols[3]:
        show_errors = st.checkbox("Show errors", key="sc_show_errors", value=True)

    size_pts = st.slider("Point Size", min_value=1, max_value=200, value=40, step=1, key="sc_size")

    # Build data from the already-filtered df_plot
    x = pd.to_numeric(df_plot[x_param], errors="coerce").to_numpy()
    y = pd.to_numeric(df_plot[y_param], errors="coerce").to_numpy()

    xerr, yerr = None, None
    if show_errors:
        x_err_minus_col, x_err_plus_col = f"{x_param}_err_minus", f"{x_param}_err_plus"
        y_err_minus_col, y_err_plus_col = f"{y_param}_err_minus", f"{y_param}_err_plus"

        if x_err_minus_col in df_plot.columns and x_err_plus_col in df_plot.columns:
            x_err_minus = pd.to_numeric(df_plot[x_err_minus_col], errors='coerce').to_numpy()
            x_err_plus = pd.to_numeric(df_plot[x_err_plus_col], errors='coerce').to_numpy()
            xerr = np.array([x_err_minus, x_err_plus])

        if y_err_minus_col in df_plot.columns and y_err_plus_col in df_plot.columns:
            y_err_minus = pd.to_numeric(df_plot[y_err_minus_col], errors='coerce').to_numpy()
            y_err_plus = pd.to_numeric(df_plot[y_err_plus_col], errors='coerce').to_numpy()
            yerr = np.array([y_err_minus, y_err_plus])

    mask = np.isfinite(x) & np.isfinite(y)
    if x_param == "P":
        mask &= (x > 0)

    cdata = None
    if color_param != "(none)":
        c = pd.to_numeric(df_plot[color_param], errors="coerce").to_numpy()
        mask &= np.isfinite(c)
        cdata = c[mask]

    xdata, ydata = x[mask], y[mask]
    xerr_data = xerr[:, mask] if xerr is not None else None
    yerr_data = yerr[:, mask] if yerr is not None else None

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    # Determine if we have a valid color array to map
    use_color_map = (color_param != "(none)" and cdata is not None and len(cdata) > 0)

    if show_errors and (xerr_data is not None or yerr_data is not None):
        # CASE 1: User wants to show errors.
        if use_color_map:
            # Subcase A: Errors AND a color map. This requires the two-step method.
            # 1. Plot just the error bars, with no points (fmt='none').
            ax.errorbar(xdata, ydata, yerr=yerr_data, xerr=xerr_data,
                        fmt='none', capsize=3, ecolor='lightgray', elinewidth=1, zorder=1)
            # 2. Plot the colored points on top using scatter.
            sc = ax.scatter(xdata, ydata, c=cdata, s=int(size_pts), alpha=0.9, zorder=2)
        else:
            # Subcase B: Errors but NO color map. A single errorbar call is perfect.
            sc = ax.errorbar(xdata, ydata, yerr=yerr_data, xerr=xerr_data,
                             fmt='o', markersize=np.sqrt(size_pts), alpha=0.9,
                             capsize=3, ecolor='gray', elinewidth=1)
    else:
        # CASE 2: User does not want errors. A single scatter call is best.
        sc = ax.scatter(xdata, ydata,
                        c=cdata if use_color_map else None,
                        s=int(size_pts), alpha=0.9)

    # Axis scales/labels (this part was already correct)
    if x_param == "P":
        ax.set_xscale("log")
    ax.set_xlabel(col_map.get(x_param, x_param))
    ax.set_ylabel(col_map.get(y_param, y_param))
    # Colorbar
    if color_param != "(none)":
        # The colorable object is in sc.lines[0] for errorbar if color is an array
        colorable = sc.lines[0] if hasattr(sc, 'lines') else sc
        cb = fig.colorbar(colorable, ax=ax)
        cb.set_label(col_map.get(color_param, color_param))

    ax.set_title(f"{col_map.get(y_param, y_param)} vs {col_map.get(x_param, x_param)}"
                 + ("" if color_param == "(none)" else f"  (color: {col_map.get(color_param, color_param)})"))
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


st.markdown("""
---
**Tips**
- The app writes per‑star outputs under the chosen *Output folder*. You can run the fit on the saved `*_CCF_RVs.csv` later from your own scripts if you prefer.
""")
