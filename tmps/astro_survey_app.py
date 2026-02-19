"""
Astro Survey Explorer — Streamlit app (BLOeM-aware)

Features
- Upload a features table (CSV/Parquet) from your local machine.
- Browse a tabular view with sorting, filtering, and column visibility controls.
- Load spectra from BLOeM directory pattern; zoom/overlay multiple spectra with Plotly.
- Load a periodogram for a selected object; zoom with Plotly (free-form path).
- Fit RV data in real-time (null model or simple SB1 circular orbit) with your own plugin
  function if available. Visualize both time-series and phase-folded RVs at a fixed period.

BLOeM directory layout used here (editable in sidebar):
- Spectra root: /Users/roeyovadia/Documents/Data/BLOeM_DR4.0_Combined
  Files are expected at: FIELD{field}/FITS/BLOeM_{field}-{object}_{epoch}_Combined.fits
- RV directory: /Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/ostars_sb1_new_list_from_coadded
  Files are: BLOeM_{field}-{object}_CCF_RVs.csv

Object IDs accepted: either BLOeM_{field}-{object} or {field}-{object}.

Run:
  streamlit run astro_survey_app.py

Dependencies (install before running):
  pip install streamlit plotly pandas numpy scipy pyarrow
  # optional: astropy for FITS support
  pip install astropy
"""
from __future__ import annotations

import io
import importlib.util
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

import streamlit as st

# -----------------------------
# Config & helpers
# -----------------------------

@dataclass
class AppConfig:
    spectra_dir: Path | None = None
    periodogram_dir: Path | None = None
    rv_dir: Path | None = None
    object_id_col: str = "object_id"  # Fallbacks: 'star_id', 'name', 'id'
    plugin_path: Optional[Path] = None


@st.cache_data(show_spinner=False)
def _read_table(uploaded: bytes, name: str) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(io.BytesIO(uploaded))
    elif suffix in {".csv", ".txt", ".dat"}:
        # Try comma first, fallback to whitespace
        try:
            return pd.read_csv(io.BytesIO(uploaded))
        except Exception:
            return pd.read_csv(io.BytesIO(uploaded), delim_whitespace=True)
    else:
        # Try CSV fallback
        return pd.read_csv(io.BytesIO(uploaded))


def _load_csv_like(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() in {".csv"}:
            return pd.read_csv(path, sep=None, engine="python")
        if path.suffix.lower() in {".txt", ".dat"}:
            # best-effort for space separated
            return pd.read_csv(path, delim_whitespace=True, header=None)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
    except Exception:
        return None
    return None


def _load_spectrum_any(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # CSV/TXT/DAT
    df = _load_csv_like(path)
    if df is not None:
        cols = [c.lower() for c in df.columns]
        # Try known names first
        w_candidates = ["wavelength", "wave", "lambda", "wl", "col0", "col1", 0]
        f_candidates = ["flux", "flx", "f", "col1", "col2", 1]
        wcol = next((c for c in w_candidates if (c in df.columns) or (isinstance(c, int) and c < len(df.columns))), None)
        fcol = next((c for c in f_candidates if (c in df.columns) or (isinstance(c, int) and c < len(df.columns))), None)
        if isinstance(wcol, int):
            wcol = df.columns[wcol]
        if isinstance(fcol, int):
            fcol = df.columns[fcol]
        if wcol is None or fcol is None:
            # Fallback: first two numeric columns
            numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if len(numeric_cols) >= 2:
                wcol, fcol = numeric_cols[:2]
            else:
                return None
        wl = pd.to_numeric(df[wcol], errors="coerce").to_numpy()
        fx = pd.to_numeric(df[fcol], errors="coerce").to_numpy()
        m = np.isfinite(wl) & np.isfinite(fx)
        if m.any():
            return wl[m], fx[m]
        return None
    # FITS (optional)
    if path.suffix.lower() in {".fits", ".fit", ".fts"}:
        try:
            from astropy.io import fits  # optional dependency
            with fits.open(path) as hdul:
                # Try table HDU first
                for hdu in hdul:
                    if hasattr(hdu, "data") and hdu.data is not None:
                        data = hdu.data
                        cols = [c.lower() for c in getattr(data, "names", []) or []]
                        if cols:
                            wname = next((c for c in ["wavelength", "wave", "lambda", "wl","WAVELENGTH"] if c in cols), None)
                            fname = next((c for c in ["flux", "flx", "f","sci_norm"] if c in cols), None)
                            if wname and fname:
                                wl = np.asarray(data[wname]).astype(float)
                                fx = np.asarray(data[fname]).astype(float)
                                m = np.isfinite(wl) & np.isfinite(fx)
                                if m.any():
                                    return wl[m], fx[m]
                # Primary image HDU as last resort (pixel index as wavelength)
                img = hdul[0].data
                if img is not None:
                    fx = np.asarray(img).astype(float).ravel()
                    wl = np.arange(len(fx), dtype=float)
                    return wl, fx
        except Exception:
            return None
    return None


# -----------------------------
# BLOeM ID parsing & path discovery
# -----------------------------
_bloem_id_re = re.compile(r"^(?:BLOeM_)?(?P<field>\d+)-(?:OBJ)?(?P<object>\d+)$", re.IGNORECASE)

def _parse_object_id(oid: str) -> Optional[Tuple[str, str, str]]:
    """Return (field, object, canonical_name) for inputs like 'BLOeM_12-345' or '12-345'."""
    if not oid:
        return None
    oid = str(oid).strip()
    m = _bloem_id_re.match(oid)
    if not m:
        return None
    field = m.group("field")
    obj = m.group("object")
    canonical = f"BLOeM_{field}-{obj}"
    return field, obj, canonical


def _find_spectra_paths(cfg: AppConfig, oid: str) -> List[Path]:
    files: List[Path] = []
    if not cfg.spectra_dir:
        return files
    parsed = _parse_object_id(oid)
    if not parsed:
        return files
    field, obj, _ = parsed
    # Pattern: ROOT/FIELD{field}/FITS/BLOeM_{field}-{object}_{epoch}_Combined.fits
    field_dir = cfg.spectra_dir / f"FIELD{field}" / "FITS"
    if field_dir.exists():
        files.extend(sorted(field_dir.glob(f"BLOeM_{field}-{obj}_*_Combined.fits")))
    return files


def _find_rv_paths(cfg: AppConfig, oid: str) -> List[Path]:
    files: List[Path] = []
    if not cfg.rv_dir:
        return files
    parsed = _parse_object_id(oid)
    if not parsed:
        return files
    field, obj, _ = parsed
    # Exact file: BLOeM_{field}-{object}_CCF_RVs.csv (no epoch)
    target = cfg.rv_dir / f"BLOeM_{field}-{obj}_CCF_RVs.csv"
    if target.exists():
        files.append(target)
    return files


# -----------------------------
# RV model & fitting (built-in fallback)
# -----------------------------

# -----------------------------
# RV model & fitting (built-in fallback)
# -----------------------------

def _sb1_circular_model(t, gamma, K, phi, P):
    # v(t) = gamma + K * sin(2π (t / P) + phi)
    return gamma + K * np.sin(2.0 * np.pi * (t / P) + phi)


def _fit_rv_builtin(times, rvs, rv_errs, period, model):
    times = np.asarray(times, float)
    rvs = np.asarray(rvs, float)
    if rv_errs is None or len(rv_errs) != len(rvs):
        rv_errs = np.ones_like(rvs)
    sigma = np.asarray(rv_errs, float)

    if model == "null":
        # Weighted mean as constant model
        w = 1.0 / np.clip(sigma, 1e-12, None) ** 2
        gamma = np.sum(w * rvs) / np.sum(w)
        rv_model = np.full_like(rvs, gamma)
        # Uncertainty on mean
        gamma_err = (1.0 / np.sqrt(np.sum(w))) if len(rvs) > 1 else np.nan
        return {
            "model": "null",
            "params": {"gamma": float(gamma)},
            "params_err": {"gamma": float(gamma_err)},
            "rv_model": rv_model,
        }

    # SB1 circular: fit gamma, K, phi at fixed P
    P = float(period)
    def f(t, gamma, K, phi):
        return _sb1_circular_model(t, gamma, K, phi, P)

    p0 = [np.nanmedian(rvs), (np.nanmax(rvs) - np.nanmin(rvs)) / 2.0, 0.0]
    try:
        popt, pcov = curve_fit(f, times, rvs, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=20000)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan, np.nan]
    except Exception:
        popt = p0
        perr = [np.nan, np.nan, np.nan]

    rv_model = f(times, *popt)
    return {
        "model": "sb1",
        "params": {"gamma": float(popt[0]), "K": float(popt[1]), "phi": float(popt[2]), "e": 0.0},
        "params_err": {"gamma": float(perr[0]), "K": float(perr[1]), "phi": float(perr[2]), "e": 0.0},
        "rv_model": rv_model,
    }


def _maybe_load_plugin(plugin_path: Optional[Path]):
    if not plugin_path:
        return None
    try:
        spec = importlib.util.spec_from_file_location("rv_plugin", str(plugin_path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "fit_rv"):
                return getattr(mod, "fit_rv")
    except Exception as e:
        st.warning(f"Failed to load plugin fitter: {e}")
    return None


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Astro Survey Explorer (BLOeM)", layout="wide")
st.title("🔭 Astro Survey Explorer — BLOeM")

with st.sidebar:
    st.header("Data inputs")
    features_file = st.file_uploader("Features table (CSV/Parquet)", type=["csv", "parquet", "pq", "txt", "dat"])
    df_features: Optional[pd.DataFrame] = None
    if features_file is not None:
        try:
            df_features = _read_table(features_file.getvalue(), features_file.name)
        except Exception as e:
            st.error(f"Could not read features table: {e}")

    st.divider()
    st.header("Directories")
    spectra_dir_str = st.text_input(
        "Spectra root (BLOeM_DR4.0_Combined)",
        value="/Users/roeyovadia/Documents/Data/BLOeM_DR4.0_Combined",
    )
    periodogram_dir_str = st.text_input("Periodogram directory (optional)", value="periodograms")
    rv_dir_str = st.text_input(
        "RV directory",
        value="/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/ostars_sb1_new_list_from_coadded",
    )

    object_id_col = st.text_input("Object ID column name", value="star_name")
    plugin_path_str = st.text_input("Plugin fitter path (optional .py)", value="")

    cfg = AppConfig(
        spectra_dir=Path(spectra_dir_str) if spectra_dir_str else None,
        periodogram_dir=Path(periodogram_dir_str) if periodogram_dir_str else None,
        rv_dir=Path(rv_dir_str) if rv_dir_str else None,
        object_id_col=object_id_col.strip() or "object_id",
        plugin_path=Path(plugin_path_str) if plugin_path_str else None,
    )

# Prepare object list
object_ids: List[str] = []
if df_features is not None:
    col_candidates = [cfg.object_id_col] + [c for c in ["star_id", "name", "id"] if c in df_features.columns]
    oid_col = next((c for c in col_candidates if c in df_features.columns), None)
    if oid_col is None:
        st.warning("Could not find an object ID column; please set it in the sidebar.")
    else:
        object_ids = list(pd.unique(df_features[oid_col].astype(str)))

# Tabs
_tab1, _tab2, _tab3, _tab4, _tab5 = st.tabs([
    "📋 Table",
    "📈 Spectra",
    "📊 Periodogram",
    "🌀 RV Fit",
    "⚙️ Settings",
])

# -----------------------------
# Tab 1: Table
# -----------------------------
with _tab1:
    st.subheader("Features table")
    if df_features is None:
        st.info("Upload a features table in the sidebar to begin.")
    else:
        # Column visibility
        cols = list(df_features.columns)
        default_cols = cols[: min(12, len(cols))]
        visible_cols = st.multiselect("Columns to show", options=cols, default=default_cols)
        # Simple text filter across string columns
        search = st.text_input("Row filter (substring in any string column)", value="")
        df_view = df_features.copy()
        if search:
            str_cols = [c for c in df_view.columns if df_view[c].dtype == "object"]
            if str_cols:
                mask = np.zeros(len(df_view), dtype=bool)
                for c in str_cols:
                    mask |= df_view[c].astype(str).str.contains(search, case=False, na=False)
                df_view = df_view[mask]
        if visible_cols:
            df_view = df_view[visible_cols]
        st.dataframe(df_view, use_container_width=True, hide_index=True)

# -----------------------------
# Tab 2: Spectra
# -----------------------------
with _tab2:
    st.subheader("Spectra viewer")
    if not object_ids:
        st.info("No objects found. Upload a features table first to populate the list.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            oid = st.selectbox("Object", object_ids)
        with c2:
            overlay_n = st.number_input("How many spectra to layer (1-6)", 1, 6, value=1)
        spectra_paths = _find_spectra_paths(cfg, oid)
        if not spectra_paths:
            st.warning("No spectra found for this object in the configured BLOeM root.")
        else:
            chosen = st.multiselect(
                "Select spectra to plot",
                options=[str(p.name) for p in spectra_paths],
                default=[p.name for p in spectra_paths[:overlay_n]],
            )
            # Build traces and collect flux for default limits
            series = []
            for name in chosen:
                path = next(p for p in spectra_paths if p.name == name)
                data = _load_spectrum_any(path)
                if data is None:
                    st.warning(f"Failed to read spectrum: {path.name}")
                    continue
                wl, fx = data
                series.append((wl, fx, path.stem))

            fig = go.Figure()
            for wl, fx, label in series:
                fig.add_trace(go.Scatter(x=wl, y=fx, mode="lines", name=label))

            # Y-axis controls
            y_all = np.concatenate([fx for _, fx, _ in series]) if series else np.array([])
            with st.expander("Y-axis controls", expanded=False):
                auto_y = st.checkbox("Auto y-scale", value=True, key=f"spec_auto_y_{oid}")
                if not auto_y and y_all.size:
                    ymin_default = float(np.nanpercentile(y_all, 1))
                    ymax_default = float(np.nanpercentile(y_all, 99))
                    cmin, cmax = st.columns(2)
                    with cmin:
                        y_min = st.number_input("Y min", value=ymin_default, format="%.6g")
                    with cmax:
                        y_max = st.number_input("Y max", value=ymax_default, format="%.6g")
                    if y_max <= y_min:
                        st.warning("Y max must be greater than Y min.")
                        y_min, y_max = ymin_default, ymax_default
                else:
                    y_min = y_max = None

            fig.update_layout(
                xaxis_title="Wavelength",
                yaxis_title="Flux",
                hovermode="x unified",
            )
            if y_min is not None and y_max is not None:
                fig.update_yaxes(range=[y_min, y_max])
            else:
                fig.update_yaxes(autorange=True)
            fig.update_xaxes(rangeslider=dict(visible=True))
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab 3: Periodogram
# -----------------------------
with _tab3:
    st.subheader("Periodogram viewer")
    if not object_ids:
        st.info("No objects found. Upload a features table first to populate the list.")
    else:
        oid = st.selectbox("Object", object_ids, key="pg_oid")
        # Free-form: allow user to point to their own periodogram folder structure
        per_paths_csv = []
        if cfg.periodogram_dir and cfg.periodogram_dir.exists():
            # Support both BLOeM-style file prefixes and generic files containing the object id
            field_obj = _parse_object_id(oid)
            if field_obj:
                field, obj, canon = field_obj
                patt1 = list(cfg.periodogram_dir.glob(f"{canon}*.*"))
                patt2 = list(cfg.periodogram_dir.glob(f"*{field}-{obj}*.*"))
                per_paths_csv = sorted({*patt1, *patt2})
        if not per_paths_csv:
            st.warning("No periodogram file found for this object (update 'Periodogram directory').")
        else:
            path = st.selectbox("Choose file", [str(p.name) for p in per_paths_csv])
            p = next(pp for pp in per_paths_csv if pp.name == path)
            dfp = _load_csv_like(p)
            if dfp is None:
                st.error("Could not read periodogram file.")
            else:
                # Detect columns
                lower = [str(c).lower() for c in dfp.columns]
                if "period" in lower:
                    xcol = dfp.columns[lower.index("period")]
                    x_label = "Period"
                elif "frequency" in lower:
                    xcol = dfp.columns[lower.index("frequency")]
                    x_label = "Frequency"
                else:
                    xcol = dfp.columns[0]
                    x_label = str(xcol)
                ycol = dfp.columns[1] if len(dfp.columns) > 1 else dfp.columns[0]
                if "power" in lower:
                    ycol = dfp.columns[lower.index("power")]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dfp[xcol], y=dfp[ycol], mode="lines", name=p.stem))
                fig.update_layout(xaxis_title=x_label, yaxis_title="Power", hovermode="x unified")
                fig.update_xaxes(rangeslider=dict(visible=True))
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab 4: RV fit
# -----------------------------
with _tab4:
    st.subheader("Radial velocity fitting")
    if not object_ids:
        st.info("No objects found. Upload a features table first to populate the list.")
    else:
        fitcol1, fitcol2, fitcol3 = st.columns([2, 1, 1])
        with fitcol1:
            oid = st.selectbox("Object", object_ids, key="rv_oid")
        with fitcol2:
            model = st.radio("Model", options=["null", "sb1"], horizontal=True)
        with fitcol3:
            period = st.number_input("Period (days)", min_value=0.0001, value=1.0, step=0.0001, format="%.6f")

        rv_paths = _find_rv_paths(cfg, oid)
        if not rv_paths:
            st.warning("No RV file found for this object in the configured RV directory.")
        else:
            path = st.selectbox("Choose RV file", [str(p.name) for p in rv_paths])
            p = next(pp for pp in rv_paths if pp.name == path)
            dfrv = _load_csv_like(p)
            if dfrv is None:
                st.error("Could not read RV file.")
            else:
                lower = [str(c).lower() for c in dfrv.columns]
                try:
                    tcol = dfrv.columns[lower.index("mjd")]
                    rvcol = dfrv.columns[lower.index("mean rv")]
                    errcol = dfrv.columns[lower.index("mean rvsig")]
                except ValueError:
                    st.error("RV file must have columns: MJD, RV, RV_ERR")
                    tcol = rvcol = errcol = None

                if tcol is not None:
                    times = pd.to_numeric(dfrv[tcol], errors="coerce").to_numpy()
                    rvs = pd.to_numeric(dfrv[rvcol], errors="coerce").to_numpy()
                    rv_errs = pd.to_numeric(dfrv[errcol], errors="coerce").to_numpy()
                    m = np.isfinite(times) & np.isfinite(rvs) & np.isfinite(rv_errs)
                    times, rvs, rv_errs = times[m], rvs[m], rv_errs[m]

                    # Choose fitter: plugin or builtin
                    fitter = _maybe_load_plugin(cfg.plugin_path)
                    if fitter is None:
                        result = _fit_rv_builtin(times, rvs, rv_errs, period, model)
                    else:
                        try:
                            result = fitter(times, rvs, rv_errs, period, model)
                        except Exception as e:
                            st.warning(f"Plugin fitter failed: {e}. Falling back to builtin.")
                            result = _fit_rv_builtin(times, rvs, rv_errs, period, model)

                    # Time-series plot
                    fig_ts = go.Figure()
                    fig_ts.add_trace(
                        go.Scatter(x=times, y=rvs, mode="markers", error_y=dict(type="data", array=rv_errs, visible=True), name="RV")
                    )
                    fig_ts.add_trace(go.Scatter(x=times, y=result["rv_model"], mode="lines", name="Model"))
                    fig_ts.update_layout(xaxis_title="MJD", yaxis_title="RV", hovermode="x")
                    fig_ts.update_xaxes(rangeslider=dict(visible=True))

                    # Phase-folded plot
                    phase = ((times % period) / period)
                    order = np.argsort(phase)
                    model_sorted = result["rv_model"][order]
                    fig_ph = go.Figure()
                    fig_ph.add_trace(
                        go.Scatter(
                            x=phase, y=rvs, mode="markers",
                            error_y=dict(type="data", array=rv_errs, visible=True), name="RV"
                        )
                    )
                    fig_ph.add_trace(go.Scatter(x=phase[order], y=model_sorted, mode="lines", name="Model"))
                    fig_ph.update_layout(xaxis_title="Phase", yaxis_title="RV", hovermode="x")

                    # --- Residuals (to mirror print_lmfit_result) ---
                    resid_ts = rvs - result["rv_model"]
                    resid_ph = rvs - np.interp(phase, phase[order], model_sorted)

                    fig_resid_ts = go.Figure()
                    fig_resid_ts.add_trace(
                        go.Scatter(x=times, y=resid_ts, mode="markers",
                                    error_y=dict(type="data", array=rv_errs, visible=True), name="O−C")
                    )
                    fig_resid_ts.add_hline(y=0, line_width=1)
                    fig_resid_ts.update_layout(xaxis_title="MJD", yaxis_title="O−C", hovermode="x")
                    fig_resid_ts.update_xaxes(rangeslider=dict(visible=True))

                    fig_resid_ph = go.Figure()
                    fig_resid_ph.add_trace(
                        go.Scatter(x=phase, y=resid_ph, mode="markers",
                                    error_y=dict(type="data", array=rv_errs, visible=True), name="O−C")
                    )
                    fig_resid_ph.add_hline(y=0, line_width=1)
                    fig_resid_ph.update_layout(xaxis_title="Phase", yaxis_title="O−C", hovermode="x")

                    # --- Stats like lmfit report: chi^2, redchi, AIC, BIC ---
                    k = 1 if model == "null" else 3  # gamma | gamma,K,phi
                    n = int(len(rvs))
                    sigma = np.asarray(rv_errs, float)
                    chisq = float(np.sum(((rvs - result["rv_model"]) / sigma) ** 2)) if n else np.nan
                    redchi = float(chisq / max(n - k, 1)) if np.isfinite(chisq) else np.nan
                    loglike_term = float(np.sum(np.log(2 * np.pi * sigma ** 2))) if n else np.nan
                    AIC = float(2 * k + chisq + loglike_term) if np.isfinite(chisq) else np.nan
                    BIC = float(k * np.log(max(n, 1)) + chisq + loglike_term) if np.isfinite(chisq) else np.nan

                    # Layout: 2x2: (RV vs time | RV vs phase) then (resid vs time | resid vs phase)
                    c11, c12 = st.columns(2)
                    with c11:
                        st.plotly_chart(fig_ts, use_container_width=True)
                    with c12:
                        st.plotly_chart(fig_ph, use_container_width=True)
                    c21, c22 = st.columns(2)
                    with c21:
                        st.plotly_chart(fig_resid_ts, use_container_width=True)
                    with c22:
                        st.plotly_chart(fig_resid_ph, use_container_width=True)

                    # Results tables
                    st.markdown("### Fit results")
                    params = result.get("params", {})
                    perrs = result.get("params_err", {})
                    rows = []
                    for kname, v in params.items():
                        rows.append({"Parameter": kname, "Value": v, "Uncertainty": perrs.get(kname, np.nan)})
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                    st.markdown("### Fit statistics")
                    st.dataframe(
                        pd.DataFrame([
                            {"n": n, "k": (1 if model == "null" else 3), "chi2": chisq, "redchi": redchi, "AIC": AIC, "BIC": BIC}
                        ]),
                        hide_index=True,
                        use_container_width=True,
                    )

# -----------------------------
# Tab 5: Settings & Help
# -----------------------------
with _tab5:
    st.subheader("Settings & Help")
    st.markdown(
        """
        **BLOeM paths**
        - Spectra: `<spectra_root>/FIELD{field}/FITS/BLOeM_{field}-{object}_{epoch}_Combined.fits`.
        - RV: `<rv_dir>/BLOeM_{field}-{object}_CCF_RVs.csv`.

        **Features table**
        - Must include an object ID column (default `object_id`). Accepts either `BLOeM_{field}-{object}` or `{field}-{object}`.

        **Spectra viewer**
        - Select and overlay multiple epochs for the same object; zoom via the range slider.

        **Periodogram viewer**
        - Point the sidebar to any folder. The app searches files that contain either the canonical `BLOeM_{field}-{object}` or `{field}-{object}` in their names.

        **RV fitting**
        - Built-in model fits a circular SB1 at fixed period (γ, K, φ). Use the plugin hook for full Keplerian models with eccentricity.

        **Performance tips**
        - Parquet is faster for large tables; FITS reading requires `astropy`.
        """
    )

    st.caption("Made with Streamlit, Plotly, NumPy/Pandas, and SciPy.")
