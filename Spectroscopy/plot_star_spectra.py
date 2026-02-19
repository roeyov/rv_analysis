#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re, webbrowser
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from astropy.io import fits
import plotly.graph_objects as go

EPOCH_REGEX = re.compile(r'_(\d+)(?:[_\.])')

def find_candidate_files(root: Path, star_name: str) -> List[Path]:
    star_lower = star_name.lower()
    return sorted([p for p in root.rglob("*.fits") if star_lower in p.name.lower()],
                  key=lambda p: p.name.lower())

def parse_epoch_from_name(fname: str) -> Optional[int]:
    m = EPOCH_REGEX.search(fname)
    if not m: return None
    try: return int(m.group(1))
    except ValueError: return None

def read_fits_spectrum(fp: Path, x_name: str, y_name: str, ext: int):
    with fits.open(fp, ignore_missing_simple=True) as hdul:
        hdr0 = dict(hdul[0].header)
        hdu = hdul[ext]
        hdr_ext = dict(hdu.header)
        data = hdu.data
        if hasattr(data, "columns"):
            x = np.array(data[x_name], dtype=float).flatten()
            y = np.array(data[y_name], dtype=float).flatten()
        else:
            naxis1 = data.shape[-1]
            crval1 = hdu.header.get("CRVAL1")
            cdelt1 = hdu.header.get("CDELT1")
            crpix1 = hdu.header.get("CRPIX1", 1.0)
            if crval1 is None or cdelt1 is None:
                raise ValueError(f"{fp.name}: missing CRVAL1/CDELT1 for wavelength axis.")
            pix = np.arange(1, naxis1 + 1, dtype=float)
            x = crval1 + (pix - crpix1) * cdelt1
            y = np.array(data, dtype=float).flatten()
        return x, y, {"primary_header": hdr0, "ext_header": hdr_ext}

def extract_mjd(headers: dict, preferred_key: str) -> Optional[float]:
    for k in [preferred_key, "MJD-OBS", "MJD", "HJD", "BMJD", "BJD"]:
        v = headers["primary_header"].get(k, headers["ext_header"].get(k))
        if v is not None:
            try: return float(v)
            except: pass
    return None

def load_template(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    sfx = path.suffix.lower()
    if sfx in {".fits", ".fit", ".fts"}:
        with fits.open(path, ignore_missing_simple=True) as hdul:
            pick = None
            for h in hdul[1:]:
                if hasattr(h.data, "columns") and h.data is not None and len(h.data.columns) >= 2:
                    pick = h; break
            if pick is None: raise ValueError("Template FITS must have a table HDU with ≥2 columns.")
            return (np.array(pick.data.field(0), float).flatten(),
                    np.array(pick.data.field(1), float).flatten())
    arr = np.loadtxt(path, ndmin=2)
    if arr.shape[1] < 2: raise ValueError("Template text file must have ≥2 columns.")
    return arr[:, 0], arr[:, 1]

# ---------- stable per-epoch color (default mode) ----------
def epoch_to_hex_color(epoch: Optional[int]) -> str:
    if epoch is None: return "#808080"
    hue = (epoch * 137) % 360
    s, v = 0.70, 0.85
    c = v * s; h_ = hue / 60.0; x = c * (1 - abs(h_ % 2 - 1))
    if   0 <= h_ < 1: r,g,b = c,x,0
    elif 1 <= h_ < 2: r,g,b = x,c,0
    elif 2 <= h_ < 3: r,g,b = 0,c,x
    elif 3 <= h_ < 4: r,g,b = 0,x,c
    elif 4 <= h_ < 5: r,g,b = x,0,c
    else:             r,g,b = c,0,x
    m = v - c; r,g,b = r+m, g+m, b+m
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))


def main():
    ap = argparse.ArgumentParser(description="Interactive multi-epoch spectra plotter")
    ap.add_argument("directory", type=str)
    ap.add_argument("star_name", type=str)
    ap.add_argument("--x-name", default="WAVELENGTH")
    ap.add_argument("--y-name", default="SCI_NORM")
    ap.add_argument("--ext", type=int, default=1)
    ap.add_argument("--time-key", default="MJD")
    ap.add_argument("--cadence-ms", type=int, default=200, help="Animation cadence per epoch (ms)")
    ap.add_argument("--template", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    root = Path(args.directory).expanduser().resolve()
    files = find_candidate_files(root, args.star_name)
    if not files:
        raise SystemExit(f"[ERR] No FITS files containing '{args.star_name}' under {root}")

    spectra = []
    for fp in files:
        try:
            x, y, headers = read_fits_spectrum(fp, args.x_name, args.y_name, args.ext)
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}");
            continue
        epoch = parse_epoch_from_name(fp.name)
        mjd = extract_mjd(headers, args.time_key)
        spectra.append({"path": fp, "epoch": epoch, "epoch_str": str(epoch) if epoch is not None else "NA",
                        "mjd": mjd, "x": x, "y": y})

    if not spectra: raise SystemExit(f"[ERR] No valid spectra for '{args.star_name}'.")

    # Sort spectra by epoch (or filename if epoch is missing)
    spectra.sort(key=lambda d: (float('inf') if d["epoch"] is None else d["epoch"], d["path"].name))

    fig = go.Figure()
    original_colors = []

    # --- traces: show ALL by default ---
    for s in spectra:
        color = epoch_to_hex_color(s["epoch"])
        original_colors.append(color)
        customdata = np.column_stack([
            np.full_like(s["x"], s["mjd"] if s["mjd"] is not None else np.nan, dtype=float),
            np.full_like(s["x"], s["epoch"] if s["epoch"] is not None else np.nan, dtype=float),
        ])
        fig.add_trace(go.Scatter(
            x=s["x"], y=s["y"], mode="lines",
            name=f"epoch {s['epoch_str']}",
            line=dict(width=2, color=color),
            customdata=customdata,
            hovertemplate=(
                "λ=%{x:.2f}<br>"
                "Flux=%{y:.4g}<br>"
                "MJD=%{customdata[0]:.5f}<br>"
                "Epoch=%{customdata[1]:.0f}"
                "<extra></extra>"
            ),
        ))

    template_index = None
    if args.template:
        try:
            tx, ty = load_template(Path(args.template).expanduser().resolve())
            template_index = len(fig.data)
            fig.add_trace(go.Scatter(
                x=tx, y=ty, mode="lines", name="Template",
                line=dict(width=3, dash="dash", color="#000000"),
                hovertemplate="Template<extra></extra>"
            ))
            original_colors.append("#000000")
        except Exception as e:
            print(f"[WARN] Failed to load template: {e}")

    n_traces = len(fig.data)
    n_spectra = len(spectra)
    anim_color = "#1f77b4"

    # -------- Animation frames: STRICT one-by-one visibility --------
    frames = []
    for i in range(n_spectra):
        # We define the visibility for ALL traces in the plot for every single frame
        vis = [False] * n_traces
        vis[i] = True  # Only show the current spectrum

        # Optional: If you want the template to always be visible in the background:
        # if template_index is not None: vis[template_index] = True

        frames.append(go.Frame(
            name=f"epoch {spectra[i]['epoch_str']}",
            data=[go.Scatter(visible=vis[j]) for j in range(n_traces)]
        ))
    fig.frames = frames

    # --- helper lists for buttons ---
    all_true = [True] * n_traces
    # Initial state for animation mode: only the first spectrum visible
    anim_start_vis = [False] * n_traces
    anim_start_vis[0] = True

    # -------- UI menus --------
    menu_default = dict(
        type="buttons", direction="left", x=0.0, y=1.15, xanchor="left", yanchor="top",
        showactive=False,
        buttons=[
            dict(label="Remove all", method="update", args=[{"visible": ["legendonly"] * n_traces}, {}]),
            dict(label="Reset all", method="update", args=[{"visible": all_true}, {}]),
            dict(
                label="Enter animation",
                method="update",
                args=[
                    {
                        "visible": anim_start_vis,  # Show only the first one immediately
                        "line.color": [anim_color] * n_spectra + (
                            [fig.data[template_index].line.color] if template_index is not None else [])
                    },
                    {
                        "updatemenus[0].visible": False,
                        "updatemenus[1].visible": True,
                        "sliders[0].visible": True,
                    }
                ]
            ),
        ]
    )

    play_args = [None, {"frame": {"duration": args.cadence_ms, "redraw": True}, "fromcurrent": False,
                        "transition": {"duration": 0}}]
    pause_args = [None, {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]

    menu_anim = dict(
        type="buttons", direction="left", x=0.0, y=1.15, xanchor="left", yanchor="top",
        visible=False, showactive=False,
        buttons=[
            dict(label="▶ Play", method="animate", args=play_args),
            dict(label="⏸ Pause", method="animate", args=pause_args),
            dict(
                label="Exit animation",
                method="update",
                args=[
                    {"visible": all_true, "line.color": original_colors},
                    {
                        "updatemenus[0].visible": True,
                        "updatemenus[1].visible": False,
                        "sliders[0].visible": False,
                    }
                ]
            ),
        ]
    )

    slider = dict(
        visible=False, active=0, currentvalue={"prefix": "Epoch: "},
        steps=[{"label": f.name, "method": "animate",
                "args": [[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0}}]}
               for f in fig.frames]
    )

    fig.update_layout(
        title=f"{args.star_name} — multi-epoch spectra",
        xaxis_title="Wavelength", yaxis_title="Flux",
        hovermode="closest",
        legend_title="Epochs",
        uirevision="keep",
        updatemenus=[menu_default, menu_anim],
        sliders=[slider],
        margin=dict(l=60, r=20, t=90, b=60)
    )

    out_path = Path(args.out) if args.out else Path(f"{args.star_name}_spectra.html")
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True, auto_play=False)
    print(f"[OK] Wrote interactive plot to: {out_path.resolve()}")
    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())

if __name__ == "__main__":
    main()
