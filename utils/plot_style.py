"""
Publication-quality plot styling for Plotly figures.

Presets
-------
INTERACTIVE : screen-optimised (current default behaviour)
PAPER       : single-column journal figure (~88 mm / ~3.3")
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PlotStyle:
    # --- dimensions (Plotly layout px, before scale) ---
    width: int
    height: int
    scale: int

    # --- fonts (Plotly pt) ---
    font_family: str
    font_base: int
    font_title: int
    font_axis_title: int
    font_tick: int
    font_legend: int
    font_annotation: int

    # --- margins ---
    margin_l: int
    margin_r: int
    margin_t: int
    margin_b: int

    # --- legend ---
    legend_orientation: str          # "h" or "v"
    legend_x: float
    legend_y: float
    legend_xanchor: str
    legend_yanchor: str
    legend_bgcolor: str

    # --- line widths ---
    line_width_main: float           # primary data trace
    line_width_model: float          # best-fit model
    line_width_reference: float      # FAL, gamma, zero-line

    # --- markers ---
    marker_size: int

    # --- colors ---
    color_data_primary: str          # time-series / periodogram data
    color_data_secondary: str        # phase-folded data
    color_jitter: str
    color_model: str
    color_residuals: str
    color_gamma: str
    color_fal: Tuple[str, ...]       # one per FAL level
    color_best_period: str

    # --- grid & axes frame ---
    show_grid: bool
    grid_color: str                  # e.g. "rgba(200,200,200,0.5)"
    show_axis_frame: bool            # draw lines along all four edges

    # --- behaviour ---
    show_title: bool
    template: str
    export_pdf: bool                 # also write a vector PDF


# ── presets ──────────────────────────────────────────────────────────────

INTERACTIVE = PlotStyle(
    width=900, height=550, scale=4,
    font_family="Arial",
    font_base=18, font_title=22, font_axis_title=20,
    font_tick=16, font_legend=16, font_annotation=14,
    margin_l=80, margin_r=20, margin_t=80, margin_b=80,
    legend_orientation="h",
    legend_x=0.5, legend_y=-0.18,
    legend_xanchor="center", legend_yanchor="top",
    legend_bgcolor="rgba(255,255,255,0)",
    line_width_main=2.0, line_width_model=3.0,
    line_width_reference=1.5,
    marker_size=8,
    color_data_primary="royalblue",
    color_data_secondary="crimson",
    color_jitter="orange",
    color_model="blue",
    color_residuals="gray",
    color_gamma="gray",
    color_fal=("red", "goldenrod", "green"),
    color_best_period="dodgerblue",
    show_grid=True,
    grid_color="rgba(200,200,200,0.3)",
    show_axis_frame=False,
    show_title=True,
    template="plotly_white",
    export_pdf=False,
)

PAPER = PlotStyle(
    width=900, height=550, scale=4,
    font_family="Arial",
    font_base=18, font_title=22, font_axis_title=20,
    font_tick=16, font_legend=16, font_annotation=14,
    margin_l=80, margin_r=20, margin_t=40, margin_b=80,
    legend_orientation="h",
    legend_x=0.5, legend_y=-0.18,
    legend_xanchor="center", legend_yanchor="top",
    legend_bgcolor="rgba(255,255,255,0)",
    line_width_main=2.0, line_width_model=3.0,
    line_width_reference=1.5,
    marker_size=8,
    color_data_primary="royalblue",
    color_data_secondary="crimson",
    color_jitter="orange",
    color_model="blue",
    color_residuals="gray",
    color_gamma="gray",
    color_fal=("red", "goldenrod", "green"),
    color_best_period="dodgerblue",
    show_grid=True,
    grid_color="rgba(200,200,200,0.5)",
    show_axis_frame=True,
    show_title=False,
    template="plotly_white",
    export_pdf=True,
)

_PRESETS = {
    "interactive": INTERACTIVE,
    "paper": PAPER,
}


def get_style(style):
    """Resolve *style* to a `PlotStyle` instance.

    Accepts a preset name string (``"interactive"``, ``"paper"``) or a
    `PlotStyle` instance directly.
    """
    if isinstance(style, PlotStyle):
        return style
    return _PRESETS[style.lower()]


def apply_style_to_layout(fig, s: PlotStyle, **overrides):
    """Apply common layout parameters from *s* to a Plotly figure."""
    axis_common = dict(
        showgrid=s.show_grid,
        gridcolor=s.grid_color,
    )
    if s.show_axis_frame:
        axis_common.update(
            showline=True, linewidth=1, linecolor="black",
            mirror=True,
        )

    kw = dict(
        template=s.template,
        font=dict(family=s.font_family, size=s.font_base),
        margin=dict(l=s.margin_l, r=s.margin_r, t=s.margin_t, b=s.margin_b),
        legend=dict(
            orientation=s.legend_orientation,
            x=s.legend_x, y=s.legend_y,
            xanchor=s.legend_xanchor, yanchor=s.legend_yanchor,
            font=dict(size=s.font_legend),
            bgcolor=s.legend_bgcolor,
        ),
        xaxis=axis_common,
        yaxis=axis_common,
    )
    kw.update(overrides)
    fig.update_layout(**kw)
