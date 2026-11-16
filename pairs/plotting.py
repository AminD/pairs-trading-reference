"""Shared Matplotlib style so every figure in the project looks the same."""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

# Categorical colours, always used in this order (colour-blind-safe set).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]

INK = "#0b0b0b"        # titles and values
INK_SOFT = "#52514e"   # axis labels, legends
MUTED = "#898781"      # tick labels, reference lines
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def use_style() -> None:
    """Apply the project style to all subsequent Matplotlib figures."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.prop_cycle": cycler(color=SERIES),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.titlecolor": INK,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SOFT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SOFT,
            "ytick.labelcolor": INK_SOFT,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SOFT,
            "font.family": "DejaVu Sans",
        }
    )


GRAY = "#898781"  # benchmark and reference series

# Dashboard identity: a navy ink on a cool grey canvas, white panels.
NAVY = "#14213D"
CANVAS = "#F2F4F7"
PANEL = "#FFFFFF"
BODY = "#1D2939"
QUIET = "#667085"
RULE = "#D9DEE7"
DISPLAY_FONT = "'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif"
BODY_FONT = "'Hanken Grotesk', 'Segoe UI', system-ui, sans-serif"


def plotly_layout(title: str | None = None, height: int = 400, **overrides) -> dict:
    """Layout shared by every interactive Plotly chart of the dashboard."""
    layout = dict(
        title=dict(text=title, x=0.0, xanchor="left", y=0.94, yanchor="top", pad=dict(l=22),
                   font=dict(family=DISPLAY_FONT, size=17, color=NAVY)) if title else None,
        height=height,
        # The chart sits on a white panel: these margins are the panel's padding.
        margin=dict(l=22, r=22, t=88 if title else 22, b=22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY_FONT, size=13, color=QUIET),
        colorway=SERIES,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor=RULE, font=dict(family=BODY_FONT, size=12, color=BODY)),
        # Legend on its own line under the title, so the two never collide.
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0, title_text=""),
        xaxis=dict(gridcolor="#EDF0F4", linecolor=RULE, zeroline=False, automargin=True),
        yaxis=dict(gridcolor="#EDF0F4", linecolor=RULE, zeroline=False, automargin=True),
    )
    layout.update(overrides)
    return layout
