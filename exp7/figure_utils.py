"""Small helpers shared by the Plotly figure builders."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Tuple

import plotly.graph_objects as go

from exp7.theme import Theme


def _translucent(hex_colour: str, alpha: float = 0.18) -> str:
    """A hex colour converted to a low-opacity rgba() string, for shaded diagram boxes."""
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _rounded_rect_path(x0: float, y0: float, x1: float, y1: float, r: float) -> str:
    """An SVG path string for a rounded rectangle, for use with `fig.add_shape(type="path", ...)`
    - Plotly's own `type="rect"` shape has no corner-radius option, so genuinely soft,
    rounded boxes (rather than sharp corners) need to be hand-drawn as a path."""
    r = max(0.0, min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    return (
        f"M {x0 + r},{y0} "
        f"L {x1 - r},{y0} Q {x1},{y0} {x1},{y0 + r} "
        f"L {x1},{y1 - r} Q {x1},{y1} {x1 - r},{y1} "
        f"L {x0 + r},{y1} Q {x0},{y1} {x0},{y1 - r} "
        f"L {x0},{y0 + r} Q {x0},{y0} {x0 + r},{y0} Z"
    )


def _build_flow_boxes(theme: Theme, stages: Sequence[Tuple[str, str, str]], height: int = 180) -> go.Figure:
    """A left-to-right flowchart of rounded boxes joined by arrows. Each stage is
    (title, subtitle, hover_description) and is hoverable for the longer explanation."""
    colour_cycle = ["PERSON", "ORG", "LOCATION", "DATE", "TERM", "MISC"]
    box_w, gap = 1.0, 0.32
    n = len(stages)

    fig = go.Figure()
    hover_x, hover_y, hover_text = [], [], []
    for i, (title, sub, desc) in enumerate(stages):
        x0 = i * (box_w + gap)
        x1 = x0 + box_w
        colour = theme.for_type(colour_cycle[i % len(colour_cycle)])
        fig.add_shape(
            type="path", path=_rounded_rect_path(x0, 0, x1, 1, r=0.09),
            line=dict(color=colour, width=1.75),
            fillcolor=_translucent(colour, 0.16)
        )
        fig.add_annotation(x=(x0 + x1) / 2, y=0.64, text=f"<b>{title}</b>", showarrow=False,
                           font=dict(size=12, color=theme.ink), align="center")
        fig.add_annotation(x=(x0 + x1) / 2, y=0.28, text=sub, showarrow=False,
                           font=dict(size=9.5, color=theme.muted), align="center")
        if i < n - 1:
            fig.add_annotation(
                x=x0 + box_w + gap, y=0.5, ax=x1, ay=0.5,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.6, arrowcolor=theme.edge
            )
        hover_x.append((x0 + x1) / 2)
        hover_y.append(0.5)
        hover_text.append(f"<b>Stage {i + 1}: {title}</b><br>{desc}")

    # An invisible marker sized to cover each box is what actually makes it hoverable -
    # `add_shape` rectangles have no hover support of their own in Plotly.
    fig.add_trace(go.Scatter(
        x=hover_x, y=hover_y, mode="markers", showlegend=False,
        marker=dict(size=90, opacity=0),
        hovertext=hover_text, hoverinfo="text"
    ))

    fig.update_xaxes(visible=False, range=[-0.15, n * (box_w + gap)], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-0.1, 1.1], fixedrange=True)
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=theme.paper_bg, plot_bgcolor=theme.card_bg,
        hoverlabel=dict(bgcolor=theme.popup_bg, font_color=theme.ink, bordercolor=theme.card_accent, align="left"),
        hovermode="closest"
    )
    return fig


def _load_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None
