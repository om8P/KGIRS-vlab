"""The interactive knowledge-graph figure and the logic that keeps it readable."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import plotly.graph_objects as go

from exp7.figure_utils import _translucent
from exp7.graph_layout import MAX_GRAPH_NODES, _graph_layout, _label_boxes, _label_positions
from exp7.theme import ENTITY_TYPE_ORDER, TYPE_SYMBOL, Theme


ALL_ENTITIES = "All entities"


def _prepare_graph_triples(
    triples: List[Dict[str, Any]],
    type_filter: Optional[List[str]] = None,
    show_weak: bool = False,
    max_entities: int = 20,
    focus: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Chooses which triples are drawn so the graph stays readable.

    Returns (triples_to_draw, info) where info counts what was left out, so the UI can say
    so honestly instead of silently dropping edges or entities."""
    working = list(triples)
    if type_filter:
        typed = [t for t in working
                 if t["subject_label"] in type_filter or t["object_label"] in type_filter]
        working = typed or working

    hidden_weak = 0
    strong = [t for t in working if t.get("cue")]
    if not show_weak and strong:
        hidden_weak = len(working) - len(strong)
        working = strong

    if focus and focus != ALL_ENTITIES:
        focused = [t for t in working if focus in (t["subject"], t["object"])]
        working = focused or working

    names = {n for t in working for n in (t["subject"], t["object"])}
    total_entities = len(names)
    if total_entities > max_entities:
        score: Dict[str, float] = {}
        for t in working:
            for n in (t["subject"], t["object"]):
                score[n] = score.get(n, 0.0) + 1.0 + 0.01 * float(t.get("confidence", 0.0))
        keep = set(sorted(score, key=lambda n: -score[n])[:max_entities])
        pruned = [t for t in working if t["subject"] in keep and t["object"] in keep]
        if not pruned:
            # Top entities aren't linked to each other: keep the strongest edges instead.
            pruned, seen = [], set()
            for t in sorted(working, key=lambda t: -float(t.get("confidence", 0.0))):
                added = {t["subject"], t["object"]} - seen
                if len(seen) + len(added) <= max_entities:
                    seen |= added
                    pruned.append(t)
        working = pruned

    shown = {n for t in working for n in (t["subject"], t["object"])}
    return working, {
        "hidden_weak": hidden_weak,
        "omitted_entities": max(0, total_entities - len(shown)),
        "total_entities": total_entities,
    }


def _build_graph_figure(
    triples: List[Dict[str, Any]],
    theme: Theme,
    show_predicates: bool = True,
    seed: int = 11,
    type_filter: Optional[List[str]] = None,
    show_weak: bool = True,
    max_entities: int = MAX_GRAPH_NODES,
    focus: Optional[str] = None,
) -> go.Figure:
    """Constructs a high-contrast, publication-grade interactive knowledge graph."""
    if not triples:
        fig = go.Figure()
        fig.add_annotation(
            text="No relationships to display.<br>Lower the confidence threshold or switch extraction method.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color=theme.muted, size=13)
        )
        fig.update_layout(
            height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False)
        )
        return fig

    filtered_triples, _ = _prepare_graph_triples(triples, type_filter, show_weak, max_entities, focus)

    # The knowledge graph itself is a networkx MultiDiGraph: a directed multigraph
    # because two entities can be joined by more than one predicate (e.g. both
    # "founded" and "leads"), and each must remain its own edge rather than collapsing.
    kg = nx.MultiDiGraph()
    for t in filtered_triples:
        kg.add_node(t["subject"], label=t["subject_label"])
        kg.add_node(t["object"], label=t["object_label"])
        kg.add_edge(t["subject"], t["object"], predicate=t["predicate"], confidence=t["confidence"])

    names: List[str] = list(kg.nodes())
    labels: Dict[str, str] = nx.get_node_attributes(kg, "label")
    index_of = {name: i for i, name in enumerate(names)}
    # Built from `filtered_triples` directly (not kg.edges()) so edges[i] stays aligned
    # with filtered_triples[i] for the zip() below - a MultiDiGraph's own edge iteration
    # groups all edges by source node, which would silently reorder them relative to the
    # original triples and misattribute confidence/cue/sentence to the wrong edge.
    edges = [(index_of[t["subject"]], index_of[t["object"]]) for t in filtered_triples]

    pos = _graph_layout(names, edges, seed=seed)
    label_at = _label_positions(pos, edges)

    # Degree (multigraph, undirected) = how many relationships touch each entity,
    # regardless of direction - drives node size, so well-connected entities stand out.
    degree: Dict[str, int] = dict(kg.to_undirected(reciprocal=False, as_view=False).degree())

    fig = go.Figure()

    node_size = {n: min(34, 16 + 2.5 * degree.get(n, 1)) for n in names}

    # Draw curved edges with directional arrows
    drawn_pairs: Dict[Tuple[int, int], int] = {}
    lbl_x, lbl_y, lbl_text, lbl_hover = [], [], [], []
    chip_x, chip_y, chip_text = [], [], []

    for t, (start, end) in zip(filtered_triples, edges):
        if start == end:
            continue
        x0, y0 = pos[start]
        x1, y1 = pos[end]
        pair = (min(start, end), max(start, end))
        rank = drawn_pairs.get(pair, 0)
        drawn_pairs[pair] = rank + 1

        length = math.hypot(x1 - x0, y1 - y0) or 1.0
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        bow = 0.0 if rank == 0 else 0.058 * ((rank + 1) // 2) * (1 if rank % 2 else -1)
        mid_x = (x0 + x1) / 2 - uy * bow
        mid_y = (y0 + y1) / 2 + ux * bow

        # Lines run centre-to-centre and the nodes are drawn on top, so an edge always
        # visibly connects its two nodes at any zoom level (no data-unit gap to inflate).
        sx, sy = x0, y0
        ex, ey = x1, y1

        # Edge weight communicates confidence at a glance: a confident, cue-backed edge is
        # drawn bolder and more opaque; a weak co-occurrence fallback fades into the
        # background and is dashed, so the eye is drawn to the reliable structure first.
        is_weak = not t.get("cue")
        confidence = float(t.get("confidence", 0.5))
        edge_opacity = max(0.35, min(0.95, 0.35 + confidence * 0.6))
        edge_width = 1.1 + confidence * 2.2
        edge_colour = _translucent(theme.edge, edge_opacity) if theme.edge.startswith("#") else theme.edge

        edge_popup = (
            f"<b>{t['subject']} &rarr; {t['object']}</b><br>"
            f"Predicate: <b>{t['predicate']}</b><br>"
            f"Confidence: {t['confidence']:.2f}<br>"
            f"Cue: {t['cue'] or 'co-occurrence (no explicit cue)'}<br>"
            f"Sentence: <i>{t['sentence']}</i>"
        )

        # Invisible wide "hit area" trace first, so hovering anywhere near the curve (not
        # just the midpoint label) shows the pop-up - Plotly's hover hit-test follows the
        # trace geometry regardless of how transparent its line colour is.
        fig.add_trace(go.Scatter(
            x=[sx, mid_x, ex], y=[sy, mid_y, ey], mode="lines",
            hoverinfo="text", hovertext=[edge_popup] * 3, showlegend=False,
            line=dict(width=14, color="rgba(0,0,0,0)", shape="spline", smoothing=0.88)
        ))
        # The visible, correctly-styled line drawn on top of the invisible hit area.
        fig.add_trace(go.Scatter(
            x=[sx, mid_x, ex], y=[sy, mid_y, ey], mode="lines",
            hoverinfo="skip", showlegend=False,
            line=dict(width=edge_width, color=edge_colour, shape="spline", smoothing=0.88,
                     dash="dot" if is_weak else "solid")
        ))

        # Arrowhead: tip at the target's centre, pulled back by a *pixel* standoff equal to
        # the node's radius so it lands on the node's rim, with a fixed 26px pixel-length
        # tail - so the head is the same size and visible at any zoom level or edge length.
        dir_x, dir_y = ex - mid_x, ey - mid_y
        dir_len = math.hypot(dir_x, dir_y) or 1.0
        dir_x, dir_y = dir_x / dir_len, dir_y / dir_len
        fig.add_annotation(
            x=ex, y=ey, ax=-dir_x * 26, ay=dir_y * 26, xref="x", yref="y", axref="pixel", ayref="pixel",
            showarrow=True, arrowhead=2, arrowsize=1.0 + confidence * 0.3,
            arrowwidth=max(1.2, edge_width * 0.75), arrowcolor=edge_colour, text="",
            standoff=node_size[names[end]] / 2 + 3
        )

        lbl_x.append(mid_x)
        lbl_y.append(mid_y)
        lbl_text.append(t["predicate"] if show_predicates else "")
        lbl_hover.append(edge_popup)
        if not is_weak:
            chip_x.append(mid_x)
            chip_y.append(mid_y)
            chip_text.append(t["predicate"])

    # Predicate labels are annotations (sized in pixels, so the box always fits its text,
    # unlike data-unit shapes that balloon on zoom). Only confident, cue-backed edges are
    # labelled; weak co-occurrence links stay unlabelled and are explained on hover.
    if show_predicates:
        for cx_, cy_, text in zip(chip_x, chip_y, chip_text):
            fig.add_annotation(
                x=cx_, y=cy_, text=text, showarrow=False, xref="x", yref="y",
                font=dict(size=10, color=theme.ink, family="ui-sans-serif, system-ui, -apple-system, sans-serif"),
                bgcolor=theme.popup_bg, bordercolor=theme.card_border, borderwidth=1,
                borderpad=2, opacity=0.95
            )
    if lbl_x:
        fig.add_trace(go.Scatter(
            x=lbl_x, y=lbl_y, mode="markers",
            marker=dict(size=9, color="rgba(0,0,0,0)"),
            customdata=lbl_hover,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False
        ))

    # Node markers grouped by entity type (gives clear legend items)
    for ent_type in ENTITY_TYPE_ORDER:
        members = [n for n in names if labels[n] == ent_type]
        if not members:
            continue
        indices = [index_of[n] for n in members]
        sizes = [node_size[n] for n in members]
        colour = theme.for_type(ent_type)

        # A soft, larger, translucent "glow" drawn directly beneath each node gives the
        # graph a bit of depth instead of flat markers sitting on a flat background.
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in indices], y=[pos[i][1] for i in indices],
            mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(
                size=[s * 1.9 for s in sizes],
                color=_translucent(colour, 0.22),
                symbol=TYPE_SYMBOL.get(ent_type, "circle"),
                line=dict(width=0)
            )
        ))

        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in indices],
            y=[pos[i][1] for i in indices],
            mode="markers+text",
            name=ent_type,
            text=[(n if len(n) <= 20 else n[:18] + "...") for n in members],
            textposition=[label_at[i] for i in indices],
            textfont=dict(
                size=11,
                color=theme.ink,
                family="ui-sans-serif, system-ui, sans-serif"
            ),
            marker=dict(
                size=sizes,
                color=colour,
                symbol=TYPE_SYMBOL.get(ent_type, "circle"),
                line=dict(width=2.5, color=theme.halo)
            ),
            customdata=[
                f"<b>Entity:</b> {n}<br>"
                f"<b>Type:</b> {ent_type}<br>"
                f"<b>Connected Relationships:</b> {degree.get(n, 0)}"
                for n in members
            ],
            hovertemplate="%{customdata}<extra></extra>"
        ))

    margin_x = float(max(_label_boxes(names)[0])) + 0.08
    margin_y = 0.12
    fig.update_xaxes(visible=False, range=[float(pos[:, 0].min()) - margin_x, float(pos[:, 0].max()) + margin_x])
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1,
                     range=[float(pos[:, 1].min()) - margin_y, float(pos[:, 1].max()) + margin_y])

    # The canvas is as tall as the layout needs (about 330px per unit of vertical extent),
    # so extra small clusters stack downwards at a readable size instead of being squeezed.
    y_extent = float(pos[:, 1].max() - pos[:, 1].min()) + 0.24
    height = int(min(1100, max(460, 320 + 20 * len(names), 170 + 330 * y_extent)))
    n_edges = sum(1 for _, (s, e) in zip(filtered_triples, edges) if s != e)
    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11, color=theme.muted),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=15, r=15, t=40, b=15),
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.card_bg,
        font=dict(color=theme.ink, size=12),
        hoverlabel=dict(
            font_size=12, bgcolor=theme.popup_bg, font_color=theme.ink,
            bordercolor=theme.card_accent, align="left"
        ),
        hoverdistance=20
    )
    fig.add_annotation(
        text=f"{len(names)} entities  &middot;  {n_edges} relationships",
        xref="paper", yref="paper", x=1, y=1.02, xanchor="right", yanchor="bottom",
        showarrow=False, font=dict(size=10.5, color=theme.muted)
    )
    return fig
