"""Charts and tables: entity types, flow (Sankey), method comparison, blind-test results, annotated text."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

import plotly.graph_objects as go

from exp7.corpora import CORPORA
from exp7.entities import extract_entities, spacy_parser_available
from exp7.extraction import evaluate_against_reference_set, extract_relationships
from exp7.theme import TYPE_HIGHLIGHT, TYPE_SLOT, Theme


def _build_flow_figure(triples: List[Dict[str, Any]], theme: Theme, limit: int = 12) -> go.Figure:
    """Sankey flow diagram: Subject Entity Type -> Relational Predicate -> Object Entity Type."""
    if not triples:
        fig = go.Figure()
        fig.add_annotation(text="No relationships available for flow visualization.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(color=theme.muted, size=13))
        fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    counts: Dict[Tuple[str, str, str], int] = {}
    for t in triples:
        key = (t["subject_label"], t["predicate"], t["object_label"])
        counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: -item[1])[:limit]

    left = sorted({k[0] for k, _ in ranked}, key=lambda l: TYPE_SLOT.get(l, 9))
    middle = sorted({k[1] for k, _ in ranked})
    right = sorted({k[2] for k, _ in ranked}, key=lambda l: TYPE_SLOT.get(l, 9))

    node_labels = (
        [f"{l} (subj)" for l in left] +
        middle +
        [f"{l} (obj)" for l in right]
    )
    node_colors = (
        [theme.for_type(l) for l in left] +
        [theme.muted for _ in middle] +
        [theme.for_type(l) for l in right]
    )

    left_idx = {l: i for i, l in enumerate(left)}
    mid_idx = {l: len(left) + i for i, l in enumerate(middle)}
    right_idx = {l: len(left) + len(middle) + i for i, l in enumerate(right)}

    sources, targets, values, link_colors = [], [], [], []

    def hex_to_rgba(hex_str: str, alpha: float = 0.35) -> str:
        hex_str = hex_str.lstrip("#")
        if len(hex_str) == 6:
            r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        return f"rgba(100,116,139,{alpha})"

    for (s_type, pred, o_type), c in ranked:
        # Subject -> Predicate
        sources.append(left_idx[s_type])
        targets.append(mid_idx[pred])
        values.append(c)
        link_colors.append(hex_to_rgba(theme.for_type(s_type), 0.38))

        # Predicate -> Object
        sources.append(mid_idx[pred])
        targets.append(right_idx[o_type])
        values.append(c)
        link_colors.append(hex_to_rgba(theme.for_type(o_type), 0.38))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=16,
            thickness=20,
            line=dict(color=theme.halo, width=1.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])
    fig.update_layout(
        height=380,
        margin=dict(l=15, r=15, t=25, b=15),
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.card_bg,
        font=dict(color=theme.ink, size=11),
        hoverlabel=dict(bgcolor=theme.popup_bg, font_color=theme.ink, font_size=11,
                       bordercolor=theme.card_accent, align="left")
    )
    return fig


def _build_entity_type_chart(entities: List[Dict[str, Any]], theme: Theme) -> go.Figure:
    """Distribution of detected entities across types."""
    counts: Dict[str, int] = {}
    for e in entities:
        counts[e["label"]] = counts.get(e["label"], 0) + 1

    fig = go.Figure(data=[go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker=dict(
            color=[theme.for_type(t) for t in counts.keys()],
            line=dict(color=theme.halo, width=1.5),
            cornerradius=8
        ),
        text=list(counts.values()),
        textposition="auto",
        hovertemplate="<b>%{x}</b>: %{y} mentions<extra></extra>"
    )])
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=25, b=25),
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.card_bg,
        font=dict(color=theme.ink, size=11),
        xaxis=dict(gridcolor=theme.grid),
        yaxis=dict(gridcolor=theme.grid, title="Mentions"),
        bargap=0.35,
        hoverlabel=dict(bgcolor=theme.popup_bg, font_color=theme.ink, bordercolor=theme.card_accent, align="left")
    )
    return fig


# Methods compared side-by-side in the Theory section's "Method Comparison" chart.
_COMPARISON_METHODS = [
    "Co-occurrence Only",
    "Pattern-Based (Lexicon Match)",
    "Hybrid (Pattern + Co-occurrence)",
    "Nearest-Verb (Generalized)",
    "Dependency Parse (SVO)",
]


SVO_METHOD = "Dependency Parse (SVO)"


def _compute_method_comparison(corpus_name: str = "Tech & Business") -> List[Dict[str, Any]]:
    """Runs every extraction method on one benchmark corpus and scores each against
    its ground-truth reference set, for a live, authentic method comparison (not
    illustrative/fabricated numbers). The Dependency Parse method is skipped when spaCy
    is not installed."""
    corpus = CORPORA[corpus_name]
    entities = extract_entities(corpus["text"])
    rows = []
    for method in _COMPARISON_METHODS:
        if method == SVO_METHOD and not spacy_parser_available():
            continue
        triples = extract_relationships(text=corpus["text"], entities=entities, method=method)
        metrics = evaluate_against_reference_set(triples, corpus["gold"])
        rows.append({
            "Method": method,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "Pair Recall": metrics["pair_recall"],
            "Triples Extracted": len(triples)
        })
    return rows


def _build_method_comparison_figure(theme: Theme, corpus_name: str = "Tech & Business") -> go.Figure:
    """Grouped bar chart: Precision / Recall / F1 / Pair Recall for each extraction method,
    computed live against the named benchmark corpus's gold reference set. Pair Recall gives
    partial credit for finding the right (subject, object) pair with any predicate, which is
    what makes Co-occurrence's real behaviour visible even though it always scores 0.0 on the
    strict, exact-predicate metrics (see the docstring on evaluate_against_reference_set)."""
    rows = _compute_method_comparison(corpus_name)
    names = [r["Method"].split(" (")[0] for r in rows]

    fig = go.Figure()
    for metric, colour_key in (("Precision", "PERSON"), ("Recall", "ORG"), ("F1", "LOCATION"), ("Pair Recall", "MISC")):
        values = [r[metric] for r in rows]
        fig.add_trace(go.Bar(
            name=metric, x=names, y=values,
            marker=dict(color=theme.for_type(colour_key), cornerradius=6),
            text=[f"{v:.2f}" for v in values], textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.2f}}<extra></extra>"
        ))
    fig.update_layout(
        barmode="group", height=340,
        margin=dict(l=20, r=20, t=30, b=25),
        paper_bgcolor=theme.paper_bg, plot_bgcolor=theme.card_bg,
        font=dict(color=theme.ink, size=11),
        yaxis=dict(gridcolor=theme.grid, title="Score", range=[0, 1.08]),
        xaxis=dict(gridcolor=theme.grid),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.28, bargroupgap=0.08,
        hoverlabel=dict(bgcolor=theme.popup_bg, font_color=theme.ink, bordercolor=theme.card_accent, align="left")
    )
    return fig


def _annotate_sentence(sentence: str, entities_in_sentence: List[Dict[str, Any]], cues: Sequence[str]) -> str:
    """Formats sentence prose with color-coded entity badges and bold relation cues."""
    sorted_ents = sorted(entities_in_sentence, key=lambda e: e["start"])
    fragments = []
    cursor = 0
    for e in sorted_ents:
        if e["start"] > cursor:
            span = sentence[cursor:e["start"]]
            fragments.append(_mark_cues(span, cues))
        highlight = TYPE_HIGHLIGHT.get(e["label"], "gray")
        fragments.append(f":{highlight}-background[**{e['text']}** ({e['label']})]")
        cursor = e["end"]
    if cursor < len(sentence):
        fragments.append(_mark_cues(sentence[cursor:], cues))
    return "".join(fragments)


def _mark_cues(fragment: str, cues: Sequence[str]) -> str:
    for c in cues:
        if not c:
            continue
        fragment = re.sub(
            re.escape(c),
            lambda m: f"**{m.group(0)}**",
            fragment,
            flags=re.IGNORECASE
        )
    return fragment
