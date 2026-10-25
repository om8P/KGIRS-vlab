"""Illustrations of the extraction pipeline and the non-BERT methods."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go

from exp7.figure_utils import _build_flow_boxes
from exp7.theme import Theme


def _build_pipeline_diagram(theme: Theme) -> go.Figure:
    """The extraction pipeline's six processing stages."""
    return _build_flow_boxes(theme, [
        ("Source Text", "Raw document or sentence",
         "The document exactly as supplied - sample corpus, typed text, or an uploaded file."),
        ("Sentence Splitting", "Split on . ! ?",
         "The text is broken into individual sentences; every later stage works one sentence at a time."),
        ("Entity Recognition", "spaCy NER + rules",
         "spaCy's trained model finds names (PERSON, ORG, LOCATION, ...), the lab's capitalisation rules "
         "add names it missed, and subject/object noun phrases become TERM concepts. Without spaCy, "
         "only the capitalisation rules run."),
        ("Candidate Pairs", "Adjacent entities per sentence",
         "Entities that sit next to each other in the same sentence become candidate subject/object pairs."),
        ("Predicate Assignment", "Lexicon / verb / grammar / meaning",
         "The relation name for each pair is decided one of three ways: matching the text between the "
         "entities against an ordered cue lexicon, taking the nearest verb, or reading the grammatical "
         "roles in the sentence (dependency parse), or comparing the clause's meaning with example clauses (BERT)."),
        ("SPO Triple", "Subject, predicate, object, confidence",
         "A pair with an assigned predicate becomes a Subject-Predicate-Object triple with a confidence score."),
    ])


def _build_nearest_verb_steps_figure(theme: Theme) -> go.Figure:
    """Step-by-step illustration of how Nearest-Verb turns the text between two entities
    into a standard predicate, using 'Microsoft, based in Redmond' as the worked case."""
    return _build_flow_boxes(theme, [
        ("Text between entities", "Microsoft [ , based in ] Redmond",
         "Only the words sitting between the two adjacent entities are examined."),
        ("First verb-like word", "'based'",
         "Words are scanned left to right; the first one that looks like a verb (ends in -ed, -ing, -s, "
         "or is a known irregular verb) is chosen. Articles, prepositions and 'is/has' are skipped."),
        ("Lemma + preposition", "base + in = 'base_in'",
         "The verb is reduced to its base form (lemma), and a preposition that follows it is attached."),
        ("Standard predicate", "'base_in' -> located_in",
         "Common verb phrases are mapped to the lab's standard predicate names. A verb with no mapping "
         "keeps its own lemma as the predicate."),
    ], height=170)


_DEP_ARC_HIGHLIGHT = {"nsubj": "PERSON", "nsubjpass": "PERSON", "dobj": "ORG", "agent": "LOCATION", "pobj": "DATE"}


def _build_dependency_arc_figure(doc: Any, theme: Theme) -> go.Figure:
    """Arc diagram of a parsed sentence: each arc runs from a word's grammatical head to the
    word that depends on it, labelled with the dependency role. The roles that carry the
    subject-verb-object meaning are coloured; the rest are muted."""
    tokens = [t for t in doc if not t.is_punct]
    index = {t.i: k for k, t in enumerate(tokens)}
    n = len(tokens)

    fig = go.Figure()
    max_peak = 0.5
    for tok in tokens:
        head = tok.head
        if head.i == tok.i or head.i not in index:
            continue
        x_head, x_dep = float(index[head.i]), float(index[tok.i])
        span = abs(x_dep - x_head)
        peak = 0.32 + 0.30 * span
        max_peak = max(max_peak, peak)
        colour = theme.for_type(_DEP_ARC_HIGHLIGHT[tok.dep_]) if tok.dep_ in _DEP_ARC_HIGHLIGHT else theme.edge
        width = 2.6 if tok.dep_ in _DEP_ARC_HIGHLIGHT else 1.3

        angles = np.linspace(0, math.pi, 40)
        centre = (x_head + x_dep) / 2.0
        xs = centre + (x_head - centre) * np.cos(angles)
        ys = peak * np.sin(angles)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color=colour, width=width),
            hoverinfo="text",
            hovertext=f"<b>{tok.text}</b> depends on <b>{head.text}</b><br>Role: <b>{tok.dep_}</b>",
            showlegend=False
        ))
        direction = 1.0 if x_dep > x_head else -1.0
        fig.add_annotation(
            x=x_dep, y=0.02, ax=-direction * 4, ay=-14, xref="x", yref="y", axref="pixel", ayref="pixel",
            showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=width, arrowcolor=colour, text=""
        )
        fig.add_annotation(
            x=centre, y=peak, text=tok.dep_, showarrow=False, yshift=0,
            font=dict(size=10, color=theme.ink), bgcolor=theme.popup_bg,
            bordercolor=colour, borderwidth=1, borderpad=2, opacity=0.95
        )

    key_tokens = {t.i for t in tokens if t.dep_ in _DEP_ARC_HIGHLIGHT}
    key_tokens |= {t.head.i for t in tokens if t.dep_ in _DEP_ARC_HIGHLIGHT}
    for k, tok in enumerate(tokens):
        bold = tok.i in key_tokens
        fig.add_annotation(
            x=k, y=-0.14, text=f"<b>{tok.text}</b>" if bold else tok.text, showarrow=False,
            font=dict(size=13, color=theme.ink if bold else theme.muted)
        )
        fig.add_annotation(
            x=k, y=-0.34, text=f"<i>{tok.lemma_}</i>", showarrow=False,
            font=dict(size=9.5, color=theme.muted)
        )
        if tok.dep_ == "ROOT":
            fig.add_annotation(
                x=k, y=-0.54, text="ROOT", showarrow=False,
                font=dict(size=10, color=theme.ink), bgcolor=theme.popup_bg,
                bordercolor=theme.edge, borderwidth=1, borderpad=2
            )

    fig.update_xaxes(visible=False, range=[-0.6, n - 0.4], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-0.7, max_peak + 0.3], fixedrange=True)
    fig.update_layout(
        height=int(min(430, 170 + 105 * max_peak)), margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=theme.paper_bg, plot_bgcolor=theme.card_bg,
        hoverlabel=dict(bgcolor=theme.popup_bg, font_color=theme.ink, bordercolor=theme.card_accent, align="left")
    )
    return fig
