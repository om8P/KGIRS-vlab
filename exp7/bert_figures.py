"""Illustrations for the BERT method."""

from __future__ import annotations

import os
import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from exp7.bert_method import BERT_MIN_PROBABILITY, BERT_MIN_SIMILARITY, _BERT_PROTOTYPES, _bert_encode, _bert_prototype_embeddings, bert_classifier_available, bert_classify_clauses, bert_rank_predicates, bert_score_clauses
from exp7.figure_utils import _build_flow_boxes, _load_json_file, _translucent
from exp7.lexicon import _RELATION_PATTERNS
from exp7.paths import PROJECT_ROOT
from exp7.theme import Theme


_BERT_EXAMPLE_CLAUSE = "X was purchased by Y"


def _build_bert_similarity_figure(clause: str, theme: Theme) -> Optional[go.Figure]:
    """Bar chart of how likely the trained classifier thinks each relation is for one masked clause
    (the winner is highlighted and the acceptance threshold is a dashed line). Without the trained
    file it shows the cosine similarity to the closest example clause of each predicate instead."""
    if bert_classifier_available():
        scored = bert_classify_clauses([clause], ["ORG"], ["ORG"])
        if scored is None:
            return None
        class_names, probs = scored
        order = np.argsort(-probs[0])[:10][::-1]
        names = [class_names[i].replace("~rev", " (reversed)") for i in order]
        values = [round(float(probs[0][i]), 3) for i in order]
        details = [f"classifier probability for '{class_names[i]}'" for i in order]
        threshold, axis_title = BERT_MIN_PROBABILITY, "Probability from the trained classifier"
    else:
        sims = bert_score_clauses([clause])
        if sims is None:
            return None
        ranked = bert_rank_predicates(sims[0])[:10][::-1]
        names = [r[0] for r in ranked]
        values = [round(r[1], 3) for r in ranked]
        details = [f"closest example: '{_BERT_PROTOTYPES[r[2]][0]}'" for r in ranked]
        threshold, axis_title = BERT_MIN_SIMILARITY, "Cosine similarity to the closest example clause"
    winner = len(names) - 1
    colours = [theme.for_type("ORG") if i == winner else _translucent(theme.edge, 0.55) if theme.edge.startswith("#") else theme.edge
               for i in range(len(names))]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker=dict(color=colours, cornerradius=5),
        text=[f"{v:.2f}" for v in values], textposition="outside", customdata=details,
        hovertemplate="<b>%{y}</b><br>score %{x:.2f}<br>%{customdata}<extra></extra>"
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color=theme.muted,
                  annotation_text=f"accept at {threshold:.2f}", annotation_position="bottom right",
                  annotation_font=dict(size=10, color=theme.muted))
    fig.update_xaxes(range=[0, 1.12], title=axis_title, gridcolor=theme.grid)
    fig.update_yaxes(gridcolor=theme.grid)
    fig.update_layout(
        height=340, margin=dict(l=10, r=20, t=10, b=40), showlegend=False,
        paper_bgcolor=theme.paper_bg, plot_bgcolor=theme.card_bg, font=dict(color=theme.ink, size=11),
        hoverlabel=dict(bgcolor=theme.popup_bg, font_color=theme.ink, bordercolor=theme.card_accent, align="left")
    )
    return fig


BERT_TRAINING_STATS_PATH = os.path.join(PROJECT_ROOT, "bert_re_training_stats.json")


def _build_bert_steps_figure(theme: Theme) -> go.Figure:
    """The BERT method as a left-to-right flow, on the passive example sentence."""
    return _build_flow_boxes(theme, [
        ("Sentence", "YouTube was purchased by Google",
         "The method works one sentence at a time, on the entities the recogniser found."),
        ("Reading rules", "pair: YouTube - Google",
         "Rules pick which entities to pair: years are skipped, 'A and B' are both subjects, the "
         "subject carries over after 'and', and after a comma aside it is the entity before it."),
        ("Mask names", "'X was purchased by Y'",
         "The entity names are replaced by X and Y so the model judges only the relationship."),
        ("BERT", "384 numbers",
         "Sentence-BERT turns the masked clause into an embedding: 384 numbers that capture its meaning."),
        ("Classifier", "+ entity types",
         "A small trained classifier reads the 384 numbers plus the two entity types and gives a "
         "probability for every relation, its reversed form, and NONE."),
        ("Triple", "Google, acquired, YouTube",
         "'acquired (reversed)' wins with high probability, so the subject and object are swapped."),
    ], height=180)


def _build_bert_embedding_heatmap(clauses: Sequence[str], theme: Theme) -> Optional[go.Figure]:
    """Each clause's 384 embedding numbers drawn as one coloured stripe: clauses with similar
    meanings share a colour pattern."""
    vecs = _bert_encode(clauses)
    if vecs is None:
        return None
    fig = go.Figure(go.Heatmap(
        z=vecs, y=[f"'{c}'" for c in clauses], x=list(range(1, vecs.shape[1] + 1)),
        colorscale="RdBu", zmid=0, showscale=False,
        hovertemplate="%{y}<br>number %{x}: %{z:.3f}<extra></extra>"
    ))
    fig.update_xaxes(title="the 384 numbers of each embedding", showgrid=False, color=theme.muted)
    fig.update_yaxes(autorange="reversed", color=theme.ink)
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=theme.paper_bg, plot_bgcolor=theme.card_bg, font=dict(color=theme.ink, size=11),
        hoverlabel=dict(bgcolor=theme.popup_bg, font_color=theme.ink, bordercolor=theme.card_accent)
    )
    return fig


def _build_bert_training_data_figure(theme: Theme) -> Optional[go.Figure]:
    """Training examples per relation, split by source (hand-written vs Re-DocRED)."""
    stats = _load_json_file(BERT_TRAINING_STATS_PATH)
    if not stats:
        return None
    rows = [(rel, c["synthetic"], c["re-docred"]) for rel, c in stats["per_relation"].items() if rel != "NONE"]
    rows.sort(key=lambda r: r[1] + r[2])
    names = [r[0] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=[r[1] for r in rows], orientation="h", name="hand-written wordings",
                         marker=dict(color=theme.for_type("ORG"), cornerradius=4),
                         hovertemplate="<b>%{y}</b><br>hand-written: %{x}<extra></extra>"))
    fig.add_trace(go.Bar(y=names, x=[r[2] for r in rows], orientation="h", name="Re-DocRED (Wikipedia)",
                         marker=dict(color=theme.for_type("LOCATION"), cornerradius=4),
                         hovertemplate="<b>%{y}</b><br>Re-DocRED: %{x}<extra></extra>"))
    fig.update_layout(
        barmode="stack", height=480, margin=dict(l=10, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        paper_bgcolor=theme.paper_bg, plot_bgcolor=theme.card_bg, font=dict(color=theme.ink, size=11),
        xaxis=dict(title="training examples", gridcolor=theme.grid), yaxis=dict(gridcolor=theme.grid),
        hoverlabel=dict(bgcolor=theme.popup_bg, font_color=theme.ink, bordercolor=theme.card_accent)
    )
    return fig


# (clause, first entity type, second entity type, correct answer) - wordings found neither in the
# Pattern lexicon nor in the BERT classifier's training data.
_BERT_UNSEEN_EXAMPLES = [
    ("X is at the helm of Y", "PERSON", "ORG", "leads"),
    ("X snapped up Y", "ORG", "ORG", "acquired"),
    ("X exchanged vows with Y", "PERSON", "PERSON", "married_to"),
    ("X was brought into being by Y", "ORG", "PERSON", "founded (reversed)"),
    ("X is up against Y", "ORG", "ORG", "competes_with"),
    ("X sparks Y", "MISC", "MISC", "causes"),
]


def _bert_unseen_wordings_table() -> Optional[pd.DataFrame]:
    """Live answers of the word list and of BERT for wordings neither has seen."""
    scored = bert_classify_clauses([c[0] for c in _BERT_UNSEEN_EXAMPLES],
                                   [c[1] for c in _BERT_UNSEEN_EXAMPLES], [c[2] for c in _BERT_UNSEEN_EXAMPLES])
    if scored is None:
        return None
    class_names, probs = scored
    rows = []
    for (clause, _, _, correct), row in zip(_BERT_UNSEEN_EXAMPLES, probs):
        lexicon = next((pred for pattern, pred, _ in _RELATION_PATTERNS
                        if re.search(pattern, clause, re.IGNORECASE)), "(nothing)")
        best = int(np.argmax(row))
        answer = class_names[best].replace("~rev", " (reversed)")
        if class_names[best] == "NONE" or row[best] < BERT_MIN_PROBABILITY:
            answer = f"(skipped: {answer})"
        rows.append({"Clause": clause, "Correct answer": correct, "Word list (Pattern-Based)": lexicon,
                     "BERT answer": answer, "BERT probability": round(float(row[best]), 2)})
    return pd.DataFrame(rows)


_BERT_MAP_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#9D755D",
                     "#72B7B2", "#EECA3B", "#FF9DA6", "#8CD17D", "#79706E", "#D67195"]


def _build_bert_meaning_map(clause: str, theme: Theme) -> Optional[go.Figure]:
    """2-D map of the meaning space: every example clause as a dot (projected from 384 numbers
    to 2 by PCA), coloured by predicate, with the chosen clause as a star. Clauses that mean
    the same thing sit together, which is the whole idea of embeddings."""
    protos = _bert_prototype_embeddings()
    vecs = _bert_encode([clause])
    if protos is None or vecs is None:
        return None
    stacked = np.vstack([protos, vecs])
    centred = stacked - stacked.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vt[:2].T
    proto_xy, clause_xy = coords[:-1], coords[-1]

    predicates = list(dict.fromkeys(pred for _, pred, _ in _BERT_PROTOTYPES))
    colour_of = {pred: _BERT_MAP_PALETTE[i % len(_BERT_MAP_PALETTE)] for i, pred in enumerate(predicates)}
    fig = go.Figure()
    for pred in predicates:
        idx = [i for i, (_, p_, _) in enumerate(_BERT_PROTOTYPES) if p_ == pred]
        fig.add_trace(go.Scatter(
            x=proto_xy[idx, 0], y=proto_xy[idx, 1], mode="markers", name=pred, showlegend=False,
            marker=dict(size=10, color=colour_of[pred], line=dict(width=1, color=theme.halo)),
            hovertext=[f"<b>{pred}</b><br>'{_BERT_PROTOTYPES[i][0]}'" for i in idx], hoverinfo="text"
        ))
        cx, cy = proto_xy[idx, 0].mean(), proto_xy[idx, 1].mean()
        fig.add_annotation(x=cx, y=cy, text=pred, showarrow=False, yshift=16,
                           font=dict(size=9.5, color=theme.ink), opacity=0.9)
    fig.add_trace(go.Scatter(
        x=[clause_xy[0]], y=[clause_xy[1]], mode="markers+text", showlegend=False,
        marker=dict(size=20, symbol="star", color=theme.ink, line=dict(width=1.5, color=theme.halo)),
        text=[f"'{clause}'"], textposition="top center", textfont=dict(size=11, color=theme.ink),
        hovertext=[f"<b>Your clause</b><br>'{clause}'"], hoverinfo="text"
    ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=theme.paper_bg, plot_bgcolor=theme.card_bg,
        hoverlabel=dict(bgcolor=theme.popup_bg, font_color=theme.ink, bordercolor=theme.card_accent, align="left")
    )
    return fig
