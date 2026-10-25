"""
Virtual Laboratory: Knowledge Graphs and Information Retrieval Systems (KGIRS)
Experiment 7: Relationship Extraction from Unstructured Text
Department of Computer Engineering / Information Technology - Virtual Laboratory Portal

Extracts semantic relationships (Subject - Predicate - Object triples) between
entities identified in text, producing entity-relationship triples suitable for
graph construction (feeds directly into Experiment 8: Create and Manage a Graph
Database, and Experiment 9: Knowledge Graph Schema & Data Import).

Features:
- Native-theme knowledge graph visualization that automatically follows light/dark mode.
- Wikipedia-style hover pop-ups on technical terms in Theory and Simulation, each with a
  plain-language explanation and a worked example, plus a searchable glossary table.
- 6 comprehensive analytical tabs: Triples, Annotated Text, Entities, Graph, Flow (Sankey), and Evaluation.
- Ground truth reference sets for empirical Precision, Recall, and F1 evaluation.
- Trial logger, conceptual quiz with instant colour-coded feedback, and PDF report generator.

Note: the page itself uses no custom CSS (Streamlit's native light/dark themes render it
seamlessly); the one deliberate exception is a small, self-contained stylesheet for the
hover pop-up cards, which pure Streamlit widgets cannot produce on their own.
"""

from __future__ import annotations

import os
import re
import io
import csv
import html
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


# ======================================================================================
# 1. THEME ENGINE: CONTRAST-CALIBRATED PALETTES FOR LIGHT & DARK MODES
# ======================================================================================

class Theme:
    """Provides high-contrast, WCAG-compliant styling tokens for Light and Dark modes."""
    def __init__(self, dark: bool = False):
        self.dark = dark
        if dark:
            self.ink = "#F8FAFC"             # Crisp off-white text
            self.muted = "#94A3B8"           # Medium slate for subtitles / cues
            self.edge = "#94A3B8"            # Clean slate line
            self.grid = "#334155"            # Subtle grid line
            self.halo = "rgba(15, 23, 42, 0.95)"  # Dark slate boundary for markers
            self.paper_bg = "rgba(0,0,0,0)"
            self.plot_bg = "rgba(0,0,0,0)"
            self.card_bg = "rgba(30, 41, 59, 0.65)"
            self.card_border = "rgba(148, 163, 184, 0.3)"
            self.card_accent = "#38BDF8"
            self.popup_bg = "#1E293B"        # Solid (not translucent) hover pop-up background
            self.badge_bg = "rgba(56, 189, 248, 0.18)"
            self.badge_text = "#38BDF8"
            self.node_colors = {
                "PERSON": "#38BDF8",     # Bright sky blue
                "ORG": "#FB923C",        # Bright coral orange
                "LOCATION": "#34D399",   # Vibrant emerald
                "DATE": "#C084FC",       # Vibrant violet
                "TERM": "#F87171",       # Salmon rose
                "MISC": "#A3A3A3",       # Neutral silver
            }
        else:
            self.ink = "#0F172A"             # Deep slate / navy text
            self.muted = "#475569"           # Slate for subtitles / cues
            self.edge = "#64748B"            # Rich slate edge
            self.grid = "#E2E8F0"            # Light grey grid line
            self.halo = "rgba(255, 255, 255, 0.95)" # White crisp boundary
            self.paper_bg = "rgba(0,0,0,0)"
            self.plot_bg = "rgba(0,0,0,0)"
            self.card_bg = "rgba(248, 250, 252, 0.85)"
            self.card_border = "rgba(100, 116, 139, 0.25)"
            self.card_accent = "#2563EB"
            self.popup_bg = "#FFFFFF"        # Solid (not translucent) hover pop-up background
            self.badge_bg = "rgba(37, 99, 235, 0.12)"
            self.badge_text = "#1D4ED8"
            self.node_colors = {
                "PERSON": "#2563EB",     # Classic royal blue
                "ORG": "#D97706",        # Amber orange
                "LOCATION": "#059669",   # Deep emerald
                "DATE": "#7C3AED",       # Deep violet
                "TERM": "#DC2626",       # Deep crimson
                "MISC": "#4B5563",       # Charcoal
            }

    def for_type(self, entity_type: str) -> str:
        return self.node_colors.get(entity_type, self.node_colors["MISC"])


def get_current_theme() -> Theme:
    """Detects the theme the viewer picked in Streamlit's own top-right "..." menu
    (System / Light / Dark), so charts and hover cards match the native page."""
    is_dark = False
    try:
        ctx_theme = getattr(st.context, "theme", None)
        if ctx_theme is not None and getattr(ctx_theme, "type", None):
            is_dark = (ctx_theme.type == "dark")
    except Exception:
        is_dark = False
    return Theme(dark=is_dark)


# Fixed marker shapes per entity type for double encoding (color + shape)
TYPE_SYMBOL = {
    "PERSON": "circle",
    "ORG": "square",
    "LOCATION": "diamond",
    "DATE": "triangle-up",
    "TERM": "hexagon",
    "MISC": "cross",
}

TYPE_SLOT = {
    "PERSON": 0,
    "ORG": 1,
    "LOCATION": 2,
    "DATE": 3,
    "TERM": 4,
    "MISC": 5,
}

# Streamlit Markdown text-highlight colors for annotated text view
TYPE_HIGHLIGHT = {
    "PERSON": "blue",
    "ORG": "orange",
    "LOCATION": "green",
    "DATE": "violet",
    "TERM": "red",
    "MISC": "gray",
}

ENTITY_TYPE_ORDER = ["PERSON", "ORG", "LOCATION", "DATE", "TERM", "MISC"]


# ======================================================================================
# 2. TECHNICAL TERMINOLOGY & INTERACTIVE POP-UP DISCLOSURE SYSTEM
# ======================================================================================

TERMS: Dict[str, Dict[str, str]] = {
    "information_retrieval": {
        "name": "Information Retrieval (IR)",
        "definition": "The science and engineering discipline of searching for, indexing, extracting, and representing relevant facts or knowledge from large collections of unstructured data.",
        "example": "Turning raw narrative news reports into queryable knowledge graph assertions.",
    },
    "relationship_extraction": {
        "name": "Relationship Extraction (RE)",
        "definition": "The NLP and IR task of detecting semantic relationships holding between two identified entities mentioned in text and mapping them into structured assertions.",
        "example": '"Larry Page co-founded Google in California" -> (Larry Page, founded, Google)',
    },
    "knowledge_graph": {
        "name": "Knowledge Graph (KG)",
        "definition": "A network structured representation of real-world knowledge where entities form nodes and semantic relationships form directed, typed edges. Enables multi-hop relational queries.",
        "example": "(Larry Page)-[:FOUNDED]->(Google)-[:LOCATED_IN]->(California)",
    },
    "triple": {
        "name": "Subject-Predicate-Object (SPO) Triple",
        "definition": "The atomic data unit in Knowledge Graphs and RDF consisting of exactly three components: (Subject Entity, Predicate / Relationship, Object Entity).",
        "example": "(Google, acquired, YouTube)\nSubject = Google, Predicate = acquired, Object = YouTube",
    },
    "subject": {
        "name": "Subject",
        "definition": "The source entity initiating or possessing the semantic relationship. In active voice English, the subject typically precedes the verbal predicate.",
        "example": 'In "Marie Curie discovered radium", the subject is Marie Curie (PERSON).',
    },
    "predicate": {
        "name": "Predicate / Relation",
        "definition": "The semantic link connecting subject and object. Normalised to snake_case without spaces to form clean edge types in graph databases.",
        "example": 'The verbs "co-founded", "founded", and "established" all map to the canonical predicate "founded".',
    },
    "object": {
        "name": "Object",
        "definition": "The target or destination entity that receives or completes the relation assertion.",
        "example": 'In "Microsoft is located in Redmond", the object is Redmond (LOCATION).',
    },
    "overfitting": {
        "name": "Overfitting",
        "definition": "When a method fits the examples it was built or trained on so closely that its score on them overstates how well it works on new text.",
        "example": "The BERT method scores 0.87 F1 on the sample corpora it was refined on, but 0.38 on the blind test.",
    },
    "training_data": {
        "name": "Training Data",
        "definition": "The labelled examples a machine-learning model learns from. A model is only as good as its training data: relations it saw few examples of are recognised less reliably.",
        "example": '"X bought Y", labelled acquired, is one training example for the BERT classifier.',
    },
    "held_out": {
        "name": "Held-out Data",
        "definition": "Data deliberately set aside and never used for training or tuning, kept only to measure how well a method works on things it has not seen.",
        "example": '64 relation wordings, such as "exchanged vows with", were kept out of BERT\'s training to test whether it generalises.',
    },
    "blind_test_set": {
        "name": "Blind Test Set",
        "definition": "A test set kept completely separate from building and tuning a method, and scored only as a whole, so its score is an honest estimate of real-world performance.",
        "example": "This lab's 106-sentence blind test: 60 Wikipedia sentences from Re-DocRED plus 46 hand-written ones.",
    },
    "noun_chunk": {
        "name": "Noun Chunk (Noun Phrase)",
        "definition": "A noun together with the words that describe it. Noun phrases that are the subject or object of a verb often name the concepts a sentence is about, even when they are not capitalised.",
        "example": 'In "Penicillin treats bacterial infections", the noun chunk "bacterial infections" becomes a TERM entity.',
    },
    "named_entity_recognition": {
        "name": "Named Entity Recognition (NER)",
        "definition": "The preceding IR stage that locates mentions of real-world objects in unstructured text and classifies them into predefined types (Experiment 6).",
        "example": "Recognizing [Sundar Pichai] as PERSON, [Google] as ORG, and [California] as LOCATION.",
    },
    "entity": {
        "name": "Entity",
        "definition": "A specific real-world object referred to in the document (person, organisation, place, or domain concept). Distinct from generic common nouns.",
        "example": '"Google" is an entity; "a software company" is a generic noun phrase.',
    },
    "entity_type": {
        "name": "Entity Type",
        "definition": "The category assigned to an entity (PERSON, ORG, LOCATION, DATE, TERM, MISC). Restricts which relational predicates can meaningfully link them.",
        "example": 'born_in requires subject: PERSON and object: LOCATION or DATE.',
    },
    "relation_cue": {
        "name": "Relation Cue / Trigger Lexicon",
        "definition": "The specific lexical token or verbal phrase between two entities that provides textual evidence for the relationship.",
        "example": 'In "Google acquired YouTube in 2006", the word "acquired" is the lexical cue.',
    },
    "pattern_based": {
        "name": "Pattern-Based Extraction",
        "definition": "Extracts triples only when an explicit relation cue matches a predefined lexical rule or regex. Achieves high precision, but lower recall due to natural language paraphrasing.",
        "example": 'Matches "X founded Y" and "X launched Y", but may miss "X was the architect behind Y".',
    },
    "cooccurrence": {
        "name": "Co-occurrence Extraction",
        "definition": "Links every pair of entities occurring in the same sentence or window with a generic related_to predicate. Achieves high recall but low precision.",
        "example": '(Einstein, related_to, Niels Bohr) with low baseline confidence 0.35.',
    },
    "hybrid": {
        "name": "Hybrid Extraction",
        "definition": "Applies pattern matching first with high confidence, and falls back to co-occurrence with lower confidence when no cue matches. Balances precision and recall.",
        "example": 'Yields (Google, acquired, YouTube) at 0.95 and (Google, related_to, Android) at 0.35.',
    },
    "confidence": {
        "name": "Confidence Score",
        "definition": "A heuristic score in [0, 1] indicating extraction reliability. Allows downstream graph ingestion to filter noise.",
        "example": "Score 0.95 for unambiguous cue 'acquired'; score 0.35 for unguided co-occurrence.",
    },
    "threshold": {
        "name": "Minimum Confidence Threshold",
        "definition": "The cutoff below which candidate triples are discarded. Higher thresholds favour precision; lower thresholds favour recall.",
        "example": "Threshold 0.70 filters out weak verbal ties and generic co-occurrence edges.",
    },
    "word_distance": {
        "name": "Maximum Word Distance",
        "definition": "The maximum permitted token distance between two entities in a sentence. Prevents linking entities in unrelated clauses.",
        "example": "At limit = 10 words, entities 22 words apart in different clauses are rejected.",
    },
    "candidate_pair": {
        "name": "Candidate Pair",
        "definition": "A pair of entities found in the same sentence evaluated by the extraction engine for a valid relation.",
        "example": '"Page and Brin founded Google" generates pairs (Page, Brin) and (Brin, Google).',
    },
    "precision": {
        "name": "Precision (P)",
        "definition": "The fraction of extracted triples that are factually correct: P = Correct Extractions / Total Extracted.",
        "example": "10 triples extracted, 8 correct -> Precision = 0.80 (80%).",
    },
    "recall": {
        "name": "Recall (R)",
        "definition": "The fraction of actual relationships present in the text that were extracted: R = Correct Extractions / Total Present in Ground Truth.",
        "example": "12 relations exist in text, 8 extracted -> Recall = 0.67 (66.7%).",
    },
    "f1": {
        "name": "F1 Score",
        "definition": "The harmonic mean of Precision and Recall: F1 = 2 * (P * R) / (P + R). Penalises extreme imbalances between precision and recall.",
        "example": "P = 0.80, R = 0.67 -> F1 = 0.73.",
    },
    "reference_set": {
        "name": "Ground Truth Reference Set",
        "definition": "A human-curated set of gold standard triples known to be asserted by the text, used to evaluate extraction accuracy.",
        "example": "The Tech & Business corpus carries 14 gold triples against which systems are benchmarked.",
    },
    "false_positive": {
        "name": "False Positive (FP)",
        "definition": "A triple extracted by the system that is unsupported or incorrect based on the text. Lowers precision.",
        "example": 'Extracting (California, founded, Google) due to word proximity.',
    },
    "false_negative": {
        "name": "False Negative (FN)",
        "definition": "A valid relationship in the text that the extractor failed to extract (an omission). Lowers recall.",
        "example": 'Failing to extract a passive-voice fact: "YouTube was bought by Google".',
    },
    "graph_query_language": {
        "name": "Graph Query Language",
        "definition": "A declarative language for asking questions of a graph database by describing a pattern of nodes and edges, such as (person)-[founded]->(company).",
        "example": "Find every (person)-[founded]->(company) pair and return both names.",
    },
    "upsert": {
        "name": "Upsert (Idempotent Import)",
        "definition": "Insert a node or relationship only if it does not already exist, otherwise reuse it. Importing the same triples twice then never creates duplicates.",
        "example": "Importing (Google, acquired, YouTube) twice still leaves exactly one Google node.",
    },
    "graphml": {
        "name": "GraphML",
        "definition": "An open, XML-based file format for graphs. Most graph databases and graph tools can import it, so it carries extracted triples to the next experiment.",
        "example": "Each entity becomes a <node>, and each triple becomes an <edge> carrying its predicate and confidence.",
    },
    "nearest_verb": {
        "name": "Nearest-Verb Extraction",
        "definition": "A relation extraction method with no fixed word list: it takes the first verb-like word between two entities, reduces it to its base form, and uses that as the predicate.",
        "example": '"Google acquired YouTube" -> the verb "acquired" -> lemma "acquire" -> predicate "acquired".',
    },
    "svo_extraction": {
        "name": "Dependency-Parse (SVO) Extraction",
        "definition": "A relation extraction method that reads the sentence grammar instead of the words between entities. The subject, verb and object are found from their grammatical roles, so word order and passive voice no longer matter.",
        "example": '"YouTube was acquired by Google" -> subject-of-action = Google, object = YouTube -> (Google, acquired, YouTube).',
    },
    "provenance": {
        "name": "Provenance",
        "definition": "The audit trail tracking the exact source sentence and text span from which a knowledge triple was derived.",
        "example": 'Triple (Google, acquired, YouTube) stores provenance: "Google acquired YouTube in 2006."',
    },
    "coreference": {
        "name": "Co-reference Resolution",
        "definition": "Resolving pronouns ('he', 'she', 'they', 'it') and descriptive noun phrases back to their antecedent entity mention.",
        "example": '"Marie Curie won the prize. She moved to Paris." -> "She" refers to Marie Curie.',
    },
    "dependency_parsing": {
        "name": "Dependency Parsing",
        "definition": "Syntactic parsing that maps grammatical relationships between words. The shortest path between entities identifies the governing verb.",
        "example": "In complex or inverted sentences, dependency parse paths bridge long entity separations.",
    },
    "open_ie": {
        "name": "Open Information Extraction (OpenIE)",
        "definition": "Relation extraction that extracts arbitrary relation phrases directly from text without adhering to a fixed, predefined schema.",
        "example": 'Extracts ("Sundar Pichai", "served as senior executive at", "Google") directly.',
    },
    "brittleness": {
        "name": "Brittleness of Rules",
        "definition": "The susceptibility of hand-crafted pattern rules to fail when encountering varied linguistic paraphrasing, metaphors, or complex syntax.",
        "example": 'A rule matching "acquired" will miss colloquial paraphrases like "swallowed up" or "brought under its umbrella".',
    },
    "error_propagation": {
        "name": "Error Propagation",
        "definition": "A cascade in a multi-stage pipeline where errors in early stages (e.g. misclassified NER entity) irrevocably corrupt later stages (RE).",
        "example": 'If "Paris" is misclassified as PERSON, RE might output (Paris, studied_at, University).',
    },
    "lemma": {
        "name": "Lemma (Base Form)",
        "definition": "The dictionary form of a word, with tense, plural and other endings removed, so different spellings of one verb can be treated as the same relation.",
        "example": '"acquired", "acquires" and "acquiring" all have the lemma "acquire".',
    },
    "passive_voice": {
        "name": "Passive Voice",
        "definition": "A sentence form where the receiver of the action comes first and the doer follows after \"by\". Methods that read left-to-right often reverse the direction of the relation.",
        "example": '"YouTube was acquired by Google" means (Google, acquired, YouTube), not (YouTube, acquired, Google).',
    },
    "bert": {
        "name": "BERT",
        "definition": "A language model (Bidirectional Encoder Representations from Transformers) trained on huge amounts of text. It reads a whole sentence at once, so the meaning it gives a word depends on the words around it.",
        "example": 'In "river bank" and "bank loan", BERT gives "bank" two different meanings because of its neighbours.',
    },
    "transformer": {
        "name": "Transformer",
        "definition": "The neural-network design behind BERT. It uses attention to let every word look at every other word in the sentence, which is how it captures context.",
        "example": 'In "Google acquired YouTube", attention links "acquired" to both "Google" and "YouTube".',
    },
    "embedding": {
        "name": "Embedding",
        "definition": "A list of numbers (a vector) that represents the meaning of a piece of text. Texts with similar meanings get similar vectors, so meaning can be compared with arithmetic.",
        "example": '"X bought Y" and "X acquired Y" get nearly the same vector; "X was born in Y" gets a very different one.',
    },
    "cosine_similarity": {
        "name": "Cosine Similarity",
        "definition": "A score from about 0 to 1 for how closely two embeddings point in the same direction. 1 means the same meaning, and lower scores mean less related meanings.",
        "example": 'similarity("X bought Y", "X acquired Y") = 0.84, while similarity("X bought Y", "X was born in Y") = 0.54.',
    },
    "sentence_bert": {
        "name": "Sentence-BERT",
        "definition": "A version of BERT trained to produce one embedding per whole sentence, so two sentences can be compared directly. This lab uses the compact all-MiniLM-L6-v2 model (about 90 MB).",
        "example": "encode(\"X acquired Y\") returns a vector of 384 numbers.",
    },
    "entity_masking": {
        "name": "Entity Masking",
        "definition": "Replacing the entity names with neutral placeholders (X and Y) before comparing meaning, so the model judges the relationship itself and not the particular companies or people.",
        "example": '"Google acquired YouTube" becomes "X acquired Y".',
    },
    "classifier": {
        "name": "Classifier",
        "definition": "A model that puts each input into one of a fixed set of categories. It learns the rule from labelled examples instead of being programmed with it.",
        "example": 'Shown "X was bought by Y", the classifier answers "acquired, with the second entity as the subject".',
    }
}

# A small, self-contained stylesheet for Wikipedia-style hover pop-ups on technical
# terms - a pure-CSS `:hover` card (no click, no JS), the one deliberate exception to
# the "no custom CSS" rule elsewhere in this file, since Streamlit's own widgets cannot
# produce an inline hover preview on their own.
_WIKI_TERM_STYLE = """
<style>
span.wikiterm {
    position: relative;
    border-bottom: 1px dotted #2563eb;
    cursor: help;
    font-weight: 500;
}
span.wikiterm > span.wikicard {
    display: none;
    position: absolute;
    left: 0;
    top: 1.5em;
    z-index: 60;
    width: max-content;
    max-width: min(22rem, 85vw);
    background: #ffffff;
    color: #0f172a;
    border: 1px solid #d8e0ec;
    border-radius: 10px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.18);
    padding: 0.7rem 0.95rem;
    text-align: left;
    white-space: normal;
    font-weight: 400;
    font-size: 0.88rem;
    line-height: 1.5;
}
span.wikiterm:hover > span.wikicard,
span.wikiterm:focus > span.wikicard {
    display: block;
}
span.wikiterm > span.wikicard > span.wikicard-title {
    display: block;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
    color: #1b365d;
}
span.wikiterm > span.wikicard > span.wikicard-example {
    display: block;
    margin-top: 0.45rem;
    padding-top: 0.4rem;
    border-top: 1px solid #edf2f9;
    font-style: italic;
    color: #475569;
}
</style>
"""


_WIKI_TERM_STYLE_DARK = """
<style>
span.wikiterm > span.wikicard {
    background: #16223a;
    color: #e6edf7;
    border-color: #2a3a55;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.5);
}
span.wikiterm > span.wikicard > span.wikicard-title { color: #8db4ea; }
span.wikiterm > span.wikicard > span.wikicard-example { color: #c5d2e6; border-top-color: #2a3a55; }
span.wikiterm { border-bottom-color: #8db4ea; }
</style>
"""


def inject_wiki_term_style():
    """Hover-card colours follow the theme chosen in the app's own menu (not the OS)."""
    st.markdown(_WIKI_TERM_STYLE + (_WIKI_TERM_STYLE_DARK if get_current_theme().dark else ""),
                unsafe_allow_html=True)


_TERM_MARKER_REGEX = re.compile(r"\[\[([a-z0-9_]+)(?:\|([^\]]+))?\]\]")
_PROPER_TERMS = {"graphml", "f1", "open_ie", "information_retrieval"}


def _format_inline_name(key: str, name: str) -> str:
    """Mid-sentence, a term reads better lower-cased (e.g. "...forms a knowledge graph")
    unless it is a proper noun/acronym, which always keeps its given capitalisation."""
    if key in _PROPER_TERMS or not name:
        return name
    return name[0].lower() + name[1:] if len(name) > 1 else name.lower()


def _wiki_term_html(key: str, display: str = "") -> str:
    """The hoverable <span> for one [[key]] / [[key|Display]] marker: a plain-language
    definition plus a concrete worked example, in the style of a Wikipedia link preview."""
    entry = TERMS.get(key)
    if not entry:
        return display or html.escape(key.replace("_", " "))
    shown = display or html.escape(_format_inline_name(key, entry["name"]))
    name = html.escape(entry["name"])
    definition = html.escape(entry["definition"])
    example = html.escape(entry.get("example", "")).replace("\n", "<br>")
    example_html = (
        f'<span class="wikicard-example">Example: {example}</span>' if example else ""
    )
    return (
        f'<span class="wikiterm">{shown}'
        f'<span class="wikicard">'
        f'<span class="wikicard-title">{name}</span>'
        f'{definition}'
        f'{example_html}'
        f'</span></span>'
    )


def _markdown_inline_to_html(fragment: str) -> str:
    """Escapes raw text, then applies inline **bold**, *italic* and `code` markdown."""
    escaped = html.escape(fragment)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code style='background:rgba(128,128,128,0.15); padding:1px 4px; border-radius:3px;'>\1</code>", escaped)
    return escaped


def _inline(text: str) -> str:
    """Inline markdown (bold/italic/code) is applied first, so a `**...**` span may safely
    wrap a `[[term]]` marker; the (now HTML-safe) marker text is substituted in afterwards,
    so it is not escaped a second time."""
    formatted = _markdown_inline_to_html(text)
    parts = []
    cursor = 0
    for match in _TERM_MARKER_REGEX.finditer(formatted):
        parts.append(formatted[cursor:match.start()])
        parts.append(_wiki_term_html(match.group(1), match.group(2) or ""))
        cursor = match.end()
    parts.append(formatted[cursor:])
    return "".join(parts)


_HEADER_LINE = re.compile(r"^(#{1,4})\s+(.*)$")
_BULLET_LINE = re.compile(r"^[-*]\s+(.*)$")
_NUMBERED_LINE = re.compile(r"^\d+\.\s+(.*)$")
_HEADER_TAG_BY_LEVEL = {1: "h3", 2: "h4", 3: "h4", 4: "h5"}


def annotate_terms(text: str) -> str:
    """Renders prose that mixes ordinary block Markdown (headers, lists, paragraphs) with
    [[key]] / [[key|Display]] markers for Wikipedia-style hover term pop-ups."""
    lines = text.strip("\n").split("\n")
    blocks: List[str] = []
    paragraph: List[str] = []

    def flush_paragraph():
        if not paragraph:
            return
        joined = " ".join(l.strip() for l in paragraph if l.strip())
        if joined:
            blocks.append(f"<p style='margin-bottom:0.75rem; line-height:1.6;'>{_inline(joined)}</p>")
        paragraph.clear()

    # An indented block of `- ` bullets directly beneath a numbered item (a common pattern
    # in this file's theory prose) is that item's nested detail, not a new top-level list -
    # so it must not reset the outer <ol> back to "1.".
    def consume_nested_bullets(start: int) -> Tuple[str, int]:
        j = start
        nested_items = []
        while j < len(lines) and lines[j][:1].isspace() and _BULLET_LINE.match(lines[j].strip()):
            nested_items.append(_BULLET_LINE.match(lines[j].strip()).group(1))
            j += 1
        if not nested_items:
            return "", start
        block_html = (
            "<ul style='margin:0.2rem 0 0.4rem 0; padding-left:1.3rem; line-height:1.55;'>"
            + "".join(f"<li>{_inline(item)}</li>" for item in nested_items) + "</ul>"
        )
        return block_html, j

    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue

        header_match = _HEADER_LINE.match(stripped)
        if header_match:
            flush_paragraph()
            tag = _HEADER_TAG_BY_LEVEL.get(len(header_match.group(1)), "h4")
            blocks.append(f"<{tag} style='margin:1.1rem 0 0.5rem 0;'>{_inline(header_match.group(2))}</{tag}>")
            i += 1
            continue

        if _BULLET_LINE.match(stripped) and not lines[i][:1].isspace():
            flush_paragraph()
            items = []
            while i < n and lines[i][:1] != " " and _BULLET_LINE.match(lines[i].strip()):
                items.append(_BULLET_LINE.match(lines[i].strip()).group(1))
                i += 1
            blocks.append(
                "<ul style='margin:0 0 0.75rem 0; padding-left:1.4rem; line-height:1.6;'>"
                + "".join(f"<li>{_inline(item)}</li>" for item in items) + "</ul>"
            )
            continue

        if _NUMBERED_LINE.match(stripped) and not lines[i][:1].isspace():
            flush_paragraph()
            items = []
            while i < n and lines[i][:1] != " " and _NUMBERED_LINE.match(lines[i].strip()):
                content = _NUMBERED_LINE.match(lines[i].strip()).group(1)
                i += 1
                nested_html, i = consume_nested_bullets(i)
                # A single blank line between numbered items is layout only, not a list
                # break, as long as another numbered item follows it.
                if i < n and not lines[i].strip():
                    lookahead = i + 1
                    if lookahead < n and lines[lookahead][:1] != " " and _NUMBERED_LINE.match(lines[lookahead].strip()):
                        i = lookahead
                items.append(f"<li>{_inline(content)}{nested_html}</li>")
            blocks.append(
                "<ol style='margin:0 0 0.75rem 0; padding-left:1.4rem; line-height:1.6;'>"
                + "".join(items) + "</ol>"
            )
            continue

        paragraph.append(lines[i])
        i += 1

    flush_paragraph()
    return "".join(blocks)


def write_annotated(text: str):
    """Renders prose with Wikipedia-style hover pop-ups on any [[key]] / [[key|Display]]
    technical terms it contains."""
    st.markdown(annotate_terms(text), unsafe_allow_html=True)


def render_glossary_expander():
    """Renders a full searchable reference table of all technical terms."""
    with st.expander("Explore Full Technical Terms Glossary (25+ IR & KG Concepts)", expanded=False):
        st.write("Browse formal definitions, IR roles, and concrete examples for terms used throughout the virtual laboratory.")
        search_query = st.text_input("Filter Glossary by keyword", placeholder="e.g. predicate, recall, confidence", key="glossary_search")
        
        rows = []
        for k, v in sorted(TERMS.items(), key=lambda item: item[1]["name"]):
            if search_query:
                q = search_query.lower()
                if q not in v["name"].lower() and q not in v["definition"].lower() and q not in v.get("example", "").lower():
                    continue
            rows.append({
                "Technical Term": v["name"],
                "Formal Definition": v["definition"],
                "Concrete Lab Example": v.get("example", "—")
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("No matching terms found in the glossary.")


# ======================================================================================
# 3. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "title": "Experiment 7: Relationship Extraction from Text",
    "course": "Knowledge Graphs and Information Retrieval Systems (KGIRS) · Course Code: D17A/B/C",
    "department": "Department of Computer Engineering / Information Technology",
    "portal": "Virtual Laboratory Learning Portal",
    "objectives": [
        "Understand how semantic relationships between entities are identified in unstructured text.",
        "Implement rule-based and pattern-based relation extraction over sentence boundaries.",
        "Represent extracted knowledge as Subject-Predicate-Object (SPO) triples with confidence scoring.",
        "Evaluate extraction precision, recall, and F1 score against human-curated gold standard reference sets.",
        "Prepare extracted knowledge graph triples for downstream graph database ingestion (Experiments 8-9)."
    ]
}

THEORY_CONTENT = {
    "background": """
### Overview & Principles
[[relationship_extraction|Relationship Extraction (RE)]] is the central task in [[information_retrieval|Information Retrieval]] and natural language processing that identifies and classifies semantic relationships holding between pairs of [[entity|entities]] mentioned in unstructured prose.

Given an input sentence such as *"Larry Page co-founded Google in California"*, an RE system automatically recognizes the entities and extracts the structured [[triple|Subject-Predicate-Object (SPO) triple]]:

`(Larry Page) --[founded]--> (Google)`

These triples represent the atomic building blocks of a [[knowledge_graph|Knowledge Graph]]:
- **Nodes** correspond to real-world entities (people, organizations, locations).
- **Directed Edges** correspond to relational predicates (`founded`, `acquired`, `located_in`).

### Extraction Approaches Compared in this Laboratory
1. **[[pattern_based|Pattern-Based Extraction]]**:
   - Matches hand-crafted lexico-syntactic patterns and verbal cues (e.g. *acquired*, *is the CEO of*, *located in*) occurring between entities.
   - **Characteristics**: High [[precision]], transparent and interpretable, but subject to linguistic [[brittleness]] when sentences contain complex clauses or unseen paraphrases.

2. **[[cooccurrence|Co-occurrence Extraction]]**:
   - Assumes that any two entities appearing within the same sentence boundary are related, assigning a generic `related_to` predicate.
   - **Characteristics**: High [[recall]], but low [[precision]] and generates unnamed, weakly-specified edges.

3. **[[hybrid|Hybrid Extraction]]**:
   - Executes pattern matching first. If an explicit [[relation_cue|relation cue]] is matched, it assigns the specific predicate with high [[confidence]]; otherwise, it defaults to a low-confidence co-occurrence link.
   - **Characteristics**: Maximizes coverage while equipping downstream graph ingestion stages with a quantitative confidence filter.

4. **[[nearest_verb|Nearest-Verb Extraction]]**:
   - Drops the fixed word list. For each adjacent entity pair it takes the first verb-like word between them, reduces it to its [[lemma]] (*acquired* becomes *acquire*), attaches a following preposition (*based in* becomes `base_in`), and maps common verbs onto the lab's standard predicate names.
   - **Characteristics**: Handles verbs the lexicon never listed and needs no extra software, but reads strictly left to right, so a [[passive_voice|passive sentence]] ("YouTube was acquired by Google") comes out with the direction reversed.

5. **[[svo_extraction|Dependency-Parse (SVO) Extraction]]**:
   - Uses [[dependency_parsing|dependency parsing]] to label every word with its grammatical role (`nsubj` subject, `dobj` object, `nsubjpass` passive subject, `agent` the "by" phrase, `pobj` object of a preposition). A triple is emitted when two entities attach to the same verb.
   - **Characteristics**: Handles passive voice, one subject with several verbs, and entities far apart, making it the most accurate of the rule-based methods. The cost is a dependency on a trained language parser (spaCy), and its mistakes become the extractor's mistakes.

6. **[[bert|BERT]] Extraction (a trained classifier on BERT embeddings)**:
   - [[bert|BERT]] is a [[transformer]] language model pretrained on huge amounts of text, so it understands words in context. Its [[sentence_bert|Sentence-BERT]] form turns a sentence into an [[embedding]] (384 numbers) that captures its meaning; [[cosine_similarity|cosine similarity]] measures how close two meanings are.
   - **Reading rules** first choose which entities to pair: years are skipped; coordinated names (*Larry Page and Sergey Brin founded Google*) are all subjects; after "and", a new verb keeps the earlier subject (*Microsoft acquired LinkedIn and partnered with OpenAI*); and after a comma aside the subject is the entity before it (*Microsoft, based in Redmond, competes with Google*).
   - For each pair the names are hidden ([[entity_masking|entity masking]]: *X acquired Y*), the clause is embedded, and a small [[classifier]] reads the embedding plus the two entity types. It answers with one of the 18 relations, the same relation reversed (passive voice: *X was acquired by Y* means Y is the subject), or *none*; its probability becomes the confidence.
   - The classifier learned from [[training_data|training data]]: about 2,000 wordings written for this lab and 6,000 Wikipedia clauses from the public Re-DocRED dataset (MIT licence). Re-DocRED has no examples for 8 of the 18 relations, so those rely on the hand-written wordings alone.
   - **Characteristics**: It compares *meaning* instead of words, so it recognises wordings no rule lists and handles the passive voice without grammar rules, and it is the most accurate method on the [[blind_test_set|blind test]]. The costs: it is a black box that cannot explain *why*, it needs a ~90 MB model download, and it is only as good as its training examples; a wording unlike anything it has seen is skipped instead of guessed.

### Systematic Pipeline Workflow
1. **[[named_entity_recognition|Entity Recognition]]**: Identify mentions of people, organizations, places, dates, and domain terms. When spaCy is installed, its trained statistical model finds the names, the lab's capitalisation rules add names it missed, and [[noun_chunk|noun phrases]] that act as subjects or objects (*radium*, *bacterial infections*) become TERM entities; without spaCy, only the capitalisation rules are used, so lowercase concepts are missed.
2. **[[candidate_pair|Candidate Pair Generation]]**: Identify all pairs of entities occurring within the [[word_distance|maximum word distance]] window in each sentence.
3. **Predicate Assignment**: Decide the relation name for each pair - by matching the intervening text against the [[relation_cue|relation lexicon]], by taking the nearest verb, by reading the sentence grammar, or by comparing meaning with BERT.
4. **[[triple|Triple Construction]]**: Emit structured facts with attached [[provenance]] (source sentence) and [[confidence|confidence score]].
5. **[[knowledge_graph|Graph Construction & Verification]]**: Preview the resulting graph, evaluate accuracy against [[reference_set|ground truth]], and export it as CSV, JSON or a portable [[graphml|GraphML]] graph file.

### Entity Recognition: the Foundation of Every Method
Every method above can only link entities that the [[named_entity_recognition|entity recogniser]] found, so its mistakes flow into every later step ([[error_propagation|error propagation]]): a missed entity can never appear in a triple, and a mistyped one (a city tagged PERSON) misleads the methods that use entity types.
- **Rule-based recognition** (always available): runs of capitalised words, typed by built-in lists and name shapes; anything unknown becomes PERSON. It cannot see lowercase concepts such as *radium* or *bacterial infections*.
- **Statistical recognition with spaCy** (when installed): a model trained on annotated text finds and types names, the capitalisation rules add names it missed, and [[noun_chunk|noun phrases]] that act as subjects or objects become **TERM** entities. On the sample corpora this change alone raised every method's score.
- **Still imperfect**: a capitalised word at the start of a sentence can be mistaken for a name, and spaCy sometimes mistypes names (a person tagged as an organisation).

### How the Methods Are Evaluated (and Why Honest Testing Matters)
- Each sample corpus comes with a [[reference_set|gold reference set]]; [[precision]], [[recall]] and [[f1|F1]] compare the extracted triples with it, and Pair Recall gives partial credit for the right entities with the wrong relation name.
- A method can look better than it really is when it is tested on the text used to build it. The rules in this lab were refined by studying the four sample corpora's mistakes, and the BERT classifier learned from its [[training_data|training data]], so scores on those texts are optimistic: this is [[overfitting]].
- Honest numbers need [[held_out|held-out]] data that played no part in building the method. This lab uses two kinds:
  1. **Unseen wordings**: 64 relation phrasings kept out of BERT's training (*is at the helm of*, *snapped up*, ...). The classifier gets about 7 in 10 right, against 3 in 10 for matching example clauses and almost none for a fixed word list.
  2. A **[[blind_test_set|blind test set]]** of 106 sentences: 60 real Wikipedia sentences from the Re-DocRED test split (Part A) and 46 hand-written sentences (Part B), built before the last round of changes and only ever scored as a whole.
- On the blind test every method scores far lower than on the sample corpora: real Wikipedia sentences are long and indirect, so related entities often sit far apart or the relation is only implied. The chart under *How Well Do They Really Work?* below shows the numbers.
""",
    "procedure": [
        "Step 1: Study the theoretical principles of relation extraction and examine key IR terms using the interactive notes.",
        "Step 2: Navigate to the Simulation section and choose a sample corpus (or upload/paste custom text).",
        "Step 3: Choose an extraction strategy (Co-occurrence, Pattern-Based, Hybrid, Nearest-Verb, Dependency Parse, or BERT Similarity) and adjust the confidence threshold.",
        "Step 4: Execute extraction and explore the 6 analysis tabs: Triples, Annotated Text, Entities, Graph, Flow, and Evaluation.",
        "Step 5: Inspect the interactive Knowledge Graph; toggle predicate labels and test legibility in Light and Dark modes.",
        "Step 6: Click 'Record Current Trial' to log the experiment parameters and results into your session logbook.",
        "Step 7: Complete at least 3 trials comparing different extraction methods and confidence cutoffs.",
        "Step 8: Export the extracted triples as CSV, JSON, or a GraphML graph file.",
        "Step 9: Complete the self-grading Conceptual Assessment Quiz to test your understanding.",
        "Step 10: Open Report Generation, enter your student credentials, and download your official PDF Laboratory Report."
    ]
}


# ======================================================================================
# 4. RELATION-EXTRACTION ENGINE (PURE CORE FUNCTIONS)
# ======================================================================================

_STOPWORD_CAPS = {
    "The", "A", "An", "In", "On", "At", "This", "That", "It", "He", "She",
    "They", "We", "I", "His", "Her", "Its", "Their", "From", "To", "With", "By",
    # Calendar words, clock abbreviations and generic capitalised nouns that the
    # capitalisation rule would otherwise mistake for named entities.
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "AM", "PM",
    "Past", "Average", "Interest", "Part", "Levels", "Metrics", "Result", "Results",
    "Among", "However", "Moreover", "Therefore", "Also", "Other", "Some", "Many", "Most",
    "Each", "Both", "Such", "These", "Those", "There", "Here", "When", "While", "Where",
    "Which", "What", "Who", "Why", "How", "After", "Before", "During", "Since", "Until",
    "For", "And", "But", "Or", "If", "As", "Of", "Is", "Are", "Was", "Were",
    # Job-title acronyms: "Sundar Pichai is the CEO of Google" must link the person to the company.
    "CEO", "CTO", "CFO", "COO", "CIO", "VP"
}

_ORG_SUFFIXES = (
    "Inc", "Inc.", "Ltd", "Ltd.", "Corp", "Corp.", "LLC", "University",
    "Institute", "Company", "Corporation", "Labs", "Laboratories",
    "Foundation", "Group", "Technologies", "Systems", "Council", "Hospital",
    "Organisation", "Organization"
)

_LOC_HINTS = (
    "City", "State", "Street", "River", "Mountain", "Island", "Ocean",
    "Valley", "County", "Province", "Republic", "Kingdom", "Nation"
)

# Relation lexicon: (regex pattern over connecting span) -> (canonical predicate, confidence)
_RELATION_PATTERNS = [
    (r"\b(co-?founded|founded|established|started|launched|set up)\b", "founded", 0.95),
    (r"\b(acquired|bought|purchased|took over|absorbed)\b", "acquired", 0.95),
    (r"\b(is|was|are|were)\s+(the\s+)?(ceo|founder|president|director|chairman|chairwoman|chairperson|leader|chief executive)\s+of\b", "leads", 0.92),
    (r"\b(leads|led|heads|directed|governs)\b", "leads", 0.90),
    (r"\b(works?\s+at|worked\s+at|employed\s+by|employee\s+of|serves?\s+at)\b", "works_at", 0.90),
    (r"\b(located\s+in|based\s+in|headquartered\s+in|situated\s+in|operates?\s+in)\b", "located_in", 0.90),
    (r"\b(born\s+in|native\s+of)\b", "born_in", 0.92),
    (r"\b(married\s+to|husband\s+of|wife\s+of|spouse\s+of)\b", "married_to", 0.92),
    (r"\b(developed|created|invented|designed|built|engineered)\b", "created", 0.90),
    (r"\b(discovered|isolated|identified)\b", "discovered", 0.92),
    (r"\b(published|authored|wrote|formulated)\b", "authored", 0.90),
    (r"\b(subsidiary\s+of|part\s+of|division\s+of|owned\s+by|unit\s+of)\b", "subsidiary_of", 0.90),
    (r"\b(collaborated\s+with|partnered\s+with|partnership\s+with|allied\s+with)\b", "collaborates_with", 0.88),
    (r"\b(merged\s+with|combined\s+with)\b", "merged_with", 0.90),
    (r"\b(studied\s+at|graduated\s+from|attended|alumnus\s+of)\b", "studied_at", 0.90),
    (r"\b(reports\s+to|managed\s+by|subordinate\s+to)\b", "reports_to", 0.85),
    (r"\b(competes\s+with|rival\s+of|competing\s+against)\b", "competes_with", 0.85),
    (r"\b(treats|alleviates|cures|prescribed\s+for)\b", "treats", 0.90),
    (r"\b(causes|leads\s+to|triggers|induces)\b", "causes", 0.88),
    (r"\b(is|was|are|were)\b", "is_a", 0.50),
]

_MAX_WORDS_BETWEEN = 12

# Verb lemma (or "lemma_preposition") -> the canonical predicate names used by the lexicon and
# the gold sets. Verbs not listed keep their raw lemma, so the two verb-based methods can
# still name relations the lexicon has never heard of.
_VERB_TO_CANONICAL: Dict[str, str] = {
    "found": "founded", "cofound": "founded", "establish": "founded", "start": "founded",
    "launch": "founded",
    "acquire": "acquired", "buy": "acquired", "purchase": "acquired", "absorb": "acquired",
    "lead": "leads", "head": "leads", "direct": "leads", "govern": "leads",
    "report_to": "reports_to", "manage_by": "reports_to",
    "compete_with": "competes_with", "rival_of": "competes_with",
    "base_in": "located_in", "headquarter_in": "located_in", "locate_in": "located_in",
    "situate_in": "located_in", "operate_in": "located_in",
    "bear_in": "born_in", "marry_to": "married_to",
    "develop": "created", "create": "created", "build": "created", "invent": "created",
    "design": "created", "engineer": "created",
    "discover": "discovered", "isolate": "discovered", "identify": "discovered",
    "publish": "authored", "write": "authored", "author": "authored", "formulate": "authored",
    "study_at": "studied_at", "graduate_from": "studied_at", "attend": "studied_at",
    "work_at": "works_at", "employ_by": "works_at", "serve_at": "works_at",
    "partner_with": "collaborates_with", "collaborate_with": "collaborates_with",
    "ally_with": "collaborates_with", "merge_with": "merged_with", "combine_with": "merged_with",
    "treat": "treats", "alleviate": "treats", "cure": "treats",
    "cause": "causes", "trigger": "causes", "induce": "causes",
}

# Nouns that, after "X is the ___ of Y", mean X leads Y (used by the Dependency Parse method).
_ROLE_NOUNS = {"ceo", "founder", "president", "director", "chairman", "chairwoman",
               "chairperson", "leader", "head", "executive"}


def _canonical_predicate(lemma_phrase: str) -> str:
    return _VERB_TO_CANONICAL.get(lemma_phrase, lemma_phrase)


# ---- Nearest-Verb helpers (no NLP library: plain string rules) ---------------------------

_IRREGULAR_VERB_LEMMAS: Dict[str, str] = {
    "was": "be", "were": "be", "is": "be", "are": "be", "been": "be", "has": "have",
    "had": "have", "led": "lead", "bought": "buy", "sold": "sell", "took": "take",
    "gave": "give", "wrote": "write", "built": "build", "grew": "grow", "began": "begin",
    "brought": "bring", "held": "hold", "kept": "keep", "left": "leave", "met": "meet",
    "ran": "run", "sent": "send", "spent": "spend", "stood": "stand", "won": "win",
    "chose": "choose", "spoke": "speak", "taught": "teach", "born": "bear", "made": "make",
    "found": "find", "knew": "know", "became": "become", "paid": "pay", "set": "set",
}

_AUXILIARY_LEMMAS = {"be", "have", "do"}

_VERB_EXCLUDE_WORDS = {
    "its", "this", "his", "hers", "ours", "yours", "theirs", "us", "yes", "plus", "thus",
    "status", "process", "focus", "across", "various", "previous", "always", "also", "as",
    "less", "unless", "business", "class", "press", "address", "series", "species",
    "nothing", "something", "everything", "during", "morning", "evening", "king", "ring",
    "thing", "cloud", "computing", "needs", "news", "mathematics", "physics", "economics",
}

_TRAILING_PREPOSITIONS = {"of", "to", "with", "in", "at", "from", "by", "for"}

_VOWELS = "aeiou"


def _lemmatize_verb(word: str) -> str:
    """Rule-based reduction of an inflected verb to its base form (approximate, no dictionary)."""
    w = word.lower()
    if w in _IRREGULAR_VERB_LEMMAS:
        return _IRREGULAR_VERB_LEMMAS[w]

    def pick(*candidates: str) -> str:
        for c in candidates:
            if c in _VERB_TO_CANONICAL or any(k.startswith(c + "_") for k in _VERB_TO_CANONICAL):
                return c
        stem = candidates[0]
        if stem.endswith(("g", "v", "c", "z", "ur", "ir", "iz", "at")):
            return stem + "e"
        return stem

    if w.endswith("ied") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ed") and len(w) > 4:
        stem = w[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in _VOWELS + "lsz":
            stem = stem[:-1]
        return pick(stem, w[:-1])
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in _VOWELS + "lsz":
            stem = stem[:-1]
        return pick(stem, stem + "e")
    if w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _looks_like_verb(token: str) -> bool:
    if not token.islower() or token in _VERB_EXCLUDE_WORDS or token in _TRAILING_PREPOSITIONS:
        return False
    if token in _IRREGULAR_VERB_LEMMAS:
        return True
    return len(token) >= 4 and token.endswith(("ed", "ing", "s")) and not token.endswith("ss")


def _find_nearest_verb(span: str) -> Optional[Tuple[str, str]]:
    """Returns (canonical_predicate, surface_text) for the first verb-like word in `span`."""
    tokens = [t for t in re.split(r"[\s,;:()]+", span.strip()) if t]
    for i, raw in enumerate(tokens):
        token = re.sub(r"^co-", "", raw.lower())
        if not _looks_like_verb(token):
            continue
        lemma = _lemmatize_verb(token)
        if lemma in _AUXILIARY_LEMMAS:
            continue
        surface = raw
        phrase = lemma
        if i + 1 < len(tokens) and tokens[i + 1].lower() in _TRAILING_PREPOSITIONS:
            phrase = f"{lemma}_{tokens[i + 1].lower()}"
            surface = f"{raw} {tokens[i + 1]}"
        predicate = _VERB_TO_CANONICAL.get(phrase) or _VERB_TO_CANONICAL.get(lemma) or phrase
        return predicate, surface
    return None


def _split_sentences(text: str) -> List[str]:
    """Splits document text into clean sentences while respecting common honorifics and abbreviations."""
    clean = text.strip().replace("\r\n", " ").replace("\n", " ")
    clean = re.sub(r"\b(Dr|Prof|Mr|Mrs|Ms|vs|Inc|Ltd|Corp|e\.g|i\.e)\.", r"\1<DOT>", clean)
    parts = re.split(r"(?<=[.!?])\s+", clean)
    sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return sentences


_KNOWN_LOCATIONS = {
    "california", "mountain view", "redmond", "warsaw", "paris", "london",
    "mumbai", "bengaluru", "chennai", "delhi", "tokyo", "new york", "san francisco",
    "india", "france", "poland", "usa", "switzerland", "cambridge", "oxford"
}
_KNOWN_ORGS = {
    "google", "alphabet", "microsoft", "youtube", "linkedin", "openai",
    "infosys", "tata group", "tata consultancy services", "curie institute",
    "university of paris", "indian space research organisation", "isro",
    "stanford university", "harvard", "mit", "who", "pfizer", "apple", "amazon"
}


def _rule_entity_type(span_text: str) -> Optional[str]:
    """Type from the lab's own lists and name-shape rules, or None when they don't decide."""
    lowered = span_text.lower()
    if lowered in _KNOWN_LOCATIONS or any(hint in span_text for hint in _LOC_HINTS):
        return "LOCATION"
    if lowered in _KNOWN_ORGS or any(span_text.endswith(sfx) for sfx in _ORG_SUFFIXES):
        return "ORG"
    if re.search(r"\b(University|Institute|College|Academy|Hospital|Board)\b", span_text):
        return "ORG"
    return None


def _extract_entities_rules(text: str) -> List[Dict[str, Any]]:
    """Rule-based NER (no library): runs of capitalised words, typed by lists and name shape
    (anything undecided becomes PERSON), plus 4-digit years as DATE. Used on its own when spaCy
    is not installed, and as a second source alongside spaCy when it is."""
    sentences = _split_sentences(text)
    entities = []
    entity_id = 1

    # Words that also occur in lower case somewhere in the document are ordinary words
    # ("Average users..." / "the average"), so a lone capitalised one at the start of a
    # sentence is capitalised only because of its position, not because it is a name.
    lowercase_words = set(re.findall(r"[a-z]+", text))

    for sent_idx, sentence in enumerate(sentences):
        # Match capitalized phrases
        # [A-Za-z]* (not just [a-z]*) so brand names with an internal capital, such as
        # "YouTube", "LinkedIn" or "OpenAI", are recognized as a single entity instead of
        # being silently skipped.
        for match in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", sentence):
            span_text = match.group(1).strip()
            span_start = match.start(1)
            # A sentence-initial article is capitalised only by position: "The Tata Group" -> "Tata Group".
            lead = span_text.split(" ", 1)
            if len(lead) == 2 and lead[0] in ("The", "A", "An"):
                span_start += len(lead[0]) + 1
                span_text = lead[1]
            if span_text in _STOPWORD_CAPS:
                continue
            if (match.start() == 0 and " " not in span_text
                    and span_text.lower() in lowercase_words):
                continue

            label = _rule_entity_type(span_text) or "PERSON"

            entities.append({
                "id": f"e{entity_id}",
                "text": span_text,
                "label": label,
                "start": span_start,
                "end": match.end(),
                "sentence_idx": sent_idx,
                "sentence": sentence
            })
            entity_id += 1

        # Match 4-digit years as DATE
        for match in re.finditer(r"\b(19\d\d|20\d\d)\b", sentence):
            year_text = match.group(1)
            entities.append({
                "id": f"e{entity_id}",
                "text": year_text,
                "label": "DATE",
                "start": match.start(),
                "end": match.end(),
                "sentence_idx": sent_idx,
                "sentence": sentence
            })
            entity_id += 1

    # Remove duplicates within identical character spans
    seen_spans = set()
    unique_entities = []
    for ent in entities:
        key = (ent["sentence_idx"], ent["start"], ent["end"])
        if key not in seen_spans:
            seen_spans.add(key)
            unique_entities.append(ent)

    return unique_entities


@st.cache_resource(show_spinner=False)
def _load_spacy_parser():
    """Loads spaCy's small English model, or returns None when spaCy/the model is missing."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def spacy_parser_available() -> bool:
    return _load_spacy_parser() is not None


SPACY_INSTALL_HINT = "pip install spacy && python -m spacy download en_core_web_sm"

_SPACY_LABEL_MAP = {
    "PERSON": "PERSON", "ORG": "ORG", "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION",
    "PRODUCT": "MISC", "WORK_OF_ART": "MISC", "EVENT": "MISC", "LAW": "MISC", "NORP": "MISC",
}
# Noun-phrase subjects / objects that are concepts ("radium", "bacterial infections") become TERM
# entities - but not generic words that only refer back to something already named.
_TERM_DEPS = {"nsubj", "nsubjpass", "dobj", "pobj", "conj"}
_GENERIC_TERM_HEADS = {
    "company", "firm", "business", "board", "husband", "wife", "chairman", "chairperson", "ceo",
    "system", "year", "time", "people", "person", "team", "deal", "market", "way", "part", "kind",
    "number", "member", "country", "city", "state", "town", "work", "product", "service",
    "organisation", "organization", "program", "programme", "use", "lot", "thing", "case", "result",
    "study", "report", "information", "fact", "day", "week", "month", "decade", "century", "rest",
    "side", "end", "interest", "level", "metric", "average", "past", "group", "head", "leader",
    "user", "customer", "student", "share", "stake", "money", "billion", "million", "percent",
    "executive", "director", "president", "manager", "founder", "officer",
}
_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "some", "any", "each", "every",
                "no", "another", "such", "its", "their"}


def _term_spans(doc) -> List[Tuple[int, int, str]]:
    """(start_char, end_char, text) of concept noun phrases in one parsed sentence."""
    spans = []
    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.pos_ != "NOUN" or root.dep_ not in _TERM_DEPS:
            continue
        # capitalised only because it starts the sentence ("Penicillin treats ...") is still a concept
        if not (root.text.islower() or chunk.start == 0):
            continue
        if root.lemma_.lower() in _GENERIC_TERM_HEADS:
            continue
        tokens = list(chunk)
        while tokens and tokens[0].dep_ == "poss":
            tokens = tokens[1:]
        if not tokens or tokens[0].lower_ in _DETERMINERS or tokens[0].pos_ == "PRON":
            continue
        if any(t.pos_ == "PROPN" for t in tokens):
            continue
        end_tok = tokens[-1]
        for child in root.children:   # attach an "of" phrase: "theory of relativity"
            if child.dep_ == "prep" and child.lower_ == "of" and child.i == end_tok.i + 1:
                pobj = [c for c in child.children if c.dep_ == "pobj"]
                if pobj and pobj[0].pos_ in ("NOUN", "PROPN"):
                    end_tok = pobj[0]
        start, end = tokens[0].idx, end_tok.idx + len(end_tok.text)
        spans.append((start, end, doc.text[start:end]))
    return spans


def _extract_entities_spacy(text: str, nlp) -> List[Dict[str, Any]]:
    """spaCy-based NER. Merges three sources per sentence (earlier sources win on overlap):
    spaCy's statistical named entities, the lab's capitalised-name rules for names spaCy missed,
    and TERM concepts from noun phrases. Years are added as DATE, as in the rule finder."""
    sentences = _split_sentences(text)
    rule_ents: Dict[int, List[Dict[str, Any]]] = {}
    for ent in _extract_entities_rules(text):
        rule_ents.setdefault(ent["sentence_idx"], []).append(ent)

    entities: List[Dict[str, Any]] = []
    for sent_idx, (sentence, doc) in enumerate(zip(sentences, nlp.pipe(sentences))):
        taken: List[Tuple[int, int]] = []

        def free(a: int, b: int) -> bool:
            return all(b <= x or a >= y for x, y in taken)

        def add(a: int, b: int, label: str) -> None:
            taken.append((a, b))
            entities.append({"id": f"e{len(entities) + 1}", "text": sentence[a:b], "label": label,
                             "start": a, "end": b, "sentence_idx": sent_idx, "sentence": sentence})

        for ent in doc.ents:
            if ent.label_ not in _SPACY_LABEL_MAP:
                continue
            toks = [t for t in ent]
            while toks and toks[0].lower_ in ("the", "a", "an"):
                toks = toks[1:]
            while toks and toks[-1].text in ("'s", "'"):
                toks = toks[:-1]
            if not toks:
                continue
            a, b = toks[0].idx, toks[-1].idx + len(toks[-1].text)
            span_text = sentence[a:b]
            if span_text in _STOPWORD_CAPS or not free(a, b):
                continue
            add(a, b, _rule_entity_type(span_text) or _SPACY_LABEL_MAP[ent.label_])

        for ent in rule_ents.get(sent_idx, []):
            if not free(ent["start"], ent["end"]):
                continue
            if ent["label"] != "DATE":
                # A capitalised word spaCy reads as an ordinary word ("Chronic smoking ...",
                # "Penicillin treats ...") is capitalised by position, not because it is a name.
                span = doc.char_span(ent["start"], ent["end"], alignment_mode="expand")
                if span is not None and not any(t.pos_ == "PROPN" for t in span):
                    continue
            add(ent["start"], ent["end"], ent["label"])

        for a, b, _ in _term_spans(doc):
            if free(a, b):
                add(a, b, "TERM")

    return entities


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Named-entity recognition for every method in the lab: spaCy's statistical model plus the
    lab's rules when spaCy is installed, the rules alone otherwise. Same output either way."""
    nlp = _load_spacy_parser()
    if nlp is None:
        return _extract_entities_rules(text)
    return _extract_entities_spacy(text, nlp)


def extract_svo_relationships(text: str, entities: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Dependency-parse extraction: finds who did what to whom from grammatical roles.

    Reuses this lab's own entities (so every method shares identical entities) and uses
    spaCy only to read the sentence grammar. Returns [] if the parser is unavailable.
    """
    nlp = _load_spacy_parser()
    if nlp is None:
        return []
    if entities is None:
        entities = extract_entities(text)

    by_sentence: Dict[int, List[Dict[str, Any]]] = {}
    for ent in entities:
        by_sentence.setdefault(ent["sentence_idx"], []).append(ent)

    triples: List[Dict[str, Any]] = []
    seen = set()

    for sent_idx in sorted(by_sentence):
        sent_ents = by_sentence[sent_idx]
        if len(sent_ents) < 2:
            continue
        sentence = sent_ents[0]["sentence"]
        # "co-founded" parses badly; blank the prefix with same-length spaces to keep offsets.
        doc = nlp(re.sub(r"\b[Cc]o-(?=[a-z])", "   ", sentence))

        tok_to_ent: Dict[int, Dict[str, Any]] = {}
        for ent in sent_ents:
            span = doc.char_span(ent["start"], ent["end"], alignment_mode="expand")
            if span is None:
                continue
            tok = span.root
            while tok.dep_ == "compound" and tok.head is not tok:
                tok = tok.head
            tok_to_ent.setdefault(tok.i, ent)

        def ents_of(tokens) -> List[Dict[str, Any]]:
            found = []
            for t in tokens:
                for cand in [t] + list(t.conjuncts):
                    if cand.i in tok_to_ent and tok_to_ent[cand.i] not in found:
                        found.append(tok_to_ent[cand.i])
            return found

        def subjects_of(verb, depth: int = 0):
            own = [c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")]
            if own:
                return own
            if depth < 3 and verb.dep_ == "conj" and verb.head is not verb:
                return subjects_of(verb.head, depth + 1)
            if verb.dep_ in ("acl", "relcl"):
                return [verb.head]
            return []

        def emit(subj, pred, obj, conf, cue):
            if subj["text"].lower() == obj["text"].lower():
                return
            key = (subj["text"].lower(), pred, obj["text"].lower())
            if key in seen:
                return
            seen.add(key)
            triples.append({
                "id": f"t{len(triples) + 1}",
                "subject": subj["text"], "subject_label": subj["label"],
                "predicate": pred,
                "object": obj["text"], "object_label": obj["label"],
                "confidence": conf, "cue": cue,
                "method": "Dependency Parse",
                "sentence": sentence, "sentence_idx": sent_idx
            })

        def is_verb_like(tok) -> bool:
            if tok.pos_ in ("VERB", "AUX"):
                return True
            # The tagger sometimes calls a coordinated verb a noun ("... and reports to ...");
            # accept it when its head is a verb and the lemma is a verb we recognise.
            lem = tok.lemma_.lower()
            return (tok.dep_ == "conj" and tok.head.pos_ in ("VERB", "AUX")
                    and (lem in _VERB_TO_CANONICAL
                         or any(k.startswith(lem + "_") for k in _VERB_TO_CANONICAL)))

        for verb in doc:
            if not is_verb_like(verb):
                continue
            subj_toks = subjects_of(verb)
            if not subj_toks:
                continue
            subj_ents = ents_of(subj_toks)
            if not subj_ents:
                continue
            passive = any(c.dep_ == "nsubjpass" for c in verb.children) or (
                not any(c.dep_ in ("nsubj", "nsubjpass") for c in verb.children)
                and any(t.dep_ == "nsubjpass" for t in subj_toks))
            lemma = verb.lemma_.lower()
            surface = verb.text

            if lemma == "be":
                for attr in (c for c in verb.children if c.dep_ == "attr"):
                    for prep in (c for c in attr.children if c.dep_ == "prep" and c.text.lower() == "of"):
                        objs = ents_of([c for c in prep.children if c.dep_ == "pobj"])
                        pred = "leads" if attr.lemma_.lower() in _ROLE_NOUNS else f"{attr.lemma_.lower()}_of"
                        for s in subj_ents:
                            for o in objs:
                                emit(s, pred, o, 0.80, f"nsubj → is {attr.text} of → pobj")
                continue

            # Direct objects: (subject, verb, object)
            for o in ents_of([c for c in verb.children if c.dep_ == "dobj"]):
                for s in subj_ents:
                    emit(s, _canonical_predicate(lemma), o, 0.85, f"nsubj → {surface} → dobj")

            for prep in (c for c in verb.children if c.dep_ in ("prep", "agent")):
                objs = ents_of([c for c in prep.children if c.dep_ == "pobj"])
                if prep.dep_ == "agent" or (passive and prep.text.lower() == "by"):
                    # Passive: "YouTube was acquired by Google" -> (Google, acquired, YouTube)
                    for agent in objs:
                        for patient in subj_ents:
                            emit(agent, _canonical_predicate(lemma), patient, 0.85,
                                 f"nsubjpass ← {surface} → agent (passive, direction swapped)")
                    continue
                phrase = f"{lemma}_{prep.text.lower()}"
                mapped = phrase in _VERB_TO_CANONICAL
                for o in objs:
                    if not mapped and o["label"] in ("DATE", "LOCATION"):
                        continue
                    for s in subj_ents:
                        emit(s, _canonical_predicate(phrase), o, 0.80, f"nsubj → {surface} {prep.text} → pobj")

    return triples


BERT_METHOD = "BERT Similarity (Sentence-BERT)"
BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BERT_INSTALL_HINT = "pip install sentence-transformers"
# A clause is accepted when its best predicate scores at least this cosine similarity AND beats
# the runner-up predicate by BERT_MIN_MARGIN (so near-ties between unrelated relations are dropped).
BERT_MIN_SIMILARITY = 0.70
BERT_MIN_MARGIN = 0.04
# The trained classifier (bert_re_classifier.npz, built by train_bert_re.py) is preferred when present.
# A clause is accepted when its most likely class is a relation (not NONE) with at least this probability.
BERT_MIN_PROBABILITY = 0.50
BERT_CLASSIFIER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_re_classifier.npz")

# Example clauses for every predicate, written with X = first entity and Y = second entity in
# reading order. The flag marks clauses where the real subject is the SECOND entity (passive
# voice), so the triple's direction is swapped.  (template, predicate, subject_is_second)
_BERT_PROTOTYPES: List[Tuple[str, str, bool]] = [
    ("X founded Y", "founded", False), ("X co-founded Y", "founded", False),
    ("X established Y", "founded", False), ("X started Y", "founded", False),
    ("X launched Y", "founded", False), ("X was founded by Y", "founded", True),
    ("X acquired Y", "acquired", False), ("X bought Y", "acquired", False),
    ("X purchased Y", "acquired", False), ("X took over Y", "acquired", False),
    ("X was acquired by Y", "acquired", True), ("X was bought by Y", "acquired", True),
    ("X is the CEO of Y", "leads", False), ("X is the chairman of Y", "leads", False),
    ("X is the president of Y", "leads", False), ("X leads Y", "leads", False),
    ("X heads Y", "leads", False), ("X is the leader of Y", "leads", False),
    ("X works at Y", "works_at", False), ("X is employed by Y", "works_at", False),
    ("X is an employee of Y", "works_at", False),
    ("X is located in Y", "located_in", False), ("X is based in Y", "located_in", False),
    ("X is headquartered in Y", "located_in", False), ("X, based in Y", "located_in", False),
    ("X is situated in Y", "located_in", False), ("X operates in Y", "located_in", False),
    ("X was born in Y", "born_in", False), ("X is a native of Y", "born_in", False),
    ("X is married to Y", "married_to", False), ("X is the husband of Y", "married_to", False),
    ("X is the wife of Y", "married_to", False),
    ("X developed Y", "created", False), ("X created Y", "created", False),
    ("X invented Y", "created", False), ("X designed Y", "created", False),
    ("X built Y", "created", False), ("X was created by Y", "created", True),
    ("X discovered Y", "discovered", False), ("X isolated Y", "discovered", False),
    ("X identified Y", "discovered", False),
    ("X published Y", "authored", False), ("X wrote Y", "authored", False),
    ("X authored Y", "authored", False), ("X was written by Y", "authored", True),
    ("X is a subsidiary of Y", "subsidiary_of", False), ("X is part of Y", "subsidiary_of", False),
    ("X is a division of Y", "subsidiary_of", False), ("X is owned by Y", "subsidiary_of", False),
    ("X collaborated with Y", "collaborates_with", False),
    ("X partnered with Y", "collaborates_with", False),
    ("X is allied with Y", "collaborates_with", False),
    ("X merged with Y", "merged_with", False), ("X combined with Y", "merged_with", False),
    ("X studied at Y", "studied_at", False), ("X graduated from Y", "studied_at", False),
    ("X attended Y", "studied_at", False),
    ("X reports to Y", "reports_to", False), ("X is managed by Y", "reports_to", False),
    ("X competes with Y", "competes_with", False), ("X is a rival of Y", "competes_with", False),
    ("X is competing against Y", "competes_with", False),
    ("X treats Y", "treats", False), ("X cures Y", "treats", False),
    ("X alleviates Y", "treats", False),
    ("X causes Y", "causes", False), ("X leads to Y", "causes", False),
    ("X triggers Y", "causes", False), ("X induces Y", "causes", False),
]


@st.cache_resource(show_spinner="Loading the BERT model (first run downloads about 90 MB)...")
def _load_bert_encoder():
    """Loads the Sentence-BERT model, or returns None when the library/model is unavailable."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer(BERT_MODEL_NAME, device=device)
    except Exception:
        return None


def bert_available() -> bool:
    return _load_bert_encoder() is not None


def _bert_encode(texts: Sequence[str]) -> Optional[np.ndarray]:
    """Unit-length BERT embeddings (one row per text), or None if BERT is unavailable."""
    encoder = _load_bert_encoder()
    if encoder is None or not texts:
        return None
    return np.asarray(encoder.encode(list(texts), normalize_embeddings=True,
                                     show_progress_bar=False, batch_size=64))


@st.cache_resource(show_spinner=False)
def _bert_prototype_embeddings() -> Optional[np.ndarray]:
    return _bert_encode([t for t, _, _ in _BERT_PROTOTYPES])


def bert_score_clauses(clauses: Sequence[str]) -> Optional[np.ndarray]:
    """Cosine similarity of each clause to every prototype: shape (clauses, prototypes)."""
    protos = _bert_prototype_embeddings()
    vecs = _bert_encode(clauses)
    if protos is None or vecs is None:
        return None
    return vecs @ protos.T


def bert_rank_predicates(sim_row: np.ndarray) -> List[Tuple[str, float, int]]:
    """Best prototype per predicate for one clause, highest similarity first.
    Each item is (predicate, similarity, prototype_index)."""
    best: Dict[str, Tuple[float, int]] = {}
    for idx, (_, predicate, _) in enumerate(_BERT_PROTOTYPES):
        sim = float(sim_row[idx])
        if predicate not in best or sim > best[predicate][0]:
            best[predicate] = (sim, idx)
    return sorted(((pred, sim, idx) for pred, (sim, idx) in best.items()), key=lambda r: -r[1])


@st.cache_resource(show_spinner=False)
def _load_bert_classifier():
    """The small relation classifier trained on top of BERT embeddings, or None if absent."""
    try:
        data = np.load(BERT_CLASSIFIER_PATH, allow_pickle=False)
        return {
            "classes": [str(c) for c in data["classes"]],
            "coef": data["coef"].astype(np.float64),
            "intercept": data["intercept"].astype(np.float64),
            "types": [str(t) for t in data["type_order"]],
        }
    except Exception:
        return None


def bert_classifier_available() -> bool:
    return _load_bert_classifier() is not None and bert_available()


def bert_classify_clauses(
    clauses: Sequence[str], first_types: Sequence[str], second_types: Sequence[str]
) -> Optional[Tuple[List[str], np.ndarray]]:
    """Class probabilities for each clause: returns (class_names, probabilities[n_clauses, n_classes])."""
    model = _load_bert_classifier()
    vecs = _bert_encode(clauses)
    if model is None or vecs is None:
        return None
    types = model["types"]
    onehot = np.zeros((len(clauses), 2 * len(types)))
    for i, (t1, t2) in enumerate(zip(first_types, second_types)):
        t1, t2 = ("MISC" if t1 == "TERM" else t1), ("MISC" if t2 == "TERM" else t2)
        onehot[i, types.index(t1) if t1 in types else types.index("MISC")] = 1.0
        onehot[i, len(types) + (types.index(t2) if t2 in types else types.index("MISC"))] = 1.0
    logits = np.hstack([vecs, onehot]) @ model["coef"].T + model["intercept"]
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return model["classes"], probs


def _bert_class_to_relation(class_name: str) -> Tuple[str, bool]:
    """'acquired~rev' -> ('acquired', True): the second entity is the subject."""
    return (class_name[:-4], True) if class_name.endswith("~rev") else (class_name, False)


_BERT_ENTITY_SEPARATOR = re.compile(r"(?:,|&|and|or|,\s*and|,\s*or)?", re.IGNORECASE)
_BERT_CONJUNCTION = re.compile(r"(?:^|[\s,])(?:and|but)\s+(?=\S)|;\s*", re.IGNORECASE)


def _bert_candidate_clauses(
    sentence: str, sent_ents: List[Dict[str, Any]], max_word_distance: int
) -> List[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    """(subject_candidate, object_candidate, clause) triples for one sentence.

    Plain "adjacent entity" pairing attaches relations to the wrong entity, so this adds the
    grammar-light rules a reader applies without thinking:
      * years are skipped (they are not relation arguments);
      * entities joined only by "and" / "," ("Larry Page and Sergey Brin") form one coordinated
        group, and every member is paired with the next group;
      * "A acquired B and partnered with C": the part after "and" is judged on its own, and A,
        the subject of the sentence so far, is its subject;
      * "Microsoft, based in Redmond, competes with Google": after a comma-delimited aside, the
        subject is the entity before the aside.
    """
    ents = sorted((e for e in sent_ents if e["label"] != "DATE"), key=lambda e: e["start"])
    groups: List[List[Dict[str, Any]]] = []
    for e in ents:
        if groups:
            last = groups[-1][-1]
            gap = sentence[last["end"]:e["start"]].strip()
            if last["label"] == e["label"] and gap and _BERT_ENTITY_SEPARATOR.fullmatch(gap):
                groups[-1].append(e)
                continue
        groups.append([e])

    candidates = []
    previous_subject = None
    pending_owner = None
    for left, right in zip(groups, groups[1:]):
        between = sentence[left[-1]["end"]:right[0]["start"]].strip()
        if not between or len(between.split()) > max_word_distance:
            continue
        subject_group, clause_text = left, between
        conj = list(_BERT_CONJUNCTION.finditer(between))
        if conj:
            clause_text = between[conj[-1].end():].strip()
            if previous_subject is not None:
                subject_group = previous_subject
        elif between.startswith(",") and pending_owner is not None:
            subject_group = pending_owner
        pending_owner = left if between.startswith(",") else None
        previous_subject = subject_group
        clause_text = clause_text.lstrip(", ").strip()
        if not clause_text:
            continue
        for subj in subject_group:
            for obj in right:
                if subj["text"].lower() != obj["text"].lower():
                    candidates.append((subj, obj, f"X {clause_text} Y"))
    return candidates


def extract_bert_relationships(
    text: str,
    entities: Optional[List[Dict[str, Any]]] = None,
    max_word_distance: int = _MAX_WORDS_BETWEEN,
) -> List[Dict[str, Any]]:
    """BERT extraction: judges the MEANING of the clause between two entities. Entity names are
    masked as X / Y so only the relationship is judged. A small classifier trained on BERT
    embeddings (plus the two entity types) picks the relation, or NONE; if that trained file is
    absent it falls back to matching example clauses by cosine similarity.
    Returns [] if BERT is unavailable."""
    if not bert_available():
        return []
    if entities is None:
        entities = extract_entities(text)

    by_sentence: Dict[int, List[Dict[str, Any]]] = {}
    for ent in entities:
        by_sentence.setdefault(ent["sentence_idx"], []).append(ent)

    candidates = []
    for sent_idx in sorted(by_sentence):
        sent_ents = by_sentence[sent_idx]
        sentence = sent_ents[0]["sentence"]
        for subj, obj, clause in _bert_candidate_clauses(sentence, sent_ents, max_word_distance):
            candidates.append((sent_idx, sentence, subj, obj, clause))
    if not candidates:
        return []

    clauses = [c[4] for c in candidates]
    triples: List[Dict[str, Any]] = []

    def emit(sent_idx, sentence, subj, obj, predicate, swap, confidence, cue):
        first, second = (obj, subj) if swap else (subj, obj)
        triples.append({
            "id": f"t{len(triples) + 1}",
            "subject": first["text"], "subject_label": first["label"],
            "predicate": predicate,
            "object": second["text"], "object_label": second["label"],
            "confidence": round(min(confidence, 0.99), 2),
            "cue": cue, "method": "BERT Similarity",
            "sentence": sentence, "sentence_idx": sent_idx
        })

    if bert_classifier_available():
        scored = bert_classify_clauses(clauses, [c[2]["label"] for c in candidates],
                                       [c[3]["label"] for c in candidates])
        if scored is None:
            return []
        class_names, probs = scored
        for (sent_idx, sentence, subj, obj, clause), row in zip(candidates, probs):
            best = int(np.argmax(row))
            if class_names[best] == "NONE" or row[best] < BERT_MIN_PROBABILITY:
                continue
            predicate, swap = _bert_class_to_relation(class_names[best])
            emit(sent_idx, sentence, subj, obj, predicate, swap, float(row[best]),
                 f"'{clause}' -> {predicate} (p={row[best]:.2f})")
        return triples

    sims = bert_score_clauses(clauses)
    if sims is None:
        return []
    for (sent_idx, sentence, subj, obj, clause), row in zip(candidates, sims):
        ranked = bert_rank_predicates(row)
        top_pred, top_sim, proto_idx = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if top_sim < BERT_MIN_SIMILARITY or (top_sim - runner_up) < BERT_MIN_MARGIN:
            continue
        template, _, swap = _BERT_PROTOTYPES[proto_idx]
        emit(sent_idx, sentence, subj, obj, top_pred, swap, top_sim,
             f"'{clause}' ~ '{template}' ({top_sim:.2f})")
    return triples


def extract_relationships(
    text: str,
    entities: Optional[List[Dict[str, Any]]] = None,
    method: str = "Hybrid (Pattern + Co-occurrence)",
    max_word_distance: int = _MAX_WORDS_BETWEEN
) -> List[Dict[str, Any]]:
    """Extracts Subject-Predicate-Object triples over sentence entity pairs."""
    if entities is None:
        entities = extract_entities(text)

    if "Dependency Parse" in method:
        return extract_svo_relationships(text, entities)
    if "BERT" in method:
        return extract_bert_relationships(text, entities, max_word_distance)

    # Group entities by sentence
    by_sentence: Dict[int, List[Dict[str, Any]]] = {}
    for ent in entities:
        by_sentence.setdefault(ent["sentence_idx"], []).append(ent)

    triples = []
    triple_id = 1

    for sent_idx, sent_ents in by_sentence.items():
        if len(sent_ents) < 2:
            continue
        # Sort by occurrence order in the sentence
        sorted_ents = sorted(sent_ents, key=lambda e: e["start"])

        # Only pair adjacent entity mentions (no other entity in between). Pairing every
        # combination in the sentence lets a relation cue between two OTHER entities get
        # wrongly attributed across an intervening entity - e.g. in "Larry Page and Sergey
        # Brin co-founded Google in California in 1998", pairing Larry Page with California
        # or 1998 directly would (wrongly) match the "co-founded" cue in between them and
        # emit a spurious high-confidence "Larry Page -> founded -> California" triple.
        # Restricting pattern/co-occurrence pairing to adjacent mentions keeps every emitted
        # triple tied to the actual local relation cue.
        for i in range(len(sorted_ents) - 1):
                subj = sorted_ents[i]
                obj = sorted_ents[i + 1]

                # If an entity is identical in string, skip self-links
                if subj["text"].lower() == obj["text"].lower():
                    continue

                sentence = subj["sentence"]
                span_between = sentence[subj["end"]:obj["start"]].strip()
                words_between = len(span_between.split())

                if words_between > max_word_distance:
                    continue

                matched_predicate = None
                confidence = 0.0
                matched_cue = None

                # Pattern matching over span
                for pattern, pred, conf in _RELATION_PATTERNS:
                    m = re.search(pattern, span_between, re.IGNORECASE)
                    if m:
                        matched_predicate = pred
                        confidence = conf
                        matched_cue = m.group(0)
                        break

                is_hybrid = "Hybrid" in method
                is_pattern = "Pattern" in method
                is_cooc = "Co-occurrence" in method and not is_hybrid

                if matched_predicate and (is_pattern or is_hybrid):
                    # Type constraint heuristic validation
                    if matched_predicate == "born_in" and subj["label"] != "PERSON":
                        confidence *= 0.6
                    if matched_predicate == "leads" and obj["label"] not in ("ORG", "LOCATION"):
                        confidence *= 0.75

                    triples.append({
                        "id": f"t{triple_id}",
                        "subject": subj["text"],
                        "subject_label": subj["label"],
                        "predicate": matched_predicate,
                        "object": obj["text"],
                        "object_label": obj["label"],
                        "confidence": round(confidence, 2),
                        "cue": matched_cue,
                        "method": "Pattern-Based",
                        "sentence": sentence,
                        "sentence_idx": sent_idx
                    })
                    triple_id += 1

                elif (is_cooc or is_hybrid) and not matched_predicate:
                    # Fallback co-occurrence relation
                    cooc_conf = max(0.20, round(0.45 - 0.02 * words_between, 2))
                    triples.append({
                        "id": f"t{triple_id}",
                        "subject": subj["text"],
                        "subject_label": subj["label"],
                        "predicate": "related_to",
                        "object": obj["text"],
                        "object_label": obj["label"],
                        "confidence": cooc_conf,
                        "cue": None,
                        "method": "Co-occurrence",
                        "sentence": sentence,
                        "sentence_idx": sent_idx
                    })
                    triple_id += 1

                elif "Nearest-Verb" in method:
                    verb_hit = _find_nearest_verb(span_between)
                    if verb_hit:
                        verb_pred, verb_cue = verb_hit
                        triples.append({
                            "id": f"t{triple_id}",
                            "subject": subj["text"],
                            "subject_label": subj["label"],
                            "predicate": verb_pred,
                            "object": obj["text"],
                            "object_label": obj["label"],
                            "confidence": 0.60,
                            "cue": verb_cue,
                            "method": "Nearest-Verb",
                            "sentence": sentence,
                            "sentence_idx": sent_idx
                        })
                        triple_id += 1

    return triples


def compute_extraction_metrics(entities: List[Dict[str, Any]], triples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summary statistics for trial logging, display metrics, and reporting."""
    n_entities = len({e["text"].lower() for e in entities})
    n_triples = len(triples)
    high_conf = sum(1 for t in triples if t["confidence"] >= 0.80)
    avg_conf = round(float(np.mean([t["confidence"] for t in triples])), 2) if triples else 0.0
    density = round(n_triples / n_entities, 2) if n_entities else 0.0
    n_predicates = len({t["predicate"] for t in triples})
    return {
        "entities": n_entities,
        "triples": n_triples,
        "high_confidence_triples": high_conf,
        "avg_confidence": avg_conf,
        "relation_density": density,
        "unique_predicates": n_predicates
    }


def evaluate_against_reference_set(
    extracted_triples: List[Dict[str, Any]],
    gold_triples: List[Tuple[str, str, str]]
) -> Dict[str, Any]:
    """Evaluates extracted triples against a human ground truth reference set.

    Precision/Recall/F1 require an *exact* match (subject, predicate, object): finding the
    right two entities with the wrong (or a generic) predicate scores zero credit. This is
    the standard, strict definition - but it also means a method that never predicts a
    specific predicate (Co-occurrence always emits the generic `related_to`) is
    mathematically guaranteed to score 0.0 on all three, on every corpus, regardless of how
    many of the right entity pairs it actually found. Pair Recall reports that separately:
    the fraction of gold (subject, object) connections found at all, ignoring the predicate
    label - the fairer number for judging a method that deliberately doesn't attempt to name
    the relationship.
    """
    gold_pairs = {(g[0].lower().strip(), g[2].lower().strip()) for g in gold_triples}
    extracted_pairs = {(t["subject"].lower().strip(), t["object"].lower().strip()) for t in extracted_triples}
    pair_hits = len(gold_pairs & extracted_pairs)
    pair_recall = round(pair_hits / len(gold_pairs), 3) if gold_pairs else 0.0

    if not gold_triples:
        return {
            "has_gold": False,
            "gold_count": 0,
            "extracted_count": len(extracted_triples),
            "tp_count": 0,
            "fp_count": len(extracted_triples),
            "fn_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "pair_recall": 0.0,
            "tp_list": [],
            "fp_list": extracted_triples,
            "fn_list": []
        }

    # Normalize tuples for robust matching
    def norm_triple(s: str, p: str, o: str) -> Tuple[str, str, str]:
        p_norm = p.lower().replace(" ", "_")
        if p_norm in ("co-founded", "started", "established"):
            p_norm = "founded"
        elif p_norm in ("bought", "purchased", "took_over"):
            p_norm = "acquired"
        elif p_norm in ("headquartered_in", "based_in"):
            p_norm = "located_in"
        return (s.lower().strip(), p_norm, o.lower().strip())

    gold_set = {norm_triple(g[0], g[1], g[2]): g for g in gold_triples}
    matched_gold = set()

    tp_list = []
    fp_list = []

    for ext in extracted_triples:
        key = norm_triple(ext["subject"], ext["predicate"], ext["object"])
        if key in gold_set:
            tp_list.append(ext)
            matched_gold.add(key)
        else:
            fp_list.append(ext)

    fn_list = [orig for key, orig in gold_set.items() if key not in matched_gold]

    tp = len(tp_list)
    fp = len(fp_list)
    fn = len(fn_list)

    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    f1 = round(2 * (precision * recall) / (precision + recall), 3) if (precision + recall) > 0 else 0.0

    return {
        "has_gold": True,
        "gold_count": len(gold_triples),
        "extracted_count": len(extracted_triples),
        "tp_count": tp,
        "fp_count": fp,
        "fn_count": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pair_recall": pair_recall,
        "tp_list": tp_list,
        "fp_list": fp_list,
        "fn_list": fn_list
    }


# ======================================================================================
# 5. SAMPLE CORPORA & GROUND TRUTH REFERENCE SETS
# ======================================================================================

CORPORA: Dict[str, Dict[str, Any]] = {
    "Tech & Business": {
        "name": "Technology and Enterprise",
        "level": "Beginner",
        "blurb": "Concise factual statements on technology companies in straightforward active voice.",
        "purpose": (
            "Every sentence is in the active voice, consists of clear subject-verb-object syntax, "
            "and places the relationship verb directly between the two entities. This represents "
            "the baseline scenario where rule-based extraction performs with high precision."
        ),
        "observe": [
            "Observe that 'Larry Page' and 'Google' are linked by 'founded' with confidence >= 0.90.",
            "Notice how 'Google acquired YouTube' generates a high-confidence 'acquired' predicate.",
            "Compare how 'Sundar Pichai is the CEO of Google' matches the 'leads' regex rule."
        ],
        "text": (
            "Larry Page and Sergey Brin co-founded Google in California in 1998. "
            "Google acquired YouTube in 2006 and developed the Android operating system. "
            "Sundar Pichai is the CEO of Google and reports to the Alphabet board. "
            "Alphabet is headquartered in Mountain View. "
            "Microsoft, based in Redmond, competes with Google in cloud computing. "
            "Satya Nadella leads Microsoft as its chief executive. "
            "Microsoft acquired LinkedIn and partnered with OpenAI."
        ),
        "gold": [
            ("Larry Page", "founded", "Google"),
            ("Sergey Brin", "founded", "Google"),
            ("Google", "located_in", "California"),
            ("Google", "acquired", "YouTube"),
            ("Google", "created", "Android"),
            ("Sundar Pichai", "leads", "Google"),
            ("Sundar Pichai", "reports_to", "Alphabet"),
            ("Alphabet", "located_in", "Mountain View"),
            ("Microsoft", "located_in", "Redmond"),
            ("Microsoft", "competes_with", "Google"),
            ("Satya Nadella", "leads", "Microsoft"),
            ("Microsoft", "acquired", "LinkedIn"),
            ("Microsoft", "collaborates_with", "OpenAI"),
        ]
    },
    "Science & Academia": {
        "name": "Science and Academia",
        "level": "Intermediate",
        "blurb": "Biographical assertions concerning scientists, universities, and discoveries.",
        "purpose": (
            "Combines PERSON, ORG, LOCATION, and DATE entities. Demonstrates the need for entity type "
            "constraints (e.g. 'born_in' accepts a PERSON subject and LOCATION or DATE object)."
        ),
        "observe": [
            "In 'Pierre Curie, her husband', the relationship is expressed via an appositive noun phrase.",
            "Notice that Marie Curie is born in Warsaw (LOCATION) and studied at University of Paris (ORG).",
            "Observe how Albert Einstein is linked to 'theory of relativity' and collaborators."
        ],
        "text": (
            "Marie Curie was born in Warsaw and studied at the University of Paris. "
            "Curie discovered radium and polonium alongside Pierre Curie, her husband. "
            "The Curie Institute was established in Paris to continue scientific research. "
            "Albert Einstein published his theory of relativity in 1905. "
            "Einstein collaborated with Niels Bohr on quantum mechanics debates."
        ),
        "gold": [
            ("Marie Curie", "born_in", "Warsaw"),
            ("Marie Curie", "studied_at", "University of Paris"),
            ("Curie", "discovered", "radium"),
            ("Curie", "married_to", "Pierre Curie"),
            ("Curie Institute", "located_in", "Paris"),
            ("Albert Einstein", "authored", "theory of relativity"),
            ("Einstein", "collaborates_with", "Niels Bohr"),
        ]
    },
    "Indian Industry & Space": {
        "name": "Indian Industry and Space Technology",
        "level": "Intermediate",
        "blurb": "Institutional and executive statements from Indian industry and space exploration.",
        "purpose": (
            "Demonstrates extraction over domain-specific organizations and regional geographical entities."
        ),
        "observe": [
            "Look at the acquisition of Jaguar Land Rover by Tata Group.",
            "Examine how 'headquartered in Bengaluru' is detected as a located_in relationship.",
            "Observe the CEO and birthplace assertions regarding Sundar Pichai."
        ],
        "text": (
            "Ratan Tata was the chairman of the Tata Group for more than two decades. "
            "The Tata Group is headquartered in Mumbai. "
            "Tata Group acquired Jaguar Land Rover in 2008. "
            "Narayana Murthy and Nandan Nilekani founded Infosys. "
            "Infosys is headquartered in Bengaluru and competes with Tata Consultancy Services. "
            "The Indian Space Research Organisation is based in Bengaluru. "
            "Sundar Pichai was born in Chennai. "
            "Sundar Pichai is the CEO of Google."
        ),
        "gold": [
            ("Ratan Tata", "leads", "Tata Group"),
            ("Tata Group", "located_in", "Mumbai"),
            ("Tata Group", "acquired", "Jaguar Land Rover"),
            ("Narayana Murthy", "founded", "Infosys"),
            ("Nandan Nilekani", "founded", "Infosys"),
            ("Infosys", "located_in", "Bengaluru"),
            ("Infosys", "competes_with", "Tata Consultancy Services"),
            ("Indian Space Research Organisation", "located_in", "Bengaluru"),
            ("Sundar Pichai", "born_in", "Chennai"),
            ("Sundar Pichai", "leads", "Google"),
        ]
    },
    "Healthcare & Medicine": {
        "name": "Healthcare and Biomedical Research",
        "level": "Advanced",
        "blurb": "Clinical and pharmaceutical relations between institutions, therapies, and indications.",
        "purpose": (
            "Introduces specialized clinical relations (treats, causes). Demonstrates how expanding the "
            "lexicon allows rule-based extractors to adapt across technical domains."
        ),
        "observe": [
            "Observe the extraction of 'treats' between pharmacological agents and clinical conditions.",
            "Notice institutional affiliations connecting researchers to research hospitals."
        ],
        "text": (
            "Alexander Fleming discovered penicillin in London in 1928. "
            "Penicillin treats bacterial infections effectively. "
            "The World Health Organisation is based in Geneva. "
            "Pfizer partnered with BioNTech to produce vaccines. "
            "Chronic smoking causes cardiovascular disease and respiratory failure."
        ),
        "gold": [
            ("Alexander Fleming", "discovered", "penicillin"),
            ("Alexander Fleming", "located_in", "London"),
            ("Penicillin", "treats", "bacterial infections"),
            ("World Health Organisation", "located_in", "Geneva"),
            ("Pfizer", "collaborates_with", "BioNTech"),
            ("smoking", "causes", "cardiovascular disease"),
        ]
    }
}


# ======================================================================================
# 6. DOCUMENT UPLOAD LOADER
# ======================================================================================

MAX_UPLOAD_CHARACTERS = 25_000


@dataclass
class LoadedDocument:
    filename: str
    extension: str
    text: str
    raw_character_count: int
    truncated: bool = False
    error: Optional[str] = None
    size_bytes: int = 0
    page_count: Optional[int] = None

    def summary(self) -> str:
        if self.error:
            return f"Failed to load `{self.filename}`: {self.error}"
        status = " (truncated to limit)" if self.truncated else ""
        return f"`{self.filename}` ({self.raw_character_count:,} characters{status})"


def load_uploaded_document(uploaded: Any) -> LoadedDocument:
    """Reads uploaded files (.txt, .md, .csv, .tsv, .pdf, .docx)."""
    name = getattr(uploaded, "name", "uploaded_file")
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    try:
        # getvalue() is safe to call on every rerun; read() returns b"" once the pointer is at the end.
        data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else (
            uploaded.read() if hasattr(uploaded, "read") else bytes(uploaded))
    except Exception as exc:
        return LoadedDocument(name, ext, "", 0, error=f"Could not read upload stream: {exc}")

    text = ""
    error = None
    page_count = None
    try:
        if ext in ("txt", "text", "log"):
            text = data.decode("utf-8", errors="replace")
        elif ext in ("md", "markdown"):
            raw = data.decode("utf-8", errors="replace")
            text = re.sub(r"#+\s*", "", raw)
            text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        elif ext in ("csv", "tsv"):
            delim = "\t" if ext == "tsv" else ","
            reader = csv.reader(io.StringIO(data.decode("utf-8", errors="replace")), delimiter=delim)
            rows = [" ".join(cell.strip() for cell in row if cell.strip()) for row in reader]
            text = ". ".join(r for r in rows if r)
        elif ext == "pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(data))
                pages = [page.extract_text() or "" for page in reader.pages]
                page_count = len(pages)
                text = " ".join(pages)
            except ImportError:
                error = "pypdf package is required for PDF parsing. Please run `pip install pypdf`."
        elif ext in ("docx", "doc"):
            try:
                import docx
                doc = docx.Document(io.BytesIO(data))
                text = " ".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                error = "python-docx package is required for Word documents. Please run `pip install python-docx`."
        else:
            text = data.decode("utf-8", errors="replace")
    except Exception as e:
        error = f"Error decoding file: {e}"

    clean_text = " ".join(text.split())
    raw_len = len(clean_text)
    truncated = False
    if raw_len > MAX_UPLOAD_CHARACTERS:
        clean_text = clean_text[:MAX_UPLOAD_CHARACTERS]
        truncated = True

    return LoadedDocument(
        filename=name,
        extension=ext,
        text=clean_text,
        raw_character_count=raw_len,
        truncated=truncated,
        error=error,
        size_bytes=len(data),
        page_count=page_count
    )


# ======================================================================================
# 7. HIGH-PERFORMANCE GRAPH LAYOUT ALGORITHM
# ======================================================================================

MAX_GRAPH_NODES = 28


def _to_networkx(count: int, edges: Sequence[Tuple[int, int]]) -> "nx.Graph":
    """Builds the networkx graph backing every node-link diagram in the lab."""
    G = nx.Graph()
    G.add_nodes_from(range(count))
    G.add_edges_from((s, e) for s, e in edges if 0 <= s < count and 0 <= e < count)
    return G


def _components(G: "nx.Graph") -> List[List[int]]:
    """Connected components (via networkx), sorted largest first."""
    return sorted((sorted(c) for c in nx.connected_components(G)), key=len, reverse=True)


def _force_layout(G: "nx.Graph", nodes: Sequence[int], seed: int = 11) -> np.ndarray:
    """Positions one connected component with networkx's spring_layout
    (Fruchterman-Reingold), normalised into the -1..1 square."""
    count = len(nodes)
    if count == 0:
        return np.zeros((0, 2))
    if count == 1:
        return np.zeros((1, 2))
    if count == 2:
        return np.array([[-0.8, 0.0], [0.8, 0.0]])

    subG = G.subgraph(nodes)
    # A larger-than-default k (ideal spring length) spreads nodes out, since every node
    # carries a text label that must not collide with its neighbours.
    raw = nx.spring_layout(subG, seed=seed, k=2.6 / math.sqrt(count), iterations=400)
    if 3 <= count <= 40:
        # Kamada-Kawai spaces small/medium graphs more evenly than spring layout; the spring
        # result seeds it so the "re-layout" button still produces a different arrangement.
        try:
            raw = nx.kamada_kawai_layout(subG, pos=raw)
        except Exception:
            pass  # scipy missing or non-convergence: keep the spring layout
    pos = np.array([raw[n] for n in nodes])
    return _normalise_layout(pos)


def _normalise_layout(pos: np.ndarray) -> np.ndarray:
    if len(pos) <= 1:
        return np.zeros_like(pos)
    low, high = pos.min(axis=0), pos.max(axis=0)
    span = high - low
    scale = max(span[0], span[1]) or 1.0
    centred = pos - (low + high) / 2.0
    return centred / (scale / 2.0)


def _label_boxes(names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    half_width = np.array([0.022 + 0.006 * len(n[:18]) for n in names], dtype=float)
    half_height = np.full(len(names), 0.060, dtype=float)
    return half_width, half_height


def _separate_labels(pos: np.ndarray, half_width: np.ndarray, half_height: np.ndarray, rounds: int = 140) -> np.ndarray:
    if len(pos) < 2:
        return pos
    pos = pos.copy()
    for _ in range(rounds):
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        need_x = half_width[:, None] + half_width[None, :]
        need_y = half_height[:, None] + half_height[None, :]
        ox = need_x - np.abs(dx)
        oy = need_y - np.abs(dy)
        colliding = (ox > 0) & (oy > 0)
        np.fill_diagonal(colliding, False)
        rows, cols = np.where(np.triu(colliding))
        if len(rows) == 0:
            break
        for f, s in zip(rows, cols):
            if ox[f, s] <= oy[f, s]:
                dir_v = np.array([1.0 if dx[f, s] >= 0 else -1.0, 0.0])
                amt = ox[f, s] / 2.0 + 1e-3
            else:
                dir_v = np.array([0.0, 1.0 if dy[f, s] >= 0 else -1.0])
                amt = oy[f, s] / 2.0 + 1e-3
            pos[f] += dir_v * amt
            pos[s] -= dir_v * amt
    return pos


def _graph_layout(names: Sequence[str], edges: Sequence[Tuple[int, int]], seed: int = 11) -> np.ndarray:
    """Node positions computed via networkx: connected components (networkx) are laid
    out independently (spring + Kamada-Kawai), with the largest cluster on top and the
    small ones packed into compact rows beneath it, so unrelated clusters never get
    dragged into one tangle."""
    count = len(names)
    if count <= 1:
        return np.zeros((count, 2))

    G = _to_networkx(count, edges)
    groups = _components(G)
    positions = np.zeros((count, 2))

    local_layouts = [_force_layout(G, grp, seed=seed) for grp in groups]

    if len(groups) == 1:
        for node, pt in zip(groups[0], _normalise_layout(local_layouts[0]) * 0.92):
            positions[node] = pt
    else:
        # The largest cluster gets a big area on top; the many small clusters are packed
        # into compact cells in rows beneath it, instead of being scattered over the canvas.
        for node, pt in zip(groups[0], local_layouts[0]):
            positions[node] = [pt[0] * 0.95, 0.75 + pt[1] * 0.60]
        others = list(zip(groups[1:], local_layouts[1:]))
        # Two columns of generously sized cells (the canvas grows taller instead of
        # squeezing clusters together), so a pair's labels and edge chip never collide.
        ncols = min(len(others), 2 if len(others) <= 6 else 3)
        cell_w, cell_h = 2.0 / ncols, 0.45
        for k, (grp, lay) in enumerate(others):
            r, c = divmod(k, ncols)
            cx = -1.0 + (c + 0.5) * cell_w
            cy = -0.05 - (r + 0.5) * cell_h
            sx, sy = cell_w * 0.40, cell_h * 0.28
            for node, pt in zip(grp, lay):
                positions[node] = [cx + pt[0] * sx, cy + pt[1] * sy]

    hw, hh = _label_boxes(names)
    positions = _separate_labels(positions, hw, hh)
    return positions


def _label_positions(pos: np.ndarray, edges: Sequence[Tuple[int, int]]) -> List[str]:
    count = len(pos)
    neighbors: Dict[int, List[int]] = {i: [] for i in range(count)}
    for s, e in edges:
        neighbors[s].append(e)
        neighbors[e].append(s)

    options = [
        ("bottom center", np.array([0.0, -1.0])),
        ("top center", np.array([0.0, 1.0])),
        ("middle right", np.array([1.0, 0.0])),
        ("middle left", np.array([-1.0, 0.0])),
    ]
    chosen = []
    for i in range(count):
        if not neighbors[i]:
            chosen.append("bottom center")
            continue
        direction = np.zeros(2)
        for nbr in neighbors[i]:
            offset = pos[nbr] - pos[i]
            l = float(np.linalg.norm(offset)) or 1.0
            direction += offset / l
        best = min(options, key=lambda opt: float(np.dot(direction, opt[1])) - (0.25 if "center" in opt[0] else 0.0))
        chosen.append(best[0])
    return chosen


# ======================================================================================
# 8. POLISHED VISUALIZATION BUILDERS (PLOTLY & GRAPH FIGURES)
# ======================================================================================

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


BLIND_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks", "blind_results.json")
BERT_TRAINING_STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_re_training_stats.json")


def _load_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


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


def _build_blind_test_figure(theme: Theme) -> Optional[go.Figure]:
    """F1 of every method on the blind test: real Wikipedia sentences, hand-written ones, and both."""
    data = _load_json_file(BLIND_RESULTS_PATH)
    if not data:
        return None
    results = data["results"][::-1]
    names = [r["method"].split(" (")[0] for r in results]
    fig = go.Figure()
    for key, label, colour_key in (("part_a", "Part A: real Wikipedia sentences", "LOCATION"),
                                   ("part_b", "Part B: hand-written sentences", "ORG"),
                                   ("both", "Both parts together", "PERSON")):
        values = [r[key]["f1"] for r in results]
        fig.add_trace(go.Bar(
            y=names, x=values, orientation="h", name=label,
            marker=dict(color=theme.for_type(colour_key), cornerradius=4),
            text=[f"{v:.2f}" for v in values], textposition="outside",
            hovertemplate=f"<b>%{{y}}</b><br>{label}<br>F1 %{{x:.2f}}<extra></extra>"
        ))
    fig.update_layout(
        barmode="group", height=460, margin=dict(l=10, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        paper_bgcolor=theme.paper_bg, plot_bgcolor=theme.card_bg, font=dict(color=theme.ink, size=11),
        xaxis=dict(title="F1 score on the blind test", range=[0, 1.05], gridcolor=theme.grid),
        yaxis=dict(gridcolor=theme.grid),
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


# Methods compared side-by-side in the Theory section's "Method Comparison" chart.
_COMPARISON_METHODS = [
    "Co-occurrence Only",
    "Pattern-Based (Lexicon Match)",
    "Hybrid (Pattern + Co-occurrence)",
    "Nearest-Verb (Generalized)",
    "Dependency Parse (SVO)",
    BERT_METHOD,
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
        if method == BERT_METHOD and not bert_available():
            st.warning(f"The BERT method needs the optional BERT library: `{BERT_INSTALL_HINT}`. "
                       "It will return no triples until installed.")
        if method == SVO_METHOD and not spacy_parser_available():
            continue
        if method == BERT_METHOD and not bert_available():
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


# ======================================================================================
# 9. PDF LABORATORY REPORT GENERATOR (FPDF)
# ======================================================================================

class LabReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "VIRTUAL LABORATORY PORTAL | KNOWLEDGE GRAPHS & INFORMATION RETRIEVAL SYSTEMS", align="R")
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Academic Virtual Lab Report | Exp-07 Relationship Extraction", align="C")


# A condensed, plain-text (no markdown/[[term]] markers) version of the Theory section's
# background, for the PDF report's own "Theoretical Background" section.
PDF_THEORY_SUMMARY = (
    "Relationship Extraction (RE) identifies and classifies semantic relationships holding "
    "between pairs of entities mentioned in unstructured text, expressing each as a "
    "Subject-Predicate-Object (SPO) triple - the atomic unit from which a Knowledge Graph is "
    "built: entities become nodes, and predicates become directed, labelled edges between them. "
    "This laboratory implements and compares six extraction strategies; four are "
    "rule-based and need no external NLP model, one uses a grammar parser and one a pretrained BERT model. Pattern-Based extraction emits a triple only when "
    "the text span between two adjacent entities matches an ordered lexicon of relation-cue "
    "regular expressions (e.g. 'acquired', 'is the CEO of'), giving high precision at the cost "
    "of recall. Co-occurrence extraction links every adjacent entity pair with a generic "
    "'related_to' predicate whose confidence decays with word distance, giving high recall at "
    "the cost of precision. Hybrid extraction applies the Pattern-Based check first and falls "
    "back to a Co-occurrence edge only when no cue matches, so no candidate pair is ever "
    "dropped outright. Nearest-Verb drops the fixed lexicon and uses the verb between two "
    "entities as the predicate, and Dependency Parse (SVO, using the optional spaCy parser) reads "
    "the sentence grammar to find who did what to whom, including in the passive voice, and BERT Similarity (Sentence-BERT) judges the meaning of the text between two entities with a classifier trained on BERT embeddings. Every emitted triple carries a numeric confidence score and its source "
    "sentence as provenance, so downstream stages (Experiments 8-9: graph database construction "
    "and schema import) can filter or audit it before ingestion."
)

# (Method, Precision, Recall, Characteristic) - the same qualitative comparison shown in the
# Theory tab's method-comparison table, condensed for the PDF.
PDF_METHOD_COMPARISON_ROWS = [
    ("Co-occurrence", "Low", "Highest", "Confidence decays with word distance; no cue required"),
    ("Pattern-Based", "High", "Lower", "Every edge traces to an explicit cue word in the text"),
    ("Hybrid", "Medium-High", "High", "Pattern-Based first, Co-occurrence fallback if no cue matches"),
    ("Nearest-Verb", "Medium", "Medium", "Nearest verb between entities becomes the predicate; no word list"),
    ("Dependency Parse", "High", "High", "Grammatical roles (subject/object) decide the triple; handles passive voice"),
    ("BERT Similarity", "High", "Medium-High", "A classifier on BERT embeddings judges the clause's meaning; no word list, but a black box"),
]


def generate_pdf_report(
    student_name: str,
    student_id: str,
    student_batch: str,
    date_str: str,
    trials_df: pd.DataFrame,
    quiz_score: int,
    quiz_total: int,
    student_notes: str,
    triples_df: pd.DataFrame,
    eval_metrics: Optional[Dict[str, Any]] = None
) -> bytes:
    """Generates an official institutional laboratory report PDF."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Title Banner
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, EXPERIMENT_CONFIG["title"], align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 5, f"{EXPERIMENT_CONFIG['course']} | {EXPERIMENT_CONFIG['department']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Student Credentials Card
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 28, 190, 24, "FD")

    pdf.set_xy(14, 30)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(30, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, student_name or "N/A", 0)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Roll No / Student ID:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(55, 5, student_id or "N/A", 1)

    pdf.set_xy(14, 38)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(30, 5, "Batch / Class:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, student_batch or "D17", 0)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Assessment:", 0)
    pdf.set_font("Helvetica", "B", 8)
    pct = int((quiz_score / quiz_total) * 100) if quiz_total else 0
    if pct >= 50:
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(55, 5, f"{quiz_score} / {quiz_total} ({pct}%)", 1)

    pdf.set_xy(14, 45)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(30, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, date_str or datetime.now().strftime("%Y-%m-%d"), 0)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Lab Status:", 0)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(55, 5, "Verified & Evaluated", 1)

    pdf.set_xy(10, 56)

    # 1. Objectives
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "1. Educational Aim & Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        pdf.cell(4, 4, "-", 0)
        pdf.cell(0, 4, f" {obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # 2. Theoretical Background
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "2. Theoretical Background", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.3, PDF_THEORY_SUMMARY)
    pdf.ln(1.5)

    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 41, 59)
    pdf.cell(45, 5, "Method", 1, 0, "L", True)
    pdf.cell(35, 5, "Precision", 1, 0, "C", True)
    pdf.cell(35, 5, "Recall", 1, 0, "C", True)
    pdf.cell(75, 5, "Characteristic", 1, 1, "L", True)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(30, 41, 59)
    fill = False
    for method_name, precision, recall, characteristic in PDF_METHOD_COMPARISON_ROWS:
        pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(45, 4.5, method_name, 1, 0, "L", fill)
        pdf.cell(35, 4.5, precision, 1, 0, "C", fill)
        pdf.cell(35, 4.5, recall, 1, 0, "C", fill)
        pdf.cell(75, 4.5, characteristic, 1, 1, "L", fill)
        fill = not fill
    pdf.ln(3)

    # 3. Evaluation Summary (if available)
    if eval_metrics and eval_metrics.get("has_gold"):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 6, "3. Ground Truth Benchmark Evaluation", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(45, 5, f"Precision: {eval_metrics['precision']:.3f}", 1, 0, "C")
        pdf.cell(45, 5, f"Recall: {eval_metrics['recall']:.3f}", 1, 0, "C")
        pdf.cell(45, 5, f"F1 Score: {eval_metrics['f1']:.3f}", 1, 0, "C")
        pdf.cell(55, 5, f"True Positives: {eval_metrics['tp_count']} / {eval_metrics['gold_count']}", 1, 1, "C")
        pdf.ln(3)

    # 4. Recorded Experimental Trials
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "4. Recorded Experimental Trials (Session Logbook)", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, "No experimental trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)
        cols = list(trials_df.columns)
        col_w = max(18, int(190 / max(1, len(cols))))
        for c in cols:
            pdf.cell(col_w, 5, str(c)[:14], 1, 0, "C", True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 7)
        fill = False
        for _, row in trials_df.iterrows():
            for c in cols:
                val = row[c]
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 4.5, val_str[:15], 1, 0, "C", fill)
            pdf.ln()
            fill = not fill
    pdf.ln(3)

    # 4. Extracted Triples Sample
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "5. Extracted Subject-Predicate-Object Knowledge Triples", new_x="LMARGIN", new_y="NEXT")

    if triples_df.empty:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, "No triples extracted in the active session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(50, 5, "Subject", 1, 0, "L", True)
        pdf.cell(45, 5, "Predicate", 1, 0, "L", True)
        pdf.cell(50, 5, "Object", 1, 0, "L", True)
        pdf.cell(20, 5, "Conf.", 1, 0, "C", True)
        pdf.cell(25, 5, "Method", 1, 1, "C", True)

        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(30, 41, 59)
        fill = False
        for _, row in triples_df.head(15).iterrows():
            # A bare ternary-as-statement here (not assigned to anything) is picked up by
            # Streamlit's "magic" auto-display and rendered as a stray `None` in the app,
            # since streamlit run auto-wraps unassigned top-level expressions in st.write().
            if fill:
                pdf.set_fill_color(248, 250, 252)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.cell(50, 4.5, str(row.get("subject", ""))[:26], 1, 0, "L", fill)
            pdf.cell(45, 4.5, str(row.get("predicate", ""))[:24], 1, 0, "L", fill)
            pdf.cell(50, 4.5, str(row.get("object", ""))[:26], 1, 0, "L", fill)
            pdf.cell(20, 4.5, f"{float(row.get('confidence', 0)):.2f}", 1, 0, "C", fill)
            pdf.cell(25, 4.5, str(row.get("method", "Pattern"))[:12], 1, 1, "C", fill)
            fill = not fill
    pdf.ln(3)

    # 5. Student Conclusions & Observations
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "6. Student Analytical Discussion & Conclusions", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.5, student_notes or "The student did not record analytical notes.")
    pdf.ln(4)

    # Signature Block
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(95, 6, "Student Signature: _______________________", 0, 0, "L")
    pdf.cell(95, 6, "Faculty Evaluator Signature: _______________________", 0, 1, "R")

    return bytes(pdf.output())


# ======================================================================================
# 10. ASSESSMENT QUIZ BANK (10 PEDAGOGICAL QUESTIONS)
# ======================================================================================

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "What are the three canonical components of a knowledge triple extracted from unstructured text?",
        "options": [
            "A) Noun, Adjective, Verb",
            "B) Subject, Predicate, Object",
            "C) Keyword, Frequency, Weight",
            "D) Document, Term, Score"
        ],
        "answer_index": 1,
        "explanation": "Subject-Predicate-Object (SPO) triples form the atomic data assertions in Knowledge Graphs and Semantic Web models."
    },
    {
        "id": 2,
        "question": "In a Knowledge Graph construction pipeline, what do the extracted entities and predicates represent?",
        "options": [
            "A) Entities become nodes, and predicates become directed labeled edges",
            "B) Entities become edge labels, and predicates become nodes",
            "C) Both entities and predicates become database tables without relationships",
            "D) Entities become vector indexes and predicates are discarded"
        ],
        "answer_index": 0,
        "explanation": "Entities (e.g. Google, California) represent network nodes, while predicates (e.g. founded, located_in) form directed edges connecting them."
    },
    {
        "id": 3,
        "question": "Why does co-occurrence based extraction typically exhibit high recall but low precision?",
        "options": [
            "A) It only considers capitalized words",
            "B) It assumes any two entities co-occurring in a sentence are related, over-generating generic links without lexical proof",
            "C) It strictly enforces grammatical dependency rules",
            "D) It drops all pairs with distance greater than 2 words"
        ],
        "answer_index": 1,
        "explanation": "Co-occurrence captures almost all true pairs (high recall) but generates many spurious or meaningless relationships (low precision)."
    },
    {
        "id": 4,
        "question": "What is the primary vulnerability (limitation) of hand-crafted pattern-based relation extraction?",
        "options": [
            "A) Linguistic brittleness when encountering paraphrases, metaphors, or complex syntactic inversions",
            "B) High computational overhead compared to neural language models",
            "C) Inability to run on CPU without GPUs",
            "D) Extreme over-generation of false positive edges"
        ],
        "answer_index": 0,
        "explanation": "Pattern rules are brittle: they reliably catch anticipated phrasings ('acquired'), but fail on unexpected paraphrases ('swallowed up')."
    },
    {
        "id": 5,
        "question": "What does Open Information Extraction (OpenIE) NOT require, unlike traditional supervised relation extraction?",
        "options": [
            "A) Input text",
            "B) A fixed, predefined schema of relation types",
            "C) Sentences",
            "D) Words"
        ],
        "answer_index": 1,
        "explanation": "OpenIE extracts arbitrary relation phrases directly from text without needing a rigid predefined ontology or relation taxonomy."
    },
    {
        "id": 6,
        "question": "What NLP pre-processing stage resolves pronouns like 'she' or 'the company' back to their antecedent entity mention?",
        "options": [
            "A) Lemmatization",
            "B) Co-reference resolution",
            "C) Stop-word pruning",
            "D) Part-of-speech tagging"
        ],
        "answer_index": 1,
        "explanation": "Co-reference resolution links pronouns ('she') to their true referent (e.g. 'Marie Curie'), enabling facts spread across sentences to be extracted."
    },
    {
        "id": 7,
        "question": "In the triple (Google) --[acquired]--> (YouTube), which element is the predicate?",
        "options": [
            "A) Google",
            "B) YouTube",
            "C) acquired",
            "D) Confidence score"
        ],
        "answer_index": 2,
        "explanation": "'acquired' is the relational predicate characterizing the semantic connection between the subject and object."
    },
    {
        "id": 8,
        "question": "Why is attaching a quantitative confidence score to each extracted triple beneficial for downstream graph ingestion?",
        "options": [
            "A) It allows downstream graph databases to filter out low-confidence noise and prioritize verified facts",
            "B) It eliminates the need for entity normalization",
            "C) It automatically translates English into a graph query language",
            "D) It guarantees 100% precision regardless of input text quality"
        ],
        "answer_index": 0,
        "explanation": "Confidence scores empower downstream ingestion pipelines to discard unreliable inferences and weight graph algorithms."
    },
    {
        "id": 9,
        "question": "Which evaluation metric represents the harmonic mean of Precision and Recall?",
        "options": [
            "A) Accuracy",
            "B) F1 Score",
            "C) Mean Reciprocal Rank (MRR)",
            "D) Discounted Cumulative Gain (DCG)"
        ],
        "answer_index": 1,
        "explanation": "The F1 score is the harmonic mean of precision and recall: 2 * (P * R) / (P + R)."
    },
    {
        "id": 10,
        "question": "When importing extracted triples into a graph database, why is an upsert (create-if-not-exists) preferred over a plain insert?",
        "options": [
            "A) An upsert is idempotent: it avoids creating duplicate nodes when an entity appears in multiple triples",
            "B) A plain insert is not supported by modern graph databases",
            "C) An upsert automatically executes relationship extraction in memory",
            "D) An upsert deletes existing nodes before adding new ones"
        ],
        "answer_index": 0,
        "explanation": "An upsert creates a node or relationship only if it does not already exist, ensuring identical entities consolidate into a connected network."
    }
]


# ======================================================================================
# 11. SECTION RENDERERS: THEORY, SIMULATION, QUIZ, REPORT
# ======================================================================================

def render_theory_section():
    """Renders Section 1: Theory, Background, Objectives, Procedure, and Glossary."""
    theme = get_current_theme()

    st.header("Theoretical Framework & Background")
    write_annotated(THEORY_CONTENT["background"])

    st.subheader("Visual Overview: The Extraction Pipeline")
    st.caption(
        "Every extraction run, regardless of method, passes through these six stages in order. "
        "Colours match the entity-type legend used throughout the Simulation section."
    )
    st.plotly_chart(_build_pipeline_diagram(theme), use_container_width=True, config={"displayModeBar": False})

    st.divider()
    st.subheader("Algorithmic Detail: How Each Extraction Method Works")
    st.write(
        "The six methods below form a progression from a fixed word list, through grammar "
        "analysis, to a language model that compares meaning. Five of them share the same "
        "first three pipeline stages (splitting, entity recognition, adjacent candidate-pair "
        "generation) and differ only in how a pair is turned into a predicate; the "
        "Dependency Parse method reads the sentence's grammar instead. Each tab shows the "
        "procedure as pseudocode, followed by that method's *live* output on the same "
        "worked example sentence."
    )

    _example_text = "Larry Page co-founded Google in California in 1998."
    _example_entities = extract_entities(_example_text)

    tab_cooc, tab_pattern, tab_hybrid, tab_verb, tab_svo, tab_bert = st.tabs([
        "Co-occurrence Only", "Pattern-Based (Lexicon Match)", "Hybrid (Pattern + Co-occurrence)",
        "Nearest-Verb (Generalized)", "Dependency Parse (SVO)", "BERT Similarity (Sentence-BERT)"
    ])

    with tab_pattern:
        write_annotated(
            "Emits a triple **only** when the text span between two adjacent entities matches "
            "one of the ordered [[relation_cue|relation-cue]] regular expressions "
            "(`_RELATION_PATTERNS`). Patterns are tried most-specific-first and the **first "
            "match wins** - so `co-founded` is caught by the `founded` family before any looser "
            "pattern gets a chance. If nothing matches, the pair is silently dropped: this is "
            "why the method is high-[[precision]] but limited in [[recall]]."
        )
        st.code(
            "for sentence in split_sentences(document):\n"
            "    entities = extract_entities(sentence)                 # NER\n"
            "    for (subject, object) in adjacent_pairs(entities):    # skip if entity in between\n"
            "        span = text_between(subject, object)\n"
            "        if word_count(span) > max_word_distance:\n"
            "            continue                                      # too far apart, skip pair\n"
            "        for (regex, predicate, base_confidence) in RELATION_PATTERNS:  # ordered\n"
            "            if regex.search(span):\n"
            "                confidence = apply_type_constraints(predicate, subject, object, base_confidence)\n"
            "                emit Triple(subject, predicate, object, confidence, cue=regex.match_text)\n"
            "                break                                     # first match wins, stop looking\n"
            "        # else: no cue found -> pair produces NO triple in Pattern-Based mode",
            language="python"
        )
        st.caption(
            "`apply_type_constraints` softens confidence when the entity types look wrong for "
            "the predicate - e.g. `born_in` is discounted (x0.6) unless the subject is a PERSON, "
            "and `leads` is discounted (x0.75) unless the object is an ORG or LOCATION."
        )
        pattern_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                                method="Pattern-Based (Lexicon Match)")
        st.write(f"**Live trace** for: *\"{_example_text}\"*")
        if pattern_triples:
            st.dataframe(pd.DataFrame(pattern_triples)[["subject", "predicate", "object", "confidence", "cue"]],
                        hide_index=True, use_container_width=True)
        else:
            st.info("No cue matched in this example, so Pattern-Based mode emits no triple here.")

    with tab_cooc:
        write_annotated(
            "Makes no attempt to read the span at all: **every** adjacent entity pair in a "
            "sentence is linked with the generic predicate `related_to`. This maximises "
            "[[recall]] (nothing is ever silently dropped) at the cost of [[precision]] (most "
            "edges carry no real semantic content). Confidence is not fixed - it **decays with "
            "distance**, so two entities separated by many words score lower than two sitting "
            "right next to each other."
        )
        st.code(
            "for sentence in split_sentences(document):\n"
            "    entities = extract_entities(sentence)\n"
            "    for (subject, object) in adjacent_pairs(entities):\n"
            "        if word_count(span) > max_word_distance:\n"
            "            continue\n"
            "        confidence = max(0.20, 0.45 - 0.02 * word_count(span))   # decays with distance\n"
            "        emit Triple(subject, 'related_to', object, confidence, cue=None)",
            language="python"
        )
        cooc_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                             method="Co-occurrence Only")
        st.write(f"**Live trace** for: *\"{_example_text}\"*")
        if cooc_triples:
            st.dataframe(pd.DataFrame(cooc_triples)[["subject", "predicate", "object", "confidence"]],
                        hide_index=True, use_container_width=True)
        else:
            st.info("No adjacent entity pairs were found in this example.")

    with tab_hybrid:
        write_annotated(
            "Runs the **Pattern-Based** check first on every candidate pair; only when no cue "
            "matches does it fall back to a **Co-occurrence** edge instead of dropping the pair. "
            "The result is a superset of what either method produces alone: the confident, "
            "correctly-labelled edges from Pattern-Based, topped up with low-confidence "
            "`related_to` edges so no relationship is lost entirely. This is the [[hybrid]] "
            "default used throughout the Simulation section."
        )
        st.code(
            "for (subject, object) in adjacent_pairs(entities):\n"
            "    if word_count(span) > max_word_distance:\n"
            "        continue\n"
            "    match = first_matching_pattern(span)          # same ordered lexicon as Pattern-Based\n"
            "    if match:\n"
            "        emit Triple(subject, match.predicate, object, match.confidence, cue=match.text)\n"
            "    else:\n"
            "        confidence = max(0.20, 0.45 - 0.02 * word_count(span))\n"
            "        emit Triple(subject, 'related_to', object, confidence, cue=None)  # fallback",
            language="python"
        )
        hybrid_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                               method="Hybrid (Pattern + Co-occurrence)")
        st.write(f"**Live trace** for: *\"{_example_text}\"*")
        if hybrid_triples:
            st.dataframe(pd.DataFrame(hybrid_triples)[["subject", "predicate", "object", "confidence", "cue"]],
                        hide_index=True, use_container_width=True)
        else:
            st.info("No adjacent entity pairs were found in this example.")

    _passive_text = "YouTube was acquired by Google in 2006."
    _passive_entities = extract_entities(_passive_text)

    with tab_verb:
        write_annotated(
            "Drops the fixed word list. For each adjacent entity pair it takes the **first "
            "verb-like word** between them, reduces it to its [[lemma]] (base form) and uses "
            "that as the predicate - so verbs the lexicon never listed (\"oversees\", "
            "\"divested\") still become relations. Common verbs are then mapped onto the same "
            "predicate names as the lexicon (`acquire` -> `acquired`). It reads strictly "
            "left-to-right, so it cannot tell who did what in a [[passive_voice|passive sentence]]."
        )
        st.caption("Illustration: how one pair of entities becomes a predicate (hover each box).")
        st.plotly_chart(_build_nearest_verb_steps_figure(theme), use_container_width=True,
                        config={"displayModeBar": False})
        st.code(
            "for (subject, object) in adjacent_pairs(entities):\n"
            "    span = text_between(subject, object)\n"
            "    verb = first_word_that_looks_like_a_verb(span)   # -ed / -ing / -s or irregular\n"
            "    if verb is None:\n"
            "        continue                                      # no verb -> no triple\n"
            "    lemma = lemmatize(verb)                           # 'acquired' -> 'acquire'\n"
            "    if next_word(verb) is a preposition:              # 'based' + 'in'\n"
            "        lemma = lemma + '_' + preposition             # 'base_in'\n"
            "    predicate = canonical_name(lemma)                 # 'base_in' -> 'located_in'\n"
            "    emit Triple(subject, predicate, object, confidence=0.60, cue=verb)",
            language="python"
        )
        verb_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                             method="Nearest-Verb (Generalized)")
        st.write(f"**Live trace** for: *\"{_example_text}\"*")
        if verb_triples:
            st.dataframe(pd.DataFrame(verb_triples)[["subject", "predicate", "object", "confidence", "cue"]],
                        hide_index=True, use_container_width=True)
        else:
            st.info("No verb was found between an adjacent entity pair in this example.")

    with tab_svo:
        write_annotated(
            "Instead of looking at the words between two entities, this method reads the "
            "sentence's grammar using [[dependency_parsing|dependency parsing]]: every word is "
            "labelled with its role (subject, object, ...) and the word it depends on. A "
            "triple is emitted when two entities hang off the same verb - so it works when "
            "the entities are far apart, when one subject has several verbs, and in the "
            "[[passive_voice|passive voice]], where it correctly swaps the direction."
        )
        st.code(
            "doc = parse(sentence)                       # grammar tree (spaCy)\n"
            "for verb in doc.verbs:\n"
            "    subjects = entities attached to verb as nsubj / nsubjpass\n"
            "    for obj in entities attached as dobj  -> Triple(subj, lemma(verb), obj)\n"
            "    for obj in entities attached via a preposition ('compete' + 'with')\n"
            "                                          -> Triple(subj, lemma_prep, obj)\n"
            "    for agent in entities attached via 'by' (passive)\n"
            "                                          -> Triple(agent, lemma(verb), subj)  # swapped\n"
            "    'X is the CEO of Y' -> Triple(X, leads, Y)\n"
            "    a verb joined by 'and' inherits the previous verb's subject",
            language="python"
        )
        if not spacy_parser_available():
            st.warning(
                "This method needs the optional spaCy parser, which is not installed on this "
                f"server. Install it with: `{SPACY_INSTALL_HINT}`"
            )
        else:
            svo_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                                method=SVO_METHOD)
            st.write(f"**Live trace** for: *\"{_example_text}\"*")
            if svo_triples:
                st.dataframe(pd.DataFrame(svo_triples)[["subject", "predicate", "object", "confidence", "cue"]],
                            hide_index=True, use_container_width=True)
            else:
                st.info("No grammatical subject-verb-object link was found in this example.")

            st.markdown("**Why grammar matters: a passive sentence**")
            st.write(f"*\"{_passive_text}\"* - the correct fact is (Google, acquired, YouTube).")
            _parse_doc = _load_spacy_parser()(_passive_text)
            st.caption(
                "Illustration: the sentence's grammar tree. Each arc runs from a word to the word "
                "that depends on it, labelled with its role. Coloured arcs carry the meaning: "
                "`nsubjpass` (receiver of the action), `agent` (the doer, after \"by\") and `pobj`. "
                "Italics show each word's lemma."
            )
            st.plotly_chart(_build_dependency_arc_figure(_parse_doc, theme), use_container_width=True,
                            config={"displayModeBar": False})
            _rows = []
            _passive_methods = ["Pattern-Based (Lexicon Match)", "Nearest-Verb (Generalized)", SVO_METHOD]
            if bert_available():
                _passive_methods.append(BERT_METHOD)
            for _m in _passive_methods:
                _t = extract_relationships(text=_passive_text, entities=_passive_entities, method=_m)
                _got = ", ".join(f"({x['subject']}, {x['predicate']}, {x['object']})" for x in _t) or "(nothing)"
                _ok = any(x["subject"] == "Google" and x["object"] == "YouTube" for x in _t)
                _rows.append({"Method": _m, "Extracted": _got, "Correct direction?": "Yes" if _ok else "No"})
            st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True)

    with tab_bert:
        write_annotated(
            "Instead of rules, this method asks a language model what the text between two "
            "entities *means*. [[bert|BERT]] turns every clause into an [[embedding]] (a "
            "vector of numbers); the names are first hidden "
            "([[entity_masking|entity masking]]) so only the relationship is judged. A small "
            "[[classifier]] then reads that embedding plus the two entity types and answers with "
            "one of the 18 relations, the same relation reversed (the "
            "[[passive_voice|passive voice]]: *X was acquired by Y* means Y is the subject), or "
            "*none*. It was trained on a few thousand example clauses: wordings written for this "
            "lab and Wikipedia sentences from the public Re-DocRED dataset (MIT licence). If the "
            "best answer is *none*, or not confident enough, the pair is skipped."
        )
        st.caption("Illustration: the BERT method step by step (hover each box).")
        st.plotly_chart(_build_bert_steps_figure(theme), use_container_width=True,
                        config={"displayModeBar": False})
        st.code(
            "# trained once (train_bert_re.py): BERT embeddings + entity types of example clauses -> classifier\n"
            "for (subject, object, clause) in candidate_clauses(entities):   # reading rules, see below\n"
            "    clause = 'X ' + text_between + ' Y'                          # names masked\n"
            "    probs = classifier(embed(clause), type(subject), type(object))\n"
            "    best = argmax(probs)\n"
            f"    if best is NONE or probs[best] < {BERT_MIN_PROBABILITY}:\n"
            "        continue                                                # no relation / not confident\n"
            "    if best is the reversed form: swap(subject, object)           # passive voice\n"
            "    emit Triple(subject, best.predicate, object, confidence=probs[best], cue=clause)\n"
            "\n"
            "# candidate_clauses: years are skipped; 'A and B <verb> C' gives both A and B as subjects;\n"
            "# 'A acquired B and partnered with C' -> the part after 'and' is judged with subject A;\n"
            "# 'M, based in R, competes with G' -> after the aside, the subject is M",
            language="python"
        )
        if not bert_available():
            st.warning(
                "This method needs the optional BERT library and model, which are not available on "
                f"this server. Install with: `{BERT_INSTALL_HINT}` (the model, about 90 MB, downloads "
                "on first use)."
            )
        else:
            bert_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                                 method=BERT_METHOD)
            st.write(f"**Live trace** for: *\"{_example_text}\"*")
            if bert_triples:
                st.dataframe(pd.DataFrame(bert_triples)[["subject", "predicate", "object", "confidence", "cue"]],
                             hide_index=True, use_container_width=True)
            else:
                st.info("No clause was similar enough to an example predicate in this sentence.")

            st.markdown("**Step 1 - Reading rules decide which entities to pair**")
            _rr_text = "Microsoft, based in Redmond, acquired LinkedIn and partnered with OpenAI."
            _rr_ents = extract_entities(_rr_text)
            _rr_sentence = _rr_ents[0]["sentence"] if _rr_ents else _rr_text
            _rr_sorted = sorted((e for e in _rr_ents if e["label"] != "DATE"), key=lambda e: e["start"])
            _rr_rows = [{"Pairing": "Neighbours only", "Subject": a["text"], "Object": b["text"],
                         "Clause judged": "X " + _rr_sentence[a["end"]:b["start"]].strip() + " Y"}
                        for a, b in zip(_rr_sorted, _rr_sorted[1:])]
            _rr_rows += [{"Pairing": "Reading rules", "Subject": subj["text"], "Object": obj["text"],
                          "Clause judged": clause}
                         for subj, obj, clause in _bert_candidate_clauses(_rr_sentence, _rr_ents, _MAX_WORDS_BETWEEN)]
            st.caption(
                f"*\"{_rr_text}\"* - pairing only neighbouring entities gives the acquisition to "
                "Redmond and the partnership to LinkedIn. The reading rules make Microsoft the subject "
                "after the comma aside and after \"and\", and judge only the words that belong to that pair."
            )
            st.dataframe(pd.DataFrame(_rr_rows), hide_index=True, use_container_width=True)
            _rr_triples = extract_relationships(text=_rr_text, entities=_rr_ents, method=BERT_METHOD)
            if _rr_triples:
                st.caption("Triples BERT extracts from this sentence:")
                st.dataframe(pd.DataFrame(_rr_triples)[["subject", "predicate", "object", "confidence"]],
                             hide_index=True, use_container_width=True)

            st.markdown("**Step 2 - BERT turns each clause into 384 numbers**")
            _fp_clauses = ["X bought Y", "X acquired Y", "X was born in Y"]
            _fp_vecs = _bert_encode(_fp_clauses)
            if _fp_vecs is not None:
                _sim = _fp_vecs @ _fp_vecs.T
                st.caption(
                    "Each stripe is one clause's embedding, coloured number by number (red = positive, "
                    "blue = negative). The two buying clauses share much of their pattern; the birth clause "
                    f"does not. Cosine similarity: bought vs acquired {_sim[0, 1]:.2f}, bought vs born in "
                    f"{_sim[0, 2]:.2f}, acquired vs born in {_sim[1, 2]:.2f}."
                )
                _hm_fig = _build_bert_embedding_heatmap(_fp_clauses, theme)
                if _hm_fig is not None:
                    st.plotly_chart(_hm_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("**Step 3 - How the decision is made**")
            st.caption(
                "The passive sentence *\"YouTube was purchased by Google\"* becomes the masked clause "
                f"`{_BERT_EXAMPLE_CLAUSE}` (YouTube = X, Google = Y). That exact wording is not among "
                "the example clauses, so this shows BERT generalising by meaning. The bars show how "
                "likely each relation is; the winner must be a relation (not NONE) and clear the dashed "
                "threshold. With the trained classifier the bars are its probabilities (the "
                "\"reversed\" form means the second entity is the subject); hover a bar for details."
            )
            _bar_fig = _build_bert_similarity_figure(_BERT_EXAMPLE_CLAUSE, theme)
            if _bar_fig is not None:
                st.plotly_chart(_bar_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("**Step 4 - The meaning space**")
            st.caption(
                "Each dot is one example clause, squeezed from 384 numbers down to two so it can "
                "be drawn; clauses with similar meanings sit together. The star is the clause above: "
                "it lands inside the *acquired* group, whatever words it uses."
            )
            _map_fig = _build_bert_meaning_map(_BERT_EXAMPLE_CLAUSE, theme)
            if _map_fig is not None:
                st.plotly_chart(_map_fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("**Wordings it has never seen**")
            _unseen = _bert_unseen_wordings_table()
            if _unseen is not None:
                st.caption(
                    "None of these wordings is in the word list or in BERT's training data. The word list "
                    "finds nothing, or only its weak catch-all is_a guess; BERT recognises some of them "
                    "by meaning but gets others wrong - its mistakes are left visible here."
                )
                st.dataframe(_unseen, hide_index=True, use_container_width=True)

            st.markdown("**What it learned from**")
            _train_fig = _build_bert_training_data_figure(theme)
            if _train_fig is not None:
                _stats = _load_json_file(BERT_TRAINING_STATS_PATH) or {}
                _none = _stats.get("per_relation", {}).get("NONE", {})
                st.caption(
                    "Training examples per relation. Re-DocRED (real Wikipedia sentences) covers only some "
                    "relations; the rest rely entirely on wordings written for this lab, which is one reason "
                    "BERT is less reliable on them. Not shown: "
                    f"{_none.get('synthetic', 0) + _none.get('re-docred', 0):,} 'no relation' examples."
                )
                st.plotly_chart(_train_fig, use_container_width=True, config={"displayModeBar": False})

            _blind = _load_json_file(BLIND_RESULTS_PATH)
            _bert_blind = next((r for r in (_blind or {}).get("results", []) if r["method"] == BERT_METHOD), None)
            _blind_note = (
                f" On the blind test it reached F1 {_bert_blind['part_a']['f1']:.2f} on real Wikipedia "
                f"sentences and {_bert_blind['part_b']['f1']:.2f} on hand-written ones - the best of the six "
                "methods, but far below its score on the sample corpora." if _bert_blind else ""
            )
            st.caption(
                "Limits: on 64 wordings it had never seen, the trained classifier got roughly 7 in 10 right, "
                "against 3 in 10 for matching example clauses and almost none for a fixed word list."
                + _blind_note +
                " It can still be wrong, it skips wordings unlike anything it has seen, and BERT cannot "
                "explain why it chose a relation."
            )

    st.divider()
    st.subheader("Illustrated Example: From Raw Sentences to a Knowledge Graph")
    _illustration_text = (
        "Larry Page and Sergey Brin co-founded Google in California in 1998. "
        "Google acquired YouTube in 2006."
    )
    _illustration_entities = extract_entities(_illustration_text)
    _illustration_triples = extract_relationships(
        text=_illustration_text, entities=_illustration_entities,
        method="Hybrid (Pattern + Co-occurrence)"
    )
    st.write("**Step 1 - Annotated source text** (entities highlighted by type, relation cues underlined):")
    _illustration_sentences = _split_sentences(_illustration_text)
    _illustration_cues = [t["cue"] for t in _illustration_triples if t.get("cue")]
    for s_idx, s_text in enumerate(_illustration_sentences):
        s_ents = [e for e in _illustration_entities if e["sentence_idx"] == s_idx]
        st.markdown(_annotate_sentence(s_text, s_ents, _illustration_cues))
    st.write("**Step 2 - Resulting knowledge graph** (same layout engine used in the Simulation section):")
    st.plotly_chart(_build_graph_figure(_illustration_triples, theme, show_predicates=True),
                    use_container_width=True)

    st.divider()
    st.subheader("Comparing the Extraction Methods")
    write_annotated(
        "The qualitative trade-offs above are confirmed empirically below: every method "
        "runs against the **Tech & Business** benchmark corpus and is scored against its "
        "human-curated [[reference_set|gold reference set]]."
    )
    st.markdown(
        "| Method | Precision | Recall | Interpretability | Typical Confidence | Best suited to |\n"
        "|---|---|---|---|---|---|\n"
        "| **Pattern-Based** | High | Lower | Every edge traces to an explicit cue word | 0.80-0.95 "
        "(fixed per pattern) | Clean, well-worded documents where missing a few facts is "
        "acceptable but wrong facts are not |\n"
        "| **Co-occurrence** | Low | Highest | Trivial (proximity only) | 0.20-0.45 (decays with "
        "distance) | Early-stage exploration, or as a recall safety-net alongside a stricter "
        "method |\n"
        "| **Hybrid** | Medium-High | High | Mostly cue-based, with visibly weaker fallback edges | "
        "Mixed (0.20-0.95) | General-purpose default; lets a confidence threshold slider do the "
        "precision/recall trade-off instead of the extraction method itself |\n"
        "| **Nearest-Verb** | Medium | Medium | Cue is the verb itself; reads left-to-right only | "
        "0.60 (fixed) | Text using verbs outside the lexicon; no extra software needed |\n"
        "| **Dependency Parse (SVO)** | High | High | Each edge shows its grammatical roles | "
        "0.80-0.85 | Varied or passive phrasing and distant entities; needs the optional spaCy parser |\n"
        "| **BERT Similarity** | High | Medium-High | Black box: shows the clause and the probability it assigned | "
        "0.50-0.99 (the classifier's probability) | Wordings no rule lists; needs the optional BERT model |\n"
    )
    st.plotly_chart(_build_method_comparison_figure(theme), use_container_width=True)
    st.caption(
        "Precision and Recall move in opposite directions between Pattern-Based and "
        "Co-occurrence - the classic precision/recall trade-off - and Hybrid's F1 lands "
        "between them because it never drops a candidate pair. Co-occurrence scores 0 on the "
        "strict metrics because it always labels edges `related_to`; Pair Recall shows the "
        "share of correct entity pairs it still found. Nearest-Verb and Dependency Parse "
        "name the relation from the sentence itself, and BERT Similarity judges meaning with a "
        "trained classifier. Dependency Parse and BERT are skipped here if spaCy or the BERT "
        "library is not installed."
    )

    st.subheader("How Well Do They Really Work? (Blind Test)")
    write_annotated(
        "The chart above scores the methods on the four sample corpora, which helped shape them, so "
        "it is optimistic ([[overfitting]]). The [[blind_test_set|blind test]] below uses sentences "
        "that played no part in building any method: 60 real Wikipedia sentences from the public "
        "Re-DocRED dataset (Part A) and 46 hand-written sentences (Part B)."
    )
    _blind_fig = _build_blind_test_figure(theme)
    if _blind_fig is None:
        st.info("Blind-test results are not available. Run `python benchmarks/evaluate_blind.py --save` to create them.")
    else:
        st.plotly_chart(_blind_fig, use_container_width=True, config={"displayModeBar": False})
        _blind_meta = _load_json_file(BLIND_RESULTS_PATH) or {}
        st.caption(
            "Every method scores far lower here than on the sample corpora. Real Wikipedia sentences are "
            "long and indirect, so related entities often sit far apart; the rule-based methods find "
            "almost none of them (recall under 3%), while BERT recovers the most. Part A can only judge "
            "the 10 relations Re-DocRED annotates. "
            f"Measured {_blind_meta.get('date', '')}"
            + (" with spaCy and the BERT classifier installed." if _blind_meta.get("spacy_available") and
               _blind_meta.get("bert_classifier_available") else ".")
        )

    st.divider()
    st.subheader("Educational Aim & Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        st.markdown(f"- **Goal {i+1}**: {obj}")

    st.divider()
    st.subheader("Standard Experimental Procedure")
    for step in THEORY_CONTENT["procedure"]:
        st.markdown(f"- {step}")

    st.divider()
    render_glossary_expander()

    st.divider()
    with st.expander("Pipeline Integration: Connection to the Complete KGIRS Architecture", expanded=False):
        st.markdown("""
This laboratory shares an standardized `st.session_state` data contract to integrate seamlessly
across the entire KGIRS curriculum:

| Session State Key | Producer Module | Consumer Modules |
|---|---|---|
| `kg_source_text` | Experiment 1-2 (Preprocessing) or this page | Experiment 6, this page |
| `kg_entities` | Experiment 6 (Named Entity Recognition) | **This page (Exp 7)**, Exp 8, Exp 9 |
| `kg_triples` | **This page (Experiment 7: Relationship Extraction)** | Experiment 8 (Graph DB Ingestion), Exp 9 (Schema Design), Exp 10-11 (Graph Querying) |

Downstream, Experiment 8 and 9 consume the extracted triples directly using an idempotent upsert:
```text
for each triple (subject, predicate, object, confidence):
    upsert node  Entity {name: subject, type: subject_label}
    upsert node  Entity {name: object,  type: object_label}
    upsert edge  subject -[RELATION {type: predicate, confidence: confidence}]-> object
```
        """)


UPLOAD_CHOICE = "Upload Document File (.txt, .pdf, .docx, .csv)"


def _render_upload_preview(loaded: LoadedDocument) -> None:
    """Shows what the extractor will actually process for an uploaded document."""
    text = loaded.text
    sentences = _split_sentences(text)
    with st.container(border=True):
        st.subheader("Document Preview")
        st.caption(loaded.summary())
        cols = st.columns(6 if loaded.page_count else 5)
        cols[0].metric("File type", f".{loaded.extension}" if loaded.extension else "n/a")
        cols[1].metric("Size", f"{loaded.size_bytes / 1024:.1f} KB")
        idx = 2
        if loaded.page_count:
            cols[idx].metric("Pages", loaded.page_count)
            idx += 1
        cols[idx].metric("Characters", f"{len(text):,}")
        cols[idx + 1].metric("Words", f"{len(text.split()):,}")
        cols[idx + 2].metric("Sentences", f"{len(sentences):,}")
        if loaded.truncated:
            st.warning(f"The file has {loaded.raw_character_count:,} characters; only the first "
                       f"{MAX_UPLOAD_CHARACTERS:,} will be processed.")

        tab_text, tab_ents = st.tabs(["Extracted Text", "Entities Preview"])
        with tab_text:
            st.caption("This is exactly the text the extraction pipeline will run on.")
            with st.container(height=260):
                st.markdown(
                    '<div style="font-family: Verdana, \'Segoe UI\', Tahoma, sans-serif; font-size: 0.82rem; '
                    f'line-height: 1.7; white-space: pre-wrap;">{html.escape(text)}</div>',
                    unsafe_allow_html=True
                )
        with tab_ents:
            entities = extract_entities(text)
            st.caption(f"{len({e['text'].lower() for e in entities})} distinct entities found across "
                       f"{len(sentences)} sentences. First sentences shown with entities highlighted:")
            for s_idx, s_text in enumerate(sentences[:5]):
                s_ents = [e for e in entities if e["sentence_idx"] == s_idx]
                st.markdown(_annotate_sentence(s_text, s_ents, []))


def render_simulation_section():
    """Renders Section 2: Interactive Sandbox, 6 Analytical Tabs, Evaluation, and Exports."""
    theme = get_current_theme()

    st.header("Interactive Relationship Extraction Sandbox")
    write_annotated(
        "Configure extraction parameters, run [[pattern_based|Pattern-Based]], "
        "[[cooccurrence|Co-occurrence]], or [[hybrid|Hybrid]] extraction, and evaluate the "
        "resulting [[triple|triples]] against a [[reference_set|ground truth]] where available."
    )

    # 1. Document Selection Card
    with st.container():
        doc_names = list(CORPORA.keys()) + ["Custom Input", UPLOAD_CHOICE]
        doc_choice = st.selectbox(
            "Select Source Document / Corpus",
            options=doc_names,
            help="Choose from benchmark pedagogical corpora with ground truth reference sets, or supply custom text."
        )

        selected_corpus = CORPORA.get(doc_choice)
        is_upload = doc_choice == UPLOAD_CHOICE
        default_text = ""
        gold_triples = []
        loaded: Optional[LoadedDocument] = None
        uploaded_file = None

        if selected_corpus:
            default_text = selected_corpus["text"]
            gold_triples = selected_corpus.get("gold", [])
            col_b1, col_b2 = st.columns([1, 4])
            with col_b1:
                st.info(f"**Level:** {selected_corpus['level']}")
            with col_b2:
                st.caption(f"**Pedagogical Purpose:** {selected_corpus['purpose']}")
            with st.expander("Things to Observe in this Document", expanded=False):
                for obs in selected_corpus.get("observe", []):
                    st.markdown(f"- {obs}")
        elif is_upload:
            uploaded_file = st.file_uploader(
                "Upload document",
                type=["txt", "md", "csv", "tsv", "pdf", "docx"],
                help="Upload a file up to 25,000 characters."
            )
            if uploaded_file is not None:
                # Parse once per file (not on every rerun) and remember the result.
                cached = st.session_state.get("exp7_uploaded_doc")
                if not cached or cached[0] != uploaded_file.file_id:
                    cached = (uploaded_file.file_id, load_uploaded_document(uploaded_file))
                    st.session_state["exp7_uploaded_doc"] = cached
                loaded = cached[1]
        else:
            upstream = st.session_state.get("kg_source_text", "")
            default_text = upstream if upstream else "Larry Page co-founded Google in California in 1998."

        # When the chosen source changes, reset the text box and drop results from the old source.
        if is_upload:
            source_sig = f"upload:{uploaded_file.file_id if uploaded_file is not None else 'none'}"
        else:
            source_sig = doc_choice
        if st.session_state.get("exp7_source_sig") != source_sig:
            st.session_state["exp7_source_sig"] = source_sig
            st.session_state["sandbox_text_area"] = default_text
            for stale in ("exp7_last_triples", "exp7_last_entities", "exp7_last_text",
                          "exp7_last_method", "exp7_last_corpus", "exp7_last_eval"):
                st.session_state.pop(stale, None)

        can_run = True
        if is_upload:
            # The uploaded document itself is processed: no editable text box in this mode.
            text_input = loaded.text if (loaded and not loaded.error) else ""
            if uploaded_file is None:
                st.info("Upload a document above to preview it and run extraction on it.")
                can_run = False
            elif loaded.error:
                st.error(loaded.error)
                can_run = False
            elif not text_input.strip():
                st.warning(
                    f"No text could be read from `{loaded.filename}`. A scanned PDF or an image-only "
                    "file has no text layer to extract from."
                )
                can_run = False
            else:
                _render_upload_preview(loaded)
        else:
            text_input = st.text_area(
                "Document Text (editable for experimentation)",
                height=140,
                key="sandbox_text_area"
            )

    # 2. Parameters Card
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        method = st.radio(
            "Extraction Strategy",
            options=[
                "Co-occurrence Only",
                "Pattern-Based (Lexicon Match)",
                "Hybrid (Pattern + Co-occurrence)",
                "Nearest-Verb (Generalized)",
                "Dependency Parse (SVO)",
                BERT_METHOD,
            ],
            index=2,
            help="Pattern-Based prioritizes precision; Co-occurrence maximizes recall; Hybrid balances both; "
                 "Nearest-Verb uses the verb between entities; Dependency Parse reads sentence grammar (needs spaCy); "
                 "BERT Similarity judges the meaning of the text between entities with a trained classifier (needs sentence-transformers)."
        )
        if method == SVO_METHOD and not spacy_parser_available():
            st.warning(f"The Dependency Parse method needs the optional spaCy parser: `{SPACY_INSTALL_HINT}`. "
                       "It will return no triples until installed.")
    with col_p2:
        min_confidence = st.slider(
            "Minimum Confidence Cutoff",
            min_value=0.10,
            max_value=1.0,
            value=0.30,
            step=0.05,
            help="Filters out triples whose confidence falls below this threshold."
        )
    with col_p3:
        max_dist = st.slider(
            "Max Word Separation",
            min_value=4,
            max_value=25,
            value=12,
            step=1,
            help="Maximum words allowed between two entities in a sentence to form a candidate pair."
        )
        use_upstream_ner = st.checkbox(
            "Reuse upstream entities (Exp 6)",
            value=True,
            help="If Experiment 6 already executed in this session, reuse its recognized entities."
        )

    run_btn = st.button("Run Relationship Extraction Pipeline", type="primary", use_container_width=True,
                        disabled=not can_run)

    # Execution or cached state retrieval
    if run_btn or ("exp7_last_triples" in st.session_state and can_run):
        if run_btn:
            upstream_ents = st.session_state.get("kg_entities")
            if use_upstream_ner and upstream_ents and st.session_state.get("kg_source_text") == text_input:
                entities = upstream_ents
            else:
                entities = extract_entities(text_input)

            triples = extract_relationships(
                text=text_input,
                entities=entities,
                method=method,
                max_word_distance=max_dist
            )
            filtered_triples = [t for t in triples if t["confidence"] >= min_confidence]

            st.session_state["exp7_last_entities"] = entities
            st.session_state["exp7_last_triples"] = filtered_triples
            st.session_state["exp7_last_text"] = text_input
            st.session_state["exp7_last_method"] = method
            st.session_state["exp7_last_corpus"] = doc_choice

            # Downstream pipeline session state contract
            st.session_state["kg_entities"] = entities
            st.session_state["kg_triples"] = filtered_triples
            st.session_state["kg_source_text"] = text_input

        entities = st.session_state.get("exp7_last_entities", [])
        triples = st.session_state.get("exp7_last_triples", [])
        metrics = compute_extraction_metrics(entities, triples)
        eval_result = evaluate_against_reference_set(triples, gold_triples)
        st.session_state["exp7_last_eval"] = eval_result

        st.divider()

        # Metrics Banner
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Entities Identified", metrics["entities"])
        m2.metric("Triples Extracted", metrics["triples"])
        m3.metric("High-Confidence", metrics["high_confidence_triples"])
        m4.metric("Avg. Confidence", f"{metrics['avg_confidence']:.2f}")
        m5.metric("Rel. Density", metrics["relation_density"])
        if eval_result.get("has_gold"):
            m6.metric("F1 Score", f"{eval_result['f1']:.2f}", delta=f"P:{eval_result['precision']:.2f} R:{eval_result['recall']:.2f}")
        else:
            m6.metric("Unique Preds", metrics["unique_predicates"])

        # 6 Comprehensive Analytical Tabs
        tab_triples, tab_annotated, tab_entities, tab_graph, tab_flow, tab_eval = st.tabs([
            "Extracted Triples (SPO)",
            "Annotated Text (Evidence)",
            "Detected Entities",
            "Knowledge Graph Preview",
            "Entity-Relationship Flow",
            "Evaluation against Ground Truth"
        ])

        # TAB 1: TRIPLES
        with tab_triples:
            if triples:
                df_triples = pd.DataFrame(triples)[[
                    "subject", "predicate", "object", "confidence",
                    "subject_label", "object_label", "method", "cue", "sentence"
                ]]
                df_triples.columns = [
                    "Subject", "Predicate", "Object", "Confidence",
                    "Subj Type", "Obj Type", "Method", "Matched Cue", "Source Sentence"
                ]
                st.dataframe(df_triples, hide_index=True, use_container_width=True)
            else:
                st.warning("No triples met the confidence threshold. Try lowering the threshold or switching to Hybrid mode.")

        # TAB 2: ANNOTATED TEXT
        with tab_annotated:
            write_annotated(
                "**Sentence Evidence Inspection:** Each sentence displays identified "
                "[[entity|entities]] with type tags and matched [[relation_cue|relation cues]] "
                "highlighted - the direct textual [[provenance]] for every extracted triple."
            )
            sentences = _split_sentences(st.session_state.get("exp7_last_text", text_input))
            cues = [t["cue"] for t in triples if t.get("cue")]

            for s_idx, s_text in enumerate(sentences):
                s_ents = [e for e in entities if e["sentence_idx"] == s_idx]
                annotated_html = _annotate_sentence(s_text, s_ents, cues)
                st.markdown(f"**Sentence {s_idx + 1}:** {annotated_html}")

        # TAB 3: DETECTED ENTITIES
        with tab_entities:
            col_e1, col_e2 = st.columns([3, 2])
            with col_e1:
                if entities:
                    ent_df = pd.DataFrame(entities).drop_duplicates(subset=["text"])[["text", "label", "sentence_idx"]]
                    ent_df.columns = ["Entity Mention", "Classified Type", "First Sentence #"]
                    st.dataframe(ent_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No entities detected.")
            with col_e2:
                if entities:
                    st.write("**Entity Type Distribution**")
                    st.plotly_chart(_build_entity_type_chart(entities, theme), use_container_width=True)

        # TAB 4: KNOWLEDGE GRAPH PREVIEW
        with tab_graph:
            col_gc1, col_gc2, col_gc3 = st.columns([2, 3, 2])
            with col_gc1:
                show_labels = st.checkbox("Show predicate labels on edges", value=True, key="graph_show_preds")
            with col_gc2:
                selected_types = st.multiselect(
                    "Filter by Entity Type",
                    options=ENTITY_TYPE_ORDER,
                    default=ENTITY_TYPE_ORDER,
                    key="graph_type_filter"
                )
            with col_gc3:
                if st.button("Re-layout / Jiggle Graph", use_container_width=True):
                    st.session_state["graph_seed"] = st.session_state.get("graph_seed", 11) + 7
                    st.rerun()

            col_gd1, col_gd2 = st.columns([2, 3])
            with col_gd1:
                show_weak = st.checkbox(
                    "Show weak links (related_to)", value=False, key="graph_show_weak",
                    help="Weak links are co-occurrence edges with no explicit relation cue. Hidden by "
                         "default to keep the graph readable (ignored if there is nothing else to draw)."
                )
            with col_gd2:
                degree_count: Dict[str, int] = {}
                for t in triples:
                    for n in (t["subject"], t["object"]):
                        degree_count[n] = degree_count.get(n, 0) + 1
                focus_options = [ALL_ENTITIES] + sorted(degree_count, key=lambda n: (-degree_count[n], n))
                focus = st.selectbox("Focus on entity", focus_options, key="graph_focus",
                                     help="Show only one entity and the entities directly linked to it.")

            curr_seed = st.session_state.get("graph_seed", 11)
            _, graph_info = _prepare_graph_triples(triples, selected_types, show_weak, MAX_GRAPH_NODES, focus)
            graph_fig = _build_graph_figure(
                triples=triples,
                theme=theme,
                show_predicates=show_labels,
                seed=curr_seed,
                type_filter=selected_types,
                show_weak=show_weak,
                max_entities=MAX_GRAPH_NODES,
                focus=focus
            )
            st.plotly_chart(graph_fig, use_container_width=True)
            notes = []
            if graph_info["hidden_weak"]:
                notes.append(f"{graph_info['hidden_weak']} weak link(s) hidden")
            if graph_info["omitted_entities"]:
                notes.append(f"{graph_info['omitted_entities']} of {graph_info['total_entities']} "
                             "entities left out (pick one under 'Focus on entity' to explore them)")
            if notes:
                st.caption("Showing the clearest part of the graph: " + "; ".join(notes) + ".")
            st.caption("Tip: Hover over any node or edge to inspect full entity attributes, source sentences, and extraction cues.")

        # TAB 5: FLOW (SANKEY)
        with tab_flow:
            st.write("**Entity Type to Predicate Transition Patterns:**")
            st.plotly_chart(_build_flow_figure(triples, theme), use_container_width=True)
            write_annotated(
                "Traces how [[entity_type|entity types]] flow from subjects into "
                "[[predicate|predicates]] and target object types."
            )

        # TAB 6: EVALUATION
        with tab_eval:
            if eval_result.get("has_gold"):
                st.subheader(f"Ground Truth Evaluation Benchmark ({selected_corpus['name'] if selected_corpus else 'Corpus'})")
                write_annotated(
                    "Every extracted [[triple]] is checked against this corpus's "
                    "[[reference_set|hand-curated gold set]]: a match is a True Positive, an "
                    "extracted triple with no match is a [[false_positive|False Positive]], and "
                    "a gold triple nothing matched is a [[false_negative|False Negative]]."
                )
                col_ev1, col_ev2, col_ev3, col_ev4 = st.columns(4)
                col_ev1.metric("Precision (P)", f"{eval_result['precision'] * 100:.1f}%", help="Fraction of extracted triples that are factually correct.")
                col_ev2.metric("Recall (R)", f"{eval_result['recall'] * 100:.1f}%", help="Fraction of ground truth relations successfully recovered.")
                col_ev3.metric("F1 Score", f"{eval_result['f1'] * 100:.1f}%", help="Harmonic mean of precision and recall.")
                col_ev4.metric("Pair Recall", f"{eval_result['pair_recall'] * 100:.1f}%", help="Fraction of ground truth (subject, object) pairs found, ignoring the predicate label. Unlike the three metrics to the left, this gives credit to a method (e.g. Co-occurrence) that finds the right entities but not the right relation name.")
                if eval_result["precision"] == 0.0 and eval_result["f1"] == 0.0 and eval_result["pair_recall"] > 0.0:
                    st.caption(
                        f"Precision/Recall/F1 are 0% because this method requires an exact predicate "
                        f"match, and none of its predicted labels match the gold set's specific relation "
                        f"names. Pair Recall shows it still found {eval_result['pair_recall'] * 100:.0f}% of "
                        f"the correct entity connections - just without naming the relationship correctly."
                    )

                st.write(f"**Summary:** True Positives: `{eval_result['tp_count']}` | False Positives: `{eval_result['fp_count']}` | False Negatives (Omissions): `{eval_result['fn_count']}`")

                col_tp, col_fp, col_fn = st.columns(3)
                with col_tp:
                    st.success(f"**True Positives ({eval_result['tp_count']})**")
                    if eval_result["tp_list"]:
                        st.dataframe(pd.DataFrame(eval_result["tp_list"])[["subject", "predicate", "object"]], hide_index=True)
                    else:
                        st.caption("None.")
                with col_fp:
                    st.warning(f"**False Positives ({eval_result['fp_count']})**")
                    if eval_result["fp_list"]:
                        st.dataframe(pd.DataFrame(eval_result["fp_list"])[["subject", "predicate", "object"]], hide_index=True)
                    else:
                        st.caption("None.")
                with col_fn:
                    st.error(f"**False Negatives ({eval_result['fn_count']})**")
                    if eval_result["fn_list"]:
                        st.dataframe(pd.DataFrame(eval_result["fn_list"], columns=["Subject", "Predicate", "Object"]), hide_index=True)
                    else:
                        st.caption("None.")
            else:
                st.info("Evaluation against ground truth is available for built-in benchmark corpora (e.g. Tech & Business, Science & Academia, Indian Industry, Healthcare). Custom and uploaded texts do not carry pre-annotated reference sets.")

        # 3. Export Card
        st.divider()
        st.subheader("Export for Downstream Graph Ingestion (Experiments 8-9)")
        col_x1, col_x2, col_x3 = st.columns(3)

        exp_df = pd.DataFrame(triples) if triples else pd.DataFrame(columns=["subject", "predicate", "object", "confidence"])
        with col_x1:
            st.download_button(
                "Download Triples (CSV)",
                data=exp_df.to_csv(index=False).encode("utf-8"),
                file_name="exp7_extracted_triples.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_x2:
            st.download_button(
                "Download Triples (JSON)",
                data=json.dumps(triples, indent=2).encode("utf-8"),
                file_name="exp7_extracted_triples.json",
                mime="application/json",
                use_container_width=True
            )
        with col_x3:
            kg_export = nx.MultiDiGraph()
            for t in triples:
                kg_export.add_node(t["subject"], type=t["subject_label"])
                kg_export.add_node(t["object"], type=t["object_label"])
                kg_export.add_edge(t["subject"], t["object"], predicate=t["predicate"],
                                   confidence=float(t["confidence"]))
            graphml_buffer = io.BytesIO()
            nx.write_graphml(kg_export, graphml_buffer)
            st.download_button(
                "Download Graph (GraphML)",
                data=graphml_buffer.getvalue(),
                file_name="exp7_knowledge_graph.graphml",
                mime="application/graphml+xml",
                use_container_width=True
            )

        # 4. Trial Logging
        st.divider()
        st.subheader("Record Current Trial to Session Logbook")
        if st.button("Record Trial to Logbook", use_container_width=True):
            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Document": doc_choice,
                "Method": method,
                "Min Conf": min_confidence,
                "Max Dist": max_dist,
                "Entities": metrics["entities"],
                "Triples": metrics["triples"],
                "High Conf": metrics["high_confidence_triples"],
                "Avg Conf": metrics["avg_confidence"],
                "F1 Score": eval_result.get("f1", "N/A") if eval_result.get("has_gold") else "N/A"
            }
            st.session_state["trials"].append(trial_record)
            st.success(f"Trial #{trial_record['Trial #']} successfully recorded into session logbook.")

    # Display Trials Table if any recorded
    if st.session_state.get("trials"):
        st.divider()
        st.subheader("Recorded Session Trials Logbook")
        st.dataframe(pd.DataFrame(st.session_state["trials"]), hide_index=True, use_container_width=True)
        if st.button("Clear Session Trials Logbook"):
            st.session_state["trials"] = []
            st.rerun()


def render_quiz_section():
    """Renders Section 3: Self-grading Conceptual Assessment Quiz."""
    st.header("Conceptual Assessment Quiz")
    st.write("Complete all questions to assess your understanding of Information Retrieval and Relationship Extraction principles.")

    user_responses = {}
    with st.form("quiz_form"):
        for q in QUIZ_QUESTIONS:
            st.subheader(f"Question {q['id']}")
            st.write(q["question"])
            key = f"quiz_q_{q['id']}"
            choice = st.radio(
                f"Options for Question {q['id']}:",
                options=q["options"],
                index=None,
                key=key,
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = choice
            st.markdown("---")

        submit_btn = st.form_submit_button("Submit Assessment for Evaluation", type="primary", use_container_width=True)

    if submit_btn:
        score = 0
        unanswered = 0
        feedback = []

        for q in QUIZ_QUESTIONS:
            selected = user_responses.get(q["id"])
            if selected is None:
                unanswered += 1
                feedback.append((q, False, "Question not answered."))
            else:
                sel_idx = q["options"].index(selected)
                is_correct = (sel_idx == q["answer_index"])
                if is_correct:
                    score += 1
                feedback.append((q, is_correct, selected))

        st.session_state["quiz_submitted"] = True
        st.session_state["quiz_score"] = score
        st.session_state["quiz_feedback"] = feedback

        pct = int((score / len(QUIZ_QUESTIONS)) * 100)
        if pct >= 80:
            st.balloons()
            st.success(f"Excellent! Your Score: **{score} / {len(QUIZ_QUESTIONS)} ({pct}%)**")
        elif pct >= 50:
            st.success(f"Good Effort! Your Score: **{score} / {len(QUIZ_QUESTIONS)} ({pct}%)**")
        else:
            st.error(f"Needs Review. Your Score: **{score} / {len(QUIZ_QUESTIONS)} ({pct}%)**")

        st.subheader("Detailed Pedagogical Review & Explanations")
        incorrect = len(feedback) - score - unanswered
        st.markdown(
            f"**Result breakdown:** :green[{score} Correct]  |  :red[{incorrect} Incorrect]"
            + (f"  |  :orange[{unanswered} Unanswered]" if unanswered else "")
        )
        for q, is_corr, sel in feedback:
            status_label = "Correct" if is_corr else "Incorrect"
            with st.expander(f"Q{q['id']}: {q['question']} — [{status_label}]", expanded=not is_corr):
                if is_corr:
                    st.success(f"**Your Answer:** {sel}\n\nThis is correct.")
                else:
                    st.error(f"**Your Answer:** {sel}\n\nThis is incorrect.")
                    st.success(f"**Correct Answer:** {q['options'][q['answer_index']]}")
                st.info(f"**Pedagogical Explanation:** {q['explanation']}")

    elif st.session_state.get("quiz_submitted", False):
        st.info(f"Quiz previously submitted in this session. Score: **{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}**")


def render_report_section():
    """Renders Section 4: Dynamic Official Laboratory Report Generation & PDF Download."""
    st.header("Official Laboratory Report Generation")
    st.write("Compile student details, session trials, extracted knowledge triples, and quiz results into a verified PDF report.")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        student_name = st.text_input("Student Full Name", value=st.session_state["student_info"].get("name", "Student Name"))
    with col_s2:
        student_id = st.text_input("Roll Number / ID", value=st.session_state["student_info"].get("id", "EXP7-001"))
    with col_s3:
        student_batch = st.text_input("Academic Batch", value=st.session_state["student_info"].get("batch", "D17A"))
    with col_s4:
        lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["batch"] = student_batch
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Student Analytical Observations & Conclusions")
    student_notes = st.text_area(
        "Enter your interpretation of results, comparisons across extraction methods, and conclusions:",
        value=st.session_state.get("student_notes", (
            "In this experiment, pattern-based relation extraction demonstrated high precision on active-voice "
            "corpora with direct verbal cues ('founded', 'acquired'). The co-occurrence baseline achieved higher "
            "recall but introduced false positives, underscoring the classic IR precision-recall trade-off. "
            "Confidence scoring successfully filtered noise prior to downstream graph construction."
        )),
        height=110
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state.get("trials", []))
    triples_df = pd.DataFrame(st.session_state.get("kg_triples", []))
    eval_metrics = st.session_state.get("exp7_last_eval")

    st.divider()
    st.subheader("Report Summary Preview")
    st.write(f"**Institution:** {EXPERIMENT_CONFIG['department']} | {EXPERIMENT_CONFIG['course']}")
    st.write(f"**Student:** {student_name} (Roll: `{student_id}`, Batch: `{student_batch}`) | **Date:** {lab_date}")
    st.write(f"**Assessment Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}` | **Recorded Trials:** `{len(trials_df)}`")

    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        student_batch=student_batch,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=student_notes,
        triples_df=triples_df,
        eval_metrics=eval_metrics
    )

    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Official Laboratory Report (.pdf)")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="Download Official lab_report.pdf",
            data=pdf_bytes,
            file_name=f"KGIRS_Exp7_Report_{student_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    with col_d2:
        st.link_button(
            "Open PDF Document in Browser Tab",
            url="/app/static/lab_report.pdf",
            use_container_width=True
        )


# ======================================================================================
# 12. INITIALIZATION & MAIN NAVIGATION
# ======================================================================================

def init_session_state():
    """Initializes session state keys."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP7-001",
            "batch": "D17A",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""
    if "kg_entities" not in st.session_state:
        st.session_state["kg_entities"] = []
    if "kg_triples" not in st.session_state:
        st.session_state["kg_triples"] = []
    if "kg_source_text" not in st.session_state:
        st.session_state["kg_source_text"] = ""
    if "graph_seed" not in st.session_state:
        st.session_state["graph_seed"] = 11


def main():
    st.set_page_config(
        page_title="Exp 7: Relationship Extraction | KGIRS Virtual Lab",
        page_icon=None,
        layout="wide"
    )

    init_session_state()
    inject_wiki_term_style()

    # Native Streamlit Title (No custom CSS, aside from the term pop-up stylesheet above).
    # Light/Dark mode is switched via Streamlit's own top-right "..." menu -> Settings.
    st.title(EXPERIMENT_CONFIG["title"])
    st.caption(f"{EXPERIMENT_CONFIG['department']} | {EXPERIMENT_CONFIG['course']}")

    # Navigation Sidebar
    section = st.sidebar.radio(
        "Lab Navigator",
        options=["Theory", "Simulation", "Quiz", "Report Generation"]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Progress Tracker")
    quiz_status = "Done" if st.session_state.get("quiz_submitted", False) else "Pending"
    st.sidebar.write(f"- **Quiz Status:** {quiz_status}")
    if st.session_state.get("quiz_submitted", False):
        st.sidebar.write(f"- **Quiz Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}`")
    st.sidebar.write(f"- **Trials Logged:** `{len(st.session_state['trials'])}`")
    st.sidebar.write(f"- **Active Triples in Session:** `{len(st.session_state.get('kg_triples', []))}`")

    st.sidebar.divider()
    st.sidebar.caption(
        "Knowledge graph triples emitted here populate `st.session_state['kg_triples']` "
        "for Experiments 8 and 9 (Graph Database Ingestion & Import)."
    )

    # Section Dispatcher
    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()


if __name__ == "__main__":
    main()
