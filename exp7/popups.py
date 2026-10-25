"""Wikipedia-style hover pop-ups for technical terms, and the markdown renderer that uses them."""

from __future__ import annotations

import html
import re
from typing import List, Tuple

import pandas as pd
import streamlit as st

from exp7.glossary import TERMS
from exp7.theme import get_current_theme


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
