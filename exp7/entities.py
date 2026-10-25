"""Named-entity recognition: capitalisation rules, spaCy NER and TERM concepts."""

from __future__ import annotations

import re
from typing import Any, Dict, List

import streamlit as st


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
    "For", "And", "But", "Or", "If", "As", "Of", "Is", "Are", "Was", "Were"
}


_ORG_SUFFIXES = (
    "Inc", "Inc.", "Ltd", "Ltd.", "Corp", "Corp.", "LLC", "University",
    "Institute", "Company", "Corporation", "Labs", "Laboratories",
    "Foundation", "Group", "Technologies", "Systems", "Council", "Hospital"
)


_LOC_HINTS = (
    "City", "State", "Street", "River", "Mountain", "Island", "Ocean",
    "Valley", "County", "Province", "Republic", "Kingdom", "Nation"
)


def _split_sentences(text: str) -> List[str]:
    """Splits document text into clean sentences while respecting common honorifics and abbreviations."""
    clean = text.strip().replace("\r\n", " ").replace("\n", " ")
    clean = re.sub(r"\b(Dr|Prof|Mr|Mrs|Ms|vs|Inc|Ltd|Corp|e\.g|i\.e)\.", r"\1<DOT>", clean)
    parts = re.split(r"(?<=[.!?])\s+", clean)
    sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return sentences


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Lightweight deterministic NER fallback identifying PERSON, ORG, LOCATION, DATE, and TERM."""
    sentences = _split_sentences(text)
    entities = []
    entity_id = 1

    known_locations = {
        "california", "mountain view", "redmond", "warsaw", "paris", "london",
        "mumbai", "bengaluru", "chennai", "delhi", "tokyo", "new york", "san francisco",
        "india", "france", "poland", "usa", "switzerland", "cambridge", "oxford"
    }
    known_orgs = {
        "google", "alphabet", "microsoft", "youtube", "linkedin", "openai",
        "infosys", "tata group", "tata consultancy services", "curie institute",
        "university of paris", "indian space research organisation", "isro",
        "stanford university", "harvard", "mit", "who", "pfizer", "apple", "amazon"
    }

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
            if span_text in _STOPWORD_CAPS:
                continue
            if (match.start() == 0 and " " not in span_text
                    and span_text.lower() in lowercase_words):
                continue

            lowered = span_text.lower()
            if lowered in known_locations or any(hint in span_text for hint in _LOC_HINTS):
                label = "LOCATION"
            elif lowered in known_orgs or any(span_text.endswith(sfx) for sfx in _ORG_SUFFIXES):
                label = "ORG"
            elif re.search(r"\b(University|Institute|College|Academy|Hospital|Board)\b", span_text):
                label = "ORG"
            else:
                label = "PERSON"

            entities.append({
                "id": f"e{entity_id}",
                "text": span_text,
                "label": label,
                "start": match.start(),
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
