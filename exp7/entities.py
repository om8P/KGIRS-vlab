"""Named-entity recognition: capitalisation rules, spaCy NER and TERM concepts."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

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
