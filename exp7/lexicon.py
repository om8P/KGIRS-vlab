"""Rule-based relation extraction helpers: cue lexicon, verb lemmatiser and nearest-verb search."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


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
