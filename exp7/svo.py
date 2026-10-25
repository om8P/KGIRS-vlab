"""Dependency-parse (subject-verb-object) relation extraction using spaCy."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from exp7.entities import _load_spacy_parser, extract_entities
from exp7.lexicon import _ROLE_NOUNS, _VERB_TO_CANONICAL, _canonical_predicate


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
