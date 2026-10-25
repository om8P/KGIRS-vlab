"""BERT relation extraction: Sentence-BERT embeddings, the trained classifier and reading rules."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import streamlit as st

from exp7.entities import extract_entities
from exp7.lexicon import _MAX_WORDS_BETWEEN
from exp7.paths import PROJECT_ROOT


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


BERT_CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "bert_re_classifier.npz")


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
