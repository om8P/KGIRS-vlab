"""The extraction entry point that dispatches to every method, plus metrics and evaluation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from exp7.entities import extract_entities
from exp7.lexicon import _MAX_WORDS_BETWEEN, _RELATION_PATTERNS, _find_nearest_verb
from exp7.svo import extract_svo_relationships


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
