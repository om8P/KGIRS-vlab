"""
Scores every extraction method in the lab on benchmarks/blind_test.json.

Matching (identical for every method):
  * entity names: lower-cased, leading "the/a/an" and trailing possessive removed, whitespace folded;
  * predicates: the same synonym folding as the lab's evaluate_against_reference_set;
  * married_to, collaborates_with, competes_with, merged_with are symmetric, so either direction counts.
Part A (Re-DocRED) can only judge the predicates its mapping covers; a triple with any other
predicate is reported as "unjudged" instead of being counted as a false positive.

Report aggregate scores only; never tune on individual errors in this set.

Usage:   python benchmarks/evaluate_blind.py          (print the table)
         python benchmarks/evaluate_blind.py --save   (also write benchmarks/blind_results.json,
                                                       which the lab's Theory page charts)
"""

import json
from datetime import date
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYMMETRIC = {"married_to", "collaborates_with", "competes_with", "merged_with"}


def load_app():
    """The lab's extraction functions, imported from the exp7 package."""
    sys.path.insert(0, str(HERE.parent))
    from types import SimpleNamespace
    from exp7.bert_method import bert_classifier_available
    from exp7.charts import _COMPARISON_METHODS
    from exp7.entities import spacy_parser_available
    from exp7.extraction import extract_relationships
    return SimpleNamespace(
        extract_relationships=extract_relationships, _COMPARISON_METHODS=_COMPARISON_METHODS,
        spacy_parser_available=spacy_parser_available, bert_classifier_available=bert_classifier_available)


def norm_entity(text):
    t = " ".join(text.lower().split())
    t = re.sub(r"^(the|a|an)\s+", "", t)
    t = re.sub(r"('s|')$", "", t)
    return t.strip(" .,;:")


def norm_predicate(p):
    p = p.lower().replace(" ", "_")
    if p in ("co-founded", "started", "established"):
        return "founded"
    if p in ("bought", "purchased", "took_over"):
        return "acquired"
    if p in ("headquartered_in", "based_in"):
        return "located_in"
    return p


def key(s, p, o):
    return norm_entity(s), norm_predicate(p), norm_entity(o)


def score(app, documents, method, judged_predicates=None):
    tp = fp = fn = unjudged = 0
    for doc in documents:
        gold = {key(*g) for g in doc["gold"]}
        matched = set()
        for t in app.extract_relationships(doc["text"], None, method):
            k = key(t["subject"], t["predicate"], t["object"])
            if judged_predicates is not None and k[1] not in judged_predicates:
                unjudged += 1
                continue
            hit = k if k in gold else (
                (k[2], k[1], k[0]) if k[1] in SYMMETRIC and (k[2], k[1], k[0]) in gold else None)
            if hit and hit not in matched:
                tp += 1
                matched.add(hit)
            elif not hit:
                fp += 1
        fn += len(gold - matched)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "unjudged": unjudged, "p": p, "r": r, "f1": f1}


def main():
    data = json.loads((HERE / "blind_test.json").read_text(encoding="utf-8"))
    judged = set(data["part_a_judged_predicates"])
    part_a = [d for d in data["documents"] if d["part"] == "A"]
    part_b = [d for d in data["documents"] if d["part"] == "B"]
    app = load_app()
    print(f"spaCy available: {app.spacy_parser_available()}   BERT classifier available: {app.bert_classifier_available()}")
    print(f"Part A: {len(part_a)} Re-DocRED sentences   Part B: {len(part_b)} hand-written sentences\n")
    header = f"{'method':34} {'part A P/R/F1':>20} {'unjudged':>9} {'part B P/R/F1':>20} {'both F1':>8}"
    print(header)
    print("-" * len(header))
    results = []
    for method in app._COMPARISON_METHODS:
        a = score(app, part_a, method, judged)
        b = score(app, part_b, method)
        tp, fp, fn = a["tp"] + b["tp"], a["fp"] + b["fp"], a["fn"] + b["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        both = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{method[:34]:34} {a['p']:.2f}/{a['r']:.2f}/{a['f1']:.2f}".ljust(55)
              + f" {a['unjudged']:>9} " + f"{b['p']:.2f}/{b['r']:.2f}/{b['f1']:.2f}".rjust(20) + f" {both:>8.2f}")
        results.append({"method": method, "part_a": a, "part_b": b,
                        "both": {"p": p, "r": r, "f1": both}})
    if "--save" in sys.argv:
        out = HERE / "blind_results.json"
        out.write_text(json.dumps({
            "date": date.today().isoformat(),
            "spacy_available": app.spacy_parser_available(),
            "bert_classifier_available": app.bert_classifier_available(),
            "part_a_sentences": len(part_a), "part_b_sentences": len(part_b),
            "results": results,
        }, indent=1), encoding="utf-8")
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
