"""
Builds benchmarks/blind_test.json: a held-out test set for the lab's extraction methods.

Part A - Re-DocRED test split (public, MIT licence, https://huggingface.co/datasets/tonytan48/Re-DocRED).
    Real Wikipedia sentences with human-annotated relations. This split is never used for training,
    for reporting dev scores, or for designing rules. Only the relations that map onto the lab's
    predicates (see REDOCRED_MAP in train_bert_re.py) are used, so Part A can judge only those.

Part B - Hand-written sentences, written before any change they are used to measure. They cover
    the relations Re-DocRED has no equivalent for (acquired, discovered, treats, causes,
    competes_with, collaborates_with, merged_with, reports_to), plus passive voice, lowercase
    concepts, coordinated subjects, asides, and distractor sentences with no relation at all.
    Note: the same author wrote the BERT classifier's synthetic training wordings.

Rule for using this file: report aggregate scores only; never tune anything on its errors.

Usage:   python benchmarks/build_blind_test.py
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_bert_re import REDOCRED_MAP, _detok  # noqa: E402

OUT = HERE / "blind_test.json"
# Stricter than the training mapping: Wikidata "country" (P17) and "location" (P276) are attached to
# almost anything (parties, events, nationality adjectives such as "Iranian"), so as test labels they
# would mostly be facts no sentence states as a location relation.
TEST_MAP = {pid: m for pid, m in REDOCRED_MAP.items() if pid not in ("P17", "P276")}
PART_A_SIZE = 60
SEED = 2026

PART_B = [
    ("Amazon bought Whole Foods in 2017.", [("Amazon", "acquired", "Whole Foods")]),
    ("Instagram was purchased by Facebook in 2012.", [("Facebook", "acquired", "Instagram")]),
    ("Disney completed its takeover of Pixar in 2006.", [("Disney", "acquired", "Pixar")]),
    ("In 2016, Microsoft paid billions to buy LinkedIn.", [("Microsoft", "acquired", "LinkedIn")]),
    ("Wilhelm Roentgen discovered X-rays in 1895.", [("Wilhelm Roentgen", "discovered", "X-rays")]),
    ("Neptune was first observed by Johann Galle in 1846.", [("Johann Galle", "discovered", "Neptune")]),
    ("Insulin was isolated by Frederick Banting and Charles Best.",
     [("Frederick Banting", "discovered", "insulin"), ("Charles Best", "discovered", "insulin")]),
    ("Aspirin relieves headaches and lowers fever.",
     [("aspirin", "treats", "headaches"), ("aspirin", "treats", "fever")]),
    ("Doctors prescribe metformin for type 2 diabetes.", [("metformin", "treats", "type 2 diabetes")]),
    ("Ibuprofen is commonly used to reduce inflammation.", [("ibuprofen", "treats", "inflammation")]),
    ("Excessive sugar intake can lead to tooth decay.", [("excessive sugar intake", "causes", "tooth decay")]),
    ("Air pollution triggers asthma attacks in children.", [("air pollution", "causes", "asthma attacks")]),
    ("Mosquito bites spread malaria.", [("mosquito bites", "causes", "malaria")]),
    ("Pepsi has long been a rival of Coca-Cola.", [("Pepsi", "competes_with", "Coca-Cola")]),
    ("Airbus competes fiercely with Boeing for aircraft orders.", [("Airbus", "competes_with", "Boeing")]),
    ("Samsung and Apple battle for the smartphone market.", [("Samsung", "competes_with", "Apple")]),
    ("Spotify partnered with Uber to let riders choose music.", [("Spotify", "collaborates_with", "Uber")]),
    ("NASA worked closely with SpaceX on the Crew Dragon program.", [("NASA", "collaborates_with", "SpaceX")]),
    ("Exxon merged with Mobil in 1999 to form ExxonMobil.", [("Exxon", "merged_with", "Mobil")]),
    ("In 2015, Kraft and Heinz merged into a single company.", [("Kraft", "merged_with", "Heinz")]),
    ("Rahul Mehta reports to Priya Sharma, the head of operations.", [("Rahul Mehta", "reports_to", "Priya Sharma")]),
    ("Lisa Chen reports to Jensen Huang.", [("Lisa Chen", "reports_to", "Jensen Huang")]),
    ("Tim Cook runs Apple from its headquarters in Cupertino.",
     [("Tim Cook", "leads", "Apple"), ("Apple", "located_in", "Cupertino")]),
    ("Satya Nadella took over as chief executive of Microsoft in 2014.", [("Satya Nadella", "leads", "Microsoft")]),
    ("Mary Barra heads General Motors, which is based in Detroit.",
     [("Mary Barra", "leads", "General Motors"), ("General Motors", "located_in", "Detroit")]),
    ("Reed Hastings and Marc Randolph co-founded Netflix in 1997.",
     [("Reed Hastings", "founded", "Netflix"), ("Marc Randolph", "founded", "Netflix")]),
    ("Wikipedia was launched by Jimmy Wales and Larry Sanger.",
     [("Jimmy Wales", "founded", "Wikipedia"), ("Larry Sanger", "founded", "Wikipedia")]),
    ("Tesla, founded in 2003, is headquartered in Austin.", [("Tesla", "located_in", "Austin")]),
    ("Ada Lovelace was born in London and studied mathematics privately.", [("Ada Lovelace", "born_in", "London")]),
    ("Srinivasa Ramanujan, who was born in Erode, later studied at Trinity College.",
     [("Srinivasa Ramanujan", "born_in", "Erode"), ("Srinivasa Ramanujan", "studied_at", "Trinity College")]),
    ("Barack Obama graduated from Columbia University and Harvard Law School.",
     [("Barack Obama", "studied_at", "Columbia University"), ("Barack Obama", "studied_at", "Harvard Law School")]),
    ("Michelle Obama is married to Barack Obama.", [("Michelle Obama", "married_to", "Barack Obama")]),
    ("George Orwell wrote Animal Farm.", [("George Orwell", "authored", "Animal Farm")]),
    ("The novel Pride and Prejudice was written by Jane Austen.", [("Jane Austen", "authored", "Pride and Prejudice")]),
    ("Linus Torvalds created Linux while he was a student in Helsinki.", [("Linus Torvalds", "created", "Linux")]),
    ("The iPhone was designed by Apple.", [("Apple", "created", "iPhone")]),
    ("WhatsApp is owned by Meta.", [("WhatsApp", "subsidiary_of", "Meta")]),
    ("Jaguar Land Rover is a part of Tata Motors.", [("Jaguar Land Rover", "subsidiary_of", "Tata Motors")]),
    ("Grace Hopper worked for the United States Navy.", [("Grace Hopper", "works_at", "United States Navy")]),
    ("Katherine Johnson was employed by NASA as a mathematician.", [("Katherine Johnson", "works_at", "NASA")]),
    ("Tokyo is the capital of Japan.", [("Tokyo", "located_in", "Japan")]),
    ("Bengaluru lies in the southern state of Karnataka.", [("Bengaluru", "located_in", "Karnataka")]),
    # distractors: no relation from the lab's list is stated
    ("Elon Musk and Jeff Bezos attended the same conference in Paris.", []),
    ("Lionel Messi met Cristiano Ronaldo before the match in Madrid.", []),
    ("Google and Microsoft both released new laptops last year.", []),
    ("Shakespeare and Marlowe lived in London at the same time.", []),
]


def build_part_a():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("tonytan48/Re-DocRED", "test_revised.json", repo_type="dataset")
    docs = json.load(open(path, encoding="utf-8"))
    candidates = []
    for doc_idx, doc in enumerate(docs):
        vertex, sents = doc["vertexSet"], doc["sents"]
        by_sentence = {}
        for lab in doc["labels"]:
            if lab["r"] not in TEST_MAP:
                continue
            predicate, subject_side = TEST_MAP[lab["r"]]
            shared = {m["sent_id"] for m in vertex[lab["h"]]} & {m["sent_id"] for m in vertex[lab["t"]]}
            for sid in shared:
                mh = next(m for m in vertex[lab["h"]] if m["sent_id"] == sid)
                mt = next(m for m in vertex[lab["t"]] if m["sent_id"] == sid)
                h_text = _detok(sents[sid][mh["pos"][0]:mh["pos"][1]])
                t_text = _detok(sents[sid][mt["pos"][0]:mt["pos"][1]])
                subj, obj = (h_text, t_text) if subject_side == "h" else (t_text, h_text)
                by_sentence.setdefault(sid, set()).add((subj, predicate, obj))
        for sid, gold in by_sentence.items():
            candidates.append((doc_idx, sid, _detok(sents[sid]), sorted(gold)))
    random.Random(SEED).shuffle(candidates)
    return [{"id": f"A{i + 1:02d}", "part": "A", "source": f"Re-DocRED test doc {d} sentence {s}",
             "text": text, "gold": [list(g) for g in gold]}
            for i, (d, s, text, gold) in enumerate(candidates[:PART_A_SIZE])]


def build_part_b():
    return [{"id": f"B{i + 1:02d}", "part": "B", "source": "hand-written",
             "text": text, "gold": [list(g) for g in gold]}
            for i, (text, gold) in enumerate(PART_B)]


def main():
    part_a, part_b = build_part_a(), build_part_b()
    judged = sorted({pred for pred, _ in TEST_MAP.values()})
    OUT.write_text(json.dumps({
        "description": "Blind test set for Experiment 7. Report aggregate scores only; never tune on it.",
        "part_a_judged_predicates": judged,
        "documents": part_a + part_b,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}: part A {len(part_a)} sentences ({sum(len(d['gold']) for d in part_a)} gold triples), "
          f"part B {len(part_b)} sentences ({sum(len(d['gold']) for d in part_b)} gold triples)")


if __name__ == "__main__":
    main()
