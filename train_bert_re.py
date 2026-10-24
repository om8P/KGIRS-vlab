"""
Trains the small relation classifier used by the lab's "BERT Similarity (Sentence-BERT)" method.

What it does
------------
Every training example is a *clause* (the text between two entities, names masked as X / Y)
plus the entity types of the two entities. BERT (Sentence-BERT, all-MiniLM-L6-v2) turns the
clause into a 384-number embedding; a multinomial logistic regression is then trained on
[embedding + entity types] to predict one of 37 classes:

    NONE            - the clause does not express any of the lab's relations
    <predicate>     - e.g. "acquired": the first entity (X) is the subject
    <predicate>~rev - e.g. "acquired~rev": the second entity (Y) is the subject (passive voice)

Training data (the four benchmark corpora in the lab are NEVER used for training, so their
scores stay an honest test):
  1. Synthetic clauses written in this file: many wordings per relation, active and passive.
  2. Re-DocRED (public, MIT licence, https://huggingface.co/datasets/tonytan48/Re-DocRED):
     Wikipedia sentences with human-annotated relations, mapped to the lab's predicates, plus
     entity pairs with no annotated relation as "NONE" examples.

Output: bert_re_classifier.npz (a few tens of KB) next to the app. The app needs only numpy to
use it. This script additionally needs: scikit-learn, sentence-transformers, huggingface_hub.

Usage:   python train_bert_re.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "bert_re_classifier.npz"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TYPE_ORDER = ["PERSON", "ORG", "LOCATION", "DATE", "TERM", "MISC"]
MAX_BETWEEN_WORDS = 12
RNG = random.Random(7)

# ----------------------------------------------------------------------------------------------
# 1. SYNTHETIC CLAUSES
# ----------------------------------------------------------------------------------------------
# predicate -> (subject types, object types, [phrases with X as subject], [phrases with Y as subject])
P, O, L, M = "PERSON", "ORG", "LOCATION", "MISC"
SYNTHETIC = {
    "founded": ([P, O], [O],
        ["founded", "co-founded", "established", "started", "launched", "set up", "has founded",
         "later founded", "went on to found", "is the founder of", "was the founder of",
         "is a co-founder of", "is the co-founder of", "started up", "set up the company"],
        ["was founded by", "was established by", "was started by", "was set up by",
         "was launched by", "was co-founded by", "has been founded by"]),
    "acquired": ([O], [O],
        ["acquired", "bought", "purchased", "took over", "absorbed", "has acquired", "later acquired",
         "agreed to acquire", "completed the acquisition of", "announced the acquisition of",
         "has bought", "purchased a majority stake in", "took control of", "paid billions for"],
        ["was acquired by", "was bought by", "was purchased by", "was taken over by",
         "was absorbed by", "has been acquired by"]),
    "leads": ([P], [O, L],
        ["is the CEO of", "was the CEO of", "is the chief executive of", "is the chairman of",
         "was the chairman of", "is the president of", "is the director of", "leads", "led", "heads",
         "runs", "is the head of", "serves as CEO of", "is the leader of", "was appointed CEO of",
         "is chairperson of", "has been the chairman of", "became the president of"],
        ["is led by", "is headed by", "is run by", "was led by"]),
    "works_at": ([P], [O],
        ["works at", "works for", "worked at", "is employed by", "was employed by", "is an employee of",
         "joined", "has joined", "is a researcher at", "is a professor at", "serves at",
         "works as an engineer at", "is a scientist at", "is a lecturer at"],
        ["employs", "has hired", "hired"]),
    "located_in": ([O, L, P], [L],
        ["is located in", "is based in", "is headquartered in", "is situated in", "operates in",
         "is found in", "lies in", "is in", "in", "is a company in", "has its headquarters in",
         "is headquartered at", "is located at", "is based out of", "has its head office in",
         "is a city in", "is a town in", "is a village in", "is a district of"],
        ["is home to", "hosts", "houses", "contains"]),
    "born_in": ([P], [L],
        ["was born in", "was born in the city of", "is a native of", "was born at", "hails from",
         "was born near", "is originally from", "was born on 12 May 1950 in", "was born on 3 March 1931 in",
         "was born in 1962 in", "was born in the town of"],
        []),
    "married_to": ([P], [P],
        ["is married to", "was married to", "married", "is the husband of", "is the wife of", "wed",
         "is the spouse of", "tied the knot with", "got married to", "has been married to", "married his wife",
         "married her husband"],
        []),
    "created": ([P, O], [M, O],
        ["developed", "created", "invented", "designed", "built", "engineered", "has developed",
         "later developed", "is the creator of", "is the developer of", "programmed", "devised",
         "came up with", "crafted", "developed the", "created the", "designed the", "co-created"],
        ["was developed by", "was created by", "was invented by", "was designed by", "was built by",
         "was engineered by"]),
    "discovered": ([P], [M],
        ["discovered", "isolated", "identified", "has discovered", "first discovered", "uncovered",
         "detected", "first isolated", "discovered the", "was the first to discover",
         "discovered the element"],
        ["was discovered by", "was isolated by", "was identified by", "was first described by"]),
    "authored": ([P], [M],
        ["wrote", "authored", "published", "has written", "penned", "published his", "published her",
         "co-wrote", "is the author of", "composed", "wrote the book", "published the paper on"],
        ["was written by", "was authored by", "was published by", "was penned by", "was composed by"]),
    "subsidiary_of": ([O], [O],
        ["is a subsidiary of", "is part of", "is a division of", "is owned by", "is a unit of",
         "is a branch of", "belongs to", "is a wholly owned subsidiary of", "is a department of",
         "operates as part of", "is controlled by", "is a subsidiary company of"],
        ["owns", "controls", "is the parent company of", "is the parent of", "wholly owns"]),
    "collaborates_with": ([O, P], [O, P],
        ["collaborated with", "partnered with", "teamed up with", "cooperated with", "has partnered with",
         "is working with", "worked together with", "joined forces with", "allied with", "is allied with",
         "entered a partnership with", "formed an alliance with", "signed a partnership with",
         "collaborates with"],
        []),
    "merged_with": ([O], [O],
        ["merged with", "combined with", "has merged with", "merged into", "fused with",
         "consolidated with", "amalgamated with", "was merged with", "is merging with"],
        []),
    "studied_at": ([P], [O],
        ["studied at", "graduated from", "attended", "was educated at", "earned a degree at", "enrolled at",
         "got a degree from", "received a degree from", "was a student at", "earned a PhD from",
         "completed a PhD at", "studied at the", "graduated from the"],
        []),
    "reports_to": ([P], [P, O],
        ["reports to", "reports directly to", "answers to", "is accountable to", "works under",
         "is supervised by", "is a subordinate of", "reports into", "reports to the", "reports to the board of"],
        []),
    "competes_with": ([O], [O],
        ["competes with", "is a rival of", "is competing against", "rivals", "is a competitor of",
         "competes against", "is in competition with", "is a major competitor to",
         "goes head to head with", "is a competitor to", "is the main rival of"],
        []),
    "treats": ([M, O], [M],
        ["treats", "cures", "alleviates", "is used to treat", "is prescribed for", "relieves",
         "is effective against", "helps treat", "is a treatment for", "can treat",
         "reduces the symptoms of", "treats effectively"],
        []),
    "causes": ([M], [M],
        ["causes", "leads to", "triggers", "induces", "results in", "is responsible for", "gives rise to",
         "brings about", "can cause", "contributes to", "may lead to", "produces", "often causes"],
        []),
}

# Wordings NOT in the training set above, used only to test how well a method generalises.
# (predicate, phrase, subject_is_second)
HELD_OUT_PARAPHRASES = [
    ("acquired", "snapped up", False), ("acquired", "swallowed up", False), ("acquired", "bought out", False),
    ("acquired", "scooped up", False), ("acquired", "was snapped up by", True),
    ("founded", "kicked off", False), ("founded", "is the entrepreneur behind", False),
    ("founded", "was brought into being by", True),
    ("leads", "is at the helm of", False), ("leads", "oversees", False), ("leads", "presides over", False),
    ("leads", "is in charge of", False), ("leads", "steers", False),
    ("works_at", "is on the staff of", False), ("works_at", "has a job at", False),
    ("works_at", "is part of the team at", False),
    ("located_in", "has its main office in", False), ("located_in", "resides in", False),
    ("located_in", "is seated in", False), ("located_in", "lies within", False),
    ("born_in", "first saw the light of day in", False), ("born_in", "came into the world in", False),
    ("born_in", "was born and raised in", False),
    ("married_to", "is wedded to", False), ("married_to", "exchanged vows with", False),
    ("married_to", "is wed to", False),
    ("created", "dreamed up", False), ("created", "put together", False), ("created", "is the inventor of", False),
    ("created", "is the brain behind", False), ("created", "coined", False),
    ("discovered", "unearthed", False), ("discovered", "stumbled upon", False), ("discovered", "spotted", False),
    ("authored", "is the writer of", False), ("authored", "released the novel", False),
    ("authored", "scribbled", False),
    ("subsidiary_of", "is under the umbrella of", False), ("subsidiary_of", "is a daughter company of", False),
    ("subsidiary_of", "sits within", False),
    ("collaborates_with", "joined hands with", False), ("collaborates_with", "teamed with", False),
    ("collaborates_with", "works hand in hand with", False),
    ("merged_with", "was folded into", False), ("merged_with", "blended with", False),
    ("merged_with", "joined together with", False),
    ("studied_at", "is an alumnus of", False), ("studied_at", "did a degree at", False),
    ("studied_at", "trained at", False),
    ("reports_to", "takes orders from", False), ("reports_to", "is answerable to", False),
    ("reports_to", "is directly under", False),
    ("competes_with", "is up against", False), ("competes_with", "vies with", False),
    ("competes_with", "is a challenger to", False), ("competes_with", "battles", False),
    ("treats", "remedies", False), ("treats", "heals", False), ("treats", "combats", False),
    ("treats", "is a remedy for", False),
    ("causes", "sparks", False), ("causes", "provokes", False), ("causes", "engenders", False),
    ("causes", "is a driver of", False),
]

# Pairs of words that are NOT one of the lab's relations. Includes "in"/"of" with a non-place object.
NONE_PHRASES = [
    "and", ",", ", and", "or", "with", "versus", "near", "after", "before", "said", "told", "visited",
    "praised", "criticised", "met", "called", "mentioned", "as well as", "along with", "of", "for",
    "to", "from the", "meanwhile", "said that the", "wrote to", "thanked", "congratulated", "asked",
    "replied to", "spoke to", "interviewed", "photographed", "followed", "joined by", "next to",
    ", while", ", but", ", then", "was seen with", "talked about", "complained about", "applauded",
    "is similar to", "is unlike", "was mentioned by", "was praised by", "was criticised by",
    "was visited by", "was followed by",
]
ADVERBS = ["also", "later", "eventually", "reportedly", "quickly", "officially", "recently", "soon"]


def _synthetic_examples():
    """(clause, subject_is_second, predicate, first_type, second_type) tuples."""
    ex = []
    for predicate, (s_types, o_types, active, passive) in SYNTHETIC.items():
        for phrases, subj_second in ((active, False), (passive, True)):
            for phrase in phrases:
                for _ in range(6):
                    text = phrase
                    if RNG.random() < 0.2 and text.split()[0] in ("is", "was", "has", "can", "may"):
                        words = text.split()
                        text = " ".join([words[0], RNG.choice(ADVERBS)] + words[1:])
                    elif RNG.random() < 0.2:
                        text = f"{RNG.choice(ADVERBS)} {text}"
                    if RNG.random() < 0.15:
                        text = f", {text}"
                    if RNG.random() < 0.25:
                        text = f"{text} the"
                    clause = f"X {text} Y".replace("X ,", "X,")
                    s_t, o_t = RNG.choice(s_types), RNG.choice(o_types)
                    first, second = (o_t, s_t) if subj_second else (s_t, o_t)
                    ex.append((clause, subj_second, predicate, first, second))
    # "X in Y" is a location relation only when Y is a place
    for _ in range(40):
        ex.append(("X in Y", False, "located_in", RNG.choice([O, L]), L))
    for phrase in NONE_PHRASES:
        for _ in range(6):
            first, second = RNG.choice(TYPE_ORDER), RNG.choice(TYPE_ORDER)
            clause = f"X {phrase} Y".replace("X ,", "X,")
            ex.append((clause, False, "NONE", first, second))
    for _ in range(40):
        ex.append(("X in Y", False, "NONE", RNG.choice([P, O, M]), RNG.choice([P, O, M])))
    return ex


# ----------------------------------------------------------------------------------------------
# 2. RE-DOCRED (public data)
# ----------------------------------------------------------------------------------------------
# Wikidata property -> (lab predicate, which side of the annotated pair is the subject: "h" or "t")
REDOCRED_MAP = {
    "P159": ("located_in", "h"), "P131": ("located_in", "h"), "P17": ("located_in", "h"),
    "P276": ("located_in", "h"),
    "P19": ("born_in", "h"), "P26": ("married_to", "h"), "P108": ("works_at", "h"),
    "P69": ("studied_at", "h"),
    "P112": ("founded", "t"), "P488": ("leads", "t"), "P6": ("leads", "t"),
    "P170": ("created", "t"), "P178": ("created", "t"), "P50": ("authored", "t"),
    "P749": ("subsidiary_of", "h"), "P355": ("subsidiary_of", "t"), "P127": ("subsidiary_of", "h"),
}
DOCRED_TYPE = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION", "TIME": "DATE", "NUM": "MISC", "MISC": "MISC"}


def _detok(tokens):
    text = " ".join(tokens)
    for a, b in ((" ,", ","), (" .", "."), (" ;", ";"), (" :", ":"), (" 's", "'s"), ("-LRB- ", "("),
                 (" -RRB-", ")"), (" )", ")"), ("( ", "(")):
        text = text.replace(a, b)
    return " ".join(text.split())


def _redocred_examples(filename, max_pos_per_class, max_none):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("tonytan48/Re-DocRED", filename, repo_type="dataset")
    docs = json.load(open(path, encoding="utf-8"))
    positives, nones = defaultdict(list), []
    for doc in docs:
        vertex, sents = doc["vertexSet"], doc["sents"]
        labelled = {(l["h"], l["t"]) for l in doc["labels"]} | {(l["t"], l["h"]) for l in doc["labels"]}

        def best_pair(h, t):
            best = None
            for mh in vertex[h]:
                for mt in vertex[t]:
                    if mh["sent_id"] != mt["sent_id"]:
                        continue
                    a, b = (mh, mt) if mh["pos"][0] < mt["pos"][0] else (mt, mh)
                    gap = b["pos"][0] - a["pos"][1]
                    if 1 <= gap <= MAX_BETWEEN_WORDS and (best is None or gap < best[0]):
                        best = (gap, a, b, mh is a)
            return best

        for lab in doc["labels"]:
            if lab["r"] not in REDOCRED_MAP:
                continue
            predicate, subject_side = REDOCRED_MAP[lab["r"]]
            found = best_pair(lab["h"], lab["t"])
            if not found:
                continue
            _, first, second, h_is_first = found
            subject_is_first = h_is_first if subject_side == "h" else not h_is_first
            toks = sents[first["sent_id"]][first["pos"][1]:second["pos"][0]]
            clause = "X " + _detok(toks) + " Y"
            clause = clause.replace("X ,", "X,")
            positives[(predicate, not subject_is_first)].append(
                (clause, not subject_is_first, predicate,
                 DOCRED_TYPE.get(first["type"], "MISC"), DOCRED_TYPE.get(second["type"], "MISC")))
        # entity pairs inside one sentence that have no annotated relation at all
        for h in range(len(vertex)):
            for t in range(h + 1, len(vertex)):
                if (h, t) in labelled:
                    continue
                found = best_pair(h, t)
                if found and RNG.random() < 0.25:
                    _, first, second, _ = found
                    toks = sents[first["sent_id"]][first["pos"][1]:second["pos"][0]]
                    clause = ("X " + _detok(toks) + " Y").replace("X ,", "X,")
                    nones.append((clause, False, "NONE", DOCRED_TYPE.get(first["type"], "MISC"),
                                  DOCRED_TYPE.get(second["type"], "MISC")))
    out = []
    for key, items in positives.items():
        RNG.shuffle(items)
        out.extend(items[:max_pos_per_class])
    RNG.shuffle(nones)
    out.extend(nones[:max_none])
    return out


# ----------------------------------------------------------------------------------------------
# 3. TRAIN
# ----------------------------------------------------------------------------------------------
def class_name(predicate, subject_is_second):
    return "NONE" if predicate == "NONE" else predicate + ("~rev" if subject_is_second else "")


def _noisy_type(t):
    """The lab's rule-based NER is rough (any capitalised name it does not recognise - companies,
    organisations AND places such as 'Geneva' - becomes PERSON), so the classifier is also trained
    on corrupted types and cannot lean on them too heavily."""
    if t in ("ORG", "MISC", "TERM", "LOCATION") and RNG.random() < 0.6:
        return "PERSON"
    return t


def featurise(encoder, rows):
    texts = [r[0] for r in rows]
    emb = np.asarray(encoder.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False))
    types = np.zeros((len(rows), 2 * len(TYPE_ORDER)))
    for i, r in enumerate(rows):
        if r[3] is not None:      # None = type unknown (all zeros)
            types[i, TYPE_ORDER.index(r[3])] = 1.0
        if r[4] is not None:
            types[i, len(TYPE_ORDER) + TYPE_ORDER.index(r[4])] = 1.0
    return np.hstack([emb, types])


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    encoder = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

    synthetic = _synthetic_examples()
    public_train = _redocred_examples("train_revised.json", max_pos_per_class=450, max_none=3000)
    public_dev = _redocred_examples("dev_revised.json", max_pos_per_class=200, max_none=600)
    print(f"synthetic: {len(synthetic)}  re-docred train: {len(public_train)}  re-docred dev: {len(public_dev)}")

    # the held-out paraphrase phrases must not appear in the training data
    training_text = " ".join(r[0] for r in synthetic).lower()
    for _, phrase, _ in HELD_OUT_PARAPHRASES:
        assert f" {phrase} " not in training_text, f"held-out phrase leaked into training: {phrase}"

    train_rows = synthetic + public_train
    noisy_rows = [(r[0], r[1], r[2], _noisy_type(r[3]), _noisy_type(r[4])) for r in train_rows]
    untyped_rows = [(r[0], r[1], r[2], None, None) for r in train_rows]
    all_rows = train_rows + noisy_rows + untyped_rows
    X = featurise(encoder, all_rows)
    y = [class_name(r[2], r[1]) for r in all_rows]
    classes = sorted(set(y))
    print("class counts:", dict(sorted(Counter(y).items())))

    clf = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")
    clf.fit(X, y)

    dev_X = featurise(encoder, public_dev)
    dev_y = [class_name(r[2], r[1]) for r in public_dev]
    print(f"Re-DocRED dev accuracy (never trained on): {accuracy_score(dev_y, clf.predict(dev_X)):.3f}")
    pos_idx = [i for i, label in enumerate(dev_y) if label != "NONE"]
    pos_acc = accuracy_score([dev_y[i] for i in pos_idx], clf.predict(dev_X[pos_idx]))
    print(f"  ... on relation clauses only: {pos_acc:.3f}  (NONE-vs-relation errors count as wrong)")

    held = []
    for predicate, phrase, rev in HELD_OUT_PARAPHRASES:
        s_types, o_types = SYNTHETIC[predicate][0], SYNTHETIC[predicate][1]
        s_t, o_t = s_types[0], o_types[0]
        first, second = (o_t, s_t) if rev else (s_t, o_t)
        held.append((f"X {phrase} Y", rev, predicate, first, second))
    held_pred = clf.predict(featurise(encoder, held))
    held_y = [class_name(r[2], r[1]) for r in held]
    print(f"Held-out paraphrase accuracy (wordings never seen): {accuracy_score(held_y, held_pred):.3f}")

    np.savez(
        OUT_PATH,
        classes=np.array(clf.classes_), coef=clf.coef_.astype(np.float32),
        intercept=clf.intercept_.astype(np.float32), type_order=np.array(TYPE_ORDER),
        model_name=np.array(MODEL_NAME),
    )
    print(f"saved {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB), {len(classes)} classes")


if __name__ == "__main__":
    main()
