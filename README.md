# KGIRS Virtual Lab: Experiment 7 - Relationship Extraction from Text

An interactive virtual lab for Experiment 7 of the Knowledge Graphs and Information Retrieval Systems (KGIRS) course. Students learn how relationships between entities are pulled out of text as Subject-Predicate-Object (SPO) triples, compare six extraction methods, including one based on BERT, and see how the results are scored against gold reference sets.

This branch extends the five-method lab on `main` with the BERT method, a blind test set, and the scripts that build and evaluate them.

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional: Dependency Parse method
streamlit run app.py
```

The BERT method is optional too. Its model (`all-MiniLM-L6-v2`, about 90 MB) downloads from Hugging Face on first use. Without `sentence-transformers` the method is hidden and everything else still works.

If you use PyTorch 2.5 or older, keep `transformers` below version 5; the requirements file already pins this.

## What the lab covers

**Theory.** Relationship extraction background, the six methods with pseudocode and a live trace on a worked example, a BERT section with step-by-step illustrations (reading rules, embeddings, the meaning space, unseen wordings, training data), entity recognition, how the methods are evaluated, and a glossary with hover definitions.

**Simulation.** Pick a sample corpus, type your own text, or upload a document (`.txt`, `.md`, `.csv`, `.tsv`, `.pdf`, `.docx`). Choose a method and a confidence cutoff, then inspect the results in six tabs: triples, annotated text, entities, an interactive knowledge graph, an entity-type flow diagram, and evaluation against the gold set. Results export as CSV, JSON, or GraphML.

**Quiz** and **Report Generation** (a PDF laboratory report).

## Extraction methods

| Method | How it decides the relation | Needs |
|---|---|---|
| Co-occurrence Only | Every adjacent entity pair gets a generic `related_to` edge | - |
| Pattern-Based (Lexicon Match) | A fixed list of relation cue phrases between two entities | - |
| Hybrid (Pattern + Co-occurrence) | Patterns first; co-occurrence fills the gaps | - |
| Nearest-Verb (Generalized) | The first verb between two entities, reduced to its base form | - |
| Dependency Parse (SVO) | Grammatical roles of subject, verb and object | spaCy and `en_core_web_sm` |
| BERT Similarity (Sentence-BERT) | A small classifier on BERT embeddings of the masked clause, with reading rules for coordinated and aside subjects | `sentence-transformers`; `bert_re_classifier.npz` ships with the repo |

## BERT method

1. Reading rules pick the entity pairs: years are skipped, coordinated names are all subjects, and a new verb after "and" keeps the earlier subject.
2. The names are masked (`X acquired Y`) and the clause is embedded with Sentence-BERT.
3. A logistic-regression classifier reads the embedding plus the two entity types and returns a probability for each of the 18 relations, the reversed form (passive voice), or NONE.

The classifier was trained on hand-written clauses plus Re-DocRED sentences (MIT licence). Re-DocRED covers only some of the lab's relations, so the others rely on the hand-written examples alone.

## Evaluation

**Sample corpora.** Each has a gold reference set; precision, recall, F1 and Pair Recall are shown in Simulation and in the Theory comparison. These scores are optimistic, because the rules and the classifier were refined on these texts.

**Blind test (`benchmarks/`).** 106 sentences that played no part in building any method:
- Part A: 60 real Wikipedia sentences from the Re-DocRED test split (judged on the 10 relations it annotates);
- Part B: 46 hand-written sentences covering the other relations, passive voice, lowercase concepts and distractors.

Latest results (F1, 2026-10-26):

| Method | Part A | Part B | Both |
|---|---|---|---|
| Co-occurrence Only | 0.00 | 0.00 | 0.00 |
| Pattern-Based | 0.04 | 0.43 | 0.18 |
| Hybrid | 0.04 | 0.29 | 0.15 |
| Nearest-Verb | 0.03 | 0.33 | 0.14 |
| Dependency Parse | 0.01 | 0.47 | 0.20 |
| BERT Similarity | 0.20 | 0.77 | 0.38 |

Real Wikipedia sentences are much harder than the sample corpora: they are long and indirect, so related entities often sit far apart. Every method scores lower there, and BERT is the most accurate of the six.

## Reproducing the blind test and the classifier

```bash
python benchmarks/build_blind_test.py          # rebuild blind_test.json (needs huggingface_hub)
python benchmarks/evaluate_blind.py --save     # score all methods, write blind_results.json
python train_bert_re.py --stats-only           # rebuild the training-data statistics only
python train_bert_re.py                        # retrain the classifier (needs scikit-learn)
```

Retraining rewrites `bert_re_classifier.npz`. Keep the blind test for final reporting only: tune on the sample corpora or the training data, not on the blind test's individual errors.

## Project layout

```
app.py                      entry point: `streamlit run app.py`
exp7/                       the lab, one module per concern
  bert_method.py            BERT extraction: embeddings, classifier, reading rules
  bert_figures.py           BERT illustrations for the Theory section
  entities.py               entity recognition (rules, spaCy, concepts)
  extraction.py             all methods, metrics and evaluation
  lexicon.py, svo.py        pattern lexicon, nearest verb, dependency parse
  corpora.py                sample corpora and gold reference sets
  graph_layout.py           graph layout (networkx)
  graph_figure.py           the interactive knowledge-graph figure
  ...                       theme, glossary, popups, content, charts, uploads,
                            report_pdf, quiz_bank, sections/, app.py
benchmarks/
  build_blind_test.py       builds blind_test.json (Re-DocRED test + hand-written)
  evaluate_blind.py         scores every method on the blind test
  blind_test.json           the blind test set
  blind_results.json        the latest scored results
train_bert_re.py            trains the BERT classifier and writes the training statistics
bert_re_classifier.npz      trained classifier (numpy only)
bert_re_training_stats.json training-data counts per relation
requirements.txt            dependencies (required and optional)
```

## Notes

- The Dependency Parse and BERT methods are hidden when their optional packages are missing.
- Uploaded documents are limited to 25,000 characters.
- Re-DocRED: Tony Tan et al., MIT licence, <https://huggingface.co/datasets/tonytan48/Re-DocRED>.
