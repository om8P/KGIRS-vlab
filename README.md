# KGIRS Virtual Lab: Experiment 7 - Relationship Extraction from Text

An interactive virtual lab for Experiment 7 of the Knowledge Graphs and Information Retrieval Systems (KGIRS) course. Students learn how relationships between entities are pulled out of text as Subject-Predicate-Object (SPO) triples, compare rule-based extraction methods, and see how the results are scored against a gold reference set.

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional, for the Dependency Parse method
streamlit run app.py
```

The lab opens in the browser. It works without the optional packages: the rule-based methods and the rule-based entity finder need only the required packages.

## What the lab covers

**Theory.** Relationship extraction background, the five extraction methods with pseudocode and a live trace on a worked example, the entity recognition step, how the methods are evaluated, and a glossary with hover definitions for each technical term.

**Simulation.** Pick a sample corpus, type your own text, or upload a document (`.txt`, `.md`, `.csv`, `.tsv`, `.pdf`, `.docx`). Choose a method and a confidence cutoff, then inspect the results in six tabs:

- extracted triples and their confidence;
- annotated text showing the evidence sentence;
- detected entities;
- an interactive knowledge graph (weak links hidden by default; focus on one entity);
- an entity-type flow diagram;
- evaluation against the gold set, where a corpus has one.

Results can be exported as CSV, JSON, or GraphML.

**Quiz** and **Report Generation** (a PDF laboratory report with the student's trial log and results).

## Extraction methods

| Method | How it decides the relation | Needs |
|---|---|---|
| Co-occurrence Only | Every adjacent entity pair is linked by a generic `related_to` edge | - |
| Pattern-Based (Lexicon Match) | A fixed list of relation cue phrases between two entities | - |
| Hybrid (Pattern + Co-occurrence) | Patterns first; co-occurrence fills the gaps | - |
| Nearest-Verb (Generalized) | The first verb between two entities, reduced to its base form | - |
| Dependency Parse (SVO) | The grammatical roles of subject, verb and object | spaCy and `en_core_web_sm` |

## Entity recognition

When spaCy is installed, its statistical model finds names, the lab's capitalisation rules add names it misses, and noun phrases used as subjects or objects are kept as concepts (for example *radium*). Without spaCy, only the capitalisation rules run, so lowercase concepts are not found.

## Evaluation

Each sample corpus has a gold reference set. Precision, recall and F1 require an exact match on subject, predicate and object. Pair Recall gives partial credit for finding the right entities with a different relation name.

Scores on the sample corpora are optimistic, because the rules were refined on them. For an honest estimate, use the blind test set in `benchmarks/` on the `bert-extraction` branch.

## Project layout

```
app.py                 entry point: `streamlit run app.py`
exp7/                  the lab, one module per concern
  theme.py             light and dark colour palettes
  glossary.py          technical-term definitions and examples
  popups.py            hover pop-ups and the markdown renderer
  content.py           experiment configuration and Theory text
  entities.py          entity recognition (rules, spaCy, concepts)
  lexicon.py           pattern lexicon and nearest-verb helpers
  svo.py               dependency-parse extraction
  extraction.py        all methods, metrics and evaluation
  corpora.py           sample corpora and gold reference sets
  uploads.py           loading .txt/.md/.csv/.tsv/.pdf/.docx files
  graph_layout.py      graph layout (networkx)
  graph_figure.py      the interactive knowledge-graph figure
  method_figures.py    pipeline and method illustrations
  charts.py            entity, flow and comparison charts
  report_pdf.py        PDF laboratory report
  quiz_bank.py         quiz questions
  sections/            Theory, Simulation, Quiz and Report sections
  app.py               session setup and navigation
requirements.txt       dependencies (required and optional)
```

## Notes

- The Dependency Parse method is shown only when spaCy and its English model are installed.
- Uploaded documents are limited to 25,000 characters.
