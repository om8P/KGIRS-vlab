"""Experiment configuration and the Theory section's written content."""

from __future__ import annotations



EXPERIMENT_CONFIG = {
    "title": "Experiment 7: Relationship Extraction from Text",
    "course": "Knowledge Graphs and Information Retrieval Systems (KGIRS) · Course Code: D17A/B/C",
    "department": "Department of Computer Engineering / Information Technology",
    "portal": "Virtual Laboratory Learning Portal",
    "objectives": [
        "Understand how semantic relationships between entities are identified in unstructured text.",
        "Implement rule-based and pattern-based relation extraction over sentence boundaries.",
        "Represent extracted knowledge as Subject-Predicate-Object (SPO) triples with confidence scoring.",
        "Evaluate extraction precision, recall, and F1 score against human-curated gold standard reference sets.",
        "Prepare extracted knowledge graph triples for downstream graph database ingestion (Experiments 8-9)."
    ]
}


THEORY_CONTENT = {
    "background": """
### Overview & Principles
[[relationship_extraction|Relationship Extraction (RE)]] is the central task in [[information_retrieval|Information Retrieval]] and natural language processing that identifies and classifies semantic relationships holding between pairs of [[entity|entities]] mentioned in unstructured prose.

Given an input sentence such as *"Larry Page co-founded Google in California"*, an RE system automatically recognizes the entities and extracts the structured [[triple|Subject-Predicate-Object (SPO) triple]]:

`(Larry Page) --[founded]--> (Google)`

These triples represent the atomic building blocks of a [[knowledge_graph|Knowledge Graph]]:
- **Nodes** correspond to real-world entities (people, organizations, locations).
- **Directed Edges** correspond to relational predicates (`founded`, `acquired`, `located_in`).

### Extraction Approaches Compared in this Laboratory
1. **[[pattern_based|Pattern-Based Extraction]]**:
   - Matches hand-crafted lexico-syntactic patterns and verbal cues (e.g. *acquired*, *is the CEO of*, *located in*) occurring between entities.
   - **Characteristics**: High [[precision]], transparent and interpretable, but subject to linguistic [[brittleness]] when sentences contain complex clauses or unseen paraphrases.

2. **[[cooccurrence|Co-occurrence Extraction]]**:
   - Assumes that any two entities appearing within the same sentence boundary are related, assigning a generic `related_to` predicate.
   - **Characteristics**: High [[recall]], but low [[precision]] and generates unnamed, weakly-specified edges.

3. **[[hybrid|Hybrid Extraction]]**:
   - Executes pattern matching first. If an explicit [[relation_cue|relation cue]] is matched, it assigns the specific predicate with high [[confidence]]; otherwise, it defaults to a low-confidence co-occurrence link.
   - **Characteristics**: Maximizes coverage while equipping downstream graph ingestion stages with a quantitative confidence filter.

4. **[[nearest_verb|Nearest-Verb Extraction]]**:
   - Drops the fixed word list. For each adjacent entity pair it takes the first verb-like word between them, reduces it to its [[lemma]] (*acquired* becomes *acquire*), attaches a following preposition (*based in* becomes `base_in`), and maps common verbs onto the lab's standard predicate names.
   - **Characteristics**: Handles verbs the lexicon never listed and needs no extra software, but reads strictly left to right, so a [[passive_voice|passive sentence]] ("YouTube was acquired by Google") comes out with the direction reversed.

5. **[[svo_extraction|Dependency-Parse (SVO) Extraction]]**:
   - Uses [[dependency_parsing|dependency parsing]] to label every word with its grammatical role (`nsubj` subject, `dobj` object, `nsubjpass` passive subject, `agent` the "by" phrase, `pobj` object of a preposition). A triple is emitted when two entities attach to the same verb.
   - **Characteristics**: Handles passive voice, one subject with several verbs, and entities far apart, with the highest accuracy of the five methods. The cost is a dependency on a trained language parser (spaCy), and its mistakes become the extractor's mistakes.

### Systematic Pipeline Workflow
1. **[[named_entity_recognition|Entity Recognition]]**: Identify mentions of people, organizations, places, dates, and domain terms.
2. **[[candidate_pair|Candidate Pair Generation]]**: Identify all pairs of entities occurring within the [[word_distance|maximum word distance]] window in each sentence.
3. **Predicate Assignment**: Decide the relation name for each pair - by matching the intervening text against the [[relation_cue|relation lexicon]], by taking the nearest verb, or by reading the sentence grammar.
4. **[[triple|Triple Construction]]**: Emit structured facts with attached [[provenance]] (source sentence) and [[confidence|confidence score]].
5. **[[knowledge_graph|Graph Construction & Verification]]**: Preview the resulting graph, evaluate accuracy against [[reference_set|ground truth]], and export it as CSV, JSON or a portable [[graphml|GraphML]] graph file.
""",
    "procedure": [
        "Step 1: Study the theoretical principles of relation extraction and examine key IR terms using the interactive notes.",
        "Step 2: Navigate to the Simulation section and choose a sample corpus (or upload/paste custom text).",
        "Step 3: Choose an extraction strategy (Co-occurrence, Pattern-Based, Hybrid, Nearest-Verb, or Dependency Parse) and adjust the confidence threshold.",
        "Step 4: Execute extraction and explore the 6 analysis tabs: Triples, Annotated Text, Entities, Graph, Flow, and Evaluation.",
        "Step 5: Inspect the interactive Knowledge Graph; toggle predicate labels and test legibility in Light and Dark modes.",
        "Step 6: Click 'Record Current Trial' to log the experiment parameters and results into your session logbook.",
        "Step 7: Complete at least 3 trials comparing different extraction methods and confidence cutoffs.",
        "Step 8: Export the extracted triples as CSV, JSON, or a GraphML graph file.",
        "Step 9: Complete the self-grading Conceptual Assessment Quiz to test your understanding.",
        "Step 10: Open Report Generation, enter your student credentials, and download your official PDF Laboratory Report."
    ]
}
