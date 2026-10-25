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
   - **Characteristics**: Handles passive voice, one subject with several verbs, and entities far apart, making it the most accurate of the rule-based methods. The cost is a dependency on a trained language parser (spaCy), and its mistakes become the extractor's mistakes.

6. **[[bert|BERT]] Extraction (a trained classifier on BERT embeddings)**:
   - [[bert|BERT]] is a [[transformer]] language model pretrained on huge amounts of text, so it understands words in context. Its [[sentence_bert|Sentence-BERT]] form turns a sentence into an [[embedding]] (384 numbers) that captures its meaning; [[cosine_similarity|cosine similarity]] measures how close two meanings are.
   - **Reading rules** first choose which entities to pair: years are skipped; coordinated names (*Larry Page and Sergey Brin founded Google*) are all subjects; after "and", a new verb keeps the earlier subject (*Microsoft acquired LinkedIn and partnered with OpenAI*); and after a comma aside the subject is the entity before it (*Microsoft, based in Redmond, competes with Google*).
   - For each pair the names are hidden ([[entity_masking|entity masking]]: *X acquired Y*), the clause is embedded, and a small [[classifier]] reads the embedding plus the two entity types. It answers with one of the 18 relations, the same relation reversed (passive voice: *X was acquired by Y* means Y is the subject), or *none*; its probability becomes the confidence.
   - The classifier learned from [[training_data|training data]]: about 2,000 wordings written for this lab and 6,000 Wikipedia clauses from the public Re-DocRED dataset (MIT licence). Re-DocRED has no examples for 8 of the 18 relations, so those rely on the hand-written wordings alone.
   - **Characteristics**: It compares *meaning* instead of words, so it recognises wordings no rule lists and handles the passive voice without grammar rules, and it is the most accurate method on the [[blind_test_set|blind test]]. The costs: it is a black box that cannot explain *why*, it needs a ~90 MB model download, and it is only as good as its training examples; a wording unlike anything it has seen is skipped instead of guessed.

### Systematic Pipeline Workflow
1. **[[named_entity_recognition|Entity Recognition]]**: Identify mentions of people, organizations, places, dates, and domain terms. When spaCy is installed, its trained statistical model finds the names, the lab's capitalisation rules add names it missed, and [[noun_chunk|noun phrases]] that act as subjects or objects (*radium*, *bacterial infections*) become TERM entities; without spaCy, only the capitalisation rules are used, so lowercase concepts are missed.
2. **[[candidate_pair|Candidate Pair Generation]]**: Identify all pairs of entities occurring within the [[word_distance|maximum word distance]] window in each sentence.
3. **Predicate Assignment**: Decide the relation name for each pair - by matching the intervening text against the [[relation_cue|relation lexicon]], by taking the nearest verb, by reading the sentence grammar, or by comparing meaning with BERT.
4. **[[triple|Triple Construction]]**: Emit structured facts with attached [[provenance]] (source sentence) and [[confidence|confidence score]].
5. **[[knowledge_graph|Graph Construction & Verification]]**: Preview the resulting graph, evaluate accuracy against [[reference_set|ground truth]], and export it as CSV, JSON or a portable [[graphml|GraphML]] graph file.

### Entity Recognition: the Foundation of Every Method
Every method above can only link entities that the [[named_entity_recognition|entity recogniser]] found, so its mistakes flow into every later step ([[error_propagation|error propagation]]): a missed entity can never appear in a triple, and a mistyped one (a city tagged PERSON) misleads the methods that use entity types.
- **Rule-based recognition** (always available): runs of capitalised words, typed by built-in lists and name shapes; anything unknown becomes PERSON. It cannot see lowercase concepts such as *radium* or *bacterial infections*.
- **Statistical recognition with spaCy** (when installed): a model trained on annotated text finds and types names, the capitalisation rules add names it missed, and [[noun_chunk|noun phrases]] that act as subjects or objects become **TERM** entities. On the sample corpora this change alone raised every method's score.
- **Still imperfect**: a capitalised word at the start of a sentence can be mistaken for a name, and spaCy sometimes mistypes names (a person tagged as an organisation).

### How the Methods Are Evaluated (and Why Honest Testing Matters)
- Each sample corpus comes with a [[reference_set|gold reference set]]; [[precision]], [[recall]] and [[f1|F1]] compare the extracted triples with it, and Pair Recall gives partial credit for the right entities with the wrong relation name.
- A method can look better than it really is when it is tested on the text used to build it. The rules in this lab were refined by studying the four sample corpora's mistakes, and the BERT classifier learned from its [[training_data|training data]], so scores on those texts are optimistic: this is [[overfitting]].
- Honest numbers need [[held_out|held-out]] data that played no part in building the method. This lab uses two kinds:
  1. **Unseen wordings**: 64 relation phrasings kept out of BERT's training (*is at the helm of*, *snapped up*, ...). The classifier gets about 7 in 10 right, against 3 in 10 for matching example clauses and almost none for a fixed word list.
  2. A **[[blind_test_set|blind test set]]** of 106 sentences: 60 real Wikipedia sentences from the Re-DocRED test split (Part A) and 46 hand-written sentences (Part B), built before the last round of changes and only ever scored as a whole.
- On the blind test every method scores far lower than on the sample corpora: real Wikipedia sentences are long and indirect, so related entities often sit far apart or the relation is only implied. The chart under *How Well Do They Really Work?* below shows the numbers.
""",
    "procedure": [
        "Step 1: Study the theoretical principles of relation extraction and examine key IR terms using the interactive notes.",
        "Step 2: Navigate to the Simulation section and choose a sample corpus (or upload/paste custom text).",
        "Step 3: Choose an extraction strategy (Co-occurrence, Pattern-Based, Hybrid, Nearest-Verb, Dependency Parse, or BERT Similarity) and adjust the confidence threshold.",
        "Step 4: Execute extraction and explore the 6 analysis tabs: Triples, Annotated Text, Entities, Graph, Flow, and Evaluation.",
        "Step 5: Inspect the interactive Knowledge Graph; toggle predicate labels and test legibility in Light and Dark modes.",
        "Step 6: Click 'Record Current Trial' to log the experiment parameters and results into your session logbook.",
        "Step 7: Complete at least 3 trials comparing different extraction methods and confidence cutoffs.",
        "Step 8: Export the extracted triples as CSV, JSON, or a GraphML graph file.",
        "Step 9: Complete the self-grading Conceptual Assessment Quiz to test your understanding.",
        "Step 10: Open Report Generation, enter your student credentials, and download your official PDF Laboratory Report."
    ]
}
