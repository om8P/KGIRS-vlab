"""The technical-terms glossary (definition + worked example for each term)."""

from __future__ import annotations

from typing import Dict


TERMS: Dict[str, Dict[str, str]] = {
    "information_retrieval": {
        "name": "Information Retrieval (IR)",
        "definition": "The science and engineering discipline of searching for, indexing, extracting, and representing relevant facts or knowledge from large collections of unstructured data.",
        "example": "Turning raw narrative news reports into queryable knowledge graph assertions.",
    },
    "relationship_extraction": {
        "name": "Relationship Extraction (RE)",
        "definition": "The NLP and IR task of detecting semantic relationships holding between two identified entities mentioned in text and mapping them into structured assertions.",
        "example": '"Larry Page co-founded Google in California" -> (Larry Page, founded, Google)',
    },
    "knowledge_graph": {
        "name": "Knowledge Graph (KG)",
        "definition": "A network structured representation of real-world knowledge where entities form nodes and semantic relationships form directed, typed edges. Enables multi-hop relational queries.",
        "example": "(Larry Page)-[:FOUNDED]->(Google)-[:LOCATED_IN]->(California)",
    },
    "triple": {
        "name": "Subject-Predicate-Object (SPO) Triple",
        "definition": "The atomic data unit in Knowledge Graphs and RDF consisting of exactly three components: (Subject Entity, Predicate / Relationship, Object Entity).",
        "example": "(Google, acquired, YouTube)\nSubject = Google, Predicate = acquired, Object = YouTube",
    },
    "subject": {
        "name": "Subject",
        "definition": "The source entity initiating or possessing the semantic relationship. In active voice English, the subject typically precedes the verbal predicate.",
        "example": 'In "Marie Curie discovered radium", the subject is Marie Curie (PERSON).',
    },
    "predicate": {
        "name": "Predicate / Relation",
        "definition": "The semantic link connecting subject and object. Normalised to snake_case without spaces to form clean edge types in graph databases.",
        "example": 'The verbs "co-founded", "founded", and "established" all map to the canonical predicate "founded".',
    },
    "object": {
        "name": "Object",
        "definition": "The target or destination entity that receives or completes the relation assertion.",
        "example": 'In "Microsoft is located in Redmond", the object is Redmond (LOCATION).',
    },
    "overfitting": {
        "name": "Overfitting",
        "definition": "When a method fits the examples it was built or trained on so closely that its score on them overstates how well it works on new text.",
        "example": "The BERT method scores 0.87 F1 on the sample corpora it was refined on, but 0.38 on the blind test.",
    },
    "training_data": {
        "name": "Training Data",
        "definition": "The labelled examples a machine-learning model learns from. A model is only as good as its training data: relations it saw few examples of are recognised less reliably.",
        "example": '"X bought Y", labelled acquired, is one training example for the BERT classifier.',
    },
    "held_out": {
        "name": "Held-out Data",
        "definition": "Data deliberately set aside and never used for training or tuning, kept only to measure how well a method works on things it has not seen.",
        "example": '64 relation wordings, such as "exchanged vows with", were kept out of BERT\'s training to test whether it generalises.',
    },
    "blind_test_set": {
        "name": "Blind Test Set",
        "definition": "A test set kept completely separate from building and tuning a method, and scored only as a whole, so its score is an honest estimate of real-world performance.",
        "example": "This lab's 106-sentence blind test: 60 Wikipedia sentences from Re-DocRED plus 46 hand-written ones.",
    },
    "noun_chunk": {
        "name": "Noun Chunk (Noun Phrase)",
        "definition": "A noun together with the words that describe it. Noun phrases that are the subject or object of a verb often name the concepts a sentence is about, even when they are not capitalised.",
        "example": 'In "Penicillin treats bacterial infections", the noun chunk "bacterial infections" becomes a TERM entity.',
    },
    "named_entity_recognition": {
        "name": "Named Entity Recognition (NER)",
        "definition": "The preceding IR stage that locates mentions of real-world objects in unstructured text and classifies them into predefined types (Experiment 6).",
        "example": "Recognizing [Sundar Pichai] as PERSON, [Google] as ORG, and [California] as LOCATION.",
    },
    "entity": {
        "name": "Entity",
        "definition": "A specific real-world object referred to in the document (person, organisation, place, or domain concept). Distinct from generic common nouns.",
        "example": '"Google" is an entity; "a software company" is a generic noun phrase.',
    },
    "entity_type": {
        "name": "Entity Type",
        "definition": "The category assigned to an entity (PERSON, ORG, LOCATION, DATE, TERM, MISC). Restricts which relational predicates can meaningfully link them.",
        "example": 'born_in requires subject: PERSON and object: LOCATION or DATE.',
    },
    "relation_cue": {
        "name": "Relation Cue / Trigger Lexicon",
        "definition": "The specific lexical token or verbal phrase between two entities that provides textual evidence for the relationship.",
        "example": 'In "Google acquired YouTube in 2006", the word "acquired" is the lexical cue.',
    },
    "pattern_based": {
        "name": "Pattern-Based Extraction",
        "definition": "Extracts triples only when an explicit relation cue matches a predefined lexical rule or regex. Achieves high precision, but lower recall due to natural language paraphrasing.",
        "example": 'Matches "X founded Y" and "X launched Y", but may miss "X was the architect behind Y".',
    },
    "cooccurrence": {
        "name": "Co-occurrence Extraction",
        "definition": "Links every pair of entities occurring in the same sentence or window with a generic related_to predicate. Achieves high recall but low precision.",
        "example": '(Einstein, related_to, Niels Bohr) with low baseline confidence 0.35.',
    },
    "hybrid": {
        "name": "Hybrid Extraction",
        "definition": "Applies pattern matching first with high confidence, and falls back to co-occurrence with lower confidence when no cue matches. Balances precision and recall.",
        "example": 'Yields (Google, acquired, YouTube) at 0.95 and (Google, related_to, Android) at 0.35.',
    },
    "confidence": {
        "name": "Confidence Score",
        "definition": "A heuristic score in [0, 1] indicating extraction reliability. Allows downstream graph ingestion to filter noise.",
        "example": "Score 0.95 for unambiguous cue 'acquired'; score 0.35 for unguided co-occurrence.",
    },
    "threshold": {
        "name": "Minimum Confidence Threshold",
        "definition": "The cutoff below which candidate triples are discarded. Higher thresholds favour precision; lower thresholds favour recall.",
        "example": "Threshold 0.70 filters out weak verbal ties and generic co-occurrence edges.",
    },
    "word_distance": {
        "name": "Maximum Word Distance",
        "definition": "The maximum permitted token distance between two entities in a sentence. Prevents linking entities in unrelated clauses.",
        "example": "At limit = 10 words, entities 22 words apart in different clauses are rejected.",
    },
    "candidate_pair": {
        "name": "Candidate Pair",
        "definition": "A pair of entities found in the same sentence evaluated by the extraction engine for a valid relation.",
        "example": '"Page and Brin founded Google" generates pairs (Page, Brin) and (Brin, Google).',
    },
    "precision": {
        "name": "Precision (P)",
        "definition": "The fraction of extracted triples that are factually correct: P = Correct Extractions / Total Extracted.",
        "example": "10 triples extracted, 8 correct -> Precision = 0.80 (80%).",
    },
    "recall": {
        "name": "Recall (R)",
        "definition": "The fraction of actual relationships present in the text that were extracted: R = Correct Extractions / Total Present in Ground Truth.",
        "example": "12 relations exist in text, 8 extracted -> Recall = 0.67 (66.7%).",
    },
    "f1": {
        "name": "F1 Score",
        "definition": "The harmonic mean of Precision and Recall: F1 = 2 * (P * R) / (P + R). Penalises extreme imbalances between precision and recall.",
        "example": "P = 0.80, R = 0.67 -> F1 = 0.73.",
    },
    "reference_set": {
        "name": "Ground Truth Reference Set",
        "definition": "A human-curated set of gold standard triples known to be asserted by the text, used to evaluate extraction accuracy.",
        "example": "The Tech & Business corpus carries 14 gold triples against which systems are benchmarked.",
    },
    "false_positive": {
        "name": "False Positive (FP)",
        "definition": "A triple extracted by the system that is unsupported or incorrect based on the text. Lowers precision.",
        "example": 'Extracting (California, founded, Google) due to word proximity.',
    },
    "false_negative": {
        "name": "False Negative (FN)",
        "definition": "A valid relationship in the text that the extractor failed to extract (an omission). Lowers recall.",
        "example": 'Failing to extract a passive-voice fact: "YouTube was bought by Google".',
    },
    "graph_query_language": {
        "name": "Graph Query Language",
        "definition": "A declarative language for asking questions of a graph database by describing a pattern of nodes and edges, such as (person)-[founded]->(company).",
        "example": "Find every (person)-[founded]->(company) pair and return both names.",
    },
    "upsert": {
        "name": "Upsert (Idempotent Import)",
        "definition": "Insert a node or relationship only if it does not already exist, otherwise reuse it. Importing the same triples twice then never creates duplicates.",
        "example": "Importing (Google, acquired, YouTube) twice still leaves exactly one Google node.",
    },
    "graphml": {
        "name": "GraphML",
        "definition": "An open, XML-based file format for graphs. Most graph databases and graph tools can import it, so it carries extracted triples to the next experiment.",
        "example": "Each entity becomes a <node>, and each triple becomes an <edge> carrying its predicate and confidence.",
    },
    "nearest_verb": {
        "name": "Nearest-Verb Extraction",
        "definition": "A relation extraction method with no fixed word list: it takes the first verb-like word between two entities, reduces it to its base form, and uses that as the predicate.",
        "example": '"Google acquired YouTube" -> the verb "acquired" -> lemma "acquire" -> predicate "acquired".',
    },
    "svo_extraction": {
        "name": "Dependency-Parse (SVO) Extraction",
        "definition": "A relation extraction method that reads the sentence grammar instead of the words between entities. The subject, verb and object are found from their grammatical roles, so word order and passive voice no longer matter.",
        "example": '"YouTube was acquired by Google" -> subject-of-action = Google, object = YouTube -> (Google, acquired, YouTube).',
    },
    "provenance": {
        "name": "Provenance",
        "definition": "The audit trail tracking the exact source sentence and text span from which a knowledge triple was derived.",
        "example": 'Triple (Google, acquired, YouTube) stores provenance: "Google acquired YouTube in 2006."',
    },
    "coreference": {
        "name": "Co-reference Resolution",
        "definition": "Resolving pronouns ('he', 'she', 'they', 'it') and descriptive noun phrases back to their antecedent entity mention.",
        "example": '"Marie Curie won the prize. She moved to Paris." -> "She" refers to Marie Curie.',
    },
    "dependency_parsing": {
        "name": "Dependency Parsing",
        "definition": "Syntactic parsing that maps grammatical relationships between words. The shortest path between entities identifies the governing verb.",
        "example": "In complex or inverted sentences, dependency parse paths bridge long entity separations.",
    },
    "open_ie": {
        "name": "Open Information Extraction (OpenIE)",
        "definition": "Relation extraction that extracts arbitrary relation phrases directly from text without adhering to a fixed, predefined schema.",
        "example": 'Extracts ("Sundar Pichai", "served as senior executive at", "Google") directly.',
    },
    "brittleness": {
        "name": "Brittleness of Rules",
        "definition": "The susceptibility of hand-crafted pattern rules to fail when encountering varied linguistic paraphrasing, metaphors, or complex syntax.",
        "example": 'A rule matching "acquired" will miss colloquial paraphrases like "swallowed up" or "brought under its umbrella".',
    },
    "error_propagation": {
        "name": "Error Propagation",
        "definition": "A cascade in a multi-stage pipeline where errors in early stages (e.g. misclassified NER entity) irrevocably corrupt later stages (RE).",
        "example": 'If "Paris" is misclassified as PERSON, RE might output (Paris, studied_at, University).',
    },
    "lemma": {
        "name": "Lemma (Base Form)",
        "definition": "The dictionary form of a word, with tense, plural and other endings removed, so different spellings of one verb can be treated as the same relation.",
        "example": '"acquired", "acquires" and "acquiring" all have the lemma "acquire".',
    },
    "passive_voice": {
        "name": "Passive Voice",
        "definition": "A sentence form where the receiver of the action comes first and the doer follows after \"by\". Methods that read left-to-right often reverse the direction of the relation.",
        "example": '"YouTube was acquired by Google" means (Google, acquired, YouTube), not (YouTube, acquired, Google).',
    },
    "bert": {
        "name": "BERT",
        "definition": "A language model (Bidirectional Encoder Representations from Transformers) trained on huge amounts of text. It reads a whole sentence at once, so the meaning it gives a word depends on the words around it.",
        "example": 'In "river bank" and "bank loan", BERT gives "bank" two different meanings because of its neighbours.',
    },
    "transformer": {
        "name": "Transformer",
        "definition": "The neural-network design behind BERT. It uses attention to let every word look at every other word in the sentence, which is how it captures context.",
        "example": 'In "Google acquired YouTube", attention links "acquired" to both "Google" and "YouTube".',
    },
    "embedding": {
        "name": "Embedding",
        "definition": "A list of numbers (a vector) that represents the meaning of a piece of text. Texts with similar meanings get similar vectors, so meaning can be compared with arithmetic.",
        "example": '"X bought Y" and "X acquired Y" get nearly the same vector; "X was born in Y" gets a very different one.',
    },
    "cosine_similarity": {
        "name": "Cosine Similarity",
        "definition": "A score from about 0 to 1 for how closely two embeddings point in the same direction. 1 means the same meaning, and lower scores mean less related meanings.",
        "example": 'similarity("X bought Y", "X acquired Y") = 0.84, while similarity("X bought Y", "X was born in Y") = 0.54.',
    },
    "sentence_bert": {
        "name": "Sentence-BERT",
        "definition": "A version of BERT trained to produce one embedding per whole sentence, so two sentences can be compared directly. This lab uses the compact all-MiniLM-L6-v2 model (about 90 MB).",
        "example": "encode(\"X acquired Y\") returns a vector of 384 numbers.",
    },
    "entity_masking": {
        "name": "Entity Masking",
        "definition": "Replacing the entity names with neutral placeholders (X and Y) before comparing meaning, so the model judges the relationship itself and not the particular companies or people.",
        "example": '"Google acquired YouTube" becomes "X acquired Y".',
    },
    "classifier": {
        "name": "Classifier",
        "definition": "A model that puts each input into one of a fixed set of categories. It learns the rule from labelled examples instead of being programmed with it.",
        "example": 'Shown "X was bought by Y", the classifier answers "acquired, with the second entity as the subject".',
    }
}
