"""The assessment quiz questions."""

from __future__ import annotations



QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "What are the three canonical components of a knowledge triple extracted from unstructured text?",
        "options": [
            "A) Noun, Adjective, Verb",
            "B) Subject, Predicate, Object",
            "C) Keyword, Frequency, Weight",
            "D) Document, Term, Score"
        ],
        "answer_index": 1,
        "explanation": "Subject-Predicate-Object (SPO) triples form the atomic data assertions in Knowledge Graphs and Semantic Web models."
    },
    {
        "id": 2,
        "question": "In a Knowledge Graph construction pipeline, what do the extracted entities and predicates represent?",
        "options": [
            "A) Entities become nodes, and predicates become directed labeled edges",
            "B) Entities become edge labels, and predicates become nodes",
            "C) Both entities and predicates become database tables without relationships",
            "D) Entities become vector indexes and predicates are discarded"
        ],
        "answer_index": 0,
        "explanation": "Entities (e.g. Google, California) represent network nodes, while predicates (e.g. founded, located_in) form directed edges connecting them."
    },
    {
        "id": 3,
        "question": "Why does co-occurrence based extraction typically exhibit high recall but low precision?",
        "options": [
            "A) It only considers capitalized words",
            "B) It assumes any two entities co-occurring in a sentence are related, over-generating generic links without lexical proof",
            "C) It strictly enforces grammatical dependency rules",
            "D) It drops all pairs with distance greater than 2 words"
        ],
        "answer_index": 1,
        "explanation": "Co-occurrence captures almost all true pairs (high recall) but generates many spurious or meaningless relationships (low precision)."
    },
    {
        "id": 4,
        "question": "What is the primary vulnerability (limitation) of hand-crafted pattern-based relation extraction?",
        "options": [
            "A) Linguistic brittleness when encountering paraphrases, metaphors, or complex syntactic inversions",
            "B) High computational overhead compared to neural language models",
            "C) Inability to run on CPU without GPUs",
            "D) Extreme over-generation of false positive edges"
        ],
        "answer_index": 0,
        "explanation": "Pattern rules are brittle: they reliably catch anticipated phrasings ('acquired'), but fail on unexpected paraphrases ('swallowed up')."
    },
    {
        "id": 5,
        "question": "What does Open Information Extraction (OpenIE) NOT require, unlike traditional supervised relation extraction?",
        "options": [
            "A) Input text",
            "B) A fixed, predefined schema of relation types",
            "C) Sentences",
            "D) Words"
        ],
        "answer_index": 1,
        "explanation": "OpenIE extracts arbitrary relation phrases directly from text without needing a rigid predefined ontology or relation taxonomy."
    },
    {
        "id": 6,
        "question": "What NLP pre-processing stage resolves pronouns like 'she' or 'the company' back to their antecedent entity mention?",
        "options": [
            "A) Lemmatization",
            "B) Co-reference resolution",
            "C) Stop-word pruning",
            "D) Part-of-speech tagging"
        ],
        "answer_index": 1,
        "explanation": "Co-reference resolution links pronouns ('she') to their true referent (e.g. 'Marie Curie'), enabling facts spread across sentences to be extracted."
    },
    {
        "id": 7,
        "question": "In the triple (Google) --[acquired]--> (YouTube), which element is the predicate?",
        "options": [
            "A) Google",
            "B) YouTube",
            "C) acquired",
            "D) Confidence score"
        ],
        "answer_index": 2,
        "explanation": "'acquired' is the relational predicate characterizing the semantic connection between the subject and object."
    },
    {
        "id": 8,
        "question": "Why is attaching a quantitative confidence score to each extracted triple beneficial for downstream graph ingestion?",
        "options": [
            "A) It allows downstream graph databases to filter out low-confidence noise and prioritize verified facts",
            "B) It eliminates the need for entity normalization",
            "C) It automatically translates English into a graph query language",
            "D) It guarantees 100% precision regardless of input text quality"
        ],
        "answer_index": 0,
        "explanation": "Confidence scores empower downstream ingestion pipelines to discard unreliable inferences and weight graph algorithms."
    },
    {
        "id": 9,
        "question": "Which evaluation metric represents the harmonic mean of Precision and Recall?",
        "options": [
            "A) Accuracy",
            "B) F1 Score",
            "C) Mean Reciprocal Rank (MRR)",
            "D) Discounted Cumulative Gain (DCG)"
        ],
        "answer_index": 1,
        "explanation": "The F1 score is the harmonic mean of precision and recall: 2 * (P * R) / (P + R)."
    },
    {
        "id": 10,
        "question": "When importing extracted triples into a graph database, why is an upsert (create-if-not-exists) preferred over a plain insert?",
        "options": [
            "A) An upsert is idempotent: it avoids creating duplicate nodes when an entity appears in multiple triples",
            "B) A plain insert is not supported by modern graph databases",
            "C) An upsert automatically executes relationship extraction in memory",
            "D) An upsert deletes existing nodes before adding new ones"
        ],
        "answer_index": 0,
        "explanation": "An upsert creates a node or relationship only if it does not already exist, ensuring identical entities consolidate into a connected network."
    }
]
