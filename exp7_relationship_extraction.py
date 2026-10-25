"""
Virtual Laboratory: Knowledge Graphs and Information Retrieval Systems (KGIRS)
Experiment 7: Relationship Extraction from Unstructured Text
Department of Computer Engineering / Information Technology - Virtual Laboratory Portal

Extracts semantic relationships (Subject - Predicate - Object triples) between
entities identified in text, producing entity-relationship triples suitable for
graph construction (feeds directly into Experiment 8: Create and Manage a Graph
Database, and Experiment 9: Knowledge Graph Schema & Data Import).

Features:
- Native-theme knowledge graph visualization that automatically follows light/dark mode.
- Wikipedia-style hover pop-ups on technical terms in Theory and Simulation, each with a
  plain-language explanation and a worked example, plus a searchable glossary table.
- 6 comprehensive analytical tabs: Triples, Annotated Text, Entities, Graph, Flow (Sankey), and Evaluation.
- Ground truth reference sets for empirical Precision, Recall, and F1 evaluation.
- Trial logger, conceptual quiz with instant colour-coded feedback, and PDF report generator.

Note: the page itself uses no custom CSS (Streamlit's native light/dark themes render it
seamlessly); the one deliberate exception is a small, self-contained stylesheet for the
hover pop-up cards, which pure Streamlit widgets cannot produce on their own.

Code layout: this file is only the entry point (`streamlit run exp7_relationship_extraction.py`);
the lab itself lives in the exp7/ package, one module per concern.
"""

from exp7.app import main

if __name__ == "__main__":
    main()
