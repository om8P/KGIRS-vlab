"""The Theory section."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from exp7.charts import SVO_METHOD, _annotate_sentence, _build_method_comparison_figure
from exp7.content import EXPERIMENT_CONFIG, THEORY_CONTENT
from exp7.entities import SPACY_INSTALL_HINT, _load_spacy_parser, _split_sentences, extract_entities, spacy_parser_available
from exp7.extraction import extract_relationships
from exp7.graph_figure import _build_graph_figure
from exp7.method_figures import _build_dependency_arc_figure, _build_nearest_verb_steps_figure, _build_pipeline_diagram
from exp7.popups import render_glossary_expander, write_annotated
from exp7.theme import get_current_theme


def render_theory_section():
    """Renders Section 1: Theory, Background, Objectives, Procedure, and Glossary."""
    theme = get_current_theme()

    st.header("Theoretical Framework & Background")
    write_annotated(THEORY_CONTENT["background"])

    st.subheader("Visual Overview: The Extraction Pipeline")
    st.caption(
        "Every extraction run, regardless of method, passes through these six stages in order. "
        "Colours match the entity-type legend used throughout the Simulation section."
    )
    st.plotly_chart(_build_pipeline_diagram(theme), use_container_width=True, config={"displayModeBar": False})

    st.divider()
    st.subheader("Algorithmic Detail: How Each Extraction Method Works")
    st.write(
        "The five methods below form a progression from a fixed word list to real grammar "
        "analysis. The first four share the same first three pipeline stages (splitting, "
        "entity recognition, adjacent candidate-pair generation) and differ only in how a "
        "pair is turned into a predicate; the Dependency Parse method reads the sentence's "
        "grammar instead. Each tab shows the procedure as pseudocode, followed by that "
        "method's *live* output on the same worked example sentence."
    )

    _example_text = "Larry Page co-founded Google in California in 1998."
    _example_entities = extract_entities(_example_text)

    tab_cooc, tab_pattern, tab_hybrid, tab_verb, tab_svo = st.tabs([
        "Co-occurrence Only", "Pattern-Based (Lexicon Match)", "Hybrid (Pattern + Co-occurrence)",
        "Nearest-Verb (Generalized)", "Dependency Parse (SVO)"
    ])

    with tab_pattern:
        write_annotated(
            "Emits a triple **only** when the text span between two adjacent entities matches "
            "one of the ordered [[relation_cue|relation-cue]] regular expressions "
            "(`_RELATION_PATTERNS`). Patterns are tried most-specific-first and the **first "
            "match wins** - so `co-founded` is caught by the `founded` family before any looser "
            "pattern gets a chance. If nothing matches, the pair is silently dropped: this is "
            "why the method is high-[[precision]] but limited in [[recall]]."
        )
        st.code(
            "for sentence in split_sentences(document):\n"
            "    entities = extract_entities(sentence)                 # NER\n"
            "    for (subject, object) in adjacent_pairs(entities):    # skip if entity in between\n"
            "        span = text_between(subject, object)\n"
            "        if word_count(span) > max_word_distance:\n"
            "            continue                                      # too far apart, skip pair\n"
            "        for (regex, predicate, base_confidence) in RELATION_PATTERNS:  # ordered\n"
            "            if regex.search(span):\n"
            "                confidence = apply_type_constraints(predicate, subject, object, base_confidence)\n"
            "                emit Triple(subject, predicate, object, confidence, cue=regex.match_text)\n"
            "                break                                     # first match wins, stop looking\n"
            "        # else: no cue found -> pair produces NO triple in Pattern-Based mode",
            language="python"
        )
        st.caption(
            "`apply_type_constraints` softens confidence when the entity types look wrong for "
            "the predicate - e.g. `born_in` is discounted (x0.6) unless the subject is a PERSON, "
            "and `leads` is discounted (x0.75) unless the object is an ORG or LOCATION."
        )
        pattern_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                                method="Pattern-Based (Lexicon Match)")
        st.write(f"**Live trace** for: *\"{_example_text}\"*")
        if pattern_triples:
            st.dataframe(pd.DataFrame(pattern_triples)[["subject", "predicate", "object", "confidence", "cue"]],
                        hide_index=True, use_container_width=True)
        else:
            st.info("No cue matched in this example, so Pattern-Based mode emits no triple here.")

    with tab_cooc:
        write_annotated(
            "Makes no attempt to read the span at all: **every** adjacent entity pair in a "
            "sentence is linked with the generic predicate `related_to`. This maximises "
            "[[recall]] (nothing is ever silently dropped) at the cost of [[precision]] (most "
            "edges carry no real semantic content). Confidence is not fixed - it **decays with "
            "distance**, so two entities separated by many words score lower than two sitting "
            "right next to each other."
        )
        st.code(
            "for sentence in split_sentences(document):\n"
            "    entities = extract_entities(sentence)\n"
            "    for (subject, object) in adjacent_pairs(entities):\n"
            "        if word_count(span) > max_word_distance:\n"
            "            continue\n"
            "        confidence = max(0.20, 0.45 - 0.02 * word_count(span))   # decays with distance\n"
            "        emit Triple(subject, 'related_to', object, confidence, cue=None)",
            language="python"
        )
        cooc_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                             method="Co-occurrence Only")
        st.write(f"**Live trace** for: *\"{_example_text}\"*")
        if cooc_triples:
            st.dataframe(pd.DataFrame(cooc_triples)[["subject", "predicate", "object", "confidence"]],
                        hide_index=True, use_container_width=True)
        else:
            st.info("No adjacent entity pairs were found in this example.")

    with tab_hybrid:
        write_annotated(
            "Runs the **Pattern-Based** check first on every candidate pair; only when no cue "
            "matches does it fall back to a **Co-occurrence** edge instead of dropping the pair. "
            "The result is a superset of what either method produces alone: the confident, "
            "correctly-labelled edges from Pattern-Based, topped up with low-confidence "
            "`related_to` edges so no relationship is lost entirely. This is the [[hybrid]] "
            "default used throughout the Simulation section."
        )
        st.code(
            "for (subject, object) in adjacent_pairs(entities):\n"
            "    if word_count(span) > max_word_distance:\n"
            "        continue\n"
            "    match = first_matching_pattern(span)          # same ordered lexicon as Pattern-Based\n"
            "    if match:\n"
            "        emit Triple(subject, match.predicate, object, match.confidence, cue=match.text)\n"
            "    else:\n"
            "        confidence = max(0.20, 0.45 - 0.02 * word_count(span))\n"
            "        emit Triple(subject, 'related_to', object, confidence, cue=None)  # fallback",
            language="python"
        )
        hybrid_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                               method="Hybrid (Pattern + Co-occurrence)")
        st.write(f"**Live trace** for: *\"{_example_text}\"*")
        if hybrid_triples:
            st.dataframe(pd.DataFrame(hybrid_triples)[["subject", "predicate", "object", "confidence", "cue"]],
                        hide_index=True, use_container_width=True)
        else:
            st.info("No adjacent entity pairs were found in this example.")

    _passive_text = "YouTube was acquired by Google in 2006."
    _passive_entities = extract_entities(_passive_text)

    with tab_verb:
        write_annotated(
            "Drops the fixed word list. For each adjacent entity pair it takes the **first "
            "verb-like word** between them, reduces it to its [[lemma]] (base form) and uses "
            "that as the predicate - so verbs the lexicon never listed (\"oversees\", "
            "\"divested\") still become relations. Common verbs are then mapped onto the same "
            "predicate names as the lexicon (`acquire` -> `acquired`). It reads strictly "
            "left-to-right, so it cannot tell who did what in a [[passive_voice|passive sentence]]."
        )
        st.caption("Illustration: how one pair of entities becomes a predicate (hover each box).")
        st.plotly_chart(_build_nearest_verb_steps_figure(theme), use_container_width=True,
                        config={"displayModeBar": False})
        st.code(
            "for (subject, object) in adjacent_pairs(entities):\n"
            "    span = text_between(subject, object)\n"
            "    verb = first_word_that_looks_like_a_verb(span)   # -ed / -ing / -s or irregular\n"
            "    if verb is None:\n"
            "        continue                                      # no verb -> no triple\n"
            "    lemma = lemmatize(verb)                           # 'acquired' -> 'acquire'\n"
            "    if next_word(verb) is a preposition:              # 'based' + 'in'\n"
            "        lemma = lemma + '_' + preposition             # 'base_in'\n"
            "    predicate = canonical_name(lemma)                 # 'base_in' -> 'located_in'\n"
            "    emit Triple(subject, predicate, object, confidence=0.60, cue=verb)",
            language="python"
        )
        verb_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                             method="Nearest-Verb (Generalized)")
        st.write(f"**Live trace** for: *\"{_example_text}\"*")
        if verb_triples:
            st.dataframe(pd.DataFrame(verb_triples)[["subject", "predicate", "object", "confidence", "cue"]],
                        hide_index=True, use_container_width=True)
        else:
            st.info("No verb was found between an adjacent entity pair in this example.")

    with tab_svo:
        write_annotated(
            "Instead of looking at the words between two entities, this method reads the "
            "sentence's grammar using [[dependency_parsing|dependency parsing]]: every word is "
            "labelled with its role (subject, object, ...) and the word it depends on. A "
            "triple is emitted when two entities hang off the same verb - so it works when "
            "the entities are far apart, when one subject has several verbs, and in the "
            "[[passive_voice|passive voice]], where it correctly swaps the direction."
        )
        st.code(
            "doc = parse(sentence)                       # grammar tree (spaCy)\n"
            "for verb in doc.verbs:\n"
            "    subjects = entities attached to verb as nsubj / nsubjpass\n"
            "    for obj in entities attached as dobj  -> Triple(subj, lemma(verb), obj)\n"
            "    for obj in entities attached via a preposition ('compete' + 'with')\n"
            "                                          -> Triple(subj, lemma_prep, obj)\n"
            "    for agent in entities attached via 'by' (passive)\n"
            "                                          -> Triple(agent, lemma(verb), subj)  # swapped\n"
            "    'X is the CEO of Y' -> Triple(X, leads, Y)\n"
            "    a verb joined by 'and' inherits the previous verb's subject",
            language="python"
        )
        if not spacy_parser_available():
            st.warning(
                "This method needs the optional spaCy parser, which is not installed on this "
                f"server. Install it with: `{SPACY_INSTALL_HINT}`"
            )
        else:
            svo_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                                method=SVO_METHOD)
            st.write(f"**Live trace** for: *\"{_example_text}\"*")
            if svo_triples:
                st.dataframe(pd.DataFrame(svo_triples)[["subject", "predicate", "object", "confidence", "cue"]],
                            hide_index=True, use_container_width=True)
            else:
                st.info("No grammatical subject-verb-object link was found in this example.")

            st.markdown("**Why grammar matters: a passive sentence**")
            st.write(f"*\"{_passive_text}\"* - the correct fact is (Google, acquired, YouTube).")
            _parse_doc = _load_spacy_parser()(_passive_text)
            st.caption(
                "Illustration: the sentence's grammar tree. Each arc runs from a word to the word "
                "that depends on it, labelled with its role. Coloured arcs carry the meaning: "
                "`nsubjpass` (receiver of the action), `agent` (the doer, after \"by\") and `pobj`. "
                "Italics show each word's lemma."
            )
            st.plotly_chart(_build_dependency_arc_figure(_parse_doc, theme), use_container_width=True,
                            config={"displayModeBar": False})
            _rows = []
            for _m in ("Pattern-Based (Lexicon Match)", "Nearest-Verb (Generalized)", SVO_METHOD):
                _t = extract_relationships(text=_passive_text, entities=_passive_entities, method=_m)
                _got = ", ".join(f"({x['subject']}, {x['predicate']}, {x['object']})" for x in _t) or "(nothing)"
                _ok = any(x["subject"] == "Google" and x["object"] == "YouTube" for x in _t)
                _rows.append({"Method": _m, "Extracted": _got, "Correct direction?": "Yes" if _ok else "No"})
            st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Illustrated Example: From Raw Sentences to a Knowledge Graph")
    _illustration_text = (
        "Larry Page and Sergey Brin co-founded Google in California in 1998. "
        "Google acquired YouTube in 2006."
    )
    _illustration_entities = extract_entities(_illustration_text)
    _illustration_triples = extract_relationships(
        text=_illustration_text, entities=_illustration_entities,
        method="Hybrid (Pattern + Co-occurrence)"
    )
    st.write("**Step 1 - Annotated source text** (entities highlighted by type, relation cues underlined):")
    _illustration_sentences = _split_sentences(_illustration_text)
    _illustration_cues = [t["cue"] for t in _illustration_triples if t.get("cue")]
    for s_idx, s_text in enumerate(_illustration_sentences):
        s_ents = [e for e in _illustration_entities if e["sentence_idx"] == s_idx]
        st.markdown(_annotate_sentence(s_text, s_ents, _illustration_cues))
    st.write("**Step 2 - Resulting knowledge graph** (same layout engine used in the Simulation section):")
    st.plotly_chart(_build_graph_figure(_illustration_triples, theme, show_predicates=True),
                    use_container_width=True)

    st.divider()
    st.subheader("Comparing the Extraction Methods")
    write_annotated(
        "The qualitative trade-offs above are confirmed empirically below: every method "
        "runs against the **Tech & Business** benchmark corpus and is scored against its "
        "human-curated [[reference_set|gold reference set]]."
    )
    st.markdown(
        "| Method | Precision | Recall | Interpretability | Typical Confidence | Best suited to |\n"
        "|---|---|---|---|---|---|\n"
        "| **Pattern-Based** | High | Lower | Every edge traces to an explicit cue word | 0.80-0.95 "
        "(fixed per pattern) | Clean, well-worded documents where missing a few facts is "
        "acceptable but wrong facts are not |\n"
        "| **Co-occurrence** | Low | Highest | Trivial (proximity only) | 0.20-0.45 (decays with "
        "distance) | Early-stage exploration, or as a recall safety-net alongside a stricter "
        "method |\n"
        "| **Hybrid** | Medium-High | High | Mostly cue-based, with visibly weaker fallback edges | "
        "Mixed (0.20-0.95) | General-purpose default; lets a confidence threshold slider do the "
        "precision/recall trade-off instead of the extraction method itself |\n"
        "| **Nearest-Verb** | Medium | Medium | Cue is the verb itself; reads left-to-right only | "
        "0.60 (fixed) | Text using verbs outside the lexicon; no extra software needed |\n"
        "| **Dependency Parse (SVO)** | High | High | Each edge shows its grammatical roles | "
        "0.80-0.85 | Varied or passive phrasing and distant entities; needs the optional spaCy parser |\n"
    )
    st.plotly_chart(_build_method_comparison_figure(theme), use_container_width=True)
    st.caption(
        "Precision and Recall move in opposite directions between Pattern-Based and "
        "Co-occurrence - the classic precision/recall trade-off - and Hybrid's F1 lands "
        "between them because it never drops a candidate pair. Co-occurrence scores 0 on the "
        "strict metrics because it always labels edges `related_to`; Pair Recall shows the "
        "share of correct entity pairs it still found. Nearest-Verb and Dependency Parse "
        "name the relation from the sentence itself; Dependency Parse is skipped here if "
        "spaCy is not installed."
    )

    st.divider()
    st.subheader("Educational Aim & Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        st.markdown(f"- **Goal {i+1}**: {obj}")

    st.divider()
    st.subheader("Standard Experimental Procedure")
    for step in THEORY_CONTENT["procedure"]:
        st.markdown(f"- {step}")

    st.divider()
    render_glossary_expander()

    st.divider()
    with st.expander("Pipeline Integration: Connection to the Complete KGIRS Architecture", expanded=False):
        st.markdown("""
This laboratory shares an standardized `st.session_state` data contract to integrate seamlessly
across the entire KGIRS curriculum:

| Session State Key | Producer Module | Consumer Modules |
|---|---|---|
| `kg_source_text` | Experiment 1-2 (Preprocessing) or this page | Experiment 6, this page |
| `kg_entities` | Experiment 6 (Named Entity Recognition) | **This page (Exp 7)**, Exp 8, Exp 9 |
| `kg_triples` | **This page (Experiment 7: Relationship Extraction)** | Experiment 8 (Graph DB Ingestion), Exp 9 (Schema Design), Exp 10-11 (Graph Querying) |

Downstream, Experiment 8 and 9 consume the extracted triples directly using an idempotent upsert:
```text
for each triple (subject, predicate, object, confidence):
    upsert node  Entity {name: subject, type: subject_label}
    upsert node  Entity {name: object,  type: object_label}
    upsert edge  subject -[RELATION {type: predicate, confidence: confidence}]-> object
```
        """)
