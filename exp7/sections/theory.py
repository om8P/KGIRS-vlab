"""The Theory section."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from exp7.bert_figures import BERT_TRAINING_STATS_PATH, _BERT_EXAMPLE_CLAUSE, _bert_unseen_wordings_table, _build_bert_embedding_heatmap, _build_bert_meaning_map, _build_bert_similarity_figure, _build_bert_steps_figure, _build_bert_training_data_figure
from exp7.bert_method import BERT_INSTALL_HINT, BERT_METHOD, BERT_MIN_PROBABILITY, _bert_candidate_clauses, _bert_encode, bert_available
from exp7.charts import BLIND_RESULTS_PATH, SVO_METHOD, _annotate_sentence, _build_blind_test_figure, _build_method_comparison_figure
from exp7.content import EXPERIMENT_CONFIG, THEORY_CONTENT
from exp7.entities import SPACY_INSTALL_HINT, _load_spacy_parser, _split_sentences, extract_entities, spacy_parser_available
from exp7.extraction import extract_relationships
from exp7.figure_utils import _load_json_file
from exp7.graph_figure import _build_graph_figure
from exp7.lexicon import _MAX_WORDS_BETWEEN
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
        "The six methods below form a progression from a fixed word list, through grammar "
        "analysis, to a language model that compares meaning. Five of them share the same "
        "first three pipeline stages (splitting, entity recognition, adjacent candidate-pair "
        "generation) and differ only in how a pair is turned into a predicate; the "
        "Dependency Parse method reads the sentence's grammar instead. Each tab shows the "
        "procedure as pseudocode, followed by that method's *live* output on the same "
        "worked example sentence."
    )

    _example_text = "Larry Page co-founded Google in California in 1998."
    _example_entities = extract_entities(_example_text)

    tab_cooc, tab_pattern, tab_hybrid, tab_verb, tab_svo, tab_bert = st.tabs([
        "Co-occurrence Only", "Pattern-Based (Lexicon Match)", "Hybrid (Pattern + Co-occurrence)",
        "Nearest-Verb (Generalized)", "Dependency Parse (SVO)", "BERT Similarity (Sentence-BERT)"
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
            _passive_methods = ["Pattern-Based (Lexicon Match)", "Nearest-Verb (Generalized)", SVO_METHOD]
            if bert_available():
                _passive_methods.append(BERT_METHOD)
            for _m in _passive_methods:
                _t = extract_relationships(text=_passive_text, entities=_passive_entities, method=_m)
                _got = ", ".join(f"({x['subject']}, {x['predicate']}, {x['object']})" for x in _t) or "(nothing)"
                _ok = any(x["subject"] == "Google" and x["object"] == "YouTube" for x in _t)
                _rows.append({"Method": _m, "Extracted": _got, "Correct direction?": "Yes" if _ok else "No"})
            st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True)

    with tab_bert:
        write_annotated(
            "Instead of rules, this method asks a language model what the text between two "
            "entities *means*. [[bert|BERT]] turns every clause into an [[embedding]] (a "
            "vector of numbers); the names are first hidden "
            "([[entity_masking|entity masking]]) so only the relationship is judged. A small "
            "[[classifier]] then reads that embedding plus the two entity types and answers with "
            "one of the 18 relations, the same relation reversed (the "
            "[[passive_voice|passive voice]]: *X was acquired by Y* means Y is the subject), or "
            "*none*. It was trained on a few thousand example clauses: wordings written for this "
            "lab and Wikipedia sentences from the public Re-DocRED dataset (MIT licence). If the "
            "best answer is *none*, or not confident enough, the pair is skipped."
        )
        st.caption("Illustration: the BERT method step by step (hover each box).")
        st.plotly_chart(_build_bert_steps_figure(theme), use_container_width=True,
                        config={"displayModeBar": False})
        st.code(
            "# trained once (train_bert_re.py): BERT embeddings + entity types of example clauses -> classifier\n"
            "for (subject, object, clause) in candidate_clauses(entities):   # reading rules, see below\n"
            "    clause = 'X ' + text_between + ' Y'                          # names masked\n"
            "    probs = classifier(embed(clause), type(subject), type(object))\n"
            "    best = argmax(probs)\n"
            f"    if best is NONE or probs[best] < {BERT_MIN_PROBABILITY}:\n"
            "        continue                                                # no relation / not confident\n"
            "    if best is the reversed form: swap(subject, object)           # passive voice\n"
            "    emit Triple(subject, best.predicate, object, confidence=probs[best], cue=clause)\n"
            "\n"
            "# candidate_clauses: years are skipped; 'A and B <verb> C' gives both A and B as subjects;\n"
            "# 'A acquired B and partnered with C' -> the part after 'and' is judged with subject A;\n"
            "# 'M, based in R, competes with G' -> after the aside, the subject is M",
            language="python"
        )
        if not bert_available():
            st.warning(
                "This method needs the optional BERT library and model, which are not available on "
                f"this server. Install with: `{BERT_INSTALL_HINT}` (the model, about 90 MB, downloads "
                "on first use)."
            )
        else:
            bert_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                                 method=BERT_METHOD)
            st.write(f"**Live trace** for: *\"{_example_text}\"*")
            if bert_triples:
                st.dataframe(pd.DataFrame(bert_triples)[["subject", "predicate", "object", "confidence", "cue"]],
                             hide_index=True, use_container_width=True)
            else:
                st.info("No clause was similar enough to an example predicate in this sentence.")

            st.markdown("**Step 1 - Reading rules decide which entities to pair**")
            _rr_text = "Microsoft, based in Redmond, acquired LinkedIn and partnered with OpenAI."
            _rr_ents = extract_entities(_rr_text)
            _rr_sentence = _rr_ents[0]["sentence"] if _rr_ents else _rr_text
            _rr_sorted = sorted((e for e in _rr_ents if e["label"] != "DATE"), key=lambda e: e["start"])
            _rr_rows = [{"Pairing": "Neighbours only", "Subject": a["text"], "Object": b["text"],
                         "Clause judged": "X " + _rr_sentence[a["end"]:b["start"]].strip() + " Y"}
                        for a, b in zip(_rr_sorted, _rr_sorted[1:])]
            _rr_rows += [{"Pairing": "Reading rules", "Subject": subj["text"], "Object": obj["text"],
                          "Clause judged": clause}
                         for subj, obj, clause in _bert_candidate_clauses(_rr_sentence, _rr_ents, _MAX_WORDS_BETWEEN)]
            st.caption(
                f"*\"{_rr_text}\"* - pairing only neighbouring entities gives the acquisition to "
                "Redmond and the partnership to LinkedIn. The reading rules make Microsoft the subject "
                "after the comma aside and after \"and\", and judge only the words that belong to that pair."
            )
            st.dataframe(pd.DataFrame(_rr_rows), hide_index=True, use_container_width=True)
            _rr_triples = extract_relationships(text=_rr_text, entities=_rr_ents, method=BERT_METHOD)
            if _rr_triples:
                st.caption("Triples BERT extracts from this sentence:")
                st.dataframe(pd.DataFrame(_rr_triples)[["subject", "predicate", "object", "confidence"]],
                             hide_index=True, use_container_width=True)

            st.markdown("**Step 2 - BERT turns each clause into 384 numbers**")
            _fp_clauses = ["X bought Y", "X acquired Y", "X was born in Y"]
            _fp_vecs = _bert_encode(_fp_clauses)
            if _fp_vecs is not None:
                _sim = _fp_vecs @ _fp_vecs.T
                st.caption(
                    "Each stripe is one clause's embedding, coloured number by number (red = positive, "
                    "blue = negative). The two buying clauses share much of their pattern; the birth clause "
                    f"does not. Cosine similarity: bought vs acquired {_sim[0, 1]:.2f}, bought vs born in "
                    f"{_sim[0, 2]:.2f}, acquired vs born in {_sim[1, 2]:.2f}."
                )
                _hm_fig = _build_bert_embedding_heatmap(_fp_clauses, theme)
                if _hm_fig is not None:
                    st.plotly_chart(_hm_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("**Step 3 - How the decision is made**")
            st.caption(
                "The passive sentence *\"YouTube was purchased by Google\"* becomes the masked clause "
                f"`{_BERT_EXAMPLE_CLAUSE}` (YouTube = X, Google = Y). That exact wording is not among "
                "the example clauses, so this shows BERT generalising by meaning. The bars show how "
                "likely each relation is; the winner must be a relation (not NONE) and clear the dashed "
                "threshold. With the trained classifier the bars are its probabilities (the "
                "\"reversed\" form means the second entity is the subject); hover a bar for details."
            )
            _bar_fig = _build_bert_similarity_figure(_BERT_EXAMPLE_CLAUSE, theme)
            if _bar_fig is not None:
                st.plotly_chart(_bar_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("**Step 4 - The meaning space**")
            st.caption(
                "Each dot is one example clause, squeezed from 384 numbers down to two so it can "
                "be drawn; clauses with similar meanings sit together. The star is the clause above: "
                "it lands inside the *acquired* group, whatever words it uses."
            )
            _map_fig = _build_bert_meaning_map(_BERT_EXAMPLE_CLAUSE, theme)
            if _map_fig is not None:
                st.plotly_chart(_map_fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("**Wordings it has never seen**")
            _unseen = _bert_unseen_wordings_table()
            if _unseen is not None:
                st.caption(
                    "None of these wordings is in the word list or in BERT's training data. The word list "
                    "finds nothing, or only its weak catch-all is_a guess; BERT recognises some of them "
                    "by meaning but gets others wrong - its mistakes are left visible here."
                )
                st.dataframe(_unseen, hide_index=True, use_container_width=True)

            st.markdown("**What it learned from**")
            _train_fig = _build_bert_training_data_figure(theme)
            if _train_fig is not None:
                _stats = _load_json_file(BERT_TRAINING_STATS_PATH) or {}
                _none = _stats.get("per_relation", {}).get("NONE", {})
                st.caption(
                    "Training examples per relation. Re-DocRED (real Wikipedia sentences) covers only some "
                    "relations; the rest rely entirely on wordings written for this lab, which is one reason "
                    "BERT is less reliable on them. Not shown: "
                    f"{_none.get('synthetic', 0) + _none.get('re-docred', 0):,} 'no relation' examples."
                )
                st.plotly_chart(_train_fig, use_container_width=True, config={"displayModeBar": False})

            _blind = _load_json_file(BLIND_RESULTS_PATH)
            _bert_blind = next((r for r in (_blind or {}).get("results", []) if r["method"] == BERT_METHOD), None)
            _blind_note = (
                f" On the blind test it reached F1 {_bert_blind['part_a']['f1']:.2f} on real Wikipedia "
                f"sentences and {_bert_blind['part_b']['f1']:.2f} on hand-written ones - the best of the six "
                "methods, but far below its score on the sample corpora." if _bert_blind else ""
            )
            st.caption(
                "Limits: on 64 wordings it had never seen, the trained classifier got roughly 7 in 10 right, "
                "against 3 in 10 for matching example clauses and almost none for a fixed word list."
                + _blind_note +
                " It can still be wrong, it skips wordings unlike anything it has seen, and BERT cannot "
                "explain why it chose a relation."
            )

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
        "| **BERT Similarity** | High | Medium-High | Black box: shows the clause and the probability it assigned | "
        "0.50-0.99 (the classifier's probability) | Wordings no rule lists; needs the optional BERT model |\n"
    )
    st.plotly_chart(_build_method_comparison_figure(theme), use_container_width=True)
    st.caption(
        "Precision and Recall move in opposite directions between Pattern-Based and "
        "Co-occurrence - the classic precision/recall trade-off - and Hybrid's F1 lands "
        "between them because it never drops a candidate pair. Co-occurrence scores 0 on the "
        "strict metrics because it always labels edges `related_to`; Pair Recall shows the "
        "share of correct entity pairs it still found. Nearest-Verb and Dependency Parse "
        "name the relation from the sentence itself, and BERT Similarity judges meaning with a "
        "trained classifier. Dependency Parse and BERT are skipped here if spaCy or the BERT "
        "library is not installed."
    )

    st.subheader("How Well Do They Really Work? (Blind Test)")
    write_annotated(
        "The chart above scores the methods on the four sample corpora, which helped shape them, so "
        "it is optimistic ([[overfitting]]). The [[blind_test_set|blind test]] below uses sentences "
        "that played no part in building any method: 60 real Wikipedia sentences from the public "
        "Re-DocRED dataset (Part A) and 46 hand-written sentences (Part B)."
    )
    _blind_fig = _build_blind_test_figure(theme)
    if _blind_fig is None:
        st.info("Blind-test results are not available. Run `python benchmarks/evaluate_blind.py --save` to create them.")
    else:
        st.plotly_chart(_blind_fig, use_container_width=True, config={"displayModeBar": False})
        _blind_meta = _load_json_file(BLIND_RESULTS_PATH) or {}
        st.caption(
            "Every method scores far lower here than on the sample corpora. Real Wikipedia sentences are "
            "long and indirect, so related entities often sit far apart; the rule-based methods find "
            "almost none of them (recall under 3%), while BERT recovers the most. Part A can only judge "
            "the 10 relations Re-DocRED annotates. "
            f"Measured {_blind_meta.get('date', '')}"
            + (" with spaCy and the BERT classifier installed." if _blind_meta.get("spacy_available") and
               _blind_meta.get("bert_classifier_available") else ".")
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
