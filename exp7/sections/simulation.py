"""The Simulation section: document selection, extraction and results."""

from __future__ import annotations

import html
import io
import json
from typing import Dict, Optional

import networkx as nx
import pandas as pd
import streamlit as st

from exp7.bert_method import BERT_METHOD
from exp7.charts import SVO_METHOD, _annotate_sentence, _build_entity_type_chart, _build_flow_figure
from exp7.corpora import CORPORA
from exp7.entities import SPACY_INSTALL_HINT, _split_sentences, extract_entities, spacy_parser_available
from exp7.extraction import compute_extraction_metrics, evaluate_against_reference_set, extract_relationships
from exp7.graph_figure import ALL_ENTITIES, _build_graph_figure, _prepare_graph_triples
from exp7.graph_layout import MAX_GRAPH_NODES
from exp7.popups import write_annotated
from exp7.theme import ENTITY_TYPE_ORDER, get_current_theme
from exp7.uploads import LoadedDocument, MAX_UPLOAD_CHARACTERS, load_uploaded_document


UPLOAD_CHOICE = "Upload Document File (.txt, .pdf, .docx, .csv)"


def _render_upload_preview(loaded: LoadedDocument) -> None:
    """Shows what the extractor will actually process for an uploaded document."""
    text = loaded.text
    sentences = _split_sentences(text)
    with st.container(border=True):
        st.subheader("Document Preview")
        st.caption(loaded.summary())
        cols = st.columns(6 if loaded.page_count else 5)
        cols[0].metric("File type", f".{loaded.extension}" if loaded.extension else "n/a")
        cols[1].metric("Size", f"{loaded.size_bytes / 1024:.1f} KB")
        idx = 2
        if loaded.page_count:
            cols[idx].metric("Pages", loaded.page_count)
            idx += 1
        cols[idx].metric("Characters", f"{len(text):,}")
        cols[idx + 1].metric("Words", f"{len(text.split()):,}")
        cols[idx + 2].metric("Sentences", f"{len(sentences):,}")
        if loaded.truncated:
            st.warning(f"The file has {loaded.raw_character_count:,} characters; only the first "
                       f"{MAX_UPLOAD_CHARACTERS:,} will be processed.")

        tab_text, tab_ents = st.tabs(["Extracted Text", "Entities Preview"])
        with tab_text:
            st.caption("This is exactly the text the extraction pipeline will run on.")
            with st.container(height=260):
                st.markdown(
                    '<div style="font-family: Verdana, \'Segoe UI\', Tahoma, sans-serif; font-size: 0.82rem; '
                    f'line-height: 1.7; white-space: pre-wrap;">{html.escape(text)}</div>',
                    unsafe_allow_html=True
                )
        with tab_ents:
            entities = extract_entities(text)
            st.caption(f"{len({e['text'].lower() for e in entities})} distinct entities found across "
                       f"{len(sentences)} sentences. First sentences shown with entities highlighted:")
            for s_idx, s_text in enumerate(sentences[:5]):
                s_ents = [e for e in entities if e["sentence_idx"] == s_idx]
                st.markdown(_annotate_sentence(s_text, s_ents, []))


def render_simulation_section():
    """Renders Section 2: Interactive Sandbox, 6 Analytical Tabs, Evaluation, and Exports."""
    theme = get_current_theme()

    st.header("Interactive Relationship Extraction Sandbox")
    write_annotated(
        "Configure extraction parameters, run [[pattern_based|Pattern-Based]], "
        "[[cooccurrence|Co-occurrence]], or [[hybrid|Hybrid]] extraction, and evaluate the "
        "resulting [[triple|triples]] against a [[reference_set|ground truth]] where available."
    )

    # 1. Document Selection Card
    with st.container():
        doc_names = list(CORPORA.keys()) + ["Custom Input", UPLOAD_CHOICE]
        doc_choice = st.selectbox(
            "Select Source Document / Corpus",
            options=doc_names,
            help="Choose from benchmark pedagogical corpora with ground truth reference sets, or supply custom text."
        )

        selected_corpus = CORPORA.get(doc_choice)
        is_upload = doc_choice == UPLOAD_CHOICE
        default_text = ""
        gold_triples = []
        loaded: Optional[LoadedDocument] = None
        uploaded_file = None

        if selected_corpus:
            default_text = selected_corpus["text"]
            gold_triples = selected_corpus.get("gold", [])
            col_b1, col_b2 = st.columns([1, 4])
            with col_b1:
                st.info(f"**Level:** {selected_corpus['level']}")
            with col_b2:
                st.caption(f"**Pedagogical Purpose:** {selected_corpus['purpose']}")
            with st.expander("Things to Observe in this Document", expanded=False):
                for obs in selected_corpus.get("observe", []):
                    st.markdown(f"- {obs}")
        elif is_upload:
            uploaded_file = st.file_uploader(
                "Upload document",
                type=["txt", "md", "csv", "tsv", "pdf", "docx"],
                help="Upload a file up to 25,000 characters."
            )
            if uploaded_file is not None:
                # Parse once per file (not on every rerun) and remember the result.
                cached = st.session_state.get("exp7_uploaded_doc")
                if not cached or cached[0] != uploaded_file.file_id:
                    cached = (uploaded_file.file_id, load_uploaded_document(uploaded_file))
                    st.session_state["exp7_uploaded_doc"] = cached
                loaded = cached[1]
        else:
            upstream = st.session_state.get("kg_source_text", "")
            default_text = upstream if upstream else "Larry Page co-founded Google in California in 1998."

        # When the chosen source changes, reset the text box and drop results from the old source.
        if is_upload:
            source_sig = f"upload:{uploaded_file.file_id if uploaded_file is not None else 'none'}"
        else:
            source_sig = doc_choice
        if st.session_state.get("exp7_source_sig") != source_sig:
            st.session_state["exp7_source_sig"] = source_sig
            st.session_state["sandbox_text_area"] = default_text
            for stale in ("exp7_last_triples", "exp7_last_entities", "exp7_last_text",
                          "exp7_last_method", "exp7_last_corpus", "exp7_last_eval"):
                st.session_state.pop(stale, None)

        can_run = True
        if is_upload:
            # The uploaded document itself is processed: no editable text box in this mode.
            text_input = loaded.text if (loaded and not loaded.error) else ""
            if uploaded_file is None:
                st.info("Upload a document above to preview it and run extraction on it.")
                can_run = False
            elif loaded.error:
                st.error(loaded.error)
                can_run = False
            elif not text_input.strip():
                st.warning(
                    f"No text could be read from `{loaded.filename}`. A scanned PDF or an image-only "
                    "file has no text layer to extract from."
                )
                can_run = False
            else:
                _render_upload_preview(loaded)
        else:
            text_input = st.text_area(
                "Document Text (editable for experimentation)",
                height=140,
                key="sandbox_text_area"
            )

    # 2. Parameters Card
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        method = st.radio(
            "Extraction Strategy",
            options=[
                "Co-occurrence Only",
                "Pattern-Based (Lexicon Match)",
                "Hybrid (Pattern + Co-occurrence)",
                "Nearest-Verb (Generalized)",
                "Dependency Parse (SVO)",
                BERT_METHOD,
            ],
            index=2,
            help="Pattern-Based prioritizes precision; Co-occurrence maximizes recall; Hybrid balances both; "
                 "Nearest-Verb uses the verb between entities; Dependency Parse reads sentence grammar (needs spaCy); "
                 "BERT Similarity judges the meaning of the text between entities with a trained classifier (needs sentence-transformers)."
        )
        if method == SVO_METHOD and not spacy_parser_available():
            st.warning(f"The Dependency Parse method needs the optional spaCy parser: `{SPACY_INSTALL_HINT}`. "
                       "It will return no triples until installed.")
    with col_p2:
        min_confidence = st.slider(
            "Minimum Confidence Cutoff",
            min_value=0.10,
            max_value=1.0,
            value=0.30,
            step=0.05,
            help="Filters out triples whose confidence falls below this threshold."
        )
    with col_p3:
        max_dist = st.slider(
            "Max Word Separation",
            min_value=4,
            max_value=25,
            value=12,
            step=1,
            help="Maximum words allowed between two entities in a sentence to form a candidate pair."
        )
        use_upstream_ner = st.checkbox(
            "Reuse upstream entities (Exp 6)",
            value=True,
            help="If Experiment 6 already executed in this session, reuse its recognized entities."
        )

    run_btn = st.button("Run Relationship Extraction Pipeline", type="primary", use_container_width=True,
                        disabled=not can_run)

    # Execution or cached state retrieval
    if run_btn or ("exp7_last_triples" in st.session_state and can_run):
        if run_btn:
            upstream_ents = st.session_state.get("kg_entities")
            if use_upstream_ner and upstream_ents and st.session_state.get("kg_source_text") == text_input:
                entities = upstream_ents
            else:
                entities = extract_entities(text_input)

            triples = extract_relationships(
                text=text_input,
                entities=entities,
                method=method,
                max_word_distance=max_dist
            )
            filtered_triples = [t for t in triples if t["confidence"] >= min_confidence]

            st.session_state["exp7_last_entities"] = entities
            st.session_state["exp7_last_triples"] = filtered_triples
            st.session_state["exp7_last_text"] = text_input
            st.session_state["exp7_last_method"] = method
            st.session_state["exp7_last_corpus"] = doc_choice

            # Downstream pipeline session state contract
            st.session_state["kg_entities"] = entities
            st.session_state["kg_triples"] = filtered_triples
            st.session_state["kg_source_text"] = text_input

        entities = st.session_state.get("exp7_last_entities", [])
        triples = st.session_state.get("exp7_last_triples", [])
        metrics = compute_extraction_metrics(entities, triples)
        eval_result = evaluate_against_reference_set(triples, gold_triples)
        st.session_state["exp7_last_eval"] = eval_result

        st.divider()

        # Metrics Banner
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Entities Identified", metrics["entities"])
        m2.metric("Triples Extracted", metrics["triples"])
        m3.metric("High-Confidence", metrics["high_confidence_triples"])
        m4.metric("Avg. Confidence", f"{metrics['avg_confidence']:.2f}")
        m5.metric("Rel. Density", metrics["relation_density"])
        if eval_result.get("has_gold"):
            m6.metric("F1 Score", f"{eval_result['f1']:.2f}", delta=f"P:{eval_result['precision']:.2f} R:{eval_result['recall']:.2f}")
        else:
            m6.metric("Unique Preds", metrics["unique_predicates"])

        # 6 Comprehensive Analytical Tabs
        tab_triples, tab_annotated, tab_entities, tab_graph, tab_flow, tab_eval = st.tabs([
            "Extracted Triples (SPO)",
            "Annotated Text (Evidence)",
            "Detected Entities",
            "Knowledge Graph Preview",
            "Entity-Relationship Flow",
            "Evaluation against Ground Truth"
        ])

        # TAB 1: TRIPLES
        with tab_triples:
            if triples:
                df_triples = pd.DataFrame(triples)[[
                    "subject", "predicate", "object", "confidence",
                    "subject_label", "object_label", "method", "cue", "sentence"
                ]]
                df_triples.columns = [
                    "Subject", "Predicate", "Object", "Confidence",
                    "Subj Type", "Obj Type", "Method", "Matched Cue", "Source Sentence"
                ]
                st.dataframe(df_triples, hide_index=True, use_container_width=True)
            else:
                st.warning("No triples met the confidence threshold. Try lowering the threshold or switching to Hybrid mode.")

        # TAB 2: ANNOTATED TEXT
        with tab_annotated:
            write_annotated(
                "**Sentence Evidence Inspection:** Each sentence displays identified "
                "[[entity|entities]] with type tags and matched [[relation_cue|relation cues]] "
                "highlighted - the direct textual [[provenance]] for every extracted triple."
            )
            sentences = _split_sentences(st.session_state.get("exp7_last_text", text_input))
            cues = [t["cue"] for t in triples if t.get("cue")]

            for s_idx, s_text in enumerate(sentences):
                s_ents = [e for e in entities if e["sentence_idx"] == s_idx]
                annotated_html = _annotate_sentence(s_text, s_ents, cues)
                st.markdown(f"**Sentence {s_idx + 1}:** {annotated_html}")

        # TAB 3: DETECTED ENTITIES
        with tab_entities:
            col_e1, col_e2 = st.columns([3, 2])
            with col_e1:
                if entities:
                    ent_df = pd.DataFrame(entities).drop_duplicates(subset=["text"])[["text", "label", "sentence_idx"]]
                    ent_df.columns = ["Entity Mention", "Classified Type", "First Sentence #"]
                    st.dataframe(ent_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No entities detected.")
            with col_e2:
                if entities:
                    st.write("**Entity Type Distribution**")
                    st.plotly_chart(_build_entity_type_chart(entities, theme), use_container_width=True)

        # TAB 4: KNOWLEDGE GRAPH PREVIEW
        with tab_graph:
            col_gc1, col_gc2, col_gc3 = st.columns([2, 3, 2])
            with col_gc1:
                show_labels = st.checkbox("Show predicate labels on edges", value=True, key="graph_show_preds")
            with col_gc2:
                selected_types = st.multiselect(
                    "Filter by Entity Type",
                    options=ENTITY_TYPE_ORDER,
                    default=ENTITY_TYPE_ORDER,
                    key="graph_type_filter"
                )
            with col_gc3:
                if st.button("Re-layout / Jiggle Graph", use_container_width=True):
                    st.session_state["graph_seed"] = st.session_state.get("graph_seed", 11) + 7
                    st.rerun()

            col_gd1, col_gd2 = st.columns([2, 3])
            with col_gd1:
                show_weak = st.checkbox(
                    "Show weak links (related_to)", value=False, key="graph_show_weak",
                    help="Weak links are co-occurrence edges with no explicit relation cue. Hidden by "
                         "default to keep the graph readable (ignored if there is nothing else to draw)."
                )
            with col_gd2:
                degree_count: Dict[str, int] = {}
                for t in triples:
                    for n in (t["subject"], t["object"]):
                        degree_count[n] = degree_count.get(n, 0) + 1
                focus_options = [ALL_ENTITIES] + sorted(degree_count, key=lambda n: (-degree_count[n], n))
                focus = st.selectbox("Focus on entity", focus_options, key="graph_focus",
                                     help="Show only one entity and the entities directly linked to it.")

            curr_seed = st.session_state.get("graph_seed", 11)
            _, graph_info = _prepare_graph_triples(triples, selected_types, show_weak, MAX_GRAPH_NODES, focus)
            graph_fig = _build_graph_figure(
                triples=triples,
                theme=theme,
                show_predicates=show_labels,
                seed=curr_seed,
                type_filter=selected_types,
                show_weak=show_weak,
                max_entities=MAX_GRAPH_NODES,
                focus=focus
            )
            st.plotly_chart(graph_fig, use_container_width=True)
            notes = []
            if graph_info["hidden_weak"]:
                notes.append(f"{graph_info['hidden_weak']} weak link(s) hidden")
            if graph_info["omitted_entities"]:
                notes.append(f"{graph_info['omitted_entities']} of {graph_info['total_entities']} "
                             "entities left out (pick one under 'Focus on entity' to explore them)")
            if notes:
                st.caption("Showing the clearest part of the graph: " + "; ".join(notes) + ".")
            st.caption("Tip: Hover over any node or edge to inspect full entity attributes, source sentences, and extraction cues.")

        # TAB 5: FLOW (SANKEY)
        with tab_flow:
            st.write("**Entity Type to Predicate Transition Patterns:**")
            st.plotly_chart(_build_flow_figure(triples, theme), use_container_width=True)
            write_annotated(
                "Traces how [[entity_type|entity types]] flow from subjects into "
                "[[predicate|predicates]] and target object types."
            )

        # TAB 6: EVALUATION
        with tab_eval:
            if eval_result.get("has_gold"):
                st.subheader(f"Ground Truth Evaluation Benchmark ({selected_corpus['name'] if selected_corpus else 'Corpus'})")
                write_annotated(
                    "Every extracted [[triple]] is checked against this corpus's "
                    "[[reference_set|hand-curated gold set]]: a match is a True Positive, an "
                    "extracted triple with no match is a [[false_positive|False Positive]], and "
                    "a gold triple nothing matched is a [[false_negative|False Negative]]."
                )
                col_ev1, col_ev2, col_ev3, col_ev4 = st.columns(4)
                col_ev1.metric("Precision (P)", f"{eval_result['precision'] * 100:.1f}%", help="Fraction of extracted triples that are factually correct.")
                col_ev2.metric("Recall (R)", f"{eval_result['recall'] * 100:.1f}%", help="Fraction of ground truth relations successfully recovered.")
                col_ev3.metric("F1 Score", f"{eval_result['f1'] * 100:.1f}%", help="Harmonic mean of precision and recall.")
                col_ev4.metric("Pair Recall", f"{eval_result['pair_recall'] * 100:.1f}%", help="Fraction of ground truth (subject, object) pairs found, ignoring the predicate label. Unlike the three metrics to the left, this gives credit to a method (e.g. Co-occurrence) that finds the right entities but not the right relation name.")
                if eval_result["precision"] == 0.0 and eval_result["f1"] == 0.0 and eval_result["pair_recall"] > 0.0:
                    st.caption(
                        f"Precision/Recall/F1 are 0% because this method requires an exact predicate "
                        f"match, and none of its predicted labels match the gold set's specific relation "
                        f"names. Pair Recall shows it still found {eval_result['pair_recall'] * 100:.0f}% of "
                        f"the correct entity connections - just without naming the relationship correctly."
                    )

                st.write(f"**Summary:** True Positives: `{eval_result['tp_count']}` | False Positives: `{eval_result['fp_count']}` | False Negatives (Omissions): `{eval_result['fn_count']}`")

                col_tp, col_fp, col_fn = st.columns(3)
                with col_tp:
                    st.success(f"**True Positives ({eval_result['tp_count']})**")
                    if eval_result["tp_list"]:
                        st.dataframe(pd.DataFrame(eval_result["tp_list"])[["subject", "predicate", "object"]], hide_index=True)
                    else:
                        st.caption("None.")
                with col_fp:
                    st.warning(f"**False Positives ({eval_result['fp_count']})**")
                    if eval_result["fp_list"]:
                        st.dataframe(pd.DataFrame(eval_result["fp_list"])[["subject", "predicate", "object"]], hide_index=True)
                    else:
                        st.caption("None.")
                with col_fn:
                    st.error(f"**False Negatives ({eval_result['fn_count']})**")
                    if eval_result["fn_list"]:
                        st.dataframe(pd.DataFrame(eval_result["fn_list"], columns=["Subject", "Predicate", "Object"]), hide_index=True)
                    else:
                        st.caption("None.")
            else:
                st.info("Evaluation against ground truth is available for built-in benchmark corpora (e.g. Tech & Business, Science & Academia, Indian Industry, Healthcare). Custom and uploaded texts do not carry pre-annotated reference sets.")

        # 3. Export Card
        st.divider()
        st.subheader("Export for Downstream Graph Ingestion (Experiments 8-9)")
        col_x1, col_x2, col_x3 = st.columns(3)

        exp_df = pd.DataFrame(triples) if triples else pd.DataFrame(columns=["subject", "predicate", "object", "confidence"])
        with col_x1:
            st.download_button(
                "Download Triples (CSV)",
                data=exp_df.to_csv(index=False).encode("utf-8"),
                file_name="exp7_extracted_triples.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_x2:
            st.download_button(
                "Download Triples (JSON)",
                data=json.dumps(triples, indent=2).encode("utf-8"),
                file_name="exp7_extracted_triples.json",
                mime="application/json",
                use_container_width=True
            )
        with col_x3:
            kg_export = nx.MultiDiGraph()
            for t in triples:
                kg_export.add_node(t["subject"], type=t["subject_label"])
                kg_export.add_node(t["object"], type=t["object_label"])
                kg_export.add_edge(t["subject"], t["object"], predicate=t["predicate"],
                                   confidence=float(t["confidence"]))
            graphml_buffer = io.BytesIO()
            nx.write_graphml(kg_export, graphml_buffer)
            st.download_button(
                "Download Graph (GraphML)",
                data=graphml_buffer.getvalue(),
                file_name="exp7_knowledge_graph.graphml",
                mime="application/graphml+xml",
                use_container_width=True
            )

        # 4. Trial Logging
        st.divider()
        st.subheader("Record Current Trial to Session Logbook")
        if st.button("Record Trial to Logbook", use_container_width=True):
            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Document": doc_choice,
                "Method": method,
                "Min Conf": min_confidence,
                "Max Dist": max_dist,
                "Entities": metrics["entities"],
                "Triples": metrics["triples"],
                "High Conf": metrics["high_confidence_triples"],
                "Avg Conf": metrics["avg_confidence"],
                "F1 Score": eval_result.get("f1", "N/A") if eval_result.get("has_gold") else "N/A"
            }
            st.session_state["trials"].append(trial_record)
            st.success(f"Trial #{trial_record['Trial #']} successfully recorded into session logbook.")

    # Display Trials Table if any recorded
    if st.session_state.get("trials"):
        st.divider()
        st.subheader("Recorded Session Trials Logbook")
        st.dataframe(pd.DataFrame(st.session_state["trials"]), hide_index=True, use_container_width=True)
        if st.button("Clear Session Trials Logbook"):
            st.session_state["trials"] = []
            st.rerun()
