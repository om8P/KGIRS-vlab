"""The Report Generation section."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from exp7.content import EXPERIMENT_CONFIG
from exp7.quiz_bank import QUIZ_QUESTIONS
from exp7.report_pdf import generate_pdf_report


def render_report_section():
    """Renders Section 4: Dynamic Official Laboratory Report Generation & PDF Download."""
    st.header("Official Laboratory Report Generation")
    st.write("Compile student details, session trials, extracted knowledge triples, and quiz results into a verified PDF report.")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        student_name = st.text_input("Student Full Name", value=st.session_state["student_info"].get("name", "Student Name"))
    with col_s2:
        student_id = st.text_input("Roll Number / ID", value=st.session_state["student_info"].get("id", "EXP7-001"))
    with col_s3:
        student_batch = st.text_input("Academic Batch", value=st.session_state["student_info"].get("batch", "D17A"))
    with col_s4:
        lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["batch"] = student_batch
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Student Analytical Observations & Conclusions")
    student_notes = st.text_area(
        "Enter your interpretation of results, comparisons across extraction methods, and conclusions:",
        value=st.session_state.get("student_notes", (
            "In this experiment, pattern-based relation extraction demonstrated high precision on active-voice "
            "corpora with direct verbal cues ('founded', 'acquired'). The co-occurrence baseline achieved higher "
            "recall but introduced false positives, underscoring the classic IR precision-recall trade-off. "
            "Confidence scoring successfully filtered noise prior to downstream graph construction."
        )),
        height=110
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state.get("trials", []))
    triples_df = pd.DataFrame(st.session_state.get("kg_triples", []))
    eval_metrics = st.session_state.get("exp7_last_eval")

    st.divider()
    st.subheader("Report Summary Preview")
    st.write(f"**Institution:** {EXPERIMENT_CONFIG['department']} | {EXPERIMENT_CONFIG['course']}")
    st.write(f"**Student:** {student_name} (Roll: `{student_id}`, Batch: `{student_batch}`) | **Date:** {lab_date}")
    st.write(f"**Assessment Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}` | **Recorded Trials:** `{len(trials_df)}`")

    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        student_batch=student_batch,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=student_notes,
        triples_df=triples_df,
        eval_metrics=eval_metrics
    )

    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Official Laboratory Report (.pdf)")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="Download Official lab_report.pdf",
            data=pdf_bytes,
            file_name=f"KGIRS_Exp7_Report_{student_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    with col_d2:
        st.link_button(
            "Open PDF Document in Browser Tab",
            url="/app/static/lab_report.pdf",
            use_container_width=True
        )
