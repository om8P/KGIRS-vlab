"""Session-state initialisation and the main navigation."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from exp7.content import EXPERIMENT_CONFIG
from exp7.popups import inject_wiki_term_style
from exp7.quiz_bank import QUIZ_QUESTIONS
from exp7.sections.quiz import render_quiz_section
from exp7.sections.report import render_report_section
from exp7.sections.simulation import render_simulation_section
from exp7.sections.theory import render_theory_section


def init_session_state():
    """Initializes session state keys."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP7-001",
            "batch": "D17A",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""
    if "kg_entities" not in st.session_state:
        st.session_state["kg_entities"] = []
    if "kg_triples" not in st.session_state:
        st.session_state["kg_triples"] = []
    if "kg_source_text" not in st.session_state:
        st.session_state["kg_source_text"] = ""
    if "graph_seed" not in st.session_state:
        st.session_state["graph_seed"] = 11


def main():
    st.set_page_config(
        page_title="Exp 7: Relationship Extraction | KGIRS Virtual Lab",
        page_icon=None,
        layout="wide"
    )

    init_session_state()
    inject_wiki_term_style()

    # Native Streamlit Title (No custom CSS, aside from the term pop-up stylesheet above).
    # Light/Dark mode is switched via Streamlit's own top-right "..." menu -> Settings.
    st.title(EXPERIMENT_CONFIG["title"])
    st.caption(f"{EXPERIMENT_CONFIG['department']} | {EXPERIMENT_CONFIG['course']}")

    # Navigation Sidebar
    section = st.sidebar.radio(
        "Lab Navigator",
        options=["Theory", "Simulation", "Quiz", "Report Generation"]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Progress Tracker")
    quiz_status = "Done" if st.session_state.get("quiz_submitted", False) else "Pending"
    st.sidebar.write(f"- **Quiz Status:** {quiz_status}")
    if st.session_state.get("quiz_submitted", False):
        st.sidebar.write(f"- **Quiz Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}`")
    st.sidebar.write(f"- **Trials Logged:** `{len(st.session_state['trials'])}`")
    st.sidebar.write(f"- **Active Triples in Session:** `{len(st.session_state.get('kg_triples', []))}`")

    st.sidebar.divider()
    st.sidebar.caption(
        "Knowledge graph triples emitted here populate `st.session_state['kg_triples']` "
        "for Experiments 8 and 9 (Graph Database Ingestion & Import)."
    )

    # Section Dispatcher
    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()
