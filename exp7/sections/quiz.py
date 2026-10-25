"""The Quiz section."""

from __future__ import annotations

import streamlit as st

from exp7.quiz_bank import QUIZ_QUESTIONS


def render_quiz_section():
    """Renders Section 3: Self-grading Conceptual Assessment Quiz."""
    st.header("Conceptual Assessment Quiz")
    st.write("Complete all questions to assess your understanding of Information Retrieval and Relationship Extraction principles.")

    user_responses = {}
    with st.form("quiz_form"):
        for q in QUIZ_QUESTIONS:
            st.subheader(f"Question {q['id']}")
            st.write(q["question"])
            key = f"quiz_q_{q['id']}"
            choice = st.radio(
                f"Options for Question {q['id']}:",
                options=q["options"],
                index=None,
                key=key,
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = choice
            st.markdown("---")

        submit_btn = st.form_submit_button("Submit Assessment for Evaluation", type="primary", use_container_width=True)

    if submit_btn:
        score = 0
        unanswered = 0
        feedback = []

        for q in QUIZ_QUESTIONS:
            selected = user_responses.get(q["id"])
            if selected is None:
                unanswered += 1
                feedback.append((q, False, "Question not answered."))
            else:
                sel_idx = q["options"].index(selected)
                is_correct = (sel_idx == q["answer_index"])
                if is_correct:
                    score += 1
                feedback.append((q, is_correct, selected))

        st.session_state["quiz_submitted"] = True
        st.session_state["quiz_score"] = score
        st.session_state["quiz_feedback"] = feedback

        pct = int((score / len(QUIZ_QUESTIONS)) * 100)
        if pct >= 80:
            st.balloons()
            st.success(f"Excellent! Your Score: **{score} / {len(QUIZ_QUESTIONS)} ({pct}%)**")
        elif pct >= 50:
            st.success(f"Good Effort! Your Score: **{score} / {len(QUIZ_QUESTIONS)} ({pct}%)**")
        else:
            st.error(f"Needs Review. Your Score: **{score} / {len(QUIZ_QUESTIONS)} ({pct}%)**")

        st.subheader("Detailed Pedagogical Review & Explanations")
        incorrect = len(feedback) - score - unanswered
        st.markdown(
            f"**Result breakdown:** :green[{score} Correct]  |  :red[{incorrect} Incorrect]"
            + (f"  |  :orange[{unanswered} Unanswered]" if unanswered else "")
        )
        for q, is_corr, sel in feedback:
            status_label = "Correct" if is_corr else "Incorrect"
            with st.expander(f"Q{q['id']}: {q['question']} — [{status_label}]", expanded=not is_corr):
                if is_corr:
                    st.success(f"**Your Answer:** {sel}\n\nThis is correct.")
                else:
                    st.error(f"**Your Answer:** {sel}\n\nThis is incorrect.")
                    st.success(f"**Correct Answer:** {q['options'][q['answer_index']]}")
                st.info(f"**Pedagogical Explanation:** {q['explanation']}")

    elif st.session_state.get("quiz_submitted", False):
        st.info(f"Quiz previously submitted in this session. Score: **{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}**")
