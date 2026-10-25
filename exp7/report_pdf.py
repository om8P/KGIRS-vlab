"""PDF laboratory report generator."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from fpdf import FPDF

from exp7.content import EXPERIMENT_CONFIG


class LabReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "VIRTUAL LABORATORY PORTAL | KNOWLEDGE GRAPHS & INFORMATION RETRIEVAL SYSTEMS", align="R")
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Academic Virtual Lab Report | Exp-07 Relationship Extraction", align="C")


# A condensed, plain-text (no markdown/[[term]] markers) version of the Theory section's
# background, for the PDF report's own "Theoretical Background" section.
PDF_THEORY_SUMMARY = (
    "Relationship Extraction (RE) identifies and classifies semantic relationships holding "
    "between pairs of entities mentioned in unstructured text, expressing each as a "
    "Subject-Predicate-Object (SPO) triple - the atomic unit from which a Knowledge Graph is "
    "built: entities become nodes, and predicates become directed, labelled edges between them. "
    "This laboratory implements and compares six extraction strategies; four are "
    "rule-based and need no external NLP model, one uses a grammar parser and one a pretrained BERT model. Pattern-Based extraction emits a triple only when "
    "the text span between two adjacent entities matches an ordered lexicon of relation-cue "
    "regular expressions (e.g. 'acquired', 'is the CEO of'), giving high precision at the cost "
    "of recall. Co-occurrence extraction links every adjacent entity pair with a generic "
    "'related_to' predicate whose confidence decays with word distance, giving high recall at "
    "the cost of precision. Hybrid extraction applies the Pattern-Based check first and falls "
    "back to a Co-occurrence edge only when no cue matches, so no candidate pair is ever "
    "dropped outright. Nearest-Verb drops the fixed lexicon and uses the verb between two "
    "entities as the predicate, and Dependency Parse (SVO, using the optional spaCy parser) reads "
    "the sentence grammar to find who did what to whom, including in the passive voice, and BERT Similarity (Sentence-BERT) judges the meaning of the text between two entities with a classifier trained on BERT embeddings. Every emitted triple carries a numeric confidence score and its source "
    "sentence as provenance, so downstream stages (Experiments 8-9: graph database construction "
    "and schema import) can filter or audit it before ingestion."
)


# (Method, Precision, Recall, Characteristic) - the same qualitative comparison shown in the
# Theory tab's method-comparison table, condensed for the PDF.
PDF_METHOD_COMPARISON_ROWS = [
    ("Co-occurrence", "Low", "Highest", "Confidence decays with word distance; no cue required"),
    ("Pattern-Based", "High", "Lower", "Every edge traces to an explicit cue word in the text"),
    ("Hybrid", "Medium-High", "High", "Pattern-Based first, Co-occurrence fallback if no cue matches"),
    ("Nearest-Verb", "Medium", "Medium", "Nearest verb between entities becomes the predicate; no word list"),
    ("Dependency Parse", "High", "High", "Grammatical roles (subject/object) decide the triple; handles passive voice"),
    ("BERT Similarity", "High", "Medium-High", "A classifier on BERT embeddings judges the clause's meaning; no word list, but a black box"),
]


def generate_pdf_report(
    student_name: str,
    student_id: str,
    student_batch: str,
    date_str: str,
    trials_df: pd.DataFrame,
    quiz_score: int,
    quiz_total: int,
    student_notes: str,
    triples_df: pd.DataFrame,
    eval_metrics: Optional[Dict[str, Any]] = None
) -> bytes:
    """Generates an official institutional laboratory report PDF."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Title Banner
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, EXPERIMENT_CONFIG["title"], align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 5, f"{EXPERIMENT_CONFIG['course']} | {EXPERIMENT_CONFIG['department']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Student Credentials Card
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 28, 190, 24, "FD")

    pdf.set_xy(14, 30)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(30, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, student_name or "N/A", 0)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Roll No / Student ID:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(55, 5, student_id or "N/A", 1)

    pdf.set_xy(14, 38)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(30, 5, "Batch / Class:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, student_batch or "D17", 0)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Assessment:", 0)
    pdf.set_font("Helvetica", "B", 8)
    pct = int((quiz_score / quiz_total) * 100) if quiz_total else 0
    if pct >= 50:
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(55, 5, f"{quiz_score} / {quiz_total} ({pct}%)", 1)

    pdf.set_xy(14, 45)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(30, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, date_str or datetime.now().strftime("%Y-%m-%d"), 0)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Lab Status:", 0)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(55, 5, "Verified & Evaluated", 1)

    pdf.set_xy(10, 56)

    # 1. Objectives
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "1. Educational Aim & Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        pdf.cell(4, 4, "-", 0)
        pdf.cell(0, 4, f" {obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # 2. Theoretical Background
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "2. Theoretical Background", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.3, PDF_THEORY_SUMMARY)
    pdf.ln(1.5)

    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 41, 59)
    pdf.cell(45, 5, "Method", 1, 0, "L", True)
    pdf.cell(35, 5, "Precision", 1, 0, "C", True)
    pdf.cell(35, 5, "Recall", 1, 0, "C", True)
    pdf.cell(75, 5, "Characteristic", 1, 1, "L", True)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(30, 41, 59)
    fill = False
    for method_name, precision, recall, characteristic in PDF_METHOD_COMPARISON_ROWS:
        pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(45, 4.5, method_name, 1, 0, "L", fill)
        pdf.cell(35, 4.5, precision, 1, 0, "C", fill)
        pdf.cell(35, 4.5, recall, 1, 0, "C", fill)
        pdf.cell(75, 4.5, characteristic, 1, 1, "L", fill)
        fill = not fill
    pdf.ln(3)

    # 3. Evaluation Summary (if available)
    if eval_metrics and eval_metrics.get("has_gold"):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 6, "3. Ground Truth Benchmark Evaluation", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(45, 5, f"Precision: {eval_metrics['precision']:.3f}", 1, 0, "C")
        pdf.cell(45, 5, f"Recall: {eval_metrics['recall']:.3f}", 1, 0, "C")
        pdf.cell(45, 5, f"F1 Score: {eval_metrics['f1']:.3f}", 1, 0, "C")
        pdf.cell(55, 5, f"True Positives: {eval_metrics['tp_count']} / {eval_metrics['gold_count']}", 1, 1, "C")
        pdf.ln(3)

    # 4. Recorded Experimental Trials
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "4. Recorded Experimental Trials (Session Logbook)", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, "No experimental trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)
        cols = list(trials_df.columns)
        col_w = max(18, int(190 / max(1, len(cols))))
        for c in cols:
            pdf.cell(col_w, 5, str(c)[:14], 1, 0, "C", True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 7)
        fill = False
        for _, row in trials_df.iterrows():
            for c in cols:
                val = row[c]
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 4.5, val_str[:15], 1, 0, "C", fill)
            pdf.ln()
            fill = not fill
    pdf.ln(3)

    # 4. Extracted Triples Sample
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "5. Extracted Subject-Predicate-Object Knowledge Triples", new_x="LMARGIN", new_y="NEXT")

    if triples_df.empty:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, "No triples extracted in the active session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(50, 5, "Subject", 1, 0, "L", True)
        pdf.cell(45, 5, "Predicate", 1, 0, "L", True)
        pdf.cell(50, 5, "Object", 1, 0, "L", True)
        pdf.cell(20, 5, "Conf.", 1, 0, "C", True)
        pdf.cell(25, 5, "Method", 1, 1, "C", True)

        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(30, 41, 59)
        fill = False
        for _, row in triples_df.head(15).iterrows():
            # A bare ternary-as-statement here (not assigned to anything) is picked up by
            # Streamlit's "magic" auto-display and rendered as a stray `None` in the app,
            # since streamlit run auto-wraps unassigned top-level expressions in st.write().
            if fill:
                pdf.set_fill_color(248, 250, 252)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.cell(50, 4.5, str(row.get("subject", ""))[:26], 1, 0, "L", fill)
            pdf.cell(45, 4.5, str(row.get("predicate", ""))[:24], 1, 0, "L", fill)
            pdf.cell(50, 4.5, str(row.get("object", ""))[:26], 1, 0, "L", fill)
            pdf.cell(20, 4.5, f"{float(row.get('confidence', 0)):.2f}", 1, 0, "C", fill)
            pdf.cell(25, 4.5, str(row.get("method", "Pattern"))[:12], 1, 1, "C", fill)
            fill = not fill
    pdf.ln(3)

    # 5. Student Conclusions & Observations
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "6. Student Analytical Discussion & Conclusions", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.5, student_notes or "The student did not record analytical notes.")
    pdf.ln(4)

    # Signature Block
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(95, 6, "Student Signature: _______________________", 0, 0, "L")
    pdf.cell(95, 6, "Faculty Evaluator Signature: _______________________", 0, 1, "R")

    return bytes(pdf.output())
