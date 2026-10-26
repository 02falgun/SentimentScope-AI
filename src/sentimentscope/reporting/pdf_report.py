"""
SentimentScope AI — PDF Report Generation Service
Generates structured executive analytical summaries using fpdf2 adhering to the
monochrome SaaS design system (pure black-and-white rules and typography).
"""

from datetime import datetime, timezone
import io
from typing import Any, Dict, List, Optional
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import pandas as pd

from src.sentimentscope.config import CANONICAL_SENTIMENTS


class SentimentReportPDF(FPDF):
    """Custom FPDF2 subclass with monochrome executive headers, footers, and section styles."""

    def header(self):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(17, 17, 17)
        self.cell(0, 8, "SentimentScope AI - Executive Analytics & Model Audit Report", border=False, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.set_draw_color(17, 17, 17)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(138, 138, 138)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | SentimentScope AI Laboratory Project | Production Release v2.0", border=False, align="C")

    def section_title(self, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(17, 17, 17)
        self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(51, 51, 51)
        self.ln(1)


def safe_text(val: Any) -> str:
    """Sanitizes text to latin-1 compatible ASCII representation for standard PDF fonts."""
    if val is None:
        return ""
    s = str(val).replace("—", "-").replace("–", "-").replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return s.encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(
    dataset_name: str,
    total_records: int,
    sentiment_dist: Dict[str, Any],
    model_artifact: Optional[Dict[str, Any]] = None,
    platform_breakdown: Optional[pd.DataFrame] = None,
    top_hashtags: Optional[List[tuple]] = None,
) -> bytes:
    """
    Builds and renders a downloadable binary PDF report in monochrome executive format.
    """
    pdf = SentimentReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Executive Summary & Metadata
    pdf.section_title("1. Executive Summary & Dataset Provenance")
    gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    meta_text = (
        f"Generated: {gen_time}\n"
        f"Target Dataset: {safe_text(dataset_name)}\n"
        f"Total Records Ingested: {total_records:,}\n"
        "Classification Engine: Scikit-learn Pipeline (Negation-Preserving TF-IDF Vectorization)"
    )
    pdf.multi_cell(0, 5, safe_text(meta_text))
    pdf.ln(3)

    # 2. Sentiment Distribution Table
    pdf.section_title("2. Aggregate Sentiment Distribution")
    counts = sentiment_dist.get("counts", {})
    percentages = sentiment_dist.get("percentages", {})

    pdf.set_draw_color(229, 229, 229)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(17, 17, 17)
    pdf.cell(50, 7, "Sentiment Category", border=1, fill=True)
    pdf.cell(40, 7, "Post Count", border=1, fill=True, align="R")
    pdf.cell(40, 7, "Percentage Share", border=1, fill=True, align="R")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 51, 51)
    for cls in CANONICAL_SENTIMENTS:
        cnt = counts.get(cls, 0)
        pct = percentages.get(cls, 0.0)
        pdf.cell(50, 6, cls, border=1)
        pdf.cell(40, 6, f"{cnt:,}", border=1, align="R")
        pdf.cell(40, 6, f"{pct:.1f}%", border=1, align="R")
        pdf.ln()

    pdf.ln(3)

    # 3. Model Performance & Evaluation Metrics
    pdf.section_title("3. Machine Learning Model Performance (Held-Out Test Data)")
    if model_artifact:
        model_name = model_artifact.get("model_name", "Production Classifier")
        metrics = model_artifact.get("metrics", {})
        acc = metrics.get("accuracy", 0.0) * 100
        mf1 = metrics.get("macro_f1", 0.0)
        wf1 = metrics.get("weighted_f1", 0.0)

        ml_text = (
            f"Active Model: {safe_text(model_name)}\n"
            f"Held-Out Test Accuracy: {acc:.2f}% (PRD Baseline: >= 80.00%)\n"
            f"Macro F1-Score: {mf1:.4f} (PRD Baseline: >= 0.7800)\n"
            f"Weighted F1-Score: {wf1:.4f}\n"
            "Evaluation Protocol: Stratified 80/20 held-out split with fixed random seed."
        )
        pdf.multi_cell(0, 5, safe_text(ml_text))
    else:
        pdf.multi_cell(0, 5, "Model metrics not currently loaded into active session.")

    pdf.ln(3)

    # 4. Cross-Platform Breakdown Table
    if platform_breakdown is not None and not platform_breakdown.empty:
        pdf.section_title("4. Cross-Platform Sentiment Breakdown")
        pdf.set_fill_color(245, 245, 245)
        pdf.set_draw_color(229, 229, 229)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(17, 17, 17)
        pdf.cell(45, 7, "Platform", border=1, fill=True)
        pdf.cell(30, 7, "Positive", border=1, fill=True, align="R")
        pdf.cell(30, 7, "Neutral", border=1, fill=True, align="R")
        pdf.cell(30, 7, "Negative", border=1, fill=True, align="R")
        pdf.cell(30, 7, "Total", border=1, fill=True, align="R")
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(51, 51, 51)
        for _, row in platform_breakdown.iterrows():
            plat_label = safe_text(str(row.get("Platform", "Other")))
            pdf.cell(45, 6, plat_label, border=1)
            pdf.cell(30, 6, str(row.get("Positive", 0)), border=1, align="R")
            pdf.cell(30, 6, str(row.get("Neutral", 0)), border=1, align="R")
            pdf.cell(30, 6, str(row.get("Negative", 0)), border=1, align="R")
            pdf.cell(30, 6, str(row.get("Total", 0)), border=1, align="R")
            pdf.ln()

        pdf.ln(3)

    # 5. Trending Hashtags
    if top_hashtags:
        pdf.section_title("5. Trending Hashtags")
        tag_str = ", ".join([f"{safe_text(tag)} ({cnt})" for tag, cnt in top_hashtags[:8]])
        pdf.multi_cell(0, 5, safe_text(f"Most frequent hashtags identified in post text:\n{tag_str}"))
        pdf.ln(3)

    # 6. Methodology & Limitations Notice
    pdf.section_title("6. Methodology & Limitations Notice")
    disclaimer = (
        "SentimentScope AI utilizes classical natural language processing models (TF-IDF with scikit-learn). "
        "Feature attributions indicate statistical word associations in training data, not causal sentiment intent. "
        "Predictions may reflect subtle domain vocabulary shifts, sarcasm, or colloquial expressions not present in training. "
        "This system operates on ephemeral user uploads without persisting personal identifiers."
    )
    pdf.multi_cell(0, 5, safe_text(disclaimer))

    out = pdf.output()
    return bytes(out) if isinstance(out, bytearray) else out
