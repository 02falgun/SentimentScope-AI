"""
SentimentScope AI — Executive Reports & Data Export Workspace
Compiles structured analytical summaries, downloads executive PDF audit briefs,
and exports processed datasets with model inference labels.
"""

import streamlit as st
import pandas as pd

from src.sentimentscope.analytics.metrics import compute_hashtag_and_emoji_stats, compute_sentiment_distribution
from src.sentimentscope.analytics.trends import compute_platform_sentiment_breakdown
from src.sentimentscope.reporting.pdf_report import generate_pdf_report
from src.sentimentscope.ui.components import (
    render_header,
    render_metric_card,
)


def render_reports_page():
    render_header(
        title="Executive Reports & Export",
        subtitle="Generate Structured Audit Summaries, Download Monochrome PDFs, and Export Processed Datasets",
        badge_text="Automated PDF Engine",
        category_label="REPORTS & EXPORT",
    )

    df: pd.DataFrame = st.session_state.get("active_df", pd.DataFrame())
    model_artifact = st.session_state.get("model_artifact")
    dataset_name = st.session_state.get("dataset_name", "Built-In Sample Dataset")

    if df.empty:
        st.warning("No dataset is currently active. Please load or upload a dataset first.")
        return

    sentiment_col = "Sentiment" if "Sentiment" in df.columns else ("Predicted_Sentiment" if "Predicted_Sentiment" in df.columns else "Sentiment")
    dist = compute_sentiment_distribution(df, sentiment_col=sentiment_col)
    plat_df = compute_platform_sentiment_breakdown(df, platform_col="Platform", sentiment_col=sentiment_col)
    social_stats = compute_hashtag_and_emoji_stats(df, text_col="Text")

    # 1. Executive Summary Preview Cards
    st.markdown("### Executive Summary Preview")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Active Corpus", dataset_name, f"{len(df):,} total records analyzed", "#111111")
    with c2:
        top_sentiment = max(dist["counts"], key=dist["counts"].get) if dist["counts"] else "Neutral"
        render_metric_card("Dominant Sentiment", top_sentiment, f"{dist['percentages'].get(top_sentiment, 0):.1f}% of total volume", "#111111")
    with c3:
        m_name = model_artifact.get("model_name", "Production Classifier") if model_artifact else "No Model Loaded"
        render_metric_card("Inference Engine", m_name, "Scikit-Learn NLP Pipeline", "#111111")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 2. Export Formats Grid
    st.markdown("### Export Formats")
    col_pdf, col_csv = st.columns(2)

    with col_pdf:
        st.markdown(
            """
            <div class="action-card" style="margin-bottom: 14px;">
                <div style="font-size: 11px; font-weight: 600; color: #8A8A8A; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                    Executive Document
                </div>
                <h4>Executive PDF Audit Report</h4>
                <p>
                    Includes narrative summary, dataset quality metrics, class distribution tables,
                    held-out evaluation benchmarks, platform breakdown, and methodology caveats in monochrome format.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("Compiling monochrome PDF report..."):
            pdf_bytes = generate_pdf_report(
                dataset_name=dataset_name,
                total_records=len(df),
                sentiment_dist=dist,
                model_artifact=model_artifact,
                platform_breakdown=plat_df,
                top_hashtags=social_stats.get("hashtags", []),
            )

        clean_slug = dataset_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        st.download_button(
            label="Download Executive Report (PDF)",
            data=pdf_bytes,
            file_name=f"sentimentscope_report_{clean_slug}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with col_csv:
        st.markdown(
            """
            <div class="action-card" style="margin-bottom: 14px;">
                <div style="font-size: 11px; font-weight: 600; color: #8A8A8A; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                    Structured Data
                </div>
                <h4>Cleaned Dataset & Inferences (CSV)</h4>
                <p>
                    Exports active processed posts, cleaned text representations, and all computed
                    model predicted sentiment classes and calibrated confidence scores.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Analyzed Posts (CSV)",
            data=csv_bytes,
            file_name=f"sentimentscope_data_{clean_slug}.csv",
            mime="text/csv",
            use_container_width=True,
        )
