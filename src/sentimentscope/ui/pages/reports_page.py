"""
SentimentScope AI — Executive Reports & Export Page
"""

import streamlit as st
import pandas as pd

from src.sentimentscope.analytics.metrics import compute_hashtag_and_emoji_stats, compute_sentiment_distribution
from src.sentimentscope.analytics.trends import compute_platform_sentiment_breakdown
from src.sentimentscope.reporting.pdf_report import generate_pdf_report
from src.sentimentscope.ui.components import render_hero, render_metric_card


def render_reports_page():
    render_hero(
        title="Executive Reports & Export",
        subtitle="Generate Structured Audit Summaries, Download PDFs, and Export Analysis Data",
        badge_text="Automated PDF Engine",
    )

    df: pd.DataFrame = st.session_state.get("active_df", pd.DataFrame())
    model_artifact = st.session_state.get("model_artifact")
    dataset_name = st.session_state.get("dataset_name", "Built-In Sample Dataset")

    if df.empty:
        st.warning("No dataset is currently active. Please load or upload data first.")
        return

    sentiment_col = "Sentiment" if "Sentiment" in df.columns else ("Predicted_Sentiment" if "Predicted_Sentiment" in df.columns else "Sentiment")
    dist = compute_sentiment_distribution(df, sentiment_col=sentiment_col)
    plat_df = compute_platform_sentiment_breakdown(df, platform_col="Platform", sentiment_col=sentiment_col)
    social_stats = compute_hashtag_and_emoji_stats(df, text_col="Text")

    # 1. Report Content Preview
    st.markdown("### 📋 Executive Summary Preview")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Dataset", dataset_name, f"{len(df):,} total records analyzed")
    with c2:
        top_sentiment = max(dist["counts"], key=dist["counts"].get) if dist["counts"] else "Neutral"
        render_metric_card("Dominant Sentiment", top_sentiment, f"{dist['percentages'].get(top_sentiment, 0):.1f}% of total volume")
    with c3:
        m_name = model_artifact.get("model_name", "Production Classifier") if model_artifact else "No Model Loaded"
        render_metric_card("Inference Engine", m_name, "Classical NLP Pipeline")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. PDF Report Generator Section
    st.markdown("### 📄 Export Formats")
    col_pdf, col_csv = st.columns(2)

    with col_pdf:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="margin: 0; color: #818CF8;">Executive PDF Report</h4>
                <p style="color: #94A3B8; font-size: 13px; margin: 8px 0 16px 0;">
                    Includes executive narrative, dataset quality metrics, class distribution tables, model held-out evaluation benchmarks, platform breakdown, and methodology caveats.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("Compiling PDF report document..."):
            pdf_bytes = generate_pdf_report(
                dataset_name=dataset_name,
                total_records=len(df),
                sentiment_dist=dist,
                model_artifact=model_artifact,
                platform_breakdown=plat_df,
                top_hashtags=social_stats.get("hashtags", []),
            )

        st.download_button(
            label="📥 Download Executive Report (PDF)",
            data=pdf_bytes,
            file_name=f"sentimentscope_report_{dataset_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

    with col_csv:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="margin: 0; color: #34D399;">Cleaned Dataset & Inferences (CSV)</h4>
                <p style="color: #94A3B8; font-size: 13px; margin: 8px 0 16px 0;">
                    Download the active processed posts, cleaned text representations, and all computed model prediction labels and confidence scores.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Analyzed Posts (CSV)",
            data=csv_bytes,
            file_name=f"sentimentscope_data_{dataset_name.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
