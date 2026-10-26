"""
SentimentScope AI — Home / Overview Page
"""

import streamlit as st
import pandas as pd

from src.sentimentscope.analytics.metrics import compute_sentiment_distribution
from src.sentimentscope.analytics.visualizations import plot_sentiment_bars, plot_sentiment_donut
from src.sentimentscope.config import SENTIMENT_COLOR_MAP
from src.sentimentscope.ui.components import render_hero, render_metric_card, render_sentiment_badge


def render_home_page():
    render_hero(
        title="SentimentScope AI",
        subtitle="Intelligent Social Media Sentiment Analysis & Trend Monitoring System",
        badge_text="Ready for Deployment",
    )

    df: pd.DataFrame = st.session_state.get("active_df", pd.DataFrame())
    model_artifact = st.session_state.get("model_artifact")
    dataset_name = st.session_state.get("dataset_name", "Built-In Sample Dataset")

    # Status Cards Row
    st.markdown("### 📊 System Status & Active Dataset")
    c1, c2, c3, c4 = st.columns(4)

    total_rows = len(df) if not df.empty else 0
    with c1:
        render_metric_card(
            label="Active Dataset",
            value=f"{total_rows:,}",
            sub_text=f"Records from {dataset_name}",
            border_accent="#6366F1",
        )

    dist = compute_sentiment_distribution(df, "Sentiment")
    counts = dist["counts"]
    pcts = dist["percentages"]

    with c2:
        render_metric_card(
            label="Positive Posts",
            value=f"{counts.get('Positive', 0):,}",
            sub_text=f"{pcts.get('Positive', 0.0):.1f}% of total",
            border_accent=SENTIMENT_COLOR_MAP["Positive"],
        )

    with c3:
        render_metric_card(
            label="Neutral Posts",
            value=f"{counts.get('Neutral', 0):,}",
            sub_text=f"{pcts.get('Neutral', 0.0):.1f}% of total",
            border_accent=SENTIMENT_COLOR_MAP["Neutral"],
        )

    with c4:
        render_metric_card(
            label="Negative Posts",
            value=f"{counts.get('Negative', 0):,}",
            sub_text=f"{pcts.get('Negative', 0.0):.1f}% of total",
            border_accent=SENTIMENT_COLOR_MAP["Negative"],
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Overview Row
    st.markdown("### 📈 Sentiment Snapshot")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        fig_donut = plot_sentiment_donut(counts, title="Sentiment Proportions")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        fig_bar = plot_sentiment_bars(counts, title="Sentiment Volumes")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Active Model Information Card
    st.markdown("### 🤖 Production Machine Learning Model")
    if model_artifact:
        metrics = model_artifact.get("metrics", {})
        m_name = model_artifact.get("model_name", "Sentiment Classifier")
        acc = metrics.get("accuracy", 0.0) * 100
        f1 = metrics.get("macro_f1", 0.0)
        dur = metrics.get("training_duration_seconds", 0.0)

        st.info(
            f"**Active Model:** `{m_name}` &nbsp;|&nbsp; "
            f"**Held-out Test Accuracy:** `{acc:.2f}%` &nbsp;|&nbsp; "
            f"**Macro F1:** `{f1:.4f}` &nbsp;|&nbsp; "
            f"**Training Time:** `{dur:.2f}s` &nbsp;|&nbsp; "
            f"**Feature Representation:** `{model_artifact.get('feature_type', 'tfidf').upper()}`"
        )
    else:
        st.warning(
            "⚠️ No production model artifact loaded. Run `python scripts/train_model.py` or use the ML Lab tab to initialize the model."
        )

    # Quick Navigation Call-To-Action
    st.markdown("### 🚀 Quick Start Actions")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="margin: 0; color: #818CF8;">1. Test Single Prediction</h4>
                <p style="color: #94A3B8; font-size: 13px; margin: 8px 0 0 0;">
                    Type or paste any social media post to test real-time sentiment inference with token explainability.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with q2:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="margin: 0; color: #34D399;">2. Upload Custom CSV</h4>
                <p style="color: #94A3B8; font-size: 13px; margin: 8px 0 0 0;">
                    Upload custom datasets with automatic column mapping, duplicate removal, and quality profiling.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with q3:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="margin: 0; color: #F472B6;">3. Explore ML Lab</h4>
                <p style="color: #94A3B8; font-size: 13px; margin: 8px 0 0 0;">
                    Inspect the 4-algorithm benchmark leaderboard, confusion matrices, and ROC curves.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
