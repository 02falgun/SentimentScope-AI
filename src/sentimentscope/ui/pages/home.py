"""
SentimentScope AI — Home / Overview Dashboard
Renders executive KPI metrics, sentiment distribution visuals, active model diagnostics,
and quick-action navigation adhering to the monochrome SaaS design system.
"""

import streamlit as st
import pandas as pd

from src.sentimentscope.analytics.metrics import compute_sentiment_distribution
from src.sentimentscope.analytics.visualizations import plot_sentiment_bars, plot_sentiment_donut
from src.sentimentscope.config import SENTIMENT_COLOR_MAP
from src.sentimentscope.ui.components import (
    render_action_card,
    render_header,
    render_metric_card,
    render_sentiment_badge,
)


def render_home_page():
    render_header(
        title="Sentiment Intelligence",
        subtitle="AI-Powered Social Media Sentiment Analysis & Trend Monitoring System",
        badge_text="Production v2.0",
        category_label="EXECUTIVE DASHBOARD",
    )

    df: pd.DataFrame = st.session_state.get("active_df", pd.DataFrame())
    model_artifact = st.session_state.get("model_artifact")
    dataset_name = st.session_state.get("dataset_name", "Built-In Sample Dataset")

    # Determine available sentiment sources
    has_ground_truth = not df.empty and "Sentiment" in df.columns
    has_predicted = not df.empty and "Predicted_Sentiment" in df.columns

    if has_ground_truth and has_predicted:
        source_choice = st.radio(
            "Sentiment Data Source:",
            options=["Ground-Truth Labels", "Model Inferences"],
            horizontal=True,
            help="Toggle between human-annotated ground-truth labels and model-generated inferences.",
        )
        sentiment_col = "Sentiment" if source_choice == "Ground-Truth Labels" else "Predicted_Sentiment"
        source_label = source_choice
    elif has_ground_truth:
        sentiment_col = "Sentiment"
        source_label = "Ground-Truth Labels"
    elif has_predicted:
        sentiment_col = "Predicted_Sentiment"
        source_label = "Model Inferences"
    else:
        sentiment_col = "Sentiment"
        source_label = "None Mapped"

    # Context Header Bar
    c_info, c_source = st.columns([3, 1])
    with c_info:
        st.markdown(
            f"""
            <div style="font-size: 15px; font-weight: 600; color: #111111; margin-bottom: 2px;">
                Active Corpus Intelligence
            </div>
            <div style="font-size: 13px; color: #666666; margin-bottom: 12px;">
                Displaying distribution for: <strong>{dataset_name}</strong> &nbsp;•&nbsp; Source: <span style="color: #111111; font-weight: 500;">{source_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 1. Primary KPI Metric Row
    c1, c2, c3, c4 = st.columns(4)
    total_rows = len(df) if not df.empty else 0

    dist = compute_sentiment_distribution(df, sentiment_col=sentiment_col) if not df.empty else {"counts": {}, "percentages": {}}
    counts = dist.get("counts", {})
    pcts = dist.get("percentages", {})

    with c1:
        render_metric_card(
            label="Total Analyzed Posts",
            value=f"{total_rows:,}",
            sub_text="Cleaned posts in active corpus",
            border_accent="#111111",
        )

    with c2:
        render_metric_card(
            label="Positive Sentiment",
            value=f"{counts.get('Positive', 0):,}",
            sub_text=f"{pcts.get('Positive', 0.0):.1f}% of total volume",
            border_accent=SENTIMENT_COLOR_MAP["Positive"],
        )

    with c3:
        render_metric_card(
            label="Neutral Sentiment",
            value=f"{counts.get('Neutral', 0):,}",
            sub_text=f"{pcts.get('Neutral', 0.0):.1f}% of total volume",
            border_accent=SENTIMENT_COLOR_MAP["Neutral"],
        )

    with c4:
        render_metric_card(
            label="Negative Sentiment",
            value=f"{counts.get('Negative', 0):,}",
            sub_text=f"{pcts.get('Negative', 0.0):.1f}% of total volume",
            border_accent=SENTIMENT_COLOR_MAP["Negative"],
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 2. Analytics Visualization Grid
    if total_rows == 0:
        st.info("No records found in active dataset. Ingest a dataset or reload the sample corpus in Dataset Explorer.")
    else:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            fig_donut = plot_sentiment_donut(counts, title=f"Sentiment Share ({source_label})")
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_right:
            fig_bar = plot_sentiment_bars(counts, title=f"Sentiment Volumes ({source_label})")
            st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Active Production ML Model Status Banner
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if model_artifact:
        metrics = model_artifact.get("metrics", {})
        m_name = model_artifact.get("model_name", "Sentiment Classifier")
        acc = metrics.get("accuracy", 0.0) * 100
        mf1 = metrics.get("macro_f1", 0.0)
        dur = metrics.get("training_duration_seconds", 0.0)
        feat = str(model_artifact.get("feature_type", "tfidf")).upper()

        st.markdown(
            f"""
            <div style="background-color: #FAFAFA; border: 1px solid #E5E5E5; border-radius: 8px; padding: 16px 20px; margin: 12px 0 24px 0;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #666666;">
                            Active Machine Learning Pipeline
                        </span>
                        <div style="font-size: 16px; font-weight: 700; color: #111111; margin-top: 2px;">
                            {m_name} ({feat})
                        </div>
                    </div>
                    <div style="display: flex; gap: 24px; font-size: 13px;">
                        <div>
                            <span style="color: #666666;">Held-Out Accuracy:</span>
                            <strong style="color: #111111; margin-left: 4px;">{acc:.2f}%</strong>
                        </div>
                        <div>
                            <span style="color: #666666;">Macro F1:</span>
                            <strong style="color: #111111; margin-left: 4px;">{mf1:.4f}</strong>
                        </div>
                        <div>
                            <span style="color: #666666;">Training Latency:</span>
                            <strong style="color: #111111; margin-left: 4px;">{dur:.2f}s</strong>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ No machine learning model currently loaded. Visit the ML Lab to benchmark and initialize.")

    # 4. Structured Workflow Call-To-Action Cards
    st.markdown("#### Primary Analytics Workflows")
    q1, q2, q3 = st.columns(3)
    with q1:
        render_action_card(
            number="01",
            title="Live Single Inference",
            description="Type or paste any customer review, tweet, or statement to test real-time inference with token-level feature attribution.",
        )
    with q2:
        render_action_card(
            number="02",
            title="Ingest Custom CSV",
            description="Upload proprietary datasets with automatic column schema mapping, deduplication, and row quality profiling.",
        )
    with q3:
        render_action_card(
            number="03",
            title="ML Lab & Generalization",
            description="Inspect the 4-algorithm benchmark leaderboard, Group-CV template-isolation audit, and confusion matrices.",
        )
