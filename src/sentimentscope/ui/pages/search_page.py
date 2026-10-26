"""
SentimentScope AI — Keyword Search & Topic Monitoring Page
"""

import streamlit as st
import pandas as pd

from src.sentimentscope.analytics.metrics import search_dataset_by_keyword
from src.sentimentscope.analytics.visualizations import plot_sentiment_donut
from src.sentimentscope.ui.components import render_hero, render_metric_card


def render_search_page():
    render_hero(
        title="Keyword Search & Opinion Mining",
        subtitle="Search Targeted Brands, Topics, or Terms Across the Active Dataset",
        badge_text="Polarity Scoring",
    )

    df: pd.DataFrame = st.session_state.get("active_df", pd.DataFrame())
    if df.empty:
        st.warning("No dataset is currently active. Please load or upload a dataset in the Explorer.")
        return

    sentiment_col = "Sentiment" if "Sentiment" in df.columns else ("Predicted_Sentiment" if "Predicted_Sentiment" in df.columns else None)

    col_input, col_action = st.columns([3, 1])
    with col_input:
        query = st.text_input(
            "Enter Topic, Keyword, or Hashtag to Track:",
            placeholder="e.g. update, performance, support, bug, battery, AI",
            value="update",
        )
    with col_action:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        search_clicked = st.button("🔎 Search Dataset", type="primary", use_container_width=True)

    if query.strip():
        search_res = search_dataset_by_keyword(df, keyword=query, text_col="Text", sentiment_col=sentiment_col or "Sentiment")

        total = search_res["total_matches"]
        avg_score = search_res["average_sentiment_score"]
        polarity_lbl = search_res["sentiment_polarity_label"]
        dist = search_res["sentiment_distribution"]
        matches_df = search_res["results_df"]

        st.markdown(f"### Results for Query: *'{query}'*")

        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card("Matching Records", f"{total:,}", f"Found in {len(df):,} total posts")
        with m2:
            accent = "#10B981" if avg_score > 0 else ("#EF4444" if avg_score < 0 else "#64748B")
            render_metric_card("Mean Sentiment Index", f"{avg_score:+.2f}", "Scale: -1.0 (Neg) to +1.0 (Pos)", accent)
        with m3:
            render_metric_card("Overall Polarity", polarity_lbl, "Weighted orientation")

        if total > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            col_chart, col_table = st.columns([1, 2])

            with col_chart:
                fig_donut = plot_sentiment_donut(dist, title=f"Sentiment Split for '{query}'")
                st.plotly_chart(fig_donut, use_container_width=True)

            with col_table:
                st.markdown(f"**Sample Matching Posts ({len(matches_df)} total):**")
                display_cols = [c for c in ["Post ID", "Text", sentiment_col, "Platform", "Date"] if c and c in matches_df.columns]
                st.dataframe(matches_df[display_cols], use_container_width=True, height=350)
        else:
            st.info(f"No records containing '{query}' were found in the current dataset.")
