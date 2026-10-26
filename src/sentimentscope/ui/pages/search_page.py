"""
SentimentScope AI — Keyword Search & Topic Monitoring Workspace
Targeted brand, topic, and phrase filtering with aggregate polarity scoring
adhering to the monochrome SaaS design system.
"""

import streamlit as st
import pandas as pd

from src.sentimentscope.analytics.metrics import search_dataset_by_keyword
from src.sentimentscope.analytics.visualizations import plot_sentiment_donut
from src.sentimentscope.config import SENTIMENT_COLOR_MAP
from src.sentimentscope.ui.components import (
    render_header,
    render_metric_card,
    render_sentiment_badge,
)


def render_search_page():
    render_header(
        title="Keyword Search & Topic Mining",
        subtitle="Search Targeted Brands, Entities, or Terminology Across the Active Corpus with Polarity Indexing",
        badge_text="Polarity Indexing",
        category_label="SEARCH & MINING",
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
            placeholder="e.g. update, performance, support, battery, crash, AI",
            value="update",
        )
    with col_action:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        search_clicked = st.button("Search Corpus", type="primary", use_container_width=True)

    if query.strip():
        search_res = search_dataset_by_keyword(df, keyword=query, text_col="Text", sentiment_col=sentiment_col or "Sentiment")

        total = search_res["total_matches"]
        avg_score = search_res["average_sentiment_score"]
        polarity_lbl = search_res["sentiment_polarity_label"]
        dist = search_res["sentiment_distribution"]
        matches_df = search_res["results_df"]

        st.markdown(f"### Query Results: *'{query}'*")

        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card("Matching Records", f"{total:,}", f"Found in {len(df):,} total posts", "#111111")
        with m2:
            accent = SENTIMENT_COLOR_MAP["Positive"] if avg_score > 0 else (SENTIMENT_COLOR_MAP["Negative"] if avg_score < 0 else SENTIMENT_COLOR_MAP["Neutral"])
            render_metric_card("Mean Sentiment Index", f"{avg_score:+.2f}", "Scale: -1.0 (Neg) to +1.0 (Pos)", accent)
        with m3:
            render_metric_card("Overall Polarity", polarity_lbl, "Weighted orientation", "#111111")

        if total > 0:
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            col_chart, col_dist = st.columns([1, 1])

            with col_chart:
                fig_donut = plot_sentiment_donut(dist, title=f"Sentiment Split for '{query}'")
                st.plotly_chart(fig_donut, use_container_width=True)

            with col_dist:
                st.markdown(f"#### Sentiment Breakdown for '{query}'")
                for s_cls in ["Positive", "Neutral", "Negative"]:
                    cnt = dist.get(s_cls, 0)
                    pct = (cnt / total * 100) if total > 0 else 0.0
                    badge = render_sentiment_badge(s_cls)
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border: 1px solid #E5E5E5; border-radius: 6px; margin-bottom: 8px; background: #FAFAFA;">
                            <div>{badge}</div>
                            <div style="font-weight: 600; color: #111111; font-size: 14px;">{cnt} posts <span style="font-weight: 400; color: #666666; font-size: 12px;">({pct:.1f}%)</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            st.markdown(f"### Matching Records ({len(matches_df)} posts found)")
            st.caption("Inspect full social media posts, sentiment annotations, and platform metadata.")

            display_cols = [c for c in ["Post ID", "Text", sentiment_col, "Platform", "Date"] if c and c in matches_df.columns]

            col_config = {}
            if "Post ID" in matches_df.columns:
                col_config["Post ID"] = st.column_config.TextColumn("ID", width="small")
            if "Text" in matches_df.columns:
                col_config["Text"] = st.column_config.TextColumn("Post Content", width="large", help="Full text of the social post")
            if sentiment_col and sentiment_col in matches_df.columns:
                col_config[sentiment_col] = st.column_config.TextColumn("Sentiment", width="small")
            if "Platform" in matches_df.columns:
                col_config["Platform"] = st.column_config.TextColumn("Platform", width="small")
            if "Date" in matches_df.columns:
                col_config["Date"] = st.column_config.TextColumn("Date", width="small")

            st.dataframe(
                matches_df[display_cols],
                column_config=col_config,
                use_container_width=True,
                height=360,
                hide_index=True,
            )
        else:
            st.info(f"No records matching query '{query}' were found in the active corpus.")
