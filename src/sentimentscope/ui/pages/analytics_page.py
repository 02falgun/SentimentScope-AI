"""
SentimentScope AI — Visual Analytics & Trends Workspace
Exploratory data intelligence across sentiments, timelines, platforms,
and lexical frequency distributions adhering to the monochrome SaaS design system.
"""

import streamlit as st
import pandas as pd

from src.sentimentscope.analytics.metrics import (
    compute_hashtag_and_emoji_stats,
    compute_post_length_stats,
    compute_sentiment_distribution,
    compute_top_words_by_sentiment,
)
from src.sentimentscope.analytics.trends import (
    compute_daily_sentiment_trends,
    compute_engagement_by_sentiment,
    compute_platform_sentiment_breakdown,
)
from src.sentimentscope.analytics.visualizations import (
    generate_wordcloud_figure,
    plot_daily_trends,
    plot_engagement_comparison,
    plot_platform_comparison,
    plot_post_length_histogram,
    plot_sentiment_bars,
    plot_sentiment_donut,
    plot_top_words_bar,
)
from src.sentimentscope.config import CANONICAL_SENTIMENTS, SENTIMENT_COLOR_MAP
from src.sentimentscope.ui.components import (
    render_header,
    render_metric_card,
)


def render_analytics_page():
    render_header(
        title="Visual Analytics & Trends",
        subtitle="Exploratory Data Intelligence Across Sentiments, Timelines, Platforms, and Vocabulary Distinctions",
        badge_text="Real Aggregations",
        category_label="ANALYTICS WORKSPACE",
    )

    df: pd.DataFrame = st.session_state.get("active_df", pd.DataFrame())
    if df.empty:
        st.warning("No dataset is currently active. Please load or upload a dataset in the Explorer.")
        return

    # Determine sentiment column to use
    sentiment_col = "Sentiment" if "Sentiment" in df.columns else ("Predicted_Sentiment" if "Predicted_Sentiment" in df.columns else None)
    if not sentiment_col:
        st.info("No sentiment labels found. Run batch model predictions in the Dataset Explorer to unlock sentiment analytics.")
        sentiment_col = "Sentiment"

    # Filter Controls Expander
    with st.expander("Filter Analytics Scope", expanded=False):
        fc1, fc2 = st.columns(2)
        with fc1:
            all_plats = list(df["Platform"].unique()) if "Platform" in df.columns else []
            sel_plats = st.multiselect("Filter Platforms:", options=all_plats, default=all_plats)
        with fc2:
            all_sents = list(df[sentiment_col].dropna().unique()) if sentiment_col in df.columns else []
            sel_sents = st.multiselect("Filter Sentiments:", options=all_sents, default=all_sents)

    # Apply filters
    filtered_df = df.copy()
    if sel_plats and "Platform" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Platform"].isin(sel_plats)]
    if sel_sents and sentiment_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[sentiment_col].isin(sel_sents)]

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        return

    # 1. Distribution Overview
    st.markdown("### Aggregate Sentiment Distribution")
    dist = compute_sentiment_distribution(filtered_df, sentiment_col=sentiment_col)

    c_donut, c_bar = st.columns([1, 1])
    with c_donut:
        fig_donut = plot_sentiment_donut(dist["counts"], title="Sentiment Share")
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_bar:
        fig_bar = plot_sentiment_bars(dist["counts"], title="Sentiment Volume Counts")
        st.plotly_chart(fig_bar, use_container_width=True)

    # 2. Timeline & Cross-Platform Trends
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### Temporal & Cross-Platform Trends")

    t_col1, t_col2 = st.columns([1, 1])
    with t_col1:
        trends_df = compute_daily_sentiment_trends(filtered_df, date_col="Date", sentiment_col=sentiment_col)
        fig_trends = plot_daily_trends(trends_df)
        st.plotly_chart(fig_trends, use_container_width=True)

    with t_col2:
        plat_df = compute_platform_sentiment_breakdown(filtered_df, platform_col="Platform", sentiment_col=sentiment_col)
        fig_plat = plot_platform_comparison(plat_df)
        st.plotly_chart(fig_plat, use_container_width=True)

    # 3. Engagement and Content Lengths
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### Engagement Metrics & Content Lengths")

    e_col1, e_col2 = st.columns([1, 1])
    with e_col1:
        eng_df = compute_engagement_by_sentiment(filtered_df, sentiment_col=sentiment_col, likes_col="Likes", shares_col="Shares")
        fig_eng = plot_engagement_comparison(eng_df)
        st.plotly_chart(fig_eng, use_container_width=True)

    with e_col2:
        length_stats = compute_post_length_stats(filtered_df, text_col="Text")
        fig_len = plot_post_length_histogram(length_stats["char_lengths"])
        st.plotly_chart(fig_len, use_container_width=True)

    # 4. Top Vocabulary Terms by Sentiment
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### Vocabulary Analysis by Sentiment Class")

    top_words = compute_top_words_by_sentiment(filtered_df, text_col="Text", sentiment_col=sentiment_col, top_n=10)
    w_col1, w_col2, w_col3 = st.columns(3)

    with w_col1:
        fig_pos = plot_top_words_bar(top_words.get("Positive", []), title="Top Positive Terms", bar_color=SENTIMENT_COLOR_MAP["Positive"])
        st.plotly_chart(fig_pos, use_container_width=True)

    with w_col2:
        fig_neu = plot_top_words_bar(top_words.get("Neutral", []), title="Top Neutral Terms", bar_color=SENTIMENT_COLOR_MAP["Neutral"])
        st.plotly_chart(fig_neu, use_container_width=True)

    with w_col3:
        fig_neg = plot_top_words_bar(top_words.get("Negative", []), title="Top Negative Terms", bar_color=SENTIMENT_COLOR_MAP["Negative"])
        st.plotly_chart(fig_neg, use_container_width=True)

    # 5. WordCloud, Hashtags, and Emojis
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### WordCloud & Social Expression Intelligence")

    wc_col, tag_col = st.columns([1.2, 1])
    with wc_col:
        st.markdown("#### Corpus WordCloud:")
        all_text = " ".join(filtered_df["Text"].dropna().astype(str))
        wc_fig = generate_wordcloud_figure(all_text, bg_color="#FFFFFF")
        if wc_fig is not None:
            st.pyplot(wc_fig, use_container_width=True)
        else:
            st.info("WordCloud figure could not be rendered for this corpus.")

    with tag_col:
        st.markdown("#### Trending Hashtags & Emojis:")
        social_stats = compute_hashtag_and_emoji_stats(filtered_df, text_col="Text", top_n=8)

        st.markdown("**Top Hashtags Identified:**")
        if social_stats["hashtags"]:
            for tag, count in social_stats["hashtags"]:
                st.markdown(f"- `{tag}` &nbsp;({count:,} posts)")
        else:
            st.write("No hashtags identified in active slice.")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("**Top Expressed Emojis:**")
        if social_stats["emojis"]:
            emoji_str = " &nbsp; ".join([f"<span style='font-size: 18px;'>{em}</span> ({cnt})" for em, cnt in social_stats["emojis"]])
            st.markdown(emoji_str, unsafe_allow_html=True)
        else:
            st.write("No emojis identified in active slice.")
