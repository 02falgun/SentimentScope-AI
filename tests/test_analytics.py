"""
Tests for analytics calculations, metrics, trends, and search functionality.
"""

import pandas as pd
import pytest

from src.sentimentscope.analytics.metrics import (
    compute_hashtag_and_emoji_stats,
    compute_post_length_stats,
    compute_sentiment_distribution,
    compute_top_words_by_sentiment,
    search_dataset_by_keyword,
)
from src.sentimentscope.analytics.trends import (
    compute_daily_sentiment_trends,
    compute_engagement_by_sentiment,
    compute_platform_sentiment_breakdown,
)
from src.sentimentscope.config import DEFAULT_SAMPLE_DATASET
from src.sentimentscope.data.loading import load_csv


@pytest.fixture(scope="module")
def sample_df():
    df, err = load_csv(DEFAULT_SAMPLE_DATASET)
    assert err is None
    return df


def test_compute_sentiment_distribution(sample_df):
    dist = compute_sentiment_distribution(sample_df, sentiment_col="Sentiment")
    assert dist["total"] == len(sample_df)
    assert "Positive" in dist["counts"]
    assert "Negative" in dist["counts"]
    assert "Neutral" in dist["counts"]
    assert sum(dist["percentages"].values()) == pytest.approx(100.0, abs=0.5)


def test_compute_top_words(sample_df):
    top_words = compute_top_words_by_sentiment(sample_df, text_col="Text", sentiment_col="Sentiment")
    assert "Positive" in top_words
    assert "Negative" in top_words
    assert len(top_words["Positive"]) > 0


def test_compute_post_length_stats(sample_df):
    stats = compute_post_length_stats(sample_df, text_col="Text")
    assert stats["avg_chars"] > 0
    assert stats["avg_words"] > 0
    assert len(stats["char_lengths"]) == len(sample_df)


def test_search_dataset_by_keyword(sample_df):
    # Search for an expected keyword in sample posts
    res = search_dataset_by_keyword(sample_df, keyword="update", text_col="Text", sentiment_col="Sentiment")
    assert res["query"] == "update"
    assert res["total_matches"] > 0
    assert not res["results_df"].empty
    assert -1.0 <= res["average_sentiment_score"] <= 1.0


def test_compute_daily_trends(sample_df):
    trends = compute_daily_sentiment_trends(sample_df, date_col="Date", sentiment_col="Sentiment")
    assert not trends.empty
    assert "Date" in trends.columns
    assert "Positive" in trends.columns
    assert "Total" in trends.columns


def test_compute_platform_breakdown(sample_df):
    breakdown = compute_platform_sentiment_breakdown(sample_df, platform_col="Platform", sentiment_col="Sentiment")
    assert not breakdown.empty
    assert "Platform" in breakdown.columns
    assert "Total" in breakdown.columns


def test_plot_sentiment_empty_data():
    from src.sentimentscope.analytics.visualizations import plot_sentiment_bars, plot_sentiment_donut
    empty_dist = {"Negative": 0, "Neutral": 0, "Positive": 0}
    fig_donut = plot_sentiment_donut(empty_dist)
    fig_bar = plot_sentiment_bars(empty_dist)
    assert fig_donut is not None
    assert fig_bar is not None
    assert "No Data" in fig_donut.layout.title.text
    assert "No Data" in fig_bar.layout.title.text
