"""
Tests for CSV loading, error recovery, column auto-detection, and data validation.
"""

import io
import pandas as pd
import pytest

from src.sentimentscope.data.loading import (
    load_csv,
    detect_all_likely_columns,
    detect_likely_column,
)
from src.sentimentscope.data.validation import (
    validate_and_profile_dataframe,
    normalize_sentiment_label,
)


def test_detect_likely_columns():
    cols = ["tweet_id", "Tweet_Content", "User_Sentiment", "created_time", "fav_count", "retweet_count", "Source"]
    detected = detect_all_likely_columns(cols)
    assert detected["text"] == "Tweet_Content"
    assert detected["sentiment"] == "User_Sentiment"
    assert detected["date"] == "created_time"
    assert detected["id"] == "tweet_id"


def test_load_csv_from_string():
    csv_data = "Text,Sentiment\nAwesome day!,Positive\nHorrible service,-1\nJust a test,neu\n"
    df, err = load_csv(io.StringIO(csv_data))
    assert err is None
    assert df is not None
    assert len(df) == 3


def test_normalize_sentiment_label():
    assert normalize_sentiment_label("positive") == "Positive"
    assert normalize_sentiment_label("POS") == "Positive"
    assert normalize_sentiment_label("1") == "Positive"
    assert normalize_sentiment_label("neutral") == "Neutral"
    assert normalize_sentiment_label("0") == "Neutral"
    assert normalize_sentiment_label("negative") == "Negative"
    assert normalize_sentiment_label("-1") == "Negative"
    assert normalize_sentiment_label("unknown_nonsense") is None


def test_validate_and_profile_dataframe_handling():
    # DataFrame with duplicates and missing values
    data = {
        "content": ["Great work!", "Great work!", "", None, "Terrible bug", "Standard update"],
        "polarity": ["Positive", "Positive", "Positive", "Negative", "Negative", "Neutral"],
    }
    df = pd.DataFrame(data)
    cleaned_df, profile = validate_and_profile_dataframe(
        df,
        text_col="content",
        sentiment_col="polarity"
    )

    assert profile.total_loaded == 6
    assert profile.duplicates_removed == 1  # 1 duplicate removed
    assert profile.missing_text_removed == 2  # 1 empty string, 1 None
    assert profile.retained_rows == 3  # Remaining valid rows
    assert "Text" in cleaned_df.columns
    assert "Sentiment" in cleaned_df.columns
    assert len(cleaned_df) == 3


def test_empty_dataframe_validation():
    df = pd.DataFrame()
    cleaned_df, profile = validate_and_profile_dataframe(df, text_col="text")
    assert cleaned_df.empty
    assert len(profile.warnings) > 0
