"""
SentimentScope AI — Trends and Platform Aggregation Module
Computes daily sentiment timelines, cross-platform breakdowns, and engagement metrics.
"""

from typing import Any, Dict, Optional
import pandas as pd

from src.sentimentscope.config import CANONICAL_SENTIMENTS


def compute_daily_sentiment_trends(
    df: pd.DataFrame,
    date_col: str = "Date",
    sentiment_col: str = "Sentiment",
) -> pd.DataFrame:
    """
    Aggregates sentiment counts grouped by date.
    Returns pivoted DataFrame with columns: [Date, Negative, Neutral, Positive, Total].
    """
    if df.empty or date_col not in df.columns or sentiment_col not in df.columns:
        return pd.DataFrame()

    df_copy = df.copy()
    # Normalize date
    df_copy["_date"] = pd.to_datetime(df_copy[date_col], errors="coerce").dt.date
    valid_dates = df_copy.dropna(subset=["_date"])

    if valid_dates.empty:
        return pd.DataFrame()

    grouped = (
        valid_dates.groupby(["_date", sentiment_col])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure canonical sentiment columns exist
    for cls in CANONICAL_SENTIMENTS:
        if cls not in grouped.columns:
            grouped[cls] = 0

    grouped = grouped.rename(columns={"_date": "Date"})
    grouped = grouped.sort_values(by="Date")
    grouped["Total"] = grouped[CANONICAL_SENTIMENTS].sum(axis=1)

    return grouped


def compute_platform_sentiment_breakdown(
    df: pd.DataFrame,
    platform_col: str = "Platform",
    sentiment_col: str = "Sentiment",
) -> pd.DataFrame:
    """
    Aggregates sentiment distribution across social media platforms.
    Returns pivoted DataFrame: Platform x Sentiment.
    """
    if df.empty or platform_col not in df.columns or sentiment_col not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby([platform_col, sentiment_col])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for cls in CANONICAL_SENTIMENTS:
        if cls not in grouped.columns:
            grouped[cls] = 0

    grouped["Total"] = grouped[CANONICAL_SENTIMENTS].sum(axis=1)
    grouped = grouped.sort_values(by="Total", ascending=False)
    return grouped


def compute_engagement_by_sentiment(
    df: pd.DataFrame,
    sentiment_col: str = "Sentiment",
    likes_col: str = "Likes",
    shares_col: str = "Shares",
) -> pd.DataFrame:
    """
    Calculates average likes and shares across sentiment classes.
    """
    if df.empty or sentiment_col not in df.columns:
        return pd.DataFrame()

    has_likes = likes_col in df.columns
    has_shares = shares_col in df.columns

    if not has_likes and not has_shares:
        return pd.DataFrame()

    agg_dict = {}
    if has_likes:
        agg_dict[likes_col] = ["mean", "sum"]
    if has_shares:
        agg_dict[shares_col] = ["mean", "sum"]

    res = df.groupby(sentiment_col).agg(agg_dict).reset_index()

    # Flatten column multi-index
    flat_cols = ["Sentiment"]
    if has_likes:
        flat_cols.extend(["Avg_Likes", "Total_Likes"])
    if has_shares:
        flat_cols.extend(["Avg_Shares", "Total_Shares"])

    res.columns = flat_cols
    if "Avg_Likes" in res.columns:
        res["Avg_Likes"] = res["Avg_Likes"].round(1)
    if "Avg_Shares" in res.columns:
        res["Avg_Shares"] = res["Avg_Shares"].round(1)

    return res
