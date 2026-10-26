"""
SentimentScope AI — Analytics Metrics Module
Computes sentiment distributions, word frequencies, post lengths,
hashtags, emojis, and keyword search aggregations.
"""

from collections import Counter
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.sentimentscope.config import CANONICAL_SENTIMENTS
from src.sentimentscope.data.preprocessing import (
    EFFECTIVE_STOPWORDS,
    clean_text,
    extract_emojis,
    extract_hashtags,
)


NUMERIC_SENTIMENT_MAP = {
    "Positive": 1.0,
    "Neutral": 0.0,
    "Negative": -1.0,
}


def compute_sentiment_distribution(
    df: pd.DataFrame,
    sentiment_col: str = "Sentiment",
) -> Dict[str, Any]:
    """
    Computes absolute counts and percentage shares for each sentiment class.
    """
    if df.empty or sentiment_col not in df.columns:
        return {
            "counts": {cls: 0 for cls in CANONICAL_SENTIMENTS},
            "percentages": {cls: 0.0 for cls in CANONICAL_SENTIMENTS},
            "total": 0,
        }

    counts_series = df[sentiment_col].value_counts()
    counts = {cls: int(counts_series.get(cls, 0)) for cls in CANONICAL_SENTIMENTS}
    total = sum(counts.values())

    percentages = {
        cls: round((cnt / total) * 100, 2) if total > 0 else 0.0
        for cls, cnt in counts.items()
    }

    return {
        "counts": counts,
        "percentages": percentages,
        "total": total,
    }


def compute_top_words_by_sentiment(
    df: pd.DataFrame,
    text_col: str = "Text",
    sentiment_col: str = "Sentiment",
    top_n: int = 15,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Extracts the most frequent words per sentiment class, filtering out stopwords.
    """
    results: Dict[str, List[Tuple[str, int]]] = {cls: [] for cls in CANONICAL_SENTIMENTS}

    if df.empty or text_col not in df.columns:
        return results

    has_sentiment = sentiment_col in df.columns

    for cls in CANONICAL_SENTIMENTS:
        if has_sentiment:
            subset = df[df[sentiment_col] == cls]
        else:
            subset = df

        if subset.empty:
            continue

        words_counter = Counter()
        for text in subset[text_col].dropna().astype(str):
            cleaned = clean_text(text, remove_stopwords=True)
            for token in cleaned.split():
                if len(token) > 2 and token not in EFFECTIVE_STOPWORDS:
                    words_counter[token] += 1

        results[cls] = words_counter.most_common(top_n)

    return results


def compute_post_length_stats(
    df: pd.DataFrame,
    text_col: str = "Text",
) -> Dict[str, Any]:
    """
    Calculates character and word length distribution statistics.
    """
    if df.empty or text_col not in df.columns:
        return {"char_lengths": [], "word_counts": [], "avg_chars": 0, "avg_words": 0}

    texts = df[text_col].dropna().astype(str)
    char_lengths = texts.apply(len).tolist()
    word_counts = texts.apply(lambda t: len(t.split())).tolist()

    return {
        "char_lengths": char_lengths,
        "word_counts": word_counts,
        "avg_chars": round(float(np.mean(char_lengths)), 1) if char_lengths else 0,
        "avg_words": round(float(np.mean(word_counts)), 1) if word_counts else 0,
    }


def compute_hashtag_and_emoji_stats(
    df: pd.DataFrame,
    text_col: str = "Text",
    top_n: int = 10,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Extracts top hashtags and top emojis across all records.
    """
    if df.empty or text_col not in df.columns:
        return {"hashtags": [], "emojis": []}

    hashtag_counter = Counter()
    emoji_counter = Counter()

    for text in df[text_col].dropna().astype(str):
        tags = extract_hashtags(text)
        emojis = extract_emojis(text)
        hashtag_counter.update(tags)
        emoji_counter.update(emojis)

    return {
        "hashtags": [(f"#{k}", v) for k, v in hashtag_counter.most_common(top_n)],
        "emojis": emoji_counter.most_common(top_n),
    }


def search_dataset_by_keyword(
    df: pd.DataFrame,
    keyword: str,
    text_col: str = "Text",
    sentiment_col: str = "Sentiment",
) -> Dict[str, Any]:
    """
    Searches dataset for query term and calculates matching count,
    sentiment breakdown, and average numerical sentiment score.
    """
    if df.empty or not keyword.strip() or text_col not in df.columns:
        return {
            "query": keyword,
            "total_matches": 0,
            "results_df": pd.DataFrame(),
            "sentiment_distribution": {cls: 0 for cls in CANONICAL_SENTIMENTS},
            "average_sentiment_score": 0.0,
            "sentiment_polarity_label": "Neutral",
        }

    pattern = re.compile(re.escape(keyword.strip()), re.IGNORECASE)
    mask = df[text_col].astype(str).str.contains(pattern, regex=True)
    matching_df = df[mask].copy()

    total_matches = len(matching_df)
    if total_matches == 0:
        return {
            "query": keyword,
            "total_matches": 0,
            "results_df": matching_df,
            "sentiment_distribution": {cls: 0 for cls in CANONICAL_SENTIMENTS},
            "average_sentiment_score": 0.0,
            "sentiment_polarity_label": "None",
        }

    # Sentiment distribution of matches
    if sentiment_col in matching_df.columns:
        counts = matching_df[sentiment_col].value_counts().to_dict()
        dist = {cls: int(counts.get(cls, 0)) for cls in CANONICAL_SENTIMENTS}
        # Calculate numeric sentiment (-1 to +1)
        numeric_scores = [NUMERIC_SENTIMENT_MAP.get(s, 0.0) for s in matching_df[sentiment_col]]
        avg_score = round(float(np.mean(numeric_scores)), 2)
    else:
        dist = {cls: 0 for cls in CANONICAL_SENTIMENTS}
        avg_score = 0.0

    if avg_score > 0.15:
        polarity = "Net Positive"
    elif avg_score < -0.15:
        polarity = "Net Negative"
    else:
        polarity = "Balanced / Neutral"

    return {
        "query": keyword,
        "total_matches": total_matches,
        "results_df": matching_df,
        "sentiment_distribution": dist,
        "average_sentiment_score": avg_score,
        "sentiment_polarity_label": polarity,
    }
