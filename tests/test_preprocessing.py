"""
Tests for text preprocessing pipeline, negation preservation, and emoji/hashtag handling.
"""

import pytest
from src.sentimentscope.data.preprocessing import (
    clean_text,
    extract_emojis,
    extract_hashtags,
    get_effective_stopwords,
    EFFECTIVE_STOPWORDS,
)
from src.sentimentscope.config import NEGATION_TOKENS


def test_clean_text_basic():
    raw = "Hello World! This is a test sentence."
    cleaned = clean_text(raw)
    assert "hello" in cleaned
    assert "test" in cleaned
    assert "sentence" in cleaned
    assert "!" not in cleaned


def test_negation_preservation():
    """Verify that negations like 'not', 'never', 'hardly' are strictly preserved."""
    raw = "The product is not good and I will never buy it."
    cleaned = clean_text(raw)
    tokens = cleaned.split()
    assert "not" in tokens, f"'not' was removed from: {cleaned}"
    assert "never" in tokens, f"'never' was removed from: {cleaned}"
    assert "good" in tokens
    assert "buy" in tokens


def test_effective_stopwords_exclude_negations():
    """Ensure NEGATION_TOKENS are not in EFFECTIVE_STOPWORDS."""
    for token in NEGATION_TOKENS:
        assert token not in EFFECTIVE_STOPWORDS, f"Negation token '{token}' should not be in stop words list!"


def test_url_and_mention_removal():
    raw = "Check out https://antigravity.ai and follow @deepmind for AI news!"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "antigravity.ai" not in cleaned
    assert "@deepmind" not in cleaned
    assert "deepmind" not in cleaned
    assert "news" in cleaned


def test_hashtag_handling():
    raw = "Loving this #MachineLearning framework! #AI"
    # When remove_hashtags=False (default), hash symbol is removed but keyword preserved
    cleaned = clean_text(raw)
    assert "#" not in cleaned
    assert "machinelearning" in cleaned or "machine" in cleaned

    tags = extract_hashtags(raw)
    assert "machinelearning" in tags
    assert "ai" in tags


def test_emoji_extraction():
    raw = "Amazing results today! 🚀✨ Great job 🎉"
    emojis = extract_emojis(raw)
    assert len(emojis) > 0
    # Clean text should strip emojis for pure text feature extraction
    cleaned = clean_text(raw)
    assert "🚀" not in cleaned
    assert "amazing" in cleaned


def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n\t  ") == ""
