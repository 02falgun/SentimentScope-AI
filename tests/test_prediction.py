"""
Tests for prediction service, latency, confidence bounds, and explainability.
"""

import pytest

from src.sentimentscope.config import DEFAULT_MODEL_PATH
from src.sentimentscope.ml.model_registry import load_model_artifact
from src.sentimentscope.ml.prediction import predict_batch_texts, predict_single_text


@pytest.fixture(scope="module")
def model_artifact():
    artifact, err = load_model_artifact(DEFAULT_MODEL_PATH)
    assert err is None, f"Could not load artifact: {err}"
    assert artifact is not None
    return artifact


def test_predict_single_text_positive(model_artifact):
    res = predict_single_text(
        "I absolutely love this new update! Everything is fast, smooth, and beautiful! 🚀",
        model_artifact
    )
    assert res.error is None
    assert res.predicted_sentiment == "Positive"
    assert res.confidence_score > 0.50
    assert "Positive" in res.probabilities
    assert res.latency_ms < 2000  # NFR-1 requirement: < 2 seconds


def test_predict_single_text_negative(model_artifact):
    res = predict_single_text(
        "This is the worst software update ever. Constantly crashes and ruined my work! 😡",
        model_artifact
    )
    assert res.error is None
    assert res.predicted_sentiment == "Negative"
    assert res.confidence_score > 0.50


def test_predict_single_text_empty(model_artifact):
    res = predict_single_text("", model_artifact)
    assert res.error is not None
    assert res.confidence_score == 0.0


def test_predict_batch_texts(model_artifact):
    texts = [
        "Love this app, five stars!",
        "Terrible bug, crashes constantly.",
        "The conference keynote starts at 10 AM.",
    ]
    df_pred = predict_batch_texts(texts, model_artifact)
    assert len(df_pred) == 3
    assert "Predicted_Sentiment" in df_pred.columns
    assert "Confidence" in df_pred.columns
    assert df_pred["Predicted_Sentiment"].iloc[0] == "Positive"
    assert df_pred["Predicted_Sentiment"].iloc[1] == "Negative"
    assert df_pred["Predicted_Sentiment"].iloc[2] == "Neutral"


def test_predict_single_text_excessive_length(model_artifact):
    long_text = "This is a great app! " * 200  # 4200 characters
    res = predict_single_text(long_text, model_artifact)
    assert res.error is None
    assert res.predicted_sentiment == "Positive"
    assert res.confidence_score > 0.0
