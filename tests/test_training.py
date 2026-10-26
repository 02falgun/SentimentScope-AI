"""
Tests for ML pipeline construction, training reproducibility, and evaluation metrics.
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.sentimentscope.config import RANDOM_SEED
from src.sentimentscope.ml.evaluation import evaluate_model_predictions
from src.sentimentscope.ml.model_registry import load_model_artifact, save_model_artifact
from src.sentimentscope.ml.training import create_pipeline, train_and_evaluate_single_model


def test_create_pipeline_structure():
    """Verify that create_pipeline returns a valid scikit-learn Pipeline with vectorizer and classifier."""
    for model_key in ["logistic_regression", "naive_bayes", "linear_svm", "random_forest"]:
        pipe = create_pipeline(model_key=model_key, feature_type="tfidf", seed=RANDOM_SEED)
        assert isinstance(pipe, Pipeline)
        assert "vectorizer" in pipe.named_steps
        assert "classifier" in pipe.named_steps


def test_train_and_evaluate_single_model():
    """Test training and evaluating on a deterministic small toy corpus."""
    X_train = [
        "Loving this wonderful application",
        "Brilliant performance and clean design",
        "Worst app ever, total disaster",
        "Terrible customer service and broken update",
        "The conference schedule is set for tomorrow",
        "Meeting minutes uploaded to shared folder",
    ]
    y_train = np.array(["Positive", "Positive", "Negative", "Negative", "Neutral", "Neutral"])

    X_test = [
        "Wonderful design and great app",
        "Disaster and horrible experience",
        "Meeting schedule details",
    ]
    y_test = np.array(["Positive", "Negative", "Neutral"])

    pipe, metrics = train_and_evaluate_single_model(
        model_key="logistic_regression",
        feature_type="tfidf",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=RANDOM_SEED,
    )

    assert isinstance(pipe, Pipeline)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert len(metrics.confusion_matrix) == 3
    assert metrics.training_duration_seconds > 0.0


def test_evaluate_model_predictions_metrics():
    y_true = np.array(["Positive", "Negative", "Neutral", "Positive"])
    y_pred = np.array(["Positive", "Negative", "Neutral", "Neutral"])

    metrics = evaluate_model_predictions(y_true, y_pred)
    assert metrics.accuracy == 0.75
    assert metrics.classes == ["Negative", "Neutral", "Positive"]
    assert "Positive" in metrics.per_class
    assert "Negative" in metrics.per_class
    assert "Neutral" in metrics.per_class
