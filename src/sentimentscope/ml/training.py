"""
SentimentScope AI — Model Training and Experimentation Pipeline
Constructs leakage-safe scikit-learn pipelines for Logistic Regression,
Multinomial Naive Bayes, Linear SVM, and Random Forest using TF-IDF and BoW.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.sentimentscope.config import (
    CANONICAL_SENTIMENTS,
    MIN_SAMPLES_PER_CLASS_FOR_STRATIFY,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.sentimentscope.data.preprocessing import clean_text
from src.sentimentscope.ml.evaluation import EvaluationMetrics, evaluate_model_predictions


SUPPORTED_MODELS = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Multinomial Naive Bayes",
    "linear_svm": "Linear Support Vector Machine",
    "random_forest": "Random Forest",
}

SUPPORTED_FEATURES = {
    "tfidf": "TF-IDF (1-2 ngrams)",
    "bow": "Bag of Words (1-2 ngrams)",
}


def build_vectorizer(feature_type: str = "tfidf", min_df: int = 1) -> Any:
    """Builds the configured text vectorizer."""
    if feature_type == "bow":
        return CountVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=5000,
        )
    # Default to TF-IDF
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=5000,
        sublinear_tf=True,
    )


def build_classifier(model_key: str, seed: int = RANDOM_SEED) -> Any:
    """Instantiates the specified classifier with fixed seeds."""
    if model_key == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
        )
    elif model_key == "naive_bayes":
        return MultinomialNB(alpha=1.0)
    elif model_key == "linear_svm":
        base_svc = LinearSVC(
            C=1.0,
            random_state=seed,
            dual="auto",
            max_iter=2000,
            class_weight="balanced",
        )
        # Calibrate via 3-fold CV on training data to provide authentic probabilities
        return CalibratedClassifierCV(estimator=base_svc, cv=3)
    elif model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported model key: {model_key}")


def create_pipeline(model_key: str, feature_type: str = "tfidf", seed: int = RANDOM_SEED) -> Pipeline:
    """Creates a scikit-learn Pipeline combining vectorizer and classifier."""
    vectorizer = build_vectorizer(feature_type)
    classifier = build_classifier(model_key, seed)
    return Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier),
    ])


def train_and_evaluate_single_model(
    model_key: str,
    feature_type: str,
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    seed: int = RANDOM_SEED,
) -> Tuple[Pipeline, EvaluationMetrics]:
    """
    Trains a pipeline on training split and evaluates strictly on held-out test split.
    """
    pipeline = create_pipeline(model_key, feature_type, seed)

    # Measure training time
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_duration = time.perf_counter() - t0

    # Measure test inference latency
    t1 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    inference_time = (time.perf_counter() - t1) * 1000.0  # ms total
    latency_per_sample = inference_time / max(len(X_test), 1)

    # Probability or decision scores for ROC-AUC
    y_scores = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_scores = pipeline.predict_proba(X_test)
        except Exception:
            y_scores = None
    elif hasattr(pipeline, "decision_function"):
        try:
            y_scores = pipeline.decision_function(X_test)
        except Exception:
            y_scores = None

    metrics = evaluate_model_predictions(
        y_true=y_test,
        y_pred=y_pred,
        y_scores=y_scores,
        classes=sorted(list(np.unique(y_train))),
        training_duration=train_duration,
        inference_latency=latency_per_sample,
    )

    return pipeline, metrics


def run_full_experiment_suite(
    df: pd.DataFrame,
    text_col: str = "Text",
    label_col: str = "Sentiment",
    feature_types: Optional[List[str]] = None,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Runs comparison across all 4 algorithms on TF-IDF (and BoW if requested).
    Returns leaderboard, detailed metrics dictionary, and trained pipelines.
    """
    if feature_types is None:
        feature_types = ["tfidf", "bow"]

    X = df[text_col].astype(str).tolist()
    y = np.array(df[label_col].astype(str).tolist())

    # Check stratification possibility
    class_counts = pd.Series(y).value_counts().to_dict()
    can_stratify = all(cnt >= MIN_SAMPLES_PER_CLASS_FOR_STRATIFY for cnt in class_counts.values())
    stratify = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=stratify,
    )

    results: Dict[str, Any] = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classes": sorted(list(np.unique(y))),
        "random_seed": seed,
        "experiments": {},
        "leaderboard": [],
    }

    best_macro_f1 = -1.0
    best_pipeline = None
    best_key = None

    for f_type in feature_types:
        for m_key in SUPPORTED_MODELS:
            exp_id = f"{m_key}_{f_type}"
            display_name = f"{SUPPORTED_MODELS[m_key]} ({f_type.upper()})"

            pipeline, metrics = train_and_evaluate_single_model(
                model_key=m_key,
                feature_type=f_type,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                seed=seed,
            )

            results["experiments"][exp_id] = {
                "display_name": display_name,
                "model_key": m_key,
                "feature_type": f_type,
                "metrics": metrics,
                "pipeline": pipeline,
            }

            results["leaderboard"].append({
                "Experiment ID": exp_id,
                "Model": SUPPORTED_MODELS[m_key],
                "Representation": f_type.upper(),
                "Accuracy": metrics.accuracy,
                "Macro F1": metrics.macro_f1,
                "Weighted F1": metrics.weighted_f1,
                "Macro Precision": metrics.precision_macro,
                "Macro Recall": metrics.recall_macro,
                "ROC-AUC": metrics.roc_auc_macro if metrics.roc_auc_macro is not None else "N/A",
                "Train Time (s)": metrics.training_duration_seconds,
                "Latency (ms/sample)": metrics.inference_latency_ms,
            })

            # Selection criteria: highest Macro F1, with Accuracy tiebreaker
            score = metrics.macro_f1 * 1000 + metrics.accuracy
            if score > best_macro_f1:
                best_macro_f1 = score
                best_pipeline = pipeline
                best_key = exp_id

    # Sort leaderboard by Macro F1 descending
    results["leaderboard"] = sorted(
        results["leaderboard"],
        key=lambda r: (r["Macro F1"], r["Accuracy"]),
        reverse=True,
    )
    results["best_model_id"] = best_key
    results["best_pipeline"] = best_pipeline

    return results
