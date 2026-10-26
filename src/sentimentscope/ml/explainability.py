"""
SentimentScope AI — Model Explainability Module
Extracts global top feature weights and local token attributions from scikit-learn pipelines.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.pipeline import Pipeline

from src.sentimentscope.data.preprocessing import clean_text


def get_top_features_per_class(
    pipeline: Pipeline,
    top_n: int = 15,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extracts top associated terms for each sentiment class.
    Returns: { 'Positive': [(word, weight), ...], 'Negative': [...], 'Neutral': [...] }
    """
    vectorizer = pipeline.named_steps.get("vectorizer")
    classifier = pipeline.named_steps.get("classifier")

    if vectorizer is None or classifier is None:
        return {}

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        return {}

    # Handle CalibratedClassifierCV
    if hasattr(classifier, "calibrated_classifiers_"):
        base = classifier.calibrated_classifiers_[0].estimator
    else:
        base = classifier

    results: Dict[str, List[Tuple[str, float]]] = {}

    if hasattr(base, "coef_"):
        coef = base.coef_
        # Multiclass: coef shape (n_classes, n_features)
        for idx, cls in enumerate(classes):
            if idx < len(coef):
                class_weights = coef[idx]
                top_indices = np.argsort(class_weights)[-top_n:][::-1]
                results[cls] = [
                    (feature_names[i], round(float(class_weights[i]), 4))
                    for i in top_indices
                ]
    elif hasattr(base, "feature_log_prob_"):
        # Naive Bayes log probability per class
        log_probs = base.feature_log_prob_
        for idx, cls in enumerate(classes):
            if idx < len(log_probs):
                class_weights = log_probs[idx]
                top_indices = np.argsort(class_weights)[-top_n:][::-1]
                results[cls] = [
                    (feature_names[i], round(float(class_weights[i]), 4))
                    for i in top_indices
                ]
    elif hasattr(base, "feature_importances_"):
        # Random Forest: overall feature importance
        importances = base.feature_importances_
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_items = [
            (feature_names[i], round(float(importances[i]), 4))
            for i in top_indices
        ]
        # Assign to all classes as general feature salience
        for cls in classes:
            results[cls] = top_items

    return results


def explain_single_prediction(
    pipeline: Pipeline,
    text: str,
    target_class: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Computes token-level importance for an individual input text.
    Returns: list of dicts: [{'word': token, 'score': float, 'polarity': 'positive'|'negative'|'neutral'}]
    """
    vectorizer = pipeline.named_steps.get("vectorizer")
    classifier = pipeline.named_steps.get("classifier")

    if vectorizer is None or classifier is None or not text.strip():
        return []

    classes = list(getattr(classifier, "classes_", []))
    if not classes:
        return []

    # Handle CalibratedClassifierCV
    if hasattr(classifier, "calibrated_classifiers_"):
        base = classifier.calibrated_classifiers_[0].estimator
    else:
        base = classifier

    cleaned = clean_text(text)
    words = cleaned.split()
    if not words:
        return []

    vocab = vectorizer.vocabulary_
    feature_names = vectorizer.get_feature_names_out()

    token_explanations = []

    # Get class index
    if target_class in classes:
        class_idx = classes.index(target_class)
    else:
        class_idx = 0

    if hasattr(base, "coef_"):
        coef = base.coef_
        weights = coef[class_idx] if class_idx < len(coef) else np.zeros(len(feature_names))
        for w in words:
            if w in vocab:
                feat_idx = vocab[w]
                score = float(weights[feat_idx])
                token_explanations.append({
                    "token": w,
                    "weight": round(score, 4),
                    "impact": "Supports Prediction" if score > 0 else "Opposes Prediction",
                })
            else:
                token_explanations.append({
                    "token": w,
                    "weight": 0.0,
                    "impact": "Neutral / Out of Vocabulary",
                })
    else:
        # Fallback for models without direct linear coefficients
        for w in words:
            token_explanations.append({
                "token": w,
                "weight": 0.0,
                "impact": "Vocabulary Present" if w in vocab else "Out of Vocabulary",
            })

    return token_explanations
