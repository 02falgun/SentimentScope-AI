"""
SentimentScope AI — Prediction Service
Serves single-text and batch predictions with honest probability distributions,
uncertainty detection, and token explainability.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.sentimentscope.ml.explainability import explain_single_prediction


@dataclass
class SinglePredictionResult:
    """Encapsulates the prediction outcome for a single text input."""
    raw_text: str
    predicted_sentiment: str
    confidence_score: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    is_uncertain: bool = False
    uncertainty_reason: str = ""
    token_explanations: List[Dict[str, Any]] = field(default_factory=list)
    model_name: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


def predict_single_text(
    text: str,
    artifact: Dict[str, Any],
    uncertainty_threshold: float = 0.50,
    margin_threshold: float = 0.12,
) -> SinglePredictionResult:
    """
    Classifies a single text input with uncertainty bounds and feature explanations.
    """
    t0 = time.perf_counter()

    if not isinstance(text, str) or not text.strip():
        return SinglePredictionResult(
            raw_text=text or "",
            predicted_sentiment="Neutral",
            confidence_score=0.0,
            error="Input text is empty or contains only whitespace.",
        )

    # Safe capping for extreme text lengths
    if len(text) > 2000:
        text = text[:2000]

    pipeline: Pipeline = artifact["pipeline"]
    model_name = artifact.get("model_name", "Sentiment Classifier")
    classes = list(artifact.get("classes", ["Negative", "Neutral", "Positive"]))

    try:
        # Run inference
        pred_label = str(pipeline.predict([text])[0])

        probs_dict: Dict[str, float] = {}
        confidence = 0.0
        is_uncertain = False
        uncertainty_reason = ""

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba([text])[0]
            # Map probabilities to classes
            model_classes = list(getattr(pipeline.named_steps["classifier"], "classes_", classes))
            for cls, p in zip(model_classes, probs):
                probs_dict[str(cls)] = round(float(p), 4)

            confidence = probs_dict.get(pred_label, round(float(np.max(probs)), 4))

            # Detect uncertainty
            sorted_probs = sorted(probs_dict.values(), reverse=True)
            if sorted_probs[0] < uncertainty_threshold:
                is_uncertain = True
                uncertainty_reason = (
                    f"Top class confidence ({sorted_probs[0]*100:.1f}%) is below the confidence threshold of {uncertainty_threshold*100:.0f}%."
                )
            elif len(sorted_probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) < margin_threshold:
                is_uncertain = True
                uncertainty_reason = (
                    f"Narrow probability margin ({(sorted_probs[0]-sorted_probs[1])*100:.1f}%) between top two classes indicates ambiguous sentiment."
                )
        else:
            # Fallback if probability not available
            confidence = 1.0
            probs_dict = {pred_label: 1.0}

        # Token explanation
        token_expl = explain_single_prediction(pipeline, text, target_class=pred_label)

        latency = (time.perf_counter() - t0) * 1000.0

        return SinglePredictionResult(
            raw_text=text,
            predicted_sentiment=pred_label,
            confidence_score=confidence,
            probabilities=probs_dict,
            is_uncertain=is_uncertain,
            uncertainty_reason=uncertainty_reason,
            token_explanations=token_expl,
            model_name=model_name,
            latency_ms=round(latency, 2),
        )

    except Exception as e:
        return SinglePredictionResult(
            raw_text=text,
            predicted_sentiment="Neutral",
            confidence_score=0.0,
            model_name=model_name,
            error=f"Prediction failed: {str(e)}",
        )


def predict_batch_texts(
    texts: List[str],
    artifact: Dict[str, Any],
) -> pd.DataFrame:
    """
    Classifies a list of text strings in batch mode.
    Returns DataFrame with columns ['Predicted_Sentiment', 'Confidence'].
    """
    pipeline: Pipeline = artifact["pipeline"]
    model_classes = list(artifact.get("classes", ["Negative", "Neutral", "Positive"]))

    clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
    preds = pipeline.predict(clean_texts)

    results = {"Predicted_Sentiment": preds}

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(clean_texts)
        # Confidence is max probability
        confidences = np.max(probs, axis=1)
        results["Confidence"] = np.round(confidences, 4)
    else:
        results["Confidence"] = 1.0

    return pd.DataFrame(results)
