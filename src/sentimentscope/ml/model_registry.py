"""
SentimentScope AI — Model Registry and Artifact Management Module
Persists and retrieves versioned model bundles and experiment logs.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
from sklearn.pipeline import Pipeline

from src.sentimentscope.config import (
    CANONICAL_SENTIMENTS,
    DEFAULT_EXPERIMENT_LOG,
    DEFAULT_MODEL_PATH,
    MODELS_DIR,
)


def save_model_artifact(
    pipeline: Pipeline,
    model_name: str,
    feature_type: str,
    metrics: Dict[str, Any],
    classes: Optional[list] = None,
    save_path: Optional[Path] = None,
    random_seed: int = 42,
) -> Path:
    """
    Serializes a complete pipeline and metadata bundle.
    """
    if save_path is None:
        save_path = DEFAULT_MODEL_PATH

    save_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "model_name": model_name,
        "feature_type": feature_type,
        "metrics": metrics,
        "classes": classes or CANONICAL_SENTIMENTS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "is_calibrated": hasattr(pipeline, "predict_proba"),
    }

    joblib.dump(artifact, save_path, compress=3)
    return save_path


def load_model_artifact(
    artifact_path: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Safely loads a serialized model bundle with schema verification.

    Returns:
        (artifact_dict, error_message)
    """
    if artifact_path is None:
        artifact_path = DEFAULT_MODEL_PATH

    p = Path(artifact_path)
    if not p.exists():
        return None, (
            f"Model artifact not found at '{p}'. "
            "Please run 'python scripts/train_model.py' to train and persist the production model."
        )

    try:
        artifact = joblib.load(p)
    except Exception as e:
        return None, f"Failed to deserialize model artifact from '{p}': {str(e)}"

    # Validate bundle schema
    required_keys = ["pipeline", "model_name", "classes"]
    for k in required_keys:
        if k not in artifact:
            return None, f"Corrupted model artifact: missing required key '{k}'."

    if not isinstance(artifact["pipeline"], Pipeline):
        return None, "Corrupted model artifact: 'pipeline' is not a valid scikit-learn Pipeline instance."

    return artifact, None


def save_experiment_history(leaderboard_data: list, path: Optional[Path] = None) -> Path:
    """Save the experiment leaderboard records to JSON."""
    if path is None:
        path = DEFAULT_EXPERIMENT_LOG

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(leaderboard_data, f, indent=2)
    return path


def load_experiment_history(path: Optional[Path] = None) -> Optional[list]:
    """Load the experiment leaderboard records from JSON."""
    if path is None:
        path = DEFAULT_EXPERIMENT_LOG

    p = Path(path)
    if not p.exists():
        return None

    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
