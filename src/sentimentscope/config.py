"""
SentimentScope AI — Central Configuration Module
Defines repository paths, dataset schema, model parameters, and platform constants.
"""

from pathlib import Path
from typing import Dict, List, Set

# Base Directories
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# Default File Paths
DEFAULT_SAMPLE_DATASET = SAMPLE_DATA_DIR / "social_media_sentiment_sample.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
DEFAULT_EXPERIMENT_LOG = MODELS_DIR / "experiment_history.json"

# Schema Specifications (from PRD Section 2.6)
REQUIRED_SCHEMA_FIELDS: List[str] = [
    "Post ID",
    "Text",
    "Platform",
    "Date",
    "Likes",
    "Shares",
    "Sentiment",
]

# Accepted Sentiment Classes & Normalized Mapping
CANONICAL_SENTIMENTS: List[str] = ["Negative", "Neutral", "Positive"]
SENTIMENT_COLOR_MAP: Dict[str, str] = {
    "Positive": "#10B981",  # Emerald Green
    "Neutral": "#64748B",   # Slate Grey
    "Negative": "#EF4444",  # Crimson Red
}

LABEL_NORMALIZATION_MAP: Dict[str, str] = {
    # Positive variations
    "positive": "Positive",
    "pos": "Positive",
    "1": "Positive",
    "4": "Positive",
    # Neutral variations
    "neutral": "Neutral",
    "neu": "Neutral",
    "0": "Neutral",
    "2": "Neutral",
    # Negative variations
    "negative": "Negative",
    "neg": "Negative",
    "-1": "Negative",
}

# Preprocessing Constants
NEGATION_TOKENS: Set[str] = {
    "no", "not", "nor", "neither", "never", "none", "nobody", "nowhere",
    "hardly", "scarcely", "barely", "cannot", "can't", "won't", "don't",
    "didn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "wouldn't", "shouldn't", "couldn't", "mustn't", "without"
}

# Machine Learning Reproducibility & Limits
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20
MIN_SAMPLES_PER_CLASS_FOR_STRATIFY: int = 2
MAX_UPLOAD_SIZE_MB: int = 20
MAX_UPLOAD_ROWS_WARN: int = 25000
