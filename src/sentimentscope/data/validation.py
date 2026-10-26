"""
SentimentScope AI — Dataset Validation and Quality Profiling Module
Validates schemas, normalizes labels, flags severe class imbalances,
and tracks row attrition (loaded, duplicate, missing, retained).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.sentimentscope.config import (
    CANONICAL_SENTIMENTS,
    LABEL_NORMALIZATION_MAP,
    MIN_SAMPLES_PER_CLASS_FOR_STRATIFY,
)


@dataclass
class QualityProfile:
    """Summary of dataset quality metrics and row attrition."""
    total_loaded: int = 0
    duplicates_removed: int = 0
    missing_text_removed: int = 0
    retained_rows: int = 0
    null_counts: Dict[str, int] = field(default_factory=dict)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    class_proportions: Dict[str, float] = field(default_factory=dict)
    unexpected_labels: List[str] = field(default_factory=list)
    has_ground_truth: bool = False
    is_severely_imbalanced: bool = False
    can_stratify: bool = True
    warnings: List[str] = field(default_factory=list)


def normalize_sentiment_label(val: Optional[str]) -> Optional[str]:
    """Normalize any casing or abbreviation of sentiment into Canonical class."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    return LABEL_NORMALIZATION_MAP.get(s, None)


def validate_and_profile_dataframe(
    df: pd.DataFrame,
    text_col: str,
    sentiment_col: Optional[str] = None,
    platform_col: Optional[str] = None,
    date_col: Optional[str] = None,
    likes_col: Optional[str] = None,
    shares_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, QualityProfile]:
    """
    Validates, cleans, maps schema, and profiles a loaded pandas DataFrame.

    Returns:
        (cleaned_df, quality_profile)
    """
    profile = QualityProfile(total_loaded=len(df))

    if df.empty:
        profile.warnings.append("The uploaded dataset is empty.")
        return pd.DataFrame(), profile

    if text_col not in df.columns:
        profile.warnings.append(f"Required text column '{text_col}' not found in dataset.")
        return pd.DataFrame(), profile

    # Record initial null counts
    profile.null_counts = {col: int(df[col].isna().sum()) for col in df.columns}

    # 1. Remove exact duplicate rows
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=[text_col]).copy()
    profile.duplicates_removed = initial_count - len(df_clean)

    # 2. Filter out null, empty, or whitespace-only text
    valid_text_mask = df_clean[text_col].notna() & (df_clean[text_col].astype(str).str.strip() != "")
    profile.missing_text_removed = int((~valid_text_mask).sum())
    df_clean = df_clean[valid_text_mask].copy()

    if df_clean.empty:
        profile.warnings.append("All rows contained empty or missing text.")
        return pd.DataFrame(), profile

    # Standardize column naming: create standard 'Text' column
    df_clean["Text"] = df_clean[text_col].astype(str).str.strip()

    # Process Sentiment column if provided
    if sentiment_col and sentiment_col in df_clean.columns:
        profile.has_ground_truth = True
        raw_labels = df_clean[sentiment_col].dropna().astype(str).str.strip().str.lower()
        unique_raw = set(raw_labels.unique())

        unexpected = [lbl for lbl in unique_raw if lbl not in LABEL_NORMALIZATION_MAP]
        if unexpected:
            profile.unexpected_labels = unexpected
            profile.warnings.append(
                f"Found {len(unexpected)} unrecognized sentiment labels: {unexpected[:5]}."
            )

        # Normalize sentiment
        df_clean["Sentiment"] = df_clean[sentiment_col].apply(normalize_sentiment_label)

        # Retain only rows with recognized sentiment if sentiment_col was chosen
        valid_sentiment_mask = df_clean["Sentiment"].notna()
        if not valid_sentiment_mask.all():
            invalid_cnt = int((~valid_sentiment_mask).sum())
            profile.warnings.append(
                f"Filtered {invalid_cnt} rows with invalid or missing sentiment labels."
            )
            df_clean = df_clean[valid_sentiment_mask].copy()

        # Class distribution and balance check
        counts = df_clean["Sentiment"].value_counts().to_dict()
        profile.class_distribution = {cls: int(counts.get(cls, 0)) for cls in CANONICAL_SENTIMENTS}
        total_valid = sum(profile.class_distribution.values())

        if total_valid > 0:
            profile.class_proportions = {
                cls: round(cnt / total_valid, 4)
                for cls, cnt in profile.class_distribution.items()
            }

            # Check if any class has < 10% representation (severe imbalance)
            min_prop = min(profile.class_proportions.values()) if profile.class_proportions else 0
            if min_prop < 0.10:
                profile.is_severely_imbalanced = True
                profile.warnings.append(
                    f"Severe class imbalance detected: minimum class proportion is {min_prop*100:.1f}%."
                )

            # Check minimum samples for stratified splitting
            min_count = min(profile.class_distribution.values()) if profile.class_distribution else 0
            if min_count < MIN_SAMPLES_PER_CLASS_FOR_STRATIFY:
                profile.can_stratify = False
                profile.warnings.append(
                    f"A class has fewer than {MIN_SAMPLES_PER_CLASS_FOR_STRATIFY} examples; stratification disabled."
                )
    else:
        profile.has_ground_truth = False

    # Standardize Platform
    if platform_col and platform_col in df_clean.columns:
        df_clean["Platform"] = df_clean[platform_col].fillna("Social Media").astype(str).str.strip()
    else:
        df_clean["Platform"] = "Social Media"

    # Standardize Date
    if date_col and date_col in df_clean.columns:
        df_clean["Date"] = pd.to_datetime(df_clean[date_col], errors="coerce").dt.date
        df_clean["Date"] = df_clean["Date"].fillna(pd.Timestamp.now().date())
    else:
        df_clean["Date"] = pd.Timestamp.now().date()

    # Standardize Likes & Shares
    if likes_col and likes_col in df_clean.columns:
        df_clean["Likes"] = pd.to_numeric(df_clean[likes_col], errors="coerce").fillna(0).astype(int)
    else:
        df_clean["Likes"] = 0

    if shares_col and shares_col in df_clean.columns:
        df_clean["Shares"] = pd.to_numeric(df_clean[shares_col], errors="coerce").fillna(0).astype(int)
    else:
        df_clean["Shares"] = 0

    # Standardize Post ID
    if id_col and id_col in df_clean.columns:
        df_clean["Post ID"] = df_clean[id_col].astype(str)
    else:
        df_clean["Post ID"] = [f"POST_{i+1:05d}" for i in range(len(df_clean))]

    profile.retained_rows = len(df_clean)
    return df_clean, profile
