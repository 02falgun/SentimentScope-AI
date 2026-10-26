"""
SentimentScope AI — Data Loading Module
Handles CSV ingestion with multi-encoding fallback, delimiter sniffing,
and intelligent schema column guessing.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


# Column auto-detection candidate names (lowercase)
CANDIDATE_COLUMN_NAMES: Dict[str, List[str]] = {
    "text": ["text", "tweet", "content", "post", "message", "body", "clean_text", "comment", "review", "sentence"],
    "sentiment": ["sentiment", "label", "target", "class", "polarity", "category"],
    "platform": ["platform", "source", "channel", "network", "media", "social_network"],
    "date": ["date", "created_at", "timestamp", "time", "published", "post_date"],
    "likes": ["likes", "favorite_count", "favorite", "upvotes", "score", "like_count"],
    "shares": ["shares", "retweet_count", "retweets", "reposts", "share_count"],
    "id": ["id", "post_id", "tweet_id", "post id", "identifier"],
}

SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "iso-8859-1", "cp1252"]


def detect_likely_column(df_columns: List[str], field_type: str) -> Optional[str]:
    """
    Heuristically matches DataFrame columns to standard schema fields.
    """
    candidates = CANDIDATE_COLUMN_NAMES.get(field_type, [])
    exact_lower_map = {col.lower().strip(): col for col in df_columns}
    normalized_map = {col.lower().strip().replace("_", " "): col for col in df_columns}

    # 1. Exact match on normalized candidates
    for cand in candidates:
        if cand in exact_lower_map:
            return exact_lower_map[cand]
        if cand in normalized_map:
            return normalized_map[cand]

    # 2. Word-level token match (avoids substring collisions, e.g. 'time' inside 'sentiment')
    for cand in candidates:
        for norm_name, orig_name in normalized_map.items():
            tokens = set(norm_name.split())
            if field_type == "text" and ("id" in tokens or norm_name.endswith("id")):
                continue
            if cand in tokens or cand == norm_name:
                return orig_name

    # 3. Substring match fallback only if candidate length >= 4
    for cand in candidates:
        if len(cand) < 4:
            continue
        for norm_name, orig_name in normalized_map.items():
            if field_type == "text" and ("id" in norm_name.split() or norm_name.endswith("id")):
                continue
            if field_type == "date" and "sentiment" in norm_name:
                continue
            if cand in norm_name:
                return orig_name

    return None


def detect_all_likely_columns(df_columns: List[str]) -> Dict[str, Optional[str]]:
    """Detect likely column matches for all standard schema fields."""
    return {
        field_type: detect_likely_column(df_columns, field_type)
        for field_type in CANDIDATE_COLUMN_NAMES
    }


def load_csv(
    file_source: Union[str, Path, io.BytesIO, io.StringIO, Any],
    encoding: Optional[str] = None,
    nrows: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely load a CSV file or buffer into a pandas DataFrame.
    Tries multiple encodings and separators automatically.

    Returns:
        (df, error_message)
    """
    encodings_to_try = [encoding] if encoding else SUPPORTED_ENCODINGS
    last_error = None

    # If file_source is bytes, read it once to allow multiple decode attempts
    raw_bytes: Optional[bytes] = None
    if hasattr(file_source, "read"):
        try:
            raw_bytes = file_source.read()
            if isinstance(raw_bytes, str):
                raw_bytes = raw_bytes.encode("utf-8")
        except Exception as e:
            return None, f"Failed reading input stream: {str(e)}"
    elif isinstance(file_source, (str, Path)):
        p = Path(file_source)
        if not p.exists():
            return None, f"File not found: {p}"
        try:
            with open(p, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            return None, f"Could not open file: {str(e)}"

    if raw_bytes is None:
        return None, "No data stream available to read."

    if len(raw_bytes) == 0:
        return pd.DataFrame(), None

    for enc in encodings_to_try:
        if enc is None:
            continue
        try:
            # First try comma separator
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding=enc,
                sep=None,
                engine="python",
                nrows=nrows,
                on_bad_lines="skip",
            )
            # Basic sanity check
            if df is not None:
                return df, None
        except Exception as e:
            last_error = str(e)
            continue

    return None, f"Failed to parse CSV with supported encodings. Error details: {last_error}"
