"""
SentimentScope AI — Text Preprocessing Pipeline
Handles text normalization, URL/mention stripping, hashtag extraction,
emoji processing, negation-preserving stopword removal, and lemmatization.
"""

import re
import string
from typing import List, Optional, Set
import pandas as pd

from src.sentimentscope.config import NEGATION_TOKENS

# Fallback basic English stopwords if NLTK is not downloaded
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "s", "same", "she", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves"
}

# Regex Patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
WHITESPACE_PATTERN = re.compile(r"\s+")

# Initialize lemmatizer safely
_LEMMATIZER = None
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    _LEMMATIZER = WordNetLemmatizer()
except Exception:
    _LEMMATIZER = None


def get_effective_stopwords() -> Set[str]:
    """
    Returns English stopwords excluding negation tokens.
    Guarantees 'not', 'never', 'no', 'hardly' etc. are never removed.
    """
    try:
        from nltk.corpus import stopwords
        base_stops = set(stopwords.words("english"))
    except Exception:
        base_stops = set(DEFAULT_STOPWORDS)

    # Strictly preserve negations
    return base_stops - NEGATION_TOKENS


EFFECTIVE_STOPWORDS = get_effective_stopwords()


def extract_hashtags(text: str) -> List[str]:
    """Extract all hashtag terms (without the leading #) from text."""
    if not isinstance(text, str):
        return []
    return [match.group(1).lower() for match in HASHTAG_PATTERN.finditer(text)]


def extract_emojis(text: str) -> List[str]:
    """Extract individual emojis found in the text."""
    if not isinstance(text, str):
        return []
    return EMOJI_PATTERN.findall(text)


def lemmatize_token(token: str) -> str:
    """Lemmatize a single token using WordNet if available."""
    if _LEMMATIZER is not None:
        try:
            return _LEMMATIZER.lemmatize(token)
        except Exception:
            return token
    return token


def clean_text(
    text: Optional[str],
    remove_stopwords: bool = True,
    apply_lemmatization: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_hashtags: bool = False,
    remove_punctuation: bool = True,
) -> str:
    """
    Preprocess raw social media text into a normalized representation.

    Parameters:
        text: Raw text string.
        remove_stopwords: If True, strips standard stopwords while strictly preserving negations.
        apply_lemmatization: If True, applies WordNet lemmatization.
        remove_urls: If True, removes http(s) URLs.
        remove_mentions: If True, removes @username tags.
        remove_hashtags: If True, removes hashtags completely; if False, strips the '#' symbol
                         keeping the word itself (e.g. #awesome -> awesome).
        remove_punctuation: If True, strips punctuation characters.

    Returns:
        Cleaned, lowercased, and whitespace-normalized string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    s = text.lower()

    if remove_urls:
        s = URL_PATTERN.sub(" ", s)

    if remove_mentions:
        s = MENTION_PATTERN.sub(" ", s)

    if remove_hashtags:
        s = HASHTAG_PATTERN.sub(" ", s)
    else:
        # Keep hashtag term, just strip the '#' symbol
        s = HASHTAG_PATTERN.sub(r" \1 ", s)

    # Strip emojis from cleaned modeling text (emojis are preserved in raw text for analytics)
    s = EMOJI_PATTERN.sub(" ", s)

    if remove_punctuation:
        # Remove punctuation, but keep apostrophes for contracted negations (e.g. don't -> dont or preserved)
        s = s.replace("'", "")
        for char in string.punctuation:
            s = s.replace(char, " ")

    # Tokenize by whitespace
    tokens = [t.strip() for t in WHITESPACE_PATTERN.split(s) if t.strip()]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in EFFECTIVE_STOPWORDS]

    if apply_lemmatization:
        tokens = [lemmatize_token(t) for t in tokens]

    return " ".join(tokens)


def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply standard clean_text across a pandas Series."""
    return series.fillna("").astype(str).apply(clean_text)
