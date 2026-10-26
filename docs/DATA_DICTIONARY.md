# SentimentScope AI — Data Dictionary

## 1. Overview

SentimentScope AI operates on social media microblogging posts. The schema reflects standard attributes commonly provided by public datasets (Sentiment140, Twitter US Airline Sentiment, Kaggle Twitter Sentiment) and platform APIs.

---

## 2. Field Specifications

| Column Name | Data Type | Constraint | Required | Description & Validation Rules |
| :--- | :--- | :--- | :--- | :--- |
| `Post ID` | String | Unique Alphanumeric | Optional | Unique post identifier (e.g. `POST_00001`). If absent in uploaded CSV, the system automatically assigns synthetic sequential IDs (`POST_00001` ... `POST_N`). |
| `Text` | String | Non-empty text | **Mandatory** | The raw social media content. Stripped of leading/trailing whitespace. Rows with null, empty, or whitespace-only text are recorded and discarded during validation. |
| `Platform` | String | Categorical | Optional | Source network (e.g. Twitter, Reddit, LinkedIn, Instagram). Defaults to `"Social Media"` if omitted. |
| `Date` | Date (`YYYY-MM-DD`) | ISO Date | Optional | Date of publication. Parsed safely with `pd.to_datetime(errors='coerce')`. Invalid dates default to the current UTC date. |
| `Likes` | Integer | $\ge 0$ | Optional | Count of user favorites / upvotes. Non-numeric or missing entries default to `0`. |
| `Shares` | Integer | $\ge 0$ | Optional | Count of retweets / shares / reposts. Non-numeric or missing entries default to `0`. |
| `Sentiment` | String | Categorical | Conditional | Ground-truth sentiment label. Required for model training and held-out evaluation; optional for pure inference and exploratory upload. |

---

## 3. Sentiment Label Normalization

Uploaded datasets frequently use non-standard naming conventions for sentiment categories. The system normalizes these using the following mapping:

| Canonical Label | Accepted Raw Aliases (Case-Insensitive) |
| :--- | :--- |
| **Positive** | `positive`, `pos`, `1`, `4` (Sentiment140 positive marker) |
| **Neutral** | `neutral`, `neu`, `0`, `2` |
| **Negative** | `negative`, `neg`, `-1` |

Any unexpected or unrecognized categories are flagged in the Data Quality Profile and excluded from supervised training to prevent corrupting the label space.

---

## 4. Preprocessing Transformations

For machine learning and word frequency calculations, raw text undergoes the following deterministic transformations:
1. **Case Normalization:** Converted to lowercase.
2. **URL Stripping:** Regex pattern `https?://\S+|www\.\S+` replaced with space.
3. **Mention Stripping:** Regex pattern `@\w+` replaced with space.
4. **Hashtag Normalization:** Retains the core keyword while removing the leading `#` (e.g. `#Tech` becomes `tech`). Raw hashtags are preserved separately for social analytics.
5. **Emoji Extraction:** Emojis are captured for social analytics and stripped from the modeling text representation.
6. **Punctuation Stripping:** Standard punctuation removed while preserving apostrophes inside contractions.
7. **Negation-Preserving Stopword Removal:** Standard English stopwords are removed, except for tokens in `NEGATION_TOKENS` (`not`, `never`, `no`, `neither`, `hardly`, `scarcely`, `barely`, `without`, `cannot`).
8. **Lemmatization:** WordNet lemmatizer reduces inflected words to their root forms (e.g. `crashes` $\to$ `crash`).
