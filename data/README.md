# SentimentScope AI — Data Directory

## Structure

- `sample/`: Contains the verified demonstration dataset `social_media_sentiment_sample.csv`.
- Uploaded user datasets are processed **in-memory** in the active Streamlit session and are **not** persisted to this directory, in strict compliance with the system's security and privacy policy (see `docs/SECURITY.md`).

## Standard Schema

Every dataset processed by SentimentScope AI maps to the following standard columns:

| Column | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `Post ID` | Integer / String | Optional (auto-generated if absent) | Unique post identifier |
| `Text` | String | **Mandatory** | Raw post content |
| `Platform` | String | Optional (defaults to "Social Media") | Source platform (e.g. Twitter, Reddit, LinkedIn) |
| `Date` | Date / String | Optional (defaults to analysis timestamp) | Post publication date |
| `Likes` | Integer | Optional (defaults to 0) | Engagement like count |
| `Shares` | Integer | Optional (defaults to 0) | Engagement share / retweet count |
| `Sentiment` | String | Optional for inference, Mandatory for training | Positive, Neutral, or Negative |
