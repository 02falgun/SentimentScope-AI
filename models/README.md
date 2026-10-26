# SentimentScope AI — Models Directory

This directory stores serialized model artifacts and experiment metadata.

## Artifact Specifications

- `sentiment_model.joblib`: The production model artifact. Contains:
  - `pipeline`: Scikit-learn Pipeline (Text Preprocessor/Vectorizer + Trained Classifier).
  - `model_name`: Human-readable identifier (e.g. "Logistic Regression (TF-IDF)").
  - `metrics`: Dictionary of evaluation metrics on held-out test data (Accuracy, Macro F1, Weighted F1, Precision, Recall).
  - `classes`: Array of canonical sentiment classes (`['Negative', 'Neutral', 'Positive']`).
  - `feature_type`: Vectorizer type (`'tfidf'` or `'bow'`).
  - `trained_at`: UTC timestamp string.
  - `random_seed`: Seed used for reproducibility.
  - `is_calibrated`: Boolean indicating whether predict_proba is calibrated.
- `experiment_history.json`: Machine-readable log of all evaluated models, configurations, and test splits for the ML Lab Leaderboard.
