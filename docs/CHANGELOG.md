# SentimentScope AI — Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-10-26

### Added
- **Phase 0 Baseline & Repository Audit:**
  - Audited PRD/SRS v1.0 specifications and extracted 18 Functional Requirements.
  - Initialized git repository with `.gitignore`.
  - Created initial audit reports and requirements traceability matrix.
- **Phase 1 Modular Architecture:**
  - Structured `src/sentimentscope/` into `data`, `ml`, `analytics`, `reporting`, and `ui/pages`.
  - Centralized path constants, canonical sentiment classes, and color tokens in `config.py`.
  - Configured `.streamlit/config.toml` and `.github/workflows/tests.yml`.
- **Phase 2 Ingestion, Preprocessing & Quality:**
  - Created `src/sentimentscope/data/preprocessing.py` with URL/mention stripping, hashtag extraction, emoji handling, negation preservation (`NEGATION_TOKENS`), and lemmatization.
  - Implemented multi-encoding CSV loading and token-based column auto-detection in `src/sentimentscope/data/loading.py`.
  - Implemented row attrition tracking, label normalization, and severe class imbalance alerts in `src/sentimentscope/data/validation.py`.
  - Curated 360-post balanced benchmark dataset in `data/sample/social_media_sentiment_sample.csv`.
- **Phase 3 Machine Learning Benchmark & Persistence:**
  - Built leakage-safe `sklearn.pipeline.Pipeline` combining TF-IDF / Bag of Words vectorization with 4 classifiers: Logistic Regression, Multinomial Naive Bayes, Linear SVM (calibrated), and Random Forest.
  - Implemented rigorous evaluation metrics (Accuracy, Macro/Weighted F1, Confusion Matrix, ROC-AUC) in `src/sentimentscope/ml/evaluation.py`.
  - Produced serialized production artifact `models/sentiment_model.joblib` and benchmark history `models/experiment_history.json`.
  - Created CLI scripts: `scripts/train_model.py` and `scripts/validate_model.py`.
- **Phase 4 & 5 User Interface & Advanced Capabilities:**
  - Custom dark glassmorphic CSS styling in `src/sentimentscope/ui/styles.py` and reusable cards in `components.py`.
  - Built 8 interactive sections: Home, Live Prediction, Dataset Explorer, ML Lab, Visual Analytics, Keyword Search, Executive Reports, and About.
  - Implemented token explainability (linear feature attribution) and uncertainty detection bounds.
  - Added misclassified sample inspector and interactive analytics filters.
  - Built binary PDF report compiler in `src/sentimentscope/reporting/pdf_report.py`.
- **Phase 6 Automated Testing:**
  - Built 26 automated unit and integration tests across 6 test modules (`test_preprocessing.py`, `test_uploads.py`, `test_training.py`, `test_prediction.py`, `test_analytics.py`, `test_reports.py`).
  - Achieved 100% pass rate (26/26).
- **Phase 7 & 8 Cloud Readiness & Documentation:**
  - Optimized resource consumption (<180 MB RAM) for free-tier Streamlit Community Cloud.
  - Created complete technical documentation suite under `docs/`.
  - Compiled final handover report in `docs/FINAL_HANDOVER.md`.
