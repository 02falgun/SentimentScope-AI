# SentimentScope AI — Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.1] - 2026-10-26 — Post-Deployment Audit & Scientific Generalization Release

### Added
- **Scientific 100% Accuracy Investigation:**
  - Implemented `evaluate_template_generalization` in `src/sentimentscope/ml/evaluation.py` to calculate Group K-Fold Cross-Validation isolating unseen base sentence templates.
  - Documented empirical finding: 93.1% base sentence template overlap in standard 80/20 train/test splits explains in-template memorization.
  - Validated out-of-template generalization benchmark: 93.15% Accuracy and 0.9193 Macro F1 across 5 folds.
  - Evaluated zero-shot generalization on 11 real-world social media posts, achieving 90.9% accuracy with calibrated uncertainty intervals.
  - Added interactive scientific audit expander to ML Lab (`src/sentimentscope/ui/pages/ml_lab.py`).
  - Added viva demonstration talking points on template leakage to About page (`src/sentimentscope/ui/pages/about_page.py`).
- **Defensive Engineering & Robustness:**
  - Added safe character capping (`len(text) <= 2000`) in `predict_single_text` (`src/sentimentscope/ml/prediction.py`) to prevent latency/memory spikes on pathological inputs.
  - Added safe empty-dataset handling to `plot_sentiment_donut` and `plot_sentiment_bars` in `src/sentimentscope/analytics/visualizations.py`.
  - Added Ground-Truth vs Model-Prediction distinction and toggling to Home page (`src/sentimentscope/ui/pages/home.py`).
- **Automated QA Expansion:**
  - Added `test_plot_sentiment_empty_data` in `tests/test_analytics.py`.
  - Added `test_evaluate_template_generalization` in `tests/test_training.py`.
  - Added `test_predict_single_text_excessive_length` in `tests/test_prediction.py`.
  - Expanded test suite to **29/29 passing tests** (100% pass rate).
- **Live Deployment Audit:**
  - Directly audited live deployment at `https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/` using browser automation.
  - Confirmed 100% operational health across all 8 dashboard sections.

---

## [1.0.0] - 2026-10-26 — Initial Production Release

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
