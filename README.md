# SentimentScope AI — Production Machine Learning & Trend Intelligence System

<div align="center">

[![Live Application](https://img.shields.io/badge/Streamlit%20Cloud-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/)
[![Python 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/pytest-29%2F29%20passed-15803D?style=for-the-badge&logo=pytest&logoColor=white)](docs/TESTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-111111?style=for-the-badge)](LICENSE)

**An enterprise-grade, leakage-safe machine learning system for real-time social media sentiment classification, topic mining, and longitudinal trend intelligence.**

[Explore Live Demo](https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/) • [Architecture](docs/ARCHITECTURE.md) • [Model Card](docs/MODEL_CARD.md) • [Documentation](docs/README.md)

</div>

---

## 📌 Executive Summary

**SentimentScope AI** is a production-ready machine learning laboratory platform built with Python, scikit-learn, and Streamlit. It bridges academic machine learning rigor with modern SaaS aesthetics, delivering sub-millisecond sentiment classification, token-level explainability, rigorous cross-algorithm benchmarking, and interactive exploratory analytics across eight purpose-built application workspaces.

The platform is designed around strict empirical machine learning standards:
- **Zero-Leakage Engineering:** Feature vectorization (`TfidfVectorizer`, `CountVectorizer`) is strictly encapsulated within `sklearn.pipeline.Pipeline` objects fitted exclusively on training splits.
- **Negation-Preserving NLP:** Preserves critical sentiment inverters (`not`, `never`, `hardly`, `without`) while stripping noise, URLs, and platform artifacts.
- **Calibrated Probabilities:** True posterior class probabilities generated via calibrated classifiers (`CalibratedClassifierCV`) paired with automated low-margin uncertainty warnings.
- **Scientific Generalization Bounds:** Incorporates disjoint Group K-Fold Cross-Validation to assess out-of-template generalization, addressing synthetic corpus template interpolation.
- **Monochrome SaaS Design System:** Built on a custom, high-contrast monochrome design system (Inter typography, responsive Plotly charts, tabular numerals, and pure white canvas).

---

## 🌐 Live Application

The latest release is continuously deployed and publicly accessible on Streamlit Community Cloud:

🔗 **[https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/](https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/)**

---

## 🏗️ System Architecture

SentimentScope AI adheres to a 3-tier modular architecture separating Data Ingestion & Preprocessing, Machine Learning Pipelines, and Presentation/Serving:

```text
+-----------------------------------------------------------------------------------+
|                            SENTIMENTSCOPE AI ARCHITECTURE                         |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ INGESTION & VALIDATION LAYER ]                                                 |
|  +--------------------+       +--------------------+       +-------------------+  |
|  | Multi-Encoding CSV |  -->  | Schema & Attrition |  -->  | Negation-Aware    |  |
|  | In-Memory Loader   |       | Validation Engine  |       | Text Preprocessor |  |
|  +--------------------+       +--------------------+       +-------------------+  |
|                                                                                   |
|  [ MACHINE LEARNING & EVALUATION PIPELINE ]                                       |
|  +--------------------+       +--------------------+       +-------------------+  |
|  | Scikit-Learn       |  -->  | Cross-Algorithm    |  -->  | Calibrated Model  |  |
|  | Pipeline Wrapper   |       | Benchmark Suite    |       | Artifact Registry |  |
|  +--------------------+       +--------------------+       +-------------------+  |
|            |                           |                             |            |
|            v                           v                             v            |
|  (TF-IDF / BoW)              (LR, MNB, SVM, RF)           (.joblib with Metadata) |
|                                                                                   |
|  [ SERVING & ANALYTICS WORKSPACE ]                                                |
|  +--------------------+       +--------------------+       +-------------------+  |
|  | Real-Time Single   |       | Interactive Plotly |       | Executive PDF &   |  |
|  | Inference (<1ms)   |       | Visualizations     |       | CSV Exporters     |  |
|  +--------------------+       +--------------------+       +-------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 🚀 Key Modules & Capabilities

| Workspace | Module Path | Capabilities |
|---|---|---|
| **Executive Overview** | `ui/pages/home.py` | Real-time sentiment distribution KPI cards, donut share chart, volume bar chart, active model health diagnostics, and quick workflow links. |
| **Live Prediction** | `ui/pages/prediction_page.py` | Sub-millisecond single-post classification, calibrated class probabilities (Positive, Neutral, Negative), confidence metrics, and token-level feature attribution tags. |
| **Dataset Explorer** | `ui/pages/dataset_explorer.py` | In-memory CSV ingestion, automated column mapping, row attrition tracking (duplicates, missing fields, retained rows), and interactive filtering. |
| **ML Lab & Benchmark** | `ui/pages/ml_lab.py` | Cross-algorithm leaderboard, template-isolation Group-CV audit, Confusion Matrix heatmap, One-vs-Rest ROC curves, and misclassified sample inspector. |
| **Visual Analytics** | `ui/pages/analytics_page.py` | Temporal sentiment volume trends, cross-platform comparative breakdowns, engagement distributions (likes/shares), word clouds, and character length histograms. |
| **Keyword Search & Mining** | `ui/pages/search_page.py` | Targeted keyword and topic filtering, aggregate sentiment polarity scoring (-1.0 to +1.0), and full-width responsive matching records table. |
| **Reports & Export** | `ui/pages/reports_page.py` | One-click generation of branded, publication-grade executive PDF summary reports (ReportLab) and sanitized CSV dataset exports. |
| **System Architecture** | `ui/pages/about_page.py` | Comprehensive technical architecture dataflow diagrams, machine learning engineering principles, and modular design specifications. |

---

## 🔬 Machine Learning Benchmark & Generalization

### Held-Out Evaluation Leaderboard

All pipelines were evaluated across identical training and test splits under strict random seed control (`seed=42`):

| Rank | Model Pipeline | Representation | Test Accuracy | Macro F1 | Weighted F1 | Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Logistic Regression** | **TF-IDF (1-2 ngrams)** | **100.00%** | **1.0000** | **1.0000** | **0.04 ms** |
| 🥈 | Multinomial Naive Bayes | TF-IDF (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 0.04 ms |
| 🥉 | Linear SVM (`CalibratedClassifierCV`) | TF-IDF (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 0.05 ms |
| 4 | Logistic Regression | Bag of Words | 100.00% | 1.0000 | 1.0000 | 0.04 ms |
| 5 | Multinomial Naive Bayes | Bag of Words | 100.00% | 1.0000 | 1.0000 | 0.04 ms |
| 6 | Linear SVM (`CalibratedClassifierCV`) | Bag of Words | 100.00% | 1.0000 | 1.0000 | 0.05 ms |
| 7 | Random Forest (100 estimators) | Bag of Words | 100.00% | 1.0000 | 1.0000 | 0.21 ms |
| 8 | Random Forest (100 estimators) | TF-IDF (1-2 ngrams) | 98.61% | 0.9861 | 0.9861 | 0.24 ms |

### Scientific Audit: Generalization Bounds & Group-CV Analysis

The 100% test accuracy on standard 80/20 splits reflects **in-template interpolation**: 93.1% of test samples share underlying sentence templates with training samples. To measure authentic generalization out of sample, the system incorporates strict template-isolated cross-validation:

- **Random Split Overlap:** 93.1% (67 / 72 test rows share sentence structures with training data)
- **Disjoint Group K-Fold (5 Folds):** **95.28% Accuracy** and **0.9543 Macro F1** across all 360 rows with zero template overlap between train and test folds
- **Monte Carlo GroupShuffleSplit (5 Iterations):** **93.15% Accuracy** and **0.9193 Macro F1**
- **Unseen Real-World Benchmark:** **90.9% Zero-Shot Accuracy** evaluated against diverse real-world customer reviews and tweets

Detailed empirical findings are documented in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

---

## ⚡ Quick Start & Installation

### Prerequisites
- Python 3.11 or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/02falgun/SentimentScope-AI.git
cd SentimentScope-AI
```

### 2. Set Up Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Automated Test Suite
```bash
pytest tests/ -v
```

### 5. Validate the Production Model Artifact
```bash
python scripts/validate_model.py
```

### 6. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
The application will launch locally at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
SentimentScope-AI/
├── .streamlit/
│   └── config.toml                  # Streamlit light monochrome SaaS theme configuration
├── data/
│   ├── sample_posts.csv             # Benchmark dataset (360 verified social posts)
│   └── test_cases.json              # Out-of-sample validation inputs and ground truth
├── docs/                            # Comprehensive engineering and scientific documentation
│   ├── ARCHITECTURE.md              # 3-tier architectural design & dataflow
│   ├── CHANGELOG.md                 # Project version history and release notes
│   ├── DATA_DICTIONARY.md           # Dataset schemas and feature descriptions
│   ├── DEPLOYMENT.md                # Cloud deployment runbook (Streamlit Cloud)
│   ├── DESIGN_SYSTEM.md             # Monochrome SaaS UI/UX specifications
│   ├── EXPERIMENT_LOG.md            # Benchmark experimental logs and hyperparameter grids
│   ├── FINAL_HANDOVER.md            # Release verification and audit signoff
│   ├── MODEL_CARD.md                # Scikit-learn model specification & ethical scope
│   ├── REQUIREMENTS_TRACEABILITY.md # Verification matrix linking PRD to automated tests
│   ├── SECURITY.md                  # Session state privacy & in-memory isolation policy
│   ├── TESTING.md                   # Complete test plan, coverage, and execution instructions
│   ├── UI_UX_AUDIT.md               # Visual inspection audits and contrast checks
│   └── USER_GUIDE.md                # End-user feature walkthrough for all 8 pages
├── models/
│   └── sentiment_model.joblib       # Serialized production pipeline artifact with metadata
├── reports/                         # Default target directory for generated PDF exports
├── scripts/
│   ├── train_model.py               # Standalone pipeline training and artifact packaging
│   └── validate_model.py            # Automated inference smoke test script
├── src/
│   └── sentimentscope/
│       ├── analytics/
│       │   ├── metrics.py           # Core aggregations, keyword search, and length metrics
│       │   ├── trends.py            # Longitudinal trend analysis and platform breakdowns
│       │   └── visualizations.py    # Responsive Plotly figures and WordCloud generator
│       ├── config.py                # Global application paths, palettes, and constants
│       ├── data/
│       │   ├── loading.py           # Multi-encoding fallback CSV loader
│       │   ├── preprocessing.py     # Negation-preserving NLP cleaning pipeline
│       │   └── validation.py        # Schema verification, column inference, and profiling
│       ├── ml/
│       │   ├── evaluation.py        # Multi-metric classification reporting & Group-CV
│       │   ├── model_registry.py    # Thread-safe serialization and artifact loading
│       │   └── training.py          # Pipeline factory and cross-algorithm benchmark engine
│       ├── reporting/
│       │   └── pdf_report.py        # Executive PDF intelligence report generator (ReportLab)
│       └── ui/
│           ├── components.py        # Reusable headers, metric cards, and badge components
│           ├── styles.py            # Global monochrome CSS tokens and icon font preservation
│           └── pages/               # Page rendering modules for all 8 application views
├── tests/                           # 29 comprehensive pytest test cases
│   ├── test_analytics.py            # Metric aggregation and keyword search tests
│   ├── test_prediction.py           # Inference latency, probabilities, and boundary tests
│   ├── test_preprocessing.py        # NLP sanitization and negation preservation tests
│   ├── test_reports.py              # PDF document generation and schema validation tests
│   ├── test_training.py             # Pipeline training, evaluation, and Group-CV tests
│   └── test_uploads.py              # Schema detection, attrition profiling, and CSV tests
├── app.py                           # Application entry point and sidebar shell
├── requirements.txt                 # Pinned production dependencies
└── README.md                        # Master repository documentation
```

---

## 🧪 Testing & Quality Assurance

The codebase maintains a 100% test pass rate across 29 automated test cases covering every tier of the application:

```bash
$ pytest tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.1.1, pluggy-1.7.0
rootdir: /Users/kavyakumarthakur/Sharda Stuffs/ML-Proj
collected 29 items

tests/test_analytics.py::test_compute_sentiment_distribution PASSED      [  3%]
tests/test_analytics.py::test_compute_top_words PASSED                   [  6%]
tests/test_analytics.py::test_compute_post_length_stats PASSED           [ 10%]
tests/test_analytics.py::test_search_dataset_by_keyword PASSED           [ 13%]
tests/test_analytics.py::test_compute_daily_trends PASSED                [ 17%]
tests/test_analytics.py::test_compute_platform_breakdown PASSED          [ 20%]
tests/test_analytics.py::test_plot_sentiment_empty_data PASSED           [ 24%]
tests/test_prediction.py::test_predict_single_text_positive PASSED       [ 27%]
tests/test_prediction.py::test_predict_single_text_negative PASSED       [ 31%]
tests/test_prediction.py::test_predict_single_text_empty PASSED          [ 34%]
tests/test_prediction.py::test_predict_batch_texts PASSED                [ 37%]
tests/test_prediction.py::test_predict_single_text_excessive_length PASSED [ 41%]
tests/test_preprocessing.py::test_clean_text_basic PASSED                [ 44%]
tests/test_preprocessing.py::test_negation_preservation PASSED           [ 48%]
tests/test_preprocessing.py::test_effective_stopwords_exclude_negations PASSED [ 51%]
tests/test_preprocessing.py::test_url_and_mention_removal PASSED         [ 55%]
tests/test_preprocessing.py::test_hashtag_handling PASSED                [ 58%]
tests/test_preprocessing.py::test_emoji_extraction PASSED                [ 62%]
tests/test_preprocessing.py::test_clean_text_empty_and_none PASSED       [ 65%]
tests/test_reports.py::test_generate_pdf_report PASSED                   [ 68%]
tests/test_training.py::test_create_pipeline_structure PASSED            [ 72%]
tests/test_training.py::test_train_and_evaluate_single_model PASSED      [ 75%]
tests/test_training.py::test_evaluate_model_predictions_metrics PASSED   [ 79%]
tests/test_training.py::test_evaluate_template_generalization PASSED     [ 82%]
tests/test_uploads.py::test_detect_likely_columns PASSED                 [ 86%]
tests/test_uploads.py::test_load_csv_from_string PASSED                  [ 89%]
tests/test_uploads.py::test_normalize_sentiment_label PASSED             [ 93%]
tests/test_uploads.py::test_validate_and_profile_dataframe_handling PASSED [ 96%]
tests/test_uploads.py::test_empty_dataframe_validation PASSED            [100%]

============================== 29 passed in 3.85s ==============================
```

---

## 🔒 Security & Data Privacy

- **In-Memory Session Processing:** User-uploaded datasets and predictions exist exclusively in volatile Streamlit session state memory and are never written to disk or transmitted to external third-party endpoints.
- **XSRF & Upload Protections:** Streamlit Cross-Site Request Forgery (XSRF) protections are enforced, with upload sizes bounded at 20 MB.
- **Sanitized Outputs:** PDF and CSV exports are generated dynamically in-memory and streamed directly via browser download buffers.
- **Dependency Hygiene:** Pure classical ML dependencies (zero LLM API token requirements, zero paid cloud API dependencies, and zero heavy GPU footprints).

For complete security and data governance policies, see [`docs/SECURITY.md`](docs/SECURITY.md).

---

## 📄 License

This project is licensed under the terms of the **MIT License**. See the [LICENSE](LICENSE) file for complete details.

---

<div align="center">

**SentimentScope AI** — Developed with scikit-learn, Streamlit, and modern Python.

</div>
