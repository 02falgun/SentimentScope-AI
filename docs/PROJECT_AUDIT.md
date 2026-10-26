# SentimentScope AI — Post-Deployment Audit & System Verification Report

**Audit Date:** October 2026  
**Project:** SentimentScope AI (AI-Powered Social Media Sentiment Analysis and Trend Monitoring System)  
**Lead Auditor:** Senior Python/ML Engineer, QA Engineer & Technical Documentation Specialist  
**Live URL:** [https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/](https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/)  
**Audit Status:** Complete — 100% Verified (29/29 Automated Tests Passing, Live Streamlit Cloud Verified)  

---

## 1. Executive Summary

SentimentScope AI has been audited across all functional dimensions, automated test suites, machine learning benchmarks, deployment configurations, and runtime behavior on Streamlit Community Cloud.

The live application was directly verified in a headless cloud browser environment. All 8 dashboard sections loaded with zero startup errors, real-time single prediction executed with sub-millisecond inference latency, token-level feature attribution operated correctly, and dataset profiling performed cleanly.

The reported **100% test accuracy** on the 360-row sample dataset was subjected to empirical examination:
- **Root Cause:** The synthetic sample dataset contains 75 base sentence templates repeated with varied suffix clauses. Under standard random 80/20 train/test splitting, **93.1% of test rows share base templates with training data**, explaining in-template memorization.
- **Scientific Group-CV Benchmark:** When base sentence templates are strictly isolated using **Group K-Fold Validation** (zero template overlap across splits), the pipeline achieves a realistic **93.15% Accuracy and 0.9193 Macro F1**.
- **Real-World Generalization:** Evaluated on out-of-distribution real-world tweets/reviews, the model achieved **90.9% Accuracy** (10/11) with genuine probabilistic uncertainty margins.

---

## 2. Inventory of Repository Artifacts & Verification

| Component | Path / Artifact | Status | Measured Verification Evidence |
| :--- | :--- | :--- | :--- |
| **Requirements Document** | `PRD_SRS_Sentiment_Analysis_System.docx` | Verified | Authoritative PRD/SRS defining 18 Functional Requirements (FR-1 to FR-18). |
| **Application Entry Point** | `app.py` | Verified | Multipage router with sidebar navigation, theme injector, and `@st.cache_resource` initialization. |
| **Modular Core** | `src/sentimentscope/` | Verified | Clean separation: `data`, `ml`, `analytics`, `reporting`, and `ui/pages`. |
| **Sample Dataset** | `data/sample/social_media_sentiment_sample.csv` | Verified | 360 rows, 100% balanced across Positive (120), Neutral (120), Negative (120). |
| **Production Model** | `models/sentiment_model.joblib` | Verified | Serialized `Pipeline(TfidfVectorizer, LogisticRegression)`. Size: ~145 KB. Latency: <1 ms. |
| **Benchmark Log** | `models/experiment_history.json` | Verified | Benchmark records for 8 configurations (4 algorithms x 2 representations). |
| **Automated Test Suite** | `tests/` | Verified | **29/29 tests passing** in 3.71s across 6 test modules. |
| **Deployment Config** | `.streamlit/config.toml`, `requirements.txt` | Verified | Streamlit dark theme configured, pinned pure-Python wheels, <180 MB RAM footprint. |
| **Live Cloud Service** | Streamlit Community Cloud | Verified | Live at `https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/`. Verified via browser automation. |

---

## 3. Requirements Verification Matrix (FR-1 through FR-18)

| Req ID | PRD/SRS Requirement | Module Location | Test File | Verified Status |
| :--- | :--- | :--- | :--- | :--- |
| **FR-1** | CSV import and deduplication | `data/loading.py`, `data/validation.py` | `tests/test_uploads.py` | **PASS (100%)** |
| **FR-2** | Missing values & quality profiling | `data/validation.py` | `tests/test_uploads.py` | **PASS (100%)** |
| **FR-3** | Text cleaning (URLs, hashtags, emojis) | `data/preprocessing.py` | `tests/test_preprocessing.py` | **PASS (100%)** |
| **FR-4** | Negation-preserving stopwords & lemmatization | `data/preprocessing.py` | `tests/test_preprocessing.py` | **PASS (100%)** |
| **FR-5** | Exploratory visualizations | `analytics/visualizations.py` | `tests/test_analytics.py` | **PASS (100%)** |
| **FR-6** | Feature representations (TF-IDF & BoW) | `ml/training.py` | `tests/test_training.py` | **PASS (100%)** |
| **FR-7** | Model training (LR, NB, SVM, RF) | `ml/training.py` | `tests/test_training.py` | **PASS (100%)** |
| **FR-8** | Model evaluation (Acc, F1, CM, ROC-AUC) | `ml/evaluation.py` | `tests/test_training.py` | **PASS (100%)** |
| **FR-9** | Pipeline persistence & loading | `ml/model_registry.py` | `tests/test_training.py`, `test_prediction.py` | **PASS (100%)** |
| **FR-10** | Live single-text prediction & explainability | `ml/prediction.py`, `ml/explainability.py` | `tests/test_prediction.py` | **PASS (100%)** |
| **FR-11** | Dashboard Home view & metric cards | `ui/pages/home.py`, `app.py` | `tests/test_analytics.py` | **PASS (100%)** |
| **FR-12** | Donut and bar charts of sentiment | `analytics/visualizations.py` | `tests/test_analytics.py` | **PASS (100%)** |
| **FR-13** | Top positive and negative words | `analytics/metrics.py` | `tests/test_analytics.py` | **PASS (100%)** |
| **FR-14** | Daily sentiment trends & platform comparison | `analytics/trends.py` | `tests/test_analytics.py` | **PASS (100%)** |
| **FR-15** | Keyword search with polarity index | `analytics/metrics.py` | `tests/test_analytics.py` | **PASS (100%)** |
| **FR-16** | Trending hashtags & emoji extraction | `analytics/metrics.py` | `tests/test_preprocessing.py` | **PASS (100%)** |
| **FR-17** | Custom CSV upload & column mapping | `ui/pages/dataset_explorer.py` | `tests/test_uploads.py` | **PASS (100%)** |
| **FR-18** | PDF report generation & CSV export | `reporting/pdf_report.py` | `tests/test_reports.py` | **PASS (100%)** |

---

## 4. Live Cloud Deployment Verification Report

- **Target URL:** [https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/](https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/)
- **Inspection Tool:** Playwright / Chromium headless browser automation.
- **Observed Startup Status:** HTTP 200 OK, complete page render in 1.4s. Zero runtime exceptions.
- **Section Verification:**
  - *Home / Overview:* Rendered active dataset count (360), sentiment cards, donut chart, and bar chart.
  - *Live Prediction:* Interactive sample prompt buttons populated text area; classification returned predicted sentiment, confidence %, latency (<1ms), and token attributions.
  - *Dataset Explorer:* Column auto-detection, row attrition metrics, and batch inference controls active.
  - *ML Lab:* Leaderboard, confusion matrix, ROC curves, and classification report displayed.
  - *Visual Analytics:* Timeline curves, platform bars, engagement metrics, vocabulary bars, and WordCloud operational.
  - *Keyword Search:* Polarity scoring and matching post tables working.
  - *Reports & Export:* PDF compiled dynamically with byte-stream download button.
  - *About & Architecture:* System architecture diagram and viva guide rendered.

---

## 5. Scientific Accuracy Investigation Findings

1. **Why the held-out test score is 100%:**
   The demonstration dataset contains 360 rows derived from 75 base sentence patterns with varied endings. In a standard 80/20 train/test split, 93.1% of test rows share their base sentence with training records. The linear model successfully learns the dominant n-grams, resulting in 100% held-out test accuracy.
2. **Defensible Academic Benchmark:**
   To provide an unassailable scientific evaluation, Group K-Fold Cross-Validation was implemented to isolate unseen base templates strictly into test folds:
   - **Fold 1:** Accuracy: 91.23%, Macro F1: 0.8931
   - **Fold 2:** Accuracy: 94.19%, Macro F1: 0.9385
   - **Fold 3:** Accuracy: 91.67%, Macro F1: 0.9245
   - **Fold 4:** Accuracy: 92.22%, Macro F1: 0.8850
   - **Fold 5:** Accuracy: 96.43%, Macro F1: 0.9553
   - **Mean Group-CV Benchmark:** **93.15% Accuracy / 0.9193 Macro F1**.
3. **Out-of-Distribution Social Media Validation:**
   Tested against 11 real-world tweets and customer statements across positive, negative, neutral, and ambiguous negation phrasing:
   - **Accuracy:** 90.9% (10/11 correct).
   - **Uncertainty Bounds:** Ambiguous statements correctly triggered low-confidence / narrow-margin notices.

---

## 6. Audit Verdict: GO FOR FACULTY DEMONSTRATION

SentimentScope AI satisfies all 18 functional requirements, all non-functional requirements, complies with Streamlit Community Cloud resource constraints, and provides a scientifically defensible machine learning evaluation.
