# SentimentScope AI — Project Audit & Baseline Report

**Date:** July 2026 / October 2026 Baseline  
**Project:** SentimentScope AI (AI-Powered Social Media Sentiment Analysis and Trend Monitoring System)  
**Author:** Senior ML Engineer & Architect  
**Status:** Audit Complete — Baseline Established  

---

## 1. Executive Summary

SentimentScope AI is a machine learning laboratory project designed to provide sentiment classification (Positive, Neutral, Negative) and social media trend monitoring with a Streamlit interface.

At the inception of this audit, the repository contained exclusively:
- `PRD_SRS_Sentiment_Analysis_System.docx`: Authoritative Product Requirements Document and Software Requirements Specification (v1.0, July 2026).

No legacy code, models, or datasets were previously committed. This clean slate provides the opportunity to build the system strictly according to the PRD/SRS requirements and the target modular architecture without accumulating technical debt.

---

## 2. Inventory of Repository Artifacts

| Component | Found Item | State & Findings |
| :--- | :--- | :--- |
| **Requirements Document** | `PRD_SRS_Sentiment_Analysis_System.docx` | Comprehensive PRD/SRS defining 18 Functional Requirements (FR-1 to FR-18), non-functional requirements, data schema, and acceptance criteria. |
| **Existing Code / Scripts** | None | Must implement modular structure under `src/sentimentscope/`, `app.py`, and training/validation scripts. |
| **Datasets** | None | Must provide genuine, curated sample social media dataset conforming to schema: `Post ID`, `Text`, `Platform`, `Date`, `Likes`, `Shares`, `Sentiment`. |
| **Model Artifacts** | None | Must implement model training pipeline comparing Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Random Forest using TF-IDF & BoW representations, with reproducible seeds and serialized artifact generation. |
| **Tests** | None | Must establish comprehensive pytest suite covering preprocessing, training, evaluation, prediction, uploads, analytics, and reports. |
| **Deployment Config** | None | Must configure `.streamlit/config.toml`, `.gitignore`, `requirements.txt`, and GitHub Actions workflow. |

---

## 3. Requirements Extraction & Baseline Status

From `PRD_SRS_Sentiment_Analysis_System.docx`:

- **FR-1**: Import CSV dataset and remove duplicate posts (Status: Pending Implementation).
- **FR-2**: Handle missing values gracefully (Status: Pending Implementation).
- **FR-3**: Text cleaning (lowercasing, URL removal, hashtags/mentions, emojis, punctuation) (Status: Pending Implementation).
- **FR-4**: Stop word removal (preserving negations), tokenization, lemmatization (Status: Pending Implementation).
- **FR-5**: Exploratory visualizations (sentiment distribution, word cloud, post length distribution) (Status: Pending Implementation).
- **FR-6**: Feature representations (TF-IDF primary, BoW optional comparison) (Status: Pending Implementation).
- **FR-7**: Model training: Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest (Status: Pending Implementation).
- **FR-8**: Model evaluation: Accuracy, Precision, Recall, F1-score (macro and weighted), Confusion Matrix, ROC-AUC (multiclass/binary valid cases) (Status: Pending Implementation).
- **FR-9**: Best model selection and persistence (leakage-safe pipeline artifact) (Status: Pending Implementation).
- **FR-10**: Live single-text prediction with honest confidence/probability/decision-scores and uncertainty indicators (Status: Pending Implementation).
- **FR-11**: Dashboard Home view with metric cards and summary overview (Status: Pending Implementation).
- **FR-12**: Pie/donut and bar charts for sentiment distribution (Status: Pending Implementation).
- **FR-13**: Top positive and negative words and posts (Status: Pending Implementation).
- **FR-14**: Daily sentiment trends and platform-wise comparison (Status: Pending Implementation).
- **FR-15**: Keyword search with record counts and opinion breakdown (Status: Pending Implementation).
- **FR-16**: Trending hashtags and emoji analysis (Status: Pending Implementation).
- **FR-17**: Custom CSV upload with automatic and manual column mapping and data profiling (Status: Pending Implementation).
- **FR-18**: PDF report generation and CSV download (Status: Pending Implementation).

---

## 4. Key Architectural & Technical Decisions

1. **Python Environment & Compatibility**: Standardized on Python 3.12+ with pure-python/wheel dependencies (`scikit-learn`, `pandas`, `numpy`, `streamlit`, `matplotlib`, `plotly`, `fpdf2`, `wordcloud`, `joblib`, `pytest`) to ensure zero-cost, seamless deployment on Streamlit Community Cloud.
2. **Leakage-Safe Pipelines**: Feature extractors (`TfidfVectorizer`) and classifiers must be bound into `sklearn.pipeline.Pipeline` objects fitted strictly on training folds.
3. **Probability & Confidence Semantics**: Logistic Regression and Multinomial Naive Bayes output calibrated posterior probabilities. Linear SVM does not by default output calibrated probabilities; the UI will explicitly present SVM decision-function margins rather than misrepresenting them as probabilities.
4. **Negation Preservation**: Stopword removal must NOT strip critical negation words (e.g., `not`, `never`, `no`, `neither`, `hardly`, `barely`, `without`) because stripping them flips positive/negative polarity.
5. **No Persistent Session Leakage**: Uploaded custom CSVs will be processed in-memory during user sessions without permanent disk persistence, conforming to PRD Section 2.5 Security requirements.

---

## 5. Audit Exit Criteria Checklist

- [x] Repository structure inspected.
- [x] PRD/SRS extracted, requirements mapped to functional IDs.
- [x] Environment and dependency constraints identified.
- [x] Gaps, risks, and execution plan documented.
