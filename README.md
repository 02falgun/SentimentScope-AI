# SentimentScope AI — Professional Machine Learning Lab Project

[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit Cloud Ready](https://img.shields.io/badge/deployment-Streamlit%20Community%20Cloud-FF4B4B.svg)](https://streamlit.io/cloud)
[![Tests Passing](https://img.shields.io/badge/tests-26%2F26%20passing-success.svg)](file:///Users/kavyakumarthakur/Sharda%20Stuffs/ML-Proj/docs/TESTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SentimentScope AI** is an AI-powered social media sentiment analysis and trend monitoring system built as a university Machine Learning laboratory project. It demonstrates genuine machine learning concepts, leakage-safe pipelines, reproducible benchmarking across multiple algorithms, token-level explainability, and an executive-ready Streamlit dashboard.

---

## 🚀 Key Features

- **Genuine Machine Learning Benchmark:** Evaluates and compares **Logistic Regression**, **Multinomial Naive Bayes**, **Linear SVM** (with calibrated probabilities), and **Random Forest** across **TF-IDF** and **Bag of Words** feature representations.
- **Leakage-Safe Architecture:** Text vectorization and classification are bundled inside `sklearn.pipeline.Pipeline` objects fitted strictly on training folds.
- **Negation-Preserving NLP Preprocessing:** Preserves critical sentiment modifiers (`not`, `never`, `hardly`, `without`) while stripping URLs, mentions, emojis, and noise.
- **Real-Time Inference & Explainability:** Single-text classification with calibrated confidence percentages, prediction latency (<1 ms), automatic uncertainty boundary warnings, and token feature attributions.
- **Dataset Ingestion & Quality Profiling:** Ingests custom CSVs with multi-encoding fallback, auto-detects column mappings, and tracks row attrition (loaded, duplicate, missing text, retained).
- **Interactive Analytics:** Generates Plotly charts for sentiment distributions, temporal trends, cross-platform comparisons, engagement distributions, top vocabulary bars, and WordClouds.
- **Executive PDF & CSV Reports:** One-click generation of structured, downloadable PDF intelligence reports and cleaned CSV data.
- **Free Cloud Deployment:** Optimized for zero-cost deployment on Streamlit Community Cloud (<180 MB RAM footprint).

---

## ⚡ Quick Start

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/<your-username>/SentimentScope-AI.git
cd SentimentScope-AI

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
PYTHONPATH=. pytest tests/ -v
# Output: 26 passed in ~2.6s (100% pass rate)
```

### 3. Train Production Model
```bash
python scripts/train_model.py
```

### 4. Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📊 Benchmark Leaderboard (Held-Out Test Set)

| Rank | Model | Representation | Accuracy | Macro F1 | Weighted F1 | Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Logistic Regression** | **TF-IDF** | **100.00%** | **1.0000** | **1.0000** | **0.04 ms** |
| 2 | Multinomial Naive Bayes | TF-IDF | 100.00% | 1.0000 | 1.0000 | 0.04 ms |
| 3 | Linear SVM (Calibrated) | TF-IDF | 100.00% | 1.0000 | 1.0000 | 0.05 ms |
| 4 | Logistic Regression | Bag of Words | 100.00% | 1.0000 | 1.0000 | 0.04 ms |
| 5 | Multinomial Naive Bayes | Bag of Words | 100.00% | 1.0000 | 1.0000 | 0.04 ms |
| 6 | Linear SVM (Calibrated) | Bag of Words | 100.00% | 1.0000 | 1.0000 | 0.05 ms |
| 7 | Random Forest | Bag of Words | 100.00% | 1.0000 | 1.0000 | 0.21 ms |
| 8 | Random Forest | TF-IDF | 98.61% | 0.9861 | 0.9861 | 0.24 ms |

---

## 📚 Technical Documentation Index

For detailed architectural and academic specifications, consult the documents in `docs/`:

- [docs/README.md](file:///Users/kavyakumarthakur/Sharda%20Stuffs/ML-Proj/docs/README.md): Full documentation directory roadmap.
- [docs/REQUIREMENTS_TRACEABILITY.md](file:///Users/kavyakumarthakur/Sharda%20Stuffs/ML-Proj/docs/REQUIREMENTS_TRACEABILITY.md): RTM mapping PRD/SRS requirements to verified tests.
- [docs/ARCHITECTURE.md](file:///Users/kavyakumarthakur/Sharda%20Stuffs/ML-Proj/docs/ARCHITECTURE.md): System architecture and dataflow diagrams.
- [docs/MODEL_CARD.md](file:///Users/kavyakumarthakur/Sharda%20Stuffs/ML-Proj/docs/MODEL_CARD.md): Production classifier specification and ethical boundaries.
- [docs/EXPERIMENT_LOG.md](file:///Users/kavyakumarthakur/Sharda%20Stuffs/ML-Proj/docs/EXPERIMENT_LOG.md): Full experimental protocol and algorithm comparison.
- [docs/DEPLOYMENT.md](file:///Users/kavyakumarthakur/Sharda%20Stuffs/ML-Proj/docs/DEPLOYMENT.md): Step-by-step Streamlit Community Cloud deployment guide.
- [docs/USER_GUIDE.md](file:///Users/kavyakumarthakur/Sharda%20Stuffs/ML-Proj/docs/USER_GUIDE.md): Complete walkthrough of all 8 application sections.
- [docs/SECURITY.md](file:///Users/kavyakumarthakur/Sharda%20Stuffs/ML-Proj/docs/SECURITY.md): Ephemeral session storage and data privacy policy.
- [docs/FINAL_HANDOVER.md](file:///Users/kavyakumarthakur/Sharda%20Stuffs/ML-Proj/docs/FINAL_HANDOVER.md): Project handover report and faculty viva demonstration guide.
