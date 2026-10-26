# SentimentScope AI — Streamlit Community Cloud Deployment Guide

This guide provides step-by-step instructions for deploying SentimentScope AI to **Streamlit Community Cloud** (100% free tier), maintaining the live deployment, and running locally.

🌐 **Live Streamlit Community Cloud Deployment:**  
[https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/](https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/)

---

## 1. Local Run Instructions

To run the application locally on your computer:

```bash
# 1. Create a Python 3.11 or 3.12 virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train or verify the production model (takes ~1.5 seconds)
python scripts/train_model.py

# 4. Launch the Streamlit dashboard
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 2. Streamlit Community Cloud Deployment Configuration

### Repository Details
- **GitHub Repository:** `02falgun/SentimentScope-AI`
- **Target Branch:** `main`
- **Entrypoint:** `app.py`
- **Python Version:** 3.12
- **Deployment URL:** [https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/](https://sentimentscope-ai-pjqvcruadb95pi6detimjb.streamlit.app/)

### How Streamlit Cloud Builds the Application
1. Detects `requirements.txt` at the repository root and installs all scientific and UI wheels.
2. Checks `.streamlit/config.toml` for custom theme colors (slate/indigo/emerald) and server configuration.
3. Launches `app.py` headlessly via Streamlit runner.
4. Executes `@st.cache_resource` for the serialized model artifact (`models/sentiment_model.joblib`) and `@st.cache_data` for the sample dataset (`data/sample/social_media_sentiment_sample.csv`).

---

## 3. Cold-Start and Resource Budget Considerations

- **Memory Footprint:** The application and models consume approximately **160 MB** of RAM, well below Streamlit Cloud's 1 GB free-tier limit.
- **Cold-Start Behavior:** Because `models/sentiment_model.joblib` is committed directly to GitHub, the application cold-starts in **< 3 seconds**. Even if the artifact were accidentally deleted, `app.py` automatically trains the winning pipeline on the sample dataset on first launch without crashing.
- **No External Secrets Required:** SentimentScope AI runs self-contained without requiring API keys, cloud tokens, or database connection strings.

---

## 4. How to Update the Live Deployment Manually

When you are ready to update the live Streamlit Cloud app with local changes:
```bash
# 1. Check your status and staged changes
git status

# 2. Commit your changes
git commit -m "feat: audit enhancements and group-CV generalization documentation"

# 3. Push to GitHub main branch
git push origin main
```

Streamlit Community Cloud is connected via webhook to `main`. It will automatically pull the newest commit and rebuild within 60–90 seconds.  
If the app does not automatically rebuild:
1. Open [https://share.streamlit.io/](https://share.streamlit.io/).
2. Click the three dots `...` next to `SentimentScope-AI`.
3. Click **"Reboot app"**.

---

## 5. Verified Post-Deployment Smoke Test Checklist

The following items were independently verified on the live Streamlit Community Cloud deployment:
- [x] **Home View:** Status cards display 360 records and model accuracy.
- [x] **Live Prediction:** Sample buttons populate text area; classification returns predicted sentiment, confidence %, latency (<1ms), and token attributions in <2 seconds.
- [x] **Dataset Explorer:** Custom CSV loading, column auto-mapping, row attrition metrics, and batch inference operational.
- [x] **ML Lab:** 8-entry model leaderboard, confusion matrix heatmap, ROC curves, and scientific generalization expander render cleanly.
- [x] **Visual Analytics:** All Plotly charts (donut, bars, daily trends, wordcloud, engagement, post lengths) render inline.
- [x] **Keyword Search:** Polarity index calculation and query matching table function properly.
- [x] **Reports:** "Download Executive Report (PDF)" compiles valid PDF binary buffer.
- [x] **About & Architecture:** System diagram and viva demonstration guide render correctly.
