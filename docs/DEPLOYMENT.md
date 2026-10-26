# SentimentScope AI — Streamlit Community Cloud Deployment Guide

This guide provides step-by-step instructions for deploying SentimentScope AI to **Streamlit Community Cloud** (100% free tier) and running locally.

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

## 2. Streamlit Community Cloud (Free Tier) Deployment

### Step A: Push Code to a GitHub Repository
1. Initialize git (if not already done) and commit the project files:
   ```bash
   git add .
   git commit -m "Complete SentimentScope AI deployment bundle"
   ```
2. Create a new GitHub repository (Public or Private) named `SentimentScope-AI`.
3. Push your commits to GitHub:
   ```bash
   git remote add origin https://github.com/<your-username>/SentimentScope-AI.git
   git branch -M main
   git push -u origin main
   ```

### Step B: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `<your-username>/SentimentScope-AI`.
4. Set **Branch**: `main`.
5. Set **Main file path**: `app.py`.
6. Click **"Deploy!"**.

Streamlit Cloud will automatically detect `requirements.txt`, install all scientific packages, and launch the application.

---

## 3. Cold-Start and Resource Budget Considerations

- **Memory Footprint:** The application and models consume approximately **160 MB** of RAM, well below Streamlit Cloud's 1 GB free-tier limit.
- **Cold-Start Behavior:** If `models/sentiment_model.joblib` is present in the repository, the app starts in **< 3 seconds**. If the model is absent, `app.py` automatically trains the winning pipeline on the sample dataset on first launch without crashing.
- **No External Secrets Required:** SentimentScope AI runs self-contained without needing API keys or database connection strings.

---

## 4. Manual Post-Deployment Smoke Test Checklist

Once deployed on Streamlit Cloud, verify the following:
- [ ] **Home View:** System status cards display 360 records and model accuracy.
- [ ] **Live Prediction:** Click a sample button (e.g. "Thrilled with update! 🚀") and click "Analyze Sentiment". Verify sentiment badge, confidence, and token attribution appear in under 2 seconds.
- [ ] **Dataset Explorer:** Upload a CSV and verify column auto-mapping and quality profiling.
- [ ] **ML Lab:** Verify the 8-entry model leaderboard and confusion matrix heatmap render cleanly.
- [ ] **Visual Analytics:** Check that all Plotly charts (donut, bars, daily trends, wordcloud) render inline.
- [ ] **Reports:** Click "Download Executive Report (PDF)" and open the resulting PDF file to verify proper formatting.
