# SentimentScope AI — User Guide

Welcome to the **SentimentScope AI** user guide. This document explains how to navigate and operate each of the 8 sections of the application dashboard.

---

## 1. Sidebar Navigation

The sidebar on the left lets you switch between the 8 core sections of the dashboard:
1. **🏠 Home / Overview:** High-level executive dashboard and status indicators.
2. **⚡ Live Prediction:** Single-text inference with uncertainty bounds and token explainability.
3. **📁 Dataset Explorer:** CSV upload, column mapping, data quality profiling, and filtering.
4. **🔬 ML Lab / Benchmark:** 4-algorithm benchmark leaderboard, confusion matrix, ROC curves, and misclassification inspector.
5. **📊 Visual Analytics:** Interactive charts for sentiment trends, platform comparisons, hashtags, and wordclouds.
6. **🔍 Keyword Search:** Query specific brands, topics, or terms and calculate sentiment polarity.
7. **📄 Reports & Export:** Download executive PDF reports and CSV datasets.
8. **📘 About & Architecture:** System architecture, academic methodology, and demonstration guide.

The sidebar also includes **live status pills** showing the active model and active dataset row count.

---

## 2. Using Live Prediction

1. Navigate to **⚡ Live Prediction**.
2. Click any of the quick sample buttons (e.g. *"I absolutely love this new update!..."*) or type your own post in the text area.
3. Click **"🔍 Analyze Sentiment"**.
4. The system displays:
   - **Predicted Sentiment:** Green badge for Positive, Slate for Neutral, Red for Negative.
   - **Confidence Score:** Calibrated posterior probability percentage.
   - **Latency:** Inference duration in milliseconds.
   - **Uncertainty Notice:** Displayed automatically if confidence is low or the top two classes are close.
   - **Probability Distribution:** Visual progress bars for all 3 sentiment classes.
   - **Token Feature Attribution:** Highlighted words showing whether each term pushed the prediction toward positive or negative polarity.
5. Below the prediction, inspect your **Session Prediction History** and click **"Download Session Predictions (CSV)"** to save your results.

---

## 3. Uploading Custom Datasets

1. Navigate to **📁 Dataset Explorer**.
2. Drag and drop any `.csv` file (up to 20 MB).
3. The system scans the header row and automatically maps candidate columns to schema fields:
   - `Text Column` (Required)
   - `Sentiment Column` (Optional)
   - `Platform Column` (Optional)
   - `Date Column` (Optional)
   - `Likes` and `Shares` (Optional)
4. Review the **Data Quality Profile**:
   - Check rows loaded, exact duplicates removed, missing text dropped, and clean retained rows.
   - If severe class imbalance is detected (<10% representation), an alert banner is displayed.
5. If your dataset lacks sentiment labels, click **"🚀 Predict Sentiments with Active Model"** to run batch inference and populate predictions.
6. Filter by platform, sentiment, or search terms, then click **"📥 Download Cleaned Dataset (CSV)"**.

---

## 4. Exploring the ML Lab

1. Navigate to **🔬 ML Lab / Benchmark**.
2. Inspect the **Model Comparison Leaderboard** comparing Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Random Forest across TF-IDF and Bag of Words.
3. Review the **Confusion Matrix Heatmap** and **ROC Curves** for the active model.
4. Use the **Misclassified Sample Inspector** to review posts where ground-truth labels differed from predictions.
5. To retrain or benchmark on your active dataset, select your desired feature representations and click **"🚀 Run ML Benchmark"**.

---

## 5. Generating PDF Reports

1. Navigate to **📄 Reports & Export**.
2. Review the live report preview summary.
3. Click **"📥 Download Executive Report (PDF)"**.
4. The system compiles a styled PDF document containing executive summary statistics, class distributions, model metrics, platform tables, trending hashtags, and methodology notes.
