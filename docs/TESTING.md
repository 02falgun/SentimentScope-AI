# SentimentScope AI — Testing & Quality Assurance Guide

## 1. Test Suite Overview

SentimentScope AI maintains a comprehensive automated testing and validation protocol:
- **Unit & Integration Suite (pytest):** 29 automated test cases covering preprocessing, data ingestion, ML pipelines, predictions, analytics, and PDF reports.
- **Model Smoke Validation (`scripts/validate_model.py`):** Direct inference validation against positive, negative, and neutral test sentences verifying latency and class probabilities.
- **Syntax & Code Hygiene:** `git diff --check` verifying zero trailing whitespace and formatting cleanliness.
- **Visual Inspection via Headless Browser Automation:** Playwright/Chromium verification across all 8 dashboard pages verifying rendering, typography, chart loading, and interactivity.

---

## 2. Test Execution Commands

Run the full pytest suite from repository root:

```bash
# Execute automated test suite
PYTHONPATH=. .venv/bin/pytest tests/ -v

# Execute model smoke validation
.venv/bin/python scripts/validate_model.py

# Verify code formatting hygiene
git diff --check
```

---

## 3. Actual Unit Test Results (29/29 Passing)

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.1.1, pluggy-1.7.0
rootdir: /Users/kavyakumarthakur/Sharda Stuffs/ML-Proj

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

============================== 29 passed in 3.72s ==============================
```

---

## 4. Visual Verification Findings (Headless Browser)

Visual inspection was conducted on `http://localhost:8505` via headless browser subagent across all sections:

1. **Overview Dashboard:**
   - Pure `#FFFFFF` background with crisp `#111111` typography and `#E5E5E5` 1px borders.
   - Sidebar displays `SentimentScope AI` with clean Model Active and Dataset Active status cards.
   - 4 KPI cards rendered with tabular numerals and restrained sentiment accents (`#15803D`, `#64748B`, `#B91C1C`).
   - Plotly donut chart and volume bar chart render with white separators and high contrast.
2. **Live Prediction:**
   - Quick test chips click smoothly and populate text area.
   - Black primary action button (`#000000`, white text) triggers real-time inference.
   - Displays Predicted Sentiment badge, confidence score (73.9%), inference latency (sub-millisecond), probability progress bars, and token attribution tags (`.token-support`, `.token-oppose`).
3. **ML Lab & Benchmark:**
   - Model Comparison Leaderboard table rendered cleanly.
   - Scientific Audit expander presents Disjoint GroupKFold (95.28% Accuracy, 0.9543 Macro F1 across all 360 disjoint samples) alongside Monte Carlo GroupShuffleSplit comparison (93.15% Accuracy, 0.9193 Macro F1 across 389 evaluations).
   - Confusion matrix renders with clean sequential slate heatmap.
   - ROC curves render with distinct line colors.
4. **Dataset Explorer, Visual Analytics, Search, Reports, About:**
   - All charts, file uploaders, data tables, and PDF generators executed without errors.
