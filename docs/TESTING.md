# SentimentScope AI — Testing & Quality Assurance Guide

## 1. Test Suite Overview

SentimentScope AI includes a rigorous automated test suite powered by `pytest`. The tests validate all layers of the system:
- **Preprocessing:** Cleaning, regex filters, negation preservation, emoji and hashtag handling.
- **Uploads & Ingestion:** File parsing, encoding fallbacks, column guessing, schema validation, duplicate and null handling.
- **ML Training & Benchmarking:** Scikit-learn pipeline integrity, stratified splitting, evaluation metrics, and Group K-Fold template isolation.
- **Prediction:** Single-text inference, batch inference, latency budget (<2s), confidence bounds, empty input recovery, and excessive length protection.
- **Analytics:** Sentiment counts, word frequencies, length metrics, temporal trends, platform breakdowns, keyword search, and empty visualization fallbacks.
- **Reporting:** Structured binary PDF report generation.

---

## 2. Test Execution Command

Run the complete test suite from the repository root:

```bash
# Using the local virtual environment
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

---

## 3. Actual Test Results (Verified)

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

============================== 29 passed in 3.71s ==============================
```

- **Total Tests:** 29
- **Passing:** 29 (100%)
- **Failures:** 0
- **Execution Time:** ~3.71 seconds
