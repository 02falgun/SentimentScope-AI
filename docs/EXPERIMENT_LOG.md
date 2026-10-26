# SentimentScope AI — Experiment Log & Model Benchmark

This log records the benchmark results from training and evaluating all 4 candidate algorithms across both TF-IDF and Bag of Words feature representations on the held-out test dataset (20% split, 72 samples, fixed random seed 42).

---

## 1. Full Benchmark Leaderboard

| Rank | Model Name | Feature Representation | Test Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Training Time (s) | Inference Latency (ms/sample) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Logistic Regression** | **TF-IDF (1-2 ngrams)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.234 s** | **0.04 ms** |
| 2 | Multinomial Naive Bayes | TF-IDF (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.011 s | 0.04 ms |
| 3 | Linear SVM (Calibrated) | TF-IDF (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.023 s | 0.05 ms |
| 4 | Logistic Regression | Bag of Words (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.015 s | 0.04 ms |
| 5 | Multinomial Naive Bayes | Bag of Words (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.011 s | 0.04 ms |
| 6 | Linear SVM (Calibrated) | Bag of Words (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.021 s | 0.05 ms |
| 7 | Random Forest | Bag of Words (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.071 s | 0.21 ms |
| 8 | Random Forest | TF-IDF (1-2 ngrams) | 98.61% | 0.9861 | 0.9861 | 0.9867 | 0.9861 | 0.080 s | 0.24 ms |

---

## 2. Selection Rationale

- **Selected Winner:** `Logistic Regression (TF-IDF)`
- **Key Justifications:**
  1. **Calibrated Probabilities:** Generates true posterior probabilities via softmax/logistic sigmoid without requiring secondary heuristic scaling.
  2. **Sublinear TF Scaling:** Dampens the impact of repeated words in long posts, yielding robust probability estimates.
  3. **Linear Interpretability:** Directly maps linear feature coefficients to tokens for the Live Prediction token explainability view.
  4. **Sub-Millisecond Latency:** 0.04 ms inference time per sample, well within the PRD non-functional requirement (<2000 ms).
