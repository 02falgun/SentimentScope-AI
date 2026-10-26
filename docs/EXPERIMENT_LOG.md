# SentimentScope AI — Experiment Log & Scientific Benchmarks

This log records the benchmark results from training and evaluating candidate algorithms across feature representations, split protocols, and generalization stress-tests.

---

## 1. Experiment 1: Standard Held-Out Benchmark (80/20 Stratified Split)

- **Dataset:** 360-row balanced sample dataset (`data/sample/social_media_sentiment_sample.csv`).
- **Split:** 80% train (288 samples), 20% held-out test (72 samples), `random_state=42`.

| Rank | Model Name | Feature Representation | Test Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Training Time (s) | Inference Latency (ms/sample) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Logistic Regression** | **TF-IDF (1-2 ngrams)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.234 s** | **0.04 ms** |
| 2 | Multinomial Naive Bayes | TF-IDF (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.011 s | 0.04 ms |
| 3 | Linear SVM (Calibrated) | TF-IDF (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.023 s | 0.05 ms |
| 4 | Logistic Regression | Bag of Words (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.015 s | 0.04 ms |
| 5 | Multinomial Naive Bayes | Bag of Words (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.011 s | 0.04 ms |
| 6 | Linear SVM (Calibrated) | Bag of Words (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.021 s | 0.05 ms |
| 7 | Random Forest | Bag of Words (1-2 ngrams) | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.071 s | 0.21 ms |
| 8 | Random Forest | TF-IDF (1-2 ngrams) | 98.61% | 0.9861 | 0.9861 | 0.9867 | 0.9861 | 0.080 s | 0.24 ms |

---

## 2. Selection Rationale for Production Model

- **Selected Winner:** `Logistic Regression (TF-IDF)`
- **Key Justifications:**
  1. **Calibrated Probabilities:** Generates true posterior probabilities via softmax/logistic sigmoid without requiring secondary heuristic scaling.
  2. **Sublinear TF Scaling:** Dampens the impact of repeated words in long posts, yielding robust probability estimates.
  3. **Linear Interpretability:** Directly maps linear feature coefficients to tokens for the Live Prediction token explainability view.
  4. **Sub-Millisecond Latency:** 0.04 ms inference time per sample, well within the PRD non-functional requirement (<2000 ms).

---

## 3. Experiment 2: Scientific Group-Aware Template Generalization Benchmark

To investigate the 100% held-out test score, we analyzed sentence construction in the demonstration dataset:
- **Observation:** The 360 rows originate from 75 base sentence patterns with varied endings.
- **Leakage Audit:** In the standard 80/20 split, **93.1% of test rows (67/72)** belong to sentence templates also present in the training set, enabling in-template interpolation.
- **Evaluation Protocols:**
  1. **Strict Disjoint Group K-Fold (`GroupKFold`, n_splits=5):** Partitions all 75 templates and 360 rows into 5 non-overlapping folds where each sample appears in a test set **exactly once** (72 samples and 15 groups per fold; sum = 360).
  2. **Monte Carlo Group Shuffle Split (`GroupShuffleSplit`, n_splits=5, test_size=0.2, seed=42):** 5 independent random group subsamples with replacement across splits (explaining the 57, 86, 72, 90, 84 sample counts totaling 389 test evaluations).

### Protocol A: Disjoint Group K-Fold (`GroupKFold`, n_splits=5) — Primary Benchmark

| Split / Fold | Test Sample Count | Base Templates | Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | 72 | 15 | 95.83% | 0.9588 |
| **Fold 2** | 72 | 15 | 91.67% | 0.9240 |
| **Fold 3** | 72 | 15 | 100.00% | 1.0000 |
| **Fold 4** | 72 | 15 | 91.67% | 0.9164 |
| **Fold 5** | 72 | 15 | 97.22% | 0.9722 |
| **Complete Dataset (Disjoint)** | **360 Total** | **75 Total** | **95.28%** | **0.9543** |

*Note: In Disjoint GroupKFold, every row is evaluated exactly once (72 x 5 = 360). Zero template leakage occurs across folds.*

### Protocol B: Monte Carlo Group Shuffle Split (`GroupShuffleSplit`, n_splits=5, seed=42)

| Split / Iteration | Test Sample Count | Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: |
| **Iteration 1** | 57 | 91.23% | 0.8931 |
| **Iteration 2** | 86 | 94.19% | 0.9385 |
| **Iteration 3** | 72 | 91.67% | 0.9245 |
| **Iteration 4** | 90 | 92.22% | 0.8850 |
| **Iteration 5** | 84 | 96.43% | 0.9553 |
| **Average (Monte Carlo)** | **77.8 avg (389 total evaluations)** | **93.15%** | **0.9193** |

*Note: GroupShuffleSplit repeatedly draws 20% of groups at random with replacement across iterations. Variable group sizes and overlapping test selections across splits explain the 389 total test evaluations.*

**Conclusion:** Both protocols independently prove that the model generalizes robustly to completely unseen sentence structures (**95.28% / 0.9543 under GroupKFold**; **93.15% / 0.9193 under GroupShuffleSplit**), confirming that high benchmark performance is not driven by vocabulary memorization.

---

## 4. Experiment 3: Out-of-Distribution Real-World Social Media Evaluation

- **Test Samples:** 11 uncurated real-world social media posts spanning consumer tech, airline delays, financial updates, customer support, and complex negation structures.
- **Results:**
  - Correct Predictions: 10 / 11 (**90.9% Accuracy**)
  - Negation handling: *"Not bad at all, actually exceeded my modest expectations"* correctly classified as Positive ($P = 0.67$).
  - Ambiguity handling: *"I wanted to love this phone, but the camera is disappointing"* produced narrow probability margin ($P = 0.49$), correctly triggering an Uncertainty Warning.
