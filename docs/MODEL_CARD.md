# SentimentScope AI — Model Card: Production Sentiment Classifier

## 1. Model Details

- **Model Name:** Logistic Regression (TF-IDF)
- **Pipeline Architecture:** Scikit-learn Pipeline
  - **Feature Extractor:** `TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000, sublinear_tf=True)`
  - **Classifier:** `LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)`
- **Target Task:** Multiclass Sentiment Classification (`Negative`, `Neutral`, `Positive`)
- **Version:** 1.0.1 (Post-Deployment Audit Release)
- **Serialization Format:** Joblib compressed artifact (`models/sentiment_model.joblib`)
- **Model Size:** ~145 KB
- **License:** MIT License / Open Academic Use

---

## 2. Intended Use

### Appropriate Uses
- Analyzing public sentiment in English social media microblogging data (tweets, comments, customer feedback).
- Real-time customer satisfaction sentiment scoring with calibrated confidence percentages.
- Keyword and brand sentiment monitoring across social networks.
- Academic demonstration of classical NLP pipelines, model evaluation, and linear explainability.

### Inappropriate / Out-of-Scope Uses
- Sole basis for critical legal, employment, medical, or financial determinations.
- Evaluation of figurative, highly poetic, or deeply sarcastic non-literal text without human review.
- Multi-language sentiment classification on non-English texts without retraining.

---

## 3. Training & Evaluation Provenance

- **Training Corpus:** Curated demonstration social media dataset (`data/sample/social_media_sentiment_sample.csv`).
- **Total Records:** 360 unique observations after deduplication.
- **Class Balance:** Perfectly balanced (120 Positive, 120 Neutral, 120 Negative).
- **Split Protocol:** Stratified 80/20 train/test split with deterministic seed (`random_state=42`).
  - Training Set: 288 samples (96 per class)
  - Held-out Test Set: 72 samples (24 per class)

---

## 4. Quantitative Evaluation Results (Held-Out Test Set)

| Metric | Measured Score | PRD Target | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **100.00%** | $\ge 80.00\%$ | **EXCEEDED** |
| **Macro F1-Score** | **1.0000** | $\ge 0.7800$ | **EXCEEDED** |
| **Weighted F1-Score** | **1.0000** | — | **OPTIMAL** |
| **Macro Precision** | **1.0000** | — | **OPTIMAL** |
| **Macro Recall** | **1.0000** | — | **OPTIMAL** |
| **ROC-AUC (OVR Macro)**| **1.0000** | — | **OPTIMAL** |
| **Inference Latency** | **0.04 ms / sample** | $< 2000\text{ ms}$ | **EXCEEDED (50,000x faster)** |
| **Training Duration** | **1.23 seconds** | — | **FAST** |

### Confusion Matrix on Held-Out Test Set:

| True \ Pred | Pred: Negative | Pred: Neutral | Pred: Positive | Total Support |
| :--- | :---: | :---: | :---: | :---: |
| **True: Negative** | **24** | 0 | 0 | 24 |
| **True: Neutral** | 0 | **24** | 0 | 24 |
| **True: Positive** | 0 | 0 | **24** | 24 |

---

## 5. Investigation into the 100% Test Accuracy & Generalization Audit

In real-world machine learning, **100% test accuracy requires thorough scientific scrutiny**. We performed a detailed forensic investigation:

### 5.1 Root Cause of 100% Accuracy
1. **Dataset Synthesis Structure:** The 360-row sample dataset contains 75 unique base sentence templates, where 45 templates are repeated 6 to 8 times with realistic suffix clauses (e.g., *"Published on official channel"*, *"Further details will follow"*, *"Avoid at all costs"*).
2. **Template Overlap in Standard Splits:** When performing an 80/20 train/test split (even stratified), **93.1% of test set samples (67/72)** belong to base sentence templates that also appear in the training set.
3. **In-Template Memorization:** Because key n-grams of the sentence are present in `X_train`, the linear classifier easily correctly predicts 100% of the held-out test split.

### 5.2 Group-Aware Out-of-Template Cross-Validation
To measure generalization on truly unseen text structures, we evaluated the pipeline under template-isolated group cross-validation:

#### Protocol A: Strict Disjoint Group K-Fold (`GroupKFold`, n_splits=5)
Partitions all 75 base templates and 360 rows into 5 mutually exclusive test folds where each sample appears in a test set **exactly once** (72 samples and 15 templates per fold; sum = 360):

| GroupKFold Split | Test Sample Count | Base Templates | Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | 72 | 15 | 95.83% | 0.9588 |
| **Fold 2** | 72 | 15 | 91.67% | 0.9240 |
| **Fold 3** | 72 | 15 | 100.00% | 1.0000 |
| **Fold 4** | 72 | 15 | 91.67% | 0.9164 |
| **Fold 5** | 72 | 15 | 97.22% | 0.9722 |
| **Complete Dataset (Disjoint)** | **360 Total** | **75 Total** | **95.28%** | **0.9543** |

#### Protocol B: Monte Carlo Group Shuffle Split (`GroupShuffleSplit`, n_splits=5, seed=42)
5 independent random 20% group subsamples with replacement across iterations (yielding 57, 86, 72, 90, 84 sample counts totaling 389 test evaluations):

| GroupShuffleSplit Iteration | Test Sample Count | Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: |
| **Iteration 1** | 57 | 91.23% | 0.8931 |
| **Iteration 2** | 86 | 94.19% | 0.9385 |
| **Iteration 3** | 72 | 91.67% | 0.9245 |
| **Iteration 4** | 90 | 92.22% | 0.8850 |
| **Iteration 5** | 84 | 96.43% | 0.9553 |
| **Average (Monte Carlo)** | **77.8 avg (389 total evaluations)** | **93.15%** | **0.9193** |

### 5.3 Out-of-Distribution Real-World Zero-Shot Validation
Evaluating the model against 11 diverse, real-world social media posts (including subtle negations, customer complaints, and neutral scheduling notices):
- **Real-World Test Accuracy:** **90.9% (10/11 correct)**.
- **Probabilistic Calibration:** Unseen real-world confidence scores ranged realistically between **38% and 78%**, accurately triggering automated uncertainty warnings on boundary cases.

---

## 6. Model Explainability & Interpretability

Because the production model employs Logistic Regression with TF-IDF, decisions are linear combinations of learned feature coefficients ($z_c = \sum w_{c, i} x_i + b_c$):
- **Top Positive Features:** `love`, `fast`, `smooth`, `beautiful`, `recommended`, `impressive`, `kudos`, `clarity`, `outstanding`, `support`.
- **Top Negative Features:** `worst`, `crashes`, `ruined`, `terrible`, `disaster`, `never`, `scam`, `horrible`, `buggy`, `outage`.
- **Top Neutral Features:** `scheduled`, `conference`, `report`, `released`, `portal`, `maintenance`, `benchmarks`, `observations`, `financial`, `quarterly`.

---

## 7. Caveats & Ethical Considerations

1. **Statistical Correlation vs. Causality:** A high positive or negative coefficient indicates that a word frequently co-occurs with that label in the training corpus; it does not constitute proof that the word causes the sentiment in isolation.
2. **Context Window Limitations:** Unigrams and bigrams capture localized negations (`not bad`, `never recommend`) effectively, but long-range rhetorical dependencies may be missed.
3. **Data Ephemerality:** The model runs entirely offline/in-memory and does not store or transmit user text.
