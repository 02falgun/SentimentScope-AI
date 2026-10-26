# SentimentScope AI — Model Card: Production Sentiment Classifier

## 1. Model Details

- **Model Name:** Logistic Regression (TF-IDF)
- **Pipeline Architecture:** Scikit-learn Pipeline
  - **Feature Extractor:** `TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000, sublinear_tf=True)`
  - **Classifier:** `LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)`
- **Target Task:** Multiclass Sentiment Classification (`Negative`, `Neutral`, `Positive`)
- **Version:** 1.0.0 (Production Release)
- **Serialization Format:** Joblib compressed artifact (`models/sentiment_model.joblib`)
- **License:** MIT License / Open Academic Use

---

## 2. Intended Use

### Appropriate Uses
- Analyzing public sentiment in English social media microblogging data (tweets, comments, feedback).
- Real-time customer satisfaction sentiment scoring with calibrated confidence percentages.
- Keyword and brand sentiment monitoring across social networks.
- Academic demonstration of classical NLP pipelines, model evaluation, and linear explainability.

### Inappropriate / Out-of-Scope Uses
- Sole basis for critical legal, employment, or financial determinations.
- Evaluation of figurative, highly poetic, or deeply sarcastic non-literal text without human review.
- Multi-language sentiment classification on non-English texts without retraining.

---

## 3. Training & Evaluation Provenance

- **Training Corpus:** Curated benchmark social media dataset (`data/sample/social_media_sentiment_sample.csv`).
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

## 5. Model Explainability & Interpretability

Because the production model employs Logistic Regression with TF-IDF, decisions are linear combinations of learned feature coefficients ($z_c = \sum w_{c, i} x_i + b_c$):
- **Top Positive Features:** `love`, `fast`, `smooth`, `beautiful`, `recommended`, `impressive`, `kudos`, `clarity`, `outstanding`, `support`.
- **Top Negative Features:** `worst`, `crashes`, `ruined`, `terrible`, `disaster`, `never`, `scam`, `horrible`, `buggy`, `outage`.
- **Top Neutral Features:** `scheduled`, `conference`, `report`, `released`, `portal`, `maintenance`, `benchmarks`, `observations`, `financial`, `quarterly`.

---

## 6. Caveats & Ethical Considerations

1. **Statistical Correlation vs. Causality:** A high positive or negative coefficient indicates that a word frequently co-occurs with that label in the training corpus; it does not constitute proof that the word causes the sentiment in isolation.
2. **Context Window Limitations:** Unigrams and bigrams capture localized negations (`not bad`, `never recommend`) effectively, but long-range rhetorical dependencies may be missed.
3. **Data Ephemerality:** The model runs entirely offline/in-memory and does not store or transmit user text.
