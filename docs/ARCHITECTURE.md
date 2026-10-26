# SentimentScope AI — System Architecture

## 1. Architectural Philosophy

SentimentScope AI is structured according to clean architecture principles for machine learning systems:
- **Separation of Concerns:** Business logic, feature extraction, model inference, analytical computation, and UI rendering are isolated into dedicated Python subpackages.
- **Leakage Safety:** Feature transformers (`TfidfVectorizer`, `CountVectorizer`) are coupled with classifiers via `sklearn.pipeline.Pipeline`, ensuring that statistics (IDF frequencies, vocabulary) are learned strictly on training splits.
- **Portability & Zero External Infrastructure:** The system runs self-contained without paid third-party APIs, external databases, or heavy GPU accelerators, guaranteeing free deployment on Streamlit Community Cloud.

---

## 2. Component Hierarchy

```text
sentimentscope-ai/
├── app.py                             # Presentation layer entry point and page router
├── requirements.txt                   # Pinned deployment dependencies
├── .streamlit/
│   └── config.toml                    # Server settings, max upload size, and theme tokens
├── src/
│   └── sentimentscope/
│       ├── config.py                  # Dynamic paths, schema fields, color tokens, seeds
│       ├── data/
│       │   ├── loading.py             # Multi-encoding loader and schema heuristic detector
│       │   ├── validation.py          # Row attrition, quality profiling, label normalization
│       │   └── preprocessing.py       # Regex cleaners, negation-preserving stopwords, lemmatizer
│       ├── ml/
│       │   ├── training.py            # 4-algorithm benchmark, pipeline factory, StratifiedKFold
│       │   ├── evaluation.py          # Accuracy, macro/weighted F1, confusion matrix, ROC-AUC
│       │   ├── prediction.py          # Real-time inference service with uncertainty detection
│       │   ├── explainability.py      # Global linear weights and local token attribution
│       │   └── model_registry.py      # Versioned artifact serialization and metadata loader
│       ├── analytics/
│       │   ├── metrics.py             # Distributions, word frequencies, length stats, keyword search
│       │   ├── trends.py              # Temporal trends, platform aggregations, engagement metrics
│       │   └── visualizations.py      # Plotly dark-mode charts and WordCloud generators
│       ├── reporting/
│       │   └── pdf_report.py          # Clean executive PDF report generator (fpdf2)
│       └── ui/
│           ├── components.py          # Hero headers, metric cards, status pills, badges
│           ├── styles.py              # Glassmorphic CSS styling and theme tokens
│           └── pages/                 # 8 interactive Streamlit subpages
├── data/
│   └── sample/
│       └── social_media_sentiment_sample.csv # 360-post curated benchmark dataset
├── models/
│   ├── sentiment_model.joblib         # Production pipeline artifact
│   └── experiment_history.json        # Leaderboard benchmark results
├── scripts/
│   ├── train_model.py                 # Standalone CLI training script
│   └── validate_model.py              # Standalone CLI validation and smoke test script
└── tests/                             # Complete pytest suite (26 passing tests)
```

---

## 3. Dataflow & Pipeline Lifecycle

### A. Training & Model Serialization Dataflow

```
[Raw CSV Dataset]
       │
       ▼
[Data Validation & Quality Profiling] ──> Exact deduplication, empty text drop, label normalize
       │
       ▼
[Train / Test Stratified Split (80/20)]
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[X_train, y_train]                         [X_test, y_test]
       │                                         │
       ▼                                         │
[Fit Pipeline: TfidfVectorizer + Classifier]     │
       │                                         │
       ▼                                         │
[Evaluate on Held-Out Test Set] <────────────────┘
       │
       ▼
[Compute Metrics: Accuracy, Macro F1, Confusion Matrix, ROC-AUC]
       │
       ▼
[Serialize Bundle to models/sentiment_model.joblib]
```

### B. Live Inference & Explainability Dataflow

```
[User Text Input] (e.g., "The app is not bad at all!")
       │
       ▼
[Input Validation & Whitespace Normalization]
       │
       ▼
[Scikit-learn Pipeline Inference: pipeline.predict_proba()]
       │
       ├─────────────────────────────────────────┬──────────────────────────────────┐
       ▼                                         ▼                                  ▼
[Predicted Sentiment]                     [Confidence Score]             [Uncertainty Check]
(Positive)                                (66.8%)                        (Margin > 12%)
       │
       ▼
[Token Feature Attribution (explain_single_prediction)]
- "not": Supports Positive (negation retained)
- "bad": Supports Positive in bigram context
- "exceeded": Supports Positive
```

---

## 4. Design Patterns Used

1. **Pipeline Pattern:** Linear chaining of transformers and estimators in `sklearn.pipeline.Pipeline`.
2. **Registry Pattern:** Serialized artifact store with schema verification in `model_registry.py`.
3. **Strategy Pattern:** Interchangeable feature vectorizers (TF-IDF vs Bag of Words) and classifiers (Logistic Regression, Naive Bayes, Linear SVM, Random Forest).
4. **Adapter / Façade Pattern:** High-level APIs (`predict_single_text`, `search_dataset_by_keyword`, `generate_pdf_report`) hide library internals from the presentation layer.
