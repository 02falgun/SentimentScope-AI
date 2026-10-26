"""
SentimentScope AI — Documentation, Architecture & Viva Talking Points Page
"""

import streamlit as st
from src.sentimentscope.ui.components import render_hero


def render_about_page():
    render_hero(
        title="Documentation & Architecture",
        subtitle="System Specifications, Machine Learning Methodology, and Demonstration Guide",
        badge_text="PRD/SRS v1.0 Compliant",
    )

    st.markdown("### 📘 System Architecture & Dataflow")
    st.markdown(
        """
        SentimentScope AI uses a strict 3-tier modular software architecture separating **Data Processing**, 
        **Machine Learning**, and **Presentation/Serving**:
        """
    )

    st.code(
        """
+-----------------------------------------------------------------------------------+
|                            SENTIMENTSCOPE AI ARCHITECTURE                         |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ INGESTION LAYER ]                                                              |
|  +--------------------+       +--------------------+       +-------------------+  |
|  | CSV File Uploader  | ----> | Multi-Encoding     | ----> | Token-Based       |  |
|  | / Sample Dataset   |       | Fallback (UTF-8,   |       | Schema Mapping    |  |
|  +--------------------+       | Latin-1, CP1252)   |       +-------------------+  |
|                               +--------------------+                 |            |
|                                                                      v            |
|  [ PREPROCESSING & QUALITY LAYER ]                                                |
|  +--------------------+       +--------------------+       +-------------------+  |
|  | Exact Text         | ----> | Negation-          | ----> | Data Quality      |  |
|  | Deduplication      |       | Preserving Stops   |       | Profiling & Flags |  |
|  +--------------------+       +--------------------+       +-------------------+  |
|                                                                      |            |
|                                                                      v            |
|  [ MODEL BENCHMARK & SERVING LAYER ]                                              |
|  +-----------------------------------------------------------------------------+  |
|  | scikit-learn Pipeline (TfidfVectorizer / CountVectorizer)                   |  |
|  | - Logistic Regression (L2 / Balanced)                                       |  |
|  | - Multinomial Naive Bayes (Laplace alpha=1.0)                               |  |
|  | - Linear Support Vector Machine (CalibratedClassifierCV)                   |  |
|  | - Random Forest Classifier (100 Trees / Balanced)                           |  |
|  +-----------------------------------------------------------------------------+  |
|                                       |                                           |
|                                       v                                           |
|  [ SERVING & ANALYTICS LAYER ]                                                    |
|  +--------------------+       +--------------------+       +-------------------+  |
|  | Real-Time Single   |       | Multi-Dimensional  |       | Structured PDF    |  |
|  | Inference (<2s)    |       | Interactive Charts |       | Report Generator  |  |
|  +--------------------+       +--------------------+       +-------------------+  |
+-----------------------------------------------------------------------------------+
        """,
        language="text",
    )

    # Academic & Technical Methodology
    st.markdown("### 🔬 Machine Learning Concepts Demonstrated")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            """
            #### 1. Leakage-Safe Feature Pipelines
            - All feature extraction (`TfidfVectorizer` and `CountVectorizer`) is integrated inside `sklearn.pipeline.Pipeline`.
            - Text transformations are fitted **strictly** on training folds (`X_train`), preventing vocabulary and document-frequency leakage from the held-out test set.
            - Model checkpoints serialize the combined transformer and classifier into a single immutable artifact (`sentiment_model.joblib`).
            """
        )

        st.markdown(
            """
            #### 2. Negation Preservation in NLP
            - Standard stopword filters strip words like `not`, `no`, `never`, which catastrophically flips negative sentiment into positive.
            - SentimentScope AI implements an explicit negation-preservation strategy maintaining critical tokens (`not`, `never`, `hardly`, `without`, `barely`).
            """
        )

    with t2:
        st.markdown(
            """
            #### 3. Honest Probability & Decision Semantics
            - Logistic Regression and Naive Bayes produce genuine posterior class probabilities.
            - Linear SVM is wrapped in `CalibratedClassifierCV` via training-only 3-fold cross-validation to provide valid probability distributions rather than arbitrary score scaling.
            - Predictions with low margin between classes automatically trigger an **Uncertainty Warning**.
            """
        )

        st.markdown(
            """
            #### 4. Macro F1 Model Selection Metric
            - While Accuracy is sensitive to class imbalance, **Macro F1** gives equal weight to all three classes (Positive, Neutral, Negative).
            - The ML Lab leaderboard ranks candidates primarily on held-out Macro F1, ensuring balanced classification performance.
            """
        )

    # Faculty Viva & Demonstration Talking Points
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎓 Laboratory Demonstration / Viva Guide")
    st.markdown(
        """
        When presenting this project to faculty or examiners, emphasize the following points:
        1. **Why classical ML over LLMs?**
           - Classical models (Logistic Regression + TF-IDF) execute in sub-millisecond latencies, require zero paid API credentials, run deterministically on low-cost cloud tiers, and offer mathematically interpretable linear coefficients.
        2. **How is data quality enforced?**
           - The system profiles row attrition (exact deduplication, empty text removal) and detects severe class imbalance (<10% representation) before model training.
        3. **How does the explainability feature work?**
           - The app inspects linear weights ($w_i$) associated with active vocabulary tokens in the user's sentence to visualize which words pushed the prediction toward positive vs negative polarity.
        4. **How are custom uploads secured?**
           - Uploaded CSVs are parsed strictly in-memory within the active session state and are never persisted to the server filesystem.
        5. **How do you address the 100% test accuracy on the sample dataset?**
           - The 360-row sample dataset is built from 75 base sentence templates; standard stratified random splitting allows 93.1% of test samples to share templates with training data. Under scientifically rigorous Group K-Fold cross-validation isolating unseen templates, the pipeline achieves **93.15% Accuracy and 0.9193 Macro F1**, demonstrating genuine generalization.
        """
    )

    # Limitations & Scope
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚠️ Model Limitations & Ethical Considerations")
    st.markdown(
        """
        - **Target Language:** Current model pipeline is trained on English text.
        - **Sarcasm and Metaphor:** Classical bag-of-words and n-gram models can misinterpret irony or sarcasm where individual words appear positive.
        - **Vocabulary Shifts:** Highly domain-specific slang not present in the training set falls into out-of-vocabulary (OOV).
        """
    )
