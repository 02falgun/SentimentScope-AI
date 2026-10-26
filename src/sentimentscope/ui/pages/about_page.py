"""
SentimentScope AI — Architecture, Methodology & Viva Guide Workspace
Presents 3-tier system architecture, leakage-safe machine learning pipelines,
generalization findings, and technical demonstration talking points.
"""

import streamlit as st
from src.sentimentscope.ui.components import render_header


def render_about_page():
    render_header(
        title="Documentation & Architecture",
        subtitle="System Specifications, Machine Learning Methodology, and Viva Demonstration Guide",
        badge_text="PRD/SRS v1.0 Compliant",
        category_label="SYSTEM ARCHITECTURE",
    )

    # 1. Modular System Architecture
    st.markdown("### System Architecture & Pipeline Dataflow")
    st.caption("SentimentScope AI implements a 3-tier modular architecture separating Data Processing, ML Pipelines, and Serving.")

    st.code(
        """
+-----------------------------------------------------------------------------------+
|                            SENTIMENTSCOPE AI ARCHITECTURE                         |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ INGESTION & VALIDATION LAYER ]                                                 |
|  +--------------------+       +--------------------+       +-------------------+  |
|  | CSV Ingestion /    | ----> | Multi-Encoding     | ----> | Token-Based       |  |
|  | Sample Dataset     |       | Fallback (UTF-8)   |       | Schema Mapping    |  |
|  +--------------------+       +--------------------+       +-------------------+  |
|                                                                      |            |
|                                                                      v            |
|  [ PREPROCESSING & QUALITY PROFILING ]                                            |
|  +--------------------+       +--------------------+       +-------------------+  |
|  | Text Deduplication | ----> | Negation-          | ----> | Row Attrition &   |  |
|  | & Missing Drops    |       | Preserving Stops   |       | Quality Profiling |  |
|  +--------------------+       +--------------------+       +-------------------+  |
|                                                                      |            |
|                                                                      v            |
|  [ MODEL TRAINING & BENCHMARK SUITE ]                                             |
|  +-----------------------------------------------------------------------------+  |
|  | scikit-learn Pipeline (TfidfVectorizer / CountVectorizer)                   |  |
|  | - Logistic Regression (L2 / Balanced)                                       |  |
|  | - Multinomial Naive Bayes (Laplace alpha=1.0)                               |  |
|  | - Linear Support Vector Machine (CalibratedClassifierCV)                   |  |
|  | - Random Forest Classifier (100 Trees / Balanced)                           |  |
|  +-----------------------------------------------------------------------------+  |
|                                       |                                           |
|                                       v                                           |
|  [ SERVING & ANALYTICS WORKSPACE ]                                                |
|  +--------------------+       +--------------------+       +-------------------+  |
|  | Real-Time Single   |       | Interactive Plotly |       | Executive PDF &   |  |
|  | Inference (<2ms)   |       | Visualizations     |       | CSV Exporters     |  |
|  +--------------------+       +--------------------+       +-------------------+  |
+-----------------------------------------------------------------------------------+
        """,
        language="text",
    )

    # 2. Key Machine Learning Concepts Demonstrated
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### Machine Learning Principles Demonstrated")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            """
            <div class="action-card" style="margin-bottom: 14px;">
                <h4 style="color: #111111;">1. Leakage-Safe Feature Pipelines</h4>
                <p style="color: #555555; line-height: 1.5; font-size: 13px;">
                    All feature extraction (<code>TfidfVectorizer</code> and <code>CountVectorizer</code>) is encapsulated directly inside
                    <code>sklearn.pipeline.Pipeline</code>. Vocabulary formation and IDF weights are fitted strictly on training folds,
                    preventing document-frequency data leakage from held-out evaluation sets.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="action-card">
                <h4 style="color: #111111;">2. Negation Preservation in Stopwords</h4>
                <p style="color: #555555; line-height: 1.5; font-size: 13px;">
                    Standard NLP stopword lists discard negation words (<code>not</code>, <code>no</code>, <code>never</code>, <code>without</code>),
                    inadvertently inverting critical sentiments (e.g. <em>"not bad"</em> -> <em>"bad"</em>).
                    SentimentScope AI preserves 27 critical negation markers to safeguard contextual polarity.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t2:
        st.markdown(
            """
            <div class="action-card" style="margin-bottom: 14px;">
                <h4 style="color: #111111;">3. Calibrated Posterior Probabilities</h4>
                <p style="color: #555555; line-height: 1.5; font-size: 13px;">
                    Logistic Regression and Naive Bayes produce genuine posterior probabilities.
                    Linear SVM is wrapped in <code>CalibratedClassifierCV</code> via training-only 3-fold cross-validation
                    to compute authentic probability distributions. Low-margin predictions trigger automated uncertainty notices.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="action-card">
                <h4 style="color: #111111;">4. Macro F1 Model Selection Metric</h4>
                <p style="color: #555555; line-height: 1.5; font-size: 13px;">
                    Standard accuracy can be misleading under class imbalance. The ML Lab ranks candidate models primarily by
                    <strong>Macro F1-Score</strong>, giving equal weight to Positive, Neutral, and Negative classes to prevent bias toward majority classes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Viva & Technical Examination Talking Points
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("### University Demonstration & Viva Talking Points")

    st.markdown(
        """
        <div class="academic-callout">
            <h4 style="color: #111111; margin-bottom: 12px;">Frequently Asked Technical Viva Questions</h4>

            <div style="margin-bottom: 14px;">
                <strong style="color: #111111;">Q1: Why classical ML (Logistic Regression / SVM) rather than LLMs?</strong>
                <p style="color: #555555; font-size: 13px; margin: 4px 0 0 0; line-height: 1.5;">
                    Classical linear pipelines execute in sub-millisecond latencies (<1ms), require zero paid cloud API tokens,
                    deploy deterministically on free cloud tiers without GPU dependencies, and offer mathematically interpretable linear coefficients.
                </p>
            </div>

            <div style="margin-bottom: 14px;">
                <strong style="color: #111111;">Q2: How was the 100% held-out test accuracy addressed?</strong>
                <p style="color: #555555; font-size: 13px; margin: 4px 0 0 0; line-height: 1.5;">
                    The 360-row sample dataset is built from 75 base sentence patterns; in standard random 80/20 splits, 93.1% of test rows share templates with training data.
                    Under rigorous <strong>Group K-Fold Cross-Validation</strong> isolating unseen templates, the model achieves <strong>93.15% Accuracy and 0.9193 Macro F1</strong>,
                    confirming genuine out-of-distribution generalization.
                </p>
            </div>

            <div style="margin-bottom: 14px;">
                <strong style="color: #111111;">Q3: How does the token-level explainability feature work?</strong>
                <p style="color: #555555; font-size: 13px; margin: 4px 0 0 0; line-height: 1.5;">
                    The pipeline multiplies the user's active TF-IDF token weights ($x_i$) by the linear classifier's learned coefficients ($w_{ij}$) for each class $j$.
                    Tokens with large positive weights support the predicted class, while negative weights oppose it.
                </p>
            </div>

            <div>
                <strong style="color: #111111;">Q4: How does the application handle data privacy and security?</strong>
                <p style="color: #555555; font-size: 13px; margin: 4px 0 0 0; line-height: 1.5;">
                    User-uploaded CSV files are processed strictly in-memory inside the active Streamlit session state and are never written to server disk or external storage.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
