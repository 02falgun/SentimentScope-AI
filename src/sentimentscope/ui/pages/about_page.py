"""
SentimentScope AI — System Architecture & Engineering Specifications
Presents 3-tier system architecture, leakage-safe machine learning pipelines,
generalization findings, and technical methodology documentation.
"""

import streamlit as st
from src.sentimentscope.ui.components import render_header


def render_about_page():
    render_header(
        title="Documentation & Architecture",
        subtitle="System Specifications, Machine Learning Methodology, and Technical Architecture",
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
