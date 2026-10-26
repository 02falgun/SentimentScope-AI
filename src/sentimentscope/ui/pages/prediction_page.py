"""
SentimentScope AI — Live Single-Text Prediction Workspace
Dedicated AI analysis workspace for real-time natural language sentiment classification,
calibrated probability estimation, and token-level feature attribution.
"""

import streamlit as st
import pandas as pd

from src.sentimentscope.config import CANONICAL_SENTIMENTS, SENTIMENT_COLOR_MAP
from src.sentimentscope.ml.prediction import predict_single_text
from src.sentimentscope.ui.components import (
    render_header,
    render_metric_card,
    render_sentiment_badge,
)

SAMPLE_PROMPTS = [
    "I absolutely love this new update! Everything is fast, smooth, and beautiful! 🚀",
    "This is the worst software update ever. Constantly crashes and ruined my work! 😡",
    "The quarterly earnings call is scheduled for Thursday at 4 PM EST.",
    "Not bad at all, actually exceeded my modest expectations.",
    "I will never recommend this service to anyone, total disaster.",
]


def render_prediction_page():
    render_header(
        title="Live Sentiment Inference",
        subtitle="Real-time Natural Language Inference with Token-Level Feature Attribution & Calibrated Probabilities",
        badge_text="Sub-Millisecond Inference",
        category_label="AI INFERENCE WORKSPACE",
    )

    model_artifact = st.session_state.get("model_artifact")
    if not model_artifact:
        st.error("⚠️ No active machine learning model artifact found. Please load or train a model in the ML Lab.")
        return

    # Initialize session history
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # Quick Sample Selection
    st.markdown("<div style='font-size: 13px; font-weight: 600; color: #111111; margin-bottom: 6px;'>Quick Test Cases:</div>", unsafe_allow_html=True)
    sample_cols = st.columns(len(SAMPLE_PROMPTS))
    chosen_sample = None
    for idx, prompt in enumerate(SAMPLE_PROMPTS):
        with sample_cols[idx]:
            short_lbl = prompt.split()[0] + " " + " ".join(prompt.split()[1:3]) + "..."
            if st.button(short_lbl, key=f"sample_btn_{idx}", help=prompt, use_container_width=True):
                chosen_sample = prompt

    # Text Input & Controls
    default_text = chosen_sample if chosen_sample else ""
    user_input = st.text_area(
        "Enter Social Media Text, Tweet, or Customer Review:",
        value=default_text,
        height=100,
        max_chars=500,
        placeholder="Type or paste any post, comment, or statement to evaluate...",
        help="Input text is processed strictly in-memory during this session and is never persisted.",
    )

    col_btn, col_stats = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    with col_stats:
        char_len = len(user_input)
        word_len = len(user_input.split())
        st.markdown(
            f"""
            <div style="font-size: 12px; color: #666666; padding-top: 10px;">
                Length: <strong>{char_len}/500</strong> characters &nbsp;•&nbsp;
                Words: <strong>{word_len}</strong> &nbsp;•&nbsp;
                Pipeline: <span style="color: #111111; font-weight: 500;">{model_artifact.get('model_name', 'Classifier')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Execution State
    if analyze_clicked or chosen_sample:
        if not user_input.strip():
            st.warning("Please provide non-empty text input for inference.")
            return

        with st.spinner("Executing inference and extracting token attributions..."):
            result = predict_single_text(user_input, model_artifact)

        if result.error:
            st.error(f"Inference error: {result.error}")
            return

        # Record in session history
        st.session_state.prediction_history.append({
            "Text": user_input,
            "Predicted_Sentiment": result.predicted_sentiment,
            "Confidence": round(result.confidence_score * 100, 1),
            "Latency_ms": round(result.latency_ms, 2),
        })

        # Structured Prediction Outcome Block
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("### Inference Diagnostics & Attribution")

        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            render_metric_card(
                label="Predicted Sentiment",
                value=result.predicted_sentiment,
                sub_text="Maximum posterior class probability",
                border_accent=SENTIMENT_COLOR_MAP.get(result.predicted_sentiment, "#111111"),
            )
        with res_c2:
            render_metric_card(
                label="Model Confidence",
                value=f"{result.confidence_score * 100:.1f}%",
                sub_text="Calibrated probability estimate",
                border_accent="#111111",
            )
        with res_c3:
            render_metric_card(
                label="Inference Latency",
                value=f"{result.latency_ms:.2f} ms",
                sub_text="Single-sample pipeline execution",
                border_accent="#111111",
            )

        # Uncertainty Notice if margins are narrow
        if result.is_uncertain:
            st.markdown(
                f"""
                <div class="badge-warning" style="margin-top: 14px;">
                    ⚠️ <strong>Prediction Uncertainty:</strong> {result.uncertainty_reason}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Class Probability Breakdown
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Calibrated Probability Distribution:")
        st.caption("Class probabilities are derived from Logistic Regression sigmoid/softmax scoring or calibrated SVM margins.")

        p_cols = st.columns(3)
        for idx, cls in enumerate(CANONICAL_SENTIMENTS):
            prob = result.probabilities.get(cls, 0.0)
            with p_cols[idx]:
                st.markdown(
                    f"""
                    <div style="background-color: #FAFAFA; border: 1px solid #E5E5E5; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: #111111;">
                            <span>{cls}</span>
                            <span style="font-variant-numeric: tabular-nums;">{prob * 100:.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(float(prob))

        # Token Attribution & Explainability
        if result.token_explanations:
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            st.markdown("#### Token-Level Feature Attribution:")
            st.caption(
                "Weights ($w_i$) reflect the linear model's learned associations in TF-IDF feature space. "
                "Tokens tagged green support the predicted class, red tokens oppose it, and neutral tokens were unweighted or out-of-vocabulary."
            )

            html_tokens = []
            for item in result.token_explanations:
                tok = item["token"]
                wt = item["weight"]
                impact = item["impact"]

                if "Supports" in impact:
                    html_tokens.append(f"<span class='token-support' title='Weight: {wt} ({impact})'>{tok} (+{wt})</span>")
                elif "Opposes" in impact:
                    html_tokens.append(f"<span class='token-oppose' title='Weight: {wt} ({impact})'>{tok} (-{wt})</span>")
                else:
                    html_tokens.append(f"<span class='token-neutral' title='Neutral / Stopword'>{tok}</span>")

            st.markdown(
                f"""
                <div style="background-color: #FAFAFA; border: 1px solid #E5E5E5; border-radius: 6px; padding: 14px 16px;">
                    {' '.join(html_tokens)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Session Prediction History Table
    if st.session_state.prediction_history:
        st.markdown("<div style='border-top: 1px solid #E5E5E5; margin: 28px 0 20px 0;'></div>", unsafe_allow_html=True)
        st.markdown("### Session Inference History")
        st.caption("Log of single inferences evaluated during this active user session.")

        hist_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(hist_df, use_container_width=True)

        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Session Inferences (CSV)",
            data=csv_bytes,
            file_name="sentimentscope_session_inferences.csv",
            mime="text/csv",
        )
