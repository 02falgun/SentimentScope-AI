"""
SentimentScope AI — Live Single-Text Prediction Page
"""

import streamlit as st
import pandas as pd

from src.sentimentscope.config import CANONICAL_SENTIMENTS, SENTIMENT_COLOR_MAP
from src.sentimentscope.ml.prediction import predict_single_text
from src.sentimentscope.ui.components import render_hero, render_sentiment_badge


SAMPLE_PROMPTS = [
    "I absolutely love this new update! Everything is fast, smooth, and beautiful! 🚀",
    "This is the worst software update ever. Constantly crashes and ruined my work! 😡",
    "The quarterly earnings call is scheduled for Thursday at 4 PM EST.",
    "Not bad at all, actually exceeded my modest expectations.",
    "I will never recommend this service to anyone, total disaster.",
]


def render_prediction_page():
    render_hero(
        title="Live Sentiment Prediction",
        subtitle="Real-Time Natural Language Inference with Token-Level Explainability",
        badge_text="Sub-2s Latency",
    )

    model_artifact = st.session_state.get("model_artifact")
    if not model_artifact:
        st.error("⚠️ No machine learning model is loaded. Please load or train a model in the ML Lab.")
        return

    # Initialize session history
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    st.markdown("#### Select a Quick Example or Enter Custom Text:")
    # Example buttons in columns
    cols = st.columns(len(SAMPLE_PROMPTS))
    chosen_sample = None
    for idx, prompt in enumerate(SAMPLE_PROMPTS):
        with cols[idx]:
            label = prompt[:20] + "..."
            if st.button(label, key=f"sample_btn_{idx}", help=prompt):
                chosen_sample = prompt

    # Text input area
    default_text = chosen_sample if chosen_sample else ""
    user_input = st.text_area(
        "Enter Social Media Text / Tweet / Comment:",
        value=default_text,
        height=110,
        max_chars=500,
        placeholder="Type or paste any customer review, tweet, or statement here...",
        help="Input is processed strictly in-memory during this session.",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    with col_info:
        st.caption(f"Character count: {len(user_input)}/500 | Target Language: English")

    if analyze_clicked or chosen_sample:
        if not user_input.strip():
            st.warning("Please enter some text before analyzing.")
            return

        with st.spinner("Analyzing text representations..."):
            result = predict_single_text(user_input, model_artifact)

        if result.error:
            st.error(result.error)
            return

        # Store in session history
        st.session_state.prediction_history.append({
            "Text": user_input,
            "Predicted_Sentiment": result.predicted_sentiment,
            "Confidence": round(result.confidence_score * 100, 1),
            "Latency_ms": result.latency_ms,
        })

        # Display Outcome Card
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Prediction Results")

        res_c1, res_c2, res_c3 = st.columns([1.2, 1, 1])

        with res_c1:
            st.markdown(f"**Predicted Sentiment:**")
            st.markdown(f"<div style='margin-top: 8px;'>{render_sentiment_badge(result.predicted_sentiment)}</div>", unsafe_allow_html=True)

        with res_c2:
            st.markdown(f"**Confidence Score:**")
            st.markdown(f"<h3 style='margin: 0; color: #F8FAFC;'>{result.confidence_score * 100:.1f}%</h3>", unsafe_allow_html=True)

        with res_c3:
            st.markdown(f"**Inference Latency:**")
            st.markdown(f"<h3 style='margin: 0; color: #38BDF8;'>{result.latency_ms:.1f} ms</h3>", unsafe_allow_html=True)

        # Uncertainty alert if applicable
        if result.is_uncertain:
            st.markdown(
                f"""
                <div class="badge-warning" style="margin-top: 16px;">
                    ⚠️ <b>Prediction Uncertainty Notice:</b> {result.uncertainty_reason}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Probabilities breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Probability Distribution:")
        p_cols = st.columns(3)
        for idx, cls in enumerate(CANONICAL_SENTIMENTS):
            prob = result.probabilities.get(cls, 0.0)
            with p_cols[idx]:
                st.markdown(f"**{cls}**: `{prob * 100:.1f}%`")
                st.progress(float(prob))

        # Token Attribution / Explainability
        if result.token_explanations:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔬 Token-Level Feature Attribution:")
            st.caption(
                "Highlighted words show learned vocabulary weights supporting or opposing this classification. "
                "Feature importance reflects statistical association in training data, not causal intent."
            )

            html_tokens = []
            for item in result.token_explanations:
                tok = item["token"]
                wt = item["weight"]
                impact = item["impact"]

                if "Supports" in impact:
                    html_tokens.append(f"<span class='token-support' title='Weight: {wt} ({impact})'>{tok} (▲ {wt})</span>")
                elif "Opposes" in impact:
                    html_tokens.append(f"<span class='token-oppose' title='Weight: {wt} ({impact})'>{tok} (▼ {wt})</span>")
                else:
                    html_tokens.append(f"<span class='token-neutral' title='Neutral / OOV'>{tok}</span>")

            st.markdown(" ".join(html_tokens), unsafe_allow_html=True)

    # Session Prediction History
    if st.session_state.prediction_history:
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("### 📜 Session Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)

        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Session Predictions (CSV)",
            data=csv_data,
            file_name="sentiment_predictions_history.csv",
            mime="text/csv",
        )
