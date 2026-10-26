"""
SentimentScope AI — Main Streamlit Application Entry Point
AI-Powered Social Media Sentiment Analysis & Trend Monitoring System
"""

import sys
from pathlib import Path

# Add project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.sentimentscope.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SAMPLE_DATASET,
    RANDOM_SEED,
)
from src.sentimentscope.data.loading import load_csv
from src.sentimentscope.data.validation import validate_and_profile_dataframe
from src.sentimentscope.ml.model_registry import load_model_artifact, save_model_artifact
from src.sentimentscope.ml.training import run_full_experiment_suite
from src.sentimentscope.ui.components import apply_theme
from src.sentimentscope.ui.pages.about_page import render_about_page
from src.sentimentscope.ui.pages.analytics_page import render_analytics_page
from src.sentimentscope.ui.pages.dataset_explorer import render_dataset_explorer
from src.sentimentscope.ui.pages.home import render_home_page
from src.sentimentscope.ui.pages.ml_lab import render_ml_lab_page
from src.sentimentscope.ui.pages.prediction_page import render_prediction_page
from src.sentimentscope.ui.pages.reports_page import render_reports_page
from src.sentimentscope.ui.pages.search_page import render_search_page

# Page Configuration
st.set_page_config(
    page_title="SentimentScope AI — AI Sentiment & ML Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Custom Monochrome Theme CSS
apply_theme()


@st.cache_resource
def get_cached_model():
    """Load model artifact once and cache across app reruns."""
    artifact, err = load_model_artifact(DEFAULT_MODEL_PATH)
    if artifact is not None:
        return artifact

    # If artifact is missing, auto-train on sample dataset for first-run zero-setup convenience
    if DEFAULT_SAMPLE_DATASET.exists():
        df_sample, _ = load_csv(DEFAULT_SAMPLE_DATASET)
        if df_sample is not None and not df_sample.empty:
            cleaned_df, _ = validate_and_profile_dataframe(df_sample, text_col="Text", sentiment_col="Sentiment")
            results = run_full_experiment_suite(cleaned_df, text_col="Text", label_col="Sentiment", feature_types=["tfidf"], seed=RANDOM_SEED)
            best_id = results["best_model_id"]
            best_exp = results["experiments"][best_id]
            save_model_artifact(
                pipeline=best_exp["pipeline"],
                model_name=best_exp["display_name"],
                feature_type=best_exp["feature_type"],
                metrics={
                    "accuracy": best_exp["metrics"].accuracy,
                    "macro_f1": best_exp["metrics"].macro_f1,
                    "weighted_f1": best_exp["metrics"].weighted_f1,
                    "per_class": best_exp["metrics"].per_class,
                    "confusion_matrix": best_exp["metrics"].confusion_matrix,
                    "training_duration_seconds": best_exp["metrics"].training_duration_seconds,
                    "inference_latency_ms": best_exp["metrics"].inference_latency_ms,
                },
                classes=best_exp["metrics"].classes,
                save_path=DEFAULT_MODEL_PATH,
                random_seed=RANDOM_SEED,
            )
            artifact, _ = load_model_artifact(DEFAULT_MODEL_PATH)
            return artifact

    return None


@st.cache_data
def get_cached_sample_data():
    """Load sample dataset once and cache."""
    df, err = load_csv(DEFAULT_SAMPLE_DATASET)
    if df is not None:
        cleaned_df, _ = validate_and_profile_dataframe(df, text_col="Text", sentiment_col="Sentiment")
        return cleaned_df
    return pd.DataFrame()


def init_session_state():
    """Ensure baseline session variables are populated on first launch."""
    if "model_artifact" not in st.session_state or st.session_state.model_artifact is None:
        st.session_state.model_artifact = get_cached_model()

    if "active_df" not in st.session_state or st.session_state.active_df is None or st.session_state.active_df.empty:
        st.session_state.active_df = get_cached_sample_data()
        st.session_state.dataset_name = "Built-In Sample Dataset (360 Posts)"


def main():
    init_session_state()

    # Sidebar Navigation & Application Shell
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 4px 0 16px 0; border-bottom: 1px solid #E5E5E5; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 20px;">🔬</span>
                    <span style="font-size: 17px; font-weight: 700; color: #111111; letter-spacing: -0.02em;">
                        SentimentScope AI
                    </span>
                </div>
                <div style="color: #666666; font-size: 12px; margin-top: 4px; margin-left: 28px;">
                    AI Sentiment & Trend Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("NAVIGATION")
        nav_choice = st.radio(
            "Select Section:",
            options=[
                "Overview Dashboard",
                "Live Prediction",
                "Dataset Explorer",
                "ML Lab & Benchmark",
                "Visual Analytics",
                "Keyword Search",
                "Reports & Export",
                "About & Architecture",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<div style='border-top: 1px solid #E5E5E5; margin: 18px 0;'></div>", unsafe_allow_html=True)

        # Active Model Sidebar Status Card
        st.caption("SYSTEM STATUS")
        model_artifact = st.session_state.get("model_artifact")
        if model_artifact:
            m_name = model_artifact.get("model_name", "Production Model")
            acc = model_artifact.get("metrics", {}).get("accuracy", 0.0) * 100
            st.markdown(
                f"""
                <div class="sidebar-status-card">
                    <div style="display: flex; align-items: center; font-size: 12px; font-weight: 600; color: #111111;">
                        <span class="status-dot-active"></span> Model Active
                    </div>
                    <div style="color: #333333; font-size: 12px; margin-top: 4px; font-weight: 500;">{m_name}</div>
                    <div style="color: #666666; font-size: 11px; margin-top: 2px;">Accuracy: {acc:.1f}% (Held-out)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="sidebar-status-card">
                    <div style="display: flex; align-items: center; font-size: 12px; font-weight: 600; color: #111111;">
                        <span class="status-dot-inactive"></span> No Model Loaded
                    </div>
                    <div style="color: #666666; font-size: 11px; margin-top: 4px;">Train in ML Lab</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Dataset Status Sidebar Card
        active_df = st.session_state.get("active_df")
        row_cnt = len(active_df) if active_df is not None and not active_df.empty else 0
        st.markdown(
            f"""
            <div class="sidebar-status-card">
                <div style="display: flex; align-items: center; font-size: 12px; font-weight: 600; color: #111111;">
                    <span class="status-dot-active"></span> Active Dataset
                </div>
                <div style="color: #333333; font-size: 12px; margin-top: 4px; font-weight: 500;">
                    {st.session_state.get('dataset_name', 'None')}
                </div>
                <div style="color: #666666; font-size: 11px; margin-top: 2px;">{row_cnt:,} Cleaned Posts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='border-top: 1px solid #E5E5E5; margin: 18px 0;'></div>", unsafe_allow_html=True)
        st.caption("SentimentScope AI | Production Release v2.0 | Pure Python & Streamlit Cloud Ready")

    # Main Router
    if nav_choice == "Overview Dashboard":
        render_home_page()
    elif nav_choice == "Live Prediction":
        render_prediction_page()
    elif nav_choice == "Dataset Explorer":
        render_dataset_explorer()
    elif nav_choice == "ML Lab & Benchmark":
        render_ml_lab_page()
    elif nav_choice == "Visual Analytics":
        render_analytics_page()
    elif nav_choice == "Keyword Search":
        render_search_page()
    elif nav_choice == "Reports & Export":
        render_reports_page()
    elif nav_choice == "About & Architecture":
        render_about_page()


if __name__ == "__main__":
    main()
