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
    page_title="SentimentScope AI — Social Sentiment & ML Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Custom Theme CSS
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

    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 8px 0 16px 0;">
                <h2 style="margin: 0; font-size: 20px; font-weight: 700; color: #FFFFFF;">
                    🔬 SentimentScope
                </h2>
                <div style="color: #94A3B8; font-size: 12px; margin-top: 4px;">
                    AI Sentiment & Trend Lab
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_choice = st.radio(
            "Navigation:",
            options=[
                "🏠 Home / Overview",
                "⚡ Live Prediction",
                "📁 Dataset Explorer",
                "🔬 ML Lab / Benchmark",
                "📊 Visual Analytics",
                "🔍 Keyword Search",
                "📄 Reports & Export",
                "📘 About & Architecture",
            ],
            index=0,
        )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.06);'>", unsafe_allow_html=True)

        # Active Model Sidebar Pill
        st.markdown("#### 🤖 Model Status")
        model_artifact = st.session_state.get("model_artifact")
        if model_artifact:
            m_name = model_artifact.get("model_name", "Production Model")
            acc = model_artifact.get("metrics", {}).get("accuracy", 0.0) * 100
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px 12px;">
                    <div style="color: #10B981; font-weight: 600; font-size: 13px;">● Model Active</div>
                    <div style="color: #F8FAFC; font-size: 12px; margin-top: 4px;">{m_name}</div>
                    <div style="color: #94A3B8; font-size: 11px;">Accuracy: {acc:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 10px 12px;">
                    <div style="color: #EF4444; font-weight: 600; font-size: 13px;">● No Model Loaded</div>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 4px;">Train in ML Lab</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Dataset Status Pill
        active_df = st.session_state.get("active_df")
        row_cnt = len(active_df) if active_df is not None and not active_df.empty else 0
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Dataset Status")
        st.markdown(
            f"""
            <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 10px 12px;">
                <div style="color: #818CF8; font-weight: 600; font-size: 13px;">● Active Dataset</div>
                <div style="color: #F8FAFC; font-size: 12px; margin-top: 4px;">{st.session_state.get('dataset_name', 'None')}</div>
                <div style="color: #94A3B8; font-size: 11px;">{row_cnt:,} Cleaned Posts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
        st.caption("SentimentScope AI | Streamlit Cloud Ready | Free Tier Optimized")

    # Main Router
    if nav_choice == "🏠 Home / Overview":
        render_home_page()
    elif nav_choice == "⚡ Live Prediction":
        render_prediction_page()
    elif nav_choice == "📁 Dataset Explorer":
        render_dataset_explorer()
    elif nav_choice == "🔬 ML Lab / Benchmark":
        render_ml_lab_page()
    elif nav_choice == "📊 Visual Analytics":
        render_analytics_page()
    elif nav_choice == "🔍 Keyword Search":
        render_search_page()
    elif nav_choice == "📄 Reports & Export":
        render_reports_page()
    elif nav_choice == "📘 About & Architecture":
        render_about_page()


if __name__ == "__main__":
    main()
