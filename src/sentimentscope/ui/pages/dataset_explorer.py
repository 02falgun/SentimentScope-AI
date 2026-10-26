"""
SentimentScope AI — Dataset Explorer and Upload Page
"""

import streamlit as st
import pandas as pd

from src.sentimentscope.config import DEFAULT_SAMPLE_DATASET, MAX_UPLOAD_SIZE_MB
from src.sentimentscope.data.loading import detect_all_likely_columns, load_csv
from src.sentimentscope.data.validation import validate_and_profile_dataframe
from src.sentimentscope.ml.prediction import predict_batch_texts
from src.sentimentscope.ui.components import render_hero, render_metric_card


def render_dataset_explorer():
    render_hero(
        title="Dataset Explorer & Ingestion",
        subtitle="Upload Custom Datasets, Validate Data Quality, and Inspect Schema Integrity",
        badge_text="Ephemeral In-Memory",
    )

    # 1. Dataset Upload or Reset Section
    st.markdown("### 📁 Dataset Source")
    col_up, col_reset = st.columns([3, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            f"Upload a CSV file (Max {MAX_UPLOAD_SIZE_MB}MB):",
            type=["csv"],
            help="Uploaded data is processed strictly in-memory during this session and is not permanently saved to disk.",
        )

    with col_reset:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset to Sample Data", use_container_width=True):
            df_sample, err = load_csv(DEFAULT_SAMPLE_DATASET)
            if df_sample is not None:
                st.session_state.active_df = df_sample
                st.session_state.dataset_name = "Built-In Sample Dataset"
                st.success("Reset to default sample dataset!")
                st.rerun()

    # Load uploaded file if provided
    if uploaded_file is not None:
        with st.spinner("Parsing uploaded CSV file..."):
            raw_df, err = load_csv(uploaded_file)
            if err or raw_df is None:
                st.error(f"Error loading CSV: {err}")
                return
            st.session_state.raw_uploaded_df = raw_df
            st.session_state.dataset_name = uploaded_file.name

    # Determine working raw DataFrame
    if "raw_uploaded_df" in st.session_state and st.session_state.raw_uploaded_df is not None:
        working_raw = st.session_state.raw_uploaded_df
    else:
        if "active_df" not in st.session_state or st.session_state.active_df is None or st.session_state.active_df.empty:
            df_sample, _ = load_csv(DEFAULT_SAMPLE_DATASET)
            st.session_state.active_df = df_sample
            st.session_state.dataset_name = "Built-In Sample Dataset"
        working_raw = st.session_state.active_df

    if working_raw.empty:
        st.warning("No dataset is currently loaded.")
        return

    # 2. Schema Mapping Controls
    st.markdown("### 🗂️ Column Schema Mapping")
    st.caption("Automatic heuristic guessing has mapped the columns below. Adjust if necessary.")

    cols = list(working_raw.columns)
    detected = detect_all_likely_columns(cols)

    c1, c2, c3 = st.columns(3)
    with c1:
        text_col = st.selectbox(
            "Text Column (Mandatory):",
            options=cols,
            index=cols.index(detected["text"]) if detected["text"] in cols else 0,
        )
        sentiment_col = st.selectbox(
            "Ground-Truth Sentiment Column (Optional):",
            options=["None"] + cols,
            index=(cols.index(detected["sentiment"]) + 1) if detected["sentiment"] in cols else 0,
        )
    with c2:
        platform_col = st.selectbox(
            "Platform Column (Optional):",
            options=["None"] + cols,
            index=(cols.index(detected["platform"]) + 1) if detected["platform"] in cols else 0,
        )
        date_col = st.selectbox(
            "Date Column (Optional):",
            options=["None"] + cols,
            index=(cols.index(detected["date"]) + 1) if detected["date"] in cols else 0,
        )
    with c3:
        likes_col = st.selectbox(
            "Likes / Upvotes (Optional):",
            options=["None"] + cols,
            index=(cols.index(detected["likes"]) + 1) if detected["likes"] in cols else 0,
        )
        shares_col = st.selectbox(
            "Shares / Retweets (Optional):",
            options=["None"] + cols,
            index=(cols.index(detected["shares"]) + 1) if detected["shares"] in cols else 0,
        )

    # Validate and profile
    cleaned_df, profile = validate_and_profile_dataframe(
        working_raw,
        text_col=text_col,
        sentiment_col=sentiment_col if sentiment_col != "None" else None,
        platform_col=platform_col if platform_col != "None" else None,
        date_col=date_col if date_col != "None" else None,
        likes_col=likes_col if likes_col != "None" else None,
        shares_col=shares_col if shares_col != "None" else None,
    )

    st.session_state.active_df = cleaned_df
    st.session_state.quality_profile = profile

    # 3. Data Quality Profile Summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🛡️ Data Quality & Row Attrition Profile")

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        render_metric_card("Total Ingested", f"{profile.total_loaded:,}", "Raw rows loaded")
    with q2:
        render_metric_card("Duplicates Dropped", f"{profile.duplicates_removed:,}", "Identical text removed", "#F59E0B")
    with q3:
        render_metric_card("Missing Text Dropped", f"{profile.missing_text_removed:,}", "Empty / null text", "#EF4444")
    with q4:
        render_metric_card("Cleaned Retained", f"{profile.retained_rows:,}", "Ready for modeling", "#10B981")

    # Quality warnings banner
    if profile.warnings:
        for w in profile.warnings:
            st.warning(f"⚠️ {w}")

    # 4. Batch Prediction Action (if sentiment is not present or user wants model inferences)
    model_artifact = st.session_state.get("model_artifact")
    if model_artifact and not cleaned_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### ⚡ Batch Model Inference")
        b1, b2 = st.columns([1, 2])
        with b1:
            if st.button("🚀 Predict Sentiments with Active Model", type="primary", use_container_width=True):
                with st.spinner("Generating batch model predictions..."):
                    preds_df = predict_batch_texts(cleaned_df["Text"].tolist(), model_artifact)
                    cleaned_df["Predicted_Sentiment"] = preds_df["Predicted_Sentiment"].values
                    cleaned_df["Model_Confidence"] = preds_df["Confidence"].values
                    st.session_state.active_df = cleaned_df
                    st.success("Batch predictions appended to active dataset!")

    # 5. Interactive Data Preview & Filtering
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Active Dataset Preview & Filter")

    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        avail_plats = list(cleaned_df["Platform"].unique()) if "Platform" in cleaned_df.columns else []
        selected_plats = st.multiselect("Filter Platform:", options=avail_plats, default=avail_plats)
    with f2:
        sentiment_col_name = "Sentiment" if "Sentiment" in cleaned_df.columns else ("Predicted_Sentiment" if "Predicted_Sentiment" in cleaned_df.columns else None)
        avail_sents = list(cleaned_df[sentiment_col_name].unique()) if sentiment_col_name else []
        selected_sents = st.multiselect("Filter Sentiment:", options=avail_sents, default=avail_sents)
    with f3:
        search_query = st.text_input("Search Text in Dataset:", placeholder="Filter by text keyword...")

    # Filter dataframe
    filtered_df = cleaned_df.copy()
    if selected_plats and "Platform" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Platform"].isin(selected_plats)]
    if selected_sents and sentiment_col_name in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[sentiment_col_name].isin(selected_sents)]
    if search_query.strip():
        filtered_df = filtered_df[filtered_df["Text"].astype(str).str.contains(search_query, case=False, na=False)]

    st.caption(f"Displaying {len(filtered_df):,} of {len(cleaned_df):,} records")
    st.dataframe(filtered_df, use_container_width=True, height=350)

    # 6. CSV Download
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Cleaned Dataset (CSV)",
        data=csv_bytes,
        file_name="sentimentscope_cleaned_dataset.csv",
        mime="text/csv",
    )
