"""
SentimentScope AI — ML Lab & Model Comparison Page
"""

import streamlit as st
import pandas as pd

from src.sentimentscope.analytics.visualizations import plot_confusion_matrix, plot_roc_curves
from src.sentimentscope.config import DEFAULT_MODEL_PATH, RANDOM_SEED
from src.sentimentscope.ml.model_registry import load_experiment_history, save_model_artifact
from src.sentimentscope.ml.prediction import predict_batch_texts
from src.sentimentscope.ml.training import run_full_experiment_suite
from src.sentimentscope.ui.components import render_hero, render_metric_card


def render_ml_lab_page():
    render_hero(
        title="ML Lab — Model Benchmark & Evaluation",
        subtitle="Rigorous Comparison Across Classical ML Algorithms, Metrics, and Error Analyses",
        badge_text="Leakage-Safe Pipelines",
    )

    model_artifact = st.session_state.get("model_artifact")
    active_df: pd.DataFrame = st.session_state.get("active_df", pd.DataFrame())

    # 1. Model Leaderboard
    st.markdown("### 🏆 Model Comparison Leaderboard")
    st.caption(
        "Evaluated on a strictly held-out test split (20%) using Stratified K-Fold validation and fixed random seed (42)."
    )

    history = load_experiment_history()
    if history:
        lb_df = pd.DataFrame(history)
        # Reorder columns nicely
        display_cols = [
            "Model", "Representation", "Accuracy", "Macro F1", "Weighted F1",
            "Macro Precision", "Macro Recall", "ROC-AUC", "Train Time (s)", "Latency (ms/sample)"
        ]
        cols_present = [c for c in display_cols if c in lb_df.columns]
        st.dataframe(lb_df[cols_present], use_container_width=True)
    else:
        st.info("Leaderboard records not found. Train or benchmark models using the controls below.")

    # 2. Detailed Evaluation of Active Model
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔬 Production Model Diagnostics")

    if model_artifact:
        metrics = model_artifact.get("metrics", {})
        m_name = model_artifact.get("model_name", "Production Model")
        classes = model_artifact.get("classes", ["Negative", "Neutral", "Positive"])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Accuracy", f"{metrics.get('accuracy', 0.0)*100:.2f}%", "PRD Target: >= 80%", "#10B981")
        with m2:
            render_metric_card("Macro F1", f"{metrics.get('macro_f1', 0.0):.4f}", "PRD Target: >= 0.78", "#6366F1")
        with m3:
            render_metric_card("Weighted F1", f"{metrics.get('weighted_f1', 0.0):.4f}", "Class-size weighted", "#38BDF8")
        with m4:
            render_metric_card("Inference Latency", f"{metrics.get('inference_latency_ms', 0.0):.2f} ms", "Per-sample latency", "#F59E0B")

        st.markdown("<br>", unsafe_allow_html=True)
        col_cm, col_roc = st.columns([1, 1])

        with col_cm:
            cm = metrics.get("confusion_matrix")
            if cm:
                fig_cm = plot_confusion_matrix(cm, classes)
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.write("Confusion matrix data unavailable.")

        with col_roc:
            roc_curves = metrics.get("roc_curves")
            fig_roc = plot_roc_curves(roc_curves)
            st.plotly_chart(fig_roc, use_container_width=True)

        # Per-Class Classification Report
        per_class = metrics.get("per_class", {})
        if per_class:
            st.markdown("#### Per-Class Performance Breakdown:")
            per_class_df = pd.DataFrame(per_class).T.reset_index().rename(columns={"index": "Class"})
            st.dataframe(per_class_df, use_container_width=True)
    else:
        st.warning("No production model metrics loaded.")

    # 3. Misclassified Sample Inspector (Phase 5 Feature)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🕵️ Misclassified Sample Inspector")
    st.caption(
        "Inspect samples where model predictions differ from ground-truth labels to diagnose ambiguous phrasing, negation nuance, or labeling noise."
    )

    if model_artifact and not active_df.empty and "Sentiment" in active_df.columns:
        # Check if predictions are computed
        if "Predicted_Sentiment" not in active_df.columns:
            if st.button("Generate Predictions on Active Dataset to Inspect Errors"):
                with st.spinner("Generating predictions..."):
                    preds_df = predict_batch_texts(active_df["Text"].tolist(), model_artifact)
                    active_df["Predicted_Sentiment"] = preds_df["Predicted_Sentiment"].values
                    active_df["Model_Confidence"] = preds_df["Confidence"].values
                    st.session_state.active_df = active_df
                    st.rerun()

        if "Predicted_Sentiment" in active_df.columns:
            misclassified = active_df[active_df["Sentiment"] != active_df["Predicted_Sentiment"]]
            total_eval = len(active_df)
            err_count = len(misclassified)
            err_rate = (err_count / total_eval) * 100 if total_eval > 0 else 0.0

            st.write(f"**Misclassified Records:** `{err_count}` / `{total_eval}` ({err_rate:.2f}% error rate)")

            if err_count > 0:
                st.dataframe(
                    misclassified[["Post ID", "Text", "Sentiment", "Predicted_Sentiment", "Model_Confidence"]],
                    use_container_width=True,
                )
            else:
                st.success("🎉 Zero misclassifications detected on this dataset!")
    else:
        st.info("Load a dataset with ground-truth sentiment labels to enable the error inspector.")

    # 4. Retraining / Benchmark Controls
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Retrain & Benchmark Models on Active Dataset")
    st.caption("Execute a complete benchmark across all 4 classical machine learning architectures.")

    c_rep, c_btn = st.columns([2, 1])
    with c_rep:
        selected_rep = st.radio(
            "Feature Representations to Benchmark:",
            options=["TF-IDF Only", "Bag of Words (BoW) Only", "Both TF-IDF and BoW"],
            index=2,
            horizontal=True,
        )

    with c_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        train_clicked = st.button("🚀 Run ML Benchmark", type="primary", use_container_width=True)

    if train_clicked:
        if active_df.empty or "Sentiment" not in active_df.columns:
            st.error("Cannot train: Active dataset is empty or lacks a ground-truth 'Sentiment' column.")
            return

        feature_map = {
            "TF-IDF Only": ["tfidf"],
            "Bag of Words (BoW) Only": ["bow"],
            "Both TF-IDF and BoW": ["tfidf", "bow"],
        }
        feats = feature_map[selected_rep]

        with st.spinner("Training and evaluating all models..."):
            suite = run_full_experiment_suite(
                active_df,
                text_col="Text",
                label_col="Sentiment",
                feature_types=feats,
                seed=RANDOM_SEED,
            )

            # Persist best model
            best_id = suite["best_model_id"]
            best_exp = suite["experiments"][best_id]
            best_pipe = best_exp["pipeline"]
            best_met = best_exp["metrics"]

            artifact_path = save_model_artifact(
                pipeline=best_pipe,
                model_name=best_exp["display_name"],
                feature_type=best_exp["feature_type"],
                metrics={
                    "accuracy": best_met.accuracy,
                    "macro_f1": best_met.macro_f1,
                    "weighted_f1": best_met.weighted_f1,
                    "precision_macro": best_met.precision_macro,
                    "recall_macro": best_met.recall_macro,
                    "per_class": best_met.per_class,
                    "confusion_matrix": best_met.confusion_matrix,
                    "roc_auc_macro": best_met.roc_auc_macro,
                    "roc_curves": best_met.roc_curves,
                    "training_duration_seconds": best_met.training_duration_seconds,
                    "inference_latency_ms": best_met.inference_latency_ms,
                },
                classes=best_met.classes,
                save_path=DEFAULT_MODEL_PATH,
                random_seed=RANDOM_SEED,
            )

            # Update session state
            st.session_state.model_artifact = {
                "pipeline": best_pipe,
                "model_name": best_exp["display_name"],
                "feature_type": best_exp["feature_type"],
                "metrics": {
                    "accuracy": best_met.accuracy,
                    "macro_f1": best_met.macro_f1,
                    "weighted_f1": best_met.weighted_f1,
                    "per_class": best_met.per_class,
                    "confusion_matrix": best_met.confusion_matrix,
                    "roc_curves": best_met.roc_curves,
                    "inference_latency_ms": best_met.inference_latency_ms,
                },
                "classes": best_met.classes,
            }

            st.success(
                f"Benchmark complete! Top model '{best_exp['display_name']}' saved with Accuracy: {best_met.accuracy*100:.2f}% and Macro F1: {best_met.macro_f1:.4f}."
            )
            st.rerun()
