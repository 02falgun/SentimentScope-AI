#!/usr/bin/env python3
"""
SentimentScope AI — Model Training Script
Trains, evaluates, benchmarks, and persists the production sentiment model pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.sentimentscope.config import (
    DEFAULT_EXPERIMENT_LOG,
    DEFAULT_MODEL_PATH,
    DEFAULT_SAMPLE_DATASET,
    RANDOM_SEED,
)
from src.sentimentscope.data.loading import load_csv
from src.sentimentscope.data.validation import validate_and_profile_dataframe
from src.sentimentscope.ml.model_registry import save_experiment_history, save_model_artifact
from src.sentimentscope.ml.training import run_full_experiment_suite


def main():
    print("=" * 70)
    print("  SentimentScope AI — Production Model Training & Benchmark Pipeline")
    print("=" * 70)

    dataset_path = DEFAULT_SAMPLE_DATASET
    print(f"\n[1/5] Loading dataset from: {dataset_path}")
    df_raw, err = load_csv(dataset_path)
    if err or df_raw is None:
        print(f"Error loading dataset: {err}")
        sys.exit(1)

    print(f"      Loaded {len(df_raw)} raw rows.")

    print("\n[2/5] Validating schema and profiling dataset quality...")
    cleaned_df, profile = validate_and_profile_dataframe(
        df_raw,
        text_col="Text",
        sentiment_col="Sentiment",
    )
    print(f"      Retained {profile.retained_rows} valid rows after deduplication.")
    print(f"      Class distribution: {profile.class_distribution}")

    if profile.warnings:
        print(f"      Quality notices: {profile.warnings}")

    print("\n[3/5] Executing rigorous benchmark across 4 algorithms & 2 representations...")
    print("      Algorithms: Logistic Regression, Naive Bayes, Linear SVM, Random Forest")
    print("      Representations: TF-IDF (primary), Bag of Words (comparison)")
    print(f"      Random Seed: {RANDOM_SEED} (Strictly reproducible)")

    experiment_results = run_full_experiment_suite(
        cleaned_df,
        text_col="Text",
        label_col="Sentiment",
        feature_types=["tfidf", "bow"],
        seed=RANDOM_SEED,
    )

    leaderboard = experiment_results["leaderboard"]
    print("\n" + "=" * 70)
    print("  MODEL BENCHMARK LEADERBOARD (Held-Out Test Set)")
    print("=" * 70)

    # Format leaderboard table
    headers = ["Model", "Rep", "Accuracy", "Macro F1", "Weighted F1", "Train(s)", "Latency(ms)"]
    table_rows = [
        [
            r["Model"],
            r["Representation"],
            f"{r['Accuracy']*100:.2f}%",
            f"{r['Macro F1']:.4f}",
            f"{r['Weighted F1']:.4f}",
            f"{r['Train Time (s)']:.3f}",
            f"{r['Latency (ms/sample)']:.2f}",
        ]
        for r in leaderboard
    ]
    try:
        from tabulate import tabulate
        print(tabulate(table_rows, headers=headers, tablefmt="grid"))
    except ImportError:
        header_str = " | ".join(f"{h:<14}" for h in headers)
        print(header_str)
        print("-" * len(header_str))
        for row in table_rows:
            print(" | ".join(f"{str(c):<14}" for c in row))

    best_id = experiment_results["best_model_id"]
    best_exp = experiment_results["experiments"][best_id]
    best_metrics = best_exp["metrics"]
    best_pipeline = best_exp["pipeline"]

    print("\n[4/5] Selected Production Model:")
    print(f"      Winner: {best_exp['display_name']}")
    print(f"      Accuracy: {best_metrics.accuracy * 100:.2f}% (PRD Target: >= 80%)")
    print(f"      Macro F1: {best_metrics.macro_f1:.4f} (PRD Target: >= 0.78)")
    print(f"      Confusion Matrix (Classes: {best_metrics.classes}):")
    for row in best_metrics.confusion_matrix:
        print(f"        {row}")

    print("\n[5/5] Persisting Model Artifact and Experiment History...")
    save_model_artifact(
        pipeline=best_pipeline,
        model_name=best_exp["display_name"],
        feature_type=best_exp["feature_type"],
        metrics={
            "accuracy": best_metrics.accuracy,
            "macro_f1": best_metrics.macro_f1,
            "weighted_f1": best_metrics.weighted_f1,
            "precision_macro": best_metrics.precision_macro,
            "recall_macro": best_metrics.recall_macro,
            "per_class": best_metrics.per_class,
            "confusion_matrix": best_metrics.confusion_matrix,
            "roc_auc_macro": best_metrics.roc_auc_macro,
            "roc_curves": best_metrics.roc_curves,
            "training_duration_seconds": best_metrics.training_duration_seconds,
            "inference_latency_ms": best_metrics.inference_latency_ms,
        },
        classes=best_metrics.classes,
        save_path=DEFAULT_MODEL_PATH,
        random_seed=RANDOM_SEED,
    )
    print(f"      Production model saved to: {DEFAULT_MODEL_PATH}")

    # Serialize experiment history (convert non-serializable objects)
    serializable_leaderboard = [
        {k: (v if not isinstance(v, float) or not pd.isna(v) else None) for k, v in row.items()}
        for row in leaderboard
    ]
    save_experiment_history(serializable_leaderboard, DEFAULT_EXPERIMENT_LOG)
    print(f"      Experiment history saved to: {DEFAULT_EXPERIMENT_LOG}")

    print("\n" + "=" * 70)
    print("  Training and Benchmark Complete Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
