#!/usr/bin/env python3
"""
SentimentScope AI — Model Validation Script
Tests the persisted production model artifact against validation test cases.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sentimentscope.config import DEFAULT_MODEL_PATH
from src.sentimentscope.ml.model_registry import load_model_artifact
from src.sentimentscope.ml.prediction import predict_single_text


def main():
    print("=" * 65)
    print("  SentimentScope AI — Production Model Smoke Validation")
    print("=" * 65)

    print(f"\n[1/3] Loading model artifact from: {DEFAULT_MODEL_PATH}")
    artifact, err = load_model_artifact(DEFAULT_MODEL_PATH)
    if err or artifact is None:
        print(f"FAILED to load model: {err}")
        sys.exit(1)

    print(f"      Model Name: {artifact['model_name']}")
    print(f"      Trained At: {artifact['trained_at']}")
    print(f"      Classes: {artifact['classes']}")
    print(f"      Held-out Accuracy: {artifact['metrics']['accuracy']*100:.2f}%")
    print(f"      Held-out Macro F1: {artifact['metrics']['macro_f1']:.4f}")

    print("\n[2/3] Executing inference validation on test cases...")
    test_cases = [
        ("I absolutely love this new update! Everything is fast, smooth, and beautiful! 🚀", "Positive"),
        ("This is the worst software update ever. Constantly crashes and ruined my work! 😡", "Negative"),
        ("The quarterly earnings call is scheduled for Thursday at 4 PM EST.", "Neutral"),
        ("Not bad at all, actually exceeded my modest expectations.", "Positive"),
        ("I will never recommend this service to anyone, total disaster.", "Negative"),
    ]

    all_passed = True
    for text, expected in test_cases:
        res = predict_single_text(text, artifact)
        is_match = (res.predicted_sentiment == expected)
        mark = "✓ PASS" if is_match else "✗ MISMATCH"
        if not is_match:
            all_passed = False
        print(f"\n      Input: '{text}'")
        print(f"      Expected: {expected} | Predicted: {res.predicted_sentiment} ({mark})")
        print(f"      Confidence: {res.confidence_score*100:.1f}% | Latency: {res.latency_ms:.2f}ms")
        print(f"      Probabilities: {res.probabilities}")

    print("\n[3/3] Validation Result:")
    if all_passed:
        print("      ALL TEST CASES PASSED VALIDATION!")
    else:
        print("      NOTICE: Some test cases differed from expected labels (review domain nuances).")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
