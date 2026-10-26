"""
SentimentScope AI — Model Evaluation Module
Calculates accuracy, precision, recall, macro/weighted F1, confusion matrices,
per-class classification metrics, and ROC-AUC curves.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.sentimentscope.config import CANONICAL_SENTIMENTS


@dataclass
class EvaluationMetrics:
    """Encapsulates all evaluation results for a single model."""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    precision_macro: float
    recall_macro: float
    per_class: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    classes: List[str]
    roc_auc_macro: Optional[float] = None
    roc_curves: Optional[Dict[str, Dict[str, List[float]]]] = None
    training_duration_seconds: float = 0.0
    inference_latency_ms: float = 0.0
    classification_report_str: str = ""


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    training_duration: float = 0.0,
    inference_latency: float = 0.0,
) -> EvaluationMetrics:
    """
    Computes rigorous metrics on held-out test predictions.
    """
    if classes is None:
        classes = [str(c) for c in sorted(list(set(y_true).union(set(y_pred))))]
    else:
        classes = [str(c) for c in classes]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    # Per-class metrics
    rep_dict = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    rep_str = classification_report(y_true, y_pred, labels=classes, zero_division=0)

    per_class: Dict[str, Dict[str, float]] = {}
    for cls in classes:
        if cls in rep_dict:
            per_class[cls] = {
                "precision": round(float(rep_dict[cls]["precision"]), 4),
                "recall": round(float(rep_dict[cls]["recall"]), 4),
                "f1": round(float(rep_dict[cls]["f1-score"]), 4),
                "support": int(rep_dict[cls]["support"]),
            }

    # ROC-AUC calculation (One-vs-Rest)
    roc_auc = None
    roc_curves = None
    if y_scores is not None and len(classes) >= 2:
        try:
            # Check if all classes present in y_true
            y_bin = label_binarize(y_true, classes=classes)
            if y_scores.ndim == 2 and y_scores.shape[1] == len(classes):
                # Ensure each class has both positive and negative cases
                valid_classes = True
                for i in range(len(classes)):
                    if len(np.unique(y_bin[:, i])) < 2:
                        valid_classes = False
                        break

                if valid_classes:
                    roc_auc = float(roc_auc_score(y_bin, y_scores, multi_class="ovr", average="macro"))
                    from sklearn.metrics import roc_curve
                    roc_curves = {}
                    for i, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
                        roc_curves[cls] = {
                            "fpr": [round(float(v), 4) for v in fpr],
                            "tpr": [round(float(v), 4) for v in tpr],
                        }
        except Exception:
            roc_auc = None
            roc_curves = None

    return EvaluationMetrics(
        accuracy=round(acc, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        precision_macro=round(prec_macro, 4),
        recall_macro=round(rec_macro, 4),
        per_class=per_class,
        confusion_matrix=cm,
        classes=classes,
        roc_auc_macro=round(roc_auc, 4) if roc_auc is not None else None,
        roc_curves=roc_curves,
        training_duration_seconds=round(training_duration, 4),
        inference_latency_ms=round(inference_latency, 2),
        classification_report_str=rep_str,
    )


COMMON_TEMPLATE_SUFFIXES = [
    "Published on official channel.",
    "Further details will follow.",
    "Under standard protocols.",
    "According to specifications.",
    "Will follow up with notes.",
    "As announced.",
    "Shared for internal review.",
    "Avoid at all costs.",
    "Expecting an immediate refund.",
    "Customer service was useless.",
    "Worst decision of the year.",
    "Absolutely furious about this.",
    "Total waste of time and money.",
    "Unbelievably bad.",
    "Definitely a step forward.",
    "Worth checking out.",
    "Solid quality overall.",
    "Five stars from me!",
    "Kudos to the team.",
]


def extract_base_template(text: str) -> str:
    """Strips common demonstration suffixes to identify base sentence templates."""
    t = str(text).strip().strip('"')
    for s in COMMON_TEMPLATE_SUFFIXES:
        if t.endswith(s):
            t = t[:-len(s)].strip()
    return t


def evaluate_template_generalization(
    df: Any,
    text_col: str = "Text",
    label_col: str = "Sentiment",
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Performs Group K-Fold cross-validation by base sentence template.
    Ensures no sentence template is shared between training and test sets,
    providing a scientifically sound out-of-template generalization benchmark.
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.pipeline import Pipeline

    if not isinstance(df, pd.DataFrame) or df.empty or text_col not in df.columns or label_col not in df.columns:
        return {"error": "Invalid dataframe provided"}

    df_eval = df.copy()
    df_eval["_base_template"] = df_eval[text_col].apply(extract_base_template)

    total_templates = int(df_eval["_base_template"].nunique())

    # Measure overlap under standard stratified 80/20 split
    strat_train, strat_test = train_test_split(
        df_eval, test_size=0.2, random_state=seed, stratify=df_eval[label_col]
    )
    train_templates = set(strat_train["_base_template"])
    test_rows_in_train = strat_test[strat_test["_base_template"].isin(train_templates)]
    standard_split_overlap_pct = round((len(test_rows_in_train) / len(strat_test)) * 100, 1)

    # Run Group Cross-Validation
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=seed)
    fold_results: List[Dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gss.split(df_eval, groups=df_eval["_base_template"])):
        train_fold = df_eval.iloc[train_idx]
        test_fold = df_eval.iloc[test_idx]

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
        ])

        pipe.fit(train_fold[text_col], train_fold[label_col])
        y_pred = pipe.predict(test_fold[text_col])

        acc = float(accuracy_score(test_fold[label_col], y_pred))
        f1 = float(f1_score(test_fold[label_col], y_pred, average="macro", zero_division=0))

        fold_results.append({
            "fold": fold_idx + 1,
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "test_samples": len(test_fold),
        })

    mean_acc = round(float(np.mean([r["accuracy"] for r in fold_results])), 4)
    mean_f1 = round(float(np.mean([r["macro_f1"] for r in fold_results])), 4)

    return {
        "total_templates": total_templates,
        "standard_split_template_overlap_pct": standard_split_overlap_pct,
        "group_cv_folds": fold_results,
        "group_cv_mean_accuracy": mean_acc,
        "group_cv_mean_macro_f1": mean_f1,
        "finding": (
            f"Under standard stratified split, {standard_split_overlap_pct}% of held-out test rows share "
            f"base templates with training data, explaining 100% test scores. Under strict group cross-validation "
            f"with isolated unseen templates, the model achieves {mean_acc*100:.2f}% accuracy and {mean_f1:.4f} Macro F1."
        ),
    }
