"""
Model Evaluation Module for SMS Spam Message Detection.
Team 2 - Natural Language Processing

Calculates comprehensive evaluation metrics:
- Accuracy, Precision, Recall, F1-Score (per-class and macro/weighted).
- Confusion Matrix: True Negatives (TN), False Positives (FP),
                    False Negatives (FN), True Positives (TP).
- Detailed classification reports and error explanations.
- Never fabricates or hardcodes metrics.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


def compute_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes complete classification metrics from actual predictions.
    
    Args:
        y_true: True binary labels (0 for ham, 1 for spam).
        y_pred: Predicted binary labels (0 for ham, 1 for spam).
        y_proba: Predicted probabilities for spam class (class 1), if available.
        model_name: Descriptive name of the model.
        
    Returns:
        Dictionary containing all metrics, confusion matrix counts, and explanations.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    
    # Class-specific metrics (Pos label = 1, which corresponds to SPAM)
    spam_precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    spam_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    spam_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    # Ham class metrics (Pos label = 0)
    ham_precision = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    ham_recall = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    ham_f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    
    # Macro and weighted averages
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    # Confusion Matrix breakdown:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    
    # ROC-AUC if probabilities provided
    roc_auc = None
    if y_proba is not None:
        try:
            if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                proba_spam = y_proba[:, 1]
            else:
                proba_spam = y_proba.ravel()
            roc_auc = float(roc_auc_score(y_true, proba_spam))
        except Exception:
            roc_auc = None

    # Classification report as structured dict
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0
    )
    
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=["ham", "spam"],
        zero_division=0
    )

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "spam_precision": round(spam_precision, 4),
        "spam_recall": round(spam_recall, 4),
        "spam_f1": round(spam_f1, 4),
        "ham_precision": round(ham_precision, 4),
        "ham_recall": round(ham_recall, 4),
        "ham_f1": round(ham_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "raw": [[tn, fp], [fn, tp]]
        },
        "report_dict": report_dict,
        "report_text": report_text,
        "pedagogical_insights": {
            "false_positive_explanation": (
                f"A False Positive (FP={fp}) occurs when a legitimate message (HAM) is incorrectly classified as SPAM. "
                "In real-world SMS filtering, high FP is dangerous because users may miss critical two-factor authentication (2FA) OTP codes, "
                "banking alerts, flight notifications, or emergency personal messages."
            ),
            "false_negative_explanation": (
                f"A False Negative (FN={fn}) occurs when an unwanted or malicious message (SPAM) is mistakenly classified as legitimate (HAM). "
                "While less catastrophic than losing a genuine message, a high FN rate exposes the user to fraudulent phishing links, financial scams, and nuisance advertising."
            )
        }
    }


def format_comparison_table(metrics_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Formats multiple model metrics dictionaries into a clean summary table for UI and logs.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m["model_name"],
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Spam Precision (%)": round(m["spam_precision"] * 100, 2),
            "Spam Recall (%)": round(m["spam_recall"] * 100, 2),
            "Spam F1-Score (%)": round(m["spam_f1"] * 100, 2),
            "Ham F1-Score (%)": round(m["ham_f1"] * 100, 2),
            "False Positives (FP)": m["confusion_matrix"]["fp"],
            "False Negatives (FN)": m["confusion_matrix"]["fn"]
        })
    return rows
