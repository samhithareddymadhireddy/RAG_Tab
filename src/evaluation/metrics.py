"""Metrics calculation module for imbalanced tabular classification.

All metrics are computed with respect to the minority class as the positive class.
"""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(
    y_true,
    y_pred,
    minority_class: str = "positive",
    majority_class: str = "negative"
) -> Dict[str, Any]:
    """Evaluates classifier predictions using metrics tailored to class imbalance.

    Calculates:
    - Precision: TP / (TP + FP)
    - Recall (Sensitivity): TP / (TP + FN)
    - F1-Score: Harmonic mean of Precision and Recall
    - Specificity (TNR): TN / (TN + FP)
    - G-Mean: sqrt(Recall * Specificity)
    - Confusion Matrix Counts: TP, FP, TN, FN

    Parameters
    ----------
    y_true : array-like
        Ground truth class labels.
    y_pred : array-like
        Predicted class labels.
    minority_class : str, default='positive'
        Label of the minority (positive) class.
    majority_class : str, default='negative'
        Label of the majority (negative) class.

    Returns
    -------
    dict
        Dictionary containing metric scores and confusion counts.
    """
    precision = precision_score(
        y_true,
        y_pred,
        pos_label=minority_class,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=minority_class,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=minority_class,
        zero_division=0
    )

    # Order rows/cols: 0: majority, 1: minority
    # Row 0: True Majority, Row 1: True Minority
    # Col 0: Pred Majority, Col 1: Pred Minority
    # [ [TN, FP],
    #   [FN, TP] ]
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[majority_class, minority_class]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = float(np.sqrt(recall * specificity))

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Specificity": float(specificity),
        "G-Mean": gmean,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }
