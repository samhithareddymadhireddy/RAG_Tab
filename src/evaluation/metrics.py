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
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    unique_classes = list(np.unique(y_true_arr))

    # Binary Classification Regime
    if len(unique_classes) <= 2 and (minority_class in unique_classes or "positive" in unique_classes):
        target_pos = minority_class if minority_class in unique_classes else "positive"
        target_neg = majority_class if majority_class in unique_classes else [c for c in unique_classes if c != target_pos][0]

        precision = precision_score(
            y_true_arr,
            y_pred_arr,
            pos_label=target_pos,
            zero_division=0
        )
        recall = recall_score(
            y_true_arr,
            y_pred_arr,
            pos_label=target_pos,
            zero_division=0
        )
        f1 = f1_score(
            y_true_arr,
            y_pred_arr,
            pos_label=target_pos,
            zero_division=0
        )

        cm = confusion_matrix(
            y_true_arr,
            y_pred_arr,
            labels=[target_neg, target_pos]
        )
        tn, fp, fn, tp = cm.ravel()
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

    # Multiclass Classification Regime (>= 3 classes)
    precision_macro = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    recall_macro = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    f1_macro = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=unique_classes)
    total_samples = len(y_true_arr)
    recalls = []
    specificities = []

    for i in range(len(unique_classes)):
        tp_c = cm[i, i]
        fn_c = cm[i, :].sum() - tp_c
        fp_c = cm[:, i].sum() - tp_c
        tn_c = total_samples - (tp_c + fn_c + fp_c)

        rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        spec_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0
        recalls.append(rec_c)
        specificities.append(spec_c)

    # Multiclass G-Mean: geometric mean of per-class recalls
    gmean = float(np.prod([max(r, 1e-4) for r in recalls]) ** (1.0 / len(recalls)))
    macro_specificity = float(np.mean(specificities))

    return {
        "Precision": float(precision_macro),
        "Recall": float(recall_macro),
        "F1": float(f1_macro),
        "F1_Weighted": float(f1_weighted),
        "Specificity": macro_specificity,
        "G-Mean": gmean,
        "Classes": [str(c) for c in unique_classes],
        "Confusion_Matrix": cm.tolist()
    }

