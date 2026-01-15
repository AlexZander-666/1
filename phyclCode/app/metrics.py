"""Lightweight binary classification metrics and operating points."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def basic_metrics(y_true: Iterable[int], probs: Iterable[float], threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(list(y_true))
    p = np.asarray(list(probs))
    preds = (p >= threshold).astype(int)
    acc = float((preds == y).mean()) if len(y) else 0.0
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    precision_pos = tp / (tp + fp + 1e-12)
    recall_pos = tp / (tp + fn + 1e-12)
    precision_neg = tn / (tn + fn + 1e-12)
    recall_neg = tn / (tn + fp + 1e-12)
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + 1e-12)
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + 1e-12)
    macro_f1 = float((f1_pos + f1_neg) / 2.0)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def operating_points(y_true: Iterable[int], probs: Iterable[float]) -> Dict[str, float]:
    """Compute FPR@TPR=95% and TPR@FPR=1%."""
    y = np.asarray(list(y_true))
    p = np.asarray(list(probs))
    if len(y) == 0:
        return {"fpr_at_tpr95": float("nan"), "tpr_at_fpr1": float("nan")}

    thresholds = np.unique(p)
    thresholds = np.concatenate([thresholds, [0.0, 1.0]])
    thresholds = np.clip(thresholds, 0.0, 1.0)
    thresholds = np.unique(thresholds)
    best_fpr_for_tpr95 = 1.0
    best_tpr_for_fpr1 = 0.0

    for t in thresholds:
        preds = (p >= t).astype(int)
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        tn = ((preds == 0) & (y == 0)).sum()
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        if tpr >= 0.95:
            best_fpr_for_tpr95 = min(best_fpr_for_tpr95, fpr)
        if fpr <= 0.01:
            best_tpr_for_fpr1 = max(best_tpr_for_fpr1, tpr)

    return {
        "fpr_at_tpr95": float(best_fpr_for_tpr95),
        "tpr_at_fpr1": float(best_tpr_for_fpr1),
    }
