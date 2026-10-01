"""Metrics with a NumPy fallback independent of scikit-learn."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        if 0 <= int(truth) < num_classes and 0 <= int(prediction) < num_classes:
            matrix[int(truth), int(prediction)] += 1
    return matrix


def classification_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], num_classes: int, class_names: list[str]
) -> tuple[dict[str, float], list[dict[str, object]], np.ndarray]:
    truth = np.asarray(list(y_true), dtype=np.int64)
    prediction = np.asarray(list(y_pred), dtype=np.int64)
    matrix = confusion_matrix(truth, prediction, num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0)
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    overall = {
        "accuracy": float((truth == prediction).mean()) if len(truth) else 0.0,
        "macro_precision": float(precision.mean()) if num_classes else 0.0,
        "macro_recall": float(recall.mean()) if num_classes else 0.0,
        "macro_f1": float(f1.mean()) if num_classes else 0.0,
    }
    per_class = [
        {
            "activity_id": index,
            "activity": class_names[index] if index < len(class_names) else str(index),
            "support": int(actual[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index in range(num_classes)
    ]
    return overall, per_class, matrix

