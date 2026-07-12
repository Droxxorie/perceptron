"""Small evaluation metrics used by both classifiers."""

from __future__ import annotations

import numpy as np


def accuracy(truth: np.ndarray, predictions: np.ndarray) -> float:
    """Return fraction of labels classified correctly."""
    expected = np.asarray(truth)
    observed = np.asarray(predictions)
    if expected.shape != observed.shape or expected.size == 0:
        raise ValueError("truth and predictions must have equal non-empty shapes")
    return float(np.mean(expected == observed))


def confusion_matrix(
    truth: np.ndarray,
    predictions: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Count predictions with true classes on rows and predictions on columns."""
    expected = np.asarray(truth, dtype=int)
    observed = np.asarray(predictions, dtype=int)
    if expected.shape != observed.shape:
        raise ValueError("truth and predictions must have equal shapes")
    if np.any((expected < 0) | (expected >= n_classes)) or np.any(
        (observed < 0) | (observed >= n_classes)
    ):
        raise ValueError("labels must be valid class indices")
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (expected, observed), 1)
    return matrix
