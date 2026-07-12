"""Decision-evidence calculations for the classifiers."""

from __future__ import annotations

import numpy as np

from .binary import RosenblattPerceptron
from .mlp import MultilayerPerceptron


def input_attributions(
    model: MultilayerPerceptron,
    features: np.ndarray,
    classes: np.ndarray | None = None,
) -> np.ndarray:
    """Return gradient-times-input evidence for selected class logits."""
    samples = np.asarray(features, dtype=float)
    return model.input_gradients(samples, classes) * samples


def binary_decision_evidence(
    model: RosenblattPerceptron,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Aggregate exact mean pixel contributions for the -1 and +1 classes."""
    samples = np.asarray(features, dtype=float)
    targets = np.asarray(labels, dtype=int)
    if len(samples) != len(targets):
        raise ValueError("features and labels must contain equal samples")
    negative = samples[targets == -1]
    positive = samples[targets == 1]
    if len(negative) == 0 or len(positive) == 0:
        raise ValueError("both -1 and +1 classes are required")
    mean_negative = negative.mean(axis=0)
    mean_positive = positive.mean(axis=0)
    return {
        "mean_negative": mean_negative,
        "mean_positive": mean_positive,
        "weights": model.weights.copy(),
        "negative_contribution": mean_negative * model.weights,
        "positive_contribution": mean_positive * model.weights,
    }


def class_average_evidence(
    model: MultilayerPerceptron,
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 10,
    correctly_classified_only: bool = True,
) -> np.ndarray:
    """Average gradient-times-input evidence for each true class."""
    samples = np.asarray(features, dtype=float)
    targets = np.asarray(labels, dtype=int)
    if len(samples) != len(targets):
        raise ValueError("features and labels must contain equal samples")
    predictions = model.predict(samples)
    rows = []
    for class_index in range(n_classes):
        mask = targets == class_index
        if correctly_classified_only:
            mask &= predictions == targets
        selected = samples[mask]
        if len(selected) == 0:
            raise ValueError(f"class {class_index} has no eligible samples")
        classes = np.full(len(selected), class_index, dtype=int)
        rows.append(input_attributions(model, selected, classes).mean(axis=0))
    return np.asarray(rows)
