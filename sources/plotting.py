"""Plot helpers for the narrative notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_history(values: list[float], ylabel: str = "Accuracy") -> plt.Figure:
    """Plot one metric against training epoch and return its figure."""
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(np.arange(1, len(values) + 1), values)
    axis.set(xlabel="Epoch", ylabel=ylabel)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def plot_confusion(matrix: np.ndarray, labels: list[str] | None = None) -> plt.Figure:
    """Render a confusion matrix with true labels on rows."""
    counts = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(counts, cmap="Blues")
    ticks = np.arange(len(counts))
    axis.set(xticks=ticks, yticks=ticks, xlabel="Predicted", ylabel="True")
    if labels is not None:
        axis.set_xticklabels(labels)
        axis.set_yticklabels(labels)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    return figure

