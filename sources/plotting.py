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


def plot_training_history(history: dict[str, list[float]]) -> plt.Figure:
    """Plot paired train/validation loss and accuracy curves."""
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    loss_axis.plot(epochs, history["train_loss"], label="Train")
    loss_axis.plot(epochs, history["validation_loss"], label="Validation")
    loss_axis.set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    accuracy_axis.plot(epochs, history["train_accuracy"], label="Train")
    accuracy_axis.plot(
        epochs, history["validation_accuracy"], label="Validation"
    )
    accuracy_axis.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    for axis in (loss_axis, accuracy_axis):
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    return figure


def plot_binary_history(history: dict[str, list[float]]) -> plt.Figure:
    """Plot binary train/validation accuracy and mistakes per epoch."""
    figure, (accuracy_axis, mistake_axis) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = np.arange(1, len(history["train_accuracy"]) + 1)
    accuracy_axis.plot(epochs, history["train_accuracy"], label="Train")
    accuracy_axis.plot(epochs, history["validation_accuracy"], label="Validation")
    accuracy_axis.set(xlabel="Epoch", ylabel="Accuracy", title="Binary accuracy")
    accuracy_axis.legend()
    mistake_axis.plot(epochs, history["mistakes"], color="tab:red")
    mistake_axis.set(xlabel="Epoch", ylabel="Mistakes", title="Training mistakes")
    for axis in (accuracy_axis, mistake_axis):
        axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def _square_side(size: int) -> int:
    side = int(np.sqrt(size))
    if side * side != size:
        raise ValueError("pixel weights must contain a square number of values")
    return side


def plot_image_weights(weights: np.ndarray, title: str = "Pixel weights") -> plt.Figure:
    """Render one flattened pixel-weight vector with a centered color scale."""
    values = np.asarray(weights, dtype=float).reshape(-1)
    side = _square_side(len(values))
    limit = max(float(np.max(np.abs(values))), 1e-12)
    figure, axis = plt.subplots(figsize=(5, 5))
    image = axis.imshow(values.reshape(side, side), vmin=-limit, vmax=limit)
    axis.set(title=title)
    axis.axis("off")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    return figure


def plot_first_layer_weights(
    weights: np.ndarray, max_neurons: int = 10
) -> plt.Figure:
    """Render input-connected hidden weights as at most ten pixel maps."""
    matrix = np.asarray(weights, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("weights must have shape (pixels, hidden neurons)")
    side = _square_side(matrix.shape[0])
    count = min(matrix.shape[1], max_neurons, 10)
    if count == 0:
        raise ValueError("weights must contain at least one hidden neuron")
    columns = min(5, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.3 * columns, 2.3 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    limit = max(float(np.max(np.abs(matrix[:, :count]))), 1e-12)
    for index, axis in enumerate(axes_array):
        if index < count:
            axis.imshow(matrix[:, index].reshape(side, side), vmin=-limit, vmax=limit)
            axis.set_title(f"Hidden neuron {index}")
        axis.axis("off")
    figure.tight_layout()
    return figure


def plot_layer_weights(weights: np.ndarray, title: str) -> plt.Figure:
    """Render non-pixel layer connections as a labeled heatmap."""
    matrix = np.asarray(weights, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("weights must be a two-dimensional matrix")
    limit = max(float(np.max(np.abs(matrix))), 1e-12)
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(matrix, vmin=-limit, vmax=limit, aspect="auto")
    axis.set(title=title, xlabel="Destination neuron", ylabel="Source neuron")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    return figure


def _symmetric_limit(values: np.ndarray) -> float:
    return max(float(np.max(np.abs(values))), 1e-12)


def plot_binary_decision_evidence(
    evidence: dict[str, np.ndarray],
) -> plt.Figure:
    """Show binary class forms, learned weights, and exact mean contributions."""
    required = {
        "mean_negative",
        "mean_positive",
        "weights",
        "negative_contribution",
        "positive_contribution",
    }
    if set(evidence) != required:
        raise ValueError("binary evidence contains unexpected or missing fields")
    side = _square_side(len(evidence["weights"]))
    figure, axes = plt.subplots(1, 5, figsize=(17, 3.6), layout="constrained")
    mean_images = [evidence["mean_negative"], evidence["mean_positive"]]
    mean_titles = ["Mean digit 0", "Mean digit 1"]
    for axis, values, title in zip(axes[:2], mean_images, mean_titles):
        axis.imshow(np.asarray(values).reshape(side, side), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")

    weight_limit = _symmetric_limit(evidence["weights"])
    weight_image = axes[2].imshow(
        np.asarray(evidence["weights"]).reshape(side, side),
        cmap="coolwarm",
        vmin=-weight_limit,
        vmax=weight_limit,
    )
    axes[2].set_title("Learned signed weights")
    axes[2].axis("off")

    contribution_values = np.asarray(
        [evidence["negative_contribution"], evidence["positive_contribution"]]
    )
    contribution_limit = _symmetric_limit(contribution_values)
    contribution_images = []
    for axis, values, title in zip(
        axes[3:],
        contribution_values,
        ["Digit 0 mean contribution", "Digit 1 mean contribution"],
    ):
        image = axis.imshow(
            values.reshape(side, side),
            cmap="coolwarm",
            vmin=-contribution_limit,
            vmax=contribution_limit,
        )
        contribution_images.append(image)
        axis.set_title(title)
        axis.axis("off")
    weight_colorbar = figure.colorbar(weight_image, ax=axes[2], shrink=0.75)
    weight_colorbar.set_label("Opposes 1  ←  weight  →  supports 1")
    contribution_colorbar = figure.colorbar(
        contribution_images[-1], ax=axes[3:], shrink=0.75
    )
    contribution_colorbar.set_label("Opposes 1  ←  pixel contribution  →  supports 1")
    figure.suptitle("Rosenblatt decision evidence")
    return figure


def plot_class_evidence(
    evidence: np.ndarray,
    labels: list[str] | None = None,
) -> plt.Figure:
    """Render one shared-scale signed decision-evidence map per class."""
    maps = np.asarray(evidence, dtype=float)
    if maps.ndim != 2:
        raise ValueError("evidence must have shape (classes, flattened pixels)")
    side = _square_side(maps.shape[1])
    count = len(maps)
    columns = min(5, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.5 * columns, 2.6 * rows), layout="constrained"
    )
    axes_array = np.atleast_1d(axes).ravel()
    limit = _symmetric_limit(maps)
    image = None
    for index, axis in enumerate(axes_array):
        if index < count:
            image = axis.imshow(
                maps[index].reshape(side, side),
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            label = labels[index] if labels is not None else str(index)
            axis.set_title(f"Evidence for {label}")
        axis.axis("off")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes_array.tolist(), shrink=0.75)
        colorbar.set_label("Opposes class  ←  evidence  →  supports class")
    figure.suptitle("Class-conditioned input evidence")
    return figure


def plot_attribution_examples(
    images: np.ndarray,
    attributions: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> plt.Figure:
    """Pair digit images with signed gradient-times-input decision evidence."""
    samples = np.asarray(images, dtype=float)
    evidence = np.asarray(attributions, dtype=float)
    truth = np.asarray(labels, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    if samples.shape != evidence.shape or len(samples) != len(truth) or len(samples) != len(predicted):
        raise ValueError("images, attributions, labels, and predictions must align")
    confidence = None
    if probabilities is not None:
        scores = np.asarray(probabilities, dtype=float)
        if scores.ndim != 2 or len(scores) != len(samples):
            raise ValueError("probabilities must have one class row per image")
        confidence = scores[np.arange(len(samples)), predicted]
    side = _square_side(samples.shape[1])
    count = min(len(samples), 10)
    if count == 0:
        raise ValueError("at least one example is required")
    figure, axes = plt.subplots(
        2,
        count,
        figsize=(2.2 * count, 4.8),
        squeeze=False,
        layout="constrained",
    )
    limit = _symmetric_limit(evidence[:count])
    evidence_image = None
    for index in range(count):
        axes[0, index].imshow(samples[index].reshape(side, side), cmap="gray", vmin=0, vmax=1)
        title = f"True {truth[index]} · Pred {predicted[index]}"
        if confidence is not None:
            title += f"\n{confidence[index]:.1%} confidence"
        axes[0, index].set_title(title)
        axes[0, index].axis("off")
        evidence_image = axes[1, index].imshow(
            evidence[index].reshape(side, side),
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axes[1, index].set_title("Decision evidence")
        axes[1, index].axis("off")
    if evidence_image is not None:
        colorbar = figure.colorbar(evidence_image, ax=axes[1, :].tolist(), shrink=0.75)
        colorbar.set_label("Opposes prediction  ←  evidence  →  supports prediction")
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
