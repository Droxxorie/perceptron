"""From-scratch ReLU multilayer perceptrons implemented with NumPy."""

from __future__ import annotations
from collections.abc import Sequence

import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    """Apply rectified linear activation element-wise."""
    return np.maximum(np.asarray(values), 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert sample-major logits into stable class probabilities."""
    scores = np.asarray(logits, dtype=float)
    if scores.ndim != 2:
        raise ValueError("logits must have shape (samples, classes)")
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def one_hot(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Encode integer labels as sample-major indicator rows."""
    targets = np.asarray(labels, dtype=int)
    if targets.ndim != 1 or np.any((targets < 0) | (targets >= n_classes)):
        raise ValueError("labels must be one-dimensional valid class indices")
    encoded = np.zeros((len(targets), n_classes), dtype=float)
    encoded[np.arange(len(targets)), targets] = 1.0
    return encoded


class MultilayerPerceptron:
    """ReLU-softmax network with one or two configurable hidden layers."""

    def __init__(
        self,
        n_features: int,
        hidden_layers: Sequence[int] = (10,),
        n_classes: int = 10,
        seed: int = 42,
    ) -> None:
        widths = tuple(int(width) for width in hidden_layers)
        if len(widths) not in (1, 2):
            raise ValueError("hidden_layers must contain one or two layer widths")
        if min((n_features, n_classes, *widths)) <= 0:
            raise ValueError("all layer sizes must be positive")

        self.hidden_layers = widths
        self._rng = np.random.default_rng(seed)
        layer_sizes = (n_features, *widths, n_classes)
        self.weights = [
            self._rng.normal(
                0.0,
                np.sqrt(2.0 / input_width),
                (input_width, output_width),
            )
            for input_width, output_width in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.zeros(width) for width in layer_sizes[1:]]
        self.best_epoch_: int | None = None
        self.stopped_early_ = False
        self._reset_velocities()

    @property
    def n_classes(self) -> int:
        """Number of output classes."""
        return self.weights[-1].shape[1]

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        samples = np.asarray(features, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != self.weights[0].shape[0]:
            raise ValueError("features have an incompatible shape")
        return samples

    def _validate_training_data(
        self, features: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        samples = self._validate_features(features)
        targets = np.asarray(labels, dtype=int)
        if len(samples) == 0 or targets.ndim != 1 or len(samples) != len(targets):
            raise ValueError("features and labels must contain equal non-zero samples")
        one_hot(targets, self.n_classes)
        return samples, targets

    def _forward_all(
        self, features: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        samples = self._validate_features(features)
        activations = [samples]
        linear_outputs: list[np.ndarray] = []
        current = samples
        for weights, bias in zip(self.weights[:-1], self.biases[:-1]):
            linear = current @ weights + bias
            linear_outputs.append(linear)
            current = relu(linear)
            activations.append(current)
        probabilities = softmax(current @ self.weights[-1] + self.biases[-1])
        return linear_outputs, activations, probabilities

    def forward(self, features: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Return each hidden activation followed by class probabilities."""
        _, activations, probabilities = self._forward_all(features)
        return activations[1:], probabilities

    def logits(self, features: np.ndarray) -> np.ndarray:
        """Return pre-softmax class scores."""
        samples = self._validate_features(features)
        current = samples
        for weights, bias in zip(self.weights[:-1], self.biases[:-1]):
            current = relu(current @ weights + bias)
        return current @ self.weights[-1] + self.biases[-1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return most probable class for each sample."""
        return self.forward(features)[1].argmax(axis=1)

    def input_gradients(
        self,
        features: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Differentiate selected class logits with respect to input pixels."""
        samples = self._validate_features(features)
        selected = self.predict(samples) if classes is None else np.asarray(classes, dtype=int)
        if selected.shape != (len(samples),) or np.any(
            (selected < 0) | (selected >= self.n_classes)
        ):
            raise ValueError("classes must contain one valid class per sample")

        linear_outputs, _, _ = self._forward_all(samples)
        gradient = self.weights[-1].T[selected]
        for layer in range(len(self.hidden_layers) - 1, -1, -1):
            gradient = gradient * (linear_outputs[layer] > 0)
            gradient = gradient @ self.weights[layer].T
        return gradient

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        """Return cross-entropy and accuracy without changing model parameters."""
        samples, targets = self._validate_training_data(features, labels)
        probabilities = self.forward(samples)[1]
        loss = -np.mean(np.log(probabilities[np.arange(len(targets)), targets] + 1e-12))
        accuracy = np.mean(probabilities.argmax(axis=1) == targets)
        return float(loss), float(accuracy)

    def _gradients(
        self, features: np.ndarray, labels: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        samples, targets = self._validate_training_data(features, labels)
        linear_outputs, activations, probabilities = self._forward_all(samples)
        gradient = (probabilities - one_hot(targets, self.n_classes)) / len(samples)
        weight_gradients: list[np.ndarray] = [np.empty(0)] * len(self.weights)
        bias_gradients: list[np.ndarray] = [np.empty(0)] * len(self.biases)

        weight_gradients[-1] = activations[-1].T @ gradient
        bias_gradients[-1] = gradient.sum(axis=0)
        for layer in range(len(self.hidden_layers) - 1, -1, -1):
            gradient = (gradient @ self.weights[layer + 1].T) * (
                linear_outputs[layer] > 0
            )
            weight_gradients[layer] = activations[layer].T @ gradient
            bias_gradients[layer] = gradient.sum(axis=0)
        return weight_gradients, bias_gradients

    def _reset_velocities(self) -> None:
        """Reset momentum state to zero for a new training run."""
        self._weight_velocities = [np.zeros_like(weight) for weight in self.weights]
        self._bias_velocities = [np.zeros_like(bias) for bias in self.biases]

    def _apply_gradients(
        self,
        weight_gradients: list[np.ndarray],
        bias_gradients: list[np.ndarray],
        learning_rate: float,
        optimizer: str,
        momentum: float,
    ) -> None:
        if optimizer not in {"sgd", "momentum"}:
            raise ValueError("optimizer must be 'sgd' or 'momentum'")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be between 0 and 1")

        for index, (weight_gradient, bias_gradient) in enumerate(
            zip(weight_gradients, bias_gradients)
        ):
            if optimizer == "momentum":
                self._weight_velocities[index] = (
                    momentum * self._weight_velocities[index]
                    + learning_rate * weight_gradient
                )
                self._bias_velocities[index] = (
                    momentum * self._bias_velocities[index]
                    + learning_rate * bias_gradient
                )
                weight_step = self._weight_velocities[index]
                bias_step = self._bias_velocities[index]
            else:
                weight_step = learning_rate * weight_gradient
                bias_step = learning_rate * bias_gradient
            self.weights[index] -= weight_step
            self.biases[index] -= bias_step

    def train_step(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        learning_rate: float,
        optimizer: str = "sgd",
        momentum: float = 0.9,
    ) -> tuple[float, float]:
        """Apply one gradient update and return post-update loss and accuracy."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        gradients = self._gradients(features, labels)
        self._apply_gradients(*gradients, learning_rate, optimizer, momentum)
        return self.evaluate(features, labels)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 1_000,
        learning_rate: float = 0.04,
        *,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        optimizer: str = "sgd",
        momentum: float = 0.9,
        batch_size: int | None = None,
        learning_rate_decay: float = 0.0,
        early_stopping: bool = False,
        patience: int = 20,
        min_delta: float = 1e-4,
        shuffle: bool = True,
    ) -> dict[str, list[float]]:
        """Train model and return per-epoch train/validation diagnostics."""
        samples, targets = self._validate_training_data(features, labels)
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        if learning_rate_decay < 0:
            raise ValueError("learning_rate_decay must be non-negative")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive or None")
        if patience <= 0 or min_delta < 0:
            raise ValueError("patience must be positive and min_delta non-negative")
        if early_stopping and validation_data is None:
            raise ValueError("early_stopping requires validation_data")
        if optimizer not in {"sgd", "momentum"}:
            raise ValueError("optimizer must be 'sgd' or 'momentum'")

        validation = (
            self._validate_training_data(*validation_data)
            if validation_data is not None
            else None
        )
        size = len(samples) if batch_size is None else min(batch_size, len(samples))
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "learning_rate": [],
        }
        self.best_epoch_ = None
        self.stopped_early_ = False
        self._reset_velocities()
        best_loss = np.inf
        best_parameters: tuple[list[np.ndarray], list[np.ndarray]] | None = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            effective_rate = learning_rate / (1.0 + learning_rate_decay * epoch)
            indices = self._rng.permutation(len(samples)) if shuffle else np.arange(len(samples))
            for start in range(0, len(samples), size):
                batch = indices[start : start + size]
                gradients = self._gradients(samples[batch], targets[batch])
                self._apply_gradients(
                    *gradients, effective_rate, optimizer, momentum
                )

            train_loss, train_accuracy = self.evaluate(samples, targets)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["learning_rate"].append(effective_rate)

            if validation is None:
                history["validation_loss"].append(float("nan"))
                history["validation_accuracy"].append(float("nan"))
                continue

            validation_loss, validation_accuracy = self.evaluate(*validation)
            history["validation_loss"].append(validation_loss)
            history["validation_accuracy"].append(validation_accuracy)
            if validation_loss < best_loss - min_delta:
                best_loss = validation_loss
                self.best_epoch_ = epoch
                best_parameters = (
                    [weight.copy() for weight in self.weights],
                    [bias.copy() for bias in self.biases],
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if early_stopping and epochs_without_improvement >= patience:
                self.stopped_early_ = True
                break

        if early_stopping and best_parameters is not None:
            self.weights, self.biases = best_parameters
        return history
