"""One-hidden-layer neural network implemented directly with NumPy."""

from __future__ import annotations

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


class OneHiddenLayerNetwork:
    """Fully connected ReLU-softmax network trained by full-batch descent."""

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 10,
        n_classes: int = 10,
        seed: int = 42,
    ) -> None:
        if min(n_features, n_hidden, n_classes) <= 0:
            raise ValueError("all layer sizes must be positive")
        rng = np.random.default_rng(seed)
        self.weights_hidden = rng.normal(0, np.sqrt(2 / n_features), (n_features, n_hidden))
        self.bias_hidden = np.zeros(n_hidden)
        self.weights_output = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_classes))
        self.bias_output = np.zeros(n_classes)

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return hidden activations and output probabilities."""
        samples = np.asarray(features, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != self.weights_hidden.shape[0]:
            raise ValueError("features have an incompatible shape")
        hidden = relu(samples @ self.weights_hidden + self.bias_hidden)
        probabilities = softmax(hidden @ self.weights_output + self.bias_output)
        return hidden, probabilities

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return most probable class for each sample."""
        return self.forward(features)[1].argmax(axis=1)

    def train_step(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        learning_rate: float,
    ) -> tuple[float, float]:
        """Apply one full-batch gradient step and return loss and accuracy."""
        samples = np.asarray(features, dtype=float)
        targets = np.asarray(labels, dtype=int)
        if len(samples) == 0 or len(samples) != len(targets):
            raise ValueError("features and labels must contain equal non-zero samples")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        hidden_linear = samples @ self.weights_hidden + self.bias_hidden
        hidden = relu(hidden_linear)
        probabilities = softmax(hidden @ self.weights_output + self.bias_output)
        encoded = one_hot(targets, self.weights_output.shape[1])
        sample_count = len(samples)

        output_gradient = (probabilities - encoded) / sample_count
        output_weight_gradient = hidden.T @ output_gradient
        output_bias_gradient = output_gradient.sum(axis=0)
        hidden_gradient = (output_gradient @ self.weights_output.T) * (hidden_linear > 0)
        hidden_weight_gradient = samples.T @ hidden_gradient
        hidden_bias_gradient = hidden_gradient.sum(axis=0)

        self.weights_output -= learning_rate * output_weight_gradient
        self.bias_output -= learning_rate * output_bias_gradient
        self.weights_hidden -= learning_rate * hidden_weight_gradient
        self.bias_hidden -= learning_rate * hidden_bias_gradient

        loss = -np.mean(np.log(probabilities[np.arange(sample_count), targets] + 1e-12))
        accuracy = float(np.mean(probabilities.argmax(axis=1) == targets))
        return float(loss), accuracy

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 1_000,
        learning_rate: float = 0.04,
    ) -> dict[str, list[float]]:
        """Train for fixed epochs and return loss and accuracy histories."""
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        history: dict[str, list[float]] = {"loss": [], "accuracy": []}
        for _ in range(epochs):
            loss, accuracy = self.train_step(features, labels, learning_rate)
            history["loss"].append(loss)
            history["accuracy"].append(accuracy)
        return history

