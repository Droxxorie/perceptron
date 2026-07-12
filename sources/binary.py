"""Rosenblatt binary perceptron implemented directly with NumPy."""

from __future__ import annotations

import numpy as np


def sign(values: np.ndarray) -> np.ndarray:
    """Map non-negative scores to +1 and negative scores to -1."""
    return np.where(np.asarray(values) >= 0, 1, -1)


class RosenblattPerceptron:
    """Binary linear classifier trained with Rosenblatt's update rule."""

    def __init__(self, n_features: int, learning_rate: float = 1e-4, seed: int = 42) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(seed)
        self.weights = self._rng.uniform(-1.0, 1.0, n_features)
        self.bias = 0.0

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Return linear scores for sample-major feature rows."""
        samples = np.asarray(features, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(self.weights):
            raise ValueError("features have an incompatible shape")
        return samples @ self.weights + self.bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict -1 or +1 for each sample."""
        return sign(self.decision_function(features))

    def update_one(self, features: np.ndarray, target: int) -> bool:
        """Apply one perceptron update; return whether sample was mistaken."""
        sample = np.asarray(features, dtype=float)
        if sample.shape != self.weights.shape:
            raise ValueError("sample has an incompatible shape")
        if target not in (-1, 1):
            raise ValueError("target must be -1 or +1")
        prediction = int(sign(np.array([sample @ self.weights + self.bias]))[0])
        if prediction == target:
            return False
        step = self.learning_rate * target
        self.weights += step * sample
        self.bias += step
        return True

    def fit(self, features: np.ndarray, labels: np.ndarray, epochs: int = 100, shuffle: bool = True) -> list[float]:
        """Train for fixed epochs and return training accuracy after each epoch."""
        samples = np.asarray(features, dtype=float)
        targets = np.asarray(labels, dtype=int)
        if len(samples) != len(targets):
            raise ValueError("features and labels must have equal sample counts")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        history: list[float] = []
        for _ in range(epochs):
            indices = self._rng.permutation(len(samples)) if shuffle else np.arange(len(samples))
            mistakes = 0
            for index in indices:
                mistakes += self.update_one(samples[index], int(targets[index]))
            history.append(float(np.mean(self.predict(samples) == targets)))
            if mistakes == 0:
                break
        return history
