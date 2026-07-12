"""Rosenblatt binary perceptron implemented directly with NumPy."""

from __future__ import annotations

import numpy as np


def sign(values: np.ndarray) -> np.ndarray:
    """Map non-negative scores to +1 and negative scores to -1."""
    return np.where(np.asarray(values) >= 0, 1, -1)


class RosenblattPerceptron:
    """Binary linear classifier trained with Rosenblatt's update rule."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 1e-4,
        seed: int = 42,
        initialization: str = "zeros",
    ) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if initialization not in {"zeros", "random"}:
            raise ValueError("initialization must be 'zeros' or 'random'")
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(seed)
        self.weights = (
            np.zeros(n_features)
            if initialization == "zeros"
            else self._rng.uniform(-1.0, 1.0, n_features)
        )
        self.bias = 0.0
        self.initial_weights = self.weights.copy()
        self.initial_bias = self.bias
        self.best_epoch_: int | None = None
        self.stopped_early_ = False

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Return linear scores for sample-major feature rows."""
        samples = np.asarray(features, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(self.weights):
            raise ValueError("features have an incompatible shape")
        return samples @ self.weights + self.bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict -1 or +1 for each sample."""
        return sign(self.decision_function(features))

    def pixel_contributions(self, features: np.ndarray) -> np.ndarray:
        """Return each pixel's exact additive contribution before bias."""
        samples = np.asarray(features, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(self.weights):
            raise ValueError("features have an incompatible shape")
        return samples * self.weights

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

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        shuffle: bool = True,
        *,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        early_stopping: bool = False,
        patience: int = 20,
        min_delta: float = 1e-4,
    ) -> dict[str, list[float]]:
        """Train and return accuracy, validation accuracy, and mistake histories."""
        samples = np.asarray(features, dtype=float)
        targets = np.asarray(labels, dtype=int)
        if len(samples) == 0 or len(samples) != len(targets):
            raise ValueError("features and labels must have equal sample counts")
        if np.any((targets != -1) & (targets != 1)):
            raise ValueError("labels must contain only -1 and +1")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if patience <= 0 or min_delta < 0:
            raise ValueError("patience must be positive and min_delta non-negative")
        if early_stopping and validation_data is None:
            raise ValueError("early_stopping requires validation_data")

        validation: tuple[np.ndarray, np.ndarray] | None = None
        if validation_data is not None:
            validation_features = np.asarray(validation_data[0], dtype=float)
            validation_labels = np.asarray(validation_data[1], dtype=int)
            if (
                len(validation_features) == 0
                or len(validation_features) != len(validation_labels)
                or validation_features.ndim != 2
                or validation_features.shape[1] != len(self.weights)
            ):
                raise ValueError("validation features and labels have incompatible shapes")
            if np.any((validation_labels != -1) & (validation_labels != 1)):
                raise ValueError("validation labels must contain only -1 and +1")
            validation = validation_features, validation_labels

        history: dict[str, list[float]] = {
            "train_accuracy": [],
            "validation_accuracy": [],
            "mistakes": [],
        }
        self.best_epoch_ = None
        self.stopped_early_ = False
        best_accuracy = -np.inf
        best_parameters: tuple[np.ndarray, float] | None = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            indices = self._rng.permutation(len(samples)) if shuffle else np.arange(len(samples))
            mistakes = 0
            for index in indices:
                mistakes += self.update_one(samples[index], int(targets[index]))
            history["train_accuracy"].append(
                float(np.mean(self.predict(samples) == targets))
            )
            history["mistakes"].append(float(mistakes))

            if validation is None:
                history["validation_accuracy"].append(float("nan"))
            else:
                validation_accuracy = float(
                    np.mean(self.predict(validation[0]) == validation[1])
                )
                history["validation_accuracy"].append(validation_accuracy)
                if validation_accuracy > best_accuracy + min_delta:
                    best_accuracy = validation_accuracy
                    self.best_epoch_ = epoch
                    best_parameters = self.weights.copy(), self.bias
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if mistakes == 0:
                break
            if early_stopping and epochs_without_improvement >= patience:
                self.stopped_early_ = True
                break

        if early_stopping and best_parameters is not None:
            self.weights, self.bias = best_parameters
        return history
