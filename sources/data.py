"""MNIST loading and deterministic preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Return MNIST pixels as floats in [0, 1] without mutating input."""
    pixels = np.asarray(images, dtype=float)
    if pixels.size == 0 or np.any((pixels < 0) | (pixels > 255)):
        raise ValueError("pixel values must be between 0 and 255")
    return pixels / 255.0


def filter_binary_digits(
    images: np.ndarray,
    labels: np.ndarray,
    negative_digit: int = 0,
    positive_digit: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Select two digits and map their labels to -1 and +1."""
    features = np.asarray(images)
    targets = np.asarray(labels, dtype=int)
    if len(features) != len(targets):
        raise ValueError("images and labels must contain the same number of samples")
    mask = (targets == negative_digit) | (targets == positive_digit)
    selected_targets = np.where(targets[mask] == positive_digit, 1, -1)
    return features[mask].copy(), selected_targets


def deterministic_split(
    images: np.ndarray,
    labels: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split sample-major arrays using a repeatable seed."""
    features = np.asarray(images)
    targets = np.asarray(labels)
    if len(features) != len(targets):
        raise ValueError("images and labels must contain the same number of samples")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    indices = np.random.default_rng(seed).permutation(len(features))
    n_test = max(1, round(len(features) * test_fraction))
    test_indices, train_indices = indices[:n_test], indices[n_test:]
    return (
        features[train_indices].copy(),
        features[test_indices].copy(),
        targets[train_indices].copy(),
        targets[test_indices].copy(),
    )


def split_mnist(
    images: np.ndarray,
    labels: np.ndarray,
    train_size: int = 60_000,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep MNIST test tail untouched and split training pool deterministically."""
    features = np.asarray(images)
    targets = np.asarray(labels)
    if len(features) != len(targets):
        raise ValueError("images and labels must contain the same number of samples")
    if not 0 < train_size < len(features):
        raise ValueError("train_size must leave at least one test sample")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    pool_images, test_images = features[:train_size], features[train_size:]
    pool_labels, test_labels = targets[:train_size], targets[train_size:]
    indices = np.random.default_rng(seed).permutation(train_size)
    validation_size = max(1, round(train_size * validation_fraction))
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]
    return (
        pool_images[training_indices].copy(),
        pool_labels[training_indices].copy(),
        pool_images[validation_indices].copy(),
        pool_labels[validation_indices].copy(),
        test_images.copy(),
        test_labels.copy(),
    )


def resize_images_nearest(images: np.ndarray, side: int = 14) -> np.ndarray:
    """Resize flattened square images with nearest-neighbor sampling."""
    features = np.asarray(images)
    if features.ndim != 2:
        raise ValueError("images must have shape (samples, flattened pixels)")
    old_side = int(np.sqrt(features.shape[1]))
    if old_side * old_side != features.shape[1]:
        raise ValueError("each flattened image must contain a square number of pixels")
    if side <= 0:
        raise ValueError("side must be positive")
    positions = np.rint(np.linspace(0, old_side - 1, side)).astype(int)
    grids = features.reshape(-1, old_side, old_side)
    return grids[:, positions][:, :, positions].reshape(len(features), side * side)


def load_mnist(cache_dir: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    """Fetch MNIST from OpenML and cache downloaded files outside Git."""
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as error:
        raise RuntimeError("install project dependencies before downloading MNIST") from error

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff",
        data_home=cache,
    )
    images = np.asarray(dataset.data, dtype=float)
    labels = np.asarray(dataset.target, dtype=int)
    if images.shape != (70_000, 784) or labels.shape != (70_000,):
        raise RuntimeError("OpenML returned an unexpected MNIST shape")
    return images, labels
