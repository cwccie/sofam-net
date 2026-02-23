"""
MLP Classifier for Intrusion Detection.

A simple multi-layer perceptron implemented in pure NumPy for classifying
network flows as benign or various attack types.
"""

from __future__ import annotations

import numpy as np


class MLPClassifier:
    """
    Multi-Layer Perceptron classifier for network intrusion detection.

    Architecture:
        Input → Dense(hidden) → ReLU → Dense(num_classes) → Softmax

    All operations use NumPy — no framework dependency.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        seed: int | None = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        rng = np.random.default_rng(seed)
        # Xavier initialization
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + num_classes))

        self.parameters: list[np.ndarray] = [
            rng.normal(0, scale1, (input_dim, hidden_dim)).astype(np.float32),
            np.zeros(hidden_dim, dtype=np.float32),
            rng.normal(0, scale2, (hidden_dim, num_classes)).astype(np.float32),
            np.zeros(num_classes, dtype=np.float32),
        ]

    def get_parameters(self) -> list[np.ndarray]:
        """Return model parameters."""
        return self.parameters

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Set model parameters."""
        self.parameters = [np.copy(p) for p in parameters]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass returning class probabilities."""
        W1, b1, W2, b2 = self.parameters
        z1 = X @ W1 + b1
        a1 = np.maximum(z1, 0)  # ReLU
        z2 = a1 @ W2 + b2
        # Softmax
        exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-10)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate model on data, returning loss and accuracy."""
        probs = self.forward(X)
        y_int = y.astype(int)

        # Cross-entropy loss
        n = len(y)
        log_probs = np.log(probs + 1e-10)
        loss = -np.mean(log_probs[np.arange(n), np.clip(y_int, 0, self.num_classes - 1)])

        # Accuracy
        preds = np.argmax(probs, axis=1)
        accuracy = float(np.mean(preds == y_int))

        # Per-class metrics
        precision_per_class = {}
        recall_per_class = {}
        for c in range(self.num_classes):
            tp = np.sum((preds == c) & (y_int == c))
            fp = np.sum((preds == c) & (y_int != c))
            fn = np.sum((preds != c) & (y_int == c))
            precision_per_class[f"precision_class_{c}"] = float(tp / (tp + fp + 1e-10))
            recall_per_class[f"recall_class_{c}"] = float(tp / (tp + fn + 1e-10))

        metrics = {"loss": float(loss), "accuracy": accuracy}
        metrics.update(precision_per_class)
        metrics.update(recall_per_class)
        return metrics

    def get_param_shapes(self) -> list[tuple[int, ...]]:
        """Return shapes of all parameters."""
        return [p.shape for p in self.parameters]

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.size for p in self.parameters)
