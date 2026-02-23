"""
Autoencoder Anomaly Detector for IDS.

Learns to reconstruct normal network traffic; anomalies produce
high reconstruction error, indicating potential intrusions.
"""

from __future__ import annotations

import numpy as np


class AnomalyAutoencoder:
    """
    Autoencoder for network anomaly detection.

    Architecture:
        Encoder: Input → Dense(encoding_dim) → ReLU
        Decoder: Dense(encoding_dim) → Dense(input_dim) → Sigmoid

    Anomalies are detected by thresholding the reconstruction error (MSE).
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 32,
        seed: int | None = None,
        threshold_percentile: float = 95.0,
    ):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.anomaly_threshold: float | None = None

        rng = np.random.default_rng(seed)
        scale_enc = np.sqrt(2.0 / (input_dim + encoding_dim))
        scale_dec = np.sqrt(2.0 / (encoding_dim + input_dim))

        self.parameters: list[np.ndarray] = [
            # Encoder
            rng.normal(0, scale_enc, (input_dim, encoding_dim)).astype(np.float32),
            np.zeros(encoding_dim, dtype=np.float32),
            # Decoder
            rng.normal(0, scale_dec, (encoding_dim, input_dim)).astype(np.float32),
            np.zeros(input_dim, dtype=np.float32),
        ]

    def get_parameters(self) -> list[np.ndarray]:
        return self.parameters

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self.parameters = [np.copy(p) for p in parameters]

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode input to latent representation."""
        W_enc, b_enc = self.parameters[0], self.parameters[1]
        z = X @ W_enc + b_enc
        return np.maximum(z, 0)  # ReLU

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent representation back to input space."""
        W_dec, b_dec = self.parameters[2], self.parameters[3]
        out = z @ W_dec + b_dec
        # Sigmoid activation
        return 1.0 / (1.0 + np.exp(-np.clip(out, -500, 500)))

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Full forward pass: encode then decode."""
        return self.decode(self.encode(X))

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample MSE reconstruction error."""
        X_hat = self.reconstruct(X)
        return np.mean((X - X_hat) ** 2, axis=1)

    def fit_threshold(self, X_normal: np.ndarray) -> float:
        """
        Fit anomaly threshold on normal (benign) data.

        Sets threshold at the given percentile of reconstruction errors.
        """
        errors = self.reconstruction_error(X_normal)
        self.anomaly_threshold = float(np.percentile(errors, self.threshold_percentile))
        return self.anomaly_threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies. Returns 1 for anomaly, 0 for normal.
        """
        if self.anomaly_threshold is None:
            raise RuntimeError("Call fit_threshold() first")
        errors = self.reconstruction_error(X)
        return (errors > self.anomaly_threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate anomaly detection performance."""
        errors = self.reconstruction_error(X)
        mean_error = float(np.mean(errors))

        if self.anomaly_threshold is not None:
            preds = (errors > self.anomaly_threshold).astype(int)
            y_int = y.astype(int)
            accuracy = float(np.mean(preds == y_int))
            tp = float(np.sum((preds == 1) & (y_int == 1)))
            fp = float(np.sum((preds == 1) & (y_int == 0)))
            fn = float(np.sum((preds == 0) & (y_int == 1)))
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            return {
                "mse": mean_error,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "threshold": self.anomaly_threshold,
            }

        return {"mse": mean_error}
