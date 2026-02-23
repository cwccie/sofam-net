"""
Federated Learning Client.

Handles local model training, gradient computation, model serialization,
and communication with the coordinator server.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Local training configuration."""

    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None
    compress_threshold: float | None = None  # Top-k gradient compression


@dataclass
class TrainingResult:
    """Result of local training."""

    parameters: list[np.ndarray]
    num_samples: int
    metrics: dict[str, float]
    training_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


class FederatedClient:
    """
    Federated learning client that trains locally and communicates with the coordinator.

    Supports:
    - Local SGD training with configurable hyperparameters
    - Gradient compression (top-k sparsification)
    - FedProx proximal term
    - SCAFFOLD control variates
    - Heartbeat reporting
    """

    def __init__(
        self,
        client_id: str | None = None,
        org_name: str = "default",
        config: TrainingConfig | None = None,
    ):
        self.client_id = client_id or f"client-{uuid.uuid4().hex[:8]}"
        self.org_name = org_name
        self.config = config or TrainingConfig()
        self.model_parameters: list[np.ndarray] | None = None
        self.last_heartbeat = time.time()
        self._control_variate: list[np.ndarray] | None = None

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load global model parameters."""
        self.model_parameters = [np.copy(p) for p in parameters]

    def get_parameters(self) -> list[np.ndarray]:
        """Return current local model parameters."""
        if self.model_parameters is None:
            raise RuntimeError("No model parameters set")
        return self.model_parameters

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        round_config: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """
        Run local training on the given data.

        Implements mini-batch SGD with support for:
        - FedProx proximal regularization (when mu is in round_config)
        - Gradient clipping
        - Momentum
        """
        if self.model_parameters is None:
            raise RuntimeError("No model parameters set. Call set_parameters() first.")

        start_time = time.time()
        config = round_config or {}
        mu = config.get("mu", 0.0)  # FedProx proximal term
        global_params = [np.copy(p) for p in self.model_parameters] if mu > 0 else None

        n_samples = len(X)
        batch_size = min(self.config.batch_size, n_samples)
        velocities = [np.zeros_like(p) for p in self.model_parameters]

        epoch_losses = []
        for epoch in range(self.config.epochs):
            indices = np.random.permutation(n_samples)
            batch_losses = []

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                # Forward pass (simple MLP-style)
                loss, gradients = self._compute_gradients(X_batch, y_batch)

                # FedProx proximal term
                if mu > 0 and global_params is not None:
                    for i, (p, gp) in enumerate(zip(self.model_parameters, global_params)):
                        gradients[i] = gradients[i] + mu * (p - gp)

                # Gradient clipping
                if self.config.gradient_clip_norm is not None:
                    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
                    if total_norm > self.config.gradient_clip_norm:
                        scale = self.config.gradient_clip_norm / (total_norm + 1e-8)
                        gradients = [g * scale for g in gradients]

                # Gradient compression (top-k sparsification)
                if self.config.compress_threshold is not None:
                    gradients = self._compress_gradients(gradients)

                # SGD with momentum
                for i, (p, g) in enumerate(zip(self.model_parameters, gradients)):
                    if self.config.momentum > 0:
                        velocities[i] = self.config.momentum * velocities[i] - self.config.learning_rate * g
                        self.model_parameters[i] = p + velocities[i]
                    else:
                        self.model_parameters[i] = p - self.config.learning_rate * g

                    # Weight decay
                    if self.config.weight_decay > 0:
                        self.model_parameters[i] *= 1.0 - self.config.learning_rate * self.config.weight_decay

                batch_losses.append(loss)

            epoch_loss = float(np.mean(batch_losses))
            epoch_losses.append(epoch_loss)

        training_time = time.time() - start_time

        # Compute accuracy on training data
        predictions = self._predict(X)
        accuracy = float(np.mean(predictions == y))

        metrics = {
            "loss": epoch_losses[-1] if epoch_losses else 0.0,
            "accuracy": accuracy,
            "epochs": self.config.epochs,
            "training_time": training_time,
        }

        metadata: dict[str, Any] = {}

        logger.info(
            "Client %s training complete: loss=%.4f, acc=%.4f, time=%.2fs",
            self.client_id, metrics["loss"], accuracy, training_time,
        )

        return TrainingResult(
            parameters=self.model_parameters,
            num_samples=n_samples,
            metrics=metrics,
            training_time=training_time,
            metadata=metadata,
        )

    def _compute_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, list[np.ndarray]]:
        """
        Compute loss and gradients for a batch.

        Uses a simple 2-layer MLP architecture:
        - Layer 0: weights (input_dim, hidden_dim)
        - Layer 1: bias (hidden_dim,)
        - Layer 2: weights (hidden_dim, output_dim)
        - Layer 3: bias (output_dim,)
        """
        params = self.model_parameters
        if params is None or len(params) < 4:
            # Fallback: treat as simple linear model
            return self._compute_linear_gradients(X, y)

        W1, b1, W2, b2 = params[0], params[1], params[2], params[3]

        # Forward
        z1 = X @ W1 + b1  # (batch, hidden)
        a1 = np.maximum(z1, 0)  # ReLU
        z2 = a1 @ W2 + b2  # (batch, output)

        # Softmax + cross-entropy
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z2 / (np.sum(exp_z2, axis=1, keepdims=True) + 1e-10)

        n = len(y)
        y_onehot = np.zeros_like(probs)
        y_int = y.astype(int)
        y_int = np.clip(y_int, 0, probs.shape[1] - 1)
        y_onehot[np.arange(n), y_int] = 1.0

        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-10), axis=1))

        # Backward
        dz2 = (probs - y_onehot) / n
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0).astype(float)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return float(loss), [dW1, db1, dW2, db2]

    def _compute_linear_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, list[np.ndarray]]:
        """Fallback gradient computation for arbitrary parameter shapes."""
        params = self.model_parameters
        if not params:
            return 0.0, []

        # Simple MSE-like gradient estimate via finite differences
        gradients = []
        base_pred = self._predict_raw(X)
        base_loss = float(np.mean((base_pred - y.reshape(-1, 1)) ** 2))
        eps = 1e-5

        for i, p in enumerate(params):
            grad = np.zeros_like(p)
            flat_p = p.ravel()
            flat_g = grad.ravel()
            # Sample a subset of parameters for efficiency
            sample_size = min(100, len(flat_p))
            sample_idx = np.random.choice(len(flat_p), sample_size, replace=False)
            for idx in sample_idx:
                old_val = flat_p[idx]
                flat_p[idx] = old_val + eps
                params[i] = flat_p.reshape(p.shape)
                pred_plus = self._predict_raw(X)
                loss_plus = float(np.mean((pred_plus - y.reshape(-1, 1)) ** 2))
                flat_g[idx] = (loss_plus - base_loss) / eps
                flat_p[idx] = old_val
            params[i] = flat_p.reshape(p.shape)
            gradients.append(flat_g.reshape(p.shape))

        return base_loss, gradients

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw forward pass returning logits."""
        params = self.model_parameters
        if params is None or len(params) < 4:
            if params and len(params) >= 2:
                return X @ params[0] + params[1]
            return np.zeros((len(X), 1))
        W1, b1, W2, b2 = params[0], params[1], params[2], params[3]
        z1 = X @ W1 + b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ W2 + b2
        return z2

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        logits = self._predict_raw(X)
        if logits.shape[1] == 1:
            return (logits.ravel() > 0).astype(int)
        return np.argmax(logits, axis=1)

    def _compress_gradients(self, gradients: list[np.ndarray]) -> list[np.ndarray]:
        """Apply top-k gradient sparsification."""
        threshold = self.config.compress_threshold
        if threshold is None:
            return gradients

        compressed = []
        for g in gradients:
            flat = g.ravel()
            k = max(1, int(len(flat) * threshold))
            top_k_idx = np.argpartition(np.abs(flat), -k)[-k:]
            sparse = np.zeros_like(flat)
            sparse[top_k_idx] = flat[top_k_idx]
            compressed.append(sparse.reshape(g.shape))
        return compressed

    def heartbeat(self) -> dict[str, Any]:
        """Generate heartbeat payload."""
        self.last_heartbeat = time.time()
        return {
            "client_id": self.client_id,
            "org_name": self.org_name,
            "timestamp": self.last_heartbeat,
            "has_model": self.model_parameters is not None,
        }
