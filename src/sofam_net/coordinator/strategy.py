"""
Federated learning aggregation strategies.

Implements FedAvg, FedProx, and SCAFFOLD without hard Flower dependency.
Each strategy defines how client model updates are combined into a global model.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    """A model update from a single client."""

    client_id: str
    parameters: list[np.ndarray]
    num_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationResult:
    """Result of aggregating client updates."""

    parameters: list[np.ndarray]
    num_clients: int
    total_samples: int
    metrics: dict[str, float] = field(default_factory=dict)


class Strategy(ABC):
    """Base class for FL aggregation strategies."""

    @abstractmethod
    def aggregate(
        self, updates: list[ClientUpdate], global_params: list[np.ndarray] | None = None
    ) -> AggregationResult:
        """Aggregate client updates into new global parameters."""

    @abstractmethod
    def configure_round(self, round_num: int) -> dict[str, Any]:
        """Return per-round configuration to send to clients."""


class FedAvgStrategy(Strategy):
    """
    Federated Averaging (McMahan et al., 2017).

    Computes a weighted average of client parameters where weights
    are proportional to each client's dataset size.
    """

    def __init__(self, min_clients: int = 2, min_samples: int = 10):
        self.min_clients = min_clients
        self.min_samples = min_samples

    def aggregate(
        self, updates: list[ClientUpdate], global_params: list[np.ndarray] | None = None
    ) -> AggregationResult:
        if len(updates) < self.min_clients:
            raise ValueError(
                f"Need at least {self.min_clients} clients, got {len(updates)}"
            )

        total_samples = sum(u.num_samples for u in updates)
        if total_samples < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} total samples, got {total_samples}"
            )

        # Weighted average by number of samples
        aggregated = []
        for layer_idx in range(len(updates[0].parameters)):
            weighted_sum = np.zeros_like(updates[0].parameters[layer_idx], dtype=np.float64)
            for update in updates:
                weight = update.num_samples / total_samples
                weighted_sum += weight * update.parameters[layer_idx].astype(np.float64)
            aggregated.append(weighted_sum.astype(updates[0].parameters[layer_idx].dtype))

        # Aggregate metrics (weighted average)
        agg_metrics: dict[str, float] = {}
        all_metric_keys = set()
        for u in updates:
            all_metric_keys.update(u.metrics.keys())
        for key in all_metric_keys:
            weighted_val = 0.0
            for u in updates:
                if key in u.metrics:
                    weighted_val += (u.num_samples / total_samples) * u.metrics[key]
            agg_metrics[key] = weighted_val

        logger.info(
            "FedAvg aggregation: %d clients, %d total samples",
            len(updates), total_samples,
        )
        return AggregationResult(
            parameters=aggregated,
            num_clients=len(updates),
            total_samples=total_samples,
            metrics=agg_metrics,
        )

    def configure_round(self, round_num: int) -> dict[str, Any]:
        return {"strategy": "fedavg", "round": round_num}


class FedProxStrategy(Strategy):
    """
    FedProx (Li et al., 2020).

    Like FedAvg but sends a proximal term coefficient (mu) to clients,
    which penalizes deviation from the global model during local training.
    """

    def __init__(self, mu: float = 0.01, min_clients: int = 2, min_samples: int = 10):
        self.mu = mu
        self.min_clients = min_clients
        self.min_samples = min_samples
        self._fedavg = FedAvgStrategy(min_clients=min_clients, min_samples=min_samples)

    def aggregate(
        self, updates: list[ClientUpdate], global_params: list[np.ndarray] | None = None
    ) -> AggregationResult:
        # FedProx aggregation is identical to FedAvg;
        # the proximal penalty is enforced during local training on the client side.
        return self._fedavg.aggregate(updates, global_params)

    def configure_round(self, round_num: int) -> dict[str, Any]:
        return {"strategy": "fedprox", "round": round_num, "mu": self.mu}


class SCAFFOLDStrategy(Strategy):
    """
    SCAFFOLD (Karimireddy et al., 2020).

    Uses control variates to correct for client drift in heterogeneous settings.
    Maintains server-level and client-level control variates.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        min_clients: int = 2,
        min_samples: int = 10,
    ):
        self.learning_rate = learning_rate
        self.min_clients = min_clients
        self.min_samples = min_samples
        self.server_control: list[np.ndarray] | None = None
        self.client_controls: dict[str, list[np.ndarray]] = {}

    def aggregate(
        self, updates: list[ClientUpdate], global_params: list[np.ndarray] | None = None
    ) -> AggregationResult:
        if len(updates) < self.min_clients:
            raise ValueError(
                f"Need at least {self.min_clients} clients, got {len(updates)}"
            )

        total_samples = sum(u.num_samples for u in updates)
        num_clients = len(updates)

        # Initialize server control variates if needed
        if self.server_control is None and global_params is not None:
            self.server_control = [np.zeros_like(p) for p in global_params]

        # Standard weighted average of parameters (like FedAvg)
        aggregated = []
        for layer_idx in range(len(updates[0].parameters)):
            weighted_sum = np.zeros_like(updates[0].parameters[layer_idx], dtype=np.float64)
            for update in updates:
                weight = update.num_samples / total_samples
                weighted_sum += weight * update.parameters[layer_idx].astype(np.float64)
            aggregated.append(weighted_sum.astype(updates[0].parameters[layer_idx].dtype))

        # Update server control variates using client delta controls
        if self.server_control is not None:
            for update in updates:
                client_id = update.client_id
                delta_controls = update.metadata.get("delta_control")
                if delta_controls is not None:
                    for i, dc in enumerate(delta_controls):
                        self.server_control[i] = (
                            self.server_control[i] + np.asarray(dc) / num_clients
                        )
                    # Store updated client control
                    old_ctrl = self.client_controls.get(client_id)
                    if old_ctrl is not None:
                        self.client_controls[client_id] = [
                            c + np.asarray(dc)
                            for c, dc in zip(old_ctrl, delta_controls)
                        ]
                    else:
                        self.client_controls[client_id] = [
                            np.asarray(dc) for dc in delta_controls
                        ]

        logger.info(
            "SCAFFOLD aggregation: %d clients, %d total samples",
            num_clients, total_samples,
        )
        return AggregationResult(
            parameters=aggregated,
            num_clients=num_clients,
            total_samples=total_samples,
            metrics={},
        )

    def configure_round(self, round_num: int) -> dict[str, Any]:
        config: dict[str, Any] = {
            "strategy": "scaffold",
            "round": round_num,
            "learning_rate": self.learning_rate,
        }
        if self.server_control is not None:
            config["server_control"] = [c.tolist() for c in self.server_control]
        return config
