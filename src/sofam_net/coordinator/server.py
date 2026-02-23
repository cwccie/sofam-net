"""
FL Coordinator Server.

Manages federated learning rounds: accepts client registrations, distributes
global model parameters, collects updates, and orchestrates aggregation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from sofam_net.coordinator.strategy import (
    AggregationResult,
    ClientUpdate,
    FedAvgStrategy,
    FedProxStrategy,
    SCAFFOLDStrategy,
    Strategy,
)

logger = logging.getLogger(__name__)


class ServerState(Enum):
    IDLE = "idle"
    WAITING_FOR_CLIENTS = "waiting_for_clients"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"


@dataclass
class ClientInfo:
    """Registered client metadata."""

    client_id: str
    org_name: str
    num_samples: int
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    rounds_participated: int = 0


@dataclass
class RoundResult:
    """Result of a single FL round."""

    round_num: int
    num_clients: int
    total_samples: int
    metrics: dict[str, float]
    duration_seconds: float
    converged: bool = False


class CoordinatorServer:
    """
    Federated Learning Coordinator Server.

    Orchestrates the FL training loop:
    1. Clients register with the coordinator
    2. Coordinator distributes global model + round config
    3. Clients train locally and submit updates
    4. Coordinator aggregates updates via selected strategy
    5. Repeat until convergence or max rounds
    """

    def __init__(
        self,
        strategy: str = "fedavg",
        num_rounds: int = 10,
        min_clients: int = 2,
        min_fit_clients: int = 2,
        convergence_threshold: float = 1e-4,
        convergence_window: int = 3,
        strategy_params: dict[str, Any] | None = None,
    ):
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.min_fit_clients = min_fit_clients
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window

        # Initialize strategy
        params = strategy_params or {}
        self.strategy: Strategy = self._create_strategy(strategy, params)

        # State
        self.state = ServerState.IDLE
        self.current_round = 0
        self.global_parameters: list[np.ndarray] | None = None
        self.clients: dict[str, ClientInfo] = {}
        self.round_results: list[RoundResult] = []
        self.pending_updates: list[ClientUpdate] = []
        self._loss_history: list[float] = []

    def _create_strategy(self, name: str, params: dict[str, Any]) -> Strategy:
        strategies = {
            "fedavg": FedAvgStrategy,
            "fedprox": FedProxStrategy,
            "scaffold": SCAFFOLDStrategy,
        }
        if name not in strategies:
            raise ValueError(f"Unknown strategy '{name}'. Available: {list(strategies.keys())}")
        return strategies[name](**params)

    def register_client(self, client_id: str, org_name: str, num_samples: int) -> dict[str, Any]:
        """Register a new client with the coordinator."""
        self.clients[client_id] = ClientInfo(
            client_id=client_id,
            org_name=org_name,
            num_samples=num_samples,
        )
        logger.info("Client registered: %s (org=%s, samples=%d)", client_id, org_name, num_samples)
        return {
            "status": "registered",
            "client_id": client_id,
            "current_round": self.current_round,
            "num_registered": len(self.clients),
        }

    def unregister_client(self, client_id: str) -> bool:
        """Remove a client from the coordinator."""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info("Client unregistered: %s", client_id)
            return True
        return False

    def heartbeat(self, client_id: str) -> dict[str, Any]:
        """Process client heartbeat."""
        if client_id in self.clients:
            self.clients[client_id].last_heartbeat = time.time()
            return {
                "status": "ok",
                "state": self.state.value,
                "current_round": self.current_round,
            }
        return {"status": "unknown_client"}

    def get_global_model(self) -> dict[str, Any]:
        """Return current global model parameters and round config."""
        config = self.strategy.configure_round(self.current_round)
        return {
            "round": self.current_round,
            "parameters": (
                [p.tolist() for p in self.global_parameters]
                if self.global_parameters is not None
                else None
            ),
            "config": config,
        }

    def submit_update(self, update: ClientUpdate) -> dict[str, Any]:
        """Accept a client model update."""
        if update.client_id not in self.clients:
            return {"status": "error", "message": "Client not registered"}

        self.pending_updates.append(update)
        self.clients[update.client_id].rounds_participated += 1

        logger.info(
            "Update received from %s (round %d, %d/%d updates)",
            update.client_id,
            self.current_round,
            len(self.pending_updates),
            self.min_fit_clients,
        )

        return {
            "status": "accepted",
            "updates_received": len(self.pending_updates),
            "updates_needed": self.min_fit_clients,
        }

    def initialize_global_model(self, parameters: list[np.ndarray]) -> None:
        """Set initial global model parameters."""
        self.global_parameters = parameters
        logger.info(
            "Global model initialized: %d layers, %d total parameters",
            len(parameters),
            sum(p.size for p in parameters),
        )

    def run_round(self) -> RoundResult | None:
        """
        Execute one FL round: aggregate pending updates and produce new global model.

        Returns None if not enough updates are available.
        """
        if len(self.pending_updates) < self.min_fit_clients:
            logger.warning(
                "Not enough updates for round %d: %d/%d",
                self.current_round,
                len(self.pending_updates),
                self.min_fit_clients,
            )
            return None

        self.state = ServerState.AGGREGATING
        start_time = time.time()

        # Aggregate
        result: AggregationResult = self.strategy.aggregate(
            self.pending_updates, self.global_parameters
        )
        self.global_parameters = result.parameters

        duration = time.time() - start_time

        # Check convergence
        loss = result.metrics.get("loss", float("inf"))
        self._loss_history.append(loss)
        converged = self._check_convergence()

        round_result = RoundResult(
            round_num=self.current_round,
            num_clients=result.num_clients,
            total_samples=result.total_samples,
            metrics=result.metrics,
            duration_seconds=duration,
            converged=converged,
        )
        self.round_results.append(round_result)

        # Advance round
        self.current_round += 1
        self.pending_updates = []
        self.state = (
            ServerState.COMPLETED if converged or self.current_round >= self.num_rounds
            else ServerState.WAITING_FOR_CLIENTS
        )

        logger.info(
            "Round %d complete: %d clients, loss=%.6f, converged=%s",
            round_result.round_num,
            round_result.num_clients,
            loss,
            converged,
        )
        return round_result

    def _check_convergence(self) -> bool:
        """Check if training has converged based on loss history."""
        if len(self._loss_history) < self.convergence_window:
            return False

        recent = self._loss_history[-self.convergence_window:]
        if all(v == float("inf") for v in recent):
            return False

        finite = [v for v in recent if v != float("inf")]
        if len(finite) < 2:
            return False

        max_delta = max(abs(finite[i] - finite[i - 1]) for i in range(1, len(finite)))
        return max_delta < self.convergence_threshold

    def get_status(self) -> dict[str, Any]:
        """Return full coordinator status."""
        return {
            "state": self.state.value,
            "current_round": self.current_round,
            "total_rounds": self.num_rounds,
            "registered_clients": len(self.clients),
            "pending_updates": len(self.pending_updates),
            "rounds_completed": len(self.round_results),
            "clients": {
                cid: {
                    "org": c.org_name,
                    "samples": c.num_samples,
                    "rounds": c.rounds_participated,
                }
                for cid, c in self.clients.items()
            },
            "history": [
                {
                    "round": r.round_num,
                    "clients": r.num_clients,
                    "samples": r.total_samples,
                    "metrics": r.metrics,
                    "duration": r.duration_seconds,
                    "converged": r.converged,
                }
                for r in self.round_results
            ],
        }
