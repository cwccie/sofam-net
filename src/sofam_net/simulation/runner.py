"""
Multi-Organization Simulation Harness.

Simulates a complete federated learning scenario with multiple organizations,
configurable data heterogeneity, network conditions, and privacy constraints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sofam_net.client.trainer import FederatedClient, TrainingConfig
from sofam_net.coordinator.server import CoordinatorServer
from sofam_net.coordinator.strategy import ClientUpdate
from sofam_net.data.splitter import DirichletSplitter
from sofam_net.data.synthetic import SyntheticIDSGenerator
from sofam_net.models.mlp import MLPClassifier
from sofam_net.privacy.accountant import RDPAccountant
from sofam_net.privacy.mechanisms import GaussianMechanism

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for FL simulation."""

    # Data
    num_samples: int = 10000
    num_features: int = 20
    num_classes: int = 5
    attack_ratio: float = 0.3

    # Federation
    num_orgs: int = 3
    org_names: list[str] | None = None
    dirichlet_alpha: float = 0.5  # Lower = more non-IID

    # Training
    num_rounds: int = 10
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.01
    strategy: str = "fedavg"
    strategy_params: dict[str, Any] = field(default_factory=dict)

    # Model
    hidden_dim: int = 64

    # Privacy
    enable_dp: bool = False
    epsilon_per_round: float = 1.0
    delta: float = 1e-5
    total_epsilon_budget: float = 10.0
    clip_norm: float = 1.0

    # Network simulation
    drop_probability: float = 0.0  # Probability of a client dropping per round
    latency_mean_ms: float = 0.0  # Simulated network latency

    # Misc
    seed: int | None = 42
    verbose: bool = True


@dataclass
class SimulationResult:
    """Complete simulation results."""

    config: SimulationConfig
    round_metrics: list[dict[str, Any]]
    final_accuracy: float
    final_loss: float
    total_time: float
    privacy_spent: dict[str, float]
    org_accuracies: dict[str, float]
    converged: bool
    convergence_round: int | None


class SimulationRunner:
    """
    Run a complete federated learning simulation.

    Orchestrates data generation, splitting, training, aggregation,
    and evaluation across simulated organizations.
    """

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def run(self) -> SimulationResult:
        """Execute the full simulation."""
        cfg = self.config
        start_time = time.time()

        if cfg.verbose:
            logger.info("=" * 60)
            logger.info("SOFAM-Net Federated Learning Simulation")
            logger.info("=" * 60)
            logger.info("Organizations: %d | Rounds: %d | Strategy: %s",
                        cfg.num_orgs, cfg.num_rounds, cfg.strategy)
            logger.info("Data: %d samples, %d features, %d classes",
                        cfg.num_samples, cfg.num_features, cfg.num_classes)
            logger.info("Dirichlet alpha: %.2f | DP: %s",
                        cfg.dirichlet_alpha, cfg.enable_dp)

        # --- Generate Data ---
        gen = SyntheticIDSGenerator(
            num_features=cfg.num_features,
            num_classes=cfg.num_classes,
            seed=cfg.seed,
        )
        dataset = gen.generate(
            num_samples=cfg.num_samples,
            attack_ratio=cfg.attack_ratio,
        )

        # Hold out test set
        n_test = int(cfg.num_samples * 0.2)
        test_idx = self.rng.choice(cfg.num_samples, n_test, replace=False)
        train_mask = np.ones(cfg.num_samples, dtype=bool)
        train_mask[test_idx] = False
        X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]
        X_train, y_train = dataset.X[train_mask], dataset.y[train_mask]

        # --- Split Data Across Orgs ---
        org_names = cfg.org_names or [f"Org-{chr(65 + i)}" for i in range(cfg.num_orgs)]
        splitter = DirichletSplitter(alpha=cfg.dirichlet_alpha, seed=cfg.seed)
        split = splitter.split(X_train, y_train, cfg.num_orgs, org_names)

        # --- Initialize Model ---
        model = MLPClassifier(
            input_dim=cfg.num_features,
            hidden_dim=cfg.hidden_dim,
            num_classes=cfg.num_classes,
            seed=cfg.seed,
        )

        # --- Initialize Coordinator ---
        coordinator = CoordinatorServer(
            strategy=cfg.strategy,
            num_rounds=cfg.num_rounds,
            min_clients=max(1, cfg.num_orgs - int(cfg.num_orgs * cfg.drop_probability)),
            min_fit_clients=max(1, cfg.num_orgs - int(cfg.num_orgs * cfg.drop_probability)),
            strategy_params=cfg.strategy_params,
        )
        coordinator.initialize_global_model(model.get_parameters())

        # --- Initialize Clients ---
        clients: dict[str, FederatedClient] = {}
        for name in org_names:
            client = FederatedClient(
                client_id=f"client-{name}",
                org_name=name,
                config=TrainingConfig(
                    epochs=cfg.local_epochs,
                    batch_size=cfg.batch_size,
                    learning_rate=cfg.learning_rate,
                ),
            )
            coordinator.register_client(client.client_id, name, split.org_sizes[name])
            clients[name] = client

        # --- Privacy Setup ---
        dp_mechanism = None
        accountant = None
        if cfg.enable_dp:
            dp_mechanism = GaussianMechanism(
                epsilon=cfg.epsilon_per_round,
                delta=cfg.delta,
            )
            accountant = RDPAccountant(target_delta=cfg.delta)
            for name in org_names:
                accountant.create_budget(name, cfg.total_epsilon_budget)

        # --- Training Loop ---
        round_metrics = []
        converged = False
        convergence_round = None

        for round_num in range(cfg.num_rounds):
            if cfg.verbose:
                logger.info("--- Round %d/%d ---", round_num + 1, cfg.num_rounds)

            # Get global model
            global_model = coordinator.get_global_model()
            global_params = [np.array(p) for p in global_model["parameters"]]

            # Each org trains locally
            for name in org_names:
                # Simulate client drop
                if cfg.drop_probability > 0 and self.rng.random() < cfg.drop_probability:
                    if cfg.verbose:
                        logger.info("  Client %s dropped (simulated network failure)", name)
                    continue

                # Simulate latency
                if cfg.latency_mean_ms > 0:
                    latency = max(0, self.rng.normal(cfg.latency_mean_ms, cfg.latency_mean_ms * 0.3))
                    time.sleep(latency / 1000.0)

                client = clients[name]
                client.set_parameters(global_params)

                org_X, org_y = split.org_data[name]
                if len(org_X) == 0:
                    continue

                # Check privacy budget
                if accountant and not accountant.can_participate(name, cfg.epsilon_per_round):
                    if cfg.verbose:
                        logger.info("  Client %s: privacy budget exhausted, skipping", name)
                    continue

                result = client.train(org_X, org_y, global_model["config"])

                # Apply DP noise
                params = result.parameters
                if dp_mechanism is not None:
                    # Clip update norm
                    update_delta = [p - gp for p, gp in zip(params, global_params)]
                    total_norm = np.sqrt(sum(float(np.sum(d**2)) for d in update_delta))
                    if total_norm > cfg.clip_norm:
                        scale = cfg.clip_norm / total_norm
                        update_delta = [d * scale for d in update_delta]
                    params = [gp + d for gp, d in zip(global_params, update_delta)]

                    params, noise_report = dp_mechanism.add_noise(params, cfg.clip_norm)
                    if accountant:
                        sigma = noise_report.noise_scale
                        accountant.step(sigma, sampling_rate=1.0)
                        accountant.record_participation(name, cfg.epsilon_per_round, round_num)

                update = ClientUpdate(
                    client_id=client.client_id,
                    parameters=params,
                    num_samples=result.num_samples,
                    metrics=result.metrics,
                )
                coordinator.submit_update(update)

            # Aggregate
            round_result = coordinator.run_round()

            if round_result is not None:
                # Evaluate global model on test set
                model.set_parameters(coordinator.global_parameters)
                eval_metrics = model.evaluate(X_test, y_test)

                round_info = {
                    "round": round_num,
                    "clients": round_result.num_clients,
                    "train_loss": round_result.metrics.get("loss", 0),
                    "train_accuracy": round_result.metrics.get("accuracy", 0),
                    "test_loss": eval_metrics["loss"],
                    "test_accuracy": eval_metrics["accuracy"],
                    "duration": round_result.duration_seconds,
                }
                round_metrics.append(round_info)

                if cfg.verbose:
                    logger.info(
                        "  Global: test_acc=%.4f, test_loss=%.4f",
                        eval_metrics["accuracy"], eval_metrics["loss"],
                    )

                if round_result.converged:
                    converged = True
                    convergence_round = round_num
                    if cfg.verbose:
                        logger.info("  Converged at round %d!", round_num)
                    break

        # --- Final Evaluation ---
        model.set_parameters(coordinator.global_parameters)
        final_eval = model.evaluate(X_test, y_test)

        # Per-org evaluation
        org_accuracies = {}
        for name in org_names:
            org_X, org_y = split.org_data[name]
            if len(org_X) > 0:
                org_eval = model.evaluate(org_X, org_y)
                org_accuracies[name] = org_eval["accuracy"]

        # Privacy summary
        privacy_spent = {}
        if accountant:
            for name, budget in accountant.budgets.items():
                privacy_spent[name] = budget.spent_epsilon

        total_time = time.time() - start_time

        if cfg.verbose:
            logger.info("=" * 60)
            logger.info("Simulation Complete")
            logger.info("Final test accuracy: %.4f", final_eval["accuracy"])
            logger.info("Total time: %.2fs", total_time)
            if privacy_spent:
                logger.info("Privacy spent: %s", privacy_spent)

        return SimulationResult(
            config=cfg,
            round_metrics=round_metrics,
            final_accuracy=final_eval["accuracy"],
            final_loss=final_eval["loss"],
            total_time=total_time,
            privacy_spent=privacy_spent,
            org_accuracies=org_accuracies,
            converged=converged,
            convergence_round=convergence_round,
        )
