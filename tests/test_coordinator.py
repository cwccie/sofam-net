"""Tests for the FL Coordinator server and strategies."""

import numpy as np
import pytest

from sofam_net.coordinator.server import CoordinatorServer
from sofam_net.coordinator.strategy import (
    ClientUpdate,
    FedAvgStrategy,
    FedProxStrategy,
    SCAFFOLDStrategy,
)


class TestFedAvgStrategy:
    def test_basic_aggregation(self, sample_params):
        strategy = FedAvgStrategy(min_clients=2)
        updates = [
            ClientUpdate("c1", [p + 0.1 for p in sample_params], 100),
            ClientUpdate("c2", [p - 0.1 for p in sample_params], 100),
        ]
        result = strategy.aggregate(updates)
        assert result.num_clients == 2
        assert result.total_samples == 200
        # Equal weights → should average to original params
        for orig, agg in zip(sample_params, result.parameters):
            np.testing.assert_allclose(agg, orig, atol=1e-6)

    def test_weighted_aggregation(self, sample_params):
        strategy = FedAvgStrategy(min_clients=2)
        updates = [
            ClientUpdate("c1", [p + 1.0 for p in sample_params], 300),
            ClientUpdate("c2", [p - 1.0 for p in sample_params], 100),
        ]
        result = strategy.aggregate(updates)
        # 300/(300+100)=0.75 weight for c1, 0.25 for c2
        for orig, agg in zip(sample_params, result.parameters):
            expected = orig + 0.75 * 1.0 - 0.25 * 1.0  # = orig + 0.5
            np.testing.assert_allclose(agg, expected, atol=1e-5)

    def test_insufficient_clients(self, sample_params):
        strategy = FedAvgStrategy(min_clients=3)
        updates = [
            ClientUpdate("c1", sample_params, 100),
            ClientUpdate("c2", sample_params, 100),
        ]
        with pytest.raises(ValueError, match="at least 3 clients"):
            strategy.aggregate(updates)

    def test_metrics_aggregation(self, sample_params):
        strategy = FedAvgStrategy(min_clients=2)
        updates = [
            ClientUpdate("c1", sample_params, 100, metrics={"loss": 0.5, "acc": 0.8}),
            ClientUpdate("c2", sample_params, 100, metrics={"loss": 0.3, "acc": 0.9}),
        ]
        result = strategy.aggregate(updates)
        assert abs(result.metrics["loss"] - 0.4) < 1e-6
        assert abs(result.metrics["acc"] - 0.85) < 1e-6

    def test_configure_round(self):
        strategy = FedAvgStrategy()
        config = strategy.configure_round(5)
        assert config["strategy"] == "fedavg"
        assert config["round"] == 5


class TestFedProxStrategy:
    def test_aggregation_same_as_fedavg(self, sample_params):
        fedavg = FedAvgStrategy(min_clients=2)
        fedprox = FedProxStrategy(mu=0.01, min_clients=2)
        updates = [
            ClientUpdate("c1", [p + 0.1 for p in sample_params], 100),
            ClientUpdate("c2", [p - 0.1 for p in sample_params], 100),
        ]
        avg_result = fedavg.aggregate(updates)
        prox_result = fedprox.aggregate(updates)
        for a, b in zip(avg_result.parameters, prox_result.parameters):
            np.testing.assert_array_equal(a, b)

    def test_configure_round_includes_mu(self):
        strategy = FedProxStrategy(mu=0.05)
        config = strategy.configure_round(3)
        assert config["strategy"] == "fedprox"
        assert config["mu"] == 0.05


class TestSCAFFOLDStrategy:
    def test_basic_aggregation(self, sample_params):
        strategy = SCAFFOLDStrategy(min_clients=2)
        updates = [
            ClientUpdate("c1", sample_params, 100),
            ClientUpdate("c2", sample_params, 100),
        ]
        result = strategy.aggregate(updates, global_params=sample_params)
        assert result.num_clients == 2

    def test_configure_round(self):
        strategy = SCAFFOLDStrategy(learning_rate=0.05)
        config = strategy.configure_round(1)
        assert config["strategy"] == "scaffold"
        assert config["learning_rate"] == 0.05


class TestCoordinatorServer:
    def test_client_registration(self):
        server = CoordinatorServer()
        result = server.register_client("c1", "OrgA", 1000)
        assert result["status"] == "registered"
        assert result["num_registered"] == 1

    def test_multiple_registrations(self):
        server = CoordinatorServer()
        server.register_client("c1", "OrgA", 500)
        server.register_client("c2", "OrgB", 700)
        assert len(server.clients) == 2

    def test_unregister_client(self):
        server = CoordinatorServer()
        server.register_client("c1", "OrgA", 500)
        assert server.unregister_client("c1")
        assert len(server.clients) == 0

    def test_heartbeat(self):
        server = CoordinatorServer()
        server.register_client("c1", "OrgA", 500)
        result = server.heartbeat("c1")
        assert result["status"] == "ok"

    def test_heartbeat_unknown_client(self):
        server = CoordinatorServer()
        result = server.heartbeat("unknown")
        assert result["status"] == "unknown_client"

    def test_initialize_model(self, sample_params):
        server = CoordinatorServer()
        server.initialize_global_model(sample_params)
        assert server.global_parameters is not None
        assert len(server.global_parameters) == 4

    def test_get_global_model(self, sample_params):
        server = CoordinatorServer()
        server.initialize_global_model(sample_params)
        model = server.get_global_model()
        assert model["round"] == 0
        assert model["parameters"] is not None

    def test_submit_update(self, sample_params):
        server = CoordinatorServer()
        server.register_client("c1", "OrgA", 500)
        update = ClientUpdate("c1", sample_params, 500)
        result = server.submit_update(update)
        assert result["status"] == "accepted"

    def test_submit_update_unregistered(self, sample_params):
        server = CoordinatorServer()
        update = ClientUpdate("unknown", sample_params, 500)
        result = server.submit_update(update)
        assert result["status"] == "error"

    def test_run_round(self, sample_params):
        server = CoordinatorServer(min_clients=2, min_fit_clients=2)
        server.initialize_global_model(sample_params)
        server.register_client("c1", "OrgA", 500)
        server.register_client("c2", "OrgB", 500)
        server.submit_update(ClientUpdate("c1", [p + 0.01 for p in sample_params], 500, {"loss": 0.5}))
        server.submit_update(ClientUpdate("c2", [p - 0.01 for p in sample_params], 500, {"loss": 0.3}))
        result = server.run_round()
        assert result is not None
        assert result.num_clients == 2
        assert server.current_round == 1

    def test_run_round_insufficient_updates(self, sample_params):
        server = CoordinatorServer(min_clients=2, min_fit_clients=2)
        server.register_client("c1", "OrgA", 500)
        server.submit_update(ClientUpdate("c1", sample_params, 500))
        result = server.run_round()
        assert result is None

    def test_get_status(self):
        server = CoordinatorServer(num_rounds=10)
        server.register_client("c1", "OrgA", 500)
        status = server.get_status()
        assert status["state"] == "idle"
        assert status["registered_clients"] == 1
        assert status["total_rounds"] == 10

    def test_strategy_selection(self):
        for name in ["fedavg", "fedprox", "scaffold"]:
            server = CoordinatorServer(strategy=name)
            assert server.strategy is not None

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            CoordinatorServer(strategy="invalid")
