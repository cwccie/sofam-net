"""Tests for the simulation harness."""


from sofam_net.simulation.runner import SimulationConfig, SimulationRunner


class TestSimulationConfig:
    def test_defaults(self):
        config = SimulationConfig()
        assert config.num_orgs == 3
        assert config.num_rounds == 10
        assert config.strategy == "fedavg"

    def test_custom(self):
        config = SimulationConfig(
            num_orgs=5, num_rounds=20, strategy="fedprox", enable_dp=True
        )
        assert config.num_orgs == 5
        assert config.enable_dp


class TestSimulationRunner:
    def test_basic_simulation(self):
        config = SimulationConfig(
            num_samples=500,
            num_features=10,
            num_classes=3,
            num_orgs=2,
            num_rounds=3,
            local_epochs=2,
            seed=42,
            verbose=False,
        )
        runner = SimulationRunner(config)
        result = runner.run()
        assert result.final_accuracy >= 0
        assert result.total_time > 0
        assert len(result.round_metrics) <= 3

    def test_simulation_with_dp(self):
        config = SimulationConfig(
            num_samples=500,
            num_features=10,
            num_classes=3,
            num_orgs=2,
            num_rounds=3,
            local_epochs=2,
            enable_dp=True,
            epsilon_per_round=1.0,
            total_epsilon_budget=10.0,
            seed=42,
            verbose=False,
        )
        runner = SimulationRunner(config)
        result = runner.run()
        assert len(result.privacy_spent) > 0

    def test_simulation_fedprox(self):
        config = SimulationConfig(
            num_samples=500,
            num_features=10,
            num_classes=3,
            num_orgs=2,
            num_rounds=3,
            strategy="fedprox",
            strategy_params={"mu": 0.01},
            seed=42,
            verbose=False,
        )
        runner = SimulationRunner(config)
        result = runner.run()
        assert result.final_accuracy >= 0

    def test_org_accuracies_populated(self):
        config = SimulationConfig(
            num_samples=500,
            num_features=10,
            num_classes=3,
            num_orgs=3,
            num_rounds=2,
            seed=42,
            verbose=False,
        )
        runner = SimulationRunner(config)
        result = runner.run()
        assert len(result.org_accuracies) > 0
