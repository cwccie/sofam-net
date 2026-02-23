"""Tests for the FL Client trainer."""

import numpy as np
import pytest

from sofam_net.client.trainer import FederatedClient, TrainingConfig, TrainingResult


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.epochs == 5
        assert config.batch_size == 32
        assert config.learning_rate == 0.01

    def test_custom(self):
        config = TrainingConfig(epochs=10, batch_size=64, learning_rate=0.001)
        assert config.epochs == 10
        assert config.batch_size == 64


class TestFederatedClient:
    def test_creation(self):
        client = FederatedClient(org_name="OrgA")
        assert client.org_name == "OrgA"
        assert client.client_id.startswith("client-")

    def test_custom_id(self):
        client = FederatedClient(client_id="my-client")
        assert client.client_id == "my-client"

    def test_set_get_parameters(self, sample_params):
        client = FederatedClient()
        client.set_parameters(sample_params)
        params = client.get_parameters()
        assert len(params) == len(sample_params)
        for orig, got in zip(sample_params, params):
            np.testing.assert_array_equal(orig, got)

    def test_get_parameters_unset(self):
        client = FederatedClient()
        with pytest.raises(RuntimeError, match="No model parameters"):
            client.get_parameters()

    def test_train(self, sample_params, sample_data):
        X, y = sample_data
        client = FederatedClient(config=TrainingConfig(epochs=2, batch_size=32))
        client.set_parameters(sample_params)
        result = client.train(X, y)
        assert isinstance(result, TrainingResult)
        assert result.num_samples == 200
        assert "loss" in result.metrics
        assert "accuracy" in result.metrics
        assert result.training_time > 0

    def test_train_unset_params(self, sample_data):
        X, y = sample_data
        client = FederatedClient()
        with pytest.raises(RuntimeError, match="No model parameters"):
            client.train(X, y)

    def test_train_with_fedprox(self, sample_params, sample_data):
        X, y = sample_data
        client = FederatedClient(config=TrainingConfig(epochs=2))
        client.set_parameters(sample_params)
        config = {"mu": 0.1}
        result = client.train(X, y, round_config=config)
        assert result.num_samples == 200

    def test_train_with_gradient_clipping(self, sample_params, sample_data):
        X, y = sample_data
        config = TrainingConfig(epochs=2, gradient_clip_norm=1.0)
        client = FederatedClient(config=config)
        client.set_parameters(sample_params)
        result = client.train(X, y)
        assert result.num_samples == 200

    def test_train_with_compression(self, sample_params, sample_data):
        X, y = sample_data
        config = TrainingConfig(epochs=2, compress_threshold=0.1)
        client = FederatedClient(config=config)
        client.set_parameters(sample_params)
        result = client.train(X, y)
        assert result.num_samples == 200

    def test_train_with_momentum(self, sample_params, sample_data):
        X, y = sample_data
        config = TrainingConfig(epochs=2, momentum=0.9)
        client = FederatedClient(config=config)
        client.set_parameters(sample_params)
        result = client.train(X, y)
        assert "loss" in result.metrics

    def test_heartbeat(self):
        client = FederatedClient(client_id="test-hb", org_name="OrgX")
        hb = client.heartbeat()
        assert hb["client_id"] == "test-hb"
        assert hb["org_name"] == "OrgX"
        assert "timestamp" in hb
