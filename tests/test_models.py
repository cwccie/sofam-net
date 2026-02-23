"""Tests for IDS models."""

import numpy as np
import pytest

from sofam_net.models.autoencoder import AnomalyAutoencoder
from sofam_net.models.gnn import FlowGNN
from sofam_net.models.mlp import MLPClassifier


class TestMLPClassifier:
    def test_creation(self):
        model = MLPClassifier(input_dim=10, hidden_dim=32, num_classes=5, seed=42)
        assert model.count_parameters() > 0
        assert len(model.get_parameters()) == 4

    def test_forward(self):
        model = MLPClassifier(input_dim=10, hidden_dim=32, num_classes=5, seed=42)
        X = np.random.randn(50, 10).astype(np.float32)
        probs = model.forward(X)
        assert probs.shape == (50, 5)
        # Should sum to 1 (softmax)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict(self):
        model = MLPClassifier(input_dim=10, num_classes=3, seed=42)
        X = np.random.randn(20, 10).astype(np.float32)
        preds = model.predict(X)
        assert preds.shape == (20,)
        assert all(0 <= p < 3 for p in preds)

    def test_evaluate(self):
        model = MLPClassifier(input_dim=10, num_classes=5, seed=42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 5, 100)
        metrics = model.evaluate(X, y)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_set_get_params(self):
        model = MLPClassifier(input_dim=10, hidden_dim=16, num_classes=2, seed=42)
        params = model.get_parameters()
        model2 = MLPClassifier(input_dim=10, hidden_dim=16, num_classes=2, seed=99)
        model2.set_parameters(params)
        for a, b in zip(model.get_parameters(), model2.get_parameters()):
            np.testing.assert_array_equal(a, b)

    def test_param_shapes(self):
        model = MLPClassifier(input_dim=20, hidden_dim=64, num_classes=3)
        shapes = model.get_param_shapes()
        assert shapes[0] == (20, 64)  # W1
        assert shapes[1] == (64,)      # b1
        assert shapes[2] == (64, 3)    # W2
        assert shapes[3] == (3,)       # b2


class TestAnomalyAutoencoder:
    def test_creation(self):
        model = AnomalyAutoencoder(input_dim=20, encoding_dim=8, seed=42)
        assert len(model.get_parameters()) == 4

    def test_encode_decode(self):
        model = AnomalyAutoencoder(input_dim=20, encoding_dim=8, seed=42)
        X = np.random.randn(50, 20).astype(np.float32)
        z = model.encode(X)
        assert z.shape == (50, 8)
        X_hat = model.decode(z)
        assert X_hat.shape == (50, 20)

    def test_reconstruct(self):
        model = AnomalyAutoencoder(input_dim=20, encoding_dim=8, seed=42)
        X = np.random.randn(50, 20).astype(np.float32)
        X_hat = model.reconstruct(X)
        assert X_hat.shape == X.shape

    def test_reconstruction_error(self):
        model = AnomalyAutoencoder(input_dim=20, encoding_dim=8, seed=42)
        X = np.random.randn(50, 20).astype(np.float32)
        errors = model.reconstruction_error(X)
        assert errors.shape == (50,)
        assert all(e >= 0 for e in errors)

    def test_fit_threshold(self):
        model = AnomalyAutoencoder(input_dim=20, encoding_dim=8, seed=42)
        X_normal = np.random.randn(100, 20).astype(np.float32)
        threshold = model.fit_threshold(X_normal)
        assert threshold > 0
        assert model.anomaly_threshold == threshold

    def test_predict(self):
        model = AnomalyAutoencoder(input_dim=20, encoding_dim=8, seed=42)
        X_normal = np.random.randn(100, 20).astype(np.float32)
        model.fit_threshold(X_normal)
        preds = model.predict(X_normal)
        assert preds.shape == (100,)
        assert all(p in [0, 1] for p in preds)

    def test_predict_without_threshold(self):
        model = AnomalyAutoencoder(input_dim=20, encoding_dim=8)
        with pytest.raises(RuntimeError, match="fit_threshold"):
            model.predict(np.random.randn(10, 20).astype(np.float32))

    def test_evaluate(self):
        model = AnomalyAutoencoder(input_dim=20, encoding_dim=8, seed=42)
        X = np.random.randn(50, 20).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        model.fit_threshold(X)
        metrics = model.evaluate(X, y)
        assert "mse" in metrics
        assert "accuracy" in metrics
        assert "f1" in metrics


class TestFlowGNN:
    def test_creation(self):
        gnn = FlowGNN(node_feature_dim=10, hidden_dim=32, output_dim=3, num_layers=2, seed=42)
        assert len(gnn.get_parameters()) == 8  # 4 params per layer * 2 layers

    def test_forward(self):
        gnn = FlowGNN(node_feature_dim=10, hidden_dim=32, output_dim=3, seed=42)
        nodes = np.random.randn(20, 10).astype(np.float32)
        adj = (np.random.rand(20, 20) > 0.7).astype(np.float32)
        adj = (adj + adj.T) / 2  # Make symmetric
        output = gnn.forward(nodes, adj)
        assert output.shape == (20, 3)

    def test_predict_graph(self):
        gnn = FlowGNN(node_feature_dim=10, output_dim=2, seed=42)
        nodes = np.random.randn(15, 10).astype(np.float32)
        adj = (np.random.rand(15, 15) > 0.5).astype(np.float32)
        preds = gnn.predict_graph(nodes, adj)
        assert preds.shape == (15,)
        assert all(0 <= p < 2 for p in preds)

    def test_graph_embedding(self):
        gnn = FlowGNN(node_feature_dim=10, output_dim=5, seed=42)
        nodes = np.random.randn(10, 10).astype(np.float32)
        adj = np.eye(10, dtype=np.float32)
        emb = gnn.graph_embedding(nodes, adj)
        assert emb.shape == (5,)

    def test_evaluate(self):
        gnn = FlowGNN(node_feature_dim=10, output_dim=3, seed=42)
        nodes = np.random.randn(20, 10).astype(np.float32)
        adj = np.eye(20, dtype=np.float32)
        labels = np.random.randint(0, 3, 20)
        metrics = gnn.evaluate(nodes, adj, labels)
        assert "accuracy" in metrics
        assert "num_nodes" in metrics
