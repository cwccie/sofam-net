"""Tests for data loading, splitting, and synthetic generation."""

import numpy as np
import pytest

from sofam_net.data.loader import IDSDataLoader
from sofam_net.data.splitter import DirichletSplitter
from sofam_net.data.synthetic import SyntheticIDSGenerator


class TestSyntheticIDSGenerator:
    def test_generation(self):
        gen = SyntheticIDSGenerator(num_features=20, num_classes=5, seed=42)
        dataset = gen.generate(num_samples=1000)
        assert dataset.X.shape == (1000, 20)
        assert dataset.y.shape == (1000,)
        assert dataset.num_classes == 5

    def test_feature_names(self):
        gen = SyntheticIDSGenerator(num_features=10, seed=42)
        dataset = gen.generate(num_samples=100)
        assert len(dataset.feature_names) == 10

    def test_attack_ratio(self):
        gen = SyntheticIDSGenerator(num_classes=2, seed=42)
        dataset = gen.generate(num_samples=10000, attack_ratio=0.5)
        benign_count = np.sum(dataset.y == 0)
        # Should be roughly 50% benign (with some variance)
        assert 3000 < benign_count < 7000

    def test_custom_distribution(self):
        gen = SyntheticIDSGenerator(num_classes=3, seed=42)
        dist = [0.7, 0.2, 0.1]
        dataset = gen.generate(num_samples=10000, class_distribution=dist)
        class_counts = [np.sum(dataset.y == c) for c in range(3)]
        # Class 0 should have most samples
        assert class_counts[0] > class_counts[1] > class_counts[2]

    def test_normalized_range(self):
        gen = SyntheticIDSGenerator(seed=42)
        dataset = gen.generate(num_samples=500)
        assert dataset.X.min() >= -0.01
        assert dataset.X.max() <= 1.01

    def test_reproducibility(self):
        gen1 = SyntheticIDSGenerator(seed=42)
        gen2 = SyntheticIDSGenerator(seed=42)
        d1 = gen1.generate(100)
        d2 = gen2.generate(100)
        np.testing.assert_array_equal(d1.X, d2.X)
        np.testing.assert_array_equal(d1.y, d2.y)


class TestDirichletSplitter:
    def test_basic_split(self):
        X = np.random.randn(1000, 10).astype(np.float32)
        y = np.random.randint(0, 5, 1000).astype(np.int64)
        splitter = DirichletSplitter(alpha=1.0, seed=42)
        split = splitter.split(X, y, num_orgs=3)
        assert len(split.org_data) == 3
        total = sum(split.org_sizes.values())
        assert total == 1000

    def test_named_orgs(self):
        X = np.random.randn(500, 10).astype(np.float32)
        y = np.random.randint(0, 3, 500)
        splitter = DirichletSplitter(alpha=0.5, seed=42)
        split = splitter.split(X, y, 3, org_names=["Alpha", "Beta", "Gamma"])
        assert "Alpha" in split.org_data
        assert "Beta" in split.org_data
        assert "Gamma" in split.org_data

    def test_low_alpha_heterogeneity(self):
        X = np.random.randn(2000, 10).astype(np.float32)
        y = np.random.randint(0, 5, 2000)
        splitter = DirichletSplitter(alpha=0.01, seed=42)
        split = splitter.split(X, y, 5)
        # With very low alpha, each org should have skewed class distribution
        for _org, dist in split.class_distributions.items():
            pass  # distributions checked below
        max_ratios = []
        for dist in split.class_distributions.values():
            if dist:
                max_ratios.append(max(dist.values()) / sum(dist.values()))
        assert any(r > 0.3 for r in max_ratios)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            DirichletSplitter(alpha=-1.0)

    def test_class_distributions_populated(self):
        X = np.random.randn(500, 5).astype(np.float32)
        y = np.random.randint(0, 3, 500)
        splitter = DirichletSplitter(alpha=1.0, seed=42)
        split = splitter.split(X, y, 3)
        for name, dist in split.class_distributions.items():
            assert isinstance(dist, dict)


class TestIDSDataLoader:
    def test_load_synthetic(self):
        loader = IDSDataLoader(seed=42)
        dataset = loader.load_synthetic(num_samples=500, num_features=10, num_classes=3)
        assert dataset.X.shape == (500, 10)
        assert dataset.num_classes == 3

    def test_load_numpy(self):
        loader = IDSDataLoader()
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        dataset = loader.load_numpy(X, y)
        assert dataset.num_classes == 2
        assert len(dataset.feature_names) == 5

    def test_train_test_split(self):
        loader = IDSDataLoader(seed=42)
        dataset = loader.load_synthetic(num_samples=1000, num_features=10)
        (X_train, y_train), (X_test, y_test) = loader.train_test_split(dataset, test_ratio=0.2)
        assert len(X_train) == 800
        assert len(X_test) == 200
        assert X_train.shape[1] == 10
