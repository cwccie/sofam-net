"""Tests for secure and Byzantine-robust aggregation."""

import numpy as np
import pytest

from sofam_net.aggregation.byzantine import KrumAggregator, TrimmedMeanAggregator
from sofam_net.aggregation.secure import MaskedUpdate, SecureAggregator
from sofam_net.coordinator.strategy import ClientUpdate


class TestSecureAggregator:
    def test_mask_generation(self):
        agg = SecureAggregator(seed=42)
        client_ids = ["c1", "c2", "c3"]
        shapes = [(10, 5), (5,)]
        masks = agg.generate_masks(client_ids, shapes)
        assert len(masks) == 3
        # Masks should sum to zero
        for layer_idx in range(len(shapes)):
            total = sum(masks[cid][layer_idx] for cid in client_ids)
            np.testing.assert_allclose(total, 0, atol=1e-10)

    def test_mask_and_aggregate(self, sample_params):
        agg = SecureAggregator(threshold=2, seed=42)
        client_ids = ["c1", "c2"]
        shapes = [p.shape for p in sample_params]
        masks = agg.generate_masks(client_ids, shapes)

        # Create masked updates
        updates = []
        for cid in client_ids:
            mu = agg.mask_update(cid, sample_params, masks[cid], 100)
            updates.append(mu)

        result = agg.aggregate(updates)
        assert result.num_clients == 2
        assert result.verified

        # Result should be close to original (since all clients have same params)
        for orig, agg_p in zip(sample_params, result.parameters):
            np.testing.assert_allclose(agg_p, orig, atol=1e-5)

    def test_insufficient_clients(self):
        agg = SecureAggregator(threshold=3)
        update = MaskedUpdate("c1", [np.zeros(10)], ["abc"], 100)
        with pytest.raises(ValueError, match="at least 3"):
            agg.aggregate([update, update])

    def test_mask_commitments(self, sample_params):
        agg = SecureAggregator(seed=42)
        masks = [np.random.randn(*p.shape) for p in sample_params]
        mu = agg.mask_update("c1", sample_params, masks, 100)
        assert len(mu.mask_commitments) == len(sample_params)
        assert all(len(c) == 64 for c in mu.mask_commitments)  # SHA-256 hex


class TestKrumAggregator:
    def test_basic(self, sample_params):
        krum = KrumAggregator(num_byzantine=0)
        updates = [
            ClientUpdate(f"c{i}", [p + np.random.randn(*p.shape) * 0.01 for p in sample_params], 100)
            for i in range(5)
        ]
        result = krum.aggregate(updates)
        assert result.num_clients >= 1

    def test_filters_outlier(self, sample_params):
        krum = KrumAggregator(num_byzantine=1)
        # Normal updates
        updates = [
            ClientUpdate(f"c{i}", sample_params, 100)
            for i in range(5)
        ]
        # Add Byzantine update (large perturbation)
        byzantine_params = [p + 1000 for p in sample_params]
        updates.append(ClientUpdate("byzantine", byzantine_params, 100))

        result = krum.aggregate(updates)
        # Krum should select a normal update, not the Byzantine one
        for orig, agg in zip(sample_params, result.parameters):
            # Should be close to normal params
            assert np.mean(np.abs(agg - orig)) < 100

    def test_multi_krum(self, sample_params):
        krum = KrumAggregator(num_byzantine=0, multi_krum_m=3)
        updates = [
            ClientUpdate(f"c{i}", sample_params, 100)
            for i in range(5)
        ]
        result = krum.aggregate(updates)
        assert result.num_clients == 3

    def test_insufficient_clients(self, sample_params):
        krum = KrumAggregator(num_byzantine=2)
        updates = [ClientUpdate(f"c{i}", sample_params, 100) for i in range(3)]
        with pytest.raises(ValueError, match="Krum requires"):
            krum.aggregate(updates)


class TestTrimmedMeanAggregator:
    def test_basic(self, sample_params):
        tm = TrimmedMeanAggregator(beta=0.1)
        updates = [
            ClientUpdate(f"c{i}", [p + np.random.randn(*p.shape) * 0.01 for p in sample_params], 100)
            for i in range(10)
        ]
        result = tm.aggregate(updates)
        assert result.num_clients > 0

    def test_filters_extreme_values(self, sample_params):
        tm = TrimmedMeanAggregator(beta=0.2)
        # 8 normal updates
        updates = [ClientUpdate(f"c{i}", sample_params, 100) for i in range(8)]
        # 2 extreme updates
        updates.append(ClientUpdate("bad1", [p + 1000 for p in sample_params], 100))
        updates.append(ClientUpdate("bad2", [p - 1000 for p in sample_params], 100))

        result = tm.aggregate(updates)
        for orig, agg in zip(sample_params, result.parameters):
            np.testing.assert_allclose(agg, orig, atol=0.5)

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="beta must be"):
            TrimmedMeanAggregator(beta=0.5)
