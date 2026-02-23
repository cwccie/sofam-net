"""Shared test fixtures for SOFAM-Net."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_params():
    """Simple 2-layer MLP parameters for testing."""
    rng = np.random.default_rng(42)
    return [
        rng.normal(0, 0.1, (10, 32)).astype(np.float32),   # W1
        np.zeros(32, dtype=np.float32),                       # b1
        rng.normal(0, 0.1, (32, 5)).astype(np.float32),     # W2
        np.zeros(5, dtype=np.float32),                        # b2
    ]


@pytest.fixture
def sample_data():
    """Simple synthetic dataset for testing."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 10)).astype(np.float32)
    y = rng.integers(0, 5, 200).astype(np.int64)
    return X, y


@pytest.fixture
def sample_data_binary():
    """Binary classification dataset."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 10)).astype(np.float32)
    y = rng.integers(0, 2, 200).astype(np.int64)
    return X, y
