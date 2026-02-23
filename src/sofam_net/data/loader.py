"""
IDS Dataset Loader.

Provides a unified interface for loading intrusion detection datasets
including CICIDS2017 (CSV), NSL-KDD, and synthetic data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sofam_net.data.synthetic import SyntheticDataset, SyntheticIDSGenerator

logger = logging.getLogger(__name__)


class IDSDataLoader:
    """
    Unified IDS dataset loader.

    Supports:
    - Synthetic data generation (no external files needed)
    - CSV file loading (CICIDS2017 format)
    - NumPy array loading
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def load_synthetic(
        self,
        num_samples: int = 10000,
        num_features: int = 76,
        num_classes: int = 5,
        attack_ratio: float = 0.3,
    ) -> SyntheticDataset:
        """Generate a synthetic IDS dataset."""
        gen = SyntheticIDSGenerator(
            num_features=num_features,
            num_classes=num_classes,
            seed=self.seed,
        )
        return gen.generate(num_samples=num_samples, attack_ratio=attack_ratio)

    def load_csv(
        self,
        filepath: str | Path,
        label_column: str = "Label",
        feature_columns: list[str] | None = None,
        max_samples: int | None = None,
    ) -> SyntheticDataset:
        """
        Load a CICIDS2017-style CSV dataset.

        Expects a CSV with numeric features and a label column.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        # Simple CSV parsing (no pandas dependency)
        with open(filepath) as f:
            header = f.readline().strip().split(",")
            header = [h.strip().strip('"') for h in header]

            if label_column not in header:
                raise ValueError(f"Label column '{label_column}' not found in: {header}")

            label_idx = header.index(label_column)
            if feature_columns:
                feat_idx = [header.index(c) for c in feature_columns]
            else:
                feat_idx = [i for i in range(len(header)) if i != label_idx]

            rows_X = []
            rows_y = []
            label_map: dict[str, int] = {}

            for line_num, line in enumerate(f):
                if max_samples and line_num >= max_samples:
                    break
                parts = line.strip().split(",")
                if len(parts) != len(header):
                    continue

                try:
                    features = [float(parts[i]) for i in feat_idx]
                except (ValueError, IndexError):
                    continue

                label_str = parts[label_idx].strip().strip('"')
                if label_str not in label_map:
                    label_map[label_str] = len(label_map)

                rows_X.append(features)
                rows_y.append(label_map[label_str])

        X = np.array(rows_X, dtype=np.float32)
        y = np.array(rows_y, dtype=np.int64)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        feature_names = [header[i] for i in feat_idx]
        attack_names = {v: k for k, v in label_map.items()}

        logger.info("Loaded CSV: %d samples, %d features, %d classes", len(X), X.shape[1], len(label_map))

        return SyntheticDataset(
            X=X, y=y,
            feature_names=feature_names,
            attack_names=attack_names,
            num_classes=len(label_map),
        )

    def load_numpy(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None
    ) -> SyntheticDataset:
        """Wrap NumPy arrays as a dataset."""
        num_classes = int(np.max(y) + 1)
        return SyntheticDataset(
            X=X.astype(np.float32),
            y=y.astype(np.int64),
            feature_names=feature_names or [f"feature_{i}" for i in range(X.shape[1])],
            attack_names={i: f"class_{i}" for i in range(num_classes)},
            num_classes=num_classes,
        )

    def train_test_split(
        self, dataset: SyntheticDataset, test_ratio: float = 0.2
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Split dataset into train and test sets."""
        rng = np.random.default_rng(self.seed)
        n = len(dataset.X)
        idx = rng.permutation(n)
        split = int(n * (1 - test_ratio))
        train_idx, test_idx = idx[:split], idx[split:]
        return (dataset.X[train_idx], dataset.y[train_idx]), (dataset.X[test_idx], dataset.y[test_idx])
