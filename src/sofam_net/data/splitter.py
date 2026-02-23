"""
Non-IID Data Splitting for Federated Learning.

Implements Dirichlet allocation to simulate realistic heterogeneous
data distributions across organizations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FederatedSplit:
    """Result of splitting data across organizations."""

    org_data: dict[str, tuple[np.ndarray, np.ndarray]]  # org_id -> (X, y)
    org_sizes: dict[str, int]
    class_distributions: dict[str, dict[int, int]]


class DirichletSplitter:
    """
    Non-IID data splitting using Dirichlet allocation.

    Lower alpha → more heterogeneous (skewed) distributions.
    Higher alpha → more homogeneous (IID-like) distributions.

    alpha=0.1: highly non-IID (each org sees mostly 1-2 classes)
    alpha=1.0: moderately non-IID
    alpha=100: approximately IID
    """

    def __init__(self, alpha: float = 0.5, seed: int | None = None):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_orgs: int,
        org_names: list[str] | None = None,
        min_samples_per_org: int = 10,
    ) -> FederatedSplit:
        """
        Split dataset across organizations using Dirichlet allocation.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            num_orgs: Number of organizations
            org_names: Optional names for organizations
            min_samples_per_org: Minimum samples per organization
        """
        if org_names is None:
            org_names = [f"org_{i}" for i in range(num_orgs)]

        n_samples = len(X)
        classes = np.unique(y)

        # Dirichlet allocation: for each class, determine fraction going to each org
        org_indices: dict[str, list[int]] = {name: [] for name in org_names}

        for c in classes:
            class_idx = np.where(y == c)[0]
            # Draw Dirichlet proportions
            proportions = self.rng.dirichlet(np.full(num_orgs, self.alpha))

            # Assign indices based on proportions
            proportions = proportions / proportions.sum()
            splits = (proportions * len(class_idx)).astype(int)
            # Handle rounding: give remainder to random orgs
            remainder = len(class_idx) - splits.sum()
            for _ in range(remainder):
                splits[self.rng.integers(num_orgs)] += 1

            self.rng.shuffle(class_idx)
            start = 0
            for org_idx, count in enumerate(splits):
                org_indices[org_names[org_idx]].extend(
                    class_idx[start:start + count].tolist()
                )
                start += count

        # Ensure minimum samples
        for name in org_names:
            if len(org_indices[name]) < min_samples_per_org:
                # Borrow samples from the largest org
                largest = max(org_names, key=lambda n: len(org_indices[n]))
                needed = min_samples_per_org - len(org_indices[name])
                if len(org_indices[largest]) > min_samples_per_org + needed:
                    borrowed = org_indices[largest][:needed]
                    org_indices[largest] = org_indices[largest][needed:]
                    org_indices[name].extend(borrowed)

        # Build result
        org_data = {}
        org_sizes = {}
        class_dists: dict[str, dict[int, int]] = {}

        for name in org_names:
            idx = np.array(org_indices[name])
            if len(idx) == 0:
                org_data[name] = (np.empty((0, X.shape[1])), np.empty(0))
                org_sizes[name] = 0
                class_dists[name] = {}
                continue

            self.rng.shuffle(idx)
            org_X = X[idx]
            org_y = y[idx]
            org_data[name] = (org_X, org_y)
            org_sizes[name] = len(idx)

            dist = {}
            for c in classes:
                count = int(np.sum(org_y == c))
                if count > 0:
                    dist[int(c)] = count
            class_dists[name] = dist

        logger.info(
            "Dirichlet split (alpha=%.2f): %d samples → %d orgs. Sizes: %s",
            self.alpha, n_samples, num_orgs,
            {n: s for n, s in org_sizes.items()},
        )

        return FederatedSplit(
            org_data=org_data,
            org_sizes=org_sizes,
            class_distributions=class_dists,
        )
