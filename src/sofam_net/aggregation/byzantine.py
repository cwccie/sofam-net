"""
Byzantine-Robust Aggregation.

Implements robust aggregation methods that tolerate malicious or faulty
clients submitting corrupted model updates.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from sofam_net.coordinator.strategy import AggregationResult, ClientUpdate

logger = logging.getLogger(__name__)


class RobustAggregator(ABC):
    """Base class for Byzantine-robust aggregation methods."""

    @abstractmethod
    def aggregate(self, updates: list[ClientUpdate]) -> AggregationResult:
        """Aggregate updates with Byzantine fault tolerance."""


class KrumAggregator(RobustAggregator):
    """
    Krum aggregation (Blanchard et al., 2017).

    Selects the update that is closest to its neighbors, effectively
    filtering out outlier (potentially Byzantine) updates.

    Multi-Krum variant averages the top-m closest updates.
    """

    def __init__(self, num_byzantine: int = 1, multi_krum_m: int = 1):
        """
        Args:
            num_byzantine: Maximum number of Byzantine clients to tolerate.
            multi_krum_m: Number of top updates to average (1 = standard Krum).
        """
        self.num_byzantine = num_byzantine
        self.multi_krum_m = multi_krum_m

    def aggregate(self, updates: list[ClientUpdate]) -> AggregationResult:
        n = len(updates)
        f = self.num_byzantine
        if n < 2 * f + 3:
            raise ValueError(
                f"Krum requires n >= 2f + 3. Got n={n}, f={f} (need {2*f+3})"
            )

        # Flatten parameters for distance computation
        flat_updates = []
        for u in updates:
            flat = np.concatenate([p.ravel() for p in u.parameters])
            flat_updates.append(flat)
        flat_updates_arr = np.array(flat_updates)

        # Compute pairwise squared distances
        n_neighbors = n - f - 2
        scores = np.zeros(n)

        for i in range(n):
            dists = np.sum((flat_updates_arr - flat_updates_arr[i]) ** 2, axis=1)
            dists[i] = np.inf  # Exclude self
            # Sum of distances to n-f-2 closest neighbors
            closest = np.sort(dists)[:n_neighbors]
            scores[i] = np.sum(closest)

        # Select top-m updates with lowest scores
        m = min(self.multi_krum_m, n)
        selected_idx = np.argsort(scores)[:m]
        selected_updates = [updates[i] for i in selected_idx]

        logger.info(
            "Krum selected %d/%d updates (f=%d). Scores: min=%.2f, max=%.2f",
            m, n, f, scores[selected_idx[0]], scores[selected_idx[-1]],
        )

        # Average selected updates
        total_samples = sum(u.num_samples for u in selected_updates)
        aggregated = []
        for layer_idx in range(len(updates[0].parameters)):
            weighted_sum = np.zeros_like(updates[0].parameters[layer_idx], dtype=np.float64)
            for u in selected_updates:
                weight = u.num_samples / total_samples
                weighted_sum += weight * u.parameters[layer_idx].astype(np.float64)
            aggregated.append(weighted_sum.astype(updates[0].parameters[layer_idx].dtype))

        return AggregationResult(
            parameters=aggregated,
            num_clients=len(selected_updates),
            total_samples=total_samples,
            metrics={"krum_score_min": float(scores[selected_idx[0]])},
        )


class TrimmedMeanAggregator(RobustAggregator):
    """
    Coordinate-wise Trimmed Mean (Yin et al., 2018).

    For each parameter coordinate, sorts the values across clients
    and removes the top and bottom beta fraction before averaging.
    """

    def __init__(self, beta: float = 0.1):
        """
        Args:
            beta: Fraction of extreme values to trim (0 to 0.5).
        """
        if not 0 <= beta < 0.5:
            raise ValueError("beta must be in [0, 0.5)")
        self.beta = beta

    def aggregate(self, updates: list[ClientUpdate]) -> AggregationResult:
        n = len(updates)
        trim_count = max(1, int(n * self.beta))

        if n <= 2 * trim_count:
            raise ValueError(
                f"Not enough clients ({n}) for trimming {trim_count} from each side"
            )

        total_samples = sum(u.num_samples for u in updates)
        aggregated = []

        for layer_idx in range(len(updates[0].parameters)):
            # Stack all client values for this layer
            stacked = np.stack([u.parameters[layer_idx] for u in updates], axis=0)

            # Sort along client axis
            sorted_vals = np.sort(stacked, axis=0)

            # Trim top and bottom
            trimmed = sorted_vals[trim_count : n - trim_count]

            # Average remaining
            mean_val = np.mean(trimmed, axis=0)
            aggregated.append(mean_val.astype(updates[0].parameters[layer_idx].dtype))

        logger.info(
            "Trimmed mean: %d clients, trimmed %d from each side (beta=%.2f)",
            n, trim_count, self.beta,
        )

        return AggregationResult(
            parameters=aggregated,
            num_clients=n - 2 * trim_count,
            total_samples=total_samples,
            metrics={"trim_count": trim_count},
        )
