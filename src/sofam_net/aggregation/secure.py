"""
Secure Aggregation Protocol.

Implements a simplified secure aggregation where clients mask their
model updates with pairwise random masks. The coordinator can only
recover the aggregate — never individual updates.

Based on: Bonawitz et al. (2017), "Practical Secure Aggregation".
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MaskedUpdate:
    """A client's masked model update."""

    client_id: str
    masked_parameters: list[np.ndarray]
    mask_commitments: list[str]  # SHA-256 commitments for verification
    num_samples: int


@dataclass
class SecureAggregationResult:
    """Result of secure aggregation."""

    parameters: list[np.ndarray]
    num_clients: int
    verified: bool
    dropped_clients: list[str] = field(default_factory=list)


class SecureAggregator:
    """
    Secure Aggregation with pairwise masking.

    Protocol:
    1. Each pair of clients (i, j) agrees on a shared random seed
    2. Client i adds mask_ij to their update, client j subtracts mask_ij
    3. When summed, masks cancel out, revealing only the aggregate

    This implementation simulates the protocol for testing/demonstration.
    In production, use proper key agreement (Diffie-Hellman).
    """

    def __init__(
        self,
        threshold: int = 2,
        seed: int | None = None,
    ):
        """
        Args:
            threshold: Minimum clients needed to reconstruct the aggregate.
            seed: Random seed for reproducibility.
        """
        self.threshold = threshold
        self._rng = np.random.default_rng(seed)
        self._shared_seeds: dict[tuple[str, str], int] = {}

    def generate_masks(
        self, client_ids: list[str], param_shapes: list[tuple[int, ...]]
    ) -> dict[str, list[np.ndarray]]:
        """
        Generate pairwise masks for all clients.

        Returns a dict mapping client_id -> list of mask arrays.
        Masks are constructed so they sum to zero across all clients.
        """
        n = len(client_ids)
        masks: dict[str, list[np.ndarray]] = {
            cid: [np.zeros(shape) for shape in param_shapes] for cid in client_ids
        }

        # Generate pairwise masks
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = client_ids[i], client_ids[j]
                seed = self._get_shared_seed(ci, cj)
                pair_rng = np.random.default_rng(seed)

                for layer_idx, shape in enumerate(param_shapes):
                    mask = pair_rng.normal(0, 1, size=shape)
                    masks[ci][layer_idx] += mask
                    masks[cj][layer_idx] -= mask  # Cancels when summed

        return masks

    def mask_update(
        self,
        client_id: str,
        parameters: list[np.ndarray],
        masks: list[np.ndarray],
        num_samples: int,
    ) -> MaskedUpdate:
        """Apply masks to a client's model update."""
        masked = [p + m for p, m in zip(parameters, masks)]

        # Generate commitments (hash of each mask for verification)
        commitments = []
        for m in masks:
            h = hashlib.sha256(m.tobytes()).hexdigest()
            commitments.append(h)

        return MaskedUpdate(
            client_id=client_id,
            masked_parameters=masked,
            mask_commitments=commitments,
            num_samples=num_samples,
        )

    def aggregate(
        self,
        masked_updates: list[MaskedUpdate],
        all_client_ids: list[str] | None = None,
    ) -> SecureAggregationResult:
        """
        Aggregate masked updates. Masks cancel out in the sum.

        If some clients dropped, their masks need to be reconstructed
        from the surviving clients' shared seeds.
        """
        if len(masked_updates) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} updates, got {len(masked_updates)}"
            )

        participating_ids = {u.client_id for u in masked_updates}
        dropped = []
        if all_client_ids is not None:
            dropped = [c for c in all_client_ids if c not in participating_ids]

        total_samples = sum(u.num_samples for u in masked_updates)

        # Sum masked parameters (masks cancel for participating pairs)
        aggregated = [
            np.zeros_like(masked_updates[0].masked_parameters[i])
            for i in range(len(masked_updates[0].masked_parameters))
        ]

        for update in masked_updates:
            weight = update.num_samples / total_samples
            for i, mp in enumerate(update.masked_parameters):
                aggregated[i] += weight * mp

        # Compensate for dropped clients' masks
        if dropped:
            logger.warning("Compensating for %d dropped clients: %s", len(dropped), dropped)
            aggregated = self._compensate_dropped(
                aggregated, list(participating_ids), dropped,
                [p.shape for p in masked_updates[0].masked_parameters],
            )

        logger.info(
            "Secure aggregation complete: %d clients, %d dropped",
            len(masked_updates), len(dropped),
        )

        return SecureAggregationResult(
            parameters=aggregated,
            num_clients=len(masked_updates),
            verified=len(dropped) == 0,
            dropped_clients=dropped,
        )

    def _compensate_dropped(
        self,
        aggregated: list[np.ndarray],
        surviving: list[str],
        dropped: list[str],
        param_shapes: list[tuple[int, ...]],
    ) -> list[np.ndarray]:
        """Reconstruct and remove masks of dropped clients."""
        for dropped_id in dropped:
            for surviving_id in surviving:
                seed = self._get_shared_seed(dropped_id, surviving_id)
                pair_rng = np.random.default_rng(seed)

                for layer_idx, shape in enumerate(param_shapes):
                    mask = pair_rng.normal(0, 1, size=shape)
                    # Determine mask direction based on ID ordering
                    if dropped_id < surviving_id:
                        aggregated[layer_idx] -= mask  # Remove dropped client's mask
                    else:
                        aggregated[layer_idx] += mask

        return aggregated

    def _get_shared_seed(self, id_a: str, id_b: str) -> int:
        """Get or create a shared seed for a pair of clients."""
        key = (min(id_a, id_b), max(id_a, id_b))
        if key not in self._shared_seeds:
            # Derive seed from client IDs (simulating key agreement)
            combined = f"{key[0]}:{key[1]}".encode()
            h = hashlib.sha256(combined).digest()
            self._shared_seeds[key] = int.from_bytes(h[:8], "big")
        return self._shared_seeds[key]
