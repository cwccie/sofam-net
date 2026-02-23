"""
Synthetic IDS Data Generator.

Generates CICIDS2017-like network flow features with realistic distributions
for benchmarking federated IDS without requiring real network data.

Feature set modeled after CIC-IDS-2017:
- Flow duration, packet counts, byte counts
- Inter-arrival times, flag counts
- Flow-level statistics (mean, std, min, max)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# CICIDS2017-like feature names
FEATURE_NAMES = [
    "flow_duration", "total_fwd_packets", "total_bwd_packets",
    "total_length_fwd", "total_length_bwd", "fwd_packet_length_max",
    "fwd_packet_length_min", "fwd_packet_length_mean", "fwd_packet_length_std",
    "bwd_packet_length_max", "bwd_packet_length_min", "bwd_packet_length_mean",
    "bwd_packet_length_std", "flow_bytes_per_sec", "flow_packets_per_sec",
    "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
    "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max",
    "fwd_iat_min", "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std",
    "bwd_iat_max", "bwd_iat_min", "fwd_psh_flags", "bwd_psh_flags",
    "fwd_urg_flags", "bwd_urg_flags", "fwd_header_length", "bwd_header_length",
    "fwd_packets_per_sec", "bwd_packets_per_sec", "min_packet_length",
    "max_packet_length", "packet_length_mean", "packet_length_std",
    "packet_length_variance", "fin_flag_count", "syn_flag_count",
    "rst_flag_count", "psh_flag_count", "ack_flag_count", "urg_flag_count",
    "cwe_flag_count", "ece_flag_count", "down_up_ratio", "avg_packet_size",
    "avg_fwd_segment_size", "avg_bwd_segment_size", "fwd_avg_bytes_bulk",
    "fwd_avg_packets_bulk", "fwd_avg_bulk_rate", "bwd_avg_bytes_bulk",
    "bwd_avg_packets_bulk", "bwd_avg_bulk_rate", "subflow_fwd_packets",
    "subflow_fwd_bytes", "subflow_bwd_packets", "subflow_bwd_bytes",
    "init_win_bytes_fwd", "init_win_bytes_bwd", "act_data_pkt_fwd",
    "min_seg_size_fwd", "active_mean", "active_std", "active_max",
    "active_min", "idle_mean", "idle_std", "idle_max", "idle_min",
]

# Attack types modeled after CICIDS2017
ATTACK_TYPES = {
    0: "BENIGN",
    1: "DoS_Hulk",
    2: "PortScan",
    3: "DDoS",
    4: "DoS_GoldenEye",
    5: "FTP_Patator",
    6: "SSH_Patator",
    7: "DoS_Slowloris",
    8: "Bot",
    9: "Web_Attack_BruteForce",
    10: "Infiltration",
}


@dataclass
class SyntheticDataset:
    """Generated synthetic IDS dataset."""

    X: np.ndarray  # Features (n_samples, n_features)
    y: np.ndarray  # Labels (n_samples,)
    feature_names: list[str]
    attack_names: dict[int, str]
    num_classes: int


class SyntheticIDSGenerator:
    """
    Generate synthetic CICIDS2017-like network flow data.

    Each attack type has distinct statistical signatures in the feature space,
    allowing ML models to learn meaningful decision boundaries.
    """

    def __init__(
        self,
        num_features: int = 76,
        num_classes: int = 5,
        seed: int | None = None,
    ):
        self.num_features = min(num_features, len(FEATURE_NAMES))
        self.num_classes = min(num_classes, len(ATTACK_TYPES))
        self.rng = np.random.default_rng(seed)

        # Generate class-specific distributions
        self._class_means = self.rng.uniform(-1, 3, (self.num_classes, self.num_features))
        self._class_stds = self.rng.uniform(0.3, 1.5, (self.num_classes, self.num_features))

        # Make benign traffic more concentrated, attacks more spread
        self._class_stds[0] *= 0.5  # Benign is tighter
        for c in range(1, self.num_classes):
            # Shift attack signatures away from benign
            shift_dims = self.rng.choice(self.num_features, self.num_features // 3, replace=False)
            self._class_means[c, shift_dims] += self.rng.uniform(1, 4, len(shift_dims))

    def generate(
        self,
        num_samples: int = 10000,
        attack_ratio: float = 0.3,
        class_distribution: list[float] | None = None,
    ) -> SyntheticDataset:
        """
        Generate a synthetic IDS dataset.

        Args:
            num_samples: Total number of samples.
            attack_ratio: Fraction of samples that are attacks (if class_distribution is None).
            class_distribution: Explicit per-class probabilities (sums to 1).
        """
        if class_distribution is not None:
            probs = np.array(class_distribution[:self.num_classes])
            probs = probs / probs.sum()
        else:
            benign_prob = 1 - attack_ratio
            attack_probs = np.ones(self.num_classes - 1) * attack_ratio / (self.num_classes - 1)
            probs = np.concatenate([[benign_prob], attack_probs])

        # Assign labels
        y = self.rng.choice(self.num_classes, size=num_samples, p=probs)

        # Generate features per class
        X = np.zeros((num_samples, self.num_features), dtype=np.float32)
        for c in range(self.num_classes):
            mask = y == c
            n_c = np.sum(mask)
            if n_c > 0:
                X[mask] = self.rng.normal(
                    self._class_means[c], self._class_stds[c], (n_c, self.num_features)
                ).astype(np.float32)

        # Add realistic feature correlations
        # Packet counts correlate with byte counts
        if self.num_features >= 5:
            X[:, 3] = np.abs(X[:, 1] * self.rng.uniform(40, 1500, num_samples) + X[:, 3])
            X[:, 4] = np.abs(X[:, 2] * self.rng.uniform(40, 1500, num_samples) + X[:, 4])

        # Ensure non-negative features where appropriate
        non_neg_features = list(range(min(self.num_features, 35)))
        X[:, non_neg_features] = np.abs(X[:, non_neg_features])

        # Normalize to [0, 1] range
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X = (X - X_min) / (X_max - X_min + 1e-10)

        attack_names = {k: v for k, v in ATTACK_TYPES.items() if k < self.num_classes}
        logger.info(
            "Generated %d samples (%d features, %d classes). Class dist: %s",
            num_samples, self.num_features, self.num_classes,
            {attack_names[c]: int(np.sum(y == c)) for c in range(self.num_classes)},
        )

        return SyntheticDataset(
            X=X,
            y=y.astype(np.int64),
            feature_names=FEATURE_NAMES[:self.num_features],
            attack_names=attack_names,
            num_classes=self.num_classes,
        )
