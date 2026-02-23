"""
Privacy Accounting — RDP accountant and per-organization budget tracking.

Implements Rényi Differential Privacy (RDP) accounting for tight composition
of privacy guarantees across multiple FL rounds.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Per-organization privacy budget tracker."""

    org_id: str
    total_epsilon: float
    total_delta: float = 1e-5
    spent_epsilon: float = 0.0
    rounds_participated: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self.spent_epsilon)

    @property
    def fraction_spent(self) -> float:
        if self.total_epsilon <= 0:
            return 1.0
        return self.spent_epsilon / self.total_epsilon

    @property
    def is_exhausted(self) -> bool:
        return self.spent_epsilon >= self.total_epsilon

    def record_spend(self, epsilon: float, round_num: int) -> None:
        """Record epsilon spent in a round."""
        self.spent_epsilon += epsilon
        self.rounds_participated += 1
        self.history.append({
            "round": round_num,
            "epsilon_spent": epsilon,
            "cumulative_epsilon": self.spent_epsilon,
        })
        logger.info(
            "Org %s: spent eps=%.4f (cumulative=%.4f/%.4f, %.1f%% used)",
            self.org_id, epsilon, self.spent_epsilon, self.total_epsilon,
            self.fraction_spent * 100,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "spent_epsilon": self.spent_epsilon,
            "remaining_epsilon": self.remaining_epsilon,
            "fraction_spent": self.fraction_spent,
            "rounds_participated": self.rounds_participated,
            "is_exhausted": self.is_exhausted,
        }


class RDPAccountant:
    """
    Rényi Differential Privacy (RDP) Accountant.

    Tracks cumulative privacy loss across FL rounds using RDP composition.
    Provides tighter bounds than basic composition theorem.

    Based on: Mironov (2017), "Rényi Differential Privacy".
    """

    # Standard RDP orders for accounting
    DEFAULT_ORDERS = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 128, 256, 512, 1024]

    def __init__(
        self,
        orders: list[float] | None = None,
        target_delta: float = 1e-5,
    ):
        self.orders = orders or self.DEFAULT_ORDERS
        self.target_delta = target_delta
        self._rdp_history: list[list[float]] = []  # RDP values per step
        self.budgets: dict[str, PrivacyBudget] = {}

    def create_budget(
        self, org_id: str, total_epsilon: float, total_delta: float | None = None
    ) -> PrivacyBudget:
        """Create a privacy budget for an organization."""
        budget = PrivacyBudget(
            org_id=org_id,
            total_epsilon=total_epsilon,
            total_delta=total_delta or self.target_delta,
        )
        self.budgets[org_id] = budget
        return budget

    def compute_rdp_gaussian(self, sigma: float, sampling_rate: float = 1.0) -> list[float]:
        """
        Compute RDP of the Sampled Gaussian Mechanism.

        For a Gaussian mechanism with noise multiplier sigma and subsampling rate q,
        this computes the RDP at each order alpha.
        """
        rdp_values = []
        for alpha in self.orders:
            if sampling_rate >= 1.0:
                # No subsampling: RDP = alpha / (2 * sigma^2)
                rdp = alpha / (2.0 * sigma**2)
            else:
                # Subsampled Gaussian RDP (simplified bound)
                rdp = self._compute_subsampled_rdp(alpha, sigma, sampling_rate)
            rdp_values.append(rdp)
        return rdp_values

    def _compute_subsampled_rdp(
        self, alpha: float, sigma: float, sampling_rate: float
    ) -> float:
        """
        Compute RDP for subsampled Gaussian mechanism.

        Uses the analytical bound from Mironov et al.
        """
        if alpha <= 1:
            return 0.0

        q = sampling_rate
        # Upper bound via log-sum-exp
        # RDP <= (1/(alpha-1)) * log( (1-q)^alpha * (1 + q*(exp((alpha-1)/sigma^2) - 1)) )
        log_term1 = alpha * math.log(1 - q)
        exp_arg = (alpha - 1) / (2 * sigma**2)

        if exp_arg > 500:
            # Overflow protection
            return alpha / (2 * sigma**2)

        inner = 1 + q * (math.exp(exp_arg) - 1)
        if inner <= 0:
            return float("inf")

        rdp = (log_term1 + math.log(inner)) / (alpha - 1)
        return max(0.0, rdp)

    def step(self, sigma: float, sampling_rate: float = 1.0) -> float:
        """
        Record one step of the mechanism and return current (epsilon, delta) guarantee.

        Returns the epsilon at the target delta.
        """
        rdp_step = self.compute_rdp_gaussian(sigma, sampling_rate)
        self._rdp_history.append(rdp_step)
        eps = self.get_epsilon()
        return eps

    def get_epsilon(self, delta: float | None = None) -> float:
        """
        Convert accumulated RDP to (epsilon, delta)-DP.

        Uses the conversion: eps = min_alpha { RDP_alpha - log(delta) / (alpha - 1) }
        """
        delta = delta or self.target_delta
        if not self._rdp_history:
            return 0.0

        # Sum RDP across all steps (composition)
        total_rdp = [0.0] * len(self.orders)
        for rdp_step in self._rdp_history:
            for i, rdp_val in enumerate(rdp_step):
                total_rdp[i] += rdp_val

        # Convert to (epsilon, delta)-DP
        best_epsilon = float("inf")
        for alpha, rdp in zip(self.orders, total_rdp):
            if alpha <= 1:
                continue
            eps = rdp - math.log(delta) / (alpha - 1)
            best_epsilon = min(best_epsilon, eps)

        return max(0.0, best_epsilon)

    def can_participate(self, org_id: str, estimated_epsilon: float) -> bool:
        """Check if an organization has enough privacy budget for another round."""
        budget = self.budgets.get(org_id)
        if budget is None:
            return True  # No budget constraint
        return budget.remaining_epsilon >= estimated_epsilon

    def record_participation(
        self, org_id: str, epsilon_spent: float, round_num: int
    ) -> None:
        """Record that an organization participated in a round."""
        budget = self.budgets.get(org_id)
        if budget is not None:
            budget.record_spend(epsilon_spent, round_num)

    def get_all_budgets(self) -> dict[str, dict[str, Any]]:
        """Return all organization privacy budgets."""
        return {org_id: b.to_dict() for org_id, b in self.budgets.items()}

    def reset(self) -> None:
        """Reset the accountant (clear history but keep budgets)."""
        self._rdp_history.clear()
