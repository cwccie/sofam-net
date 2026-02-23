"""
Differential Privacy Noise Mechanisms.

Implements Gaussian and Laplace mechanisms for adding calibrated noise
to model updates before aggregation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoiseReport:
    """Report of noise applied to parameters."""

    mechanism: str
    noise_scale: float
    sensitivity: float
    num_parameters: int
    total_noise_norm: float


class DPMechanism(ABC):
    """Base class for differential privacy mechanisms."""

    @abstractmethod
    def add_noise(
        self,
        parameters: list[np.ndarray],
        sensitivity: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[list[np.ndarray], NoiseReport]:
        """Add calibrated noise to model parameters."""

    @abstractmethod
    def compute_noise_scale(self, sensitivity: float) -> float:
        """Compute the noise scale for the given sensitivity."""


class GaussianMechanism(DPMechanism):
    """
    Gaussian Mechanism for (epsilon, delta)-differential privacy.

    Adds Gaussian noise calibrated to the L2 sensitivity of the query.
    Noise scale sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon.
    """

    def __init__(self, epsilon: float, delta: float = 1e-5):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        self.epsilon = epsilon
        self.delta = delta

    def compute_noise_scale(self, sensitivity: float) -> float:
        """Compute Gaussian noise standard deviation."""
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def add_noise(
        self,
        parameters: list[np.ndarray],
        sensitivity: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[list[np.ndarray], NoiseReport]:
        rng = rng or np.random.default_rng()
        sigma = self.compute_noise_scale(sensitivity)

        noisy_params = []
        total_noise_norm_sq = 0.0
        total_params = 0

        for p in parameters:
            noise = rng.normal(0, sigma, size=p.shape).astype(p.dtype)
            noisy_params.append(p + noise)
            total_noise_norm_sq += float(np.sum(noise**2))
            total_params += p.size

        report = NoiseReport(
            mechanism="gaussian",
            noise_scale=sigma,
            sensitivity=sensitivity,
            num_parameters=total_params,
            total_noise_norm=float(np.sqrt(total_noise_norm_sq)),
        )
        logger.debug(
            "Gaussian noise applied: sigma=%.6f, eps=%.4f, delta=%.2e",
            sigma, self.epsilon, self.delta,
        )
        return noisy_params, report


class LaplaceMechanism(DPMechanism):
    """
    Laplace Mechanism for epsilon-differential privacy.

    Adds Laplace noise calibrated to the L1 sensitivity of the query.
    Noise scale b = sensitivity / epsilon.
    """

    def __init__(self, epsilon: float):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        self.epsilon = epsilon

    def compute_noise_scale(self, sensitivity: float) -> float:
        """Compute Laplace noise scale parameter."""
        return sensitivity / self.epsilon

    def add_noise(
        self,
        parameters: list[np.ndarray],
        sensitivity: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[list[np.ndarray], NoiseReport]:
        rng = rng or np.random.default_rng()
        b = self.compute_noise_scale(sensitivity)

        noisy_params = []
        total_noise_norm_sq = 0.0
        total_params = 0

        for p in parameters:
            noise = rng.laplace(0, b, size=p.shape).astype(p.dtype)
            noisy_params.append(p + noise)
            total_noise_norm_sq += float(np.sum(noise**2))
            total_params += p.size

        report = NoiseReport(
            mechanism="laplace",
            noise_scale=b,
            sensitivity=sensitivity,
            num_parameters=total_params,
            total_noise_norm=float(np.sqrt(total_noise_norm_sq)),
        )
        logger.debug("Laplace noise applied: b=%.6f, eps=%.4f", b, self.epsilon)
        return noisy_params, report
