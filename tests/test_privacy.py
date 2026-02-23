"""Tests for the Differential Privacy engine."""

import numpy as np
import pytest

from sofam_net.privacy.accountant import PrivacyBudget, RDPAccountant
from sofam_net.privacy.mechanisms import GaussianMechanism, LaplaceMechanism


class TestGaussianMechanism:
    def test_noise_scale(self):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5)
        sigma = mech.compute_noise_scale(1.0)
        assert sigma > 0
        # sigma = sqrt(2*ln(1.25/1e-5)) / 1.0 ≈ 4.92
        assert 4.0 < sigma < 6.0

    def test_add_noise(self, sample_params):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5)
        noisy, report = mech.add_noise(sample_params, sensitivity=1.0)
        assert len(noisy) == len(sample_params)
        assert report.mechanism == "gaussian"
        assert report.noise_scale > 0
        # Noisy params should differ from original
        total_diff = sum(np.sum(np.abs(n - o)) for n, o in zip(noisy, sample_params))
        assert total_diff > 0

    def test_reproducibility(self, sample_params):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        noisy1, _ = mech.add_noise(sample_params, 1.0, rng=rng1)
        noisy2, _ = mech.add_noise(sample_params, 1.0, rng=rng2)
        for a, b in zip(noisy1, noisy2):
            np.testing.assert_array_equal(a, b)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            GaussianMechanism(epsilon=-1.0)

    def test_invalid_delta(self):
        with pytest.raises(ValueError, match="Delta must be in"):
            GaussianMechanism(epsilon=1.0, delta=0)

    def test_higher_epsilon_less_noise(self, sample_params):
        mech_low = GaussianMechanism(epsilon=0.1, delta=1e-5)
        mech_high = GaussianMechanism(epsilon=10.0, delta=1e-5)
        assert mech_low.compute_noise_scale(1.0) > mech_high.compute_noise_scale(1.0)


class TestLaplaceMechanism:
    def test_noise_scale(self):
        mech = LaplaceMechanism(epsilon=1.0)
        b = mech.compute_noise_scale(1.0)
        assert b == 1.0

    def test_add_noise(self, sample_params):
        mech = LaplaceMechanism(epsilon=1.0)
        noisy, report = mech.add_noise(sample_params, sensitivity=1.0)
        assert report.mechanism == "laplace"
        assert len(noisy) == len(sample_params)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            LaplaceMechanism(epsilon=0)


class TestPrivacyBudget:
    def test_creation(self):
        budget = PrivacyBudget(org_id="org1", total_epsilon=10.0)
        assert budget.remaining_epsilon == 10.0
        assert not budget.is_exhausted
        assert budget.fraction_spent == 0.0

    def test_spend(self):
        budget = PrivacyBudget(org_id="org1", total_epsilon=10.0)
        budget.record_spend(1.0, round_num=0)
        assert budget.remaining_epsilon == 9.0
        assert budget.rounds_participated == 1

    def test_exhaustion(self):
        budget = PrivacyBudget(org_id="org1", total_epsilon=2.0)
        budget.record_spend(1.0, 0)
        budget.record_spend(1.0, 1)
        assert budget.is_exhausted
        assert budget.remaining_epsilon == 0.0

    def test_to_dict(self):
        budget = PrivacyBudget(org_id="org1", total_epsilon=5.0)
        budget.record_spend(1.5, 0)
        d = budget.to_dict()
        assert d["org_id"] == "org1"
        assert d["spent_epsilon"] == 1.5
        assert d["remaining_epsilon"] == 3.5


class TestRDPAccountant:
    def test_creation(self):
        acct = RDPAccountant()
        assert acct.get_epsilon() == 0.0

    def test_step(self):
        acct = RDPAccountant(target_delta=1e-5)
        eps = acct.step(sigma=1.0)
        assert eps > 0

    def test_composition(self):
        acct = RDPAccountant(target_delta=1e-5)
        eps1 = acct.step(sigma=1.0)
        eps2 = acct.step(sigma=1.0)
        # After two steps, epsilon should be higher
        assert eps2 > eps1

    def test_create_budget(self):
        acct = RDPAccountant()
        budget = acct.create_budget("org1", total_epsilon=10.0)
        assert budget.org_id == "org1"
        assert "org1" in acct.budgets

    def test_can_participate(self):
        acct = RDPAccountant()
        acct.create_budget("org1", total_epsilon=5.0)
        assert acct.can_participate("org1", 1.0)
        acct.record_participation("org1", 4.5, 0)
        assert not acct.can_participate("org1", 1.0)

    def test_get_all_budgets(self):
        acct = RDPAccountant()
        acct.create_budget("org1", 10.0)
        acct.create_budget("org2", 5.0)
        budgets = acct.get_all_budgets()
        assert len(budgets) == 2
        assert "org1" in budgets

    def test_reset(self):
        acct = RDPAccountant()
        acct.step(sigma=1.0)
        acct.reset()
        assert acct.get_epsilon() == 0.0

    def test_higher_sigma_lower_epsilon(self):
        acct1 = RDPAccountant(target_delta=1e-5)
        acct2 = RDPAccountant(target_delta=1e-5)
        eps1 = acct1.step(sigma=0.5)
        eps2 = acct2.step(sigma=5.0)
        assert eps1 > eps2  # More noise = better privacy
