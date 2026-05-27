"""Tests for the block bootstrap return generator."""

from __future__ import annotations

import numpy as np
import pytest

from bootstrap import block_bootstrap_returns, load_bootstrap_annual_returns


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _synthetic_returns(n: int = 50, seed: int = 0) -> np.ndarray:
    """Deterministic synthetic annual returns for unit tests."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.08, 0.18, n)


# ── TestLoadBootstrapReturns ─────────────────────────────────────────────────

class TestLoadBootstrapReturns:
    def test_returns_1d_array(self):
        arr = load_bootstrap_annual_returns()
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1

    def test_reasonable_length(self):
        arr = load_bootstrap_annual_returns()
        assert len(arr) >= 30

    def test_reasonable_values(self):
        arr = load_bootstrap_annual_returns()
        assert arr.min() > -1.0
        assert arr.max() < 2.0
        assert -0.05 < arr.mean() < 0.25

    def test_caching(self):
        a = load_bootstrap_annual_returns()
        b = load_bootstrap_annual_returns()
        assert a is b


# ── TestBlockBootstrapReturns ────────────────────────────────────────────────

class TestBlockBootstrapReturns:
    def test_output_shape(self):
        hist = _synthetic_returns()
        result = block_bootstrap_returns(_rng(), hist, 100, 15)
        assert result.shape == (100, 15)

    def test_deterministic_with_same_seed(self):
        hist = _synthetic_returns()
        a = block_bootstrap_returns(_rng(7), hist, 50, 10)
        b = block_bootstrap_returns(_rng(7), hist, 50, 10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        hist = _synthetic_returns()
        a = block_bootstrap_returns(_rng(1), hist, 50, 10)
        b = block_bootstrap_returns(_rng(2), hist, 50, 10)
        assert not np.array_equal(a, b)

    def test_mean_shift_accuracy(self):
        hist = _synthetic_returns(99)
        target = 0.06
        result = block_bootstrap_returns(_rng(), hist, 10_000, 15, target_mean=target)
        assert abs(result.mean() - target) < 0.001

    def test_no_mean_shift(self):
        hist = _synthetic_returns(99)
        result = block_bootstrap_returns(_rng(), hist, 10_000, 15, target_mean=None)
        # Without mean-shift, mean should be close to hist mean, not exactly
        assert abs(result.mean() - hist.mean()) < 0.02

    def test_floor_at_minus_099(self):
        # Inject an extreme return
        hist = np.full(20, -0.95)
        result = block_bootstrap_returns(_rng(), hist, 100, 10, target_mean=None)
        assert result.min() >= -0.99

    def test_block_structure_autocorrelation(self):
        """Bootstrap with block_size > 1 should show higher lag-1 autocorrelation
        than independent draws (block_size=1)."""
        hist = _synthetic_returns(99)
        bs5 = block_bootstrap_returns(_rng(0), hist, 5000, 20, block_size=5, target_mean=None)
        bs1 = block_bootstrap_returns(_rng(0), hist, 5000, 20, block_size=1, target_mean=None)

        def _lag1_autocorr(arr: np.ndarray) -> float:
            flat = arr.ravel()
            x = flat[:-1] - flat.mean()
            y = flat[1:] - flat.mean()
            return float(np.sum(x * y) / np.sum(x**2))

        ac_block = _lag1_autocorr(bs5)
        ac_indep = _lag1_autocorr(bs1)
        assert ac_block > ac_indep

    def test_short_history_raises(self):
        hist = np.array([0.05, 0.10, 0.08])
        with pytest.raises(ValueError, match="at least"):
            block_bootstrap_returns(_rng(), hist, 100, 10, block_size=5)

    def test_various_block_sizes(self):
        hist = _synthetic_returns(50)
        for bs in [1, 3, 5, 10]:
            result = block_bootstrap_returns(_rng(), hist, 100, 15, block_size=bs)
            assert result.shape == (100, 15)

    def test_years_longer_than_history(self):
        hist = _synthetic_returns(20)
        result = block_bootstrap_returns(_rng(), hist, 100, 40, block_size=5)
        assert result.shape == (100, 40)


# ── TestBootstrapIntegration ─────────────────────────────────────────────────

class TestBootstrapIntegration:
    """Test _generate_mc_returns with bootstrap vs parametric."""

    def test_bootstrap_returns_same_shape(self):
        from monte_carlo import _generate_mc_returns
        hist = _synthetic_returns(50)
        rng = _rng()
        sr, inf, by = _generate_mc_returns(
            rng, 100, 15,
            0.06, 0.15, 0.023, 0.02, 0.031, 0.01,
            bootstrap_annual_returns=hist,
            bootstrap_block_size=5,
            bootstrap_mean_shift=True,
        )
        assert sr.shape == (100, 15)
        assert inf.shape == (100, 15)
        assert by.shape == (100, 15)

    def test_none_bootstrap_matches_parametric(self):
        from monte_carlo import _generate_mc_returns
        kwargs = dict(
            n_simulations=50, years=10,
            stock_mean=0.06, stock_std=0.15,
            inflation_mean=0.023, inflation_std=0.02,
            bonds_yield=0.031, bond_std=0.01,
        )
        a_sr, a_inf, a_by = _generate_mc_returns(_rng(42), **kwargs)
        b_sr, b_inf, b_by = _generate_mc_returns(
            _rng(42), **kwargs,
            bootstrap_annual_returns=None,
        )
        np.testing.assert_array_equal(a_sr, b_sr)
        np.testing.assert_array_equal(a_inf, b_inf)
        np.testing.assert_array_equal(a_by, b_by)

    def test_bootstrap_stock_returns_differ_from_parametric(self):
        from monte_carlo import _generate_mc_returns
        hist = _synthetic_returns(50)
        kwargs = dict(
            n_simulations=500, years=15,
            stock_mean=0.06, stock_std=0.15,
            inflation_mean=0.023, inflation_std=0.02,
            bonds_yield=0.031, bond_std=0.01,
        )
        param_sr, _, _ = _generate_mc_returns(_rng(42), **kwargs)
        boot_sr, _, _ = _generate_mc_returns(
            _rng(42), **kwargs,
            bootstrap_annual_returns=hist,
        )
        # Distributions should differ (different generation method)
        assert abs(param_sr.std() - boot_sr.std()) > 0.001 or \
               abs(param_sr.mean() - boot_sr.mean()) > 0.001 or \
               not np.array_equal(param_sr, boot_sr)
