import numpy as np
import pytest

from finance import run_scenario
from models import Config
from monte_carlo import _generate_mc_returns, run_monte_carlo_simulation


def _make_config(**overrides) -> Config:
    defaults = dict(
        current_stock_value=100_000.0,
        current_bonds_value=10_000.0,
        current_stock_unrealized_gain=20_000.0,
        current_cash=50_000.0,
        cash_reserve_at_horizon=10_000.0,
        interest_on_cash_pa=2.0,
        annual_net_saving=24_000.0,
        annual_contribution_growth_percent=0.0,
        maximum_stock_monthly_investment_pct=0.5,
        maximum_bonds_monthly_investment_pct=0.3,
        target_monthly_retirement_income=3_000.0,
        years=10.0,
        expected_inflation_percent=2.0,
        expected_yearly_growth_stocks_percent=7.0,
        bonds_percentage=40.0,
        yield_bonds=3.0,
        tax_rate=0.26,
        tax_rate_bond_italy=0.125,
        withdrawal_percentage_stock=3.3,
        basiszins=0.0229,
        freibetrag=1_000.0,
        teilfreistellung=0.7,
        optimize_grid_step=50_000.0,
        mc_n_simulations=500,
        mc_stock_return_std_percent=15.0,
        mc_inflation_std_percent=2.0,
        mc_bond_yield_std_percent=1.0,
        mc_target_success_percent=95,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_scenario(config: Config):
    return run_scenario(
        name="TEST",
        config=config,
        future_monthly_income_value=4_000.0,
        target_annual_net_income=48_000.0,
        stock_tax_rate=config.tax_rate,
        bond_tax_rate=config.tax_rate,
        stock_expected_yield=config.expected_yearly_growth_stocks_percent / 100,
        bonds_yield=config.yield_bonds / 100,
        withdrawal_rate=config.withdrawal_percentage_stock / 100,
        stock_gain_fraction=0.3,
        stock_teilfreistellung=config.teilfreistellung,
        annual_contribution_growth=config.annual_contribution_growth_percent / 100,
    )


class TestRunMonteCarloSimulation:
    def test_success_rate_in_range(self):
        config = _make_config()
        scenario = _make_scenario(config)
        result = run_monte_carlo_simulation(
            config=config, scenario=scenario,
            n_simulations=200, stock_std=0.15, inflation_std=0.02, seed=42,
        )
        assert 0.0 <= result.success_rate <= 1.0

    def test_deterministic_with_same_seed(self):
        config = _make_config()
        scenario = _make_scenario(config)
        kwargs = dict(config=config, scenario=scenario, n_simulations=200,
                      stock_std=0.15, inflation_std=0.02, seed=42)
        r1 = run_monte_carlo_simulation(**kwargs)
        r2 = run_monte_carlo_simulation(**kwargs)
        assert r1.success_rate == r2.success_rate

    def test_different_seeds_produce_different_results(self):
        config = _make_config()
        scenario = _make_scenario(config)
        r1 = run_monte_carlo_simulation(config=config, scenario=scenario,
                                        n_simulations=500, stock_std=0.15,
                                        inflation_std=0.02, seed=1)
        r2 = run_monte_carlo_simulation(config=config, scenario=scenario,
                                        n_simulations=500, stock_std=0.15,
                                        inflation_std=0.02, seed=999)
        assert r1.success_rate != r2.success_rate

    def test_final_portfolio_percentiles_are_sorted(self):
        config = _make_config()
        scenario = _make_scenario(config)
        result = run_monte_carlo_simulation(
            config=config, scenario=scenario,
            n_simulations=200, stock_std=0.15, inflation_std=0.02, seed=42,
        )
        p = result.final_portfolio_percentiles
        tail = 100 - config.mc_target_success_percent
        target = config.mc_target_success_percent
        assert p[tail] <= p[25] <= p[50] <= p[75] <= p[target]

    def test_split_metrics_in_range(self):
        config = _make_config()
        scenario = _make_scenario(config)
        result = run_monte_carlo_simulation(
            config=config, scenario=scenario,
            n_simulations=300, stock_std=0.15, inflation_std=0.02, seed=42,
        )
        assert 0.0 <= result.success_rate_stock_only <= 1.0
        assert 0.0 <= result.success_rate_bond_only <= 1.0
        assert 0.0 <= result.success_rate_income_only <= 1.0

    def test_split_metrics_ordering(self):
        # Q1 combined (stocks AND bonds) ≤ each individual metric
        config = _make_config()
        scenario = _make_scenario(config)
        result = run_monte_carlo_simulation(
            config=config, scenario=scenario,
            n_simulations=500, stock_std=0.15, inflation_std=0.02, seed=42,
        )
        assert result.success_rate <= result.success_rate_stock_only
        assert result.success_rate <= result.success_rate_bond_only

    def test_income_metric_increases_with_large_portfolio(self):
        config = _make_config()
        scenario = _make_scenario(config)
        common = dict(config=config, scenario=scenario, n_simulations=500,
                      stock_std=0.15, inflation_std=0.02, seed=42)
        baseline = run_monte_carlo_simulation(**common)
        rich = run_monte_carlo_simulation(**common, initial_stock_override=5_000_000.0)
        assert rich.success_rate_income_only > baseline.success_rate_income_only

    def test_large_initial_stock_raises_success_rate(self):
        config = _make_config()
        scenario = _make_scenario(config)
        common = dict(config=config, scenario=scenario, n_simulations=500,
                      stock_std=0.15, inflation_std=0.02, seed=42)
        mc_baseline = run_monte_carlo_simulation(**common)
        mc_rich = run_monte_carlo_simulation(
            **common, initial_stock_override=5_000_000.0
        )
        assert mc_rich.success_rate > mc_baseline.success_rate

    def test_result_name_matches_scenario(self):
        config = _make_config()
        scenario = _make_scenario(config)
        result = run_monte_carlo_simulation(
            config=config, scenario=scenario,
            n_simulations=100, stock_std=0.15, inflation_std=0.02, seed=42,
        )
        assert result.name == scenario.name

    def test_required_portfolio_grows_with_inflation(self):
        config_low = _make_config(expected_inflation_percent=1.0)
        config_high = _make_config(expected_inflation_percent=5.0)
        scenario_low = _make_scenario(config_low)
        scenario_high = _make_scenario(config_high)
        r_low = run_monte_carlo_simulation(config=config_low, scenario=scenario_low,
                                           n_simulations=500, stock_std=0.15,
                                           inflation_std=0.01, seed=42)
        r_high = run_monte_carlo_simulation(config=config_high, scenario=scenario_high,
                                            n_simulations=500, stock_std=0.15,
                                            inflation_std=0.01, seed=42)
        # Higher inflation → larger required portfolio (median)
        assert r_high.required_portfolio_percentiles[50] > r_low.required_portfolio_percentiles[50]


class TestAR1Inflation:
    _COMMON = dict(
        n_simulations=2000, years=15,
        stock_mean=0.07, stock_std=0.15,
        inflation_mean=0.02, inflation_std=0.02,
        bonds_yield=0.03, bond_std=0.01,
    )

    def _rng(self, seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_phi0_matches_default(self):
        """phi=0 must be identical to omitting the argument (same code path, same RNG state)."""
        _, inf_phi0, _ = _generate_mc_returns(self._rng(), **self._COMMON, inflation_ar1_phi=0.0)
        _, inf_default, _ = _generate_mc_returns(self._rng(), **self._COMMON)
        np.testing.assert_array_equal(inf_phi0, inf_default)

    def test_deterministic_with_same_seed(self):
        _, inf1, _ = _generate_mc_returns(self._rng(), **self._COMMON, inflation_ar1_phi=0.6)
        _, inf2, _ = _generate_mc_returns(self._rng(), **self._COMMON, inflation_ar1_phi=0.6)
        np.testing.assert_array_equal(inf1, inf2)

    def test_stationary_variance_preserved(self):
        """Marginal variance of inflation should ≈ inflation_std² for any phi (allowing for clipping)."""
        _, inflations, _ = _generate_mc_returns(self._rng(), **self._COMMON, inflation_ar1_phi=0.6)
        var_mid = np.var(inflations[:, 7])
        expected_var = self._COMMON["inflation_std"] ** 2
        assert abs(var_mid - expected_var) / expected_var < 0.3

    def test_autocorrelation_present_with_phi(self):
        """Adjacent-year inflation should be positively correlated when phi=0.6."""
        _, inflations, _ = _generate_mc_returns(self._rng(0), **self._COMMON, inflation_ar1_phi=0.6)
        corr = np.corrcoef(inflations[:, 5], inflations[:, 6])[0, 1]
        assert corr > 0.3, f"Expected positive autocorrelation, got {corr:.3f}"

    def test_phi0_no_autocorrelation(self):
        """Adjacent years should be near-zero correlated when phi=0."""
        _, inflations, _ = _generate_mc_returns(self._rng(0), **self._COMMON, inflation_ar1_phi=0.0)
        corr = np.corrcoef(inflations[:, 5], inflations[:, 6])[0, 1]
        assert abs(corr) < 0.1, f"Expected no autocorrelation with phi=0, got {corr:.3f}"

    def test_fatter_tails_on_cumulative_inflation(self):
        """phi=0.6 should widen cumulative 15-year inflation spread vs phi=0."""
        common = {**self._COMMON, "n_simulations": 5000}
        _, inf0, _ = _generate_mc_returns(self._rng(1), **common, inflation_ar1_phi=0.0)
        _, inf6, _ = _generate_mc_returns(self._rng(1), **common, inflation_ar1_phi=0.6)
        spread0 = np.percentile(inf0.sum(axis=1), 95) - np.percentile(inf0.sum(axis=1), 5)
        spread6 = np.percentile(inf6.sum(axis=1), 95) - np.percentile(inf6.sum(axis=1), 5)
        assert spread6 > spread0, (
            f"AR(1) phi=0.6 should widen cumulative inflation spread: "
            f"phi=0={spread0:.4f}, phi=0.6={spread6:.4f}"
        )
