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


class TestStudentTFatTails:
    _COMMON = dict(
        n_simulations=2_000, years=15,
        stock_mean=0.07, stock_std=0.15,
        inflation_mean=0.02, inflation_std=0.02,
        bonds_yield=0.03, bond_std=0.01,
    )

    def _rng(self, seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_success_rate_in_range(self):
        config = _make_config(mc_stock_return_df=5.0)
        scenario = _make_scenario(config)
        result = run_monte_carlo_simulation(
            config=config, scenario=scenario,
            n_simulations=500, stock_std=0.15, inflation_std=0.02, seed=42,
        )
        assert 0.0 <= result.success_rate <= 1.0

    def test_deterministic_with_same_seed(self):
        config = _make_config(mc_stock_return_df=5.0)
        scenario = _make_scenario(config)
        kwargs = dict(config=config, scenario=scenario, n_simulations=500,
                      stock_std=0.15, inflation_std=0.02, seed=42)
        r1 = run_monte_carlo_simulation(**kwargs)
        r2 = run_monte_carlo_simulation(**kwargs)
        assert r1.success_rate == r2.success_rate

    def test_std_preserved(self):
        """Student-t scaled draws should have std ≈ target std."""
        rng = self._rng(0)
        returns, _, _ = _generate_mc_returns(**self._COMMON, rng=rng, stock_return_df=5.0)
        assert abs(float(np.std(returns)) - 0.15) < 0.015

    def test_fatter_tails_than_normal(self):
        """t(5) should produce more extreme negative returns at P1 than normal."""
        rng_n = self._rng(1)
        normal, _, _ = _generate_mc_returns(
            **{**self._COMMON, "n_simulations": 50_000, "stock_mean": 0.0},
            rng=rng_n, stock_return_df=1000.0,
        )
        rng_t = self._rng(1)
        fat, _, _ = _generate_mc_returns(
            **{**self._COMMON, "n_simulations": 50_000, "stock_mean": 0.0},
            rng=rng_t, stock_return_df=5.0,
        )
        assert float(np.percentile(fat, 1)) < float(np.percentile(normal, 1))

    def test_large_df_close_to_normal(self):
        """df=1000 should behave statistically like normal."""
        rng1 = self._rng(0)
        r1, _, _ = _generate_mc_returns(**self._COMMON, rng=rng1, stock_return_df=1000.0)
        rng2 = self._rng(0)
        r2, _, _ = _generate_mc_returns(**self._COMMON, rng=rng2)
        # Same seed → same underlying z_stock → results differ only due to RNG order
        # Just check statistical proximity (mean and std should be very close)
        assert abs(float(np.mean(r1)) - float(np.mean(r2))) < 0.01
        assert abs(float(np.std(r1)) - float(np.std(r2))) < 0.01


class TestStockBondCorrelation:
    _COMMON = dict(
        n_simulations=50_000, years=1,
        stock_mean=0.05, stock_std=0.15,
        inflation_mean=0.02, inflation_std=0.02,
        bonds_yield=0.03, bond_std=0.10,
    )

    def _rng(self, seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_success_rate_in_range(self):
        config = _make_config(mc_stock_bond_correlation=-0.3)
        scenario = _make_scenario(config)
        result = run_monte_carlo_simulation(
            config=config, scenario=scenario,
            n_simulations=500, stock_std=0.15, inflation_std=0.02, seed=42,
        )
        assert 0.0 <= result.success_rate <= 1.0

    def test_deterministic_with_same_seed(self):
        config = _make_config(mc_stock_bond_correlation=-0.3)
        scenario = _make_scenario(config)
        kwargs = dict(config=config, scenario=scenario, n_simulations=500,
                      stock_std=0.15, inflation_std=0.02, seed=42)
        r1 = run_monte_carlo_simulation(**kwargs)
        r2 = run_monte_carlo_simulation(**kwargs)
        assert r1.success_rate == r2.success_rate

    def test_negative_correlation_preserved(self):
        """Empirical correlation should be close to the requested value."""
        rng = self._rng(0)
        stocks, _, bonds = _generate_mc_returns(**self._COMMON, rng=rng, stock_bond_corr=-0.5)
        emp_corr = float(np.corrcoef(stocks[:, 0], bonds[:, 0])[0, 1])
        assert abs(emp_corr - (-0.5)) < 0.02

    def test_positive_correlation_preserved(self):
        rng = self._rng(0)
        stocks, _, bonds = _generate_mc_returns(**self._COMMON, rng=rng, stock_bond_corr=0.4)
        emp_corr = float(np.corrcoef(stocks[:, 0], bonds[:, 0])[0, 1])
        assert abs(emp_corr - 0.4) < 0.02

    def test_zero_correlation_near_zero(self):
        rng = self._rng(0)
        stocks, _, bonds = _generate_mc_returns(**self._COMMON, rng=rng, stock_bond_corr=0.0)
        emp_corr = float(np.corrcoef(stocks[:, 0], bonds[:, 0])[0, 1])
        assert abs(emp_corr) < 0.03


class TestStockReturnAR1:
    _COMMON = dict(
        n_simulations=2_000, years=15,
        stock_mean=0.07, stock_std=0.15,
        inflation_mean=0.02, inflation_std=0.02,
        bonds_yield=0.03, bond_std=0.01,
    )

    def _rng(self, seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_success_rate_in_range(self):
        config = _make_config(mc_stock_return_ar1_phi=-0.25)
        scenario = _make_scenario(config)
        result = run_monte_carlo_simulation(
            config=config, scenario=scenario,
            n_simulations=500, stock_std=0.15, inflation_std=0.02, seed=42,
        )
        assert 0.0 <= result.success_rate <= 1.0

    def test_deterministic_with_same_seed(self):
        config = _make_config(mc_stock_return_ar1_phi=-0.25)
        scenario = _make_scenario(config)
        kwargs = dict(config=config, scenario=scenario, n_simulations=500,
                      stock_std=0.15, inflation_std=0.02, seed=42)
        r1 = run_monte_carlo_simulation(**kwargs)
        r2 = run_monte_carlo_simulation(**kwargs)
        assert r1.success_rate == r2.success_rate

    def test_marginal_std_preserved(self):
        """AR(1) z-scores remain marginally N(0,1), so return std should equal target."""
        rng = self._rng(0)
        returns, _, _ = _generate_mc_returns(**self._COMMON, rng=rng, stock_return_ar1_phi=-0.25)
        assert abs(float(np.std(returns)) - 0.15) < 0.02

    def test_negative_autocorrelation_in_returns(self):
        """Adjacent-year returns should be negatively correlated with phi < 0."""
        common = {**self._COMMON, "n_simulations": 20_000}
        rng = self._rng(0)
        returns, _, _ = _generate_mc_returns(**common, rng=rng, stock_return_ar1_phi=-0.25)
        corr = float(np.corrcoef(returns[:, 5], returns[:, 6])[0, 1])
        assert corr < -0.10, f"Expected negative autocorrelation, got {corr:.3f}"

    def test_zero_phi_same_as_default(self):
        """Explicit phi=0 must produce identical output to omitting the argument."""
        s1, _, _ = _generate_mc_returns(self._rng(), **self._COMMON, stock_return_ar1_phi=0.0)
        s2, _, _ = _generate_mc_returns(self._rng(), **self._COMMON)
        np.testing.assert_array_equal(s1, s2)


class TestParameterUncertainty:
    _COMMON = dict(
        n_simulations=2_000, years=15,
        stock_mean=0.07, stock_std=0.15,
        inflation_mean=0.02, inflation_std=0.02,
        bonds_yield=0.03, bond_std=0.01,
    )

    def _rng(self, seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_success_rate_in_range(self):
        config = _make_config(mc_stock_mean_uncertainty=0.015, mc_stock_vol_uncertainty=0.03)
        scenario = _make_scenario(config)
        result = run_monte_carlo_simulation(
            config=config, scenario=scenario,
            n_simulations=500, stock_std=0.15, inflation_std=0.02, seed=42,
        )
        assert 0.0 <= result.success_rate <= 1.0

    def test_deterministic_with_same_seed(self):
        config = _make_config(mc_stock_mean_uncertainty=0.015, mc_stock_vol_uncertainty=0.03)
        scenario = _make_scenario(config)
        kwargs = dict(config=config, scenario=scenario, n_simulations=500,
                      stock_std=0.15, inflation_std=0.02, seed=42)
        r1 = run_monte_carlo_simulation(**kwargs)
        r2 = run_monte_carlo_simulation(**kwargs)
        assert r1.success_rate == r2.success_rate

    def test_uncertainty_widens_outcome_spread(self):
        """Parameter uncertainty should widen the interquartile range of final portfolios."""
        kwargs = dict(n_simulations=5_000, stock_std=0.15, inflation_std=0.02, seed=42)
        config_base = _make_config()
        scenario = _make_scenario(config_base)
        r_base = run_monte_carlo_simulation(config=config_base, scenario=scenario, **kwargs)

        config_unc = _make_config(mc_stock_mean_uncertainty=0.02, mc_stock_vol_uncertainty=0.04)
        r_unc = run_monte_carlo_simulation(config=config_unc, scenario=scenario, **kwargs)

        spread_base = r_base.final_portfolio_percentiles[75] - r_base.final_portfolio_percentiles[25]
        spread_unc  = r_unc.final_portfolio_percentiles[75]  - r_unc.final_portfolio_percentiles[25]
        assert spread_unc > spread_base

    def test_zero_uncertainty_same_as_default(self):
        """Explicit zero uncertainty must produce identical output to the no-uncertainty default."""
        r1, _, _ = _generate_mc_returns(
            self._rng(), **self._COMMON, stock_mean_uncertainty=0.0, stock_vol_uncertainty=0.0
        )
        r2, _, _ = _generate_mc_returns(self._rng(), **self._COMMON)
        np.testing.assert_array_equal(r1, r2)


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
