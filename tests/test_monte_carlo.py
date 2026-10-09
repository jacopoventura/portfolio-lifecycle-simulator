import numpy as np
import pytest

from finance import run_scenario
from models import Config
from monte_carlo import _generate_mc_returns, compute_inflation_cagr_grid, run_monte_carlo_simulation


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
        stock_monthly_investment_pct=0.5,
        bonds_monthly_investment_pct=0.3,
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
        # 1% yield std keeps draws clear of the −1% yield floor, which would
        # otherwise clip the distribution and bias the measured correlation.
        bonds_yield=0.03, bond_std=0.01,
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

    # stock_bond_corr is defined on bond PRICE returns, which move opposite to
    # yields — so with the legacy single-year draw (κ=1, β=0) the yield level
    # correlates with stocks at −ρ and the bond price shock (−yield) at +ρ.

    def test_negative_correlation_preserved(self):
        """Empirical stock/bond-price correlation should match the requested value."""
        rng = self._rng(0)
        stocks, _, bonds = _generate_mc_returns(**self._COMMON, rng=rng, stock_bond_corr=-0.5)
        emp_corr = float(np.corrcoef(stocks[:, 0], -bonds[:, 0])[0, 1])
        assert abs(emp_corr - (-0.5)) < 0.02

    def test_positive_correlation_preserved(self):
        rng = self._rng(0)
        stocks, _, bonds = _generate_mc_returns(**self._COMMON, rng=rng, stock_bond_corr=0.4)
        emp_corr = float(np.corrcoef(stocks[:, 0], -bonds[:, 0])[0, 1])
        assert abs(emp_corr - 0.4) < 0.02

    def test_zero_correlation_near_zero(self):
        rng = self._rng(0)
        stocks, _, bonds = _generate_mc_returns(**self._COMMON, rng=rng, stock_bond_corr=0.0)
        emp_corr = float(np.corrcoef(stocks[:, 0], -bonds[:, 0])[0, 1])
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


class TestInflationCagrGrid:
    def _make_config(self, **kw):
        from tests.test_monte_carlo import _make_config as mc
        return mc(**kw)

    def _scenario(self, config):
        from finance import run_scenario
        return run_scenario(
            name="TEST", config=config,
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

    def test_grid_dimensions(self):
        from monte_carlo import _INFLATION_VALUES
        config = _make_config()
        scenario = _make_scenario(config)
        grid = compute_inflation_cagr_grid(
            config=config, scenario=scenario,
            stock_std=0.15, inflation_std=0.02,
            monthly_stocks=2_000.0, monthly_bonds=500.0,
            n_simulations=200, seed=42,
        )
        assert len(grid.inflation_values) == len(_INFLATION_VALUES)
        assert len(grid.grid) == len(_INFLATION_VALUES)
        assert all(len(row) == len(grid.cagr_values) for row in grid.grid)

    def test_success_rates_in_range(self):
        config = _make_config()
        scenario = _make_scenario(config)
        grid = compute_inflation_cagr_grid(
            config=config, scenario=scenario,
            stock_std=0.15, inflation_std=0.02,
            monthly_stocks=2_000.0, monthly_bonds=500.0,
            n_simulations=200, seed=42,
        )
        for row in grid.grid:
            for cell in row:
                assert 0.0 <= cell.success_income <= 1.0

    def test_higher_cagr_higher_success(self):
        """At fixed inflation, higher CAGR should give higher or equal success."""
        config = _make_config()
        scenario = _make_scenario(config)
        grid = compute_inflation_cagr_grid(
            config=config, scenario=scenario,
            stock_std=0.15, inflation_std=0.02,
            monthly_stocks=2_000.0, monthly_bonds=500.0,
            n_simulations=500, seed=42,
        )
        # Check middle inflation row: success should be non-decreasing with CAGR
        mid_row = grid.grid[len(grid.grid) // 2]
        rates = [cell.success_income for cell in mid_row]
        assert rates[-1] >= rates[0], f"Expected higher CAGR → higher success: {rates}"

    def test_higher_inflation_lower_success(self):
        """At fixed CAGR, higher inflation should give lower or equal success."""
        config = _make_config()
        scenario = _make_scenario(config)
        grid = compute_inflation_cagr_grid(
            config=config, scenario=scenario,
            stock_std=0.15, inflation_std=0.02,
            monthly_stocks=2_000.0, monthly_bonds=500.0,
            n_simulations=500, seed=42,
        )
        # Check last CAGR column: success should be non-increasing with inflation
        col_idx = len(grid.cagr_values) - 1
        rates = [grid.grid[i][col_idx].success_income for i in range(len(grid.inflation_values))]
        assert rates[0] >= rates[-1], f"Expected higher inflation → lower success: {rates}"

    def test_base_values_recorded(self):
        config = _make_config(
            expected_yearly_growth_stocks_percent=5.0,
            expected_inflation_percent=2.0,
        )
        scenario = _make_scenario(config)
        grid = compute_inflation_cagr_grid(
            config=config, scenario=scenario,
            stock_std=0.15, inflation_std=0.02,
            monthly_stocks=2_000.0, monthly_bonds=500.0,
            n_simulations=200, seed=42,
        )
        assert grid.base_cagr == pytest.approx(0.05)
        assert grid.base_inflation == pytest.approx(0.02)

    def test_deterministic_with_same_seed(self):
        config = _make_config()
        scenario = _make_scenario(config)
        kwargs = dict(
            config=config, scenario=scenario,
            stock_std=0.15, inflation_std=0.02,
            monthly_stocks=2_000.0, monthly_bonds=500.0,
            n_simulations=200, seed=7,
        )
        g1 = compute_inflation_cagr_grid(**kwargs)
        g2 = compute_inflation_cagr_grid(**kwargs)
        for i, row in enumerate(g1.grid):
            for j, cell in enumerate(row):
                assert cell.success_income == g2.grid[i][j].success_income


class TestBondYieldModel:
    """Persistent, inflation-linked bond yields (κ = mean reversion, β = inflation link)."""

    _COMMON = dict(
        n_simulations=20_000, years=30,
        stock_mean=0.06, stock_std=0.15,
        inflation_mean=0.023, inflation_std=0.02,
        bonds_yield=0.03, bond_std=0.007,
        inflation_ar1_phi=0.6,
    )
    _DURATION = 7.0

    def _draw(self, seed: int = 0, **kw):
        return _generate_mc_returns(**{**self._COMMON, **kw}, rng=np.random.default_rng(seed))

    @staticmethod
    def _lag1(x: np.ndarray) -> float:
        return float(np.corrcoef(x[:, 1:].ravel(), x[:, :-1].ravel())[0, 1])

    def test_legacy_defaults_are_independent_each_year(self):
        _, _, y = self._draw()  # function defaults: κ=1, β=0
        assert abs(self._lag1(y)) < 0.02

    def test_mean_reversion_creates_persistent_regimes(self):
        _, _, y = self._draw(bond_yield_mean_reversion=0.15)
        assert self._lag1(y) > 0.7

    def test_yields_rise_with_inflation(self):
        _, infl, y = self._draw(bond_yield_mean_reversion=0.15, bond_yield_inflation_beta=0.3)
        assert float(np.corrcoef(infl.ravel(), y.ravel())[0, 1]) > 0.2
        _, infl0, y0 = self._draw(bond_yield_mean_reversion=0.15, bond_yield_inflation_beta=0.0)
        assert abs(float(np.corrcoef(infl0.ravel(), y0.ravel())[0, 1])) < 0.03

    def test_long_run_mean_is_configured_yield(self):
        _, _, y = self._draw(bond_yield_mean_reversion=0.15, bond_yield_inflation_beta=0.3)
        assert abs(float(np.mean(y)) - 0.03) < 0.002

    def test_bond_return_volatility_is_realistic(self):
        """Coupon + duration price change: ~4–7.5% annual vol (EUR aggregate ≈ 5–6%)."""
        _, _, y = self._draw(bond_yield_mean_reversion=0.15, bond_yield_inflation_beta=0.3)
        ret = y[:, 1:] - self._DURATION * np.diff(y, axis=1)
        assert 0.04 < float(np.std(ret)) < 0.075

    def test_bonds_hedge_stocks_with_negative_correlation(self):
        stocks, _, y = self._draw(
            bond_yield_mean_reversion=0.15, bond_yield_inflation_beta=0.3, stock_bond_corr=-0.3,
        )
        price_ret = -np.diff(y, axis=1)
        assert float(np.corrcoef(stocks[:, 1:].ravel(), price_ret.ravel())[0, 1]) < -0.1

    def test_yield_floor(self):
        _, _, y = self._draw(bond_std=0.05, bond_yield_mean_reversion=0.05)
        assert float(y.min()) >= -0.01 - 1e-12

    def test_config_values_reach_the_generator(self):
        """Config-driven decumulation must differ between legacy and persistent bonds."""
        from monte_carlo import simulate_decumulation
        legacy = _make_config(mc_bond_yield_mean_reversion=1.0, mc_bond_yield_inflation_beta=0.0)
        modern = _make_config(mc_bond_yield_mean_reversion=0.15, mc_bond_yield_inflation_beta=0.3)
        kw = dict(initial_portfolio=1_000_000.0, n_simulations=2_000, seed=1,
                  return_final_portfolios=True)
        a = simulate_decumulation(legacy, **kw).final_portfolios
        b = simulate_decumulation(modern, **kw).final_portfolios
        assert a != b


class TestCagrInput:
    """expected_yearly_growth_stocks_percent is a CAGR: paths must compound at it."""

    def test_conversion_matches_lognormal_rule_of_thumb(self):
        from monte_carlo import cagr_to_arithmetic_mean
        mu = cagr_to_arithmetic_mean(0.06, 0.15)
        assert 0.069 < mu < 0.072          # ≈ g + σ²/2
        assert cagr_to_arithmetic_mean(0.06, 0.0) == pytest.approx(0.06)

    @pytest.mark.parametrize("kw", [
        dict(),
        dict(stock_return_df=5.0, stock_return_ar1_phi=-0.25,
             stock_mean_uncertainty=0.015, stock_vol_uncertainty=0.03),
    ])
    def test_median_compounded_growth_equals_input(self, kw):
        stocks, _, _ = _generate_mc_returns(
            rng=np.random.default_rng(3), n_simulations=20_000, years=30,
            stock_mean=0.06, stock_std=0.15, inflation_mean=0.02, inflation_std=0.02,
            bonds_yield=0.03, bond_std=0.007, stock_mean_is_cagr=True, **kw,
        )
        cagr = np.prod(1 + stocks, axis=1) ** (1 / 30) - 1
        assert float(np.median(cagr)) == pytest.approx(0.06, abs=0.003)

    def test_legacy_flag_keeps_arithmetic_mean(self):
        stocks, _, _ = _generate_mc_returns(
            rng=np.random.default_rng(3), n_simulations=20_000, years=30,
            stock_mean=0.06, stock_std=0.15, inflation_mean=0.02, inflation_std=0.02,
            bonds_yield=0.03, bond_std=0.007, stock_mean_is_cagr=False,
        )
        assert float(np.mean(stocks)) == pytest.approx(0.06, abs=0.002)


class TestDecumRebalancing:
    """Annual rebalancing must not depend on whether a glide path is active."""

    def _survival(self, stock_frac: float, glide_end: float, **kw) -> float:
        from monte_carlo import _run_decum_loop
        n, ny = 5_000, 30
        s, i, b = _generate_mc_returns(
            rng=np.random.default_rng(42), n_simulations=n, years=ny,
            stock_mean=0.06, stock_std=0.15, inflation_mean=0.023, inflation_std=0.01,
            bonds_yield=0.03, bond_std=0.007, inflation_ar1_phi=0.6,
            bond_yield_mean_reversion=0.15, bond_yield_inflation_beta=0.3,
        )
        W = 1_000_000.0
        return _run_decum_loop(
            s, i, b, np.full(n, W * stock_frac), np.full(n, W * (1 - stock_frac)),
            W * 0.035 / 12, ny, "fixed", 0.0, 0.0, 999,
            stock_pct_end=glide_end, glide_years=20, **kw,
        ).survival_rate

    def test_no_jump_where_start_equals_glide_end(self):
        # Before 2026-10 the 30%-stock point (= glide end) lost rebalancing and
        # survival dropped ~18pp versus its neighbours.
        neighbours = [self._survival(0.29, 0.30), self._survival(0.31, 0.30)]
        at_end = self._survival(0.30, 0.30)
        assert min(neighbours) - 0.03 < at_end < max(neighbours) + 0.03

    def test_legacy_stocks_first_still_available(self):
        assert self._survival(0.30, 0.30, rebalance=False) != self._survival(0.30, 0.30)
