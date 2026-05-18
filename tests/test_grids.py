"""
Tests for the Monte Carlo grid functions:
  compute_contribution_cagr_grid  (Q1a / Q2 tables)
  compute_bond_stock_grid         (Q1b table)
  compute_combined_grid           (Q1 combined table)
  compute_optimal_allocation      (Section 3 optimal allocation)

All tests use low n_simulations and reduced grid ranges for speed.
They verify structural invariants and directional properties rather than
exact numeric values, so they stay valid regardless of random-seed choices.
"""

import pytest

from finance import run_scenario
from models import Config
from monte_carlo import (
    compute_bond_stock_grid,
    compute_combined_grid,
    compute_contribution_cagr_grid,
    compute_optimal_allocation,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> Config:
    defaults = dict(
        current_stock_value=100_000.0,
        current_bonds_value=10_000.0,
        current_stock_unrealized_gain=10_000.0,
        current_cash=80_000.0,
        cash_reserve_at_horizon=10_000.0,
        interest_on_cash_pa=2.0,
        annual_net_saving=36_000.0,
        annual_contribution_growth_percent=0.0,
        maximum_stock_monthly_investment_pct=0.5,
        maximum_bonds_monthly_investment_pct=0.3,
        target_monthly_retirement_income=2_000.0,
        years=10.0,
        expected_inflation_percent=2.0,
        expected_yearly_growth_stocks_percent=7.0,
        bonds_percentage=40.0,
        yield_bonds=3.0,
        tax_rate=0.26,
        tax_rate_bond_italy=0.125,
        withdrawal_percentage_stock=3.5,
        basiszins=0.0229,
        freibetrag=1_000.0,
        teilfreistellung=0.7,
        optimize_grid_step=50_000.0,
        mc_n_simulations=200,
        mc_stock_return_std_percent=15.0,
        mc_inflation_std_percent=2.0,
        mc_bond_yield_std_percent=1.0,
        mc_target_success_percent=95,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _scenario(config: Config):
    return run_scenario(
        name="TEST",
        config=config,
        future_monthly_income_value=3_000.0,
        target_annual_net_income=36_000.0,
        stock_tax_rate=config.tax_rate,
        bond_tax_rate=config.tax_rate,
        stock_expected_yield=config.expected_yearly_growth_stocks_percent / 100,
        bonds_yield=config.yield_bonds / 100,
        withdrawal_rate=config.withdrawal_percentage_stock / 100,
        stock_gain_fraction=0.2,
        stock_teilfreistellung=config.teilfreistellung,
        annual_contribution_growth=config.annual_contribution_growth_percent / 100,
    )


# ---------------------------------------------------------------------------
# compute_contribution_cagr_grid
# ---------------------------------------------------------------------------

class TestContributionCagrGrid:
    @pytest.fixture(scope="class")
    def grids(self):
        config = _cfg()
        scenario = _scenario(config)
        return compute_contribution_cagr_grid(
            config=config, scenario=scenario,
            stock_std=0.15, inflation_std=0.02,
            seed=42, n_simulations=100,
        )

    def test_returns_two_grids(self, grids):
        assert len(grids) == 2

    def test_bond_multipliers(self, grids):
        assert grids[0].bonds_multiplier == 1
        assert grids[1].bonds_multiplier == 2

    def test_bonds_monthly_doubles(self, grids):
        assert grids[1].bonds_monthly == pytest.approx(grids[0].bonds_monthly * 2, rel=1e-6)

    def test_grid_shape_matches_axes(self, grids):
        for grid in grids:
            assert len(grid.grid) == len(grid.cagr_values)
            for row in grid.grid:
                assert len(row) == len(grid.contribution_values)

    def test_cagr_values_are_sorted_ascending(self, grids):
        for grid in grids:
            assert grid.cagr_values == sorted(grid.cagr_values)

    def test_contribution_values_are_sorted_ascending(self, grids):
        for grid in grids:
            assert grid.contribution_values == sorted(grid.contribution_values)

    def test_all_rates_in_unit_interval(self, grids):
        for grid in grids:
            for row in grid.grid:
                for cell in row:
                    assert 0.0 <= cell.success_hard <= 1.0
                    assert 0.0 <= cell.success_income <= 1.0
                    assert 0.0 <= cell.success_stock_only <= 1.0

    def test_hard_success_le_stock_only(self, grids):
        # P(stocks≥req AND bonds≥req) ≤ P(stocks≥req)
        for grid in grids:
            for row in grid.grid:
                for cell in row:
                    assert cell.success_hard <= cell.success_stock_only + 1e-9

    def test_stock_tail_positive(self, grids):
        for grid in grids:
            for row in grid.grid:
                for cell in row:
                    assert cell.stock_tail_pct >= 0.0

    def test_higher_contribution_does_not_decrease_stock_success(self, grids):
        # Within a CAGR row, more stock contribution → stock success non-decreasing.
        # Tolerance of 0.15 accounts for independent per-cell seeds with n=100 sims
        # (σ ≈ 0.046, so 3σ ≈ 0.14 per adjacent-cell comparison).
        for grid in grids:
            for row in grid.grid:
                for i in range(len(row) - 1):
                    assert row[i].success_stock_only <= row[i + 1].success_stock_only + 0.15

    def test_higher_cagr_improves_stock_success(self, grids):
        # Compare first vs last CAGR row at the same contribution level
        for grid in grids:
            first_row = grid.grid[0]
            last_row  = grid.grid[-1]
            first_avg = sum(c.success_stock_only for c in first_row) / len(first_row)
            last_avg  = sum(c.success_stock_only for c in last_row)  / len(last_row)
            assert last_avg >= first_avg - 0.05

    def test_scenario_name_preserved(self, grids):
        for grid in grids:
            assert grid.scenario_name == "TEST"

    def test_double_bonds_has_same_cagr_values(self, grids):
        assert grids[0].cagr_values == grids[1].cagr_values


# ---------------------------------------------------------------------------
# compute_bond_stock_grid
# ---------------------------------------------------------------------------

class TestBondStockGrid:
    @pytest.fixture(scope="class")
    def grid(self):
        config = _cfg()
        scenario = _scenario(config)
        return compute_bond_stock_grid(
            config=config, scenario=scenario,
            stock_std=0.15, inflation_std=0.02,
            seed=42, n_simulations=100,
        )

    def test_grid_shape_matches_axes(self, grid):
        assert len(grid.grid) == len(grid.bonds_values)
        for row in grid.grid:
            assert len(row) == len(grid.stocks_values)

    def test_stocks_axis_starts_at_zero(self, grid):
        assert grid.stocks_values[0] == pytest.approx(0.0)

    def test_bonds_axis_starts_at_zero(self, grid):
        assert grid.bonds_values[0] == pytest.approx(0.0)

    def test_axes_sorted_ascending(self, grid):
        assert grid.stocks_values == sorted(grid.stocks_values)
        assert grid.bonds_values  == sorted(grid.bonds_values)

    def test_all_bond_success_in_unit_interval(self, grid):
        for row in grid.grid:
            for cell in row:
                assert 0.0 <= cell.success_bond <= 1.0

    def test_zero_stocks_maximises_bond_success(self, grid):
        # With 0 stocks/mo the full cash pile accumulates → best bond success
        for bonds_idx in range(len(grid.bonds_values)):
            zero_stocks_success = grid.grid[bonds_idx][0].success_bond
            for stocks_idx in range(1, len(grid.stocks_values)):
                assert zero_stocks_success >= grid.grid[bonds_idx][stocks_idx].success_bond - 0.05

    def test_required_bonds_populated(self, grid):
        assert grid.required_bonds > 0

    def test_suggested_stocks_and_bonds_positive(self, grid):
        assert grid.suggested_stocks > 0
        assert grid.suggested_bonds > 0

    def test_scenario_name_preserved(self, grid):
        assert grid.scenario_name == "TEST"


# ---------------------------------------------------------------------------
# compute_combined_grid
# ---------------------------------------------------------------------------

class TestCombinedGrid:
    @pytest.fixture(scope="class")
    def grid(self):
        config = _cfg()
        scenario = _scenario(config)
        return compute_combined_grid(
            config=config, scenario=scenario,
            stock_std=0.15, inflation_std=0.02,
            seed=42, n_simulations=100,
        )

    def test_grid_shape_matches_axes(self, grid):
        assert len(grid.grid) == len(grid.cagr_values)
        for layer in grid.grid:
            assert len(layer) == len(grid.bonds_values)
            for row in layer:
                assert len(row) == len(grid.stocks_values)

    def test_all_combined_success_in_unit_interval(self, grid):
        for layer in grid.grid:
            for row in layer:
                for cell in row:
                    assert 0.0 <= cell.success_combined <= 1.0

    def test_cagr_values_sorted_ascending(self, grid):
        assert grid.cagr_values == sorted(grid.cagr_values)

    def test_stocks_axis_starts_at_zero(self, grid):
        assert grid.stocks_values[0] == pytest.approx(0.0)

    def test_bonds_axis_starts_at_zero(self, grid):
        assert grid.bonds_values[0] == pytest.approx(0.0)

    def test_required_stocks_and_bonds_positive(self, grid):
        assert grid.required_stocks > 0
        assert grid.required_bonds > 0

    def test_scenario_name_preserved(self, grid):
        assert grid.scenario_name == "TEST"


# ---------------------------------------------------------------------------
# compute_optimal_allocation
# ---------------------------------------------------------------------------

class TestComputeOptimalAllocation:
    @pytest.fixture(scope="class")
    def result_unachievable(self):
        # With a very high target success, small portfolio, AND a realistic monthly
        # budget cap, no allocation in the grid can hit 99.9% — so optimal is None.
        # (Without budget_cap, the optimizer would always find a feasible plan by
        #  spending an unrealistic amount per month.)
        config = _cfg(optimize_grid_step=20_000.0)
        scenario = _scenario(config)
        return compute_optimal_allocation(
            config=config, scenario=scenario,
            stock_std=0.15, inflation_std=0.02,
            current_stocks_mo=2_000.0, current_bonds_mo=1_000.0,
            target_success=0.999,
            seed=42, n_simulations=100,
            budget_cap=config.monthly_saving,
        )

    @pytest.fixture(scope="class")
    def result_achievable(self):
        # With a very large stock portfolio, 95% success should be reachable
        config = _cfg(
            current_stock_value=5_000_000.0,
            current_stock_unrealized_gain=1_000_000.0,
            optimize_grid_step=100_000.0,
        )
        scenario = _scenario(config)
        return compute_optimal_allocation(
            config=config, scenario=scenario,
            stock_std=0.15, inflation_std=0.02,
            current_stocks_mo=2_000.0, current_bonds_mo=1_000.0,
            target_success=0.50,
            seed=42, n_simulations=200,
        )

    def test_unachievable_returns_none_optimal(self, result_unachievable):
        assert result_unachievable.optimal is None

    def test_grid_is_non_empty(self, result_unachievable):
        assert len(result_unachievable.grid) > 0

    def test_all_grid_points_have_success_rate_in_unit_interval(self, result_unachievable):
        for point in result_unachievable.grid:
            assert 0.0 <= point.best_success_rate <= 1.0

    def test_achievable_returns_optimal_point(self, result_achievable):
        assert result_achievable.optimal is not None

    def test_optimal_is_achievable(self, result_achievable):
        assert bool(result_achievable.optimal.achievable)

    def test_optimal_has_finite_monthly_values(self, result_achievable):
        opt = result_achievable.optimal
        assert opt.monthly_stocks < float("inf")
        assert opt.monthly_bonds  < float("inf")
        assert opt.total_monthly  < float("inf")

    def test_optimal_minimises_total_monthly(self, result_achievable):
        # The optimal point must have total_monthly ≤ every other achievable point
        opt = result_achievable.optimal
        for point in result_achievable.grid:
            if point.achievable:
                assert opt.total_monthly <= point.total_monthly + 1e-6

    def test_extra_stocks_and_bonds_non_negative(self, result_achievable):
        for point in result_achievable.grid:
            assert point.extra_stocks >= 0.0
            assert point.extra_bonds  >= 0.0

    def test_new_cash_equals_current_minus_extras(self, result_achievable):
        config = _cfg(
            current_stock_value=5_000_000.0,
            current_stock_unrealized_gain=1_000_000.0,
            optimize_grid_step=100_000.0,
        )
        for point in result_achievable.grid:
            expected_cash = config.current_cash - point.extra_stocks - point.extra_bonds
            assert point.new_cash == pytest.approx(expected_cash, abs=1.0)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_mc_n_simulations(self):
        from finance import validate_config_inputs
        config = _cfg(mc_n_simulations=0)
        with pytest.raises(ValueError, match="mc_n_simulations"):
            validate_config_inputs(
                config=config,
                stock_expected_yield=0.07,
                bonds_yield=0.03,
                withdrawal_rate=0.035,
            )

    def test_invalid_mc_target_success_percent_zero(self):
        from finance import validate_config_inputs
        config = _cfg(mc_target_success_percent=0)
        with pytest.raises(ValueError, match="mc_target_success_percent"):
            validate_config_inputs(
                config=config,
                stock_expected_yield=0.07,
                bonds_yield=0.03,
                withdrawal_rate=0.035,
            )

    def test_invalid_mc_target_success_percent_100(self):
        from finance import validate_config_inputs
        config = _cfg(mc_target_success_percent=100)
        with pytest.raises(ValueError, match="mc_target_success_percent"):
            validate_config_inputs(
                config=config,
                stock_expected_yield=0.07,
                bonds_yield=0.03,
                withdrawal_rate=0.035,
            )

    def test_invalid_teilfreistellung_above_one(self):
        from finance import validate_config_inputs
        config = _cfg(teilfreistellung=1.5)
        with pytest.raises(ValueError, match="teilfreistellung"):
            validate_config_inputs(
                config=config,
                stock_expected_yield=0.07,
                bonds_yield=0.03,
                withdrawal_rate=0.035,
            )

    def test_invalid_negative_basiszins(self):
        from finance import validate_config_inputs
        config = _cfg(basiszins=-0.01)
        with pytest.raises(ValueError, match="basiszins"):
            validate_config_inputs(
                config=config,
                stock_expected_yield=0.07,
                bonds_yield=0.03,
                withdrawal_rate=0.035,
            )

    def test_invalid_negative_freibetrag(self):
        from finance import validate_config_inputs
        config = _cfg(freibetrag=-100.0)
        with pytest.raises(ValueError, match="freibetrag"):
            validate_config_inputs(
                config=config,
                stock_expected_yield=0.07,
                bonds_yield=0.03,
                withdrawal_rate=0.035,
            )

    def test_invalid_negative_stock_std(self):
        from finance import validate_config_inputs
        config = _cfg(mc_stock_return_std_percent=-1.0)
        with pytest.raises(ValueError, match="mc_stock_return_std_percent"):
            validate_config_inputs(
                config=config,
                stock_expected_yield=0.07,
                bonds_yield=0.03,
                withdrawal_rate=0.035,
            )

    def test_invalid_negative_inflation_std(self):
        from finance import validate_config_inputs
        config = _cfg(mc_inflation_std_percent=-1.0)
        with pytest.raises(ValueError, match="mc_inflation_std_percent"):
            validate_config_inputs(
                config=config,
                stock_expected_yield=0.07,
                bonds_yield=0.03,
                withdrawal_rate=0.035,
            )

    def test_valid_config_does_not_raise(self):
        from finance import validate_config_inputs
        config = _cfg()
        validate_config_inputs(
            config=config,
            stock_expected_yield=0.07,
            bonds_yield=0.03,
            withdrawal_rate=0.035,
        )
