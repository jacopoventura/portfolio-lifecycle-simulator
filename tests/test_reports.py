"""
Smoke tests for all report functions.
Each test verifies that the function runs without raising and produces output.
Fixture data is minimal but structurally correct.
"""

import pytest

from models import (
    AllocationGridPoint,
    BondStockCell,
    BondStockGrid,
    CagrContributionCell,
    CombinedGrid,
    CombinedGridCell,
    Config,
    ContributionCagrGrid,
    FinancialFreedomResult,
    MonteCarloResult,
    OptimalAllocationResult,
    ProjectionResult,
    ProjectionYieldScenarioResult,
    ScenarioResult,
)
from reports import (
    _cagr_line,
    _contribution_cagr_frontier_table,
    _find_best_combined,
    _find_min_contrib,
    _find_min_contrib_stock,
    _fmt_cash,
    _fmt_stocks,
    _print_bond_stock_grid,
    _print_cagr_grid_verbose,
    _print_section,
    _print_subsection,
    _print_top_allocations,
    _q_analysis_header,
    _total_invested,
    clear_terminal,
    format_currency,
    format_currency_signed,
    format_percent_value,
    format_ratio_as_percent,
    format_ratio_delta_as_percentage_points,
    print_combined_q1_analysis,
    print_current_situation_mc,
    print_financial_freedom_report,
    print_monte_carlo_report,
    print_monte_carlo_suggestions,
    print_optimal_allocation_report,
    print_plan_b_hard,
    print_plan_b_income,
    print_projection_report,
    print_projection_yield_table,
    print_q1_analysis,
    print_q1a_analysis,
    print_q1b_analysis,
    print_q2_analysis,
    print_scenarios_report,
    print_sensitivity_table,
    print_stock_allocation_sweep,
)


# ---------------------------------------------------------------------------
# Shared builders
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


def _scenario(name: str = "DE") -> ScenarioResult:
    return ScenarioResult(
        name=name,
        stock_tax_rate=0.26,
        bond_tax_rate=0.26,
        stock_gain_fraction=0.3,
        effective_stock_tax_rate=0.054,
        stock_teilfreistellung=0.7,
        years=10.0,
        current_stock_value=100_000.0,
        stock_expected_yield=0.07,
        bonds_percentage=40.0,
        yield_bonds=0.03,
        withdrawal_rate=0.035,
        portfolio_gross_income_rate=0.04,
        portfolio_net_income_rate=0.032,
        future_monthly_income_value=2_438.0,
        target_annual_net_income=29_256.0,
        required_annual_gross_income_from_portfolio=30_000.0,
        required_total_portfolio=750_000.0,
        required_stock_amount=450_000.0,
        required_bonds_amount=300_000.0,
        generated_annual_gross_income=30_000.0,
        generated_annual_net_income=29_000.0,
        required_monthly_contribution=1_200.0,
        required_monthly_bonds_contribution=800.0,
    )


_PCTS = {5: 0.0, 25: 50_000.0, 50: 100_000.0, 75: 150_000.0, 95: 200_000.0}
_SURPLUS = {5: -50_000.0, 25: 0.0, 50: 50_000.0, 75: 100_000.0, 95: 150_000.0}


def _mc(name: str = "DE") -> MonteCarloResult:
    return MonteCarloResult(
        name=name,
        n_simulations=100,
        stock_std=0.15,
        inflation_std=0.02,
        bond_std=0.01,
        success_rate=0.80,
        final_portfolio_percentiles=dict(_PCTS),
        required_portfolio_percentiles={k: 80_000.0 for k in _PCTS},
        surplus_percentiles=dict(_SURPLUS),
        portfolio_net_income_rate=0.032,
        final_stock_percentiles=dict(_PCTS),
        final_bond_percentiles={k: v * 0.4 for k, v in _PCTS.items()},
        success_rate_income_only=0.85,
        success_rate_stock_only=0.82,
        success_rate_bond_only=0.78,
    )


def _projection() -> ProjectionResult:
    return ProjectionResult(
        years=10.0,
        stock_expected_yield=0.07,
        cash_interest_rate=0.02,
        annual_net_saving=36_000.0,
        annual_contribution_growth_percent=0.0,
        maximum_stock_monthly_investment_pct=0.5,
        maximum_bonds_monthly_investment_pct=0.3,
        initial_stock_value=100_000.0,
        initial_stock_unrealized_gain=10_000.0,
        initial_cash_value=80_000.0,
        monthly_stock_contribution=1_500.0,
        monthly_bonds_contribution=900.0,
        monthly_cash_contribution=600.0,
        final_stock_value=350_000.0,
        final_bonds_from_contributions=120_000.0,
        final_cash_before_conversion=40_000.0,
        final_cash_converted_to_bonds=30_000.0,
        final_remaining_cash=10_000.0,
        final_bonds_value=150_000.0,
        final_portfolio_value=500_000.0,
        cost_basis_stock=280_000.0,
        total_vorabpauschale=5_000.0,
        total_vorabpauschale_tax=900.0,
    )


def _proj_yield(name: str = "DE") -> ProjectionYieldScenarioResult:
    return ProjectionYieldScenarioResult(
        name=name,
        stock_tax_rate=0.26,
        bond_tax_rate=0.26,
        annual_gross_income_stock=10_000.0,
        annual_net_income_stock=8_500.0,
        annual_gross_income_bond=4_000.0,
        annual_net_income_bond=3_000.0,
        annual_gross_income_total=14_000.0,
        annual_net_income_total=11_500.0,
        bond_net_contribution_share=0.26,
        stock_gain_fraction=0.3,
        effective_stock_tax_rate=0.054,
        portfolio_gross_yield=0.028,
        portfolio_net_yield=0.023,
    )


def _ff(years: int = 8) -> FinancialFreedomResult:
    return FinancialFreedomResult(
        years=years,
        success_rate=0.95,
        portfolio_p50=600_000.0,
        portfolio_tail_pct=400_000.0,
        tail_pct=5,
        monthly_income_p50=1_600.0,
    )


def _allocation_point(extra_stocks: float = 0.0, extra_bonds: float = 0.0) -> AllocationGridPoint:
    cash = 80_000.0 - extra_stocks - extra_bonds
    return AllocationGridPoint(
        extra_stocks=extra_stocks,
        extra_bonds=extra_bonds,
        new_stocks=100_000.0 + extra_stocks,
        new_bonds=10_000.0 + extra_bonds,
        new_cash=cash,
        monthly_stocks=1_500.0,
        monthly_bonds=900.0,
        total_monthly=2_400.0,
        achievable=True,
        best_success_rate=0.85,
    )


def _optimal_result(name: str = "DE") -> OptimalAllocationResult:
    grid = [_allocation_point(0.0), _allocation_point(10_000.0), _allocation_point(20_000.0)]
    return OptimalAllocationResult(
        scenario_name=name,
        target_success=0.95,
        min_cash_today=10_000.0,
        grid=grid,
        optimal=grid[0],
    )


def _cagr_grid(name: str = "DE", multiplier: int = 1) -> ContributionCagrGrid:
    cagr_values = [0.05, 0.07, 0.09]
    contribution_values = [500.0, 1_000.0, 1_500.0]
    cell = CagrContributionCell(
        success_hard=0.70, success_income=0.80,
        success_stock_only=0.75, stock_tail_pct=300_000.0,
    )
    grid = [[cell] * len(contribution_values) for _ in cagr_values]
    return ContributionCagrGrid(
        scenario_name=name,
        bonds_monthly=500.0 * multiplier,
        bonds_multiplier=multiplier,
        cagr_values=cagr_values,
        contribution_values=contribution_values,
        suggested_contribution=1_000.0,
        grid=grid,
    )


def _bond_stock_grid(name: str = "DE") -> BondStockGrid:
    stocks = [0.0, 500.0, 1_000.0]
    bonds = [300.0, 600.0, 900.0]
    cell = BondStockCell(success_bond=0.75)
    grid = [[cell] * len(stocks) for _ in bonds]
    return BondStockGrid(
        scenario_name=name,
        stocks_values=stocks,
        bonds_values=bonds,
        required_bonds=200_000.0,
        suggested_stocks=1_000.0,
        suggested_bonds=600.0,
        grid=grid,
    )


def _combined_grid(name: str = "DE") -> CombinedGrid:
    cagr_values = [0.05, 0.07]
    stocks = [0.0, 500.0, 1_000.0]
    bonds = [300.0, 600.0]
    cell = CombinedGridCell(success_combined=0.65)
    grid = [[[cell] * len(stocks) for _ in bonds] for _ in cagr_values]
    return CombinedGrid(
        scenario_name=name,
        cagr_values=cagr_values,
        stocks_values=stocks,
        bonds_values=bonds,
        required_stocks=400_000.0,
        required_bonds=200_000.0,
        suggested_stocks=1_000.0,
        suggested_bonds=600.0,
        grid=grid,
    )


# ---------------------------------------------------------------------------
# formatters
# ---------------------------------------------------------------------------

class TestFormatters:
    def test_format_currency(self):
        assert format_currency(1234.5) == "1,234.50"

    def test_format_currency_signed_positive(self):
        assert format_currency_signed(100.0).startswith("+")

    def test_format_currency_signed_negative(self):
        assert format_currency_signed(-100.0).startswith("-")

    def test_format_ratio_as_percent(self):
        assert format_ratio_as_percent(0.075) == "7.50%"

    def test_format_percent_value(self):
        assert format_percent_value(7.5) == "7.50%"

    def test_format_ratio_delta_as_percentage_points(self):
        result = format_ratio_delta_as_percentage_points(0.01)
        assert "pp" in result

    def test_print_section_produces_output(self, capsys):
        _print_section(1, "Test Title")
        out = capsys.readouterr().out
        assert "Test Title" in out

    def test_print_subsection_produces_output(self, capsys):
        _print_subsection("Sub Title")
        out = capsys.readouterr().out
        assert "Sub Title" in out


# ---------------------------------------------------------------------------
# section1
# ---------------------------------------------------------------------------

class TestSection1:
    def test_print_scenarios_report_runs(self, capsys):
        s1 = _scenario("DE")
        s2 = _scenario("IT")
        print_scenarios_report(
            s1=s1, s2=s2,
            current_bonds_value=10_000.0,
            current_cash=80_000.0,
            cash_interest_rate_pa=2.0,
            estimated_final_cash=95_000.0,
            cash_to_bonds_at_horizon=30_000.0,
            cash_reserve_at_horizon=10_000.0,
            annual_net_saving=36_000.0,
        )
        assert capsys.readouterr().out != ""

    def test_print_scenarios_report_verbose(self, capsys):
        s1 = _scenario("DE")
        s2 = _scenario("IT")
        print_scenarios_report(
            s1=s1, s2=s2,
            current_bonds_value=10_000.0,
            current_cash=80_000.0,
            cash_interest_rate_pa=2.0,
            estimated_final_cash=95_000.0,
            cash_to_bonds_at_horizon=30_000.0,
            cash_reserve_at_horizon=10_000.0,
            annual_net_saving=36_000.0,
            verbose=True,
        )
        assert capsys.readouterr().out != ""

    def test_print_sensitivity_table_runs(self, capsys):
        print_sensitivity_table(
            config=_cfg(),
            base_inflation_rate=0.02,
            base_bonds_yield=0.03,
            base_withdrawal_rate=0.035,
            stock_expected_yield=0.07,
            stock_gain_fraction=0.3,
        )
        assert capsys.readouterr().out != ""


# ---------------------------------------------------------------------------
# section2
# ---------------------------------------------------------------------------

class TestSection2:
    def test_print_monte_carlo_report_runs(self, capsys):
        print_monte_carlo_report(
            mc1=_mc("DE"), mc2=_mc("IT"),
            current_stock_mo=1_500.0, current_bonds_mo=900.0,
            config=_cfg(),
        )
        assert capsys.readouterr().out != ""

    def test_print_monte_carlo_report_verbose(self, capsys):
        print_monte_carlo_report(
            mc1=_mc("DE"), mc2=_mc("IT"),
            current_stock_mo=1_500.0, current_bonds_mo=900.0,
            config=_cfg(), verbose=True,
        )
        assert capsys.readouterr().out != ""

    def test_print_monte_carlo_suggestions_runs(self, capsys):
        scan_rows = [
            {
                "stocks_mo": 0.0, "cash_mo": 3_000.0, "rate": 0.70,
                "rate_income": 0.72, "median_stocks": 200_000.0,
                "median_bonds": 100_000.0, "p75_bonds": 110_000.0,
                "p50_target_income": 30_000.0, "is_current": False,
            },
            {
                "stocks_mo": 1_500.0, "cash_mo": 600.0, "rate": 0.82,
                "rate_income": 0.85, "median_stocks": 350_000.0,
                "median_bonds": 120_000.0, "p75_bonds": 130_000.0,
                "p50_target_income": 30_000.0, "is_current": True,
            },
        ]
        print_monte_carlo_suggestions(
            scan_rows=scan_rows,
            scenario=_scenario("DE"),
            current_stock_mean_percent=7.0,
            config=_cfg(),
            fixed_monthly_bonds=900.0,
        )
        assert capsys.readouterr().out != ""

    def test_cagr_line_achievable(self):
        line = _cagr_line("Scenario 1", 0.08, 7.0)
        assert "8.0%" in line

    def test_cagr_line_none(self):
        line = _cagr_line("Scenario 1", None, 7.0)
        assert "not achievable" in line

    def test_print_plan_b_hard_runs(self, capsys):
        print_plan_b_hard(cagr_1=0.08, cagr_2=None, config=_cfg())
        assert capsys.readouterr().out != ""

    def test_print_plan_b_income_runs(self, capsys):
        print_plan_b_income(cagr_1=0.07, cagr_2=0.065, config=_cfg())
        assert capsys.readouterr().out != ""

    def test_print_stock_allocation_sweep_runs(self, capsys):
        sweep = [
            {"extra": 0.0, "stocks_t0": 100_000.0, "cash_t0": 80_000.0, "rate_1": 0.80, "rate_2": 0.82},
            {"extra": 50_000.0, "stocks_t0": 150_000.0, "cash_t0": 30_000.0, "rate_1": 0.85, "rate_2": 0.87},
        ]
        print_stock_allocation_sweep(stock_sweep=sweep, config=_cfg())
        assert capsys.readouterr().out != ""

    def test_find_min_contrib_found(self):
        grid = _cagr_grid()
        result = _find_min_contrib(grid, cagr_idx=0, use_hard=False, target_success=0.5)
        assert result is not None
        assert result == 500.0

    def test_find_min_contrib_not_found(self):
        grid = _cagr_grid()
        result = _find_min_contrib(grid, cagr_idx=0, use_hard=True, target_success=0.999)
        assert result is None

    def test_fmt_stocks_found(self):
        s = _fmt_stocks(1_000.0, 12)
        assert "1,000" in s

    def test_fmt_stocks_none(self):
        s = _fmt_stocks(None, 12)
        assert "n/a" in s

    def test_fmt_cash_found(self):
        s = _fmt_cash(500.0, 300.0, 3_000.0, 12)
        assert s.strip() != ""

    def test_fmt_cash_none(self):
        s = _fmt_cash(None, 300.0, 3_000.0, 12)
        assert "—" in s

    def test_contribution_cagr_frontier_table_runs(self, capsys):
        grids = [_cagr_grid("DE", 1), _cagr_grid("DE", 2)]
        _contribution_cagr_frontier_table(
            grids=grids, monthly_saving=3_000.0,
            goal_label="Test goal", use_hard=True, target_success=0.95,
        )
        assert capsys.readouterr().out != ""

    def test_print_cagr_grid_verbose_runs(self, capsys):
        _print_cagr_grid_verbose(grid=_cagr_grid(), target_success=0.95)
        assert capsys.readouterr().out != ""

    def test_q_analysis_header_runs(self, capsys):
        _q_analysis_header(monthly_saving=3_000.0, target_success_pct=95)
        assert capsys.readouterr().out != ""

    def test_print_q1_analysis_runs(self, capsys):
        grids = [_cagr_grid("DE", 1), _cagr_grid("DE", 2)]
        print_q1_analysis(
            grids_de=grids, grids_it=grids,
            monthly_saving=3_000.0, target_success=0.95,
        )
        assert capsys.readouterr().out != ""

    def test_print_q2_analysis_runs(self, capsys):
        grids = [_cagr_grid("DE", 1), _cagr_grid("DE", 2)]
        print_q2_analysis(
            grids_de=grids, grids_it=grids,
            monthly_saving=3_000.0, target_success=0.95,
        )
        assert capsys.readouterr().out != ""

    def test_find_min_contrib_stock_found(self):
        grid = _cagr_grid()
        result = _find_min_contrib_stock(grid, cagr_idx=0, target_success=0.5)
        assert result is not None
        assert result[0] == 500.0

    def test_find_min_contrib_stock_not_found(self):
        grid = _cagr_grid()
        result = _find_min_contrib_stock(grid, cagr_idx=0, target_success=0.999)
        assert result is None

    def test_total_invested_no_growth(self):
        result = _total_invested(monthly=1_000.0, years=10.0, growth=0.0, current_cost_basis=50_000.0)
        assert result == pytest.approx(50_000.0 + 1_000.0 * 12 * 10.0)

    def test_total_invested_with_growth(self):
        result = _total_invested(monthly=1_000.0, years=10.0, growth=0.05, current_cost_basis=50_000.0)
        assert result > 50_000.0 + 1_000.0 * 12 * 10.0

    def test_print_q1a_analysis_runs(self, capsys):
        grids = [_cagr_grid("DE", 1), _cagr_grid("DE", 2)]
        print_q1a_analysis(
            grids_de=grids, grids_it=grids,
            config=_cfg(), target_success=0.95,
        )
        assert capsys.readouterr().out != ""

    def test_print_bond_stock_grid_runs(self, capsys):
        _print_bond_stock_grid(
            grid=_bond_stock_grid(),
            current_stock_mo=500.0, current_bonds_mo=600.0,
            monthly_saving=3_000.0, target_success=0.95,
        )
        assert capsys.readouterr().out != ""

    def test_print_q1b_analysis_runs(self, capsys):
        print_q1b_analysis(
            bond_grid_de=_bond_stock_grid("DE"),
            bond_grid_it=_bond_stock_grid("IT"),
            current_stock_mo=500.0, current_bonds_mo=600.0,
            monthly_saving=3_000.0, target_success=0.95,
        )
        assert capsys.readouterr().out != ""

    def test_find_best_combined(self):
        grid = _combined_grid()
        result = _find_best_combined(grid, cagr_idx=0)
        assert len(result) == 3
        stocks_mo, bonds_mo, rate = result
        assert rate >= 0.0

    def test_print_combined_q1_analysis_runs(self, capsys):
        print_combined_q1_analysis(
            grid_de=_combined_grid("DE"),
            grid_it=_combined_grid("IT"),
            config=_cfg(), target_success=0.95,
        )
        assert capsys.readouterr().out != ""


# ---------------------------------------------------------------------------
# section3
# ---------------------------------------------------------------------------

class TestSection3:
    def test_print_projection_report_runs(self, capsys):
        print_projection_report(
            result=_projection(),
            scenario_de=_scenario("DE"),
            scenario_it=_scenario("IT"),
        )
        assert capsys.readouterr().out != ""

    def test_print_financial_freedom_report_runs(self, capsys):
        print_financial_freedom_report(
            ff_de=_ff(8), ff_it=_ff(9), config=_cfg()
        )
        assert capsys.readouterr().out != ""

    def test_print_financial_freedom_report_not_achievable(self, capsys):
        ff_none = FinancialFreedomResult(
            years=None, success_rate=0.0,
            portfolio_p50=0.0, portfolio_tail_pct=0.0,
            tail_pct=5, monthly_income_p50=0.0,
        )
        print_financial_freedom_report(ff_de=ff_none, ff_it=ff_none, config=_cfg())
        out = capsys.readouterr().out
        assert "not achievable" in out

    def test_print_projection_yield_table_runs(self, capsys):
        print_projection_yield_table(
            scenario_1=_proj_yield("DE"),
            scenario_2=_proj_yield("IT"),
            portfolio_value=500_000.0,
        )
        assert capsys.readouterr().out != ""

    def test_print_current_situation_mc_runs(self, capsys):
        print_current_situation_mc(
            mc1=_mc("DE"), mc2=_mc("IT"),
            current_stock_mo=1_500.0, current_bonds_mo=900.0,
            config=_cfg(),
        )
        assert capsys.readouterr().out != ""

    def test_print_top_allocations_runs(self, capsys):
        _print_top_allocations(result=_optimal_result())
        assert capsys.readouterr().out != ""

    def test_print_optimal_allocation_report_runs(self, capsys):
        print_optimal_allocation_report(
            result_1=_optimal_result("DE"),
            result_2=_optimal_result("IT"),
            config=_cfg(),
            n_simulations_per_point=5_000,
        )
        assert capsys.readouterr().out != ""

    def test_print_optimal_allocation_report_no_optimal(self, capsys):
        result = OptimalAllocationResult(
            scenario_name="DE", target_success=0.95,
            min_cash_today=10_000.0, grid=[], optimal=None,
        )
        print_optimal_allocation_report(
            result_1=result, result_2=result,
            config=_cfg(), n_simulations_per_point=5_000,
        )
        out = capsys.readouterr().out
        assert "not achievable" in out

    def test_print_optimal_allocation_report_verbose(self, capsys):
        print_optimal_allocation_report(
            result_1=_optimal_result("DE"),
            result_2=_optimal_result("IT"),
            config=_cfg(),
            n_simulations_per_point=5_000,
            verbose=True,
        )
        assert capsys.readouterr().out != ""
