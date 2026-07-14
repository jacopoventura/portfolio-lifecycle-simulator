"""
Integration tests: run the full pipeline with fixed inputs and assert key outputs.

These tests serve two purposes:
  1. Regression guard — any change to finance.py / monte_carlo.py that shifts a
     numerical result will immediately be caught here.
  2. Correctness anchor — the key deterministic values (future income, cost basis,
     required portfolio) are independently verified below in comments so the expected
     values can be audited without running the code.

Config used throughout:
  years=15, inflation=2%, target=2500/mo, stock yield=6%, bonds=3%
  current_stock=80k, cost_basis=70k (80k-10k unrealised), cash=120k
  annual_saving=30k, no contribution growth, 50% stocks / 30% bonds / 20% cash
"""

import pytest

from finance import (
    calculate_future_income_value,
    cash_converted_to_bonds_at_horizon,
    run_projection,
    run_scenario,
)
from models import Config
from monte_carlo import (
    compute_years_to_financial_freedom_mc,
    run_monte_carlo_simulation,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config(
        current_stock_value=80_000.0,
        current_bonds_value=5_000.0,
        current_stock_unrealized_gain=10_000.0,
        current_cash=120_000.0,
        cash_reserve_at_horizon=15_000.0,
        interest_on_cash_pa=2.0,
        annual_net_saving=30_000.0,
        annual_contribution_growth_percent=0.0,
        stock_monthly_investment_pct=0.5,
        bonds_monthly_investment_pct=0.3,
        target_monthly_retirement_income=2_500.0,
        years=15.0,
        expected_inflation_percent=2.0,
        expected_yearly_growth_stocks_percent=6.0,
        bonds_percentage=50.0,
        yield_bonds=3.0,
        tax_rate=0.26,
        tax_rate_bond_italy=0.125,
        withdrawal_percentage_stock=3.5,
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


@pytest.fixture(scope="module")
def derived(cfg):
    stock_yield   = cfg.expected_yearly_growth_stocks_percent / 100
    bonds_yield   = cfg.yield_bonds / 100
    inflation     = cfg.expected_inflation_percent / 100
    withdrawal    = cfg.withdrawal_percentage_stock / 100

    fut_income       = calculate_future_income_value(cfg.target_monthly_retirement_income, cfg.years, inflation)
    target_annual_net = fut_income * 12
    projection       = run_projection(cfg, stock_yield)

    gain_stock = max(0.0, projection.final_stock_value - projection.cost_basis_stock)
    gain_frac  = gain_stock / projection.final_stock_value if projection.final_stock_value > 0 else 0.0

    scenario_de = run_scenario(
        name="DE",
        config=cfg,
        future_monthly_income_value=fut_income,
        target_annual_net_income=target_annual_net,
        stock_tax_rate=cfg.tax_rate,
        bond_tax_rate=cfg.tax_rate,
        stock_expected_yield=stock_yield,
        bonds_yield=bonds_yield,
        withdrawal_rate=withdrawal,
        stock_gain_fraction=gain_frac,
        stock_teilfreistellung=cfg.teilfreistellung,
        annual_contribution_growth=cfg.annual_contribution_growth_percent / 100,
    )
    scenario_it = run_scenario(
        name="IT",
        config=cfg,
        future_monthly_income_value=fut_income,
        target_annual_net_income=target_annual_net,
        stock_tax_rate=cfg.tax_rate,
        bond_tax_rate=cfg.tax_rate_bond_italy,
        stock_expected_yield=stock_yield,
        bonds_yield=bonds_yield,
        withdrawal_rate=withdrawal,
        stock_gain_fraction=gain_frac,
        stock_teilfreistellung=1.0,
        annual_contribution_growth=cfg.annual_contribution_growth_percent / 100,
    )

    cash_pct     = max(0.0, 1.0 - cfg.stock_monthly_investment_pct
                             - cfg.bonds_monthly_investment_pct)
    monthly_cash = cfg.annual_net_saving / 12 * cash_pct
    ctb = cash_converted_to_bonds_at_horizon(
        current_cash=cfg.current_cash,
        cash_rate=cfg.interest_on_cash_pa / 100,
        years=cfg.years,
        cash_reserve=cfg.cash_reserve_at_horizon,
        monthly_contribution=monthly_cash,
        contribution_growth_rate=cfg.annual_contribution_growth_percent / 100,
    )

    return dict(
        stock_yield=stock_yield,
        bonds_yield=bonds_yield,
        withdrawal=withdrawal,
        fut_income=fut_income,
        target_annual_net=target_annual_net,
        projection=projection,
        scenario_de=scenario_de,
        scenario_it=scenario_it,
        cash_to_bonds=ctb,
    )


# ---------------------------------------------------------------------------
# Step 1 — Inflation-adjusted income
# manual: 2500 × 1.02^15 = 3364.670846
# ---------------------------------------------------------------------------

class TestFutureIncomeValue:
    def test_golden_value(self, cfg, derived):
        assert derived["fut_income"] == pytest.approx(3_364.670846, rel=1e-6)

    def test_annual_target(self, cfg, derived):
        # 3364.67 × 12 = 40376.05
        assert derived["target_annual_net"] == pytest.approx(40_376.050150, rel=1e-6)

    def test_higher_inflation_increases_income(self, cfg):
        low  = calculate_future_income_value(2_500, 15, 0.01)
        high = calculate_future_income_value(2_500, 15, 0.04)
        assert high > low

    def test_longer_horizon_increases_income(self, cfg):
        short = calculate_future_income_value(2_500, 10, 0.02)
        long_ = calculate_future_income_value(2_500, 20, 0.02)
        assert long_ > short


# ---------------------------------------------------------------------------
# Step 2 — Deterministic projection
# cost_basis: (80k − 10k) + 1250/mo × 12 × 15 = 295,000 (verified manually)
# ---------------------------------------------------------------------------

class TestProjection:
    def test_cost_basis_no_contribution_growth(self, derived):
        # initial cost basis = 80k − 10k = 70k
        # annual stock contrib = 30k/yr × 0.5 × 15y = 225k  →  total = 295k
        assert derived["projection"].cost_basis_stock == pytest.approx(295_000.0, rel=1e-9)

    def test_final_stock_value(self, derived):
        assert derived["projection"].final_stock_value == pytest.approx(550_364.534, rel=1e-5)

    def test_final_bonds_value(self, derived):
        assert derived["projection"].final_bonds_value == pytest.approx(428_681.833, rel=1e-5)

    def test_cash_converted_to_bonds(self, derived):
        # cash reserve kept = 15k; rest converts to bonds
        assert derived["projection"].final_remaining_cash == pytest.approx(15_000.0, abs=1.0)
        assert derived["projection"].final_cash_converted_to_bonds == pytest.approx(251_212.439, rel=1e-5)

    def test_cash_to_bonds_helper_matches_projection(self, derived):
        assert derived["cash_to_bonds"] == pytest.approx(
            derived["projection"].final_cash_converted_to_bonds, rel=1e-9
        )

    def test_vorabpauschale_positive(self, derived):
        assert derived["projection"].total_vorabpauschale > 0
        assert derived["projection"].total_vorabpauschale_tax > 0

    def test_vorabpauschale_golden(self, derived):
        assert derived["projection"].total_vorabpauschale     == pytest.approx(43_731.844, rel=1e-4)
        assert derived["projection"].total_vorabpauschale_tax == pytest.approx( 4_094.916, rel=1e-4)

    def test_stocks_exceed_bonds(self, derived):
        # 50% to stocks monthly vs 30% to bonds → stocks should dominate
        proj = derived["projection"]
        assert proj.final_stock_value > proj.final_bonds_value


# ---------------------------------------------------------------------------
# Step 3 — Scenario required portfolio
# manually: net_rate ≈ 0.02712, required = 40376/0.02712 ≈ 1,488,672
# ---------------------------------------------------------------------------

class TestScenario:
    def test_required_total_portfolio_de(self, derived):
        assert derived["scenario_de"].required_total_portfolio == pytest.approx(1_488_672.38, rel=1e-4)

    def test_required_stock_and_bond_split(self, derived):
        sc = derived["scenario_de"]
        assert sc.required_stock_amount == pytest.approx(sc.required_total_portfolio * 0.5, rel=1e-6)
        assert sc.required_bonds_amount == pytest.approx(sc.required_total_portfolio * 0.5, rel=1e-6)

    def test_portfolio_net_income_rate(self, derived):
        assert derived["scenario_de"].portfolio_net_income_rate == pytest.approx(0.027122, rel=1e-3)

    def test_it_requires_smaller_portfolio_than_de(self, derived):
        # Lower bond tax rate in IT → higher net income rate → smaller required portfolio
        assert derived["scenario_it"].required_total_portfolio < derived["scenario_de"].required_total_portfolio

    def test_it_required_portfolio_golden(self, derived):
        # IT bond tax 12.5% vs DE 26% → net bond income higher → smaller portfolio needed
        sc_it = derived["scenario_it"]
        assert sc_it.required_total_portfolio < 1_488_672.38

    def test_monthly_contributions_positive(self, derived):
        sc = derived["scenario_de"]
        assert sc.required_monthly_contribution > 0
        assert sc.required_monthly_bonds_contribution > 0

    def test_required_monthly_contribution_golden(self, derived):
        assert derived["scenario_de"].required_monthly_contribution == pytest.approx(1_926.07, rel=1e-3)
        assert derived["scenario_de"].required_monthly_bonds_contribution == pytest.approx(2_145.22, rel=1e-3)


# ---------------------------------------------------------------------------
# Step 4 — Monte Carlo invariants and split metrics
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    @pytest.fixture(scope="class")
    def mc_de(self, cfg, derived):
        current_stock_mo = cfg.annual_net_saving / 12 * cfg.stock_monthly_investment_pct
        current_bonds_mo = cfg.annual_net_saving / 12 * cfg.bonds_monthly_investment_pct
        return run_monte_carlo_simulation(
            config=cfg, scenario=derived["scenario_de"],
            n_simulations=2_000,
            stock_std=cfg.mc_stock_return_std_percent / 100,
            inflation_std=cfg.mc_inflation_std_percent / 100,
            seed=42,
            monthly_stock_contribution_override=current_stock_mo,
            monthly_bonds_contribution_override=current_bonds_mo,
        )

    # --- structural invariants (never break regardless of config) -----------
    def test_all_rates_in_unit_interval(self, mc_de):
        assert 0.0 <= mc_de.success_rate <= 1.0
        assert 0.0 <= mc_de.success_rate_stock_only <= 1.0
        assert 0.0 <= mc_de.success_rate_bond_only <= 1.0
        assert 0.0 <= mc_de.success_rate_income_only <= 1.0

    def test_joint_rate_le_individual_rates(self, mc_de):
        # P(A ∩ B) ≤ P(A) and P(A ∩ B) ≤ P(B)
        assert mc_de.success_rate <= mc_de.success_rate_stock_only
        assert mc_de.success_rate <= mc_de.success_rate_bond_only

    def test_percentiles_are_ordered(self, cfg, mc_de):
        tail   = 100 - cfg.mc_target_success_percent
        target = cfg.mc_target_success_percent
        p = mc_de.final_portfolio_percentiles
        assert p[tail] <= p[25] <= p[50] <= p[75] <= p[target]

    def test_stock_and_bond_percentiles_positive(self, cfg, mc_de):
        assert mc_de.final_stock_percentiles[50] > 0
        assert mc_de.final_bond_percentiles[50] > 0

    # --- directional checks (higher stock = better stock success) -----------
    def test_large_stock_override_improves_stock_success(self, cfg, derived):
        common = dict(
            config=cfg, scenario=derived["scenario_de"],
            n_simulations=500,
            stock_std=cfg.mc_stock_return_std_percent / 100,
            inflation_std=cfg.mc_inflation_std_percent / 100,
            seed=42,
        )
        baseline = run_monte_carlo_simulation(**common)
        rich     = run_monte_carlo_simulation(**common, initial_stock_override=5_000_000.0)
        assert rich.success_rate_stock_only > baseline.success_rate_stock_only
        assert rich.success_rate_income_only > baseline.success_rate_income_only

    def test_large_bond_override_improves_bond_success(self, cfg, derived):
        common = dict(
            config=cfg, scenario=derived["scenario_de"],
            n_simulations=500,
            stock_std=cfg.mc_stock_return_std_percent / 100,
            inflation_std=cfg.mc_inflation_std_percent / 100,
            seed=42,
        )
        baseline = run_monte_carlo_simulation(**common)
        rich     = run_monte_carlo_simulation(**common, initial_bonds_override=5_000_000.0)
        assert rich.success_rate_bond_only > baseline.success_rate_bond_only

    # --- fixed-seed golden values (regression guard) -----------------------
    def test_golden_success_rate(self, mc_de):
        assert mc_de.success_rate == pytest.approx(0.0, abs=0.01)

    def test_golden_income_success_rate(self, mc_de):
        assert mc_de.success_rate_income_only == pytest.approx(0.0635, abs=0.02)

    def test_golden_stock_success_rate(self, mc_de):
        assert mc_de.success_rate_stock_only == pytest.approx(0.1815, abs=0.03)

    def test_golden_median_portfolio(self, mc_de):
        assert mc_de.final_portfolio_percentiles[50] == pytest.approx(930_270, rel=0.05)


# ---------------------------------------------------------------------------
# Step 5 — End-to-end pipeline consistency
# ---------------------------------------------------------------------------

class TestPipelineConsistency:
    def test_projection_stocks_exceed_scenario_requirement_at_high_yield(self):
        # With a high assumed yield, the projection should exceed the required stock amount
        cfg_high = Config(
            current_stock_value=80_000.0,
            current_bonds_value=5_000.0,
            current_stock_unrealized_gain=10_000.0,
            current_cash=120_000.0,
            cash_reserve_at_horizon=15_000.0,
            interest_on_cash_pa=2.0,
            annual_net_saving=30_000.0,
            annual_contribution_growth_percent=0.0,
            stock_monthly_investment_pct=0.5,
            bonds_monthly_investment_pct=0.3,
            target_monthly_retirement_income=2_500.0,
            years=15.0,
            expected_inflation_percent=2.0,
            expected_yearly_growth_stocks_percent=10.0,
            bonds_percentage=50.0,
            yield_bonds=3.0,
            tax_rate=0.26,
            tax_rate_bond_italy=0.125,
            withdrawal_percentage_stock=3.5,
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
        inflation  = cfg_high.expected_inflation_percent / 100
        stock_yield = cfg_high.expected_yearly_growth_stocks_percent / 100
        bonds_yield = cfg_high.yield_bonds / 100
        withdrawal  = cfg_high.withdrawal_percentage_stock / 100

        fut = calculate_future_income_value(cfg_high.target_monthly_retirement_income, cfg_high.years, inflation)
        proj = run_projection(cfg_high, stock_yield)

        gain_frac = max(0.0, proj.final_stock_value - proj.cost_basis_stock) / proj.final_stock_value
        sc = run_scenario(
            name="DE", config=cfg_high,
            future_monthly_income_value=fut,
            target_annual_net_income=fut * 12,
            stock_tax_rate=cfg_high.tax_rate,
            bond_tax_rate=cfg_high.tax_rate,
            stock_expected_yield=stock_yield,
            bonds_yield=bonds_yield,
            withdrawal_rate=withdrawal,
            stock_gain_fraction=gain_frac,
            stock_teilfreistellung=cfg_high.teilfreistellung,
            annual_contribution_growth=0.0,
        )
        assert proj.final_stock_value > sc.required_stock_amount

    def test_higher_saving_increases_mc_success(self, cfg, derived):
        cfg_rich = Config(
            **{**cfg.__dict__,
               "annual_net_saving": cfg.annual_net_saving * 3,
               "mc_n_simulations": 500},
        )
        stock_yield = cfg_rich.expected_yearly_growth_stocks_percent / 100
        bonds_yield  = cfg_rich.yield_bonds / 100
        withdrawal   = cfg_rich.withdrawal_percentage_stock / 100
        inflation    = cfg_rich.expected_inflation_percent / 100

        fut  = calculate_future_income_value(cfg_rich.target_monthly_retirement_income, cfg_rich.years, inflation)
        proj = run_projection(cfg_rich, stock_yield)
        gain_frac = max(0.0, proj.final_stock_value - proj.cost_basis_stock) / proj.final_stock_value if proj.final_stock_value > 0 else 0.0
        sc = run_scenario(
            name="DE", config=cfg_rich,
            future_monthly_income_value=fut,
            target_annual_net_income=fut * 12,
            stock_tax_rate=cfg_rich.tax_rate,
            bond_tax_rate=cfg_rich.tax_rate,
            stock_expected_yield=stock_yield,
            bonds_yield=bonds_yield,
            withdrawal_rate=withdrawal,
            stock_gain_fraction=gain_frac,
            stock_teilfreistellung=cfg_rich.teilfreistellung,
            annual_contribution_growth=0.0,
        )
        stock_mo = cfg_rich.annual_net_saving / 12 * cfg_rich.stock_monthly_investment_pct
        bonds_mo = cfg_rich.annual_net_saving / 12 * cfg_rich.bonds_monthly_investment_pct

        mc_base = run_monte_carlo_simulation(
            config=cfg, scenario=derived["scenario_de"],
            n_simulations=500,
            stock_std=cfg.mc_stock_return_std_percent / 100,
            inflation_std=cfg.mc_inflation_std_percent / 100,
            seed=42,
            monthly_stock_contribution_override=cfg.annual_net_saving / 12 * cfg.stock_monthly_investment_pct,
            monthly_bonds_contribution_override=cfg.annual_net_saving / 12 * cfg.bonds_monthly_investment_pct,
        )
        mc_rich = run_monte_carlo_simulation(
            config=cfg_rich, scenario=sc,
            n_simulations=500,
            stock_std=cfg_rich.mc_stock_return_std_percent / 100,
            inflation_std=cfg_rich.mc_inflation_std_percent / 100,
            seed=42,
            monthly_stock_contribution_override=stock_mo,
            monthly_bonds_contribution_override=bonds_mo,
        )
        assert mc_rich.success_rate_income_only > mc_base.success_rate_income_only
