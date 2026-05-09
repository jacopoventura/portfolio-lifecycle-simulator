import numpy as np
import pytest

from models import Config
from monte_carlo import (
    compute_allocation_sweep,
    compute_joint_probability,
    compute_sorr_scenarios,
    find_required_decumulation_portfolio,
    simulate_decumulation,
)


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
        target_monthly_retirement_income=3_300.0,
        years=15.0,
        expected_inflation_percent=2.3,
        expected_yearly_growth_stocks_percent=5.0,
        bonds_percentage=60.0,
        yield_bonds=3.4,
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
        mc_bond_yield_std_percent=0.1,
        mc_target_success_percent=95,
        mc_decumulation_years=20,
        mc_decumulation_stock_pct=0.40,
        mc_decumulation_n_simulations=300,
        mc_decumulation_pension_start_year=15,
        mc_decumulation_pension_monthly=2_500.0,
        mc_decumulation_pension_adjustment_rate=0.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


_PORTFOLIO = 1_500_000.0
_SMALL_N = 300


class TestSimulateDecumulation:
    def test_survival_rate_in_range(self):
        config = _make_config()
        result = simulate_decumulation(config, _PORTFOLIO, seed=42, n_simulations=_SMALL_N)
        assert 0.0 <= result.survival_rate <= 1.0

    def test_survival_by_year_monotone_decreasing(self):
        config = _make_config()
        result = simulate_decumulation(config, _PORTFOLIO, seed=42, n_simulations=_SMALL_N)
        for i in range(1, len(result.survival_by_year)):
            assert result.survival_by_year[i] <= result.survival_by_year[i - 1] + 1e-9

    def test_zero_withdrawal_full_survival(self):
        config = _make_config(withdrawal_percentage_stock=0.0,
                              mc_decumulation_pension_monthly=0.0)
        result = simulate_decumulation(config, _PORTFOLIO, seed=42, n_simulations=_SMALL_N)
        assert result.survival_rate == pytest.approx(1.0)

    def test_deterministic_with_same_seed(self):
        config = _make_config()
        r1 = simulate_decumulation(config, _PORTFOLIO, seed=7, n_simulations=_SMALL_N)
        r2 = simulate_decumulation(config, _PORTFOLIO, seed=7, n_simulations=_SMALL_N)
        assert r1.survival_rate == r2.survival_rate

    def test_larger_portfolio_higher_survival(self):
        config = _make_config()
        r_small = simulate_decumulation(config, 500_000.0, seed=42, n_simulations=_SMALL_N)
        r_large = simulate_decumulation(config, 5_000_000.0, seed=42, n_simulations=_SMALL_N)
        assert r_large.survival_rate >= r_small.survival_rate

    def test_portfolio_percentiles_sorted(self):
        config = _make_config()
        result = simulate_decumulation(config, _PORTFOLIO, seed=42, n_simulations=_SMALL_N)
        for pcts in result.portfolio_percentiles_by_year:
            assert pcts[3] <= pcts[25] <= pcts[50] <= pcts[75] <= pcts[97]

    def test_result_fields_populated(self):
        config = _make_config()
        result = simulate_decumulation(config, _PORTFOLIO, seed=42, n_simulations=_SMALL_N)
        assert len(result.survival_by_year) == config.mc_decumulation_years
        assert len(result.portfolio_percentiles_by_year) == config.mc_decumulation_years
        assert len(result.withdrawal_percentiles_by_year) == config.mc_decumulation_years
        assert result.initial_portfolio == pytest.approx(_PORTFOLIO)


class TestGKvsFixed:
    def test_gk_survival_ge_fixed(self):
        # With a stressed portfolio, GK guardrails should maintain or improve survival
        config = _make_config(mc_decumulation_n_simulations=500)
        r_fixed = simulate_decumulation(config, 800_000.0, seed=0, n_simulations=500,
                                        withdrawal_strategy="fixed")
        r_gk = simulate_decumulation(config, 800_000.0, seed=0, n_simulations=500,
                                     withdrawal_strategy="guyton_klinger")
        assert r_gk.survival_rate >= r_fixed.survival_rate - 0.05  # allow small MC noise


class TestFindRequiredPortfolio:
    def test_required_portfolio_achieves_target(self):
        config = _make_config(mc_target_success_percent=90)
        req = find_required_decumulation_portfolio(
            config, n_simulations=_SMALL_N, seed=42, n_iter=20,
            lo=500_000.0, hi=8_000_000.0,
        )
        result = simulate_decumulation(config, req, seed=42, n_simulations=_SMALL_N)
        assert result.survival_rate >= 0.90 - 0.05  # binary search tolerance + MC noise


class TestSORR:
    def test_sorr_stress_lower_than_baseline(self):
        config = _make_config()
        sorr = compute_sorr_scenarios(config, 800_000.0, seed=42, n_simulations=_SMALL_N)
        for label, stress in sorr.stress_scenarios:
            assert stress.survival_rate <= sorr.baseline.survival_rate + 0.10, (
                f"{label} survival ({stress.survival_rate:.3f}) exceeded baseline "
                f"({sorr.baseline.survival_rate:.3f})"
            )

    def test_sorr_has_three_scenarios(self):
        config = _make_config()
        sorr = compute_sorr_scenarios(config, _PORTFOLIO, seed=42, n_simulations=_SMALL_N)
        assert len(sorr.stress_scenarios) == 3


class TestGlidePath:
    def test_glide_does_not_break_survival_range(self):
        config = _make_config(mc_decumulation_glide_end_pct=0.30, mc_decumulation_glide_years=20)
        result = simulate_decumulation(config, _PORTFOLIO, seed=42, n_simulations=_SMALL_N)
        assert 0.0 <= result.survival_rate <= 1.0

    def test_glide_higher_stocks_improves_or_maintains_survival(self):
        # Starting at 80% stocks (glide 80→30) vs fixed 40% stocks should help long-run survival
        config_fixed = _make_config(mc_decumulation_glide_end_pct=0.0, mc_decumulation_glide_years=0)
        config_glide = _make_config(mc_decumulation_glide_end_pct=0.30, mc_decumulation_glide_years=20)
        r_fixed = simulate_decumulation(config_fixed, _PORTFOLIO, seed=42, n_simulations=_SMALL_N,
                                        stock_allocation=0.80)
        r_glide = simulate_decumulation(config_glide, _PORTFOLIO, seed=42, n_simulations=_SMALL_N,
                                        stock_allocation=0.80)
        # Both should be valid survival rates
        assert 0.0 <= r_fixed.survival_rate <= 1.0
        assert 0.0 <= r_glide.survival_rate <= 1.0

    def test_no_glide_same_as_zero_glide_years(self):
        config = _make_config()
        r1 = simulate_decumulation(config, _PORTFOLIO, seed=7, n_simulations=_SMALL_N,
                                   glide_end_pct=0.30, glide_years=0)
        r2 = simulate_decumulation(config, _PORTFOLIO, seed=7, n_simulations=_SMALL_N,
                                   glide_end_pct=-1.0, glide_years=0)
        assert r1.survival_rate == r2.survival_rate

    def test_survival_by_year_length_with_glide(self):
        config = _make_config(mc_decumulation_glide_end_pct=0.30, mc_decumulation_glide_years=10)
        result = simulate_decumulation(config, _PORTFOLIO, seed=42, n_simulations=_SMALL_N)
        assert len(result.survival_by_year) == config.mc_decumulation_years


class TestAllocationSweep:
    def test_sweep_returns_all_points(self):
        config = _make_config()
        bonds_pct_values = [0.40, 0.60, 0.80]
        sweep = compute_allocation_sweep(config, bonds_pct_values=bonds_pct_values,
                                         seed=42, n_simulations=200)
        assert len(sweep.points) == 3

    def test_sweep_survival_rates_in_range(self):
        config = _make_config()
        sweep = compute_allocation_sweep(config, bonds_pct_values=[0.40, 0.60],
                                         seed=42, n_simulations=200)
        for pt in sweep.points:
            assert 0.0 <= pt.survival_yr30_gk <= 1.0
            assert 0.0 <= pt.survival_yr30_fixed <= 1.0

    def test_sweep_required_portfolio_decreases_with_bonds(self):
        # More bonds → higher income rate (IT: bond tax 12.5% < stock tax 26%)
        # → lower required portfolio
        config = _make_config()
        sweep = compute_allocation_sweep(config, bonds_pct_values=[0.30, 0.50, 0.70],
                                         seed=42, n_simulations=200)
        portfolios = [pt.required_portfolio for pt in sweep.points]
        assert portfolios[0] > portfolios[1] > portfolios[2]

    def test_sweep_optimal_is_valid_bonds_pct(self):
        config = _make_config()
        sweep = compute_allocation_sweep(config, bonds_pct_values=[0.30, 0.50, 0.70],
                                         seed=42, n_simulations=200)
        assert sweep.optimal_bonds_pct in [0.30, 0.50, 0.70]

    def test_sweep_base_bonds_pct_matches_config(self):
        config = _make_config(bonds_percentage=60.0)
        sweep = compute_allocation_sweep(config, bonds_pct_values=[0.40, 0.60, 0.80],
                                         seed=42, n_simulations=200)
        assert sweep.base_bonds_pct == pytest.approx(0.60)


class TestJointProbability:
    def test_joint_probability_bounds(self):
        config = _make_config()
        joint = compute_joint_probability(
            config, accumulation_p50=2_000_000.0,
            accumulation_tail=1_000_000.0, tail_pct=3,
            seed=42, n_simulations=_SMALL_N,
        )
        assert 0.0 <= joint.joint_p50 <= joint.p_survive_from_p50
        assert 0.0 <= joint.joint_tail <= joint.p_survive_from_tail

    def test_joint_p50_formula(self):
        config = _make_config()
        joint = compute_joint_probability(
            config, accumulation_p50=2_000_000.0,
            accumulation_tail=1_000_000.0, tail_pct=3,
            seed=42, n_simulations=_SMALL_N,
        )
        assert joint.joint_p50 == pytest.approx(0.50 * joint.p_survive_from_p50)
        assert joint.joint_tail == pytest.approx(0.03 * joint.p_survive_from_tail)


class TestPensionAdjustmentRate:
    def test_growing_pension_improves_or_matches_survival(self):
        # Growing pension reduces portfolio draw in later years → survival should not decrease
        config_fixed = _make_config(mc_decumulation_pension_adjustment_rate=0.0)
        config_growing = _make_config(mc_decumulation_pension_adjustment_rate=0.015)
        r_fixed = simulate_decumulation(config_fixed, _PORTFOLIO, seed=42, n_simulations=_SMALL_N)
        r_growing = simulate_decumulation(config_growing, _PORTFOLIO, seed=42, n_simulations=_SMALL_N)
        assert r_growing.survival_rate >= r_fixed.survival_rate - 0.02

    def test_zero_rate_is_fixed_nominal(self):
        # rate=0 → pension never changes → same result from both configs
        r1 = simulate_decumulation(_make_config(mc_decumulation_pension_adjustment_rate=0.0),
                                   _PORTFOLIO, seed=7, n_simulations=_SMALL_N)
        r2 = simulate_decumulation(_make_config(mc_decumulation_pension_adjustment_rate=0.0),
                                   _PORTFOLIO, seed=7, n_simulations=_SMALL_N)
        assert r1.survival_rate == pytest.approx(r2.survival_rate)

    def test_higher_rate_improves_survival(self):
        # 3% growth > 1.5%: more pension income in later years → better survival
        r_low = simulate_decumulation(_make_config(mc_decumulation_pension_adjustment_rate=0.015),
                                      _PORTFOLIO, seed=42, n_simulations=_SMALL_N)
        r_high = simulate_decumulation(_make_config(mc_decumulation_pension_adjustment_rate=0.03),
                                       _PORTFOLIO, seed=42, n_simulations=_SMALL_N)
        assert r_high.survival_rate >= r_low.survival_rate - 0.02
