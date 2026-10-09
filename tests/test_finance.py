import math
import pytest

from finance import (
    calculate_future_income_value,
    calculate_required_monthly_contribution,
    cash_converted_to_bonds_at_horizon,
    run_projection,
    run_scenario,
    validate_config_inputs,
    validate_projection_inputs,
)
from models import Config


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
        years=1.0,
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


class TestCalculateFutureIncomeValue:
    def test_no_inflation(self):
        assert calculate_future_income_value(1_000.0, 10, 0.0) == pytest.approx(1_000.0)

    def test_zero_years(self):
        assert calculate_future_income_value(1_000.0, 0, 0.05) == pytest.approx(1_000.0)

    def test_known_value(self):
        expected = 1_000.0 * (1.02 ** 10)
        assert calculate_future_income_value(1_000.0, 10, 0.02) == pytest.approx(expected, rel=1e-9)


class TestCalculateRequiredMonthlyContribution:
    def test_zero_when_target_already_met(self):
        assert calculate_required_monthly_contribution(
            current_amount=1_000_000.0, target_amount=100.0, years=10, annual_yield=0.07
        ) == pytest.approx(0.0)

    def test_zero_rate_simple_division(self):
        # 0% yield, no growth: monthly = (target - current) / months
        result = calculate_required_monthly_contribution(
            current_amount=0.0, target_amount=1_200.0, years=1, annual_yield=0.0
        )
        assert result == pytest.approx(100.0, rel=1e-9)

    def test_zero_rate_with_initial_amount(self):
        result = calculate_required_monthly_contribution(
            current_amount=500.0, target_amount=1_200.0, years=1, annual_yield=0.0
        )
        assert result == pytest.approx(700.0 / 12, rel=1e-9)

    def test_positive_yield_reduces_required_contribution(self):
        c_no_yield = calculate_required_monthly_contribution(0.0, 10_000.0, 5, 0.0)
        c_with_yield = calculate_required_monthly_contribution(0.0, 10_000.0, 5, 0.07)
        assert c_with_yield < c_no_yield

    def test_contribution_growth_reduces_first_payment(self):
        c_flat = calculate_required_monthly_contribution(0.0, 10_000.0, 5, 0.07, 0.0)
        c_growing = calculate_required_monthly_contribution(0.0, 10_000.0, 5, 0.07, 0.03)
        assert c_growing < c_flat


class TestCashConvertedToBondsAtHorizon:
    def test_no_rate_no_reserve(self):
        result = cash_converted_to_bonds_at_horizon(10_000.0, 0.0, 5, 0.0)
        assert result == pytest.approx(10_000.0)

    def test_no_rate_with_reserve(self):
        result = cash_converted_to_bonds_at_horizon(10_000.0, 0.0, 5, 2_000.0)
        assert result == pytest.approx(8_000.0)

    def test_below_reserve_returns_zero(self):
        result = cash_converted_to_bonds_at_horizon(1_000.0, 0.0, 5, 2_000.0)
        assert result == pytest.approx(0.0)

    def test_positive_rate_grows_cash(self):
        result_0 = cash_converted_to_bonds_at_horizon(10_000.0, 0.0, 5, 0.0)
        result_r = cash_converted_to_bonds_at_horizon(10_000.0, 0.02, 5, 0.0)
        assert result_r > result_0

    def test_with_monthly_contribution(self):
        # monthly_contribution adds to the final amount
        result_no_contrib = cash_converted_to_bonds_at_horizon(10_000.0, 0.0, 1, 0.0)
        result_contrib = cash_converted_to_bonds_at_horizon(10_000.0, 0.0, 1, 0.0, monthly_contribution=100.0)
        assert result_contrib == pytest.approx(result_no_contrib + 100.0 * 12, rel=1e-6)


class TestValidateConfigInputs:
    def test_years_zero_raises(self):
        config = _make_config(years=0.0)
        with pytest.raises(ValueError, match="Years"):
            validate_config_inputs(config, 0.07, 0.03, 0.033)

    def test_invalid_bonds_percentage_raises(self):
        config = _make_config(bonds_percentage=150.0)
        with pytest.raises(ValueError, match="bonds_percentage"):
            validate_config_inputs(config, 0.07, 0.03, 0.033)

    def test_invalid_tax_rate_raises(self):
        config = _make_config(tax_rate=1.5)
        with pytest.raises(ValueError, match="tax_rate"):
            validate_config_inputs(config, 0.07, 0.03, 0.033)

    def test_negative_stock_value_raises(self):
        config = _make_config(current_stock_value=-1.0, current_stock_unrealized_gain=0.0)
        with pytest.raises(ValueError, match="current_stock_value"):
            validate_config_inputs(config, 0.07, 0.03, 0.033)

    def test_valid_config_does_not_raise(self):
        config = _make_config()
        validate_config_inputs(config, 0.07, 0.03, 0.033)  # must not raise


class TestValidateProjectionInputs:
    def test_reserve_exceeds_cash_clamps_to_current_cash(self):
        config = _make_config(cash_reserve_at_horizon=100_000.0)
        validate_projection_inputs(config, 0.02)  # warns, does not raise
        assert config.cash_reserve_at_horizon == config.current_cash

    def test_negative_reserve_raises(self):
        config = _make_config(cash_reserve_at_horizon=-1.0)
        with pytest.raises(ValueError, match="cash_reserve_at_horizon"):
            validate_projection_inputs(config, 0.02)

    def test_allocation_sum_exceeds_one_raises(self):
        config = _make_config(
            stock_monthly_investment_pct=0.9,
            bonds_monthly_investment_pct=0.2,
        )
        with pytest.raises(ValueError):
            validate_projection_inputs(config, 0.02)

    def test_negative_saving_raises(self):
        config = _make_config(annual_net_saving=-1.0)
        with pytest.raises(ValueError, match="annual_net_saving"):
            validate_projection_inputs(config, 0.02)

    def test_valid_config_does_not_raise(self):
        config = _make_config()
        validate_projection_inputs(config, 0.02)  # must not raise


class TestRunScenario:
    def _scenario(self, config=None, **kwargs):
        config = config or _make_config(years=10.0)
        defaults = dict(
            name="TEST",
            config=config,
            future_monthly_income_value=3_600.0,
            target_annual_net_income=43_200.0,
            stock_tax_rate=0.26,
            bond_tax_rate=0.26,
            stock_expected_yield=0.07,
            bonds_yield=0.03,
            withdrawal_rate=0.033,
            stock_gain_fraction=0.3,
            stock_teilfreistellung=0.7,
            annual_contribution_growth=0.0,
        )
        defaults.update(kwargs)
        return run_scenario(**defaults)

    def test_portfolio_composition(self):
        result = self._scenario()
        assert result.required_stock_amount + result.required_bonds_amount == pytest.approx(
            result.required_total_portfolio, rel=1e-9
        )

    def test_net_income_matches_target(self):
        target = 43_200.0
        result = self._scenario(target_annual_net_income=target)
        assert result.generated_annual_net_income == pytest.approx(target, rel=1e-9)

    def test_contributions_nonnegative(self):
        result = self._scenario()
        assert result.required_monthly_contribution >= 0.0
        assert result.required_monthly_bonds_contribution >= 0.0

    def test_higher_target_requires_higher_portfolio(self):
        r_low = self._scenario(target_annual_net_income=30_000.0)
        r_high = self._scenario(target_annual_net_income=60_000.0)
        assert r_high.required_total_portfolio > r_low.required_total_portfolio

    def test_net_rate_positive(self):
        result = self._scenario()
        assert result.portfolio_net_income_rate > 0.0


class TestRunProjection:
    def test_stock_grows_with_contributions(self):
        config = _make_config()
        result = run_projection(config, stock_expected_yield=0.07)
        assert result.final_stock_value > config.current_stock_value

    def test_cost_basis_is_initial_plus_contributions(self):
        config = _make_config()
        result = run_projection(config, stock_expected_yield=0.07)
        initial_cost_basis = config.current_stock_value - config.current_stock_unrealized_gain
        monthly_stock = config.annual_net_saving / 12 * config.stock_monthly_investment_pct
        expected_cost_basis = initial_cost_basis + monthly_stock * 12  # 1 year, no growth
        assert result.cost_basis_stock == pytest.approx(expected_cost_basis, rel=1e-9)

    def test_cash_converted_and_remaining(self):
        config = _make_config()
        result = run_projection(config, stock_expected_yield=0.07)
        # Cash at horizon is well above reserve, so remaining cash = reserve
        assert result.final_remaining_cash == pytest.approx(config.cash_reserve_at_horizon, rel=1e-6)
        assert result.final_cash_converted_to_bonds > 0.0

    def test_monthly_contributions_sum_to_saving(self):
        config = _make_config()
        result = run_projection(config, stock_expected_yield=0.07)
        monthly_saving = config.annual_net_saving / 12
        total = (result.monthly_stock_contribution
                 + result.monthly_bonds_contribution
                 + result.monthly_cash_contribution)
        assert total == pytest.approx(monthly_saving, rel=1e-9)

    def test_vorabpauschale_nonnegative(self):
        config = _make_config()
        result = run_projection(config, stock_expected_yield=0.07)
        assert result.total_vorabpauschale >= 0.0
        assert result.total_vorabpauschale_tax >= 0.0


class TestValidateConfigInputsMCAdvanced:
    """Validation of advanced MC parameters added for P7/P8/P8b/P8c."""

    def _call(self, config):
        validate_config_inputs(config, 0.07, 0.03, 0.033)

    def test_correlation_plus_one_raises(self):
        with pytest.raises(ValueError, match="mc_stock_bond_correlation"):
            self._call(_make_config(mc_stock_bond_correlation=1.0))

    def test_correlation_minus_one_raises(self):
        with pytest.raises(ValueError, match="mc_stock_bond_correlation"):
            self._call(_make_config(mc_stock_bond_correlation=-1.0))

    def test_correlation_out_of_range_raises(self):
        with pytest.raises(ValueError, match="mc_stock_bond_correlation"):
            self._call(_make_config(mc_stock_bond_correlation=1.5))

    def test_correlation_valid_does_not_raise(self):
        self._call(_make_config(mc_stock_bond_correlation=-0.2))

    def test_bond_mean_reversion_zero_raises(self):
        with pytest.raises(ValueError, match="mc_bond_yield_mean_reversion"):
            self._call(_make_config(mc_bond_yield_mean_reversion=0.0))

    def test_bond_mean_reversion_above_one_raises(self):
        with pytest.raises(ValueError, match="mc_bond_yield_mean_reversion"):
            self._call(_make_config(mc_bond_yield_mean_reversion=1.5))

    def test_bond_inflation_beta_negative_raises(self):
        with pytest.raises(ValueError, match="mc_bond_yield_inflation_beta"):
            self._call(_make_config(mc_bond_yield_inflation_beta=-0.1))

    def test_bond_model_valid_does_not_raise(self):
        self._call(_make_config(mc_bond_yield_mean_reversion=0.15, mc_bond_yield_inflation_beta=0.3))

    def test_stock_return_df_le_2_raises(self):
        with pytest.raises(ValueError, match="mc_stock_return_df"):
            self._call(_make_config(mc_stock_return_df=2.0))

    def test_stock_return_df_exactly_2_raises(self):
        with pytest.raises(ValueError, match="mc_stock_return_df"):
            self._call(_make_config(mc_stock_return_df=2.0))

    def test_stock_return_df_above_2_valid(self):
        self._call(_make_config(mc_stock_return_df=5.0))

    def test_stock_ar1_phi_at_plus_one_raises(self):
        with pytest.raises(ValueError, match="mc_stock_return_ar1_phi"):
            self._call(_make_config(mc_stock_return_ar1_phi=1.0))

    def test_stock_ar1_phi_at_minus_one_raises(self):
        with pytest.raises(ValueError, match="mc_stock_return_ar1_phi"):
            self._call(_make_config(mc_stock_return_ar1_phi=-1.0))

    def test_stock_ar1_phi_valid(self):
        self._call(_make_config(mc_stock_return_ar1_phi=-0.25))

    def test_inflation_ar1_phi_at_one_raises(self):
        with pytest.raises(ValueError, match="mc_inflation_ar1_phi"):
            self._call(_make_config(mc_inflation_ar1_phi=1.0))

    def test_inflation_ar1_phi_valid(self):
        self._call(_make_config(mc_inflation_ar1_phi=0.6))

    def test_negative_mean_uncertainty_raises(self):
        with pytest.raises(ValueError, match="mc_stock_mean_uncertainty"):
            self._call(_make_config(mc_stock_mean_uncertainty=-0.01))

    def test_negative_vol_uncertainty_raises(self):
        with pytest.raises(ValueError, match="mc_stock_vol_uncertainty"):
            self._call(_make_config(mc_stock_vol_uncertainty=-0.01))

    def test_decumulation_years_zero_raises(self):
        with pytest.raises(ValueError, match="mc_decumulation_years"):
            self._call(_make_config(mc_decumulation_years=0))

    def test_decumulation_stock_pct_out_of_range_raises(self):
        with pytest.raises(ValueError, match="mc_decumulation_stock_pct"):
            self._call(_make_config(mc_decumulation_stock_pct=1.5))

    def test_decumulation_glide_end_out_of_range_raises(self):
        with pytest.raises(ValueError, match="mc_decumulation_glide_end_pct"):
            self._call(_make_config(mc_decumulation_glide_end_pct=-0.1))

    def test_negative_glide_years_raises(self):
        with pytest.raises(ValueError, match="mc_decumulation_glide_years"):
            self._call(_make_config(mc_decumulation_glide_years=-1))

    def test_negative_pension_raises(self):
        with pytest.raises(ValueError, match="mc_decumulation_pension_monthly"):
            self._call(_make_config(mc_decumulation_pension_monthly=-100.0))

    def test_valid_full_config_does_not_raise(self):
        self._call(_make_config(
            mc_stock_bond_correlation=-0.2,
            mc_stock_return_df=5.0,
            mc_stock_return_ar1_phi=-0.25,
            mc_inflation_ar1_phi=0.6,
            mc_stock_mean_uncertainty=0.015,
            mc_stock_vol_uncertainty=0.03,
            mc_decumulation_years=40,
            mc_decumulation_stock_pct=0.4,
            mc_decumulation_glide_end_pct=0.3,
            mc_decumulation_glide_years=20,
            mc_decumulation_pension_monthly=2500.0,
        ))
