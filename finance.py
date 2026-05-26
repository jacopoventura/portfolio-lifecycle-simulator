from models import (
    Config,
    ProjectionResult,
    ProjectionYieldScenarioResult,
    ScenarioResult,
)
from src.private_saving_plan import calculate_yearly_vorabpauschale


def validate_config_inputs(
    config: Config,
    stock_expected_yield: float,
    bonds_yield: float,
    withdrawal_rate: float,
) -> None:
    if config.years <= 0:
        raise ValueError("Years must be greater than 0.")
    if stock_expected_yield <= -1:
        raise ValueError("expected_yearly_growth_stocks_percent must be greater than -100.")
    if bonds_yield <= -1:
        raise ValueError("yield_bonds must be greater than -100.")
    if not 0 <= config.bonds_percentage <= 100:
        raise ValueError("bonds_percentage must be between 0 and 100.")
    if not 0 <= config.tax_rate < 1:
        raise ValueError("tax_rate must be between 0 and 1 (for example 0.26).")
    if not 0 <= config.tax_rate_bond_italy < 1:
        raise ValueError("tax_rate_bond_italy must be between 0 and 1 (for example 0.125).")
    if withdrawal_rate <= 0:
        raise ValueError("withdrawal_percentage_stock must be greater than 0.")
    if config.current_stock_value < 0:
        raise ValueError("current_stock_value must be greater than or equal to 0.")
    if config.current_bonds_value < 0:
        raise ValueError("current_bonds_value must be greater than or equal to 0.")
    if not 0 <= config.current_stock_unrealized_gain <= config.current_stock_value:
        raise ValueError("current_stock_unrealized_gain must be between 0 and current_stock_value.")
    if not 0 <= config.teilfreistellung <= 1:
        raise ValueError("teilfreistellung must be between 0 and 1.")
    if config.basiszins < 0:
        raise ValueError("basiszins must be greater than or equal to 0.")
    if config.freibetrag < 0:
        raise ValueError("freibetrag must be greater than or equal to 0.")
    if config.mc_n_simulations <= 0:
        raise ValueError("mc_n_simulations must be greater than 0.")
    if not 1 <= config.mc_target_success_percent <= 99:
        raise ValueError("mc_target_success_percent must be an integer between 1 and 99.")
    if config.mc_stock_return_std_percent < 0:
        raise ValueError("mc_stock_return_std_percent must be greater than or equal to 0.")
    if config.mc_inflation_std_percent < 0:
        raise ValueError("mc_inflation_std_percent must be greater than or equal to 0.")
    if config.mc_bond_yield_std_percent < 0:
        raise ValueError("mc_bond_yield_std_percent must be >= 0.")
    if not (-1.0 < config.mc_inflation_ar1_phi < 1.0):
        raise ValueError("mc_inflation_ar1_phi must be strictly between -1 and 1 (stationarity requires |φ| < 1).")
    if not (-1.0 < config.mc_stock_bond_correlation < 1.0):
        raise ValueError("mc_stock_bond_correlation must be strictly between -1 and 1.")
    if config.mc_stock_return_df <= 2.0:
        raise ValueError("mc_stock_return_df must be > 2 (t-distribution variance is infinite for df ≤ 2).")
    if not (-1.0 < config.mc_stock_return_ar1_phi < 1.0):
        raise ValueError("mc_stock_return_ar1_phi must be strictly between -1 and 1 (stationarity requires |φ| < 1).")
    if config.mc_stock_mean_uncertainty < 0:
        raise ValueError("mc_stock_mean_uncertainty must be >= 0.")
    if config.mc_stock_vol_uncertainty < 0:
        raise ValueError("mc_stock_vol_uncertainty must be >= 0.")
    if config.mc_decumulation_years <= 0:
        raise ValueError("mc_decumulation_years must be > 0.")
    if config.mc_decumulation_n_simulations <= 0:
        raise ValueError("mc_decumulation_n_simulations must be > 0.")
    if not (0.0 <= config.mc_decumulation_stock_pct <= 1.0):
        raise ValueError("mc_decumulation_stock_pct must be between 0 and 1.")
    if not (0.0 <= config.mc_decumulation_glide_end_pct <= 1.0):
        raise ValueError("mc_decumulation_glide_end_pct must be between 0 and 1.")
    if config.mc_decumulation_glide_years < 0:
        raise ValueError("mc_decumulation_glide_years must be >= 0.")
    if config.mc_decumulation_pension_monthly < 0:
        raise ValueError("mc_decumulation_pension_monthly must be >= 0.")
    if config.mc_decumulation_pension_start_year < 0:
        raise ValueError("mc_decumulation_pension_start_year must be >= 0.")


def validate_projection_inputs(config: Config, cash_interest_rate: float) -> None:
    if config.current_cash < 0:
        raise ValueError("current_cash must be greater than or equal to 0.")
    if not 0 <= config.cash_reserve_at_horizon <= config.current_cash:
        raise ValueError(
            f"cash_reserve_at_horizon ({config.cash_reserve_at_horizon:,.0f}) must be between 0 "
            f"and current_cash ({config.current_cash:,.0f})."
        )
    if config.annual_net_saving < 0:
        raise ValueError("annual_net_saving must be greater than or equal to 0.")
    if not 0 <= config.maximum_stock_monthly_investment_pct <= 1:
        raise ValueError("maximum_stock_monthly_investment_pct must be between 0 and 1.")
    if not 0 <= config.maximum_bonds_monthly_investment_pct <= 1:
        raise ValueError("maximum_bonds_monthly_investment_pct must be between 0 and 1.")
    total_pct = config.maximum_stock_monthly_investment_pct + config.maximum_bonds_monthly_investment_pct
    if total_pct > 1.0 + 1e-9:
        raise ValueError(
            f"maximum_stock_monthly_investment_pct + maximum_bonds_monthly_investment_pct "
            f"({total_pct:.4f}) must be <= 1.0  (remainder goes to cash)."
        )
    if cash_interest_rate <= -1:
        raise ValueError("interest_on_cash_pa must be greater than -100.")
    months_float = config.years * 12
    if abs(months_float - round(months_float)) > 1e-9:
        raise ValueError("years must map to whole months for projection (steps of 1/12).")


def _annuity_factor(monthly_rate: float, monthly_growth: float, months: float) -> float:
    if abs(monthly_rate - monthly_growth) < 1e-12:
        return months * (1 + monthly_rate) ** (months - 1)
    return (
        ((1 + monthly_rate) ** months - (1 + monthly_growth) ** months)
        / (monthly_rate - monthly_growth)
    )


def calculate_future_income_value(income_value: float, years: float, inflation_rate: float) -> float:
    return income_value * (1 + inflation_rate) ** years


def calculate_required_total_portfolio(
    target_annual_net_income: float,
    net_income_rate: float,
) -> float:
    return target_annual_net_income / net_income_rate


def calculate_required_monthly_contribution(
    current_amount: float,
    target_amount: float,
    years: float,
    annual_yield: float,
    annual_contribution_growth: float = 0.0,
) -> float:
    months = years * 12
    monthly_rate = (1 + annual_yield) ** (1 / 12) - 1
    monthly_growth = (1 + annual_contribution_growth) ** (1 / 12) - 1

    future_amount_without_contributions = current_amount * (1 + monthly_rate) ** months
    if target_amount <= future_amount_without_contributions:
        return 0.0

    remaining = target_amount - future_amount_without_contributions

    return remaining / _annuity_factor(monthly_rate, monthly_growth, months)


def cash_converted_to_bonds_at_horizon(
    current_cash: float,
    cash_rate: float,
    years: float,
    cash_reserve: float,
    monthly_contribution: float = 0.0,
    contribution_growth_rate: float = 0.0,
) -> float:
    months = int(round(years * 12))
    monthly_rate = (1 + cash_rate) ** (1 / 12) - 1
    fv_initial = current_cash * (1 + monthly_rate) ** months
    if monthly_contribution > 0:
        monthly_growth = (1 + contribution_growth_rate) ** (1 / 12) - 1
        fv_contributions = monthly_contribution * _annuity_factor(monthly_rate, monthly_growth, months)
    else:
        fv_contributions = 0.0
    return max(0.0, fv_initial + fv_contributions - cash_reserve)


def required_portfolio_and_contribution(
    target_annual_net_income: float,
    bonds_percentage: float,
    bonds_yield: float,
    withdrawal_rate: float,
    stock_gain_fraction: float,
    stock_tax_rate: float,
    bond_tax_rate: float,
    stock_teilfreistellung: float,
    current_stock_value: float,
    current_bonds_value: float,
    cash_to_bonds_at_horizon: float,
    years: float,
    stock_expected_yield: float,
    annual_contribution_growth: float,
) -> tuple[float, float, float]:
    bonds_weight = bonds_percentage / 100
    stocks_weight = 1 - bonds_weight
    effective_stock_tax = stock_gain_fraction * stock_teilfreistellung * stock_tax_rate
    net_rate = (
        bonds_weight * bonds_yield * (1 - bond_tax_rate)
        + stocks_weight * withdrawal_rate * (1 - effective_stock_tax)
    )
    if net_rate <= 0:
        return float("inf"), float("inf"), float("inf")
    required_portfolio = target_annual_net_income / net_rate
    required_stocks = required_portfolio * stocks_weight
    required_bonds = required_portfolio * bonds_weight
    monthly_stocks = calculate_required_monthly_contribution(
        current_amount=current_stock_value,
        target_amount=required_stocks,
        years=years,
        annual_yield=stock_expected_yield,
        annual_contribution_growth=annual_contribution_growth,
    )
    adjusted_bonds_target = max(0.0, required_bonds - cash_to_bonds_at_horizon)
    monthly_bonds = calculate_required_monthly_contribution(
        current_amount=current_bonds_value,
        target_amount=adjusted_bonds_target,
        years=years,
        annual_yield=bonds_yield,
        annual_contribution_growth=annual_contribution_growth,
    )
    return required_portfolio, monthly_stocks, monthly_bonds


def run_scenario(
    name: str,
    config: Config,
    future_monthly_income_value: float,
    target_annual_net_income: float,
    stock_tax_rate: float,
    bond_tax_rate: float,
    stock_expected_yield: float,
    bonds_yield: float,
    withdrawal_rate: float,
    stock_gain_fraction: float,
    stock_teilfreistellung: float = 1.0,
    annual_contribution_growth: float = 0.0,
) -> ScenarioResult:
    """Build a ScenarioResult sized to the income-formula required portfolio
    (target_annual_net_income ÷ portfolio_net_income_rate)."""
    bonds_weight = config.bonds_percentage / 100
    stocks_weight = 1 - bonds_weight
    effective_stock_tax_rate = stock_gain_fraction * stock_teilfreistellung * stock_tax_rate
    portfolio_gross_income_rate = (
        (bonds_weight * bonds_yield)
        + (stocks_weight * withdrawal_rate)
    )
    portfolio_net_income_rate = (
        (bonds_weight * bonds_yield * (1 - bond_tax_rate))
        + (stocks_weight * withdrawal_rate * (1 - effective_stock_tax_rate))
    )
    if portfolio_gross_income_rate <= 0:
        raise ValueError("Portfolio gross income rate must be greater than 0.")
    if portfolio_net_income_rate <= 0:
        raise ValueError("Portfolio net income rate must be greater than 0.")

    required_total_portfolio = calculate_required_total_portfolio(
        target_annual_net_income=target_annual_net_income,
        net_income_rate=portfolio_net_income_rate,
    )
    required_bonds_amount = required_total_portfolio * bonds_weight
    required_stock_amount = required_total_portfolio * stocks_weight
    required_monthly_contribution = calculate_required_monthly_contribution(
        current_amount=config.current_stock_value,
        target_amount=required_stock_amount,
        years=config.years,
        annual_yield=stock_expected_yield,
        annual_contribution_growth=annual_contribution_growth,
    )
    cash_pct = max(0.0, 1.0 - config.maximum_stock_monthly_investment_pct - config.maximum_bonds_monthly_investment_pct)
    monthly_cash_contrib = config.annual_net_saving / 12 * cash_pct
    cash_to_bonds = cash_converted_to_bonds_at_horizon(
        current_cash=config.current_cash,
        cash_rate=config.interest_on_cash_pa / 100,
        years=config.years,
        cash_reserve=config.cash_reserve_at_horizon,
        monthly_contribution=monthly_cash_contrib,
        contribution_growth_rate=annual_contribution_growth,
    )
    adjusted_bonds_target = max(0.0, required_bonds_amount - cash_to_bonds)
    required_monthly_bonds_contribution = calculate_required_monthly_contribution(
        current_amount=config.current_bonds_value,
        target_amount=adjusted_bonds_target,
        years=config.years,
        annual_yield=bonds_yield,
        annual_contribution_growth=annual_contribution_growth,
    )
    generated_annual_gross_income = required_total_portfolio * portfolio_gross_income_rate
    generated_annual_net_income = required_total_portfolio * portfolio_net_income_rate

    return ScenarioResult(
        name=name,
        stock_tax_rate=stock_tax_rate,
        bond_tax_rate=bond_tax_rate,
        stock_gain_fraction=stock_gain_fraction,
        effective_stock_tax_rate=effective_stock_tax_rate,
        stock_teilfreistellung=stock_teilfreistellung,
        years=config.years,
        current_stock_value=config.current_stock_value,
        stock_expected_yield=stock_expected_yield,
        bonds_percentage=config.bonds_percentage,
        yield_bonds=bonds_yield,
        withdrawal_rate=withdrawal_rate,
        portfolio_gross_income_rate=portfolio_gross_income_rate,
        portfolio_net_income_rate=portfolio_net_income_rate,
        future_monthly_income_value=future_monthly_income_value,
        target_annual_net_income=target_annual_net_income,
        required_annual_gross_income_from_portfolio=generated_annual_gross_income,
        required_total_portfolio=required_total_portfolio,
        required_stock_amount=required_stock_amount,
        required_bonds_amount=required_bonds_amount,
        generated_annual_gross_income=generated_annual_gross_income,
        generated_annual_net_income=generated_annual_net_income,
        required_monthly_contribution=required_monthly_contribution,
        required_monthly_bonds_contribution=required_monthly_bonds_contribution,
    )


def compute_it_required_portfolio(
    config: Config,
    *,
    stock_expected_yield: float | None = None,
) -> float:
    """Canonical IT required portfolio at horizon (income-formula bar).

    Single source of truth for the "expected portfolio IT" / "required portfolio
    IT" number printed by main_portfolio_projection's §1. Equals
    `target_annual_inflated / portfolio_net_income_rate` where the net yield
    uses IT tax parameters (bond_tax_rate = tax_rate_bond_italy 12.5%,
    stock_teilfreistellung = 1.0, stock_gain_fraction from the deterministic
    projection at retirement).

    This is what the IT scenario aims at — *not* what it delivers (that's
    `compute_it_deterministic_projection`).
    """
    inflation_rate = config.expected_inflation_percent / 100.0
    stock_yield = stock_expected_yield if stock_expected_yield is not None else config.planning_cagr
    bonds_yield = config.yield_bonds / 100.0
    withdrawal_rate = config.withdrawal_percentage_stock / 100.0
    bonds_weight = config.bonds_percentage / 100.0
    stocks_weight = 1.0 - bonds_weight

    future_monthly_income = calculate_future_income_value(
        income_value=config.target_monthly_retirement_income,
        years=config.years,
        inflation_rate=inflation_rate,
    )
    target_annual_net_income = future_monthly_income * 12.0

    # Stock gain fraction at retirement, from the deterministic projection.
    # Survives config.years==0 because run_projection's loop just skips.
    projection = run_projection(config, stock_yield)
    gain_stock = max(0.0, projection.final_stock_value - projection.cost_basis_stock)
    stock_gain_fraction = (
        gain_stock / projection.final_stock_value if projection.final_stock_value > 0 else 0.0
    )

    # IT portfolio_net_income_rate (mirrors run_scenario's IT branch). Inlined
    # here so callers with config.years==0 (e.g. pillar 4's "retire today" probe)
    # don't trip run_scenario's required-monthly-contribution divide-by-zero.
    effective_stock_tax_rate = stock_gain_fraction * 1.0 * config.tax_rate  # IT teilfreistellung=1.0
    portfolio_net_income_rate = (
        bonds_weight * bonds_yield * (1.0 - config.tax_rate_bond_italy)
        + stocks_weight * withdrawal_rate * (1.0 - effective_stock_tax_rate)
    )
    if portfolio_net_income_rate <= 0:
        return float("inf")
    return target_annual_net_income / portfolio_net_income_rate


def compute_it_deterministic_projection(
    config: Config,
    *,
    stock_expected_yield: float | None = None,
    monthly_stock_contribution_override: float | None = None,
    monthly_bonds_contribution_override: float | None = None,
) -> float:
    """Canonical deterministic projection of total wealth at retirement (IT scenario).

    Single source of truth for "expected portfolio at horizon". Mirrors the
    wealth-tracking loop in `run_projection` (monthly compounding, contribution
    step-up, cash-above-reserve→bonds at horizon), but returns just the scalar
    and accepts contribution overrides (used by decumulation Q2 back-solve).

    When called with `stock_expected_yield=None` and no overrides, returns
    exactly `run_projection(config, config.planning_cagr).final_portfolio_value`
    by construction.
    """
    cash_interest_rate = config.interest_on_cash_pa / 100.0
    bonds_yield = config.yield_bonds / 100.0
    stock_yield = stock_expected_yield if stock_expected_yield is not None else config.planning_cagr

    months = int(round(config.years * 12))
    stock_m = (1 + stock_yield) ** (1 / 12) - 1
    bond_m = (1 + bonds_yield) ** (1 / 12) - 1
    cash_m = (1 + cash_interest_rate) ** (1 / 12) - 1

    monthly_net_saving = config.annual_net_saving / 12
    init_stock_c = (
        monthly_stock_contribution_override
        if monthly_stock_contribution_override is not None
        else monthly_net_saving * config.maximum_stock_monthly_investment_pct
    )
    init_bonds_c = (
        monthly_bonds_contribution_override
        if monthly_bonds_contribution_override is not None
        else monthly_net_saving * config.maximum_bonds_monthly_investment_pct
    )
    init_cash_c = max(0.0, monthly_net_saving - init_stock_c - init_bonds_c)

    contribution_growth_rate = config.annual_contribution_growth_percent / 100.0
    stock_v = float(config.current_stock_value)
    bond_v = float(config.current_bonds_value)
    cash_v = float(config.current_cash)

    for month in range(months):
        year = month // 12
        g = (1 + contribution_growth_rate) ** year
        stock_v = stock_v * (1 + stock_m) + init_stock_c * g
        bond_v = bond_v * (1 + bond_m) + init_bonds_c * g
        cash_v = cash_v * (1 + cash_m) + init_cash_c * g

    cash_to_bonds = max(0.0, cash_v - config.cash_reserve_at_horizon)
    remaining_cash = cash_v - cash_to_bonds
    return stock_v + bond_v + cash_to_bonds + remaining_cash


def run_projection(config: Config, stock_expected_yield: float) -> ProjectionResult:
    cash_interest_rate = config.interest_on_cash_pa / 100
    validate_projection_inputs(config, cash_interest_rate)

    months = int(round(config.years * 12))
    stock_monthly_rate = (1 + stock_expected_yield) ** (1 / 12) - 1
    cash_monthly_rate = (1 + cash_interest_rate) ** (1 / 12) - 1

    stock_value = config.current_stock_value
    cash_value = config.current_cash

    bonds_yield = config.yield_bonds / 100
    bond_monthly_rate = (1 + bonds_yield) ** (1 / 12) - 1

    contribution_growth_rate = config.annual_contribution_growth_percent / 100
    monthly_net_saving = config.annual_net_saving / 12
    initial_monthly_stock_contribution = monthly_net_saving * config.maximum_stock_monthly_investment_pct
    initial_monthly_bonds_contribution = monthly_net_saving * config.maximum_bonds_monthly_investment_pct
    initial_monthly_cash_contribution = (
        monthly_net_saving - initial_monthly_stock_contribution - initial_monthly_bonds_contribution
    )

    bond_value = config.current_bonds_value
    initial_cost_basis = config.current_stock_value - config.current_stock_unrealized_gain
    tranches = [{'purchase_month': -1, 'original_amount': initial_cost_basis}]
    running_cost_basis = initial_cost_basis

    total_stock_contributions = 0.0
    total_vorabpauschale = 0.0
    total_vorabpauschale_tax = 0.0

    for month in range(months):
        year = month // 12
        growth_factor = (1 + contribution_growth_rate) ** year
        monthly_stock = initial_monthly_stock_contribution * growth_factor
        monthly_bonds = initial_monthly_bonds_contribution * growth_factor
        monthly_cash = initial_monthly_cash_contribution * growth_factor
        stock_value = stock_value * (1 + stock_monthly_rate) + monthly_stock
        bond_value = bond_value * (1 + bond_monthly_rate) + monthly_bonds
        cash_value = cash_value * (1 + cash_monthly_rate) + monthly_cash
        total_stock_contributions += monthly_stock
        running_cost_basis += monthly_stock
        tranches.append({'purchase_month': month, 'original_amount': monthly_stock})

        if (month + 1) % 12 == 0:
            vorabpauschale = calculate_yearly_vorabpauschale(
                tranches, month, stock_value, running_cost_basis, config.basiszins
            )
            taxable = max(0.0, vorabpauschale * config.teilfreistellung - config.freibetrag)
            total_vorabpauschale += vorabpauschale
            total_vorabpauschale_tax += taxable * config.tax_rate

    cost_basis_stock = (
        config.current_stock_value - config.current_stock_unrealized_gain + total_stock_contributions
    )
    final_bonds_from_contributions = bond_value
    final_cash_converted_to_bonds = max(0.0, cash_value - config.cash_reserve_at_horizon)
    final_remaining_cash = cash_value - final_cash_converted_to_bonds
    final_bonds_value = final_bonds_from_contributions + final_cash_converted_to_bonds
    final_portfolio_value = stock_value + final_bonds_value + final_remaining_cash

    return ProjectionResult(
        years=config.years,
        stock_expected_yield=stock_expected_yield,
        cash_interest_rate=cash_interest_rate,
        annual_net_saving=config.annual_net_saving,
        annual_contribution_growth_percent=config.annual_contribution_growth_percent,
        maximum_stock_monthly_investment_pct=config.maximum_stock_monthly_investment_pct,
        maximum_bonds_monthly_investment_pct=config.maximum_bonds_monthly_investment_pct,
        initial_stock_value=config.current_stock_value,
        initial_stock_unrealized_gain=config.current_stock_unrealized_gain,
        initial_cash_value=config.current_cash,
        monthly_stock_contribution=initial_monthly_stock_contribution,
        monthly_bonds_contribution=initial_monthly_bonds_contribution,
        monthly_cash_contribution=initial_monthly_cash_contribution,
        final_stock_value=stock_value,
        final_bonds_from_contributions=final_bonds_from_contributions,
        final_cash_before_conversion=cash_value,
        final_cash_converted_to_bonds=final_cash_converted_to_bonds,
        final_remaining_cash=final_remaining_cash,
        final_bonds_value=final_bonds_value,
        final_portfolio_value=final_portfolio_value,
        cost_basis_stock=cost_basis_stock,
        total_vorabpauschale=total_vorabpauschale,
        total_vorabpauschale_tax=total_vorabpauschale_tax,
    )


def calculate_projection_yield_for_scenario(
    name: str,
    projection: ProjectionResult,
    withdrawal_rate: float,
    bonds_yield: float,
    stock_tax_rate: float,
    bond_tax_rate: float,
    stock_teilfreistellung: float = 1.0,
) -> ProjectionYieldScenarioResult:
    stock_bond_capital = projection.final_stock_value + projection.final_bonds_value
    if stock_bond_capital <= 0:
        raise ValueError("Projected stock+bond capital must be greater than 0.")

    gain = max(0.0, projection.final_stock_value - projection.cost_basis_stock)
    stock_gain_fraction = gain / projection.final_stock_value if projection.final_stock_value > 0 else 0.0
    effective_stock_tax_rate = stock_gain_fraction * stock_teilfreistellung * stock_tax_rate

    annual_gross_income_stock = projection.final_stock_value * withdrawal_rate
    annual_net_income_stock = annual_gross_income_stock * (1 - effective_stock_tax_rate)
    annual_gross_income_bond = projection.final_bonds_value * bonds_yield
    annual_net_income_bond = annual_gross_income_bond * (1 - bond_tax_rate)

    annual_gross_income_total = annual_gross_income_stock + annual_gross_income_bond
    annual_net_income_total = annual_net_income_stock + annual_net_income_bond
    bond_net_contribution_share = annual_net_income_bond / annual_net_income_total

    portfolio_gross_yield = annual_gross_income_total / stock_bond_capital
    portfolio_net_yield = annual_net_income_total / stock_bond_capital

    return ProjectionYieldScenarioResult(
        name=name,
        stock_tax_rate=stock_tax_rate,
        bond_tax_rate=bond_tax_rate,
        annual_gross_income_stock=annual_gross_income_stock,
        annual_net_income_stock=annual_net_income_stock,
        annual_gross_income_bond=annual_gross_income_bond,
        annual_net_income_bond=annual_net_income_bond,
        annual_gross_income_total=annual_gross_income_total,
        annual_net_income_total=annual_net_income_total,
        bond_net_contribution_share=bond_net_contribution_share,
        stock_gain_fraction=stock_gain_fraction,
        effective_stock_tax_rate=effective_stock_tax_rate,
        portfolio_gross_yield=portfolio_gross_yield,
        portfolio_net_yield=portfolio_net_yield,
    )
