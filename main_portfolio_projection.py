from dataclasses import dataclass
import os
from pathlib import Path
import tomllib


@dataclass
class Config:
    current_portfolio_value: float
    current_stock_value: float
    current_cash: float
    pct_cash_to_be_allocated_in_stock_now: float
    interest_on_cash_pa: float
    annual_net_saving: float
    maximum_stock_monthly_investment_pct: float
    pct_final_cash_to_convert_to_bonds: float
    income_value: float
    years: float
    expected_inflation_percent: float
    expected_yearly_growth_stocks_percent: float
    bonds_percentage: float
    yield_bonds: float
    tax_rate: float
    tax_rate_bond_italy: float
    withdrawal_percentage_stock: float

    @classmethod
    def from_dict(cls, config: dict[str, object]) -> "Config":
        current_portfolio_value = float(config["current_portfolio_value"])
        return cls(
            current_portfolio_value=current_portfolio_value,
            current_stock_value=float(config.get("current_stock_value", current_portfolio_value)),
            current_cash=float(config.get("current_cash", 0.0)),
            pct_cash_to_be_allocated_in_stock_now=float(
                config.get("pct_cash_to_be_allocated_in_stock_now", 0.0)
            ),
            interest_on_cash_pa=float(config.get("interest_on_cash_pa", 0.0)),
            annual_net_saving=float(config.get("annual_net_saving", 0.0)),
            maximum_stock_monthly_investment_pct=float(
                config.get("maximum_stock_monthly_investment_pct", 0.0)
            ),
            pct_final_cash_to_convert_to_bonds=float(
                config.get("pct_final_cash_to_convert_to_bonds", 0.8)
            ),
            income_value=float(config["income_value"]),
            years=float(config["years"]),
            expected_inflation_percent=float(config["expected_inflation_percent"]),
            expected_yearly_growth_stocks_percent=float(config["expected_yearly_growth_stocks_percent"]),
            bonds_percentage=float(config["bonds_percentage"]),
            yield_bonds=float(config["yield_bonds"]),
            tax_rate=float(config["tax_rate"]),
            tax_rate_bond_italy=float(config.get("tax_rate_bond_italy", config["tax_rate"])),
            withdrawal_percentage_stock=float(config["withdrawal_percentage_stock"]),
        )


@dataclass
class ScenarioResult:
    name: str
    stock_tax_rate: float
    bond_tax_rate: float
    years: float
    current_portfolio_value: float
    current_stock_value: float
    stock_expected_yield: float
    bonds_percentage: float
    yield_bonds: float
    withdrawal_rate: float
    portfolio_gross_income_rate: float
    portfolio_net_income_rate: float
    future_monthly_income_value: float
    target_annual_net_income: float
    required_annual_gross_income_from_portfolio: float
    required_total_portfolio: float
    required_stock_amount: float
    required_bonds_amount: float
    generated_annual_gross_income: float
    generated_annual_net_income: float
    required_monthly_contribution: float


@dataclass
class ProjectionResult:
    years: float
    stock_expected_yield: float
    cash_interest_rate: float
    annual_net_saving: float
    maximum_stock_monthly_investment_pct: float
    initial_stock_value: float
    initial_cash_value: float
    initial_cash_allocated_to_stock: float
    monthly_stock_contribution: float
    monthly_cash_contribution: float
    final_stock_value: float
    final_cash_before_conversion: float
    final_cash_converted_to_bonds: float
    final_remaining_cash: float
    final_bonds_value: float
    final_portfolio_value: float


@dataclass
class ProjectionYieldScenarioResult:
    name: str
    stock_tax_rate: float
    bond_tax_rate: float
    annual_gross_income_stock: float
    annual_net_income_stock: float
    annual_gross_income_bond: float
    annual_net_income_bond: float
    annual_gross_income_total: float
    annual_net_income_total: float
    bond_net_contribution_share: float
    portfolio_gross_yield: float
    portfolio_net_yield: float


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
) -> float:
    months = years * 12
    monthly_rate = (1 + annual_yield) ** (1 / 12) - 1

    future_amount_without_contributions = current_amount * (1 + monthly_rate) ** months
    if target_amount <= future_amount_without_contributions:
        return 0.0

    if abs(monthly_rate) < 1e-12:
        return (target_amount - current_amount) / months

    annuity_factor = ((1 + monthly_rate) ** months - 1) / monthly_rate
    return (target_amount - future_amount_without_contributions) / annuity_factor


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
    if config.current_portfolio_value < 0:
        raise ValueError("current_portfolio_value must be greater than or equal to 0.")
    if config.current_stock_value < 0:
        raise ValueError("current_stock_value must be greater than or equal to 0.")


def validate_projection_inputs(config: Config, cash_interest_rate: float) -> None:
    if config.current_cash < 0:
        raise ValueError("current_cash must be greater than or equal to 0.")
    if config.annual_net_saving < 0:
        raise ValueError("annual_net_saving must be greater than or equal to 0.")
    if not 0 <= config.pct_cash_to_be_allocated_in_stock_now <= 1:
        raise ValueError("pct_cash_to_be_allocated_in_stock_now must be between 0 and 1.")
    if not 0 <= config.maximum_stock_monthly_investment_pct <= 1:
        raise ValueError("maximum_stock_monthly_investment_pct must be between 0 and 1.")
    if not 0 <= config.pct_final_cash_to_convert_to_bonds <= 1:
        raise ValueError("pct_final_cash_to_convert_to_bonds must be between 0 and 1.")
    if cash_interest_rate <= -1:
        raise ValueError("interest_on_cash_pa must be greater than -100.")
    months_float = config.years * 12
    if abs(months_float - round(months_float)) > 1e-9:
        raise ValueError("years must map to whole months for projection (steps of 1/12).")


def run_projection(config: Config, stock_expected_yield: float) -> ProjectionResult:
    cash_interest_rate = config.interest_on_cash_pa / 100
    validate_projection_inputs(config, cash_interest_rate)

    months = int(round(config.years * 12))
    stock_monthly_rate = (1 + stock_expected_yield) ** (1 / 12) - 1
    cash_monthly_rate = (1 + cash_interest_rate) ** (1 / 12) - 1

    initial_cash_allocated_to_stock = (
        config.current_cash * config.pct_cash_to_be_allocated_in_stock_now
    )
    stock_value = config.current_stock_value + initial_cash_allocated_to_stock
    cash_value = config.current_cash - initial_cash_allocated_to_stock

    monthly_net_saving = config.annual_net_saving / 12
    monthly_stock_contribution = (
        monthly_net_saving * config.maximum_stock_monthly_investment_pct
    )
    monthly_cash_contribution = monthly_net_saving - monthly_stock_contribution

    for _ in range(months):
        stock_value = stock_value * (1 + stock_monthly_rate) + monthly_stock_contribution
        cash_value = cash_value * (1 + cash_monthly_rate) + monthly_cash_contribution

    final_cash_converted_to_bonds = (
        cash_value * config.pct_final_cash_to_convert_to_bonds
    )
    final_remaining_cash = cash_value - final_cash_converted_to_bonds
    final_bonds_value = final_cash_converted_to_bonds
    final_portfolio_value = stock_value + final_bonds_value + final_remaining_cash

    return ProjectionResult(
        years=config.years,
        stock_expected_yield=stock_expected_yield,
        cash_interest_rate=cash_interest_rate,
        annual_net_saving=config.annual_net_saving,
        maximum_stock_monthly_investment_pct=config.maximum_stock_monthly_investment_pct,
        initial_stock_value=config.current_stock_value,
        initial_cash_value=config.current_cash,
        initial_cash_allocated_to_stock=initial_cash_allocated_to_stock,
        monthly_stock_contribution=monthly_stock_contribution,
        monthly_cash_contribution=monthly_cash_contribution,
        final_stock_value=stock_value,
        final_cash_before_conversion=cash_value,
        final_cash_converted_to_bonds=final_cash_converted_to_bonds,
        final_remaining_cash=final_remaining_cash,
        final_bonds_value=final_bonds_value,
        final_portfolio_value=final_portfolio_value,
    )


def calculate_projection_yield_for_scenario(
    name: str,
    projection: ProjectionResult,
    withdrawal_rate: float,
    bonds_yield: float,
    stock_tax_rate: float,
    bond_tax_rate: float,
) -> ProjectionYieldScenarioResult:
    stock_bond_capital = projection.final_stock_value + projection.final_bonds_value
    if stock_bond_capital <= 0:
        raise ValueError("Projected stock+bond capital must be greater than 0.")

    annual_gross_income_stock = projection.final_stock_value * withdrawal_rate
    annual_net_income_stock = annual_gross_income_stock * (1 - stock_tax_rate)
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
        portfolio_gross_yield=portfolio_gross_yield,
        portfolio_net_yield=portfolio_net_yield,
    )


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
) -> ScenarioResult:
    bonds_weight = config.bonds_percentage / 100
    stocks_weight = 1 - bonds_weight
    portfolio_gross_income_rate = (
        (bonds_weight * bonds_yield)
        + (stocks_weight * withdrawal_rate)
    )
    portfolio_net_income_rate = (
        (bonds_weight * bonds_yield * (1 - bond_tax_rate))
        + (stocks_weight * withdrawal_rate * (1 - stock_tax_rate))
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
    )
    generated_annual_gross_income = required_total_portfolio * portfolio_gross_income_rate
    generated_annual_net_income = required_total_portfolio * portfolio_net_income_rate

    return ScenarioResult(
        name=name,
        stock_tax_rate=stock_tax_rate,
        bond_tax_rate=bond_tax_rate,
        years=config.years,
        current_portfolio_value=config.current_portfolio_value,
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
    )


def format_currency(value: float) -> str:
    return f"{value:,.2f}"


def format_currency_signed(value: float) -> str:
    return f"{value:+,.2f}"


def format_ratio_as_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_percent_value(value: float) -> str:
    return f"{value:.2f}%"


def format_ratio_delta_as_percentage_points(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def print_scenario_report(result: ScenarioResult) -> None:
    print(f"{result.name}: Income from portfolio")
    print("Assumptions")
    print(f"  Horizon: {result.years:.1f} years")
    print(f"  Current portfolio value: {format_currency(result.current_portfolio_value)}")
    print(f"  Current stock value: {format_currency(result.current_stock_value)}")
    print(f"  Stocks expected yield: {format_ratio_as_percent(result.stock_expected_yield)}")
    print(f"  Bonds allocation: {format_percent_value(result.bonds_percentage)}")
    print(f"  Bonds yield: {format_ratio_as_percent(result.yield_bonds)}")
    print(f"  Stocks withdrawal rate: {format_ratio_as_percent(result.withdrawal_rate)}")
    print()
    print("Tax assumptions")
    print(f"  Stocks tax rate: {format_ratio_as_percent(result.stock_tax_rate)}")
    print(f"  Bonds tax rate: {format_ratio_as_percent(result.bond_tax_rate)}")
    print()
    print("Income target at horizon")
    print(f"  Future monthly net income: {format_currency(result.future_monthly_income_value)}")
    print(f"  Target annual net income: {format_currency(result.target_annual_net_income)}")
    print(
        "  Required annual gross income from portfolio: "
        f"{format_currency(result.required_annual_gross_income_from_portfolio)}"
    )
    print()
    print("Required portfolio")
    print(f"  Total portfolio: {format_currency(result.required_total_portfolio)}")
    print(f"  Required stocks amount: {format_currency(result.required_stock_amount)}")
    print(f"  Required bonds amount: {format_currency(result.required_bonds_amount)}")
    print(
        "  Portfolio gross yield: "
        f"{format_ratio_as_percent(result.portfolio_gross_income_rate)}"
    )
    print(
        "  Portfolio net yield (after tax): "
        f"{format_ratio_as_percent(result.portfolio_net_income_rate)}"
    )
    print()
    print("Validation")
    print(
        "  Generated annual gross income: "
        f"{format_currency(result.generated_annual_gross_income)}"
    )
    print(
        "  Generated annual net income: "
        f"{format_currency(result.generated_annual_net_income)}"
    )
    print()
    print("Savings plan")
    print(
        "  Required monthly contribution (stocks): "
        f"{format_currency(result.required_monthly_contribution)}"
    )


def print_projection_report(result: ProjectionResult, conversion_pct: float) -> None:
    monthly_net_saving = result.annual_net_saving / 12
    final_stock_and_bonds_value = result.final_stock_value + result.final_bonds_value

    print("Assumptions")
    print(f"  Horizon: {result.years:.1f} years")
    print(f"  Initial stock: {format_currency(result.initial_stock_value)}")
    print(f"  Expected stock yearly growth: {format_ratio_as_percent(result.stock_expected_yield)}")
    print(f"  Interest on cash (p.a.): {format_ratio_as_percent(result.cash_interest_rate)}")
    print()
    print("=" * 70)
    print(f"{'PROJECTION: Current accumulation path':^70}")
    print("=" * 70)
    print(
        "  Cash allocated to stocks now: "
        f"{format_ratio_as_percent(result.initial_cash_allocated_to_stock / result.initial_cash_value) if result.initial_cash_value > 0 else '0.00%'} "
        f"({format_currency(result.initial_cash_allocated_to_stock)})"
    )
    print()
    print("Flow")
    print(f"  Annual net saving (assumption): {format_currency(result.annual_net_saving)}")
    print(f"  Monthly net saving: {format_currency(monthly_net_saving)}")
    print(
        "  Monthly contribution to stocks "
        f"({format_ratio_as_percent(result.maximum_stock_monthly_investment_pct)}): "
        f"{format_currency(result.monthly_stock_contribution)}"
    )
    print(f"  Monthly contribution to cash: {format_currency(result.monthly_cash_contribution)}")
    print()
    print("Projection at horizon")
    print(
        "  Final portfolio value (Stocks + Bonds): "
        f"{format_currency(final_stock_and_bonds_value)}"
    )
    print(f"  Stocks: {format_currency(result.final_stock_value)}")
    print(
        f"  Bonds ({format_ratio_as_percent(conversion_pct)}): "
        f"{format_currency(result.final_cash_converted_to_bonds)}"
    )
    print(
        "  Final cash (before conversion in bonds): "
        f"{format_currency(result.final_cash_before_conversion)}"
    )
    print(f"  Remaining cash: {format_currency(result.final_remaining_cash)}")
    print()


def print_projection_yield_table(
    scenario_1: ProjectionYieldScenarioResult,
    scenario_2: ProjectionYieldScenarioResult,
    portfolio_value: float,
) -> None:
    metric_width = 38
    value_width = 16
    delta_width = 16

    rows = [
        (
            "Stock gross income",
            format_currency(scenario_1.annual_gross_income_stock),
            format_currency(scenario_2.annual_gross_income_stock),
            format_currency_signed(
                scenario_2.annual_gross_income_stock - scenario_1.annual_gross_income_stock
            ),
        ),
        (
            "Stock net income",
            format_currency(scenario_1.annual_net_income_stock),
            format_currency(scenario_2.annual_net_income_stock),
            format_currency_signed(
                scenario_2.annual_net_income_stock - scenario_1.annual_net_income_stock
            ),
        ),
        (
            "Bond gross income",
            format_currency(scenario_1.annual_gross_income_bond),
            format_currency(scenario_2.annual_gross_income_bond),
            format_currency_signed(
                scenario_2.annual_gross_income_bond - scenario_1.annual_gross_income_bond
            ),
        ),
        (
            "Bond net income",
            format_currency(scenario_1.annual_net_income_bond),
            format_currency(scenario_2.annual_net_income_bond),
            format_currency_signed(
                scenario_2.annual_net_income_bond - scenario_1.annual_net_income_bond
            ),
        ),
        (
            "Total gross income (stock + bond)",
            format_currency(scenario_1.annual_gross_income_total),
            format_currency(scenario_2.annual_gross_income_total),
            format_currency_signed(
                scenario_2.annual_gross_income_total - scenario_1.annual_gross_income_total
            ),
        ),
        (
            "Total net income (stock + bond)",
            format_currency(scenario_1.annual_net_income_total),
            format_currency(scenario_2.annual_net_income_total),
            format_currency_signed(
                scenario_2.annual_net_income_total - scenario_1.annual_net_income_total
            ),
        ),
        (
            "Gross yield at horizon (stock + bond)",
            format_ratio_as_percent(scenario_1.portfolio_gross_yield),
            format_ratio_as_percent(scenario_2.portfolio_gross_yield),
            format_ratio_delta_as_percentage_points(
                scenario_2.portfolio_gross_yield - scenario_1.portfolio_gross_yield
            ),
        ),
        (
            "Net yield at horizon (stock + bond)",
            format_ratio_as_percent(scenario_1.portfolio_net_yield),
            format_ratio_as_percent(scenario_2.portfolio_net_yield),
            format_ratio_delta_as_percentage_points(
                scenario_2.portfolio_net_yield - scenario_1.portfolio_net_yield
            ),
        ),
    ]

    separator = (
        f"+-{'-' * metric_width}-+-{'-' * value_width}-+-{'-' * value_width}-+-{'-' * delta_width}-+"
    )
    title = (
        "PROJECTED INCOME BREAKDOWN FROM PORTFOLIO "
        f"{format_currency(portfolio_value)}"
    )

    print()
    print(f"{title:^{len(separator)}}")
    print()
    print(separator)
    print(
        f"| {'Metric':<{metric_width}} "
        f"| {'Scenario 1':>{value_width}} "
        f"| {'Scenario 2':>{value_width}} "
        f"| {'Delta (S2-S1)':>{delta_width}} |"
    )
    print(separator)
    for metric, value_1, value_2, delta in rows:
        print(
            f"| {metric:<{metric_width}} "
            f"| {value_1:>{value_width}} "
            f"| {value_2:>{value_width}} "
            f"| {delta:>{delta_width}} |"
        )
    print(separator)


def print_summary_table(scenario_1: ScenarioResult, scenario_2: ScenarioResult) -> None:
    metric_width = 34
    value_width = 16
    delta_width = 16

    rows = [
        (
            "Stocks tax rate",
            format_ratio_as_percent(scenario_1.stock_tax_rate),
            format_ratio_as_percent(scenario_2.stock_tax_rate),
            format_ratio_delta_as_percentage_points(
                scenario_2.stock_tax_rate - scenario_1.stock_tax_rate
            ),
        ),
        (
            "Bonds tax rate",
            format_ratio_as_percent(scenario_1.bond_tax_rate),
            format_ratio_as_percent(scenario_2.bond_tax_rate),
            format_ratio_delta_as_percentage_points(
                scenario_2.bond_tax_rate - scenario_1.bond_tax_rate
            ),
        ),
        (
            "Portfolio net yield (after tax)",
            format_ratio_as_percent(scenario_1.portfolio_net_income_rate),
            format_ratio_as_percent(scenario_2.portfolio_net_income_rate),
            format_ratio_delta_as_percentage_points(
                scenario_2.portfolio_net_income_rate - scenario_1.portfolio_net_income_rate
            ),
        ),
        (
            "Annual gross income from portfolio",
            format_currency(scenario_1.generated_annual_gross_income),
            format_currency(scenario_2.generated_annual_gross_income),
            format_currency_signed(
                scenario_2.generated_annual_gross_income - scenario_1.generated_annual_gross_income
            ),
        ),
        (
            "Total portfolio",
            format_currency(scenario_1.required_total_portfolio),
            format_currency(scenario_2.required_total_portfolio),
            format_currency_signed(
                scenario_2.required_total_portfolio - scenario_1.required_total_portfolio
            ),
        ),
        (
            "Required stocks amount",
            format_currency(scenario_1.required_stock_amount),
            format_currency(scenario_2.required_stock_amount),
            format_currency_signed(
                scenario_2.required_stock_amount - scenario_1.required_stock_amount
            ),
        ),
        (
            "Required bonds amount",
            format_currency(scenario_1.required_bonds_amount),
            format_currency(scenario_2.required_bonds_amount),
            format_currency_signed(
                scenario_2.required_bonds_amount - scenario_1.required_bonds_amount
            ),
        ),
        (
            "Monthly contribution (stocks)",
            format_currency(scenario_1.required_monthly_contribution),
            format_currency(scenario_2.required_monthly_contribution),
            format_currency_signed(
                scenario_2.required_monthly_contribution - scenario_1.required_monthly_contribution
            ),
        ),
    ]

    separator = (
        f"+-{'-' * metric_width}-+-{'-' * value_width}-+-{'-' * value_width}-+-{'-' * delta_width}-+"
    )

    print()
    print(
        f"{'SUMMARY TABLE':^{len(separator)}}"
    )
    print(separator)
    print(
        f"| {'Metric':<{metric_width}} "
        f"| {'Scenario 1':>{value_width}} "
        f"| {'Scenario 2':>{value_width}} "
        f"| {'Delta (S2-S1)':>{delta_width}} |"
    )
    print(separator)
    for metric, value_1, value_2, delta in rows:
        print(
            f"| {metric:<{metric_width}} "
            f"| {value_1:>{value_width}} "
            f"| {value_2:>{value_width}} "
            f"| {delta:>{delta_width}} |"
        )
    print(separator)


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def main() -> None:
    clear_terminal()
    config_path = Path(__file__).with_name("config.toml")

    try:
        with config_path.open("rb") as config_file:
            config = Config.from_dict(tomllib.load(config_file))
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return
    except KeyError as error:
        print(f"Missing required config key: {error}")
        return
    except (ValueError, tomllib.TOMLDecodeError):
        print("Config values must be valid numbers in valid TOML.")
        return

    try:
        inflation_rate = config.expected_inflation_percent / 100
        stock_expected_yield = config.expected_yearly_growth_stocks_percent / 100
        bonds_yield = config.yield_bonds / 100
        withdrawal_rate = config.withdrawal_percentage_stock / 100

        validate_config_inputs(
            config=config,
            stock_expected_yield=stock_expected_yield,
            bonds_yield=bonds_yield,
            withdrawal_rate=withdrawal_rate,
        )

        future_monthly_income_value = calculate_future_income_value(
            income_value=config.income_value,
            years=config.years,
            inflation_rate=inflation_rate,
        )
        target_annual_net_income = future_monthly_income_value * 12
        projection_result = run_projection(config, stock_expected_yield)

        scenario_1_result = run_scenario(
            name="SCENARIO 1",
            config=config,
            future_monthly_income_value=future_monthly_income_value,
            target_annual_net_income=target_annual_net_income,
            stock_tax_rate=config.tax_rate,
            bond_tax_rate=config.tax_rate,
            stock_expected_yield=stock_expected_yield,
            bonds_yield=bonds_yield,
            withdrawal_rate=withdrawal_rate,
        )
        scenario_2_result = run_scenario(
            name="SCENARIO 2",
            config=config,
            future_monthly_income_value=future_monthly_income_value,
            target_annual_net_income=target_annual_net_income,
            stock_tax_rate=config.tax_rate,
            bond_tax_rate=config.tax_rate_bond_italy,
            stock_expected_yield=stock_expected_yield,
            bonds_yield=bonds_yield,
            withdrawal_rate=withdrawal_rate,
        )
        projected_yield_scenario_1 = calculate_projection_yield_for_scenario(
            name="SCENARIO 1",
            projection=projection_result,
            withdrawal_rate=withdrawal_rate,
            bonds_yield=bonds_yield,
            stock_tax_rate=config.tax_rate,
            bond_tax_rate=config.tax_rate,
        )
        projected_yield_scenario_2 = calculate_projection_yield_for_scenario(
            name="SCENARIO 2",
            projection=projection_result,
            withdrawal_rate=withdrawal_rate,
            bonds_yield=bonds_yield,
            stock_tax_rate=config.tax_rate,
            bond_tax_rate=config.tax_rate_bond_italy,
        )
    except ValueError as error:
        print(error)
        return

    print_projection_report(
        result=projection_result,
        conversion_pct=config.pct_final_cash_to_convert_to_bonds,
    )
    print_projection_yield_table(
        projected_yield_scenario_1,
        projected_yield_scenario_2,
        projection_result.final_stock_value + projection_result.final_bonds_value,
    )
    print("\n" * 3, end="")
    print("=" * 70)
    print(f"{'REQUIRED STOCK SAVING PLAN CALCULATION':^70}")
    print("=" * 70)
    print_scenario_report(scenario_1_result)
    print()
    print_scenario_report(scenario_2_result)
    print_summary_table(scenario_1_result, scenario_2_result)


if __name__ == "__main__":
    main()
