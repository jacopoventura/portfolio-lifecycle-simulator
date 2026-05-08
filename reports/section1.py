from finance import (
    calculate_future_income_value,
    cash_converted_to_bonds_at_horizon,
    required_portfolio_and_contribution,
)
from models import Config, ScenarioResult, SensitivityWaterfallResult

from reports.formatters import (
    format_currency,
    format_currency_signed,
    format_percent_value,
    format_ratio_as_percent,
    format_ratio_delta_as_percentage_points,
)


def print_scenarios_report(
    s1: ScenarioResult,
    s2: ScenarioResult,
    current_bonds_value: float,
    current_cash: float,
    cash_interest_rate_pa: float,
    estimated_final_cash: float,
    cash_to_bonds_at_horizon: float,
    cash_reserve_at_horizon: float,
    annual_net_saving: float,
    verbose: bool = False,
) -> None:
    print("Common assumptions")
    print(f"  Horizon: {s1.years:.1f} years")
    print(f"  Monthly net saving: {format_currency(annual_net_saving / 12)}/mo"
          f"  |  Annual: {format_currency(annual_net_saving)}/yr")
    print(f"  Current stock: {format_currency(s1.current_stock_value)}"
          f"  |  Expected yield: {format_ratio_as_percent(s1.stock_expected_yield)}"
          f"  |  Withdrawal rate: {format_ratio_as_percent(s1.withdrawal_rate)}")
    print(f"  Bonds: {format_percent_value(s1.bonds_percentage)} at {format_ratio_as_percent(s1.yield_bonds)}")
    if verbose:
        print(f"  Stocks nominal tax: {format_ratio_as_percent(s1.stock_tax_rate)}"
              f"  |  Gain fraction: {format_ratio_as_percent(s1.stock_gain_fraction)}"
              f"  |  Effective tax: {format_ratio_as_percent(s1.effective_stock_tax_rate)}")
    fv_current_stocks = s1.current_stock_value * (1 + s1.stock_expected_yield) ** s1.years
    fv_current_bonds = current_bonds_value * (1 + s1.yield_bonds) ** s1.years
    stocks_to_build_de = max(0.0, s1.required_stock_amount - fv_current_stocks)
    stocks_to_build_it = max(0.0, s2.required_stock_amount - fv_current_stocks)
    bonds_to_build_de = max(0.0, s1.required_bonds_amount - cash_to_bonds_at_horizon - fv_current_bonds)
    bonds_to_build_it = max(0.0, s2.required_bonds_amount - cash_to_bonds_at_horizon - fv_current_bonds)
    print(f"  Current stocks: {format_currency(s1.current_stock_value)}"
          f"  |  Estimated at horizon: {format_currency(fv_current_stocks)}")
    print(f"  Required stocks (DE): {format_currency(s1.required_stock_amount)}"
          f"  |  Stocks amount to build until horizon: {format_currency(stocks_to_build_de)}")
    print(f"  Required stocks (IT): {format_currency(s2.required_stock_amount)}"
          f"  |  Stocks amount to build until horizon: {format_currency(stocks_to_build_it)}")
    print(f"  Current bonds: {format_currency(current_bonds_value)}"
          f"  |  Estimated at horizon: {format_currency(fv_current_bonds)}")
    print(f"  Current cash: {format_currency(current_cash)}"
          f"  |  Yield: {format_percent_value(cash_interest_rate_pa)}/yr"
          f"  |  Estimated at horizon: {format_currency(estimated_final_cash)}")
    print(f"  Cash converted to bonds at horizon: {format_currency(cash_to_bonds_at_horizon)}"
          f"  |  Reserve kept: {format_currency(cash_reserve_at_horizon)} (part of current cash)")
    print(f"  Required bonds (DE): {format_currency(s1.required_bonds_amount)}"
          f"  |  Bonds amount to build until horizon: {format_currency(bonds_to_build_de)}")
    print(f"  Required bonds (IT): {format_currency(s2.required_bonds_amount)}"
          f"  |  Bonds amount to build until horizon: {format_currency(bonds_to_build_it)}")
    print()

    metric_width = 40
    value_width = 16
    delta_width = 16
    separator = (
        f"+-{'-' * metric_width}-+-{'-' * value_width}-+-{'-' * value_width}-+-{'-' * delta_width}-+"
    )

    def _row(label: str, v1: str, v2: str, delta: str) -> None:
        print(
            f"| {label:<{metric_width}} "
            f"| {v1:>{value_width}} "
            f"| {v2:>{value_width}} "
            f"| {delta:>{delta_width}} |"
        )

    print(separator)
    print(
        f"| {'Metric':<{metric_width}} "
        f"| {'Scenario 1 (DE)':>{value_width}} "
        f"| {'Scenario 2 (IT)':>{value_width}} "
        f"| {'Delta (IT-DE)':>{delta_width}} |"
    )
    print(separator)

    _row("Total portfolio",
         format_currency(s1.required_total_portfolio),
         format_currency(s2.required_total_portfolio),
         format_currency_signed(s2.required_total_portfolio - s1.required_total_portfolio))
    _row("  of which stocks",
         format_currency(s1.required_stock_amount),
         format_currency(s2.required_stock_amount),
         format_currency_signed(s2.required_stock_amount - s1.required_stock_amount))
    _row("  of which bonds",
         format_currency(s1.required_bonds_amount),
         format_currency(s2.required_bonds_amount),
         format_currency_signed(s2.required_bonds_amount - s1.required_bonds_amount))
    _row("Gross income from portfolio",
         format_currency(s1.required_annual_gross_income_from_portfolio),
         format_currency(s2.required_annual_gross_income_from_portfolio),
         format_currency_signed(
             s2.required_annual_gross_income_from_portfolio
             - s1.required_annual_gross_income_from_portfolio))
    _row("Net income from portfolio",
         format_currency(s1.generated_annual_net_income),
         format_currency(s2.generated_annual_net_income),
         format_currency_signed(s2.generated_annual_net_income - s1.generated_annual_net_income))
    _row("Monthly contribution, yr 1 (stocks)",
         format_currency(s1.required_monthly_contribution),
         format_currency(s2.required_monthly_contribution),
         format_currency_signed(s2.required_monthly_contribution - s1.required_monthly_contribution))
    _row("Monthly contribution, yr 1 (bonds)",
         format_currency(s1.required_monthly_bonds_contribution),
         format_currency(s2.required_monthly_bonds_contribution),
         format_currency_signed(
             s2.required_monthly_bonds_contribution - s1.required_monthly_bonds_contribution))
    monthly_saving = annual_net_saving / 12
    cash_s1 = max(0.0, monthly_saving - s1.required_monthly_contribution - s1.required_monthly_bonds_contribution)
    cash_s2 = max(0.0, monthly_saving - s2.required_monthly_contribution - s2.required_monthly_bonds_contribution)
    _row("Monthly contribution, yr 1 (cash)",
         format_currency(cash_s1),
         format_currency(cash_s2),
         format_currency_signed(cash_s2 - cash_s1))

    if verbose:
        print(separator)
        _row("Bonds tax rate",
             format_ratio_as_percent(s1.bond_tax_rate),
             format_ratio_as_percent(s2.bond_tax_rate),
             format_ratio_delta_as_percentage_points(s2.bond_tax_rate - s1.bond_tax_rate))
        _row("Stock nominal tax rate",
             format_ratio_as_percent(s1.stock_tax_rate),
             format_ratio_as_percent(s2.stock_tax_rate),
             format_ratio_delta_as_percentage_points(s2.stock_tax_rate - s1.stock_tax_rate))
        _row("Stock Teilfreistellung (taxable portion)",
             format_ratio_as_percent(s1.stock_teilfreistellung),
             format_ratio_as_percent(s2.stock_teilfreistellung),
             format_ratio_delta_as_percentage_points(
                 s2.stock_teilfreistellung - s1.stock_teilfreistellung))
        _row("Stock gain fraction",
             format_ratio_as_percent(s1.stock_gain_fraction),
             format_ratio_as_percent(s2.stock_gain_fraction),
             format_ratio_delta_as_percentage_points(
                 s2.stock_gain_fraction - s1.stock_gain_fraction))
        _row("Stock effective tax (on withdrawal)",
             format_ratio_as_percent(s1.effective_stock_tax_rate),
             format_ratio_as_percent(s2.effective_stock_tax_rate),
             format_ratio_delta_as_percentage_points(
                 s2.effective_stock_tax_rate - s1.effective_stock_tax_rate))
        _row("Portfolio gross yield",
             format_ratio_as_percent(s1.portfolio_gross_income_rate),
             format_ratio_as_percent(s2.portfolio_gross_income_rate),
             format_ratio_delta_as_percentage_points(
                 s2.portfolio_gross_income_rate - s1.portfolio_gross_income_rate))
        _row("Portfolio net yield (after tax)",
             format_ratio_as_percent(s1.portfolio_net_income_rate),
             format_ratio_as_percent(s2.portfolio_net_income_rate),
             format_ratio_delta_as_percentage_points(
                 s2.portfolio_net_income_rate - s1.portfolio_net_income_rate))

    print(separator)


def print_sensitivity_table(
    config: Config,
    base_inflation_rate: float,
    base_bonds_yield: float,
    base_withdrawal_rate: float,
    stock_expected_yield: float,
    stock_gain_fraction: float,
) -> None:
    annual_contribution_growth = config.annual_contribution_growth_percent / 100
    base_target_annual = (
        calculate_future_income_value(
            config.target_monthly_retirement_income, config.years, base_inflation_rate
        )
        * 12
    )

    col_w = 13
    sep = (
        f"+-{'-' * 14}-+-{'-' * col_w}-+-{'-' * col_w}-+-{'-' * col_w}"
        f"-+-{'-' * col_w}-+-{'-' * col_w}-+-{'-' * col_w}-+"
    )

    def print_header(param_label: str) -> None:
        print(f"\n  Sensitivity: {param_label}")
        print(sep)
        print(
            f"| {'Value':<14} "
            f"| {'Portfolio DE':>{col_w}} "
            f"| {'Stocks/mo DE':>{col_w}} "
            f"| {'Bonds/mo DE':>{col_w}} "
            f"| {'Portfolio IT':>{col_w}} "
            f"| {'Stocks/mo IT':>{col_w}} "
            f"| {'Bonds/mo IT':>{col_w}} |"
        )
        print(sep)

    def print_row(
        label: str, is_base: bool,
        p1: float, ms1: float, mb1: float,
        p2: float, ms2: float, mb2: float,
    ) -> None:
        marker = "* " if is_base else "  "
        def fmt(v: float) -> str:
            return format_currency(v) if v != float("inf") else "N/A"
        print(
            f"|{marker}{label:<13} "
            f"| {fmt(p1):>{col_w}} "
            f"| {fmt(ms1):>{col_w}} "
            f"| {fmt(mb1):>{col_w}} "
            f"| {fmt(p2):>{col_w}} "
            f"| {fmt(ms2):>{col_w}} "
            f"| {fmt(mb2):>{col_w}} |"
        )

    cash_pct = max(0.0, 1.0 - config.maximum_stock_monthly_investment_pct - config.maximum_bonds_monthly_investment_pct)
    monthly_cash_contribution = config.annual_net_saving / 12 * cash_pct
    cash_to_bonds = cash_converted_to_bonds_at_horizon(
        current_cash=config.current_cash,
        cash_rate=config.interest_on_cash_pa / 100,
        years=config.years,
        cash_reserve=config.cash_reserve_at_horizon,
        monthly_contribution=monthly_cash_contribution,
        contribution_growth_rate=annual_contribution_growth,
    )

    def both(
        target_annual: float, bonds_pct: float, b_yield: float, w_rate: float,
    ) -> tuple[float, float, float, float, float, float]:
        p1, ms1, mb1 = required_portfolio_and_contribution(
            target_annual, bonds_pct, b_yield, w_rate,
            stock_gain_fraction, config.tax_rate, config.tax_rate,
            config.teilfreistellung,
            config.current_stock_value, config.current_bonds_value, cash_to_bonds,
            config.years, stock_expected_yield, annual_contribution_growth,
        )
        p2, ms2, mb2 = required_portfolio_and_contribution(
            target_annual, bonds_pct, b_yield, w_rate,
            stock_gain_fraction, config.tax_rate, config.tax_rate_bond_italy,
            1.0,
            config.current_stock_value, config.current_bonds_value, cash_to_bonds,
            config.years, stock_expected_yield, annual_contribution_growth,
        )
        return p1, ms1, mb1, p2, ms2, mb2

    def is_base_val(a: float, b: float) -> bool:
        return abs(a - b) < 1e-9

    print_header("Bonds allocation  (* = current)")
    for pct in [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]:
        p1, ms1, mb1, p2, ms2, mb2 = both(base_target_annual, pct, base_bonds_yield, base_withdrawal_rate)
        print_row(f"{pct:.0f}%", is_base_val(pct, config.bonds_percentage), p1, ms1, mb1, p2, ms2, mb2)
    print(sep)

    print_header("Bonds yield  (* = current)")
    for y in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        p1, ms1, mb1, p2, ms2, mb2 = both(base_target_annual, config.bonds_percentage, y / 100, base_withdrawal_rate)
        print_row(f"{y:.1f}%", is_base_val(y / 100, base_bonds_yield), p1, ms1, mb1, p2, ms2, mb2)
    print(sep)

    print_header("Stock withdrawal rate  (* = current)")
    for r in [2.0, 2.5, 3.0, 3.3, 4.0, 4.5]:
        p1, ms1, mb1, p2, ms2, mb2 = both(base_target_annual, config.bonds_percentage, base_bonds_yield, r / 100)
        print_row(f"{r:.1f}%", is_base_val(r / 100, base_withdrawal_rate), p1, ms1, mb1, p2, ms2, mb2)
    print(sep)

    print_header("Inflation  (* = current)")
    for inf in [1.5, 2.0, 2.3, 2.5, 3.0, 3.5]:
        target_annual = (
            calculate_future_income_value(config.target_monthly_retirement_income, config.years, inf / 100) * 12
        )
        p1, ms1, mb1, p2, ms2, mb2 = both(target_annual, config.bonds_percentage, base_bonds_yield, base_withdrawal_rate)
        print_row(f"{inf:.1f}%", is_base_val(inf / 100, base_inflation_rate), p1, ms1, mb1, p2, ms2, mb2)
    print(sep)


def print_sensitivity_waterfall(
    result_de: SensitivityWaterfallResult,
    result_it: SensitivityWaterfallResult,
) -> None:
    base_de = result_de.entries[0].success_base if result_de.entries else 0.0
    base_it = result_it.entries[0].success_base if result_it.entries else 0.0
    print(f"  Base income success rate:  DE {base_de * 100:.1f}%   IT {base_it * 100:.1f}%")
    print()

    col_param = 24
    col_base  = 12
    col_delta = 10
    col_val   = 9

    header = (
        f"  {'Parameter':<{col_param}}"
        f"{'Base':>{col_base}}"
        f"{'Delta':>{col_delta}}"
        f"{'DE +δ':>{col_val}}"
        f"{'DE −δ':>{col_val}}"
        f"{'IT +δ':>{col_val}}"
        f"{'IT −δ':>{col_val}}"
    )
    sep = "  " + "─" * (col_param + col_base + col_delta + col_val * 4)
    print(header)
    print(sep)

    de_map = {e.parameter: e for e in result_de.entries}
    for e_it in result_it.entries:
        e_de = de_map.get(e_it.parameter)

        def _pp(val: float, base: float) -> str:
            diff = (val - base) * 100
            sign = "+" if diff >= 0 else ""
            return f"{sign}{diff:.1f}pp"

        de_up   = _pp(e_de.success_up,   e_de.success_base) if e_de else "n/a"
        de_down = _pp(e_de.success_down, e_de.success_base) if e_de else "n/a"
        it_up   = _pp(e_it.success_up,   e_it.success_base)
        it_down = _pp(e_it.success_down, e_it.success_base)

        print(
            f"  {e_it.parameter:<{col_param}}"
            f"{e_it.base_value:>{col_base}}"
            f"{e_it.delta_label:>{col_delta}}"
            f"{de_up:>{col_val}}"
            f"{de_down:>{col_val}}"
            f"{it_up:>{col_val}}"
            f"{it_down:>{col_val}}"
        )
    print(sep)
    print("  Sorted by IT impact (largest first). +δ = parameter increases, −δ = decreases.")
