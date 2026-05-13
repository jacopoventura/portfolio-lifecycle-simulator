import datetime

from models import (
    Config,
    FinancialFreedomResult,
    MonteCarloResult,
    OptimalAllocationResult,
    ProjectionResult,
    ProjectionYieldScenarioResult,
    ScenarioResult,
)

from reports.formatters import (
    format_currency,
    format_currency_signed,
    format_ratio_as_percent,
    format_ratio_delta_as_percentage_points,
)


def print_projection_report(
    result: ProjectionResult,
    scenario_de: ScenarioResult,
    scenario_it: ScenarioResult,
) -> None:
    conversion_pct = (result.final_cash_converted_to_bonds / result.final_cash_before_conversion
                      if result.final_cash_before_conversion > 0 else 0.0)
    monthly_net_saving = result.annual_net_saving / 12
    final_stock_and_bonds_value = result.final_stock_value + result.final_bonds_value

    print("Assumptions")
    print(f"  Horizon: {result.years:.1f} years")
    print(f"  Initial stock: {format_currency(result.initial_stock_value)}")
    print(f"  Initial stock unrealized gain: {format_currency(result.initial_stock_unrealized_gain)}")
    print(f"  Expected stock yearly growth: {format_ratio_as_percent(result.stock_expected_yield)}")
    print(f"  Interest on cash (p.a.): {format_ratio_as_percent(result.cash_interest_rate)}")
    print()
    print(f"{'PROJECTION: Current accumulation path':^70}")
    print("Flow")
    print(f"  Annual net saving (assumption): {format_currency(result.annual_net_saving)}")
    print(f"  Monthly net saving: {format_currency(monthly_net_saving)}")
    print(
        "  Monthly contribution to stocks, year 1 "
        f"({format_ratio_as_percent(result.maximum_stock_monthly_investment_pct)}): "
        f"{format_currency(result.monthly_stock_contribution)}"
    )
    print(
        "  Monthly contribution to bonds, year 1  "
        f"({format_ratio_as_percent(result.maximum_bonds_monthly_investment_pct)}): "
        f"{format_currency(result.monthly_bonds_contribution)}"
    )
    print(f"  Monthly contribution to cash, year 1   ({format_ratio_as_percent(1.0 - result.maximum_stock_monthly_investment_pct - result.maximum_bonds_monthly_investment_pct)}): {format_currency(result.monthly_cash_contribution)}")
    print(f"  Annual contribution growth: {format_ratio_as_percent(result.annual_contribution_growth_percent / 100)}")
    print()
    print("Projection at horizon")
    print(
        "  Final portfolio value (Stocks + Bonds): "
        f"{format_currency(final_stock_and_bonds_value)}"
    )
    print(f"  Stocks: {format_currency(result.final_stock_value)}")
    for sc in (scenario_de, scenario_it):
        diff = result.final_stock_value - sc.required_stock_amount
        label = "surplus" if diff >= 0 else "shortfall"
        print(f"    vs required {sc.name}: {format_currency(sc.required_stock_amount)}"
              f"  ({label} {format_currency_signed(diff)})")
    print(f"  Bonds (total): {format_currency(result.final_bonds_value)}")
    for sc in (scenario_de, scenario_it):
        diff = result.final_bonds_value - sc.required_bonds_amount
        label = "surplus" if diff >= 0 else "shortfall"
        print(f"    vs required {sc.name}: {format_currency(sc.required_bonds_amount)}"
              f"  ({label} {format_currency_signed(diff)})")
    print(f"    of which from contributions: {format_currency(result.final_bonds_from_contributions)}")
    print(
        f"    of which from cash conversion ({format_ratio_as_percent(conversion_pct)}): "
        f"{format_currency(result.final_cash_converted_to_bonds)}"
    )
    print(
        "  Final cash (before conversion in bonds): "
        f"{format_currency(result.final_cash_before_conversion)}"
    )
    print(f"  Remaining cash: {format_currency(result.final_remaining_cash)}")
    print()
    print("Vorabpauschale (stock ETF pre-tax)")
    print(f"  Total Vorabpauschale over horizon: {format_currency(result.total_vorabpauschale)}")
    print(f"  Total tax paid on Vorabpauschale: {format_currency(result.total_vorabpauschale_tax)}")
    print()


def print_financial_freedom_report(
    ff_de: FinancialFreedomResult,
    ff_it: FinancialFreedomResult,
    config: Config,
) -> None:
    current_year = datetime.date.today().year
    plan_years = int(round(config.years))
    plan_year = current_year + plan_years

    w = 18
    sep = f"  +{'-' * 30}+{'-' * (w + 2)}+{'-' * (w + 2)}+"

    def _year_str(ff: FinancialFreedomResult) -> str:
        if ff.years is None:
            return "not achievable"
        return f"{current_year + ff.years} ({ff.years}y)"

    def _delta_str(ff: FinancialFreedomResult) -> str:
        if ff.years is None:
            return "—"
        delta = ff.years - plan_years
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta}y vs plan ({plan_year})"

    def _val(ff: FinancialFreedomResult, v: float) -> str:
        return format_currency(v) if ff.years is not None else "N/A"

    def _pct(ff: FinancialFreedomResult) -> str:
        return f"{ff.success_rate * 100:.1f}%" if ff.years is not None else "N/A"

    target_pct = config.mc_target_success_percent
    tail_pct = 100 - target_pct
    print(f"Financial freedom horizon  (MC ≥{target_pct}% success, current contributions)")
    print(sep)
    print(f"  | {'':30}| {'Scenario 1 (DE)':>{w}} | {'Scenario 2 (IT)':>{w}} |")
    print(sep)
    print(f"  | {'Freedom year':30}| {_year_str(ff_de):>{w}} | {_year_str(ff_it):>{w}} |")
    print(f"  | {'vs plan':30}| {_delta_str(ff_de):>{w}} | {_delta_str(ff_it):>{w}} |")
    print(f"  | {'MC success rate at freedom':30}| {_pct(ff_de):>{w}} | {_pct(ff_it):>{w}} |")
    print(f"  | {'Portfolio p50 at freedom':30}| {_val(ff_de, ff_de.portfolio_p50):>{w}} | {_val(ff_it, ff_it.portfolio_p50):>{w}} |")
    tail_label = f"Portfolio p{tail_pct}  at freedom"
    print(f"  | {tail_label:30}| {_val(ff_de, ff_de.portfolio_tail_pct):>{w}} | {_val(ff_it, ff_it.portfolio_tail_pct):>{w}} |")
    print(f"  | {'Net income/mo p50 at freedom':30}| {_val(ff_de, ff_de.monthly_income_p50):>{w}} | {_val(ff_it, ff_it.monthly_income_p50):>{w}} |")
    print(sep)
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
        f"| {'Scenario 1 (DE)':>{value_width}} "
        f"| {'Scenario 2 (IT)':>{value_width}} "
        f"| {'Delta (IT-DE)':>{delta_width}} |"
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


def print_current_situation_mc(
    mc1: MonteCarloResult,
    mc2: MonteCarloResult,
    current_stock_mo: float,
    current_bonds_mo: float,
    config: Config,
) -> None:
    w = 14
    sep = f"  +{'-' * 34}+{'-' * (w + 2)}+{'-' * (w + 2)}+"
    print(f"  Current monthly contributions (no change to plan):")
    print(f"    Stocks : {format_currency(current_stock_mo)}/mo")
    print(f"    Bonds  : {format_currency(current_bonds_mo)}/mo")
    print(f"    Cash   : {format_currency(config.annual_net_saving / 12 - current_stock_mo - current_bonds_mo)}/mo  (remainder)")
    print()
    print(sep)
    print(f"  | {'Metric':<34}| {'Scenario 1 (DE)':>{w}} | {'Scenario 2 (IT)':>{w}} |")
    tail_pct = 100 - config.mc_target_success_percent
    print(sep)
    print(f"  | {'MC success rate':<34}| {mc1.success_rate * 100:>{w - 1}.1f}% | {mc2.success_rate * 100:>{w - 1}.1f}% |")
    print(f"  | {'Median final portfolio (p50)':<34}"
          f"| {format_currency(mc1.final_portfolio_percentiles[50]):>{w}} "
          f"| {format_currency(mc2.final_portfolio_percentiles[50]):>{w}} |")
    bad_luck_label = f"Bad-luck final portfolio (p{tail_pct})"
    print(f"  | {bad_luck_label:<34}"
          f"| {format_currency(mc1.final_portfolio_percentiles[tail_pct]):>{w}} "
          f"| {format_currency(mc2.final_portfolio_percentiles[tail_pct]):>{w}} |")
    print(f"  | {'Median surplus(+)/shortfall(−)':<34}| {format_currency_signed(mc1.surplus_percentiles[50]):>{w}} | {format_currency_signed(mc2.surplus_percentiles[50]):>{w}} |")
    print(sep)


def _print_top_allocations(result: OptimalAllocationResult, top_n: int = 5) -> None:
    """Top N upfront allocations ranked by income success rate."""
    cw = 13
    sep = f"  +{'-' * (cw + 2)}+{('-' * (cw + 2) + '+') * 6}"
    hdr = (
        f"  | {'→Stocks':^{cw}} | {'→Bonds':^{cw}} | {'Cash kept':^{cw}}"
        f" | {'Stocks/mo*':^{cw}} | {'Bonds/mo*':^{cw}} | {'Best P(income)':^{cw}} |"
    )
    ranked = sorted(result.grid, key=lambda p: p.best_success_rate, reverse=True)[:top_n]
    print(sep)
    print(hdr)
    print(sep)
    for point in ranked:
        print(
            f"  | {format_currency(point.extra_stocks):^{cw}}"
            f" | {format_currency(point.extra_bonds):^{cw}}"
            f" | {format_currency(point.new_cash):^{cw}}"
            f" | {format_currency(point.income_stock_mo):^{cw}}"
            f" | {format_currency(point.income_bonds_mo):^{cw}}"
            f" | {f'{point.best_success_rate * 100:.1f}%':^{cw}} |"
        )
    print(sep)
    print(f"  * Stocks/mo and Bonds/mo are the optimised monthly split that maximises"
          f" income success rate for each upfront allocation.")


def _print_top_portfolio_allocations(result: OptimalAllocationResult, top_n: int = 5) -> None:
    """Top N upfront allocations ranked by lowest total monthly to reach target portfolio
    (stocks ≥ required AND bonds ≥ required)."""
    cw = 13
    sep = f"  +{'-' * (cw + 2)}+{('-' * (cw + 2) + '+') * 6}"
    hdr = (
        f"  | {'→Stocks':^{cw}} | {'→Bonds':^{cw}} | {'Cash kept':^{cw}}"
        f" | {'Stocks/mo':^{cw}} | {'Bonds/mo':^{cw}} | {'Total/mo':^{cw}} |"
    )
    achievable = [p for p in result.grid if p.achievable]
    if not achievable:
        print(f"  Target portfolio not achievable with any tested cash allocation.")
        return
    ranked = sorted(achievable, key=lambda p: p.monthly_stocks + p.monthly_bonds)[:top_n]
    print(sep)
    print(hdr)
    print(sep)
    for point in ranked:
        total = point.monthly_stocks + point.monthly_bonds
        print(
            f"  | {format_currency(point.extra_stocks):^{cw}}"
            f" | {format_currency(point.extra_bonds):^{cw}}"
            f" | {format_currency(point.new_cash):^{cw}}"
            f" | {format_currency(point.monthly_stocks):^{cw}}"
            f" | {format_currency(point.monthly_bonds):^{cw}}"
            f" | {format_currency(total):^{cw}} |"
        )
    print(sep)


def print_optimal_allocation_report(
    result_1: OptimalAllocationResult,
    result_2: OptimalAllocationResult,
    config: Config,
    n_simulations_per_point: int,
    verbose: bool = False,
) -> None:
    total_assets = config.current_stock_value + config.current_bonds_value + config.current_cash
    available = config.current_cash - result_1.min_cash_today

    target_pct = int(round(result_1.target_success * 100))
    print("\n\n")
    print(f"{'OPTIMAL CURRENT ALLOCATION':^70}")
    print(f"  Total current assets : {format_currency(total_assets)}")
    print(f"  Min cash kept today  : {format_currency(result_1.min_cash_today)}")
    print(f"  Redeployable cash    : {format_currency(available)}")
    print(f"  Grid step            : {format_currency(config.optimize_grid_step)}")
    print(f"  Simulations per point: {n_simulations_per_point:,}  (5k for speed; suggestions table used 10k)")

    # ── Part 1: minimum monthly to reach target success ───────────────────────
    print()
    print(f"  Part 1 — Minimum monthly contribution to reach {target_pct}% success")
    print(f"  (searches the cash reallocation that minimises stocks/mo + bonds/mo)")
    for result in [result_1, result_2]:
        opt = result.optimal
        print(f"  {'─' * 66}")
        print(f"  {result.scenario_name}")
        print(f"  {'─' * 66}")

        if opt is None:
            print(f"  {target_pct}% target not achievable with any tested allocation.")
            continue

        print(f"  OPTIMAL — minimum total monthly contribution to reach {target_pct}% success:")
        print()
        print(f"    Move to stocks today  : {format_currency(opt.extra_stocks)}")
        print(f"    Move to bonds today   : {format_currency(opt.extra_bonds)}")
        print(f"    Keep as cash today    : {format_currency(opt.new_cash)}")
        print()
        print(f"    Stocks at t=0  : {format_currency(opt.new_stocks)}")
        print(f"    Bonds at t=0   : {format_currency(opt.new_bonds)}")
        print(f"    Cash at t=0    : {format_currency(opt.new_cash)}")
        print()
        print(f"    Monthly stocks : {format_currency(opt.monthly_stocks)}/mo")
        print(f"    Monthly bonds  : {format_currency(opt.monthly_bonds)}/mo")
        print(f"    Total monthly  : {format_currency(opt.total_monthly)}/mo")
        print()

        if verbose:
            cw = 14
            vsep = f"  +{'-' * (cw + 2)}+{('-' * (cw + 2) + '+') * 7}"
            hdr = (
                f"  | {'→Stocks':^{cw}} | {'→Bonds':^{cw}} | {'Cash kept':^{cw}}"
                f" | {'Stocks t=0':^{cw}} | {'Bonds t=0':^{cw}}"
                f" | {'Stocks/mo':^{cw}} | {'Total/mo':^{cw}} |"
            )
            print(f"  Full grid (sorted by total monthly, * = optimal):")
            print()
            print(vsep)
            print(hdr)
            print(vsep)
            sorted_grid = sorted(result.grid, key=lambda p: p.total_monthly if p.achievable else float("inf"))
            for point in sorted_grid:
                if not point.achievable:
                    break
                marker = " *" if (
                    abs(point.extra_stocks - opt.extra_stocks) < 1
                    and abs(point.extra_bonds - opt.extra_bonds) < 1
                ) else "  "
                print(
                    f"  | {format_currency(point.extra_stocks) + marker:^{cw}}"
                    f" | {format_currency(point.extra_bonds):^{cw}}"
                    f" | {format_currency(point.new_cash):^{cw}}"
                    f" | {format_currency(point.new_stocks):^{cw}}"
                    f" | {format_currency(point.new_bonds):^{cw}}"
                    f" | {format_currency(point.monthly_stocks):^{cw}}"
                    f" | {format_currency(point.total_monthly):^{cw}} |"
                )
            print(vsep)
            print()

    # ── Part 2: top allocations by best achievable success rate ───────────────
    print()
    print(f"  Part 2 — Top cash reallocations by income success rate (Q2)")
    print(f"  (monthly stock contribution kept at current plan; only today's cash reallocation varies)")
    for result in [result_1, result_2]:
        print(f"  {'─' * 66}")
        print(f"  {result.scenario_name}")
        print(f"  {'─' * 66}")
        _print_top_allocations(result)
