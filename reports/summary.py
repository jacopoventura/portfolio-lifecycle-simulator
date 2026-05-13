import datetime

from models import (
    Config,
    ContributionCagrGrid,
    CombinedGrid,
    MonteCarloResult,
    OptimalAllocationResult,
    ProjectionResult,
    ScenarioResult,
    YearsToTargetsResult,
)
from reports.formatters import (
    format_currency,
    format_currency_signed,
    format_ratio_as_percent,
)
from reports.section2 import _find_best_combined, _find_best_income_split, _find_min_contrib
from reports.section3 import _print_top_allocations, _print_top_portfolio_allocations


def _q_section(title: str) -> None:
    print()
    print(f"{title}")
    print()


def _q_subsection(title: str) -> None:
    print(f"{title}")
    print()


def print_summary_q0(
    config: Config,
    projection_result: ProjectionResult,
    scenario_1: ScenarioResult,
    scenario_2: ScenarioResult,
    mc1: MonteCarloResult,
    mc2: MonteCarloResult,
    years_de: YearsToTargetsResult,
    years_it: YearsToTargetsResult,
) -> None:
    _q_section("CURRENT SITUATION")

    total = config.current_stock_value + config.current_bonds_value + config.current_cash
    monthly = config.annual_net_saving / 12
    s_mo = monthly * config.maximum_stock_monthly_investment_pct
    b_mo = monthly * config.maximum_bonds_monthly_investment_pct
    c_mo = max(0.0, monthly - s_mo - b_mo)

    print(f"  Portfolio today : {format_currency(total)}")
    print(f"    stocks {format_currency(config.current_stock_value)}"
          f"  |  bonds {format_currency(config.current_bonds_value)}"
          f"  |  cash {format_currency(config.current_cash)}")
    print(f"  Annual saving   : {format_currency(config.annual_net_saving)}/yr")
    print(f"    stocks {format_currency(s_mo)}/mo"
          f"  |  bonds {format_currency(b_mo)}/mo"
          f"  |  cash {format_currency(c_mo)}/mo")
    print()

    tgt = config.mc_target_success_percent
    current_year = datetime.date.today().year
    plan_years = int(round(config.years))
    plan_year = current_year + plan_years

    def _yr(y: int | None) -> str:
        if y is None:
            return "not achievable"
        delta = y - plan_years
        sign = "+" if delta > 0 else ""
        return f"{current_year + y}  ({sign}{delta}y)"

    rw, pw, yw = 22, 7, 16
    sep2 = f"  +{'-' * (rw + 2)}+{'-' * (pw + 2)}+{'-' * (yw + 2)}+{'-' * (pw + 2)}+{'-' * (yw + 2)}+"
    print(f"  MC success — current plan  (target ≥{tgt}%, plan year {plan_year}):")
    print(sep2)
    print(f"  | {'':<{rw}} "
          f"| {'DE %':>{pw}} | {f'DE yr → {tgt}%':>{yw}} "
          f"| {'IT %':>{pw}} | {f'IT yr → {tgt}%':>{yw}} |")
    print(sep2)

    def _row(label: str, r_de: float, y_de: int | None, r_it: float, y_it: int | None) -> None:
        print(f"  | {label:<{rw}} "
              f"| {r_de * 100:>{pw - 1}.1f}% | {_yr(y_de):>{yw}} "
              f"| {r_it * 100:>{pw - 1}.1f}% | {_yr(y_it):>{yw}} |")

    _row("Stocks ≥ req",
         mc1.success_rate_stock_only, years_de.years_to_stocks,
         mc2.success_rate_stock_only, years_it.years_to_stocks)
    _row("Bonds ≥ req",
         mc1.success_rate_bond_only, years_de.years_to_bonds,
         mc2.success_rate_bond_only, years_it.years_to_bonds)
    _row("Stocks AND Bonds ≥ req",
         mc1.success_rate, years_de.years_to_both,
         mc2.success_rate, years_it.years_to_both)
    _row("Income ≥ target",
         mc1.success_rate_income_only, years_de.years_to_income,
         mc2.success_rate_income_only, years_it.years_to_income)
    print(sep2)

    target_success = tgt / 100

    def _status_portfolio(rate: float) -> str:
        if rate >= target_success:
            return "on track"
        return f"WARNING — financial plan must be revised to improve the success rate to {tgt}%"

    def _status_income(rate: float) -> str:
        if rate >= target_success:
            return "on track to reach the required income"
        return f"WARNING — financial plan must be revised to improve the income success rate to {tgt}%"

    print()
    print(f"  Portfolio composition (stocks AND bonds ≥ req):")
    print(f"    DE: {_status_portfolio(mc1.success_rate)}")
    print(f"    IT: {_status_portfolio(mc2.success_rate)}")


def print_summary_q1(
    config: Config,
    scenario_1: ScenarioResult,
    scenario_2: ScenarioResult,
    future_monthly_income: float,
    target_annual_net_income: float
) -> None:
    _q_section(
        f"WHAT YOU NEED AT HORIZON  ({config.years:.0f}y, {config.mc_target_success_percent}% MC confidence)",
    )
    stock_pct = 100 - config.bonds_percentage
    print(f"  Target income today           : {format_currency(config.target_monthly_retirement_income)}/mo")
    print(f"  Inflation-adjusted at horizon : {format_currency(future_monthly_income)}/mo"
          f"  ({format_currency(target_annual_net_income)}/yr)")
    print(f"  Target portfolio              : stock {stock_pct}% / bonds {config.bonds_percentage}%")
    print()
    print(f"  Deterministic estimation")

    cw = 16
    dw = 12
    sep = f"  +{'-' * 32}+{'-' * (cw + 2)}+{'-' * (cw + 2)}+{'-' * (dw + 2)}+"
    print(sep)
    print(f"  | {'Metric':<32}| {'DE':>{cw}} | {'IT':>{cw}} | {'IT − DE':>{dw}} |")
    print(sep)

    def _row(label: str, v1: float, v2: float) -> None:
        print(f"  | {label:<32}"
              f"| {format_currency(v1):>{cw}} "
              f"| {format_currency(v2):>{cw}} "
              f"| {format_currency_signed(v2 - v1):>{dw}} |")

    def _row_pct(label: str, v1: float, v2: float) -> None:
        delta = f"{(v2 - v1) * 100:+.2f} pp"
        print(f"  | {label:<32}"
              f"| {format_ratio_as_percent(v1):>{cw}} "
              f"| {format_ratio_as_percent(v2):>{cw}} "
              f"| {delta:>{dw}} |")

    _row("Required total portfolio", scenario_1.required_total_portfolio, scenario_2.required_total_portfolio)
    _row("  of which stocks", scenario_1.required_stock_amount, scenario_2.required_stock_amount)
    _row("  of which bonds", scenario_1.required_bonds_amount, scenario_2.required_bonds_amount)
    # _row("Required stocks/mo (yr 1)", scenario_1.required_monthly_contribution, scenario_2.required_monthly_contribution)
    # _row("Required bonds/mo (yr 1)", scenario_1.required_monthly_bonds_contribution, scenario_2.required_monthly_bonds_contribution)
    # _row_pct("Portfolio net income rate", scenario_1.portfolio_net_income_rate, scenario_2.portfolio_net_income_rate)
    print(sep)


def print_summary_q2a(
    combined_de: CombinedGrid,
    combined_it: CombinedGrid,
    config: Config,
    target_success: float,
    cagr_stock_s1: float | None,
    cagr_stock_s2: float | None,
) -> None:
    print()
    _q_section("HOW TO GET THERE")
    _q_subsection(f"Problem 1 — Optimization of monthly contribution for different stock CAGR (keep current cash)")
    print(f"Exact split  (stocks ≥ req AND bonds ≥ req, {config.mc_target_success_percent}% confidence)")

    assumed = config.expected_yearly_growth_stocks_percent / 100
    target_pct = config.mc_target_success_percent
    monthly_saving = config.annual_net_saving / 12

    def _cagr_fmt(cagr: float | None) -> str:
        if cagr is None:
            return "not achievable in scanned range"
        delta = cagr * 100 - config.expected_yearly_growth_stocks_percent
        return f"{cagr * 100:.1f}%  ({delta:+.1f}pp vs assumed {config.expected_yearly_growth_stocks_percent:.1f}%)"

    print(f"  Min CAGR for {target_pct}% exact split (stocks AND bonds):")
    print(f"    DE : {_cagr_fmt(cagr_stock_s1)}")
    print(f"    IT : {_cagr_fmt(cagr_stock_s2)}")
    print()

    cw = 12
    pw = 8
    grp = cw * 2 + pw + 8
    sep = f"  {'─' * (8 + grp * 2 + 2)}"
    print(f"  Best P(stocks ≥ req AND bonds ≥ req) at each CAGR  (* = assumed {assumed * 100:.0f}%):")
    print(f"  {'CAGR':6}  {'── Scenario 1 (DE) ──':^{grp}}  {'── Scenario 2 (IT) ──':^{grp}}")
    print(f"  {'':6}  {'Stocks/mo':>{cw}}  {'Bonds/mo':>{cw + 1}}  {'P(both)':>{pw}}"
          f"  {'Stocks/mo':>{cw}}  {'Bonds/mo':>{cw + 1}}  {'P(both)':>{pw}}")
    print(sep)

    for cagr_idx, cagr in enumerate(combined_de.cagr_values):
        s_de, b_de, r_de = _find_best_combined(combined_de, cagr_idx)
        s_it, b_it, r_it = _find_best_combined(combined_it, cagr_idx)
        marker = "*" if abs(cagr - assumed) < 1e-4 else " "
        over_de = "†" if (s_de + b_de) > monthly_saving + 0.01 else " "
        over_it = "†" if (s_it + b_it) > monthly_saving + 0.01 else " "
        flag_de = "✓" if r_de >= target_success else " "
        flag_it = "✓" if r_it >= target_success else " "
        b_de_str = format_currency(b_de) + over_de
        b_it_str = format_currency(b_it) + over_it
        print(f"  {int(cagr * 100):2d}% {marker} "
              f" {format_currency(s_de):>{cw}}"
              f"  {b_de_str:>{cw + 1}}"
              f"  {flag_de}{r_de * 100:>{pw - 2}.1f}%"
              f"  {format_currency(s_it):>{cw}}"
              f"  {b_it_str:>{cw + 1}}"
              f"  {flag_it}{r_it * 100:>{pw - 2}.1f}%")

    print(sep)
    print(f"  ✓ meets {target_pct}% target  |  † stocks + bonds exceed monthly saving ({format_currency(monthly_saving)}/mo)")


def print_summary_q2b(
    grids_de: list[ContributionCagrGrid],
    grids_it: list[ContributionCagrGrid],
    cagr_income_s1: float | None,
    cagr_income_s2: float | None,
    config: Config,
    target_success: float,
) -> None:
    print()
    _q_subsection(f"Any split  (income ≥ target, {config.mc_target_success_percent}% confidence)")

    assumed = config.expected_yearly_growth_stocks_percent
    target_pct = config.mc_target_success_percent
    monthly_saving = config.annual_net_saving / 12

    def _cagr_fmt(cagr: float | None) -> str:
        if cagr is None:
            return "not achievable in scanned range"
        delta = cagr * 100 - assumed
        return f"{cagr * 100:.1f}%  ({delta:+.1f}pp vs assumed {assumed:.1f}%)"

    print(f"  Min CAGR for {target_pct}% income target (any stock/bond split):")
    print(f"    DE : {_cagr_fmt(cagr_income_s1)}")
    print(f"    IT : {_cagr_fmt(cagr_income_s2)}")
    print()

    assumed_frac = assumed / 100
    cw = 12
    mw = 5
    grp = cw * 3 + 4
    sep = f"  {'─' * (mw + 2 + grp * 2 + 2)}"
    print(f"  Min stocks/mo for {target_pct}% income target at {assumed:.1f}%/yr CAGR:")
    print(sep)
    print(f"  {'Mult':<{mw}}  {'── Scenario 1 (DE) ──':^{grp}}  {'── Scenario 2 (IT) ──':^{grp}}")
    print(f"  {'':<{mw}}  {'Bonds/mo':>{cw}}  {'Stocks/mo':>{cw}}  {'Cash/mo':>{cw}}"
          f"  {'Bonds/mo':>{cw}}  {'Stocks/mo':>{cw}}  {'Cash/mo':>{cw}}")
    print(sep)

    for grid_de, grid_it in zip(grids_de, grids_it):
        cagr_idx = min(
            range(len(grid_de.cagr_values)),
            key=lambda i: abs(grid_de.cagr_values[i] - assumed_frac),
        )
        min_de = _find_min_contrib(grid_de, cagr_idx, use_hard=False, target_success=target_success)
        min_it = _find_min_contrib(grid_it, cagr_idx, use_hard=False, target_success=target_success)

        b_label = f"{grid_de.bonds_multiplier}×"
        b_de = format_currency(grid_de.bonds_monthly)
        b_it = format_currency(grid_it.bonds_monthly)
        s_de = format_currency(min_de) if min_de is not None else "n/a"
        c_de = format_currency(max(0.0, monthly_saving - (min_de or 0.0) - grid_de.bonds_monthly)) if min_de is not None else "—"
        s_it = format_currency(min_it) if min_it is not None else "n/a"
        c_it = format_currency(max(0.0, monthly_saving - (min_it or 0.0) - grid_it.bonds_monthly)) if min_it is not None else "—"

        print(f"  {b_label:<{mw}}  {b_de:>{cw}}  {s_de:>{cw}}  {c_de:>{cw}}"
              f"  {b_it:>{cw}}  {s_it:>{cw}}  {c_it:>{cw}}")

    print(sep)

    # ── Best P(income) at each CAGR ───────────────────────────────────────────
    cw2, pw = 12, 10
    grp = cw2 * 2 + pw + 8
    sep2 = f"  {'─' * (8 + grp * 2 + 2)}"
    print()
    print(f"  Best P(portfolio income ≥ required income) at each CAGR  (* = assumed {assumed:.0f}%):")
    print(f"  {'CAGR':6}  {'── Scenario 1 (DE) ──':^{grp}}  {'── Scenario 2 (IT) ──':^{grp}}")
    print(f"  {'':6}  {'Stocks/mo':>{cw2}}  {'Bonds/mo':>{cw2}}  {'P(income)':>{pw}}"
          f"  {'Stocks/mo':>{cw2}}  {'Bonds/mo':>{cw2}}  {'P(income)':>{pw}}")
    print(sep2)

    cagr_values = grids_de[0].cagr_values
    for cagr_idx, cagr in enumerate(cagr_values):
        s_de, b_de, r_de = _find_best_income_split(grids_de, cagr_idx)
        s_it, b_it, r_it = _find_best_income_split(grids_it, cagr_idx)
        marker = "*" if abs(cagr - assumed_frac) < 1e-4 else " "
        flag_de = "✓" if r_de >= target_success else " "
        flag_it = "✓" if r_it >= target_success else " "
        print(f"  {int(cagr * 100):2d}% {marker} "
              f" {format_currency(s_de):>{cw2}}"
              f"  {format_currency(b_de):>{cw2}}"
              f"  {flag_de}{r_de * 100:>{pw - 2}.1f}%"
              f"  {format_currency(s_it):>{cw2}}"
              f"  {format_currency(b_it):>{cw2}}"
              f"  {flag_it}{r_it * 100:>{pw - 2}.1f}%")

    print(sep2)
    print(f"  ✓ meets {target_pct}% target")


def print_summary_q4(
    opt_result_1: OptimalAllocationResult,
    opt_result_2: OptimalAllocationResult,
    config: Config,
) -> None:
    print()
    _q_subsection("Problem 2 — Optimization of monthly contribution and current asset allocation for different stock CAGR ")

    total_assets = config.current_stock_value + config.current_bonds_value + config.current_cash
    available = config.current_cash - opt_result_1.min_cash_today
    target_pct = int(round(opt_result_1.target_success * 100))

    print(f"  Total assets     : {format_currency(total_assets)}")
    print(f"  Min cash kept    : {format_currency(opt_result_1.min_cash_today)}")
    print(f"  Redeployable     : {format_currency(available)}")

    print()
    print(f"  2a — Maximize income probability  (P(income ≥ target))")
    print(f"       Top 3 upfront allocations ranked by best income success rate.")
    for result in [opt_result_1, opt_result_2]:
        print(f"  {'─' * 66}")
        print(f"  {result.scenario_name}")
        _print_top_allocations(result, top_n=3)

    print()
    print(f"  2b — Reach target portfolio  (stocks ≥ req AND bonds ≥ req, {target_pct}% confidence)")
    print(f"       Top 3 upfront allocations ranked by lowest total monthly contribution.")
    for result in [opt_result_1, opt_result_2]:
        print(f"  {'─' * 66}")
        print(f"  {result.scenario_name}")
        _print_top_portfolio_allocations(result, top_n=3)
