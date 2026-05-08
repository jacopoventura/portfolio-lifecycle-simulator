import datetime

from models import (
    Config,
    ContributionCagrGrid,
    CombinedGrid,
    FinancialFreedomResult,
    MonteCarloResult,
    OptimalAllocationResult,
    ProjectionResult,
    ScenarioResult,
)
from reports.formatters import (
    format_currency,
    format_currency_signed,
    format_ratio_as_percent,
)
from reports.section2 import _find_best_combined, _find_min_contrib
from reports.section3 import _print_top_allocations


def _q_section(label: str, title: str) -> None:
    print("\n\n")
    print("=" * 70)
    print(f"  {label}.  {title}")
    print("=" * 70)
    print()


def print_summary_q0(
    config: Config,
    projection_result: ProjectionResult,
    scenario_1: ScenarioResult,
    scenario_2: ScenarioResult,
    mc1: MonteCarloResult,
    mc2: MonteCarloResult,
) -> None:
    _q_section(" ", "WHERE YOU ARE NOW")

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

    cw = 14
    sep = f"  +{'-' * 14}+{('-' * (cw + 2) + '+') * 4}"
    print(f"  At horizon ({config.years:.0f}y) — current plan vs target:")
    print(sep)
    print(f"  | {'':14}| {'Projected':>{cw}} | {'Required DE':>{cw}} | {'Gap DE':>{cw}} | {'Gap IT':>{cw}} |")
    print(sep)

    proj_s = projection_result.final_stock_value
    proj_b = projection_result.final_bonds_value

    print(f"  | {'Stocks':<14}"
          f"| {format_currency(proj_s):>{cw}} "
          f"| {format_currency(scenario_1.required_stock_amount):>{cw}} "
          f"| {format_currency_signed(proj_s - scenario_1.required_stock_amount):>{cw}} "
          f"| {format_currency_signed(proj_s - scenario_2.required_stock_amount):>{cw}} |")
    print(f"  | {'Bonds':<14}"
          f"| {format_currency(proj_b):>{cw}} "
          f"| {format_currency(scenario_1.required_bonds_amount):>{cw}} "
          f"| {format_currency_signed(proj_b - scenario_1.required_bonds_amount):>{cw}} "
          f"| {format_currency_signed(proj_b - scenario_2.required_bonds_amount):>{cw}} |")
    print(sep)
    print()

    tgt = config.mc_target_success_percent
    lw = 36
    print(f"  MC success rate — current plan  (target ≥{tgt}%):")
    print(f"    {'Q1a  Stocks ≥ required':<{lw}}"
          f": DE {mc1.success_rate_stock_only * 100:>5.1f}%   IT {mc2.success_rate_stock_only * 100:>5.1f}%")
    print(f"    {'Q1b  Bonds ≥ required':<{lw}}"
          f": DE {mc1.success_rate_bond_only * 100:>5.1f}%   IT {mc2.success_rate_bond_only * 100:>5.1f}%")
    print(f"    {'Q2   Income ≥ target (any split)':<{lw}}"
          f": DE {mc1.success_rate_income_only * 100:>5.1f}%   IT {mc2.success_rate_income_only * 100:>5.1f}%")


def print_summary_q1(
    config: Config,
    scenario_1: ScenarioResult,
    scenario_2: ScenarioResult,
    future_monthly_income: float,
    target_annual_net_income: float,
) -> None:
    _q_section(
        " ",
        f"WHAT YOU NEED AT HORIZON  ({config.years:.0f}y, {config.mc_target_success_percent}% MC confidence)",
    )

    print(f"  Target income today           : {format_currency(config.target_monthly_retirement_income)}/mo")
    print(f"  Inflation-adjusted at horizon : {format_currency(future_monthly_income)}/mo"
          f"  ({format_currency(target_annual_net_income)}/yr)")
    print()

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
    _row("Required stocks/mo (yr 1)", scenario_1.required_monthly_contribution, scenario_2.required_monthly_contribution)
    _row("Required bonds/mo (yr 1)", scenario_1.required_monthly_bonds_contribution, scenario_2.required_monthly_bonds_contribution)
    _row_pct("Portfolio net income rate", scenario_1.portfolio_net_income_rate, scenario_2.portfolio_net_income_rate)
    print(sep)


def print_summary_q2a(
    combined_de: CombinedGrid,
    combined_it: CombinedGrid,
    config: Config,
    target_success: float,
    cagr_stock_s1: float | None,
    cagr_stock_s2: float | None,
) -> None:
    _q_section(
        " ",
        f"HOW TO GET THERE — EXACT SPLIT  (stocks ≥ req AND bonds ≥ req, {config.mc_target_success_percent}% confidence)",
    )

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
    grp = cw * 3 + pw + 10
    sep = f"  {'─' * (8 + grp * 2 + 2)}"
    print(f"  Best P(stocks ≥ req AND bonds ≥ req) at each CAGR  (* = assumed {assumed * 100:.0f}%):")
    print(f"  {'CAGR':6}  {'── Scenario 1 (DE) ──':^{grp}}  {'── Scenario 2 (IT) ──':^{grp}}")
    print(f"  {'':6}  {'Stocks/mo':>{cw}}  {'Bonds/mo':>{cw}}  {'Total/mo':>{cw}}  {'P(both)':>{pw}}"
          f"  {'Stocks/mo':>{cw}}  {'Bonds/mo':>{cw}}  {'Total/mo':>{cw}}  {'P(both)':>{pw}}")
    print(sep)

    for cagr_idx, cagr in enumerate(combined_de.cagr_values):
        s_de, b_de, r_de = _find_best_combined(combined_de, cagr_idx)
        s_it, b_it, r_it = _find_best_combined(combined_it, cagr_idx)
        marker = "*" if abs(cagr - assumed) < 1e-4 else " "
        over_de = "†" if (s_de + b_de) > monthly_saving + 0.01 else " "
        over_it = "†" if (s_it + b_it) > monthly_saving + 0.01 else " "
        flag_de = "✓" if r_de >= target_success else " "
        flag_it = "✓" if r_it >= target_success else " "
        t_de = format_currency(s_de + b_de) + over_de
        t_it = format_currency(s_it + b_it) + over_it
        print(f"  {int(cagr * 100):2d}% {marker} "
              f" {format_currency(s_de):>{cw}}"
              f"  {format_currency(b_de):>{cw}}"
              f"  {t_de:>{cw + 1}}"
              f"  {flag_de}{r_de * 100:>{pw - 2}.1f}%"
              f"  {format_currency(s_it):>{cw}}"
              f"  {format_currency(b_it):>{cw}}"
              f"  {t_it:>{cw + 1}}"
              f"  {flag_it}{r_it * 100:>{pw - 2}.1f}%")

    print(sep)
    print(f"  ✓ meets {target_pct}% target  |  † total exceeds monthly saving ({format_currency(monthly_saving)}/mo)")


def print_summary_q2b(
    grids_de: list[ContributionCagrGrid],
    grids_it: list[ContributionCagrGrid],
    cagr_income_s1: float | None,
    cagr_income_s2: float | None,
    config: Config,
    target_success: float,
) -> None:
    _q_section(
        " ",
        f"HOW TO GET THERE — ANY SPLIT  (income ≥ target, {config.mc_target_success_percent}% confidence)",
    )

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
    cw = 14
    sep = f"  {'─' * (10 + (cw + 2) * 4)}"
    print(f"  Min stocks/mo for {target_pct}% income target at {assumed:.1f}%/yr CAGR:")
    print(sep)
    print(f"  {'Bonds':10}  {'DE stocks/mo':>{cw}}  {'DE cash/mo':>{cw}}  {'IT stocks/mo':>{cw}}  {'IT cash/mo':>{cw}}")
    print(sep)

    for grid_de, grid_it in zip(grids_de, grids_it):
        cagr_idx = min(
            range(len(grid_de.cagr_values)),
            key=lambda i: abs(grid_de.cagr_values[i] - assumed_frac),
        )
        min_de = _find_min_contrib(grid_de, cagr_idx, use_hard=False, target_success=target_success)
        min_it = _find_min_contrib(grid_it, cagr_idx, use_hard=False, target_success=target_success)

        b_label = f"{grid_de.bonds_multiplier}× bonds"
        s_de = format_currency(min_de) if min_de is not None else "n/a"
        c_de = format_currency(max(0.0, monthly_saving - (min_de or 0.0) - grid_de.bonds_monthly)) if min_de is not None else "—"
        s_it = format_currency(min_it) if min_it is not None else "n/a"
        c_it = format_currency(max(0.0, monthly_saving - (min_it or 0.0) - grid_it.bonds_monthly)) if min_it is not None else "—"

        print(f"  {b_label:10}  {s_de:>{cw}}  {c_de:>{cw}}  {s_it:>{cw}}  {c_it:>{cw}}")

    print(sep)


def print_summary_q3(
    ff_de: FinancialFreedomResult,
    ff_it: FinancialFreedomResult,
    config: Config,
) -> None:
    _q_section(" ", "FINANCIAL FREEDOM ETA  (current contributions, any portfolio composition)")

    current_year = datetime.date.today().year
    plan_years = int(round(config.years))
    plan_year = current_year + plan_years
    tgt = config.mc_target_success_percent
    tail_pct = 100 - tgt

    def _year(ff: FinancialFreedomResult) -> str:
        return "not achievable" if ff.years is None else f"{current_year + ff.years} ({ff.years}y)"

    def _delta(ff: FinancialFreedomResult) -> str:
        if ff.years is None:
            return "—"
        d = ff.years - plan_years
        return f"{'+'if d > 0 else ''}{d}y vs plan ({plan_year})"

    def _pct(ff: FinancialFreedomResult) -> str:
        return f"{ff.success_rate * 100:.1f}%" if ff.years is not None else "N/A"

    def _val(ff: FinancialFreedomResult, v: float) -> str:
        return format_currency(v) if ff.years is not None else "N/A"

    print(f"  Planned horizon : {plan_year} ({plan_years}y)  |  MC confidence target: ≥{tgt}%")
    print()

    w = 20
    sep = f"  +{'-' * 32}+{'-' * (w + 2)}+{'-' * (w + 2)}+"
    print(sep)
    print(f"  | {'':32}| {'Scenario 1 (DE)':>{w}} | {'Scenario 2 (IT)':>{w}} |")
    print(sep)

    def _row(label: str, v_de: str, v_it: str) -> None:
        print(f"  | {label:<32}| {v_de:>{w}} | {v_it:>{w}} |")

    _row("Freedom year", _year(ff_de), _year(ff_it))
    _row("vs planned horizon", _delta(ff_de), _delta(ff_it))
    _row("MC success rate at freedom", _pct(ff_de), _pct(ff_it))
    _row("Portfolio p50 at freedom", _val(ff_de, ff_de.portfolio_p50), _val(ff_it, ff_it.portfolio_p50))
    _row(f"Portfolio p{tail_pct} at freedom", _val(ff_de, ff_de.portfolio_tail_pct), _val(ff_it, ff_it.portfolio_tail_pct))
    _row("Net income/mo p50 at freedom", _val(ff_de, ff_de.monthly_income_p50), _val(ff_it, ff_it.monthly_income_p50))
    print(sep)


def print_summary_q4(
    opt_result_1: OptimalAllocationResult,
    opt_result_2: OptimalAllocationResult,
    config: Config,
) -> None:
    _q_section(" ", "BEST CASH ALLOCATION TODAY")

    total_assets = config.current_stock_value + config.current_bonds_value + config.current_cash
    available = config.current_cash - opt_result_1.min_cash_today
    target_pct = int(round(opt_result_1.target_success * 100))

    print(f"  Total assets     : {format_currency(total_assets)}")
    print(f"  Min cash kept    : {format_currency(opt_result_1.min_cash_today)}")
    print(f"  Redeployable     : {format_currency(available)}")
    print()

    print(f"  Part 1 — Minimum monthly to reach {target_pct}% success:")
    for result in [opt_result_1, opt_result_2]:
        opt = result.optimal
        print(f"  {'─' * 66}")
        print(f"  {result.scenario_name}")
        if opt is None:
            print(f"    {target_pct}% not achievable with any tested allocation.")
        else:
            print(f"    Move today    : +{format_currency(opt.extra_stocks)} stocks"
                  f"  |  +{format_currency(opt.extra_bonds)} bonds"
                  f"  |  keep {format_currency(opt.new_cash)} cash")
            print(f"    Monthly plan  : {format_currency(opt.monthly_stocks)}/mo stocks"
                  f"  |  {format_currency(opt.monthly_bonds)}/mo bonds"
                  f"  |  total {format_currency(opt.total_monthly)}/mo")

    print()
    print(f"  Part 2 — Top 3 cash reallocations by income success rate (Q2):")
    for result in [opt_result_1, opt_result_2]:
        print(f"  {'─' * 66}")
        print(f"  {result.scenario_name}")
        _print_top_allocations(result, top_n=3)
