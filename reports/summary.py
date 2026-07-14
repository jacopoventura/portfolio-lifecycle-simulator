import datetime

from models import (
    CagrRealismSnapshot,
    Config,
    ContributionCagrGrid,
    CombinedGrid,
    DecumulationSummary,
    FinancialFreedomResult,
    MonteCarloResult,
    OptimalAllocationResult,
    ProjectionResult,
    ScenarioResult,
    SensitivityWaterfallResult,
    VerdictInputs,
    YearsToTargetsResult,
)
from reports.formatters import (
    format_currency,
    format_currency_signed,
    format_millions,
    format_ratio_as_percent,
)
from reports.section2 import (
    _find_min_total_combined,
    _find_min_total_income_split,
)
from reports.section3 import _print_top_allocations, _print_top_portfolio_allocations


def _q_section(title: str) -> None:
    print()
    print(f"{title}")
    print()


def _q_subsection(title: str) -> None:
    print(f"{title}")
    print()


def print_current_situation_summary(
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
    s_mo = monthly * config.stock_monthly_investment_pct
    b_mo = monthly * config.bonds_monthly_investment_pct
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

    def _yr(y: int | None) -> str:
        if y is None:
            return "not achievable"
        delta = y - plan_years
        sign = "+" if delta > 0 else ""
        return f"{current_year + y}  ({sign}{delta}y)"

    rw, pw, yw = 22, 7, 16
    sep2 = f"  +{'-' * (rw + 2)}+{'-' * (pw + 2)}+{'-' * (yw + 2)}+{'-' * (pw + 2)}+{'-' * (yw + 2)}+"
    print(f"  Success probability (Monte Carlo) : target probability ≥ {tgt}%, {plan_years} years horizon")
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


def print_what_you_need_at_horizon(
    config: Config,
    scenario_1: ScenarioResult,
    scenario_2: ScenarioResult,
    future_monthly_income: float,
    target_annual_net_income: float
) -> None:
    """Render §1 'Required portfolio at horizon' table.

    Required portfolios here use the *at-retirement* stock gain fraction (computed once
    at the projection horizon, ~0.40 for a 15y accumulation). The decumulation MC uses
    a *horizon-averaged* 0.65 gain-fraction proxy in its tax expression and reports
    MC-minimum required wealth (always ≥ this deterministic number by the SORR premium).
    """
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


def print_how_to_get_there_portfolio(
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
    print(f"1a. Target portfolio with required stock/bond split, {config.mc_target_success_percent}% success rate")

    assumed = config.expected_yearly_growth_stocks_percent / 100
    target_pct = config.mc_target_success_percent
    monthly_saving = config.annual_net_saving / 12

    def _cagr_fmt(cagr: float | None) -> str:
        if cagr is None:
            return "not achievable in scanned range"
        delta = cagr * 100 - config.expected_yearly_growth_stocks_percent
        return f"{cagr * 100:.1f}%  ({delta:+.1f}pp vs assumed {config.expected_yearly_growth_stocks_percent:.1f}%)"

    print(f"  Min required CAGR:")
    print(f"    DE : {_cagr_fmt(cagr_stock_s1)}")
    print(f"    IT : {_cagr_fmt(cagr_stock_s2)}")
    print()

    max_stock_mo = max(combined_de.stocks_values)
    max_bonds_de = max(combined_de.bonds_values)
    max_bonds_it = max(combined_it.bonds_values)
    cw = 12
    pw = 8
    grp = cw * 2 + pw + 8
    sep = f"  {'─' * (8 + grp * 2 + 2)}"
    print(f"  Min total contribution per month (* = current base assumption;"
          f" search caps: stocks {format_currency(max_stock_mo)}/mo"
          f" · bonds DE {format_currency(max_bonds_de)}/mo"
          f" · bonds IT {format_currency(max_bonds_it)}/mo):")
    print(f"  {'CAGR':6}  {'── Scenario 1 (DE) ──':^{grp}}  {'── Scenario 2 (IT) ──':^{grp}}")
    print(f"  {'':6}  {'Stocks/mo':>{cw}}  {'Bonds/mo':>{cw + 1}}  {'P(both)':>{pw}}"
          f"  {'Stocks/mo':>{cw}}  {'Bonds/mo':>{cw + 1}}  {'P(both)':>{pw}}")
    print(sep)

    for cagr_idx, cagr in enumerate(combined_de.cagr_values):
        s_de, b_de, r_de = _find_min_total_combined(combined_de, cagr_idx, target_success)
        s_it, b_it, r_it = _find_min_total_combined(combined_it, cagr_idx, target_success)
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
    print(f"  ✓ meets {target_pct}% target; rows without ✓ show best achievable rate (target unreachable in scan range)")
    print(f"  † stocks + bonds exceed monthly saving ({format_currency(monthly_saving)}/mo)")


def print_how_to_get_there_income(
    grids_de: list[ContributionCagrGrid],
    grids_it: list[ContributionCagrGrid],
    cagr_income_s1: float | None,
    cagr_income_s2: float | None,
    config: Config,
    target_success: float,
) -> None:
    print()
    _q_subsection(f"1b. Target income without required stock/bond split (income ≥ target), {config.mc_target_success_percent}% success rate)")

    assumed = config.expected_yearly_growth_stocks_percent
    target_pct = config.mc_target_success_percent
    monthly_saving = config.annual_net_saving / 12

    def _cagr_fmt(cagr: float | None) -> str:
        if cagr is None:
            return "not achievable in scanned range"
        delta = cagr * 100 - assumed
        return f"{cagr * 100:.1f}%  ({delta:+.1f}pp vs assumed {assumed:.1f}%)"

    print(f"  Min required CAGR:")
    print(f"    DE : {_cagr_fmt(cagr_income_s1)}")
    print(f"    IT : {_cagr_fmt(cagr_income_s2)}")
    print()

    assumed_frac = assumed / 100
    max_stock_mo = max(grids_de[0].contribution_values)
    max_bonds_de = max(g.bonds_monthly for g in grids_de)
    max_bonds_it = max(g.bonds_monthly for g in grids_it)

    # ── Best P(income) at each CAGR ───────────────────────────────────────────
    cw2, pw = 12, 10
    grp = cw2 * 2 + pw + 8
    sep2 = f"  {'─' * (8 + grp * 2 + 2)}"
    print()
    print(f"  Min total contribution per month (* = current base assumption;"
          f" search caps: stocks {format_currency(max_stock_mo)}/mo"
          f" · bonds DE {format_currency(max_bonds_de)}/mo"
          f" · bonds IT {format_currency(max_bonds_it)}/mo):")
    print(f"  {'CAGR':6}  {'── Scenario 1 (DE) ──':^{grp}}  {'── Scenario 2 (IT) ──':^{grp}}")
    print(f"  {'':6}  {'Stocks/mo':>{cw2}}  {'Bonds/mo':>{cw2}}  {'P(income)':>{pw}}"
          f"  {'Stocks/mo':>{cw2}}  {'Bonds/mo':>{cw2}}  {'P(income)':>{pw}}")
    print(sep2)

    cagr_values = grids_de[0].cagr_values
    for cagr_idx, cagr in enumerate(cagr_values):
        s_de, b_de, r_de = _find_min_total_income_split(grids_de, cagr_idx, target_success)
        s_it, b_it, r_it = _find_min_total_income_split(grids_it, cagr_idx, target_success)
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
    print(f"  ✓ meets {target_pct}% target; rows without ✓ show best achievable rate (target unreachable in scan range)")


def print_redeploy_current_assets(
    opt_result_1: OptimalAllocationResult,
    opt_result_2: OptimalAllocationResult,
    config: Config,
    opt_result_1_capped: OptimalAllocationResult | None = None,
    opt_result_2_capped: OptimalAllocationResult | None = None,
) -> None:
    print()
    print()
    _q_subsection(
        f"Problem 2 — Optimal upfront allocation + monthly contribution "
        f"(at expected stock CAGR {config.expected_yearly_growth_stocks_percent:.1f}%)"
    )

    total_assets = config.current_stock_value + config.current_bonds_value + config.current_cash
    available = config.current_cash - opt_result_1.min_cash_today
    target_pct = int(round(opt_result_1.target_success * 100))

    print(f"  Total assets     : {format_currency(total_assets)}")
    print(f"  Min cash kept    : {format_currency(opt_result_1.min_cash_today)}")
    print(f"  Redeployable     : {format_currency(available)}")

    print()
    print(f"  2a — Reach target portfolio  (stocks ≥ req AND bonds ≥ req, {target_pct}% confidence)")
    print(f"       Top 3 upfront allocations ranked by lowest total monthly contribution. No budget constraint.")
    for result in [opt_result_1, opt_result_2]:
        print(f"  {'─' * 66}")
        print(f"  {result.scenario_name}")
        _print_top_portfolio_allocations(result, top_n=3)

    monthly_saving = config.annual_net_saving / 12

    if opt_result_1_capped is not None and opt_result_2_capped is not None:
        print()
        print(f"  2a (budget-constrained) — Reach target portfolio  "
              f"(stocks ≥ req AND bonds ≥ req, {target_pct}% confidence)")
        print(f"       Same optimization as 2a, but stocks/mo + bonds/mo ≤ "
              f"{format_currency(monthly_saving)}/mo (full monthly saving; cash = residual, may be 0).")
        for result in [opt_result_1_capped, opt_result_2_capped]:
            print(f"  {'─' * 66}")
            print(f"  {result.scenario_name}")
            _print_top_portfolio_allocations(result, top_n=3)

    print()
    print(f"  2b — Maximize income probability  (P(income ≥ target))")
    print(f"       Budget constraint: stocks/mo + bonds/mo ≤ {format_currency(monthly_saving)}/mo"
          f"  (full monthly saving; cash = residual, may be 0).")
    print(f"       Top 3 upfront allocations ranked by best income success rate.")
    for result in [opt_result_1, opt_result_2]:
        print(f"  {'─' * 66}")
        print(f"  {result.scenario_name}")
        _print_top_allocations(result, top_n=3)


def _closest_cagr_idx(cagr_values: list[float], target: float) -> int:
    return min(range(len(cagr_values)), key=lambda i: abs(cagr_values[i] - target))


def _best_income_point(opt_result: OptimalAllocationResult):
    if not opt_result.grid:
        return None
    return max(opt_result.grid, key=lambda p: p.best_success_rate)


def _worst_negative_entries(result: SensitivityWaterfallResult, top_n: int = 2):
    def _worst(e):
        return min(e.success_up - e.success_base, e.success_down - e.success_base)
    return sorted(result.entries, key=_worst)[:top_n]


def print_takeaway(
    config: Config,
    scenario_1: ScenarioResult,
    scenario_2: ScenarioResult,
    mc1: MonteCarloResult,
    mc2: MonteCarloResult,
    combined_de: CombinedGrid,
    combined_it: CombinedGrid,
    grids_de: list[ContributionCagrGrid],
    grids_it: list[ContributionCagrGrid],
    opt_result_1: OptimalAllocationResult,
    opt_result_2: OptimalAllocationResult,
    opt_result_1_capped: OptimalAllocationResult,
    opt_result_2_capped: OptimalAllocationResult,
    sensitivity_de: SensitivityWaterfallResult,
    sensitivity_it: SensitivityWaterfallResult,
    target_success: float,
    future_monthly_income: float,
    target_annual_net_income: float,
    invested_stock_amount: float,
    years_de: YearsToTargetsResult | None = None,
    years_it: YearsToTargetsResult | None = None,
    ff_de: FinancialFreedomResult | None = None,
    ff_it: FinancialFreedomResult | None = None,
) -> None:
    assumed = config.expected_yearly_growth_stocks_percent / 100
    target_pct = int(round(target_success * 100))
    stock_pct = 100 - config.bonds_percentage
    years = int(round(config.years))
    monthly_saving = config.annual_net_saving / 12

    _q_section("TAKE AWAY  —  portfolio projection")

    # 1. Required portfolio at horizon
    print(f"  1. Required portfolio at horizon ({years}y)")
    print(f"     Target income : {format_currency(future_monthly_income)}/mo"
          f"  ({format_currency(target_annual_net_income)}/yr, inflation-adjusted)")
    print(f"     Composition   : stocks {stock_pct}% · bonds {config.bonds_percentage}%")
    for s in (scenario_1, scenario_2):
        tag = s.name.split()[-1].strip("()")
        print(f"     {tag} : {format_currency(s.required_total_portfolio):>14}"
              f"  (stocks {format_currency(s.required_stock_amount)}"
              f" · bonds {format_currency(s.required_bonds_amount)})")
    print()

    # 2. Current situation with expected success rate
    total = config.current_stock_value + config.current_bonds_value + config.current_cash
    s_mo = monthly_saving * config.stock_monthly_investment_pct
    b_mo = monthly_saving * config.bonds_monthly_investment_pct
    c_mo = max(0.0, monthly_saving - s_mo - b_mo)
    current_year = datetime.date.today().year
    print(f"  2. Current situation  (target success ≥ {target_pct}% over {years}y)")
    print(f"     Portfolio : {format_currency(total)}"
          f"  (stocks {format_currency(config.current_stock_value)}"
          f" · bonds {format_currency(config.current_bonds_value)}"
          f" · cash {format_currency(config.current_cash)})")
    print(f"     Saving    : {format_currency(config.annual_net_saving)}/yr"
          f"  ⇒ stocks {format_currency(s_mo)}/mo"
          f" · bonds {format_currency(b_mo)}/mo"
          f" · cash {format_currency(c_mo)}/mo")

    tail_pct = 100 - target_pct
    if invested_stock_amount > 0 and 50 in mc1.final_stock_percentiles and tail_pct in mc1.final_stock_percentiles:
        gain_p50 = mc1.final_stock_percentiles[50] - invested_stock_amount
        gain_tail = mc1.final_stock_percentiles[tail_pct] - invested_stock_amount
        ratio_p50 = gain_p50 / invested_stock_amount * 100
        ratio_tail = gain_tail / invested_stock_amount * 100
        print(f"     Stock gain (vs {format_currency(invested_stock_amount)} invested):"
              f"  P50 {format_currency_signed(gain_p50)} ({ratio_p50:+.1f}%)"
              f"  ·  P{tail_pct} {format_currency_signed(gain_tail)} ({ratio_tail:+.1f}%)")

    def _yr_seg(label: str, yrs: int | None) -> str:
        if yrs is None:
            return f"{label} not achievable"
        delta = yrs - years
        sign = "+" if delta >= 0 else ""
        return f"{label} {current_year + yrs} ({sign}{delta}y vs plan)"

    for mc, yrs_target, ff in ((mc1, years_de, ff_de), (mc2, years_it, ff_it)):
        tag = mc.name.split()[-1].strip("()")
        flag_p = "✓" if mc.success_rate >= target_success else "✗"
        flag_i = "✓" if mc.success_rate_income_only >= target_success else "✗"
        print(f"     {tag} : portfolio {flag_p} {mc.success_rate * 100:>4.1f}%"
              f"  (stocks {mc.success_rate_stock_only * 100:.1f}%"
              f" · bonds {mc.success_rate_bond_only * 100:.1f}%)"
              f"  · income {flag_i} {mc.success_rate_income_only * 100:>4.1f}%")
        # Sub-row: years needed at current contributions.
        #   portfolio  = stocks AND bonds each ≥ required allocation (Q1 sense)
        #   net income = portfolio income ≥ target, with year-specific
        #                taxable-gain accounting via ff (more accurate than
        #                the constant effective_stock_tax_rate used in
        #                years_to_targets.years_to_income).
        # Both use the *income-formula* FI bar (W₀ ≈ target/yield); the
        # preservation-calibrated answer lives in decum's tradeoff table.
        if yrs_target is not None or ff is not None:
            yrs_both = yrs_target.years_to_both if yrs_target is not None else None
            yrs_inc  = ff.years if ff is not None else None
            print(f"          reaches {target_pct}% in: "
                  f"{_yr_seg('portfolio', yrs_both)}"
                  f"  ·  {_yr_seg('net income', yrs_inc)}")
    print()

    # 3. Reach TARGET ALLOCATION
    print(f"  3. Improve success rate to reach TARGET ALLOCATION (stocks AND bonds ≥ req)"
          f" with fixed CAGR of {assumed * 100:.1f}%")
    print(f"     a) Keep current allocation — min monthly contribution:")
    cagr_idx_de = _closest_cagr_idx(combined_de.cagr_values, assumed)
    cagr_idx_it = _closest_cagr_idx(combined_it.cagr_values, assumed)
    for grid, idx, mc in ((combined_de, cagr_idx_de, mc1), (combined_it, cagr_idx_it, mc2)):
        s, b, r = _find_min_total_combined(grid, idx, target_success)
        tag = grid.scenario_name.split()[-1].strip("()")
        flag = "✓" if r >= target_success else " "
        over = "†" if (s + b) > monthly_saving + 0.01 else ""
        delta = (r - mc.success_rate) * 100
        print(f"        {tag} : stocks {format_currency(s)}/mo"
              f" · bonds {format_currency(b)}/mo"
              f" · total {format_currency(s + b)}/mo{over}"
              f"  P(both)={flag}{r * 100:.1f}%"
              f"  (baseline {mc.success_rate * 100:.1f}%, {delta:+.1f}pp)")

    print(f"     b) Optimize current allocation + monthly contribution"
          f" (redeploy cash beyond {format_currency(opt_result_1_capped.min_cash_today)} reserve,"
          f" stocks/mo + bonds/mo ≤ {format_currency(monthly_saving)}/mo full saving):")
    for opt, mc in ((opt_result_1_capped, mc1), (opt_result_2_capped, mc2)):
        tag = opt.scenario_name.split()[-1].strip("()")
        p = opt.optimal
        if p is not None:
            moves = []
            if p.extra_stocks > 0:
                moves.append(f"cash → stocks {format_currency(p.extra_stocks)}")
            if p.extra_bonds > 0:
                moves.append(f"cash → bonds {format_currency(p.extra_bonds)}")
            move_str = "; ".join(moves) if moves else "no upfront move"
            delta = (opt.target_success - mc.success_rate) * 100
            over = "†" if p.total_monthly > monthly_saving + 0.01 else ""
            print(f"        {tag} : {move_str}"
                  f"  ⇒ stocks {format_currency(p.monthly_stocks)}/mo"
                  f" · bonds {format_currency(p.monthly_bonds)}/mo"
                  f" · total {format_currency(p.total_monthly)}/mo{over}"
                  f"  P(both)=✓{opt.target_success * 100:.1f}%"
                  f"  (baseline {mc.success_rate * 100:.1f}%, {delta:+.1f}pp)")
            continue
        # Target unreachable in scanned range — report the best achievable P(both)
        # along with the (upfront move, stocks/mo, bonds/mo) that produced it.
        best = max(opt.grid, key=lambda q: q.portfolio_ceiling_rate, default=None)
        if best is None or best.portfolio_ceiling_rate <= 0:
            print(f"        {tag} : not achievable in scanned range")
            continue
        delta = (best.portfolio_ceiling_rate - mc.success_rate) * 100
        moves = []
        if best.extra_stocks > 0:
            moves.append(f"cash → stocks {format_currency(best.extra_stocks)}")
        if best.extra_bonds > 0:
            moves.append(f"cash → bonds {format_currency(best.extra_bonds)}")
        move_str = "; ".join(moves) if moves else "no upfront move"
        print(f"        {tag} : target unreachable — best achievable: {move_str}"
              f"  ⇒ stocks {format_currency(best.ceiling_stocks_mo)}/mo"
              f" · bonds {format_currency(best.ceiling_bonds_mo)}/mo"
              f"  P(both)= {best.portfolio_ceiling_rate * 100:.1f}%"
              f"  (baseline {mc.success_rate * 100:.1f}%, {delta:+.1f}pp)")
    print()

    # 4. Reach TARGET INCOME
    print(f"  4. Improve success rate to reach TARGET INCOME (P(income ≥ target))"
          f" with fixed CAGR of {assumed * 100:.1f}%")
    print(f"     a) Keep current allocation — min monthly contribution:")
    cagr_idx_inc = _closest_cagr_idx(grids_de[0].cagr_values, assumed)
    for grids, label in ((grids_de, scenario_1.name), (grids_it, scenario_2.name)):
        s, b, r = _find_min_total_income_split(grids, cagr_idx_inc, target_success)
        tag = label.split()[-1].strip("()")
        flag = "✓" if r >= target_success else " "
        over = "†" if (s + b) > monthly_saving + 0.01 else ""
        print(f"        {tag} : stocks {format_currency(s)}/mo"
              f" · bonds {format_currency(b)}/mo"
              f" · total {format_currency(s + b)}/mo{over}"
              f"  P(income)={flag}{r * 100:.1f}%")

    print(f"     b) Optimize current allocation + monthly contribution"
          f" (redeploy cash beyond {format_currency(opt_result_1.min_cash_today)} reserve,"
          f" stocks/mo + bonds/mo ≤ {format_currency(monthly_saving)}/mo full saving):")
    for opt, mc in ((opt_result_1, mc1), (opt_result_2, mc2)):
        tag = opt.scenario_name.split()[-1].strip("()")
        p = _best_income_point(opt)
        if p is None:
            print(f"        {tag} : not available")
            continue
        moves = []
        if p.extra_stocks > 0:
            moves.append(f"cash → stocks {format_currency(p.extra_stocks)}")
        if p.extra_bonds > 0:
            moves.append(f"cash → bonds {format_currency(p.extra_bonds)}")
        move_str = "; ".join(moves) if moves else "no upfront move"
        delta = (p.best_success_rate - mc.success_rate_income_only) * 100
        if p.best_success_rate >= target_success:
            prefix = f"{move_str}"
            rate_str = f"P(income)=✓{p.best_success_rate * 100:.1f}%"
        else:
            prefix = f"target unreachable — best achievable: {move_str}"
            rate_str = f"P(income)= {p.best_success_rate * 100:.1f}%"
        print(f"        {tag} : {prefix}"
              f"  ⇒ stocks {format_currency(p.income_stock_mo)}/mo"
              f" · bonds {format_currency(p.income_bonds_mo)}/mo"
              f"  {rate_str}"
              f"  (baseline {mc.success_rate_income_only * 100:.1f}%, {delta:+.1f}pp)")
    print()

    # 5. Major risks
    print(f"  5. Major risks  (top 2 negative impacts on income success)")
    for result in (sensitivity_de, sensitivity_it):
        tag = result.scenario_name.split()[-1].strip("()")
        top = _worst_negative_entries(result, top_n=2)
        parts = []
        for e in top:
            worst = min(e.success_up - e.success_base, e.success_down - e.success_base)
            side = "+δ" if (e.success_up - e.success_base) <= (e.success_down - e.success_base) else "−δ"
            parts.append(f"{e.parameter} ({side} {e.delta_label}) → {worst * 100:+.1f}pp")
        print(f"     {tag} : " + "  ·  ".join(parts) if parts else f"     {tag} : n/a")
    print()


# ── Plan check — verdict-first report header ─────────────────────────────────
#
# Replaces the old bottom-of-run FINAL VERDICT (removed 2026-07). Printed FIRST,
# fully self-contained: every claim carries its number, its bar, and its unit.
# Layout: header → 3 pillars → since-last-run → levers → three-bars legend.

_PILLAR_INDENT = " " * 18  # continuation indent under "  ✓ PILLARNAME    "


def _fmt_k(v: float) -> str:
    """Compact thousands: 108_000 → '108k'."""
    return f"{v / 1e3:,.0f}k"


def _pillar(mark: str, name: str, first_line: str) -> None:
    print(f"  {mark} {name:<14}{first_line}")


def _pillar_cont(line: str) -> None:
    print(f"{_PILLAR_INDENT}{line}")


def _read_cape_ratio() -> float | None:
    """CAPE ratio from config.toml (not part of Config). None if unavailable."""
    try:
        import tomllib
        with open("config.toml", "rb") as f:
            raw = tomllib.load(f).get("cape_ratio")
        return float(raw) if raw is not None else None
    except Exception:
        return None


def print_plan_check(
    verdict: VerdictInputs,
    decum_summary: DecumulationSummary | None,
    realism: CagrRealismSnapshot | None,
    income_bar: float,
    drift_prev: dict | None,
) -> None:
    """Verdict-first plan check: 30-second top block with conclusions.

    income_bar:  IT income-formula required portfolio; 0 when unavailable.
    drift_prev:  previous month's drift row (main._read_prev_drift_row) or None.
    Pulls only from data already computed — no new MC.
    """
    config = verdict.config
    mc = verdict.mc_it
    target_success = verdict.target_success
    target_pct = int(round(target_success * 100))
    horizon = int(round(config.years))
    today = datetime.date.today()
    retire_year = today.year + horizon
    target_today = config.target_monthly_retirement_income
    infl_factor = (1.0 + config.expected_inflation_percent / 100.0) ** config.years
    stock_pct = int(round(100 - config.bonds_percentage))
    bonds_pct = int(round(config.bonds_percentage))
    preserve_pct = int(round(config.fi_preservation_ratio * 100))
    total_today = config.current_stock_value + config.current_bonds_value + config.current_cash

    # MC distribution of the plan's final wealth, deflated to today's €.
    # Keys follow the configured target: P{100-target} / P50 / P{target}.
    tail_key = 100 - target_pct
    mc_pcts = mc.final_portfolio_percentiles
    mc_today = {
        k: mc_pcts[k] / infl_factor
        for k in (tail_key, 50, target_pct)
        if k in mc_pcts and infl_factor > 0
    }

    # ── Pillar states ─────────────────────────────────────────────────────────
    income_p = mc.success_rate_income_only
    income_ok = income_p >= target_success

    preserve_ok: bool | None = None
    safe_yr = max_safe = None
    if decum_summary is not None:
        safe_yr = decum_summary.tradeoff_first_safe_year_from_now
        max_safe = decum_summary.max_safe_wd_at_plan_horizon_today
        preserve_ok = safe_yr is not None and safe_yr <= horizon

    planning = config.planning_cagr
    cape = realism.cape_nominal_cagr if realism is not None else None
    return_ok: bool | None = None if cape is None else (planning - cape) * 100 <= 2.0

    states = [income_ok, preserve_ok, return_ok]
    evaluated = sum(1 for s in states if s is not None)
    passed = sum(1 for s in states if s is True)
    on_track = income_ok and preserve_ok is not False and return_ok is not False

    # ── Header ────────────────────────────────────────────────────────────────
    bar = "═" * 80
    print(bar)
    print(f"  RETIREMENT PLAN CHECK — {today.isoformat()}")
    print(f"  Plan   : retire {retire_year} ({horizon}y, age {config.current_age + horizon}) · "
          f"spend €{format_currency(target_today)}/mo today's € · IT taxes")
    print(f"  Assumes: stock CAGR {planning * 100:.1f}% · "
          f"inflation {config.expected_inflation_percent:.1f}% · "
          f"{stock_pct}/{bonds_pct} stocks/bonds at retirement")
    print(f"  Today  : €{format_currency(total_today)} "
          f"(stocks {_fmt_k(config.current_stock_value)} · "
          f"bonds {_fmt_k(config.current_bonds_value)} · "
          f"cash {_fmt_k(config.current_cash)})")
    monthly_saving = config.annual_net_saving / 12.0
    hdr_stocks_mo = monthly_saving * config.stock_monthly_investment_pct
    hdr_bonds_mo = monthly_saving * config.bonds_monthly_investment_pct
    hdr_cash_mo = max(0.0, monthly_saving - hdr_stocks_mo - hdr_bonds_mo)
    print(f"  Saving : €{format_currency(config.annual_net_saving)}/yr ⇒ "
          f"stocks €{format_currency(hdr_stocks_mo)}/mo · "
          f"bonds €{format_currency(hdr_bonds_mo)}/mo · "
          f"cash €{format_currency(hdr_cash_mo)}/mo")
    print(bar)
    print()
    status = "ON TRACK" if on_track else "NOT ON TRACK"
    print(f"  VERDICT — {status}  ({passed} of {evaluated} pillars pass)")
    print()

    # ── INCOME pillar ─────────────────────────────────────────────────────────
    conf_word = "with" if income_ok else "with only"
    _pillar("✓" if income_ok else "✗", "INCOME",
            f"You can fund €{format_currency(target_today)}/mo from {retire_year} "
            f"{conf_word} {income_p * 100:.1f}% confidence")
    line = f"(min: {target_pct}%)."
    if income_bar > 0:
        # Same units as the MC outcome line below (today's €); the nominal
        # figure is kept in parens because the evidence tables use 2042 prices.
        if infl_factor > 0:
            line += (f" Wealth needed: {format_millions(income_bar / infl_factor)} "
                     f"today's € ({format_millions(income_bar)} at {retire_year} prices).")
        else:
            line += f" Wealth needed: {format_millions(income_bar)}."
    _pillar_cont(line)
    if len(mc_today) == 3:
        # MC of the CURRENT situation: today's portfolio + current contribution
        # split, no levers applied (run_monte_carlo_simulation with the
        # current_stock_mo/current_bonds_mo overrides).
        _pillar_cont(f"Current plan outcome (MC, today's €): "
                     f"P{tail_key} {format_millions(mc_today[tail_key])} · "
                     f"P50 {format_millions(mc_today[50])} · "
                     f"P{target_pct} {format_millions(mc_today[target_pct])}.")

    # Mix risk — the composition test (stocks AND bonds each reach their 50/50
    # quota). Income can pass while this fails: bonds/cash overshoot compensates
    # a lagging stock sleeve. Shown only when it fails, so S1's ✗ isn't mistaken
    # for an income problem. The fix (re-split) is lever D below.
    mix_risk_fails = mc.success_rate < target_success
    if mix_risk_fails:
        _pillar_cont(f"Mix risk: the {stock_pct}/{bonds_pct} shape itself is only "
                     f"{mc.success_rate * 100:.1f}% likely "
                     f"(stocks {mc.success_rate_stock_only * 100:.1f}% · "
                     f"bonds {mc.success_rate_bond_only * 100:.1f}%).")

    # ── PRESERVATION pillar ───────────────────────────────────────────────────
    if decum_summary is None:
        _pillar("-", "PRESERVATION", "not computed (decumulation skipped this run)")
    else:
        end_year = retire_year + decum_summary.n_years
        if preserve_ok:
            _pillar("✓", "PRESERVATION",
                    f"Spending €{format_currency(target_today)}/mo from {retire_year} keeps "
                    f"≥{preserve_pct}% of real wealth")
            line = f"through {end_year}"
            if max_safe is not None:
                line += f" (max safe spend €{format_currency(max_safe)}/mo ≥ target)"
            _pillar_cont(line + ".")
        else:
            _pillar("✗", "PRESERVATION",
                    f"Spending €{format_currency(target_today)}/mo from {retire_year} depletes "
                    f"real wealth over {decum_summary.n_years}y.")
            if max_safe is not None:
                gap = target_today - max_safe
                _pillar_cont(f"Max spend that keeps ≥{preserve_pct}% real wealth: "
                             f"€{format_currency(max_safe)}/mo — a €{format_currency(gap)}/mo gap.")
            if safe_yr is not None:
                _pillar_cont(f"Retiring ~{today.year + safe_yr} instead closes it "
                             f"with no extra saving.")
            else:
                _pillar_cont("No retirement year within the tradeoff window closes it "
                             "at current saving.")

    # ── RETURN pillar ─────────────────────────────────────────────────────────
    if cape is None:
        _pillar("-", "RETURN", "CAPE data unavailable — realism unchecked this run.")
    else:
        cape_ratio = _read_cape_ratio()
        cape_str = f"CAPE ({cape_ratio:.0f})" if cape_ratio is not None else "CAPE"
        if return_ok:
            label = "conservative" if (planning - cape) * 100 < -2.0 else "aligned"
            _pillar("✓", "RETURN",
                    f"Planning CAGR {planning * 100:.1f}% vs {cape_str}-implied "
                    f"{cape * 100:.1f}% — {label}.")
        else:
            _pillar("✗", "RETURN",
                    f"The {planning * 100:.1f}% CAGR assumption is optimistic — {cape_str} implies")
            _pillar_cont(f"~{cape * 100:.1f}%. Both pillars above inherit this optimism.")
            midpoint = (planning + cape) / 2.0 * 100  # halfway sanity re-run, matches S3 advice
            _pillar_cont(f"Cross-check: python main.py --cagr {midpoint:.0f}")

    # ── Since last run ────────────────────────────────────────────────────────
    print()
    if drift_prev is None:
        print("  SINCE LAST RUN: first recorded run — no history yet")
    else:
        delta_pp = (income_p - drift_prev["sr_total"]) * 100
        changes: list[str] = []
        if abs(drift_prev["bonds_pct"] - config.bonds_percentage) > 0.01:
            changes.append(f"bonds {drift_prev['bonds_pct']:.0f}%→{bonds_pct}%")
        if drift_prev["horizon"] != horizon:
            changes.append(f"horizon {drift_prev['horizon']}y→{horizon}y")
        if abs(drift_prev["target_income"] - target_today) > 0.01:
            changes.append(f"target €{format_currency(drift_prev['target_income'])}"
                           f"→€{format_currency(target_today)}/mo")
        print(f"  SINCE LAST RUN ({drift_prev['month']})")
        if changes:
            # Different plan on file — a success-rate delta would compare apples
            # to oranges, so name the changes instead.
            print(f"    plan changed: {' · '.join(changes)} — success delta not comparable")
        else:
            print(f"    income success {income_p * 100:.1f}% ({delta_pp:+.1f}pp) · "
                  f"plan params unchanged ({bonds_pct}% bonds, {horizon}y, "
                  f"€{format_currency(target_today)}/mo)")

    # ── Levers, primary first ─────────────────────────────────────────────────
    print()
    print("  WHAT MOVES THE NEEDLE")

    if income_ok:
        lever_a = "not needed: income pillar already passes"
    else:
        assumed = config.expected_yearly_growth_stocks_percent / 100
        idx = _closest_cagr_idx(verdict.grids_it[0].cagr_values, assumed)
        a_s, a_b, a_p = _find_min_total_income_split(verdict.grids_it, idx, target_success)
        lever_a = (f"stocks €{format_currency(a_s)}/mo + bonds €{format_currency(a_b)}/mo "
                   f"= €{format_currency(a_s + a_b)}/mo → income {a_p * 100:.1f}%")

    if decum_summary is not None and safe_yr is not None:
        wd = decum_summary.tradeoff_first_safe_wd_today or 0.0
        lever_b = f"~{today.year + safe_yr}: max safe spend €{format_currency(wd)}/mo ≥ target"
    else:
        lever_b = "no retirement year in the tradeoff window reaches target spend"

    pt = _best_income_point(verdict.opt_result_it)
    if pt is not None:
        moves = []
        if pt.extra_stocks > 0:
            moves.append(f"€{_fmt_k(pt.extra_stocks)}→stocks")
        if pt.extra_bonds > 0:
            moves.append(f"€{_fmt_k(pt.extra_bonds)}→bonds")
        move_str = " + ".join(moves) if moves else "no upfront move"
        note = " (margin, not preservation)" if preserve_ok is False else ""
        lever_c = f"{move_str}: income {pt.best_success_rate * 100:.1f}%{note}"
    else:
        lever_c = "optimizer result unavailable"

    primary = "A" if not income_ok else ("B" if preserve_ok is False else None)
    levers = [("A. Save more", lever_a), ("B. Retire later", lever_b),
              ("C. Redeploy cash", lever_c)]

    # Lever D — re-split monthly saving to fix the 50/50 mix risk flagged under
    # INCOME. Only offered when the shape test fails and the grid is available.
    # The grid cells use fewer sims (mc_grid_n_simulations) than the headline MC,
    # so its argmax carries ~±0.3pp noise: only recommend a re-split when it
    # meets the target or beats the current split by more than that noise band.
    if mix_risk_fails and verdict.combined_it is not None:
        assumed = config.expected_yearly_growth_stocks_percent / 100
        c_idx = _closest_cagr_idx(verdict.combined_it.cagr_values, assumed)
        c_s, c_b, c_r = _find_min_total_combined(verdict.combined_it, c_idx, target_success)
        cur_s = config.annual_net_saving / 12 * config.stock_monthly_investment_pct
        cur_b = config.annual_net_saving / 12 * config.bonds_monthly_investment_pct
        gain_pp = (c_r - mc.success_rate) * 100
        if c_r >= target_success or gain_pp > 1.0:
            reach = "" if c_r >= target_success else " (best reachable)"
            levers.append((
                "D. Re-split saving",
                f"stocks €{format_currency(c_s)} + bonds €{format_currency(c_b)}/mo "
                f"(now €{format_currency(cur_s)}+€{format_currency(cur_b)}) "
                f"→ shape {c_r * 100:.1f}%{reach}, {gain_pp:+.1f}pp",
            ))
        else:
            levers.append((
                "D. Re-split saving",
                f"no gain: best grid split {c_r * 100:.1f}% ≈ current "
                f"{mc.success_rate * 100:.1f}% — {target_pct}% needs more than a re-split",
            ))

    # Primary lever first, actionable levers next, non-actionable ones last.
    def _lever_rank(lv: tuple[str, str]) -> int:
        if primary is not None and lv[0].startswith(primary):
            return 0
        return 2 if lv[1].startswith(("not needed", "no gain")) else 1
    levers.sort(key=_lever_rank)
    for label, text in levers:
        suffix = "  ← primary lever" if primary is not None and label.startswith(primary) else ""
        print(f"    {label:<19}→ {text}{suffix}")

    sens = verdict.sensitivity_it
    if sens is not None and sens.entries:
        top = _worst_negative_entries(sens, top_n=1)[0]
        worst = min(top.success_up - top.success_base, top.success_down - top.success_base)
        side = "+" if (top.success_up - top.success_base) <= (top.success_down - top.success_base) else "−"
        delta_label = top.delta_label.lstrip("±")
        print(f"    Watch: {side}{delta_label} {top.parameter.lower()} → "
              f"{worst * 100:+.0f}pp income success (biggest single risk)")

    # ── Three-numbers legend ──────────────────────────────────────────────────
    print()
    print("  THREE WEALTH NUMBERS BELOW — don't confuse them (all today's €; "
          f"evidence tables use {retire_year} prices):")
    if income_bar > 0 and infl_factor > 0:
        print(f"    {format_millions(income_bar / infl_factor):<8} income min        "
              f"yield covers €{format_currency(target_today)}/mo; may deplete "
              f"(= {format_millions(income_bar)} in tables)")
    if 50 in mc_today:
        print(f"    {format_millions(mc_today[50]):<8} plan P50 (MC)     "
              f"typical wealth the current plan reaches by {retire_year}")
    if decum_summary is not None and infl_factor > 0:
        pres = decum_summary.required_portfolio_fi_canonical
        print(f"    {format_millions(pres / infl_factor):<8} "
              f"preservation min  keeps ≥{preserve_pct}% real wealth to "
              f"{retire_year + decum_summary.n_years} "
              f"(= {format_millions(pres)} in tables)")

    # ── Glossary ──────────────────────────────────────────────────────────────
    print()
    print("  GLOSSARY")
    print(f"    {'MC':<5} Monte Carlo: {mc.n_simulations:,} simulated market futures, "
          f"not a single forecast")
    if len(mc_today) == 3:
        print(f"    {f'P{tail_key}':<5} bad luck — worst {tail_key}% of MC futures "
              f"end below this value")
        print(f"    {'P50':<5} typical — median MC future; half end below, half above")
        print(f"    {f'P{target_pct}':<5} good luck — only {100 - target_pct}% of MC futures "
              f"end above this value")
    print()
