from models import (
    BondStockGrid,
    CombinedGrid,
    Config,
    ContributionCagrGrid,
    InflationCagrGrid,
    MonteCarloResult,
    ScenarioResult,
)

from reports.formatters import (
    format_currency,
    format_currency_signed,
)


def print_monte_carlo_report(
    mc1: MonteCarloResult,
    mc2: MonteCarloResult,
    current_stock_mo: float,
    current_bonds_mo: float,
    config: Config,
    verbose: bool = False,
) -> None:
    tail_pct = 100 - config.mc_target_success_percent
    target_pct = config.mc_target_success_percent
    pcts = sorted({tail_pct, 25, 50, 75, target_pct})
    metric_w = 22
    col_w = 13
    sep = f"+-{'-' * metric_w}-+" + f"-{'-' * col_w}-+" * len(pcts)

    def _pct_label(p: int) -> str:
        if p == tail_pct:
            return f"bad luck({p}%)"
        if p == target_pct:
            return f"good luck({p}%)"
        if p == 50:
            return "median(50%)"
        return f"p{p}"

    def section(mc: MonteCarloResult) -> None:
        monthly_income_impact = {
            p: mc.surplus_percentiles[p] * mc.portfolio_net_income_rate / 12
            for p in pcts
        }
        print(f"\n  {mc.name}  —  current plan")
        print(f"  Alloc OK (stocks AND bonds ≥ required):  {mc.success_rate * 100:.1f}%")
        print(f"  Target reached (actual split, any alloc): {mc.success_rate_income_only * 100:.1f}%")
        print(f"  Surplus(+) / Shortfall(−): final portfolio minus the portfolio needed")
        print(f"  to sustain your inflation-adjusted income at the planned withdrawal rate.")
        print(f"  Monthly income impact: surplus × net yield ÷ 12  "
              f"(extra or missing income vs your target)")
        print()
        print(sep)
        col_labels = {p: _pct_label(p) for p in pcts}
        header = f"| {'Metric':<{metric_w}} |"
        for p in pcts:
            header += f" {col_labels[p]:>{col_w}} |"
        print(header)
        print(sep)
        rows = [
            ("Final portfolio", mc.final_portfolio_percentiles),
            ("Required portfolio", mc.required_portfolio_percentiles),
            ("Surplus(+) / Shortfall(−)", mc.surplus_percentiles),
            ("Monthly income impact", monthly_income_impact),
        ]
        for label, data in rows:
            row = f"| {label:<{metric_w}} |"
            for p in pcts:
                v = data[p]
                row += f" {format_currency_signed(v) if label != 'Final portfolio' and label != 'Required portfolio' else format_currency(v):>{col_w}} |"
            print(row)
        print(sep)

    stock_mean = config.expected_yearly_growth_stocks_percent
    stock_std_pct = mc1.stock_std * 100
    inf_mean = config.expected_inflation_percent
    inf_std_pct = mc1.inflation_std * 100
    bond_mean = config.yield_bonds
    bond_std_pct = mc1.bond_std * 100
    monthly_saving = config.annual_net_saving / 12
    cash_mo = max(0.0, monthly_saving - current_stock_mo - current_bonds_mo)

    print(f"  Stock  : CAGR {stock_mean:.1f}% ± {stock_std_pct:.1f}%  |  "
          f"Bonds: yield {bond_mean:.1f}% ± {bond_std_pct:.1f}%  |  "
          f"Inflation: {inf_mean:.1f}% ± {inf_std_pct:.1f}%")
    print(f"  Current plan : stocks {format_currency(current_stock_mo)}/mo  |  "
          f"bonds {format_currency(current_bonds_mo)}/mo  |  "
          f"cash {format_currency(cash_mo)}/mo")
    print()

    lw, cw = 50, 10
    sep = f"  +{'-' * lw}+{'-' * (cw + 2)}+{'-' * (cw + 2)}+"
    print(sep)
    print(f"  | {'Success rate — current plan':<{lw}}| {'DE':^{cw}} | {'IT':^{cw}} |")
    print(sep)
    print(f"  | {'Q1a  Stock target  (stocks ≥ required stocks)':<{lw}}"
          f"| {mc1.success_rate_stock_only * 100:>{cw - 1}.1f}% | {mc2.success_rate_stock_only * 100:>{cw - 1}.1f}% |")
    print(f"  | {'Q1b  Bond target   (bonds ≥ required bonds)':<{lw}}"
          f"| {mc1.success_rate_bond_only * 100:>{cw - 1}.1f}% | {mc2.success_rate_bond_only * 100:>{cw - 1}.1f}% |")
    print(f"  | {'Q2   Income target (portfolio income ≥ target)':<{lw}}"
          f"| {mc1.success_rate_income_only * 100:>{cw - 1}.1f}% | {mc2.success_rate_income_only * 100:>{cw - 1}.1f}% |")
    print(sep)

    if verbose:
        section(mc1)
        section(mc2)


def print_monte_carlo_suggestions(
    scan_rows: list[dict],
    scenario: ScenarioResult,
    current_stock_mean_percent: float,
    config: Config,
    fixed_monthly_bonds: float,
) -> None:
    monthly_saving = config.annual_net_saving / 12
    req_bonds = scenario.required_bonds_amount
    print(f"\n  {scenario.name} — stock contribution scan")
    print(f"  Bonds fixed at {format_currency(fixed_monthly_bonds)}/mo  |  "
          f"remaining goes to cash → converted to bonds at horizon")
    print(f"  Required bonds at horizon: {format_currency(req_bonds)}  "
          f"(deterministic target — Bond@T p50/p75 vs this shows the structural gap)")
    print(f"  {'─' * 148}")
    target_pct = config.mc_target_success_percent
    target_success = target_pct / 100.0
    print(f"  Total monthly saving: {format_currency(monthly_saving)}/mo  |  * = current plan  |  ▶ ≥{target_pct}%")
    print(f"  Alloc S/B reached  = stocks AND bonds both reach their required amounts (40/60 split respected)")
    print(f"  Tgt reached w/o split = portfolio × actual income rate ≥ target, regardless of split")
    print()
    cw = 12
    vw = 14
    sep = (f"  +{'-' * (cw + 2)}+{'-' * (cw + 2)}+{'-' * (cw + 2)}"
           f"+{'-' * (vw + 2)}+{'-' * (vw + 2)}+{'-' * (vw + 2)}+{'-' * (vw + 2)}+{'-' * (vw + 2)}"
           f"+{'-' * (cw + 2)}+{'-' * (cw + 2)}+")
    h1 = (f"  | {'Stocks/mo':^{cw}} | {'Cash/mo':^{cw}} | {'Bonds/mo':^{cw}}"
          f" | {'Stock@T':^{vw}} | {'Bond@T':^{vw}} | {'Bond@T':^{vw}} | {'Net inc/yr':^{vw}} | {'Target/yr':^{vw}}"
          f" | {'Stock Q1a':^{cw}} | {'Income Q2':^{cw}} |")
    h2 = (f"  | {'at ' + f'{current_stock_mean_percent:.1f}%/yr':^{cw}} | {'→bonds at T':^{cw}} | {'fixed':^{cw}}"
          f" | {'(p50)':^{vw}} | {'(p50)':^{vw}} | {'(p75)':^{vw}} | {'(p50, nom.)':^{vw}} | {'(p50, infl.)':^{vw}}"
          f" | {'reached':^{cw}} | {'reached':^{cw}} |")
    print(sep)
    print(h1)
    print(h2)
    print(sep)
    for row in scan_rows:
        sm = row["stocks_mo"]
        cm = row["cash_mo"]
        rate_stock = row.get("rate_stock", row["rate"])
        rate_income = row["rate_income"]
        median_stocks = row["median_stocks"]
        median_bonds = row["median_bonds"]
        p75_bonds = row.get("p75_bonds", median_bonds)
        net_income = (median_stocks + median_bonds) * scenario.portfolio_net_income_rate
        p50_target = row["p50_target_income"]
        marker = " *" if row["is_current"] else "  "
        flag_s = "▶" if rate_stock >= target_success else " "
        flag_i = "▶" if rate_income >= target_success else " "
        print(f"  | {format_currency(sm) + marker:^{cw}} "
              f"| {format_currency(cm):>{cw}} "
              f"| {format_currency(fixed_monthly_bonds):>{cw}} "
              f"| {format_currency(median_stocks):>{vw}} "
              f"| {format_currency(median_bonds):>{vw}} "
              f"| {format_currency(p75_bonds):>{vw}} "
              f"| {format_currency(net_income):>{vw}} "
              f"| {format_currency(p50_target):>{vw}} "
              f"| {flag_s}{rate_stock * 100:>{cw - 2}.1f}% "
              f"| {flag_i}{rate_income * 100:>{cw - 2}.1f}% |")
    print(sep)

    p50_target = scan_rows[0]["p50_target_income"] if scan_rows else 0.0
    first_target_income = next((r for r in scan_rows if r["rate_income"] >= target_success), None)
    first_target_stock = next((r for r in scan_rows if r.get("rate_stock", 0) >= target_success), None)

    if first_target_income:
        net_tgt = (first_target_income["median_stocks"] + first_target_income["median_bonds"]) * scenario.portfolio_net_income_rate
        margin_tgt = (net_tgt / p50_target - 1) * 100 if p50_target > 0 else 0.0
        print(f"  → Q2 income ▶{target_pct}% first reached at {format_currency(first_target_income['stocks_mo'])}/mo stocks  |  "
              f"Net inc/yr {format_currency(net_tgt)} = +{margin_tgt:.0f}% above p50 inflation-adjusted target.")
    else:
        print(f"  → Q2 income target not reached in this scan range — see Q2 grid below.")

    if first_target_stock:
        print(f"  → Q1a stock target ▶{target_pct}% first reached at {format_currency(first_target_stock['stocks_mo'])}/mo stocks.")
    else:
        print(f"  → Q1a stock target not reached in this scan range — see Q1a grid below.")

    # Structural bond cap observation: p75 ≈ p50 once cash hits zero
    high_stock_rows = [r for r in scan_rows if r["stocks_mo"] >= monthly_saving * 0.7]
    if high_stock_rows:
        b50 = high_stock_rows[0]["median_bonds"]
        b75 = high_stock_rows[0].get("p75_bonds", b50)
        spread_pct = abs(b75 - b50) / max(b50, 1) * 100
        if spread_pct < 5.0:
            print(f"  ! Bond@T p75 ≈ p50 ({format_currency(b75)} vs {format_currency(b50)}) at high stock contributions: "
                  f"bonds are structurally capped by the cash→bonds conversion, not by CAGR variance.")
            print(f"    The required {format_currency(req_bonds)} cannot be reached by raising stocks — "
                  f"the binding constraint is the bond amount, not CAGR.")

    # Note when €0 stocks already clears the Q2 income threshold
    zero_row = next((r for r in scan_rows if r["stocks_mo"] < 1.0), None)
    if zero_row and zero_row["rate_income"] >= target_success:
        current_row = next((r for r in scan_rows if r["is_current"]), None)
        if current_row and current_row["rate_income"] < zero_row["rate_income"]:
            print(f"  ! Q2 income target already met with €0 stocks ({zero_row['rate_income'] * 100:.1f}% success): "
                  f"routing all cash to the bond conversion covers the income target. "
                  f"Adding stocks ({format_currency(current_row['stocks_mo'])}/mo) diverts cash away from bonds "
                  f"and reduces Q2 success to {current_row['rate_income'] * 100:.1f}%. "
                  f"The binding constraint is bond income, not stock CAGR.")


def _cagr_line(label: str, cagr: float | None, assumed: float) -> str:
    _HIST_EQUITY_CAGR = 7.0
    if cagr is None:
        return f"  {label}: not achievable even at 35%/yr — increase contributions instead"
    delta = cagr * 100 - assumed
    sign = "+" if delta >= 0 else ""
    vs_hist = cagr * 100 - _HIST_EQUITY_CAGR
    hist_flag = "above historical avg — consider increasing contributions" if vs_hist > 0 else "within historical range"
    return (
        f"  {label}: needs {cagr * 100:.1f}%/yr  "
        f"(your assumption {assumed:.1f}%/yr,  Δ {sign}{delta:.1f}pp  |  "
        f"vs ~{_HIST_EQUITY_CAGR:.0f}% historical: {vs_hist:+.1f}pp → {hist_flag})"
    )


def print_plan_b_hard(
    cagr_1: float | None,
    cagr_2: float | None,
    config: Config,
) -> None:
    assumed = config.expected_yearly_growth_stocks_percent
    _HIST_EQUITY_CAGR = 7.0
    target_pct = config.mc_target_success_percent
    print(f"\n  Plan B Q1a — min stock CAGR for {target_pct}% exact split (stocks AND bonds ≥ req)")
    print(f"  Required: stocks ≥ required_stocks AND bonds ≥ required_bonds at horizon with {target_pct}% confidence")
    print(f"  Reference: global equity funds have returned ~{_HIST_EQUITY_CAGR:.0f}%/yr nominal historically")
    print(f"  {'─' * 80}")
    for label, cagr in [("Scenario 1 (DE)", cagr_1), ("Scenario 2 (IT)", cagr_2)]:
        print(_cagr_line(label, cagr, assumed))
    print()


def print_plan_b_income(
    cagr_1: float | None,
    cagr_2: float | None,
    config: Config,
) -> None:
    assumed = config.expected_yearly_growth_stocks_percent
    _HIST_EQUITY_CAGR = 7.0
    target_pct = config.mc_target_success_percent
    print(f"\n  Plan B — min CAGR for {target_pct}% success, any stock/bond split")
    print(f"  Target: portfolio × actual income rate ≥ inflation-adjusted target income")
    print(f"  Reference: global equity funds have returned ~{_HIST_EQUITY_CAGR:.0f}%/yr nominal historically")
    print(f"  {'─' * 80}")
    for label, cagr in [("Scenario 1 (DE)", cagr_1), ("Scenario 2 (IT)", cagr_2)]:
        print(_cagr_line(label, cagr, assumed))
    print()


def print_stock_allocation_sweep(
    stock_sweep: list[dict[str, float]],
    config: Config,
) -> None:
    print("\n" * 2, end="")
    print(f"{'SENSITIVITY: INITIAL STOCK ALLOCATION':^70}")
    print()
    print("  Rate DE / Rate IT = probability that your final portfolio is large")
    print("  enough to sustain your inflation-adjusted retirement income, under")
    print("  German (DE) or Italian (IT) tax rules.")
    print()
    print(f"  Moving cash → stocks at t=0 (monthly contributions unchanged)")
    print(f"  Steps of €50,000; * = current allocation\n")
    c_w = 14
    sep = f"  +{'-' * (c_w + 2)}+{('-' * (c_w + 2) + '+') * 4}"
    print(sep)
    print(f"  | {'Extra→stocks':^{c_w}} | {'Stocks t=0':^{c_w}} | {'Cash t=0':^{c_w}} | {'Rate DE':^{c_w}} | {'Rate IT':^{c_w}} |")
    print(sep)
    for row in stock_sweep:
        marker = " *" if row["extra"] == 0 else "  "
        print(f"  | {format_currency(row['extra']) + marker:^{c_w}} "
              f"| {format_currency(row['stocks_t0']):>{c_w}} "
              f"| {format_currency(row['cash_t0']):>{c_w}} "
              f"| {row['rate_1'] * 100:>{c_w - 1}.1f}% "
              f"| {row['rate_2'] * 100:>{c_w - 1}.1f}% |")
    print(sep)


def _find_min_contrib(
    grid: ContributionCagrGrid, cagr_idx: int, use_hard: bool, target_success: float
) -> float | None:
    for contrib_idx, contrib in enumerate(grid.contribution_values):
        cell = grid.grid[cagr_idx][contrib_idx]
        rate = cell.success_hard if use_hard else cell.success_income
        if rate >= target_success:
            return contrib
    return None


def _find_best_income_split(
    grids: list[ContributionCagrGrid], cagr_idx: int
) -> tuple[float, float, float]:
    """Best (stocks/mo, bonds/mo, P(income)) across all bonds levels for a given CAGR."""
    best_rate, best_stocks, best_bonds = -1.0, 0.0, 0.0
    for g in grids:
        for contrib_idx, contrib in enumerate(g.contribution_values):
            rate = g.grid[cagr_idx][contrib_idx].success_income
            if rate > best_rate:
                best_rate, best_stocks, best_bonds = rate, contrib, g.bonds_monthly
    return best_stocks, best_bonds, best_rate


def _fmt_stocks(min_stocks: float | None, cw: int) -> str:
    if min_stocks is None:
        return f"{'n/a':^{cw}}"
    return f"{format_currency(min_stocks):>{cw}}"


def _fmt_cash(min_stocks: float | None, bonds: float, monthly_saving: float, cw: int) -> str:
    if min_stocks is None:
        return f"{'—':^{cw}}"
    cash = max(0.0, monthly_saving - min_stocks - bonds)
    over_budget = (min_stocks + bonds) > monthly_saving + 0.01
    cash_str = format_currency(cash) + ("†" if over_budget else " ")
    return f"{cash_str:>{cw}}"


def _contribution_cagr_frontier_table(
    grids: list[ContributionCagrGrid],
    monthly_saving: float,
    goal_label: str,
    use_hard: bool,
    target_success: float,
) -> None:
    grid_1x = grids[0]
    grid_2x = grids[1]
    b1 = grid_1x.bonds_monthly
    b2 = grid_2x.bonds_monthly
    cw = 12
    sep_w = 72
    sep = f"  {'─' * sep_w}"
    target_pct = int(round(target_success * 100))

    print(f"\n  {goal_label}")
    print(sep)
    b1_label = f"Bonds 1× ({format_currency(b1)}/mo)"
    b2_label = f"Bonds 2× ({format_currency(b2)}/mo)"
    group_w = cw * 2 + 5
    print(f"  {'CAGR':<5}  {b1_label:^{group_w}}  {b2_label:^{group_w}}")
    print(f"  {'':5}  {'Stocks/mo':>{cw}}  {'Cash/mo':>{cw}}    {'Stocks/mo':>{cw}}  {'Cash/mo':>{cw}}")
    print(sep)

    for cagr_idx, cagr in enumerate(grid_1x.cagr_values):
        min_1x = _find_min_contrib(grid_1x, cagr_idx, use_hard, target_success)
        min_2x = _find_min_contrib(grid_2x, cagr_idx, use_hard, target_success)
        s1 = _fmt_stocks(min_1x, cw)
        c1 = _fmt_cash(min_1x, b1, monthly_saving, cw)
        s2 = _fmt_stocks(min_2x, cw)
        c2 = _fmt_cash(min_2x, b2, monthly_saving, cw)
        print(f"  {int(cagr * 100):2d}%    {s1}  {c1}    {s2}  {c2}")

    print(sep)

    if use_hard:
        any_target = any(
            grid.grid[ci][ki].success_hard >= target_success
            for grid in grids
            for ci in range(len(grid.cagr_values))
            for ki in range(len(grid.contribution_values))
        )
        if not any_target:
            best_rate, best_cagr, best_contrib, best_mult = 0.0, 0.0, 0.0, 1
            for grid in grids:
                for ci, cagr in enumerate(grid.cagr_values):
                    for ki, contrib in enumerate(grid.contribution_values):
                        rate = grid.grid[ci][ki].success_hard
                        if rate > best_rate:
                            best_rate, best_cagr = rate, cagr
                            best_contrib, best_mult = contrib, grid.bonds_multiplier
            print(f"  ! {target_pct}% not reached in the scanned range.")
            print(f"    Best result : {best_rate * 100:.1f}% hard success at CAGR "
                  f"{best_cagr * 100:.0f}%, stocks {format_currency(best_contrib)}/mo, "
                  f"bonds {best_mult}x")
            print(f"    -> Redeploy cash to stocks today (see allocation optimizer below), "
                  f"then rerun.")


def _print_cagr_grid_verbose(grid: ContributionCagrGrid, target_success: float) -> None:
    cw = 8
    cagr_w = 5
    n_cols = len(grid.contribution_values)
    sep = f"  +{'-' * cagr_w}+" + f"{'-' * (cw + 2)}+" * n_cols
    target_pct = int(round(target_success * 100))

    print(f"\n  Full grid — {grid.scenario_name}  "
          f"Bonds {grid.bonds_multiplier}× ({format_currency(grid.bonds_monthly)}/mo)")
    print(f"  Cells: Hard% / Income%  |  ▶ = ≥{target_pct}%  |  * = suggested contribution")
    print()

    header = f"  | {'CAGR':^{cagr_w}} |"
    for i, contrib in enumerate(grid.contribution_values):
        marker = "*" if i == 0 else " "
        label = f"{int(contrib):>5,}{marker}"
        header += f" {label:^{cw}} |"

    print(sep)
    print(header)
    print(sep)

    for cagr_idx, cagr in enumerate(grid.cagr_values):
        row = f"  | {int(cagr * 100):2d}%   |"
        for contrib_idx in range(n_cols):
            cell = grid.grid[cagr_idx][contrib_idx]
            fh = "▶" if cell.success_hard >= target_success else " "
            fs = "▶" if cell.success_income >= target_success else " "
            h = int(round(cell.success_hard * 100))
            s = int(round(cell.success_income * 100))
            cell_str = f"{fh}{h}/{fs}{s}"
            row += f" {cell_str:^{cw}} |"
        print(row)

    print(sep)
    print()


def _q_analysis_header(monthly_saving: float, target_success_pct: int) -> None:
    print(f"  Monthly saving : {format_currency(monthly_saving)}/mo")
    print(f"  Bonds          : fixed at 1× or 2× the deterministic required contribution")
    print(f"  Cash           : max(0, saving − stocks − bonds) goes to cash savings")
    print(f"  †              : stocks+bonds exceed monthly saving — drawing from cash pile")
    print(f"  Table shows min stocks/mo to reach ≥{target_success_pct}% success at each CAGR level")


def print_q1_analysis(
    grids_de: list[ContributionCagrGrid],
    grids_it: list[ContributionCagrGrid],
    monthly_saving: float,
    target_success: float,
    verbose: bool = False,
) -> None:
    _q_analysis_header(monthly_saving, int(round(target_success * 100)))
    print()
    for label, grids in [("SCENARIO 1 (DE)", grids_de), ("SCENARIO 2 (IT)", grids_it)]:
        print(f"  {'─' * 66}")
        print(f"  {label}")
        print(f"  {'─' * 66}")
        _contribution_cagr_frontier_table(
            grids, monthly_saving,
            "Goal 1 — Target portfolio with required composition  (stocks AND bonds split)",
            use_hard=True,
            target_success=target_success,
        )
        print()
        if verbose:
            for grid in grids:
                _print_cagr_grid_verbose(grid, target_success)


def print_q2_analysis(
    grids_de: list[ContributionCagrGrid],
    grids_it: list[ContributionCagrGrid],
    monthly_saving: float,
    target_success: float,
    verbose: bool = False,
) -> None:
    _q_analysis_header(monthly_saving, int(round(target_success * 100)))
    print()
    for label, grids in [("SCENARIO 1 (DE)", grids_de), ("SCENARIO 2 (IT)", grids_it)]:
        print(f"  {'─' * 66}")
        print(f"  {label}")
        print(f"  {'─' * 66}")
        _contribution_cagr_frontier_table(
            grids, monthly_saving,
            "Goal 2 — Required net income, any portfolio composition",
            use_hard=False,
            target_success=target_success,
        )
        print()
        if verbose:
            for grid in grids:
                _print_cagr_grid_verbose(grid, target_success)


def _find_min_contrib_stock(
    grid: ContributionCagrGrid, cagr_idx: int, target_success: float
) -> tuple[float, float] | None:
    """Return (min_stocks_mo, stock_tail_pct) for the first cell hitting target_success, or None."""
    for contrib_idx, contrib in enumerate(grid.contribution_values):
        cell = grid.grid[cagr_idx][contrib_idx]
        if cell.success_stock_only >= target_success:
            return contrib, cell.stock_tail_pct
    return None


def _total_invested(monthly: float, years: float, growth: float, current_cost_basis: float) -> float:
    """Total nominal cash in stocks: current cost basis + future contributions with annual step-up."""
    if abs(growth) > 1e-9:
        contributions = monthly * 12 * ((1 + growth) ** years - 1) / growth
    else:
        contributions = monthly * 12 * years
    return current_cost_basis + contributions


def print_q1a_analysis(
    grids_de: list[ContributionCagrGrid],
    grids_it: list[ContributionCagrGrid],
    config: Config,
    target_success: float,
) -> None:
    grid_de = grids_de[0]
    grid_it = grids_it[0]
    cw = 16
    tail_pct = 100 - int(round(target_success * 100))
    target_pct = int(round(target_success * 100))
    years = config.years
    growth = config.annual_contribution_growth_percent / 100
    current_cost_basis = config.current_stock_value - config.current_stock_unrealized_gain
    sep = f"  {'─' * (8 + (cw * 3 + 6) * 2)}"
    p_label = f"Stock@T p{tail_pct}"
    tot_label = "Total invested"

    print(f"\n  Min stocks/mo for ≥{target_pct}% stock success at each CAGR")
    print(f"  Stock@T p{tail_pct} : stock portfolio at horizon in the bad-luck {tail_pct}% of simulations")
    print(f"  Total invested: current stock cost basis ({format_currency(current_cost_basis)}) + contributions over {years:.0f}y (incl. {growth * 100:.1f}%/yr step-up)")
    print(f"  Stock outcome is independent of bond level and cash allocation")
    print(sep)
    grp = cw * 3 + 6
    print(f"  {'CAGR':5}  {'── Scenario 1 (DE) ──':^{grp}}  {'── Scenario 2 (IT) ──':^{grp}}")
    print(f"  {'':5}  {'Min stocks/mo':>{cw}}  {p_label:>{cw}}  {tot_label:>{cw}}"
          f"  {'Min stocks/mo':>{cw}}  {p_label:>{cw}}  {tot_label:>{cw}}")
    print(sep)

    for cagr_idx, cagr in enumerate(grid_de.cagr_values):
        res_de = _find_min_contrib_stock(grid_de, cagr_idx, target_success)
        res_it = _find_min_contrib_stock(grid_it, cagr_idx, target_success)
        s_de = format_currency(res_de[0]) if res_de else "n/a"
        t_de = format_currency(res_de[1]) if res_de else "—"
        i_de = format_currency(_total_invested(res_de[0], years, growth, current_cost_basis)) if res_de else "—"
        s_it = format_currency(res_it[0]) if res_it else "n/a"
        t_it = format_currency(res_it[1]) if res_it else "—"
        i_it = format_currency(_total_invested(res_it[0], years, growth, current_cost_basis)) if res_it else "—"
        print(f"  {int(cagr * 100):2d}%     {s_de:>{cw}}  {t_de:>{cw}}  {i_de:>{cw}}"
              f"  {s_it:>{cw}}  {t_it:>{cw}}  {i_it:>{cw}}")

    print(sep)


def _print_bond_stock_grid(
    grid: BondStockGrid,
    current_stock_mo: float,
    current_bonds_mo: float,
    monthly_saving: float,
    target_success: float,
) -> None:
    cw = 8     # cell width: flag(1) + value(5) + %(1) + dagger(1) = 8
    lw = 12    # label width for bonds/mo column

    n_stocks = len(grid.stocks_values)
    sep = f"  {'─' * (lw + 4 + n_stocks * (cw + 2))}"

    print(f"  {grid.scenario_name}  —  Required bonds: {format_currency(grid.required_bonds)}")
    print()

    header = f"  {'Bonds/mo ↓':>{lw}}  "
    for stocks_mo in grid.stocks_values:
        mark = "*" if abs(stocks_mo - current_stock_mo) < 1.0 else " "
        label = f"{int(stocks_mo):>5,}{mark}"
        header += f"  {label:>{cw}}"
    print(f"  {'Stocks/mo →':>{lw}}")
    print(header)
    print(sep)

    for bonds_idx, bonds_mo in enumerate(grid.bonds_values):
        is_det = abs(bonds_mo - grid.suggested_bonds) < 1.0
        is_cur = abs(bonds_mo - current_bonds_mo) < 1.0
        prefix = "←" if is_det else " "
        suffix = "*" if is_cur and not is_det else ("*" if is_cur else " ")
        label = f"{prefix}{format_currency(bonds_mo)}{suffix}"
        row = f"  {label:>{lw}}  "

        for stocks_idx, stocks_mo in enumerate(grid.stocks_values):
            cell = grid.grid[bonds_idx][stocks_idx]
            r = cell.success_bond
            flag = "▶" if r >= target_success else " "
            over = (stocks_mo + bonds_mo) > monthly_saving + 0.01
            dagger = "†" if over else " "
            cell_str = f"{flag}{r * 100:5.1f}%{dagger}"   # 8 chars
            row += f"  {cell_str}"
        print(row)

    print(sep)
    print()


def print_q1b_analysis(
    bond_grid_de: BondStockGrid,
    bond_grid_it: BondStockGrid,
    current_stock_mo: float,
    current_bonds_mo: float,
    monthly_saving: float,
    target_success: float,
) -> None:
    target_pct = int(round(target_success * 100))
    print(f"  Monthly saving : {format_currency(monthly_saving)}/mo")
    print(f"  Cash = max(0, saving − stocks − bonds) → converts to bonds at horizon")
    print(f"  ▶ = ≥{target_pct}%  |  ← = deterministic required bonds  |  * = current plan  |  † = stocks+bonds > saving")
    print()
    _print_bond_stock_grid(bond_grid_de, current_stock_mo, current_bonds_mo, monthly_saving, target_success)
    _print_bond_stock_grid(bond_grid_it, current_stock_mo, current_bonds_mo, monthly_saving, target_success)


def _find_best_combined(
    grid: CombinedGrid, cagr_idx: int
) -> tuple[float, float, float]:
    """Return (stocks_mo, bonds_mo, best_success_rate) maximising P(both), ties broken by min total."""
    best_rate = -1.0
    best_total = float("inf")
    best: tuple[float, float, float] = (0.0, 0.0, 0.0)
    for bonds_idx, bonds_mo in enumerate(grid.bonds_values):
        for stocks_idx, stocks_mo in enumerate(grid.stocks_values):
            rate = grid.grid[cagr_idx][bonds_idx][stocks_idx].success_combined
            total = stocks_mo + bonds_mo
            if rate > best_rate or (rate == best_rate and total < best_total):
                best_rate = rate
                best_total = total
                best = (stocks_mo, bonds_mo, rate)
    return best


def print_combined_q1_analysis(
    grid_de: CombinedGrid,
    grid_it: CombinedGrid,
    config: Config,
    target_success: float,
) -> None:
    target_pct = int(round(target_success * 100))
    monthly_saving = config.annual_net_saving / 12
    cw = 12
    pw = 7
    grp = cw * 3 + pw + 8
    sep = f"  {'─' * (8 + grp * 2)}"

    print(f"\n  Combined Q1 — best achievable P(stocks ≥ required AND bonds ≥ required)")
    print(f"  Shows the (stocks/mo, bonds/mo) split that maximises the joint success rate at each CAGR")
    print(f"  ✓ = meets the {target_pct}% target  |  † = total exceeds monthly saving ({format_currency(monthly_saving)}/mo)")
    print(f"  Required stocks (DE): {format_currency(grid_de.required_stocks)}"
          f"  |  Required bonds (DE): {format_currency(grid_de.required_bonds)}")
    print(f"  Required stocks (IT): {format_currency(grid_it.required_stocks)}"
          f"  |  Required bonds (IT): {format_currency(grid_it.required_bonds)}")
    print(sep)
    print(f"  {'CAGR':5}  {'── Scenario 1 (DE) ──':^{grp}}  {'── Scenario 2 (IT) ──':^{grp}}")
    print(f"  {'':5}  {'Stocks/mo':>{cw}}  {'Bonds/mo':>{cw}}  {'Total/mo':>{cw}}  {'P(both)':>{pw}}"
          f"  {'Stocks/mo':>{cw}}  {'Bonds/mo':>{cw}}  {'Total/mo':>{cw}}  {'P(both)':>{pw}}")
    print(sep)

    def _fmt_row(opt: tuple[float, float, float]) -> str:
        stocks_mo, bonds_mo, rate = opt
        total = stocks_mo + bonds_mo
        over = total > monthly_saving + 0.01
        meets = rate >= target_success
        total_str = format_currency(total) + ("†" if over else " ")
        flag = "✓" if meets else " "
        return (f"{format_currency(stocks_mo):>{cw}}  {format_currency(bonds_mo):>{cw}}"
                f"  {total_str:>{cw + 1}}  {flag}{rate * 100:>{pw - 2}.1f}%")

    for cagr_idx, cagr in enumerate(grid_de.cagr_values):
        opt_de = _find_best_combined(grid_de, cagr_idx)
        opt_it = _find_best_combined(grid_it, cagr_idx)
        print(f"  {int(cagr * 100):2d}%     {_fmt_row(opt_de)}  {_fmt_row(opt_it)}")

    print(sep)


def print_inflation_cagr_heatmap(
    grid_1: InflationCagrGrid,
    grid_2: InflationCagrGrid,
    target_success: float,
) -> None:
    """Print a 2D heatmap of income success rate vs (inflation, CAGR) for two scenarios."""
    def _cell(rate: float, is_base: bool) -> str:
        s = f"{rate * 100:.0f}%"
        return f"[{s}]" if is_base else f" {s} "

    def _print_grid(grid: InflationCagrGrid) -> None:
        col_w = 7
        header = f"  {'CPI↓  CAGR→':>12}"
        for cagr in grid.cagr_values:
            marker = "*" if abs(cagr - grid.base_cagr) < 1e-4 else " "
            header += f"  {int(round(cagr * 100)):>2}%{marker}"
        print(header)
        sep = "  " + "-" * (13 + len(grid.cagr_values) * (col_w - 1))
        print(sep)

        for i, inf in enumerate(grid.inflation_values):
            is_base_inf = abs(inf - grid.base_inflation) < 1e-4
            inf_label = f"{inf * 100:.1f}%{'←' if is_base_inf else ' '}"
            row = f"  {inf_label:>12}"
            for j, cagr in enumerate(grid.cagr_values):
                rate = grid.grid[i][j].success_income
                is_base = is_base_inf and abs(cagr - grid.base_cagr) < 1e-4
                row += f"  {_cell(rate, is_base):>5}"
            print(row)

        print(sep)
        print(f"  * = base CAGR ({grid.base_cagr * 100:.1f}%)   "
              f"← = base CPI ({grid.base_inflation * 100:.1f}%)   "
              f"[XX%] = base case   target ≥ {target_success * 100:.0f}%")
        print(f"  Contributions: stocks €{grid.monthly_stocks:,.0f}/mo  "
              f"bonds €{grid.monthly_bonds:,.0f}/mo")

    print(f"\n  {grid_1.scenario_name}")
    _print_grid(grid_1)
    print(f"\n  {grid_2.scenario_name}")
    _print_grid(grid_2)
