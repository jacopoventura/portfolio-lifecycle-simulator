from models import (
    DecumInflationCagrCell,
    DecumInflationCagrGrid,
    FocusedPreservationTable,
    WealthPreservationUnderSORR,
    YearsToRetirementTradeoff,
)
from reports.formatters import _print_section, _print_subsection, format_currency


def print_focused_preservation_table(table: FocusedPreservationTable) -> None:
    """Decumulation Q1/Q2/Q3 anchored on the user's actual trajectory.

    Q1: At the projected retirement wealth (deterministic AND MC P50), what's
        P(real final ≥ preserve_ratio · projection)?
    Q2: Required monthly contribution to grow the deterministic projection up
        to Q3's target wealth.
    Q3: Preservation-required wealth for P(preserve ≥X) ≥ confidence at best alloc.

    Terminology (see models.py "TARGET PORTFOLIO TERMINOLOGY" block):
      - IT required portfolio        = income-formula yield bar (€2.07M for user).
      - Preservation-required wealth = Q3 answer, single-confidence (€5.39M for user).
      - Canonical FI bar (dual-conf) = computed separately by main.py.
    """
    _print_section(4, "DECUMULATION — PROJECTED WEALTH vs PRESERVATION TARGET")

    conf_pct = int(round(table.confidence * 100))
    preserve_pct = int(round(table.preserve_ratio * 100))
    cur_bonds_pct = int(round(table.current_bonds_pct * 100))
    cur_stocks_pct = 100 - cur_bonds_pct

    print()
    print(f"  Setup")
    print(f"    Preservation target   : {preserve_pct}% of initial real wealth at horizon  (config: fi_preservation_ratio)")
    print(f"    Confidence target     : {conf_pct}% (real-€)")
    print(f"    Horizon (decum)       : {table.horizon_years} yr")
    print(f"    Current allocation    : {cur_bonds_pct}% bonds / {cur_stocks_pct}% stocks")
    print(f"    Current contribution  : €{format_currency(table.current_monthly_contribution)}/mo (stocks+bonds)")
    print(f"    Target withdrawal     : €{format_currency(table.target_monthly_withdrawal)}/mo (nominal at retirement)")
    print()

    # ── Q1 ────────────────────────────────────────────────────────────────────
    _print_subsection(f"Q1.  At the IT required portfolio, P(preserve ≥{preserve_pct}% real)?")
    print()
    print(f"    {'Anchor':<32} {'Wealth at retirement':>22}    {'P(preserve)':>11}")
    print("    " + "─" * 74)
    flag_req = "✓" if table.q1_p_at_it_required >= table.confidence else "✗"
    flag_proj = "✓" if table.q1_p_at_it_projected >= table.confidence else "✗"
    print(f"    {'IT required (the plan target)':<32} €{format_currency(table.it_required_portfolio):>21}    "
          f"{table.q1_p_at_it_required * 100:>9.1f}% {flag_req}")
    print(f"    {'IT projected (plan delivers)':<32} €{format_currency(table.it_projected_portfolio):>21}    "
          f"{table.q1_p_at_it_projected * 100:>9.1f}% {flag_proj}")
    print(f"\n    ✓ = clears {conf_pct}% confidence    ✗ = below {conf_pct}% bar")
    print()

    # ── Q2 ────────────────────────────────────────────────────────────────────
    _print_subsection(f"Q2.  Required monthly contribution to reach {conf_pct}% confidence (lever 1)")
    print()
    delta = table.q2_required_contribution - table.current_monthly_contribution
    delta_sign = "+" if delta >= 0 else "−"
    if not table.q2_reachable:
        print(f"    Required contribution : >€{format_currency(table.q2_required_contribution)}/mo  "
              f"(target unreachable in {table.horizon_years}y at any contribution level)")
    else:
        print(f"    Current contribution  : €{format_currency(table.current_monthly_contribution)}/mo")
        print(f"    Required contribution : €{format_currency(table.q2_required_contribution)}/mo  "
              f"({delta_sign}€{format_currency(abs(delta))}/mo vs current)")
        print(f"    → grows deterministic projection up to Q3 target (€{format_currency(table.q3_required_wealth)}).")
    print()

    # ── Q3 ────────────────────────────────────────────────────────────────────
    grid_lo = int(round(min(table.bonds_pct_grid) * 100))
    grid_hi = int(round(max(table.bonds_pct_grid) * 100))
    _print_subsection(f"Q3.  Preservation-required wealth for {conf_pct}% confidence (lever 2)")
    print()
    best_bonds_pct = int(round(table.q3_best_bonds_pct * 100))
    best_stocks_pct = 100 - best_bonds_pct
    mult_req = (
        table.q3_required_wealth / table.it_required_portfolio
        if table.it_required_portfolio > 0 else 0.0
    )
    mult_proj = (
        table.q3_required_wealth / table.it_projected_portfolio
        if table.it_projected_portfolio > 0 else 0.0
    )
    unreach = "  (search ceiling — bar truly higher)" if not table.q3_reachable else ""
    print(f"    Preservation-required wealth : €{format_currency(table.q3_required_wealth)}{unreach}")
    print(f"    Best allocation              : {best_bonds_pct}% bonds / {best_stocks_pct}% stocks  "
          f"(search grid: {grid_lo}–{grid_hi}% bonds)")
    print(f"    × IT required portfolio      : {mult_req:.2f}×  (vs the income-formula yield bar)")
    print(f"    × IT projected portfolio     : {mult_proj:.2f}×  (vs what the plan delivers)")
    print()


def print_years_to_retirement_tradeoff(t: YearsToRetirementTradeoff) -> None:
    """Sensitivity table: how preservation outcomes vary with retirement year.

    Columns:
      • Years to retirement (and age at retirement)
      • IT projected portfolio at that year, in today's €
      • Implied real WR at the user's target spend (€/mo today's €)
      • Max safe monthly WD for `confidence`% preserve ≥`preserve_ratio`% (today's €)
      • P50 / P5 real final wealth, expressed as % of the projected portfolio,
        evaluated at the user's target WR.
    """
    conf_pct = int(round(t.confidence * 100))
    preserve_pct = int(round(t.preserve_ratio * 100))
    _print_subsection(f"When can I retire: Years-to-retirement tradeoff (target spend = €{format_currency(t.target_monthly_today_eur)}/mo today's €)")
    print()
    print(f"    {'Yrs':>3}  {'P50 Portfolio €':>17}  "
          f"{'P50 stock gain':>14}  {'P50 stock gain €':>17}  "
          f"{'Max safe WD (today €/mo)':>26}  "
          f"{'P50 final*':>10}  {'P5 final*':>9}  "
          f"{'30y decrement €':>17}")
    print("    " + "─" * 134)
    for r in t.rows:
        proj_str = f"€{format_currency(r.projected_portfolio_today_eur)}"
        max_wd_str = f"€{format_currency(r.max_safe_monthly_today_eur)}"
        gain_pct_str = f"{r.p50_stock_gain_pct * 100:+.1f}%"
        def _signed_eur(v: float) -> str:
            return f"-€{format_currency(-v)}" if v < 0 else f"€{format_currency(v)}"
        gain_eur_str = _signed_eur(r.p50_stock_gain_eur)
        dec_str = _signed_eur(r.decrement_30y_today_eur)
        print(f"    {r.years_to_retirement:>3}  {proj_str:>17}  "
              f"{gain_pct_str:>14}  {gain_eur_str:>17}  "
              f"{max_wd_str:>21}  ({r.max_safe_wr * 100:>4.2f}%)  "
              f"{r.p50_real_final_ratio * 100:>8.0f}%  "
              f"{r.p5_real_final_ratio * 100:>7.0f}%  "
              f"{dec_str:>17}")
    print()
    print(f"    * = withdrawal of today's target income (€{format_currency(t.target_monthly_today_eur)}/mo)")
    print()
    print(f"    Notes")
    print(f"    • 'P50 Portfolio €' = MC P50 (median) accumulation wealth at that horizon, today's €.")
    print(f"      Median of the actually-reachable distribution — not the deterministic upper-bias path.")
    print(f"    • 'P50 stock gain' (% and €) = (MC P50 stock at year N − nominal money invested in stocks).")
    print(f"      Nominal (not inflation-adjusted) — same basis as the accumulation takeaway.")
    print(f"    • 'Max safe WD' = back-solved real €/mo (today's €) at which P(preserve ≥{preserve_pct}%) ≥ {conf_pct}%.")
    print(f"    • 'P50/P5 final*' = real final wealth as % of P50 Portfolio, at end of each row's natural decum horizon.")
    print(f"    • '30y decrement €' = P50 Portfolio − P50 wealth after a fixed 30y decum (today's €), withdrawing target spend.")
    print(f"      Apples-to-apples across rows. Negative = portfolio grew in real terms over 30y.")
    print()


def print_focused_takeaway(
    table: FocusedPreservationTable,
    *,
    tradeoff: "YearsToRetirementTradeoff | None" = None,
) -> None:
    """End-of-report bottom line: years-to-retirement tradeoff table plus the
    IT-required-portfolio stats block. Designed to be the LAST block printed."""
    conf_pct = int(round(table.confidence * 100))
    preserve_pct = int(round(table.preserve_ratio * 100))

    wd_pct = table.real_withdrawal_rate * 100
    surv_pct = table.survival_at_it_required * 100
    median_preserve = table.p50_real_final_at_it_required * 100
    p5_preserve = table.p5_real_final_at_it_required * 100
    max_wd_pct = table.max_sustainable_real_wd_rate * 100
    max_wd_eur = table.max_sustainable_monthly_withdrawal
    bonds_pct = int(round(table.current_bonds_pct * 100))
    stocks_pct = 100 - bonds_pct
    q3_best_bonds_pct = int(round(table.q3_best_bonds_pct * 100))
    q3_best_stocks_pct = 100 - q3_best_bonds_pct

    horizon = table.horizon_years
    median_end_eur = table.it_required_portfolio * table.p50_real_final_at_it_required
    p5_end_eur     = table.it_required_portfolio * table.p5_real_final_at_it_required

    print(f"  At the IT required portfolio (the plan target)")
    print(f"    IT required portfolio                       : €{format_currency(table.it_required_portfolio)}  "
          f"({bonds_pct}% bonds / {stocks_pct}% stocks)")
    print(f"    Target withdrawal rate                      : {wd_pct:.2f}%  "
          f"(€{format_currency(table.real_monthly_withdrawal)}/mo today's €, compare to classic 4% rule)")
    label_w = 44  # chars before the colon, matches the static rows above
    prob_label = f"Probability final wealth > 0 after {horizon}y"
    med_label  = f"Median end wealth after {horizon}y (real €)"
    bad_label  = f"Bad-luck end wealth after {horizon}y (P5, real €)"
    print(f"    {prob_label:<{label_w}}: {surv_pct:.1f}%  — *won't run out*")
    print(f"    {med_label:<{label_w}}: "
          f"€{format_currency(median_end_eur)}  ({median_preserve:.0f}% of starting wealth)")
    print(f"    {bad_label:<{label_w}}: "
          f"€{format_currency(p5_end_eur)}  ({p5_preserve:.0f}% of starting — worst 5% of scenarios)")
    print(f"    Max safe withdrawal to preserve ≥{preserve_pct}% real ({conf_pct}% conf) : "
          f"€{format_currency(max_wd_eur)}/mo today's €  ({max_wd_pct:.2f}% WR)")
    print(f"    Wealth needed for €{format_currency(table.real_monthly_withdrawal)}/mo income AND ≥{preserve_pct}% preservation : "
          f"€{format_currency(table.q3_required_wealth)}  ({q3_best_bonds_pct}% bonds / {q3_best_stocks_pct}% stocks, {conf_pct}% conf)")
    print(f"  Reference — IT projected portfolio (plan delivers): "
          f"€{format_currency(table.it_projected_portfolio)}")
    print()

    if tradeoff is not None:
        print_years_to_retirement_tradeoff(tradeoff)


def _milestone_years(horizon: int, pension_start: int) -> tuple[int, int]:
    """Return (early, late) reporting milestones for the §4.10 heatmap."""
    late = max(1, horizon)
    early_candidate = pension_start if pension_start > 0 else late // 2
    early = max(1, min(early_candidate, late))
    return early, late


def _surv_at(survival_by_year: list[float], year: int) -> float:
    """Survival at year N (1-based). Returns 0 if year is out of range or list is empty."""
    if not survival_by_year or year < 1 or year > len(survival_by_year):
        return 0.0
    return survival_by_year[year - 1]


def print_decum_inflation_cagr_heatmap(result: DecumInflationCagrGrid) -> None:
    _print_subsection("4.10  Decumulation Sensitivity — Inflation × Stock CAGR")
    print()
    early, late = _milestone_years(result.horizon_years, result.pension_start_year)
    if early == late:
        cell_caption = f"survival at year {late}"
    else:
        cell_caption = f"survival at year {early} / year {late}"
    print(f"  Portfolio: €{format_currency(result.display_portfolio)}  "
          f"(fixed withdrawal, {cell_caption})")
    print()

    def _cell(cell: "DecumInflationCagrCell", is_base: bool) -> str:
        s_early = _surv_at(cell.survival_by_year, early) * 100
        s_late = _surv_at(cell.survival_by_year, late) * 100
        s = f"{s_late:.0f}" if early == late else f"{s_early:.0f}/{s_late:.0f}"
        return f"[{s}]" if is_base else f" {s} "

    col_w = 9
    header = f"  {'CPI↓  CAGR→':>12}"
    for cagr in result.cagr_values:
        marker = "*" if abs(cagr - result.base_cagr) < 1e-4 else " "
        header += f"  {int(round(cagr * 100)):>2}%{marker} "
    print(header)
    sep = "  " + "-" * (13 + len(result.cagr_values) * col_w)
    print(sep)

    for i, inf in enumerate(result.inflation_values):
        is_base_inf = abs(inf - result.base_inflation) < 1e-4
        inf_label = f"{inf * 100:.1f}%{'←' if is_base_inf else ' '}"
        row = f"  {inf_label:>12}"
        for j in range(len(result.cagr_values)):
            cell = result.grid[i][j]
            is_base = is_base_inf and abs(result.cagr_values[j] - result.base_cagr) < 1e-4
            row += f"  {_cell(cell, is_base):>7}"
        print(row)

    print(sep)
    cell_legend = (f"yr{late}%" if early == late else f"yr{early}%/yr{late}%")
    print(f"  Values = {cell_legend} survival.  * = base CAGR ({result.base_cagr * 100:.1f}%)  "
          f"← = base CPI ({result.base_inflation * 100:.1f}%)  [x] = base case")


def print_wealth_preservation_under_sorr(surface: WealthPreservationUnderSORR) -> None:
    _print_subsection(
        f"4.11  Wealth Preservation under SORR "
        f"(yr-{surface.horizon_years}, fixed withdrawal, optimal allocation)"
    )
    print()
    print(f"  Initial wealth = €{format_currency(surface.initial_wealth)}")
    grid_pct = ", ".join(f"{int(round(b * 100))}%" for b in surface.bonds_pct_grid)
    print(f"  Each cell      : best P(final wealth ≥ X% of initial) across bonds% "
          f"∈ {{{grid_pct}}}")
    print(f"  Target         : ≥ {surface.target_success_pct}% for FI to be worth the risk")
    print()

    col_w = 8
    name_w = 28
    header = f"  {'Scenario':<{name_w}}"
    for ratio in surface.end_wealth_ratios:
        header += f"  {'P(≥' + str(int(round(ratio * 100))) + '%)':>{col_w}}"
    header += f"   {'Best @≥50%':>11}"
    print(header)
    print("  " + "─" * (name_w + 2 + (col_w + 2) * len(surface.end_wealth_ratios) + 14))

    for row in surface.rows:
        line = f"  {row.scenario_label:<{name_w}}"
        for p in row.best_p_by_ratio:
            line += f"  {p * 100:>{col_w - 1}.1f}%"
        best_alloc_50 = int(round(row.best_alloc_by_ratio[0] * 100))
        line += f"   {best_alloc_50:>4}% bonds"
        print(line)

    # ── Interpretation: median-target P degradation under stress ──────────────
    print()
    target_idx = next(
        (i for i, r in enumerate(surface.end_wealth_ratios) if abs(r - 1.0) < 1e-6),
        None,
    )
    if target_idx is not None and len(surface.rows) >= 2:
        baseline_p = surface.rows[0].best_p_by_ratio[target_idx] * 100
        worst_row = min(surface.rows[1:], key=lambda r: r.best_p_by_ratio[target_idx])
        worst_p = worst_row.best_p_by_ratio[target_idx] * 100
        print(f"  Preserving 100% of initial wealth: {baseline_p:.0f}% under baseline, "
              f"{worst_p:.0f}% under {worst_row.scenario_label}.")
    if surface.end_wealth_ratios and abs(surface.end_wealth_ratios[0] - 0.5) < 1e-6 and len(surface.rows) >= 2:
        baseline_p = surface.rows[0].best_p_by_ratio[0] * 100
        worst_row = min(surface.rows[1:], key=lambda r: r.best_p_by_ratio[0])
        worst_p = worst_row.best_p_by_ratio[0] * 100
        print(f"  Preserving 50% of initial wealth : {baseline_p:.0f}% under baseline, "
              f"{worst_p:.0f}% under {worst_row.scenario_label}.")

    # ── Pre-pension bridge milestone ─────────────────────────────────────────
    print()
    cur_bonds_pct = int(round(surface.current_bonds_pct * 100))
    by = surface.bridge_year
    med_retention = surface.bridge_p50 / surface.initial_wealth * 100.0
    p3_retention = surface.bridge_p3 / surface.initial_wealth * 100.0
    print(f"  Pre-pension bridge — portfolio at yr-{by} (pension start), "
          f"current {cur_bonds_pct}% bonds, baseline returns:")
    print(f"    Bad luck (P3)  : €{format_currency(surface.bridge_p3):>12}  "
          f"({p3_retention:>5.1f}% of initial)")
    print(f"    Median (P50)   : €{format_currency(surface.bridge_p50):>12}  "
          f"({med_retention:>5.1f}% of initial)")
    print(f"    Good luck (P97): €{format_currency(surface.bridge_p97):>12}")
    print(f"    Survival to yr-{by}: {surface.bridge_survival * 100:.1f}%")
