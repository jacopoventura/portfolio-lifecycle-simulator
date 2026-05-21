from models import (
    AllocationSweepResult,
    CanonicalFIRequiredScan,
    ChainedMCResult,
    DecumInflationCagrGrid,
    DecumulationResult,
    SORRResult,
    SWRSweepResult,
    WealthPreservationUnderSORR,
)
from reports.formatters import _print_section, _print_subsection, format_currency


def _milestone_years(horizon: int, pension_start: int) -> tuple[int, int]:
    """Return (early, late) reporting milestones derived from config.

    Early year defaults to pension_start (or horizon//2 if pension lands at/past horizon).
    Late year is always the full horizon. Used by the §4.7, §4.9, and §4.10 printers
    so the displayed columns adapt to whatever mc_decumulation_years is set to.
    """
    late = max(1, horizon)
    early_candidate = pension_start if pension_start > 0 else late // 2
    early = max(1, min(early_candidate, late))
    return early, late


def _surv_at(survival_by_year: list[float], year: int) -> float:
    """Survival at year N (1-based). Returns 0 if year is out of range or list is empty."""
    if not survival_by_year or year < 1 or year > len(survival_by_year):
        return 0.0
    return survival_by_year[year - 1]


def _survival_bar(rate: float, width: int = 30) -> str:
    filled = round(rate * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {rate * 100:5.1f}%"


def _portfolio_row(label: str, pcts: dict[int, float]) -> str:
    return (
        f"  {label:<22}"
        f"  {format_currency(pcts[3]):>12}"
        f"  {format_currency(pcts[25]):>12}"
        f"  {format_currency(pcts[50]):>12}"
        f"  {format_currency(pcts[75]):>12}"
        f"  {format_currency(pcts[97]):>12}"
    )


def _print_survival_curve(result: DecumulationResult, label: str = "") -> None:
    tag = f" ({label})" if label else ""
    print(f"\n  Survival curve{tag} — {result.withdrawal_strategy}, "
          f"initial portfolio €{format_currency(result.initial_portfolio)}, "
          f"withdrawal {result.initial_monthly_withdrawal / result.initial_portfolio * 100 * 12:.2f}% pa")
    print()
    key_years = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    for y in key_years:
        if y <= result.n_years:
            idx = y - 1
            rate = result.survival_by_year[idx]
            print(f"    Year {y:>2}:  {_survival_bar(rate)}")
    print(f"\n  Final survival rate (year {result.n_years}): "
          f"{result.survival_rate * 100:.1f}%")
    if result.ruin_year_median is not None:
        print(f"  Median year of ruin (failed sims): year {result.ruin_year_median}")


def _print_portfolio_fan(result: DecumulationResult) -> None:
    print()
    print(f"  {'Year':<22}  {'P3':>12}  {'P25':>12}  {'P50':>12}  {'P75':>12}  {'P97':>12}")
    print("  " + "-" * 80)
    years_to_show = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    for y in years_to_show:
        if y <= result.n_years:
            pcts = result.portfolio_percentiles_by_year[y - 1]
            print(_portfolio_row(f"Year {y}", pcts))


def print_decumulation_report(
    result_fixed: DecumulationResult,
    result_gk: DecumulationResult,
    required_fixed: float,
    required_gk: float,
    sorr: SORRResult,
    deterministic_target: float,
    target_success_pct: int,
    fi_scan: CanonicalFIRequiredScan,
) -> None:
    _print_section(4, "DECUMULATION — WEALTH PRESERVATION ANALYSIS")

    # ── 4.1 Required portfolio (canonical FI bar + survival-only diagnostics) ─
    _print_subsection("4.1  Required Portfolio at Retirement")
    canonical = fi_scan.canonical_required_wealth
    canonical_alloc_pct = int(round(fi_scan.canonical_optimal_bonds_pct * 100))
    preserve_pct = int(round(fi_scan.canonical_ratio * 100))
    base_pct = int(round(fi_scan.confidence_baseline * 100))
    stress_pct = int(round(fi_scan.confidence_stress * 100))

    print(f"  Canonical FI bar  : preserve ≥{preserve_pct}% of initial wealth "
          f"at yr {fi_scan.horizon_years}")
    print(f"                      baseline P≥{base_pct}%, worst-SORR P≥{stress_pct}% "
          f"(worst = {fi_scan.worst_stress_label})")
    print(f"  Required wealth   : €{format_currency(canonical)}  "
          f"(optimal alloc {canonical_alloc_pct}% bonds)")
    print(f"  Planned wealth    : €{format_currency(deterministic_target)}")
    gap_fi = canonical - deterministic_target
    if abs(gap_fi) < 1.0:
        print(f"  Gap vs FI bar     : €0  (exactly at FI bar)  ✓ on track")
    elif gap_fi < 0:
        print(f"  Gap vs FI bar     : €{format_currency(-gap_fi)} surplus  ✓ on track")
    else:
        print(f"  Gap vs FI bar     : €{format_currency(gap_fi)} short    ✗ off track")
    print()
    print(f"  Diagnostic — survival-only minima (balance > €0 at yr-{result_fixed.n_years}, "
          f"P≥{target_success_pct}%):")
    gap_fixed = required_fixed - canonical
    gap_gk    = required_gk    - canonical
    print(f"    Fixed withdrawal MC min    : €{format_currency(required_fixed):>12}  "
          f"({'€' + format_currency(gap_fixed) if gap_fixed >= 0 else '−€' + format_currency(-gap_fixed)} vs FI bar)")
    print(f"    Guyton-Klinger MC min      : €{format_currency(required_gk):>12}  "
          f"({'€' + format_currency(gap_gk) if gap_gk >= 0 else '−€' + format_currency(-gap_gk)} vs FI bar)")
    print()
    print("  Note: survival is the permissive 'balance > €0' bar. The FI canonical bar")
    print("        is the preservation-calibrated requirement — what accumulation should target.")

    # ── 4.2 Survival curves ──────────────────────────────────────────────────
    _print_subsection("4.2  Survival Curves")
    _print_survival_curve(result_fixed, "fixed withdrawal")
    _print_survival_curve(result_gk,    "Guyton-Klinger")

    # ── 4.3 Portfolio fan chart ───────────────────────────────────────────────
    _print_subsection("4.3  Portfolio Fan (fixed withdrawal, €)")
    _print_portfolio_fan(result_fixed)

    # ── 4.4 Monthly withdrawal range ─────────────────────────────────────────
    _print_subsection("4.4  Monthly Withdrawal Range (Guyton-Klinger, €/mo)")
    print()
    print(f"  {'Year':<10}  {'P10':>12}  {'P50':>12}  {'P90':>12}")
    print("  " + "-" * 50)
    years_to_show = [1, 5, 10, 15, 20, 25, 30]
    for y in years_to_show:
        if y <= result_gk.n_years:
            w = result_gk.withdrawal_percentiles_by_year[y - 1]
            print(f"  {'Year ' + str(y):<10}  "
                  f"€{format_currency(w[10]):>11}  "
                  f"€{format_currency(w[50]):>11}  "
                  f"€{format_currency(w[90]):>11}")

    # ── 4.5 SORR stress scenarios ─────────────────────────────────────────────
    _print_subsection("4.5  Sequence-of-Returns Risk (SORR) Stress Scenarios")
    print()
    print(f"  {'Scenario':<30}  {'Survival rate':>14}  {'vs baseline':>12}")
    print("  " + "-" * 60)
    baseline_rate = sorr.baseline.survival_rate
    print(f"  {'Baseline (random)':<30}  {baseline_rate * 100:>13.1f}%  {'—':>12}")
    for label, stress in sorr.stress_scenarios:
        delta = (stress.survival_rate - baseline_rate) * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<30}  {stress.survival_rate * 100:>13.1f}%  "
              f"{sign}{delta:>10.1f} pp")
    print()
    print("  Note: 'survival' = balance > €0 at year-N, a permissive bar. For the")
    print("        FI-worth-it question, see §4.11 (end-wealth preservation under SORR).")


def print_allocation_sweep(sweep: AllocationSweepResult) -> None:
    _print_subsection("4.7  Optimal Stock/Bond Allocation — Joint Sweep")
    glide_desc = (
        f"Glide path: stock% declines from starting allocation → "
        f"{sweep.glide_end_pct * 100:.0f}% over {sweep.glide_years} yr"
        if sweep.glide_years > 0 else "No glide path (fixed allocation)"
    )
    print(f"\n  {glide_desc}")
    horizon = sweep.horizon_years
    early, late = _milestone_years(horizon, sweep.pension_start_year)
    print(f"  Optimal (highest yr-{late} GK survival): bonds = "
          f"{sweep.optimal_bonds_pct * 100:.0f}%  / stocks = "
          f"{(1 - sweep.optimal_bonds_pct) * 100:.0f}%")
    print()

    if early == late:
        hdr = (f"  {'bonds%':>6}  {'stocks%':>7}  {'Required €':>14}  "
               f"{'Yr' + str(late) + ' fix':>9}  {'Yr' + str(late) + ' GK':>9}")
    else:
        hdr = (f"  {'bonds%':>6}  {'stocks%':>7}  {'Required €':>14}  "
               f"{'Yr' + str(early) + ' fix':>9}  {'Yr' + str(late) + ' fix':>9}  "
               f"{'Yr' + str(early) + ' GK':>8}  {'Yr' + str(late) + ' GK':>8}")
    print(hdr)
    print("  " + "-" * 80)

    for pt in sweep.points:
        is_base    = abs(pt.bonds_pct - sweep.base_bonds_pct) < 0.01
        is_optimal = abs(pt.bonds_pct - sweep.optimal_bonds_pct) < 0.01
        if is_optimal and is_base:
            tag = " ★←"
        elif is_optimal:
            tag = " ★"
        elif is_base:
            tag = " ←"
        else:
            tag = ""
        row = (
            f"  {pt.bonds_pct * 100:>5.0f}%"
            f"  {pt.stock_pct_start * 100:>6.0f}%"
            f"  €{format_currency(pt.required_portfolio):>13}"
        )
        if early != late:
            row += f"  {_surv_at(pt.survival_by_year_fixed, early) * 100:>8.1f}%"
        row += f"  {_surv_at(pt.survival_by_year_fixed, late) * 100:>8.1f}%"
        if early != late:
            row += f"  {_surv_at(pt.survival_by_year_gk, early) * 100:>7.1f}%"
        row += f"  {_surv_at(pt.survival_by_year_gk, late) * 100:>7.1f}%{tag}"
        print(row)
    print()
    print(f"  ← current plan   ★ optimal (max yr-{late} GK survival)")
    print("  Note: required portfolio computed from income formula; "
          "all rows use identical random draws.")


def print_chained_mc(result: ChainedMCResult) -> None:
    _print_subsection("4.8  Chained Accumulation → Decumulation MC")
    pcts = result.starting_portfolio_percentiles
    print()
    print(f"  {result.n_chained_paths:,} joint paths: accumulation phase feeds directly into decumulation.")
    print()
    print(f"  Accumulated wealth at horizon — distribution:")
    print(f"    P5 : €{format_currency(pcts[5]):>12}   P25: €{format_currency(pcts[25]):>12}"
          f"   P50: €{format_currency(pcts[50]):>12}")
    print(f"    P75: €{format_currency(pcts[75]):>12}   P95: €{format_currency(pcts[95]):>12}")
    print()
    if result.secondary_required_portfolio is not None and result.secondary_accum_success_frac is not None:
        primary_lbl = "decum survival-min"
        secondary_lbl = result.secondary_required_label or "accum target"
        print(f"  Paths reaching ≥ required at retirement:")
        print(f"    ≥ €{format_currency(result.required_portfolio)} ({primary_lbl})"
              f"      : {result.accum_success_frac * 100:5.1f}%")
        print(f"    ≥ €{format_currency(result.secondary_required_portfolio)} ({secondary_lbl})"
              f"  : {result.secondary_accum_success_frac * 100:5.1f}%")
    else:
        print(f"  Required portfolio at retirement: €{format_currency(result.required_portfolio)}")
        print(f"  Paths reaching ≥ required at retirement: {result.accum_success_frac * 100:.1f}%")
    print()

    key_years = [10, 20, 30, 40]
    n_years = len(result.survival_by_year)
    print(f"  {'Year':<8}  {'Joint survival':>15}")
    print("  " + "-" * 26)
    for y in key_years:
        if y <= n_years:
            rate = result.survival_by_year[y - 1]
            bar = f"[{'█' * round(rate * 20)}{'░' * (20 - round(rate * 20))}]"
            print(f"  Year {y:<4}  {bar}  {rate * 100:5.1f}%")
    print()

    def _fmt_surv(v: float) -> str:
        return f"{v * 100:.1f}%" if v == v else "n/a"  # nan check

    print(f"  Conditional survival at year {n_years}:")
    print(f"    Started with ≥ required ({result.accum_success_frac * 100:.1f}% of paths) "
          f"→ {_fmt_surv(result.cond_survival_accum_success)} survive")
    fail_frac = 1.0 - result.accum_success_frac
    print(f"    Started with < required ({fail_frac * 100:.1f}% of paths) "
          f"→ {_fmt_surv(result.cond_survival_accum_fail)} survive")
    print()
    print(f"  True joint P(accumulate AND survive {n_years} yr) : "
          f"{result.joint_survival_rate * 100:.1f}%")


def print_swr_sweep(result: SWRSweepResult) -> None:
    _print_subsection("4.9  Withdrawal Rate Sensitivity")
    print()
    print(f"  Portfolio: €{format_currency(result.display_portfolio)}  "
          f"(all rows share identical random draws — differences reflect SWR only)")
    print()

    early, late = _milestone_years(result.horizon_years, result.pension_start_year)
    if early == late:
        print(f"  {'SWR':>6}  {'Monthly €':>12}  {'Yr ' + str(late):>8}")
    else:
        print(f"  {'SWR':>6}  {'Monthly €':>12}  {'Yr ' + str(early):>8}  {'Yr ' + str(late):>8}")
    print("  " + "-" * 52)
    for pt in result.points:
        tag = " ←" if abs(pt.swr - result.config_swr) < 0.001 else ""
        row = (
            f"  {pt.swr * 100:>5.2f}%"
            f"  €{format_currency(pt.monthly_withdrawal):>11}"
        )
        if early != late:
            row += f"  {_surv_at(pt.survival_by_year, early) * 100:>7.1f}%"
        row += f"  {_surv_at(pt.survival_by_year, late) * 100:>7.1f}%{tag}"
        print(row)
    print()
    print("  ← config SWR")


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

    # Header: scenario name | one column per end-wealth ratio | best alloc for 50%
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
    # Find the 100% column (preservation = initial) and report baseline vs worst stress
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
    # Preservation at 50% under worst stress
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


def print_decumulation_takeaway(
    *,
    result_fixed: DecumulationResult,
    result_gk: DecumulationResult,
    chained: ChainedMCResult,
    sorr: SORRResult,
    swr_sweep: SWRSweepResult,
    decum_grid: DecumInflationCagrGrid,
    sweep: AllocationSweepResult,
    mc_min_fixed: float | None,
    mc_min_gk: float | None,
    display_portfolio: float,
    target_monthly_nom: float,
    target_success_pct: int,
    horizon_years: int,
    fi_scan: CanonicalFIRequiredScan,
    wealth_surface: WealthPreservationUnderSORR,
    fi_year: int | None = None,
) -> None:
    """Final actionable takeaway: verdict (preservation-led), risks, levers, status.

    The verdict block now answers 'is FI worth doing?' via preservation under both
    baseline AND worst-SORR — the canonical philosophy. Survival rates are kept as
    a supporting note since survival ≠ FI.
    """
    target = target_success_pct / 100.0
    surv_fixed = result_fixed.survival_rate
    surv_gk = result_gk.survival_rate
    surv_chained = chained.joint_survival_rate

    _print_subsection(
        f"TAKEAWAY — DECUMULATION ({horizon_years}y horizon, "
        f"€{format_currency(target_monthly_nom)}/mo target at retirement)"
    )

    # ── VERDICT (preservation-led) ───────────────────────────────────────────
    canonical_required = fi_scan.canonical_required_wealth
    gap_to_fi = display_portfolio - canonical_required
    preserve_pct_label = int(round(fi_scan.canonical_ratio * 100))
    Y_pct = int(round(fi_scan.confidence_baseline * 100))
    Z_pct = int(round(fi_scan.confidence_stress * 100))

    print(f"  VERDICT  at planned €{format_currency(display_portfolio)} retirement wealth")
    print()
    print(f"  FI canonical bar  : €{format_currency(canonical_required)}  "
          f"(preserve ≥{preserve_pct_label}% · baseline P≥{Y_pct}% · "
          f"worst-SORR P≥{Z_pct}%)")
    if abs(gap_to_fi) < 1.0:
        print(f"  Gap vs FI bar     : €0  (exactly at FI bar)  ✓ on track")
    elif gap_to_fi > 0:
        print(f"  Gap vs FI bar     : €{format_currency(gap_to_fi)} surplus  "
              f"✓ on track")
    else:
        print(f"  Gap vs FI bar     : €{format_currency(-gap_to_fi)} short    "
              f"✗ off track")
    print()

    # Preservation matrix at the planned wealth (from §4.11)
    # Locate baseline + worst-stress rows and the 50% / 100% ratio columns.
    ratios = wealth_surface.end_wealth_ratios
    try:
        col_50 = next(i for i, r in enumerate(ratios) if abs(r - 0.50) < 1e-6)
    except StopIteration:
        col_50 = 0
    col_100 = next(
        (i for i, r in enumerate(ratios) if abs(r - 1.00) < 1e-6), len(ratios) - 1
    )
    baseline_row = wealth_surface.rows[0]
    worst_row = min(
        wealth_surface.rows[1:],
        key=lambda r: r.best_p_by_ratio[col_50],
    ) if len(wealth_surface.rows) > 1 else baseline_row

    def _flag_baseline(p: float) -> str:
        return "✓" if p >= fi_scan.confidence_baseline else "✗"

    def _flag_stress(p: float) -> str:
        return "✓" if p >= fi_scan.confidence_stress else "✗"

    print(f"  Preservation at planned wealth (best-allocation, from §4.11):")
    print(f"                                Baseline      {worst_row.scenario_label}")
    print(f"    P(preserve ≥50%)            "
          f"{baseline_row.best_p_by_ratio[col_50] * 100:>5.1f}% {_flag_baseline(baseline_row.best_p_by_ratio[col_50])}  "
          f"     {worst_row.best_p_by_ratio[col_50] * 100:>5.1f}% {_flag_stress(worst_row.best_p_by_ratio[col_50])}")
    print(f"    P(preserve ≥100%)           "
          f"{baseline_row.best_p_by_ratio[col_100] * 100:>5.1f}%    "
          f"     {worst_row.best_p_by_ratio[col_100] * 100:>5.1f}%")
    print()

    # ── SURVIVAL (footnote) ──────────────────────────────────────────────────
    print(f"  Survival (permissive: balance > €0 at yr-{horizon_years}):")
    print(f"    Fixed {surv_fixed * 100:.1f}%  ·  GK {surv_gk * 100:.1f}%  ·  "
          f"Chained {surv_chained * 100:.1f}%")
    print("  (Survival is a lower bar than preservation — see §4.5 / §4.11.)")
    print()

    # ── Verdict prose ────────────────────────────────────────────────────────
    on_fi = gap_to_fi >= 0
    stress_p = worst_row.best_p_by_ratio[col_50]
    if on_fi and stress_p >= fi_scan.confidence_stress:
        print("  → On track: planned wealth clears the FI canonical bar and the worst-SORR")
        print("    floor. Equity risk is being rewarded by preservation.")
    elif on_fi and stress_p < fi_scan.confidence_stress:
        print("  → On track for baseline preservation, but FRAGILE under worst-SORR.")
        print("    Worth considering a higher wealth target or more bonds at retirement.")
    else:
        print(f"  → Off track: need €{format_currency(-gap_to_fi)} more wealth to clear the FI bar.")
        print("    See LEVERS below for adjustment options.")

    # ── TOP RISKS ─────────────────────────────────────────────────────────────
    print()
    print("  TOP RISKS  (what kills the plan)")

    baseline_surv = sorr.baseline.survival_rate
    worst = max(sorr.stress_scenarios, key=lambda kv: baseline_surv - kv[1].survival_rate)
    worst_label, worst_res = worst
    drop_sorr = (baseline_surv - worst_res.survival_rate) * 100
    print(f"    1. {worst_label:<22}  survival {baseline_surv * 100:.1f}% → "
          f"{worst_res.survival_rate * 100:.1f}%   (−{drop_sorr:.0f} pp)")

    # CPI stress: highest inflation row in grid at base CAGR column
    try:
        cagr_idx = next(
            i for i, c in enumerate(decum_grid.cagr_values)
            if abs(c - decum_grid.base_cagr) < 1e-4
        )
    except StopIteration:
        cagr_idx = len(decum_grid.cagr_values) // 2
    hi_inf = decum_grid.inflation_values[-1]
    base_inf_row = next(
        (i for i, v in enumerate(decum_grid.inflation_values)
         if abs(v - decum_grid.base_inflation) < 1e-4),
        len(decum_grid.inflation_values) // 2,
    )
    surv_base_grid = _surv_at(decum_grid.grid[base_inf_row][cagr_idx].survival_by_year, horizon_years)
    surv_hi_cpi = _surv_at(decum_grid.grid[-1][cagr_idx].survival_by_year, horizon_years)
    cpi_drop = (surv_base_grid - surv_hi_cpi) * 100
    print(f"    2. Persistent CPI {hi_inf * 100:.1f}%   yr-{horizon_years} survival → "
          f"{surv_hi_cpi * 100:.0f}%   (−{cpi_drop:.0f} pp)")

    # SWR creep: next SWR point above config
    swr_points = sorted(swr_sweep.points, key=lambda p: p.swr)
    next_swr = next((p for p in swr_points if p.swr > swr_sweep.config_swr + 1e-6), None)
    if next_swr is not None:
        cur_swr_pt = next(
            (p for p in swr_points if abs(p.swr - swr_sweep.config_swr) < 1e-6),
            None,
        )
        base_swr_final = (
            _surv_at(cur_swr_pt.survival_by_year, horizon_years) if cur_swr_pt else surv_fixed
        )
        next_swr_final = _surv_at(next_swr.survival_by_year, horizon_years)
        creep_drop = (base_swr_final - next_swr_final) * 100
        print(f"    3. SWR creep to {next_swr.swr * 100:.1f}%   yr-{horizon_years} survival → "
              f"{next_swr_final * 100:.0f}%   (−{creep_drop:.0f} pp)")

    # ── LEVERS ────────────────────────────────────────────────────────────────
    print()
    print("  LEVERS  (what moves the needle)")
    if mc_min_fixed is not None and mc_min_gk is not None and mc_min_gk < mc_min_fixed:
        save = mc_min_fixed - mc_min_gk
        print(f"    • Adopt GK guardrails               → −€{format_currency(save)} required wealth")
    cur_bonds = int(round(sweep.base_bonds_pct * 100))
    opt_bonds = int(round(sweep.optimal_bonds_pct * 100))
    if opt_bonds != cur_bonds:
        print(f"    • Reweight to {opt_bonds}% bonds / {100 - opt_bonds}% stocks"
              f"  (current {cur_bonds}%)")
        print(f"      → highest yr-{horizon_years} GK survival in the allocation sweep")
    print("    • Hold 2-yr cash reserve at retirement start  → directly mitigates risk #1")

    # ── ONE-LINE STATUS ───────────────────────────────────────────────────────
    print()
    on_track = surv_gk >= target and surv_chained >= target
    status = "On track" if on_track else ("Borderline" if surv_gk >= target else "Off track")
    fi_clause = f" for {fi_year} FI" if fi_year is not None else ""
    print("  ONE-LINE STATUS:")
    print(f"    {status}{fi_clause} under GK + current allocation.")

    fragile = []
    if surv_fixed < target:
        fragile.append("you require fixed spending")
    if surv_hi_cpi < 0.60:
        fragile.append(f"CPI runs > {hi_inf * 100 - 0.5:.0f}%")
    if fragile:
        print(f"    Fragile if {' or '.join(fragile)}.")
