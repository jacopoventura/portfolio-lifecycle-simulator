from models import AllocationSweepResult, ChainedMCResult, DecumInflationCagrGrid, DecumulationResult, JointProbabilityResult, SORRResult, (
    SWRSweepResult)
from reports.formatters import _print_section, _print_subsection, format_currency


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
    joint: JointProbabilityResult,
    deterministic_target: float,
    target_success_pct: int,
) -> None:
    _print_section(4, "DECUMULATION — MONTE CARLO SURVIVAL ANALYSIS")

    # ── 4.1 Required portfolio comparison ────────────────────────────────────
    _print_subsection("4.1  Required Portfolio at Retirement")
    gap_fixed = required_fixed - deterministic_target
    gap_gk    = required_gk    - deterministic_target
    print(f"  Target success rate            : {target_success_pct}%")
    print(f"  Deterministic accumulation plan: €{format_currency(deterministic_target)}")
    print()
    print(f"  {'Strategy':<30}  {'Required portfolio':>20}  {'Gap vs plan':>15}")
    print("  " + "-" * 70)
    print(f"  {'Fixed withdrawal':<30}  €{format_currency(required_fixed):>19}  "
          f"{'€' + format_currency(gap_fixed):>15}")
    print(f"  {'Guyton-Klinger guardrails':<30}  €{format_currency(required_gk):>19}  "
          f"{'€' + format_currency(gap_gk):>15}")
    print()
    if gap_fixed <= 0:
        print("  ✓ Deterministic plan is sufficient for fixed-withdrawal survival.")
    else:
        print(f"  ! Gap (fixed):  plan needs €{format_currency(gap_fixed)} more at retirement.")
    if gap_gk <= 0:
        print("  ✓ Deterministic plan is sufficient with Guyton-Klinger guardrails.")
    else:
        print(f"  ! Gap (GK):     plan needs €{format_currency(gap_gk)} more at retirement.")

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

    # ── 4.6 Joint probability ─────────────────────────────────────────────────
    _print_subsection("4.6  Joint Probability (Accumulate AND Survive)")
    print()
    print(f"  Accumulation portfolio P50  : €{format_currency(joint.accumulation_p50)}")
    print(f"  Accumulation portfolio P{joint.tail_pct:<2}  : €{format_currency(joint.accumulation_tail)}")
    print()
    print(f"  {'Scenario':<35}  {'P(survive)':>12}  {'Joint P':>12}")
    print("  " + "-" * 63)
    print(f"  {'Reach P50, then survive':<35}  "
          f"{joint.p_survive_from_p50 * 100:>11.1f}%  "
          f"{joint.joint_p50 * 100:>11.1f}%")
    print(f"  {'Reach P' + str(joint.tail_pct) + ' (tail), then survive':<35}  "
          f"{joint.p_survive_from_tail * 100:>11.1f}%  "
          f"{joint.joint_tail * 100:>11.1f}%")
    print()
    print("  Note: joint probability = P(accumulate to Px) × P(survive from Px).")
    print("        Independence assumption — accumulation/decumulation risks may be correlated.")


def print_allocation_sweep(sweep: AllocationSweepResult) -> None:
    _print_subsection("4.7  Optimal Stock/Bond Allocation — Joint Sweep")
    glide_desc = (
        f"Glide path: stock% declines from starting allocation → "
        f"{sweep.glide_end_pct * 100:.0f}% over {sweep.glide_years} yr"
        if sweep.glide_years > 0 else "No glide path (fixed allocation)"
    )
    print(f"\n  {glide_desc}")
    print(f"  Optimal (highest yr-30 GK survival): bonds = {sweep.optimal_bonds_pct * 100:.0f}%  "
          f"/ stocks = {(1 - sweep.optimal_bonds_pct) * 100:.0f}%")
    print()

    hdr = (f"  {'bonds%':>6}  {'stocks%':>7}  {'Required €':>14}  "
           f"{'Yr20 fix':>9}  {'Yr30 fix':>9}  {'Yr30 GK':>8}  "
           f"{'Yr35 GK':>8}  {'Yr40 GK':>8}")
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
        print(
            f"  {pt.bonds_pct * 100:>5.0f}%"
            f"  {pt.stock_pct_start * 100:>6.0f}%"
            f"  €{format_currency(pt.required_portfolio):>13}"
            f"  {pt.survival_yr20_fixed * 100:>8.1f}%"
            f"  {pt.survival_yr30_fixed * 100:>8.1f}%"
            f"  {pt.survival_yr30_gk * 100:>7.1f}%"
            f"  {pt.survival_yr35_gk * 100:>7.1f}%"
            f"  {pt.survival_yr40_gk * 100:>7.1f}%"
            f"{tag}"
        )
    print()
    print("  ← current plan   ★ optimal (max yr-30 GK survival)")
    print("  Note: required portfolio computed from income formula; "
          "all rows use identical random draws.")


def print_chained_mc(result: ChainedMCResult) -> None:
    _print_subsection("4.8  Chained Accumulation → Decumulation MC")
    pcts = result.starting_portfolio_percentiles
    print()
    print(f"  {result.n_chained_paths:,} joint paths: accumulation phase feeds directly into decumulation.")
    print(f"  Required portfolio at retirement: €{format_currency(result.required_portfolio)}")
    print()
    print(f"  Accumulated wealth at horizon — distribution:")
    print(f"    P5 : €{format_currency(pcts[5]):>12}   P25: €{format_currency(pcts[25]):>12}"
          f"   P50: €{format_currency(pcts[50]):>12}")
    print(f"    P75: €{format_currency(pcts[75]):>12}   P95: €{format_currency(pcts[95]):>12}")
    print()
    print(f"  Paths reaching ≥ required at year-N : {result.accum_success_frac * 100:.1f}%")
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
    print(f"  {'SWR':>6}  {'Monthly €':>12}  {'Yr 20':>8}  {'Yr 30':>8}  {'Yr 40':>8}")
    print("  " + "-" * 52)
    for pt in result.points:
        tag = " ←" if abs(pt.swr - result.config_swr) < 0.001 else ""
        print(
            f"  {pt.swr * 100:>5.2f}%"
            f"  €{format_currency(pt.monthly_withdrawal):>11}"
            f"  {pt.survival_yr20 * 100:>7.1f}%"
            f"  {pt.survival_yr30 * 100:>7.1f}%"
            f"  {pt.survival_yr40 * 100:>7.1f}%"
            f"{tag}"
        )
    print()
    print("  ← config SWR")


def print_decum_inflation_cagr_heatmap(result: DecumInflationCagrGrid) -> None:
    _print_subsection("4.10  Decumulation Sensitivity — Inflation × Stock CAGR")
    print()
    print(f"  Portfolio: €{format_currency(result.display_portfolio)}  "
          f"(fixed withdrawal, survival at year 30 / year 40)")
    print()

    def _cell(cell: "DecumInflationCagrCell", is_base: bool) -> str:
        s = f"{cell.survival_yr30 * 100:.0f}/{cell.survival_yr40 * 100:.0f}"
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
    print(f"  Values = yr30%/yr40% survival.  * = base CAGR ({result.base_cagr * 100:.1f}%)  "
          f"← = base CPI ({result.base_inflation * 100:.1f}%)  [x] = base case")
