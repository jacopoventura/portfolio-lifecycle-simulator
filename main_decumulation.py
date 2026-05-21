"""
Standalone decumulation analysis.

Usage:
    python main_decumulation.py
    python main_decumulation.py --portfolio 1864445
    python main_decumulation.py --cagr 4
"""
import argparse
import datetime
import tomllib

from models import CanonicalFIRequiredScan, Config, DecumulationSummary
from monte_carlo import (
    compute_allocation_sweep,
    compute_canonical_fi_required_scan,
    compute_chained_mc,
    compute_decum_inflation_cagr_grid,
    compute_sorr_scenarios,
    compute_swr_sweep,
    compute_wealth_preservation_under_sorr,
    simulate_decumulation,
)
from reports.formatters import clear_terminal
from reports.section4 import (
    print_allocation_sweep,
    print_chained_mc,
    print_decum_inflation_cagr_heatmap,
    print_decumulation_report,
    print_decumulation_takeaway,
    print_swr_sweep,
    print_wealth_preservation_under_sorr,
)


def _load_config(path: str = "config.toml") -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Config.from_dict(raw)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo decumulation analysis")
    p.add_argument("--config", default="config.toml", help="Path to config file")
    p.add_argument("--portfolio", type=float, default=None,
                   help="Portfolio to display survival curves for (e.g. accumulation plan output)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-clear", action="store_true",
                   help="Do not clear terminal before output")
    p.add_argument("--cagr", default=None,
                   help="override stock CAGR for the decum MC "
                        "(accepts 4, 4.0, 0.04, '4%%'); per-run only")
    return p.parse_known_args()[0]


def _find_min_portfolio_mc(
    config: Config,
    target_monthly_nom: float,
    target_pct: float,
    seed: int,
    n_simulations: int = 10_000,
    tolerance: float = 1_000.0,
    hi: float = 10_000_000.0,
    withdrawal_strategy: str = "fixed",
) -> float | None:
    """Binary search for the minimum starting portfolio that achieves target_pct survival rate.

    Returns None if even `hi` does not achieve the target survival rate, so callers
    don't mistake the search-range ceiling for a real answer.
    Survival rate is monotonically increasing with starting portfolio, so binary search
    converges reliably regardless of pension timing.
    Uses a reduced n_simulations for speed (10k gives ~±0.5pp accuracy).
    """
    def survival_pct(portfolio: float) -> float:
        return simulate_decumulation(
            config, portfolio, seed=seed, n_simulations=n_simulations,
            withdrawal_strategy=withdrawal_strategy,
            monthly_withdrawal_override=target_monthly_nom,
        ).survival_rate * 100

    if survival_pct(hi) < target_pct:
        return None

    lo = 100_000.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if survival_pct(mid) >= target_pct:
            hi = mid
        else:
            lo = mid
        if hi - lo < tolerance:
            break
    return hi


def main(
    *,
    config_path: str = "config.toml",
    portfolio_override: float | None = None,
    accum_required_override: float | None = None,
    canonical_fi_scan: "CanonicalFIRequiredScan | None" = None,
    seed: int = 42,
    no_clear: bool = False,
    cagr: str | None = None,
) -> DecumulationSummary | None:
    """Run decumulation analysis. Prints the report to stdout, returns a typed summary.

    All inputs are keyword-only so programmatic callers (e.g. main.py) don't need to
    scrub sys.argv. The standalone CLI in __main__ parses argv and forwards here.

    portfolio_override       : starting wealth for survival curves, fan, and withdrawal range
    accum_required_override  : accum IT income-formula required portfolio (for §4.8 secondary threshold)
    cagr                     : per-run override for stock CAGR (decimal or percent string)
    """
    config = _load_config(config_path)
    config.planning_cagr_override = Config.parse_planning_cagr(cagr)

    if not no_clear:
        clear_terminal()

    n = config.mc_decumulation_n_simulations
    target_pct = config.mc_target_success_percent

    print(f"\nDecumulation Monte Carlo  |  {n:,} simulations  |  "
          f"horizon {config.mc_decumulation_years} yr  |  "
          f"target survival {target_pct}%")
    adj = config.mc_decumulation_pension_adjustment_rate
    pension_note = f"  +{adj * 100:.1f}%/yr" if adj > 0 else "  (fixed nominal)"
    print(f"Withdrawal rate: {config.withdrawal_percentage_stock:.2f}% pa  |  "
          f"Stock allocation: {config.mc_decumulation_stock_pct * 100:.0f}%  |  "
          f"Stock return μ: {config.planning_cagr * 100:.1f}%  |  "
          f"Pension: €{config.mc_decumulation_pension_monthly:,.0f}/mo "
          f"from year {config.mc_decumulation_pension_start_year}{pension_note}")
    print()

    # ── Target withdrawal at retirement (today's € inflated to retirement day) ─
    infl_factor = (1.0 + config.expected_inflation_percent / 100.0) ** config.years
    target_monthly_nom = config.target_monthly_retirement_income * infl_factor
    print(f"  Target withdrawal : €{target_monthly_nom:,.0f}/mo  "
          f"(€{config.target_monthly_retirement_income:,.0f} today × inflation {config.years:.0f}yr)",
          flush=True)

    # ── Required portfolio: MC minimum for target survival, per strategy ──────
    # These are the canonical "required wealth" anchors — fixed-withdrawal MC min
    # is used wherever the report needs a single deterministic-style reference.
    search_hi = 10_000_000.0

    def _print_mc_min(label: str, mc_min: float | None, delta_anchor: float | None = None,
                      delta_note: str = "") -> None:
        prefix = f"\r  MC minimum ({label:<5}): "
        if mc_min is None:
            print(f"{prefix}>€{search_hi:,.0f}  "
                  f"(target {target_pct:.0f}% survival unreachable in search range)   ")
            return
        if delta_anchor is None:
            print(f"{prefix}€{mc_min:,.0f}                                        ")
        else:
            delta = mc_min - delta_anchor
            print(f"{prefix}€{mc_min:,.0f}  ({delta:+,.0f} {delta_note})"
                  f"                    ")

    print(f"  MC minimum (fixed, {target_pct:.0f}% survival, 10k sims)…",
          end="", flush=True)
    mc_min_fixed = _find_min_portfolio_mc(
        config, target_monthly_nom, target_pct, seed=seed, hi=search_hi,
        withdrawal_strategy="fixed",
    )
    _print_mc_min("fixed", mc_min_fixed)

    print(f"  MC minimum (GK,    {target_pct:.0f}% survival, 10k sims)…",
          end="", flush=True)
    mc_min_gk = _find_min_portfolio_mc(
        config, target_monthly_nom, target_pct, seed=seed, hi=search_hi,
        withdrawal_strategy="guyton_klinger",
    )
    _print_mc_min("GK", mc_min_gk, delta_anchor=mc_min_fixed, delta_note="vs fixed: guardrails benefit")

    # ── Canonical FI required wealth (preservation-calibrated, scanned) ──────
    if canonical_fi_scan is not None:
        fi_scan = canonical_fi_scan
        print(f"  Canonical FI scan: reusing precomputed (worst-SORR: {fi_scan.worst_stress_label})")
    else:
        print(f"  Canonical FI scan (preserve ≥{int(round(config.fi_preservation_ratio * 100))}% baseline P≥"
              f"{int(round(config.fi_preservation_confidence_baseline * 100))}% "
              f"& worst-SORR P≥{int(round(config.fi_preservation_confidence_stress * 100))}%)…",
              end="", flush=True)
        fi_scan = compute_canonical_fi_required_scan(config, target_monthly_nom, seed=seed)
        print(f"\r  Canonical FI scan computed (worst-SORR: {fi_scan.worst_stress_label})         ")
    print()
    print(f"  FI required wealth scan (worst-SORR = {fi_scan.worst_stress_label}):")
    print(f"    {'Preserve':>9}   {'Required €':>13}   {'Opt bonds':>10}   "
          f"{'Baseline P':>10}   {'Stress P':>9}")
    print(f"    {'─' * 66}")
    for row in fi_scan.rows:
        unreach = "" if row.reachable else "  (unreached)"
        print(f"    {int(round(row.preserve_ratio * 100)):>7}%   "
              f"€{row.required_wealth_eur:>12,.0f}   "
              f"{int(round(row.optimal_bonds_pct * 100)):>8}%    "
              f"{row.achieved_baseline_p * 100:>9.1f}%   "
              f"{row.achieved_stress_p * 100:>8.1f}%{unreach}")
    print()
    print(f"  Canonical FI bar (preserve ≥{int(round(fi_scan.canonical_ratio * 100))}%): "
          f"€{fi_scan.canonical_required_wealth:,.0f}  "
          f"at {int(round(fi_scan.canonical_optimal_bonds_pct * 100))}% bonds"
          f"{'' if fi_scan.canonical_reachable else '  (UNREACHED in search range)'}")
    print()

    # Canonical "required wealth" anchor for the rest of this run.
    # Phase 2: canonical FI bar (preservation-calibrated) replaces the old survival-only
    # MC fixed minimum. Falls back to search_hi if the canonical was unreachable.
    if fi_scan.canonical_reachable:
        required_anchor = fi_scan.canonical_required_wealth
    else:
        required_anchor = search_hi
    # `mc_min_fixed` and `mc_min_gk` are kept around as diagnostic lower bounds in §4.1.

    display_portfolio = portfolio_override if portfolio_override is not None else required_anchor
    survival_label = (
        "(accumulation plan target)" if portfolio_override is not None
        else "(FI canonical bar)"
    )
    print(f"  Survival curves   : €{display_portfolio:,.0f}  {survival_label}")
    print()

    # ── Full simulations for display ──────────────────────────────────────────
    print(f"Simulating fixed withdrawal at €{display_portfolio:,.0f}…", flush=True)
    result_fixed = simulate_decumulation(
        config, display_portfolio, seed=seed, n_simulations=n,
        withdrawal_strategy="fixed",
        monthly_withdrawal_override=target_monthly_nom,
    )

    print(f"Simulating Guyton-Klinger at €{display_portfolio:,.0f}…", flush=True)
    result_gk = simulate_decumulation(
        config, display_portfolio, seed=seed, n_simulations=n,
        withdrawal_strategy="guyton_klinger",
        monthly_withdrawal_override=target_monthly_nom,
    )

    # ── SORR stress scenarios ─────────────────────────────────────────────────
    print("Computing SORR stress scenarios…", flush=True)
    sorr = compute_sorr_scenarios(
        config, display_portfolio, seed=seed, n_simulations=n,
        withdrawal_strategy="fixed",
        monthly_withdrawal_override=target_monthly_nom,
    )

    # ── Allocation sweep ──────────────────────────────────────────────────────
    print("Running allocation sweep (glide path optimisation)…", flush=True)
    sweep = compute_allocation_sweep(config, seed=seed, n_simulations=min(n, 5_000))

    # ── Chained accumulation → decumulation MC ────────────────────────────────
    print("Running chained accumulation → decumulation MC…", flush=True)
    monthly_stock = config.annual_net_saving / 12 * config.maximum_stock_monthly_investment_pct
    monthly_bonds = config.annual_net_saving / 12 * config.maximum_bonds_monthly_investment_pct
    chained = compute_chained_mc(
        config=config,
        monthly_stock_contribution=monthly_stock,
        monthly_bond_contribution=monthly_bonds,
        required_portfolio=required_anchor,
        n_paths=n,
        seed=seed,
        secondary_required_portfolio=accum_required_override,
        secondary_required_label="accum income formula" if accum_required_override else None,
    )

    # ── SWR sensitivity sweep ─────────────────────────────────────────────────
    print("Running SWR sensitivity sweep…", flush=True)
    swr_sweep = compute_swr_sweep(config, display_portfolio, seed=seed)

    # ── Decumulation inflation × CAGR grid ────────────────────────────────────
    print("Running decumulation inflation × CAGR grid…", flush=True)
    decum_grid = compute_decum_inflation_cagr_grid(config, display_portfolio, seed=seed)

    # ── Deterministic reference (accumulation plan target) ───────────────────
    deterministic_target = display_portfolio

    # ── Report ────────────────────────────────────────────────────────────────
    # Unreachable GK MC search falls back to search_hi for the §4.1 row.
    required_gk_report = mc_min_gk if mc_min_gk is not None else search_hi
    print_decumulation_report(
        result_fixed=result_fixed,
        result_gk=result_gk,
        required_fixed=mc_min_fixed if mc_min_fixed is not None else search_hi,
        required_gk=required_gk_report,
        sorr=sorr,
        deterministic_target=deterministic_target,
        target_success_pct=target_pct,
        fi_scan=fi_scan,
    )
    print_allocation_sweep(sweep)
    print_chained_mc(chained)
    print_swr_sweep(swr_sweep)
    print_decum_inflation_cagr_heatmap(decum_grid)

    # ── Wealth preservation under SORR ────────────────────────────────────────
    # Runs at display_portfolio (planned wealth), not the canonical bar, so the
    # matrix answers "at the wealth I expect to have, what's the preservation?"
    print("Running wealth preservation under SORR…", flush=True)
    wealth_surface = compute_wealth_preservation_under_sorr(
        config,
        initial_wealth=display_portfolio,
        target_monthly_nom=target_monthly_nom,
        seed=seed,
    )
    print_wealth_preservation_under_sorr(wealth_surface)

    fi_year = datetime.date.today().year + int(round(config.years))
    print_decumulation_takeaway(
        result_fixed=result_fixed,
        result_gk=result_gk,
        chained=chained,
        sorr=sorr,
        swr_sweep=swr_sweep,
        decum_grid=decum_grid,
        sweep=sweep,
        mc_min_fixed=mc_min_fixed,
        mc_min_gk=mc_min_gk,
        display_portfolio=display_portfolio,
        target_monthly_nom=target_monthly_nom,
        target_success_pct=target_pct,
        horizon_years=config.mc_decumulation_years,
        fi_scan=fi_scan,
        wealth_surface=wealth_surface,
        fi_year=fi_year,
    )

    return DecumulationSummary(
        required_portfolio_fi_canonical=required_anchor,
        target_monthly_withdrawal=target_monthly_nom,
        n_years=config.mc_decumulation_years,
        survival_fixed_final=result_fixed.survival_by_year[-1] if result_fixed.survival_by_year else result_fixed.survival_rate,
        survival_gk_final=result_gk.survival_by_year[-1] if result_gk.survival_by_year else result_gk.survival_rate,
        joint_survival_chained=chained.joint_survival_rate,
    )


if __name__ == "__main__":
    _args = _parse_args()
    main(
        config_path=_args.config,
        portfolio_override=_args.portfolio,
        seed=_args.seed,
        no_clear=_args.no_clear,
        cagr=_args.cagr,
    )
