"""
Standalone decumulation analysis.

Usage:
    python main_decumulation.py --portfolio 1864445 --p50 1864445 --tail 1200000
    python main_decumulation.py --portfolio 2500000 --p50 2500000 --tail 1500000 --tail-pct 5
"""
import argparse
import tomllib

from models import Config
from monte_carlo import (
    compute_allocation_sweep,
    compute_chained_mc,
    compute_decum_inflation_cagr_grid,
    compute_joint_probability,
    compute_sorr_scenarios,
    compute_swr_sweep,
    simulate_decumulation,
)
from reports.formatters import clear_terminal
from reports.section4 import (
    print_allocation_sweep,
    print_chained_mc,
    print_decum_inflation_cagr_heatmap,
    print_decumulation_report,
    print_swr_sweep,
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
    p.add_argument("--p50", type=float, default=None,
                   help="Accumulation P50 for joint probability (defaults to --portfolio)")
    p.add_argument("--tail", type=float, default=None,
                   help="Accumulation tail portfolio for joint probability")
    p.add_argument("--tail-pct", type=int, default=5,
                   help="Tail percentile label (default: 5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-clear", action="store_true",
                   help="Do not clear terminal before output")
    return p.parse_args()


def _income_formula_portfolio(config: Config) -> tuple[float, float]:
    """Return (required_portfolio, target_monthly_nom) using the deterministic income formula."""
    infl_factor = (1.0 + config.expected_inflation_percent / 100.0) ** config.years
    target_monthly_nom = config.target_monthly_retirement_income * infl_factor
    target_annual_nom = target_monthly_nom * 12.0

    sa = config.mc_decumulation_stock_pct
    bonds_pct = 1.0 - sa
    swr = config.withdrawal_percentage_stock / 100.0
    yield_net = config.yield_bonds / 100.0 * (1.0 - config.tax_rate_bond_italy)
    net_rate = sa * swr + bonds_pct * yield_net
    required_portfolio = target_annual_nom / net_rate if net_rate > 0 else float("inf")
    return required_portfolio, target_monthly_nom


def _find_min_portfolio_mc(
    config: Config,
    target_monthly_nom: float,
    target_pct: float,
    seed: int,
    n_simulations: int = 10_000,
    tolerance: float = 1_000.0,
) -> float:
    """Binary search for the minimum starting portfolio that achieves target_pct survival rate.

    Survival rate is monotonically increasing with starting portfolio, so binary search
    converges reliably regardless of pension timing.
    Uses a reduced n_simulations for speed (10k gives ~±0.5pp accuracy).
    """
    lo, hi = 100_000.0, 10_000_000.0
    for _ in range(40):
        mid = (lo + hi) / 2
        result = simulate_decumulation(
            config, mid, seed=seed, n_simulations=n_simulations,
            withdrawal_strategy="fixed",
            monthly_withdrawal_override=target_monthly_nom,
        )
        if result.survival_rate * 100 >= target_pct:
            hi = mid
        else:
            lo = mid
        if hi - lo < tolerance:
            break
    return hi


def main(
    portfolio_override: float | None = None,
    p50_override: float | None = None,
    tail_override: float | None = None,
) -> "DecumulationSummary | None":
    """Run decumulation analysis.

    When called programmatically from main.py, pass portfolio_override / p50_override /
    tail_override to wire accumulation bridge values directly instead of CLI flags.
    Returns the full report text (captured from stdout) for the AI agent.
    """
    args = _parse_args()
    config = _load_config(args.config)

    if not args.no_clear:
        clear_terminal()

    n = config.mc_decumulation_n_simulations
    seed = args.seed
    target_pct = config.mc_target_success_percent

    print(f"\nDecumulation Monte Carlo  |  {n:,} simulations  |  "
          f"horizon {config.mc_decumulation_years} yr  |  "
          f"target survival {target_pct}%")
    adj = config.mc_decumulation_pension_adjustment_rate
    pension_note = f"  +{adj * 100:.1f}%/yr" if adj > 0 else "  (fixed nominal)"
    print(f"Withdrawal rate: {config.withdrawal_percentage_stock:.2f}% pa  |  "
          f"Stock allocation: {config.mc_decumulation_stock_pct * 100:.0f}%  |  "
          f"Stock return μ: {config.expected_yearly_growth_stocks_percent:.1f}%  |  "
          f"Pension: €{config.mc_decumulation_pension_monthly:,.0f}/mo "
          f"from year {config.mc_decumulation_pension_start_year}{pension_note}")
    print()

    # ── Required portfolio: income formula ───────────────────────────────────
    required_portfolio, target_monthly_nom = _income_formula_portfolio(config)
    print(f"  Target withdrawal : €{target_monthly_nom:,.0f}/mo  "
          f"(€{config.target_monthly_retirement_income:,.0f} today × inflation {config.years:.0f}yr)",
          flush=True)
    print(f"  Income formula    : €{required_portfolio:,.0f}  "
          f"(deterministic: stock SWR + bond yield = target income)")

    # ── Required portfolio: MC minimum for target survival ────────────────────
    print(f"  MC minimum ({target_pct:.0f}% survival, 10k sims)…", end="", flush=True)
    mc_min_portfolio = _find_min_portfolio_mc(
        config, target_monthly_nom, target_pct, seed=seed,
    )
    delta = mc_min_portfolio - required_portfolio
    delta_str = f"{delta:+,.0f}"
    print(f"\r  MC minimum        : €{mc_min_portfolio:,.0f}  "
          f"({delta_str} vs income formula)                    ")

    display_portfolio = required_portfolio
    if args.portfolio is not None:
        display_portfolio = args.portfolio
    print(f"  Survival curves   : €{display_portfolio:,.0f}  (income formula target)")
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

    # ── Joint probability ─────────────────────────────────────────────────────
    # Bridge: programmatic overrides from main.py take precedence over CLI args
    p50 = p50_override if p50_override is not None else (args.p50 if args.p50 is not None else display_portfolio)
    tail = tail_override if tail_override is not None else (args.tail if args.tail is not None else required_portfolio)
    tail_pct = args.tail_pct

    print("Computing joint probability…", flush=True)
    joint = compute_joint_probability(
        config,
        accumulation_p50=p50,
        accumulation_tail=tail,
        tail_pct=tail_pct,
        seed=seed,
        n_simulations=n,
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
        required_portfolio=required_portfolio,
        n_paths=n,
        seed=seed,
    )

    # ── SWR sensitivity sweep ─────────────────────────────────────────────────
    print("Running SWR sensitivity sweep…", flush=True)
    swr_sweep = compute_swr_sweep(config, display_portfolio, seed=seed)

    # ── Decumulation inflation × CAGR grid ────────────────────────────────────
    print("Running decumulation inflation × CAGR grid…", flush=True)
    decum_grid = compute_decum_inflation_cagr_grid(config, display_portfolio, seed=seed)

    # ── Deterministic reference (accumulation plan target) ───────────────────
    deterministic_target = portfolio_override or (args.portfolio if args.portfolio else display_portfolio)

    # ── Report ────────────────────────────────────────────────────────────────
    print_decumulation_report(
        result_fixed=result_fixed,
        result_gk=result_gk,
        required_fixed=required_portfolio,
        required_gk=required_portfolio,
        sorr=sorr,
        joint=joint,
        deterministic_target=deterministic_target,
        target_success_pct=target_pct,
    )
    print_allocation_sweep(sweep)
    print_chained_mc(chained)
    print_swr_sweep(swr_sweep)
    print_decum_inflation_cagr_heatmap(decum_grid)

    from models import DecumulationSummary
    return DecumulationSummary(
        required_portfolio_income=required_portfolio,
        target_monthly_withdrawal=target_monthly_nom,
        n_years=config.mc_decumulation_years,
        survival_fixed_final=result_fixed.survival_by_year[-1] if result_fixed.survival_by_year else result_fixed.survival_rate,
        survival_gk_final=result_gk.survival_by_year[-1] if result_gk.survival_by_year else result_gk.survival_rate,
        joint_survival_chained=chained.joint_survival_rate,
    )


if __name__ == "__main__":
    main()
