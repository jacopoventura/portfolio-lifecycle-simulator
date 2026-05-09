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
    compute_joint_probability,
    compute_sorr_scenarios,
    find_required_decumulation_portfolio,
    simulate_decumulation,
)
from reports.formatters import clear_terminal
from reports.section4 import print_allocation_sweep, print_decumulation_report


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


def main(
    portfolio_override: float | None = None,
    p50_override: float | None = None,
    tail_override: float | None = None,
) -> str | None:
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

    _SEARCH_HI = 10_000_000.0

    # ── Binary search: required portfolio for each strategy ──────────────────
    print("Running binary search for required portfolio (fixed withdrawal)…", flush=True)
    required_fixed = find_required_decumulation_portfolio(
        config, seed=seed, n_simulations=n, withdrawal_strategy="fixed",
        lo=200_000.0, hi=_SEARCH_HI, n_iter=30,
    )

    print("Running binary search for required portfolio (Guyton-Klinger)…", flush=True)
    required_gk = find_required_decumulation_portfolio(
        config, seed=seed, n_simulations=n, withdrawal_strategy="guyton_klinger",
        lo=200_000.0, hi=_SEARCH_HI, n_iter=30,
    )

    if required_fixed >= _SEARCH_HI * 0.9999:
        print(f"\n  WARNING: {target_pct}% survival over {config.mc_decumulation_years} yr "
              f"with {config.withdrawal_percentage_stock:.2f}% withdrawal exceeds search ceiling "
              f"(€{_SEARCH_HI:,.0f}). Consider lowering mc_target_success_percent or "
              f"mc_decumulation_years in config.toml.\n", flush=True)

    # ── Choose display portfolio ──────────────────────────────────────────────
    if args.portfolio:
        display_portfolio = args.portfolio
    elif required_fixed < _SEARCH_HI * 0.9999:
        display_portfolio = required_fixed
    else:
        display_portfolio = required_gk  # GK needs less capital — use as fallback display

    # ── Full simulations for display ──────────────────────────────────────────
    print(f"Simulating fixed withdrawal at €{display_portfolio:,.0f}…", flush=True)
    result_fixed = simulate_decumulation(
        config, display_portfolio, seed=seed, n_simulations=n,
        withdrawal_strategy="fixed",
    )

    print(f"Simulating Guyton-Klinger at €{display_portfolio:,.0f}…", flush=True)
    result_gk = simulate_decumulation(
        config, display_portfolio, seed=seed, n_simulations=n,
        withdrawal_strategy="guyton_klinger",
    )

    # ── SORR stress scenarios ─────────────────────────────────────────────────
    print("Computing SORR stress scenarios…", flush=True)
    sorr = compute_sorr_scenarios(
        config, display_portfolio, seed=seed, n_simulations=n,
        withdrawal_strategy="fixed",
    )

    # ── Joint probability ─────────────────────────────────────────────────────
    # Bridge: programmatic overrides from main.py take precedence over CLI args
    p50 = p50_override if p50_override is not None else (args.p50 if args.p50 is not None else display_portfolio)
    tail = tail_override if tail_override is not None else (args.tail if args.tail is not None else required_gk)
    tail_pct = args.tail_pct

    print("Computing joint probability…", flush=True)
    joint = compute_joint_probability(
        config,
        accumulation_p50=p50,
        accumulation_tail=tail,
        tail_pct=tail_pct,
        seed=seed,
        n_simulations=n,
    )

    # ── Allocation sweep ──────────────────────────────────────────────────────
    print("Running allocation sweep (glide path optimisation)…", flush=True)
    sweep = compute_allocation_sweep(config, seed=seed, n_simulations=min(n, 5_000))

    # ── Deterministic reference (accumulation plan target) ───────────────────
    deterministic_target = portfolio_override or (args.portfolio if args.portfolio else display_portfolio)

    # ── Report ────────────────────────────────────────────────────────────────
    print_decumulation_report(
        result_fixed=result_fixed,
        result_gk=result_gk,
        required_fixed=required_fixed,
        required_gk=required_gk,
        sorr=sorr,
        joint=joint,
        deterministic_target=deterministic_target,
        target_success_pct=target_pct,
    )
    print_allocation_sweep(sweep)


if __name__ == "__main__":
    main()
