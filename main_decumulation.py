"""
Standalone decumulation analysis — focused real-€ preservation.

Answers three questions, anchored on the IT required portfolio (income-formula
yield bar from pillar 1):

  Q1. At the IT required portfolio, what is P(real final ≥ preserve_ratio · W)?
      (Reports both IT required and IT projected anchors.)
  Q2. What monthly contribution would grow the deterministic projection up
      to the preservation-required wealth?  (Accumulation lever.)
  Q3. What is the preservation-required wealth at the best allocation?

Plus a years-to-retirement tradeoff table and §4.14 survival matrix inside the
Takeaway (always shown). In --verbose mode: full Q1/Q2/Q3 tables + §4.10
inflation × CAGR grid + §4.11 SORR preservation matrix + §4.12 price-of-
constraint + §4.13 two-bucket comparison + §4.15 crash-timing sensitivity.

Usage:
    python main_decumulation.py
    python main_decumulation.py --verbose
    python main_decumulation.py --portfolio 1864445
    python main_decumulation.py --cagr 4
"""
import argparse
import tomllib

from models import Config, DecumulationSummary, resolve_config_path
from monte_carlo import (
    compute_canonical_fi_required_scan,
    compute_crash_timing_matrix,
    compute_decum_inflation_cagr_grid,
    compute_focused_preservation_table,
    compute_price_of_constraint_matrix,
    compute_survival_matrix,
    compute_two_bucket_required_wealth,
    compute_wealth_preservation_under_sorr,
    compute_years_to_retirement_tradeoff,
)
from reports.formatters import clear_terminal
from reports.section4 import (
    print_crash_timing_matrix,
    print_decum_inflation_cagr_heatmap,
    print_focused_preservation_table,
    print_focused_takeaway,
    print_price_of_constraint_matrix,
    print_survival_matrix,
    print_two_bucket_comparison,
    print_wealth_preservation_under_sorr,
)


def _load_config(path: str | None = None) -> Config:
    with open(path or resolve_config_path(), "rb") as f:
        raw = tomllib.load(f)
    return Config.from_dict(raw)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo decumulation — focused preservation")
    p.add_argument("--config", default=None, help="Path to config file (default: FIRE_CONFIG env var, else config.toml)")
    p.add_argument("--portfolio", type=float, default=None,
                   help="Starting wealth for §4.10/§4.11 supporting context (defaults to IT required portfolio)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-clear", action="store_true",
                   help="Do not clear terminal before output")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print full report (header, Q1/Q2/Q3 tables, §4.10, §4.11). "
                        "Without --verbose only the Takeaway is shown.")
    p.add_argument("--cagr", default=None,
                   help="override stock CAGR for the decum MC "
                        "(accepts 4, 4.0, 0.04, '4%%'); per-run only")
    return p.parse_known_args()[0]


def main(
    *,
    config_path: str | None = None,
    portfolio_override: float | None = None,
    seed: int = 42,
    no_clear: bool = False,
    verbose: bool = False,
    cagr: str | None = None,
) -> DecumulationSummary | None:
    """Run focused decumulation analysis. Prints the report and returns a summary.

    portfolio_override: starting wealth for §4.10/§4.11; defaults to the IT required portfolio.
    verbose: if False (default), print the Takeaway (years-to-retirement
        tradeoff + §4.14 survival matrix). If True, also print the Q1/Q2/Q3
        tables, §4.10-§4.13, and §4.15 crash-timing sensitivity.
    """
    config = _load_config(config_path)
    config.planning_cagr_override = Config.parse_planning_cagr(cagr)

    if not no_clear:
        clear_terminal()

    n = config.mc_decumulation_n_simulations

    if verbose:
        print(f"\nDecumulation — focused real-€ preservation  |  "
              f"{n:,} sims  |  horizon {config.mc_decumulation_years} yr")
        adj = config.mc_decumulation_pension_adjustment_rate
        pension_note = f"  +{adj * 100:.1f}%/yr" if adj > 0 else "  (fixed nominal)"
        print(f"Withdrawal rate: {config.withdrawal_percentage_stock:.2f}% pa  |  "
              f"Stock allocation: {config.mc_decumulation_stock_pct * 100:.0f}%  |  "
              f"Stock return μ: {config.planning_cagr * 100:.1f}%  |  "
              f"Pension: €{config.mc_decumulation_pension_monthly:,.0f}/mo "
              f"from year {config.mc_decumulation_pension_start_year}{pension_note}")
        print()

    # Target withdrawal at retirement (today's € inflated to retirement day)
    infl_factor = (1.0 + config.expected_inflation_percent / 100.0) ** config.years
    target_monthly_nom = config.target_monthly_retirement_income * infl_factor
    if verbose:
        print(f"  Target withdrawal : €{target_monthly_nom:,.0f}/mo  "
              f"(€{config.target_monthly_retirement_income:,.0f} today × inflation {config.years:.0f}yr)",
              flush=True)
        print()

    # ── Focused preservation table — always computed (Takeaway needs it) ─────
    if verbose:
        print("Computing focused preservation table…", flush=True)
    else:
        print("Computing decumulation…", flush=True)
    focused = compute_focused_preservation_table(
        config, target_monthly_nom,
        confidence=config.fi_preservation_confidence_baseline,
        seed=seed,
        n_simulations=min(n, 5_000),
    )
    if verbose:
        print_focused_preservation_table(focused)

    # ── Years-to-retirement tradeoff — printed inside the Takeaway below ─────
    if verbose:
        print("Computing years-to-retirement tradeoff…", flush=True)
    tradeoff = compute_years_to_retirement_tradeoff(
        config,
        confidence=config.fi_preservation_confidence_baseline,
        preserve_ratio=config.fi_preservation_ratio,
        seed=seed,
        n_simulations=min(n, 2_000),
    )

    # ── Preservation-ratio sensitivity — always computed (plan-check needs it) ─
    # Dual-confidence (baseline + worst-SORR) required wealth at a handful of
    # preserve_ratio values, so plan-check never states "how much do I need"
    # at a single frozen ratio without showing what that choice costs.
    if verbose:
        print("Computing preservation-ratio sensitivity…", flush=True)
    sensitivity_ratios = sorted({0.10, 0.25, 0.50, config.fi_preservation_ratio})
    sensitivity_scan = compute_canonical_fi_required_scan(
        config, target_monthly_nom,
        preserve_ratios=sensitivity_ratios,
        seed=seed,
        n_simulations=min(n, 4_000),
    )

    # ── Survival matrix wealth points — always built, reused by §4.14/§4.15 ───
    # Sourced entirely from already-computed data (no extra MC): the plan target
    # plus one point per preservation ratio in the sensitivity scan above.
    wealth_points: list[tuple[str, float]] = [
        ("IT required portfolio", focused.it_required_portfolio),
    ]
    for ratio, row in zip(sensitivity_ratios, sensitivity_scan.rows):
        label = f"{int(round(ratio * 100))}% preservation"
        if abs(ratio - config.fi_preservation_ratio) < 1e-6:
            label += " (your target)"
        wealth_points.append((label, row.required_wealth_eur))

    # Supporting context (§4.10/§4.11) — only computed and shown in verbose mode.
    # Anchored on the IT required portfolio (matches focused Q1's primary anchor).
    if verbose:
        display_portfolio = (
            portfolio_override if portfolio_override is not None
            else focused.it_required_portfolio
        )

        print("Computing inflation × CAGR sensitivity…", flush=True)
        decum_grid = compute_decum_inflation_cagr_grid(
            config, display_portfolio, seed=seed, n_simulations=min(n, 5_000),
        )
        print_decum_inflation_cagr_heatmap(decum_grid)

        print("Computing wealth preservation under SORR…", flush=True)
        wealth_surface = compute_wealth_preservation_under_sorr(
            config,
            initial_wealth=display_portfolio,
            target_monthly_nom=target_monthly_nom,
            seed=seed,
            n_simulations=min(n, 10_000),
        )
        print_wealth_preservation_under_sorr(wealth_surface)

        print("Computing price of the SORR safety layer…", flush=True)
        constraint_matrix = compute_price_of_constraint_matrix(
            config,
            target_monthly_nom,
            seed=seed,
            n_simulations=min(n, 5_000),
        )
        print_price_of_constraint_matrix(constraint_matrix)

        print("Computing two-bucket structure comparison…", flush=True)
        two_bucket = compute_two_bucket_required_wealth(
            config,
            target_monthly_nom,
            seed=seed,
            n_simulations=min(n, 5_000),
        )
        print_two_bucket_comparison(two_bucket)
        wealth_points.append(("two-bucket total", two_bucket.required_total_wealth))

    # ── Survival matrix (§4.14) — always computed and printed, not gated by
    # --verbose: this is the direct answer to "what portfolio survives under
    # which conditions," so it belongs in the terse default output too.
    print("Computing survival matrix…", flush=True)
    survival_matrix = compute_survival_matrix(
        config,
        target_monthly_nom,
        wealth_points,
        seed=seed,
        n_simulations=min(n, 5_000),
    )
    print_survival_matrix(survival_matrix)

    if verbose:
        print("Computing crash-timing sensitivity…", flush=True)
        crash_timing = compute_crash_timing_matrix(
            config,
            target_monthly_nom,
            wealth_points,
            seed=seed,
            n_simulations=min(n, 5_000),
        )
        print_crash_timing_matrix(crash_timing)

    # ── Takeaway (always printed; LAST block) ────────────────────────────────
    print_focused_takeaway(focused, tradeoff=tradeoff, config=config)

    # For the final verdict (Lever B): first tradeoff row where Max safe WD
    # in today's € meets the user's target spend in today's €.
    target_today = float(config.target_monthly_retirement_income)
    first_safe = next(
        (r for r in tradeoff.rows if r.max_safe_monthly_today_eur >= target_today),
        None,
    )

    # Max safe WD at the tradeoff row closest to the plan horizon — plan-check's
    # preservation pillar uses it to quantify the spending gap at the plan date.
    plan_years = int(round(config.years))
    plan_row = min(
        tradeoff.rows, key=lambda r: abs(r.years_to_retirement - plan_years),
        default=None,
    ) if tradeoff.rows else None

    # Canonical "required" for main.py's Plan Snapshot = Q3 wealth at the
    # configured preservation ratio.
    return DecumulationSummary(
        required_portfolio_fi_canonical=focused.q3_required_wealth,
        target_monthly_withdrawal=target_monthly_nom,
        n_years=config.mc_decumulation_years,
        tradeoff_first_safe_year_from_now=first_safe.years_to_retirement if first_safe else None,
        tradeoff_first_safe_wd_today=first_safe.max_safe_monthly_today_eur if first_safe else None,
        max_safe_wd_at_plan_horizon_today=plan_row.max_safe_monthly_today_eur if plan_row else None,
        p50_final_real_wealth=focused.it_required_portfolio * focused.p50_real_final_at_it_required,
        p50_final_real_ratio=focused.p50_real_final_at_it_required,
        preservation_sensitivity_ratios=sensitivity_ratios,
        preservation_sensitivity_wealth_eur=[
            row.required_wealth_eur for row in sensitivity_scan.rows
        ],
    )


if __name__ == "__main__":
    _args = _parse_args()
    main(
        config_path=_args.config,
        portfolio_override=_args.portfolio,
        seed=_args.seed,
        no_clear=_args.no_clear,
        verbose=_args.verbose,
        cagr=_args.cagr,
    )
