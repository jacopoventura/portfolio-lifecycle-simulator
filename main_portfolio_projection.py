import argparse
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
import tomllib

from finance import (
    calculate_future_income_value,
    calculate_projection_yield_for_scenario,
    cash_converted_to_bonds_at_horizon,
    run_projection,
    run_scenario,
    validate_config_inputs,
)
from models import (
    BondStockGrid,
    CombinedGrid,
    Config,
    ContributionCagrGrid,
    FinancialFreedomResult,
    InflationCagrGrid,
    MonteCarloResult,
    OptimalAllocationResult,
    ProjectionResult,
    ProjectionYieldScenarioResult,
    ScenarioResult,
    SensitivityWaterfallResult,
    VerdictInputs,
    YearsToTargetsResult,
)
from monte_carlo import (
    _OPTIMIZE_N_SIMULATIONS,
    compute_bond_stock_grid,
    compute_combined_grid,
    compute_contribution_cagr_grid,
    compute_inflation_cagr_grid,
    compute_optimal_allocation,
    compute_required_cagr_for_success,
    compute_sensitivity_waterfall,
    compute_stock_allocation_sweep,
    compute_years_to_financial_freedom_mc,
    compute_years_to_targets_mc,
    run_monte_carlo_simulation,
)
from reports import (
    _print_section,
    _print_subsection,
    clear_terminal,
    print_combined_q1_analysis,
    print_current_situation_mc,
    print_inflation_cagr_heatmap,
    print_financial_freedom_report,
    print_monte_carlo_report,
    print_optimal_allocation_report,
    print_plan_b_hard,
    print_plan_b_income,
    print_projection_report,
    print_projection_yield_table,
    print_q1a_analysis,
    print_q1b_analysis,
    print_q2_analysis,
    print_scenarios_report,
    print_sensitivity_table,
    print_sensitivity_waterfall,
    print_stock_allocation_sweep,
    print_current_situation_summary,
    print_what_you_need_at_horizon,
    print_how_to_get_there_portfolio,
    print_how_to_get_there_income,
    print_redeploy_current_assets,
    print_takeaway,
)


@dataclass
class ReportContext:
    """Aggregated inputs for verbose / summary / takeaway rendering.
    Owned by main(); built once and passed to renderers."""
    config: Config
    args: argparse.Namespace
    future_monthly_income_value: float
    target_annual_net_income: float
    projection_result: ProjectionResult
    stock_gain_fraction: float
    scenario_1_result: ScenarioResult
    scenario_2_result: ScenarioResult
    projected_yield_scenario_1: ProjectionYieldScenarioResult
    projected_yield_scenario_2: ProjectionYieldScenarioResult
    cash_to_bonds_at_horizon_val: float
    estimated_final_cash: float
    inflation_rate: float
    stock_expected_yield: float
    bonds_yield: float
    withdrawal_rate: float
    stock_std: float
    inflation_std: float
    target_success: float
    current_stock_mo: float
    current_bonds_mo: float
    bonds_mo_de: float
    bonds_mo_it: float
    mc1: MonteCarloResult
    mc2: MonteCarloResult
    cagr_stock_s1: float | None
    cagr_stock_s2: float | None
    grids_de: list[ContributionCagrGrid]
    grids_it: list[ContributionCagrGrid]
    bond_grid_de: BondStockGrid
    bond_grid_it: BondStockGrid
    combined_de: CombinedGrid
    combined_it: CombinedGrid
    cagr_income_s1: float | None
    cagr_income_s2: float | None
    ff_de: FinancialFreedomResult
    ff_it: FinancialFreedomResult
    opt_result_1: OptimalAllocationResult
    opt_result_2: OptimalAllocationResult
    opt_result_1_capped: OptimalAllocationResult
    opt_result_2_capped: OptimalAllocationResult
    sensitivity_de: SensitivityWaterfallResult
    sensitivity_it: SensitivityWaterfallResult
    years_de: YearsToTargetsResult
    years_it: YearsToTargetsResult
    stock_sweep: list[dict[str, float]] | None
    heatmap_de: InflationCagrGrid
    heatmap_it: InflationCagrGrid


def _print_verbose_report(ctx: ReportContext) -> None:
    config = ctx.config
    args = ctx.args
    future_monthly_income_value = ctx.future_monthly_income_value
    target_annual_net_income = ctx.target_annual_net_income
    projection_result = ctx.projection_result
    stock_gain_fraction = ctx.stock_gain_fraction
    scenario_1_result = ctx.scenario_1_result
    scenario_2_result = ctx.scenario_2_result
    projected_yield_scenario_1 = ctx.projected_yield_scenario_1
    projected_yield_scenario_2 = ctx.projected_yield_scenario_2
    cash_to_bonds_at_horizon_val = ctx.cash_to_bonds_at_horizon_val
    estimated_final_cash = ctx.estimated_final_cash
    inflation_rate = ctx.inflation_rate
    stock_expected_yield = ctx.stock_expected_yield
    bonds_yield = ctx.bonds_yield
    withdrawal_rate = ctx.withdrawal_rate
    stock_std = ctx.stock_std
    inflation_std = ctx.inflation_std
    target_success = ctx.target_success
    current_stock_mo = ctx.current_stock_mo
    current_bonds_mo = ctx.current_bonds_mo
    mc1 = ctx.mc1
    mc2 = ctx.mc2
    cagr_stock_s1 = ctx.cagr_stock_s1
    cagr_stock_s2 = ctx.cagr_stock_s2
    grids_de = ctx.grids_de
    grids_it = ctx.grids_it
    bond_grid_de = ctx.bond_grid_de
    bond_grid_it = ctx.bond_grid_it
    combined_de = ctx.combined_de
    combined_it = ctx.combined_it
    cagr_income_s1 = ctx.cagr_income_s1
    cagr_income_s2 = ctx.cagr_income_s2
    ff_de = ctx.ff_de
    ff_it = ctx.ff_it
    opt_result_1 = ctx.opt_result_1
    opt_result_2 = ctx.opt_result_2
    stock_sweep = ctx.stock_sweep
    heatmap_de = ctx.heatmap_de
    heatmap_it = ctx.heatmap_it
    # ── Section 1: Assumptions & Target ───────────────────────────────────────
    _print_section(1, "ASSUMPTIONS & TARGET  —  DETERMINISTIC")
    current_portfolio_total = (
        config.current_stock_value + config.current_bonds_value + config.current_cash
    )
    saving_pct = (
        config.annual_net_saving / current_portfolio_total * 100
        if current_portfolio_total > 0 else 0.0
    )
    print(f"  Current portfolio : {current_portfolio_total:,.0f}  "
          f"(stocks {config.current_stock_value:,.0f}"
          f"  |  bonds {config.current_bonds_value:,.0f}"
          f"  |  cash {config.current_cash:,.0f})")
    print(f"  Annual saving     : {config.annual_net_saving:,.0f}/yr"
          f"  =  {saving_pct:.1f}% of current portfolio")
    print()
    print(f"  Target monthly retirement income (today's money) : "
          f"{config.target_monthly_retirement_income:,.0f}/mo")
    print(f"  Inflation-adjusted at horizon ({config.years:.0f}y)          : "
          f"{future_monthly_income_value:,.0f}/mo")
    print(f"  Annual net income inflation-adjusted at horizon  : "
          f"{target_annual_net_income:,.0f}/yr")
    print()
    print_scenarios_report(
        s1=scenario_1_result, s2=scenario_2_result,
        current_bonds_value=config.current_bonds_value,
        current_cash=config.current_cash,
        cash_interest_rate_pa=config.interest_on_cash_pa,
        estimated_final_cash=estimated_final_cash,
        cash_to_bonds_at_horizon=cash_to_bonds_at_horizon_val,
        cash_reserve_at_horizon=config.cash_reserve_at_horizon,
        annual_net_saving=config.annual_net_saving,
        verbose=True,
    )
    print_sensitivity_table(
        config=config,
        base_inflation_rate=inflation_rate,
        base_bonds_yield=bonds_yield,
        base_withdrawal_rate=withdrawal_rate,
        stock_expected_yield=stock_expected_yield,
        stock_gain_fraction=stock_gain_fraction,
    )

    # ── Section 2: Monte Carlo Analysis ───────────────────────────────────────
    _print_section(
        2,
        f"MONTE CARLO ANALYSIS  —  {config.mc_target_success_percent}% success rate",
    )
    print_monte_carlo_report(mc1, mc2, current_stock_mo, current_bonds_mo, config, verbose=True)

    _print_subsection(
        f"Q1a — Stock target: P(stocks ≥ required) ≥ {config.mc_target_success_percent}%"
    )
    print_plan_b_hard(cagr_stock_s1, cagr_stock_s2, config)
    print_q1a_analysis(grids_de, grids_it, config, target_success=target_success)

    _print_subsection(
        f"Q1b — Bond target: P(bonds ≥ required) ≥ {config.mc_target_success_percent}%"
    )
    print_q1b_analysis(
        bond_grid_de, bond_grid_it,
        current_stock_mo, current_bonds_mo,
        monthly_saving=config.monthly_saving,
        target_success=target_success,
    )

    _print_subsection("Q1 Combined — best achievable P(stocks ≥ req AND bonds ≥ req)")
    print_combined_q1_analysis(combined_de, combined_it, config, target_success=target_success)

    _print_subsection(
        f"Q2 — Income target: P(portfolio income ≥ target) ≥ {config.mc_target_success_percent}%"
    )
    print_plan_b_income(cagr_income_s1, cagr_income_s2, config)
    print_q2_analysis(
        grids_de, grids_it,
        config.monthly_saving,
        target_success=target_success,
        verbose=True,
    )

    if stock_sweep:
        print_stock_allocation_sweep(stock_sweep=stock_sweep, config=config)

    _print_subsection("Q2d — Joint sensitivity: income success vs Inflation × CAGR")
    print_inflation_cagr_heatmap(heatmap_de, heatmap_it, target_success=target_success)

    # ── Section 3: Current Situation & Recommendation ─────────────────────────
    _print_section(3, "CURRENT SITUATION & RECOMMENDATION")
    #print_projection_report(
    #    result=projection_result,
    #    scenario_de=scenario_1_result,
    #    scenario_it=scenario_2_result,
    #)
    print_financial_freedom_report(ff_de, ff_it, config)
    print_projection_yield_table(
        projected_yield_scenario_1,
        projected_yield_scenario_2,
        projection_result.final_stock_value + projection_result.final_bonds_value,
    )
    print_current_situation_mc(mc1, mc2, current_stock_mo, current_bonds_mo, config)
    print_optimal_allocation_report(
        opt_result_1, opt_result_2, config,
        n_simulations_per_point=_OPTIMIZE_N_SIMULATIONS,
        verbose=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portfolio projection and Monte Carlo analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main_portfolio_projection.py               # clean Q0-Q4 summary\n"
            "  python main_portfolio_projection.py --verbose     # full grids + tables + summary\n"
            "  python main_portfolio_projection.py --action-plan # summary + Claude AI action plan\n"
            "\n"
            "config: edit config.toml to set your portfolio values and assumptions\n"
            "api:    set ANTHROPIC_API_KEY in .env to use --action-plan"
        ),
    )
    parser.add_argument("--verbose", action="store_true",
                        help="also print full grids, sensitivity tables and MC scans")
    parser.add_argument("--action-plan", action="store_true",
                        help="call Claude Haiku to generate 5 prioritised action bullets (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--no-clear", action="store_true",
                        help="suppress terminal clear (used when called from main.py)")
    parser.add_argument("--cagr", default=None,
                        help="override stock CAGR for projection / MC planning "
                             "(accepts 4, 4.0, 0.04, '4%%'); per-run only")
    args, _ = parser.parse_known_args()
    return args


def _load_config() -> tuple[Config, str] | None:
    """Load config.toml. On error, print a user-facing message and return None."""
    config_path = Path(__file__).with_name("config.toml")
    try:
        with config_path.open("rb") as f:
            config = Config.from_dict(tomllib.load(f))
        return config, config_path.read_text()
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
    except KeyError as error:
        print(f"Missing required config key: {error}")
    except (ValueError, tomllib.TOMLDecodeError):
        print("Config values must be valid numbers in valid TOML.")
    return None


def _compute_all(config: Config, args: argparse.Namespace) -> ReportContext:
    """Run every projection / Monte Carlo / optimization needed for the report.
    Raises ValueError on validation failures (caught by main())."""
    inflation_rate = config.expected_inflation_percent / 100
    # Use planning_cagr (override-aware) for all projection / MC math; the raw
    # config baseline is kept for the realism check that surfaces the override.
    stock_expected_yield = config.planning_cagr
    bonds_yield = config.yield_bonds / 100
    withdrawal_rate = config.withdrawal_percentage_stock / 100
    target_success = config.mc_target_success_percent / 100

    validate_config_inputs(
        config=config,
        stock_expected_yield=stock_expected_yield,
        bonds_yield=bonds_yield,
        withdrawal_rate=withdrawal_rate,
    )

    future_monthly_income_value = calculate_future_income_value(
        income_value=config.target_monthly_retirement_income,
        years=config.years,
        inflation_rate=inflation_rate,
    )
    target_annual_net_income = future_monthly_income_value * 12
    projection_result = run_projection(config, stock_expected_yield)

    # At-retirement gain fraction: (final_stock - cost_basis) / final_stock on day 1
    # of decumulation. Captures the tax rate paid on year-1 withdrawals only. As decum
    # progresses, cost basis erodes and this fraction rises toward 1.0; the decumulation
    # MC uses a 0.65 horizon-averaged proxy (monte_carlo.py:1497, 1628, 1706, 2008).
    # Keep this at the at-retirement value here: scenario_yield breakdowns and contribution
    # sizing reason about day-1 economics, not lifetime averages.
    gain_stock = max(
        0.0, projection_result.final_stock_value - projection_result.cost_basis_stock
    )
    stock_gain_fraction = (
        gain_stock / projection_result.final_stock_value
        if projection_result.final_stock_value > 0 else 0.0
    )

    scenario_1_result = run_scenario(
        name="SCENARIO 1 (DE)",
        config=config,
        future_monthly_income_value=future_monthly_income_value,
        target_annual_net_income=target_annual_net_income,
        stock_tax_rate=config.tax_rate,
        bond_tax_rate=config.tax_rate,
        stock_expected_yield=stock_expected_yield,
        bonds_yield=bonds_yield,
        withdrawal_rate=withdrawal_rate,
        stock_gain_fraction=stock_gain_fraction,
        stock_teilfreistellung=config.teilfreistellung,
        annual_contribution_growth=config.annual_contribution_growth_percent / 100,
    )
    scenario_2_result = run_scenario(
        name="SCENARIO 2 (IT)",
        config=config,
        future_monthly_income_value=future_monthly_income_value,
        target_annual_net_income=target_annual_net_income,
        stock_tax_rate=config.tax_rate,
        bond_tax_rate=config.tax_rate_bond_italy,
        stock_expected_yield=stock_expected_yield,
        bonds_yield=bonds_yield,
        withdrawal_rate=withdrawal_rate,
        stock_gain_fraction=stock_gain_fraction,
        stock_teilfreistellung=1.0,
        annual_contribution_growth=config.annual_contribution_growth_percent / 100,
    )
    projected_yield_scenario_1 = calculate_projection_yield_for_scenario(
        name="SCENARIO 1 (DE)",
        projection=projection_result,
        withdrawal_rate=withdrawal_rate,
        bonds_yield=bonds_yield,
        stock_tax_rate=config.tax_rate,
        bond_tax_rate=config.tax_rate,
        stock_teilfreistellung=config.teilfreistellung,
    )
    projected_yield_scenario_2 = calculate_projection_yield_for_scenario(
        name="SCENARIO 2 (IT)",
        projection=projection_result,
        withdrawal_rate=withdrawal_rate,
        bonds_yield=bonds_yield,
        stock_tax_rate=config.tax_rate,
        bond_tax_rate=config.tax_rate_bond_italy,
        stock_teilfreistellung=1.0,
    )

    cash_pct = max(
        0.0,
        1.0 - config.maximum_stock_monthly_investment_pct
        - config.maximum_bonds_monthly_investment_pct,
    )
    monthly_cash = config.monthly_saving * cash_pct
    cash_to_bonds_at_horizon_val = cash_converted_to_bonds_at_horizon(
        current_cash=config.current_cash,
        cash_rate=config.interest_on_cash_pa / 100,
        years=config.years,
        cash_reserve=config.cash_reserve_at_horizon,
        monthly_contribution=monthly_cash,
        contribution_growth_rate=config.annual_contribution_growth_percent / 100,
    )
    estimated_final_cash = cash_to_bonds_at_horizon_val + config.cash_reserve_at_horizon

    stock_std = config.mc_stock_return_std_percent / 100
    inflation_std = config.mc_inflation_std_percent / 100
    current_stock_mo = config.monthly_saving * config.maximum_stock_monthly_investment_pct
    current_bonds_mo = config.monthly_saving * config.maximum_bonds_monthly_investment_pct
    bonds_mo_de = scenario_1_result.required_monthly_bonds_contribution
    bonds_mo_it = scenario_2_result.required_monthly_bonds_contribution

    mc1 = run_monte_carlo_simulation(
        config=config, scenario=scenario_1_result,
        n_simulations=config.mc_n_simulations,
        stock_std=stock_std, inflation_std=inflation_std,
        monthly_stock_contribution_override=current_stock_mo,
        monthly_bonds_contribution_override=current_bonds_mo,
    )
    mc2 = run_monte_carlo_simulation(
        config=config, scenario=scenario_2_result,
        n_simulations=config.mc_n_simulations,
        stock_std=stock_std, inflation_std=inflation_std,
        monthly_stock_contribution_override=current_stock_mo,
        monthly_bonds_contribution_override=current_bonds_mo,
    )

    cagr_stock_s1 = compute_required_cagr_for_success(
        config=config, scenario=scenario_1_result,
        stock_std=stock_std, inflation_std=inflation_std,
        fixed_monthly_bonds=bonds_mo_de,
        monthly_stock_contribution=current_stock_mo,
        target_success=target_success,
    )
    cagr_stock_s2 = compute_required_cagr_for_success(
        config=config, scenario=scenario_2_result,
        stock_std=stock_std, inflation_std=inflation_std,
        fixed_monthly_bonds=bonds_mo_it,
        monthly_stock_contribution=current_stock_mo,
        target_success=target_success,
    )

    grids_de = compute_contribution_cagr_grid(
        config=config, scenario=scenario_1_result,
        stock_std=stock_std, inflation_std=inflation_std,
    )
    grids_it = compute_contribution_cagr_grid(
        config=config, scenario=scenario_2_result,
        stock_std=stock_std, inflation_std=inflation_std,
    )

    bond_grid_de = compute_bond_stock_grid(
        config=config, scenario=scenario_1_result,
        stock_std=stock_std, inflation_std=inflation_std,
    )
    bond_grid_it = compute_bond_stock_grid(
        config=config, scenario=scenario_2_result,
        stock_std=stock_std, inflation_std=inflation_std,
    )

    combined_de = compute_combined_grid(
        config=config, scenario=scenario_1_result,
        stock_std=stock_std, inflation_std=inflation_std,
    )
    combined_it = compute_combined_grid(
        config=config, scenario=scenario_2_result,
        stock_std=stock_std, inflation_std=inflation_std,
    )

    cagr_income_s1 = compute_required_cagr_for_success(
        config=config, scenario=scenario_1_result,
        stock_std=stock_std, inflation_std=inflation_std,
        fixed_monthly_bonds=bonds_mo_de,
        monthly_stock_contribution=current_stock_mo,
        target_success=target_success, use_income_metric=True,
    )
    cagr_income_s2 = compute_required_cagr_for_success(
        config=config, scenario=scenario_2_result,
        stock_std=stock_std, inflation_std=inflation_std,
        fixed_monthly_bonds=bonds_mo_it,
        monthly_stock_contribution=current_stock_mo,
        target_success=target_success, use_income_metric=True,
    )

    ff_de = compute_years_to_financial_freedom_mc(
        config=config,
        withdrawal_rate=withdrawal_rate,
        stock_tax_rate=config.tax_rate,
        bond_tax_rate=config.tax_rate,
        stock_teilfreistellung=config.teilfreistellung,
        stock_std=stock_std, inflation_std=inflation_std,
        monthly_stock=current_stock_mo, monthly_bonds=current_bonds_mo,
        target_success=target_success,
    )
    ff_it = compute_years_to_financial_freedom_mc(
        config=config,
        withdrawal_rate=withdrawal_rate,
        stock_tax_rate=config.tax_rate,
        bond_tax_rate=config.tax_rate_bond_italy,
        stock_teilfreistellung=1.0,
        stock_std=stock_std, inflation_std=inflation_std,
        monthly_stock=current_stock_mo, monthly_bonds=current_bonds_mo,
        target_success=target_success,
    )

    years_de = compute_years_to_targets_mc(
        config=config, scenario=scenario_1_result,
        monthly_stock=current_stock_mo, monthly_bonds=current_bonds_mo,
        stock_std=stock_std, inflation_std=inflation_std,
        target_success=target_success,
    )
    years_it = compute_years_to_targets_mc(
        config=config, scenario=scenario_2_result,
        monthly_stock=current_stock_mo, monthly_bonds=current_bonds_mo,
        stock_std=stock_std, inflation_std=inflation_std,
        target_success=target_success,
    )

    sensitivity_de = compute_sensitivity_waterfall(
        config=config, scenario=scenario_1_result,
        stock_std=stock_std, inflation_std=inflation_std,
        current_stock_mo=current_stock_mo, current_bonds_mo=current_bonds_mo,
    )
    sensitivity_it = compute_sensitivity_waterfall(
        config=config, scenario=scenario_2_result,
        stock_std=stock_std, inflation_std=inflation_std,
        current_stock_mo=current_stock_mo, current_bonds_mo=current_bonds_mo,
    )

    opt_result_1 = compute_optimal_allocation(
        config=config, scenario=scenario_1_result,
        stock_std=stock_std, inflation_std=inflation_std,
        current_stocks_mo=current_stock_mo, current_bonds_mo=current_bonds_mo,
        step=config.optimize_grid_step, target_success=target_success,
    )
    opt_result_2 = compute_optimal_allocation(
        config=config, scenario=scenario_2_result,
        stock_std=stock_std, inflation_std=inflation_std,
        current_stocks_mo=current_stock_mo, current_bonds_mo=current_bonds_mo,
        step=config.optimize_grid_step, target_success=target_success,
    )

    # Same optimization, but constrained to stocks/mo + bonds/mo ≤ full monthly saving
    # (cash slice = residual = monthly_saving − stocks − bonds, can be 0).
    opt_result_1_capped = compute_optimal_allocation(
        config=config, scenario=scenario_1_result,
        stock_std=stock_std, inflation_std=inflation_std,
        current_stocks_mo=current_stock_mo, current_bonds_mo=current_bonds_mo,
        step=config.optimize_grid_step, target_success=target_success,
        budget_cap=config.monthly_saving,
    )
    opt_result_2_capped = compute_optimal_allocation(
        config=config, scenario=scenario_2_result,
        stock_std=stock_std, inflation_std=inflation_std,
        current_stocks_mo=current_stock_mo, current_bonds_mo=current_bonds_mo,
        step=config.optimize_grid_step, target_success=target_success,
        budget_cap=config.monthly_saving,
    )

    heatmap_de = compute_inflation_cagr_grid(
        config=config, scenario=scenario_1_result,
        stock_std=stock_std, inflation_std=inflation_std,
        monthly_stocks=current_stock_mo, monthly_bonds=current_bonds_mo,
    )
    heatmap_it = compute_inflation_cagr_grid(
        config=config, scenario=scenario_2_result,
        stock_std=stock_std, inflation_std=inflation_std,
        monthly_stocks=current_stock_mo, monthly_bonds=current_bonds_mo,
    )

    stock_sweep = None
    if args.verbose:
        stock_sweep = compute_stock_allocation_sweep(
            config=config,
            scenario_1=scenario_1_result, scenario_2=scenario_2_result,
            stock_std=stock_std, inflation_std=inflation_std,
            current_stock_mo=current_stock_mo,
            current_bonds_mo_1=bonds_mo_de, current_bonds_mo_2=bonds_mo_it,
        )

    return ReportContext(
        config=config, args=args,
        future_monthly_income_value=future_monthly_income_value,
        target_annual_net_income=target_annual_net_income,
        projection_result=projection_result,
        stock_gain_fraction=stock_gain_fraction,
        scenario_1_result=scenario_1_result,
        scenario_2_result=scenario_2_result,
        projected_yield_scenario_1=projected_yield_scenario_1,
        projected_yield_scenario_2=projected_yield_scenario_2,
        cash_to_bonds_at_horizon_val=cash_to_bonds_at_horizon_val,
        estimated_final_cash=estimated_final_cash,
        inflation_rate=inflation_rate,
        stock_expected_yield=stock_expected_yield,
        bonds_yield=bonds_yield,
        withdrawal_rate=withdrawal_rate,
        stock_std=stock_std, inflation_std=inflation_std,
        target_success=target_success,
        current_stock_mo=current_stock_mo, current_bonds_mo=current_bonds_mo,
        bonds_mo_de=bonds_mo_de, bonds_mo_it=bonds_mo_it,
        mc1=mc1, mc2=mc2,
        cagr_stock_s1=cagr_stock_s1, cagr_stock_s2=cagr_stock_s2,
        grids_de=grids_de, grids_it=grids_it,
        bond_grid_de=bond_grid_de, bond_grid_it=bond_grid_it,
        combined_de=combined_de, combined_it=combined_it,
        cagr_income_s1=cagr_income_s1, cagr_income_s2=cagr_income_s2,
        ff_de=ff_de, ff_it=ff_it,
        opt_result_1=opt_result_1, opt_result_2=opt_result_2,
        opt_result_1_capped=opt_result_1_capped,
        opt_result_2_capped=opt_result_2_capped,
        sensitivity_de=sensitivity_de, sensitivity_it=sensitivity_it,
        years_de=years_de, years_it=years_it,
        stock_sweep=stock_sweep,
        heatmap_de=heatmap_de, heatmap_it=heatmap_it,
    )


def _capture_stdout(render_fn) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        render_fn()
    return buf.getvalue()


def _render_all(ctx: ReportContext) -> tuple[str, str, str]:
    """Build the three string outputs: verbose report, clean summary, compact takeaway."""
    verbose_report = _capture_stdout(lambda: _print_verbose_report(ctx))

    def _render_summary() -> None:
        print_what_you_need_at_horizon(
            ctx.config, ctx.scenario_1_result, ctx.scenario_2_result,
            ctx.future_monthly_income_value, ctx.target_annual_net_income,
        )
        print_current_situation_summary(
            ctx.config, ctx.projection_result, ctx.scenario_1_result, ctx.scenario_2_result,
            ctx.mc1, ctx.mc2, ctx.years_de, ctx.years_it,
        )
        print_how_to_get_there_portfolio(
            ctx.combined_de, ctx.combined_it, ctx.config, ctx.target_success,
            ctx.cagr_stock_s1, ctx.cagr_stock_s2,
        )
        print_how_to_get_there_income(
            ctx.grids_de, ctx.grids_it, ctx.cagr_income_s1, ctx.cagr_income_s2,
            ctx.config, ctx.target_success,
        )
        print_redeploy_current_assets(
            ctx.opt_result_1, ctx.opt_result_2, ctx.config,
            opt_result_1_capped=ctx.opt_result_1_capped,
            opt_result_2_capped=ctx.opt_result_2_capped,
        )
        print()
        print("SENSITIVITY — income success vs each input assumption")
        print()
        print_sensitivity_waterfall(ctx.sensitivity_de, ctx.sensitivity_it)
        if ctx.args.verbose:
            print("\n")
            print("JOINT SENSITIVITY — income success vs Inflation × Stock CAGR")
            print_inflation_cagr_heatmap(ctx.heatmap_de, ctx.heatmap_it, target_success=ctx.target_success)

    def _render_takeaway() -> None:
        print_takeaway(
            config=ctx.config,
            scenario_1=ctx.scenario_1_result, scenario_2=ctx.scenario_2_result,
            mc1=ctx.mc1, mc2=ctx.mc2,
            combined_de=ctx.combined_de, combined_it=ctx.combined_it,
            grids_de=ctx.grids_de, grids_it=ctx.grids_it,
            opt_result_1=ctx.opt_result_1, opt_result_2=ctx.opt_result_2,
            opt_result_1_capped=ctx.opt_result_1_capped,
            opt_result_2_capped=ctx.opt_result_2_capped,
            sensitivity_de=ctx.sensitivity_de, sensitivity_it=ctx.sensitivity_it,
            target_success=ctx.target_success,
            future_monthly_income=ctx.future_monthly_income_value,
            target_annual_net_income=ctx.target_annual_net_income,
            invested_stock_amount=(
                ctx.projection_result.cost_basis_stock
                + ctx.config.current_stock_unrealized_gain
            ),
            years_de=ctx.years_de, years_it=ctx.years_it,
            ff_de=ctx.ff_de, ff_it=ctx.ff_it,
        )

    return verbose_report, _capture_stdout(_render_summary), _capture_stdout(_render_takeaway)


def _print_action_plan(config_toml_text: str, verbose_report: str, clean_summary: str) -> None:
    try:
        from agent import generate_action_plan
        print("\n\n")
        print(f"{'  ACTION PLAN':^70}")
        print()
        print("  Generating via Claude Sonnet — reading full report + summary...")
        print()
        plan = generate_action_plan(config_toml_text, verbose_report, clean_summary)
        print(plan)
        print()
    except ImportError:
        print("\n  Install the 'anthropic' package to use --action-plan.")
    except Exception as error:
        print(f"\n  Action plan failed: {error}")


def main() -> tuple[float, float, float | None, int, float, str, str, str, VerdictInputs] | None:
    args = _parse_args()
    if not args.no_clear:
        clear_terminal()

    loaded = _load_config()
    if loaded is None:
        return None
    config, config_toml_text = loaded

    # Apply CLI --cagr override (decimal or percent); no-op when absent
    config.planning_cagr_override = Config.parse_planning_cagr(args.cagr)

    try:
        ctx = _compute_all(config, args)
    except ValueError as error:
        print(error)
        return None

    verbose_report, clean_summary, takeaway = _render_all(ctx)
    print(clean_summary if args.verbose else takeaway, end="")

    if args.action_plan:
        _print_action_plan(config_toml_text, verbose_report, clean_summary)

    tail_pct_int = int(round(100 - config.mc_target_success_percent))
    verdict_inputs = VerdictInputs(
        config=ctx.config,
        mc_it=ctx.mc2,
        grids_it=ctx.grids_it,
        opt_result_it=ctx.opt_result_2,
        target_success=ctx.target_success,
    )
    return (
        ctx.mc2.final_portfolio_percentiles.get(50, 0.0),
        ctx.mc2.final_portfolio_percentiles.get(tail_pct_int, 0.0),
        ctx.cagr_stock_s2,
        tail_pct_int,
        ctx.scenario_2_result.required_total_portfolio,
        verbose_report,
        clean_summary,
        config_toml_text,
        verdict_inputs,
    )


if __name__ == "__main__":
    main()
