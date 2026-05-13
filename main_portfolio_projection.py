import argparse
import contextlib
import io
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
from models import Config
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
    print_summary_q0,
    print_summary_q1,
    print_summary_q2a,
    print_summary_q2b,
    print_summary_q4,
)


def _print_verbose_report(
    config, args,
    future_monthly_income_value, target_annual_net_income,
    projection_result, stock_gain_fraction,
    scenario_1_result, scenario_2_result,
    projected_yield_scenario_1, projected_yield_scenario_2,
    cash_to_bonds_at_horizon_val, estimated_final_cash,
    inflation_rate, stock_expected_yield, bonds_yield, withdrawal_rate,
    stock_std, inflation_std, target_success,
    current_stock_mo, current_bonds_mo, bonds_mo_de, bonds_mo_it,
    mc1, mc2,
    cagr_stock_s1, cagr_stock_s2,
    grids_de, grids_it,
    bond_grid_de, bond_grid_it,
    combined_de, combined_it,
    cagr_income_s1, cagr_income_s2,
    ff_de, ff_it,
    opt_result_1, opt_result_2,
    stock_sweep,
    heatmap_de, heatmap_it,
) -> None:
    # ── Section 1: Assumptions & Target ───────────────────────────────────────
    _print_section(1, "ASSUMPTIONS & TARGET  —  DETERMINISTIC")
    current_portfolio_total = (
        config.current_stock_value + config.current_bonds_value + config.current_cash
    )
    saving_pct = (
        config.annual_net_saving / current_portfolio_total * 100
        if current_portfolio_total > 0 else 0.0
    )
    print(f"  Current portfolio : {current_portfolio_total:,.2f}  "
          f"(stocks {config.current_stock_value:,.0f}"
          f"  |  bonds {config.current_bonds_value:,.0f}"
          f"  |  cash {config.current_cash:,.0f})")
    print(f"  Annual saving     : {config.annual_net_saving:,.2f}/yr"
          f"  =  {saving_pct:.1f}% of current portfolio")
    print()
    print(f"  Target monthly retirement income (today's money) : "
          f"{config.target_monthly_retirement_income:,.2f}/mo")
    print(f"  Inflation-adjusted at horizon ({config.years:.0f}y)          : "
          f"{future_monthly_income_value:,.2f}/mo")
    print(f"  Annual net income inflation-adjusted at horizon  : "
          f"{target_annual_net_income:,.2f}/yr")
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
        monthly_saving=config.annual_net_saving / 12,
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
        config.annual_net_saving / 12,
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


def main() -> None:
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
    args, _ = parser.parse_known_args()

    if not args.no_clear:
        clear_terminal()
    config_path = Path(__file__).with_name("config.toml")

    try:
        with config_path.open("rb") as f:
            config = Config.from_dict(tomllib.load(f))
        config_toml_text = config_path.read_text()
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return
    except KeyError as error:
        print(f"Missing required config key: {error}")
        return
    except (ValueError, tomllib.TOMLDecodeError):
        print("Config values must be valid numbers in valid TOML.")
        return

    # ── All computations ──────────────────────────────────────────────────────
    try:
        inflation_rate = config.expected_inflation_percent / 100
        stock_expected_yield = config.expected_yearly_growth_stocks_percent / 100
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
    except ValueError as error:
        print(error)
        return

    _cash_pct = max(
        0.0,
        1.0 - config.maximum_stock_monthly_investment_pct
        - config.maximum_bonds_monthly_investment_pct,
    )
    _monthly_cash = config.annual_net_saving / 12 * _cash_pct
    cash_to_bonds_at_horizon_val = cash_converted_to_bonds_at_horizon(
        current_cash=config.current_cash,
        cash_rate=config.interest_on_cash_pa / 100,
        years=config.years,
        cash_reserve=config.cash_reserve_at_horizon,
        monthly_contribution=_monthly_cash,
        contribution_growth_rate=config.annual_contribution_growth_percent / 100,
    )
    estimated_final_cash = cash_to_bonds_at_horizon_val + config.cash_reserve_at_horizon

    stock_std = config.mc_stock_return_std_percent / 100
    inflation_std = config.mc_inflation_std_percent / 100
    current_stock_mo = config.annual_net_saving / 12 * config.maximum_stock_monthly_investment_pct
    current_bonds_mo = config.annual_net_saving / 12 * config.maximum_bonds_monthly_investment_pct
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
        step=config.optimize_grid_step, target_success=target_success,
    )
    opt_result_2 = compute_optimal_allocation(
        config=config, scenario=scenario_2_result,
        stock_std=stock_std, inflation_std=inflation_std,
        step=config.optimize_grid_step, target_success=target_success,
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

    # ── Capture verbose report (always, for Claude context) ───────────────────
    verbose_buf = io.StringIO()
    with contextlib.redirect_stdout(verbose_buf):
        _print_verbose_report(
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
            stock_sweep=stock_sweep,
            heatmap_de=heatmap_de, heatmap_it=heatmap_it,
        )
    verbose_report = verbose_buf.getvalue()

    if args.verbose:
        print(verbose_report, end="")

    # ── Print clean summary (always) ───────────────────────────────────
    summary_buf = io.StringIO()
    with contextlib.redirect_stdout(summary_buf):
        print_summary_q1(config, scenario_1_result, scenario_2_result,
                         future_monthly_income_value, target_annual_net_income, )
        print_summary_q0(config, projection_result, scenario_1_result, scenario_2_result,
                         mc1, mc2, years_de, years_it)
        print_summary_q2a(combined_de, combined_it, config, target_success,
                          cagr_stock_s1, cagr_stock_s2)
        print_summary_q2b(grids_de, grids_it, cagr_income_s1, cagr_income_s2,
                          config, target_success)
        print_summary_q4(opt_result_1, opt_result_2, config)
        print()
        print("SENSITIVITY — income success vs each input assumption")
        print()
        print_sensitivity_waterfall(sensitivity_de, sensitivity_it)
        if args.verbose:
            print("\n")
            print("JOINT SENSITIVITY — income success vs Inflation × Stock CAGR")
            print_inflation_cagr_heatmap(heatmap_de, heatmap_it, target_success=target_success)
    clean_summary = summary_buf.getvalue()
    print(clean_summary, end="")

    # ── Optional action plan via Claude ───────────────────────────────────────
    if args.action_plan:
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

    # Bridge values for unified main.py
    tail_pct_int = int(round(100 - config.mc_target_success_percent))
    return (
        mc2.final_portfolio_percentiles.get(50, 0.0),
        mc2.final_portfolio_percentiles.get(tail_pct_int, 0.0),
        cagr_stock_s2,
        tail_pct_int,
        verbose_report,
        clean_summary,
        config_toml_text,
    )


if __name__ == "__main__":
    main()
