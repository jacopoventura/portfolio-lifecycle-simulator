"""One-shot probe: print golden values for fixture regression tests.

Run: python -m tests._fixture_probe
Bake the printed values into test_fixtures.py snapshots.
"""
from models import Config
from monte_carlo import (
    compute_canonical_fi_required_scan,
    compute_chained_mc,
    compute_sorr_scenarios,
    compute_wealth_preservation_under_sorr,
    simulate_decumulation,
)


def _fixture_config() -> Config:
    return Config(
        current_stock_value=100_000.0,
        current_bonds_value=10_000.0,
        current_stock_unrealized_gain=20_000.0,
        current_cash=50_000.0,
        cash_reserve_at_horizon=10_000.0,
        interest_on_cash_pa=2.0,
        annual_net_saving=24_000.0,
        annual_contribution_growth_percent=0.0,
        maximum_stock_monthly_investment_pct=0.5,
        maximum_bonds_monthly_investment_pct=0.3,
        target_monthly_retirement_income=3_300.0,
        years=15.0,
        expected_inflation_percent=2.3,
        expected_yearly_growth_stocks_percent=5.0,
        bonds_percentage=60.0,
        yield_bonds=3.4,
        tax_rate=0.26,
        tax_rate_bond_italy=0.125,
        withdrawal_percentage_stock=3.3,
        basiszins=0.0229,
        freibetrag=1_000.0,
        teilfreistellung=0.7,
        optimize_grid_step=50_000.0,
        mc_n_simulations=500,
        mc_stock_return_std_percent=15.0,
        mc_inflation_std_percent=2.0,
        mc_bond_yield_std_percent=0.1,
        mc_target_success_percent=95,
        mc_decumulation_years=20,
        mc_decumulation_stock_pct=0.40,
        mc_decumulation_n_simulations=300,
        mc_decumulation_pension_start_year=10,
        mc_decumulation_pension_monthly=2_000.0,
        mc_decumulation_pension_adjustment_rate=0.0,
    )


def main() -> None:
    config = _fixture_config()
    seed = 42
    n = 500
    target_monthly = 4_500.0
    portfolio = 1_500_000.0

    print("# === FIXTURE SNAPSHOTS ===")

    r = simulate_decumulation(
        config, initial_portfolio=portfolio, seed=seed, n_simulations=n,
        monthly_withdrawal_override=target_monthly,
    )
    print(f"SIMULATE_DECUM_SURVIVAL_RATE = {r.survival_rate:.6f}")
    print(f"SIMULATE_DECUM_FINAL_YEAR    = {r.survival_by_year[-1]:.6f}")

    sorr = compute_sorr_scenarios(
        config, initial_portfolio=portfolio, seed=seed, n_simulations=n,
        monthly_withdrawal_override=target_monthly,
    )
    print(f"SORR_BASELINE_SURVIVAL       = {sorr.baseline.survival_rate:.6f}")
    print(f"SORR_NUM_STRESS_SCENARIOS    = {len(sorr.stress_scenarios)}")

    chained = compute_chained_mc(
        config=config,
        monthly_stock_contribution=1_000.0,
        monthly_bond_contribution=500.0,
        required_portfolio=1_500_000.0,
        n_paths=n, seed=seed,
    )
    print(f"CHAINED_JOINT_SURVIVAL_RATE  = {chained.joint_survival_rate:.6f}")
    print(f"CHAINED_ACCUM_SUCCESS_FRAC   = {chained.accum_success_frac:.6f}")

    scan = compute_canonical_fi_required_scan(
        config, target_monthly, seed=seed, n_simulations=n,
    )
    print(f"CANONICAL_REQUIRED_WEALTH    = {scan.canonical_required_wealth:.2f}")
    print(f"CANONICAL_OPTIMAL_BONDS_PCT  = {scan.canonical_optimal_bonds_pct:.6f}")
    print(f"CANONICAL_REACHABLE          = {scan.canonical_reachable}")
    print(f"CANONICAL_WORST_STRESS_LABEL = {scan.worst_stress_label!r}")

    surface = compute_wealth_preservation_under_sorr(
        config, initial_wealth=portfolio, target_monthly_nom=target_monthly,
        seed=seed, n_simulations=n,
    )
    base_row = surface.rows[0]
    print(f"WEALTH_SURFACE_NUM_ROWS      = {len(surface.rows)}")
    print(f"WEALTH_SURFACE_BASELINE_P50  = {base_row.best_p_by_ratio[0]:.6f}")
    print(f"WEALTH_SURFACE_BASELINE_P100 = {base_row.best_p_by_ratio[5]:.6f}")
    print(f"WEALTH_SURFACE_BRIDGE_P50    = {surface.bridge_p50:.2f}")


if __name__ == "__main__":
    main()
