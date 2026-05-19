import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace as _dc_replace
from typing import NamedTuple

import numpy as np

from finance import cash_converted_to_bonds_at_horizon, calculate_required_monthly_contribution
from models import (
    AllocationGridPoint,
    AllocationSweepPoint,
    AllocationSweepResult,
    BondStockCell,
    BondStockGrid,
    CagrContributionCell,
    ChainedMCResult,
    CombinedGrid,
    DecumInflationCagrCell,
    DecumInflationCagrGrid,
    SWRSweepPoint,
    SWRSweepResult,
    CombinedGridCell,
    Config,
    ContributionCagrGrid,
    DecumulationResult,
    FinancialFreedomResult,
    InflationCagrCell,
    InflationCagrGrid,
    JointProbabilityResult,
    MonteCarloResult,
    OptimalAllocationResult,
    ScenarioResult,
    SensitivityEntry,
    SensitivityWaterfallResult,
    SORRResult,
    YearsToTargetsResult,
)

_SENSITIVITY_N_SIMULATIONS = 10_000
_SUGGESTIONS_N_SIMULATIONS = 15_000
_SUGGESTIONS_SCAN_STEP_MAIN = 1_000.0
_SUGGESTIONS_SCAN_MAX_STOCK = 15_000.0
_OPTIMIZE_N_SIMULATIONS = 5_000
_OPTIMIZE_STEP = 25_000.0
_OPTIMIZE_MIN_CASH_TODAY = 50_000.0
_OPTIMIZE_BONDS_LEVELS = 20  # bonds/mo search resolution per upfront cell
_GRID_N_SIMULATIONS = 15_000
_GRID_CAGR_MIN = 0.03
_GRID_CAGR_MAX = 0.09
_GRID_CAGR_STEP = 0.01
_GRID_CONTRIBUTION_STEP = 250.0
_BOND_GRID_STOCK_MAX = 7_000.0
_BOND_GRID_STOCK_STEP = 1_000.0
_BOND_GRID_BOND_STEP = 500.0
_COMBINED_GRID_N_SIMULATIONS = 10_000
_COMBINED_GRID_STOCK_STEP = 500.0
_COMBINED_GRID_BOND_STEP = 500.0


def _generate_mc_returns(
    rng: np.random.Generator,
    n_simulations: int,
    years: int,
    stock_mean: float,
    stock_std: float,
    inflation_mean: float,
    inflation_std: float,
    bonds_yield: float,
    bond_std: float,
    inflation_ar1_phi: float = 0.0,
    stock_return_df: float = 1000.0,
    stock_bond_corr: float = 0.0,
    stock_return_ar1_phi: float = 0.0,
    stock_mean_uncertainty: float = 0.0,
    stock_vol_uncertainty: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (n_simulations, years)

    # AR(1) mean reversion in z-space: z_t = φ·z_{t-1} + √(1−φ²)·ε_t
    # Marginal distribution of each z_t remains N(0,1), so t-transform is still exact.
    if abs(stock_return_ar1_phi) > 1e-9:
        innovation_scale = np.sqrt(max(1.0 - stock_return_ar1_phi ** 2, 0.0))
        z_innovations = rng.standard_normal(shape)
        z_stock = np.empty(shape)
        z_prev = np.zeros(n_simulations)
        for t in range(years):
            z_stock[:, t] = stock_return_ar1_phi * z_prev + innovation_scale * z_innovations[:, t]
            z_prev = z_stock[:, t]
    else:
        z_stock = rng.standard_normal(shape)

    # Cholesky stock/bond correlation
    z_bond_raw = rng.standard_normal(shape)
    if abs(stock_bond_corr) > 1e-9:
        z_bond = stock_bond_corr * z_stock + np.sqrt(1.0 - stock_bond_corr ** 2) * z_bond_raw
    else:
        z_bond = z_bond_raw

    # Parameter uncertainty: each simulation draws its own μ and σ from a prior.
    # Models epistemic uncertainty — we don't know the true long-run CAGR or volatility.
    if stock_mean_uncertainty > 1e-9:
        eff_mean = rng.normal(stock_mean, stock_mean_uncertainty, (n_simulations, 1))
    else:
        eff_mean = stock_mean
    if stock_vol_uncertainty > 1e-9:
        eff_std = np.maximum(rng.normal(stock_std, stock_vol_uncertainty, (n_simulations, 1)), 0.01)
    else:
        eff_std = stock_std

    # Student-t fat tails for stocks when df < 100 (preserves target std)
    if stock_return_df < 100.0:
        df = stock_return_df
        chi2 = rng.chisquare(df, shape) / df
        t_draw = z_stock / np.sqrt(chi2)
        stock_returns = eff_mean + t_draw * eff_std * np.sqrt((df - 2.0) / df)
    else:
        stock_returns = eff_mean + z_stock * eff_std
    stock_returns = np.maximum(stock_returns, -0.99)

    # Inflation — independent of stock/bond correlation
    if inflation_ar1_phi == 0.0:
        inflations = rng.normal(inflation_mean, inflation_std, shape)
    else:
        # AR(1): inflation_t = μ + φ·(inflation_{t-1} − μ) + ε_t
        # innovation_std keeps marginal variance = inflation_std²
        innovation_std = inflation_std * np.sqrt(1 - inflation_ar1_phi ** 2)
        inflations = np.empty(shape)
        prev = np.full(n_simulations, inflation_mean)
        for t in range(years):
            eps = rng.normal(0.0, innovation_std, n_simulations)
            inflations[:, t] = inflation_mean + inflation_ar1_phi * (prev - inflation_mean) + eps
            prev = inflations[:, t]
    inflations = np.where(inflations < 0, inflations / 2, inflations)

    # Bond yields from correlated z-scores
    bond_yields = np.maximum(bonds_yield + z_bond * bond_std, -0.5)
    return stock_returns, inflations, bond_yields


def _vec_annuity_factor(annual_rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fv_factor, annuity_factor) arrays for a vector of annual rates."""
    monthly_rates = (1 + annual_rates) ** (1 / 12) - 1
    fv_factors = (1 + monthly_rates) ** 12
    annuity_factors = np.where(
        np.abs(monthly_rates) < 1e-12, 12.0, (fv_factors - 1) / monthly_rates
    )
    return fv_factors, annuity_factors


class _MCSetup(NamedTuple):
    annual_stock_returns: np.ndarray
    annual_inflations: np.ndarray
    annual_bond_yields: np.ndarray
    stock_values: np.ndarray
    bond_values: np.ndarray
    cash_values: np.ndarray
    cash_fv_factor: float
    cash_annuity: float
    bonds_yield: float
    annual_growth: float
    bond_std: float


def _setup_mc(
    config: Config,
    n_simulations: int,
    n_years: int,
    seed: int | None,
    stock_mean: float,
    stock_std: float,
    inflation_std: float,
    initial_stock: float,
    initial_bonds: float,
    initial_cash: float,
) -> _MCSetup:
    bond_std = config.mc_bond_yield_std_percent / 100
    bonds_yield = config.yield_bonds / 100
    cash_rate = config.interest_on_cash_pa / 100
    annual_growth = config.annual_contribution_growth_percent / 100

    rng = np.random.default_rng(seed)
    annual_stock_returns, annual_inflations, annual_bond_yields = _generate_mc_returns(
        rng, n_simulations, n_years,
        stock_mean, stock_std,
        config.expected_inflation_percent / 100, inflation_std,
        bonds_yield, bond_std,
        inflation_ar1_phi=config.mc_inflation_ar1_phi,
        stock_return_df=config.mc_stock_return_df,
        stock_bond_corr=config.mc_stock_bond_correlation,
        stock_return_ar1_phi=config.mc_stock_return_ar1_phi,
        stock_mean_uncertainty=config.mc_stock_mean_uncertainty,
        stock_vol_uncertainty=config.mc_stock_vol_uncertainty,
    )
    stock_values = np.full(n_simulations, float(initial_stock))
    bond_values = np.full(n_simulations, float(initial_bonds))
    cash_values = np.full(n_simulations, float(initial_cash))

    monthly_cash_rate = (1 + cash_rate) ** (1 / 12) - 1
    cash_fv_factor = (1 + monthly_cash_rate) ** 12
    cash_annuity = (cash_fv_factor - 1) / monthly_cash_rate if monthly_cash_rate > 1e-12 else 12.0

    return _MCSetup(
        annual_stock_returns=annual_stock_returns,
        annual_inflations=annual_inflations,
        annual_bond_yields=annual_bond_yields,
        stock_values=stock_values,
        bond_values=bond_values,
        cash_values=cash_values,
        cash_fv_factor=cash_fv_factor,
        cash_annuity=cash_annuity,
        bonds_yield=bonds_yield,
        annual_growth=annual_growth,
        bond_std=bond_std,
    )


def run_monte_carlo_simulation(
    config: Config,
    scenario: ScenarioResult,
    n_simulations: int,
    stock_std: float,
    inflation_std: float,
    seed: int | None = None,
    monthly_stock_contribution_override: float | None = None,
    monthly_bonds_contribution_override: float | None = None,
    stock_mean_override: float | None = None,
    initial_stock_override: float | None = None,
    initial_bonds_override: float | None = None,
    initial_cash_override: float | None = None,
) -> MonteCarloResult:
    years = int(round(config.years))
    stock_mean = stock_mean_override if stock_mean_override is not None else config.planning_cagr
    initial_stock = initial_stock_override if initial_stock_override is not None else config.current_stock_value
    initial_bonds = initial_bonds_override if initial_bonds_override is not None else config.current_bonds_value
    initial_cash = initial_cash_override if initial_cash_override is not None else config.current_cash

    mc = _setup_mc(config, n_simulations, years, seed, stock_mean, stock_std, inflation_std,
                   initial_stock, initial_bonds, initial_cash)

    cumulative_inflation = np.prod(1 + mc.annual_inflations, axis=1)
    stock_values = mc.stock_values
    bond_values = mc.bond_values
    cash_values = mc.cash_values

    base_stock_contrib = monthly_stock_contribution_override if monthly_stock_contribution_override is not None else scenario.required_monthly_contribution
    base_bond_contrib = monthly_bonds_contribution_override if monthly_bonds_contribution_override is not None else scenario.required_monthly_bonds_contribution

    for year in range(years):
        stock_fv_factor, stock_annuity = _vec_annuity_factor(mc.annual_stock_returns[:, year])
        bond_fv_factor, bond_annuity = _vec_annuity_factor(mc.annual_bond_yields[:, year])
        g = (1 + mc.annual_growth) ** year
        c_stock = base_stock_contrib * g
        c_bond = base_bond_contrib * g
        c_cash = max(0.0, config.annual_net_saving / 12 * g - c_stock - c_bond)
        stock_values = stock_values * stock_fv_factor + c_stock * stock_annuity
        bond_values = bond_values * bond_fv_factor + c_bond * bond_annuity
        cash_values = cash_values * mc.cash_fv_factor + c_cash * mc.cash_annuity

    cash_converted = np.maximum(0.0, cash_values - config.cash_reserve_at_horizon)
    bond_values += cash_converted
    final_portfolios = stock_values + bond_values

    target_annual = config.target_monthly_retirement_income * cumulative_inflation * 12
    required_portfolios = target_annual / scenario.portfolio_net_income_rate
    surpluses = final_portfolios - required_portfolios

    bonds_weight = scenario.bonds_percentage / 100
    stocks_weight = 1.0 - bonds_weight
    required_stock_per_sim = required_portfolios * stocks_weight
    required_bonds_per_sim = required_portfolios * bonds_weight
    success = (stock_values >= required_stock_per_sim) & (bond_values >= required_bonds_per_sim)
    success_stock_only = stock_values >= required_stock_per_sim
    success_bond_only = bond_values >= required_bonds_per_sim

    s_frac = np.where(final_portfolios > 0, stock_values / final_portfolios, 0.0)
    b_frac = np.where(final_portfolios > 0, bond_values / final_portfolios, 0.0)
    actual_net_rate = (
        s_frac * scenario.withdrawal_rate * (1.0 - scenario.effective_stock_tax_rate)
        + b_frac * scenario.yield_bonds * (1.0 - scenario.bond_tax_rate)
    )
    success_income = final_portfolios * actual_net_rate >= target_annual

    tail_pct = 100 - config.mc_target_success_percent
    target_pct = config.mc_target_success_percent
    pcts = sorted({tail_pct, 25, 50, 75, target_pct})
    return MonteCarloResult(
        name=scenario.name,
        n_simulations=n_simulations,
        stock_std=stock_std,
        inflation_std=inflation_std,
        success_rate=float(np.mean(success)),
        success_rate_income_only=float(np.mean(success_income)),
        success_rate_stock_only=float(np.mean(success_stock_only)),
        success_rate_bond_only=float(np.mean(success_bond_only)),
        final_portfolio_percentiles={p: float(np.percentile(final_portfolios, p)) for p in pcts},
        required_portfolio_percentiles={p: float(np.percentile(required_portfolios, p)) for p in pcts},
        surplus_percentiles={p: float(np.percentile(surpluses, p)) for p in pcts},
        portfolio_net_income_rate=scenario.portfolio_net_income_rate,
        final_stock_percentiles={p: float(np.percentile(stock_values, p)) for p in pcts},
        final_bond_percentiles={p: float(np.percentile(bond_values, p)) for p in pcts},
        bond_std=mc.bond_std,
    )


def _mc_success_rate(
    config: Config,
    scenario: ScenarioResult,
    n_simulations: int,
    stock_std: float,
    inflation_std: float,
    seed: int,
    monthly_stock_contribution_override: float | None = None,
    monthly_bonds_contribution_override: float | None = None,
    stock_mean_override: float | None = None,
    initial_stock_override: float | None = None,
    initial_bonds_override: float | None = None,
    initial_cash_override: float | None = None,
    use_income_metric: bool = False,
    use_stock_metric: bool = False,
) -> float:
    result = run_monte_carlo_simulation(
        config=config,
        scenario=scenario,
        n_simulations=n_simulations,
        stock_std=stock_std,
        inflation_std=inflation_std,
        seed=seed,
        monthly_stock_contribution_override=monthly_stock_contribution_override,
        monthly_bonds_contribution_override=monthly_bonds_contribution_override,
        stock_mean_override=stock_mean_override,
        initial_stock_override=initial_stock_override,
        initial_bonds_override=initial_bonds_override,
        initial_cash_override=initial_cash_override,
    )
    if use_income_metric:
        return result.success_rate_income_only
    if use_stock_metric:
        return result.success_rate_stock_only
    return result.success_rate


def _mc_grid_worker(args: tuple):
    """Module-level picklable worker for parallel grid computation."""
    config, scenario, n_sims, stock_std, inflation_std, seed, smo, bmo, cagr = args
    return run_monte_carlo_simulation(
        config=config, scenario=scenario,
        n_simulations=n_sims, stock_std=stock_std, inflation_std=inflation_std,
        seed=seed,
        monthly_stock_contribution_override=smo,
        monthly_bonds_contribution_override=bmo,
        stock_mean_override=cagr,
    )


def _resolve_workers(n_workers: int | None, n_tasks: int, n_simulations: int) -> int:
    """Return worker count; forces sequential for tiny sims (e.g. tests)."""
    if n_simulations < 1_000 or n_tasks < 2:
        return 1
    if n_workers is None:
        return min(os.cpu_count() or 1, n_tasks)
    return max(1, n_workers)


def _parallel_map(args_list: list, n_workers: int, worker=_mc_grid_worker) -> list:
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(worker, args_list))
    return [worker(a) for a in args_list]


def compute_suggestions_scan(
    config: Config,
    scenario: ScenarioResult,
    stock_std: float,
    inflation_std: float,
    fixed_monthly_bonds: float,
    current_stock_mo: float,
    seed: int = 42,
) -> list[dict]:
    """Scan stock contribution from 0 to _SUGGESTIONS_SCAN_MAX_STOCK; cash = max(0, saving - stocks - bonds)."""
    monthly_saving = config.annual_net_saving / 12
    common = dict(config=config, scenario=scenario, n_simulations=_SUGGESTIONS_N_SIMULATIONS,
                  stock_std=stock_std, inflation_std=inflation_std, seed=seed,
                  monthly_bonds_contribution_override=fixed_monthly_bonds)
    scan_values: list[float] = []
    c = 0.0
    while c <= _SUGGESTIONS_SCAN_MAX_STOCK + _SUGGESTIONS_SCAN_STEP_MAIN / 2:
        scan_values.append(min(c, _SUGGESTIONS_SCAN_MAX_STOCK))
        if c >= _SUGGESTIONS_SCAN_MAX_STOCK:
            break
        c += _SUGGESTIONS_SCAN_STEP_MAIN
    if not any(abs(v - current_stock_mo) < 1.0 for v in scan_values):
        scan_values.append(current_stock_mo)
        scan_values.sort()

    rows = []
    for stocks_mo in scan_values:
        cash_mo = max(0.0, monthly_saving - stocks_mo - fixed_monthly_bonds)
        mc = run_monte_carlo_simulation(**common, monthly_stock_contribution_override=stocks_mo)
        rows.append({
            "stocks_mo": stocks_mo,
            "cash_mo": cash_mo,
            "rate": mc.success_rate,
            "rate_income": mc.success_rate_income_only,
            "rate_stock": mc.success_rate_stock_only,
            "median_stocks": mc.final_stock_percentiles[50],
            "median_bonds": mc.final_bond_percentiles[50],
            "p75_bonds": mc.final_bond_percentiles[75],
            "p50_target_income": mc.required_portfolio_percentiles[50] * scenario.portfolio_net_income_rate,
            "is_current": abs(stocks_mo - current_stock_mo) < 1.0,
        })
    return rows


def compute_required_cagr_for_success(
    config: Config,
    scenario: ScenarioResult,
    stock_std: float,
    inflation_std: float,
    fixed_monthly_bonds: float,
    monthly_stock_contribution: float,
    target_success: float = 0.95,
    seed: int = 42,
    use_income_metric: bool = False,
    use_stock_metric: bool = False,
) -> float | None:
    """Binary-search for the minimum stock CAGR to hit target_success, contributions fixed."""
    common = dict(
        config=config, scenario=scenario,
        n_simulations=_SUGGESTIONS_N_SIMULATIONS,
        stock_std=stock_std, inflation_std=inflation_std, seed=seed,
        monthly_stock_contribution_override=monthly_stock_contribution,
        monthly_bonds_contribution_override=fixed_monthly_bonds,
        use_income_metric=use_income_metric,
        use_stock_metric=use_stock_metric,
    )
    hi = 0.35
    if _mc_success_rate(**common, stock_mean_override=hi) < target_success:
        return None
    lo = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _mc_success_rate(**common, stock_mean_override=mid) >= target_success:
            hi = mid
        else:
            lo = mid
    return hi


def compute_years_to_financial_freedom_mc(
    config: Config,
    withdrawal_rate: float,
    stock_tax_rate: float,
    bond_tax_rate: float,
    stock_teilfreistellung: float,
    stock_std: float,
    inflation_std: float,
    monthly_stock: float,
    monthly_bonds: float,
    target_success: float = 0.95,
    n_simulations: int = _SUGGESTIONS_N_SIMULATIONS,
    seed: int = 42,
    max_years: int = 40,
) -> FinancialFreedomResult:
    tail_pct = 100 - config.mc_target_success_percent
    stock_mean = config.planning_cagr
    cost_basis = config.current_stock_value - config.current_stock_unrealized_gain

    mc = _setup_mc(
        config, n_simulations, max_years, seed,
        stock_mean, stock_std, inflation_std,
        config.current_stock_value, config.current_bonds_value, config.current_cash,
    )

    stock_values = mc.stock_values
    bond_values = mc.bond_values
    cash_values = mc.cash_values
    cumulative_inflation = np.ones(n_simulations)

    for year in range(max_years):
        T = year + 1
        g = (1 + mc.annual_growth) ** year
        c_stock = monthly_stock * g
        c_bond = monthly_bonds * g
        c_cash = max(0.0, config.annual_net_saving / 12 * g - c_stock - c_bond)

        stock_fv_yr, stock_ann_yr = _vec_annuity_factor(mc.annual_stock_returns[:, year])
        bond_fv_yr, bond_ann_yr = _vec_annuity_factor(mc.annual_bond_yields[:, year])

        stock_values = stock_values * stock_fv_yr + c_stock * stock_ann_yr
        bond_values = bond_values * bond_fv_yr + c_bond * bond_ann_yr
        cash_values = cash_values * mc.cash_fv_factor + c_cash * mc.cash_annuity
        cost_basis += c_stock * 12
        cumulative_inflation = cumulative_inflation * (1 + mc.annual_inflations[:, year])

        cash_conv = np.maximum(0.0, cash_values - config.cash_reserve_at_horizon)
        total_bonds = bond_values + cash_conv
        portfolio = stock_values + total_bonds

        gain = np.maximum(0.0, stock_values - cost_basis)
        gain_frac = np.where(stock_values > 0, gain / stock_values, 0.0)
        s_frac = np.where(portfolio > 0, stock_values / portfolio, 0.0)
        b_frac = np.where(portfolio > 0, total_bonds / portfolio, 0.0)
        eff_tax = gain_frac * stock_teilfreistellung * stock_tax_rate
        net_rate = s_frac * withdrawal_rate * (1 - eff_tax) + b_frac * mc.annual_bond_yields[:, year] * (1 - bond_tax_rate)

        target_annual = config.target_monthly_retirement_income * cumulative_inflation * 12
        success = portfolio * net_rate >= target_annual
        success_rate = float(np.mean(success))

        if success_rate >= target_success:
            net_income_annual = portfolio * net_rate
            return FinancialFreedomResult(
                years=T,
                success_rate=success_rate,
                portfolio_p50=float(np.percentile(portfolio, 50)),
                portfolio_tail_pct=float(np.percentile(portfolio, tail_pct)),
                tail_pct=tail_pct,
                monthly_income_p50=float(np.percentile(net_income_annual, 50)) / 12,
            )

    return FinancialFreedomResult(years=None, success_rate=0.0, portfolio_p50=0.0,
                                  portfolio_tail_pct=0.0, tail_pct=tail_pct, monthly_income_p50=0.0)


def compute_years_to_targets_mc(
    config: Config,
    scenario: ScenarioResult,
    monthly_stock: float,
    monthly_bonds: float,
    stock_std: float,
    inflation_std: float,
    target_success: float = 0.95,
    n_simulations: int = _SUGGESTIONS_N_SIMULATIONS,
    seed: int = 42,
    max_years: int = 40,
) -> YearsToTargetsResult:
    stock_mean = config.planning_cagr
    bonds_weight = scenario.bonds_percentage / 100
    stocks_weight = 1.0 - bonds_weight

    mc = _setup_mc(
        config, n_simulations, max_years, seed,
        stock_mean, stock_std, inflation_std,
        config.current_stock_value, config.current_bonds_value, config.current_cash,
    )

    stock_values = mc.stock_values
    bond_values = mc.bond_values
    cash_values = mc.cash_values
    cumulative_inflation = np.ones(n_simulations)

    year_stocks: int | None = None
    year_bonds: int | None = None
    year_both: int | None = None
    year_income: int | None = None

    for year in range(max_years):
        T = year + 1
        g = (1 + mc.annual_growth) ** year
        c_stock = monthly_stock * g
        c_bond = monthly_bonds * g
        c_cash = max(0.0, config.annual_net_saving / 12 * g - c_stock - c_bond)

        stock_fv_yr, stock_ann_yr = _vec_annuity_factor(mc.annual_stock_returns[:, year])
        bond_fv_yr, bond_ann_yr = _vec_annuity_factor(mc.annual_bond_yields[:, year])

        stock_values = stock_values * stock_fv_yr + c_stock * stock_ann_yr
        bond_values = bond_values * bond_fv_yr + c_bond * bond_ann_yr
        cash_values = cash_values * mc.cash_fv_factor + c_cash * mc.cash_annuity
        cumulative_inflation = cumulative_inflation * (1 + mc.annual_inflations[:, year])

        cash_conv = np.maximum(0.0, cash_values - config.cash_reserve_at_horizon)
        total_bonds = bond_values + cash_conv
        portfolio = stock_values + total_bonds

        target_annual = config.target_monthly_retirement_income * cumulative_inflation * 12
        required_portfolio = target_annual / scenario.portfolio_net_income_rate
        required_stock = required_portfolio * stocks_weight
        required_bonds = required_portfolio * bonds_weight

        success_stock = stock_values >= required_stock
        success_bonds = total_bonds >= required_bonds
        success_both = success_stock & success_bonds

        s_frac = np.where(portfolio > 0, stock_values / portfolio, 0.0)
        b_frac = np.where(portfolio > 0, total_bonds / portfolio, 0.0)
        actual_net_rate = (
            s_frac * scenario.withdrawal_rate * (1.0 - scenario.effective_stock_tax_rate)
            + b_frac * scenario.yield_bonds * (1.0 - scenario.bond_tax_rate)
        )
        success_income = portfolio * actual_net_rate >= target_annual

        if year_stocks is None and float(np.mean(success_stock)) >= target_success:
            year_stocks = T
        if year_bonds is None and float(np.mean(success_bonds)) >= target_success:
            year_bonds = T
        if year_both is None and float(np.mean(success_both)) >= target_success:
            year_both = T
        if year_income is None and float(np.mean(success_income)) >= target_success:
            year_income = T

        if all(v is not None for v in (year_stocks, year_bonds, year_both, year_income)):
            break

    return YearsToTargetsResult(
        years_to_stocks=year_stocks,
        years_to_bonds=year_bonds,
        years_to_both=year_both,
        years_to_income=year_income,
    )


def _golden_section_max(fn, lo: float, hi: float, n_iter: int = 30) -> float:
    """Golden-section search for the maximum of a unimodal function on [lo, hi]."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.618
    c = hi - phi * (hi - lo)
    d = lo + phi * (hi - lo)
    fc, fd = fn(c), fn(d)
    for _ in range(n_iter - 2):
        if fc < fd:
            lo = c
            c, fc = d, fd
            d = lo + phi * (hi - lo)
            fd = fn(d)
        else:
            hi = d
            d, fd = c, fc
            c = hi - phi * (hi - lo)
            fc = fn(c)
    return (lo + hi) / 2.0


def _optimize_cell_worker(args: tuple):
    """Per-upfront-cell 2D search over (stocks/mo, bonds/mo) for Q1 hard target,
    plus golden-section income split, plus an explicit baseline-contribution floor.
    Returns a dict consumed by compute_optimal_allocation."""
    (config, scenario, n_simulations, stock_std, inflation_std, seed,
     target_success, max_contrib, extra_stocks, extra_bonds,
     new_stocks, new_bonds, new_cash, deterministic_bonds_mo,
     current_stocks_mo, current_bonds_mo, investable_monthly, bond_levels,
     budget_cap) = args

    _base = dict(
        config=config, scenario=scenario,
        n_simulations=n_simulations, stock_std=stock_std,
        inflation_std=inflation_std, seed=seed,
        initial_stock_override=new_stocks,
        initial_bonds_override=new_bonds,
        initial_cash_override=new_cash,
    )

    def _rate(stocks_mo: float, bonds_mo: float) -> float:
        return _mc_success_rate(
            **_base,
            monthly_stock_contribution_override=stocks_mo,
            monthly_bonds_contribution_override=bonds_mo,
        )

    # Candidate bond levels: uniform grid + deterministic + user's current bonds/mo
    bond_candidates = sorted(set(
        [round(b, 2) for b in bond_levels]
        + [round(deterministic_bonds_mo, 2), round(current_bonds_mo, 2)]
    ))

    best_total = float("inf")
    best_stocks_mo = float("inf")
    best_bonds_mo = deterministic_bonds_mo
    best_rate = 0.0
    ceiling_max = 0.0
    ceiling_stocks_mo = 0.0
    ceiling_bonds_mo = deterministic_bonds_mo

    for bonds_mo in bond_candidates:
        # When a budget cap is set, the search ceiling on stocks_mo is the remaining
        # headroom (budget_cap - bonds_mo). Skip bond levels that leave no headroom.
        if budget_cap is not None:
            stocks_headroom = budget_cap - bonds_mo
            if stocks_headroom <= 0:
                continue
            search_hi = min(max_contrib, stocks_headroom)
        else:
            search_hi = max_contrib
        ceiling = _rate(search_hi, bonds_mo)
        if ceiling > ceiling_max:
            ceiling_max = ceiling
            ceiling_stocks_mo = search_hi
            ceiling_bonds_mo = bonds_mo
        if ceiling < target_success:
            continue
        lo, hi = 0.0, search_hi
        rate_at_hi = ceiling
        for _ in range(30):
            mid = (lo + hi) / 2
            r = _rate(mid, bonds_mo)
            if r >= target_success:
                hi = mid
                rate_at_hi = r
            else:
                lo = mid
        total = hi + bonds_mo
        if total < best_total:
            best_total = total
            best_stocks_mo = hi
            best_bonds_mo = bonds_mo
            best_rate = rate_at_hi

    # Baseline floor: user's actual current contributions at this upfront allocation.
    # Guarantees optimizer ≥ baseline whenever baseline already hits target_success
    # AND is within the budget cap (if any).
    baseline_total_candidate = current_stocks_mo + current_bonds_mo
    if budget_cap is None or baseline_total_candidate <= budget_cap + 1e-6:
        baseline_rate = _rate(current_stocks_mo, current_bonds_mo)
        if baseline_rate >= target_success:
            if baseline_total_candidate < best_total:
                best_total = baseline_total_candidate
                best_stocks_mo = current_stocks_mo
                best_bonds_mo = current_bonds_mo
                best_rate = baseline_rate
        if baseline_rate > ceiling_max:
            ceiling_max = baseline_rate
            ceiling_stocks_mo = current_stocks_mo
            ceiling_bonds_mo = current_bonds_mo

    # Income optimiser: vary stocks/bonds split over full monthly saving;
    # the simulation absorbs (monthly_saving − stocks − bonds) into cash, so cash
    # is the residual and may be 0 when stocks+bonds = monthly_saving.
    income_budget = config.annual_net_saving / 12
    def income_rate_for_f(f: float) -> float:
        stocks_mo = f * income_budget
        bonds_mo = (1.0 - f) * income_budget
        return _mc_success_rate(
            **_base,
            monthly_stock_contribution_override=stocks_mo,
            monthly_bonds_contribution_override=bonds_mo,
            use_income_metric=True,
        )

    best_f = _golden_section_max(income_rate_for_f, 0.0, 1.0, n_iter=30)
    best_income_rate = income_rate_for_f(best_f)
    opt_stocks_mo = best_f * income_budget
    opt_bonds_mo = (1.0 - best_f) * income_budget

    achievable = best_total < float("inf")
    return {
        "extra_stocks": extra_stocks,
        "extra_bonds": extra_bonds,
        "new_stocks": new_stocks,
        "new_bonds": new_bonds,
        "new_cash": new_cash,
        "achievable": achievable,
        "monthly_stocks": best_stocks_mo if achievable else float("inf"),
        "monthly_bonds": best_bonds_mo if achievable else deterministic_bonds_mo,
        "total_monthly": best_total if achievable else float("inf"),
        "q1_success_rate": best_rate,
        "portfolio_ceiling_rate": ceiling_max,
        "ceiling_stocks_mo": ceiling_stocks_mo,
        "ceiling_bonds_mo": ceiling_bonds_mo,
        "income_stock_mo": opt_stocks_mo,
        "income_bonds_mo": opt_bonds_mo,
        "best_income_rate": best_income_rate,
    }


def compute_optimal_allocation(
    config: Config,
    scenario: ScenarioResult,
    stock_std: float,
    inflation_std: float,
    current_stocks_mo: float,
    current_bonds_mo: float,
    target_success: float = 0.95,
    min_cash_today: float = _OPTIMIZE_MIN_CASH_TODAY,
    step: float = _OPTIMIZE_STEP,
    seed: int = 42,
    n_simulations: int = _OPTIMIZE_N_SIMULATIONS,
    n_bond_levels: int = _OPTIMIZE_BONDS_LEVELS,
    n_workers: int | None = None,
    budget_cap: float | None = None,
) -> OptimalAllocationResult:
    min_cash_today = max(min_cash_today, config.cash_reserve_at_horizon)
    available = max(0.0, config.current_cash - min_cash_today)
    extra_stocks_values = np.arange(0.0, available + step * 0.5, step)
    master_rng = np.random.default_rng(seed)

    _cash_pct = max(0.0, 1.0 - config.maximum_stock_monthly_investment_pct
                    - config.maximum_bonds_monthly_investment_pct)
    monthly_cash_contrib = config.annual_net_saving / 12 * _cash_pct
    cash_rate = config.interest_on_cash_pa / 100
    contribution_growth = config.annual_contribution_growth_percent / 100
    monthly_saving = config.annual_net_saving / 12
    # Monthly budget available for stocks+bonds (cash slice is pre-allocated)
    investable_monthly = monthly_saving * (1.0 - _cash_pct)
    max_contrib = max(scenario.required_monthly_contribution * 25, 25_000.0)
    # Bond search range: 0 to 2× monthly saving (covers over-budget plans)
    bond_levels = list(np.linspace(0.0, 2.0 * monthly_saving, n_bond_levels))

    # Build per-cell args
    args_list = []
    for extra_stocks in extra_stocks_values:
        remaining = available - extra_stocks
        extra_bonds_values = np.arange(0.0, remaining + step * 0.5, step)
        for extra_bonds in extra_bonds_values:
            new_stocks = config.current_stock_value + extra_stocks
            new_bonds = config.current_bonds_value + extra_bonds
            new_cash = config.current_cash - extra_stocks - extra_bonds

            cash_to_bonds = cash_converted_to_bonds_at_horizon(
                current_cash=new_cash,
                cash_rate=cash_rate,
                years=config.years,
                cash_reserve=config.cash_reserve_at_horizon,
                monthly_contribution=monthly_cash_contrib,
                contribution_growth_rate=contribution_growth,
            )
            adjusted_bonds_target = max(0.0, scenario.required_bonds_amount - cash_to_bonds)
            deterministic_bonds_mo = calculate_required_monthly_contribution(
                current_amount=new_bonds,
                target_amount=adjusted_bonds_target,
                years=config.years,
                annual_yield=config.yield_bonds / 100,
                annual_contribution_growth=config.annual_contribution_growth_percent / 100,
            )

            point_seed = int(master_rng.integers(0, 2**31))
            args_list.append((
                config, scenario, n_simulations, stock_std, inflation_std, point_seed,
                target_success, max_contrib, float(extra_stocks), float(extra_bonds),
                float(new_stocks), float(new_bonds), float(new_cash),
                float(deterministic_bonds_mo),
                float(current_stocks_mo), float(current_bonds_mo),
                float(investable_monthly), bond_levels,
                float(budget_cap) if budget_cap is not None else None,
            ))

    workers = _resolve_workers(n_workers, len(args_list), n_simulations)
    results = _parallel_map(args_list, workers, worker=_optimize_cell_worker)

    grid: list[AllocationGridPoint] = [
        AllocationGridPoint(
            extra_stocks=r["extra_stocks"],
            extra_bonds=r["extra_bonds"],
            new_stocks=r["new_stocks"],
            new_bonds=r["new_bonds"],
            new_cash=r["new_cash"],
            monthly_stocks=r["monthly_stocks"],
            monthly_bonds=r["monthly_bonds"],
            total_monthly=r["total_monthly"],
            achievable=r["achievable"],
            best_success_rate=r["best_income_rate"],
            income_stock_mo=r["income_stock_mo"],
            income_bonds_mo=r["income_bonds_mo"],
            portfolio_ceiling_rate=r["portfolio_ceiling_rate"],
            ceiling_stocks_mo=r["ceiling_stocks_mo"],
            ceiling_bonds_mo=r["ceiling_bonds_mo"],
            q1_success_rate=r["q1_success_rate"],
        )
        for r in results
    ]

    achievable = [p for p in grid if p.achievable]
    optimal = min(achievable, key=lambda p: p.total_monthly) if achievable else None
    return OptimalAllocationResult(
        scenario_name=scenario.name,
        target_success=target_success,
        min_cash_today=min_cash_today,
        grid=grid,
        optimal=optimal,
    )


def compute_contribution_cagr_grid(
    config: Config,
    scenario: ScenarioResult,
    stock_std: float,
    inflation_std: float,
    seed: int = 42,
    n_simulations: int | None = None,
    n_workers: int | None = None,
) -> list[ContributionCagrGrid]:
    n_sims = config.mc_grid_n_simulations if n_simulations is None else n_simulations

    cagr_values: list[float] = []
    c = config.mc_grid_cagr_min
    while c <= config.mc_grid_cagr_max + 1e-9:
        cagr_values.append(round(c, 4))
        c = round(c + config.mc_grid_cagr_step, 4)

    suggested = scenario.required_monthly_contribution
    start = float(round(suggested / 1_000) * 1_000)
    contribution_values: list[float] = []
    s = start
    while s <= config.mc_grid_contribution_max + 1e-9:
        contribution_values.append(s)
        s += _GRID_CONTRIBUTION_STEP
    if not contribution_values:
        contribution_values = [start]

    tail_pct = 100 - config.mc_target_success_percent
    master_rng = np.random.default_rng(seed)

    # Generate all seeds and args upfront to preserve determinism
    args_list = []
    for multiplier in [1, 2]:
        bonds_monthly = scenario.required_monthly_bonds_contribution * multiplier
        for cagr in cagr_values:
            for stocks_mo in contribution_values:
                cell_seed = int(master_rng.integers(0, 2**31))
                args_list.append((config, scenario, n_sims, stock_std, inflation_std,
                                  cell_seed, stocks_mo, bonds_monthly, cagr))

    results = _parallel_map(args_list, _resolve_workers(n_workers, len(args_list), n_sims))

    # Reassemble in the same iteration order
    grids: list[ContributionCagrGrid] = []
    idx = 0
    for multiplier in [1, 2]:
        bonds_monthly = scenario.required_monthly_bonds_contribution * multiplier
        grid_rows: list[list[CagrContributionCell]] = []
        for _ in cagr_values:
            row: list[CagrContributionCell] = []
            for _ in contribution_values:
                r = results[idx]; idx += 1
                row.append(CagrContributionCell(
                    success_hard=r.success_rate,
                    success_income=r.success_rate_income_only,
                    success_stock_only=r.success_rate_stock_only,
                    stock_tail_pct=r.final_stock_percentiles[tail_pct],
                ))
            grid_rows.append(row)
        grids.append(ContributionCagrGrid(
            scenario_name=scenario.name,
            bonds_monthly=bonds_monthly,
            bonds_multiplier=multiplier,
            cagr_values=cagr_values,
            contribution_values=contribution_values,
            suggested_contribution=suggested,
            grid=grid_rows,
        ))
    return grids


def compute_bond_stock_grid(
    config: Config,
    scenario: ScenarioResult,
    stock_std: float,
    inflation_std: float,
    seed: int = 42,
    n_simulations: int | None = None,
    n_workers: int | None = None,
) -> BondStockGrid:
    stocks_values: list[float] = []
    s = 0.0
    while s <= _BOND_GRID_STOCK_MAX + 1e-9:
        stocks_values.append(s)
        s += _BOND_GRID_STOCK_STEP

    det_bonds = scenario.required_monthly_bonds_contribution
    bonds_max = round(det_bonds * 2.0 / _BOND_GRID_BOND_STEP) * _BOND_GRID_BOND_STEP
    bonds_values: list[float] = []
    b = 0.0
    while b <= bonds_max + 1e-9:
        bonds_values.append(round(b, 2))
        b += _BOND_GRID_BOND_STEP
    if not any(abs(v - det_bonds) < 1.0 for v in bonds_values):
        bonds_values.append(round(det_bonds, 2))
        bonds_values.sort()

    n_sims = config.mc_grid_n_simulations if n_simulations is None else n_simulations
    master_rng = np.random.default_rng(seed)
    args_list = []
    for bonds_mo in bonds_values:
        for stocks_mo in stocks_values:
            cell_seed = int(master_rng.integers(0, 2**31))
            args_list.append((config, scenario, n_sims, stock_std, inflation_std,
                              cell_seed, stocks_mo, bonds_mo, None))

    results = _parallel_map(args_list, _resolve_workers(n_workers, len(args_list), n_sims))

    grid_rows: list[list[BondStockCell]] = []
    idx = 0
    for _ in bonds_values:
        row: list[BondStockCell] = []
        for _ in stocks_values:
            row.append(BondStockCell(success_bond=results[idx].success_rate_bond_only))
            idx += 1
        grid_rows.append(row)

    return BondStockGrid(
        scenario_name=scenario.name,
        stocks_values=stocks_values,
        bonds_values=bonds_values,
        required_bonds=scenario.required_bonds_amount,
        suggested_stocks=scenario.required_monthly_contribution,
        suggested_bonds=det_bonds,
        grid=grid_rows,
    )


def compute_combined_grid(
    config: Config,
    scenario: ScenarioResult,
    stock_std: float,
    inflation_std: float,
    seed: int = 42,
    n_simulations: int = _COMBINED_GRID_N_SIMULATIONS,
    n_workers: int | None = None,
) -> CombinedGrid:
    stocks_values: list[float] = []
    s = 0.0
    while s <= config.mc_grid_contribution_max + 1e-9:
        stocks_values.append(round(s, 2))
        s += _COMBINED_GRID_STOCK_STEP

    det_bonds = scenario.required_monthly_bonds_contribution
    bonds_max = round(det_bonds * 2.0 / _COMBINED_GRID_BOND_STEP) * _COMBINED_GRID_BOND_STEP
    bonds_values: list[float] = []
    b = 0.0
    while b <= bonds_max + 1e-9:
        bonds_values.append(round(b, 2))
        b += _COMBINED_GRID_BOND_STEP

    cagr_values: list[float] = []
    c = config.mc_grid_cagr_min
    while c <= config.mc_grid_cagr_max + 1e-9:
        cagr_values.append(round(c, 4))
        c = round(c + config.mc_grid_cagr_step, 4)

    master_rng = np.random.default_rng(seed)
    args_list = []
    for cagr in cagr_values:
        for bonds_mo in bonds_values:
            for stocks_mo in stocks_values:
                cell_seed = int(master_rng.integers(0, 2**31))
                args_list.append((config, scenario, n_simulations, stock_std, inflation_std,
                                  cell_seed, stocks_mo, bonds_mo, cagr))

    results = _parallel_map(args_list, _resolve_workers(n_workers, len(args_list), n_simulations))

    grid: list[list[list[CombinedGridCell]]] = []
    idx = 0
    for _ in cagr_values:
        cagr_layer: list[list[CombinedGridCell]] = []
        for _ in bonds_values:
            row: list[CombinedGridCell] = []
            for _ in stocks_values:
                row.append(CombinedGridCell(success_combined=results[idx].success_rate))
                idx += 1
            cagr_layer.append(row)
        grid.append(cagr_layer)

    return CombinedGrid(
        scenario_name=scenario.name,
        cagr_values=cagr_values,
        stocks_values=stocks_values,
        bonds_values=bonds_values,
        required_stocks=scenario.required_stock_amount,
        required_bonds=scenario.required_bonds_amount,
        suggested_stocks=scenario.required_monthly_contribution,
        suggested_bonds=det_bonds,
        grid=grid,
    )


def compute_sensitivity_waterfall(
    config: Config,
    scenario: ScenarioResult,
    stock_std: float,
    inflation_std: float,
    current_stock_mo: float,
    current_bonds_mo: float,
    n_simulations: int = _SENSITIVITY_N_SIMULATIONS,
    seed: int = 42,
    n_workers: int | None = None,
) -> SensitivityWaterfallResult:
    cagr = config.planning_cagr

    # Each tuple: (param_name, base_value_str, delta_label, up_args, dn_args)
    # Args format matches _mc_grid_worker: (config, scenario, n_sims, stock_std, inflation_std, seed, smo, bmo, cagr)
    def _a(cfg=config, ss=stock_std, smo=current_stock_mo, bmo=current_bonds_mo, sm=None):
        return (cfg, scenario, n_simulations, ss, inflation_std, seed, smo, bmo, sm)

    param_defs = [
        # Display the planning CAGR (override-aware) — that's the centre point
        # of the ±2pp sensitivity perturbation below.
        ("Stock CAGR", f"{config.planning_cagr*100:.1f}%", "±2pp",
         _a(sm=cagr + 0.02),
         _a(sm=max(0.001, cagr - 0.02))),

        ("Inflation", f"{config.expected_inflation_percent:.1f}%", "±1pp",
         _a(cfg=_dc_replace(config, expected_inflation_percent=config.expected_inflation_percent + 1.0)),
         _a(cfg=_dc_replace(config, expected_inflation_percent=max(0.0, config.expected_inflation_percent - 1.0)))),

        ("Stock contribution", f"{current_stock_mo:,.0f}€/mo", "±500€/mo",
         _a(smo=current_stock_mo + 500),
         _a(smo=max(0.0, current_stock_mo - 500))),

        ("Bond contribution", f"{current_bonds_mo:,.0f}€/mo", "±500€/mo",
         _a(bmo=current_bonds_mo + 500),
         _a(bmo=max(0.0, current_bonds_mo - 500))),

        ("Bond yield", f"{config.yield_bonds:.1f}%", "±0.5pp",
         _a(cfg=_dc_replace(config, yield_bonds=config.yield_bonds + 0.5)),
         _a(cfg=_dc_replace(config, yield_bonds=max(0.0, config.yield_bonds - 0.5)))),

        ("Contrib. growth", f"{config.annual_contribution_growth_percent:.1f}%", "±0.5pp",
         _a(cfg=_dc_replace(config, annual_contribution_growth_percent=config.annual_contribution_growth_percent + 0.5)),
         _a(cfg=_dc_replace(config, annual_contribution_growth_percent=max(0.0, config.annual_contribution_growth_percent - 0.5)))),

        ("Stock volatility (σ)", f"{stock_std * 100:.0f}%", "±5pp",
         _a(ss=stock_std + 0.05),
         _a(ss=max(0.01, stock_std - 0.05))),
    ]

    # Flat list: [base, up_0, dn_0, up_1, dn_1, ...]
    args_list = [_a()]
    for _, _, _, up_args, dn_args in param_defs:
        args_list.append(up_args)
        args_list.append(dn_args)

    results = _parallel_map(args_list, _resolve_workers(n_workers, len(args_list), n_simulations))

    base = results[0].success_rate_income_only
    entries = [
        SensitivityEntry(
            parameter=p, base_value=bv, delta_label=dl,
            success_base=base,
            success_up=results[1 + i * 2].success_rate_income_only,
            success_down=results[2 + i * 2].success_rate_income_only,
        )
        for i, (p, bv, dl, _, _) in enumerate(param_defs)
    ]
    entries.sort(
        key=lambda e: max(abs(e.success_up - e.success_base), abs(e.success_down - e.success_base)),
        reverse=True,
    )
    return SensitivityWaterfallResult(scenario_name=scenario.name, entries=entries)


def compute_stock_allocation_sweep(
    config: Config,
    scenario_1: ScenarioResult,
    scenario_2: ScenarioResult,
    stock_std: float,
    inflation_std: float,
    current_stock_mo: float,
    current_bonds_mo_1: float,
    current_bonds_mo_2: float,
    seed: int = 42,
) -> list[dict[str, float]]:
    max_extra = config.current_cash - config.cash_reserve_at_horizon
    common_1 = dict(config=config, scenario=scenario_1, n_simulations=_SUGGESTIONS_N_SIMULATIONS,
                    stock_std=stock_std, inflation_std=inflation_std, seed=seed,
                    monthly_stock_contribution_override=current_stock_mo,
                    monthly_bonds_contribution_override=current_bonds_mo_1)
    common_2 = dict(config=config, scenario=scenario_2, n_simulations=_SUGGESTIONS_N_SIMULATIONS,
                    stock_std=stock_std, inflation_std=inflation_std, seed=seed,
                    monthly_stock_contribution_override=current_stock_mo,
                    monthly_bonds_contribution_override=current_bonds_mo_2)
    results = []
    extra = 0.0
    while extra <= max_extra + 1:
        new_stock = config.current_stock_value + extra
        new_cash = config.current_cash - extra
        results.append({
            "extra": extra,
            "stocks_t0": new_stock,
            "cash_t0": new_cash,
            "rate_1": _mc_success_rate(**common_1, initial_stock_override=new_stock, initial_cash_override=new_cash),
            "rate_2": _mc_success_rate(**common_2, initial_stock_override=new_stock, initial_cash_override=new_cash),
        })
        extra += 50_000.0
    return results


# ── Inflation × CAGR sensitivity heatmap ─────────────────────────────────────

_INFLATION_CAGR_N_SIMULATIONS = 5_000
_INFLATION_VALUES = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]


def _mc_inflation_cagr_worker(args: tuple):
    config, scenario, n_sims, stock_std, inflation_std, seed, smo, bmo, cagr, inflation_pct = args
    cfg = _dc_replace(config, expected_inflation_percent=inflation_pct * 100.0)
    return run_monte_carlo_simulation(
        config=cfg, scenario=scenario,
        n_simulations=n_sims, stock_std=stock_std, inflation_std=inflation_std,
        seed=seed,
        monthly_stock_contribution_override=smo,
        monthly_bonds_contribution_override=bmo,
        stock_mean_override=cagr,
    )


def compute_inflation_cagr_grid(
    config: Config,
    scenario: ScenarioResult,
    stock_std: float,
    inflation_std: float,
    monthly_stocks: float,
    monthly_bonds: float,
    seed: int = 42,
    n_simulations: int = _INFLATION_CAGR_N_SIMULATIONS,
    n_workers: int | None = None,
) -> InflationCagrGrid:
    """2D sensitivity grid: income success rate vs (inflation, CAGR).

    Both axes vary around the base-case config values, using current
    monthly contributions. Metric: P(portfolio income >= inflation-adjusted target).
    """
    cagr_values: list[float] = []
    c = config.mc_grid_cagr_min
    while c <= config.mc_grid_cagr_max + 1e-9:
        cagr_values.append(round(c, 4))
        c = round(c + config.mc_grid_cagr_step, 4)

    master_rng = np.random.default_rng(seed)
    args_list = []
    for inf in _INFLATION_VALUES:
        for cagr in cagr_values:
            cell_seed = int(master_rng.integers(0, 2**31))
            args_list.append((
                config, scenario, n_simulations, stock_std, inflation_std,
                cell_seed, monthly_stocks, monthly_bonds, cagr, inf,
            ))

    n_w = _resolve_workers(n_workers, len(args_list), n_simulations)
    if n_w > 1:
        with ProcessPoolExecutor(max_workers=n_w) as pool:
            results = list(pool.map(_mc_inflation_cagr_worker, args_list))
    else:
        results = [_mc_inflation_cagr_worker(a) for a in args_list]

    grid: list[list[InflationCagrCell]] = []
    idx = 0
    for _ in _INFLATION_VALUES:
        row: list[InflationCagrCell] = []
        for _ in cagr_values:
            row.append(InflationCagrCell(success_income=results[idx].success_rate_income_only))
            idx += 1
        grid.append(row)

    return InflationCagrGrid(
        scenario_name=scenario.name,
        cagr_values=cagr_values,
        inflation_values=_INFLATION_VALUES,
        base_cagr=config.planning_cagr,
        base_inflation=config.expected_inflation_percent / 100.0,
        monthly_stocks=monthly_stocks,
        monthly_bonds=monthly_bonds,
        grid=grid,
    )


def _mc_decum_inflation_cagr_worker(args: tuple):
    config, portfolio, n_sims, seed, cagr, inflation_pct = args
    cfg = _dc_replace(
        config,
        expected_yearly_growth_stocks_percent=cagr * 100.0,
        expected_inflation_percent=inflation_pct * 100.0,
    )
    res = simulate_decumulation(cfg, portfolio, seed=seed, n_simulations=n_sims)
    surv30 = res.survival_by_year[29] if len(res.survival_by_year) >= 30 else 0.0
    surv40 = res.survival_by_year[39] if len(res.survival_by_year) >= 40 else 0.0
    return surv30, surv40


def compute_decum_inflation_cagr_grid(
    config: Config,
    display_portfolio: float,
    seed: int = 42,
    n_simulations: int = _INFLATION_CAGR_N_SIMULATIONS,
    n_workers: int | None = None,
) -> DecumInflationCagrGrid:
    """2D sensitivity grid: decumulation survival rate vs (inflation, stock CAGR).

    Axes use the same ranges as the accumulation grid (mc_grid_cagr_* in config)
    and the shared _INFLATION_VALUES constant. All cells use simulate_decumulation
    at display_portfolio with the modified return/inflation parameters.
    """
    cagr_values: list[float] = []
    c = config.mc_grid_cagr_min
    while c <= config.mc_grid_cagr_max + 1e-9:
        cagr_values.append(round(c, 4))
        c = round(c + config.mc_grid_cagr_step, 4)

    master_rng = np.random.default_rng(seed)
    args_list = [
        (config, display_portfolio, n_simulations,
         int(master_rng.integers(0, 2**31)), cagr, inf)
        for inf in _INFLATION_VALUES
        for cagr in cagr_values
    ]

    n_w = _resolve_workers(n_workers, len(args_list), n_simulations)
    if n_w > 1:
        with ProcessPoolExecutor(max_workers=n_w) as pool:
            raw = list(pool.map(_mc_decum_inflation_cagr_worker, args_list))
    else:
        raw = [_mc_decum_inflation_cagr_worker(a) for a in args_list]

    grid: list[list[DecumInflationCagrCell]] = []
    idx = 0
    for _ in _INFLATION_VALUES:
        row: list[DecumInflationCagrCell] = []
        for _ in cagr_values:
            s30, s40 = raw[idx]
            row.append(DecumInflationCagrCell(survival_yr30=s30, survival_yr40=s40))
            idx += 1
        grid.append(row)

    return DecumInflationCagrGrid(
        display_portfolio=display_portfolio,
        cagr_values=cagr_values,
        inflation_values=_INFLATION_VALUES,
        base_cagr=config.planning_cagr,
        base_inflation=config.expected_inflation_percent / 100.0,
        grid=grid,
    )


# ── Decumulation MC ───────────────────────────────────────────────────────────

_SORR_SEQUENCES: list[tuple[str, list[float]]] = [
    ("Forced crash (yr 1-5)",  [-0.30, -0.15, +0.10, +0.05, +0.08]),
    ("Historical 2000-2002",   [-0.09, -0.12, -0.22]),
    ("Historical 2008-2009",   [-0.37, +0.26]),
]


def _run_decum_loop(
    annual_stock_returns: np.ndarray,
    annual_inflations: np.ndarray,
    annual_bond_yields: np.ndarray,
    initial_stock_v: np.ndarray,
    initial_bond_v: np.ndarray,
    initial_monthly_wd: float,
    n_years: int,
    withdrawal_strategy: str,
    bond_tax_rate: float,
    pension_monthly: float,
    pension_start_year: int,
    pension_adjustment_rate: float = 0.0,
    stock_pct_end: float = -1.0,
    glide_years: int = 0,
    effective_stock_tax_rate: float = 0.0,
) -> DecumulationResult:
    """Core decumulation year-loop. All simulation work is vectorised.

    Glide path: when glide_years > 0, stock allocation linearly shifts from the
    initial fraction toward stock_pct_end over glide_years years, then stays fixed.
    Annual rebalancing runs after stock growth, before withdrawals.
    """
    n_sims = len(initial_stock_v)
    stock_v = initial_stock_v.copy()
    bond_v = initial_bond_v.copy()
    initial_total = float(initial_stock_v[0] + initial_bond_v[0])
    initial_annual_wd = initial_monthly_wd * 12.0
    stock_pct_start = float(initial_stock_v[0] / initial_total) if initial_total > 0 else 0.0
    stock_allocation = stock_pct_start
    _glide_active = glide_years > 0 and stock_pct_end >= 0.0 and stock_pct_end != stock_pct_start

    alive = np.ones(n_sims, dtype=bool)
    cum_inf = np.ones(n_sims)
    gk_multiplier = np.ones(n_sims)
    ruin_years = np.full(n_sims, -1, dtype=int)

    pension_annual = pension_monthly * 12.0
    _PCTS = (3, 25, 50, 75, 97)

    survival_by_year: list[float] = []
    portfolio_pcts_by_year: list[dict[int, float]] = []
    withdrawal_pcts_by_year: list[dict[int, float]] = []

    for year in range(n_years):
        # 1. Grow stocks (bonds stay at par — coupon extracted below)
        stock_fv, _ = _vec_annuity_factor(annual_stock_returns[:, year])
        stock_v = stock_v * stock_fv

        # 2. Bond coupon: yield × principal × (1 − tax), net annual income
        bond_coupon = bond_v * np.maximum(annual_bond_yields[:, year], 0.0) * (1.0 - bond_tax_rate)

        # 3. Cumulative inflation (per-simulation, AR(1) model)
        cum_inf = cum_inf * (1.0 + annual_inflations[:, year])

        # 3b. Glide path rebalancing (after growth, before withdrawals)
        if _glide_active:
            progress = min(year + 1, glide_years) / glide_years
            target_frac = stock_pct_start + (stock_pct_end - stock_pct_start) * progress
            total_v = stock_v + bond_v
            rebalance_mask = alive & (total_v > 0.0)
            stock_v = np.where(rebalance_mask, total_v * target_frac, stock_v)
            bond_v  = np.where(rebalance_mask, total_v * (1.0 - target_frac), bond_v)

        # 4. Inflation-adjusted target and pension income
        target_annual = initial_annual_wd * cum_inf
        if year >= pension_start_year:
            pension_income = pension_annual * (1.0 + pension_adjustment_rate) ** (year - pension_start_year)
        else:
            pension_income = 0.0

        # 5. Guyton-Klinger adjustment on total target
        if withdrawal_strategy == "guyton_klinger":
            portfolio_total = stock_v + bond_v
            baseline = initial_total * cum_inf
            cut_mask = alive & (portfolio_total < 0.80 * baseline)
            raise_mask = alive & (portfolio_total > 1.20 * baseline)
            gk_multiplier = np.where(cut_mask, gk_multiplier * 0.90, gk_multiplier)
            gk_multiplier = np.where(raise_mask, gk_multiplier * 1.10, gk_multiplier)
            gk_multiplier = np.clip(gk_multiplier, 0.85, 1.20)
            target_annual = target_annual * gk_multiplier

        # 6. Residual from stocks after passive income; surplus is reinvested
        passive_income = bond_coupon + pension_income
        residual = np.maximum(0.0, target_annual - passive_income)
        surplus = np.maximum(0.0, passive_income - target_annual)

        # 7. Deduct / reinvest (alive sims only).
        # Scale withdrawal up by 1/(1−eff_tax) so we sell enough stock to net
        # `residual` after capital-gains tax on the gain portion. Otherwise the
        # simulation silently skips the tax bill and overstates survival.
        residual_gross = (
            residual / (1.0 - effective_stock_tax_rate)
            if effective_stock_tax_rate > 0 else residual
        )
        stock_v = stock_v - np.where(alive, residual_gross, 0.0)
        stock_v = stock_v + np.where(alive, surplus, 0.0)

        # 8. Fallback: if stocks exhausted, sell bond principal to cover shortfall
        shortfall = np.maximum(0.0, -stock_v)
        stock_v = np.maximum(stock_v, 0.0)
        bond_v = np.maximum(bond_v - np.where(alive, shortfall, 0.0), 0.0)

        # 9. Detect ruin (both exhausted)
        portfolio_total = stock_v + bond_v
        newly_dead = alive & (portfolio_total <= 0.0)
        ruin_years = np.where(newly_dead, year + 1, ruin_years)
        alive = alive & (portfolio_total > 0.0)

        # 10. Record
        survival_by_year.append(float(np.mean(alive)))
        portfolio_pcts_by_year.append(
            {p: float(np.percentile(portfolio_total, p)) for p in _PCTS}
        )
        eff_wd = np.where(alive, (residual + passive_income) / 12.0, np.nan)
        withdrawal_pcts_by_year.append(
            {p: float(np.nanpercentile(eff_wd, p)) for p in (10, 50, 90)}
        )

    failed = ruin_years > 0
    ruin_year_median = int(np.median(ruin_years[failed])) if np.any(failed) else None

    return DecumulationResult(
        n_simulations=n_sims,
        n_years=n_years,
        withdrawal_strategy=withdrawal_strategy,
        initial_portfolio=initial_total,
        initial_monthly_withdrawal=initial_monthly_wd,
        stock_allocation=stock_allocation,
        survival_rate=survival_by_year[-1],
        survival_by_year=survival_by_year,
        portfolio_percentiles_by_year=portfolio_pcts_by_year,
        withdrawal_percentiles_by_year=withdrawal_pcts_by_year,
        ruin_year_median=ruin_year_median,
    )


def simulate_decumulation(
    config: Config,
    initial_portfolio: float,
    stock_allocation: float | None = None,
    n_years: int | None = None,
    stock_std: float | None = None,
    inflation_std: float | None = None,
    seed: int = 42,
    n_simulations: int | None = None,
    withdrawal_strategy: str = "fixed",
    bond_tax_rate: float | None = None,
    glide_end_pct: float | None = None,
    glide_years: int | None = None,
    monthly_withdrawal_override: float | None = None,
    effective_stock_tax_rate: float | None = None,
) -> DecumulationResult:
    sa  = stock_allocation if stock_allocation is not None else config.mc_decumulation_stock_pct
    ny  = n_years          if n_years          is not None else config.mc_decumulation_years
    ss  = stock_std        if stock_std        is not None else config.mc_stock_return_std_percent / 100
    is_ = inflation_std    if inflation_std    is not None else config.mc_inflation_std_percent / 100
    n   = n_simulations    if n_simulations    is not None else config.mc_decumulation_n_simulations
    btr = bond_tax_rate    if bond_tax_rate    is not None else config.tax_rate_bond_italy
    gep = glide_end_pct    if glide_end_pct    is not None else config.mc_decumulation_glide_end_pct
    gyr = glide_years      if glide_years      is not None else config.mc_decumulation_glide_years
    # Capital-gains tax on stock withdrawals. Default = gain-fraction proxy 0.65
    # (typical for a long-held stock portfolio) × teilfreistellung × tax_rate.
    # The sizing functions use the same convention; this matches them in the loop.
    est = (
        effective_stock_tax_rate if effective_stock_tax_rate is not None
        else 0.65 * config.teilfreistellung * config.tax_rate
    )

    rng = np.random.default_rng(seed)
    stock_returns, inflations, bond_yields = _generate_mc_returns(
        rng, n, ny,
        config.planning_cagr, ss,
        config.expected_inflation_percent / 100, is_,
        config.yield_bonds / 100, config.mc_bond_yield_std_percent / 100,
        inflation_ar1_phi=config.mc_inflation_ar1_phi,
        stock_return_df=config.mc_stock_return_df,
        stock_bond_corr=config.mc_stock_bond_correlation,
        stock_return_ar1_phi=config.mc_stock_return_ar1_phi,
        stock_mean_uncertainty=config.mc_stock_mean_uncertainty,
        stock_vol_uncertainty=config.mc_stock_vol_uncertainty,
    )
    initial_stock_v = np.full(n, initial_portfolio * sa)
    initial_bond_v  = np.full(n, initial_portfolio * (1.0 - sa))
    initial_monthly_wd = (
        monthly_withdrawal_override if monthly_withdrawal_override is not None
        else initial_portfolio * (config.withdrawal_percentage_stock / 100) / 12.0
    )

    return _run_decum_loop(
        annual_stock_returns=stock_returns,
        annual_inflations=inflations,
        annual_bond_yields=bond_yields,
        initial_stock_v=initial_stock_v,
        initial_bond_v=initial_bond_v,
        initial_monthly_wd=initial_monthly_wd,
        n_years=ny,
        withdrawal_strategy=withdrawal_strategy,
        bond_tax_rate=btr,
        pension_monthly=config.mc_decumulation_pension_monthly,
        pension_start_year=config.mc_decumulation_pension_start_year,
        pension_adjustment_rate=config.mc_decumulation_pension_adjustment_rate,
        stock_pct_end=gep,
        glide_years=gyr,
        effective_stock_tax_rate=est,
    )


def find_required_decumulation_portfolio(
    config: Config,
    stock_allocation: float | None = None,
    target_survival: float | None = None,
    stock_std: float | None = None,
    inflation_std: float | None = None,
    seed: int = 42,
    n_simulations: int | None = None,
    withdrawal_strategy: str = "fixed",
    lo: float = 100_000.0,
    hi: float = 10_000_000.0,
    n_iter: int = 40,
) -> float:
    """Binary search for the minimum portfolio giving target survival rate."""
    target = target_survival if target_survival is not None else config.mc_target_success_percent / 100

    def _survival(portfolio: float) -> float:
        return simulate_decumulation(
            config=config, initial_portfolio=portfolio,
            stock_allocation=stock_allocation, stock_std=stock_std,
            inflation_std=inflation_std, seed=seed, n_simulations=n_simulations,
            withdrawal_strategy=withdrawal_strategy,
        ).survival_rate

    if _survival(hi) < target:
        return hi  # target unachievable in range

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        if _survival(mid) >= target:
            hi = mid
        else:
            lo = mid

    return (lo + hi) / 2.0


def compute_sorr_scenarios(
    config: Config,
    initial_portfolio: float,
    stock_allocation: float | None = None,
    stock_std: float | None = None,
    inflation_std: float | None = None,
    seed: int = 42,
    n_simulations: int | None = None,
    withdrawal_strategy: str = "fixed",
    monthly_withdrawal_override: float | None = None,
) -> SORRResult:
    """Run baseline decumulation + three historical stress sequences."""
    sa  = stock_allocation if stock_allocation is not None else config.mc_decumulation_stock_pct
    ny  = config.mc_decumulation_years
    ss  = stock_std     if stock_std     is not None else config.mc_stock_return_std_percent / 100
    is_ = inflation_std if inflation_std is not None else config.mc_inflation_std_percent / 100
    n   = n_simulations if n_simulations is not None else config.mc_decumulation_n_simulations

    rng = np.random.default_rng(seed)
    stock_returns, inflations, bond_yields = _generate_mc_returns(
        rng, n, ny,
        config.planning_cagr, ss,
        config.expected_inflation_percent / 100, is_,
        config.yield_bonds / 100, config.mc_bond_yield_std_percent / 100,
        inflation_ar1_phi=config.mc_inflation_ar1_phi,
        stock_return_df=config.mc_stock_return_df,
        stock_bond_corr=config.mc_stock_bond_correlation,
        stock_return_ar1_phi=config.mc_stock_return_ar1_phi,
        stock_mean_uncertainty=config.mc_stock_mean_uncertainty,
        stock_vol_uncertainty=config.mc_stock_vol_uncertainty,
    )
    initial_stock_v = np.full(n, initial_portfolio * sa)
    initial_bond_v  = np.full(n, initial_portfolio * (1.0 - sa))
    initial_monthly_wd = (
        monthly_withdrawal_override if monthly_withdrawal_override is not None
        else initial_portfolio * (config.withdrawal_percentage_stock / 100) / 12.0
    )

    _common = dict(
        annual_inflations=inflations,
        annual_bond_yields=bond_yields,
        initial_stock_v=initial_stock_v,
        initial_bond_v=initial_bond_v,
        initial_monthly_wd=initial_monthly_wd,
        n_years=ny,
        withdrawal_strategy=withdrawal_strategy,
        bond_tax_rate=config.tax_rate_bond_italy,
        pension_monthly=config.mc_decumulation_pension_monthly,
        pension_start_year=config.mc_decumulation_pension_start_year,
        pension_adjustment_rate=config.mc_decumulation_pension_adjustment_rate,
        stock_pct_end=config.mc_decumulation_glide_end_pct,
        glide_years=config.mc_decumulation_glide_years,
        effective_stock_tax_rate=0.65 * config.teilfreistellung * config.tax_rate,
    )

    baseline = _run_decum_loop(annual_stock_returns=stock_returns, **_common)

    stress_scenarios: list[tuple[str, DecumulationResult]] = []
    for label, seq in _SORR_SEQUENCES:
        patched = stock_returns.copy()
        for t, r in enumerate(seq):
            if t < ny:
                patched[:, t] = r
        stress_scenarios.append((label, _run_decum_loop(annual_stock_returns=patched, **_common)))

    return SORRResult(baseline=baseline, stress_scenarios=stress_scenarios)


def compute_joint_probability(
    config: Config,
    accumulation_p50: float,
    accumulation_tail: float,
    tail_pct: int,
    stock_allocation: float | None = None,
    stock_std: float | None = None,
    inflation_std: float | None = None,
    seed: int = 42,
    n_simulations: int | None = None,
    monthly_withdrawal_override: float | None = None,
) -> JointProbabilityResult:
    """Chain accumulation percentile portfolios into decumulation survival rates."""
    _kw = dict(
        config=config, stock_allocation=stock_allocation,
        stock_std=stock_std, inflation_std=inflation_std,
        seed=seed, n_simulations=n_simulations,
        monthly_withdrawal_override=monthly_withdrawal_override,
    )
    dec_p50  = simulate_decumulation(initial_portfolio=accumulation_p50,  **_kw)
    dec_tail = simulate_decumulation(initial_portfolio=accumulation_tail, **_kw)
    return JointProbabilityResult(
        accumulation_p50=accumulation_p50,
        accumulation_tail=accumulation_tail,
        tail_pct=tail_pct,
        p_survive_from_p50=dec_p50.survival_rate,
        p_survive_from_tail=dec_tail.survival_rate,
        joint_p50=0.50 * dec_p50.survival_rate,
        joint_tail=(tail_pct / 100.0) * dec_tail.survival_rate,
    )


def _survival_at_year(result: "DecumulationResult", year: int) -> float:
    """Return survival rate at a specific year (1-based). Returns 0 if beyond horizon."""
    if year <= 0 or year > result.n_years:
        return 0.0
    return result.survival_by_year[year - 1]


def compute_allocation_sweep(
    config: Config,
    effective_stock_tax_rate: float | None = None,
    bonds_pct_values: list[float] | None = None,
    seed: int = 42,
    n_simulations: int | None = None,
) -> AllocationSweepResult:
    """Sweep bonds% at retirement, run decumulation for each, find optimal allocation.

    For each bonds_pct:
      1. Compute required portfolio deterministically (income formula).
      2. Run fixed-withdrawal and Guyton-Klinger decumulation at that portfolio,
         with glide path from (1-bonds_pct) down to config.mc_decumulation_glide_end_pct.
      3. Record survival at key years.

    All allocations share the same random draws (same seed) so differences in survival
    rates reflect allocation differences, not Monte Carlo noise.
    """
    if bonds_pct_values is None:
        bonds_pct_values = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    # Effective tax on stock withdrawals. Conservative default: 65% gain fraction.
    if effective_stock_tax_rate is None:
        effective_stock_tax_rate = 0.65 * config.teilfreistellung * config.tax_rate

    ny  = config.mc_decumulation_years
    n   = n_simulations if n_simulations is not None else min(
        config.mc_decumulation_n_simulations, 5_000
    )
    btr = config.tax_rate_bond_italy
    glide_end = config.mc_decumulation_glide_end_pct
    glide_yrs = config.mc_decumulation_glide_years

    # Income target at retirement in nominal terms
    inflation_factor = (1.0 + config.expected_inflation_percent / 100.0) ** int(round(config.years))
    target_annual = config.target_monthly_retirement_income * inflation_factor * 12.0

    # Generate ONE set of random draws shared across all allocations
    rng = np.random.default_rng(seed)
    stock_returns, inflations, bond_yields = _generate_mc_returns(
        rng, n, ny,
        config.planning_cagr,
        config.mc_stock_return_std_percent / 100,
        config.expected_inflation_percent / 100,
        config.mc_inflation_std_percent / 100,
        config.yield_bonds / 100,
        config.mc_bond_yield_std_percent / 100,
        inflation_ar1_phi=config.mc_inflation_ar1_phi,
        stock_return_df=config.mc_stock_return_df,
        stock_bond_corr=config.mc_stock_bond_correlation,
        stock_return_ar1_phi=config.mc_stock_return_ar1_phi,
        stock_mean_uncertainty=config.mc_stock_mean_uncertainty,
        stock_vol_uncertainty=config.mc_stock_vol_uncertainty,
    )

    wd_rate = config.withdrawal_percentage_stock / 100.0
    bond_yield = config.yield_bonds / 100.0
    points: list[AllocationSweepPoint] = []

    for bonds_pct in bonds_pct_values:
        stocks_w = 1.0 - bonds_pct
        net_rate = (
            bonds_pct * bond_yield * (1.0 - btr)
            + stocks_w * wd_rate * (1.0 - effective_stock_tax_rate)
        )
        if net_rate <= 0:
            continue
        required_portfolio = target_annual / net_rate
        stock_pct_start = stocks_w
        # Only glide downward; if already at or below glide_end, no rebalancing
        eff_glide_end = min(glide_end, stock_pct_start)
        eff_glide_yrs = glide_yrs if stock_pct_start > glide_end else 0

        initial_stock_v = np.full(n, required_portfolio * stock_pct_start)
        initial_bond_v  = np.full(n, required_portfolio * bonds_pct)
        initial_monthly_wd = required_portfolio * wd_rate / 12.0

        _loop_kw = dict(
            annual_inflations=inflations,
            annual_bond_yields=bond_yields,
            initial_stock_v=initial_stock_v,
            initial_bond_v=initial_bond_v,
            initial_monthly_wd=initial_monthly_wd,
            n_years=ny,
            bond_tax_rate=btr,
            pension_monthly=config.mc_decumulation_pension_monthly,
            pension_start_year=config.mc_decumulation_pension_start_year,
            pension_adjustment_rate=config.mc_decumulation_pension_adjustment_rate,
            stock_pct_end=eff_glide_end,
            glide_years=eff_glide_yrs,
            effective_stock_tax_rate=effective_stock_tax_rate,
        )
        r_fixed = _run_decum_loop(
            annual_stock_returns=stock_returns, withdrawal_strategy="fixed", **_loop_kw
        )
        r_gk = _run_decum_loop(
            annual_stock_returns=stock_returns, withdrawal_strategy="guyton_klinger", **_loop_kw
        )

        points.append(AllocationSweepPoint(
            bonds_pct=bonds_pct,
            stock_pct_start=stock_pct_start,
            required_portfolio=required_portfolio,
            net_income_rate=net_rate,
            survival_yr20_fixed=_survival_at_year(r_fixed, 20),
            survival_yr30_fixed=_survival_at_year(r_fixed, 30),
            survival_yr30_gk=_survival_at_year(r_gk, 30),
            survival_yr35_gk=_survival_at_year(r_gk, 35),
            survival_yr40_gk=_survival_at_year(r_gk, 40),
        ))

    optimal = max(points, key=lambda p: p.survival_yr30_gk)
    return AllocationSweepResult(
        points=points,
        optimal_bonds_pct=optimal.bonds_pct,
        glide_end_pct=glide_end,
        glide_years=glide_yrs,
        base_bonds_pct=config.bonds_percentage / 100.0,
    )


def compute_chained_mc(
    config: Config,
    monthly_stock_contribution: float,
    monthly_bond_contribution: float,
    required_portfolio: float,
    n_paths: int | None = None,
    seed: int = 42,
) -> ChainedMCResult:
    """Two-phase chained Monte Carlo: accumulation → decumulation.

    Phase 1: simulate n_paths joint portfolio paths over config.years using the
    same return model as run_monte_carlo_simulation.

    Phase 2: use each path's final portfolio as starting wealth in decumulation,
    with a fixed nominal withdrawal equal to target_monthly_retirement_income
    inflated to the retirement date. No independence assumption.
    """
    n = n_paths or config.mc_decumulation_n_simulations
    ny_accum = int(round(config.years))
    ny_decum = config.mc_decumulation_years
    stock_mean = config.planning_cagr
    bond_yield = config.yield_bonds / 100
    stock_std = config.mc_stock_return_std_percent / 100
    infl_std = config.mc_inflation_std_percent / 100
    bond_std_pct = config.mc_bond_yield_std_percent / 100

    rng = np.random.default_rng(seed)

    # ── Phase 1: Accumulation ─────────────────────────────────────────────────
    stock_r_a, infl_r_a, bond_r_a = _generate_mc_returns(
        rng, n, ny_accum, stock_mean, stock_std,
        config.expected_inflation_percent / 100, infl_std,
        bond_yield, bond_std_pct,
        inflation_ar1_phi=config.mc_inflation_ar1_phi,
        stock_return_df=config.mc_stock_return_df,
        stock_bond_corr=config.mc_stock_bond_correlation,
        stock_return_ar1_phi=config.mc_stock_return_ar1_phi,
        stock_mean_uncertainty=config.mc_stock_mean_uncertainty,
        stock_vol_uncertainty=config.mc_stock_vol_uncertainty,
    )

    stock_v = np.full(n, config.current_stock_value)
    bond_v = np.full(n, config.current_bonds_value)
    cash_v = config.current_cash  # deterministic — same across paths

    cash_rate = config.interest_on_cash_pa / 100
    monthly_cash_rate = (1 + cash_rate) ** (1 / 12) - 1
    cash_fv_factor = (1 + monthly_cash_rate) ** 12
    cash_annuity = (cash_fv_factor - 1) / monthly_cash_rate if monthly_cash_rate > 1e-12 else 12.0
    annual_growth = config.annual_contribution_growth_percent / 100
    max_monthly = config.annual_net_saving / 12

    for year in range(ny_accum):
        stock_fv, stock_ann = _vec_annuity_factor(stock_r_a[:, year])
        bond_fv, bond_ann = _vec_annuity_factor(bond_r_a[:, year])
        g = (1 + annual_growth) ** year
        cs = monthly_stock_contribution * g
        cb = monthly_bond_contribution * g
        cc = max(0.0, max_monthly * g - cs - cb)
        stock_v = stock_v * stock_fv + cs * stock_ann
        bond_v = bond_v * bond_fv + cb * bond_ann
        cash_v = cash_v * cash_fv_factor + cc * cash_annuity

    cash_converted = max(0.0, cash_v - config.cash_reserve_at_horizon)
    bond_v = bond_v + cash_converted
    final_portfolios = stock_v + bond_v

    # ── Phase 2: Decumulation ─────────────────────────────────────────────────
    sa = config.mc_decumulation_stock_pct
    stock_v = final_portfolios * sa
    bond_v = final_portfolios * (1.0 - sa)

    infl_factor = (1.0 + config.expected_inflation_percent / 100) ** config.years
    target_annual_nom = config.target_monthly_retirement_income * infl_factor * 12.0

    stock_r_d, infl_r_d, bond_r_d = _generate_mc_returns(
        rng, n, ny_decum, stock_mean, stock_std,
        config.expected_inflation_percent / 100, infl_std,
        bond_yield, bond_std_pct,
        inflation_ar1_phi=config.mc_inflation_ar1_phi,
        stock_return_df=config.mc_stock_return_df,
        stock_bond_corr=config.mc_stock_bond_correlation,
        stock_return_ar1_phi=config.mc_stock_return_ar1_phi,
        stock_mean_uncertainty=config.mc_stock_mean_uncertainty,
        stock_vol_uncertainty=config.mc_stock_vol_uncertainty,
    )

    alive = np.ones(n, dtype=bool)
    cum_inf = np.ones(n)
    btr = config.tax_rate_bond_italy
    pension_annual = config.mc_decumulation_pension_monthly * 12.0
    pension_start = config.mc_decumulation_pension_start_year
    pension_adj = config.mc_decumulation_pension_adjustment_rate
    gep = config.mc_decumulation_glide_end_pct
    gyr = config.mc_decumulation_glide_years
    _glide = gyr > 0 and abs(gep - sa) > 1e-6

    survival_by_year: list[float] = []

    for year in range(ny_decum):
        stock_fv, _ = _vec_annuity_factor(stock_r_d[:, year])
        stock_v = stock_v * stock_fv
        bond_coupon = bond_v * np.maximum(bond_r_d[:, year], 0.0) * (1.0 - btr)
        cum_inf = cum_inf * (1.0 + infl_r_d[:, year])

        if _glide:
            progress = min(year + 1, gyr) / gyr
            tgt_frac = sa + (gep - sa) * progress
            total_v = stock_v + bond_v
            mask = alive & (total_v > 0.0)
            stock_v = np.where(mask, total_v * tgt_frac, stock_v)
            bond_v = np.where(mask, total_v * (1.0 - tgt_frac), bond_v)

        pension_inc = (
            pension_annual * (1.0 + pension_adj) ** (year - pension_start)
            if year >= pension_start else 0.0
        )
        target = target_annual_nom * cum_inf
        passive = bond_coupon + pension_inc
        residual = np.maximum(0.0, target - passive)
        surplus = np.maximum(0.0, passive - target)

        stock_v = stock_v - np.where(alive, residual, 0.0) + np.where(alive, surplus, 0.0)
        shortfall = np.maximum(0.0, -stock_v)
        stock_v = np.maximum(stock_v, 0.0)
        bond_v = np.maximum(bond_v - np.where(alive, shortfall, 0.0), 0.0)

        portfolio = stock_v + bond_v
        alive = alive & (portfolio > 0.0)
        survival_by_year.append(float(np.mean(alive)))

    accum_ok = final_portfolios >= required_portfolio
    accum_success_frac = float(np.mean(accum_ok))
    cond_surv_ok = float(np.mean(alive[accum_ok])) if accum_ok.any() else float("nan")
    cond_surv_fail = float(np.mean(alive[~accum_ok])) if (~accum_ok).any() else float("nan")

    return ChainedMCResult(
        n_chained_paths=n,
        required_portfolio=required_portfolio,
        joint_survival_rate=survival_by_year[-1],
        survival_by_year=survival_by_year,
        accum_success_frac=accum_success_frac,
        cond_survival_accum_success=cond_surv_ok,
        cond_survival_accum_fail=cond_surv_fail,
        starting_portfolio_percentiles={
            p: float(np.percentile(final_portfolios, p)) for p in (5, 25, 50, 75, 95)
        },
    )


def compute_swr_sweep(
    config: Config,
    display_portfolio: float,
    swr_values: list[float] | None = None,
    seed: int = 42,
    n_simulations: int | None = None,
) -> SWRSweepResult:
    """Sweep withdrawal rates at a fixed portfolio, sharing identical random draws.

    Survival rates at years 20/30/40 are computed for each SWR so that differences
    reflect only the withdrawal rate change, not Monte Carlo noise.
    """
    if swr_values is None:
        swr_values = [0.020, 0.025, 0.028, 0.030, 0.035, 0.040]
    n = n_simulations or min(config.mc_decumulation_n_simulations, 5_000)
    ny = config.mc_decumulation_years
    sa = config.mc_decumulation_stock_pct
    btr = config.tax_rate_bond_italy

    rng = np.random.default_rng(seed)
    stock_returns, inflations, bond_yields = _generate_mc_returns(
        rng, n, ny,
        config.planning_cagr,
        config.mc_stock_return_std_percent / 100,
        config.expected_inflation_percent / 100,
        config.mc_inflation_std_percent / 100,
        config.yield_bonds / 100,
        config.mc_bond_yield_std_percent / 100,
        inflation_ar1_phi=config.mc_inflation_ar1_phi,
        stock_return_df=config.mc_stock_return_df,
        stock_bond_corr=config.mc_stock_bond_correlation,
        stock_return_ar1_phi=config.mc_stock_return_ar1_phi,
        stock_mean_uncertainty=config.mc_stock_mean_uncertainty,
        stock_vol_uncertainty=config.mc_stock_vol_uncertainty,
    )

    points: list[SWRSweepPoint] = []
    for swr in swr_values:
        monthly_wd = display_portfolio * swr / 12.0
        res = _run_decum_loop(
            annual_stock_returns=stock_returns,
            annual_inflations=inflations,
            annual_bond_yields=bond_yields,
            initial_stock_v=np.full(n, display_portfolio * sa),
            initial_bond_v=np.full(n, display_portfolio * (1.0 - sa)),
            initial_monthly_wd=monthly_wd,
            n_years=ny,
            withdrawal_strategy="fixed",
            bond_tax_rate=btr,
            pension_monthly=config.mc_decumulation_pension_monthly,
            pension_start_year=config.mc_decumulation_pension_start_year,
            pension_adjustment_rate=config.mc_decumulation_pension_adjustment_rate,
            stock_pct_end=config.mc_decumulation_glide_end_pct,
            glide_years=config.mc_decumulation_glide_years,
            effective_stock_tax_rate=0.65 * config.teilfreistellung * config.tax_rate,
        )
        points.append(SWRSweepPoint(
            swr=swr,
            monthly_withdrawal=monthly_wd,
            survival_yr20=_survival_at_year(res, 20),
            survival_yr30=_survival_at_year(res, 30),
            survival_yr40=_survival_at_year(res, 40),
        ))

    return SWRSweepResult(
        points=points,
        config_swr=config.withdrawal_percentage_stock / 100,
        display_portfolio=display_portfolio,
    )
