from dataclasses import dataclass, field


@dataclass
class Config:
    current_stock_value: float
    current_bonds_value: float
    current_stock_unrealized_gain: float
    current_cash: float
    cash_reserve_at_horizon: float
    interest_on_cash_pa: float
    annual_net_saving: float
    annual_contribution_growth_percent: float
    maximum_stock_monthly_investment_pct: float
    maximum_bonds_monthly_investment_pct: float
    target_monthly_retirement_income: float
    years: float
    expected_inflation_percent: float
    expected_yearly_growth_stocks_percent: float
    bonds_percentage: float
    yield_bonds: float
    tax_rate: float
    tax_rate_bond_italy: float
    withdrawal_percentage_stock: float
    basiszins: float
    freibetrag: float
    teilfreistellung: float
    optimize_grid_step: float
    mc_n_simulations: int
    mc_stock_return_std_percent: float
    mc_inflation_std_percent: float
    mc_bond_yield_std_percent: float
    mc_target_success_percent: int
    mc_inflation_ar1_phi: float = 0.0
    mc_decumulation_years: int = 40
    mc_decumulation_stock_pct: float = 0.40  # derived from bonds_percentage; kept for override
    mc_decumulation_n_simulations: int = 10_000
    mc_decumulation_pension_start_year: int = 15
    mc_decumulation_pension_monthly: float = 2_500.0
    mc_decumulation_pension_adjustment_rate: float = 0.0  # nominal annual pension growth (0 = fixed nominal)
    mc_decumulation_glide_end_pct: float = 0.0   # stock% at end of glide (0 = no glide)
    mc_decumulation_glide_years: int = 0          # years to complete glide (0 = no glide)
    # Canonical FI required portfolio criteria — see config.toml documentation
    fi_preservation_ratio: float = 0.50
    fi_preservation_confidence_baseline: float = 0.95
    fi_preservation_confidence_stress: float = 0.50
    # Teilfreistellung applied during decumulation. Defaults to 1.0 (no discount: IT
    # regime, where capital gains on stocks are taxed at full tax_rate). Set to the
    # German value (e.g. 0.7 for equity ETFs) if decumulating under DE tax rules.
    decum_teilfreistellung: float = 1.0
    mc_grid_cagr_min: float = 0.03
    mc_grid_cagr_max: float = 0.09
    mc_grid_cagr_step: float = 0.01
    mc_grid_n_simulations: int = 15_000
    mc_grid_contribution_max: float = 7_000.0
    mc_stock_return_df: float = 1000.0         # Student-t df for stocks (large = normal, 5 = fat tails)
    mc_stock_bond_correlation: float = 0.0    # Cholesky correlation between stock and bond draws
    mc_stock_return_ar1_phi: float = 0.0      # AR(1) mean reversion (negative = post-crash recovery)
    mc_stock_mean_uncertainty: float = 0.0    # prior std on stock CAGR (epistemic uncertainty)
    mc_stock_vol_uncertainty: float = 0.0     # prior std on stock volatility (epistemic uncertainty)
    # Optional CAGR override used for projection/MC planning. Stored as decimal
    # (0.04 = 4%) so the property can return it directly. None falls back to the
    # config "expected_yearly_growth_stocks_percent" baseline. Set this when the
    # CAGR realism check flags the baseline as "optimistic" vs CAPE.
    planning_cagr_override: float | None = None

    @property
    def monthly_saving(self) -> float:
        return self.annual_net_saving / 12

    @property
    def planning_cagr(self) -> float:
        """Stock CAGR used for projection / MC planning. Equals the override if
        set (incl. zero or negative), otherwise the config baseline."""
        if self.planning_cagr_override is not None:
            return self.planning_cagr_override
        return self.expected_yearly_growth_stocks_percent / 100.0

    @classmethod
    def from_dict(cls, config: dict[str, object]) -> "Config":
        return cls(
            current_stock_value=float(config["current_stock_value"]),
            current_bonds_value=float(config.get("current_bonds_value", 0.0)),
            current_stock_unrealized_gain=float(config.get("current_stock_unrealized_gain", 0.0)),
            current_cash=float(config.get("current_cash", 0.0)),
            cash_reserve_at_horizon=float(config.get("cash_reserve_at_horizon", 0.0)),
            interest_on_cash_pa=float(config.get("interest_on_cash_pa", 0.0)),
            annual_net_saving=float(config.get("annual_net_saving", 0.0)),
            annual_contribution_growth_percent=float(config.get("annual_contribution_growth_percent", 0.0)),
            maximum_stock_monthly_investment_pct=float(
                config.get("maximum_stock_monthly_investment_pct", 0.0)
            ),
            maximum_bonds_monthly_investment_pct=float(
                config.get("maximum_bonds_monthly_investment_pct", 0.0)
            ),
            target_monthly_retirement_income=float(config["target_monthly_retirement_income"]),
            years=float(config["years"]),
            expected_inflation_percent=float(config["expected_inflation_percent"]),
            expected_yearly_growth_stocks_percent=float(config["expected_yearly_growth_stocks_percent"]),
            bonds_percentage=float(config["bonds_percentage"]),
            yield_bonds=float(config["yield_bonds"]),
            tax_rate=float(config["tax_rate"]),
            tax_rate_bond_italy=float(config.get("tax_rate_bond_italy", config["tax_rate"])),
            withdrawal_percentage_stock=float(config["withdrawal_percentage_stock"]),
            basiszins=float(config.get("basiszins", 0.0229)),
            freibetrag=float(config.get("freibetrag", 1000.0)),
            teilfreistellung=float(config.get("teilfreistellung", 0.7)),
            optimize_grid_step=float(config.get("optimize_grid_step", 50000.0)),
            mc_n_simulations=int(config.get("mc_n_simulations", 10000)),
            mc_stock_return_std_percent=float(config.get("mc_stock_return_std_percent", 15.0)),
            mc_inflation_std_percent=float(config.get("mc_inflation_std_percent", 3.0)),
            mc_bond_yield_std_percent=float(config.get("mc_bond_yield_std_percent", 1.0)),
            mc_target_success_percent=int(config.get("mc_target_success_percent", 95)),
            mc_grid_cagr_min=float(config.get("mc_grid_cagr_min", 0.03)),
            mc_grid_cagr_max=float(config.get("mc_grid_cagr_max", 0.09)),
            mc_grid_cagr_step=float(config.get("mc_grid_cagr_step", 0.01)),
            mc_inflation_ar1_phi=float(config.get("mc_inflation_ar1_phi", 0.0)),
            mc_decumulation_years=int(config.get("mc_decumulation_years", 40)),
            mc_decumulation_stock_pct=float(config.get(
                "mc_decumulation_stock_pct",
                1.0 - float(config["bonds_percentage"]) / 100.0,
            )),
            mc_decumulation_n_simulations=int(config.get("mc_decumulation_n_simulations", 10_000)),
            mc_decumulation_pension_start_year=int(config.get("mc_decumulation_pension_start_year", 15)),
            mc_decumulation_pension_monthly=float(config.get("mc_decumulation_pension_monthly", 2_500.0)),
            mc_decumulation_pension_adjustment_rate=float(config.get("mc_decumulation_pension_adjustment_rate", 0.0)),
            mc_decumulation_glide_end_pct=float(config.get(
                "mc_decumulation_glide_end_pct",
                1.0 - float(config["bonds_percentage"]) / 100.0,  # default = start = no glide
            )),
            mc_decumulation_glide_years=int(config.get("mc_decumulation_glide_years", 0)),
            fi_preservation_ratio=float(config.get("fi_preservation_ratio", 0.50)),
            fi_preservation_confidence_baseline=float(config.get("fi_preservation_confidence_baseline", 0.95)),
            fi_preservation_confidence_stress=float(config.get("fi_preservation_confidence_stress", 0.50)),
            decum_teilfreistellung=float(config.get("decum_teilfreistellung", 1.0)),
            mc_grid_n_simulations=int(config.get("mc_grid_n_simulations", 15_000)),
            mc_grid_contribution_max=float(config.get("mc_grid_contribution_max", 7_000.0)),
            mc_stock_return_df=float(config.get("mc_stock_return_df", 1000.0)),
            mc_stock_bond_correlation=float(config.get("mc_stock_bond_correlation", 0.0)),
            mc_stock_return_ar1_phi=float(config.get("mc_stock_return_ar1_phi", 0.0)),
            mc_stock_mean_uncertainty=float(config.get("mc_stock_mean_uncertainty", 0.0)),
            mc_stock_vol_uncertainty=float(config.get("mc_stock_vol_uncertainty", 0.0)),
            # planning_cagr_override is CLI-only (per-run) — never read from config.toml
        )

    @staticmethod
    def parse_planning_cagr(raw: object) -> float | None:
        """Parse a CLI --cagr argument. Accepts:
          0.04, 4.0, "4.0", "4%", "" / None → returns decimal float, or None.
        Values > 1 are interpreted as percent (4.0 → 0.04); ≤ 1 as already decimal."""
        if raw is None or raw == "":
            return None
        try:
            s = str(raw).strip().rstrip("%")
            if not s:
                return None
            v = float(s)
        except (TypeError, ValueError):
            return None
        return v / 100.0 if v > 1.0 else v


@dataclass
class ScenarioResult:
    name: str
    stock_tax_rate: float
    bond_tax_rate: float
    stock_gain_fraction: float
    effective_stock_tax_rate: float
    stock_teilfreistellung: float
    years: float
    current_stock_value: float
    stock_expected_yield: float
    bonds_percentage: float
    yield_bonds: float
    withdrawal_rate: float
    portfolio_gross_income_rate: float
    portfolio_net_income_rate: float
    future_monthly_income_value: float
    target_annual_net_income: float
    required_annual_gross_income_from_portfolio: float
    required_total_portfolio: float
    required_stock_amount: float
    required_bonds_amount: float
    generated_annual_gross_income: float
    generated_annual_net_income: float
    required_monthly_contribution: float
    required_monthly_bonds_contribution: float


@dataclass
class MonteCarloResult:
    name: str
    n_simulations: int
    stock_std: float
    inflation_std: float
    success_rate: float
    final_portfolio_percentiles: dict[int, float]
    required_portfolio_percentiles: dict[int, float]
    surplus_percentiles: dict[int, float]
    portfolio_net_income_rate: float
    final_stock_percentiles: dict[int, float]
    final_bond_percentiles: dict[int, float]
    bond_std: float
    success_rate_income_only: float
    success_rate_stock_only: float = 0.0   # P(stocks ≥ required stocks)
    success_rate_bond_only: float = 0.0    # P(bonds ≥ required bonds)


@dataclass
class ProjectionResult:
    years: float
    stock_expected_yield: float
    cash_interest_rate: float
    annual_net_saving: float
    annual_contribution_growth_percent: float
    maximum_stock_monthly_investment_pct: float
    maximum_bonds_monthly_investment_pct: float
    initial_stock_value: float
    initial_stock_unrealized_gain: float
    initial_cash_value: float
    monthly_stock_contribution: float
    monthly_bonds_contribution: float
    monthly_cash_contribution: float
    final_stock_value: float
    final_bonds_from_contributions: float
    final_cash_before_conversion: float
    final_cash_converted_to_bonds: float
    final_remaining_cash: float
    final_bonds_value: float
    final_portfolio_value: float
    cost_basis_stock: float
    total_vorabpauschale: float
    total_vorabpauschale_tax: float


@dataclass
class ProjectionYieldScenarioResult:
    name: str
    stock_tax_rate: float
    bond_tax_rate: float
    annual_gross_income_stock: float
    annual_net_income_stock: float
    annual_gross_income_bond: float
    annual_net_income_bond: float
    annual_gross_income_total: float
    annual_net_income_total: float
    bond_net_contribution_share: float
    stock_gain_fraction: float
    effective_stock_tax_rate: float
    portfolio_gross_yield: float
    portfolio_net_yield: float


@dataclass
class FinancialFreedomResult:
    years: int | None
    success_rate: float
    portfolio_p50: float
    portfolio_tail_pct: float
    tail_pct: int
    monthly_income_p50: float


@dataclass
class YearsToTargetsResult:
    years_to_stocks: int | None
    years_to_bonds: int | None
    years_to_both: int | None
    years_to_income: int | None


@dataclass
class AllocationGridPoint:
    extra_stocks: float
    extra_bonds: float
    new_stocks: float
    new_bonds: float
    new_cash: float
    monthly_stocks: float       # min stocks/mo for Q1 hard target; inf if not achievable
    monthly_bonds: float
    total_monthly: float
    achievable: bool
    best_success_rate: float = 0.0
    income_stock_mo: float = 0.0   # optimal stocks/mo for income success (optimised split)
    income_bonds_mo: float = 0.0   # optimal bonds/mo for income success (optimised split)
    portfolio_ceiling_rate: float = 0.0  # P(both) at max stock contribution — best achievable for Q1
    ceiling_stocks_mo: float = 0.0       # stocks/mo that produced portfolio_ceiling_rate
    ceiling_bonds_mo: float = 0.0        # bonds/mo that produced portfolio_ceiling_rate
    q1_success_rate: float = 0.0  # P(both) measured at (monthly_stocks, monthly_bonds) — the achieved rate at the optimum


@dataclass
class OptimalAllocationResult:
    scenario_name: str
    target_success: float
    min_cash_today: float
    grid: list[AllocationGridPoint]
    optimal: AllocationGridPoint | None


@dataclass
class CagrContributionCell:
    success_hard: float         # stocks AND bonds both reach required allocation
    success_income: float       # portfolio income >= target, any composition
    success_stock_only: float = 0.0  # P(stocks >= required stocks)
    stock_tail_pct: float = 0.0      # stock value at the bad-luck tail percentile


@dataclass
class CombinedGridCell:
    success_combined: float  # P(stocks >= required AND bonds >= required)


@dataclass
class CombinedGrid:
    scenario_name: str
    cagr_values: list[float]
    stocks_values: list[float]
    bonds_values: list[float]
    required_stocks: float
    required_bonds: float
    suggested_stocks: float
    suggested_bonds: float
    grid: list[list[list[CombinedGridCell]]]  # [cagr_idx][bonds_idx][stocks_idx]


@dataclass
class BondStockCell:
    success_bond: float   # P(bonds at horizon >= required bonds)


@dataclass
class BondStockGrid:
    scenario_name: str
    stocks_values: list[float]   # x-axis (columns): monthly stock contributions
    bonds_values: list[float]    # y-axis (rows): monthly bond contributions
    required_bonds: float
    suggested_stocks: float      # deterministic required monthly stocks
    suggested_bonds: float       # deterministic required monthly bonds
    grid: list[list[BondStockCell]]  # [bonds_idx][stocks_idx]


@dataclass
class SensitivityEntry:
    parameter: str
    base_value: str
    delta_label: str
    success_base: float
    success_up: float    # success rate at +delta
    success_down: float  # success rate at -delta


@dataclass
class SensitivityWaterfallResult:
    scenario_name: str
    entries: list["SensitivityEntry"]  # sorted by max impact descending


@dataclass
class ContributionCagrGrid:
    scenario_name: str
    bonds_monthly: float          # fixed bonds contribution for this sub-grid
    bonds_multiplier: int         # 1 or 2 (× deterministic bonds)
    cagr_values: list[float]
    contribution_values: list[float]   # stock monthly € axis
    suggested_contribution: float      # unrounded deterministic suggestion
    grid: list[list[CagrContributionCell]]  # [cagr_idx][contrib_idx]


@dataclass
class DecumulationResult:
    n_simulations: int
    n_years: int
    withdrawal_strategy: str           # "fixed" | "guyton_klinger"
    initial_portfolio: float
    initial_monthly_withdrawal: float
    stock_allocation: float
    survival_rate: float               # P(portfolio > 0 at year n_years)
    survival_by_year: list[float]      # length n_years
    portfolio_percentiles_by_year: list[dict[int, float]]   # {3,25,50,75,97} per year
    withdrawal_percentiles_by_year: list[dict[int, float]]  # {10,50,90} per year
    ruin_year_median: int | None       # median year of ruin among failed sims
    final_portfolios: list[float] | None = None  # per-path wealth at year n_years; opt-in


@dataclass
class SORRResult:
    baseline: "DecumulationResult"
    stress_scenarios: list[tuple[str, "DecumulationResult"]]  # (label, result)


@dataclass
class AllocationSweepPoint:
    bonds_pct: float            # bonds allocation at retirement start
    stock_pct_start: float      # = 1 - bonds_pct (glide path start)
    required_portfolio: float   # deterministic required portfolio for this allocation
    net_income_rate: float      # portfolio net income rate
    survival_by_year_fixed: list[float]   # length = horizon, fixed strategy
    survival_by_year_gk: list[float]      # length = horizon, GK strategy


@dataclass
class AllocationSweepResult:
    points: list[AllocationSweepPoint]
    optimal_bonds_pct: float    # bonds_pct with highest yr-{horizon} GK survival
    glide_end_pct: float        # stock% at end of glide
    glide_years: int            # years to complete glide
    base_bonds_pct: float       # bonds_pct from config (current plan)
    horizon_years: int          # decumulation horizon (config-derived)
    pension_start_year: int     # pension start year (config-derived; used for milestone selection)


@dataclass
class InflationCagrCell:
    success_income: float   # P(portfolio income >= inflation-adjusted target)


@dataclass
class InflationCagrGrid:
    scenario_name: str
    cagr_values: list[float]       # x-axis (decimal, e.g. 0.05)
    inflation_values: list[float]  # y-axis (decimal, e.g. 0.02)
    base_cagr: float               # config expected_yearly_growth_stocks_percent / 100
    base_inflation: float          # config expected_inflation_percent / 100
    monthly_stocks: float
    monthly_bonds: float
    grid: list[list[InflationCagrCell]]  # [inflation_idx][cagr_idx]


@dataclass
class DecumInflationCagrCell:
    survival_by_year: list[float]   # length = horizon


@dataclass
class DecumInflationCagrGrid:
    display_portfolio: float
    cagr_values: list[float]        # x-axis (decimal)
    inflation_values: list[float]   # y-axis (decimal)
    base_cagr: float
    base_inflation: float
    grid: list[list[DecumInflationCagrCell]]  # [inflation_idx][cagr_idx]
    horizon_years: int              # decumulation horizon (config-derived)
    pension_start_year: int         # used for milestone selection


@dataclass
class SWRSweepPoint:
    swr: float              # annual withdrawal rate (decimal, e.g. 0.028)
    monthly_withdrawal: float
    survival_by_year: list[float]   # length = horizon


@dataclass
class SWRSweepResult:
    points: list[SWRSweepPoint]
    config_swr: float           # SWR from config (decimal)
    display_portfolio: float    # portfolio used for all rows
    horizon_years: int          # decumulation horizon (config-derived)
    pension_start_year: int     # used for milestone selection


@dataclass
class ChainedMCResult:
    """Result of a two-phase chained accumulation → decumulation Monte Carlo."""
    n_chained_paths: int
    required_portfolio: float
    joint_survival_rate: float           # P(survive ny_decum years | start from accum distribution)
    survival_by_year: list[float]        # length = mc_decumulation_years
    accum_success_frac: float            # P(accumulated portfolio >= required_portfolio)
    cond_survival_accum_success: float   # P(survive | accum >= required)
    cond_survival_accum_fail: float      # P(survive | accum < required)
    starting_portfolio_percentiles: dict[int, float]   # P5/P25/P50/P75/P95 of year-N wealth
    # Optional secondary threshold for the "Paths reaching ≥ required" row only —
    # used to display the stricter accum income-formula bar alongside the looser
    # decum survival-minimum bar. Both fields populated together or both None.
    secondary_required_portfolio: float | None = None
    secondary_required_label: str | None = None
    secondary_accum_success_frac: float | None = None


@dataclass
class WealthPreservationRow:
    """One scenario row of the SORR × wealth preservation matrix.

    For each end-wealth ratio, reports the best P(final ≥ ratio × initial) achievable
    across the bonds-allocation grid, plus the allocation that achieves it.
    """
    scenario_label: str
    best_p_by_ratio: list[float]         # parallel to end_wealth_ratios
    best_alloc_by_ratio: list[float]     # bonds_pct that achieves best_p_by_ratio
    p_at_current_by_ratio: list[float]   # P at current config allocation


@dataclass
class WealthPreservationUnderSORR:
    """SORR × end-wealth preservation matrix at fixed starting wealth.

    Replaces the survival-centric framing with a preservation-centric one: for each
    SORR scenario × end-wealth target, what's the best P of preserving that fraction
    of starting wealth, and what allocation achieves it?
    """
    horizon_years: int
    initial_wealth: float                # fixed starting portfolio
    target_success_pct: int              # threshold for ✓/✗ on individual cells
    current_bonds_pct: float
    bonds_pct_grid: list[float]
    end_wealth_ratios: list[float]       # column headers as decimal (0.50 → 1.10)
    rows: list[WealthPreservationRow]    # baseline + 3 SORR scenarios = 4 rows
    # Pre-pension bridge milestone (baseline scenario, current allocation only)
    bridge_year: int
    bridge_p3: float
    bridge_p50: float
    bridge_p97: float
    bridge_survival: float


@dataclass
class CanonicalFIRequiredRow:
    """One row of the canonical FI required scan: smallest wealth W where
    P(final ≥ preserve_ratio × W) clears both confidence thresholds at the
    optimal allocation."""
    preserve_ratio: float                # X (decimal: 0.50 → 1.00)
    required_wealth_eur: float           # W*(X) — smallest wealth satisfying both bounds
    optimal_bonds_pct: float             # allocation that achieves it (decimal)
    achieved_baseline_p: float           # P(final ≥ X·W) under baseline at the optimum
    achieved_stress_p: float             # P(final ≥ X·W) under worst-SORR at the optimum
    reachable: bool                      # False if no W in search range satisfies both


@dataclass
class CanonicalFIRequiredScan:
    """Scan of W*(X) for X ∈ {0.50, 0.60, …, 1.00}, plus the canonical pick.

    The canonical row is the one whose preserve_ratio matches
    config.fi_preservation_ratio — that row's required_wealth_eur is the bar
    accumulation targets.
    """
    horizon_years: int
    target_monthly_withdrawal: float
    confidence_baseline: float           # Y from config
    confidence_stress: float             # Z from config
    worst_stress_label: str              # which SORR sequence binds
    bonds_pct_grid: list[float]          # allocations searched
    rows: list[CanonicalFIRequiredRow]   # one per preserve_ratio
    canonical_ratio: float               # = config.fi_preservation_ratio
    canonical_required_wealth: float     # = rows[canonical_ratio].required_wealth_eur
    canonical_optimal_bonds_pct: float   # alloc that achieves the canonical bar
    canonical_reachable: bool            # was the canonical bar achievable in search range?


@dataclass
class PlanVerdictSummary:
    """Typed values from main_cagr_analysis Plan Verdict block — used by Plan Snapshot."""
    required_portfolio: float                # income-formula required portfolio at horizon
    projected_portfolio_config: float        # projected with config stock CAGR
    projected_portfolio_cape: float | None   # projected with CAPE-implied CAGR (None when CAPE unavailable)
    gap_config: float                        # projected_config − required (positive = surplus)
    on_track: bool                           # gap_config >= 0
    joint_survival_rate: float | None        # chained MC P(accumulate AND survive), if computed


@dataclass
class CagrRealismSnapshot:
    """One-row realism check: does the world produce returns close to what we need?
    All values nominal, EUR. None where data is insufficient."""
    cape_nominal_cagr: float | None          # CAPE-implied forward CAGR
    sp500_10y_rolling_eur_p50: float | None  # P50 of rolling 10y EUR CAGRs (1988–today)
    sp500_10y_rolling_eur_p5:  float | None  # P5 of same — entry-point bad-luck floor
    optimised_etf_p50: float | None          # selected strategy's rolling 10y P50
    optimised_etf_p5:  float | None          # selected strategy's rolling 10y P5 (None if history < 10y)


@dataclass
class DecumulationSummary:
    """Typed values from main_decumulation — used by Plan Snapshot."""
    required_portfolio_fi_canonical: float   # canonical FI bar (preservation-calibrated); see fi_preservation_* config
    target_monthly_withdrawal: float         # nominal monthly withdrawal at horizon
    n_years: int                              # decumulation horizon
    survival_fixed_final: float              # final-year survival, fixed-withdrawal strategy
    survival_gk_final: float                 # final-year survival, Guyton-Klinger strategy
    joint_survival_chained: float            # chained accum→decum joint probability
