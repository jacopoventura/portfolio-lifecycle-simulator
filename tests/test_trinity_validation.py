"""External validation: does the engine agree with published withdrawal research?

Every other test in this suite checks the engine against *itself* — internal
consistency, invariants, refactor safety. None of them can catch a
systematically wrong model. These check it against two published results:

  * Bengen (1994), "Determining Withdrawal Rates Using Historical Data",
    Journal of Financial Planning — origin of the 4% rule; SAFEMAX (highest
    withdrawal rate surviving every historical 30-year window) ≈ 4.15%.

  * Cooley, Hubbard & Walz (1998), "Retirement Savings: Choosing a Withdrawal
    Rate That Is Sustainable" (the "Trinity study") — a 4% inflation-adjusted
    withdrawal from a 50/50 portfolio succeeded in ~95% of 30-year periods,
    1926-1995.

WHAT IS CLAIMED
---------------
Not bit-exact reproduction — that is impossible and not the goal. Trinity
resampled actual historical sequences; this engine draws from a parametric
process. Real sequences mean-revert; independent draws do not, which makes
sequence risk strictly harsher here. The engine therefore lands BELOW the
published success rate by design, and that is the acceptable direction: a
planning tool that is more pessimistic than history is safe, one that is more
optimistic is dangerous. The test asserts the result sits in the right region
and reproduces the qualitative structure of the literature.

Measured (October 2026 engine): 4% / 50/50 / 30y gives ~93% here against
Trinity's ~95%. The remaining ~2 points are plausibly the mean reversion that
independent parametric draws lack.

NEUTRALISED FOR THE COMPARISON
------------------------------
Features Trinity did not model are switched off: taxes, pension income, the
equity glide path, Student-t fat tails, and stock/bond correlation.

Inflation persistence is deliberately NOT switched off — see FINDINGS below.

Market assumptions are Trinity's 1926-1995 US figures: S&P 500 ≈ 10.3% nominal
compounded (σ ≈ 20%), high-grade corporate bonds ≈ 5.2%, inflation ≈ 3.1%.

FINDINGS
--------
1. THE FIRST RUN (August 2026) GAVE ~80%, AND THE CAUSE WAS SETUP, NOT THE
   RETURN PROCESS. It was attributed to missing mean reversion. Re-running one
   change at a time (October 2026) showed two engine conventions explained
   most of the gap:

       setup                                                   4% / 50-50 / 30y
       10.3% used as ARITHMETIC mean, no rebalancing (old)           80.1%
       10.3% treated as a CAGR (arith. mean ≈ 12.3%)                 89.3%
       annual rebalancing to 50/50                                   86.2%
       both (current engine)                                         ~93%
       Trinity, published                                            95%

   (a) A CAGR used as the arithmetic mean of annual returns compounds at
       ~CAGR − σ²/2 (≈ 2pp/yr at σ = 20%). The engine now converts the input
       CAGR to the arithmetic mean that compounds at it
       (`mc_stock_mean_is_cagr`, default True; see cagr_to_arithmetic_mean).
   (b) Without a glide path the loop used to never rebalance — "50/50" meant
       spend stocks first, bonds as last resort. Trinity and Bengen rebalanced
       annually; the engine now does too.

2. THE PHI=0 TRAP. The first version of this file set
   `mc_inflation_ar1_phi = 0`, which made an all-bond portfolio look SAFER
   than a balanced one — inverting Trinity's central finding. Inflation
   persistence is what erodes nominal bonds over 30 years, so it stays on.
   Under the current engine the inversion no longer appears even at phi=0
   (all bonds ≈ 90% vs 50/50 ≈ 95%): stocks compound at the stated CAGR and
   bond yields now rise with inflation (persistent, inflation-linked yield
   model, see monte_carlo._generate_mc_returns).

   Survival at 4% / 30y by allocation, current engine:

       all bonds    50/50     all stocks
         ~71%       ~93%        ~88%

3. MODEL LIMITATION STILL PRESENT. The engine ranks all-stocks BELOW 50/50,
   whereas the Trinity tables show stock-heavy portfolios at least as safe as
   50/50 at 4% / 30y. Independent annual draws have no long-horizon mean
   reversion, so a 100%-stock path that starts badly never "earns it back" the
   way historical sequences did. Asserted in TestKnownLimitations.

4. BENGEN. Bengen's SAFEMAX (~4.15%) is the rate that survived EVERY
   historical 30-year window — a worst case over ~50 overlapping windows.
   The test below uses the highest rate with ≥95% Monte Carlo survival,
   which is a different statistic; it currently sits at ~3.75%. The band is
   therefore a plausibility check, not a reproduction.
"""

import pytest

from models import Config
from monte_carlo import simulate_decumulation

# ── Study parameters ─────────────────────────────────────────────────────────
TRINITY_STOCK_CAGR_PCT = 10.3      # S&P 500, 1926-1995, nominal, compounded
TRINITY_STOCK_STD_PCT = 20.0
TRINITY_BOND_YIELD_PCT = 5.2       # long-term high-grade corporate bonds
TRINITY_INFLATION_PCT = 3.1        # US CPI, same period
TRINITY_INFLATION_STD_PCT = 3.0    # historical CPI volatility
TRINITY_INFLATION_PHI = 0.6        # inflation persistence (the 1970s mechanism)
TRINITY_HORIZON_YEARS = 30
TRINITY_STOCK_ALLOCATION = 0.50
TRINITY_PORTFOLIO = 1_000_000.0

TRINITY_PUBLISHED_SUCCESS = 0.95   # Cooley/Hubbard/Walz, 50/50, 30y, 4%
BENGEN_SAFEMAX_PCT = 4.15

N_SIMS = 10_000
SEED = 42


def _trinity_config(**overrides) -> Config:
    """Config matching Trinity's assumptions, engine-specific extras removed."""
    defaults = dict(
        # accumulation fields — required by the dataclass, unused in decumulation
        current_stock_value=0.0,
        current_bonds_value=0.0,
        current_stock_unrealized_gain=0.0,
        current_cash=0.0,
        cash_reserve_at_horizon=0.0,
        interest_on_cash_pa=0.0,
        annual_net_saving=0.0,
        annual_contribution_growth_percent=0.0,
        stock_monthly_investment_pct=0.5,
        bonds_monthly_investment_pct=0.5,
        target_monthly_retirement_income=0.0,
        years=1.0,
        optimize_grid_step=50_000.0,
        mc_n_simulations=100,
        mc_target_success_percent=95,
        basiszins=0.0,
        freibetrag=0.0,
        teilfreistellung=1.0,
        # Trinity market assumptions
        expected_yearly_growth_stocks_percent=TRINITY_STOCK_CAGR_PCT,
        mc_stock_return_std_percent=TRINITY_STOCK_STD_PCT,
        yield_bonds=TRINITY_BOND_YIELD_PCT,
        expected_inflation_percent=TRINITY_INFLATION_PCT,
        mc_inflation_std_percent=TRINITY_INFLATION_STD_PCT,
        mc_inflation_ar1_phi=TRINITY_INFLATION_PHI,
        bonds_percentage=(1.0 - TRINITY_STOCK_ALLOCATION) * 100,
        withdrawal_percentage_stock=4.0,
        mc_bond_yield_std_percent=0.5,
        # features Trinity did not model
        tax_rate=0.0,
        tax_rate_bond_italy=0.0,
        decum_teilfreistellung=1.0,
        mc_decumulation_pension_monthly=0.0,
        mc_decumulation_pension_start_year=999,
        mc_decumulation_pension_adjustment_rate=0.0,
        mc_decumulation_glide_end_pct=0.0,
        mc_decumulation_glide_years=0,
        mc_stock_return_df=1000.0,          # ≈ normal
        mc_stock_return_ar1_phi=0.0,
        mc_stock_bond_correlation=0.0,
        mc_stock_mean_uncertainty=0.0,
        mc_stock_vol_uncertainty=0.0,
        mc_return_method="parametric",
        mc_decumulation_years=TRINITY_HORIZON_YEARS,
        mc_decumulation_stock_pct=TRINITY_STOCK_ALLOCATION,
        mc_decumulation_n_simulations=N_SIMS,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _survival(withdrawal_rate_pct: float,
              stock_allocation: float = TRINITY_STOCK_ALLOCATION,
              n_years: int = TRINITY_HORIZON_YEARS,
              **config_overrides) -> float:
    """P(portfolio > 0 at horizon) under a fixed, inflation-adjusted withdrawal."""
    config = _trinity_config(withdrawal_percentage_stock=withdrawal_rate_pct,
                             **config_overrides)
    return simulate_decumulation(
        config,
        initial_portfolio=TRINITY_PORTFOLIO,
        stock_allocation=stock_allocation,
        n_years=n_years,
        seed=SEED,
        n_simulations=N_SIMS,
        withdrawal_strategy="fixed",     # inflation-adjusted real withdrawal
        bond_tax_rate=0.0,
        effective_stock_tax_rate=0.0,
        glide_end_pct=0.0,
        glide_years=0,
    ).survival_rate


class TestPublishedLevels:
    """Absolute results, compared against the literature."""

    def test_4pct_50_50_30y_sits_in_the_published_region(self):
        survival = _survival(4.0)

        assert 0.88 <= survival <= 0.97, (
            f"4% / 50-50 / 30y gave {survival:.1%}; Trinity reports "
            f"{TRINITY_PUBLISHED_SUCCESS:.0%}. Outside this band the "
            f"discrepancy is too large to attribute to the return process."
        )

    def test_engine_is_conservative_relative_to_history(self):
        # Independent draws carry no mean reversion, so the engine should not
        # be MORE optimistic than historical resampling. If it ever is, the
        # return generator is producing sequences kinder than the past.
        assert _survival(4.0) <= TRINITY_PUBLISHED_SUCCESS

    def test_bengen_safemax_lies_in_the_published_range(self):
        rate, safemax = 2.0, None
        while rate <= 7.0:
            if _survival(rate) >= 0.95:
                safemax = rate
            rate += 0.25

        assert safemax is not None, "no withdrawal rate reached 95% survival"
        # MC 95%-survival rate vs Bengen's all-windows worst case: related but
        # not identical statistics (see FINDINGS 4), hence the band.
        assert 3.25 <= safemax <= 4.75, (
            f"engine SAFEMAX = {safemax:.2f}%; Bengen reports "
            f"~{BENGEN_SAFEMAX_PCT}%"
        )


class TestQualitativeStructure:
    """Relationships that must hold whatever the return process."""

    @pytest.mark.parametrize("lower,higher", [(3.0, 4.0), (4.0, 5.0), (5.0, 6.0)])
    def test_higher_withdrawal_rate_lowers_survival(self, lower, higher):
        assert _survival(lower) > _survival(higher)

    def test_longer_horizon_lowers_survival(self):
        assert _survival(5.0, n_years=15) > _survival(5.0, n_years=30)

    def test_balanced_portfolio_beats_all_bonds(self):
        # Trinity's central finding: for a 30-year inflation-adjusted
        # withdrawal, an all-bond portfolio is not the safe choice — nominal
        # bonds cannot outpace sustained inflation. Reproduced here only
        # because inflation persistence is modelled (see TestKnownLimitations).
        assert _survival(4.0, stock_allocation=0.50) > _survival(4.0, stock_allocation=0.0)


class TestKnownLimitations:
    """Documented model artifacts, asserted so they cannot be silently lost.

    If a test here fails, the model has changed and METHODOLOGY.md needs
    updating — that is the point of encoding them.
    """

    def test_all_stocks_ranks_below_balanced(self):
        # Historical sequences mean-revert over decades, so stock-heavy
        # portfolios did at least as well as 50/50 at 4% / 30y. Independent
        # annual draws do not mean-revert, so here all-stocks falls BELOW
        # 50/50 (~88% vs ~93%). See FINDINGS 3.
        all_stocks = _survival(4.0, stock_allocation=1.0)
        balanced = _survival(4.0, stock_allocation=0.50)

        assert all_stocks < balanced, (
            "All-stocks no longer ranks below 50/50 — the return process may "
            "now carry long-horizon mean reversion. Update METHODOLOGY.md and "
            "this test."
        )


class TestResolvedArtifacts:
    """Former known limitations, now fixed — kept so they cannot come back."""

    def test_bonds_no_longer_look_safest_without_inflation_persistence(self):
        # Used to fail the other way: with phi=0, all bonds (~96%) beat
        # 50/50 (~83%). Resolved by the CAGR conversion, annual rebalancing
        # and inflation-linked bond yields (October 2026).
        kwargs = dict(mc_inflation_ar1_phi=0.0, mc_inflation_std_percent=2.0)
        all_bonds = _survival(4.0, stock_allocation=0.0, **kwargs)
        balanced = _survival(4.0, stock_allocation=0.50, **kwargs)

        assert balanced > all_bonds

    def test_legacy_conventions_reproduce_the_old_pessimism(self):
        # The arithmetic-mean convention alone costs several points of survival.
        legacy = _survival(4.0, mc_stock_mean_is_cagr=False)
        current = _survival(4.0)

        assert current - legacy > 0.04
