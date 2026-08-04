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

Measured at the time of writing: 4% / 50/50 / 30y gives ~82% here against
Trinity's ~95%. Switching `mc_return_method="bootstrap"` (block-resampled
historical returns) moves it to ~85%, i.e. toward the published figure, which
is consistent with mean reversion being the explanation for the gap.

NEUTRALISED FOR THE COMPARISON
------------------------------
Features Trinity did not model are switched off: taxes, pension income, the
equity glide path, Student-t fat tails, and stock/bond correlation.

Inflation persistence is deliberately NOT switched off — see FINDINGS below.

Market assumptions are Trinity's 1926-1995 US figures: S&P 500 ≈ 10.3% nominal
(σ ≈ 20%), high-grade corporate bonds ≈ 5.2%, inflation ≈ 3.1%.

FINDINGS (first run, August 2026)
---------------------------------
1. LEVEL. 4% / 50-50 / 30y → ~82% survival against Trinity's ~95%. The gap is
   attributable to the return process, not a defect: independent parametric
   draws contain no mean reversion, so a bad decade is never followed by a
   compensating one and sequence risk bites harder than it did historically.
   Supporting evidence — switching `mc_return_method="bootstrap"` (block
   resampling of real return sequences) moves the result to ~85%, i.e. toward
   the published figure. The engine is therefore CONSERVATIVE relative to
   history, which is the correct direction of error for a planning tool.

2. THE PHI=0 TRAP. The first version of this file set
   `mc_inflation_ar1_phi = 0` on the reasoning that Trinity did not model
   inflation persistence. That inverted Trinity's central finding: an all-bond
   portfolio came out SAFER than a balanced one. The arithmetic is sound given
   the inputs — 5.2% nominal bonds against 3.1% inflation is a ~2.0% real
   annuity, which funds a 4% real withdrawal for ~35 years — but it is not
   true of real bonds. Sustained inflation is the mechanism that destroys
   nominal bond portfolios over 30 years, and switching it off removed the
   very risk the comparison exists to test.

   Survival at 4% / 30y by allocation:

       inflation model        all bonds    50/50     all stocks
       phi=0.0, sigma=2%        96.2%      83.1%       79.8%     <- inverted
       phi=0.6, sigma=2%        84.1%      81.9%       79.1%
       phi=0.6, sigma=3%        74.0%      80.1%       78.1%     <- Trinity's
                                                                    ordering

   With historically realistic inflation persistence and volatility the engine
   reproduces the published conclusion: a balanced portfolio beats both
   extremes. That the AR(1) inflation model is what restores it is meaningful
   validation of that component.

3. MODEL LIMITATION EXPOSED. The bond leg is a nominal coupon carrying no
   duration or credit risk; stochastic inflation is the only force eroding it.
   Bond-heavy allocations therefore look safer here than the historical record
   supports, and that bias grows as assumed inflation volatility falls. This
   matters directly for any plan built on a high bond share. Asserted in
   TestKnownLimitations so it cannot be silently forgotten.
"""

import pytest

from models import Config
from monte_carlo import simulate_decumulation

# ── Study parameters ─────────────────────────────────────────────────────────
TRINITY_STOCK_CAGR_PCT = 10.3      # S&P 500, 1926-1995, nominal
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

        assert 0.75 <= survival <= 0.97, (
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
        assert 2.5 <= safemax <= 6.0, (
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

    def test_without_inflation_persistence_bonds_look_too_safe(self):
        # With phi=0 and low inflation volatility, the bond leg is effectively
        # a deterministic ~2% real annuity, which funds a 4% real withdrawal
        # for ~35 years — so an all-bond portfolio outranks a balanced one,
        # contradicting the historical record. The engine's bond leg carries
        # no duration or credit risk; only stochastic inflation erodes it.
        kwargs = dict(mc_inflation_ar1_phi=0.0, mc_inflation_std_percent=2.0)
        all_bonds = _survival(4.0, stock_allocation=0.0, **kwargs)
        balanced = _survival(4.0, stock_allocation=0.50, **kwargs)

        assert all_bonds > balanced, (
            "The all-bonds-look-safest artifact has disappeared — the bond "
            "model may have been improved. Update METHODOLOGY.md and this test."
        )
