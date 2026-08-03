"""Property tests: invariants that any healthy MC run must satisfy.

These catch silent regressions when MC internals change. Each test uses a fixed
seed and small n_simulations (200-500) so shared RNG draws make the comparisons
deterministic — tightening monotonicity bounds beyond what raw MC noise would allow.
"""
import numpy as np
import pytest

from models import Config
from monte_carlo import (
    compute_canonical_fi_required_scan,
    compute_chained_mc,
    compute_crash_timing_matrix,
    compute_sorr_scenarios,
    compute_survival_matrix,
    compute_two_bucket_required_wealth,
    compute_wealth_preservation_under_sorr,
    simulate_decumulation,
)


def _make_config(**overrides) -> Config:
    defaults = dict(
        current_stock_value=100_000.0,
        current_bonds_value=10_000.0,
        current_stock_unrealized_gain=20_000.0,
        current_cash=50_000.0,
        cash_reserve_at_horizon=10_000.0,
        interest_on_cash_pa=2.0,
        annual_net_saving=24_000.0,
        annual_contribution_growth_percent=0.0,
        stock_monthly_investment_pct=0.5,
        bonds_monthly_investment_pct=0.3,
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
    defaults.update(overrides)
    return Config(**defaults)


_PORTFOLIO = 1_500_000.0
_WITHDRAWAL = 3_500.0
_N = 300
_SEED = 42


# ── Bounds: all probabilities are in [0, 1] ──────────────────────────────────


class TestBounds:
    def test_simulate_decumulation_survival_in_unit_interval(self):
        config = _make_config()
        result = simulate_decumulation(
            config, initial_portfolio=_PORTFOLIO, seed=_SEED, n_simulations=_N,
            monthly_withdrawal_override=_WITHDRAWAL,
        )
        assert 0.0 <= result.survival_rate <= 1.0
        for s in result.survival_by_year:
            assert 0.0 <= s <= 1.0

    def test_chained_mc_probabilities_in_unit_interval(self):
        config = _make_config()
        chained = compute_chained_mc(
            config=config,
            monthly_stock_contribution=1_000.0,
            monthly_bond_contribution=500.0,
            required_portfolio=1_500_000.0,
            n_paths=_N, seed=_SEED,
        )
        assert 0.0 <= chained.joint_survival_rate <= 1.0
        assert 0.0 <= chained.accum_success_frac <= 1.0
        for s in chained.survival_by_year:
            assert 0.0 <= s <= 1.0

    def test_wealth_preservation_probabilities_in_unit_interval(self):
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        surface = compute_wealth_preservation_under_sorr(
            config, initial_wealth=_PORTFOLIO, target_monthly_nom=target_monthly,
            seed=_SEED, n_simulations=_N,
        )
        for row in surface.rows:
            for p in row.best_p_by_ratio:
                assert 0.0 <= p <= 1.0
            for p in row.p_at_current_by_ratio:
                assert 0.0 <= p <= 1.0

    def test_canonical_scan_confidences_in_unit_interval(self):
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        scan = compute_canonical_fi_required_scan(
            config, target_monthly, seed=_SEED, n_simulations=_N,
        )
        for row in scan.rows:
            assert 0.0 <= row.achieved_baseline_p <= 1.0
            assert 0.0 <= row.achieved_stress_p <= 1.0


# ── Time monotonicity: survival only decreases ────────────────────────────────


class TestTimeMonotonicity:
    def test_survival_does_not_increase_over_time(self):
        config = _make_config()
        result = simulate_decumulation(
            config, initial_portfolio=_PORTFOLIO, seed=_SEED, n_simulations=_N,
            monthly_withdrawal_override=_WITHDRAWAL,
        )
        for prev, curr in zip(result.survival_by_year, result.survival_by_year[1:]):
            assert curr <= prev + 1e-9, (
                f"Survival increased from {prev} to {curr} — paths can't resurrect"
            )

    def test_chained_survival_does_not_increase_over_time(self):
        config = _make_config()
        chained = compute_chained_mc(
            config=config,
            monthly_stock_contribution=1_000.0,
            monthly_bond_contribution=500.0,
            required_portfolio=1_500_000.0,
            n_paths=_N, seed=_SEED,
        )
        for prev, curr in zip(chained.survival_by_year, chained.survival_by_year[1:]):
            assert curr <= prev + 1e-9


# ── Input monotonicity: more wealth → higher survival/preservation ───────────


class TestWealthMonotonicity:
    def test_more_wealth_increases_survival(self):
        # Same RNG draws → comparison is deterministic. More wealth must give ≥ survival.
        config = _make_config()
        kw = dict(seed=_SEED, n_simulations=_N, monthly_withdrawal_override=_WITHDRAWAL)
        low = simulate_decumulation(config, initial_portfolio=1_000_000.0, **kw)
        high = simulate_decumulation(config, initial_portfolio=3_000_000.0, **kw)
        assert high.survival_rate >= low.survival_rate - 1e-9, (
            f"More wealth gave LOWER survival: {high.survival_rate} < {low.survival_rate}"
        )

    def test_more_wealth_increases_preservation(self):
        # P(final ≥ fixed threshold) must rise with starting wealth
        config = _make_config()
        kw = dict(seed=_SEED, n_simulations=_N, monthly_withdrawal_override=_WITHDRAWAL,
                  return_final_portfolios=True)
        low = simulate_decumulation(config, initial_portfolio=1_500_000.0, **kw)
        high = simulate_decumulation(config, initial_portfolio=3_000_000.0, **kw)
        threshold = 1_000_000.0
        p_low = float(np.mean(np.asarray(low.final_portfolios) >= threshold))
        p_high = float(np.mean(np.asarray(high.final_portfolios) >= threshold))
        assert p_high >= p_low - 1e-9


# ── Input monotonicity: higher withdrawal → lower survival ───────────────────


class TestWithdrawalMonotonicity:
    def test_higher_withdrawal_decreases_survival(self):
        config = _make_config()
        kw = dict(seed=_SEED, n_simulations=_N, initial_portfolio=_PORTFOLIO)
        low = simulate_decumulation(config, monthly_withdrawal_override=2_000.0, **kw)
        high = simulate_decumulation(config, monthly_withdrawal_override=6_000.0, **kw)
        # Higher withdrawal must produce ≤ survival. Allow tiny float slack.
        assert high.survival_rate <= low.survival_rate + 1e-9


# ── Inflation monotonicity: higher inflation → lower survival ────────────────


class TestInflationMonotonicity:
    def test_higher_inflation_does_not_increase_survival(self):
        # Higher inflation raises real withdrawal cost → less survival (or equal).
        # Same seed, but inflation parameters differ in MC draw mean → not strictly
        # deterministic, allow modest MC noise tolerance.
        cfg_low = _make_config(expected_inflation_percent=1.0)
        cfg_high = _make_config(expected_inflation_percent=4.0)
        kw = dict(seed=_SEED, n_simulations=_N, initial_portfolio=_PORTFOLIO,
                  monthly_withdrawal_override=_WITHDRAWAL)
        low = simulate_decumulation(cfg_low, **kw)
        high = simulate_decumulation(cfg_high, **kw)
        assert high.survival_rate <= low.survival_rate + 0.05  # 5pp MC slack


# ── Pension monotonicity: earlier/larger pension → higher survival ───────────


class TestPensionMonotonicity:
    def test_later_pension_does_not_increase_survival(self):
        cfg_early = _make_config(mc_decumulation_pension_start_year=5)
        cfg_late = _make_config(mc_decumulation_pension_start_year=15)
        kw = dict(seed=_SEED, n_simulations=_N, initial_portfolio=_PORTFOLIO,
                  monthly_withdrawal_override=_WITHDRAWAL)
        early = simulate_decumulation(cfg_early, **kw)
        late = simulate_decumulation(cfg_late, **kw)
        assert late.survival_rate <= early.survival_rate + 1e-9

    def test_larger_pension_does_not_decrease_survival(self):
        cfg_small = _make_config(mc_decumulation_pension_monthly=1_000.0)
        cfg_large = _make_config(mc_decumulation_pension_monthly=3_000.0)
        kw = dict(seed=_SEED, n_simulations=_N, initial_portfolio=_PORTFOLIO,
                  monthly_withdrawal_override=_WITHDRAWAL)
        small = simulate_decumulation(cfg_small, **kw)
        large = simulate_decumulation(cfg_large, **kw)
        assert large.survival_rate >= small.survival_rate - 1e-9


# ── Canonical FI bar monotonicity ────────────────────────────────────────────


class TestCanonicalMonotonicity:
    def test_higher_preservation_ratio_does_not_decrease_canonical_wealth(self):
        # Preserving more wealth costs more starting wealth.
        # Compare across the scan rows of a single run — both reachable.
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        scan = compute_canonical_fi_required_scan(
            config, target_monthly, seed=_SEED, n_simulations=_N,
        )
        # Sort rows by preserve_ratio ascending; required wealth should be monotonic
        sorted_rows = sorted(scan.rows, key=lambda r: r.preserve_ratio)
        for prev, curr in zip(sorted_rows, sorted_rows[1:]):
            if prev.reachable and curr.reachable:
                # Allow small slack from binary-search tolerance (5k €)
                assert curr.required_wealth_eur >= prev.required_wealth_eur - 10_000.0

    def test_higher_baseline_confidence_does_not_decrease_canonical_wealth(self):
        cfg_lax = _make_config(fi_preservation_confidence_baseline=0.80)
        cfg_strict = _make_config(fi_preservation_confidence_baseline=0.95)
        infl = (1 + cfg_lax.expected_inflation_percent / 100) ** cfg_lax.years
        target_monthly = cfg_lax.target_monthly_retirement_income * infl
        scan_lax = compute_canonical_fi_required_scan(
            cfg_lax, target_monthly, seed=_SEED, n_simulations=_N,
        )
        scan_strict = compute_canonical_fi_required_scan(
            cfg_strict, target_monthly, seed=_SEED, n_simulations=_N,
        )
        if scan_lax.canonical_reachable and scan_strict.canonical_reachable:
            assert scan_strict.canonical_required_wealth >= scan_lax.canonical_required_wealth - 50_000.0


# ── Cross-section consistency ────────────────────────────────────────────────


class TestConsistency:
    def test_sorr_baseline_survival_matches_default_simulate(self):
        # SORR baseline should match a vanilla simulate_decumulation with the same params.
        # Same seed + same n_simulations → same MC draws → identical survival.
        config = _make_config()
        kw = dict(seed=_SEED, n_simulations=_N, monthly_withdrawal_override=_WITHDRAWAL)
        baseline_simulate = simulate_decumulation(
            config, initial_portfolio=_PORTFOLIO, **kw,
        )
        sorr = compute_sorr_scenarios(
            config, initial_portfolio=_PORTFOLIO, **kw,
        )
        # Allow tiny noise from float ordering in vectorised paths
        assert abs(sorr.baseline.survival_rate - baseline_simulate.survival_rate) <= 0.02

    def test_canonical_scan_canonical_row_in_rows(self):
        # The canonical_required_wealth must match one of the scan rows.
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        scan = compute_canonical_fi_required_scan(
            config, target_monthly, seed=_SEED, n_simulations=_N,
        )
        matching = [r for r in scan.rows
                    if abs(r.preserve_ratio - scan.canonical_ratio) < 1e-6]
        assert matching, "canonical_ratio not found in scan rows"
        assert matching[0].required_wealth_eur == scan.canonical_required_wealth


# ── Two-bucket (E3) invariants ────────────────────────────────────────────────


class TestTwoBucketProperties:
    def test_total_equals_sum_of_buckets(self):
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        result = compute_two_bucket_required_wealth(
            config, target_monthly, seed=_SEED, n_simulations=_N,
        )
        assert result.required_total_wealth == pytest.approx(
            result.required_retirement_wealth + result.required_bequest_wealth
        )

    def test_probabilities_in_unit_interval(self):
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        result = compute_two_bucket_required_wealth(
            config, target_monthly, seed=_SEED, n_simulations=_N,
        )
        for p in (result.retirement_baseline_survival, result.retirement_stress_survival,
                  result.bequest_baseline_p, result.bequest_stress_p):
            assert 0.0 <= p <= 1.0

    def test_higher_preserve_ratio_does_not_decrease_bequest_wealth(self):
        # R doesn't depend on preserve_ratio at all (survival bar only), so this
        # isolates B's own monotonicity: preserving more of a larger total (X·(B+R))
        # costs more bequest-bucket capital. Ratios chosen so both are reachable
        # for this fixture (0.75 saturates the search ceiling here).
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        lax = compute_two_bucket_required_wealth(
            config, target_monthly, preserve_ratio=0.10, seed=_SEED, n_simulations=_N,
        )
        strict = compute_two_bucket_required_wealth(
            config, target_monthly, preserve_ratio=0.30, seed=_SEED, n_simulations=_N,
        )
        if lax.reachable and strict.reachable:
            assert strict.required_bequest_wealth >= lax.required_bequest_wealth - 10_000.0

    def test_retirement_wealth_independent_of_preserve_ratio(self):
        # R's survival bar has no notion of preserve_ratio — same R* regardless.
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        low = compute_two_bucket_required_wealth(
            config, target_monthly, preserve_ratio=0.10, seed=_SEED, n_simulations=_N,
        )
        high = compute_two_bucket_required_wealth(
            config, target_monthly, preserve_ratio=0.75, seed=_SEED, n_simulations=_N,
        )
        assert low.required_retirement_wealth == pytest.approx(
            high.required_retirement_wealth, rel=0.01
        )


# ── Survival matrix invariants ────────────────────────────────────────────────


class TestSurvivalMatrixProperties:
    def test_probabilities_and_ratios_well_formed(self):
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        matrix = compute_survival_matrix(
            config, target_monthly, [("P", _PORTFOLIO)], seed=_SEED, n_simulations=_N,
        )
        row = matrix.rows[0]
        n_scenarios = len(matrix.scenario_labels)
        assert len(row.survival_by_scenario) == n_scenarios
        assert len(row.median_preserved_by_scenario) == n_scenarios
        assert len(row.p5_preserved_by_scenario) == n_scenarios
        for p in row.survival_by_scenario:
            assert 0.0 <= p <= 1.0
        for ratio in row.median_preserved_by_scenario + row.p5_preserved_by_scenario:
            assert ratio >= 0.0

    def test_median_preserved_does_not_exceed_survival_implied_bound(self):
        # P5 (worse luck) preserved ratio should never exceed the median for the
        # same scenario — P5 is the 5th percentile of a real-final distribution.
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        matrix = compute_survival_matrix(
            config, target_monthly, [("P", _PORTFOLIO)], seed=_SEED, n_simulations=_N,
        )
        row = matrix.rows[0]
        for p5, med in zip(row.p5_preserved_by_scenario, row.median_preserved_by_scenario):
            assert p5 <= med + 1e-9

    def test_higher_wealth_does_not_decrease_survival(self):
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        matrix = compute_survival_matrix(
            config, target_monthly,
            [("low", _PORTFOLIO), ("high", _PORTFOLIO * 2.0)],
            seed=_SEED, n_simulations=_N,
        )
        low_row, high_row = matrix.rows
        for low_p, high_p in zip(low_row.survival_by_scenario, high_row.survival_by_scenario):
            assert high_p >= low_p - 0.02

    def test_baseline_survival_matches_default_simulate(self):
        # Same seed + same n_simulations -> same MC draws -> identical baseline survival.
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        matrix = compute_survival_matrix(
            config, target_monthly, [("P", _PORTFOLIO)], seed=_SEED, n_simulations=_N,
        )
        baseline_simulate = simulate_decumulation(
            config, initial_portfolio=_PORTFOLIO, seed=_SEED, n_simulations=_N,
            monthly_withdrawal_override=target_monthly,
        )
        assert abs(matrix.rows[0].survival_by_scenario[0] - baseline_simulate.survival_rate) <= 0.02


# ── Crash-timing invariants ───────────────────────────────────────────────────


class TestCrashTimingProperties:
    def test_lists_well_formed(self):
        config = _make_config()
        matrix = compute_crash_timing_matrix(
            config, 3_500.0, [("P", 700_000.0)],
            offsets=[0, 5, 10, 15], seed=_SEED, n_simulations=_N,
        )
        row = matrix.rows[0]
        assert len(row.survival_by_offset) == 4
        assert len(row.median_preserved_by_offset) == 4
        assert len(row.p5_preserved_by_offset) == 4
        for p in row.survival_by_offset:
            assert 0.0 <= p <= 1.0

    def test_later_crash_does_not_decrease_survival(self):
        # A crash landing later (closer to / after pension start) should not be
        # more damaging than the same crash at retirement day. Uses a stressed
        # portfolio where survival isn't already saturated at 100%, so the
        # comparison is meaningful.
        config = _make_config()
        matrix = compute_crash_timing_matrix(
            config, 3_500.0, [("P", 700_000.0)],
            offsets=[0, 5, 10, 15], seed=_SEED, n_simulations=_N,
        )
        row = matrix.rows[0]
        for prev, curr in zip(row.survival_by_offset, row.survival_by_offset[1:]):
            assert curr >= prev - 0.05

    def test_offset_zero_matches_survival_matrix_scenario(self):
        # Crash at offset 0 IS the same scenario compute_survival_matrix reports
        # under its named label — same seed/wealth/config should match exactly.
        config = _make_config()
        infl = (1 + config.expected_inflation_percent / 100) ** config.years
        target_monthly = config.target_monthly_retirement_income * infl
        timing = compute_crash_timing_matrix(
            config, target_monthly, [("P", _PORTFOLIO)],
            offsets=[0], seed=_SEED, n_simulations=_N,
        )
        surface = compute_survival_matrix(
            config, target_monthly, [("P", _PORTFOLIO)],
            seed=_SEED, n_simulations=_N,
        )
        crash_idx = surface.scenario_labels.index(timing.crash_label)
        assert timing.rows[0].survival_by_offset[0] == pytest.approx(
            surface.rows[0].survival_by_scenario[crash_idx], abs=1e-9
        )
