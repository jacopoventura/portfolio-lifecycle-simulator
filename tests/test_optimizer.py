"""Tests for the ETF portfolio optimiser (compute_optimal_portfolio)."""
import numpy as np
import pandas as pd
import pytest

from main_cagr_analysis import (
    EU_TICKERS,
    OptimalPortfolioResult,
    PortfolioStrategy,
    TickerResult,
    WindowStats,
    compute_optimal_portfolio,
)


def _make_prices(n_days: int, cagr: float, vol: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(cagr / 252, vol / np.sqrt(252), n_days)
    prices = 100.0 * np.cumprod(1 + daily_ret)
    idx = pd.date_range("2013-01-01", periods=n_days, freq="B")
    return pd.Series(prices, index=idx)


def _make_result(ticker: str, prices: pd.Series, eu: bool) -> TickerResult:
    return TickerResult(
        ticker=ticker, name=ticker, currency_original="EUR",
        data_from=prices.index[0].date(), data_to=prices.index[-1].date(),
        ann_vol_eur=0.15, ann_vol_5y_eur=0.15, ann_vol_10y_eur=0.15,
        max_drawdown_eur=-0.35, sharpe_eur=0.5, sortino_eur=0.7, calmar_eur=0.2,
        windows={10: WindowStats(200, 0.07, 0.03, 0.01, 0.15, 0.05, 0.25, 0.85)},
        eu_available=eu,
    )


N = 12 * 252   # ~12 years of trading days — enough for 10y rolling windows


@pytest.fixture(scope="module")
def synthetic_data():
    prices = {
        "IWDA.AS": _make_prices(N, 0.07, 0.14, 1),
        "EIMI.AS": _make_prices(N, 0.06, 0.18, 2),
        "XDWD.DE": _make_prices(N, 0.08, 0.20, 3),
        "QQQ":     _make_prices(N, 0.10, 0.22, 4),
    }
    results = [
        _make_result("IWDA.AS", prices["IWDA.AS"], eu=True),
        _make_result("EIMI.AS", prices["EIMI.AS"], eu=True),
        _make_result("XDWD.DE", prices["XDWD.DE"], eu=True),
        _make_result("QQQ",     prices["QQQ"],     eu=False),
    ]
    return results, prices


@pytest.fixture(scope="module")
def opt_result(synthetic_data):
    results, prices = synthetic_data
    return compute_optimal_portfolio(
        results, prices,
        optimize_tickers=["IWDA.AS", "EIMI.AS", "XDWD.DE", "QQQ"],
        max_weight_per_ticker=0.50,
        qqq_max_weight=0.25,
        min_weight_threshold=0.05,
        min_history_years=5.0,
    )


class TestOptimizerReturnsResult:
    def test_result_not_none(self, opt_result):
        assert opt_result is not None

    def test_all_strategies_present(self, opt_result):
        assert set(opt_result.strategies.keys()) == {
            "max_sortino", "max_sharpe", "min_variance", "max_cagr",
            "max_p50_cagr", "max_p1_cagr", "equal_weight",
        }

    def test_recommended_is_valid_key(self, opt_result):
        assert opt_result.recommended in opt_result.strategies

    def test_recommended_has_highest_p50_cagr(self, opt_result):
        rec_cagr = opt_result.strategies[opt_result.recommended].portfolio_cagr_p50
        for key, strat in opt_result.strategies.items():
            other = strat.portfolio_cagr_p50
            if rec_cagr is not None and other is not None:
                assert rec_cagr >= other - 1e-6, f"{key} has higher P50 CAGR than recommended"

    def test_eligible_tickers_populated(self, opt_result):
        assert len(opt_result.eligible_tickers) == 4

    def test_common_history_populated(self, opt_result):
        assert opt_result.n_trading_days > 0
        assert opt_result.n_years > 5.0


class TestWeightConstraints:
    @pytest.mark.parametrize("key", [
        "max_sortino", "max_sharpe", "min_variance", "max_cagr",
        "max_p50_cagr", "max_p1_cagr", "equal_weight",
    ])
    def test_weights_sum_to_one(self, opt_result, key):
        w = list(opt_result.strategies[key].weights.values())
        assert abs(sum(w) - 1.0) < 1e-5

    @pytest.mark.parametrize("key", [
        "max_sortino", "max_sharpe", "min_variance", "max_cagr",
        "max_p50_cagr", "max_p1_cagr", "equal_weight",
    ])
    def test_no_negative_weights(self, opt_result, key):
        assert all(v >= 0 for v in opt_result.strategies[key].weights.values())

    @pytest.mark.parametrize("key", [
        "max_sortino", "max_sharpe", "min_variance", "max_cagr",
        "max_p50_cagr", "max_p1_cagr",
    ])
    def test_qqq_cap_respected(self, opt_result, key):
        qqq_w = opt_result.strategies[key].weights.get("QQQ", 0.0)
        assert qqq_w <= 0.25 + 1e-4, f"{key}: QQQ={qqq_w:.3f} > 0.25"

    @pytest.mark.parametrize("key", [
        "max_sortino", "max_sharpe", "min_variance", "max_cagr",
        "max_p50_cagr", "max_p1_cagr",
    ])
    def test_max_weight_per_ticker(self, opt_result, key):
        assert all(v <= 0.50 + 1e-4 for v in opt_result.strategies[key].weights.values())


class TestStrategyStats:
    def test_all_cagrs_positive(self, opt_result):
        for key, strat in opt_result.strategies.items():
            assert strat.portfolio_cagr > 0, f"{key} has non-positive CAGR"

    def test_max_cagr_beats_minvar(self, opt_result):
        cagr_mc = opt_result.strategies["max_cagr"].portfolio_cagr
        cagr_mv = opt_result.strategies["min_variance"].portfolio_cagr
        assert cagr_mc >= cagr_mv - 1e-6, "max_cagr should have CAGR >= min_variance"

    def test_all_vols_positive(self, opt_result):
        for key, strat in opt_result.strategies.items():
            assert strat.portfolio_vol_5y > 0, f"{key} has non-positive vol"

    def test_mdd_negative(self, opt_result):
        for key, strat in opt_result.strategies.items():
            if strat.portfolio_mdd is not None:
                assert strat.portfolio_mdd < 0, f"{key} MDD should be negative"

    def test_sortino_better_than_minvar(self, opt_result):
        s_sortino = opt_result.strategies["max_sortino"].portfolio_sortino or 0
        s_minvar  = opt_result.strategies["min_variance"].portfolio_sortino or 0
        assert s_sortino >= s_minvar - 0.05

    def test_minvar_lowest_vol(self, opt_result):
        vols = {k: s.portfolio_vol_5y for k, s in opt_result.strategies.items()
                if k != "equal_weight"}
        assert vols["min_variance"] == min(vols.values())

    def test_max_p50_cagr_beats_max_cagr_on_p50(self, opt_result):
        p50_p50 = opt_result.strategies["max_p50_cagr"].portfolio_cagr_p50
        p50_mc  = opt_result.strategies["max_cagr"].portfolio_cagr_p50
        if p50_p50 is not None and p50_mc is not None:
            assert p50_p50 >= p50_mc - 1e-4, "max_p50_cagr should have P50 >= max_cagr"

    def test_p50_10y_cagr_populated(self, opt_result):
        # 12yr common history → rolling 10y windows exist
        for key, strat in opt_result.strategies.items():
            assert strat.portfolio_cagr_p50 is not None, f"{key}: p50 CAGR missing"

    def test_p5_10y_cagr_populated_and_below_p50(self, opt_result):
        for key, strat in opt_result.strategies.items():
            assert strat.portfolio_cagr_p5 is not None, f"{key}: p5 CAGR missing"
            assert strat.portfolio_cagr_p5 <= strat.portfolio_cagr_p50 + 1e-6, (
                f"{key}: P5 should be <= P50"
            )

    def test_p1_10y_cagr_populated_and_below_p5(self, opt_result):
        for key, strat in opt_result.strategies.items():
            assert strat.portfolio_cagr_p1 is not None, f"{key}: p1 CAGR missing"
            assert strat.portfolio_cagr_p1 <= strat.portfolio_cagr_p5 + 1e-6, (
                f"{key}: P1 should be <= P5"
            )

    def test_annual_ret_std_populated(self, opt_result):
        for key, strat in opt_result.strategies.items():
            assert strat.portfolio_annual_ret_std is not None, f"{key}: annual_ret_std missing"
            assert strat.portfolio_annual_ret_std >= 0, f"{key}: annual_ret_std negative"

    def test_annual_ret_std_consistent_with_mdd(self, opt_result):
        # σ-annual should be in a plausible range relative to MDD (not 1% for a -30% MDD portfolio)
        for key, strat in opt_result.strategies.items():
            std = strat.portfolio_annual_ret_std
            mdd = strat.portfolio_mdd
            if std is not None and mdd is not None:
                assert std > 0.05, f"{key}: annual std {std:.2%} implausibly low"
                assert std < 1.0,  f"{key}: annual std {std:.2%} implausibly high"


class TestEdgeCases:
    def test_too_few_tickers_returns_none(self, synthetic_data):
        results, prices = synthetic_data
        result = compute_optimal_portfolio(
            results, prices, optimize_tickers=["IWDA.AS"],
        )
        assert result is None

    def test_unknown_ticker_excluded(self, synthetic_data):
        results, prices = synthetic_data
        result = compute_optimal_portfolio(
            results, prices,
            optimize_tickers=["IWDA.AS", "EIMI.AS", "NONEXISTENT"],
        )
        assert result is not None
        assert any(t == "NONEXISTENT" for t, _ in result.excluded)
        assert "NONEXISTENT" not in result.eligible_tickers

    def test_short_history_ticker_excluded(self, synthetic_data):
        results, prices = synthetic_data
        short_p = _make_prices(2 * 252, 0.07, 0.15, 99)
        short_r = _make_result("SHORT.DE", short_p, eu=True)
        result = compute_optimal_portfolio(
            results + [short_r], {**prices, "SHORT.DE": short_p},
            optimize_tickers=["IWDA.AS", "EIMI.AS", "SHORT.DE"],
            min_history_years=5.0,
        )
        assert result is not None
        assert "SHORT.DE" in [t for t, _ in result.excluded]
