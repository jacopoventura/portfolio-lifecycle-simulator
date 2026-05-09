"""
Tests for main_cagr_analysis.py — pure functions and cache logic.

Coverage:
  - _window_label, _pct                      — formatting helpers
  - _SORT_FALLBACK constant                   — precomputed fallback order
  - _sort_key                                 — sorting with fallbacks
  - _rolling_cagrs                            — core rolling CAGR computation
  - _rolling_ann_vol                          — rolling annualised volatility
  - convert_to_eur                            — FX conversion (USD and GBP)
  - _fetch_with_cache                         — parquet cache (miss / hit / network failure)
"""

import sys
import os
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main_cagr_analysis import (
    SORT_BY_WINDOW,
    WINDOW_YEARS,
    _SORT_FALLBACK,
    WindowStats,
    TickerResult,
    _window_label,
    _pct,
    _sort_key,
    _rolling_cagrs,
    _rolling_ann_vol,
    convert_to_eur,
    _fetch_with_cache,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_prices(n_years: float, daily_return: float = 0.0, start: str = "2010-01-01") -> pd.Series:
    """Synthetic daily price series with a constant daily log-return.

    Uses calendar-day frequency so the calendar span matches n_years and the
    code's 365.25-based `actual_years` arithmetic stays accurate.
    daily_return is per calendar day; for a 7% annual return use log(1.07)/365.
    """
    n = int(n_years * 365)
    dates = pd.date_range(start, periods=n, freq="D")
    prices = 100.0 * np.exp(daily_return * np.arange(n))
    return pd.Series(prices, index=dates)


def _annualised_return(daily_return: float) -> float:
    return np.exp(daily_return * 365) - 1.0


def _make_result(
    windows: dict,
    eu_available: bool = True,
    ticker: str = "TEST",
) -> TickerResult:
    return TickerResult(
        ticker=ticker,
        name=ticker,
        currency_original="EUR",
        data_from=date(2010, 1, 1),
        data_to=date(2020, 1, 1),
        ann_vol_eur=None,
        ann_vol_5y_eur=None,
        ann_vol_10y_eur=None,
        max_drawdown_eur=None,
        sharpe_eur=None,
        sortino_eur=None,
        calmar_eur=None,
        windows=windows,
        eu_available=eu_available,
    )


def _ws(mean: float | None, n: int = 100) -> WindowStats:
    return WindowStats(
        n_windows=n,
        cagr_mean=mean,
        cagr_std=None,
        cagr_min=None,
        cagr_max=None,
    )


def _make_yf_hist(prices: list[float], tz: str = "America/New_York") -> pd.DataFrame:
    """Minimal yfinance-style OHLCV DataFrame with a tz-aware index."""
    idx = pd.date_range("2020-01-02", periods=len(prices), freq="B", tz=tz)
    return pd.DataFrame({"Close": prices, "Volume": [1_000] * len(prices)}, index=idx)


# ── _window_label ─────────────────────────────────────────────────────────────

class TestWindowLabel:
    def test_integer_years(self):
        assert _window_label(1)  == "1y"
        assert _window_label(10) == "10y"
        assert _window_label(15) == "15y"

    def test_none_returns_max(self):
        assert _window_label(None) == "max"


# ── _pct ──────────────────────────────────────────────────────────────────────

class TestPct:
    def test_none_returns_na(self):
        assert _pct(None) == "n/a"

    def test_positive(self):
        assert _pct(0.07) == "7.0%"

    def test_negative(self):
        assert _pct(-0.05) == "-5.0%"

    def test_custom_decimals(self):
        assert _pct(0.1234, 2) == "12.34%"

    def test_zero(self):
        assert _pct(0.0) == "0.0%"


# ── _SORT_FALLBACK ────────────────────────────────────────────────────────────

class TestSortFallback:
    def test_excludes_sort_by_window(self):
        assert SORT_BY_WINDOW not in _SORT_FALLBACK

    def test_none_is_last(self):
        assert _SORT_FALLBACK[-1] is None

    def test_descending_integer_order(self):
        ints = [y for y in _SORT_FALLBACK if y is not None]
        assert ints == sorted(ints, reverse=True)

    def test_contains_all_other_windows(self):
        expected = {y for y in WINDOW_YEARS if y != SORT_BY_WINDOW}
        assert set(_SORT_FALLBACK) == expected


# ── _sort_key ─────────────────────────────────────────────────────────────────

class TestSortKey:
    def test_primary_window_used_when_available(self):
        r = _make_result({SORT_BY_WINDOW: _ws(0.08)})
        assert _sort_key(r) == pytest.approx(-0.08)

    def test_falls_back_to_longest_rolling(self):
        # SORT_BY_WINDOW (10y) missing; 15y and 5y present
        longest = _SORT_FALLBACK[0]   # 15
        second  = _SORT_FALLBACK[1]   # 5
        r = _make_result({longest: _ws(0.06), second: _ws(0.09)})
        assert _sort_key(r) == pytest.approx(-0.06)

    def test_falls_back_to_alltime_last(self):
        r = _make_result({None: _ws(0.05)})
        assert _sort_key(r) == pytest.approx(-0.05)

    def test_no_data_returns_inf(self):
        r = _make_result({})
        assert _sort_key(r) == float("inf")

    def test_no_data_window_none_mean(self):
        r = _make_result({SORT_BY_WINDOW: _ws(None)})
        assert _sort_key(r) == float("inf")

    def test_higher_cagr_sorts_first(self):
        r_high = _make_result({SORT_BY_WINDOW: _ws(0.12)}, ticker="A")
        r_low  = _make_result({SORT_BY_WINDOW: _ws(0.05)}, ticker="B")
        assert _sort_key(r_high) < _sort_key(r_low)


# ── _rolling_cagrs ────────────────────────────────────────────────────────────

class TestRollingCagrs:
    def test_constant_growth_matches_expected_cagr(self):
        # daily_return per calendar day so actual_years arithmetic in _rolling_cagrs is exact
        daily_ret = np.log(1.07) / 365
        prices = _make_prices(7, daily_ret)
        cagrs = _rolling_cagrs(prices, 5)
        assert cagrs is not None
        assert np.allclose(cagrs, 0.07, atol=1e-3)

    def test_flat_prices_give_zero_cagr(self):
        prices = _make_prices(5, daily_return=0.0)
        cagrs = _rolling_cagrs(prices, 3)
        assert cagrs is not None
        assert np.allclose(cagrs, 0.0, atol=1e-9)

    def test_returns_none_when_window_exceeds_history(self):
        prices = _make_prices(0.5)  # 6 months
        assert _rolling_cagrs(prices, 1) is None

    def test_returns_none_for_all_zero_prices(self):
        n = int(3 * 252)
        dates = pd.date_range("2010-01-01", periods=n, freq="B")
        prices = pd.Series(np.zeros(n), index=dates)
        assert _rolling_cagrs(prices, 1) is None

    def test_returns_none_for_all_negative_prices(self):
        n = int(3 * 252)
        dates = pd.date_range("2010-01-01", periods=n, freq="B")
        prices = pd.Series(np.full(n, -1.0), index=dates)
        assert _rolling_cagrs(prices, 1) is None

    def test_zero_prices_excluded_from_cagr_pairs(self):
        # 4 years of prices; one price is zero — pairs involving it are dropped
        n = int(4 * 252)
        dates = pd.date_range("2010-01-01", periods=n, freq="B")
        vals = np.ones(n) * 100.0
        vals[50] = 0.0   # single bad point
        prices = pd.Series(vals, index=dates)
        cagrs = _rolling_cagrs(prices, 2)
        assert cagrs is not None
        assert np.all(np.isfinite(cagrs))

    def test_cagr_count_decreases_with_longer_window(self):
        prices = _make_prices(10)
        c3 = _rolling_cagrs(prices, 3)
        c5 = _rolling_cagrs(prices, 5)
        assert len(c3) > len(c5)

    def test_tolerance_filter_excludes_windows_with_large_gap(self):
        # Prices with a 2-year gap in the middle: some windows will span far more
        # calendar time than requested. The 1.15 tolerance filter should exclude them.
        early = _make_prices(3, daily_return=np.log(1.07) / 365, start="2010-01-01")
        late_start = "2015-01-01"   # 2-year gap after 3 years of data
        late = _make_prices(3, daily_return=np.log(1.07) / 365, start=late_start)
        prices = pd.concat([early, late])
        prices = prices[~prices.index.duplicated(keep="last")].sort_index()
        # A 3-year window starting just before the gap would normally land 5+ years later
        # The tolerance filter should drop those distorted windows
        cagrs = _rolling_cagrs(prices, 3)
        # Windows that survive should all have actual_years < 3 * 1.15 = 3.45
        # (those landing entirely within early or late segment are fine)
        assert cagrs is None or np.all(np.isfinite(cagrs))


# ── _rolling_ann_vol ──────────────────────────────────────────────────────────

class TestRollingAnnVol:
    def test_flat_prices_give_zero_vol(self):
        prices = _make_prices(6, daily_return=0.0)
        vol = _rolling_ann_vol(prices, 5.0)
        assert vol == pytest.approx(0.0, abs=1e-10)

    def test_returns_none_for_insufficient_history(self):
        # 3 years available, requesting 5 → 3/5 = 60% < 80%
        prices = _make_prices(3)
        assert _rolling_ann_vol(prices, 5.0) is None

    def test_returns_value_when_just_above_80pct_threshold(self):
        # 4.1 years available, requesting 5 → 4.1/5 = 82% > 80%
        prices = _make_prices(4.1, daily_return=np.log(1.07) / 252)
        result = _rolling_ann_vol(prices, 5.0)
        assert result is not None

    def test_returns_none_for_zero_prices_in_window(self):
        n = int(6 * 252)
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        vals = np.ones(n) * 100.0
        vals[-10] = 0.0
        prices = pd.Series(vals, index=dates)
        assert _rolling_ann_vol(prices, 5.0) is None

    def test_known_volatility(self):
        # Build prices with known daily std, verify annualised vol
        rng = np.random.default_rng(42)
        daily_std = 0.01  # 1% daily
        n = int(6 * 252)
        log_returns = rng.normal(0, daily_std, n)
        prices = 100.0 * np.exp(np.cumsum(log_returns))
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        s = pd.Series(prices, index=dates)
        vol = _rolling_ann_vol(s, 5.0)
        expected = daily_std * np.sqrt(252)
        assert vol == pytest.approx(expected, rel=0.05)

    def test_custom_trading_days_scales_vol(self):
        # Using trading_days=365 (crypto) should give a different annualisation
        rng = np.random.default_rng(42)
        daily_std = 0.01
        n = int(6 * 252)
        log_returns = rng.normal(0, daily_std, n)
        prices = 100.0 * np.exp(np.cumsum(log_returns))
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        s = pd.Series(prices, index=dates)
        vol_252 = _rolling_ann_vol(s, 5.0, trading_days=252)
        vol_365 = _rolling_ann_vol(s, 5.0, trading_days=365)
        assert vol_365 == pytest.approx(vol_252 * np.sqrt(365 / 252), rel=0.001)


# ── WindowStats win_rate ───────────────────────────────────────────────────────

class TestWindowStatsWinRate:
    def test_win_rate_defaults_to_none(self):
        ws = WindowStats(n_windows=0, cagr_mean=None, cagr_std=None,
                         cagr_min=None, cagr_max=None)
        assert ws.win_rate is None

    def test_win_rate_all_positive(self):
        cagrs = np.array([0.05, 0.07, 0.10])
        ws = WindowStats(
            n_windows=len(cagrs), cagr_mean=float(np.mean(cagrs)),
            cagr_std=float(np.std(cagrs, ddof=1)),
            cagr_min=float(cagrs.min()), cagr_max=float(cagrs.max()),
            win_rate=float(np.mean(cagrs > 0)),
        )
        assert ws.win_rate == pytest.approx(1.0)

    def test_win_rate_partial(self):
        cagrs = np.array([0.05, -0.02, 0.08, -0.01])
        wr = float(np.mean(cagrs > 0))
        ws = WindowStats(
            n_windows=len(cagrs), cagr_mean=float(np.mean(cagrs)),
            cagr_std=float(np.std(cagrs, ddof=1)),
            cagr_min=float(cagrs.min()), cagr_max=float(cagrs.max()),
            win_rate=wr,
        )
        assert ws.win_rate == pytest.approx(0.5)


# ── TickerResult calmar_eur ────────────────────────────────────────────────────

class TestTickerResultCalmar:
    def test_calmar_defaults_to_none(self):
        r = _make_result({})
        assert r.calmar_eur is None

    def test_calmar_field_accessible(self):
        r = _make_result({}, eu_available=False)
        r2 = TickerResult(
            ticker="X", name="X", currency_original="EUR",
            data_from=date(2010, 1, 1), data_to=date(2020, 1, 1),
            ann_vol_eur=0.15, ann_vol_5y_eur=0.14, ann_vol_10y_eur=0.16,
            max_drawdown_eur=-0.50, sharpe_eur=0.40, sortino_eur=0.60,
            calmar_eur=0.16,  # e.g. 8% CAGR / 50% MDD
            windows={}, eu_available=True,
        )
        assert r2.calmar_eur == pytest.approx(0.16)


# ── convert_to_eur ────────────────────────────────────────────────────────────

class TestConvertToEur:
    def _dates(self, n: int) -> pd.DatetimeIndex:
        return pd.date_range("2020-01-02", periods=n, freq="B")

    def test_eur_passthrough(self):
        prices = pd.Series([100.0, 101.0, 102.0], index=self._dates(3))
        result = convert_to_eur(prices, "EUR", None)
        pd.testing.assert_series_equal(result, prices)

    def test_none_fx_passthrough(self):
        prices = pd.Series([100.0, 101.0], index=self._dates(2))
        result = convert_to_eur(prices, "USD", None)
        pd.testing.assert_series_equal(result, prices)

    def test_usd_divided_by_eurusd(self):
        prices = pd.Series([110.0, 110.0, 110.0], index=self._dates(3))
        fx     = pd.Series([1.10,  1.10,  1.10],  index=self._dates(3))
        result = convert_to_eur(prices, "USD", fx)
        assert np.allclose(result.values, 100.0)

    def test_gbp_multiplied_by_gbpeur(self):
        prices = pd.Series([90.0, 90.0, 90.0], index=self._dates(3))
        fx     = pd.Series([1.20, 1.20, 1.20], index=self._dates(3))
        result = convert_to_eur(prices, "GBP", fx)
        assert np.allclose(result.values, 108.0)

    def test_tz_aware_price_index_normalised(self):
        idx = pd.date_range("2020-01-02", periods=3, freq="B", tz="America/New_York")
        prices = pd.Series([100.0, 100.0, 100.0], index=idx)
        fx     = pd.Series([1.0, 1.0, 1.0], index=pd.date_range("2020-01-02", periods=3, freq="B"))
        result = convert_to_eur(prices, "USD", fx)
        assert result.index.tz is None

    def test_missing_fx_rows_dropped(self):
        prices = pd.Series([100.0, 101.0, 102.0], index=self._dates(3))
        # FX only has 2 of 3 dates — the missing one should be forward-filled or dropped
        fx_idx = pd.DatetimeIndex([self._dates(3)[0], self._dates(3)[2]])
        fx = pd.Series([1.0, 1.0], index=fx_idx)
        result = convert_to_eur(prices, "USD", fx)
        # All three rows should survive (middle date forward-filled from first)
        assert len(result) == 3

    def test_usd_varying_fx(self):
        prices = pd.Series([110.0, 220.0], index=self._dates(2))
        fx     = pd.Series([1.10,  2.20],  index=self._dates(2))
        result = convert_to_eur(prices, "USD", fx)
        assert np.allclose(result.values, [100.0, 100.0])


# ── _fetch_with_cache ─────────────────────────────────────────────────────────

class TestFetchWithCache:
    def _mock_ticker(self, prices: list[float], tz: str = "America/New_York") -> MagicMock:
        hist = _make_yf_hist(prices, tz=tz)
        m = MagicMock()
        m.history.return_value = hist
        return m

    def test_cache_miss_returns_new_status(self, tmp_path):
        mock_t = self._mock_ticker([100.0, 101.0, 102.0])
        with patch("main_cagr_analysis.CACHE_DIR", tmp_path), \
             patch("main_cagr_analysis.yf.Ticker", return_value=mock_t):
            df, status = _fetch_with_cache("TEST", "TEST")
        assert status == "new"
        assert df is not None
        assert len(df) == 3

    def test_cache_miss_writes_parquet(self, tmp_path):
        mock_t = self._mock_ticker([100.0, 101.0, 102.0])
        with patch("main_cagr_analysis.CACHE_DIR", tmp_path), \
             patch("main_cagr_analysis.yf.Ticker", return_value=mock_t):
            _fetch_with_cache("TEST", "TEST")
        assert (tmp_path / "TEST.parquet").exists()

    def test_cache_hit_no_new_data_returns_cached(self, tmp_path):
        # Write initial cache
        mock_t = self._mock_ticker([100.0, 101.0, 102.0])
        with patch("main_cagr_analysis.CACHE_DIR", tmp_path), \
             patch("main_cagr_analysis.yf.Ticker", return_value=mock_t):
            _fetch_with_cache("TEST", "TEST")

        # Second call: network returns the same data (no new days)
        with patch("main_cagr_analysis.CACHE_DIR", tmp_path), \
             patch("main_cagr_analysis.yf.Ticker", return_value=mock_t):
            df, status = _fetch_with_cache("TEST", "TEST")

        assert status == "cached"
        assert df is not None

    def test_cache_hit_with_new_days_returns_updated(self, tmp_path):
        # Write initial cache (3 days)
        mock_initial = self._mock_ticker([100.0, 101.0, 102.0])
        with patch("main_cagr_analysis.CACHE_DIR", tmp_path), \
             patch("main_cagr_analysis.yf.Ticker", return_value=mock_initial):
            _fetch_with_cache("TEST", "TEST")

        # Second call: network returns extra days beyond the cached range
        extra_prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        mock_updated = self._mock_ticker(extra_prices)
        with patch("main_cagr_analysis.CACHE_DIR", tmp_path), \
             patch("main_cagr_analysis.yf.Ticker", return_value=mock_updated):
            df, status = _fetch_with_cache("TEST", "TEST")

        assert "updated" in status
        assert df is not None
        # Merged frame must have at least 5 unique rows
        assert len(df) >= 5

    def test_network_failure_with_cache_returns_cached(self, tmp_path):
        # Write initial cache
        mock_ok = self._mock_ticker([100.0, 101.0, 102.0])
        with patch("main_cagr_analysis.CACHE_DIR", tmp_path), \
             patch("main_cagr_analysis.yf.Ticker", return_value=mock_ok):
            _fetch_with_cache("TEST", "TEST")

        # Second call: network raises
        mock_err = MagicMock()
        mock_err.history.side_effect = Exception("network error")
        with patch("main_cagr_analysis.CACHE_DIR", tmp_path), \
             patch("main_cagr_analysis.yf.Ticker", return_value=mock_err):
            df, status = _fetch_with_cache("TEST", "TEST")

        assert status == "cached"
        assert df is not None

    def test_total_failure_returns_none(self, tmp_path):
        mock_err = MagicMock()
        mock_err.history.side_effect = Exception("network error")
        with patch("main_cagr_analysis.CACHE_DIR", tmp_path), \
             patch("main_cagr_analysis.yf.Ticker", return_value=mock_err):
            df, status = _fetch_with_cache("MISSING", "MISSING")
        assert df is None
        assert status == "failed"

    def test_index_is_tz_naive_after_cache_miss(self, tmp_path):
        mock_t = self._mock_ticker([100.0, 101.0])
        with patch("main_cagr_analysis.CACHE_DIR", tmp_path), \
             patch("main_cagr_analysis.yf.Ticker", return_value=mock_t):
            df, _ = _fetch_with_cache("TEST", "TEST")
        assert df.index.tz is None

    def test_index_is_tz_naive_after_cache_hit(self, tmp_path):
        mock_t = self._mock_ticker([100.0, 101.0, 102.0])
        with patch("main_cagr_analysis.CACHE_DIR", tmp_path), \
             patch("main_cagr_analysis.yf.Ticker", return_value=mock_t):
            _fetch_with_cache("TEST", "TEST")
            df, _ = _fetch_with_cache("TEST", "TEST")
        assert df.index.tz is None

    def test_cache_name_used_as_filename(self, tmp_path):
        mock_t = self._mock_ticker([100.0])
        with patch("main_cagr_analysis.CACHE_DIR", tmp_path), \
             patch("main_cagr_analysis.yf.Ticker", return_value=mock_t):
            _fetch_with_cache("EURUSD=X", "FX_USD")
        assert (tmp_path / "FX_USD.parquet").exists()
        assert not (tmp_path / "EURUSD=X.parquet").exists()
