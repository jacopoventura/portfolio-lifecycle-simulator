"""
main_cagr_analysis.py

Fetches historical price data from Yahoo Finance for a list of ETFs and
reports rolling CAGR statistics for multiple time windows.

Rolling method: for each window of N years, every single trading day in the
full available history is used as a start date. The N-year CAGR is computed
from that day to the nearest trading day exactly N years later. Mean, Std,
Min, and Max are reported across all those rolling windows.

Std of rolling CAGRs measures entry-point consistency: how much the N-year
CAGR varies depending on when you started investing. Low std means the ETF
reliably delivers near its mean CAGR regardless of entry date. This is
distinct from Ann. σ (daily price volatility) which feeds into Monte Carlo.

FX conversion: all prices are converted to EUR before computation so results
are directly comparable for a EUR-based investor. EUR-priced ETFs are left
unchanged (EUR/USD is already embedded in their NAV). USD and GBP prices are
converted using historical daily EUR/USD and GBP/EUR rates.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ── Configuration ─────────────────────────────────────────────────────────────

# EU-available UCITS ETFs (all EUR-denominated unless noted)
EU_TICKERS: list[str] = [
    "IWDA.AS",   # iShares Core MSCI World Acc (Amsterdam)
    "CSPX.L",    # iShares Core S&P 500 Acc (London, USD)
    "VEVE.AS",   # Vanguard FTSE Developed World Acc (Amsterdam)
    "IS3R.DE",   # iShares Edge MSCI World Momentum Factor (Xetra)
    "XDEQ.DE",   # Xtrackers MSCI World Quality Factor (Xetra)
    "XDWT.DE",   # Xtrackers MSCI World Information Technology (Xetra)
    "TDIV.AS",   # VanEck Morningstar Developed Markets Dividend Leaders (Amsterdam)
    "IEMA.AS",   # iShares MSCI EM IMI Acc (Amsterdam)
    "IGLN.L",    # iShares Physical Gold ETC (London, USD)
]

# US-listed benchmarks — not available to EU retail investors under MiFID II/PRIIPs
# Kept for historical comparison (longer track records). Prices converted to EUR.
US_TICKERS: list[str] = [
    "SPY",       # SPDR S&P 500
    "QQQ",       # Invesco QQQ (NASDAQ-100)
    "VTI",       # Vanguard Total Stock Market
    "VEA",       # Vanguard Developed Markets ex-US
    "VWO",       # Vanguard Emerging Markets
    "GLD",       # SPDR Gold Shares (since 2004, USD) — longer history than IGLN.L
    "BTC-USD",   # Bitcoin spot price (USD) — alternative asset benchmark
]

TICKERS = EU_TICKERS + US_TICKERS

# Tickers treated as gold/commodity for the gold_max_weight cap in the optimizer
GOLD_TICKERS: frozenset[str] = frozenset({"IGLN.L", "GLD"})

# Time windows in years shown as columns. None = single endpoint CAGR (inception→today).
WINDOW_YEARS: list[int | None] = [1, 3, 5, 10, 15, None]

# Window used for the main sort and for showing Min/Max in the summary
SORT_BY_WINDOW: int | None = 10

# Fallback sort order when SORT_BY_WINDOW has no data: longest rolling window
# first, single-endpoint all-time last. Precomputed — used in every _sort_key call.
_SORT_FALLBACK: list[int | None] = (
    [y for y in sorted((x for x in WINDOW_YEARS if x is not None), reverse=True)
     if y != SORT_BY_WINDOW]
    + [None]
)

# Yahoo Finance tickers for FX rates (fetched once at startup)
FX_TICKERS = {
    "USD": "EURUSD=X",   # USD per 1 EUR  →  price_EUR = price_USD / rate
    "GBP": "GBPEUR=X",   # EUR per 1 GBP  →  price_EUR = price_GBP * rate
}

# Fallback currency map: used when t.info fails and to validate unexpected
# currencies returned by Yahoo Finance.
_KNOWN_CURRENCIES: dict[str, str] = {
    "IWDA.AS": "EUR",
    "VEVE.AS": "EUR",
    "IS3R.DE": "EUR", "XDEQ.DE": "EUR", "XDWT.DE": "EUR",
    "TDIV.AS": "EUR", "IEMA.AS": "EUR",
    "CSPX.L": "USD",   # London-listed, USD share class
    "IGLN.L": "USD",   # London-listed physical gold ETC, USD share class
    "SPY": "USD", "QQQ": "USD", "VTI": "USD", "VEA": "USD", "VWO": "USD",
    "GLD": "USD", "BTC-USD": "USD",
    "^GSPC": "USD", "^IXIC": "USD", "^W5000": "USD",
    "^STOXX50E": "EUR",
}

# Parquet cache: one file per ticker/FX rate in this folder
CACHE_DIR = Path("historical_data")
# Re-download this many trailing days on cache hit to catch retroactive yfinance
# split/dividend adjustments
CACHE_OVERLAP_DAYS = 30

# Annual risk-free rate used for Sharpe and Sortino (ECB deposit rate proxy)
RISK_FREE_RATE = 0.025


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class WindowStats:
    n_windows: int
    cagr_mean: float | None
    cagr_std: float | None   # std of rolling CAGRs = entry-point timing risk
    cagr_min: float | None
    cagr_max: float | None
    cagr_p05: float | None = None   # 5th percentile — bad-luck planning input
    cagr_p25: float | None = None   # 25th percentile
    win_rate: float | None = None   # fraction of rolling windows with CAGR > 0


@dataclass
class TickerResult:
    ticker: str
    name: str
    currency_original: str          # as reported by Yahoo Finance
    data_from: date
    data_to: date
    ann_vol_eur: float | None       # annualised daily σ, full history
    ann_vol_5y_eur: float | None    # annualised daily σ, last 5 years
    ann_vol_10y_eur: float | None   # annualised daily σ, last 10 years
    max_drawdown_eur: float | None  # peak-to-trough max drawdown (negative)
    sharpe_eur: float | None        # (all-time CAGR − RFR) / ann_vol_eur
    sortino_eur: float | None       # (all-time CAGR − RFR) / downside_vol_eur
    calmar_eur: float | None        # all-time CAGR / |MDD| — SORR risk measure
    windows: dict[int | None, WindowStats]
    eu_available: bool              # True = UCITS, False = US-only benchmark


# ── Parquet cache ─────────────────────────────────────────────────────────────

def _fetch_with_cache(yf_symbol: str, cache_name: str) -> tuple[pd.DataFrame | None, str]:
    """Download OHLCV history with parquet caching.

    Returns (df, status):
      "new"         — full download, no prior cache
      "updated +Nd" — cache hit, N new trading days appended
      "cached"      — cache hit, already up to date
      "failed"      — network error and no usable cache
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_name}.parquet"

    cached_df: pd.DataFrame | None = None
    if cache_path.exists():
        try:
            cached_df = pd.read_parquet(cache_path)
            # Index was saved tz-naive; just ensure it is a proper DatetimeIndex
            cached_df.index = pd.to_datetime(cached_df.index).normalize()
        except Exception:
            cached_df = None

    if cached_df is not None and not cached_df.empty:
        last_date = cached_df.index.max()
        overlap_start = last_date - pd.Timedelta(days=CACHE_OVERLAP_DAYS)
        try:
            new_raw = yf.Ticker(yf_symbol).history(
                start=overlap_start.strftime("%Y-%m-%d"), auto_adjust=True
            )
        except Exception:
            return cached_df, "cached"

        if new_raw.empty:
            return cached_df, "cached"

        new_raw.index = pd.to_datetime(new_raw.index).tz_localize(None).normalize()
        new_raw = new_raw[~new_raw.index.duplicated(keep="last")]

        merged = pd.concat([cached_df, new_raw])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.sort_index(inplace=True)

        n_new = int((merged.index > last_date).sum())
        try:
            merged.to_parquet(cache_path)
        except Exception:
            pass

        status = f"updated +{n_new}d" if n_new > 0 else "cached"
        return merged, status

    # Cache miss — full download
    try:
        df = yf.Ticker(yf_symbol).history(period="max", auto_adjust=True)
    except Exception:
        return None, "failed"

    if df is None or df.empty:
        return None, "failed"

    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    try:
        df.to_parquet(cache_path)
    except Exception:
        pass

    return df, "new"


# ── FX helpers ────────────────────────────────────────────────────────────────

def fetch_fx_series(currency: str) -> tuple[pd.Series | None, str]:
    if currency == "EUR":
        return None, "n/a"
    fx_ticker = FX_TICKERS.get(currency)
    if fx_ticker is None:
        return None, "failed"

    cache_name = f"FX_{currency}"
    df, status = _fetch_with_cache(fx_ticker, cache_name)

    if df is None or df.empty or "Close" not in df.columns:
        return None, "failed"

    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s, status


def convert_to_eur(prices: pd.Series, currency: str, fx: pd.Series | None) -> pd.Series:
    if currency == "EUR" or fx is None:
        return prices
    # Normalise ETF index to tz-naive date
    prices = prices.copy()
    idx = pd.to_datetime(prices.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    prices.index = idx.normalize()
    prices = prices[~prices.index.duplicated(keep="last")]
    # Align FX to ETF dates with forward-fill for missing market days
    aligned_fx = fx.reindex(prices.index, method="ffill")
    # Drop rows where FX is unavailable
    mask = aligned_fx.notna()
    prices = prices[mask]
    aligned_fx = aligned_fx[mask]
    if currency == "USD":
        return prices / aligned_fx       # price_EUR = price_USD / EURUSD
    elif currency == "GBP":
        return prices * aligned_fx       # price_EUR = price_GBP * GBPEUR
    return prices


# ── Rolling CAGR ──────────────────────────────────────────────────────────────

def _window_label(years: int | None) -> str:
    return "max" if years is None else f"{years}y"


def _rolling_cagrs(prices: pd.Series, years: int) -> np.ndarray | None:
    """
    For every trading day in prices, find the day exactly `years` calendar
    years later (nearest available trading day), compute the CAGR.
    Fully vectorised using pandas datetime arithmetic.
    """
    n_days = int(years * 365.25)
    target_dates = prices.index + pd.to_timedelta(n_days, unit="D")
    end_positions = prices.index.searchsorted(target_dates, side="left")

    valid = end_positions < len(prices)
    si = np.where(valid)[0]
    ei = end_positions[si]

    if len(si) == 0:
        return None

    actual_days = (prices.index[ei] - prices.index[si]).days.values
    actual_years = actual_days / 365.25

    pv = prices.values.astype(float)
    # Drop pairs where either price is zero/negative or end date is >15% farther
    # than requested (protects against large data gaps distorting CAGR windows)
    ok = (pv[si] > 0) & (pv[ei] > 0) & (actual_years < years * 1.15)
    if not np.any(ok):
        return None
    si, ei, actual_years = si[ok], ei[ok], actual_years[ok]

    cagrs = (pv[ei] / pv[si]) ** (1.0 / actual_years) - 1.0
    return cagrs


# ── Volatility helper ─────────────────────────────────────────────────────────

def _rolling_ann_vol(prices: pd.Series, years: float, trading_days: float = 252) -> float | None:
    """Annualised daily log-return σ over the last `years` calendar years.
    Returns None if fewer than 80% of the requested window is available,
    or if any price in the window is zero/negative.
    trading_days: actual trading days per year (252 for equities, 365 for crypto)."""
    cutoff = prices.index[-1] - pd.Timedelta(days=int(years * 365.25))
    subset = prices[prices.index >= cutoff]
    available = (prices.index[-1] - prices.index[0]).days / 365.25
    if available < years * 0.8:
        return None
    vals = subset.values.astype(float)
    if np.any(vals <= 0):
        return None
    log_ret = np.diff(np.log(vals))
    return float(np.std(log_ret, ddof=1) * np.sqrt(trading_days)) if len(log_ret) > 1 else None


# ── Fetch & compute ───────────────────────────────────────────────────────────

def fetch_and_compute(
    ticker: str,
    fx_cache: dict[str, pd.Series | None],
    eu_available: bool,
) -> tuple[TickerResult | None, str, pd.Series | None]:
    df, cache_status = _fetch_with_cache(ticker, ticker)

    if df is None or df.empty or "Close" not in df.columns:
        if cache_status != "failed":
            print(f"  [!] {ticker}: no data returned")
        return None, "failed", None

    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
        currency = (info.get("currency") or "EUR").upper()
    except Exception:
        name = ticker
        currency = _KNOWN_CURRENCIES.get(ticker, "")
        if not currency:
            currency = "EUR"
            print(f"  [!] {ticker}: t.info failed, currency unknown — assuming EUR (CAGR may be wrong)")

    if currency != "EUR" and currency not in FX_TICKERS:
        known = _KNOWN_CURRENCIES.get(ticker)
        if known and known != currency and (known == "EUR" or known in FX_TICKERS):
            print(f"  [!] {ticker}: Yahoo returned currency '{currency}', expected '{known}' — using known value")
            currency = known
        else:
            # No safe conversion path: refuse rather than silently mislabel as EUR.
            print(f"  [!] {ticker}: unsupported currency '{currency}' (no FX series, no known fallback) — skipping")
            return None, "failed", None

    raw_prices = df["Close"].dropna()
    if len(raw_prices) < 2:
        print(f"  [!] {ticker}: insufficient price history")
        return None, "failed", None

    # Convert to EUR
    fx = fx_cache.get(currency)
    if currency != "EUR" and fx is None:
        print(f"  [!] {ticker}: FX series for {currency} unavailable — skipping")
        return None, "failed", None
    prices = convert_to_eur(raw_prices, currency, fx)
    if len(prices) < 2:
        print(f"  [!] {ticker}: FX alignment left insufficient data")
        return None, "failed", None

    # Normalise index for consistent rolling computation
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices[~prices.index.duplicated(keep="last")]

    data_from = prices.index[0].date()
    data_to   = prices.index[-1].date()

    # Calendar span and actual trading-days-per-year detection.
    # Crypto (BTC-USD) trades every calendar day (~365/yr); equities trade ~252/yr.
    # Using the wrong factor understates crypto vol by ~√(365/252) ≈ 20%.
    _span_years = (data_to - data_from).days / 365.25
    trading_days_per_year = len(prices) / max(_span_years, 1.0)

    # Annualised daily-return volatility in EUR terms — full history and rolling windows
    _pv_all = prices.values.astype(float)
    if len(_pv_all) > 1 and np.all(_pv_all > 0):
        log_ret = np.diff(np.log(_pv_all))
        ann_vol = float(np.std(log_ret, ddof=1) * np.sqrt(trading_days_per_year))
    else:
        ann_vol = None
    ann_vol_5y  = _rolling_ann_vol(prices, 5.0,  trading_days_per_year)
    ann_vol_10y = _rolling_ann_vol(prices, 10.0, trading_days_per_year)

    # Max drawdown (peak-to-trough in EUR terms) — guard against zero/negative prices
    if np.all(_pv_all > 0):
        rolling_max = prices.cummax()
        max_dd: float | None = float(((prices - rolling_max) / rolling_max).min())
    else:
        max_dd = None

    windows: dict[int | None, WindowStats] = {}
    available_years = _span_years

    for years in WINDOW_YEARS:
        if years is None:
            if available_years >= 1.0 and np.all(_pv_all > 0):
                cagr = (_pv_all[-1] / _pv_all[0]) ** (1.0 / available_years) - 1.0
                windows[None] = WindowStats(
                    n_windows=1, cagr_mean=float(cagr), cagr_std=None,
                    cagr_min=float(cagr), cagr_max=float(cagr),
                    win_rate=1.0 if cagr > 0 else 0.0,
                )
            else:
                windows[None] = WindowStats(n_windows=0, cagr_mean=None, cagr_std=None,
                                            cagr_min=None, cagr_max=None)
            continue

        if available_years < years:
            windows[years] = WindowStats(n_windows=0, cagr_mean=None, cagr_std=None,
                                         cagr_min=None, cagr_max=None)
            continue

        cagrs = _rolling_cagrs(prices, years)
        if cagrs is None or len(cagrs) < 2:
            windows[years] = WindowStats(n_windows=0 if cagrs is None else len(cagrs),
                                         cagr_mean=None, cagr_std=None,
                                         cagr_min=None, cagr_max=None)
            continue

        windows[years] = WindowStats(
            n_windows=len(cagrs),
            cagr_mean=float(np.mean(cagrs)),
            cagr_std=float(np.std(cagrs, ddof=1)),
            cagr_min=float(np.min(cagrs)),
            cagr_max=float(np.max(cagrs)),
            cagr_p05=float(np.percentile(cagrs, 5)),
            cagr_p25=float(np.percentile(cagrs, 25)),
            win_rate=float(np.mean(cagrs > 0)),
        )

    # Sharpe and Sortino — full-history daily simple returns vs all-time CAGR
    sharpe: float | None = None
    sortino: float | None = None
    calmar: float | None = None
    w_alltime = windows.get(None)
    if ann_vol is not None and w_alltime and w_alltime.cagr_mean is not None:
        excess = w_alltime.cagr_mean - RISK_FREE_RATE
        sharpe = excess / ann_vol
        # Downside deviation: daily simple returns below the daily risk-free threshold.
        # Use per-ticker trading_days_per_year so BTC isn't mis-annualised.
        simple_ret = prices.pct_change().dropna().values.astype(float)
        if len(simple_ret) > 1:
            daily_rfr = (1 + RISK_FREE_RATE) ** (1.0 / trading_days_per_year) - 1
            downside = np.minimum(simple_ret - daily_rfr, 0.0)
            sortino_vol = float(np.sqrt(np.mean(downside ** 2) * trading_days_per_year))
            sortino = excess / sortino_vol if sortino_vol > 0 else None

    # Calmar ratio: all-time CAGR / |MDD| — return per unit of worst drawdown
    if w_alltime and w_alltime.cagr_mean is not None and max_dd is not None and max_dd < 0:
        calmar = w_alltime.cagr_mean / abs(max_dd)

    return TickerResult(
        ticker=ticker, name=name,
        currency_original=currency,
        data_from=data_from, data_to=data_to,
        ann_vol_eur=ann_vol,
        ann_vol_5y_eur=ann_vol_5y,
        ann_vol_10y_eur=ann_vol_10y,
        max_drawdown_eur=max_dd,
        sharpe_eur=sharpe,
        sortino_eur=sortino,
        calmar_eur=calmar,
        windows=windows,
        eu_available=eu_available,
    ), cache_status, prices


# ── Formatting ────────────────────────────────────────────────────────────────

def _pct(v: float | None, d: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def _sort_key(r: TickerResult) -> float:
    w = r.windows.get(SORT_BY_WINDOW)
    if w and w.cagr_mean is not None:
        return -w.cagr_mean
    # Fallback: longest rolling window first, single-endpoint all-time last
    for y in _SORT_FALLBACK:
        ww = r.windows.get(y)
        if ww and ww.cagr_mean is not None:
            return -ww.cagr_mean
    return float("inf")  # no data → sort last


def _name_col_width(fixed_overhead: int, min_w: int = 20) -> int:
    cols = shutil.get_terminal_size((220, 50)).columns
    return max(min_w, cols - fixed_overhead)


def _short_ticker(ticker: str) -> str:
    return ticker.split(".")[0]


# ── Final summary ─────────────────────────────────────────────────────────────

def print_final_summary(results: list[TickerResult]) -> None:
    """Compact ranked table: key metrics only, with a consistency ratio."""
    ref = SORT_BY_WINDOW
    ref_label = _window_label(ref)

    rk_w    = 2
    tick_w  = 8
    yr_w    = 4
    cw      = 7
    cons_w  = 6
    av_w    = 6   # volatility columns (3 of them)
    mdd_w   = 7   # max drawdown
    ratio_w = 6   # Sharpe / Sortino / Calmar
    wr_w    = 6   # win rate for reference window
    # fixed chars in sep excluding name_w: 27+29+20+27+10+27+9 = 149
    name_w = _name_col_width(
        (2+rk_w+3+tick_w+3+3+yr_w+2) +
        (1+cw+3+cw+3+cons_w+2) +
        (1+cw+3+cw+2) +
        (1+av_w+3+av_w+3+av_w+2) +
        (1+mdd_w+2) +
        (1+ratio_w+3+ratio_w+3+ratio_w+2) +
        (1+wr_w+2)
    )

    sep = (
        f"+-{'-'*rk_w}-+-{'-'*tick_w}-+-{'-'*name_w}-+-{'-'*yr_w}-+"
        f"-{'-'*cw}-+-{'-'*cw}-+-{'-'*cons_w}-+"
        f"-{'-'*cw}-+-{'-'*cw}-+"
        f"-{'-'*av_w}-+-{'-'*av_w}-+-{'-'*av_w}-+"
        f"-{'-'*mdd_w}-+"
        f"-{'-'*ratio_w}-+-{'-'*ratio_w}-+-{'-'*ratio_w}-+"
        f"-{'-'*wr_w}-+"
    )
    hdr = (
        f"| {'#':>{rk_w}} | {'Ticker':<{tick_w}} | {'Name':<{name_w}} "
        f"| {'From':>{yr_w}} |"
        f" {ref_label+' avg':>{cw}} | {'P5-'+ref_label:>{cw}} | {'Cons.':>{cons_w}} |"
        f" {'5y avg':>{cw}} | {'All-time':>{cw}} |"
        f" {'σ-5y':>{av_w}} | {'σ-10y':>{av_w}} | {'σ-all':>{av_w}} |"
        f" {'MDD':>{mdd_w}} |"
        f" {'Sharpe':>{ratio_w}} | {'Sortino':>{ratio_w}} | {'Calmar':>{ratio_w}} |"
        f" {'WR-'+ref_label:>{wr_w}} |"
    )

    def _cons(r: TickerResult) -> float | None:
        w = r.windows.get(ref)
        if w and w.cagr_mean is not None and w.cagr_std is not None and w.cagr_std > 0:
            return w.cagr_mean / w.cagr_std
        return None

    all_results = sorted(results, key=_sort_key)

    def _row(rank: int, r: TickerResult) -> None:
        w_ref = r.windows.get(ref)
        w_5y  = r.windows.get(5)
        w_max = r.windows.get(None)
        cons  = _cons(r)
        def _ratio(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.2f}"
        wr_s = (
            f"{w_ref.win_rate * 100:.0f}%"
            if w_ref and w_ref.win_rate is not None else "n/a"
        )

        print(
            f"| {rank:>{rk_w}} | {_short_ticker(r.ticker):<{tick_w}} | {r.name[:name_w]:<{name_w}} "
            f"| {r.data_from.year:>{yr_w}} |"
            f" {_pct(w_ref.cagr_mean if w_ref else None):>{cw}} |"
            f" {_pct(w_ref.cagr_p05  if w_ref else None):>{cw}} |"
            f" {(f'{cons:.1f}' if cons is not None else 'n/a'):>{cons_w}} |"
            f" {_pct(w_5y.cagr_mean  if w_5y  else None):>{cw}} |"
            f" {_pct(w_max.cagr_mean if w_max  else None):>{cw}} |"
            f" {_pct(r.ann_vol_5y_eur):>{av_w}} |"
            f" {_pct(r.ann_vol_10y_eur):>{av_w}} |"
            f" {_pct(r.ann_vol_eur):>{av_w}} |"
            f" {_pct(r.max_drawdown_eur):>{mdd_w}} |"
            f" {_ratio(r.sharpe_eur):>{ratio_w}} | {_ratio(r.sortino_eur):>{ratio_w}} | {_ratio(r.calmar_eur):>{ratio_w}} |"
            f" {wr_s:>{wr_w}} |"
        )

    print(sep)
    print(hdr)
    print(sep)
    for i, r in enumerate(all_results, 1):
        _row(i, r)
    print(sep)

# ── Correlation matrix ────────────────────────────────────────────────────────

def print_correlation_matrix(results: list[TickerResult], prices: dict[str, pd.Series]) -> None:
    """Pairwise Pearson correlation of EUR daily log returns over the common date range."""
    tickers = [r.ticker for r in results if r.ticker in prices]
    eu_set  = {r.ticker for r in results if r.eu_available}
    if len(tickers) < 2:
        return

    start = max(prices[t].index[0] for t in tickers)
    end   = min(prices[t].index[-1] for t in tickers)
    if start >= end:
        print("  [!] No overlapping date range — correlation matrix skipped.")
        return

    log_ret: dict[str, pd.Series] = {}
    for t in tickers:
        sub = prices[t]
        sub = sub[(sub.index >= start) & (sub.index <= end)].astype(float)
        sub = sub[sub > 0]
        lr = np.log(sub).diff().dropna()
        if len(lr) > 1:
            log_ret[t] = lr

    tickers = list(log_ret.keys())
    n = len(tickers)
    if n < 2:
        return

    df_ret = pd.DataFrame(log_ret)
    corr   = df_ret.corr()

    n_years = (end - start).days / 365.25
    n_days  = int(df_ret.count().mean())

    stickers = [_short_ticker(t) for t in tickers]
    stw = max(len(s) for s in stickers)  # column/row label width

    print(f"  Common period : {start.date()} → {end.date()}"
          f"  ({n_years:.1f} years  |  ~{n_days:,} days/pair)")
    print()

    # Column header row
    hdr = f"  {'':>{stw}}"
    for st in stickers:
        hdr += f"  {st:>{stw}}"
    print(hdr)

    # Lower-triangular rows (diagonal shown as dashes)
    for i, (ti, st_i) in enumerate(zip(tickers, stickers)):
        row = f"  {st_i:>{stw}}"
        for j in range(i):
            v = corr.loc[ti, tickers[j]]
            row += "   n/a" if pd.isna(v) else f"  {v:>{stw}.2f}"
        row += "  " + "─" * stw
        print(row)


# ── Portfolio Optimiser ───────────────────────────────────────────────────────

@dataclass
class PortfolioStrategy:
    name: str
    weights: dict[str, float]         # ticker → weight (0.0 = excluded by threshold)
    portfolio_cagr: float             # realized CAGR from common history
    portfolio_cagr_p50: float | None       # P50 rolling 10y CAGR (None if < 10yr data)
    portfolio_cagr_p5: float | None        # P5 rolling 10y CAGR — pessimistic entry-point scenario
    portfolio_cagr_p1: float | None        # P1 rolling 10y CAGR — worst-case entry-point scenario
    portfolio_annual_ret_std: float | None # std of calendar-year returns — portfolio volatility proxy
    portfolio_vol_5y: float                # 5y annualized daily vol of combined series
    portfolio_sortino: float | None
    portfolio_sharpe: float | None
    portfolio_mdd: float | None
    portfolio_avg_corr: float | None  # weighted avg pairwise correlation across ETF pairs


@dataclass
class OptimalPortfolioResult:
    eligible_tickers: list[str]
    ticker_names: dict[str, str]             # ticker → full ETF name
    excluded: list[tuple[str, str]]          # (ticker, reason)
    common_from: date
    common_to: date
    n_trading_days: int
    n_years: float
    strategies: dict[str, PortfolioStrategy]
    recommended: str                          # key of recommended strategy
    shrinkage_info: dict | None = field(default=None)


def _portfolio_stats(
    port_ret: np.ndarray,
    dates: pd.DatetimeIndex,
    trading_days_per_year: float,
) -> tuple[float, float | None, float | None, float | None, float | None, float, float | None, float | None, float | None]:
    """Return (cagr, p50_10y, p5_10y, p1_10y, annual_ret_std, vol_5y, sortino, sharpe, mdd)."""
    n = len(port_ret)
    total_ret = float((1.0 + port_ret).prod())
    cagr = total_ret ** (trading_days_per_year / n) - 1.0 if n > 0 else 0.0

    # Rolling 10y percentiles (entry-point timing distribution)
    cum_prices = pd.Series((1.0 + port_ret).cumprod() * 100.0, index=dates)
    rolling = _rolling_cagrs(cum_prices, years=10)
    if rolling is not None and len(rolling) >= 5:
        p50_10y = float(np.percentile(rolling, 50))
        p5_10y  = float(np.percentile(rolling, 5))
        p1_10y  = float(np.percentile(rolling, 1))
    else:
        p50_10y = p5_10y = p1_10y = None

    # Std of calendar-year returns — true portfolio volatility proxy (consistent with MDD)
    ret_series = pd.Series(port_ret, index=dates)
    annual_rets = (1 + ret_series).groupby(ret_series.index.year).prod() - 1
    annual_ret_std = float(annual_rets.std(ddof=1)) if len(annual_rets) >= 3 else None

    cutoff = dates[-1] - pd.Timedelta(days=5 * 365.25)
    recent_ret = port_ret[dates > cutoff]
    vol_src = recent_ret if len(recent_ret) > 20 else port_ret
    vol_5y = float(np.std(vol_src, ddof=1) * np.sqrt(trading_days_per_year))

    daily_rfr = (1.0 + RISK_FREE_RATE) ** (1.0 / trading_days_per_year) - 1.0
    downside = np.minimum(port_ret - daily_rfr, 0.0)
    sortino_vol = float(np.sqrt(np.mean(downside ** 2) * trading_days_per_year))
    sortino = (cagr - RISK_FREE_RATE) / sortino_vol if sortino_vol > 1e-8 else None
    sharpe  = (cagr - RISK_FREE_RATE) / vol_5y if vol_5y > 1e-8 else None

    cum = (1.0 + port_ret).cumprod()
    peak = np.maximum.accumulate(cum)
    mdd = float(((cum - peak) / peak).min()) if len(cum) > 0 else None

    return cagr, p50_10y, p5_10y, p1_10y, annual_ret_std, vol_5y, sortino, sharpe, mdd


def _run_optimization(
    ret_array: np.ndarray,
    n: int,
    objective_fn,
    bounds,
    constraints: list,
    n_restarts: int = 15,
    seed: int = 42,
) -> np.ndarray:
    """SLSQP minimisation with multiple random restarts; returns best weight vector."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    best_w = np.ones(n) / n
    best_val = np.inf

    # Extract the scalar upper box-bound so restart points start feasible w.r.t.
    # max_weight_per_ticker (SLSQP can recover but converges faster from feasible).
    _ub = getattr(bounds, "ub", None)
    ub_scalar = float(np.atleast_1d(_ub).max()) if _ub is not None else 1.0

    for _ in range(n_restarts):
        w0 = rng.dirichlet(np.ones(n))
        # Clip-and-redistribute any component above the box bound onto remaining
        # capacity; bails if the cap is infeasible (sum of caps < 1).
        for _ in range(5):
            if w0.max() <= ub_scalar + 1e-9:
                break
            excess = (w0 - ub_scalar).clip(min=0).sum()
            w0 = np.minimum(w0, ub_scalar)
            room = (ub_scalar - w0).clip(min=0)
            if room.sum() <= 1e-9:
                break
            w0 = w0 + excess * room / room.sum()
        try:
            res = minimize(
                objective_fn, w0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 500},
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_w = res.x
        except Exception:
            pass

    return best_w


def _oas_covariance(X_centered: np.ndarray) -> tuple[np.ndarray, float]:
    """Oracle Approximating Shrinkage estimator (Chen et al. 2010).
    Shrinks sample covariance toward (trace(S)/p) * I.
    Returns (cov_shrunk, delta) where delta is the shrinkage coefficient."""
    T, p = X_centered.shape
    S = X_centered.T @ X_centered / T
    trace_S  = np.trace(S)
    trace_S2 = np.trace(S @ S)
    mu_target = trace_S / p
    num = (1.0 - 2.0 / p) * trace_S2 + trace_S ** 2
    den = (T + 1.0 - 2.0 / p) * (trace_S2 - trace_S ** 2 / p)
    delta = float(np.clip(num / den, 0.0, 1.0)) if den > 1e-12 else 1.0
    cov_shrunk = (1.0 - delta) * S + delta * mu_target * np.eye(p)
    return cov_shrunk, delta


def compute_optimal_portfolio(
    results: list[TickerResult],
    prices_eur: dict[str, pd.Series],
    optimize_tickers: list[str],
    max_weight_per_ticker: float = 0.50,
    qqq_max_weight: float = 0.25,
    gold_max_weight: float = 0.10,
    min_weight_threshold: float = 0.05,
    min_history_years: float = 5.0,
    selected_strategy: str | None = None,
    use_bayesian_shrinkage: bool = False,
    shrink_mean_intensity: float = 0.5,
    seed: int = 42,
) -> OptimalPortfolioResult | None:
    """Find optimal portfolio weights via Max-Sortino, Max-Sharpe, Min-Variance, and Equal Weight.

    Recommended strategy is the one with the highest P50 rolling 10y CAGR.
    """
    from scipy.optimize import Bounds

    result_map = {r.ticker: r for r in results}

    eligible: list[str] = []
    excluded: list[tuple[str, str]] = []
    for ticker in optimize_tickers:
        if ticker not in result_map:
            excluded.append((ticker, "not in CAGR results"))
            continue
        r = result_map[ticker]
        span = (r.data_to - r.data_from).days / 365.25
        if span < min_history_years:
            excluded.append((ticker, f"only {span:.1f}yr history"))
            continue
        if ticker not in prices_eur:
            excluded.append((ticker, "price series unavailable"))
            continue
        eligible.append(ticker)

    if len(eligible) < 2:
        print("  [!] Optimizer: fewer than 2 eligible tickers — skipping.")
        return None

    # Align on common trading dates
    common_idx = prices_eur[eligible[0]].index
    for t in eligible[1:]:
        common_idx = common_idx.intersection(prices_eur[t].index)
    common_idx = common_idx.sort_values()

    if len(common_idx) < 252:
        print(f"  [!] Optimizer: common history only {len(common_idx)} days — skipping.")
        return None

    ret_df = pd.DataFrame(
        {t: prices_eur[t].loc[common_idx].pct_change() for t in eligible}
    ).dropna()
    dates      = ret_df.index
    ret_array  = ret_df.values.astype(float)
    n          = len(eligible)
    n_years    = (dates[-1] - dates[0]).days / 365.25
    tdpy       = len(dates) / max(n_years, 1.0)   # trading days per year

    # Precompute: special-cap ticker indices and 5-year recent mask
    qqq_idx  = eligible.index("QQQ") if "QQQ" in eligible else -1
    gold_idxs = [i for i, t in enumerate(eligible) if t in GOLD_TICKERS]
    cutoff_5y   = dates[-1] - pd.Timedelta(days=5 * 365.25)
    recent_mask = np.asarray(dates > cutoff_5y)  # numpy bool array — used in Sharpe obj

    # ── Constraints ───────────────────────────────────────────────────────────
    bounds_box   = Bounds(lb=0.0, ub=max_weight_per_ticker)
    constraints: list[dict] = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    if qqq_idx >= 0:
        constraints.append({
            "type": "ineq",
            "fun": lambda w, i=qqq_idx: qqq_max_weight - w[i],
        })
    for gi in gold_idxs:
        constraints.append({
            "type": "ineq",
            "fun": lambda w, i=gi: gold_max_weight - w[i],
        })

    # ── Objective functions ───────────────────────────────────────────────────
    daily_rfr = (1.0 + RISK_FREE_RATE) ** (1.0 / tdpy) - 1.0
    w10 = max(1, int(round(10.0 * tdpy)))  # 10-year window in trading days

    def _rolling_cagrs_td(pr: np.ndarray) -> np.ndarray | None:
        """All rolling 10y CAGR values over every possible start date, indexed by
        trading days (distinct from module-level _rolling_cagrs which uses calendar
        days on a price series). Returns None when history is shorter than 10y."""
        cum_ext = np.concatenate([[1.0], np.cumprod(1.0 + pr)])
        if len(cum_ext) <= w10:
            return None
        ratios = cum_ext[w10:] / cum_ext[:-w10]
        return ratios ** (tdpy / w10) - 1.0

    def _mean_rolling_cagr(pr: np.ndarray) -> float:
        """Mean of all rolling 10y CAGRs; falls back to endpoint CAGR if history < 10y."""
        cagrs = _rolling_cagrs_td(pr)
        if cagrs is None:
            return float((1.0 + pr).prod()) ** (tdpy / len(pr)) - 1.0
        return float(np.mean(cagrs))

    def _neg_sortino(w: np.ndarray) -> float:
        pr = ret_array @ w
        cagr = _mean_rolling_cagr(pr)
        dv = float(np.sqrt(np.mean(np.minimum(pr - daily_rfr, 0.0) ** 2) * tdpy))
        return -(cagr - RISK_FREE_RATE) / dv if dv > 1e-8 else 0.0

    def _neg_sharpe(w: np.ndarray) -> float:
        pr = ret_array @ w
        cagr = _mean_rolling_cagr(pr)
        recent = pr[recent_mask] if recent_mask.sum() > 20 else pr
        vol = float(np.std(recent, ddof=1) * np.sqrt(tdpy))
        return -(cagr - RISK_FREE_RATE) / vol if vol > 1e-8 else 0.0

    def _variance(w: np.ndarray) -> float:
        return float(np.var(ret_array @ w, ddof=1) * tdpy)

    def _neg_cagr(w: np.ndarray) -> float:
        return -_mean_rolling_cagr(ret_array @ w)

    def _rolling_pct(w: np.ndarray, pct: float) -> float:
        """Rolling 10y CAGR percentile — used for max_p50_cagr / max_p1_cagr."""
        pr = ret_array @ w
        cagrs = _rolling_cagrs_td(pr)
        if cagrs is None:
            return float((1.0 + pr).prod()) ** (tdpy / len(pr)) - 1.0
        return float(np.percentile(cagrs, pct))

    def _neg_p50_cagr(w: np.ndarray) -> float:
        return -_rolling_pct(w, 50)

    def _neg_p1_cagr(w: np.ndarray) -> float:
        return -_rolling_pct(w, 1)

    # ── Weight cleanup — enforces threshold and all per-ticker caps ──────────
    def _clean_weights(w_raw: np.ndarray) -> np.ndarray:
        w = w_raw.copy()
        # Zero tiny positions and renorm
        w[w < min_weight_threshold] = 0.0
        total = w.sum()
        if total > 0:
            w /= total
        # Per-ticker caps: tighter qqq/gold caps where applicable.
        # Iteratively redistribute excess until all caps satisfied (usually 1-3 passes).
        caps = np.full(n, max_weight_per_ticker)
        if qqq_idx >= 0:
            caps[qqq_idx] = min(qqq_max_weight, max_weight_per_ticker)
        for gi in gold_idxs:
            caps[gi] = min(gold_max_weight, max_weight_per_ticker)
        for _ in range(20):
            over = w > caps + 1e-9
            if not over.any():
                break
            excess = (w[over] - caps[over]).sum()
            w[over] = caps[over]
            free = ~over & (w > 1e-9)
            if free.sum() > 0:
                w[free] += excess * w[free] / w[free].sum()
                continue
            # No allocated free positions left — threshold-zeroing wiped them.
            # Fall back to capacity-weighted fill across non-over slots so the
            # portfolio stays fully invested (caps win over min-weight threshold).
            capacity = np.maximum(caps - w, 0.0) * (~over)
            if capacity.sum() > 1e-9:
                w += excess * capacity / capacity.sum()
            else:
                break  # caps genuinely sum to < 1 — cannot stay fully invested
        return w

    # ── Correlation matrix (computed once, reused per strategy) ──────────────
    corr_matrix = np.corrcoef(ret_array.T)  # shape [n, n]

    def _weighted_avg_corr(w: np.ndarray) -> float | None:
        """Weighted average pairwise correlation: (w·C·w − Σwᵢ²) / (1 − Σwᵢ²).
        Returns None when only one ETF has non-zero weight (no pairs exist)."""
        hhi = float(np.sum(w ** 2))
        denom = 1.0 - hhi
        if denom < 1e-9:
            return None
        return float((w @ corr_matrix @ w - hhi) / denom)

    # ── Bayesian shrinkage (optional, --advanced-optimization) ───────────────
    shrinkage_info: dict | None = None
    if use_bayesian_shrinkage:
        mu_sample_daily = ret_array.mean(axis=0)
        mu_prior_daily  = np.full(n, mu_sample_daily.mean())
        mu_shrunk_daily = ((1.0 - shrink_mean_intensity) * mu_sample_daily
                           + shrink_mean_intensity * mu_prior_daily)
        mu_shrunk_ann   = mu_shrunk_daily * tdpy

        X_centered          = ret_array - mu_sample_daily
        cov_shrunk_daily, delta_oas = _oas_covariance(X_centered)
        cov_shrunk_ann      = cov_shrunk_daily * tdpy

        shrinkage_info = {
            "delta_oas":            delta_oas,
            "kappa_mean":           shrink_mean_intensity,
            "grand_mean_ann":       float(mu_prior_daily.mean() * tdpy),
            "mu_sample_ann":        {t: float(mu_sample_daily[i] * tdpy) for i, t in enumerate(eligible)},
            "mu_shrunk_ann":        {t: float(mu_shrunk_ann[i])          for i, t in enumerate(eligible)},
        }

        # Redefine affected objectives to use shrunken parameters.
        # P50/P1 objectives intentionally left unchanged (rolling pct already conservative).
        def _neg_sortino(w: np.ndarray) -> float:
            # Asymmetric on purpose: numerator uses the shrunken annual mean
            # (James-Stein), denominator uses raw daily downside deviation.
            # Downside vol isn't a covariance estimate, so OAS shrinkage doesn't
            # apply to it; shrinking only the mean is the standard treatment.
            pr = ret_array @ w
            port_ret = float(w @ mu_shrunk_ann)
            dv = float(np.sqrt(np.mean(np.minimum(pr - daily_rfr, 0.0) ** 2) * tdpy))
            return -(port_ret - RISK_FREE_RATE) / dv if dv > 1e-8 else 0.0

        def _neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(w @ mu_shrunk_ann)
            vol = float(np.sqrt(max(float(w @ cov_shrunk_ann @ w), 0.0)))
            return -(port_ret - RISK_FREE_RATE) / vol if vol > 1e-8 else 0.0

        def _variance(w: np.ndarray) -> float:
            return float(max(float(w @ cov_shrunk_ann @ w), 0.0))

        def _neg_cagr(w: np.ndarray) -> float:
            return -float(w @ mu_shrunk_ann)

    # ── Run all strategies ────────────────────────────────────────────────────
    def _make_strategy(name: str, w_raw: np.ndarray) -> PortfolioStrategy:
        w = _clean_weights(w_raw)
        pr = ret_array @ w
        cagr, p50, p5, p1, annual_std, vol_5y, sortino, sharpe, mdd = _portfolio_stats(pr, dates, tdpy)
        return PortfolioStrategy(
            name=name,
            weights={t: float(w[i]) for i, t in enumerate(eligible)},
            portfolio_cagr=cagr,
            portfolio_cagr_p50=p50,
            portfolio_cagr_p5=p5,
            portfolio_cagr_p1=p1,
            portfolio_annual_ret_std=annual_std,
            portfolio_vol_5y=vol_5y,
            portfolio_sortino=sortino,
            portfolio_sharpe=sharpe,
            portfolio_mdd=mdd,
            portfolio_avg_corr=_weighted_avg_corr(w),
        )

    w_sortino = _run_optimization(ret_array, n, _neg_sortino, bounds_box, constraints, seed=seed)
    w_sharpe  = _run_optimization(ret_array, n, _neg_sharpe,  bounds_box, constraints, seed=seed)
    w_minvar  = _run_optimization(ret_array, n, _variance,    bounds_box, constraints, seed=seed)
    w_cagr    = _run_optimization(ret_array, n, _neg_cagr,    bounds_box, constraints, seed=seed)
    w_p50     = _run_optimization(ret_array, n, _neg_p50_cagr, bounds_box, constraints, seed=seed)
    w_p1      = _run_optimization(ret_array, n, _neg_p1_cagr,  bounds_box, constraints, seed=seed)
    # Fix 4: equal weight also goes through _clean_weights so QQQ cap is enforced
    w_equal   = np.ones(n) / n

    strategies = {
        "max_sortino":  _make_strategy("Max Sortino",  w_sortino),
        "max_sharpe":   _make_strategy("Max Sharpe",   w_sharpe),
        "min_variance": _make_strategy("Min Variance", w_minvar),
        "max_cagr":     _make_strategy("Max CAGR",     w_cagr),
        "max_p50_cagr": _make_strategy("Max P50 CAGR", w_p50),
        "max_p1_cagr":  _make_strategy("Max P1 CAGR",  w_p1),
        "equal_weight": _make_strategy("Equal Weight", w_equal),
    }

    # Recommended = selected_strategy if valid, else best by (P5 available, P5/P50/CAGR).
    # Strategies whose common history is too short to compute rolling 10y P5 get
    # deprioritised — a 40yr decumulation plan must be optimised on P5 survival,
    # not P50 alone (P5=n/a means we can't size the worst-case downside).
    def _rec_score(s: "PortfolioStrategy") -> tuple[int, float]:
        if s.portfolio_cagr_p5 is not None:
            return (1, s.portfolio_cagr_p5)
        if s.portfolio_cagr_p50 is not None:
            return (0, s.portfolio_cagr_p50)
        return (0, s.portfolio_cagr)

    if selected_strategy and selected_strategy in strategies:
        recommended = selected_strategy
    else:
        recommended = max(strategies, key=lambda k: _rec_score(strategies[k]))

    ticker_names = {t: result_map[t].name for t in eligible if t in result_map}

    return OptimalPortfolioResult(
        eligible_tickers=eligible,
        ticker_names=ticker_names,
        excluded=excluded,
        common_from=dates[0].date(),
        common_to=dates[-1].date(),
        n_trading_days=len(dates),
        n_years=n_years,
        strategies=strategies,
        recommended=recommended,
        shrinkage_info=shrinkage_info,
    )


_SHORT_LABELS: dict[str, str] = {
    "max_sortino":  "Sortino",
    "max_sharpe":   "Sharpe",
    "min_variance": "MinVar",
    "max_cagr":     "MaxCAGR",
    "max_p50_cagr": "P50-CAGR",
    "max_p1_cagr":  "P1-CAGR",
    "equal_weight": "EqWt",
}


def print_optimal_portfolio(result: OptimalPortfolioResult) -> None:
    KEYS   = list(result.strategies.keys())
    LABELS = [_SHORT_LABELS.get(k, k) for k in KEYS]
    TC, NC, SC = 10, 28, 10
    W = TC + NC + SC * len(KEYS) + 2   # +2 for leading "  " indent

    print(f"  Common history : {result.common_from} → {result.common_to}"
          f"  ({result.n_years:.1f} yr,  {result.n_trading_days:,} days)")
    if result.excluded:
        for t, reason in result.excluded:
            print(f"  Excluded       : {t}  ({reason})")

    if result.shrinkage_info is not None:
        si = result.shrinkage_info
        print()
        print(f"  [Advanced] Bayesian shrinkage active")
        print(f"    Covariance : OAS estimator  "
              f"(δ = {si['delta_oas']:.2f}  →  diagonal target)")
        print(f"    Returns    : James-Stein     "
              f"(κ = {si['kappa_mean']:.2f}  →  grand mean "
              f"{si['grand_mean_ann']*100:.1f}%/yr)")
        print(f"    {'Ticker':>6}   {'Sample':>8}   {'Shrunk':>8}   {'Δ':>6}")
        for tkr in result.eligible_tickers:
            s = si["mu_sample_ann"].get(tkr, 0.0)
            sh = si["mu_shrunk_ann"].get(tkr, 0.0)
            print(f"    {_short_ticker(tkr):>6}   {s*100:>+7.1f}%   {sh*100:>+7.1f}%"
                  f"   {(sh-s)*100:>+5.1f}pp")
    print()

    # ── Legend ───────────────────────────────────────────────────────────────
    n_rolling = max(0, int((result.n_years - 10) * 252))
    print(f"  Metrics explained:")
    print(f"  Sharpe  = (CAGR − risk-free) / 5y-vol.  Measures total risk-adjusted return.")
    print(f"  Sortino = (CAGR − risk-free) / downside-vol.  Like Sharpe but penalises only")
    print(f"            negative returns — more relevant for sequence-of-returns risk.")
    print(f"  P50/P5/P1 CAGR = median / 5th / 1st percentile of rolling 10y CAGRs in the sample.")
    print(f"            P5: 95% of start dates beat it.  P1: extreme worst-case entry point.")
    print(f"  σ annual  = std of calendar-year returns over the full history ({int(result.n_years)} years).")
    print(f"            Consistent with MDD: a -30% MDD portfolio typically shows σ ≈ 15–20%.")
    print(f"  ⚠ Only {n_rolling} rolling 10y windows available ({result.n_years:.1f}yr history − 10yr).")
    print(f"    P50/P5 reflect entry-point noise within one bull-market cycle, not")
    print(f"    the full range of regimes. Historical MSCI World worst 10yr: ~0% (2000–2010).")
    print(f"  ⚠ Data starts {result.common_from} — post-financial-crisis bull market.")
    print(f"    Realized CAGRs of 13–15% likely overstate forward expectations.")
    print(f"    Conservative long-run assumption: 6–8% real / 8–10% nominal.")
    print()

    # ── Header ───────────────────────────────────────────────────────────────
    hdr = f"  {'Ticker':<{TC}}{'Name':<{NC}}"
    for label in LABELS:
        hdr += f"{label:>{SC}}"
    print(hdr)
    sep = "  " + "─" * (TC + NC + SC * len(KEYS))
    print(sep)

    # Weight rows
    for ticker in result.eligible_tickers:
        eu_flag = " ✓" if ticker in EU_TICKERS else ""
        label = (_short_ticker(ticker) + eu_flag)[:TC]
        raw_name = result.ticker_names.get(ticker, "")
        for suffix in (" UCITS ETF USD (Acc)", " UCITS ETF USD Distributing",
                       " UCITS ETF 1C", " UCITS ETF", " Trust", " ETC"):
            raw_name = raw_name.replace(suffix, "")
        name = raw_name[:NC]
        row = f"  {label:<{TC}}{name:<{NC}}"
        for key in KEYS:
            w = result.strategies[key].weights.get(ticker, 0.0)
            cell = f"{w * 100:.0f}%" if w >= 0.005 else "—"
            row += f"{cell:>{SC}}"
        print(row)

    print(sep)

    # Stats rows
    def _r(label: str, vals: list[str]) -> None:
        row = f"  {label:<{TC + NC}}"
        for v in vals:
            row += f"{v:>{SC}}"
        print(row)

    def _pv(v: float | None, d: int = 1) -> str:
        return "n/a" if v is None else f"{v * 100:.{d}f}%"

    def _rv(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.2f}"

    use_p50 = result.n_years >= 10
    cagr_label = "P50-10y CAGR" if use_p50 else f"CAGR ({result.n_years:.0f}yr)"
    p5_label   = "P5-10y CAGR"  if use_p50 else "P5 CAGR"
    p1_label   = "P1-10y CAGR"  if use_p50 else "P1 CAGR"
    _r(cagr_label,    [_pv(result.strategies[k].portfolio_cagr_p50 if use_p50
                           else result.strategies[k].portfolio_cagr) for k in KEYS])
    _r(p5_label,      [_pv(result.strategies[k].portfolio_cagr_p5)  for k in KEYS])
    _r(p1_label,      [_pv(result.strategies[k].portfolio_cagr_p1)  for k in KEYS])
    _r("σ annual ret", [_pv(result.strategies[k].portfolio_annual_ret_std) for k in KEYS])
    _r("Ann. σ-5y",   [_pv(result.strategies[k].portfolio_vol_5y)   for k in KEYS])
    _r("Sortino",     [_rv(result.strategies[k].portfolio_sortino)   for k in KEYS])
    _r("Sharpe",      [_rv(result.strategies[k].portfolio_sharpe)    for k in KEYS])
    _r("MDD",         [_pv(result.strategies[k].portfolio_mdd)       for k in KEYS])
    _r("Avg corr",    [_rv(result.strategies[k].portfolio_avg_corr)  for k in KEYS])
    print(sep)

    rec = result.strategies[result.recommended]
    rec_cagr = rec.portfolio_cagr_p50 if use_p50 else rec.portfolio_cagr
    rec_p5   = rec.portfolio_cagr_p5
    rec_p1   = rec.portfolio_cagr_p1
    rec_std  = rec.portfolio_annual_ret_std
    cagr_label_footer = "P50-10y CAGR" if use_p50 else f"CAGR ({result.n_years:.0f}yr)"
    cagr_str = f"  {cagr_label_footer} = {rec_cagr * 100:.1f}%" if rec_cagr is not None else ""
    p5_str   = f"  P5 = {rec_p5 * 100:.1f}%" if rec_p5 is not None else ""
    p1_str   = f"  P1 = {rec_p1 * 100:.1f}%" if rec_p1 is not None else ""
    std_str  = f"  σ-annual = {rec_std * 100:.1f}%" if rec_std is not None else ""
    print(f"\n  ★ Recommended : {rec.name}{cagr_str}{p5_str}{p1_str}{std_str}")
    print(f'    Set  selected_strategy = "{result.recommended}"  in [etf_optimizer] to use these weights.')
    print()


# ── Cardinality sweep ─────────────────────────────────────────────────────────

def compute_cardinality_sweep(
    results: list[TickerResult],
    prices_eur: dict[str, pd.Series],
    eligible: list[str],
    max_weight_per_ticker: float,
    qqq_max_weight: float,
    gold_max_weight: float,
    min_history_years: float,
    strategy_key: str = "max_sortino",
    max_k: int | None = None,
    use_bayesian_shrinkage: bool = False,
    shrink_mean_intensity: float = 0.5,
) -> list[tuple[int, OptimalPortfolioResult]]:
    """For each k from 2 to min(len(eligible), max_k), enumerate all C(n,k) subsets,
    run the optimizer, and return the best portfolio per k for the given strategy."""
    import contextlib
    import hashlib
    import io
    from itertools import combinations

    k_max = min(len(eligible), max_k) if max_k is not None else len(eligible)
    sweep: list[tuple[int, OptimalPortfolioResult]] = []

    for k in range(2, k_max + 1):
        best_score: tuple[int, float] = (-1, -float("inf"))
        best_result: OptimalPortfolioResult | None = None

        for subset in combinations(eligible, k):
            # Deterministic per-subset seed so each subset explores its own
            # restart starts but the full sweep is still reproducible.
            subset_seed = int(
                hashlib.md5(",".join(subset).encode()).hexdigest()[:8], 16
            )
            with contextlib.redirect_stdout(io.StringIO()):
                opt = compute_optimal_portfolio(
                    results=results,
                    prices_eur=prices_eur,
                    optimize_tickers=list(subset),
                    max_weight_per_ticker=max_weight_per_ticker,
                    qqq_max_weight=qqq_max_weight,
                    gold_max_weight=gold_max_weight,
                    min_weight_threshold=0.0,
                    min_history_years=min_history_years,
                    use_bayesian_shrinkage=use_bayesian_shrinkage,
                    shrink_mean_intensity=shrink_mean_intensity,
                    seed=subset_seed,
                )
            if opt is None or strategy_key not in opt.strategies:
                continue
            strat = opt.strategies[strategy_key]
            # Tuple score: P5-available subsets always rank above P5=None ones, so
            # the sweep never recommends a combo whose worst-case can't be measured.
            if strat.portfolio_cagr_p5 is not None:
                score = (1, strat.portfolio_cagr_p5)
            elif strat.portfolio_cagr_p50 is not None:
                score = (0, strat.portfolio_cagr_p50)
            else:
                score = (0, strat.portfolio_cagr)
            if best_result is None or score > best_score:
                best_score = score
                best_result = opt

        if best_result is not None:
            sweep.append((k, best_result))

    return sweep


def print_portfolio_final_summary(
    opt_result: "OptimalPortfolioResult",
    strategy_key: str,
    sweep: "list[tuple[int, OptimalPortfolioResult]] | None" = None,
    sp500_prices: "pd.Series | None" = None,
) -> None:
    """Compact table — k | Allocation | P50 | P5 | Sharpe | Sortino | MDD | Avg corr.
    σ is dropped vs the verbose optimiser table; everything else relevant to a
    40yr decumulation decision is kept."""
    strat_label = {
        "max_sortino": "Max Sortino", "max_sharpe": "Max Sharpe",
        "min_variance": "Min Variance", "max_cagr": "Max CAGR",
        "max_p50_cagr": "Max P50 CAGR", "max_p1_cagr": "Max P1 CAGR",
        "equal_weight": "Equal Weight",
    }.get(strategy_key, strategy_key)

    k_w     = 4   # widened from 3 to fit "★all" / "★2" / "☆3" prefixes
    p50_w   = 6
    p5_w    = 6
    ratio_w = 6   # Sharpe / Sortino
    mdd_w   = 7
    corr_w  = 6
    fixed = (2 + k_w + 2 + 2 + p50_w + 2 + p5_w + 2 + ratio_w + 2
             + ratio_w + 2 + mdd_w + 2 + corr_w)
    alloc_w = max(40, shutil.get_terminal_size((120, 50)).columns - fixed)

    def _alloc(result: "OptimalPortfolioResult", key: str) -> str:
        strat = result.strategies[key]
        parts = [
            f"{_short_ticker(t)} {strat.weights.get(t, 0) * 100:.0f}%"
            for t in result.eligible_tickers
            if strat.weights.get(t, 0) >= 0.005
        ]
        return " + ".join(parts)

    def _row(k_label: str, alloc: str, p50: str, p5: str,
             sharpe: str, sortino: str, mdd: str, corr: str) -> None:
        print(f"  {k_label:>{k_w}}  {alloc:<{alloc_w}}"
              f"  {p50:>{p50_w}}  {p5:>{p5_w}}"
              f"  {sharpe:>{ratio_w}}  {sortino:>{ratio_w}}"
              f"  {mdd:>{mdd_w}}  {corr:>{corr_w}}")

    def _fmt_strat(strat: "PortfolioStrategy", n_years: float) -> tuple[str, str, str, str, str, str]:
        use_p50 = n_years >= 10
        cagr   = strat.portfolio_cagr_p50 if use_p50 else strat.portfolio_cagr
        p50    = f"{cagr * 100:.1f}%" if cagr is not None else "n/a"
        p5     = f"{strat.portfolio_cagr_p5 * 100:.1f}%" if strat.portfolio_cagr_p5 is not None else "n/a"
        sharpe = f"{strat.portfolio_sharpe:.2f}" if strat.portfolio_sharpe is not None else "n/a"
        srt    = f"{strat.portfolio_sortino:.2f}" if strat.portfolio_sortino is not None else "n/a"
        mdd    = f"{strat.portfolio_mdd * 100:.1f}%" if strat.portfolio_mdd is not None else "n/a"
        corr   = f"{strat.portfolio_avg_corr:.2f}" if strat.portfolio_avg_corr is not None else "n/a"
        return p50, p5, sharpe, srt, mdd, corr

    def _flag_alloc(result: "OptimalPortfolioResult", key: str) -> str:
        strat = result.strategies.get(key)
        prefix = "⚠ " if (strat is not None and strat.portfolio_cagr_p5 is None) else ""
        return prefix + _alloc(result, key)

    # Tuple score per row: (P5-available flag, P5/P50/CAGR). Same ranking
    # used by the recommender — a row with measurable P5 always beats a row
    # with P5=n/a (history < 10y), since a 40yr decum needs a real worst-case.
    def _row_score(strat: "PortfolioStrategy") -> tuple[int, float]:
        if strat.portfolio_cagr_p5 is not None:
            return (1, strat.portfolio_cagr_p5)
        if strat.portfolio_cagr_p50 is not None:
            return (0, strat.portfolio_cagr_p50)
        return (0, strat.portfolio_cagr)

    # Collect candidates in display order, then rank to find top-1 / top-2.
    candidates: list[tuple[str, "OptimalPortfolioResult", "PortfolioStrategy"]] = []
    if strategy_key in opt_result.strategies:
        candidates.append(("all", opt_result, opt_result.strategies[strategy_key]))
    if sweep:
        for k, result in sweep:
            if strategy_key in result.strategies:
                candidates.append((str(k), result, result.strategies[strategy_key]))

    ranked_labels = [c[0] for c in sorted(candidates,
                                          key=lambda c: _row_score(c[2]),
                                          reverse=True)]
    top1 = ranked_labels[0] if len(ranked_labels) >= 1 else None
    top2 = ranked_labels[1] if len(ranked_labels) >= 2 else None

    def _mark(label: str) -> str:
        if label == top1:
            return f"★{label}"
        if label == top2:
            return f"☆{label}"
        return label

    W = fixed + alloc_w
    print(f"{'OPTIMAL PORTFOLIO — FINAL SUMMARY':^{W}}")
    print(f"  Strategy : {strat_label}   ·   ★ recommended   ☆ runner-up   ·   ⚠ = P5 n/a (history < 10y)")
    print()

    _row("k", "Allocation", "P50", "P5", "Sharpe", "Sortino", "MDD", "Avg corr")
    print("  " + "─" * (W - 2))

    for label, result, strat in candidates:
        p50, p5, sharpe, srt, mdd, corr = _fmt_strat(strat, result.n_years)
        _row(_mark(label), _flag_alloc(result, strategy_key),
             p50, p5, sharpe, srt, mdd, corr)

    if sp500_prices is not None:
        ref = _era_stats(sp500_prices, "2000-01-01", None)
        if ref["cagr"] is not None:
            print("  " + "─" * (W - 2))
            _row(
                "ref",
                f"S&P 500  2000–today  [{ref['yr_from']}–{ref['yr_to']}]",
                f"{ref['cagr'] * 100:.1f}%", "—",
                "—", "—",
                f"{ref['mdd']  * 100:.1f}%", "—",
            )

    print()


# ── Benchmark indices ─────────────────────────────────────────────────────────

BENCHMARK_WINDOWS: list[int] = [5, 10, 15, 20, 25]

# Human-readable names (Yahoo Finance .info can be slow for indices; hardcode fallback)
_BENCHMARK_NAMES: dict[str, str] = {
    "^GSPC":     "S&P 500",
    "^IXIC":     "NASDAQ Composite",
    "^STOXX50E": "Euro Stoxx 50",
}


def fetch_benchmark_series(
    ticker: str,
) -> tuple[pd.Series | None, str, str]:
    """Fetch and cache a benchmark index series in its native currency.
    Returns (prices, status_string, currency).
    No EUR conversion — avoids truncation from the short EURUSD=X history (2003+).
    """
    cache_name = ticker.replace("^", "IDX_")   # "^" is invalid in filenames
    df, status = _fetch_with_cache(ticker, cache_name)
    if df is None or df.empty or "Close" not in df.columns:
        return None, "failed", ""

    raw = df["Close"].dropna()
    if len(raw) < 2:
        return None, "failed", ""

    prices = raw.copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices[~prices.index.duplicated(keep="last")]
    currency = _KNOWN_CURRENCIES.get(ticker, "USD")
    return prices, status, currency


def compute_benchmark_rolling(
    prices: pd.Series,
    windows: list[int] = BENCHMARK_WINDOWS,
) -> dict[int, dict[str, float | None]]:
    """For each window (years), return mean/P5/P50/P95/n of all rolling CAGRs."""
    out: dict[int, dict[str, float | None]] = {}
    for yrs in windows:
        cagrs = _rolling_cagrs(prices, yrs)
        if cagrs is None or len(cagrs) == 0:
            out[yrs] = {"mean": None, "p1": None, "p5": None, "p50": None, "p95": None, "n": 0}
        else:
            out[yrs] = {
                "mean": float(np.mean(cagrs)),
                "p1":   float(np.percentile(cagrs, 1)),
                "p5":   float(np.percentile(cagrs, 5)),
                "p50":  float(np.percentile(cagrs, 50)),
                "p95":  float(np.percentile(cagrs, 95)),
                "n":    len(cagrs),
            }
    return out


def print_benchmark_table(
    benchmark_data: list[tuple[str, pd.Series, str, dict[int, dict[str, float | None]]]],
    windows: list[int] = BENCHMARK_WINDOWS,
    title: str = "BENCHMARK INDICES — ROLLING CAGR",
    note: str = "",
) -> None:
    """Print rolling CAGR table for benchmark indices (mean of all start dates)."""
    if not benchmark_data:
        return

    W   = 94
    NC  = 22   # name column
    DC  = 14   # history + currency column
    WC  = 10   # per-window column

    print(f"{title:^{W}}")
    if note:
        print(f"  {note}")
    print(f"  Each cell = mean CAGR across every possible start date for that window.")
    print(f"  P5/P95 show the spread due to entry-point luck.")
    print()

    # Header
    hdr = f"  {'Index':<{NC}}  {'History (ccy)':<{DC}}"
    for y in windows:
        hdr += f"  {f'{y}y mean':>{WC}}"
    print(hdr)

    print("  " + "─" * (W - 2))

    for ticker, prices, currency, rolling in benchmark_data:
        name = _BENCHMARK_NAMES.get(ticker, ticker)
        span = f"{prices.index[0].year}–{prices.index[-1].year} [{currency}]"
        row  = f"  {name:<{NC}}  {span:<{DC}}"
        for y in windows:
            s = rolling[y]
            cell = f"{s['mean'] * 100:.1f}%" if s["mean"] is not None else "n/a"
            row += f"  {cell:>{WC}}"
        print(row)

        # P5–P95 sub-row
        sub_range = f"  {'':<{NC}}  {'P5–P95:':>{DC}}"
        for y in windows:
            s = rolling[y]
            cell = f"{s['p5']*100:.1f}%–{s['p95']*100:.1f}%" if s["p5"] is not None else ""
            sub_range += f"  {cell:>{WC}}"
        print(sub_range)

        # P1 sub-row
        sub_p1 = f"  {'':<{NC}}  {'P1:':>{DC}}"
        for y in windows:
            s = rolling[y]
            cell = f"{s['p1'] * 100:.1f}%" if s["p1"] is not None else ""
            sub_p1 += f"  {cell:>{WC}}"
        print(sub_p1)

    print()


# Era boundaries: (label, start_inclusive, end_exclusive)  None = data boundary
_ERAS: list[tuple[str, str | None, str | None]] = [
    ("Pre-2000",   None,         "2000-01-01"),
    ("2000–2009",  "2000-01-01", "2010-01-01"),
    ("2009–today", "2009-01-01", None),
    ("2000–today", "2000-01-01", None),
]


def _era_stats(prices: pd.Series, start: str | None, end: str | None) -> dict:
    """Point-to-point CAGR, MDD, and annualised vol for a price slice."""
    s = prices.copy()
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    if end:
        s = s[s.index < pd.Timestamp(end)]
    if len(s) < 20:
        return {"cagr": None, "mdd": None, "vol": None, "yr_from": None, "yr_to": None}
    n_years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / n_years) - 1
    cum  = s / s.iloc[0]
    peak = cum.cummax()
    mdd  = float(((cum - peak) / peak).min())
    log_ret = np.log(s).diff().dropna()
    tdpy = len(log_ret) / n_years          # trading days per year (handles crypto)
    vol  = float(np.std(log_ret, ddof=1) * np.sqrt(tdpy))
    return {
        "cagr":    float(cagr),
        "mdd":     mdd,
        "vol":     vol,
        "yr_from": s.index[0].year,
        "yr_to":   s.index[-1].year,
    }


def _era_rolling_stats(prices: pd.Series, start: str | None, end: str | None) -> dict:
    """p5/p10/p50/p90 of CAGR, MDD, and vol across all entry days in the first year of the era."""
    _null = {k: None for k in (
        "cagr_p5",  "cagr_p10", "cagr_p50", "cagr_p90",
        "mdd_p10",  "mdd_p50",  "mdd_p90",
        "vol_p10",  "vol_p50",  "vol_p90",
        "yr_from",  "yr_to",
    )}
    s = prices.copy()
    if end:
        s = s[s.index < pd.Timestamp(end)]
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    if len(s) < 20:
        return _null

    year_one_end = s.index[0] + pd.DateOffset(years=1)
    entry_days   = s.index[s.index < year_one_end]

    cagrs, mdds, vols = [], [], []
    yr_to = None
    for day in entry_days:
        st = _era_stats(prices, day.strftime("%Y-%m-%d"), end)
        if st["cagr"] is None:
            continue
        cagrs.append(st["cagr"])
        mdds.append(st["mdd"])
        vols.append(st["vol"])
        yr_to = st["yr_to"]

    if not cagrs:
        return _null

    ca = np.array(cagrs)
    md = np.array(mdds)
    vo = np.array(vols)
    return {
        "cagr_p5":  float(np.percentile(ca, 5)),
        "cagr_p10": float(np.percentile(ca, 10)), "cagr_p50": float(np.percentile(ca, 50)), "cagr_p90": float(np.percentile(ca, 90)),
        "mdd_p10":  float(np.percentile(md, 10)), "mdd_p50":  float(np.percentile(md, 50)), "mdd_p90":  float(np.percentile(md, 90)),
        "vol_p10":  float(np.percentile(vo, 10)), "vol_p50":  float(np.percentile(vo, 50)), "vol_p90":  float(np.percentile(vo, 90)),
        "yr_from": s.index[0].year,
        "yr_to":   yr_to,
    }


def _fmt_pct_triple(p10: float, p50: float, p90: float, sign: bool = False) -> str:
    fmt = "+.1f" if sign else ".1f"
    return f"{p10*100:{fmt}}/{p50*100:{fmt}}/{p90*100:{fmt}}%"


def print_benchmark_era_table(
    benchmark_data: list[tuple[str, pd.Series, str, dict]],
    era_horizon_year: int | None = None,
    horizon_years: int | None = None,
) -> None:
    """Per-regime p10/p50/p90 CAGR, MDD, and vol — rolling over entry days in year one of each era."""
    if not benchmark_data:
        return

    eras = list(_ERAS)
    if horizon_years is not None:
        end_year = 2000 + horizon_years
        eras.append((f"2000–{end_year}", "2000-01-01", f"{end_year + 1}-01-01"))
    if era_horizon_year is not None:
        eras.append((f"2000–{era_horizon_year}", "2000-01-01", f"{era_horizon_year + 1}-01-01"))

    NC, EC, CC, VC = 20, 14, 12, 22   # name, era, coverage, triple-value col widths
    W = NC + EC + CC + VC * 3 + 14

    print("BENCHMARK INDICES — ERA ANALYSIS")
    print("  p10 / p50 / p90 across all entry days in the first calendar year of each era.")
    print("  p10 = bad-luck entry, p50 = typical, p90 = good-luck entry.")
    if era_horizon_year is not None:
        print(f"  2000–{era_horizon_year} : lost-decade stress test — {era_horizon_year - 2000}yr horizon starting at the dot-com peak.")
    print()
    hdr = (f"  {'Index':<{NC}}  {'Era':<{EC}}  {'Coverage':<{CC}}"
           f"  {'CAGR p10/p50/p90':>{VC}}  {'MDD p10/p50/p90':>{VC}}  {'σ-ann p10/p50/p90':>{VC}}")
    print(hdr)

    for ticker, prices, currency, _ in benchmark_data:
        name = _BENCHMARK_NAMES.get(ticker, ticker)
        print("  " + "─" * (W - 2))
        for era_label, era_start, era_end in eras:
            s = _era_rolling_stats(prices, era_start, era_end)
            if s["cagr_p50"] is None:
                continue
            coverage = f"{s['yr_from']}–{s['yr_to']}"
            cagr_s = _fmt_pct_triple(s["cagr_p10"], s["cagr_p50"], s["cagr_p90"], sign=True)
            mdd_s  = _fmt_pct_triple(s["mdd_p10"],  s["mdd_p50"],  s["mdd_p90"])
            vol_s  = _fmt_pct_triple(s["vol_p10"],  s["vol_p50"],  s["vol_p90"])
            print(f"  {name:<{NC}}  {era_label:<{EC}}  {coverage:<{CC}}"
                  f"  {cagr_s:>{VC}}  {mdd_s:>{VC}}  {vol_s:>{VC}}")
            name = ""

    print()


def print_forward_cagr_estimates(
    benchmark_data: list[tuple[str, pd.Series, str, dict]],
    horizon_years: int,
    inflation: float,
    cape_ratio: float | None = None,
    cape_long_run_mean: float = 17.0,
    cape_eps_growth: float = 0.015,
    n_simulations: int = 50_000,
    initial_stock: float = 0.0,
    monthly_contribution: float = 0.0,
    verbose: bool = True,
) -> dict:
    """CAPE-implied forward CAGR + two-phase lost-decade Monte Carlo stress test.
    Always computes; printing is gated by `verbose`. Returns a dict of key metrics
    for downstream consumers (e.g. the take-away recap)."""
    summary: dict = {}
    W = 70
    if verbose:
        title = f"FORWARD CAGR ESTIMATES — HORIZON {horizon_years}yr"
        print(f"{title:^{W}}")
        print()

    # ── CAPE-implied forward CAGR ─────────────────────────────────────────────
    if cape_ratio is not None and cape_ratio > 0:
        earnings_yield = 1.0 / cape_ratio
        # Valuation drag: CAPE mean-reverts to long-run mean over the horizon
        # contribution = (CAPE_mean/CAPE)^(1/N) - 1  (negative when CAPE > mean)
        reversion = (cape_long_run_mean / cape_ratio) ** (1.0 / horizon_years) - 1.0
        real_cagr    = earnings_yield + cape_eps_growth + reversion
        nominal_cagr = (1.0 + real_cagr) * (1.0 + inflation) - 1.0
        summary["cape_ratio"]         = cape_ratio
        summary["cape_long_run_mean"] = cape_long_run_mean
        summary["cape_real_cagr"]     = real_cagr
        summary["cape_nominal_cagr"]  = nominal_cagr

        if verbose:
            print("  CAPE-implied forward CAGR:")
            print(f"    Current CAPE          : {cape_ratio:.1f}")
            print(f"    Earnings yield (1/CAPE): {earnings_yield * 100:+.2f}%")
            print(f"    Real EPS growth       : {cape_eps_growth * 100:+.2f}%")
            print(f"    CAPE reversion drag   : {reversion * 100:+.2f}%"
                  f"  ({cape_ratio:.0f} → {cape_long_run_mean:.0f} over {horizon_years}yr)")
            print(f"    {'─' * 38}")
            print(f"    Implied real CAGR     : {real_cagr * 100:.1f}%")
            print(f"    Implied nominal CAGR  : {nominal_cagr * 100:.1f}%"
                  f"  (+ {inflation * 100:.1f}% inflation)")
            print()

    # ── Two-phase stress test ─────────────────────────────────────────────────
    # Use S&P 500 era stats (2000-2009 = lost decade, 2009-today = recovery)
    sp500_prices = next(
        (prices for ticker, prices, _, _ in benchmark_data if ticker == "^GSPC"), None
    )
    if sp500_prices is None:
        return summary

    # Phase 1 = lost decade (2000-2009, fixed 10y). Phase 2 = early recovery
    # within the user's scenario horizon: end at year (2000 + horizon_years) so
    # a 15yr horizon samples the 2009-2015 sub-window rather than 2009-today.
    # This makes the MC's "2000-2015 redux" actually match the historical
    # 2000-2015 row in the take-away table.
    rec_end = f"{2000 + horizon_years + 1}-01-01"
    lost = _era_stats(sp500_prices, "2000-01-01", "2010-01-01")
    rec  = _era_stats(sp500_prices, "2009-01-01", rec_end)
    if lost["cagr"] is None or rec["cagr"] is None:
        return summary

    # Geometric → arithmetic mean:  arithmetic ≈ geometric + σ²/2
    phase1_mean = lost["cagr"] + 0.5 * lost["vol"] ** 2
    phase2_mean = rec["cagr"]  + 0.5 * rec["vol"]  ** 2
    phase1_std  = lost["vol"]
    phase2_std  = rec["vol"]
    phase1_yrs  = min(10, horizon_years)
    phase2_yrs  = max(0, horizon_years - phase1_yrs)

    rng = np.random.default_rng(42)
    r1  = np.maximum(rng.normal(phase1_mean, phase1_std, (n_simulations, phase1_yrs)), -0.99)
    fv  = np.prod(1.0 + r1, axis=1)
    if phase2_yrs > 0:
        r2  = np.maximum(rng.normal(phase2_mean, phase2_std, (n_simulations, phase2_yrs)), -0.99)
        fv  = fv * np.prod(1.0 + r2, axis=1)
    cagrs = fv ** (1.0 / horizon_years) - 1.0

    p5, p25, p50, p75, p95 = (float(np.percentile(cagrs, p)) for p in (5, 25, 50, 75, 95))
    prob_positive = float(np.mean(cagrs > 0)) * 100
    prob_above5   = float(np.mean(cagrs > 0.05)) * 100
    summary["lump_p5"]        = p5
    summary["lump_p25"]       = p25
    summary["lump_p50"]       = p50
    summary["lump_p75"]       = p75
    summary["lump_p95"]       = p95
    summary["lump_prob_pos"]  = prob_positive
    summary["lump_prob_5pct"] = prob_above5

    if verbose:
        print("  Two-phase stress test  (S&P 500 as equity proxy, native USD):")
        print(f"    Phase 1 — {phase1_yrs:2d}yr lost decade : "
              f"mean={phase1_mean * 100:+.1f}%  σ={phase1_std * 100:.1f}%"
              f"  ({lost['yr_from']}–{lost['yr_to']})")
        if phase2_yrs > 0:
            print(f"    Phase 2 — {phase2_yrs:2d}yr recovery    : "
                  f"mean={phase2_mean * 100:+.1f}%  σ={phase2_std * 100:.1f}%"
                  f"  ({rec['yr_from']}–{rec['yr_to']})")
        print(f"    {n_simulations:,} simulations  |  total horizon {horizon_years}yr")
        print(f"    {'─' * 38}")
        print(f"    P5  CAGR : {p5  * 100:.1f}%   (worst 5% of outcomes)")
        print(f"    P25 CAGR : {p25 * 100:.1f}%")
        print(f"    P50 CAGR : {p50 * 100:.1f}%   (median outcome)")
        print(f"    P75 CAGR : {p75 * 100:.1f}%")
        print(f"    P95 CAGR : {p95 * 100:.1f}%   (best 5% of outcomes)")
        print(f"    P(CAGR>0) : {prob_positive:.0f}%   P(CAGR>5%) : {prob_above5:.0f}%")
        print()

    # ── DCA stress test ───────────────────────────────────────────────────────
    # Same two-phase scenario but with ongoing monthly contributions. Money-weighted
    # IRR is the apples-to-apples comparison vs lump-sum CAGR (gives each contribution
    # credit for the time it was actually invested).
    if monthly_contribution > 0:
        annual_C = monthly_contribution * 12.0
        total_invested = initial_stock + annual_C * horizon_years

        # Simulate portfolio year-by-year; contributions arrive monthly, so use
        # the annuity FV formula for each year's cash flows.
        rng2 = np.random.default_rng(43)
        r1_dca = np.maximum(
            rng2.normal(phase1_mean, phase1_std, (n_simulations, phase1_yrs)), -0.99
        )
        r2_dca = (
            np.maximum(
                rng2.normal(phase2_mean, phase2_std, (n_simulations, phase2_yrs)), -0.99
            )
            if phase2_yrs > 0
            else np.empty((n_simulations, 0))
        )
        all_r = np.concatenate([r1_dca, r2_dca], axis=1)  # (n_sim, horizon_years)

        portfolio = np.full(n_simulations, initial_stock, dtype=float)
        for yr in range(horizon_years):
            annual_r = all_r[:, yr]
            # Monthly rate for annuity factor; guard against near-zero annual return
            monthly_r = (1.0 + annual_r) ** (1.0 / 12) - 1.0
            safe = np.abs(monthly_r) > 1e-9
            annuity_fv = np.where(
                safe,
                monthly_contribution * ((1.0 + monthly_r) ** 12 - 1.0) / monthly_r,
                annual_C,
            )
            portfolio = portfolio * (1.0 + annual_r) + annuity_fv

        # Money-weighted IRR per simulation: annual r solving
        #   p0*(1+r)^N + monthly_c*((1+r_m)^(12N)-1)/r_m == final_portfolio
        # Vectorised bisection (monotonic in r; ~50 iters is overkill for 1e-7 precision).
        months = 12 * horizon_years
        lo = np.full(n_simulations, -0.99)
        hi = np.full(n_simulations,  5.00)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            r_m = (1.0 + mid) ** (1.0 / 12) - 1.0
            ok  = np.abs(r_m) > 1e-12
            annuity_proj = np.where(
                ok,
                monthly_contribution * ((1.0 + r_m) ** months - 1.0) / r_m,
                monthly_contribution * months,
            )
            proj = initial_stock * (1.0 + mid) ** horizon_years + annuity_proj
            too_low = proj < portfolio
            lo = np.where(too_low, mid, lo)
            hi = np.where(too_low, hi, mid)
        irr = 0.5 * (lo + hi)

        rp5, rp25, rp50, rp75, rp95 = (
            float(np.percentile(irr, p)) for p in (5, 25, 50, 75, 95)
        )
        fp5, fp25, fp50, fp75, fp95 = (
            float(np.percentile(portfolio / 1e3, p)) for p in (5, 25, 50, 75, 95)
        )
        invested_k = total_invested / 1e3
        pl5, pl25, pl50, pl75, pl95 = (
            fp - invested_k for fp in (fp5, fp25, fp50, fp75, fp95)
        )
        prob_irr_pos  = float(np.mean(irr > 0))   * 100
        prob_irr_5pct = float(np.mean(irr > 0.05)) * 100
        summary["dca_p5_irr"]         = rp5
        summary["dca_p25_irr"]        = rp25
        summary["dca_p50_irr"]        = rp50
        summary["dca_p75_irr"]        = rp75
        summary["dca_p95_irr"]        = rp95
        summary["dca_p5_final_k"]     = fp5
        summary["dca_p50_final_k"]    = fp50
        summary["dca_p95_final_k"]    = fp95
        summary["dca_p5_pl_k"]        = pl5
        summary["dca_p50_pl_k"]       = pl50
        summary["dca_p95_pl_k"]       = pl95
        summary["dca_total_invested"] = total_invested

        if verbose:
            print("  DCA stress test  (same two-phase scenario + monthly contributions):")
            print(f"    Initial stock portfolio : €{initial_stock:>10,.0f}")
            print(f"    Monthly contribution    : €{monthly_contribution:>10,.0f}")
            print(f"    Total invested (horizon): €{total_invested:>10,.0f}")
            print(f"    {'─' * 70}")
            print(f"    {'':20}  {'IRR (ann.)':>11}   {'Final portfolio (k€)':>20}   {'P&L (k€)':>12}")
            print(f"    {'P5  (worst 5%)':20}  {rp5*100:>10.1f}%   {fp5:>20.0f}   {pl5:>+12,.0f}")
            print(f"    {'P25':20}  {rp25*100:>10.1f}%   {fp25:>20.0f}   {pl25:>+12,.0f}")
            print(f"    {'P50  (median)':20}  {rp50*100:>10.1f}%   {fp50:>20.0f}   {pl50:>+12,.0f}")
            print(f"    {'P75':20}  {rp75*100:>10.1f}%   {fp75:>20.0f}   {pl75:>+12,.0f}")
            print(f"    {'P95  (best 5%)':20}  {rp95*100:>10.1f}%   {fp95:>20.0f}   {pl95:>+12,.0f}")
            print(f"    {'─' * 70}")
            print(f"    P(IRR>0) : {prob_irr_pos:.0f}%   P(IRR>5%) : {prob_irr_5pct:.0f}%")
            print(f"    vs lump-sum P50 CAGR: {p50*100:.1f}%  →  DCA P50 IRR: {rp50*100:.1f}%"
                  f"  (Δ={(rp50 - p50) * 100:+.1f}pp,  money-weighted)")
            print()

    return summary


# ── Take-away recap ───────────────────────────────────────────────────────────

def print_take_away(
    forward_summary: dict,
    native_data: list,
    opt_result: "OptimalPortfolioResult | None",
    sweep: "list[tuple[int, OptimalPortfolioResult]] | None",
    sweep_strategy: str,
    horizon_years: int,
    config_stock_cagr: float,
    fx_cache: dict | None = None,
    planning_cagr_override: float | None = None,
) -> None:
    """Compact recap — instrument selection + CAGR realism check.
    Decumulation survival lives in main_decumulation.py (run that for SORR analysis)."""
    W = 70
    print(f"{'TAKE AWAY — CAGR ANALYSIS':^{W}}")
    print()

    # ── 1. Assumed CAGR vs the real world ────────────────────────────────────
    # Verdict applies to the ACTIVE planning value (override if set, else config).
    # When --cagr overrides the baseline, the verdict moves to the Planning line
    # — otherwise the user sees "config 6% → optimistic" while planning at 4%.
    nom_cagr   = forward_summary.get("cape_nominal_cagr") if forward_summary else None
    active     = planning_cagr_override if planning_cagr_override is not None else config_stock_cagr
    delta      = active - nom_cagr if nom_cagr is not None else None
    if delta is None:
        verdict = ""
    elif delta > 0.02:
        verdict = "   → optimistic"
    elif delta < -0.02:
        verdict = "   → conservative"
    else:
        verdict = "   → in range"

    print("  1. Expected CAGR — does the assumed return exist in the real world?")
    # Baseline line gets the verdict only when there's no override.
    baseline_suffix = "" if planning_cagr_override is not None else verdict
    print(f"     Assumed CAGR (config) : {config_stock_cagr*100:+.1f}% / yr{baseline_suffix}")
    if nom_cagr is not None:
        print(f"     CAPE-implied nominal  : {nom_cagr*100:+.1f}% / yr")
    if planning_cagr_override is not None:
        print(f"     Planning CAGR (CLI)   : {planning_cagr_override*100:+.1f}% / yr{verdict}"
              f"   (--cagr override active)")
    elif nom_cagr is not None and delta is not None and delta > 0.02:
        # No override set, baseline reads optimistic — suggest a value just
        # above CAPE (CAPE + 2pp, the tolerance-band edge between "optimistic"
        # and "in range"). Capped at config so we never suggest going higher.
        suggested = min(config_stock_cagr, nom_cagr + 0.02)
        print(f"     → To plan with CAPE-aligned returns, re-run with "
              f"--cagr {suggested*100:.1f}")
    print()

    # Rolling-CAGR table — historical S&P (USD + EUR) and forward MC scenarios
    sp500_native = next((p for t, p, _, _ in native_data if t == "^GSPC"), None)
    sp500_eur: pd.Series | None = None
    if sp500_native is not None and fx_cache is not None:
        try:
            sp_eur = convert_to_eur(sp500_native, "USD", fx_cache.get("USD"))
            if sp_eur is not None and len(sp_eur) > 0:
                sp_eur = sp_eur.copy()
                sp_eur.index = pd.to_datetime(sp_eur.index).normalize()
                sp_eur = sp_eur[~sp_eur.index.duplicated(keep="last")]
                sp500_eur = sp_eur
        except Exception:
            sp500_eur = None

    rows: list[tuple[str, str, float | None, float | None]] = []
    if sp500_native is not None:
        era = _era_rolling_stats(sp500_native, "2000-01-01", None)
        if era["cagr_p5"] is not None:
            rows.append(("S&P 500  USD", f"{era['yr_from']}–{era['yr_to']}",
                         era["cagr_p5"], era["cagr_p50"]))
        end_year = 2000 + horizon_years
        era2 = _era_rolling_stats(sp500_native, "2000-01-01", f"{end_year + 1}-01-01")
        if era2["cagr_p5"] is not None:
            rows.append(("S&P 500  USD", f"{era2['yr_from']}–{era2['yr_to']}",
                         era2["cagr_p5"], era2["cagr_p50"]))
    if sp500_eur is not None:
        era3 = _era_rolling_stats(sp500_eur, None, None)
        if era3["cagr_p5"] is not None:
            rows.append(("S&P 500  EUR", f"{era3['yr_from']}–{era3['yr_to']}",
                         era3["cagr_p5"], era3["cagr_p50"]))

    # Label the MC scenario by the historical era it samples from (2000-onwards),
    # not by forward calendar years — makes the rows directly comparable to the
    # historical S&P 500 2000-{2000+horizon} row above.
    scenario_range = f"2000–{2000 + horizon_years}"
    lp50 = forward_summary.get("lump_p50") if forward_summary else None
    lp5  = forward_summary.get("lump_p5")  if forward_summary else None
    if lp50 is not None and lp5 is not None:
        rows.append(("Lump-sum 2-phase MC", scenario_range, lp5, lp50))
    rp50 = forward_summary.get("dca_p50_irr") if forward_summary else None
    rp5  = forward_summary.get("dca_p5_irr")  if forward_summary else None
    if rp50 is not None and rp5 is not None:
        rows.append(("DCA      2-phase MC", scenario_range, rp5, rp50))

    if rows:
        print("     Real-world CAGR distributions  (P5 = bad-luck entry, P50 = typical):")
        print()
        print(f"     {'Source':<22}  {'Period':<13}  {'P5':>6}  {'P50':>6}")
        print(f"     {'─' * 22}  {'─' * 13}  {'─' * 6}  {'─' * 6}")
        for src, period, p5v, p50v in rows:
            print(f"     {src:<22}  {period:<13}"
                  f"  {p5v*100:>+5.1f}%  {p50v*100:>+5.1f}%")
        print()
        print("     P50s matching across historical and MC rows validates the model:")
        print("     the MC's median outcome reproduces the lost-decade-plus-recovery")
        print("     historical CAGR. The MC's tail (P5) tells you what the distribution")
        print("     of possible 2000-2015-like sequences looks like, not just the one")
        print("     we actually got. Historical P5 is narrow because all 2000 entries")
        print("     land near the same point by 2015; MC P5 is wide because Gaussian")
        print("     resampling produces many alternative sequences with the same mean+σ.")
        print()

    # ── 2. Optimal portfolio summary table ───────────────────────────────────
    if opt_result:
        sp500_prices = next((p for t, p, _, _ in native_data if t == "^GSPC"), None)
        print("  2. Optimal portfolio")
        print_portfolio_final_summary(
            opt_result,
            strategy_key=sweep_strategy,
            sweep=sweep,
            sp500_prices=sp500_prices,
        )


# ── Realism snapshot ──────────────────────────────────────────────────────────

def _build_realism_snapshot(
    forward_summary: dict,
    benchmark_prices: dict,
    fx_cache: dict,
    opt_result: "OptimalPortfolioResult | None",
    sweep: "list[tuple[int, OptimalPortfolioResult]] | None",
    sweep_strategy: str,
) -> "CagrRealismSnapshot | None":
    """Build the one-row realism check used by main.py's Plan Snapshot.
    Returns None on import failure so callers degrade gracefully."""
    try:
        from models import CagrRealismSnapshot
    except ImportError:
        return None

    cape_nom = forward_summary.get("cape_nominal_cagr") if forward_summary else None

    # S&P 500 10y rolling EUR P50/P5 — convert to EUR first, then roll.
    sp_p50 = sp_p5 = None
    sp500_native = benchmark_prices.get("^GSPC", (None, None))[0]
    if sp500_native is not None:
        sp_eur = convert_to_eur(sp500_native, "USD", fx_cache.get("USD"))
        if sp_eur is not None and len(sp_eur) > 0:
            sp_eur = sp_eur.copy()
            sp_eur.index = pd.to_datetime(sp_eur.index).normalize()
            sp_eur = sp_eur[~sp_eur.index.duplicated(keep="last")]
            cagrs = _rolling_cagrs(sp_eur, 10)
            if cagrs is not None and len(cagrs) > 0:
                sp_p50 = float(np.percentile(cagrs, 50))
                sp_p5  = float(np.percentile(cagrs, 5))

    # Optimised ETF P50/P5 — pick the SAME row the table's ★ marks: highest
    # rank by tuple score (P5_available, P5/P50/CAGR). Previously this picked
    # the largest-k sweep row, which could disagree with the recommendation.
    def _score(s: "PortfolioStrategy") -> tuple[int, float]:
        if s.portfolio_cagr_p5 is not None:
            return (1, s.portfolio_cagr_p5)
        if s.portfolio_cagr_p50 is not None:
            return (0, s.portfolio_cagr_p50)
        return (0, s.portfolio_cagr)

    candidates: list[PortfolioStrategy] = []
    if opt_result is not None:
        s = opt_result.strategies.get(sweep_strategy)
        if s is not None:
            candidates.append(s)
    if sweep:
        for _k, sub in sweep:
            s = sub.strategies.get(sweep_strategy)
            if s is not None:
                candidates.append(s)
    etf_p50 = etf_p5 = None
    if candidates:
        pick = max(candidates, key=_score)
        etf_p50 = pick.portfolio_cagr_p50
        etf_p5  = pick.portfolio_cagr_p5

    return CagrRealismSnapshot(
        cape_nominal_cagr=cape_nom,
        sp500_10y_rolling_eur_p50=sp_p50,
        sp500_10y_rolling_eur_p5=sp_p5,
        optimised_etf_p50=etf_p50,
        optimised_etf_p5=etf_p5,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="ETF CAGR analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "analyses:\n"
            "  cagr       ETF rolling CAGR summary + key metrics\n"
            "  optimizer  portfolio optimiser + cardinality sweep + final summary\n"
            "  benchmark  benchmark index rolling CAGR + era analysis\n"
            "  forward    CAPE-implied CAGR + two-phase MC stress test + DCA\n"
            "  full       all of the above  (default)\n"
        ),
    )
    parser.add_argument(
        "analysis", nargs="?", default="full",
        choices=["cagr", "optimizer", "benchmark", "forward", "full"],
        help="which analysis to run (default: full)",
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="print detailed tables (final summary, benchmarks, era analysis, forward, correlation)")
    parser.add_argument("--no-advanced-optimization", action="store_false", dest="advanced_opt",
                        help="disable Bayesian shrinkage (OAS covariance + James-Stein means) in optimizer")
    parser.add_argument("--no-clear", action="store_true",
                        help="suppress terminal clear (used when called from main.py)")
    parser.add_argument("--cagr", default=None,
                        help="override stock CAGR for the realism check display "
                             "(accepts 4, 4.0, 0.04, '4%%'); per-run only")
    parser.set_defaults(advanced_opt=True)
    args = parser.parse_args()
    verbose  = args.verbose
    analysis = args.analysis
    advanced_opt = args.advanced_opt
    from models import Config as _CfgParser
    cli_planning_cagr_override = _CfgParser.parse_planning_cagr(args.cagr)

    run_cagr      = analysis in ("cagr", "optimizer", "full")
    run_optimizer = analysis in ("optimizer", "full")
    run_benchmark = analysis in ("benchmark", "forward", "full")
    run_forward   = analysis in ("forward", "full")

    if not args.no_clear:
        os.system("cls" if os.name == "nt" else "clear")

    print("SINGLE ETF CAGR ANALYSIS  —  ROLLING DAILY WINDOWS  (EUR-adjusted)")
    if verbose:
        print(f"  Instruments: {len(TICKERS)} total  ({len(EU_TICKERS)} EU-eligible ✓, {len(US_TICKERS)} US-listed)")
        print(f"  Windows    : {', '.join(_window_label(y) for y in WINDOW_YEARS)}")
        print(f"  Sort by    : {_window_label(SORT_BY_WINDOW)} mean rolling CAGR  (unified ranking)")
        print(f"  FX         : USD→EUR via EURUSD=X  |  GBP→EUR via GBPEUR=X")
        print(f"  EU flag ✓  : UCITS ETF directly purchasable under MiFID II")
    print()

    # ── Load config ───────────────────────────────────────────────────────────
    shrink_mean_intensity: float = 0.5  # default; overwritten from config if loaded
    try:
        import tomllib
        with open("config.toml", "rb") as _f:
            _raw_cfg = tomllib.load(_f)
        cfg_raw: dict                     = _raw_cfg
        benchmark_tickers: list[str]      = _raw_cfg.get("benchmark_index", [])
        benchmark_start_year: int | None  = _raw_cfg.get("benchmark_start_year", None)
        benchmark_era_horizon_year: int | None = _raw_cfg.get("benchmark_era_horizon_year", None)
        cape_ratio: float | None          = _raw_cfg.get("cape_ratio", None)
        cape_long_run_mean: float         = float(_raw_cfg.get("cape_long_run_mean", 17.0))
        cape_eps_growth: float            = float(_raw_cfg.get("cape_eps_growth", 0.015))
        cape_n_simulations: int           = int(_raw_cfg.get("cape_n_simulations", 50_000))
        horizon_years: int                = int(_raw_cfg.get("years", 15))
        inflation: float                  = float(_raw_cfg.get("expected_inflation_percent", 2.3)) / 100
        config_stock_cagr: float          = float(_raw_cfg.get("expected_yearly_growth_stocks_percent", 6.0)) / 100
        planning_cagr_override: float | None = cli_planning_cagr_override
        _initial_stock: float             = float(_raw_cfg.get("current_stock_value", 0.0))
        _annual_saving: float             = float(_raw_cfg.get("annual_net_saving", 0.0))
        _stock_pct: float                 = float(_raw_cfg.get("maximum_stock_monthly_investment_pct", 0.0))
        dca_monthly: float                = (_annual_saving / 12.0) * _stock_pct
        opt_cfg: dict                     = _raw_cfg.get("etf_optimizer", {})
        optimize_tickers: list[str]       = opt_cfg.get("optimize_tickers", [])
        shrink_mean_intensity: float      = float(opt_cfg.get("bayesian_shrink_mean", 0.5))
    except Exception:
        benchmark_tickers      = []
        benchmark_start_year   = None
        benchmark_era_horizon_year = None
        cape_ratio             = None
        cape_long_run_mean     = 17.0
        cape_eps_growth        = 0.015
        cape_n_simulations     = 50_000
        horizon_years          = 15
        inflation              = 0.023
        config_stock_cagr      = 0.06
        planning_cagr_override = None
        _initial_stock         = 0.0
        dca_monthly            = 0.0
        opt_cfg                = {}
        optimize_tickers       = []
        cfg_raw                = {}

    # ── Fetch FX rates (always needed for EUR conversion) ─────────────────────
    if verbose:
        print("  Fetching FX rate series...")
    fx_cache: dict[str, pd.Series | None] = {}
    for ccy, fx_ticker in FX_TICKERS.items():
        s, fx_status = fetch_fx_series(ccy)
        fx_cache[ccy] = s
        if verbose:
            if s is not None:
                range_str = f"{s.index[0].date()} → {s.index[-1].date()}  ({len(s)} days)"
                print(f"  {fx_ticker}  [{fx_status}]  →  {range_str}")
            else:
                print(f"  {fx_ticker}  [failed]")

    # ── Fetch ETF price series (cagr / optimizer) ─────────────────────────────
    results: list[TickerResult] = []
    prices_eur: dict[str, pd.Series] = {}

    if run_cagr or run_optimizer:
        print(f"  Fetching {len(TICKERS)} tickers…", end="" if not verbose else "\n", flush=True)
        failed: list[str] = []
        for ticker in TICKERS:
            eu = ticker in EU_TICKERS
            if verbose:
                print(f"  → {ticker}", end="", flush=True)
            result, cache_status, eur_prices = fetch_and_compute(ticker, fx_cache, eu)
            if result is not None:
                if verbose:
                    orig = result.currency_original
                    fx_note = "" if orig == "EUR" else f" [{orig}→EUR]"
                    print(f"  ✓  [{cache_status}]  {result.data_from} → {result.data_to}{fx_note}")
                results.append(result)
                prices_eur[ticker] = eur_prices
            else:
                failed.append(ticker)
                if verbose:
                    print(f"  ✗  [{cache_status}]")
        ok_count = len(results)
        fail_str = f"  {len(failed)} failed: {', '.join(failed)}" if failed else ""
        if not verbose:
            print(f"  {ok_count} ok{fail_str}")

        if not results:
            print("\n  No data retrieved.")
            return

    # ── Fetch benchmark index series (benchmark / forward) ────────────────────
    benchmark_prices: dict[str, tuple] = {}

    if run_benchmark or run_forward:
        if benchmark_tickers:
            print(f"  Fetching {len(benchmark_tickers)} benchmark indices…",
                  end="" if not verbose else "\n", flush=True)
            bfailed: list[str] = []
            for bticker in benchmark_tickers:
                prices_b, bstatus, bccy = fetch_benchmark_series(bticker)
                if prices_b is not None:
                    benchmark_prices[bticker] = (prices_b, bccy)
                    if verbose:
                        print(f"  {bticker}  [{bstatus}]  "
                              f"{prices_b.index[0].year}–{prices_b.index[-1].year}  "
                              f"({len(prices_b):,} days)  [{bccy}]")
                else:
                    bfailed.append(bticker)
            if not verbose:
                ok_b = len(benchmark_prices)
                fail_b = f"  {len(bfailed)} failed: {', '.join(bfailed)}" if bfailed else ""
                print(f"  {ok_b} ok{fail_b}")

    # ── CAGR analysis ─────────────────────────────────────────────────────────
    if run_cagr and results:
        ref_label = _window_label(SORT_BY_WINDOW)

        if verbose:
            print()
            print(f"FINAL SUMMARY — KEY METRICS AT A GLANCE:")
            print(f"  Ranked by {ref_label} mean CAGR  |  Cons. = mean÷std (timing consistency)\n")
            print_final_summary(results)

            print()
            print(f"{'CORRELATION MATRIX — PAIRWISE PEARSON  (EUR daily log returns)':^70}")
            print("  Values near 1.0 = redundant.")
            print()
            print_correlation_matrix(results, prices_eur)

    # ── Portfolio optimiser ───────────────────────────────────────────────────
    opt_result: OptimalPortfolioResult | None = None
    sweep: list[tuple[int, OptimalPortfolioResult]] | None = None
    sweep_strategy: str = opt_cfg.get("selected_strategy", "max_sortino")

    if run_optimizer and results and optimize_tickers:
        if verbose:
            print()
            print(f"{'PORTFOLIO OPTIMISER':^70}")
            print()
        print("  Running optimisation (15 restarts per strategy)…", flush=True)
        opt_result = compute_optimal_portfolio(
            results=results,
            prices_eur=prices_eur,
            optimize_tickers=optimize_tickers,
            max_weight_per_ticker=float(opt_cfg.get("max_weight_per_ticker", 0.50)),
            qqq_max_weight=float(opt_cfg.get("qqq_max_weight", 0.25)),
            gold_max_weight=float(opt_cfg.get("gold_max_weight", 0.10)),
            min_weight_threshold=float(opt_cfg.get("min_weight_if_included", 0.05)),
            min_history_years=float(opt_cfg.get("min_history_years", 5.0)),
            selected_strategy=opt_cfg.get("selected_strategy", None),
            use_bayesian_shrinkage=advanced_opt,
            shrink_mean_intensity=shrink_mean_intensity,
        )
        if opt_result and verbose:
            print_optimal_portfolio(opt_result)

        if opt_result and opt_cfg.get("cardinality_sweep", False):
            max_k = opt_cfg.get("cardinality_sweep_max_k", None)
            n_eligible = len(opt_result.eligible_tickers)
            k_limit = min(n_eligible, int(max_k)) if max_k is not None else n_eligible
            from math import comb
            n_subsets = sum(comb(n_eligible, k) for k in range(2, k_limit + 1))
            print()
            print(f"  Running cardinality sweep (k=2..{k_limit}, {n_subsets} subsets)…",
                  flush=True)
            sweep = compute_cardinality_sweep(
                results=results,
                prices_eur=prices_eur,
                eligible=opt_result.eligible_tickers,
                max_weight_per_ticker=float(opt_cfg.get("max_weight_per_ticker", 0.50)),
                qqq_max_weight=float(opt_cfg.get("qqq_max_weight", 0.25)),
                gold_max_weight=float(opt_cfg.get("gold_max_weight", 0.10)),
                min_history_years=float(opt_cfg.get("min_history_years", 5.0)),
                strategy_key=sweep_strategy,
                max_k=k_limit,
                use_bayesian_shrinkage=advanced_opt,
                shrink_mean_intensity=shrink_mean_intensity,
            )

    # ── Benchmark rolling CAGR + era analysis ────────────────────────────────
    native_data: list = []

    if run_benchmark and benchmark_prices:
        def _clip(prices: pd.Series) -> pd.Series:
            if benchmark_start_year is None:
                return prices
            return prices[prices.index >= pd.Timestamp(f"{benchmark_start_year}-01-01")]

        for bticker, (bprices, bccy) in benchmark_prices.items():
            clipped = _clip(bprices)
            if len(clipped) < 2:
                continue
            rolling = compute_benchmark_rolling(clipped)
            native_data.append((bticker, clipped, bccy, rolling))

        eur_data = []
        for bticker, (bprices, bccy) in benchmark_prices.items():
            if bccy == "EUR":
                prices_eur_b = bprices
            else:
                fx = fx_cache.get(bccy)
                if fx is None:
                    continue
                prices_eur_b = convert_to_eur(bprices, bccy, fx)
                prices_eur_b.index = pd.to_datetime(prices_eur_b.index).normalize()  # type: ignore[union-attr]
                prices_eur_b = prices_eur_b[~prices_eur_b.index.duplicated(keep="last")]  # type: ignore[union-attr]
            clipped_eur = _clip(prices_eur_b)
            if len(clipped_eur) < 2:
                continue
            rolling_eur = compute_benchmark_rolling(clipped_eur)
            eur_data.append((bticker, clipped_eur, "EUR", rolling_eur))

        if verbose:
            print()
            print_benchmark_table(
                native_data,
                title="BENCHMARK INDICES — ROLLING CAGR (native currency, full history)",
                note="Native currency — no EUR conversion. Avoids EURUSD=X 2003 truncation. Use for long-window plausibilization.",
            )
            print_benchmark_table(
                eur_data,
                title="BENCHMARK INDICES — ROLLING CAGR (EUR-converted, 2003+)",
                note="EUR-converted via EURUSD=X. Shows actual EUR investor experience including FX impact.",
            )
            print_benchmark_era_table(
                native_data,
                era_horizon_year=benchmark_era_horizon_year,
                horizon_years=horizon_years,
            )

    # ── Forward CAGR estimates (always compute; print gated by verbose) ─────
    # SORR / decumulation survival lives in main_decumulation.py — this script
    # only models the accumulation-CAGR side.
    forward_summary: dict = {}
    if run_forward and benchmark_prices:
        if not native_data:
            def _clip(prices: pd.Series) -> pd.Series:
                if benchmark_start_year is None:
                    return prices
                return prices[prices.index >= pd.Timestamp(f"{benchmark_start_year}-01-01")]
            for bticker, (bprices, bccy) in benchmark_prices.items():
                clipped = _clip(bprices)
                if len(clipped) < 2:
                    continue
                rolling = compute_benchmark_rolling(clipped)
                native_data.append((bticker, clipped, bccy, rolling))

        if verbose:
            print()
        forward_summary = print_forward_cagr_estimates(
            native_data,
            horizon_years=horizon_years,
            inflation=inflation,
            cape_ratio=cape_ratio,
            cape_long_run_mean=cape_long_run_mean,
            cape_eps_growth=cape_eps_growth,
            n_simulations=cape_n_simulations,
            initial_stock=_initial_stock,
            monthly_contribution=dca_monthly,
            verbose=verbose,
        )

    # ── TAKE AWAY (always printed) ────────────────────────────────────────────
    print()
    print_take_away(
        forward_summary=forward_summary,
        native_data=native_data,
        opt_result=opt_result,
        sweep=sweep,
        sweep_strategy=sweep_strategy,
        horizon_years=horizon_years,
        config_stock_cagr=config_stock_cagr,
        fx_cache=fx_cache,
        planning_cagr_override=planning_cagr_override,
    )

    # ── Realism snapshot (for Plan Snapshot in main.py) ───────────────────────
    realism = _build_realism_snapshot(
        forward_summary=forward_summary,
        benchmark_prices=benchmark_prices,
        fx_cache=fx_cache,
        opt_result=opt_result,
        sweep=sweep,
        sweep_strategy=sweep_strategy,
    )
    return results, opt_result, realism


if __name__ == "__main__":
    main()
