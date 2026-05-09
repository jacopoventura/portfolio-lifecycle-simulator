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
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ── Configuration ─────────────────────────────────────────────────────────────

# EU-available UCITS ETFs (all EUR-denominated unless noted)
EU_TICKERS: list[str] = [
    "VWCE.DE",   # Vanguard FTSE All-World Acc (Xetra)
    "IWDA.AS",   # iShares Core MSCI World Acc (Amsterdam)
    "EUNL.DE",   # iShares Core MSCI World Dis (Xetra)
    "VUSA.AS",   # Vanguard S&P 500 Dis (Amsterdam)
    "CSPX.L",    # iShares Core S&P 500 Acc (London, USD)
    "XDWD.DE",   # Xtrackers MSCI World Swap Acc (Xetra)
    "VEVE.AS",   # Vanguard FTSE Developed World Acc (Amsterdam)
    "IS3R.DE",   # iShares Edge MSCI World Momentum Factor (Xetra)
    "XDEQ.DE",   # Xtrackers MSCI World Quality Factor (Xetra)
    "XDWT.DE",   # Xtrackers MSCI World Information Technology (Xetra)
    "TDIV.AS",   # VanEck Morningstar Developed Markets Dividend Leaders (Amsterdam)
    "VFEM.AS",   # Vanguard FTSE Emerging Markets (Amsterdam)
    "IEMA.AS",   # iShares MSCI EM IMI Acc (Amsterdam)
    "EXS1.DE",   # iShares Core DAX Acc (Xetra)
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
    "VWCE.DE": "EUR", "IWDA.AS": "EUR", "EUNL.DE": "EUR",
    "VUSA.AS": "EUR", "XDWD.DE": "EUR", "VEVE.AS": "EUR",
    "IS3R.DE": "EUR", "XDEQ.DE": "EUR", "XDWT.DE": "EUR",
    "TDIV.AS": "EUR", "VFEM.AS": "EUR", "IEMA.AS": "EUR",
    "EXS1.DE": "EUR",
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
        if known and known != currency:
            print(f"  [!] {ticker}: Yahoo returned currency '{currency}', expected '{known}' — using known value")
            currency = known
        else:
            print(f"  [!] {ticker}: unrecognised currency '{currency}' — treating as EUR (CAGR may be wrong)")
            currency = "EUR"

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


def print_summary_table(results: list[TickerResult]) -> None:
    labels = [_window_label(y) for y in WINDOW_YEARS]
    tick_w = 10
    name_w = 36
    dt_w = 8    # start / end date columns
    cw = 7      # CAGR columns
    av_w = 7    # Ann. σ
    ref_cw = 8  # min / max / std of reference window

    window_block = "".join(f"-{'-' * cw}-+" for _ in labels)
    sep = (
        f"+-{'-' * tick_w}-+-{'-' * name_w}-+--+"
        f"-{'-' * dt_w}-+-{'-' * dt_w}-+"
        f"{window_block}"
        f"-{'-' * av_w}-+"
        f"-{'-' * ref_cw}-+-{'-' * ref_cw}-+-{'-' * ref_cw}-+-{'-' * ref_cw}-+"
    )

    ref_label = _window_label(SORT_BY_WINDOW)
    header_windows = "".join(f" {label:>{cw}} |" for label in labels)
    hdr = (
        f"| {'Ticker':<{tick_w}} | {'Name':<{name_w}} |EU"
        f"| {'Start':>{dt_w}} | {'End':>{dt_w}} |"
        f"{header_windows}"
        f" {'σ-5y':>{av_w}} |"
        f" {'P5-' + ref_label:>{ref_cw}} | {'Std' + ref_label:>{ref_cw}} | {'Min' + ref_label:>{ref_cw}} | {'Max' + ref_label:>{ref_cw}} |"
    )

    # Fixed-width blank prefix for the n_windows sub-row
    blank_prefix = (
        f"| {'':>{tick_w}} | {'':>{name_w}} |  "
        f"| {'':>{dt_w}} | {'':>{dt_w}} |"
    )
    blank_suffix = f" {'':>{av_w}} | {'':>{ref_cw}} | {'':>{ref_cw}} | {'':>{ref_cw}} | {'':>{ref_cw}} |"

    print(sep)
    print(hdr)
    print(sep)

    def print_row(r: TickerResult) -> None:
        eu_flag = "✓ " if r.eu_available else "  "
        row_w = "".join(
            f" {_pct(r.windows[y].cagr_mean) if y in r.windows else 'n/a':>{cw}} |"
            for y in WINDOW_YEARS
        )
        ref_w = r.windows.get(SORT_BY_WINDOW)
        p05_s = _pct(ref_w.cagr_p05) if ref_w else "n/a"
        std_s = _pct(ref_w.cagr_std) if ref_w else "n/a"
        min_s = _pct(ref_w.cagr_min) if ref_w else "n/a"
        max_s = _pct(ref_w.cagr_max) if ref_w else "n/a"
        print(
            f"| {r.ticker:<{tick_w}} | {r.name[:name_w]:<{name_w}} |{eu_flag}"
            f"| {r.data_from.strftime('%Y-%m'):>{dt_w}} | {r.data_to.strftime('%Y-%m'):>{dt_w}} |"
            f"{row_w}"
            f" {_pct(r.ann_vol_5y_eur):>{av_w}} |"
            f" {p05_s:>{ref_cw}} | {std_s:>{ref_cw}} | {min_s:>{ref_cw}} | {max_s:>{ref_cw}} |"
        )
        # Sub-row: n_windows per window (omit for "max" window — always 1)
        row_n = "".join(
            f" {('n=' + str(r.windows[y].n_windows)) if (y in r.windows and r.windows[y].n_windows > 0 and y is not None) else '':>{cw}} |"
            for y in WINDOW_YEARS
        )
        print(f"{blank_prefix}{row_n}{blank_suffix}")

    for r in sorted(results, key=_sort_key):
        print_row(r)
    print(sep)


def print_detail_table(results: list[TickerResult], years: int | None) -> None:
    label = _window_label(years)
    valid = [
        (r, r.windows[years])
        for r in results
        if years in r.windows and r.windows[years].cagr_mean is not None
    ]
    valid.sort(key=lambda x: -(x[1].cagr_mean or 0))

    tick_w = 10
    name_w = 36
    cw = 9
    is_single = (years is None)

    if is_single:
        sep = f"+-{'-' * tick_w}-+-{'-' * name_w}-+--+-{'-' * 8}-+-{'-' * cw}-+-{'-' * cw}-+"
        hdr = (
            f"| {'Ticker':<{tick_w}} | {'Name':<{name_w}} |EU"
            f"| {'From':>8} | {'CAGR (EUR)':>{cw}} | {'Ann. σ':>{cw}} |"
        )
    else:
        sep = (
            f"+-{'-' * tick_w}-+-{'-' * name_w}-+--+"
            f"-{'-' * cw}-+-{'-' * cw}-+-{'-' * cw}-+-{'-' * cw}-+-{'-' * cw}-+-{'-' * cw}-+-{'-' * cw}-+"
            f"-{'-' * 8}-+"
        )
        hdr = (
            f"| {'Ticker':<{tick_w}} | {'Name':<{name_w}} |EU"
            f"| {'Mean CAGR':>{cw}} | {'P5 CAGR':>{cw}} | {'P25 CAGR':>{cw}} | {'Std CAGR':>{cw}} | {'Min CAGR':>{cw}} | {'Max CAGR':>{cw}} | {'Ann. σ':>{cw}} |"
            f" {'Windows':>8} |"
        )

    print(f"\n  Window: {label.upper()}")
    print(sep)
    print(hdr)
    print(sep)

    for r, w in valid:
        eu_flag = "✓ " if r.eu_available else "  "
        if is_single:
            print(
                f"| {r.ticker:<{tick_w}} | {r.name[:name_w]:<{name_w}} |{eu_flag}"
                f"| {r.data_from.strftime('%Y-%m'):>8} | {_pct(w.cagr_mean):>{cw}} | {_pct(r.ann_vol_eur):>{cw}} |"
            )
        else:
            print(
                f"| {r.ticker:<{tick_w}} | {r.name[:name_w]:<{name_w}} |{eu_flag}"
                f"| {_pct(w.cagr_mean):>{cw}} | {_pct(w.cagr_p05):>{cw}} | {_pct(w.cagr_p25):>{cw}} | {_pct(w.cagr_std):>{cw}} | {_pct(w.cagr_min):>{cw}} | {_pct(w.cagr_max):>{cw}} | {_pct(r.ann_vol_eur):>{cw}} |"
                f" {w.n_windows:>8,} |"
            )
    print(sep)
    if not valid:
        print("  No tickers with sufficient data for this window.")


# ── Final summary ─────────────────────────────────────────────────────────────

def print_final_summary(results: list[TickerResult]) -> None:
    """Compact ranked table: key metrics only, with a consistency ratio."""
    ref = SORT_BY_WINDOW
    ref_label = _window_label(ref)

    rk_w    = 2
    tick_w  = 10
    name_w  = 28
    yr_w    = 4
    cw      = 7
    nw      = 6   # n_windows column width
    cons_w  = 6
    av_w    = 6   # volatility columns (3 of them)
    mdd_w   = 7   # max drawdown
    ratio_w = 6   # Sharpe / Sortino / Calmar
    wr_w    = 6   # win rate for reference window

    sep = (
        f"+-{'-'*rk_w}-+-{'-'*tick_w}-+-{'-'*name_w}-+--+-{'-'*yr_w}-+"
        f"-{'-'*cw}-+-{'-'*nw}-+-{'-'*cw}-+-{'-'*cons_w}-+"
        f"-{'-'*cw}-+-{'-'*cw}-+"
        f"-{'-'*av_w}-+-{'-'*av_w}-+-{'-'*av_w}-+"
        f"-{'-'*mdd_w}-+"
        f"-{'-'*ratio_w}-+-{'-'*ratio_w}-+-{'-'*ratio_w}-+"
        f"-{'-'*wr_w}-+"
    )
    hdr = (
        f"| {'#':>{rk_w}} | {'Ticker':<{tick_w}} | {'Name':<{name_w}} |EU"
        f"| {'From':>{yr_w}} |"
        f" {ref_label+' avg':>{cw}} | {'n':>{nw}} | {'P5-'+ref_label:>{cw}} | {'Cons.':>{cons_w}} |"
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

    def _nw_str(w: WindowStats | None) -> str:
        if w is None or w.n_windows == 0:
            return "n/a"
        return f"{w.n_windows:,}"

    all_results = sorted(results, key=_sort_key)

    def _row(rank: int, r: TickerResult) -> None:
        eu_flag = "✓ " if r.eu_available else "  "
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
            f"| {rank:>{rk_w}} | {r.ticker:<{tick_w}} | {r.name[:name_w]:<{name_w}} |{eu_flag}"
            f"| {r.data_from.year:>{yr_w}} |"
            f" {_pct(w_ref.cagr_mean if w_ref else None):>{cw}} |"
            f" {_nw_str(w_ref):>{nw}} |"
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
    print(
        f"  Cons.   = {ref_label} mean CAGR ÷ {ref_label} std  "
        f"(higher = more consistent outcome regardless of entry date)\n"
        f"  P5-{ref_label} = 5th-percentile {ref_label} rolling CAGR — conservative planning input\n"
        f"  MDD     = max peak-to-trough drawdown (e.g. -55% = lost 55% from peak)\n"
        f"  σ-5y/10y/all = annualised daily vol — crypto annualised at 365 days, equities at ~252\n"
        f"  Sharpe  = (all-time CAGR − {RISK_FREE_RATE*100:.1f}% RFR) / σ-all\n"
        f"  Sortino = (all-time CAGR − {RISK_FREE_RATE*100:.1f}% RFR) / downside-σ  "
        f"(only days below daily RFR threshold)\n"
        f"  Calmar  = all-time CAGR / |MDD| — return per unit of worst drawdown (SORR proxy)\n"
        f"  WR-{ref_label} = % of {ref_label} rolling windows with CAGR > 0 (historical win probability)"
    )


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

    tick_w = max(len(t) for t in tickers)
    cell_w = 6   # "  0.97" or " -0.05"

    print(f"  Common period : {start.date()} → {end.date()}"
          f"  ({n_years:.1f} years  |  ~{n_days:,} days/pair)")
    print()

    # Legend
    for i, t in enumerate(tickers, 1):
        eu_flag = "✓" if t in eu_set else " "
        print(f"    {i:>2}  {eu_flag}  {t}")
    print()

    # Column header row
    hdr = f"  {'':>{tick_w}}"
    for i in range(1, n + 1):
        hdr += f"  {i:>4}"
    print(hdr)

    # Lower-triangular rows (diagonal shown as "────")
    for i, ti in enumerate(tickers):
        row = f"  {ti:>{tick_w}}"
        for j in range(i):
            v = corr.loc[ti, tickers[j]]
            row += "   n/a" if pd.isna(v) else f"  {v:>4.2f}"
        row += "  ────"
        print(row)


# ── Agent summary ─────────────────────────────────────────────────────────────

def print_agent_summary(results: list[TickerResult]) -> None:
    """Structured one-line-per-ticker summary optimised for AI agent consumption."""
    from datetime import date as _date

    all_r = sorted(results, key=_sort_key)
    eu    = [r for r in all_r if r.eu_available]

    def _cons_val(r: TickerResult) -> float | None:
        w = r.windows.get(SORT_BY_WINDOW)
        if w and w.cagr_mean is not None and w.cagr_std is not None and w.cagr_std > 0:
            return w.cagr_mean / w.cagr_std
        return None

    def _ratio(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.2f}"

    def _ticker_line(rank: int, r: TickerResult) -> str:
        w10 = r.windows.get(SORT_BY_WINDOW)
        w5  = r.windows.get(5)
        wat = r.windows.get(None)
        cons = _cons_val(r)
        wr_s = (
            f"{w10.win_rate * 100:.0f}%"
            if w10 and w10.win_rate is not None else "n/a"
        )
        return (
            f"  {rank:>2}. {r.ticker:<10}"
            f"  10y={_pct(w10.cagr_mean if w10 else None)}"
            f"  P5-10y={_pct(w10.cagr_p05 if w10 else None)}"
            f"  Cons={_ratio(cons)}"
            f"  5y={_pct(w5.cagr_mean if w5 else None)}"
            f"  all={_pct(wat.cagr_mean if wat else None)}"
            f"  σ5y={_pct(r.ann_vol_5y_eur)}"
            f"  MDD={_pct(r.max_drawdown_eur)}"
            f"  Sharpe={_ratio(r.sharpe_eur)}"
            f"  Sortino={_ratio(r.sortino_eur)}"
            f"  Calmar={_ratio(r.calmar_eur)}"
            f"  WR-10y={wr_s}"
            f"  from={r.data_from.year}"
        )

    ref_label = _window_label(SORT_BY_WINDOW)
    print(f"  Generated  : {_date.today()}")
    print(f"  Parameters : RFR={RISK_FREE_RATE*100:.1f}%  |  sort={ref_label} mean CAGR  |  prices in EUR")
    print()
    print("  QUESTION TO ANSWER:")
    print("  Given a target portfolio value or target monthly income at the retirement")
    print("  horizon — which instruments and combinations maximise the probability of")
    print("  reaching that goal? Recommend: instrument list + allocation rationale.")
    print()
    print("  METRIC GUIDE:")
    print(f"  P5-10y  : conservative return input for Monte Carlo (5th-pct of rolling {ref_label} CAGRs)")
    print("  σ5y     : forward-looking vol input for Monte Carlo (recent 5-year regime; crypto at 365d/yr)")
    print("  Cons    : mean÷std of rolling CAGRs — how stable is the return regardless of entry date")
    print("  MDD     : peak-to-trough drawdown — proxy for sequence-of-returns risk at decumulation")
    print("  Sortino : risk-adjusted return penalising only downside vol — preferred over Sharpe for SORR")
    print("  Calmar  : all-time CAGR / |MDD| — higher = better return per unit of worst drawdown")
    print(f"  WR-10y  : % of rolling {ref_label} windows with CAGR > 0 — historical win probability")
    print("  Corr    : see correlation matrix above — corr > 0.95 = redundant pair, pick one")
    print()

    print("  ALL INSTRUMENTS — ranked by 10y mean CAGR (✓ = EU-eligible UCITS):")
    for i, r in enumerate(all_r, 1):
        print(_ticker_line(i, r))

    # Key observations — combination-relevant highlights for the AI
    print()
    print("  KEY OBSERVATIONS FOR COMBINATION SELECTION:")

    def _best(lst: list[TickerResult], key, label: str, fmt) -> None:
        valid = [(r, key(r)) for r in lst if key(r) is not None]
        if valid:
            r, v = max(valid, key=lambda x: x[1])
            print(f"  - {label:<28}: {r.ticker}  ({fmt(v)})")

    def _worst(lst: list[TickerResult], key, label: str, fmt) -> None:
        valid = [(r, key(r)) for r in lst if key(r) is not None]
        if valid:
            r, v = min(valid, key=lambda x: x[1])
            print(f"  - {label:<28}: {r.ticker}  ({fmt(v)})")

    _best(all_r, lambda r: r.sharpe_eur,   "Best Sharpe",       lambda v: f"{v:.2f}")
    _best(all_r, lambda r: r.sortino_eur,  "Best Sortino",      lambda v: f"{v:.2f}")
    _best(all_r, lambda r: r.calmar_eur,   "Best Calmar",       lambda v: f"{v:.2f}")
    _best(all_r, _cons_val,                "Most consistent",   lambda v: f"Cons={v:.1f}")
    _best(all_r, lambda r: (r.windows[SORT_BY_WINDOW].cagr_p05
                            if SORT_BY_WINDOW in r.windows else None),
              f"Best P5-{ref_label}",   _pct)
    _worst(all_r, lambda r: r.ann_vol_5y_eur, "Lowest σ-5y",    _pct)
    _worst(all_r, lambda r: r.max_drawdown_eur, "Shallowest MDD", _pct)


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

    cutoff = dates[-1] - pd.Timedelta(days=5 * 365)
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

    for _ in range(n_restarts):
        w0 = rng.dirichlet(np.ones(n))
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
    cutoff_5y   = dates[-1] - pd.Timedelta(days=5 * 365)
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

    def _rolling_cagrs(pr: np.ndarray) -> np.ndarray | None:
        """All rolling 10y CAGR values over every possible start date.
        Returns None when history is shorter than the 10y window."""
        cum_ext = np.concatenate([[1.0], np.cumprod(1.0 + pr)])
        if len(cum_ext) <= w10:
            return None
        ratios = cum_ext[w10:] / cum_ext[:-w10]
        return ratios ** (tdpy / w10) - 1.0

    def _mean_rolling_cagr(pr: np.ndarray) -> float:
        """Mean of all rolling 10y CAGRs; falls back to endpoint CAGR if history < 10y."""
        cagrs = _rolling_cagrs(pr)
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
        cagrs = _rolling_cagrs(pr)
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
            if free.sum() == 0:
                break
            w[free] += excess * w[free] / w[free].sum()
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

    w_sortino = _run_optimization(ret_array, n, _neg_sortino, bounds_box, constraints)
    w_sharpe  = _run_optimization(ret_array, n, _neg_sharpe,  bounds_box, constraints)
    w_minvar  = _run_optimization(ret_array, n, _variance,    bounds_box, constraints)
    w_cagr    = _run_optimization(ret_array, n, _neg_cagr,    bounds_box, constraints)
    w_p50     = _run_optimization(ret_array, n, _neg_p50_cagr, bounds_box, constraints)
    w_p1      = _run_optimization(ret_array, n, _neg_p1_cagr,  bounds_box, constraints)
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

    # Recommended = use selected_strategy if valid; else highest P50-10y CAGR
    if selected_strategy and selected_strategy in strategies:
        recommended = selected_strategy
    else:
        recommended = max(
            strategies,
            key=lambda k: strategies[k].portfolio_cagr_p50
                          if strategies[k].portfolio_cagr_p50 is not None
                          else strategies[k].portfolio_cagr,
        )

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

    print("=" * W)
    print(f"{'PORTFOLIO OPTIMISER':^{W}}")
    print("=" * W)
    print(f"  Common history : {result.common_from} → {result.common_to}"
          f"  ({result.n_years:.1f} yr,  {result.n_trading_days:,} days)")
    if result.excluded:
        for t, reason in result.excluded:
            print(f"  Excluded       : {t}  ({reason})")
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
        label = (ticker + eu_flag)[:TC]
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
) -> list[tuple[int, OptimalPortfolioResult]]:
    """For each k from 2 to min(len(eligible), max_k), enumerate all C(n,k) subsets,
    run the optimizer, and return the best portfolio per k for the given strategy."""
    import contextlib
    import io
    from itertools import combinations

    k_max = min(len(eligible), max_k) if max_k is not None else len(eligible)
    sweep: list[tuple[int, OptimalPortfolioResult]] = []

    for k in range(2, k_max + 1):
        best_score = -np.inf
        best_result: OptimalPortfolioResult | None = None

        for subset in combinations(eligible, k):
            with contextlib.redirect_stdout(io.StringIO()):
                opt = compute_optimal_portfolio(
                    results=results,
                    prices_eur=prices_eur,
                    optimize_tickers=list(subset),
                    max_weight_per_ticker=max_weight_per_ticker,
                    qqq_max_weight=qqq_max_weight,
                    gold_max_weight=gold_max_weight,
                    min_weight_threshold=0.0,  # don't zero out — enforce exactly k tickers
                    min_history_years=min_history_years,
                )
            if opt is None or strategy_key not in opt.strategies:
                continue
            strat = opt.strategies[strategy_key]
            score = (strat.portfolio_cagr_p50 if strat.portfolio_cagr_p50 is not None
                     else strat.portfolio_cagr)
            if score > best_score:
                best_score = score
                best_result = opt

        if best_result is not None:
            sweep.append((k, best_result))

    return sweep


def print_cardinality_sweep(
    sweep: list[tuple[int, OptimalPortfolioResult]],
    strategy_key: str,
) -> None:
    W = 92
    STRAT_LABELS = {
        "max_sortino": "Max Sortino", "max_sharpe": "Max Sharpe",
        "min_variance": "Min Variance", "max_cagr": "Max CAGR",
        "max_p50_cagr": "Max P50 CAGR", "max_p1_cagr": "Max P1 CAGR",
        "equal_weight": "Equal Weight",
    }
    label = STRAT_LABELS.get(strategy_key, strategy_key)

    print("=" * W)
    print(f"{'CARDINALITY SWEEP — ' + label:^{W}}")
    print("=" * W)
    print(f"  Best portfolio found by exhaustive subset search for each portfolio size.\n")

    # Column header
    print(f"  {'k':<3}  {'Tickers & weights':<48}  {'History':>7}  "
          f"{'P50-CAGR':>9}  {'σ-annual':>8}  {'Sortino':>8}  {'MDD':>7}  {'Avg corr':>8}")
    print("  " + "─" * 98)

    for k, result in sweep:
        strat = result.strategies[strategy_key]
        use_p50 = result.n_years >= 10
        cagr    = strat.portfolio_cagr_p50 if use_p50 else strat.portfolio_cagr
        cagr_s  = f"{cagr * 100:.1f}%" if cagr is not None else "n/a"
        ann_s   = (f"{strat.portfolio_annual_ret_std * 100:.1f}%"
                   if strat.portfolio_annual_ret_std is not None else "n/a")
        srt_s   = f"{strat.portfolio_sortino:.2f}" if strat.portfolio_sortino is not None else "n/a"
        mdd_s   = f"{strat.portfolio_mdd * 100:.1f}%" if strat.portfolio_mdd is not None else "n/a"
        corr_s  = f"{strat.portfolio_avg_corr:.2f}" if strat.portfolio_avg_corr is not None else "n/a"
        hist_s  = f"{result.n_years:.1f}yr"

        # Build compact "TICK w% + TICK w%" string (strip exchange suffix for brevity)
        def _abbr(t: str) -> str:
            for sfx in (".AS", ".DE", ".L", "-USD"):
                t = t.replace(sfx, "")
            return t

        parts = []
        for t in result.eligible_tickers:
            w = strat.weights.get(t, 0.0)
            if w >= 0.005:
                parts.append(f"{_abbr(t)} {w * 100:.0f}%")
        tickers_s = " + ".join(parts)
        if len(tickers_s) > 48:
            tickers_s = tickers_s[:45] + "..."

        print(f"  {k:<3}  {tickers_s:<48}  {hist_s:>7}  "
              f"{cagr_s:>9}  {ann_s:>8}  {srt_s:>8}  {mdd_s:>7}  {corr_s:>8}")

    print()


# ── Benchmark indices ─────────────────────────────────────────────────────────

BENCHMARK_WINDOWS: list[int] = [5, 10, 15, 20, 25]

# Human-readable names (Yahoo Finance .info can be slow for indices; hardcode fallback)
_BENCHMARK_NAMES: dict[str, str] = {
    "^GSPC":    "S&P 500",
    "^IXIC":    "NASDAQ Composite",
    "^STOXX50E": "Euro Stoxx 50",
    "^W5000":   "Wilshire 5000",
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

    print("=" * W)
    print(f"{title:^{W}}")
    print("=" * W)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ETF CAGR analysis")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="print rolling CAGR tables and notes")
    args, _ = parser.parse_known_args()
    verbose = args.verbose

    os.system("cls" if os.name == "nt" else "clear")

    print("=" * 70)
    print(f"{'ETF CAGR ANALYSIS  —  ROLLING DAILY WINDOWS  (EUR-adjusted)':^70}")
    print("=" * 70)
    if verbose:
        print(f"  Instruments: {len(TICKERS)} total  ({len(EU_TICKERS)} EU-eligible ✓, {len(US_TICKERS)} US-listed)")
        print(f"  Windows    : {', '.join(_window_label(y) for y in WINDOW_YEARS)}")
        print(f"  Sort by    : {_window_label(SORT_BY_WINDOW)} mean rolling CAGR  (unified ranking)")
        print(f"  FX         : USD→EUR via EURUSD=X  |  GBP→EUR via GBPEUR=X")
        print(f"  EU flag ✓  : UCITS ETF directly purchasable under MiFID II")
    print()

    # Fetch FX rates
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

    # Fetch benchmark index series (cached separately from ETFs)
    try:
        import tomllib
        with open("config.toml", "rb") as _f:
            _raw_cfg = tomllib.load(_f)
        benchmark_tickers: list[str] = _raw_cfg.get("benchmark_index", [])
        benchmark_start_year: int | None = _raw_cfg.get("benchmark_start_year", None)
    except Exception:
        benchmark_tickers = []
        benchmark_start_year = None

    benchmark_prices: dict[str, pd.Series] = {}
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

    # Fetch ETF price series
    print(f"  Fetching {len(TICKERS)} tickers…", end="" if not verbose else "\n", flush=True)
    results: list[TickerResult] = []
    prices_eur: dict[str, pd.Series] = {}
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

    ref_label = _window_label(SORT_BY_WINDOW)

    # Verbose-only: rolling CAGR tables
    if verbose:
        print()
        print("=" * 70)
        print(f"{'SUMMARY — MEAN ROLLING CAGR BY WINDOW  (all prices in EUR)':^70}")
        print("=" * 70)
        print(f"  Sorted by {ref_label} mean CAGR  |  Min/Max = worst/best {ref_label} period ever")
        print(f"  EU✓ = UCITS ETF available to EU retail investors\n")
        print_summary_table(results)

        print()
        print("=" * 70)
        print(f"{'DETAIL — RANKED BY MEAN CAGR PER WINDOW':^70}")
        print("=" * 70)
        print("  Min = worst N-year period | Max = best | Ann.σ = daily vol (EUR)")
        for years in WINDOW_YEARS:
            print_detail_table(results, years)

    # Always: final summary
    print()
    print("=" * 70)
    print(f"{'FINAL SUMMARY — KEY METRICS AT A GLANCE':^70}")
    print("=" * 70)
    print(f"  Ranked by {ref_label} mean CAGR  |  Cons. = mean÷std (timing consistency)\n")
    print_final_summary(results)

    # Verbose-only: notes
    if verbose:
        print()
        print("=" * 70)
        print(f"{'NOTES':^70}")
        print("=" * 70)
        print("  Std CAGR: std dev of rolling N-year CAGRs across all daily start dates.")
        print("            Measures entry-point timing risk: how much the N-year outcome")
        print("            varies depending on when you started. Low std = consistent.")
        print()
        print("  P5/P25  : 5th and 25th percentile of rolling CAGRs. P5 is the conservative")
        print("            planning input — 95% of all entry dates beat this return.")
        print("            More robust than Min (which can be a single outlier period).")
        print()
        print("  MDD     : max peak-to-trough drawdown in EUR. E.g. -55% means the ETF")
        print("            lost 55% from its high at worst. Key input for SORR assessment.")
        print()
        print("  σ-5y / σ-10y / σ-all: annualised daily log-return vol over last 5 years,")
        print("            last 10 years, and full available history. σ-5y feeds Monte Carlo.")
        print()
        print("  EUR pricing: UCITS ETFs (.DE/.AS) are NAV-priced in EUR — price already")
        print("  embeds EUR/USD. US ETFs converted via EURUSD=X / GBPEUR=X historical rates.")
        print()
        print("  MiFID II / PRIIPs: EU retail investors cannot buy US-domiciled ETFs")
        print("  (SPY, QQQ, VTI, VEA, VWO) directly. Shown as benchmarks only.")
        print()

    print()
    print("=" * 70)
    print(f"{'CORRELATION MATRIX — PAIRWISE PEARSON  (EUR daily log returns)':^70}")
    print("=" * 70)
    print("  Lower triangle. 1.00 on diagonal (omitted). Values near 1.0 = redundant.")
    print()
    print_correlation_matrix(results, prices_eur)

    # Always: agent summary (captured by main.py pipeline)
    print()
    print("=" * 70)
    print(f"{'AGENT SUMMARY — STRUCTURED FOR AI RECOMMENDATION':^70}")
    print("=" * 70)
    print()
    print_agent_summary(results)
    print()

    # ── Portfolio optimiser ───────────────────────────────────────────────────
    opt_result: OptimalPortfolioResult | None = None
    try:
        import tomllib
        with open("config.toml", "rb") as _f:
            _raw = tomllib.load(_f)
        opt_cfg = _raw.get("etf_optimizer", {})
        optimize_tickers: list[str] = opt_cfg.get("optimize_tickers", [])
    except Exception:
        optimize_tickers = []
        opt_cfg = {}

    if optimize_tickers:
        print()
        print("=" * 70)
        print(f"{'PORTFOLIO OPTIMISER':^70}")
        print("=" * 70)
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
        )
        if opt_result:
            print_optimal_portfolio(opt_result)

        # ── Cardinality sweep ─────────────────────────────────────────────────
        if opt_result and opt_cfg.get("cardinality_sweep", False):
            sweep_strategy = opt_cfg.get("selected_strategy", "max_sortino")
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
            )
            if sweep:
                print_cardinality_sweep(sweep, sweep_strategy)

    # ── Benchmark index rolling CAGR tables ──────────────────────────────────
    if benchmark_prices:
        def _clip(prices: pd.Series) -> pd.Series:
            if benchmark_start_year is None:
                return prices
            return prices[prices.index >= pd.Timestamp(f"{benchmark_start_year}-01-01")]

        # Table 1: native currency — full history (clipped to benchmark_start_year), no FX truncation
        native_data = []
        for bticker, (bprices, bccy) in benchmark_prices.items():
            clipped = _clip(bprices)
            if len(clipped) < 2:
                continue
            rolling = compute_benchmark_rolling(clipped)
            native_data.append((bticker, clipped, bccy, rolling))

        # Table 2: EUR-converted — shows actual EUR investor experience (2003+)
        eur_data = []
        for bticker, (bprices, bccy) in benchmark_prices.items():
            if bccy == "EUR":
                prices_eur_b = bprices
            else:
                fx = fx_cache.get(bccy)
                if fx is None:
                    continue
                prices_eur_b = convert_to_eur(bprices, bccy, fx)
                prices_eur_b.index = pd.to_datetime(prices_eur_b.index).normalize()
                prices_eur_b = prices_eur_b[~prices_eur_b.index.duplicated(keep="last")]
            clipped_eur = _clip(prices_eur_b)
            if len(clipped_eur) < 2:
                continue
            rolling_eur = compute_benchmark_rolling(clipped_eur)
            eur_data.append((bticker, clipped_eur, "EUR", rolling_eur))

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

    return results, opt_result


if __name__ == "__main__":
    main()
