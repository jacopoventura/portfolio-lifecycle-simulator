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
    "VEVE.AS",   # Vanguard FTSE Developed World Acc (Amsterdam)   ← new
    "IS3R.DE",   # iShares Edge MSCI World Momentum Factor (Xetra) ← new
    "XDEQ.DE",   # Xtrackers MSCI World Quality Factor (Xetra)     ← new  (QDEV)
    "XDWT.DE",   # Xtrackers MSCI World Information Technology (Xetra) ← new
    "TDIV.AS",   # VanEck Morningstar Developed Markets Dividend Leaders (Amsterdam) ← new
    "VFEM.AS",   # Vanguard FTSE Emerging Markets (Amsterdam)
    "IEMA.AS",   # iShares MSCI EM IMI Acc (Amsterdam)
    "EXS1.DE",   # iShares Core DAX Acc (Xetra)
]

# US-listed benchmarks — not available to EU retail investors under MiFID II/PRIIPs
# Kept for historical comparison (longer track records). Prices converted to EUR.
US_TICKERS: list[str] = [
    "SPY",       # SPDR S&P 500
    "QQQ",       # Invesco QQQ (NASDAQ-100)
    "VTI",       # Vanguard Total Stock Market
    "VEA",       # Vanguard Developed Markets ex-US
    "VWO",       # Vanguard Emerging Markets
]

TICKERS = EU_TICKERS + US_TICKERS

# Time windows in years shown as columns. None = single endpoint CAGR (inception→today).
WINDOW_YEARS: list[int | None] = [1, 3, 5, 10, 15, None]

# Window used for the main sort and for showing Min/Max in the summary
SORT_BY_WINDOW: int | None = 10

# Yahoo Finance tickers for FX rates (fetched once at startup)
FX_TICKERS = {
    "USD": "EURUSD=X",   # USD per 1 EUR  →  price_EUR = price_USD / rate
    "GBP": "GBPEUR=X",   # EUR per 1 GBP  →  price_EUR = price_GBP * rate
}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class WindowStats:
    n_windows: int
    cagr_mean: float | None
    cagr_std: float | None   # std of rolling CAGRs = entry-point timing risk
    cagr_min: float | None
    cagr_max: float | None


@dataclass
class TickerResult:
    ticker: str
    name: str
    currency_original: str       # as reported by Yahoo Finance
    data_from: date
    data_to: date
    ann_vol_eur: float | None    # annualised daily σ in EUR terms
    windows: dict[int | None, WindowStats]
    eu_available: bool           # True = UCITS, False = US-only benchmark


# ── FX helpers ────────────────────────────────────────────────────────────────

def fetch_fx_series(currency: str) -> pd.Series | None:
    if currency == "EUR":
        return None
    fx_ticker = FX_TICKERS.get(currency)
    if fx_ticker is None:
        return None
    try:
        hist = yf.Ticker(fx_ticker).history(period="max", auto_adjust=True)
        if hist.empty:
            return None
        s = hist["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s
    except Exception:
        return None


def convert_to_eur(prices: pd.Series, currency: str, fx: pd.Series | None) -> pd.Series:
    if currency == "EUR" or fx is None:
        return prices
    # Normalise ETF index to tz-naive date
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    # Align FX to ETF dates with forward-fill for missing market days
    aligned_fx = fx.reindex(prices.index, method="ffill").reindex(prices.index)
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
    cagrs = (pv[ei] / pv[si]) ** (1.0 / actual_years) - 1.0
    return cagrs


# ── Fetch & compute ───────────────────────────────────────────────────────────

def fetch_and_compute(
    ticker: str,
    fx_cache: dict[str, pd.Series | None],
    eu_available: bool,
) -> TickerResult | None:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="max", auto_adjust=True)
    except Exception as exc:
        print(f"  [!] {ticker}: download error — {exc}")
        return None

    if hist.empty or "Close" not in hist.columns:
        print(f"  [!] {ticker}: no data returned")
        return None

    try:
        info = t.info
        name = info.get("longName") or info.get("shortName") or ticker
        currency = (info.get("currency") or "EUR").upper()
    except Exception:
        name = ticker
        currency = "EUR"

    raw_prices = hist["Close"].dropna()
    if len(raw_prices) < 2:
        print(f"  [!] {ticker}: insufficient price history")
        return None

    # Convert to EUR
    fx = fx_cache.get(currency)
    prices = convert_to_eur(raw_prices, currency, fx)
    if len(prices) < 2:
        print(f"  [!] {ticker}: FX alignment left insufficient data")
        return None

    # Normalise index for consistent rolling computation
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()

    data_from = prices.index[0].date()
    data_to = prices.index[-1].date()

    # Annualised daily-return volatility in EUR terms
    log_ret = np.diff(np.log(prices.values.astype(float)))
    ann_vol = float(np.std(log_ret, ddof=1) * np.sqrt(252)) if len(log_ret) > 1 else None

    windows: dict[int | None, WindowStats] = {}
    available_years = (data_to - data_from).days / 365.25

    for years in WINDOW_YEARS:
        if years is None:
            if available_years >= 1.0:
                pv = prices.values.astype(float)
                cagr = (pv[-1] / pv[0]) ** (1.0 / available_years) - 1.0
                windows[None] = WindowStats(n_windows=1, cagr_mean=float(cagr),
                                            cagr_std=None,
                                            cagr_min=float(cagr), cagr_max=float(cagr))
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
        )

    return TickerResult(
        ticker=ticker, name=name,
        currency_original=currency,
        data_from=data_from, data_to=data_to,
        ann_vol_eur=ann_vol,
        windows=windows,
        eu_available=eu_available,
    )


# ── Formatting ────────────────────────────────────────────────────────────────

def _pct(v: float | None, d: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def _sort_key(r: TickerResult) -> float:
    w = r.windows.get(SORT_BY_WINDOW)
    if w and w.cagr_mean is not None:
        return -w.cagr_mean
    for y in reversed(WINDOW_YEARS):
        ww = r.windows.get(y)
        if ww and ww.cagr_mean is not None:
            return -ww.cagr_mean
    return 0.0


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
        f"-{'-' * ref_cw}-+-{'-' * ref_cw}-+-{'-' * ref_cw}-+"
    )

    ref_label = _window_label(SORT_BY_WINDOW)
    header_windows = "".join(f" {label:>{cw}} |" for label in labels)
    hdr = (
        f"| {'Ticker':<{tick_w}} | {'Name':<{name_w}} |EU"
        f"| {'Start':>{dt_w}} | {'End':>{dt_w}} |"
        f"{header_windows}"
        f" {'Ann.σ':>{av_w}} |"
        f" {'Std' + ref_label:>{ref_cw}} | {'Min' + ref_label:>{ref_cw}} | {'Max' + ref_label:>{ref_cw}} |"
    )

    # Fixed-width blank prefix for the n_windows sub-row
    blank_prefix = (
        f"| {'':>{tick_w}} | {'':>{name_w}} |  "
        f"| {'':>{dt_w}} | {'':>{dt_w}} |"
    )
    blank_suffix = f" {'':>{av_w}} | {'':>{ref_cw}} | {'':>{ref_cw}} | {'':>{ref_cw}} |"

    print(sep)
    print(hdr)
    print(sep)

    eu_results = [r for r in results if r.eu_available]
    us_results = [r for r in results if not r.eu_available]

    def print_row(r: TickerResult) -> None:
        eu_flag = "✓ " if r.eu_available else "  "
        row_w = "".join(
            f" {_pct(r.windows[y].cagr_mean) if y in r.windows else 'n/a':>{cw}} |"
            for y in WINDOW_YEARS
        )
        ref_w = r.windows.get(SORT_BY_WINDOW)
        std_s = _pct(ref_w.cagr_std) if ref_w else "n/a"
        min_s = _pct(ref_w.cagr_min) if ref_w else "n/a"
        max_s = _pct(ref_w.cagr_max) if ref_w else "n/a"
        print(
            f"| {r.ticker:<{tick_w}} | {r.name[:name_w]:<{name_w}} |{eu_flag}"
            f"| {r.data_from.strftime('%Y-%m'):>{dt_w}} | {r.data_to.strftime('%Y-%m'):>{dt_w}} |"
            f"{row_w}"
            f" {_pct(r.ann_vol_eur):>{av_w}} |"
            f" {std_s:>{ref_cw}} | {min_s:>{ref_cw}} | {max_s:>{ref_cw}} |"
        )
        # Sub-row: n_windows per window (omit for "max" window — always 1)
        row_n = "".join(
            f" {('n=' + str(r.windows[y].n_windows)) if (y in r.windows and r.windows[y].n_windows > 0 and y is not None) else '':>{cw}} |"
            for y in WINDOW_YEARS
        )
        print(f"{blank_prefix}{row_n}{blank_suffix}")

    for r in sorted(eu_results, key=_sort_key):
        print_row(r)
    if us_results:
        print(f"| {'── US benchmarks (not EU-eligible, prices converted to EUR) ──':<{len(sep)-4}} |")
        for r in sorted(us_results, key=_sort_key):
            print_row(r)
    print(sep)


def print_detail_table(results: list[TickerResult], years: int | None) -> None:
    label = _window_label(years)
    valid = [
        (r, r.windows[years])
        for r in results
        if years in r.windows and r.windows[years].cagr_mean is not None
    ]
    valid.sort(key=lambda x: (not x[0].eu_available, -(x[1].cagr_mean or 0)))

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
            f"-{'-' * cw}-+-{'-' * cw}-+-{'-' * cw}-+-{'-' * cw}-+-{'-' * cw}-+"
            f"-{'-' * 8}-+"
        )
        hdr = (
            f"| {'Ticker':<{tick_w}} | {'Name':<{name_w}} |EU"
            f"| {'Mean CAGR':>{cw}} | {'Std CAGR':>{cw}} | {'Min CAGR':>{cw}} | {'Max CAGR':>{cw}} | {'Ann. σ':>{cw}} |"
            f" {'Windows':>8} |"
        )

    print(f"\n  Window: {label.upper()}")
    print(sep)
    print(hdr)
    print(sep)

    prev_eu = True
    for r, w in valid:
        if prev_eu and not r.eu_available:
            print(f"| {'── US benchmarks ──':<{len(sep)-4}} |")
            prev_eu = False
        eu_flag = "✓ " if r.eu_available else "  "
        if is_single:
            print(
                f"| {r.ticker:<{tick_w}} | {r.name[:name_w]:<{name_w}} |{eu_flag}"
                f"| {r.data_from.strftime('%Y-%m'):>8} | {_pct(w.cagr_mean):>{cw}} | {_pct(r.ann_vol_eur):>{cw}} |"
            )
        else:
            print(
                f"| {r.ticker:<{tick_w}} | {r.name[:name_w]:<{name_w}} |{eu_flag}"
                f"| {_pct(w.cagr_mean):>{cw}} | {_pct(w.cagr_std):>{cw}} | {_pct(w.cagr_min):>{cw}} | {_pct(w.cagr_max):>{cw}} | {_pct(r.ann_vol_eur):>{cw}} |"
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

    rk_w   = 2
    tick_w = 10
    name_w = 28
    yr_w   = 4
    cw     = 7
    nw     = 6   # n_windows column width
    cons_w = 6
    av_w   = 7

    sep = (
        f"+-{'-'*rk_w}-+-{'-'*tick_w}-+-{'-'*name_w}-+--+-{'-'*yr_w}-+"
        f"-{'-'*cw}-+-{'-'*nw}-+-{'-'*cw}-+-{'-'*cons_w}-+"
        f"-{'-'*cw}-+-{'-'*nw}-+-{'-'*cw}-+-{'-'*av_w}-+"
    )
    hdr = (
        f"| {'#':>{rk_w}} | {'Ticker':<{tick_w}} | {'Name':<{name_w}} |EU"
        f"| {'From':>{yr_w}} |"
        f" {ref_label+' avg':>{cw}} | {'n':>{nw}} | {ref_label+' std':>{cw}} | {'Cons.':>{cons_w}} |"
        f" {'5y avg':>{cw}} | {'n':>{nw}} | {'All-time':>{cw}} | {'Ann.σ':>{av_w}} |"
    )

    def _cons(r: TickerResult) -> float | None:
        w = r.windows.get(ref)
        if w and w.cagr_mean is not None and w.cagr_std and w.cagr_std > 0:
            return w.cagr_mean / w.cagr_std
        return None

    def _sort(r: TickerResult) -> tuple:
        w = r.windows.get(ref)
        mean = w.cagr_mean if (w and w.cagr_mean is not None) else None
        return (mean is None, -(mean or 0))

    def _nw_str(w: WindowStats | None) -> str:
        if w is None or w.n_windows == 0:
            return "n/a"
        return f"{w.n_windows:,}"

    eu_results = sorted([r for r in results if r.eu_available], key=_sort)
    us_results = sorted([r for r in results if not r.eu_available], key=_sort)

    def _row(rank: int, r: TickerResult) -> None:
        eu_flag = "✓ " if r.eu_available else "  "
        w_ref = r.windows.get(ref)
        w_5y  = r.windows.get(5)
        w_max = r.windows.get(None)
        cons  = _cons(r)
        print(
            f"| {rank:>{rk_w}} | {r.ticker:<{tick_w}} | {r.name[:name_w]:<{name_w}} |{eu_flag}"
            f"| {r.data_from.year:>{yr_w}} |"
            f" {_pct(w_ref.cagr_mean if w_ref else None):>{cw}} |"
            f" {_nw_str(w_ref):>{nw}} |"
            f" {_pct(w_ref.cagr_std  if w_ref else None):>{cw}} |"
            f" {(f'{cons:.1f}' if cons is not None else 'n/a'):>{cons_w}} |"
            f" {_pct(w_5y.cagr_mean  if w_5y  else None):>{cw}} |"
            f" {_nw_str(w_5y):>{nw}} |"
            f" {_pct(w_max.cagr_mean if w_max  else None):>{cw}} |"
            f" {_pct(r.ann_vol_eur):>{av_w}} |"
        )

    print(sep)
    print(hdr)
    print(sep)
    for i, r in enumerate(eu_results, 1):
        _row(i, r)
    if us_results:
        bench_label = "── US benchmarks (EUR-adjusted, not EU-eligible) ──"
        print(f"| {'':>{rk_w}} | {bench_label:<{len(sep)-rk_w-8}} |")
        for i, r in enumerate(us_results, 1):
            _row(i, r)
    print(sep)
    print(
        f"  Cons. = {ref_label} mean CAGR ÷ {ref_label} std  "
        f"(higher = more consistent outcome regardless of entry date)"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.system("cls" if os.name == "nt" else "clear")

    print("=" * 70)
    print(f"{'ETF CAGR ANALYSIS  —  ROLLING DAILY WINDOWS  (EUR-adjusted)':^70}")
    print("=" * 70)
    print(f"  EU tickers : {len(EU_TICKERS)}")
    print(f"  US bench.  : {len(US_TICKERS)}  (converted to EUR, not EU-eligible)")
    print(f"  Windows    : {', '.join(_window_label(y) for y in WINDOW_YEARS)}")
    print(f"  Sort by    : {_window_label(SORT_BY_WINDOW)} mean rolling CAGR")
    print(f"  FX         : USD→EUR via EURUSD=X  |  GBP→EUR via GBPEUR=X")
    print()
    print("  Fetching FX rate series...")

    fx_cache: dict[str, pd.Series | None] = {}
    for ccy, fx_ticker in FX_TICKERS.items():
        s = fetch_fx_series(ccy)
        fx_cache[ccy] = s
        status = f"{len(s)} days from {s.index[0].date()} to {s.index[-1].date()}" if s is not None else "FAILED"
        print(f"  {fx_ticker}  →  {status}")

    print()
    print("  Fetching ETF price series...")

    results: list[TickerResult] = []
    for ticker in TICKERS:
        eu = ticker in EU_TICKERS
        print(f"  → {ticker}", end="", flush=True)
        result = fetch_and_compute(ticker, fx_cache, eu)
        if result is not None:
            orig = result.currency_original
            fx_note = "" if orig == "EUR" else f" [{orig}→EUR]"
            print(f"  ✓  {result.data_from} → {result.data_to}{fx_note}")
            results.append(result)
        else:
            print("  ✗")

    if not results:
        print("\n  No data retrieved.")
        return

    ref_label = _window_label(SORT_BY_WINDOW)
    print()
    print("=" * 70)
    print(f"{'SUMMARY — MEAN ROLLING CAGR BY WINDOW  (all prices in EUR)':^70}")
    print("=" * 70)
    print(f"  Sorted by {ref_label} mean CAGR  |  Min/Max columns = worst/best {ref_label} period ever")
    print(f"  EU✓ = UCITS ETF available to EU retail investors\n")
    print_summary_table(results)

    print()
    print("=" * 70)
    print(f"{'DETAIL — RANKED BY MEAN CAGR PER WINDOW':^70}")
    print("=" * 70)
    print("  Min = worst N-year period recorded | Max = best | Ann.σ = daily vol (EUR)")
    for years in WINDOW_YEARS:
        print_detail_table(results, years)

    print()
    print("=" * 70)
    print(f"{'FINAL SUMMARY — KEY METRICS AT A GLANCE':^70}")
    print("=" * 70)
    print(f"  Ranked by {ref_label} mean CAGR  |  Cons. = mean÷std (timing consistency)\n")
    print_final_summary(results)

    print()
    print("=" * 70)
    print(f"{'NOTES':^70}")
    print("=" * 70)
    print("  Std CAGR: std dev of rolling N-year CAGRs across all daily start dates.")
    print("            Measures entry-point timing risk: how much the N-year outcome")
    print("            varies depending on when you started. Low std = consistent.")
    print()
    print("  Ann. σ  : annualised std dev of daily log returns in EUR — the key")
    print("            volatility input for Monte Carlo in main_portfolio_projection.py")
    print("            (currently set to 15%). Different from Std CAGR.")
    print()
    print("  EUR pricing: UCITS ETFs (.DE/.AS) are NAV-priced in EUR daily — their")
    print("  price already embeds EUR/USD movements. US ETFs are price-converted")
    print("  using historical EURUSD=X / GBPEUR=X rates so CAGR is comparable.")
    print()
    print("  MiFID II / PRIIPs: EU retail investors cannot buy US-domiciled ETFs")
    print("  (SPY, QQQ, VTI, VEA, VWO) directly. They are shown as benchmarks only.")
    print()


if __name__ == "__main__":
    main()
