"""Block bootstrap return generator for Monte Carlo simulations.

Draws contiguous blocks from historical annual stock returns (circular
block bootstrap), preserving fat tails, volatility clustering, and
autocorrelation without parametric assumptions.

Data source: spliced S&P 500 total returns — SPY (1993+, total return)
plus ^GSPC (pre-1993, price-only with a constant dividend adjustment
derived from the overlap period).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_cache: dict[str, np.ndarray] = {}

_GSPC_PARQUET = "IDX_GSPC.parquet"
_SPY_PARQUET = "SPY.parquet"


def _annual_returns_from_parquet(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    close = df["Close"].dropna()
    close.index = pd.to_datetime(close.index).normalize()
    yearly = close.resample("YE").last().dropna()
    return yearly.pct_change().dropna()


def load_bootstrap_annual_returns(
    cache_dir: str = "historical_data",
) -> np.ndarray:
    """Load and splice S&P 500 annual total returns.

    Returns a 1-D numpy array of ~99 annual return observations:
    - 1993+: SPY total return (dividends included via auto_adjust)
    - Pre-1993: ^GSPC price return + constant dividend adjustment
      (gap = mean(SPY) − mean(GSPC) over the overlap period)

    Falls back to GSPC-only if SPY parquet is missing.
    """
    cache_key = cache_dir
    if cache_key in _cache:
        return _cache[cache_key]

    base = Path(cache_dir)
    gspc_path = base / _GSPC_PARQUET
    spy_path = base / _SPY_PARQUET

    if not gspc_path.exists():
        raise FileNotFoundError(
            f"Bootstrap requires {gspc_path} — run main_cagr_analysis.py first "
            f"to populate the historical data cache."
        )

    gspc = _annual_returns_from_parquet(gspc_path)

    if spy_path.exists():
        spy = _annual_returns_from_parquet(spy_path)
        overlap = gspc.index.intersection(spy.index)
        if len(overlap) >= 5:
            div_gap = float(spy.loc[overlap].mean() - gspc.loc[overlap].mean())
            pre_overlap = gspc[gspc.index < overlap[0]] + div_gap
            result = pd.concat([pre_overlap, spy.loc[overlap[0]:]]).sort_index()
        else:
            result = gspc
    else:
        result = gspc

    arr = result.values.astype(np.float64)
    _cache[cache_key] = arr
    return arr


def block_bootstrap_returns(
    rng: np.random.Generator,
    annual_returns: np.ndarray,
    n_simulations: int,
    years: int,
    block_size: int = 5,
    target_mean: float | None = None,
) -> np.ndarray:
    """Circular block bootstrap from historical annual returns.

    Draws contiguous blocks of `block_size` years (wrapping at array end),
    concatenates them into `(n_simulations, years)` paths, and optionally
    mean-shifts to match `target_mean`.

    Returns an array of shape ``(n_simulations, years)`` with the same
    contract as the parametric stock return arrays.
    """
    n_obs = len(annual_returns)
    min_obs = max(10, 2 * block_size)
    if n_obs < min_obs:
        raise ValueError(
            f"Block bootstrap requires at least {min_obs} annual return "
            f"observations; got {n_obs}. Use a longer-history source or "
            f"reduce mc_bootstrap_block_size."
        )

    n_blocks = int(np.ceil(years / block_size))
    starts = rng.integers(0, n_obs, size=(n_simulations, n_blocks))
    offsets = np.arange(block_size)
    indices = (starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]) % n_obs
    flat = annual_returns[indices.reshape(n_simulations, -1)]
    result = flat[:, :years]

    if target_mean is not None:
        result = result + (target_mean - result.mean())

    return np.maximum(result, -0.99)
