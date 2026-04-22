"""Market data loading from yfinance and CSV."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config.settings import DEFAULT_LOOKBACK_DAYS


def _get_yf():
    """Lazy-import yfinance to avoid hard dep at module level."""
    try:
        import yfinance as yf
        return yf
    except ImportError:
        raise ImportError(
            "yfinance is required for live data. Install it: pip install yfinance\n"
            "Alternatively, use load_from_csv() with a local CSV file."
        )


def load_from_yahoo(
    symbol: str,
    period_days: int = DEFAULT_LOOKBACK_DAYS,
    interval: str = "1d",
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance.

    Args:
        symbol: Ticker symbol (e.g. "AAPL", "BTCUSD=X").
        period_days: Number of days of history.
        interval: Candle interval — '1m','5m','15m','1h','1d','1wk','1mo'.
        end_date: End date (defaults to today).

    Returns:
        DataFrame with columns: open, high, low, close, volume (lowercase).
    """
    yf = _get_yf()

    end = end_date or datetime.now()
    start = end - timedelta(days=period_days)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} ({interval}, last {period_days}d)")

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Keep only OHLCV
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[required].copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)  # remove timezone for compatibility
    df.dropna(inplace=True)

    return df


def load_from_csv(filepath: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Expected columns (case-insensitive): date/datetime, open, high, low, close, volume.
    """
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Identify date column
    date_col = None
    for candidate in ["date", "datetime", "time", "timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[required].copy()
    df.dropna(inplace=True)

    return df


def generate_sample_data(
    symbol: str = "SAMPLE",
    days: int = 365,
    start_price: float = 100.0,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing when yfinance is unavailable.

    Creates realistic-looking price data with trends, mean-reversion, and volume.
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")  # business days

    prices = [start_price]
    for _ in range(1, days):
        # Random walk with slight drift
        change = np.random.normal(0.0003, volatility)
        prices.append(prices[-1] * (1 + change))

    closes = np.array(prices)
    # Generate OHLV from close
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, volatility / 2, days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, volatility / 2, days)))
    volumes = np.random.randint(1_000_000, 50_000_000, days).astype(float)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=dates[:days])

    return df


def _is_xauusd_symbol(symbol: str) -> bool:
    s = symbol.upper().replace("=X", "").replace("/", "").replace("_", "").replace("-", "")
    return s.startswith("XAUUSD") or s in ("GOLD", "XAU", "GCUSD")


def get_market_data(symbol: str, period_days: int = DEFAULT_LOOKBACK_DAYS, interval: str = "1d") -> pd.DataFrame:
    """High-level data retrieval with instrument-aware routing.

    For XAUUSD (spot gold), try OANDA first, then a local CSV cache, then yfinance.
    Fail loudly rather than silently returning synthetic data — intraday gold
    synthetics would hide every real bug in the strategy.

    For other symbols, preserve legacy behavior: Yahoo → synthetic fallback.
    """
    if _is_xauusd_symbol(symbol):
        errors = []

        # 1. OANDA (preferred — clean 1m/5m bars back years)
        try:
            from backtesting.data_oanda import fetch_xauusd
            return fetch_xauusd(days=period_days, interval=interval)
        except Exception as e:
            errors.append(f"OANDA: {e}")

        # 2. Local CSV cache (Dukascopy-style)
        import os
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "xauusd_csv",
        )
        if os.path.isdir(cache_dir):
            try:
                from backtesting.data_csv_xauusd import load_xauusd_csv_dir, resample_to
                raw = load_xauusd_csv_dir(cache_dir)
                return resample_to(raw, interval) if interval != "1m" else raw
            except Exception as e:
                errors.append(f"CSV: {e}")

        # 3. yfinance GC=F (futures proxy — limited intraday history)
        try:
            return load_from_yahoo("GC=F", period_days, interval)
        except Exception as e:
            errors.append(f"yfinance: {e}")

        raise RuntimeError(
            f"Could not load XAUUSD data from any source. Tried:\n  "
            + "\n  ".join(errors)
            + "\nSet OANDA_API_KEY or drop CSVs in data/xauusd_csv/."
        )

    # Non-XAUUSD: legacy path
    try:
        return load_from_yahoo(symbol, period_days, interval)
    except Exception as e:
        print(f"[WARNING] Could not fetch live data for {symbol}: {e}")
        print("[WARNING] Using synthetic data for demonstration...")
        return generate_sample_data(symbol=symbol, days=period_days)
