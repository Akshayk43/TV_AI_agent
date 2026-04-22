"""OANDA v20 REST loader for XAUUSD (and other FX/metals).

OANDA's v20 REST API is the recommended source for intraday XAUUSD data:
- Free practice account — no cost for historical candles
- 5000 candles per request (paginate for longer ranges)
- Native XAU_USD instrument
- Granularities: S5, S10, S30, M1, M5, M15, M30, H1, H4, D

Auth: set OANDA_API_KEY and OANDA_ACCOUNT_ID environment variables. Optionally
set OANDA_ENV=practice (default) or OANDA_ENV=live for the real-money endpoint.

Usage:
    from backtesting.data_oanda import fetch_xauusd
    df = fetch_xauusd(days=30, interval="5m")
"""

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd


OANDA_ENDPOINTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

INTERVAL_MAP = {
    "5s": "S5", "10s": "S10", "30s": "S30",
    "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
    "1h": "H1", "4h": "H4", "1d": "D",
}

DEFAULT_INSTRUMENT = "XAU_USD"


class OANDAError(RuntimeError):
    """Raised when OANDA returns an error or env is misconfigured."""


def _get_credentials() -> tuple[str, str]:
    api_key = os.getenv("OANDA_API_KEY", "").strip()
    account_id = os.getenv("OANDA_ACCOUNT_ID", "").strip()
    if not api_key:
        raise OANDAError(
            "OANDA_API_KEY not set. Create a free practice account at "
            "https://www.oanda.com/demo-account/tpa/personal_token and export "
            "OANDA_API_KEY and OANDA_ACCOUNT_ID."
        )
    return api_key, account_id


def _get_endpoint() -> str:
    env = os.getenv("OANDA_ENV", "practice").lower()
    if env not in OANDA_ENDPOINTS:
        raise OANDAError(f"OANDA_ENV must be 'practice' or 'live', got {env!r}")
    return OANDA_ENDPOINTS[env]


def _fetch_candles(
    instrument: str,
    granularity: str,
    from_time: datetime,
    to_time: datetime,
    price: str = "M",
) -> list[dict]:
    """Fetch up to 5000 candles. Paginates if range exceeds the API limit."""
    import requests

    api_key, _ = _get_credentials()
    endpoint = _get_endpoint()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept-Datetime-Format": "RFC3339",
    }
    url = f"{endpoint}/v3/instruments/{instrument}/candles"

    candles: list[dict] = []
    cursor = from_time

    while cursor < to_time:
        params = {
            "granularity": granularity,
            "from": cursor.isoformat().replace("+00:00", "Z"),
            "to": to_time.isoformat().replace("+00:00", "Z"),
            "price": price,
            "count": 5000,
        }
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=30)
                break
            except requests.RequestException as e:
                if attempt == 2:
                    raise OANDAError(f"OANDA request failed: {e}") from e
                time.sleep(2 ** attempt)

        if resp.status_code != 200:
            raise OANDAError(f"OANDA {resp.status_code}: {resp.text[:500]}")

        batch = resp.json().get("candles", [])
        if not batch:
            break

        candles.extend(batch)

        # Advance cursor past the last candle returned
        last_ts = datetime.fromisoformat(batch[-1]["time"].replace("Z", "+00:00"))
        if last_ts <= cursor:
            break
        cursor = last_ts + timedelta(seconds=1)

        if len(batch) < 5000:
            break  # Got everything the server has for this range

    return candles


def fetch_xauusd(
    days: int = 30,
    interval: str = "5m",
    end: datetime | None = None,
    instrument: str = DEFAULT_INSTRUMENT,
) -> pd.DataFrame:
    """Fetch XAU_USD OHLCV from OANDA.

    Args:
        days: Lookback window in days.
        interval: One of '5s','10s','30s','1m','5m','15m','30m','1h','4h','1d'.
        end: End timestamp (UTC). Defaults to now.
        instrument: OANDA instrument name (default XAU_USD).

    Returns DataFrame with columns open, high, low, close, volume. UTC index.
    """
    if interval not in INTERVAL_MAP:
        raise ValueError(f"Unsupported interval: {interval}. Use one of {list(INTERVAL_MAP)}")

    granularity = INTERVAL_MAP[interval]
    end = end or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=days)

    candles = _fetch_candles(instrument, granularity, start, end)

    rows = []
    for c in candles:
        if not c.get("complete", False):
            continue
        mid = c.get("mid", {})
        rows.append({
            "time": c["time"],
            "open": float(mid.get("o", 0)),
            "high": float(mid.get("h", 0)),
            "low": float(mid.get("l", 0)),
            "close": float(mid.get("c", 0)),
            "volume": float(c.get("volume", 0)),
        })

    if not rows:
        raise OANDAError(
            f"No candles returned for {instrument} {granularity} over {days}d. "
            f"Possible causes: weekend, market closed, or invalid credentials."
        )

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)
    df.index = df.index.tz_convert(None)  # strip tz for engine compatibility
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df
