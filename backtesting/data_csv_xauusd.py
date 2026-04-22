"""CSV fallback loader for XAUUSD historical bars.

Use when OANDA credentials are unavailable. Supports two common formats:
  1. Dukascopy CSV export: https://www.dukascopy.com/swiss/english/marketwatch/historical/
     Columns: Gmt time, Open, High, Low, Close, Volume
  2. HistData.com ASCII 1-minute: timestamp,open,high,low,close,volume

Dukascopy gives cleaner tick → 1m aggregation and is free without login for
up to 1 month at a time (download monthly, concatenate locally).
"""

import glob
import os
from datetime import datetime

import pandas as pd


def load_dukascopy(path: str) -> pd.DataFrame:
    """Load a Dukascopy historical CSV (semicolon- or comma-delimited).

    Dukascopy exports look like:
        Gmt time,Open,High,Low,Close,Volume
        01.01.2024 22:00:00.000,2063.50,2064.12,2063.28,2063.95,12.34

    Returns DataFrame with lowercase OHLCV and UTC-naive datetime index.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    time_col = next((c for c in ("gmt_time", "time", "date", "datetime", "timestamp") if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"No time column found in {path}. Columns: {list(df.columns)}")

    df[time_col] = pd.to_datetime(df[time_col], format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")
    if df[time_col].isna().all():
        # Fall back to flexible parser
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df = df.dropna(subset=[time_col])
    df.set_index(time_col, inplace=True)

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {path}")
    df = df[required].astype(float).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_histdata(path: str) -> pd.DataFrame:
    """Load a HistData.com generic ASCII 1-minute file.

    Format: YYYYMMDD HHMMSS;open;high;low;close;volume
    """
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["time", "open", "high", "low", "close", "volume"],
        dtype={"time": str},
    )
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d %H%M%S", errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def load_xauusd_csv_dir(directory: str, pattern: str = "*.csv") -> pd.DataFrame:
    """Load and concatenate all XAUUSD CSV files in a directory.

    Auto-detects Dukascopy vs HistData format based on first-row sniff.
    Files are assumed chronologically ordered; duplicates are dropped.
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSV files match {os.path.join(directory, pattern)}")

    frames = []
    for f in files:
        with open(f, "r") as fh:
            first = fh.readline().strip()
        try:
            if "gmt" in first.lower() or "," in first and "open" in first.lower():
                frames.append(load_dukascopy(f))
            else:
                frames.append(load_histdata(f))
        except Exception as e:
            print(f"[WARNING] Skipping {f}: {e}")

    if not frames:
        raise ValueError(f"No parseable CSVs in {directory}")

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def resample_to(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample 1-minute bars up to 5m / 15m / etc."""
    rule_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1H", "4h": "4H"}
    rule = rule_map.get(interval, interval)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna()
    return out
