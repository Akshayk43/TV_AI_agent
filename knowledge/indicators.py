"""Technical indicator knowledge and computation helpers."""

import numpy as np
import pandas as pd


# ── Indicator reference for the agent ─────────────────────────────────────────

INDICATOR_KNOWLEDGE = {
    "moving_averages": {
        "SMA": {
            "description": "Simple Moving Average — arithmetic mean of last N closes.",
            "use_cases": ["Trend direction", "Dynamic support/resistance"],
            "common_periods": [9, 20, 50, 100, 200],
            "pine": "ta.sma(close, length)",
        },
        "EMA": {
            "description": "Exponential Moving Average — weighted towards recent prices.",
            "use_cases": ["Faster trend detection", "Crossover signals"],
            "common_periods": [9, 12, 21, 26, 50, 200],
            "pine": "ta.ema(close, length)",
        },
        "VWAP": {
            "description": "Volume Weighted Average Price — average price weighted by volume.",
            "use_cases": ["Intraday fair value", "Institutional benchmark"],
            "pine": "ta.vwap(hlc3)",
        },
    },
    "momentum": {
        "RSI": {
            "description": "Relative Strength Index — momentum oscillator (0-100).",
            "overbought": 70,
            "oversold": 30,
            "use_cases": ["Overbought/oversold", "Divergence", "Trend confirmation"],
            "pine": "ta.rsi(close, 14)",
        },
        "MACD": {
            "description": "Moving Average Convergence Divergence — trend-following momentum.",
            "components": ["MACD line (12 EMA - 26 EMA)", "Signal line (9 EMA of MACD)", "Histogram"],
            "use_cases": ["Crossover signals", "Divergence", "Momentum strength"],
            "pine": "[macdLine, signalLine, hist] = ta.macd(close, 12, 26, 9)",
        },
        "Stochastic": {
            "description": "Stochastic Oscillator — compares close to high-low range.",
            "overbought": 80,
            "oversold": 20,
            "pine": "ta.stoch(close, high, low, 14)",
        },
    },
    "volatility": {
        "ATR": {
            "description": "Average True Range — volatility measure based on price range.",
            "use_cases": ["Stop-loss sizing", "Volatility filter", "Position sizing"],
            "common_periods": [14],
            "pine": "ta.atr(14)",
        },
        "Bollinger_Bands": {
            "description": "Moving average ± N standard deviations — volatility envelope.",
            "use_cases": ["Volatility squeeze", "Mean reversion", "Breakout detection"],
            "pine": "[middle, upper, lower] = ta.bb(close, 20, 2)",
        },
    },
    "volume": {
        "OBV": {
            "description": "On-Balance Volume — cumulative volume based on close direction.",
            "use_cases": ["Confirm trend", "Divergence detection"],
            "pine": "ta.obv",
        },
        "Volume_SMA": {
            "description": "Volume moving average — to spot above/below-average volume.",
            "use_cases": ["Volume confirmation", "Breakout validation"],
            "pine": "ta.sma(volume, 20)",
        },
    },
}


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    middle = compute_sma(series, period)
    std = series.rolling(window=period).std()
    return {
        "upper": middle + std_dev * std,
        "middle": middle,
        "lower": middle - std_dev * std,
    }


def compute_stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> dict:
    low_min = df["low"].rolling(window=period).min()
    high_max = df["high"].rolling(window=period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k_smooth = k.rolling(window=smooth_k).mean()
    d_smooth = k_smooth.rolling(window=smooth_d).mean()
    return {"k": k_smooth, "d": d_smooth}


def compute_obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff())
    sign.iloc[0] = 0
    return (sign * df["volume"]).cumsum()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_vwap = (typical_price * df["volume"]).cumsum()
    return cum_vwap / cum_vol


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and add all common indicators to the DataFrame."""
    df = df.copy()

    # Moving averages
    for p in [9, 20, 50, 200]:
        df[f"sma_{p}"] = compute_sma(df["close"], p)
        df[f"ema_{p}"] = compute_ema(df["close"], p)

    # RSI
    df["rsi_14"] = compute_rsi(df["close"], 14)

    # MACD
    macd = compute_macd(df["close"])
    df["macd"] = macd["macd"]
    df["macd_signal"] = macd["signal"]
    df["macd_histogram"] = macd["histogram"]

    # ATR
    df["atr_14"] = compute_atr(df, 14)

    # Bollinger Bands
    bb = compute_bollinger_bands(df["close"])
    df["bb_upper"] = bb["upper"]
    df["bb_middle"] = bb["middle"]
    df["bb_lower"] = bb["lower"]

    # Stochastic
    stoch = compute_stochastic(df)
    df["stoch_k"] = stoch["k"]
    df["stoch_d"] = stoch["d"]

    # OBV
    df["obv"] = compute_obv(df)

    # VWAP
    df["vwap"] = compute_vwap(df)

    # Volume SMA
    df["volume_sma_20"] = compute_sma(df["volume"], 20)

    return df


def get_indicator_summary(df: pd.DataFrame) -> str:
    """Generate an indicator-based market summary for the agent."""
    df = add_all_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = last["close"]

    lines = [
        "=== Technical Indicator Summary ===",
        f"Close: {close:.2f}",
        "",
        "— Moving Averages —",
    ]

    for p in [9, 20, 50, 200]:
        sma_val = last.get(f"sma_{p}")
        ema_val = last.get(f"ema_{p}")
        if pd.notna(sma_val):
            pos = "ABOVE" if close > sma_val else "BELOW"
            lines.append(f"  SMA({p}): {sma_val:.2f} — price {pos}")
        if pd.notna(ema_val):
            pos = "ABOVE" if close > ema_val else "BELOW"
            lines.append(f"  EMA({p}): {ema_val:.2f} — price {pos}")

    lines.append("")
    lines.append("— Momentum —")

    rsi = last.get("rsi_14")
    if pd.notna(rsi):
        zone = "OVERBOUGHT" if rsi > 70 else ("OVERSOLD" if rsi < 30 else "NEUTRAL")
        lines.append(f"  RSI(14): {rsi:.1f} — {zone}")

    macd_val = last.get("macd")
    sig_val = last.get("macd_signal")
    hist_val = last.get("macd_histogram")
    if pd.notna(macd_val):
        cross = "BULLISH" if macd_val > sig_val else "BEARISH"
        hist_dir = "expanding" if abs(hist_val) > abs(prev.get("macd_histogram", 0)) else "contracting"
        lines.append(f"  MACD: {macd_val:.4f} | Signal: {sig_val:.4f} | Histogram: {hist_val:.4f} — {cross}, {hist_dir}")

    stk = last.get("stoch_k")
    std = last.get("stoch_d")
    if pd.notna(stk):
        zone = "OVERBOUGHT" if stk > 80 else ("OVERSOLD" if stk < 20 else "NEUTRAL")
        lines.append(f"  Stochastic K: {stk:.1f} D: {std:.1f} — {zone}")

    lines.append("")
    lines.append("— Volatility —")

    atr = last.get("atr_14")
    if pd.notna(atr):
        atr_pct = (atr / close) * 100
        lines.append(f"  ATR(14): {atr:.2f} ({atr_pct:.2f}% of price)")

    bb_u = last.get("bb_upper")
    bb_l = last.get("bb_lower")
    bb_m = last.get("bb_middle")
    if pd.notna(bb_u):
        bb_width = (bb_u - bb_l) / bb_m * 100
        bb_pos = (close - bb_l) / (bb_u - bb_l) * 100 if (bb_u - bb_l) > 0 else 50
        lines.append(f"  BB(20,2): Upper={bb_u:.2f} Mid={bb_m:.2f} Lower={bb_l:.2f}")
        lines.append(f"  BB Width: {bb_width:.2f}% | Price at {bb_pos:.0f}% of band")

    lines.append("")
    lines.append("— Volume —")

    vol = last.get("volume", 0)
    vol_sma = last.get("volume_sma_20", 0)
    if vol_sma and vol_sma > 0:
        vol_ratio = vol / vol_sma
        vol_status = "ABOVE average" if vol_ratio > 1.2 else ("BELOW average" if vol_ratio < 0.8 else "AVERAGE")
        lines.append(f"  Volume: {vol:,.0f} | SMA(20): {vol_sma:,.0f} | Ratio: {vol_ratio:.2f} — {vol_status}")

    obv = last.get("obv")
    if pd.notna(obv):
        obv_trend = "rising" if obv > prev.get("obv", obv) else "falling"
        lines.append(f"  OBV: {obv:,.0f} — {obv_trend}")

    return "\n".join(lines)
