"""Candlestick pattern definitions and detection logic."""

import pandas as pd
import numpy as np


# ── Pattern catalog (used by the agent as reference knowledge) ────────────────

CANDLESTICK_PATTERNS = {
    # ── Single-candle patterns ────────────────────────────────────────────
    "doji": {
        "type": "neutral",
        "description": "Open and close are virtually equal. Signals indecision.",
        "significance": "Potential reversal when appearing after a trend.",
        "pine_condition": "math.abs(close - open) <= (high - low) * 0.1",
    },
    "hammer": {
        "type": "bullish_reversal",
        "description": "Small body at the top, long lower shadow (≥2x body). Appears in downtrends.",
        "significance": "Strong bullish reversal signal at support levels.",
        "pine_condition": (
            "(min(open, close) - low) >= 2 * math.abs(close - open) and "
            "(high - max(open, close)) <= math.abs(close - open) * 0.3"
        ),
    },
    "inverted_hammer": {
        "type": "bullish_reversal",
        "description": "Small body at the bottom, long upper shadow. Appears in downtrends.",
        "significance": "Potential bullish reversal, needs confirmation.",
        "pine_condition": (
            "(high - max(open, close)) >= 2 * math.abs(close - open) and "
            "(min(open, close) - low) <= math.abs(close - open) * 0.3"
        ),
    },
    "hanging_man": {
        "type": "bearish_reversal",
        "description": "Same shape as hammer but appears in uptrends.",
        "significance": "Bearish reversal signal at resistance levels.",
        "pine_condition": (
            "(min(open, close) - low) >= 2 * math.abs(close - open) and "
            "(high - max(open, close)) <= math.abs(close - open) * 0.3"
        ),
    },
    "shooting_star": {
        "type": "bearish_reversal",
        "description": "Small body at the bottom, long upper shadow. Appears in uptrends.",
        "significance": "Strong bearish reversal signal.",
        "pine_condition": (
            "(high - max(open, close)) >= 2 * math.abs(close - open) and "
            "(min(open, close) - low) <= math.abs(close - open) * 0.3"
        ),
    },
    "marubozu_bullish": {
        "type": "bullish_continuation",
        "description": "Long bullish body with no/tiny shadows.",
        "significance": "Strong buying pressure, bullish continuation.",
        "pine_condition": (
            "close > open and "
            "(high - close) <= (close - open) * 0.05 and "
            "(open - low) <= (close - open) * 0.05"
        ),
    },
    "marubozu_bearish": {
        "type": "bearish_continuation",
        "description": "Long bearish body with no/tiny shadows.",
        "significance": "Strong selling pressure, bearish continuation.",
        "pine_condition": (
            "open > close and "
            "(high - open) <= (open - close) * 0.05 and "
            "(close - low) <= (open - close) * 0.05"
        ),
    },
    "spinning_top": {
        "type": "neutral",
        "description": "Small body with upper and lower shadows of similar length.",
        "significance": "Indecision in the market, potential reversal.",
        "pine_condition": (
            "math.abs(close - open) <= (high - low) * 0.3 and "
            "(high - max(open, close)) > math.abs(close - open) and "
            "(min(open, close) - low) > math.abs(close - open)"
        ),
    },

    # ── Two-candle patterns ───────────────────────────────────────────────
    "bullish_engulfing": {
        "type": "bullish_reversal",
        "description": "Bearish candle followed by a larger bullish candle that engulfs it.",
        "significance": "Strong bullish reversal, especially at support.",
        "pine_condition": (
            "close[1] < open[1] and close > open and "
            "open <= close[1] and close >= open[1]"
        ),
    },
    "bearish_engulfing": {
        "type": "bearish_reversal",
        "description": "Bullish candle followed by a larger bearish candle that engulfs it.",
        "significance": "Strong bearish reversal, especially at resistance.",
        "pine_condition": (
            "close[1] > open[1] and close < open and "
            "open >= close[1] and close <= open[1]"
        ),
    },
    "piercing_line": {
        "type": "bullish_reversal",
        "description": "Bearish candle followed by bullish candle that opens below prior low and closes above midpoint.",
        "significance": "Bullish reversal at support levels.",
        "pine_condition": (
            "close[1] < open[1] and close > open and "
            "open < low[1] and close > (open[1] + close[1]) / 2"
        ),
    },
    "dark_cloud_cover": {
        "type": "bearish_reversal",
        "description": "Bullish candle followed by bearish candle that opens above prior high and closes below midpoint.",
        "significance": "Bearish reversal at resistance levels.",
        "pine_condition": (
            "close[1] > open[1] and close < open and "
            "open > high[1] and close < (open[1] + close[1]) / 2"
        ),
    },
    "tweezer_top": {
        "type": "bearish_reversal",
        "description": "Two candles with matching highs at top of uptrend.",
        "significance": "Bearish reversal signal.",
        "pine_condition": (
            "math.abs(high - high[1]) <= (high - low) * 0.02 and "
            "close[1] > open[1] and close < open"
        ),
    },
    "tweezer_bottom": {
        "type": "bullish_reversal",
        "description": "Two candles with matching lows at bottom of downtrend.",
        "significance": "Bullish reversal signal.",
        "pine_condition": (
            "math.abs(low - low[1]) <= (high - low) * 0.02 and "
            "close[1] < open[1] and close > open"
        ),
    },

    # ── Three-candle patterns ─────────────────────────────────────────────
    "morning_star": {
        "type": "bullish_reversal",
        "description": "Bearish candle, small-body candle (gap down), then bullish candle closing above first candle midpoint.",
        "significance": "Strong bullish reversal pattern.",
        "pine_condition": (
            "close[2] < open[2] and "
            "math.abs(close[1] - open[1]) < math.abs(close[2] - open[2]) * 0.3 and "
            "close > open and close > (open[2] + close[2]) / 2"
        ),
    },
    "evening_star": {
        "type": "bearish_reversal",
        "description": "Bullish candle, small-body candle (gap up), then bearish candle closing below first candle midpoint.",
        "significance": "Strong bearish reversal pattern.",
        "pine_condition": (
            "close[2] > open[2] and "
            "math.abs(close[1] - open[1]) < math.abs(close[2] - open[2]) * 0.3 and "
            "close < open and close < (open[2] + close[2]) / 2"
        ),
    },
    "three_white_soldiers": {
        "type": "bullish_continuation",
        "description": "Three consecutive bullish candles, each closing higher.",
        "significance": "Strong bullish momentum continuation.",
        "pine_condition": (
            "close > open and close[1] > open[1] and close[2] > open[2] and "
            "close > close[1] and close[1] > close[2] and "
            "open > open[1] and open[1] > open[2]"
        ),
    },
    "three_black_crows": {
        "type": "bearish_continuation",
        "description": "Three consecutive bearish candles, each closing lower.",
        "significance": "Strong bearish momentum continuation.",
        "pine_condition": (
            "close < open and close[1] < open[1] and close[2] < open[2] and "
            "close < close[1] and close[1] < close[2] and "
            "open < open[1] and open[1] < open[2]"
        ),
    },
}


def detect_patterns(df: pd.DataFrame) -> dict[str, list[int]]:
    """Detect candlestick patterns in OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume

    Returns:
        Dict mapping pattern name → list of bar indices where pattern was detected.
    """
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    body = np.abs(c - o)
    rng = h - l
    rng[rng == 0] = 1e-10  # avoid division by zero

    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - l

    detected: dict[str, list[int]] = {}

    # ── Single-candle ─────────────────────────────────────────────────────
    # Doji
    mask = body <= rng * 0.1
    detected["doji"] = list(np.where(mask)[0])

    # Hammer (bullish reversal)
    mask = (lower_shadow >= 2 * body) & (upper_shadow <= body * 0.3) & (body > rng * 0.05)
    detected["hammer"] = list(np.where(mask)[0])

    # Inverted hammer
    mask = (upper_shadow >= 2 * body) & (lower_shadow <= body * 0.3) & (body > rng * 0.05)
    detected["inverted_hammer"] = list(np.where(mask)[0])

    # Shooting star (same shape as inverted hammer but in uptrend context)
    detected["shooting_star"] = list(np.where(mask)[0])

    # Marubozu bullish
    mask = (c > o) & (upper_shadow <= body * 0.05) & (lower_shadow <= body * 0.05) & (body > rng * 0.8)
    detected["marubozu_bullish"] = list(np.where(mask)[0])

    # Marubozu bearish
    mask = (o > c) & (upper_shadow <= body * 0.05) & (lower_shadow <= body * 0.05) & (body > rng * 0.8)
    detected["marubozu_bearish"] = list(np.where(mask)[0])

    # Spinning top
    mask = (body <= rng * 0.3) & (upper_shadow > body) & (lower_shadow > body) & (body > rng * 0.05)
    detected["spinning_top"] = list(np.where(mask)[0])

    # ── Two-candle (start from index 1) ───────────────────────────────────
    n = len(df)
    if n < 2:
        return detected

    # Bullish engulfing
    indices = []
    for i in range(1, n):
        if c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] <= c[i - 1] and c[i] >= o[i - 1]:
            indices.append(i)
    detected["bullish_engulfing"] = indices

    # Bearish engulfing
    indices = []
    for i in range(1, n):
        if c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] >= c[i - 1] and c[i] <= o[i - 1]:
            indices.append(i)
    detected["bearish_engulfing"] = indices

    # Piercing line
    indices = []
    for i in range(1, n):
        mid_prev = (o[i - 1] + c[i - 1]) / 2
        if c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] < l[i - 1] and c[i] > mid_prev:
            indices.append(i)
    detected["piercing_line"] = indices

    # Dark cloud cover
    indices = []
    for i in range(1, n):
        mid_prev = (o[i - 1] + c[i - 1]) / 2
        if c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] > h[i - 1] and c[i] < mid_prev:
            indices.append(i)
    detected["dark_cloud_cover"] = indices

    # Tweezer top
    indices = []
    for i in range(1, n):
        if (abs(h[i] - h[i - 1]) <= rng[i] * 0.02 and c[i - 1] > o[i - 1] and c[i] < o[i]):
            indices.append(i)
    detected["tweezer_top"] = indices

    # Tweezer bottom
    indices = []
    for i in range(1, n):
        if (abs(l[i] - l[i - 1]) <= rng[i] * 0.02 and c[i - 1] < o[i - 1] and c[i] > o[i]):
            indices.append(i)
    detected["tweezer_bottom"] = indices

    # ── Three-candle (start from index 2) ─────────────────────────────────
    if n < 3:
        return detected

    # Morning star
    indices = []
    for i in range(2, n):
        body2 = abs(c[i - 2] - o[i - 2])
        if body2 == 0:
            continue
        mid_first = (o[i - 2] + c[i - 2]) / 2
        if (c[i - 2] < o[i - 2]
                and abs(c[i - 1] - o[i - 1]) < body2 * 0.3
                and c[i] > o[i]
                and c[i] > mid_first):
            indices.append(i)
    detected["morning_star"] = indices

    # Evening star
    indices = []
    for i in range(2, n):
        body2 = abs(c[i - 2] - o[i - 2])
        if body2 == 0:
            continue
        mid_first = (o[i - 2] + c[i - 2]) / 2
        if (c[i - 2] > o[i - 2]
                and abs(c[i - 1] - o[i - 1]) < body2 * 0.3
                and c[i] < o[i]
                and c[i] < mid_first):
            indices.append(i)
    detected["evening_star"] = indices

    # Three white soldiers
    indices = []
    for i in range(2, n):
        if (c[i] > o[i] and c[i - 1] > o[i - 1] and c[i - 2] > o[i - 2]
                and c[i] > c[i - 1] > c[i - 2]
                and o[i] > o[i - 1] > o[i - 2]):
            indices.append(i)
    detected["three_white_soldiers"] = indices

    # Three black crows
    indices = []
    for i in range(2, n):
        if (c[i] < o[i] and c[i - 1] < o[i - 1] and c[i - 2] < o[i - 2]
                and c[i] < c[i - 1] < c[i - 2]
                and o[i] < o[i - 1] < o[i - 2]):
            indices.append(i)
    detected["three_black_crows"] = indices

    return detected


def get_pattern_summary(detected: dict[str, list[int]], lookback: int = 10) -> str:
    """Return a human-readable summary of recently detected patterns."""
    lines = []
    for name, indices in detected.items():
        if not indices:
            continue
        recent = [i for i in indices if i >= (max(max(v) for v in detected.values() if v) - lookback)]
        if recent:
            info = CANDLESTICK_PATTERNS.get(name, {})
            lines.append(
                f"  {name}: detected at bars {recent} — "
                f"{info.get('type', 'unknown')} — {info.get('significance', '')}"
            )
    if not lines:
        return "No significant candlestick patterns detected in recent bars."
    return "Candlestick Patterns Detected:\n" + "\n".join(lines)
