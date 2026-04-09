"""Price action analysis: support/resistance, trend, structure."""

import numpy as np
import pandas as pd


# ── Knowledge reference for the agent ─────────────────────────────────────────

PRICE_ACTION_CONCEPTS = {
    "support_resistance": {
        "description": (
            "Key horizontal price levels where buying (support) or selling (resistance) "
            "pressure has historically been significant."
        ),
        "identification": [
            "Previous swing highs and lows",
            "Areas of price consolidation",
            "Round psychological numbers",
            "High-volume nodes from volume profile",
        ],
        "trading_rules": [
            "Buy near support with confirmation",
            "Sell near resistance with confirmation",
            "Breakout above resistance → new support",
            "Breakdown below support → new resistance",
        ],
    },
    "trend_structure": {
        "description": "Market structure defined by higher highs/lows (uptrend) or lower highs/lows (downtrend).",
        "uptrend": "Higher highs (HH) and higher lows (HL)",
        "downtrend": "Lower highs (LH) and lower lows (LL)",
        "ranging": "Price oscillating between support and resistance without clear HH/HL or LH/LL",
    },
    "breakout_pullback": {
        "description": "Price breaking through a key level, then pulling back to retest it.",
        "rules": [
            "Wait for candle close above/below the level",
            "Look for pullback to the broken level",
            "Enter on confirmation (e.g., bullish candle at new support)",
            "Volume should increase on the breakout",
        ],
    },
    "order_blocks": {
        "description": "Institutional supply/demand zones — the last opposing candle before an impulsive move.",
        "bullish_ob": "Last bearish candle before a strong bullish move up",
        "bearish_ob": "Last bullish candle before a strong bearish move down",
    },
    "fair_value_gaps": {
        "description": "Imbalance zones (FVG) where price moved so fast it left a gap between candle wicks.",
        "bullish_fvg": "Gap between candle[2].high and candle[0].low in an up-move",
        "bearish_fvg": "Gap between candle[0].high and candle[2].low in a down-move",
        "trading": "Price tends to return to fill these gaps.",
    },
}


def find_swing_points(df: pd.DataFrame, window: int = 5) -> tuple[list[int], list[int]]:
    """Identify swing highs and swing lows.

    Returns:
        (swing_high_indices, swing_low_indices)
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_highs, swing_lows = [], []

    for i in range(window, n - window):
        if all(highs[i] >= highs[i - j] for j in range(1, window + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, window + 1)):
            swing_highs.append(i)

        if all(lows[i] <= lows[i - j] for j in range(1, window + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, window + 1)):
            swing_lows.append(i)

    return swing_highs, swing_lows


def find_support_resistance(df: pd.DataFrame, window: int = 5, merge_pct: float = 0.01) -> dict:
    """Find support and resistance levels from swing points.

    Args:
        df: OHLCV DataFrame.
        window: Swing detection lookback/lookahead.
        merge_pct: Percentage threshold to merge nearby levels.

    Returns:
        Dict with 'support' and 'resistance' level lists.
    """
    swing_highs, swing_lows = find_swing_points(df, window)

    resistance_levels = sorted(set(df["high"].iloc[i] for i in swing_highs))
    support_levels = sorted(set(df["low"].iloc[i] for i in swing_lows))

    def merge_levels(levels: list[float], pct: float) -> list[float]:
        if not levels:
            return []
        merged = [levels[0]]
        for lv in levels[1:]:
            if abs(lv - merged[-1]) / merged[-1] < pct:
                merged[-1] = (merged[-1] + lv) / 2  # average nearby levels
            else:
                merged.append(lv)
        return merged

    return {
        "support": merge_levels(support_levels, merge_pct),
        "resistance": merge_levels(resistance_levels, merge_pct),
    }


def detect_trend(df: pd.DataFrame, window: int = 5) -> dict:
    """Detect current market trend using swing-point structure.

    Returns:
        Dict with 'trend' (uptrend/downtrend/ranging), 'strength', and 'details'.
    """
    swing_highs, swing_lows = find_swing_points(df, window)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"trend": "unknown", "strength": 0, "details": "Not enough swing points."}

    highs = df["high"].values
    lows = df["low"].values

    recent_sh = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
    recent_sl = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows

    hh_count = sum(1 for i in range(1, len(recent_sh)) if highs[recent_sh[i]] > highs[recent_sh[i - 1]])
    hl_count = sum(1 for i in range(1, len(recent_sl)) if lows[recent_sl[i]] > lows[recent_sl[i - 1]])
    lh_count = sum(1 for i in range(1, len(recent_sh)) if highs[recent_sh[i]] < highs[recent_sh[i - 1]])
    ll_count = sum(1 for i in range(1, len(recent_sl)) if lows[recent_sl[i]] < lows[recent_sl[i - 1]])

    bull_score = hh_count + hl_count
    bear_score = lh_count + ll_count

    if bull_score > bear_score and bull_score >= 2:
        trend = "uptrend"
        strength = min(bull_score / 4, 1.0)
    elif bear_score > bull_score and bear_score >= 2:
        trend = "downtrend"
        strength = min(bear_score / 4, 1.0)
    else:
        trend = "ranging"
        strength = 0.0

    return {
        "trend": trend,
        "strength": round(strength, 2),
        "details": (
            f"HH={hh_count} HL={hl_count} LH={lh_count} LL={ll_count} | "
            f"Recent swing highs at bars {recent_sh}, lows at bars {recent_sl}"
        ),
    }


def detect_fair_value_gaps(df: pd.DataFrame, min_gap_pct: float = 0.002) -> list[dict]:
    """Detect Fair Value Gaps (imbalance zones).

    Returns:
        List of dicts with 'type', 'bar', 'top', 'bottom' for each FVG.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    fvgs = []

    for i in range(2, len(df)):
        # Bullish FVG: candle[i-2].high < candle[i].low
        if lows[i] > highs[i - 2]:
            gap = lows[i] - highs[i - 2]
            if gap / closes[i] >= min_gap_pct:
                fvgs.append({
                    "type": "bullish_fvg",
                    "bar": i,
                    "top": lows[i],
                    "bottom": highs[i - 2],
                })

        # Bearish FVG: candle[i].high < candle[i-2].low
        if highs[i] < lows[i - 2]:
            gap = lows[i - 2] - highs[i]
            if gap / closes[i] >= min_gap_pct:
                fvgs.append({
                    "type": "bearish_fvg",
                    "bar": i,
                    "top": lows[i - 2],
                    "bottom": highs[i],
                })

    return fvgs


def get_price_action_summary(df: pd.DataFrame) -> str:
    """Generate a comprehensive price action summary for the agent."""
    trend = detect_trend(df)
    sr = find_support_resistance(df)
    fvgs = detect_fair_value_gaps(df)

    current_price = df["close"].iloc[-1]

    lines = [
        f"Current Price: {current_price:.2f}",
        f"Trend: {trend['trend']} (strength: {trend['strength']}) — {trend['details']}",
        "",
        f"Support Levels: {[round(s, 2) for s in sr['support'][-5:]]}",
        f"Resistance Levels: {[round(r, 2) for r in sr['resistance'][-5:]]}",
    ]

    # Nearest S/R
    supports_below = [s for s in sr["support"] if s < current_price]
    resistance_above = [r for r in sr["resistance"] if r > current_price]
    if supports_below:
        lines.append(f"Nearest Support: {supports_below[-1]:.2f}")
    if resistance_above:
        lines.append(f"Nearest Resistance: {resistance_above[0]:.2f}")

    # Recent FVGs
    recent_fvgs = fvgs[-5:] if fvgs else []
    if recent_fvgs:
        lines.append("")
        lines.append("Recent Fair Value Gaps:")
        for fvg in recent_fvgs:
            lines.append(f"  {fvg['type']} at bar {fvg['bar']}: {fvg['bottom']:.2f} — {fvg['top']:.2f}")

    return "\n".join(lines)
