"""Volume Profile analysis — distribution of volume across price levels."""

import numpy as np
import pandas as pd


VOLUME_PROFILE_KNOWLEDGE = {
    "description": (
        "Volume Profile plots the total volume traded at each price level over a period. "
        "Unlike time-based volume bars, it shows WHERE the most trading occurred."
    ),
    "key_concepts": {
        "POC": {
            "name": "Point of Control",
            "description": "The price level with the highest traded volume. Acts as a magnet for price.",
            "trading": "Price tends to gravitate towards POC; strong S/R level.",
        },
        "VAH": {
            "name": "Value Area High",
            "description": "Upper boundary of the value area (70% of volume).",
            "trading": "Resistance level; breakout above is bullish.",
        },
        "VAL": {
            "name": "Value Area Low",
            "description": "Lower boundary of the value area (70% of volume).",
            "trading": "Support level; breakdown below is bearish.",
        },
        "HVN": {
            "name": "High Volume Node",
            "description": "Price levels with significantly above-average volume.",
            "trading": "Act as support/resistance; price consolidates around HVNs.",
        },
        "LVN": {
            "name": "Low Volume Node",
            "description": "Price levels with significantly below-average volume.",
            "trading": "Price moves quickly through LVNs; potential breakout/breakdown zones.",
        },
    },
    "strategies": {
        "value_area_rotation": (
            "If price opens inside the value area, expect rotation between VAH and VAL. "
            "Trade mean-reversion between these levels."
        ),
        "value_area_breakout": (
            "If price opens outside the value area, look for acceptance (multiple closes) "
            "above VAH (bullish) or below VAL (bearish)."
        ),
        "poc_rejection": (
            "If price approaches POC and shows rejection (wicks, reversal candles), "
            "trade the bounce. POC is the fairest price."
        ),
        "naked_poc": (
            "A POC from a prior session that hasn't been revisited. Acts as a magnet — "
            "price tends to seek these levels."
        ),
    },
}


def compute_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
    value_area_pct: float = 0.70,
) -> dict:
    """Compute volume profile from OHLCV data.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns.
        num_bins: Number of price bins.
        value_area_pct: Percentage of total volume for the value area (default 70%).

    Returns:
        Dict with 'poc', 'vah', 'val', 'hvn', 'lvn', 'profile' (price→volume mapping).
    """
    price_low = df["low"].min()
    price_high = df["high"].max()

    if price_high == price_low:
        mid = price_low
        return {
            "poc": mid,
            "vah": mid,
            "val": mid,
            "hvn": [mid],
            "lvn": [],
            "profile": {mid: df["volume"].sum()},
        }

    bin_edges = np.linspace(price_low, price_high, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_volumes = np.zeros(num_bins)

    # Distribute each bar's volume across the price bins it spans
    for _, row in df.iterrows():
        bar_low, bar_high, vol = row["low"], row["high"], row["volume"]
        if bar_high == bar_low:
            idx = np.searchsorted(bin_edges, bar_low, side="right") - 1
            idx = min(max(idx, 0), num_bins - 1)
            bin_volumes[idx] += vol
        else:
            for j in range(num_bins):
                overlap_low = max(bin_edges[j], bar_low)
                overlap_high = min(bin_edges[j + 1], bar_high)
                if overlap_high > overlap_low:
                    fraction = (overlap_high - overlap_low) / (bar_high - bar_low)
                    bin_volumes[j] += vol * fraction

    # POC — bin with highest volume
    poc_idx = int(np.argmax(bin_volumes))
    poc = float(bin_centers[poc_idx])

    # Value Area — expand from POC until we capture value_area_pct of total volume
    total_vol = bin_volumes.sum()
    if total_vol == 0:
        return {
            "poc": poc,
            "vah": float(bin_centers[-1]),
            "val": float(bin_centers[0]),
            "hvn": [],
            "lvn": [],
            "profile": dict(zip(bin_centers.tolist(), bin_volumes.tolist())),
        }

    va_vol = bin_volumes[poc_idx]
    low_idx, high_idx = poc_idx, poc_idx

    while va_vol / total_vol < value_area_pct:
        add_low = bin_volumes[low_idx - 1] if low_idx > 0 else 0
        add_high = bin_volumes[high_idx + 1] if high_idx < num_bins - 1 else 0

        if low_idx <= 0 and high_idx >= num_bins - 1:
            break

        if add_high >= add_low and high_idx < num_bins - 1:
            high_idx += 1
            va_vol += bin_volumes[high_idx]
        elif low_idx > 0:
            low_idx -= 1
            va_vol += bin_volumes[low_idx]
        elif high_idx < num_bins - 1:
            high_idx += 1
            va_vol += bin_volumes[high_idx]
        else:
            break

    vah = float(bin_centers[high_idx])
    val = float(bin_centers[low_idx])

    # HVN / LVN — bins significantly above/below average
    avg_vol = total_vol / num_bins
    hvn_threshold = avg_vol * 1.5
    lvn_threshold = avg_vol * 0.5

    hvn = [float(bin_centers[i]) for i in range(num_bins) if bin_volumes[i] > hvn_threshold]
    lvn = [float(bin_centers[i]) for i in range(num_bins)
           if 0 < bin_volumes[i] < lvn_threshold]

    profile = dict(zip([round(p, 4) for p in bin_centers.tolist()], [round(v, 2) for v in bin_volumes.tolist()]))

    return {
        "poc": round(poc, 4),
        "vah": round(vah, 4),
        "val": round(val, 4),
        "hvn": [round(h, 4) for h in hvn],
        "lvn": [round(l, 4) for l in lvn],
        "profile": profile,
    }


def get_volume_profile_summary(df: pd.DataFrame) -> str:
    """Generate a human-readable volume profile summary."""
    vp = compute_volume_profile(df)
    current_price = df["close"].iloc[-1]

    # Determine position relative to value area
    if current_price > vp["vah"]:
        position = "ABOVE value area (bullish breakout zone)"
    elif current_price < vp["val"]:
        position = "BELOW value area (bearish breakdown zone)"
    else:
        position = "INSIDE value area (mean-reversion zone)"

    lines = [
        "=== Volume Profile Analysis ===",
        f"Current Price: {current_price:.2f} — {position}",
        f"Point of Control (POC): {vp['poc']:.2f}",
        f"Value Area High (VAH): {vp['vah']:.2f}",
        f"Value Area Low (VAL): {vp['val']:.2f}",
        f"High Volume Nodes: {[round(h, 2) for h in vp['hvn'][:8]]}",
        f"Low Volume Nodes: {[round(l, 2) for l in vp['lvn'][:8]]}",
    ]

    # Distance metrics
    dist_poc = ((current_price - vp["poc"]) / vp["poc"]) * 100
    lines.append(f"Distance to POC: {dist_poc:+.2f}%")

    return "\n".join(lines)
