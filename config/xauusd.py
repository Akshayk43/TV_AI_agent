"""XAUUSD (spot gold) instrument-specific configuration.

Retail spot gold is quoted in USD per troy ounce. Typical conventions:
- Tick size: $0.01 per oz (some brokers: $0.10)
- No percentage commission — cost is embedded in the bid/ask spread
- Typical retail spread: $0.20 - $0.50 per oz (Dukascopy, OANDA, IC Markets)
- Pip value: $0.10 per oz per pip (where 1 pip = $0.01)

These constants are used by backtesting.cost_model.XAUUSDSpreadCost and by
the session/news filters.
"""

from datetime import time


# ── Instrument microstructure ────────────────────────────────────────────
TICK_SIZE = 0.01        # USD per oz
TICK_VALUE = 0.01       # USD per oz per tick (1:1 for spot gold)
MIN_SPREAD = 0.15       # Best-case ECN spread in USD per oz
TYPICAL_SPREAD = 0.30   # Default retail spread used in backtests
STRESS_SPREAD = 0.60    # 2× typical, used to test if edge is spread-captive
LOT_SIZE = 100.0        # Standard lot = 100 oz (MT4/MT5 convention)


# ── Session windows (UTC) ────────────────────────────────────────────────
# These drive 80%+ of intraday XAUUSD moves. Entries outside these windows
# tend to be noisy and spread-unfavorable.
SESSIONS = {
    # London cash session open — liquidity injection, frequent breakouts
    "LONDON_OPEN": (time(7, 0), time(10, 0)),
    # NY AM — peak liquidity, US data releases land here
    "NY_AM": (time(13, 30), time(16, 0)),
    # London PM fix — 15:00 UK time, fixing-driven order flow
    "LONDON_FIX": (time(14, 45), time(15, 15)),
    # Asia session — generally avoid for breakout strategies (choppy)
    "ASIA": (time(23, 0), time(6, 0)),
}


# ── News blackout windows (UTC, approximate) ─────────────────────────────
# Static high-impact US data release schedule. For production use,
# replace with a ForexFactory scraper or econ calendar API.
# Windows: 5m before release to 30m after — avoid entries during this.
NEWS_RELEASES_UTC = [
    # US NFP: 1st Friday of month, 13:30 UTC
    {"name": "NFP", "time": time(13, 30), "day": "first_friday", "pre_min": 5, "post_min": 30},
    # US CPI: monthly ~mid-month, 13:30 UTC
    {"name": "CPI", "time": time(13, 30), "day": "monthly_cpi", "pre_min": 5, "post_min": 30},
    # FOMC: 8x/year, 19:00 UTC
    {"name": "FOMC", "time": time(19, 0), "day": "fomc_day", "pre_min": 10, "post_min": 60},
]


# ── Cost and sizing defaults for XAUUSD ──────────────────────────────────
# At spot ~$2300/oz, $100k account, 2% risk per trade, 1×ATR stop (~$5/oz)
# → position size ~400 oz ($920k notional). That's 4 mini-lots or 40 micro.
DEFAULT_POSITION_SIZE_PCT = 2.0      # Risk 2% of equity per trade
DEFAULT_STOP_ATR_MULT = 1.0
DEFAULT_TAKE_ATR_MULT = 2.0


def is_in_session(ts, session_name: str) -> bool:
    """Check if a timestamp falls within a named session window.

    Handles wrap-around (e.g. ASIA 23:00 → 06:00).
    """
    if session_name not in SESSIONS:
        raise ValueError(f"Unknown session: {session_name}. Available: {list(SESSIONS)}")

    start, end = SESSIONS[session_name]
    t = ts.time() if hasattr(ts, "time") else ts

    if start <= end:
        return start <= t <= end
    # Wraps midnight
    return t >= start or t <= end
