"""Time-of-day and session filtering for entries.

A SessionFilter is a callable: `(timestamp) -> bool` where True means
"entries are allowed at this bar". The engine consults it before opening
a new position; exits are NEVER blocked by the filter (always allow flat).
"""

from datetime import time, datetime, timedelta
from typing import Callable, Iterable

import pandas as pd

from config.xauusd import SESSIONS, is_in_session


SessionFilter = Callable[[pd.Timestamp], bool]


def always_open() -> SessionFilter:
    """No filter — 24/7 trading allowed."""
    return lambda ts: True


def by_sessions(names: Iterable[str]) -> SessionFilter:
    """Allow entries only during the named sessions (from config.xauusd.SESSIONS).

    Example: by_sessions(["LONDON_OPEN", "NY_AM"])
    """
    names = list(names)
    for n in names:
        if n not in SESSIONS:
            raise ValueError(f"Unknown session '{n}'. Available: {list(SESSIONS)}")

    def _filter(ts: pd.Timestamp) -> bool:
        return any(is_in_session(ts, n) for n in names)

    return _filter


def by_time_window(start: time, end: time) -> SessionFilter:
    """Allow entries within a custom UTC time window (handles midnight wrap)."""

    def _filter(ts: pd.Timestamp) -> bool:
        t = ts.time()
        if start <= end:
            return start <= t <= end
        return t >= start or t <= end

    return _filter


def exclude_news_windows(
    releases: list[dict],
    reference_date: datetime | None = None,
) -> SessionFilter:
    """Block entries in ±minutes around scheduled news releases.

    releases: list of dicts with 'time' (datetime.time), 'pre_min', 'post_min'.
    This v1 treats releases as happening every day at the given UTC time.
    For real use, combine with a separate date-specific schedule.
    """

    def _filter(ts: pd.Timestamp) -> bool:
        current = ts.time() if hasattr(ts, "time") else ts
        current_dt = datetime.combine(ts.date() if hasattr(ts, "date") else datetime.today().date(), current)
        for rel in releases:
            rel_dt = datetime.combine(current_dt.date(), rel["time"])
            blackout_start = rel_dt - timedelta(minutes=rel.get("pre_min", 5))
            blackout_end = rel_dt + timedelta(minutes=rel.get("post_min", 30))
            if blackout_start <= current_dt <= blackout_end:
                return False
        return True

    return _filter


def combine_all(*filters: SessionFilter) -> SessionFilter:
    """Logical AND of multiple filters — entry allowed only if all pass."""
    filters = tuple(f for f in filters if f is not None)
    if not filters:
        return always_open()
    return lambda ts: all(f(ts) for f in filters)
