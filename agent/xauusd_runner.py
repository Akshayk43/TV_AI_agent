"""XAUUSD intraday workflow runner.

Ties together: OANDA data loader → XAUUSDSpreadCost → session filter → walk-forward
harness → ranking of seed strategies. Optional TV cross-check via the MCP if
mode='tradingview'.

This is intentionally separate from OODAAgent so we don't entangle the XAUUSD
flow with the general-purpose vision/Pine pipeline.

Usage:
    from agent.xauusd_runner import run_xauusd_cycle
    result = run_xauusd_cycle(timeframe="5m", period_days=90)
    print(result["report"])
"""

import glob
import json
import os
from typing import Optional

import pandas as pd

from config.settings import (
    INITIAL_CAPITAL,
    XAUUSD_DEFAULT_SPREAD,
    XAUUSD_STRESS_SPREAD,
    XAUUSD_COMMISSION_PER_OZ,
    WF_TRAIN_DAYS,
    WF_TEST_DAYS,
    WF_STEP_DAYS,
)
from backtesting.cost_model import XAUUSDSpreadCost
from backtesting.data_loader import get_market_data
from backtesting.session_filter import by_sessions, always_open
from backtesting.walk_forward import walk_forward, format_walk_forward_report


SEEDS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "strategies",
    "seeds",
)


def load_seed_strategies(directory: str = SEEDS_DIR) -> list[dict]:
    """Load all xauusd_*.json seed files."""
    files = sorted(glob.glob(os.path.join(directory, "xauusd_*.json")))
    out = []
    for f in files:
        with open(f, "r") as fh:
            out.append(json.load(fh))
    return out


def run_xauusd_cycle(
    timeframe: str = "5m",
    period_days: int = 90,
    spread: float = XAUUSD_DEFAULT_SPREAD,
    commission_per_oz: float = XAUUSD_COMMISSION_PER_OZ,
    train_days: int = WF_TRAIN_DAYS,
    test_days: int = WF_TEST_DAYS,
    step_days: int = WF_STEP_DAYS,
    strategies: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
    use_session_filter: bool = True,
) -> dict:
    """Run walk-forward on all XAUUSD seed strategies, return ranked results.

    Args:
        timeframe: '1m' or '5m' (the 1m seed uses 1m, others use 5m).
        period_days: History lookback.
        spread: Typical per-oz spread used in cost model.
        commission_per_oz: Optional ECN-style commission.
        strategies: Override seed list.
        df: Optional pre-loaded DataFrame (skips data fetch).
        use_session_filter: If True, restrict entries to each strategy's
            `session_config.allowed_sessions`.

    Returns dict with: 'ranked' (list of {strategy, wf}), 'best', 'report'.
    """
    strategies = strategies or load_seed_strategies()
    if not strategies:
        raise RuntimeError(f"No seed strategies found under {SEEDS_DIR}")

    cost_model = XAUUSDSpreadCost(spread=spread, commission_per_oz=commission_per_oz)

    data_cache: dict[str, pd.DataFrame] = {}
    ranked: list[dict] = []

    for strat in strategies:
        tf = strat.get("timeframe", timeframe)

        if df is not None and strat is strategies[0]:
            # Explicit df passes through for one timeframe
            data = df
        elif tf not in data_cache:
            data_cache[tf] = get_market_data("XAUUSD", period_days=period_days, interval=tf)
            data = data_cache[tf]
        else:
            data = data_cache[tf]

        session_filter = None
        if use_session_filter:
            allowed = strat.get("session_config", {}).get("allowed_sessions")
            if allowed:
                session_filter = by_sessions(allowed)

        wf = walk_forward(
            data,
            strat,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            initial_capital=INITIAL_CAPITAL,
            cost_model=cost_model,
            session_filter=session_filter,
        )

        ranked.append({
            "strategy": strat,
            "wf": wf,
            "oos_sharpe": wf["oos_metrics"].get("sharpe_ratio", 0),
            "oos_pf": wf["oos_metrics"].get("profit_factor", 0),
            "oos_trades": wf["oos_metrics"].get("total_trades", 0),
        })

    ranked.sort(key=lambda r: r["oos_sharpe"], reverse=True)

    report_lines = [
        f"\n=== XAUUSD Walk-Forward Report ===",
        f"Timeframe: {timeframe}   Period: {period_days}d   "
        f"Spread: ${spread}/oz   Commission: ${commission_per_oz}/oz",
        "",
    ]
    for i, r in enumerate(ranked, 1):
        report_lines.append(f"[{i}] {r['strategy']['name']}")
        report_lines.append(format_walk_forward_report(r["wf"]))
        report_lines.append("")

    return {
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
        "report": "\n".join(report_lines),
    }


def run_stress_test(base_result: dict, stress_spread: float = XAUUSD_STRESS_SPREAD) -> dict:
    """Re-run the best strategy with a wider spread to see if edge survives.

    Uses the dataframe already cached inside `base_result` is not feasible
    because that isn't stored — caller should re-invoke run_xauusd_cycle
    with spread=stress_spread and strategies=[base['strategy']].
    """
    if not base_result.get("best"):
        raise ValueError("No best strategy in base_result")
    best = base_result["best"]["strategy"]
    return run_xauusd_cycle(
        timeframe=best.get("timeframe", "5m"),
        period_days=90,
        spread=stress_spread,
        strategies=[best],
    )
