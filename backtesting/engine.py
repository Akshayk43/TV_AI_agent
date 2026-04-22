"""Backtesting engine — executes strategy rules on historical OHLCV data."""

import json
from typing import Callable, Optional

import numpy as np
import pandas as pd

from knowledge.indicators import add_all_indicators
from backtesting.metrics import compute_metrics
from backtesting.cost_model import CostModel, PercentCost
from config.settings import INITIAL_CAPITAL, COMMISSION_PCT, SLIPPAGE_PCT


class BacktestEngine:
    """Event-driven backtester that processes strategy rules bar-by-bar.

    A strategy is defined as a JSON dict with:
    {
        "name": "Strategy Name",
        "description": "...",
        "timeframe": "1d",
        "rules": {
            "long_entry": [{"indicator": "rsi_14", "operator": "<", "value": 30}, ...],
            "long_exit":  [{"indicator": "rsi_14", "operator": ">", "value": 70}, ...],
            "short_entry": [...],  # optional
            "short_exit":  [...],  # optional
        },
        "risk_management": {
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "position_size_pct": 10.0,  # % of equity per trade
            "max_positions": 1,
        }
    }
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        commission_pct: float = COMMISSION_PCT,
        slippage_pct: float = SLIPPAGE_PCT,
        cost_model: Optional[CostModel] = None,
        session_filter: Optional[Callable[[pd.Timestamp], bool]] = None,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        # If no explicit cost_model is passed, mirror the legacy percent-based behavior
        # using the caller's commission_pct / slippage_pct values.
        self.cost_model = cost_model or PercentCost(
            commission_pct=commission_pct, slippage_pct=slippage_pct
        )
        self.session_filter = session_filter

    def run(self, df: pd.DataFrame, strategy: dict) -> dict:
        """Run the backtest.

        Args:
            df: OHLCV DataFrame.
            strategy: Strategy definition dict.

        Returns:
            Dict with 'trades', 'metrics', 'equity_curve', 'strategy'.
        """
        df = add_all_indicators(df)
        rules = strategy.get("rules", {})
        risk = strategy.get("risk_management", {})

        stop_loss_pct = risk.get("stop_loss_pct", 2.0) / 100
        take_profit_pct = risk.get("take_profit_pct", 4.0) / 100
        position_size_pct = risk.get("position_size_pct", 10.0) / 100
        max_positions = risk.get("max_positions", 1)

        equity = self.initial_capital
        trades: list[dict] = []
        open_positions: list[dict] = []
        equity_curve = [equity]

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            date = df.index[i]

            # ── Check exits on open positions ─────────────────────────────
            closed = []
            for pos in open_positions:
                exit_price = None
                exit_reason = None

                # Stop loss / take profit
                if pos["direction"] == "long":
                    if row["low"] <= pos["stop_loss"]:
                        exit_price = pos["stop_loss"]
                        exit_reason = "stop_loss"
                    elif row["high"] >= pos["take_profit"]:
                        exit_price = pos["take_profit"]
                        exit_reason = "take_profit"
                    elif self._check_conditions(rules.get("long_exit", []), df, i):
                        exit_price = row["close"]
                        exit_reason = "signal_exit"
                else:  # short
                    if row["high"] >= pos["stop_loss"]:
                        exit_price = pos["stop_loss"]
                        exit_reason = "stop_loss"
                    elif row["low"] <= pos["take_profit"]:
                        exit_price = pos["take_profit"]
                        exit_reason = "take_profit"
                    elif self._check_conditions(rules.get("short_exit", []), df, i):
                        exit_price = row["close"]
                        exit_reason = "signal_exit"

                if exit_price is not None:
                    # Apply cost model (spread + commission) via the configured model
                    exit_price = self.cost_model.exit_price(exit_price, pos["direction"])

                    # Compute PnL
                    if pos["direction"] == "long":
                        pnl = (exit_price - pos["entry_price"]) * pos["size"]
                    else:
                        pnl = (pos["entry_price"] - exit_price) * pos["size"]

                    commission = self.cost_model.commission(exit_price, pos["size"])
                    pnl -= commission
                    equity += pnl

                    trade = {
                        "direction": pos["direction"],
                        "entry_price": pos["entry_price"],
                        "exit_price": round(exit_price, 4),
                        "size": pos["size"],
                        "entry_bar": pos["entry_bar"],
                        "exit_bar": i,
                        "entry_date": str(pos["entry_date"]),
                        "exit_date": str(date),
                        "pnl": round(pnl, 2),
                        "exit_reason": exit_reason,
                    }
                    trades.append(trade)
                    closed.append(pos)

            for pos in closed:
                open_positions.remove(pos)

            # ── Check entries ─────────────────────────────────────────────
            # Session filter only gates new entries; it never blocks exits.
            entries_allowed = (
                len(open_positions) < max_positions
                and (self.session_filter is None or self.session_filter(date))
            )
            if entries_allowed:
                # Long entry
                if rules.get("long_entry") and self._check_conditions(rules["long_entry"], df, i):
                    entry_price = self.cost_model.entry_price(row["close"], "long")
                    size_value = equity * position_size_pct
                    size = size_value / entry_price
                    commission = self.cost_model.commission(entry_price, size)
                    equity -= commission

                    open_positions.append({
                        "direction": "long",
                        "entry_price": round(entry_price, 4),
                        "size": round(size, 6),
                        "entry_bar": i,
                        "entry_date": date,
                        "stop_loss": round(entry_price * (1 - stop_loss_pct), 4),
                        "take_profit": round(entry_price * (1 + take_profit_pct), 4),
                    })

                # Short entry
                elif rules.get("short_entry") and self._check_conditions(rules["short_entry"], df, i):
                    entry_price = self.cost_model.entry_price(row["close"], "short")
                    size_value = equity * position_size_pct
                    size = size_value / entry_price
                    commission = self.cost_model.commission(entry_price, size)
                    equity -= commission

                    open_positions.append({
                        "direction": "short",
                        "entry_price": round(entry_price, 4),
                        "size": round(size, 6),
                        "entry_bar": i,
                        "entry_date": date,
                        "stop_loss": round(entry_price * (1 + stop_loss_pct), 4),
                        "take_profit": round(entry_price * (1 - take_profit_pct), 4),
                    })

            equity_curve.append(equity)

        # Close remaining positions at last bar
        last_row = df.iloc[-1]
        for pos in open_positions:
            if pos["direction"] == "long":
                pnl = (last_row["close"] - pos["entry_price"]) * pos["size"]
            else:
                pnl = (pos["entry_price"] - last_row["close"]) * pos["size"]
            equity += pnl
            trades.append({
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "exit_price": round(last_row["close"], 4),
                "size": pos["size"],
                "entry_bar": pos["entry_bar"],
                "exit_bar": len(df) - 1,
                "entry_date": str(pos["entry_date"]),
                "exit_date": str(df.index[-1]),
                "pnl": round(pnl, 2),
                "exit_reason": "end_of_data",
            })

        metrics = compute_metrics(trades, self.initial_capital)

        return {
            "strategy": strategy,
            "trades": trades,
            "metrics": metrics,
            "equity_curve": equity_curve,
            "data_range": f"{df.index[0]} to {df.index[-1]}",
            "total_bars": len(df),
        }

    def _check_conditions(self, conditions: list[dict], df: pd.DataFrame, bar_idx: int) -> bool:
        """Evaluate a list of AND conditions at a specific bar.

        Each condition: {"indicator": "rsi_14", "operator": "<", "value": 30}
        Supports cross-references: {"indicator": "close", "operator": ">", "ref": "sma_50"}
        """
        if not conditions:
            return False

        for cond in conditions:
            indicator = cond.get("indicator", "")
            operator = cond.get("operator", "")

            # Get the indicator value
            if indicator in df.columns:
                val = df[indicator].iloc[bar_idx]
            elif indicator == "close":
                val = df["close"].iloc[bar_idx]
            elif indicator == "open":
                val = df["open"].iloc[bar_idx]
            elif indicator == "high":
                val = df["high"].iloc[bar_idx]
            elif indicator == "low":
                val = df["low"].iloc[bar_idx]
            elif indicator == "volume":
                val = df["volume"].iloc[bar_idx]
            else:
                return False

            if pd.isna(val):
                return False

            # Get comparison value
            if "ref" in cond:
                ref = cond["ref"]
                if ref in df.columns:
                    compare_val = df[ref].iloc[bar_idx]
                else:
                    return False
                if pd.isna(compare_val):
                    return False
            elif "value" in cond:
                compare_val = cond["value"]
            else:
                return False

            # Compare
            if operator == ">":
                if not (val > compare_val):
                    return False
            elif operator == "<":
                if not (val < compare_val):
                    return False
            elif operator == ">=":
                if not (val >= compare_val):
                    return False
            elif operator == "<=":
                if not (val <= compare_val):
                    return False
            elif operator == "==":
                if not (val == compare_val):
                    return False
            elif operator == "crosses_above":
                if bar_idx < 1:
                    return False
                prev_val = df[indicator].iloc[bar_idx - 1]
                if "ref" in cond:
                    prev_compare = df[cond["ref"]].iloc[bar_idx - 1]
                else:
                    prev_compare = compare_val
                if not (prev_val <= prev_compare and val > compare_val):
                    return False
            elif operator == "crosses_below":
                if bar_idx < 1:
                    return False
                prev_val = df[indicator].iloc[bar_idx - 1]
                if "ref" in cond:
                    prev_compare = df[cond["ref"]].iloc[bar_idx - 1]
                else:
                    prev_compare = compare_val
                if not (prev_val >= prev_compare and val < compare_val):
                    return False
            else:
                return False

        return True


def run_backtest(
    df: pd.DataFrame,
    strategy: dict,
    cost_model: Optional[CostModel] = None,
    session_filter: Optional[Callable[[pd.Timestamp], bool]] = None,
) -> dict:
    """Convenience function to run a backtest.

    Pass an optional cost_model (e.g. XAUUSDSpreadCost) and session_filter
    for instrument-accurate backtests. Defaults preserve legacy behavior.
    """
    engine = BacktestEngine(cost_model=cost_model, session_filter=session_filter)
    return engine.run(df, strategy)
