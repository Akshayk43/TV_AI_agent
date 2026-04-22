"""Walk-forward out-of-sample validation harness.

A single in-sample backtest tells you almost nothing about real-world performance
on minute-bar data — parameter search overfits in 1-2 iterations. Walk-forward
splits history into rolling (train, test) windows and evaluates the strategy
on the OOS segment of each window.

Default schedule (configurable):
  train_days = 60   → enough bars to stabilize indicators
  test_days  = 10   → ~2 trading weeks OOS per fold
  step_days  = 10   → non-overlapping test windows

For a 365-day history that gives ~30 folds.

Output: list[FoldResult] + aggregate OOS equity curve stitched from test
segments only. The aggregate equity curve is what honestly answers "would
this strategy have made money?"
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from backtesting.metrics import compute_metrics
from backtesting.cost_model import CostModel


@dataclass
class FoldResult:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    in_sample: dict = field(default_factory=dict)   # metrics on train
    out_sample: dict = field(default_factory=dict)  # metrics on test
    trades: list = field(default_factory=list)       # OOS trades only

    def summary(self) -> str:
        is_m = self.in_sample
        oos = self.out_sample
        return (
            f"Fold #{self.fold_index:>2}: "
            f"test={self.test_start.date()}→{self.test_end.date()} "
            f"IS Sharpe={is_m.get('sharpe_ratio', 0):.2f} "
            f"OOS Sharpe={oos.get('sharpe_ratio', 0):.2f} "
            f"OOS trades={oos.get('total_trades', 0)} "
            f"OOS PF={oos.get('profit_factor', 0):.2f} "
            f"OOS ret={oos.get('total_return_pct', 0):.2f}%"
        )


def walk_forward(
    df: pd.DataFrame,
    strategy: dict,
    train_days: int = 60,
    test_days: int = 10,
    step_days: int = 10,
    initial_capital: float = 100_000.0,
    cost_model: Optional[CostModel] = None,
    session_filter: Optional[Callable[[pd.Timestamp], bool]] = None,
    refit: Optional[Callable[[dict, pd.DataFrame], dict]] = None,
) -> dict:
    """Run rolling walk-forward validation.

    Args:
        df: Full OHLCV DataFrame with datetime index.
        strategy: Strategy dict (passed as-is unless `refit` is provided).
        train_days, test_days, step_days: Window geometry in days.
        initial_capital: Starting equity per fold (keeps folds comparable).
        cost_model: Optional cost model passed to BacktestEngine.
        session_filter: Optional entry-time filter passed to BacktestEngine.
        refit: Optional function(strategy, train_df) → tuned strategy. If provided,
               it's called once per fold on the training window and the returned
               strategy is used for the OOS test. If None, the same strategy is
               used in both segments (still useful — measures out-of-time stability).

    Returns dict:
        {
          "folds": [FoldResult...],
          "oos_metrics": aggregate metrics across all OOS trades,
          "oos_equity": stitched OOS equity curve,
          "is_oos_gap": {sharpe, pf, win_rate} mean absolute gap (overfit diagnostic),
          "n_folds": int,
        }
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("walk_forward requires a DatetimeIndex on df")

    engine = BacktestEngine(
        initial_capital=initial_capital,
        cost_model=cost_model,
        session_filter=session_filter,
    )

    start = df.index[0]
    end = df.index[-1]
    train_td = pd.Timedelta(days=train_days)
    test_td = pd.Timedelta(days=test_days)
    step_td = pd.Timedelta(days=step_days)

    folds: list[FoldResult] = []
    all_oos_trades: list = []
    oos_equity = [initial_capital]
    gaps = {"sharpe_ratio": [], "profit_factor": [], "win_rate": []}

    fold_idx = 0
    cursor = start

    while cursor + train_td + test_td <= end:
        train_start = cursor
        train_end = cursor + train_td
        test_start = train_end
        test_end = min(test_start + test_td, end)

        train_df = df.loc[train_start:train_end]
        test_df = df.loc[test_start:test_end]

        if len(train_df) < 50 or len(test_df) < 10:
            cursor += step_td
            continue

        fold_strategy = refit(strategy, train_df) if refit else strategy

        # In-sample eval (for overfit diagnostic — we don't report IS as success)
        is_results = engine.run(train_df, fold_strategy)

        # OOS eval — this is what actually counts
        oos_results = engine.run(test_df, fold_strategy)

        fold = FoldResult(
            fold_index=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            in_sample=is_results["metrics"],
            out_sample=oos_results["metrics"],
            trades=oos_results["trades"],
        )
        folds.append(fold)
        all_oos_trades.extend(oos_results["trades"])

        # Stitch OOS equity segment (reset to the running total, not the fold's initial)
        for trade in oos_results["trades"]:
            oos_equity.append(oos_equity[-1] + trade.get("pnl", 0))

        # Overfit gap — positive = worse OOS than IS
        for k in gaps:
            if is_results["metrics"].get("total_trades", 0) > 0 and oos_results["metrics"].get("total_trades", 0) > 0:
                gaps[k].append(is_results["metrics"].get(k, 0) - oos_results["metrics"].get(k, 0))

        fold_idx += 1
        cursor += step_td

    oos_aggregate = compute_metrics(all_oos_trades, initial_capital=initial_capital)
    is_oos_gap = {k: float(np.mean(v)) if v else 0.0 for k, v in gaps.items()}

    return {
        "folds": folds,
        "oos_metrics": oos_aggregate,
        "oos_equity": oos_equity,
        "is_oos_gap": is_oos_gap,
        "n_folds": len(folds),
    }


def format_walk_forward_report(wf: dict) -> str:
    """Human-readable summary of a walk-forward run."""
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        "║           WALK-FORWARD OOS VALIDATION                ║",
        "╠══════════════════════════════════════════════════════╣",
        f"║ Folds: {wf['n_folds']:<45}║",
        "╠══════════════════════════════════════════════════════╣",
        "║ Per-fold OOS:                                        ║",
    ]
    for f in wf["folds"]:
        lines.append(f"║ {f.summary()[:52]:<52} ║")

    lines.append("╠══════════════════════════════════════════════════════╣")
    lines.append("║ Aggregate OOS (stitched across all folds):           ║")
    m = wf["oos_metrics"]
    lines.append(f"║   Trades:      {m.get('total_trades', 0):<37}  ║")
    lines.append(f"║   Win rate:    {m.get('win_rate', 0):<37.2%}  ║")
    lines.append(f"║   Sharpe:      {m.get('sharpe_ratio', 0):<37.2f}  ║")
    lines.append(f"║   Profit fac:  {m.get('profit_factor', 0):<37.2f}  ║")
    lines.append(f"║   Max DD:      {m.get('max_drawdown_pct', 0):<36.2f}%  ║")
    lines.append(f"║   Total ret:   {m.get('total_return_pct', 0):<36.2f}%  ║")

    lines.append("╠══════════════════════════════════════════════════════╣")
    lines.append("║ Overfit diagnostic (IS−OOS; lower = more stable):    ║")
    g = wf["is_oos_gap"]
    lines.append(f"║   Sharpe gap:   {g.get('sharpe_ratio', 0):<36.2f}  ║")
    lines.append(f"║   PF gap:       {g.get('profit_factor', 0):<36.2f}  ║")
    lines.append(f"║   Win rate gap: {g.get('win_rate', 0):<36.2%}  ║")
    lines.append("╚══════════════════════════════════════════════════════╝")
    return "\n".join(lines)
