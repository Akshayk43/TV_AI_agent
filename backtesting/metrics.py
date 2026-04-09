"""Performance metrics for backtesting results."""

import numpy as np
import pandas as pd


def compute_metrics(trades: list[dict], initial_capital: float = 100_000.0) -> dict:
    """Compute comprehensive performance metrics from a list of trades.

    Each trade dict must have: 'entry_price', 'exit_price', 'direction' ('long'/'short'),
    'size', 'entry_bar', 'exit_bar', 'entry_date', 'exit_date'.

    Returns:
        Dict of performance metrics.
    """
    if not trades:
        return _empty_metrics()

    pnls = []
    for t in trades:
        if t["direction"] == "long":
            pnl = (t["exit_price"] - t["entry_price"]) * t["size"]
        else:
            pnl = (t["entry_price"] - t["exit_price"]) * t["size"]
        pnls.append(pnl)

    pnls = np.array(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    total_pnl = pnls.sum()
    total_return_pct = (total_pnl / initial_capital) * 100

    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Risk/reward
    avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Equity curve for drawdown and Sharpe
    equity = [initial_capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)

    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_drawdown = drawdown.max()
    max_drawdown_pct = max_drawdown * 100

    # Sharpe ratio (annualized, assuming daily trades)
    if len(pnls) > 1 and pnls.std() > 0:
        sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino ratio
    downside = pnls[pnls < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (pnls.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    # Consecutive wins/losses
    max_consec_wins, max_consec_losses = 0, 0
    current_wins, current_losses = 0, 0
    for p in pnls:
        if p > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    return {
        "total_trades": len(pnls),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_risk_reward": round(avg_rr, 2),
        "profit_factor": round(profit_factor, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "expectancy": round(expectancy, 2),
        "initial_capital": initial_capital,
        "final_equity": round(equity[-1], 2),
        "equity_curve": equity.tolist(),
    }


def _empty_metrics() -> dict:
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "total_return_pct": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "avg_risk_reward": 0.0,
        "profit_factor": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "expectancy": 0.0,
        "initial_capital": 100_000.0,
        "final_equity": 100_000.0,
        "equity_curve": [100_000.0],
    }


def format_metrics_report(metrics: dict) -> str:
    """Format metrics into a readable report string."""
    return (
        "╔══════════════════════════════════════════╗\n"
        "║       BACKTEST PERFORMANCE REPORT        ║\n"
        "╠══════════════════════════════════════════╣\n"
        f"║ Total Trades:         {metrics['total_trades']:>16}  ║\n"
        f"║ Winning Trades:       {metrics['winning_trades']:>16}  ║\n"
        f"║ Losing Trades:        {metrics['losing_trades']:>16}  ║\n"
        f"║ Win Rate:             {metrics['win_rate']:>15.1%}  ║\n"
        "╠══════════════════════════════════════════╣\n"
        f"║ Total P&L:           ${metrics['total_pnl']:>15,.2f}  ║\n"
        f"║ Total Return:         {metrics['total_return_pct']:>15.2f}%  ║\n"
        f"║ Profit Factor:        {metrics['profit_factor']:>16.2f}  ║\n"
        f"║ Expectancy:          ${metrics['expectancy']:>15,.2f}  ║\n"
        "╠══════════════════════════════════════════╣\n"
        f"║ Avg Win:             ${metrics['avg_win']:>15,.2f}  ║\n"
        f"║ Avg Loss:            ${metrics['avg_loss']:>15,.2f}  ║\n"
        f"║ Avg Risk/Reward:      {metrics['avg_risk_reward']:>16.2f}  ║\n"
        "╠══════════════════════════════════════════╣\n"
        f"║ Sharpe Ratio:         {metrics['sharpe_ratio']:>16.2f}  ║\n"
        f"║ Sortino Ratio:        {metrics['sortino_ratio']:>16.2f}  ║\n"
        f"║ Max Drawdown:         {metrics['max_drawdown_pct']:>15.2f}%  ║\n"
        "╠══════════════════════════════════════════╣\n"
        f"║ Initial Capital:     ${metrics['initial_capital']:>15,.2f}  ║\n"
        f"║ Final Equity:        ${metrics['final_equity']:>15,.2f}  ║\n"
        f"║ Max Consec. Wins:     {metrics['max_consecutive_wins']:>16}  ║\n"
        f"║ Max Consec. Losses:   {metrics['max_consecutive_losses']:>16}  ║\n"
        "╚══════════════════════════════════════════╝"
    )
