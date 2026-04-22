"""Strategy optimizer — iteratively improves strategies using Claude."""

import json
from typing import Callable, Optional

import anthropic

from config.settings import (
    ANTHROPIC_API_KEY,
    MODEL_NAME,
    MAX_OPTIMIZATION_ITERATIONS,
    WALK_FORWARD_MAX_ITERATIONS,
    TARGET_WIN_RATE,
    TARGET_PROFIT_FACTOR,
    TARGET_SHARPE_RATIO,
    WF_TRAIN_DAYS,
    WF_TEST_DAYS,
    WF_STEP_DAYS,
)
from agent.prompts.system_prompts import TRADING_AGENT_SYSTEM_PROMPT, STRATEGY_IMPROVEMENT_PROMPT
from backtesting.engine import run_backtest
from backtesting.metrics import format_metrics_report
from backtesting.cost_model import CostModel
from backtesting.walk_forward import walk_forward, format_walk_forward_report


class StrategyOptimizer:
    """Iteratively improves a strategy by backtesting and asking Claude to refine it."""

    def __init__(
        self,
        max_iterations: int = MAX_OPTIMIZATION_ITERATIONS,
        target_win_rate: float = TARGET_WIN_RATE,
        target_profit_factor: float = TARGET_PROFIT_FACTOR,
        target_sharpe: float = TARGET_SHARPE_RATIO,
    ):
        self.max_iterations = max_iterations
        self.target_win_rate = target_win_rate
        self.target_profit_factor = target_profit_factor
        self.target_sharpe = target_sharpe
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.history: list[dict] = []

    def optimize(self, df, initial_strategy: dict, callback=None) -> dict:
        """Run the optimization loop.

        Args:
            df: OHLCV DataFrame.
            initial_strategy: Starting strategy dict.
            callback: Optional function called each iteration with (iteration, strategy, results).

        Returns:
            Dict with 'best_strategy', 'best_results', 'history', 'iterations'.
        """
        current_strategy = initial_strategy
        best_strategy = initial_strategy
        best_score = -float("inf")
        best_results = None

        for iteration in range(1, self.max_iterations + 1):
            # ── Backtest current strategy ─────────────────────────────────
            results = run_backtest(df, current_strategy)
            metrics = results["metrics"]
            score = self._score_strategy(metrics)

            self.history.append({
                "iteration": iteration,
                "strategy": current_strategy,
                "metrics": metrics,
                "score": score,
            })

            if callback:
                callback(iteration, current_strategy, results)

            # Track best
            if score > best_score:
                best_score = score
                best_strategy = current_strategy
                best_results = results

            # Check if targets met
            if self._targets_met(metrics):
                break

            # ── Ask Claude to improve ─────────────────────────────────────
            improved = self._get_improvement(current_strategy, metrics)
            if improved:
                current_strategy = improved
            else:
                break

        return {
            "best_strategy": best_strategy,
            "best_results": best_results,
            "history": self.history,
            "iterations": len(self.history),
        }

    def _score_strategy(self, metrics: dict, is_oos_gap: Optional[dict] = None) -> float:
        """Compute a composite score for ranking strategies.

        If `is_oos_gap` is provided (walk-forward mode), penalize large IS→OOS
        degradation — that's the single most reliable overfit signal on
        minute-bar data.
        """
        if metrics["total_trades"] < 5:
            return -100  # Penalize too few trades

        win_rate_score = metrics["win_rate"] * 30
        pf_score = min(metrics["profit_factor"], 5) * 15
        sharpe_score = min(max(metrics["sharpe_ratio"], -2), 5) * 20
        dd_score = max(0, 20 - metrics["max_drawdown_pct"]) * 1.5
        trade_score = min(metrics["total_trades"] / 20, 1) * 10
        return_score = min(max(metrics["total_return_pct"], -50), 100) * 0.5

        base = win_rate_score + pf_score + sharpe_score + dd_score + trade_score + return_score

        if is_oos_gap:
            # Heavy penalty when OOS Sharpe is much worse than IS Sharpe
            sharpe_gap = max(0, is_oos_gap.get("sharpe_ratio", 0))
            pf_gap = max(0, is_oos_gap.get("profit_factor", 0))
            base -= sharpe_gap * 15
            base -= pf_gap * 10

        return base

    def optimize_walk_forward(
        self,
        df,
        initial_strategy: dict,
        cost_model: Optional[CostModel] = None,
        session_filter: Optional[Callable] = None,
        max_iterations: int = WALK_FORWARD_MAX_ITERATIONS,
        train_days: int = WF_TRAIN_DAYS,
        test_days: int = WF_TEST_DAYS,
        step_days: int = WF_STEP_DAYS,
        callback=None,
    ) -> dict:
        """Walk-forward optimization — scores on OOS metrics, not in-sample.

        Each iteration runs a full rolling WF and asks Claude to improve the
        strategy based on aggregate OOS performance. Best strategy is the one
        with the highest OOS score after applying the IS→OOS gap penalty.
        """
        current_strategy = initial_strategy
        best_strategy = initial_strategy
        best_score = -float("inf")
        best_wf = None

        for iteration in range(1, max_iterations + 1):
            wf = walk_forward(
                df,
                current_strategy,
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
                cost_model=cost_model,
                session_filter=session_filter,
            )
            oos_metrics = wf["oos_metrics"]
            gap = wf["is_oos_gap"]
            score = self._score_strategy(oos_metrics, is_oos_gap=gap)

            self.history.append({
                "iteration": iteration,
                "strategy": current_strategy,
                "metrics": oos_metrics,
                "is_oos_gap": gap,
                "n_folds": wf["n_folds"],
                "score": score,
            })

            if callback:
                callback(iteration, current_strategy, wf)

            if score > best_score:
                best_score = score
                best_strategy = current_strategy
                best_wf = wf

            if self._targets_met(oos_metrics):
                break

            improved = self._get_improvement(current_strategy, oos_metrics)
            if improved:
                current_strategy = improved
            else:
                break

        return {
            "best_strategy": best_strategy,
            "best_wf": best_wf,
            "history": self.history,
            "iterations": len(self.history),
        }

    def _targets_met(self, metrics: dict) -> bool:
        """Check if all target thresholds are met."""
        return (
            metrics["win_rate"] >= self.target_win_rate
            and metrics["profit_factor"] >= self.target_profit_factor
            and metrics["sharpe_ratio"] >= self.target_sharpe
            and metrics["max_drawdown_pct"] <= 15
            and metrics["total_trades"] >= 10
        )

    def _get_improvement(self, strategy: dict, metrics: dict) -> dict | None:
        """Ask Claude to improve the strategy."""
        metrics_report = format_metrics_report(metrics)

        prompt = STRATEGY_IMPROVEMENT_PROMPT.format(
            metrics_report=metrics_report,
            strategy_json=json.dumps(strategy, indent=2),
            target_win_rate=int(self.target_win_rate * 100),
            target_profit_factor=self.target_profit_factor,
            target_sharpe=self.target_sharpe,
        )

        # Build conversation with history for context
        history_summary = ""
        if len(self.history) > 1:
            history_summary = "\n\n## Optimization History\n"
            for h in self.history[-3:]:  # Last 3 iterations
                history_summary += (
                    f"Iteration {h['iteration']}: "
                    f"WR={h['metrics']['win_rate']:.1%} "
                    f"PF={h['metrics']['profit_factor']:.2f} "
                    f"Sharpe={h['metrics']['sharpe_ratio']:.2f} "
                    f"DD={h['metrics']['max_drawdown_pct']:.1f}% "
                    f"Trades={h['metrics']['total_trades']}\n"
                )

        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=4000,
            system=TRADING_AGENT_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": prompt + history_summary + (
                        "\n\nOutput the improved strategy JSON inside a ```json code block. "
                        "Keep the same structure but modify the rules and/or risk management."
                    ),
                }
            ],
        )

        response_text = response.content[0].text

        # Extract JSON from response
        return self._extract_strategy_json(response_text)

    def _extract_strategy_json(self, text: str) -> dict | None:
        """Extract strategy JSON from Claude's response."""
        # Try to find JSON in code blocks
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            json_str = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            json_str = text[start:end].strip()
        else:
            # Try to find raw JSON
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
            else:
                return None

        try:
            strategy = json.loads(json_str)
            # Validate structure
            if "rules" in strategy and isinstance(strategy["rules"], dict):
                return strategy
        except json.JSONDecodeError:
            pass

        return None


def format_optimization_report(result: dict) -> str:
    """Format the optimization result into a readable report."""
    lines = [
        "╔══════════════════════════════════════════╗",
        "║      STRATEGY OPTIMIZATION REPORT        ║",
        "╠══════════════════════════════════════════╣",
        f"║ Iterations Run: {result['iterations']:>24} ║",
        "╠══════════════════════════════════════════╣",
        "║ Iteration History:                       ║",
    ]

    for h in result["history"]:
        m = h["metrics"]
        lines.append(
            f"║ #{h['iteration']:>2}: WR={m['win_rate']:.0%} PF={m['profit_factor']:.1f} "
            f"Sh={m['sharpe_ratio']:.1f} DD={m['max_drawdown_pct']:.0f}% "
            f"T={m['total_trades']:>3} ║"
        )

    lines.append("╠══════════════════════════════════════════╣")
    lines.append("║ Best Strategy Performance:               ║")

    if result["best_results"]:
        bm = result["best_results"]["metrics"]
        lines.extend([
            f"║ Win Rate:      {bm['win_rate']:>24.1%} ║",
            f"║ Profit Factor: {bm['profit_factor']:>24.2f} ║",
            f"║ Sharpe Ratio:  {bm['sharpe_ratio']:>24.2f} ║",
            f"║ Max Drawdown:  {bm['max_drawdown_pct']:>23.2f}% ║",
            f"║ Total Return:  {bm['total_return_pct']:>23.2f}% ║",
            f"║ Total Trades:  {bm['total_trades']:>24} ║",
        ])

    lines.append("╚══════════════════════════════════════════╝")
    return "\n".join(lines)
