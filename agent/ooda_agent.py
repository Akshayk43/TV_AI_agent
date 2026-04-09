"""OODA Agent — the main orchestrator that drives the AI trading workflow.

OODA Loop:
  Observe  → Gather market data, render charts, compute indicators
  Orient   → Analyze data via vision + quantitative analysis
  Decide   → Create or refine a trading strategy
  Act      → Backtest, generate Pine Script, optimize
"""

import json
import os
from datetime import datetime

import anthropic
import pandas as pd

from config.settings import ANTHROPIC_API_KEY, MODEL_NAME, STRATEGIES_DIR, CHARTS_OUTPUT_DIR
from agent.prompts.system_prompts import TRADING_AGENT_SYSTEM_PROMPT
from agent.chart_analyzer import analyze_chart_with_data
from agent.pine_script_generator import generate_pine_script, strategy_to_pine_template
from agent.strategy_optimizer import StrategyOptimizer, format_optimization_report
from backtesting.data_loader import get_market_data
from backtesting.engine import run_backtest
from backtesting.metrics import format_metrics_report
from charts.renderer import render_candlestick_chart, render_equity_curve
from knowledge.indicators import get_indicator_summary, add_all_indicators
from knowledge.price_action import get_price_action_summary, find_support_resistance
from knowledge.volume_profile import get_volume_profile_summary
from knowledge.candlestick_patterns import detect_patterns, get_pattern_summary


class OODAAgent:
    """AI Trading Agent that operates in an OODA loop.

    Workflow:
    1. observe()  — Fetch data, render chart, compute all analyses.
    2. orient()   — Have Claude analyze the chart + data (vision + text).
    3. decide()   — Have Claude create a strategy based on the analysis.
    4. act()      — Backtest, optimize, generate Pine Script.
    """

    def __init__(self, symbol: str = "AAPL", timeframe: str = "1d", period_days: int = 365):
        self.symbol = symbol
        self.timeframe = timeframe
        self.period_days = period_days
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # State
        self.df: pd.DataFrame | None = None
        self.chart_b64: str = ""
        self.analysis: str = ""
        self.strategy: dict | None = None
        self.backtest_results: dict | None = None
        self.pine_script: str = ""
        self.optimization_result: dict | None = None

        # Analysis summaries
        self.indicator_summary: str = ""
        self.volume_profile_summary: str = ""
        self.price_action_summary: str = ""
        self.candlestick_summary: str = ""

        # Conversation history for context
        self.messages: list[dict] = []

    def run_full_cycle(self, optimize: bool = True, callback=None) -> dict:
        """Execute a complete OODA cycle.

        Args:
            optimize: Whether to run the optimization loop.
            callback: Optional function called at each phase with (phase_name, data).

        Returns:
            Dict with all results from the cycle.
        """
        if callback:
            callback("start", {"symbol": self.symbol, "timeframe": self.timeframe})

        # ── OBSERVE ───────────────────────────────────────────────────────
        if callback:
            callback("observe", {"status": "Loading market data..."})
        self.observe()
        if callback:
            callback("observe_done", {"bars": len(self.df)})

        # ── ORIENT ────────────────────────────────────────────────────────
        if callback:
            callback("orient", {"status": "Analyzing chart and data..."})
        self.orient()
        if callback:
            callback("orient_done", {"analysis_length": len(self.analysis)})

        # ── DECIDE ────────────────────────────────────────────────────────
        if callback:
            callback("decide", {"status": "Creating trading strategy..."})
        self.decide()
        if callback:
            callback("decide_done", {"strategy_name": self.strategy.get("name", "Unknown")})

        # ── ACT ───────────────────────────────────────────────────────────
        if callback:
            callback("act", {"status": "Backtesting and generating Pine Script..."})
        self.act(optimize=optimize, callback=callback)
        if callback:
            callback("act_done", {})

        return self.get_results()

    def observe(self):
        """OBSERVE phase: gather all data and render the chart."""
        print(f"[OBSERVE] Fetching {self.symbol} data ({self.timeframe}, {self.period_days}d)...")
        self.df = get_market_data(self.symbol, self.period_days, self.timeframe)
        print(f"[OBSERVE] Loaded {len(self.df)} bars from {self.df.index[0]} to {self.df.index[-1]}")

        # Compute all analyses
        print("[OBSERVE] Computing technical indicators...")
        self.indicator_summary = get_indicator_summary(self.df)

        print("[OBSERVE] Computing volume profile...")
        self.volume_profile_summary = get_volume_profile_summary(self.df)

        print("[OBSERVE] Analyzing price action...")
        self.price_action_summary = get_price_action_summary(self.df)

        print("[OBSERVE] Detecting candlestick patterns...")
        patterns = detect_patterns(self.df)
        self.candlestick_summary = get_pattern_summary(patterns)

        # Find S/R for chart overlay
        sr_levels = find_support_resistance(self.df)

        # Render chart with all overlays
        print("[OBSERVE] Rendering chart with indicators and volume profile...")
        self.chart_b64 = render_candlestick_chart(
            self.df,
            symbol=self.symbol,
            timeframe=self.timeframe,
            show_volume=True,
            show_indicators=["sma_20", "sma_50", "ema_200"],
            show_volume_profile=True,
            show_support_resistance=True,
            sr_levels=sr_levels,
            last_n_bars=120,
            save_path=os.path.join(CHARTS_OUTPUT_DIR, f"{self.symbol}_{self.timeframe}_analysis.png"),
        )
        print("[OBSERVE] Observation complete.\n")

    def orient(self):
        """ORIENT phase: Claude analyzes the chart image + computed data."""
        print("[ORIENT] Sending chart and data to Claude for analysis...")

        self.analysis = analyze_chart_with_data(
            image_b64=self.chart_b64,
            data_summary=f"Symbol: {self.symbol}, Timeframe: {self.timeframe}, Bars: {len(self.df)}",
            indicator_summary=self.indicator_summary,
            volume_profile_summary=self.volume_profile_summary,
            candlestick_summary=self.candlestick_summary,
            price_action_summary=self.price_action_summary,
        )

        print(f"[ORIENT] Analysis complete ({len(self.analysis)} chars).\n")

    def decide(self):
        """DECIDE phase: Claude creates a strategy based on the analysis."""
        print("[DECIDE] Asking Claude to create a trading strategy...")

        prompt = f"""Based on your analysis of {self.symbol} ({self.timeframe}), create a trading strategy.

## Your Analysis
{self.analysis}

## Market Data Summary
{self.indicator_summary}

{self.volume_profile_summary}

{self.price_action_summary}

{self.candlestick_summary}

## Task
Create a complete trading strategy optimized for this specific market condition.
Output the strategy in the required JSON format with:
- Specific entry/exit rules using available indicators
- Appropriate risk management for the current volatility
- Rules that align with the identified trend and key levels

Output the strategy JSON inside a ```json code block."""

        self.messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=4000,
            system=TRADING_AGENT_SYSTEM_PROMPT,
            messages=self.messages,
        )

        response_text = response.content[0].text
        self.messages.append({"role": "assistant", "content": response_text})

        # Extract strategy JSON
        self.strategy = self._extract_json(response_text)

        if self.strategy:
            print(f"[DECIDE] Strategy created: {self.strategy.get('name', 'Unnamed')}")
            print(f"[DECIDE] Rules: {json.dumps(self.strategy.get('rules', {}), indent=2)[:500]}\n")
        else:
            print("[DECIDE] WARNING: Could not extract strategy JSON. Using default strategy.\n")
            self.strategy = self._default_strategy()

    def act(self, optimize: bool = True, callback=None):
        """ACT phase: backtest, optimize, and generate Pine Script."""
        # ── Initial backtest ──────────────────────────────────────────────
        print("[ACT] Running initial backtest...")
        self.backtest_results = run_backtest(self.df, self.strategy)
        metrics = self.backtest_results["metrics"]
        print(format_metrics_report(metrics))
        print()

        # ── Optimization loop ─────────────────────────────────────────────
        if optimize:
            print("[ACT] Starting optimization loop...")
            optimizer = StrategyOptimizer()

            def opt_callback(iteration, strategy, results):
                m = results["metrics"]
                print(
                    f"  Iteration {iteration}: "
                    f"WR={m['win_rate']:.1%} PF={m['profit_factor']:.2f} "
                    f"Sharpe={m['sharpe_ratio']:.2f} DD={m['max_drawdown_pct']:.1f}% "
                    f"Trades={m['total_trades']}"
                )
                if callback:
                    callback("optimization_iteration", {"iteration": iteration, "metrics": m})

            self.optimization_result = optimizer.optimize(
                self.df, self.strategy, callback=opt_callback
            )

            # Update to best strategy
            self.strategy = self.optimization_result["best_strategy"]
            self.backtest_results = self.optimization_result["best_results"]

            print()
            print(format_optimization_report(self.optimization_result))
            print()

        # ── Generate Pine Script ──────────────────────────────────────────
        print("[ACT] Generating Pine Script v5...")
        try:
            self.pine_script = generate_pine_script(
                self.strategy, symbol=self.symbol, timeframe=self.timeframe
            )
        except Exception as e:
            print(f"[ACT] API Pine Script generation failed ({e}), using template...")
            self.pine_script = strategy_to_pine_template(self.strategy)

        # Save outputs
        self._save_outputs()
        print("[ACT] All outputs saved.\n")

    def get_results(self) -> dict:
        """Return all results from the OODA cycle."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "data_bars": len(self.df) if self.df is not None else 0,
            "analysis": self.analysis,
            "strategy": self.strategy,
            "backtest_metrics": self.backtest_results["metrics"] if self.backtest_results else None,
            "pine_script": self.pine_script,
            "optimization": self.optimization_result,
            "chart_b64": self.chart_b64,
        }

    def _save_outputs(self):
        """Save strategy, Pine Script, and charts to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.symbol}_{self.timeframe}_{timestamp}"

        # Strategy JSON
        strategy_path = os.path.join(STRATEGIES_DIR, f"{prefix}_strategy.json")
        with open(strategy_path, "w") as f:
            json.dump(self.strategy, f, indent=2)
        print(f"  Strategy saved: {strategy_path}")

        # Pine Script
        pine_path = os.path.join(STRATEGIES_DIR, f"{prefix}_strategy.pine")
        with open(pine_path, "w") as f:
            f.write(self.pine_script)
        print(f"  Pine Script saved: {pine_path}")

        # Equity curve
        if self.backtest_results:
            eq_b64 = render_equity_curve(
                self.backtest_results["equity_curve"],
                title=f"{self.symbol} Equity Curve — {self.strategy.get('name', 'Strategy')}",
            )
            eq_path = os.path.join(CHARTS_OUTPUT_DIR, f"{prefix}_equity.png")
            import base64
            with open(eq_path, "wb") as f:
                f.write(base64.b64decode(eq_b64))
            print(f"  Equity curve saved: {eq_path}")

        # Backtest report
        if self.backtest_results:
            report_path = os.path.join(STRATEGIES_DIR, f"{prefix}_report.txt")
            with open(report_path, "w") as f:
                f.write(f"Strategy: {self.strategy.get('name', 'Unknown')}\n")
                f.write(f"Symbol: {self.symbol}\n")
                f.write(f"Timeframe: {self.timeframe}\n")
                f.write(f"Data Range: {self.backtest_results.get('data_range', 'N/A')}\n\n")
                f.write(format_metrics_report(self.backtest_results["metrics"]))
                f.write("\n\nStrategy JSON:\n")
                f.write(json.dumps(self.strategy, indent=2))
            print(f"  Report saved: {report_path}")

    def _extract_json(self, text: str) -> dict | None:
        """Extract JSON from Claude's response."""
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            json_str = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            json_str = text[start:end].strip()
        else:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
            else:
                return None

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _default_strategy(self) -> dict:
        """Fallback strategy if Claude's response can't be parsed."""
        return {
            "name": f"{self.symbol} Default Strategy",
            "description": "RSI + SMA crossover strategy with trend filter",
            "timeframe": self.timeframe,
            "rules": {
                "long_entry": [
                    {"indicator": "rsi_14", "operator": "<", "value": 35},
                    {"indicator": "close", "operator": ">", "ref": "sma_50"},
                    {"indicator": "ema_20", "operator": ">", "ref": "sma_50"},
                ],
                "long_exit": [
                    {"indicator": "rsi_14", "operator": ">", "value": 70},
                ],
                "short_entry": [
                    {"indicator": "rsi_14", "operator": ">", "value": 65},
                    {"indicator": "close", "operator": "<", "ref": "sma_50"},
                    {"indicator": "ema_20", "operator": "<", "ref": "sma_50"},
                ],
                "short_exit": [
                    {"indicator": "rsi_14", "operator": "<", "value": 30},
                ],
            },
            "risk_management": {
                "stop_loss_pct": 2.5,
                "take_profit_pct": 5.0,
                "position_size_pct": 10.0,
                "max_positions": 1,
            },
        }


class InteractiveAgent:
    """Interactive wrapper for conversational use of the OODA agent."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.agent: OODAAgent | None = None
        self.messages: list[dict] = []

    def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        Supports commands:
        - "analyze SYMBOL" — run full OODA cycle
        - "optimize" — re-run optimization on current strategy
        - "pine" — regenerate Pine Script
        - "backtest" — re-run backtest
        - "show strategy" — display current strategy
        - Free-form questions about trading
        """
        self.messages.append({"role": "user", "content": user_message})

        # Build context from agent state
        context = self._build_context()

        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=6000,
            system=TRADING_AGENT_SYSTEM_PROMPT + "\n\n" + context,
            messages=self.messages,
        )

        assistant_msg = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_msg})

        return assistant_msg

    def _build_context(self) -> str:
        """Build context string from current agent state."""
        if not self.agent:
            return "No analysis has been run yet. Suggest the user analyze a symbol."

        parts = [f"Current Symbol: {self.agent.symbol} ({self.agent.timeframe})"]

        if self.agent.analysis:
            parts.append(f"\nLatest Analysis:\n{self.agent.analysis[:2000]}")

        if self.agent.strategy:
            parts.append(f"\nCurrent Strategy:\n{json.dumps(self.agent.strategy, indent=2)}")

        if self.agent.backtest_results:
            parts.append(f"\nBacktest Results:\n{format_metrics_report(self.agent.backtest_results['metrics'])}")

        return "\n".join(parts)
