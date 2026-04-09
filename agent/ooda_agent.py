"""OODA Agent — the main orchestrator that drives the AI trading workflow.

OODA Loop:
  Observe  → Capture TradingView chart / gather market data
  Orient   → Analyze chart via vision + quantitative analysis
  Decide   → Create or refine a trading strategy
  Act      → Deploy Pine Script to TradingView, read backtest results, optimize

Supports two modes:
  - "tradingview" (default): drives TradingView Desktop via the tv_controller
  - "local": uses the built-in backtesting engine (no TradingView required)
"""

import json
import os
import base64
from datetime import datetime

import anthropic
import pandas as pd

from config.settings import ANTHROPIC_API_KEY, MODEL_NAME, STRATEGIES_DIR, CHARTS_OUTPUT_DIR
from agent.prompts.system_prompts import TRADING_AGENT_SYSTEM_PROMPT
from agent.chart_analyzer import analyze_chart_with_data, analyze_chart_image
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

    Args:
        symbol: Ticker symbol to analyze.
        timeframe: Chart timeframe.
        period_days: Lookback period for data.
        mode: "tradingview" to drive TradingView Desktop, "local" for built-in backtester.
    """

    def __init__(
        self,
        symbol: str = "AAPL",
        timeframe: str = "1d",
        period_days: int = 365,
        mode: str = "tradingview",
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.period_days = period_days
        self.mode = mode
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # TradingView controller (lazy-loaded)
        self._tv: "TradingViewController | None" = None

        # State
        self.df: pd.DataFrame | None = None
        self.chart_b64: str = ""
        self.analysis: str = ""
        self.strategy: dict | None = None
        self.backtest_results: dict | None = None
        self.tv_backtest_results: dict | None = None  # results read from TradingView
        self.pine_script: str = ""
        self.optimization_result: dict | None = None

        # Analysis summaries
        self.indicator_summary: str = ""
        self.volume_profile_summary: str = ""
        self.price_action_summary: str = ""
        self.candlestick_summary: str = ""

        # Conversation history
        self.messages: list[dict] = []

    @property
    def tv(self):
        """Lazy-load the TradingView controller."""
        if self._tv is None:
            from tradingview_mcp.tv_controller import TradingViewController
            self._tv = TradingViewController()
        return self._tv

    # ══════════════════════════════════════════════════════════════════════
    #  FULL CYCLE
    # ══════════════════════════════════════════════════════════════════════

    def run_full_cycle(self, optimize: bool = True, max_iterations: int = 5, callback=None) -> dict:
        """Execute a complete OODA cycle.

        Args:
            optimize: Whether to run the optimization loop.
            max_iterations: Max optimization iterations (TradingView mode).
            callback: Optional function called at each phase with (phase_name, data).

        Returns:
            Dict with all results from the cycle.
        """
        if callback:
            callback("start", {"symbol": self.symbol, "timeframe": self.timeframe, "mode": self.mode})

        # ── OBSERVE ───────────────────────────────────────────────────────
        if callback:
            callback("observe", {"status": "Observing market..."})
        self.observe()
        if callback:
            callback("observe_done", {"bars": len(self.df) if self.df is not None else 0})

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
            callback("act", {"status": "Deploying and backtesting..."})
        self.act(optimize=optimize, max_iterations=max_iterations, callback=callback)
        if callback:
            callback("act_done", {})

        return self.get_results()

    # ══════════════════════════════════════════════════════════════════════
    #  OBSERVE
    # ══════════════════════════════════════════════════════════════════════

    def observe(self):
        """OBSERVE phase: gather data and capture the chart."""
        if self.mode == "tradingview":
            self._observe_tradingview()
        else:
            self._observe_local()

    def _observe_tradingview(self):
        """Observe via TradingView Desktop — capture live chart."""
        print(f"[OBSERVE] Focusing TradingView Desktop...")
        self.tv.focus_tradingview()

        print(f"[OBSERVE] Switching to {self.symbol} ({self.timeframe})...")
        self.tv.change_symbol(self.symbol)
        self.tv.change_timeframe(self.timeframe)

        print("[OBSERVE] Capturing chart screenshot...")
        self.chart_b64 = self.tv.capture_chart_area()

        # Also fetch data for quantitative analysis
        print(f"[OBSERVE] Fetching market data for quantitative analysis...")
        try:
            self.df = get_market_data(self.symbol, self.period_days, self.timeframe)
            print(f"[OBSERVE] Loaded {len(self.df)} bars")

            print("[OBSERVE] Computing indicators and price action...")
            self.indicator_summary = get_indicator_summary(self.df)
            self.volume_profile_summary = get_volume_profile_summary(self.df)
            self.price_action_summary = get_price_action_summary(self.df)
            patterns = detect_patterns(self.df)
            self.candlestick_summary = get_pattern_summary(patterns)
        except Exception as e:
            print(f"[OBSERVE] Data fetch failed ({e}), relying on visual analysis only.")
            self.df = None

        print("[OBSERVE] Observation complete.\n")

    def _observe_local(self):
        """Observe via local data — fetch + render chart."""
        print(f"[OBSERVE] Fetching {self.symbol} data ({self.timeframe}, {self.period_days}d)...")
        self.df = get_market_data(self.symbol, self.period_days, self.timeframe)
        print(f"[OBSERVE] Loaded {len(self.df)} bars from {self.df.index[0]} to {self.df.index[-1]}")

        print("[OBSERVE] Computing technical indicators...")
        self.indicator_summary = get_indicator_summary(self.df)

        print("[OBSERVE] Computing volume profile...")
        self.volume_profile_summary = get_volume_profile_summary(self.df)

        print("[OBSERVE] Analyzing price action...")
        self.price_action_summary = get_price_action_summary(self.df)

        print("[OBSERVE] Detecting candlestick patterns...")
        patterns = detect_patterns(self.df)
        self.candlestick_summary = get_pattern_summary(patterns)

        sr_levels = find_support_resistance(self.df)

        print("[OBSERVE] Rendering chart...")
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

    # ══════════════════════════════════════════════════════════════════════
    #  ORIENT
    # ══════════════════════════════════════════════════════════════════════

    def orient(self):
        """ORIENT phase: Claude analyzes the chart + data."""
        print("[ORIENT] Sending chart to Claude for visual analysis...")

        if self.df is not None:
            self.analysis = analyze_chart_with_data(
                image_b64=self.chart_b64,
                data_summary=f"Symbol: {self.symbol}, Timeframe: {self.timeframe}",
                indicator_summary=self.indicator_summary,
                volume_profile_summary=self.volume_profile_summary,
                candlestick_summary=self.candlestick_summary,
                price_action_summary=self.price_action_summary,
            )
        else:
            # TradingView mode without data — pure visual analysis
            self.analysis = analyze_chart_image(
                image_b64=self.chart_b64,
                context=f"Symbol: {self.symbol}, Timeframe: {self.timeframe}",
            )

        print(f"[ORIENT] Analysis complete ({len(self.analysis)} chars).\n")

    # ══════════════════════════════════════════════════════════════════════
    #  DECIDE
    # ══════════════════════════════════════════════════════════════════════

    def decide(self):
        """DECIDE phase: Claude creates a strategy based on the analysis."""
        print("[DECIDE] Creating trading strategy via Claude...")

        data_context = ""
        if self.indicator_summary:
            data_context = f"""
## Market Data Summary
{self.indicator_summary}

{self.volume_profile_summary}

{self.price_action_summary}

{self.candlestick_summary}
"""

        prompt = f"""Based on your analysis of {self.symbol} ({self.timeframe}), create a trading strategy.

## Your Analysis
{self.analysis}
{data_context}
## Task
Create a complete trading strategy optimized for this market condition.
Output the strategy as a JSON object inside a ```json code block with this structure:
{{
    "name": "Strategy Name",
    "description": "What the strategy does and why",
    "timeframe": "{self.timeframe}",
    "rules": {{
        "long_entry": [{{"indicator": "...", "operator": "...", "value": ...}}],
        "long_exit": [...],
        "short_entry": [...],
        "short_exit": [...]
    }},
    "risk_management": {{
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
        "position_size_pct": 10.0,
        "max_positions": 1
    }}
}}

Available indicators: close, open, high, low, volume, sma_9, sma_20, sma_50, sma_200,
ema_9, ema_20, ema_50, ema_200, rsi_14, macd, macd_signal, macd_histogram, atr_14,
bb_upper, bb_middle, bb_lower, stoch_k, stoch_d, obv, vwap, volume_sma_20.

Operators: >, <, >=, <=, ==, crosses_above, crosses_below."""

        self.messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=4000,
            system=TRADING_AGENT_SYSTEM_PROMPT,
            messages=self.messages,
        )

        response_text = response.content[0].text
        self.messages.append({"role": "assistant", "content": response_text})

        self.strategy = self._extract_json(response_text)

        if self.strategy:
            print(f"[DECIDE] Strategy: {self.strategy.get('name', 'Unnamed')}")
            print(f"[DECIDE] Rules: {json.dumps(self.strategy.get('rules', {}), indent=2)[:500]}\n")
        else:
            print("[DECIDE] WARNING: Could not parse strategy JSON. Using default.\n")
            self.strategy = self._default_strategy()

    # ══════════════════════════════════════════════════════════════════════
    #  ACT
    # ══════════════════════════════════════════════════════════════════════

    def act(self, optimize: bool = True, max_iterations: int = 5, callback=None):
        """ACT phase: deploy strategy and iterate."""
        if self.mode == "tradingview":
            self._act_tradingview(optimize=optimize, max_iterations=max_iterations, callback=callback)
        else:
            self._act_local(optimize=optimize, callback=callback)

    def _act_tradingview(self, optimize: bool = True, max_iterations: int = 5, callback=None):
        """Deploy to TradingView, read results, and iterate."""
        # ── Generate Pine Script ──────────────────────────────────────────
        print("[ACT] Generating Pine Script v5...")
        try:
            self.pine_script = generate_pine_script(
                self.strategy, symbol=self.symbol, timeframe=self.timeframe
            )
        except Exception as e:
            print(f"[ACT] API generation failed ({e}), using template...")
            self.pine_script = strategy_to_pine_template(self.strategy)

        # ── Deploy to TradingView ─────────────────────────────────────────
        print("[ACT] Deploying strategy to TradingView Desktop...")
        self.tv.deploy_strategy(self.pine_script)

        # ── Read backtest results ─────────────────────────────────────────
        print("[ACT] Reading backtest results from Strategy Tester...")
        from tradingview_mcp.screen_reader import (
            read_backtest_results,
            format_results_for_agent,
        )

        overview_b64 = self.tv.capture_backtest_overview()
        self.tv_backtest_results = read_backtest_results(overview_b64)
        results_text = format_results_for_agent(self.tv_backtest_results)
        print(results_text)

        # ── Optimization loop (TradingView mode) ─────────────────────────
        if optimize:
            print(f"\n[ACT] Starting optimization loop (max {max_iterations} iterations)...")

            for iteration in range(1, max_iterations + 1):
                print(f"\n  === Optimization Iteration {iteration} ===")

                # Check if results are good enough
                if self._tv_results_acceptable():
                    print(f"  Target metrics met! Stopping optimization.")
                    break

                # Capture additional screenshots for Claude
                perf_b64 = self.tv.capture_backtest_performance()
                chart_b64 = self.tv.capture_chart_area()

                # Ask Claude to improve
                improved = self._get_tv_improvement(
                    results_text, chart_b64, perf_b64, iteration
                )

                if not improved:
                    print(f"  Could not generate improvement. Stopping.")
                    break

                self.strategy = improved

                # Re-generate and re-deploy
                print(f"  Generating improved Pine Script...")
                try:
                    self.pine_script = generate_pine_script(
                        self.strategy, symbol=self.symbol, timeframe=self.timeframe
                    )
                except Exception:
                    self.pine_script = strategy_to_pine_template(self.strategy)

                print(f"  Deploying improved strategy to TradingView...")
                self.tv.deploy_strategy(self.pine_script)

                # Read new results
                overview_b64 = self.tv.capture_backtest_overview()
                self.tv_backtest_results = read_backtest_results(overview_b64)
                results_text = format_results_for_agent(self.tv_backtest_results)
                print(results_text)

                if callback:
                    callback("optimization_iteration", {
                        "iteration": iteration,
                        "results": self.tv_backtest_results,
                    })

        # ── Save outputs ──────────────────────────────────────────────────
        self._save_outputs()
        print("[ACT] All outputs saved.\n")

    def _act_local(self, optimize: bool = True, callback=None):
        """Run local backtest, optimize, and generate Pine Script."""
        print("[ACT] Running local backtest...")
        self.backtest_results = run_backtest(self.df, self.strategy)
        metrics = self.backtest_results["metrics"]
        print(format_metrics_report(metrics))
        print()

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

            self.strategy = self.optimization_result["best_strategy"]
            self.backtest_results = self.optimization_result["best_results"]

            print()
            print(format_optimization_report(self.optimization_result))
            print()

        print("[ACT] Generating Pine Script v5...")
        try:
            self.pine_script = generate_pine_script(
                self.strategy, symbol=self.symbol, timeframe=self.timeframe
            )
        except Exception as e:
            print(f"[ACT] API Pine Script generation failed ({e}), using template...")
            self.pine_script = strategy_to_pine_template(self.strategy)

        self._save_outputs()
        print("[ACT] All outputs saved.\n")

    # ══════════════════════════════════════════════════════════════════════
    #  TV OPTIMIZATION HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _tv_results_acceptable(self) -> bool:
        """Check if TradingView backtest results meet targets."""
        r = self.tv_backtest_results
        if not r or r.get("parse_error"):
            return False

        win_rate = r.get("win_rate_pct")
        pf = r.get("profit_factor")
        total_trades = r.get("total_closed_trades")

        if win_rate is not None and pf is not None and total_trades is not None:
            try:
                return (
                    float(win_rate) >= 50
                    and float(pf) >= 1.5
                    and int(total_trades) >= 10
                )
            except (ValueError, TypeError):
                pass

        return False

    def _get_tv_improvement(
        self, results_text: str, chart_b64: str, perf_b64: str, iteration: int
    ) -> dict | None:
        """Ask Claude to improve the strategy based on TradingView results."""
        # Build image content for vision
        user_content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": chart_b64},
            },
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": perf_b64},
            },
            {
                "type": "text",
                "text": f"""Iteration {iteration}: Improve this trading strategy based on the TradingView backtest results.

## Current Backtest Results (from TradingView Strategy Tester)
{results_text}

## Current Strategy JSON
```json
{json.dumps(self.strategy, indent=2)}
```

## Target Metrics
- Win Rate: >= 50%
- Profit Factor: >= 1.5
- Reasonable drawdown (< 15%)

## Instructions
1. Look at the chart screenshot — are the entry/exit signals well-placed?
2. Look at the performance summary — what is the biggest weakness?
3. Make 1-3 SPECIFIC changes to improve the weakest metric
4. Output the IMPROVED strategy JSON inside a ```json code block

Keep the same JSON structure. Explain your changes briefly before the JSON.""",
            },
        ]

        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=4000,
            system=TRADING_AGENT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        response_text = response.content[0].text
        print(f"  Claude's reasoning: {response_text[:300]}...")

        return self._extract_json(response_text)

    # ══════════════════════════════════════════════════════════════════════
    #  RESULTS & HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def get_results(self) -> dict:
        """Return all results from the OODA cycle."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "mode": self.mode,
            "data_bars": len(self.df) if self.df is not None else 0,
            "analysis": self.analysis,
            "strategy": self.strategy,
            "backtest_metrics": self.backtest_results["metrics"] if self.backtest_results else None,
            "tv_backtest_results": self.tv_backtest_results,
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

        # Equity curve (local mode only)
        if self.backtest_results:
            eq_b64 = render_equity_curve(
                self.backtest_results["equity_curve"],
                title=f"{self.symbol} Equity Curve — {self.strategy.get('name', 'Strategy')}",
            )
            eq_path = os.path.join(CHARTS_OUTPUT_DIR, f"{prefix}_equity.png")
            with open(eq_path, "wb") as f:
                f.write(base64.b64decode(eq_b64))
            print(f"  Equity curve saved: {eq_path}")

        # Report
        report_path = os.path.join(STRATEGIES_DIR, f"{prefix}_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Strategy: {self.strategy.get('name', 'Unknown')}\n")
            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Timeframe: {self.timeframe}\n")
            f.write(f"Mode: {self.mode}\n\n")

            if self.backtest_results:
                f.write("=== Local Backtest ===\n")
                f.write(format_metrics_report(self.backtest_results["metrics"]))
                f.write("\n\n")

            if self.tv_backtest_results:
                from tradingview_mcp.screen_reader import format_results_for_agent
                f.write("=== TradingView Backtest ===\n")
                f.write(format_results_for_agent(self.tv_backtest_results))
                f.write("\n\n")

            f.write("Strategy JSON:\n")
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
        """Fallback strategy."""
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

    def __init__(self, mode: str = "tradingview"):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.mode = mode
        self.agent: OODAAgent | None = None
        self.messages: list[dict] = []

    def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's response."""
        self.messages.append({"role": "user", "content": user_message})

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
        if not self.agent:
            return "No analysis has been run yet. Suggest the user analyze a symbol."

        parts = [f"Current Symbol: {self.agent.symbol} ({self.agent.timeframe}) [mode: {self.agent.mode}]"]

        if self.agent.analysis:
            parts.append(f"\nLatest Analysis:\n{self.agent.analysis[:2000]}")

        if self.agent.strategy:
            parts.append(f"\nCurrent Strategy:\n{json.dumps(self.agent.strategy, indent=2)}")

        if self.agent.backtest_results:
            parts.append(f"\nLocal Backtest:\n{format_metrics_report(self.agent.backtest_results['metrics'])}")

        if self.agent.tv_backtest_results:
            from tradingview_mcp.screen_reader import format_results_for_agent
            parts.append(f"\nTradingView Backtest:\n{format_results_for_agent(self.agent.tv_backtest_results)}")

        return "\n".join(parts)
