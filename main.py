#!/usr/bin/env python3
"""AI Trading Agent — Main entry point.

An OODA-loop AI agent powered by Claude Opus that:
- Drives TradingView Desktop to write Pine Script and run backtests
- Analyzes markets using price action, indicators, and volume profile
- Reads candlestick charts visually to identify patterns
- Iteratively optimizes strategies by reading TradingView backtest results

Modes:
  --mode tradingview   (default) Automate TradingView Desktop directly
  --mode local         Use built-in backtester (no TradingView needed)

Usage:
    python main.py                              # Interactive mode (TradingView)
    python main.py analyze AAPL                 # Full OODA cycle on AAPL
    python main.py analyze BTCUSD 1h 90         # Symbol, timeframe, days
    python main.py --mode local analyze TSLA    # Local backtester mode
    python main.py --no-optimize MSFT           # Skip optimization loop
    python main.py mcp                          # Start MCP server (for Claude Code)
"""

import argparse
import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import ANTHROPIC_API_KEY
from agent.ooda_agent import OODAAgent, InteractiveAgent
from backtesting.metrics import format_metrics_report


def print_banner(mode: str = "tradingview"):
    mode_label = "TradingView Desktop" if mode == "tradingview" else "Local Backtester"
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           AI TRADING AGENT — Powered by Claude Opus         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OODA Loop: Observe → Orient → Decide → Act                 ║
║  Mode: {mode_label:<52}║
║                                                              ║
║  Capabilities:                                               ║
║  • Visual chart analysis (candlestick pattern recognition)   ║
║  • Price action & market structure analysis                  ║
║  • Technical indicators (RSI, MACD, BB, Stoch, ATR, etc.)   ║
║  • Volume profile analysis (POC, VAH, VAL, HVN, LVN)        ║
║  • Pine Script v5 strategy generation for TradingView        ║
║  • Deploy & backtest directly in TradingView Desktop         ║
║  • Iterative strategy optimization via AI                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def run_analysis(
    symbol: str,
    timeframe: str = "1d",
    period_days: int = 365,
    optimize: bool = True,
    mode: str = "tradingview",
):
    """Run a full OODA cycle analysis on a symbol."""
    print(f"\nStarting OODA cycle for {symbol} ({timeframe}, {period_days}d lookback)")
    print(f"Mode: {mode}")
    print("=" * 60)

    agent = OODAAgent(symbol=symbol, timeframe=timeframe, period_days=period_days, mode=mode)

    def callback(phase, data):
        if phase == "start":
            print(f"\n{'='*60}")
            print(f"  Analyzing: {data['symbol']} ({data['timeframe']}) [{data['mode']}]")
            print(f"{'='*60}")
        elif phase == "observe":
            print(f"\n[1/4] OBSERVE — {data['status']}")
        elif phase == "observe_done":
            bars = data['bars']
            print(f"  Done ({bars} bars)" if bars else "  Done (visual only)")
        elif phase == "orient":
            print(f"\n[2/4] ORIENT — {data['status']}")
        elif phase == "orient_done":
            print(f"  Analysis complete")
        elif phase == "decide":
            print(f"\n[3/4] DECIDE — {data['status']}")
        elif phase == "decide_done":
            print(f"  Strategy: {data['strategy_name']}")
        elif phase == "act":
            print(f"\n[4/4] ACT — {data['status']}")
        elif phase == "optimization_iteration":
            if "metrics" in data:
                m = data["metrics"]
                print(
                    f"  Iteration {data['iteration']}: "
                    f"WR={m['win_rate']:.1%}  PF={m['profit_factor']:.2f}  "
                    f"Sharpe={m['sharpe_ratio']:.2f}  DD={m['max_drawdown_pct']:.1f}%"
                )
            else:
                print(f"  Iteration {data['iteration']}: results captured from TradingView")

    results = agent.run_full_cycle(optimize=optimize, callback=callback)

    # ── Print final summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)

    if results["backtest_metrics"]:
        print(format_metrics_report(results["backtest_metrics"]))

    if results.get("tv_backtest_results"):
        from tradingview_mcp.screen_reader import format_results_for_agent
        print(format_results_for_agent(results["tv_backtest_results"]))

    print(f"\nStrategy: {results['strategy'].get('name', 'Unknown')}")
    print(f"Description: {results['strategy'].get('description', '')}")

    print("\nFiles saved:")
    print(f"  Strategies dir: strategies/generated/")
    print(f"  Charts dir: charts_output/")

    # Show Pine Script preview
    if results["pine_script"]:
        lines = results["pine_script"].split("\n")
        print(f"\nPine Script Preview (first 20 lines):")
        print("-" * 40)
        for line in lines[:20]:
            print(f"  {line}")
        if len(lines) > 20:
            print(f"  ... ({len(lines) - 20} more lines)")

    return results


def interactive_mode(mode: str = "tradingview"):
    """Run the agent in interactive conversational mode."""
    print_banner(mode)
    print("Interactive Mode — Chat with the AI Trading Agent")
    print("Commands:")
    print("  analyze <SYMBOL> [timeframe] [days]  — Run full OODA analysis")
    print("  capture                              — Screenshot TradingView chart")
    print("  results                              — Read TradingView backtest results")
    print("  deploy                               — Re-deploy current Pine Script")
    print("  pine                                 — Show Pine Script")
    print("  strategy                             — Show current strategy")
    print("  metrics                              — Show backtest metrics")
    print("  quit / exit                          — Exit")
    print("  Anything else                        — Chat with the agent")
    print("-" * 60)

    agent = None
    interactive = InteractiveAgent(mode=mode)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        if lower in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        elif lower.startswith("analyze"):
            parts = user_input.split()
            symbol = parts[1].upper() if len(parts) > 1 else "AAPL"
            timeframe = parts[2] if len(parts) > 2 else "1d"
            days = int(parts[3]) if len(parts) > 3 else 365

            try:
                results = run_analysis(symbol, timeframe, days, mode=mode)
                agent = OODAAgent(symbol=symbol, timeframe=timeframe, period_days=days, mode=mode)
                agent.strategy = results["strategy"]
                agent.pine_script = results.get("pine_script", "")
                agent.backtest_results = results.get("backtest_metrics")
                agent.tv_backtest_results = results.get("tv_backtest_results")
                interactive.agent = agent
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

        elif lower == "capture" and mode == "tradingview":
            try:
                from tradingview_mcp.tv_controller import TradingViewController
                from tradingview_mcp.screen_reader import analyze_chart
                tv = TradingViewController()
                b64 = tv.capture_chart_area()
                analysis = analyze_chart(b64)
                print(f"\n{analysis}")
            except Exception as e:
                print(f"Error capturing: {e}")

        elif lower == "results" and mode == "tradingview":
            try:
                from tradingview_mcp.tv_controller import TradingViewController
                from tradingview_mcp.screen_reader import read_backtest_results, format_results_for_agent
                tv = TradingViewController()
                b64 = tv.capture_backtest_overview()
                results = read_backtest_results(b64)
                print(format_results_for_agent(results))
            except Exception as e:
                print(f"Error reading results: {e}")

        elif lower == "deploy" and agent and agent.pine_script:
            try:
                from tradingview_mcp.tv_controller import TradingViewController
                tv = TradingViewController()
                tv.deploy_strategy(agent.pine_script)
                print("Strategy deployed to TradingView.")
            except Exception as e:
                print(f"Error deploying: {e}")

        elif lower == "pine" and agent and agent.pine_script:
            print("\n" + agent.pine_script)

        elif lower == "strategy" and agent and agent.strategy:
            print(json.dumps(agent.strategy, indent=2))

        elif lower == "metrics":
            if agent and agent.backtest_results:
                print(format_metrics_report(agent.backtest_results["metrics"]))
            if agent and agent.tv_backtest_results:
                from tradingview_mcp.screen_reader import format_results_for_agent
                print(format_results_for_agent(agent.tv_backtest_results))

        else:
            # Free-form chat
            try:
                response = interactive.chat(user_input)
                print(f"\n{response}")
            except Exception as e:
                print(f"Error: {e}")


def run_mcp_server():
    """Start the TradingView MCP server for Claude Code integration."""
    print("Starting TradingView MCP Server (stdio transport)...")
    print("This server exposes TradingView Desktop automation as MCP tools.")
    print("Connect via Claude Code or any MCP client.\n")

    from tradingview_mcp.server import main as mcp_main
    mcp_main()


def main():
    parser = argparse.ArgumentParser(
        description="AI Trading Agent — OODA loop powered by Claude Opus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Interactive mode (TradingView)
  python main.py analyze AAPL                 # Full OODA on Apple via TradingView
  python main.py analyze BTCUSD 1h 90         # BTC hourly, 90 days
  python main.py --mode local analyze TSLA    # Use local backtester
  python main.py --no-optimize MSFT           # Skip optimization
  python main.py mcp                          # Start MCP server for Claude Code
        """,
    )
    parser.add_argument("command", nargs="?", default=None,
                        help="Command: 'analyze', 'mcp'")
    parser.add_argument("symbol", nargs="?", default="AAPL",
                        help="Ticker symbol (default: AAPL)")
    parser.add_argument("timeframe", nargs="?", default="1d",
                        help="Timeframe (default: 1d)")
    parser.add_argument("days", nargs="?", type=int, default=365,
                        help="Lookback days (default: 365)")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Skip optimization loop")
    parser.add_argument("--mode", choices=["tradingview", "local"], default="tradingview",
                        help="Mode: 'tradingview' (drive TV Desktop) or 'local' (built-in backtester)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward OOS validation on XAUUSD seed strategies")
    parser.add_argument("--spread", type=float, default=None,
                        help="XAUUSD spread in USD per oz for cost model (default: config XAUUSD_DEFAULT_SPREAD)")
    parser.add_argument("--stress", action="store_true",
                        help="After walk-forward, re-run best strategy with stress spread")

    args = parser.parse_args()

    # MCP server mode — doesn't need API key validation
    if args.command == "mcp":
        run_mcp_server()
        return

    # Walk-forward XAUUSD mode — bypasses the OODA cycle
    if args.walk_forward:
        from agent.xauusd_runner import run_xauusd_cycle, run_stress_test
        from config.settings import XAUUSD_DEFAULT_SPREAD, XAUUSD_STRESS_SPREAD

        spread = args.spread if args.spread is not None else XAUUSD_DEFAULT_SPREAD
        print(f"\nRunning walk-forward on XAUUSD seed strategies...")
        print(f"  Timeframe: {args.timeframe}   Period: {args.days}d   Spread: ${spread}/oz\n")
        result = run_xauusd_cycle(
            timeframe=args.timeframe,
            period_days=args.days,
            spread=spread,
        )
        print(result["report"])

        if args.stress and result.get("best"):
            print(f"\n=== Stress test (spread ${XAUUSD_STRESS_SPREAD}/oz) ===\n")
            stressed = run_stress_test(result, stress_spread=XAUUSD_STRESS_SPREAD)
            print(stressed["report"])
        return

    # Validate API key
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Set it in your .env file or environment:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    if args.command == "analyze":
        print_banner(args.mode)
        run_analysis(
            symbol=args.symbol,
            timeframe=args.timeframe,
            period_days=args.days,
            optimize=not args.no_optimize,
            mode=args.mode,
        )
    elif args.command is None:
        interactive_mode(mode=args.mode)
    else:
        # Treat command as symbol
        print_banner(args.mode)
        run_analysis(
            symbol=args.command.upper(),
            timeframe=args.timeframe,
            period_days=args.days,
            optimize=not args.no_optimize,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
