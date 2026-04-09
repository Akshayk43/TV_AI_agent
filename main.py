#!/usr/bin/env python3
"""AI Trading Agent — Main entry point.

An OODA-loop AI agent powered by Claude Opus that:
- Analyzes markets using price action, indicators, and volume profile
- Reads candlestick charts visually to identify patterns
- Creates TradingView Pine Script strategies
- Backtests strategies and iteratively optimizes them

Usage:
    python main.py                          # Interactive mode
    python main.py analyze AAPL             # Analyze a symbol
    python main.py analyze BTCUSD 1h 90     # Symbol, timeframe, days
    python main.py --no-optimize TSLA       # Skip optimization loop
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


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           AI TRADING AGENT — Powered by Claude Opus         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OODA Loop: Observe → Orient → Decide → Act                 ║
║                                                              ║
║  Capabilities:                                               ║
║  • Visual chart analysis (candlestick pattern recognition)   ║
║  • Price action & market structure analysis                  ║
║  • Technical indicators (RSI, MACD, BB, Stoch, ATR, etc.)   ║
║  • Volume profile analysis (POC, VAH, VAL, HVN, LVN)        ║
║  • Pine Script v5 strategy generation for TradingView        ║
║  • Backtesting with performance metrics                      ║
║  • Iterative strategy optimization                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def run_analysis(symbol: str, timeframe: str = "1d", period_days: int = 365, optimize: bool = True):
    """Run a full OODA cycle analysis on a symbol."""
    print(f"\nStarting OODA cycle for {symbol} ({timeframe}, {period_days}d lookback)")
    print("=" * 60)

    agent = OODAAgent(symbol=symbol, timeframe=timeframe, period_days=period_days)

    def callback(phase, data):
        if phase == "start":
            print(f"\n{'='*60}")
            print(f"  Analyzing: {data['symbol']} ({data['timeframe']})")
            print(f"{'='*60}")
        elif phase == "observe":
            print(f"\n[1/4] OBSERVE — {data['status']}")
        elif phase == "observe_done":
            print(f"  ✓ Loaded {data['bars']} bars of data")
        elif phase == "orient":
            print(f"\n[2/4] ORIENT — {data['status']}")
        elif phase == "orient_done":
            print(f"  ✓ Analysis complete")
        elif phase == "decide":
            print(f"\n[3/4] DECIDE — {data['status']}")
        elif phase == "decide_done":
            print(f"  ✓ Strategy: {data['strategy_name']}")
        elif phase == "act":
            print(f"\n[4/4] ACT — {data['status']}")
        elif phase == "optimization_iteration":
            m = data["metrics"]
            print(
                f"  Iteration {data['iteration']}: "
                f"WR={m['win_rate']:.1%}  PF={m['profit_factor']:.2f}  "
                f"Sharpe={m['sharpe_ratio']:.2f}  DD={m['max_drawdown_pct']:.1f}%"
            )

    results = agent.run_full_cycle(optimize=optimize, callback=callback)

    # ── Print final summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)

    if results["backtest_metrics"]:
        print(format_metrics_report(results["backtest_metrics"]))

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


def interactive_mode():
    """Run the agent in interactive conversational mode."""
    print_banner()
    print("Interactive Mode — Chat with the AI Trading Agent")
    print("Commands:")
    print("  analyze <SYMBOL> [timeframe] [days]  — Run full OODA analysis")
    print("  optimize                             — Optimize current strategy")
    print("  pine                                 — Show Pine Script")
    print("  strategy                             — Show current strategy")
    print("  metrics                              — Show backtest metrics")
    print("  quit / exit                          — Exit")
    print("  Anything else                        — Chat with the agent")
    print("-" * 60)

    agent = None
    interactive = InteractiveAgent()

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

            agent = OODAAgent(symbol=symbol, timeframe=timeframe, period_days=days)
            interactive.agent = agent

            try:
                results = run_analysis(symbol, timeframe, days)
                agent.strategy = results["strategy"]
                agent.backtest_results = results.get("backtest_metrics")
            except Exception as e:
                print(f"Error: {e}")

        elif lower == "pine" and agent and agent.pine_script:
            print("\n" + agent.pine_script)

        elif lower == "strategy" and agent and agent.strategy:
            print(json.dumps(agent.strategy, indent=2))

        elif lower == "metrics" and agent and agent.backtest_results:
            print(format_metrics_report(agent.backtest_results["metrics"]))

        else:
            # Free-form chat
            try:
                response = interactive.chat(user_input)
                print(f"\n{response}")
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Trading Agent — OODA loop powered by Claude Opus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode
  python main.py analyze AAPL             # Analyze Apple stock
  python main.py analyze BTCUSD 1h 90     # BTC hourly, 90 days
  python main.py analyze TSLA 1d 180      # Tesla daily, 180 days
  python main.py --no-optimize MSFT       # Skip optimization
        """,
    )
    parser.add_argument("command", nargs="?", default=None, help="Command: 'analyze'")
    parser.add_argument("symbol", nargs="?", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("timeframe", nargs="?", default="1d", help="Timeframe (default: 1d)")
    parser.add_argument("days", nargs="?", type=int, default=365, help="Lookback days (default: 365)")
    parser.add_argument("--no-optimize", action="store_true", help="Skip optimization loop")

    args = parser.parse_args()

    # Validate API key
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Set it in your .env file or environment:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    if args.command == "analyze":
        print_banner()
        run_analysis(
            symbol=args.symbol,
            timeframe=args.timeframe,
            period_days=args.days,
            optimize=not args.no_optimize,
        )
    elif args.command is None:
        interactive_mode()
    else:
        # Treat command as symbol
        print_banner()
        run_analysis(
            symbol=args.command.upper(),
            timeframe=args.timeframe,
            period_days=args.days,
            optimize=not args.no_optimize,
        )


if __name__ == "__main__":
    main()
