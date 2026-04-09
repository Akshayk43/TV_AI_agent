"""TradingView MCP Server — exposes TradingView Desktop automation as MCP tools.

This server bridges Claude Code / the AI agent to TradingView Desktop via
the Model Context Protocol (MCP). It provides tools to:
- Write Pine Script in the editor
- Compile and run strategies
- Capture and analyze charts
- Read backtest results
- Change symbols and timeframes

Usage:
    # Run standalone (stdio transport for Claude Code):
    python -m tradingview_mcp.server

    # Or register in Claude Code settings / MCP config
"""

import json
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

from tradingview_mcp.tv_controller import TradingViewController
from tradingview_mcp.screen_reader import (
    read_backtest_results,
    read_performance_summary,
    read_trades_list,
    analyze_chart,
    read_compilation_errors,
    format_results_for_agent,
)
from agent.pine_script_generator import generate_pine_script, strategy_to_pine_template
from agent.prompts.system_prompts import TRADING_AGENT_SYSTEM_PROMPT

# ── Create MCP Server ────────────────────────────────────────────────────────

mcp = FastMCP(
    "tradingview",
    instructions=(
        "TradingView Desktop automation server. Use these tools to write Pine Script "
        "strategies in TradingView's Pine Editor, run backtests, capture charts, and "
        "read strategy performance results. The agent operates in an OODA loop:\n"
        "1. Observe: capture chart, analyze market\n"
        "2. Orient: identify patterns, levels, trends\n"
        "3. Decide: create/refine strategy\n"
        "4. Act: deploy to TradingView, read results, iterate"
    ),
)

# Shared controller instance
_controller = TradingViewController()


# ═════════════════════════════════════════════════════════════════════════════
#  TOOLS
# ═════════════════════════════════════════════════════════════════════════════


@mcp.tool()
def focus_tradingview() -> str:
    """Bring TradingView Desktop to the foreground.

    Call this first to make sure TradingView is visible and active.
    """
    ok = _controller.focus_tradingview()
    return "TradingView focused successfully." if ok else "Could not find TradingView window."


@mcp.tool()
def change_symbol(symbol: str) -> str:
    """Change the chart symbol in TradingView.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL', 'BTCUSD', 'EURUSD', 'NIFTY')
    """
    _controller.change_symbol(symbol)
    return f"Symbol changed to {symbol}. Waiting for chart to load."


@mcp.tool()
def change_timeframe(timeframe: str) -> str:
    """Change the chart timeframe in TradingView.

    Args:
        timeframe: Timeframe string. Examples: '1m', '5m', '15m', '1h', '4h', '1d', '1w'
    """
    _controller.change_timeframe(timeframe)
    return f"Timeframe changed to {timeframe}."


@mcp.tool()
def write_pine_script(code: str) -> str:
    """Write Pine Script code into TradingView's Pine Editor.

    This will:
    1. Open the Pine Editor panel
    2. Select all existing code
    3. Paste the new code

    Args:
        code: Complete Pine Script v5 code to write.
    """
    _controller.write_pine_script(code)
    return "Pine Script written to editor. Call compile_and_run() to execute it."


@mcp.tool()
def compile_and_run() -> str:
    """Compile the Pine Script and add the strategy to the chart.

    Saves the script, compiles it, and applies it to the active chart.
    Wait a few seconds after this for the backtest to complete.
    """
    _controller.compile_and_add_to_chart()
    return (
        "Script compiled and added to chart. "
        "Use capture_backtest_results() to read the Strategy Tester output."
    )


@mcp.tool()
def deploy_strategy(
    pine_script: str,
    symbol: str = "",
    timeframe: str = "",
) -> str:
    """Full workflow: change symbol/timeframe, write script, compile, and run.

    This is the all-in-one tool that:
    1. Changes to the specified symbol (if provided)
    2. Changes to the specified timeframe (if provided)
    3. Writes the Pine Script to the editor
    4. Compiles and adds the strategy to the chart
    5. Waits for backtest to complete

    Args:
        pine_script: Complete Pine Script v5 code.
        symbol: Optional ticker symbol to switch to.
        timeframe: Optional timeframe to switch to.
    """
    _controller.deploy_strategy(
        pine_script,
        symbol=symbol or None,
        timeframe=timeframe or None,
    )
    return (
        f"Strategy deployed"
        f"{' on ' + symbol if symbol else ''}"
        f"{' (' + timeframe + ')' if timeframe else ''}. "
        "Backtest should be running. "
        "Use capture_backtest_results() to see the results."
    )


@mcp.tool()
def capture_chart() -> str:
    """Capture a screenshot of the TradingView chart area.

    Returns a description of the chart. Use analyze_chart_screenshot() for
    detailed AI analysis of the chart.
    """
    b64 = _controller.capture_chart_area()
    return f"Chart screenshot captured ({len(b64)} bytes base64). Use analyze_chart_screenshot() for AI analysis."


@mcp.tool()
def analyze_chart_screenshot(context: str = "") -> str:
    """Capture the chart and run AI visual analysis on it.

    Uses Claude's vision to identify:
    - Trend direction and strength
    - Support/resistance levels
    - Candlestick patterns
    - Chart patterns
    - Indicator readings
    - Volume analysis
    - Trade setup recommendations

    Args:
        context: Optional context (e.g., 'Looking for reversal patterns at resistance')
    """
    b64 = _controller.capture_chart_area()
    analysis = analyze_chart(b64, context=context)
    return analysis


@mcp.tool()
def capture_backtest_results() -> str:
    """Capture and read the Strategy Tester backtest results.

    Takes a screenshot of the Strategy Tester Overview tab and uses AI
    to extract all performance metrics (net profit, win rate, profit factor,
    drawdown, Sharpe ratio, etc.).

    Returns:
        Formatted backtest results summary.
    """
    b64 = _controller.capture_backtest_overview()
    results = read_backtest_results(b64)
    return format_results_for_agent(results)


@mcp.tool()
def capture_backtest_results_raw() -> str:
    """Capture backtest results as raw JSON for programmatic use.

    Returns:
        JSON string of parsed metrics from Strategy Tester.
    """
    b64 = _controller.capture_backtest_overview()
    results = read_backtest_results(b64)
    return json.dumps(results, indent=2)


@mcp.tool()
def capture_performance_summary() -> str:
    """Capture the Performance Summary tab from Strategy Tester.

    Shows detailed performance breakdown including long/short trade statistics.
    """
    b64 = _controller.capture_backtest_performance()
    results = read_performance_summary(b64)
    return json.dumps(results, indent=2) if not results.get("parse_error") else results.get("raw_text", "Could not read")


@mcp.tool()
def capture_trades_list() -> str:
    """Capture the List of Trades from Strategy Tester.

    Shows individual trades with entry/exit prices, P&L, etc.
    """
    b64 = _controller.capture_backtest_trades()
    trades = read_trades_list(b64)
    return json.dumps(trades, indent=2)


@mcp.tool()
def check_compilation_status() -> str:
    """Check if the Pine Script compiled successfully.

    Captures the Pine Editor area and checks for error messages.

    Returns:
        Compilation status and any error messages.
    """
    b64 = _controller.capture_full_screen()
    status = read_compilation_errors(b64)

    if status.get("parse_error"):
        return f"Could not determine compilation status. Raw: {status.get('raw_text', 'N/A')}"

    if status.get("has_errors"):
        errors = status.get("errors", [])
        return f"Compilation FAILED with {len(errors)} error(s):\n" + "\n".join(f"  - {e}" for e in errors)
    else:
        return "Script compiled successfully (no errors detected)."


@mcp.tool()
def generate_strategy_pine_script(
    strategy_json: str,
    symbol: str = "",
    timeframe: str = "1d",
) -> str:
    """Generate Pine Script v5 code from a strategy definition JSON.

    Uses Claude to create production-ready TradingView Pine Script with
    proper inputs, plots, alerts, and strategy execution logic.

    Args:
        strategy_json: Strategy definition as JSON string with rules and risk_management.
        symbol: Symbol context for the strategy.
        timeframe: Timeframe context.

    Returns:
        Complete Pine Script v5 code ready to paste into TradingView.
    """
    strategy = json.loads(strategy_json)

    try:
        code = generate_pine_script(strategy, symbol=symbol, timeframe=timeframe)
    except Exception:
        code = strategy_to_pine_template(strategy)

    return code


@mcp.tool()
def generate_and_deploy_strategy(
    strategy_json: str,
    symbol: str = "",
    timeframe: str = "",
) -> str:
    """Generate Pine Script from strategy JSON AND deploy it to TradingView.

    Complete workflow:
    1. Generate Pine Script v5 from the strategy JSON
    2. Switch to the specified symbol/timeframe
    3. Write the script to TradingView's Pine Editor
    4. Compile and add to chart
    5. Wait for backtest

    Args:
        strategy_json: Strategy definition JSON string.
        symbol: Optional ticker symbol.
        timeframe: Optional timeframe.
    """
    strategy = json.loads(strategy_json)

    try:
        code = generate_pine_script(strategy, symbol=symbol, timeframe=timeframe)
    except Exception:
        code = strategy_to_pine_template(strategy)

    _controller.deploy_strategy(
        code,
        symbol=symbol or None,
        timeframe=timeframe or None,
    )

    return (
        f"Strategy '{strategy.get('name', 'unnamed')}' generated and deployed. "
        "Use capture_backtest_results() to see performance."
    )


@mcp.tool()
def scroll_chart(direction: str = "left", amount: int = 10) -> str:
    """Scroll the chart left or right to view more history.

    Args:
        direction: 'left' for older data, 'right' for newer data.
        amount: Number of scroll steps.
    """
    _controller.scroll_chart(direction, amount)
    return f"Chart scrolled {direction} by {amount} steps."


@mcp.tool()
def zoom_chart(direction: str = "in", steps: int = 3) -> str:
    """Zoom the chart in or out.

    Args:
        direction: 'in' for closer view, 'out' for wider view.
        steps: Number of zoom steps.
    """
    _controller.zoom_chart(direction, steps)
    return f"Chart zoomed {direction} by {steps} steps."


@mcp.tool()
def capture_full_screenshot() -> str:
    """Capture a full screenshot of the entire TradingView window.

    Useful for debugging or getting a complete view of the application state.
    """
    b64 = _controller.capture_full_screen()
    return f"Full screenshot captured ({len(b64)} bytes base64)."


@mcp.tool()
def get_trading_knowledge(topic: str) -> str:
    """Get the agent's built-in trading knowledge on a topic.

    Args:
        topic: One of 'price_action', 'indicators', 'volume_profile',
               'candlestick_patterns', 'all'
    """
    from knowledge.price_action import PRICE_ACTION_CONCEPTS
    from knowledge.indicators import INDICATOR_KNOWLEDGE
    from knowledge.volume_profile import VOLUME_PROFILE_KNOWLEDGE
    from knowledge.candlestick_patterns import CANDLESTICK_PATTERNS

    if topic == "price_action":
        return json.dumps(PRICE_ACTION_CONCEPTS, indent=2)
    elif topic == "indicators":
        return json.dumps(INDICATOR_KNOWLEDGE, indent=2, default=str)
    elif topic == "volume_profile":
        return json.dumps(VOLUME_PROFILE_KNOWLEDGE, indent=2)
    elif topic == "candlestick_patterns":
        summary = {k: {"type": v["type"], "description": v["description"]}
                   for k, v in CANDLESTICK_PATTERNS.items()}
        return json.dumps(summary, indent=2)
    elif topic == "all":
        return json.dumps({
            "price_action": PRICE_ACTION_CONCEPTS,
            "indicators": {cat: {k: v["description"] for k, v in inds.items()}
                          for cat, inds in INDICATOR_KNOWLEDGE.items()},
            "volume_profile": VOLUME_PROFILE_KNOWLEDGE,
            "candlestick_patterns": {k: v["type"] + ": " + v["description"]
                                     for k, v in CANDLESTICK_PATTERNS.items()},
        }, indent=2)
    else:
        return f"Unknown topic '{topic}'. Use: price_action, indicators, volume_profile, candlestick_patterns, all"


# ═════════════════════════════════════════════════════════════════════════════
#  RESOURCES
# ═════════════════════════════════════════════════════════════════════════════


@mcp.resource("tradingview://system-prompt")
def get_system_prompt() -> str:
    """Get the full trading agent system prompt with all knowledge."""
    return TRADING_AGENT_SYSTEM_PROMPT


@mcp.resource("tradingview://strategy-template")
def get_strategy_template() -> str:
    """Get a template strategy JSON that can be customized."""
    template = {
        "name": "My Strategy",
        "description": "Description of the strategy logic",
        "timeframe": "1d",
        "rules": {
            "long_entry": [
                {"indicator": "rsi_14", "operator": "<", "value": 30},
                {"indicator": "close", "operator": ">", "ref": "sma_50"},
            ],
            "long_exit": [
                {"indicator": "rsi_14", "operator": ">", "value": 70},
            ],
            "short_entry": [],
            "short_exit": [],
        },
        "risk_management": {
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "position_size_pct": 10.0,
            "max_positions": 1,
        },
    }
    return json.dumps(template, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ═════════════════════════════════════════════════════════════════════════════


@mcp.prompt()
def analyze_and_create_strategy(symbol: str = "AAPL", timeframe: str = "1d") -> str:
    """OODA loop prompt: analyze a symbol and create a Pine Script strategy.

    This prompt guides the agent through the full cycle:
    1. Observe the chart
    2. Analyze it
    3. Create a strategy
    4. Deploy and backtest in TradingView
    """
    return f"""Execute a full OODA trading analysis cycle for {symbol} on the {timeframe} timeframe.

## Step 1: OBSERVE
- Call change_symbol('{symbol}') and change_timeframe('{timeframe}')
- Call analyze_chart_screenshot() to visually analyze the current chart
- Note the trend, key levels, patterns, and indicator readings

## Step 2: ORIENT
- Based on the chart analysis, identify:
  - Current market regime (trending up/down, ranging, volatile)
  - Key support and resistance levels
  - Any active candlestick or chart patterns
  - Indicator confluence (where multiple signals agree)
  - Volume profile zones if visible (POC, VAH, VAL)

## Step 3: DECIDE
- Design a trading strategy with specific rules:
  - Entry conditions (what indicators and levels to use)
  - Exit conditions (take profit, stop loss, signal exits)
  - Risk management (position size, max drawdown tolerance)
- Create the strategy JSON definition

## Step 4: ACT
- Call generate_and_deploy_strategy() with your strategy JSON
- Wait for the backtest to complete
- Call capture_backtest_results() to read the performance
- Analyze the results — if win rate < 50% or profit factor < 1.5:
  - Identify what went wrong
  - Modify the strategy
  - Re-deploy and re-test (up to 3 iterations)

## Final Output
- Show the final strategy performance metrics
- Provide the final Pine Script code
- Give a summary of what the strategy does and why"""


@mcp.prompt()
def improve_current_strategy() -> str:
    """Prompt to improve the currently running strategy based on backtest results."""
    return """Improve the current trading strategy based on its backtest results.

## Steps
1. Call capture_backtest_results() to get current performance
2. Call capture_performance_summary() for detailed breakdown
3. Call capture_trades_list() to see individual trades
4. Call analyze_chart_screenshot() to check if strategy signals make visual sense
5. Identify the biggest weakness:
   - Low win rate → tighten entries, add confirmation
   - Low profit factor → improve risk/reward, widen TP or tighten SL
   - High drawdown → reduce position size, add trend filter
   - Few trades → loosen entry conditions
6. Generate improved Pine Script and deploy it
7. Compare new results vs old results
8. Repeat if needed (max 3 iterations)"""


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """Run the MCP server with stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
