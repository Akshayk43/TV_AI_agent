"""Screen reader — uses Claude's vision API to extract data from TradingView screenshots.

Reads backtest results, chart patterns, and strategy tester metrics from
screenshots captured by the tv_controller.
"""

import json
import anthropic

from config.settings import ANTHROPIC_API_KEY, MODEL_NAME


def _make_vision_request(image_b64: str, prompt: str, system: str = "") -> str:
    """Send an image + prompt to Claude's vision API."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4000,
        system=system or "You are a precise data extraction assistant. Extract exactly what is asked.",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    return response.content[0].text


def read_backtest_results(overview_b64: str) -> dict:
    """Read backtest metrics from a Strategy Tester Overview screenshot.

    Extracts: net profit, total trades, win rate, profit factor,
    max drawdown, Sharpe ratio, etc.

    Returns:
        Dict of parsed metrics.
    """
    prompt = """Extract ALL backtest metrics visible in this TradingView Strategy Tester screenshot.

Return a JSON object with these fields (use null if not visible):
{
    "net_profit": <number or string>,
    "net_profit_pct": <number>,
    "total_closed_trades": <int>,
    "win_rate_pct": <number>,
    "profit_factor": <number>,
    "max_drawdown": <number or string>,
    "max_drawdown_pct": <number>,
    "sharpe_ratio": <number>,
    "sortino_ratio": <number>,
    "avg_trade": <number or string>,
    "avg_winning_trade": <number or string>,
    "avg_losing_trade": <number or string>,
    "largest_winning_trade": <number or string>,
    "largest_losing_trade": <number or string>,
    "avg_bars_in_trade": <number>,
    "commission_paid": <number or string>,
    "total_open_trades": <int>,
    "number_winning_trades": <int>,
    "number_losing_trades": <int>,
    "gross_profit": <number or string>,
    "gross_loss": <number or string>,
    "buy_and_hold_return": <number or string>,
    "buy_and_hold_return_pct": <number>,
    "max_contracts_held": <number>,
    "open_pl": <number or string>,
    "ratio_avg_win_avg_loss": <number>,
    "max_consecutive_wins": <int>,
    "max_consecutive_losses": <int>
}

Output ONLY the JSON, no other text. Parse numbers from the screenshot precisely."""

    text = _make_vision_request(overview_b64, prompt)

    # Extract JSON
    try:
        if "```" in text:
            start = text.index("```")
            if text[start:start + 7] == "```json":
                start += 7
            else:
                start += 3
            end = text.index("```", start)
            text = text[start:end].strip()

        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"raw_text": text, "parse_error": True}


def read_performance_summary(performance_b64: str) -> dict:
    """Read the Performance Summary tab data."""
    prompt = """Extract the performance summary data from this TradingView Strategy Tester screenshot.

Return a JSON object with all visible metrics. Include any tables or statistics shown.
Group them logically (e.g., "overall", "long_trades", "short_trades" if such breakdown exists).

Output ONLY the JSON, no other text."""

    text = _make_vision_request(performance_b64, prompt)

    try:
        if "```" in text:
            start = text.index("```")
            if text[start:start + 7] == "```json":
                start += 7
            else:
                start += 3
            end = text.index("```", start)
            text = text[start:end].strip()
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"raw_text": text, "parse_error": True}


def read_trades_list(trades_b64: str) -> list[dict]:
    """Read the List of Trades from a Strategy Tester screenshot.

    Returns:
        List of trade dicts with entry/exit info.
    """
    prompt = """Extract the list of trades from this TradingView Strategy Tester screenshot.

Return a JSON array of trade objects. Each trade should have:
{
    "trade_number": <int>,
    "type": "Long" or "Short",
    "signal": <entry signal name>,
    "date_time": <entry datetime string>,
    "price": <entry price>,
    "contracts": <number>,
    "profit": <profit/loss string>,
    "profit_pct": <percentage>,
    "cumulative_profit": <running total>,
    "run_up": <max favorable>,
    "drawdown": <max adverse>
}

Extract as many trades as visible. Output ONLY the JSON array."""

    text = _make_vision_request(trades_b64, prompt)

    try:
        if "```" in text:
            start = text.index("```")
            if text[start:start + 7] == "```json":
                start += 7
            else:
                start += 3
            end = text.index("```", start)
            text = text[start:end].strip()
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return [{"raw_text": text, "parse_error": True}]


def analyze_chart(chart_b64: str, context: str = "") -> str:
    """Full visual analysis of a TradingView chart screenshot.

    Returns:
        Detailed technical analysis text.
    """
    prompt = f"""Analyze this TradingView chart in detail.

Context: {context}

Provide a comprehensive analysis:
1. **Instrument & Timeframe**: What symbol and timeframe is shown?
2. **Current Trend**: Direction, strength, structure (HH/HL or LH/LL)
3. **Key Levels**: Support, resistance, round numbers visible
4. **Candlestick Patterns**: Any significant patterns in recent bars
5. **Chart Patterns**: Triangles, channels, H&S, flags, wedges, etc.
6. **Indicators**: Read any visible indicators (moving averages, RSI panel, MACD, etc.)
7. **Volume**: If visible, assess buying/selling pressure
8. **Volume Profile**: If visible, identify POC, VAH, VAL
9. **Strategy Signals**: If entry/exit markers are visible, assess their quality
10. **Overall Bias**: Bullish/bearish/neutral with confidence (1-10)
11. **Trade Idea**: If a setup exists, describe entry, stop, target

Be specific about price levels you can read from the chart."""

    system = (
        "You are an expert technical analyst with deep knowledge of price action, "
        "candlestick patterns, chart patterns, indicators, and volume profile. "
        "You are looking at a real TradingView chart. Be precise about what you see."
    )

    return _make_vision_request(chart_b64, prompt, system)


def read_compilation_errors(editor_b64: str) -> dict:
    """Read any Pine Script compilation errors from the editor screenshot.

    Returns:
        Dict with 'has_errors', 'errors' list, and 'raw_text'.
    """
    prompt = """Look at this TradingView Pine Editor screenshot.

Are there any compilation errors visible? Check for:
- Red error messages at the bottom of the editor
- Red underlines in the code
- Error popups or warnings

Return a JSON object:
{
    "has_errors": true/false,
    "errors": ["error message 1", "error message 2"],
    "warnings": ["warning 1"],
    "status": "compiled_ok" or "compilation_failed" or "unknown"
}

Output ONLY the JSON."""

    text = _make_vision_request(editor_b64, prompt)

    try:
        if "```" in text:
            start = text.index("```")
            if text[start:start + 7] == "```json":
                start += 7
            else:
                start += 3
            end = text.index("```", start)
            text = text[start:end].strip()
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"has_errors": None, "raw_text": text, "parse_error": True}


def format_results_for_agent(results: dict) -> str:
    """Format parsed backtest results into a readable summary for the agent."""
    if results.get("parse_error"):
        return f"Could not parse results cleanly. Raw text:\n{results.get('raw_text', 'N/A')}"

    lines = ["=== TradingView Backtest Results ==="]

    fields = [
        ("Net Profit", "net_profit"),
        ("Net Profit %", "net_profit_pct"),
        ("Total Trades", "total_closed_trades"),
        ("Win Rate", "win_rate_pct"),
        ("Profit Factor", "profit_factor"),
        ("Max Drawdown", "max_drawdown"),
        ("Max Drawdown %", "max_drawdown_pct"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Sortino Ratio", "sortino_ratio"),
        ("Avg Trade", "avg_trade"),
        ("Avg Win", "avg_winning_trade"),
        ("Avg Loss", "avg_losing_trade"),
        ("Max Consec. Wins", "max_consecutive_wins"),
        ("Max Consec. Losses", "max_consecutive_losses"),
        ("Buy & Hold Return", "buy_and_hold_return_pct"),
    ]

    for label, key in fields:
        val = results.get(key)
        if val is not None:
            if isinstance(val, float):
                lines.append(f"  {label}: {val:.2f}")
            else:
                lines.append(f"  {label}: {val}")

    return "\n".join(lines)
