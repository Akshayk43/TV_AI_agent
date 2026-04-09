"""Pine Script v5 code generation from strategy definitions."""

import json
import anthropic

from config.settings import ANTHROPIC_API_KEY, MODEL_NAME
from agent.prompts.system_prompts import PINE_SCRIPT_PROMPT


def generate_pine_script(strategy: dict, symbol: str = "", timeframe: str = "1d") -> str:
    """Use Claude to generate Pine Script v5 code from a strategy definition.

    Args:
        strategy: Strategy dict with rules and risk_management.
        symbol: Ticker symbol for context.
        timeframe: Timeframe for context.

    Returns:
        Pine Script v5 source code string.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_prompt = f"""Generate a complete TradingView Pine Script v5 strategy based on this definition:

**Symbol**: {symbol}
**Timeframe**: {timeframe}

**Strategy Definition**:
```json
{json.dumps(strategy, indent=2)}
```

Requirements:
1. Translate ALL entry/exit conditions from the JSON rules into Pine Script
2. Implement the risk management settings (stop-loss, take-profit, position sizing)
3. Add input() parameters for all key values so users can tune them
4. Add plot overlays for key indicators used in the strategy
5. Add background color to highlight long/short zones
6. Include alertcondition() calls for entry/exit signals
7. Add a strategy performance table if possible

Output ONLY the Pine Script code, no explanation needed."""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=8000,
        system=PINE_SCRIPT_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    code = response.content[0].text

    # Extract code from markdown fence if present
    if "```" in code:
        lines = code.split("\n")
        in_code = False
        code_lines = []
        for line in lines:
            if line.strip().startswith("```"):
                if in_code:
                    break
                in_code = True
                continue
            if in_code:
                code_lines.append(line)
        if code_lines:
            code = "\n".join(code_lines)

    return code


def strategy_to_pine_template(strategy: dict) -> str:
    """Generate a basic Pine Script template without calling the API.

    Useful as a fallback or for quick generation.
    """
    name = strategy.get("name", "AI Strategy")
    desc = strategy.get("description", "AI-generated trading strategy")
    rules = strategy.get("rules", {})
    risk = strategy.get("risk_management", {})

    sl_pct = risk.get("stop_loss_pct", 2.0)
    tp_pct = risk.get("take_profit_pct", 4.0)

    def conditions_to_pine(conditions: list[dict]) -> str:
        parts = []
        for c in conditions:
            ind = c.get("indicator", "close")
            op = c.get("operator", ">")
            if "ref" in c:
                ref = c["ref"]
                ref_pine = _indicator_to_pine(ref)
            elif "value" in c:
                ref_pine = str(c["value"])
            else:
                continue

            ind_pine = _indicator_to_pine(ind)

            if op == "crosses_above":
                parts.append(f"ta.crossover({ind_pine}, {ref_pine})")
            elif op == "crosses_below":
                parts.append(f"ta.crossunder({ind_pine}, {ref_pine})")
            else:
                parts.append(f"{ind_pine} {op} {ref_pine}")

        return " and ".join(parts) if parts else "false"

    long_entry = conditions_to_pine(rules.get("long_entry", []))
    long_exit = conditions_to_pine(rules.get("long_exit", []))
    short_entry = conditions_to_pine(rules.get("short_entry", []))
    short_exit = conditions_to_pine(rules.get("short_exit", []))

    # Collect all indicators used
    all_indicators = set()
    for rule_list in rules.values():
        for cond in rule_list:
            all_indicators.add(cond.get("indicator", ""))
            if "ref" in cond:
                all_indicators.add(cond["ref"])
    all_indicators.discard("")

    indicator_defs = _generate_indicator_definitions(all_indicators)

    pine = f"""//@version=5
strategy("{name}", overlay=true, default_qty_type=strategy.percent_of_equity,
         default_qty_value=10, initial_capital=100000, commission_type=strategy.commission.percent,
         commission_value=0.1)

// ── Description ──────────────────────────────────────────────────────────────
// {desc}

// ── Inputs ───────────────────────────────────────────────────────────────────
i_sl_pct = input.float({sl_pct}, "Stop Loss %", minval=0.1, step=0.1)
i_tp_pct = input.float({tp_pct}, "Take Profit %", minval=0.1, step=0.1)

// ── Indicator Calculations ───────────────────────────────────────────────────
{indicator_defs}

// ── Entry Conditions ─────────────────────────────────────────────────────────
longCondition = {long_entry}
shortCondition = {short_entry}

// ── Exit Conditions ──────────────────────────────────────────────────────────
longExitCondition = {long_exit}
shortExitCondition = {short_exit}

// ── Strategy Execution ───────────────────────────────────────────────────────
if longCondition
    strategy.entry("Long", strategy.long)
    strategy.exit("Long TP/SL", "Long",
                   profit=close * i_tp_pct / 100 / syminfo.mintick,
                   loss=close * i_sl_pct / 100 / syminfo.mintick)

if shortCondition
    strategy.entry("Short", strategy.short)
    strategy.exit("Short TP/SL", "Short",
                   profit=close * i_tp_pct / 100 / syminfo.mintick,
                   loss=close * i_sl_pct / 100 / syminfo.mintick)

if longExitCondition
    strategy.close("Long")

if shortExitCondition
    strategy.close("Short")

// ── Plots ────────────────────────────────────────────────────────────────────
plotshape(longCondition, "Long Signal", shape.triangleup, location.belowbar, color.new(color.green, 0), size=size.small)
plotshape(shortCondition, "Short Signal", shape.triangledown, location.abovebar, color.new(color.red, 0), size=size.small)

// ── Background ───────────────────────────────────────────────────────────────
bgcolor(strategy.position_size > 0 ? color.new(color.green, 90) : strategy.position_size < 0 ? color.new(color.red, 90) : na)

// ── Alerts ───────────────────────────────────────────────────────────────────
alertcondition(longCondition, "Long Entry", "Long entry signal triggered")
alertcondition(shortCondition, "Short Entry", "Short entry signal triggered")
alertcondition(longExitCondition, "Long Exit", "Long exit signal triggered")
alertcondition(shortExitCondition, "Short Exit", "Short exit signal triggered")
"""
    return pine


def _indicator_to_pine(name: str) -> str:
    """Map internal indicator names to Pine Script variable names."""
    mapping = {
        "close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "sma_9": "sma9",
        "sma_20": "sma20",
        "sma_50": "sma50",
        "sma_200": "sma200",
        "ema_9": "ema9",
        "ema_20": "ema20",
        "ema_50": "ema50",
        "ema_200": "ema200",
        "rsi_14": "rsi14",
        "macd": "macdLine",
        "macd_signal": "signalLine",
        "macd_histogram": "hist",
        "atr_14": "atr14",
        "bb_upper": "bbUpper",
        "bb_middle": "bbMiddle",
        "bb_lower": "bbLower",
        "stoch_k": "stochK",
        "stoch_d": "stochD",
        "obv": "obvValue",
        "vwap": "vwapValue",
        "volume_sma_20": "volSma20",
    }
    return mapping.get(name, name)


def _generate_indicator_definitions(indicators: set[str]) -> str:
    """Generate Pine Script variable definitions for the indicators used."""
    lines = []
    seen = set()

    defs = {
        "sma_9": 'sma9 = ta.sma(close, input.int(9, "SMA 9 Length"))',
        "sma_20": 'sma20 = ta.sma(close, input.int(20, "SMA 20 Length"))',
        "sma_50": 'sma50 = ta.sma(close, input.int(50, "SMA 50 Length"))',
        "sma_200": 'sma200 = ta.sma(close, input.int(200, "SMA 200 Length"))',
        "ema_9": 'ema9 = ta.ema(close, input.int(9, "EMA 9 Length"))',
        "ema_20": 'ema20 = ta.ema(close, input.int(20, "EMA 20 Length"))',
        "ema_50": 'ema50 = ta.ema(close, input.int(50, "EMA 50 Length"))',
        "ema_200": 'ema200 = ta.ema(close, input.int(200, "EMA 200 Length"))',
        "rsi_14": 'rsi14 = ta.rsi(close, input.int(14, "RSI Length"))',
        "macd": '[macdLine, signalLine, hist] = ta.macd(close, input.int(12, "MACD Fast"), input.int(26, "MACD Slow"), input.int(9, "MACD Signal"))',
        "macd_signal": '[macdLine, signalLine, hist] = ta.macd(close, input.int(12, "MACD Fast"), input.int(26, "MACD Slow"), input.int(9, "MACD Signal"))',
        "macd_histogram": '[macdLine, signalLine, hist] = ta.macd(close, input.int(12, "MACD Fast"), input.int(26, "MACD Slow"), input.int(9, "MACD Signal"))',
        "atr_14": 'atr14 = ta.atr(input.int(14, "ATR Length"))',
        "bb_upper": '[bbMiddle, bbUpper, bbLower] = ta.bb(close, input.int(20, "BB Length"), input.float(2.0, "BB StdDev"))',
        "bb_middle": '[bbMiddle, bbUpper, bbLower] = ta.bb(close, input.int(20, "BB Length"), input.float(2.0, "BB StdDev"))',
        "bb_lower": '[bbMiddle, bbUpper, bbLower] = ta.bb(close, input.int(20, "BB Length"), input.float(2.0, "BB StdDev"))',
        "stoch_k": 'stochK = ta.sma(ta.stoch(close, high, low, input.int(14, "Stoch Length")), 3)\nstochD = ta.sma(stochK, 3)',
        "stoch_d": 'stochK = ta.sma(ta.stoch(close, high, low, input.int(14, "Stoch Length")), 3)\nstochD = ta.sma(stochK, 3)',
        "obv": "obvValue = ta.obv",
        "vwap": "vwapValue = ta.vwap(hlc3)",
        "volume_sma_20": 'volSma20 = ta.sma(volume, input.int(20, "Volume SMA Length"))',
    }

    for ind in sorted(indicators):
        if ind in defs and defs[ind] not in seen:
            lines.append(defs[ind])
            seen.add(defs[ind])

    # Add plot lines for moving averages
    plot_lines = []
    for ind in sorted(indicators):
        pine_name = _indicator_to_pine(ind)
        if ind.startswith("sma_"):
            plot_lines.append(f'plot({pine_name}, "{ind.upper()}", color=color.blue, linewidth=1)')
        elif ind.startswith("ema_"):
            plot_lines.append(f'plot({pine_name}, "{ind.upper()}", color=color.orange, linewidth=1)')

    if plot_lines:
        lines.append("")
        lines.extend(plot_lines)

    return "\n".join(lines)
