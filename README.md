# AI Trading Agent — OODA Loop Powered by Claude Opus

An AI-powered trading agent that **directly drives TradingView Desktop** to write Pine Script strategies, run backtests, read results, and iteratively optimize — all via Claude Opus.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         OODA LOOP                                │
│                                                                  │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐      │
│  │    OBSERVE     │──>│    ORIENT     │──>│    DECIDE     │      │
│  │                │   │               │   │               │      │
│  │ • Capture TV   │   │ • Claude      │   │ • Claude      │      │
│  │   Desktop      │   │   vision      │   │   creates     │      │
│  │   chart        │   │   analysis    │   │   strategy    │      │
│  │ • Fetch data   │   │ • Price       │   │ • Entry/exit  │      │
│  │ • Compute      │   │   action      │   │   rules       │      │
│  │   indicators   │   │ • Volume      │   │ • Risk mgmt   │      │
│  │                │   │   profile     │   │               │      │
│  └───────────────┘   └───────────────┘   └──────┬────────┘      │
│       ▲                                          │               │
│       │            ┌───────────────┐             │               │
│       │            │      ACT      │<────────────┘               │
│       │            │               │                             │
│       │            │ • Generate    │                             │
│       └────────────│   Pine Script │                             │
│      (iterate)     │ • Deploy to   │                             │
│                    │   TradingView │                             │
│                    │ • Read        │                             │
│                    │   backtest    │                             │
│                    │   results     │                             │
│                    │ • Optimize    │                             │
│                    └───────────────┘                             │
└──────────────────────────────────────────────────────────────────┘
```

### Integration Flow

```
Claude Opus (AI Brain)
        │
        ├── Analyzes charts visually (vision API)
        ├── Creates trading strategies
        ├── Generates Pine Script v5
        └── Reviews results & optimizes
                │
        ┌───────┴────────┐
        │  MCP Server /  │
        │  TV Controller │
        └───────┬────────┘
                │  (pyautogui automation)
        ┌───────┴────────┐
        │   TradingView  │
        │    Desktop     │
        │                │
        │ • Pine Editor  │
        │ • Strategy     │
        │   Tester       │
        │ • Charts       │
        └────────────────┘
```

## Features

### TradingView Desktop Integration
- **Direct Pine Editor control**: Writes Pine Script directly into TradingView's editor
- **Auto-compile & deploy**: Compiles scripts and adds strategies to charts
- **Backtest result reading**: Captures Strategy Tester screenshots, uses Claude vision to extract metrics
- **Symbol & timeframe switching**: Changes charts programmatically
- **Screenshot capture**: Captures chart and strategy tester for AI analysis

### MCP Server (for Claude Code)
- Full MCP server exposing 18+ tools for TradingView automation
- Use directly from Claude Code as an MCP server
- Tools: `deploy_strategy`, `capture_backtest_results`, `analyze_chart_screenshot`, etc.

### Market Analysis
- **Visual Chart Analysis**: Claude reads candlestick charts and identifies patterns
- **Price Action**: Support/resistance, trend structure (HH/HL/LH/LL), breakouts, fair value gaps
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR, OBV, VWAP
- **Volume Profile**: POC, VAH, VAL, High/Low Volume Nodes
- **Candlestick Patterns**: 18+ patterns (doji, hammer, engulfing, morning star, etc.)

### Strategy Generation & Optimization
- AI creates strategies based on comprehensive chart + data analysis
- Generates production-ready Pine Script v5 with inputs, alerts, and overlays
- **Iterative optimization**: Deploys to TradingView → reads results → Claude improves → repeat

## Setup

### Prerequisites
- **Python 3.11+**
- **TradingView Desktop** installed and running
- **Anthropic API key** with Claude Opus access

### Installation

```bash
# 1. Clone and install
cd TV_AI_agent
pip install -r requirements.txt

# 2. Platform-specific dependencies
# Linux:
sudo apt install wmctrl xdotool scrot
# Windows:
pip install pygetwindow
# macOS: grant Accessibility permissions to Terminal/Python

# 3. Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 4. Open TradingView Desktop (must be running before using the agent)
```

### Quick Start

```bash
# TradingView mode (default) — drives TradingView Desktop directly
python main.py                              # Interactive mode
python main.py analyze AAPL                 # Full OODA cycle on Apple
python main.py analyze BTCUSD 1h 90         # BTC hourly, 90 days

# Local mode — uses built-in backtester (no TradingView needed)
python main.py --mode local analyze TSLA

# Skip optimization
python main.py --no-optimize MSFT

# Start MCP server (for Claude Code integration)
python main.py mcp
```

## Using as an MCP Server with Claude Code

The agent includes a full MCP server that you can connect to Claude Code. This lets Claude Code directly control TradingView Desktop.

### 1. Add to Claude Code settings

Add this to your Claude Code MCP config (`~/.claude/settings.json` or project `.claude/settings.json`):

```json
{
  "mcpServers": {
    "tradingview": {
      "command": "python",
      "args": ["-m", "tradingview_mcp.server"],
      "cwd": "/path/to/TV_AI_agent",
      "env": {
        "ANTHROPIC_API_KEY": "your-api-key"
      }
    }
  }
}
```

### 2. Available MCP Tools

| Tool | Description |
|------|-------------|
| `focus_tradingview` | Bring TradingView to foreground |
| `change_symbol` | Switch chart symbol |
| `change_timeframe` | Switch chart timeframe |
| `write_pine_script` | Write code to Pine Editor |
| `compile_and_run` | Compile and add strategy to chart |
| `deploy_strategy` | All-in-one: write + compile + run |
| `analyze_chart_screenshot` | Capture chart + AI visual analysis |
| `capture_backtest_results` | Read Strategy Tester metrics |
| `capture_performance_summary` | Read detailed performance breakdown |
| `capture_trades_list` | Read individual trades list |
| `check_compilation_status` | Check for Pine Script errors |
| `generate_strategy_pine_script` | Generate Pine Script from strategy JSON |
| `generate_and_deploy_strategy` | Generate + deploy in one step |
| `scroll_chart` | Scroll chart left/right |
| `zoom_chart` | Zoom chart in/out |
| `get_trading_knowledge` | Access built-in trading knowledge base |

### 3. MCP Prompts

| Prompt | Description |
|--------|-------------|
| `analyze_and_create_strategy` | Full OODA cycle for a symbol |
| `improve_current_strategy` | Optimize the running strategy |

## Project Structure

```
TV_AI_agent/
├── main.py                         # CLI entry point (interactive, analyze, mcp)
├── claude_mcp_config.json          # Example MCP config for Claude Code
├── config/
│   └── settings.py                 # Configuration & API keys
│
├── tradingview_mcp/                # ★ TradingView MCP Server
│   ├── server.py                   # MCP server (18+ tools, resources, prompts)
│   ├── tv_controller.py            # Desktop automation (pyautogui)
│   ├── screen_reader.py            # Screenshot → Claude vision → parsed data
│   └── __main__.py                 # python -m tradingview_mcp
│
├── agent/                          # AI Agent core
│   ├── ooda_agent.py               # OODA loop orchestrator (TV + local modes)
│   ├── chart_analyzer.py           # Vision-based chart analysis
│   ├── pine_script_generator.py    # Pine Script v5 generation
│   ├── strategy_optimizer.py       # Iterative optimization loop
│   └── prompts/
│       └── system_prompts.py       # System prompts with trading knowledge
│
├── knowledge/                      # Trading domain expertise
│   ├── price_action.py             # S/R, trends, FVGs, order blocks
│   ├── indicators.py               # Technical indicator computation
│   ├── volume_profile.py           # POC, VAH, VAL, HVN, LVN
│   └── candlestick_patterns.py     # 18 pattern detection
│
├── backtesting/                    # Local backtesting (fallback)
│   ├── engine.py                   # Event-driven backtester
│   ├── data_loader.py              # Yahoo Finance + CSV + synthetic data
│   └── metrics.py                  # Performance metrics
│
├── charts/                         # Chart rendering
│   └── renderer.py                 # mplfinance candlestick rendering
│
├── strategies/generated/           # Output: strategy JSON + Pine Script files
└── charts_output/                  # Output: chart screenshots
```

## How It Works

### TradingView Mode (default)

#### 1. OBSERVE
- Focuses TradingView Desktop and switches to the target symbol/timeframe
- Captures a screenshot of the live chart
- Fetches historical data for quantitative analysis (indicators, volume profile, etc.)

#### 2. ORIENT
- Sends the TradingView chart screenshot to Claude's vision API
- Claude visually identifies trends, patterns, key levels, and indicator readings
- Combines visual analysis with computed quantitative data

#### 3. DECIDE
- Claude creates a trading strategy with specific entry/exit rules
- Strategy is defined as a structured JSON with indicator conditions and risk management

#### 4. ACT
- Generates Pine Script v5 from the strategy
- **Writes the script directly into TradingView's Pine Editor**
- **Compiles and adds the strategy to the chart**
- **Waits for backtest to run, then captures Strategy Tester results**
- Uses Claude vision to read the metrics (win rate, profit factor, etc.)
- If optimization is enabled:
  - Claude reviews results + chart screenshot
  - Proposes improvements to the strategy
  - Re-generates Pine Script, re-deploys, re-reads results
  - Repeats up to 5 iterations until targets are met

### Local Mode (fallback)

Same OODA loop but uses the built-in backtesting engine instead of TradingView Desktop. Useful for testing without TradingView installed.

```bash
python main.py --mode local analyze AAPL
```

## Requirements

- Python 3.11+
- Anthropic API key (Claude Opus)
- TradingView Desktop (for TradingView mode)
- Internet connection (for market data and API calls)

### System Dependencies

| Platform | Requirements |
|----------|-------------|
| **Linux** | `wmctrl`, `xdotool`, `scrot` (for window management & screenshots) |
| **Windows** | `pygetwindow` (pip install) |
| **macOS** | Grant Accessibility permissions to Terminal/Python |
