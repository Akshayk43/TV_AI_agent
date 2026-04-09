# AI Trading Agent — OODA Loop Powered by Claude Opus

An AI-powered trading agent that uses Claude Opus to analyze markets, create TradingView Pine Script strategies, backtest them, and iteratively optimize for better performance.

## Architecture

The agent operates in an **OODA Loop** (Observe → Orient → Decide → Act):

```
┌─────────────────────────────────────────────────────┐
│                    OODA LOOP                        │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ OBSERVE  │───>│  ORIENT  │───>│  DECIDE  │      │
│  │          │    │          │    │          │      │
│  │• Fetch   │    │• Vision  │    │• Create  │      │
│  │  data    │    │  chart   │    │  strategy│      │
│  │• Compute │    │  analysis│    │  rules   │      │
│  │  indicators│  │• Price   │    │• Set risk│      │
│  │• Render  │    │  action  │    │  params  │      │
│  │  charts  │    │• Volume  │    │          │      │
│  │          │    │  profile │    │          │      │
│  └──────────┘    └──────────┘    └─────┬────┘      │
│       ▲                                │            │
│       │          ┌──────────┐          │            │
│       │          │   ACT    │<─────────┘            │
│       │          │          │                       │
│       │          │• Backtest│                       │
│       └──────────│• Optimize│                       │
│     (iterate)    │• Generate│                       │
│                  │  Pine    │                       │
│                  │  Script  │                       │
│                  └──────────┘                       │
└─────────────────────────────────────────────────────┘
```

## Features

### Market Analysis
- **Visual Chart Analysis**: Uses Claude's vision to read candlestick charts and identify patterns
- **Price Action**: Support/resistance, trend structure (HH/HL/LH/LL), breakouts, fair value gaps
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR, OBV, VWAP
- **Volume Profile**: POC, VAH, VAL, High/Low Volume Nodes
- **Candlestick Patterns**: 20+ patterns (doji, hammer, engulfing, morning star, etc.)

### Strategy Generation
- Creates strategies based on comprehensive market analysis
- Outputs both backtestable JSON rules and TradingView Pine Script v5
- Pine Scripts include inputs, alerts, and visual overlays

### Backtesting & Optimization
- Full backtesting engine with realistic commission and slippage
- Performance metrics: Win rate, profit factor, Sharpe ratio, max drawdown, etc.
- **Iterative optimization**: Claude reviews backtest results and refines the strategy over multiple iterations

## Setup

```bash
# 1. Clone and install
cd TV_AI_agent
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your Anthropic API key

# 3. Run
python main.py                          # Interactive mode
python main.py analyze AAPL             # Analyze a symbol
python main.py analyze BTCUSD 1h 90     # Custom timeframe and lookback
python main.py --no-optimize TSLA       # Skip optimization
```

## Project Structure

```
TV_AI_agent/
├── main.py                         # CLI entry point
├── config/
│   └── settings.py                 # Configuration
├── agent/
│   ├── ooda_agent.py               # Main OODA loop orchestrator
│   ├── chart_analyzer.py           # Vision-based chart analysis
│   ├── pine_script_generator.py    # Pine Script v5 generation
│   ├── strategy_optimizer.py       # Iterative optimization loop
│   └── prompts/
│       └── system_prompts.py       # All agent prompts
├── knowledge/
│   ├── price_action.py             # S/R, trends, FVGs
│   ├── indicators.py               # Technical indicator computation
│   ├── volume_profile.py           # Volume profile analysis
│   └── candlestick_patterns.py     # Pattern detection
├── backtesting/
│   ├── engine.py                   # Backtesting engine
│   ├── data_loader.py              # Market data (Yahoo Finance)
│   └── metrics.py                  # Performance metrics
├── charts/
│   └── renderer.py                 # Chart rendering (mplfinance)
├── strategies/generated/           # Output: strategy JSON + Pine Script
└── charts_output/                  # Output: chart images
```

## How It Works

### 1. OBSERVE
- Downloads market data via Yahoo Finance
- Computes all technical indicators (RSI, MACD, BB, etc.)
- Analyzes volume profile (POC, VAH, VAL)
- Detects candlestick patterns
- Identifies support/resistance levels and trend structure
- Renders a full chart with overlays

### 2. ORIENT
- Sends the rendered chart image to Claude via the vision API
- Claude visually identifies chart patterns, trend lines, and formations
- Combines visual analysis with computed indicator data
- Produces a comprehensive market analysis

### 3. DECIDE
- Claude creates a trading strategy based on the analysis
- Defines specific entry/exit rules using available indicators
- Sets risk management parameters (stop-loss, take-profit, position size)

### 4. ACT
- Backtests the strategy on historical data
- If optimization is enabled, enters an iterative improvement loop:
  - Claude reviews backtest results
  - Proposes specific improvements
  - Re-backtests the improved strategy
  - Repeats until targets are met or max iterations reached
- Generates TradingView Pine Script v5 code
- Saves all outputs (strategy JSON, Pine Script, charts, reports)

## Strategy JSON Format

```json
{
    "name": "Trend Following RSI Strategy",
    "description": "Long on RSI oversold + price above SMA 50",
    "timeframe": "1d",
    "rules": {
        "long_entry": [
            {"indicator": "rsi_14", "operator": "<", "value": 30},
            {"indicator": "close", "operator": ">", "ref": "sma_50"}
        ],
        "long_exit": [
            {"indicator": "rsi_14", "operator": ">", "value": 70}
        ]
    },
    "risk_management": {
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
        "position_size_pct": 10.0,
        "max_positions": 1
    }
}
```

## Available Indicators

| Indicator | Key | Description |
|-----------|-----|-------------|
| SMA | `sma_9`, `sma_20`, `sma_50`, `sma_200` | Simple Moving Average |
| EMA | `ema_9`, `ema_20`, `ema_50`, `ema_200` | Exponential Moving Average |
| RSI | `rsi_14` | Relative Strength Index |
| MACD | `macd`, `macd_signal`, `macd_histogram` | MACD components |
| Bollinger | `bb_upper`, `bb_middle`, `bb_lower` | Bollinger Bands |
| Stochastic | `stoch_k`, `stoch_d` | Stochastic Oscillator |
| ATR | `atr_14` | Average True Range |
| OBV | `obv` | On-Balance Volume |
| VWAP | `vwap` | Volume Weighted Average Price |
| Volume | `volume_sma_20` | Volume Moving Average |

## Requirements

- Python 3.11+
- Anthropic API key (Claude Opus access)
- Internet connection (for market data and API calls)
