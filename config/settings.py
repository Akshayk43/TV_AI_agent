"""Configuration settings for the AI Trading Agent."""

import os
from dotenv import load_dotenv

load_dotenv()

# Anthropic API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL_NAME = "claude-opus-4-20250514"

# Trading defaults
DEFAULT_SYMBOL = "AAPL"
DEFAULT_TIMEFRAME = "1d"
DEFAULT_LOOKBACK_DAYS = 365

# Backtesting defaults
INITIAL_CAPITAL = 100_000.0
COMMISSION_PCT = 0.001  # 0.1%
SLIPPAGE_PCT = 0.0005   # 0.05%

# Strategy optimization
MAX_OPTIMIZATION_ITERATIONS = 5
TARGET_WIN_RATE = 0.55
TARGET_PROFIT_FACTOR = 1.5
TARGET_SHARPE_RATIO = 1.0

# Chart rendering
CHART_WIDTH = 1600
CHART_HEIGHT = 900
CHART_STYLE = "charles"

# Output directories
STRATEGIES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "strategies", "generated")
CHARTS_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "charts_output")

os.makedirs(STRATEGIES_DIR, exist_ok=True)
os.makedirs(CHARTS_OUTPUT_DIR, exist_ok=True)
