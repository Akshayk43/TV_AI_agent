"""Chart rendering for visual analysis by Claude (vision API)."""

import os
import io
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from knowledge.indicators import compute_sma, compute_ema, compute_rsi, compute_macd, compute_bollinger_bands
from knowledge.volume_profile import compute_volume_profile
from config.settings import CHART_WIDTH, CHART_HEIGHT, CHARTS_OUTPUT_DIR


def render_candlestick_chart(
    df: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "1d",
    show_volume: bool = True,
    show_indicators: list[str] | None = None,
    show_volume_profile: bool = False,
    show_support_resistance: bool = False,
    sr_levels: dict | None = None,
    trades: list[dict] | None = None,
    save_path: str | None = None,
    last_n_bars: int | None = None,
) -> str:
    """Render a candlestick chart and return as base64-encoded PNG.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        symbol: Ticker symbol for the title.
        timeframe: Timeframe string for the title.
        show_volume: Show volume bars.
        show_indicators: List of indicator overlays — 'sma_20', 'ema_50', 'bb', etc.
        show_volume_profile: Show volume profile on the right side.
        show_support_resistance: Show S/R horizontal lines.
        sr_levels: Dict with 'support' and 'resistance' lists.
        trades: List of trade dicts to plot entry/exit markers.
        save_path: Optional file path to save the PNG.
        last_n_bars: Only show the last N bars.

    Returns:
        Base64-encoded PNG string.
    """
    plot_df = df.copy()
    if last_n_bars and last_n_bars < len(plot_df):
        plot_df = plot_df.iloc[-last_n_bars:]

    # Ensure DatetimeIndex
    if not isinstance(plot_df.index, pd.DatetimeIndex):
        plot_df.index = pd.to_datetime(plot_df.index)

    # ── Build overlay plots ───────────────────────────────────────────────
    add_plots = []
    if show_indicators:
        for ind in show_indicators:
            if ind.startswith("sma_"):
                period = int(ind.split("_")[1])
                vals = compute_sma(plot_df["close"], period)
                add_plots.append(mpf.make_addplot(vals, color="blue", width=1.2, label=ind))
            elif ind.startswith("ema_"):
                period = int(ind.split("_")[1])
                vals = compute_ema(plot_df["close"], period)
                add_plots.append(mpf.make_addplot(vals, color="orange", width=1.2, label=ind))
            elif ind == "bb":
                bb = compute_bollinger_bands(plot_df["close"])
                add_plots.append(mpf.make_addplot(bb["upper"], color="gray", linestyle="dashed", width=0.8))
                add_plots.append(mpf.make_addplot(bb["middle"], color="gray", width=0.8))
                add_plots.append(mpf.make_addplot(bb["lower"], color="gray", linestyle="dashed", width=0.8))

    # ── Trade markers ─────────────────────────────────────────────────────
    if trades:
        buy_signals = pd.Series(np.nan, index=plot_df.index)
        sell_signals = pd.Series(np.nan, index=plot_df.index)
        for t in trades:
            entry_bar = t.get("entry_bar", 0)
            exit_bar = t.get("exit_bar", 0)
            if entry_bar < len(plot_df):
                buy_signals.iloc[entry_bar] = plot_df["low"].iloc[entry_bar] * 0.995
            if exit_bar < len(plot_df):
                sell_signals.iloc[exit_bar] = plot_df["high"].iloc[exit_bar] * 1.005

        if buy_signals.notna().any():
            add_plots.append(mpf.make_addplot(buy_signals, type="scatter", markersize=100,
                                               marker="^", color="green"))
        if sell_signals.notna().any():
            add_plots.append(mpf.make_addplot(sell_signals, type="scatter", markersize=100,
                                               marker="v", color="red"))

    # ── Plot ──────────────────────────────────────────────────────────────
    dpi = 100
    figsize = (CHART_WIDTH / dpi, CHART_HEIGHT / dpi)

    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        marketcolors=mpf.make_marketcolors(
            up="green", down="red",
            edge="inherit",
            wick="inherit",
            volume={"up": "green", "down": "red"},
        ),
        gridstyle="-",
        gridcolor="#2a2a2a",
        facecolor="#1a1a1a",
        figcolor="#1a1a1a",
        y_on_right=True,
    )

    kwargs = {
        "type": "candle",
        "style": style,
        "volume": show_volume,
        "figsize": figsize,
        "title": f"\n{symbol} — {timeframe}",
        "warn_too_much_data": 10000,
        "returnfig": True,
    }

    if add_plots:
        kwargs["addplot"] = add_plots

    fig, axes = mpf.plot(plot_df, **kwargs)

    # ── Volume profile overlay ────────────────────────────────────────────
    if show_volume_profile:
        vp = compute_volume_profile(plot_df, num_bins=40)
        profile = vp["profile"]
        prices = list(profile.keys())
        volumes = list(profile.values())
        max_vol = max(volumes) if volumes else 1

        # Draw horizontal volume bars on the price axis
        ax_price = axes[0]
        x_max = len(plot_df)
        bar_width = x_max * 0.15  # 15% of chart width

        for price, vol in zip(prices, volumes):
            width = (vol / max_vol) * bar_width
            color = "#4444ff55" if price < vp["poc"] else "#ff444455"
            if abs(price - vp["poc"]) < (prices[1] - prices[0]) * 1.5:
                color = "#ffff0066"
            ax_price.barh(price, width, left=x_max - bar_width, height=(prices[1] - prices[0]) * 0.8,
                         color=color, edgecolor="none")

        # Mark POC, VAH, VAL
        ax_price.axhline(y=vp["poc"], color="yellow", linestyle="--", linewidth=1, alpha=0.7)
        ax_price.axhline(y=vp["vah"], color="cyan", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_price.axhline(y=vp["val"], color="cyan", linestyle="--", linewidth=0.8, alpha=0.5)

    # ── Support / Resistance lines ────────────────────────────────────────
    if show_support_resistance and sr_levels:
        ax_price = axes[0]
        for s in sr_levels.get("support", []):
            ax_price.axhline(y=s, color="green", linestyle=":", linewidth=0.8, alpha=0.6)
        for r in sr_levels.get("resistance", []):
            ax_price.axhline(y=r, color="red", linestyle=":", linewidth=0.8, alpha=0.6)

    # Title styling
    for ax in axes:
        ax.title.set_color("white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333333")

    # ── Export ─────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            buf.seek(0)
            f.write(buf.read())

    return b64


def render_multi_timeframe(
    dfs: dict[str, pd.DataFrame],
    symbol: str = "",
) -> str:
    """Render multiple timeframes side by side and return base64 PNG."""
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(CHART_WIDTH / 100 * n, CHART_HEIGHT / 100))
    fig.patch.set_facecolor("#1a1a1a")

    if n == 1:
        axes = [axes]

    for ax, (tf, df) in zip(axes, dfs.items()):
        ax.set_facecolor("#1a1a1a")
        ax.set_title(f"{symbol} — {tf}", color="white")

        # Simple candlestick rendering via matplotlib
        for i in range(len(df)):
            row = df.iloc[i]
            color = "green" if row["close"] >= row["open"] else "red"
            ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.5)
            ax.plot([i, i], [row["open"], row["close"]], color=color, linewidth=2.5)

        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333333")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_equity_curve(equity_curve: list[float], title: str = "Equity Curve") -> str:
    """Render equity curve and return base64 PNG."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    ax.plot(equity_curve, color="cyan", linewidth=1.5)
    ax.axhline(y=equity_curve[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.fill_between(range(len(equity_curve)), equity_curve, equity_curve[0],
                     where=[e >= equity_curve[0] for e in equity_curve],
                     color="green", alpha=0.2)
    ax.fill_between(range(len(equity_curve)), equity_curve, equity_curve[0],
                     where=[e < equity_curve[0] for e in equity_curve],
                     color="red", alpha=0.2)
    ax.set_title(title, color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.set_xlabel("Bar", color="white")
    ax.set_ylabel("Equity ($)", color="white")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
