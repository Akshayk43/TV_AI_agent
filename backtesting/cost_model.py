"""Cost models for backtesting.

The original engine applied `commission_pct` and `slippage_pct` multiplicatively,
which is fine for equities but wildly wrong for FX/metals where cost is
dollar-denominated via the bid/ask spread.

A cost model answers two questions per trade:
  1. entry_adjustment(ref_price, direction) → executed entry price
  2. exit_adjustment(ref_price, direction)  → executed exit price
  3. commission(price, size)                → absolute USD commission

Longs pay the ask (ref + half-spread) on entry and hit the bid (ref - half-spread)
on exit. Shorts are mirrored.
"""

from abc import ABC, abstractmethod


class CostModel(ABC):
    @abstractmethod
    def entry_price(self, ref_price: float, direction: str) -> float: ...

    @abstractmethod
    def exit_price(self, ref_price: float, direction: str) -> float: ...

    @abstractmethod
    def commission(self, price: float, size: float) -> float: ...


class PercentCost(CostModel):
    """Legacy percent-based model — preserves original engine behavior.

    Slippage is applied as ±slippage_pct of price; commission as commission_pct
    of notional.
    """

    def __init__(self, commission_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def entry_price(self, ref_price: float, direction: str) -> float:
        return ref_price * (1 + self.slippage_pct) if direction == "long" else ref_price * (1 - self.slippage_pct)

    def exit_price(self, ref_price: float, direction: str) -> float:
        return ref_price * (1 - self.slippage_pct) if direction == "long" else ref_price * (1 + self.slippage_pct)

    def commission(self, price: float, size: float) -> float:
        return price * size * self.commission_pct


class XAUUSDSpreadCost(CostModel):
    """Spot gold cost model — fixed dollar spread, no explicit commission.

    `spread` is the full bid/ask spread in USD per oz (e.g. 0.30 for a
    typical retail broker). Half the spread is applied on each side of
    the trade. An optional per-oz commission captures ECN brokers that
    bill explicitly (e.g. IC Markets raw: spread ~0.10 + $0.07/oz commission).
    """

    def __init__(self, spread: float = 0.30, commission_per_oz: float = 0.0):
        self.spread = spread
        self.half_spread = spread / 2.0
        self.commission_per_oz = commission_per_oz

    def entry_price(self, ref_price: float, direction: str) -> float:
        # Long buys at the ask (higher); short sells at the bid (lower)
        return ref_price + self.half_spread if direction == "long" else ref_price - self.half_spread

    def exit_price(self, ref_price: float, direction: str) -> float:
        # Long sells at the bid (lower); short buys at the ask (higher)
        return ref_price - self.half_spread if direction == "long" else ref_price + self.half_spread

    def commission(self, price: float, size: float) -> float:
        return self.commission_per_oz * size


def default_cost_for_symbol(symbol: str) -> CostModel:
    """Pick a reasonable cost model from the instrument symbol."""
    s = symbol.upper().replace("=X", "").replace("/", "").replace("_", "")
    if s.startswith("XAUUSD") or s in ("GOLD", "GC", "GCUSD"):
        from config.xauusd import TYPICAL_SPREAD
        return XAUUSDSpreadCost(spread=TYPICAL_SPREAD)
    return PercentCost()
