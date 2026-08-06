"""
VWAP Breakout Strategy.

Session VWAP is computed from the recent trade tape (market_data.TradeStore,
same source observation_engine.py's VWAP filter reads). Long when price is
trading meaningfully above VWAP with buy-side volume dominating the
window, short when meaningfully below with sell-side volume dominating.

Much simpler than observation_engine.py: one distance-from-VWAP test plus
one volume-confirmation test, no trend/candle checks, no observation
window or expiry. Every tick is evaluated fresh from scratch, same "no
direction locked in across ticks" philosophy as the other strategies —
see observation_engine.py's module docstring for why that matters for a
fast scalp read.

Implements strategy.base.StrategyEngine — see that module's docstring
for the interface tracker.py talks to. Swap to this strategy by setting
tracker.py's STRATEGY_NAME = "vwap_stg".
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from market_data import MarketDataStore, TradeStore, Signal, DEFAULT_SYMBOL_WHITELIST
from .base import StrategyContext, StrategyEngine

log = logging.getLogger("okx_futures.vwap_stg")


def compute_vwap(trades: List[dict]) -> Optional[float]:
    """Volume-weighted average price across `trades`: sum(price*qty) /
    sum(qty). Returns None if there's no volume to weight against."""
    total_qty = sum(t["qty"] for t in trades)
    if total_qty <= 0:
        return None
    return sum(t["price"] * t["qty"] for t in trades) / total_qty


def compute_side_volume_ratio(trades: List[dict], side: str) -> float:
    """Fraction (0-1) of the window's total volume that traded on
    `side` ("buy"/"sell"). 0.0 if the window has no volume at all."""
    side_vol = sum(t["qty"] for t in trades if t["side"] == side)
    total_vol = sum(t["qty"] for t in trades)
    return side_vol / total_vol if total_vol > 0 else 0.0


@dataclass
class VwapConfig:
    window_ms: int = 600_000  # 10 minutes of trade tape for the session VWAP
    min_data_trade_count: int = 20  # don't trust a VWAP built from too few prints
    breakout_pct: float = 0.0025  # price must sit at least this far (0.25%) from VWAP to count as a breakout
    min_side_volume_ratio: float = 0.60  # the breakout side must own at least this share of the window's volume
    # Only symbols in this set are ever accepted into the watchlist —
    # same hard backstop observation_engine.py uses.
    symbol_whitelist: Optional[frozenset] = field(default_factory=lambda: DEFAULT_SYMBOL_WHITELIST)


@dataclass
class VwapCandidate:
    symbol: str
    direction: str = ""
    vwap: Optional[float] = None
    price: float = 0.0
    distance_pct: float = 0.0
    side_volume_ratio: float = 0.0
    data_ready: bool = False
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.started_at

    def status_line(self) -> str:
        vwap_text = f"{self.vwap:.6g}" if self.vwap is not None else "-"
        base = (
            f"{self.symbol} direction={self.direction or '-'} price={self.price:.6g} "
            f"vwap={vwap_text} dist={self.distance_pct:+.2%} side_vol={self.side_volume_ratio:.0%}"
        )
        if not self.data_ready:
            base += " (warming up)"
        return base


class VwapStrategy(StrategyEngine):
    name = "vwap_stg"

    def __init__(
        self,
        trade_store: TradeStore,
        market_data: MarketDataStore,
        config: Optional[VwapConfig] = None,
    ) -> None:
        self._trade_store = trade_store
        self._market_data = market_data
        self.config = config or VwapConfig()
        self._candidates: Dict[str, VwapCandidate] = {}
        self._lock = asyncio.Lock()

    async def sync_watchlist(self, watchlist_symbols) -> None:
        watchlist_symbols = set(watchlist_symbols)
        whitelist = self.config.symbol_whitelist
        if whitelist:
            rejected = watchlist_symbols - whitelist
            watchlist_symbols &= whitelist
            if rejected:
                log.debug(f"[vwap] ignoring {len(rejected)} non-whitelisted symbol(s): {sorted(rejected)}")
        async with self._lock:
            for symbol in watchlist_symbols:
                if symbol not in self._candidates:
                    self._candidates[symbol] = VwapCandidate(symbol=symbol)
                    log.info(f"[vwap] {symbol} added — watching session VWAP")
            dropped = [s for s in self._candidates if s not in watchlist_symbols]
            for symbol in dropped:
                del self._candidates[symbol]

    async def snapshot(self) -> List[VwapCandidate]:
        async with self._lock:
            return list(self._candidates.values())

    async def evaluate(self, symbol: str) -> Optional[Signal]:
        cfg = self.config
        async with self._lock:
            candidate = self._candidates.get(symbol)
        if candidate is None:
            return None

        market = await self._market_data.get(symbol)
        if not market:
            return None
        price = market["last_price"]
        candidate.price = price

        trades = await self._trade_store.get_window(symbol, cfg.window_ms)
        candidate.data_ready = len(trades) >= cfg.min_data_trade_count
        if not candidate.data_ready:
            return None

        vwap = compute_vwap(trades)
        candidate.vwap = vwap
        if not vwap:
            return None

        distance_pct = (price - vwap) / vwap
        candidate.distance_pct = distance_pct

        if distance_pct >= cfg.breakout_pct:
            direction, side = "long", "buy"
        elif distance_pct <= -cfg.breakout_pct:
            direction, side = "short", "sell"
        else:
            # Inside the VWAP band — no breakout this tick, nothing
            # carried forward, re-read fresh next tick.
            candidate.direction = ""
            return None

        side_ratio = compute_side_volume_ratio(trades, side)
        candidate.side_volume_ratio = side_ratio
        candidate.direction = direction

        if side_ratio < cfg.min_side_volume_ratio:
            return None

        log.info(f"[vwap] {symbol} ACCEPTED — {candidate.status_line()}")
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=1.0,
            entry_price=price,
            take_profit=price,  # unused — execution_engine computes its own TP/SL
            stop_loss=price,
            timestamp=time.time(),
            reasons=[
                f"vwap_distance={distance_pct:+.2%}",
                f"side_volume_ratio={side_ratio:.0%}",
            ],
        )


def build(ctx: StrategyContext) -> VwapStrategy:
    """strategy.load_strategy()'s entry point — see strategy/base.py's
    module docstring for the contract every strategy module follows."""
    cfg = ctx.build_config(VwapConfig)
    return VwapStrategy(ctx.trade_store, ctx.market_data, config=cfg)
