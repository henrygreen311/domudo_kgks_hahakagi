"""
EMA Crossover Strategy.

Classic fast/slow EMA crossover read off closed candles, fetched the
same way observation_engine.py's trend check does (via
okx_client.get_candles, passed through as `candle_fetcher`). Long the
tick the fast EMA crosses above the slow EMA, short the tick it crosses
below.

This is the only one of the three bundled strategies that never touches
the trade tape (market_data.TradeStore) at all — candles are the only
input — which is exactly the point of the strategy.base.StrategyContext
design: each strategy only takes what it actually needs.

Implements strategy.base.StrategyEngine — see that module's docstring
for the interface tracker.py talks to. Swap to this strategy by setting
tracker.py's STRATEGY_NAME = "ema_stg".
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from market_data import MarketDataStore, Signal, DEFAULT_SYMBOL_WHITELIST
from .base import StrategyContext, StrategyEngine

log = logging.getLogger("okx_futures.ema_stg")

CandleFetcher = Callable[[str, str, int], Awaitable[List[dict]]]


def compute_ema_series(closes: List[float], period: int) -> List[float]:
    """Standard EMA, seeded with a simple average of the first `period`
    closes. `closes` must be ordered oldest-to-newest. Returns one EMA
    value per close from index `period - 1` onward — empty if there
    aren't even `period` closes to seed from."""
    if len(closes) < period:
        return []
    k = 2.0 / (period + 1)
    ema_values = [sum(closes[:period]) / period]
    for close in closes[period:]:
        ema_values.append(close * k + ema_values[-1] * (1 - k))
    return ema_values


def _closed_candles_oldest_first(raw_candles: List[dict]) -> List[dict]:
    """Drops any still-forming candle (confirm=="0") and sorts the rest
    oldest-to-newest — a missing/unknown confirm value is treated as
    closed, same conservative default observation_engine.py's
    _split_forming_and_closed uses."""
    closed = [c for c in raw_candles if str(c.get("confirm", "1")) != "0"]
    return sorted(closed, key=lambda c: c["ts"])


@dataclass
class EmaConfig:
    candle_bar: str = "3m"
    fast_period: int = 9
    slow_period: int = 21
    # Extra candles fetched beyond slow_period so there's still a full
    # slow-EMA series after the newest (possibly still-forming) one is
    # dropped.
    candle_fetch_buffer: int = 3
    # Fast/slow must differ by at least this much (as a fraction of the
    # slow EMA), post-cross, for the cross to count as real rather than
    # noise sitting right at the two lines touching.
    min_separation_pct: float = 0.0005
    # Only symbols in this set are ever accepted into the watchlist —
    # same hard backstop observation_engine.py uses.
    symbol_whitelist: Optional[frozenset] = field(default_factory=lambda: DEFAULT_SYMBOL_WHITELIST)


@dataclass
class EmaCandidate:
    symbol: str
    direction: str = ""
    fast_ema: float = 0.0
    slow_ema: float = 0.0
    price: float = 0.0
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.started_at

    def status_line(self) -> str:
        return (
            f"{self.symbol} direction={self.direction or '-'} price={self.price:.6g} "
            f"fast_ema={self.fast_ema:.6g} slow_ema={self.slow_ema:.6g}"
        )


class EmaStrategy(StrategyEngine):
    name = "ema_stg"

    def __init__(
        self,
        market_data: MarketDataStore,
        candle_fetcher: CandleFetcher,
        config: Optional[EmaConfig] = None,
    ) -> None:
        self._market_data = market_data
        self._candle_fetcher = candle_fetcher
        self.config = config or EmaConfig()
        self._candidates: Dict[str, EmaCandidate] = {}
        # Last-seen fast-vs-slow relationship ("above"/"below") per
        # symbol, so a signal only fires the tick the relationship
        # actually flips — not on every subsequent tick it happens to
        # still be on the favorable side.
        self._last_relation: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def sync_watchlist(self, watchlist_symbols) -> None:
        watchlist_symbols = set(watchlist_symbols)
        whitelist = self.config.symbol_whitelist
        if whitelist:
            rejected = watchlist_symbols - whitelist
            watchlist_symbols &= whitelist
            if rejected:
                log.debug(f"[ema] ignoring {len(rejected)} non-whitelisted symbol(s): {sorted(rejected)}")
        async with self._lock:
            for symbol in watchlist_symbols:
                if symbol not in self._candidates:
                    self._candidates[symbol] = EmaCandidate(symbol=symbol)
                    log.info(
                        f"[ema] {symbol} added — watching {self.config.fast_period}/{self.config.slow_period} "
                        f"EMA cross on {self.config.candle_bar}"
                    )
            dropped = [s for s in self._candidates if s not in watchlist_symbols]
            for symbol in dropped:
                del self._candidates[symbol]
                self._last_relation.pop(symbol, None)

    async def snapshot(self) -> List[EmaCandidate]:
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
        candidate.price = market["last_price"]

        try:
            raw_candles = await self._candle_fetcher(symbol, cfg.candle_bar, cfg.slow_period + cfg.candle_fetch_buffer)
        except Exception as exc:
            log.warning(f"[ema] {symbol} — could not fetch candles: {exc}")
            return None

        closes = [c["close"] for c in _closed_candles_oldest_first(raw_candles)]
        if len(closes) < cfg.slow_period:
            return None

        fast_series = compute_ema_series(closes, cfg.fast_period)
        slow_series = compute_ema_series(closes, cfg.slow_period)
        if not fast_series or not slow_series:
            return None

        fast_ema = fast_series[-1]
        slow_ema = slow_series[-1]
        candidate.fast_ema = fast_ema
        candidate.slow_ema = slow_ema

        relation = "above" if fast_ema > slow_ema else "below"
        prev_relation = self._last_relation.get(symbol)
        self._last_relation[symbol] = relation

        if prev_relation is None or prev_relation == relation:
            # First read for this symbol, or fast/slow simply hasn't
            # crossed since the last tick — no signal, nothing carried
            # forward beyond the relation bookkeeping above.
            candidate.direction = ""
            return None

        separation_pct = abs(fast_ema - slow_ema) / slow_ema if slow_ema else 0.0
        if separation_pct < cfg.min_separation_pct:
            return None

        direction = "long" if relation == "above" else "short"
        candidate.direction = direction

        log.info(f"[ema] {symbol} ACCEPTED — {candidate.status_line()} (crossed {prev_relation}->{relation})")
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=1.0,
            entry_price=candidate.price,
            take_profit=candidate.price,  # unused — execution_engine computes its own TP/SL
            stop_loss=candidate.price,
            timestamp=time.time(),
            reasons=[
                f"ema_cross={prev_relation}->{relation}",
                f"fast_ema={fast_ema:.6g}",
                f"slow_ema={slow_ema:.6g}",
            ],
        )


def build(ctx: StrategyContext) -> EmaStrategy:
    """strategy.load_strategy()'s entry point — see strategy/base.py's
    module docstring for the contract every strategy module follows."""
    cfg = ctx.build_config(EmaConfig)
    return EmaStrategy(ctx.market_data, ctx.candle_fetcher, config=cfg)
