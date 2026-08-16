"""
Flow Ignition Engine — order-flow burst detection for ETH-USDT scalping.

Where vwap_3stage_engine.py routes off WHERE price sits relative to
VWAP, this engine ignores location and instead watches HOW the trade
tape is behaving right now against its own recent pace. Every tick
pulls a baseline_window_ms trade-tape window (default 3 min) and tests
its most recent ignition_window_ms slice (default 8 sec) against six
gates: cooldown/daily cap, a realized-range regime filter, an ignition
z-score (the slice's signed buy/sell delta vs. the baseline's own
empirical distribution of same-length-slice deltas), tape acceleration
(trades/sec vs. baseline pace), price displacement (the burst must be
moving price, not just absorbing volume), and dominant-side trade count
(blocks a single block print from faking a burst). All six must pass
for evaluate() to return a Signal.

No VWAP, swing levels, candles, or classical indicators — at this scalp
scale the trade tape itself reacts faster than anything candle-based.

The only state remembered tick-to-tick is last_signal_at per symbol,
purely for cooldown_sec. max_signals_per_day is a hard daily ceiling on
top of that.

Same Signal/StrategyEngine/TradeStore/MarketDataStore contract as
vwap_3stage_engine.py, including its take_profit=price/stop_loss=price
placeholder pattern — execution_engine computes the real TP/SL from
tracker.py's target_net_profit_usdt/target_stop_loss_usdt, not from the
strategy module. Switch to this strategy with tracker.py's
STRATEGY_NAME = "flow_ignition_engine".
"""

import asyncio
import calendar
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from market_data import MarketDataStore, TradeStore, Signal
from .base import StrategyContext, StrategyEngine

log = logging.getLogger("okx_futures.flowignition")

CandleFetcher = Callable[[str, str, int], Awaitable[List[dict]]]


def compute_vwap(trades: List[dict]) -> Optional[float]:
    """Volume-weighted average price across `trades`."""
    total_qty = sum(t["qty"] for t in trades)
    if total_qty <= 0:
        return None
    return sum(t["price"] * t["qty"] for t in trades) / total_qty


def _bucketize_by_time(trades: List[dict], bucket_count: int) -> List[List[dict]]:
    """Splits `trades` into `bucket_count` equal-duration chronological slices."""
    if not trades or bucket_count < 1:
        return [[] for _ in range(max(bucket_count, 1))]
    ordered = sorted(trades, key=lambda t: t["timestamp"])
    start, end = ordered[0]["timestamp"], ordered[-1]["timestamp"]
    span = end - start
    buckets: List[List[dict]] = [[] for _ in range(bucket_count)]
    if span <= 0:
        buckets[-1] = ordered
        return buckets
    bucket_span = span / bucket_count
    for t in ordered:
        idx = min(int((t["timestamp"] - start) / bucket_span), bucket_count - 1)
        buckets[idx].append(t)
    return buckets


def _signed_delta(trades: List[dict]) -> float:
    buy = sum(t["qty"] for t in trades if t["side"] == "buy")
    sell = sum(t["qty"] for t in trades if t["side"] == "sell")
    return buy - sell


def compute_realized_range_pct(baseline_trades: List[dict]) -> float:
    """(max_price - min_price) / mean_price across `baseline_trades`."""
    if not baseline_trades:
        return 0.0
    prices = [t["price"] for t in baseline_trades]
    mean_price = sum(prices) / len(prices)
    if mean_price <= 0:
        return 0.0
    return (max(prices) - min(prices)) / mean_price


def compute_baseline_delta_stats(baseline_trades: List[dict], slice_count: int) -> Dict:
    """Mean/stdev of signed delta across `slice_count` equal-duration slices of `baseline_trades`."""
    buckets = _bucketize_by_time(baseline_trades, slice_count)
    deltas = [_signed_delta(b) for b in buckets if b]
    if len(deltas) < 8:
        return {"mean": 0.0, "stdev": 0.0, "slice_count_used": len(deltas)}
    mean = statistics.fmean(deltas)
    stdev = statistics.pstdev(deltas, mu=mean)
    return {"mean": mean, "stdev": stdev, "slice_count_used": len(deltas)}


def compute_ignition_zscore(ignition_trades: List[dict], baseline_stats: Dict) -> Dict:
    """Z-scores the ignition window's signed delta against `baseline_stats`."""
    delta = _signed_delta(ignition_trades)
    stdev = baseline_stats["stdev"]
    if stdev <= 0:
        return {"delta": delta, "zscore": 0.0}
    return {"delta": delta, "zscore": (delta - baseline_stats["mean"]) / stdev}


def compute_tape_acceleration(
    ignition_trades: List[dict], baseline_trades: List[dict], ignition_window_ms: int, baseline_window_ms: int
) -> Dict:
    """Ignition trades/sec vs. baseline trades/sec."""
    ignition_rate = len(ignition_trades) / max(ignition_window_ms / 1000.0, 1e-9)
    baseline_rate = len(baseline_trades) / max(baseline_window_ms / 1000.0, 1e-9)
    multiplier = (ignition_rate / baseline_rate) if baseline_rate > 0 else 0.0
    return {
        "ignition_rate": round(ignition_rate, 3),
        "baseline_rate": round(baseline_rate, 3),
        "multiplier": round(multiplier, 3),
    }


def compute_micro_displacement(ignition_trades: List[dict]) -> Dict:
    """Price displacement from the older half to the newer half of the ignition window, by VWAP."""
    if len(ignition_trades) < 4:
        return {"displacement_pct": 0.0, "direction": "flat"}
    ordered = sorted(ignition_trades, key=lambda t: t["timestamp"])
    mid = len(ordered) // 2
    early_vwap = compute_vwap(ordered[:mid])
    late_vwap = compute_vwap(ordered[mid:])
    if not early_vwap or not late_vwap:
        return {"displacement_pct": 0.0, "direction": "flat"}
    displacement_pct = (late_vwap - early_vwap) / early_vwap
    direction = "up" if displacement_pct > 0 else ("down" if displacement_pct < 0 else "flat")
    return {"displacement_pct": round(displacement_pct, 6), "direction": direction}


def dominant_side_trade_count(ignition_trades: List[dict], direction: str) -> int:
    """Count of ignition trades on `direction`'s side."""
    side = "buy" if direction == "long" else "sell"
    return sum(1 for t in ignition_trades if t["side"] == side)


@dataclass
class FlowIgnitionConfig:
    symbol_whitelist: Optional[frozenset] = field(default_factory=lambda: frozenset({"ETH-USDT-SWAP"}))

    baseline_window_ms: int = 180_000
    ignition_window_ms: int = 8_000
    min_baseline_trade_count: int = 40
    min_ignition_trade_count: int = 6

    min_regime_range_pct: float = 0.0006
    max_regime_range_pct: float = 0.006

    ignition_min_zscore: float = 2.5
    tape_min_acceleration: float = 1.8
    min_displacement_pct: float = 0.00025
    min_dominant_trade_count: int = 4

    cooldown_sec: float = 180.0
    max_signals_per_day: int = 20


@dataclass
class SymbolFlowState:
    symbol: str
    last_signal_at: float = 0.0
    last_checked_at: float = 0.0

    last_zscore: float = 0.0
    last_acceleration: float = 0.0
    last_displacement_pct: float = 0.0
    last_regime_range_pct: float = 0.0
    last_reject_reason: str = ""

    def status_line(self) -> str:
        return (
            f"{self.symbol} z={self.last_zscore:+.2f} accel={self.last_acceleration:.2f}x "
            f"disp={self.last_displacement_pct:+.3%} regime={self.last_regime_range_pct:.3%} "
            f"reject={self.last_reject_reason or '-'}"
        )


class FlowIgnitionEngine(StrategyEngine):
    """Tracks one SymbolFlowState per watchlisted symbol and tests its trade tape for an order-flow ignition each tick."""

    name = "flow_ignition_engine"

    def __init__(
        self,
        trade_store: TradeStore,
        market_data: MarketDataStore,
        candle_fetcher: CandleFetcher,
        config: Optional[FlowIgnitionConfig] = None,
    ) -> None:
        self._trade_store = trade_store
        self._market_data = market_data
        self._candle_fetcher = candle_fetcher
        self.config = config or FlowIgnitionConfig()
        self._states: Dict[str, SymbolFlowState] = {}
        self._lock = asyncio.Lock()
        self._signals_today = 0
        self._day_started_at = self._utc_day_start()

    @staticmethod
    def _utc_day_start(ts: Optional[float] = None) -> float:
        t = time.gmtime(ts if ts is not None else time.time())
        return float(calendar.timegm((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, 0)))

    def _roll_day_if_needed(self, now: float) -> None:
        if now >= self._day_started_at + 86_400:
            if self._signals_today:
                log.info(f"[flow_ignition] day rollover — {self._signals_today} signal(s) fired in the prior 24h")
            self._signals_today = 0
            self._day_started_at = self._utc_day_start(now)

    async def sync_watchlist(self, watchlist_symbols) -> None:
        watchlist_symbols = set(watchlist_symbols)
        whitelist = self.config.symbol_whitelist
        if whitelist:
            rejected = watchlist_symbols - whitelist
            watchlist_symbols &= whitelist
            if rejected:
                log.debug(f"[flow_ignition] ignoring {len(rejected)} non-whitelisted symbol(s): {sorted(rejected)}")
        async with self._lock:
            for symbol in watchlist_symbols:
                if symbol not in self._states:
                    self._states[symbol] = SymbolFlowState(symbol=symbol)
                    log.info(f"[flow_ignition] {symbol} added — watching trade tape for order-flow ignitions")
            dropped = [s for s in self._states if s not in watchlist_symbols]
            for symbol in dropped:
                del self._states[symbol]

    async def snapshot(self) -> List[SymbolFlowState]:
        async with self._lock:
            return list(self._states.values())

    async def evaluate(self, symbol: str) -> Optional[Signal]:
        cfg = self.config
        async with self._lock:
            state = self._states.get(symbol)
        if state is None:
            return None

        now = time.time()
        self._roll_day_if_needed(now)
        state.last_checked_at = now

        if self._signals_today >= cfg.max_signals_per_day:
            state.last_reject_reason = "daily_cap_reached"
            return None
        if now - state.last_signal_at < cfg.cooldown_sec:
            state.last_reject_reason = "cooldown"
            return None

        try:
            baseline_trades = await self._trade_store.get_window(symbol, cfg.baseline_window_ms)
        except Exception as exc:
            log.warning(f"[flow_ignition] {symbol} — could not fetch baseline window: {exc}")
            return None
        if len(baseline_trades) < cfg.min_baseline_trade_count:
            state.last_reject_reason = "baseline_too_thin"
            return None

        ignition_cutoff_ms = now * 1000.0 - cfg.ignition_window_ms
        ignition_trades = [t for t in baseline_trades if t["timestamp"] >= ignition_cutoff_ms]
        if len(ignition_trades) < cfg.min_ignition_trade_count:
            state.last_reject_reason = "ignition_too_thin"
            return None

        regime_range_pct = compute_realized_range_pct(baseline_trades)
        state.last_regime_range_pct = regime_range_pct
        if not (cfg.min_regime_range_pct <= regime_range_pct <= cfg.max_regime_range_pct):
            state.last_reject_reason = "regime_out_of_band"
            return None

        slice_count = max(3, cfg.baseline_window_ms // cfg.ignition_window_ms)
        baseline_stats = compute_baseline_delta_stats(baseline_trades, slice_count)
        z = compute_ignition_zscore(ignition_trades, baseline_stats)
        state.last_zscore = z["zscore"]
        if abs(z["zscore"]) < cfg.ignition_min_zscore:
            state.last_reject_reason = "zscore_below_threshold"
            return None

        direction = "long" if z["zscore"] > 0 else "short"

        accel = compute_tape_acceleration(ignition_trades, baseline_trades, cfg.ignition_window_ms, cfg.baseline_window_ms)
        state.last_acceleration = accel["multiplier"]
        if accel["multiplier"] < cfg.tape_min_acceleration:
            state.last_reject_reason = "tape_not_accelerating"
            return None

        disp = compute_micro_displacement(ignition_trades)
        state.last_displacement_pct = disp["displacement_pct"]
        wants_up = direction == "long"
        displacement_ok = (
            (wants_up and disp["displacement_pct"] >= cfg.min_displacement_pct)
            or (not wants_up and disp["displacement_pct"] <= -cfg.min_displacement_pct)
        )
        if not displacement_ok:
            state.last_reject_reason = "no_price_confirmation"
            return None

        dominant_count = dominant_side_trade_count(ignition_trades, direction)
        if dominant_count < cfg.min_dominant_trade_count:
            state.last_reject_reason = "single_print_burst"
            return None

        market = await self._market_data.get(symbol)
        if not market:
            state.last_reject_reason = "no_market_data"
            return None
        price = market["last_price"]

        state.last_reject_reason = ""
        state.last_signal_at = now
        self._signals_today += 1

        log.info(
            f"[flow_ignition] SIGNAL: {symbol} {direction.upper()} (#{self._signals_today}/{cfg.max_signals_per_day} today)\n"
            f"  entry={price:.6g}\n"
            f"  zscore={z['zscore']:+.2f} acceleration={accel['multiplier']:.2f}x "
            f"displacement={disp['displacement_pct']:+.3%} regime={regime_range_pct:.3%} "
            f"dominant_trades={dominant_count}"
        )
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=1.0,
            entry_price=price,
            take_profit=price,
            stop_loss=price,
            timestamp=now,
            reasons=[
                "engine=flow_ignition",
                f"zscore={z['zscore']:+.2f}",
                f"tape_acceleration={accel['multiplier']:.2f}x",
                f"displacement={disp['displacement_pct']:+.3%}",
                f"regime_range={regime_range_pct:.3%}",
                f"dominant_trades={dominant_count}",
            ],
        )


def build(ctx: StrategyContext) -> FlowIgnitionEngine:
    cfg = ctx.build_config(FlowIgnitionConfig)
    return FlowIgnitionEngine(ctx.trade_store, ctx.market_data, ctx.candle_fetcher, config=cfg)
