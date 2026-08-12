"""
VWAP 3-Stage Engine — location-based signal routing off distance from VWAP.

This is a fork of observation_engine.py that keeps every one of its
existing building blocks (micro-trend detection, buy/sell pressure with
acceleration scoring, volume expansion scoring, VWAP-from-trade-tape, and
candle confirmation — all reused verbatim below, not reimplemented) but
throws out its single combined "trend + pressure + volume + vwap-side +
candle, all must agree" gate. That gate always evaluated the same four
checks regardless of where price actually was relative to VWAP, which
means it happily chased price straight into an already-overextended
move as long as trend/pressure/volume all agreed — exactly the "chasing
price" behavior this version is meant to stop.

Instead, price's distance from VWAP is classified into one of three
zones every tick, and only ONE of three independent engines ever runs,
picked by that zone:

  FAR ABOVE VWAP  (distance_pct > vwap_far_threshold_pct)
      -> Engine 1: short-only exhaustion/pullback play. Refuses to chase
         the move further; instead waits for price to reach a recent
         swing-high resistance level AND sellers to be visibly taking
         over (pressure + expanding volume) before opening SHORT.

  NEAR VWAP       (|distance_pct| < vwap_near_threshold_pct)
      -> Engine 2: continuation battle. Whichever side the established
         micro-trend favors, checks whether THAT side is actually
         defending/continuing through the VWAP retest (pressure +
         volume + candle confirmation) before opening in the trend's
         direction. If the trend is bullish this checks buyers; if
         bearish, sellers -- only one side is ever checked, matching
         whichever direction the trend already favors.

  FAR BELOW VWAP  (distance_pct < -vwap_far_threshold_pct)
      -> Engine 3: mirror of Engine 1 -- long-only exhaustion/pullback
         play off a swing-low support level, gated on buyer pressure +
         expanding buy volume.

  Anything in between (near the far thresholds but not within the near
  band either) is NEUTRAL: no engine runs, no signal, re-read fresh next
  tick. This is deliberate -- the bot should sit out the ambiguous
  middle ground rather than force one of the three reads to fit.

Same async structure, same market_data.Signal return type, same
TradeStore/MarketDataStore usage, and the same "no locked-in direction
across ticks" philosophy as observation_engine.py -- every tick is an
independent read; a failed check is never remembered into the next one.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from market_data import MarketDataStore, TradeStore, Signal, DEFAULT_SYMBOL_WHITELIST
from .base import StrategyContext, StrategyEngine

log = logging.getLogger("okx_futures.vwap3stage")

CandleFetcher = Callable[[str, str, int], Awaitable[List[dict]]]

TREND_CANDLE_BAR = "1m"
WINDOW_MS = 240_000
VWAP_WINDOW_MS = 1_800_000
MIN_ENTRY_TREND_STRENGTH_PCT = 60.0
MIN_ENTRY_PRESSURE_PCT = 60.0
MIN_ENTRY_VOLUME_EXPANSION_PCT = 30.0


def compute_trend_strength(candles: List[dict]) -> Dict:
    if not candles or len(candles) < 2:
        return {"direction": "sideways", "strength_pct": 0.0, "net_move_pct": 0.0}

    ordered = sorted(candles, key=lambda c: c["ts"])
    open_price, close_price = ordered[0]["open"], ordered[-1]["close"]
    net_move_pct = (close_price - open_price) / open_price if open_price else 0.0

    bull_weight = bear_weight = 0.0
    for c in ordered:
        o, cl = c["open"], c["close"]
        if not o:
            continue
        move = (cl - o) / o
        weight = abs(move)
        if move > 0:
            bull_weight += weight
        elif move < 0:
            bear_weight += weight

    total_weight = bull_weight + bear_weight
    if total_weight <= 0:
        return {"direction": "sideways", "strength_pct": 0.0, "net_move_pct": round(net_move_pct, 5)}

    if bull_weight > bear_weight:
        direction, dominant = "long", bull_weight
    elif bear_weight > bull_weight:
        direction, dominant = "short", bear_weight
    else:
        direction, dominant = "sideways", 0.0

    strength_pct = round(100.0 * dominant / total_weight, 2)
    return {"direction": direction, "strength_pct": strength_pct, "net_move_pct": round(net_move_pct, 5)}


def _bucketize_trades(trades: List[dict], bucket_count: int) -> List[List[dict]]:
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


def _slice_slope_score(values: List[float], full_range: float) -> float:
    n = len(values)
    if n < 2:
        return 50.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        return 50.0
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom
    max_slope = full_range / (n - 1)
    slope_norm = max(-1.0, min(1.0, slope / max_slope)) if max_slope > 0 else 0.0
    return (slope_norm + 1.0) * 50.0


def compute_buy_pressure_strength(trades: List[dict], direction: str, bucket_count: int) -> Dict:
    side = "buy" if direction == "long" else "sell"
    other = "sell" if direction == "long" else "buy"
    buckets = _bucketize_trades(trades, bucket_count)

    pcts = []
    for bucket in buckets:
        side_vol = sum(t["qty"] for t in bucket if t["side"] == side)
        other_vol = sum(t["qty"] for t in bucket if t["side"] == other)
        total = side_vol + other_vol
        if total > 0:
            pcts.append(side_vol / total)

    if len(pcts) < 2:
        current_ratio = pcts[-1] if pcts else 0.0
        return {"strength_pct": 0.0, "current_ratio": round(current_ratio, 4), "accelerating": False}

    slope_score = _slice_slope_score(pcts, full_range=0.5)
    level_score = pcts[-1] * 100.0
    strength_pct = round(0.5 * slope_score + 0.5 * level_score, 2)
    return {"strength_pct": strength_pct, "current_ratio": round(pcts[-1], 4), "accelerating": pcts[-1] > pcts[0]}


def compute_volume_expansion_strength(trades: List[dict], direction: str, bucket_count: int, target_multiplier: float) -> Dict:
    side = "buy" if direction == "long" else "sell"
    buckets = _bucketize_trades(trades, bucket_count)
    volumes = [sum(t["qty"] for t in bucket if t["side"] == side) for bucket in buckets]

    n = len(volumes)
    if n < 2 or all(v <= 0 for v in volumes):
        return {"strength_pct": 0.0, "expanding": False}

    increases = sum(1 for i in range(1, n) if volumes[i] >= volumes[i - 1])
    monotonic_ratio = increases / (n - 1)

    half = max(1, n // 2)
    early_avg = sum(volumes[:half]) / half
    late_avg = sum(volumes[-half:]) / half
    if early_avg > 0:
        growth_ratio = late_avg / early_avg
        growth_score = max(0.0, min(1.0, growth_ratio / target_multiplier))
    else:
        growth_score = 1.0 if late_avg > 0 else 0.0

    strength_pct = round(0.5 * monotonic_ratio * 100.0 + 0.5 * growth_score * 100.0, 2)
    return {"strength_pct": strength_pct, "expanding": strength_pct >= 50.0}


def compute_vwap(trades: List[dict]) -> Optional[float]:
    total_qty = sum(t["qty"] for t in trades)
    if total_qty <= 0:
        return None
    return sum(t["price"] * t["qty"] for t in trades) / total_qty


def _vwap_supports_direction(price: float, vwap: Optional[float], direction: str) -> bool:
    if vwap is None:
        return False
    if direction == "long":
        return price > vwap
    if direction == "short":
        return price < vwap
    return False


def _split_forming_and_closed(candles: List[dict]):
    forming = None
    closed: List[dict] = []
    ordered = sorted(candles, key=lambda c: c.get("ts", 0), reverse=True)
    for c in ordered:
        if forming is None and str(c.get("confirm")) == "0":
            forming = c
        else:
            closed.append(c)
    return forming, closed


def _candle_supports_direction(candle: Optional[dict], direction: str) -> bool:
    if not candle:
        return False
    o, c = candle.get("open"), candle.get("close")
    if not o or c is None:
        return False
    if direction == "long":
        return c > o
    if direction == "short":
        return c < o
    return False


def get_swing_levels(closed_candles: List[dict], lookback: int) -> Dict[str, Optional[float]]:
    if not closed_candles:
        return {"swing_high": None, "swing_low": None}
    ordered = sorted(closed_candles, key=lambda c: c.get("ts", 0), reverse=True)[:lookback]
    highs = [c["high"] for c in ordered if c.get("high") is not None]
    lows = [c["low"] for c in ordered if c.get("low") is not None]
    return {
        "swing_high": max(highs) if highs else None,
        "swing_low": min(lows) if lows else None,
    }


def classify_vwap_zone(distance_pct: float, cfg: "Vwap3StageConfig") -> str:
    if distance_pct > cfg.vwap_far_threshold_pct:
        return "far_above"
    if distance_pct < -cfg.vwap_far_threshold_pct:
        return "far_below"
    if abs(distance_pct) < cfg.vwap_near_threshold_pct:
        return "near"
    return "neutral"


@dataclass
class Vwap3StageConfig:
    max_observation_minutes: float = 6.0
    trend_candle_bar: str = TREND_CANDLE_BAR
    bucket_count: int = 5
    window_ms: int = WINDOW_MS
    vwap_window_ms: int = VWAP_WINDOW_MS
    min_data_warmup_sec: float = 45.0
    min_data_trade_count: int = 15
    candle_fetch_buffer: int = 2
    symbol_whitelist: Optional[frozenset] = field(default_factory=lambda: DEFAULT_SYMBOL_WHITELIST)

    vwap_far_threshold_pct: float = 0.005
    vwap_near_threshold_pct: float = 0.0010

    swing_lookback: int = 20
    swing_proximity_pct: float = 0.002

    reversal_min_pressure_pct: float = 70.0
    reversal_require_pressure_accelerating: bool = True
    reversal_min_volume_expansion_strength_pct: float = 55.0
    reversal_volume_expansion_multiplier: float = 1.4

    continuation_min_trend_strength_pct: float = 65.0
    continuation_min_net_move_pct: float = 0.0015
    continuation_min_pressure_pct: float = 70.0
    continuation_require_pressure_accelerating: bool = True
    continuation_min_volume_expansion_strength_pct: float = 55.0
    continuation_volume_expansion_multiplier: float = 1.4
    continuation_require_candle_confirmation: bool = True

    min_entry_trend_strength_pct: float = MIN_ENTRY_TREND_STRENGTH_PCT
    min_entry_pressure_pct: float = MIN_ENTRY_PRESSURE_PCT
    min_entry_volume_expansion_pct: float = MIN_ENTRY_VOLUME_EXPANSION_PCT


@dataclass
class CandidateObservation:
    symbol: str
    direction: str = ""
    status: str = "OBSERVING"
    started_at: float = field(default_factory=time.time)
    last_checked_at: float = 0.0

    data_ready: bool = False

    trend: str = "sideways"
    trend_strength_pct: float = 0.0

    buy_pressure_strength_pct: float = 0.0
    volume_strength_pct: float = 0.0

    vwap: Optional[float] = None
    vwap_distance_pct: float = 0.0
    vwap_zone: str = "neutral"

    swing_high: Optional[float] = None
    swing_low: Optional[float] = None

    engine_used: str = ""

    entry_price: float = 0.0

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.started_at

    @property
    def direction_letter(self) -> str:
        return self.direction[0].upper() if self.direction else "?"

    def status_line(self) -> str:
        vwap_text = f"{self.vwap:.6g}" if self.vwap is not None else "-"
        base = (
            f"{self.symbol} status={self.status} zone={self.vwap_zone} "
            f"direction={self.direction.upper() or '-'} elapsed={self.elapsed_sec:.0f}s "
            f"vwap={vwap_text} dist={self.vwap_distance_pct:+.2%} "
            f"pressure={self.buy_pressure_strength_pct:.0f}% volume={self.volume_strength_pct:.0f}%"
        )
        if self.engine_used:
            base += f" engine={self.engine_used}"
        if not self.data_ready:
            base += " (warming up)"
        return base


class Vwap3StageEngine(StrategyEngine):

    name = "vwap_3stage_engine"

    def __init__(
        self,
        trade_store: TradeStore,
        market_data: MarketDataStore,
        candle_fetcher: CandleFetcher,
        config: Optional[Vwap3StageConfig] = None,
    ) -> None:
        self._trade_store = trade_store
        self._market_data = market_data
        self._candle_fetcher = candle_fetcher
        self.config = config or Vwap3StageConfig()
        self._candidates: Dict[str, CandidateObservation] = {}
        self._lock = asyncio.Lock()

    async def sync_watchlist(self, watchlist_symbols) -> None:
        watchlist_symbols = set(watchlist_symbols)
        whitelist = self.config.symbol_whitelist
        if whitelist:
            rejected = watchlist_symbols - whitelist
            watchlist_symbols &= whitelist
            if rejected:
                log.debug(
                    f"[vwap3stage] ignoring {len(rejected)} non-whitelisted symbol(s) from the feed: "
                    f"{sorted(rejected)}"
                )
        async with self._lock:
            for symbol in watchlist_symbols:
                if symbol not in self._candidates:
                    self._candidates[symbol] = CandidateObservation(symbol=symbol)
                    log.info(
                        f"[vwap3stage] {symbol} added — observing for up to "
                        f"{self.config.max_observation_minutes:.0f}m"
                    )
            dropped = [s for s in self._candidates if s not in watchlist_symbols]
            for symbol in dropped:
                del self._candidates[symbol]

    async def snapshot(self) -> List[CandidateObservation]:
        async with self._lock:
            return list(self._candidates.values())

    async def evaluate(self, symbol: str) -> Optional[Signal]:
        cfg = self.config
        async with self._lock:
            candidate = self._candidates.get(symbol)
        if candidate is None or candidate.status != "OBSERVING":
            return None

        if candidate.elapsed_sec >= cfg.max_observation_minutes * 60.0:
            candidate.status = "EXPIRED"
            log.info(f"[vwap3stage] {symbol} EXPIRED after {candidate.elapsed_sec / 60.0:.1f}m — discarding")
            async with self._lock:
                self._candidates.pop(symbol, None)
            return None

        market = await self._market_data.get(symbol)
        if not market:
            return None
        candidate.last_checked_at = time.time()
        price = market["last_price"]
        candidate.entry_price = price

        fetch_count = max(cfg.bucket_count, cfg.swing_lookback) + cfg.candle_fetch_buffer
        try:
            raw_candles = await self._candle_fetcher(symbol, cfg.trend_candle_bar, fetch_count)
        except Exception as exc:
            log.warning(f"[vwap3stage] {symbol} — could not fetch candles: {exc}")
            return None
        forming_candle, closed_candles = _split_forming_and_closed(raw_candles)
        support_candle = forming_candle or (closed_candles[0] if closed_candles else None)

        trend_result = compute_trend_strength(closed_candles[: cfg.bucket_count])
        candidate.trend = trend_result["direction"]
        candidate.trend_strength_pct = trend_result["strength_pct"]

        swing = get_swing_levels(closed_candles, cfg.swing_lookback)
        candidate.swing_high = swing["swing_high"]
        candidate.swing_low = swing["swing_low"]

        window_trades = await self._trade_store.get_window(symbol, cfg.window_ms)
        was_ready = candidate.data_ready
        candidate.data_ready = (
            candidate.elapsed_sec >= cfg.min_data_warmup_sec
            and len(window_trades) >= cfg.min_data_trade_count
        )
        if candidate.data_ready and not was_ready:
            log.info(
                f"[vwap3stage] {symbol} data warm-up complete after {candidate.elapsed_sec:.0f}s "
                f"({len(window_trades)} trades in window) — zone checks now active"
            )
        if not candidate.data_ready:
            return None

        try:
            vwap_trades = await self._trade_store.get_window(symbol, cfg.vwap_window_ms)
        except Exception as exc:
            log.warning(f"[vwap3stage] {symbol} — could not fetch VWAP window: {exc}")
            return None
        vwap = compute_vwap(vwap_trades)
        candidate.vwap = vwap
        if not vwap:
            return None

        distance_pct = (price - vwap) / vwap
        candidate.vwap_distance_pct = distance_pct
        zone = classify_vwap_zone(distance_pct, cfg)
        candidate.vwap_zone = zone

        if zone == "far_above":
            candidate.engine_used = "engine1_short_exhaustion"
            return await self._evaluate_engine1_short(candidate, price, distance_pct, window_trades, trend_result)
        if zone == "near":
            candidate.engine_used = "engine2_continuation"
            return await self._evaluate_engine2_continuation(candidate, price, trend_result, window_trades, support_candle)
        if zone == "far_below":
            candidate.engine_used = "engine3_long_exhaustion"
            return await self._evaluate_engine3_long(candidate, price, distance_pct, window_trades)

        candidate.engine_used = ""
        candidate.direction = ""
        return None

    async def _evaluate_engine1_short(
        self,
        candidate: CandidateObservation,
        price: float,
        distance_pct: float,
        window_trades: List[dict],
        trend_result: Dict,
    ) -> Optional[Signal]:
        cfg = self.config
        swing_high = candidate.swing_high
        if not swing_high:
            candidate.direction = ""
            return None

        proximity_pct = abs(price - swing_high) / swing_high
        if proximity_pct >= cfg.swing_proximity_pct:
            candidate.direction = ""
            return None

        direction = "short"
        pressure = compute_buy_pressure_strength(window_trades, direction, cfg.bucket_count)
        volume = compute_volume_expansion_strength(
            window_trades, direction, cfg.bucket_count, cfg.reversal_volume_expansion_multiplier
        )
        candidate.buy_pressure_strength_pct = pressure["strength_pct"]
        candidate.volume_strength_pct = volume["strength_pct"]

        pressure_ok = pressure["strength_pct"] >= cfg.reversal_min_pressure_pct
        if cfg.reversal_require_pressure_accelerating:
            pressure_ok = pressure_ok and pressure["accelerating"]
        volume_ok = volume["expanding"] and volume["strength_pct"] >= cfg.reversal_min_volume_expansion_strength_pct

        entry_reqs_ok = (
            trend_result["strength_pct"] >= cfg.min_entry_trend_strength_pct
            and pressure["strength_pct"] >= cfg.min_entry_pressure_pct
            and volume["strength_pct"] >= cfg.min_entry_volume_expansion_pct
        )

        if not (pressure_ok and volume_ok and entry_reqs_ok):
            candidate.direction = ""
            return None

        candidate.direction = direction
        symbol = candidate.symbol
        log.info(
            f"[vwap3stage] ENGINE 1 ACCEPTED: {symbol}\n"
            f"  Price far above VWAP (distance={distance_pct:+.2%})\n"
            f"  Swing resistance reached (price={price:.6g}, swing_high={swing_high:.6g})\n"
            f"  Seller pressure {pressure['strength_pct']:.0f}%\n"
            f"  Sell volume expanding"
        )
        async with self._lock:
            self._candidates.pop(symbol, None)
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=1.0,
            entry_price=price,
            take_profit=price,
            stop_loss=price,
            timestamp=time.time(),
            reasons=[
                "engine=1_short_exhaustion",
                f"vwap_distance={distance_pct:+.2%}",
                f"swing_high={swing_high:.6g}",
                f"seller_pressure={pressure['strength_pct']:.0f}%",
                f"sell_volume_expansion={volume['strength_pct']:.0f}%",
            ],
        )

    async def _evaluate_engine3_long(
        self, candidate: CandidateObservation, price: float, distance_pct: float, window_trades: List[dict]
    ) -> Optional[Signal]:
        cfg = self.config
        swing_low = candidate.swing_low
        if not swing_low:
            candidate.direction = ""
            return None

        proximity_pct = abs(price - swing_low) / swing_low
        if proximity_pct >= cfg.swing_proximity_pct:
            candidate.direction = ""
            return None

        direction = "long"
        pressure = compute_buy_pressure_strength(window_trades, direction, cfg.bucket_count)
        volume = compute_volume_expansion_strength(
            window_trades, direction, cfg.bucket_count, cfg.reversal_volume_expansion_multiplier
        )
        candidate.buy_pressure_strength_pct = pressure["strength_pct"]
        candidate.volume_strength_pct = volume["strength_pct"]

        pressure_ok = pressure["strength_pct"] >= cfg.reversal_min_pressure_pct
        if cfg.reversal_require_pressure_accelerating:
            pressure_ok = pressure_ok and pressure["accelerating"]
        volume_ok = volume["expanding"] and volume["strength_pct"] >= cfg.reversal_min_volume_expansion_strength_pct

        if not (pressure_ok and volume_ok):
            candidate.direction = ""
            return None

        candidate.direction = direction
        symbol = candidate.symbol
        log.info(
            f"[vwap3stage] ENGINE 3 ACCEPTED: {symbol}\n"
            f"  Price far below VWAP (distance={distance_pct:+.2%})\n"
            f"  Swing support reached (price={price:.6g}, swing_low={swing_low:.6g})\n"
            f"  Buyer pressure {pressure['strength_pct']:.0f}%\n"
            f"  Buy volume expanding"
        )
        async with self._lock:
            self._candidates.pop(symbol, None)
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=1.0,
            entry_price=price,
            take_profit=price,
            stop_loss=price,
            timestamp=time.time(),
            reasons=[
                "engine=3_long_exhaustion",
                f"vwap_distance={distance_pct:+.2%}",
                f"swing_low={swing_low:.6g}",
                f"buyer_pressure={pressure['strength_pct']:.0f}%",
                f"buy_volume_expansion={volume['strength_pct']:.0f}%",
            ],
        )

    async def _evaluate_engine2_continuation(
        self,
        candidate: CandidateObservation,
        price: float,
        trend_result: Dict,
        window_trades: List[dict],
        support_candle: Optional[dict],
    ) -> Optional[Signal]:
        cfg = self.config
        trend_ok = (
            trend_result["direction"] != "sideways"
            and trend_result["strength_pct"] >= cfg.continuation_min_trend_strength_pct
            and abs(trend_result["net_move_pct"]) >= cfg.continuation_min_net_move_pct
        )
        if not trend_ok:
            candidate.direction = ""
            return None

        direction = trend_result["direction"]
        pressure = compute_buy_pressure_strength(window_trades, direction, cfg.bucket_count)
        volume = compute_volume_expansion_strength(
            window_trades, direction, cfg.bucket_count, cfg.continuation_volume_expansion_multiplier
        )
        candidate.buy_pressure_strength_pct = pressure["strength_pct"]
        candidate.volume_strength_pct = volume["strength_pct"]

        pressure_ok = pressure["strength_pct"] >= cfg.continuation_min_pressure_pct
        if cfg.continuation_require_pressure_accelerating:
            pressure_ok = pressure_ok and pressure["accelerating"]
        volume_ok = volume["expanding"] and volume["strength_pct"] >= cfg.continuation_min_volume_expansion_strength_pct
        candle_ok = (
            not cfg.continuation_require_candle_confirmation
            or _candle_supports_direction(support_candle, direction)
        )

        entry_reqs_ok = (
            trend_result["strength_pct"] >= cfg.min_entry_trend_strength_pct
            and pressure["strength_pct"] >= cfg.min_entry_pressure_pct
            and volume["strength_pct"] >= cfg.min_entry_volume_expansion_pct
        )

        if not (pressure_ok and volume_ok and candle_ok and entry_reqs_ok):
            candidate.direction = ""
            return None

        candidate.direction = direction
        symbol = candidate.symbol
        trend_label = "Bull" if direction == "long" else "Bear"
        side_label = "Buyer" if direction == "long" else "Seller"
        log.info(
            f"[vwap3stage] ENGINE 2 ACCEPTED: {symbol}\n"
            f"  VWAP continuation\n"
            f"  {trend_label} trend ({trend_result['strength_pct']:.0f}% strength)\n"
            f"  {side_label} dominance {pressure['strength_pct']:.0f}%"
        )
        async with self._lock:
            self._candidates.pop(symbol, None)
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=1.0,
            entry_price=price,
            take_profit=price,
            stop_loss=price,
            timestamp=time.time(),
            reasons=[
                "engine=2_continuation",
                f"trend={trend_result['direction']}:{trend_result['strength_pct']:.0f}%",
                f"dominance={pressure['strength_pct']:.0f}%",
                f"volume_expansion={volume['strength_pct']:.0f}%",
            ],
        )


def build(ctx: StrategyContext) -> Vwap3StageEngine:
    cfg = ctx.build_config(Vwap3StageConfig)
    return Vwap3StageEngine(ctx.trade_store, ctx.market_data, ctx.candle_fetcher, config=cfg)