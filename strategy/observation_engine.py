"""
Fast-Scalp Observation Engine — Momentum Breakout signal system.

This replaces the previous "Observation Window" engine's MONITORING /
exhaustion / TREND_RECOVERED / REVERSAL_CONFIRMED state machine, which was
built around a 30-minute reference window and up to 20 minutes of patient
waiting for a trend to either recover or genuinely reverse. That design
made sense for a slower swing-style entry, but it's a poor match for a
bot targeting trades that resolve in single-digit minutes: by the time a
candidate finished its exhaustion → monitoring → recovery/reversal detour,
the scalp opportunity it was reading was already stale.

This version keeps the two ideas from the old engine that were genuinely
good (candles weighted by move size rather than counted 1-for-1, and
pressure/volume scored as ACCELERATING/EXPANDING rather than merely
"above some level") but drops the multi-stage waiting logic entirely.
Every tick independently asks one question: is there real, currently-
building momentum in some direction right now? If yes, open immediately.
If no, change nothing and ask again next tick. There is no "locked in"
direction that persists across ticks while conditions weaken — a
candidate's direction is only ever what THIS tick's data says, which is
what keeps the read fresh for a fast scalp.

Mechanics: a symbol entering the watchlist becomes a candidate immediately
and is checked every tick for up to `max_observation_minutes` (6 by
default — short, because a scalp candidate that hasn't fired within a
few minutes is reading stale context, not a 20-minute one). Every tick:

  1. Compute the current micro-trend from the last `bucket_count` CLOSED
     `trend_candle_bar` candles (1-minute by default, ~5 minutes total —
     not the old engine's 5-minute/30-minute setup). If there's no clear,
     sufficiently strong, sufficiently large-net-move direction, the tick
     ends here; nothing is remembered from a prior tick.

  2. If a direction is established, check whether that side's pressure is
     accelerating and its volume is expanding (same scored functions as
     before, just fed a much shorter trade-tape window — `window_ms`,
     4 minutes by default instead of 30), AND the live (currently-forming)
     candle itself agrees with the direction, AND price is on the correct
     side of the session VWAP (long wants price above VWAP — buyers
     paying up, not chasing a spike; short wants price below it). All
     four true -> open immediately.

  3. Anything less than that -> stay OBSERVING, re-checked fresh next
     tick, until `max_observation_minutes` elapses and the candidate is
     dropped.

There is no cross-tick "exhausted side" bookkeeping and no reversal-
structure check — if the market actually reverses, the next tick's fresh
micro-trend read simply shows the new direction and re-evaluates it on
its own merits, which is faster and simpler than waiting out a dedicated
reversal-confirmation step.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from market_data import MarketDataStore, TradeStore, Signal, DEFAULT_SYMBOL_WHITELIST
from .base import StrategyContext, StrategyEngine

log = logging.getLogger("okx_futures.observation")

CandleFetcher = Callable[[str, str, int], Awaitable[List[dict]]]


# ---------------------------------------------------------------------------
# Pure signal functions — unchanged from the previous engine. They're
# generic over whatever candle/trade timeframe is fed in, so the fast-scalp
# config below just calls them with shorter windows; the scoring logic
# itself (weighted-move trend, acceleration-scored pressure, consistency-
# scored volume expansion) didn't need to change.
# ---------------------------------------------------------------------------


def compute_trend_strength(candles: List[dict]) -> Dict:
    """Direction + strength from a sequence of {"ts","open","close"}
    candles (only these three keys are required — high/low aren't used),
    ordered oldest to newest internally regardless of input order.

    Each candle's move is weighted by its own size, so a +3% candle pulls
    much harder than a +0.1% candle instead of every candle counting as
    one equal "vote". strength_pct is the dominant side's share of the
    total weighted movement across all candles.

    Returns {"direction": "long"/"short"/"sideways", "strength_pct": 0-100,
    "net_move_pct": open-to-close move across the whole window}."""
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
    """Splits trades into `bucket_count` equal chronological slices
    spanning the trades' own oldest-to-newest timestamp range."""
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
    """0-100 score for how strongly `values` trend upward across their
    index, via a simple least-squares slope normalized against the
    largest slope that would be plausible for values living in
    `full_range`. 50 = flat, 100 = accelerating hard, 0 = reversing hard."""
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
    """Splits the window into `bucket_count` chronological slices and
    scores whether executed buy/sell pressure in `direction`'s favor is
    BUILDING across those slices, not just present on net. Combines two
    things equally: (a) the current (most recent slice) dominance level,
    and (b) whether that dominance has been accelerating slice to slice.

    Returns {"strength_pct": 0-100, "current_ratio": 0-1, "accelerating":
    bool}."""
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
    """Splits the window into `bucket_count` chronological slices and
    scores whether directional participation is expanding CONSISTENTLY
    across those slices rather than via a single spike. Combines equally:
    (a) how many consecutive slices increased vs the one before, and
    (b) overall growth of the back half vs the front half, capped at
    `target_multiplier`x.

    Returns {"strength_pct": 0-100, "expanding": bool}."""
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


# ---------------------------------------------------------------------------
# VWAP filter — added to guard against entering an already-extended move.
# Trend + pressure + volume can all agree while price is nonetheless
# stretched well past where the actual volume in the window traded; VWAP
# catches that case specifically. Deliberately the ONLY new signal added
# here — order book depth was considered too (top-5 levels can vanish
# between being read and an order landing, so it's noisier) and left out
# until VWAP alone has proven itself in practice.
# ---------------------------------------------------------------------------


def compute_vwap(trades: List[dict]) -> Optional[float]:
    """Volume-weighted average price across `trades`: sum(price*qty) /
    sum(qty). Fed the same `window_ms` trade-tape window already used for
    the pressure/volume checks (4 minutes by default), so this costs no
    extra fetch. Returns None if there's no volume to weight against,
    which the caller treats as "can't confirm" rather than a pass."""
    total_qty = sum(t["qty"] for t in trades)
    if total_qty <= 0:
        return None
    return sum(t["price"] * t["qty"] for t in trades) / total_qty


def _vwap_supports_direction(price: float, vwap: Optional[float], direction: str) -> bool:
    """True if `price` is on the side of `vwap` that direction wants:
    long wants price trading above VWAP (buyers paying up into strength,
    not chasing an extended move below where volume actually traded),
    short wants price below it. A missing VWAP (no volume in the window)
    never counts as support."""
    if vwap is None:
        return False
    if direction == "long":
        return price > vwap
    if direction == "short":
        return price < vwap
    return False


# ---------------------------------------------------------------------------
# Forming-candle helper
# ---------------------------------------------------------------------------


def _split_forming_and_closed(candles: List[dict]):
    """OKX's /market/candles response carries a `confirm` flag ("0" =
    still forming/live, "1" = closed). Splits `candles` (any order) into
    (forming_candle_or_None, closed_candles_newest_first) so the trend
    check only ever sees confirmed bars while the live-candle-agreement
    check can still read the current one. A missing/unknown confirm value
    is treated as closed (conservative: never mistakes a stale/malformed
    row for a live one)."""
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
    """True if `candle`'s own open->close move agrees with `direction`
    (long wants a bullish/green candle, short wants bearish/red). A
    missing candle never counts as support, which only makes it harder to
    open, never easier."""
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


# ---------------------------------------------------------------------------
# Observation state
# ---------------------------------------------------------------------------


@dataclass
class ObservationConfig:
    # Was 20 minutes. A scalp candidate that hasn't fired within a few
    # minutes is reading context that's already gone stale for a trade
    # meant to resolve in 3-10 minutes -- no point holding it longer.
    max_observation_minutes: float = 6.0

    # Was 5-minute candles / 6 buckets (~30 minutes). A fast scalp needs a
    # CURRENT read: 1-minute candles, 5 buckets, ~5 minutes of reference.
    trend_candle_bar: str = "1m"
    bucket_count: int = 5

    # Was 30 minutes (1_800_000ms) of trade tape. Shortened to 4 minutes
    # so the pressure/volume read reflects what's happening right now,
    # not what happened half an hour ago.
    window_ms: int = 240_000

    # Loosened slightly from the old 70%/0.3% -- a 5-minute window is
    # naturally noisier than a 30-minute one, and the profit target is
    # small, so demanding the same strictness would starve entries.
    min_trend_strength_pct: float = 65.0
    min_net_move_pct: float = 0.0015  # 0.15%

    # Was 300s/20 trades, sized for a 30-minute window. Scaled down with
    # window_ms: enough real trade prints to trust a bucket reading
    # without eating most of the (much shorter) observation budget.
    min_data_warmup_sec: float = 45.0
    min_data_trade_count: int = 15

    min_buy_pressure_strength_pct: float = 65.0
    min_volume_expansion_strength_pct: float = 55.0
    volume_expansion_multiplier: float = 1.4

    # Require price to be on the correct side of the session VWAP (see
    # compute_vwap). Kept as a toggle since it's new and unproven —
    # set False to instantly fall back to the pre-VWAP behavior.
    require_vwap_confirmation: bool = True

    # How many extra candles to fetch beyond bucket_count so there's still
    # bucket_count CLOSED candles even when the newest row is a
    # still-forming candle.
    candle_fetch_buffer: int = 2

    # Only symbols in this set are ever accepted into the watchlist.
    symbol_whitelist: Optional[frozenset] = field(default_factory=lambda: DEFAULT_SYMBOL_WHITELIST)


@dataclass
class CandidateObservation:
    symbol: str
    direction: str = ""  # "" whenever the current tick has no qualifying trend; recomputed fresh every tick, never "locked in"
    status: str = "OBSERVING"  # OBSERVING / ACCEPTED / EXPIRED
    started_at: float = field(default_factory=time.time)
    last_checked_at: float = 0.0

    data_ready: bool = False  # False until enough real trade-tape data has accumulated to trust pressure/volume

    trend: str = "sideways"
    trend_strength_pct: float = 0.0
    trend_ok: bool = False

    buy_pressure_strength_pct: float = 0.0
    buy_pressure_ratio: float = 0.0
    buy_pressure_ok: bool = False

    volume_strength_pct: float = 0.0
    volume_ok: bool = False

    vwap: Optional[float] = None
    vwap_ok: bool = False

    entry_price: float = 0.0

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.started_at

    @property
    def direction_letter(self) -> str:
        """Single-letter direction ('L'/'S') for compact logging. Returns
        '?' whenever the current tick has no qualifying direction."""
        return self.direction[0].upper() if self.direction else "?"

    def status_line(self) -> str:
        vwap_text = f"{self.vwap:.6g}" if self.vwap is not None else "-"
        base = (
            f"{self.symbol} status={self.status} direction={self.direction.upper() or '-'} "
            f"elapsed={self.elapsed_sec:.0f}s "
            f"trend={self.trend}:{self.trend_strength_pct:.0f}% "
            f"pressure={self.buy_pressure_strength_pct:.0f}% "
            f"volume={self.volume_strength_pct:.0f}% "
            f"vwap={vwap_text}"
        )
        if not self.data_ready:
            base += " (warming up)"
        return base


class ObservationWindowManager(StrategyEngine):
    """Tracks one CandidateObservation per watchlisted symbol and
    re-evaluates it fresh every tick — no persisted state machine, so a
    candidate's read can never go stale between ticks. Fully async and
    keyed per-symbol, so many candidates are observed concurrently
    without blocking each other or the rest of the trading loop.

    Implements strategy.base.StrategyEngine — see that module's
    docstring for the interface tracker.py talks to. This is the
    default strategy; switch to a different one by setting tracker.py's
    STRATEGY_NAME to e.g. "vwap_stg" or "ema_stg"."""

    name = "observation_engine"

    def __init__(
        self,
        trade_store: TradeStore,
        market_data: MarketDataStore,
        candle_fetcher: CandleFetcher,
        config: Optional[ObservationConfig] = None,
    ) -> None:
        self._trade_store = trade_store
        self._market_data = market_data
        self._candle_fetcher = candle_fetcher
        self.config = config or ObservationConfig()
        self._candidates: Dict[str, CandidateObservation] = {}
        self._lock = asyncio.Lock()

    async def sync_watchlist(self, watchlist_symbols) -> None:
        """Starts observing any symbol newly present in the watchlist and
        drops local state for any symbol that fell off it.

        Whatever the caller passes in is first filtered down to
        `config.symbol_whitelist` — the hard backstop that keeps the bot
        from ever watching (and therefore ever trading) a pair outside
        the approved list, regardless of what the upstream ranking/feed
        logic surfaces."""
        watchlist_symbols = set(watchlist_symbols)
        whitelist = self.config.symbol_whitelist
        if whitelist:
            rejected = watchlist_symbols - whitelist
            watchlist_symbols &= whitelist
            if rejected:
                log.debug(
                    f"[observation] ignoring {len(rejected)} non-whitelisted symbol(s) from the feed: "
                    f"{sorted(rejected)}"
                )
        async with self._lock:
            for symbol in watchlist_symbols:
                if symbol not in self._candidates:
                    self._candidates[symbol] = CandidateObservation(symbol=symbol)
                    log.info(
                        f"[observation] {symbol} added — observing for up to "
                        f"{self.config.max_observation_minutes:.0f}m"
                    )
            dropped = [s for s in self._candidates if s not in watchlist_symbols]
            for symbol in dropped:
                del self._candidates[symbol]

    async def snapshot(self) -> List[CandidateObservation]:
        async with self._lock:
            return list(self._candidates.values())

    async def evaluate(self, symbol: str) -> Optional[Signal]:
        """Runs one fast-scalp check for `symbol`. Returns a ready-to-open
        market_data.Signal once it's ACCEPTED (the candidate is removed
        from tracking immediately after). Returns None on every other
        tick — including a tick where trend or pressure/volume don't
        currently qualify, which simply leaves the candidate OBSERVING to
        be re-checked fresh next tick, with no memory of the failed check
        carried forward."""
        cfg = self.config
        async with self._lock:
            candidate = self._candidates.get(symbol)
        if candidate is None or candidate.status != "OBSERVING":
            return None

        if candidate.elapsed_sec >= cfg.max_observation_minutes * 60.0:
            candidate.status = "EXPIRED"
            log.info(
                f"[observation] {symbol} EXPIRED after {candidate.elapsed_sec / 60.0:.1f}m — discarding"
            )
            async with self._lock:
                self._candidates.pop(symbol, None)
            return None

        market = await self._market_data.get(symbol)
        if not market:
            return None
        candidate.last_checked_at = time.time()
        candidate.entry_price = market["last_price"]

        # --- Micro-trend, recomputed fresh every tick. ---
        try:
            raw_candles = await self._candle_fetcher(
                symbol, cfg.trend_candle_bar, cfg.bucket_count + cfg.candle_fetch_buffer
            )
        except Exception as exc:
            log.warning(f"[observation] {symbol} — could not fetch candles for the trend check: {exc}")
            return None
        forming_candle, closed_candles = _split_forming_and_closed(raw_candles)
        support_candle = forming_candle or (closed_candles[0] if closed_candles else None)

        trend_result = compute_trend_strength(closed_candles[: cfg.bucket_count])
        candidate.trend = trend_result["direction"]
        candidate.trend_strength_pct = trend_result["strength_pct"]
        trend_ok = (
            trend_result["direction"] != "sideways"
            and trend_result["strength_pct"] >= cfg.min_trend_strength_pct
            and abs(trend_result["net_move_pct"]) >= cfg.min_net_move_pct
        )
        candidate.trend_ok = trend_ok
        if not trend_ok:
            # No qualifying direction this tick -- nothing carried
            # forward, stays OBSERVING, re-read fresh next tick.
            candidate.direction = ""
            return None

        direction = trend_result["direction"]
        candidate.direction = direction

        window_trades = await self._trade_store.get_window(symbol, cfg.window_ms)
        was_ready = candidate.data_ready
        candidate.data_ready = (
            candidate.elapsed_sec >= cfg.min_data_warmup_sec
            and len(window_trades) >= cfg.min_data_trade_count
        )
        if candidate.data_ready and not was_ready:
            log.info(
                f"[observation] {symbol} data warm-up complete after {candidate.elapsed_sec:.0f}s "
                f"({len(window_trades)} trades in window) — pressure/volume checks now active"
            )
        if not candidate.data_ready:
            # Not enough real trade-tape history yet to trust a
            # pressure/volume reading off it.
            return None

        pressure = compute_buy_pressure_strength(window_trades, direction, cfg.bucket_count)
        volume = compute_volume_expansion_strength(window_trades, direction, cfg.bucket_count, cfg.volume_expansion_multiplier)
        candidate.buy_pressure_strength_pct = pressure["strength_pct"]
        candidate.buy_pressure_ratio = pressure["current_ratio"]
        candidate.volume_strength_pct = volume["strength_pct"]
        candidate.buy_pressure_ok = (
            pressure["accelerating"] and pressure["strength_pct"] >= cfg.min_buy_pressure_strength_pct
        )
        candidate.volume_ok = volume["expanding"] and volume["strength_pct"] >= cfg.min_volume_expansion_strength_pct

        # VWAP, computed from the same window_trades already fetched
        # above for pressure/volume -- no extra fetch needed.
        candidate.vwap = compute_vwap(window_trades)
        candidate.vwap_ok = (
            _vwap_supports_direction(candidate.entry_price, candidate.vwap, direction)
            if cfg.require_vwap_confirmation
            else True
        )

        healthy = (
            candidate.buy_pressure_ok
            and candidate.volume_ok
            and candidate.vwap_ok
            and _candle_supports_direction(support_candle, direction)
        )
        if healthy:
            candidate.status = "ACCEPTED"
            async with self._lock:
                self._candidates.pop(symbol, None)
            log.info(f"[observation] {symbol} ACCEPTED — {candidate.status_line()}")
            return Signal(
                symbol=candidate.symbol,
                direction=candidate.direction,
                confidence=1.0,  # not used for any gating decision — see execution_engine.py
                entry_price=candidate.entry_price,
                take_profit=candidate.entry_price,  # unused — execution_engine computes its own TP from target_net_profit_usdt
                stop_loss=candidate.entry_price,  # unused — execution_engine computes its own SL from target_stop_loss_usdt
                timestamp=time.time(),
                reasons=[
                    f"trend={candidate.trend}:{candidate.trend_strength_pct:.0f}%",
                    f"buy_pressure={candidate.buy_pressure_strength_pct:.0f}%",
                    f"volume_expansion={candidate.volume_strength_pct:.0f}%",
                ],
            )

        return None


def build(ctx: StrategyContext) -> ObservationWindowManager:
    """strategy.load_strategy()'s entry point — see strategy/base.py's
    module docstring for the contract every strategy module follows."""
    cfg = ctx.build_config(ObservationConfig)
    return ObservationWindowManager(ctx.trade_store, ctx.market_data, ctx.candle_fetcher, config=cfg)
