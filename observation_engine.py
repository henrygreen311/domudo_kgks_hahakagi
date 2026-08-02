"""
Observation Window signal system.

This replaces the entire legacy evidence/scoring pipeline as the trading
bot's sole trade-decision logic:

  - evidence_pipeline.py (RollingEvidenceAccumulator / PersistenceValidator
    / weighted scoring / direction-consistency / raw signal counts)
  - event_confirmation.py (whale trades, order-book sweeps, momentum
    confirmation)
  - confidence_engine.py-driven confidence scores/percentages
  - signal_store.py (the signals_histories table those evidence snapshots
    were persisted to)

None of the above are imported here or anywhere in the new pipeline. See
tracker.py's run_trading_loop for how this module is wired in; those four
files are no longer referenced anywhere in the project and can be deleted.

Mechanics: a symbol entering the watchlist becomes a candidate immediately
and is observed for up to `max_observation_minutes` (20 by default). Every
tick, three required signals are re-evaluated, all measured as a STRENGTH
over the same lookback (`bucket_count` slices — 6 x 5-minute candles/
buckets = 30 minutes by default) rather than a rigid one-shot pass/fail:

  1. Trend strength: candles are weighted by how big their move was (a
     +3% candle counts far more than a +0.1% candle), not just counted as
     bullish/bearish. strength_pct = the winning side's share of total
     weighted movement.
  2. Buy/sell pressure strength: the window is split into `bucket_count`
     chronological slices and the dominant side's share is tracked slice
     by slice — this rewards buying/selling pressure that is *building*
     (e.g. 58/61/66/71/76/82) over pressure that is merely present but
     flat (e.g. 71/70/71/70/71/70).
  3. Volume expansion strength: same slicing, applied to directional
     volume — rewards steady, accelerating participation over a single
     one-off spike.

...plus cross-exchange confirmation of the same three signals (see
cross_exchange_validator.py, which reuses the pure functions below). The
trade opens the instant all of this holds simultaneously — observation
does not wait out the full window once conditions are met. If the window
elapses first, the candidate is discarded.

Net-aggressive-delta persistence (the old "every slice must agree with
direction or the whole thing fails" check) has been removed. Its intent
— reward flow that keeps pushing the same way — is now covered by the
buy/sell pressure *strength* score above, which measures that directly
instead of as a binary all-or-nothing gate.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from market_data import MarketDataStore, TradeStore

log = logging.getLogger("okx_futures.observation")

CandleFetcher = Callable[[str, str, int], Awaitable[List[dict]]]


# ---------------------------------------------------------------------------
# Pure signal functions — shared with cross_exchange_validator.py so the
# per-exchange confirmation checks the *same* logic against the *same*
# thresholds, not a parallel reimplementation that could quietly drift.
# ---------------------------------------------------------------------------


def compute_trend_strength(candles: List[dict]) -> Dict:
    """Direction + strength from a sequence of {"ts","open","close"}
    candles (only these three keys are required — high/low aren't used),
    ordered oldest to newest internally regardless of input order.

    Each candle's move is weighted by its own size, so a +3% candle pulls
    much harder than a +0.1% candle instead of every candle counting as
    one equal "vote". strength_pct is the dominant side's share of the
    total weighted movement across all candles (e.g. 5 bullish candles
    against 1 small bearish one, weighted, might land at 83%).

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
    spanning the trades' own oldest-to-newest timestamp range. Shared by
    the two strength functions below so both slice the window the exact
    same way."""
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
    `full_range` (e.g. a ratio moving 0.5->1.0, dominant-side share, over
    the whole window). 50 = flat, 100 = accelerating hard, 0 = reversing
    hard."""
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
    and (b) whether that dominance has been accelerating slice to slice —
    58/61/66/71/76/82 scores far higher than a flat 71/70/71/70/71/70
    even though the flat case's overall ratio looks fine in isolation.

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

    # Dominant-side share realistically ranges 0.5 (even) -> 1.0 (total
    # dominance), so that's the "full range" a max-strength slope covers.
    slope_score = _slice_slope_score(pcts, full_range=0.5)
    level_score = pcts[-1] * 100.0
    strength_pct = round(0.5 * slope_score + 0.5 * level_score, 2)
    return {"strength_pct": strength_pct, "current_ratio": round(pcts[-1], 4), "accelerating": pcts[-1] > pcts[0]}


def compute_volume_expansion_strength(trades: List[dict], direction: str, bucket_count: int, target_multiplier: float) -> Dict:
    """Splits the window into `bucket_count` chronological slices and
    scores whether directional participation (buy volume for a long
    candidate, sell volume for a short one) is expanding CONSISTENTLY
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
# Observation state
# ---------------------------------------------------------------------------


@dataclass
class ObservationConfig:
    max_observation_minutes: float = 20.0

    # Shared 30-minute lookback (6 x 5-minute buckets) used by all three
    # strength checks below — trend, buy pressure, and volume expansion —
    # so "over the last 30 minutes" means the same window everywhere.
    trend_candle_bar: str = "5m"
    bucket_count: int = 6  # 6 x 5m = ~30 minutes
    window_ms: int = 1_800_000  # 30 minutes of executed trades, sliced into bucket_count buckets

    min_trend_strength_pct: float = 70.0
    min_net_move_pct: float = 0.003  # still require some real overall move, not just a lopsided-but-flat window

    min_buy_pressure_strength_pct: float = 70.0

    min_volume_expansion_strength_pct: float = 60.0
    volume_expansion_multiplier: float = 1.5  # target growth (back-half vs front-half of the window) for a max volume score

    require_cross_exchange: bool = True
    min_agreeing_exchanges: int = 5
    total_exchanges: int = 7


@dataclass
class CandidateObservation:
    symbol: str
    direction: str = "long"
    status: str = "OBSERVING"  # OBSERVING / ACCEPTED / EXPIRED
    started_at: float = field(default_factory=time.time)
    last_checked_at: float = 0.0

    trend: str = "sideways"
    trend_strength_pct: float = 0.0
    trend_ok: bool = False

    buy_pressure_strength_pct: float = 0.0
    buy_pressure_ratio: float = 0.0
    buy_pressure_ok: bool = False

    volume_strength_pct: float = 0.0
    volume_ok: bool = False

    cross_exchange_ok: bool = False
    cross_exchange_agreeing: int = 0
    cross_exchange_reason: str = ""

    entry_price: float = 0.0

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.started_at

    @property
    def local_conditions_met(self) -> bool:
        return self.trend_ok and self.buy_pressure_ok and self.volume_ok

    @property
    def all_conditions_met(self) -> bool:
        return self.local_conditions_met and self.cross_exchange_ok

    def status_line(self) -> str:
        return (
            f"{self.symbol} status={self.status} direction={self.direction.upper()} "
            f"elapsed={self.elapsed_sec:.0f}s "
            f"trend={self.trend}:{self.trend_strength_pct:.0f}%({'OK' if self.trend_ok else 'no'}) "
            f"buy_pressure={self.buy_pressure_strength_pct:.0f}%({'OK' if self.buy_pressure_ok else 'no'}) "
            f"volume={self.volume_strength_pct:.0f}%({'OK' if self.volume_ok else 'no'}) "
            f"cross_exchange={self.cross_exchange_agreeing}({'OK' if self.cross_exchange_ok else 'no'})"
        )


class ObservationWindowManager:
    """Tracks one CandidateObservation per watchlisted symbol and
    re-evaluates it every tick. Fully async and keyed per-symbol, so many
    candidates are observed concurrently without blocking each other or
    the rest of the trading loop — each evaluate() call only touches its
    own symbol's state under the shared lock for the brief dict
    read/write, never for the actual (awaited) data fetching."""

    def __init__(
        self,
        trade_store: TradeStore,
        market_data: MarketDataStore,
        candle_fetcher: CandleFetcher,
        cross_exchange_validator=None,
        config: Optional[ObservationConfig] = None,
    ) -> None:
        self._trade_store = trade_store
        self._market_data = market_data
        self._candle_fetcher = candle_fetcher
        self._cross_exchange_validator = cross_exchange_validator
        self.config = config or ObservationConfig()
        self._candidates: Dict[str, CandidateObservation] = {}
        self._lock = asyncio.Lock()

    async def sync_watchlist(self, watchlist_symbols) -> None:
        """Starts observing any symbol newly present in the watchlist and
        drops local state for any symbol that fell off it — a symbol that
        leaves the watchlist is no longer a candidate, full stop."""
        watchlist_symbols = set(watchlist_symbols)
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

    async def evaluate(self, symbol: str) -> Optional[CandidateObservation]:
        """Runs one observation tick for `symbol`. Returns the candidate
        once (and only once — it's removed from tracking immediately
        after) it reaches ACCEPTED; returns None while still observing,
        on this tick's failure, or once it expires."""
        cfg = self.config
        async with self._lock:
            candidate = self._candidates.get(symbol)
        if candidate is None or candidate.status != "OBSERVING":
            return None

        if candidate.elapsed_sec >= cfg.max_observation_minutes * 60.0:
            candidate.status = "EXPIRED"
            log.info(
                f"[observation] {symbol} EXPIRED after {candidate.elapsed_sec / 60.0:.1f}m "
                f"without meeting all conditions — discarding"
            )
            async with self._lock:
                self._candidates.pop(symbol, None)
            return None

        market = await self._market_data.get(symbol)
        if not market:
            return None
        price = market["last_price"]

        try:
            candles = await self._candle_fetcher(symbol, cfg.trend_candle_bar, cfg.bucket_count)
        except Exception as exc:
            log.warning(f"[observation] {symbol} — could not fetch candles for the trend check: {exc}")
            candles = []
        trend_result = compute_trend_strength(candles)
        trend = trend_result["direction"]
        candidate.trend = trend
        candidate.trend_strength_pct = trend_result["strength_pct"]
        candidate.last_checked_at = time.time()
        candidate.entry_price = price

        trend_ok = (
            trend != "sideways"
            and trend_result["strength_pct"] >= cfg.min_trend_strength_pct
            and abs(trend_result["net_move_pct"]) >= cfg.min_net_move_pct
        )
        candidate.trend_ok = trend_ok

        if trend == "sideways":
            candidate.buy_pressure_ok = False
            candidate.volume_ok = False
            candidate.cross_exchange_ok = False
            return None

        direction = trend
        candidate.direction = direction

        if not trend_ok:
            candidate.buy_pressure_ok = False
            candidate.volume_ok = False
            candidate.cross_exchange_ok = False
            return None

        window_trades = await self._trade_store.get_window(symbol, cfg.window_ms)

        pressure = compute_buy_pressure_strength(window_trades, direction, cfg.bucket_count)
        candidate.buy_pressure_strength_pct = pressure["strength_pct"]
        candidate.buy_pressure_ratio = pressure["current_ratio"]
        candidate.buy_pressure_ok = pressure["strength_pct"] >= cfg.min_buy_pressure_strength_pct

        volume = compute_volume_expansion_strength(window_trades, direction, cfg.bucket_count, cfg.volume_expansion_multiplier)
        candidate.volume_strength_pct = volume["strength_pct"]
        candidate.volume_ok = volume["strength_pct"] >= cfg.min_volume_expansion_strength_pct

        if not candidate.local_conditions_met:
            candidate.cross_exchange_ok = False
            return None

        if cfg.require_cross_exchange and self._cross_exchange_validator is not None:
            cx_result = await self._cross_exchange_validator.validate(symbol, direction)
            candidate.cross_exchange_agreeing = cx_result.agreeing_exchanges
            candidate.cross_exchange_reason = cx_result.reason or ""
            candidate.cross_exchange_ok = cx_result.decision == "accepted"
            if not candidate.cross_exchange_ok:
                return None
        else:
            candidate.cross_exchange_ok = True

        candidate.status = "ACCEPTED"
        async with self._lock:
            self._candidates.pop(symbol, None)
        log.info(f"[observation] {symbol} ACCEPTED — {candidate.status_line()}")
        return candidate
