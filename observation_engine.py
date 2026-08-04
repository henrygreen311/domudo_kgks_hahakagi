"""
Observation Window signal system — Trend Continuation & Reversal
Prediction Engine.

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
tracker.py's run_trading_loop for how this module is wired in.

It also replaces this module's OWN first design: a static-threshold
checker (trend >= 70%, pressure >= 70%, volume >= 60%, all-or-nothing,
discard the instant any one dropped). That version entered a lot of fake
breakouts and then missed the reversal that followed, because a single
soft dip in volume was enough to throw the candidate away right before
the real move. This version never does that — a weakening trend moves
into a MONITORING state instead of being discarded, and is only resolved
once the market actually shows its hand.

Mechanics: a symbol entering the watchlist becomes a candidate immediately
and is observed for up to `max_observation_minutes` (20 by default,
unchanged). Every tick:

  1. Determine the current dominant direction from the last
     `bucket_count` CLOSED 5-minute candles (`compute_trend_strength`,
     unchanged) — this is the reference direction everything below reacts
     to. No trade opens off this alone.

  2. Health check — is that direction still gaining strength? Checked via
     the same `compute_buy_pressure_strength` / `compute_volume_expansion_strength`
     used before, but now specifically for whether the dominant side's
     pressure is *accelerating* and its volume is *expanding* (not just
     above a static bar), plus whether the currently-forming 5m candle
     itself supports that direction. All three healthy -> open the trade
     immediately, same as the old engine's instant-open behavior.

  3. If not healthy, check for exhaustion: the dominant side's pressure
     AND volume both flat-or-falling, while the OPPOSITE side's pressure
     AND volume are both building. Only this specific combination moves
     the candidate into MONITORING — a merely soft/ambiguous tick (e.g.
     volume alone easing off) changes nothing and is re-checked next tick.
     This is the fix for the old discard-on-first-dip behavior.

  4. While MONITORING, every tick re-checks two, and only two, outcomes
     for the ORIGINAL (exhausted) direction:
       a. TREND_RECOVERED — the exhausted side's pressure/volume are
          building again and the current candle supports it once more.
          Opens in the ORIGINAL direction.
       b. REVERSAL_CONFIRMED — the opposite side's pressure/volume are
          building AND the last two CLOSED candles show a genuine
          structure break (close beyond the prior candle's low/high by at
          least `reversal_velocity_pct`, not just a wick through it).
          Opens in the OPPOSITE direction.
     Neither outcome -> stays in MONITORING, still not discarded, up to
     the overall `max_observation_minutes` ceiling.

Cross-exchange confirmation (cross_exchange_validator.py) is not part of
this decision path — it's still available as a standalone module reusing
the pure functions below, but ObservationWindowManager doesn't call one.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from market_data import MarketDataStore, TradeStore, DEFAULT_SYMBOL_WHITELIST

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
# Forming-candle / reversal-structure helpers
# ---------------------------------------------------------------------------


def _split_forming_and_closed(candles: List[dict]):
    """OKX's /market/candles response carries a `confirm` flag ("0" =
    still forming/live, "1" = closed) which okx_futures_client.get_candles
    passes straight through. Splits `candles` (any order) into
    (forming_candle_or_None, closed_candles_newest_first) so the trend and
    reversal-structure checks only ever see confirmed bars while the
    health/recovery checks can still read the live one. A missing/unknown
    confirm value is treated as closed (conservative: never mistakes a
    stale/malformed row for a live one)."""
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
    (long wants a bullish/green candle, short wants bearish/red). Used for
    the still-forming 5m candle when one's available, and as a
    same-shaped fallback on the latest closed candle otherwise (e.g. a
    fetcher/test double that doesn't report a confirm flag) — a missing
    candle never counts as support, which only makes it harder to open or
    recover a trade, never easier."""
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


def _reversal_confirmed(closed_candles: List[dict], exhausted_direction: str, velocity_pct: float) -> bool:
    """Confirms a genuine structure break using the last two CLOSED 5m
    candles (`closed_candles`, newest-first): for an exhausted LONG, the
    most recently closed candle must close below the prior candle's low
    by at least `velocity_pct` — not merely wick through it intraday; for
    an exhausted SHORT, it must close above the prior candle's high by
    the same margin. Pressure/volume alone (see the reversal-building
    check in ObservationWindowManager.evaluate) only earn a candidate the
    right to be checked against this — this is what actually opens the
    reversal trade."""
    if len(closed_candles) < 2:
        return False
    last_closed, prior_closed = closed_candles[0], closed_candles[1]
    if exhausted_direction == "long":
        level = prior_closed.get("low")
        if not level:
            return False
        return last_closed["close"] <= level * (1 - velocity_pct)
    else:
        level = prior_closed.get("high")
        if not level:
            return False
        return last_closed["close"] >= level * (1 + velocity_pct)


# ---------------------------------------------------------------------------
# Observation state
# ---------------------------------------------------------------------------


# DEFAULT_SYMBOL_WHITELIST (only these pairs may ever be watchlisted/
# observed/traded) now lives in market_data.py as the single source of
# truth, since SymbolRanker there needs the same set — imported above.


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

    # Reversal-confirmation velocity filter (see _reversal_confirmed):
    # requires the close to move at least this far beyond the prior
    # candle's low/high, not just touch or wick through it, before a
    # MONITORING candidate's reversal is treated as real.
    reversal_velocity_pct: float = 0.0015  # 0.15%

    # How many extra candles to fetch beyond bucket_count so there's
    # still bucket_count CLOSED candles for the trend calc even when the
    # newest row is a still-forming candle, plus enough closed candles
    # left over for the last-two-closed reversal-structure check.
    candle_fetch_buffer: int = 2

    # Only symbols in this set are ever accepted into the watchlist — see
    # sync_watchlist() below. An empty/None set disables filtering (every
    # symbol the feed offers gets watchlisted), so leave this populated.
    symbol_whitelist: Optional[frozenset] = field(default_factory=lambda: DEFAULT_SYMBOL_WHITELIST)


@dataclass
class CandidateObservation:
    symbol: str
    direction: str = ""  # "" until Step 1 establishes a dominant direction; then "long"/"short"
    status: str = "OBSERVING"  # OBSERVING / ACCEPTED / EXPIRED -- ObservationWindowManager's own bookkeeping
    started_at: float = field(default_factory=time.time)
    last_checked_at: float = 0.0

    # State-machine phase, distinct from `status` above -- this is the
    # finer-grained read of *why* (OBSERVING / MONITORING / TREND_RECOVERED
    # / REVERSAL_CONFIRMED), logged on every transition.
    state: str = "OBSERVING"
    monitoring: bool = False
    exhausted_side: str = ""  # "long"/"short" -- the direction currently being monitored for recovery vs reversal
    exhaustion_count: int = 0  # ticks spent in MONITORING since the current exhaustion began

    trend: str = "sideways"
    trend_strength_pct: float = 0.0
    trend_ok: bool = False

    # Dominant-side (i.e. candidate.direction's side) pressure/volume as of
    # the last tick -- "buy_" naming kept for backward compatibility with
    # callers (tracker.py's Signal.reasons) built before short candidates
    # existed; these reflect whichever side is currently dominant, not
    # literally "buy" for a short candidate.
    buy_pressure_strength_pct: float = 0.0
    buy_pressure_ratio: float = 0.0
    buy_pressure_ok: bool = False

    volume_strength_pct: float = 0.0
    volume_ok: bool = False

    entry_price: float = 0.0

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.started_at

    @property
    def direction_letter(self) -> str:
        """Single-letter direction ('L'/'S') for compact per-candidate
        logging, e.g. tracker.py's periodic watch-summary line. Returns
        '?' before Step 1 establishes a direction (`direction` starts as
        "" and stays that way for any candidate whose 30-min trend hasn't
        cleared yet) rather than making every caller guard against
        indexing into an empty string themselves."""
        return self.direction[0].upper() if self.direction else "?"

    def status_line(self) -> str:
        base = (
            f"{self.symbol} status={self.status} state={self.state} direction={self.direction.upper() or '-'} "
            f"elapsed={self.elapsed_sec:.0f}s "
            f"trend={self.trend}:{self.trend_strength_pct:.0f}% "
            f"pressure={self.buy_pressure_strength_pct:.0f}% "
            f"volume={self.volume_strength_pct:.0f}%"
        )
        if self.monitoring:
            base += f" monitoring_since_ticks={self.exhaustion_count} exhausted_side={self.exhausted_side.upper()}"
        return base


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
        drops local state for any symbol that fell off it — a symbol that
        leaves the watchlist is no longer a candidate, full stop.

        Whatever the caller passes in is first filtered down to
        `config.symbol_whitelist` — this is the hard backstop that keeps
        the bot from ever watching (and therefore ever trading) a pair
        outside the user's approved list, regardless of what the upstream
        ranking/feed logic surfaces."""
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

    async def evaluate(self, symbol: str) -> Optional[CandidateObservation]:
        """Runs one observation tick for `symbol`. Returns the candidate
        once (and only once — it's removed from tracking immediately
        after) it reaches ACCEPTED, whether via a healthy continuing
        trend, a recovered trend, or a confirmed reversal. Returns None
        on every other tick, including a weakening trend moved into
        MONITORING — that candidate is NOT discarded, only expiry
        (max_observation_minutes) or a real acceptance ever removes it."""
        cfg = self.config
        async with self._lock:
            candidate = self._candidates.get(symbol)
        if candidate is None or candidate.status != "OBSERVING":
            return None

        if candidate.elapsed_sec >= cfg.max_observation_minutes * 60.0:
            candidate.status = "EXPIRED"
            log.info(
                f"[observation] {symbol} EXPIRED after {candidate.elapsed_sec / 60.0:.1f}m "
                f"(last state={candidate.state}) — discarding"
            )
            async with self._lock:
                self._candidates.pop(symbol, None)
            return None

        market = await self._market_data.get(symbol)
        if not market:
            return None
        price = market["last_price"]
        candidate.last_checked_at = time.time()
        candidate.entry_price = price

        # --- Step 1: dominant direction from the last `bucket_count`
        # CLOSED 5m candles. Fetches a few extra so there's still
        # bucket_count closed candles even when the newest row is the
        # currently-forming one, plus enough closed candles left over for
        # the reversal-structure check later. ---
        try:
            raw_candles = await self._candle_fetcher(
                symbol, cfg.trend_candle_bar, cfg.bucket_count + cfg.candle_fetch_buffer
            )
        except Exception as exc:
            log.warning(f"[observation] {symbol} — could not fetch candles for the trend check: {exc}")
            raw_candles = []
        forming_candle, closed_candles = _split_forming_and_closed(raw_candles)
        # Fallback support-check candle when the fetcher doesn't report a
        # confirm flag at all: the latest closed candle stands in for the
        # forming one rather than support defaulting to False every tick.
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

        if not candidate.direction:
            # Direction hasn't been established yet -- Step 1 isn't a
            # trade decision on its own, just the reference point
            # everything below reacts to.
            if not trend_ok:
                return None
            candidate.direction = trend_result["direction"]
            candidate.state = "OBSERVING"
            log.info(
                f"[observation] {symbol} STATE=OBSERVING direction established="
                f"{candidate.direction.upper()} trend_strength={trend_result['strength_pct']:.0f}%"
            )

        direction = candidate.direction
        opposite = "short" if direction == "long" else "long"

        window_trades = await self._trade_store.get_window(symbol, cfg.window_ms)
        dom_pressure = compute_buy_pressure_strength(window_trades, direction, cfg.bucket_count)
        dom_volume = compute_volume_expansion_strength(window_trades, direction, cfg.bucket_count, cfg.volume_expansion_multiplier)
        opp_pressure = compute_buy_pressure_strength(window_trades, opposite, cfg.bucket_count)
        opp_volume = compute_volume_expansion_strength(window_trades, opposite, cfg.bucket_count, cfg.volume_expansion_multiplier)

        candidate.buy_pressure_strength_pct = dom_pressure["strength_pct"]
        candidate.buy_pressure_ratio = dom_pressure["current_ratio"]
        candidate.volume_strength_pct = dom_volume["strength_pct"]

        dominant_healthy = (
            dom_pressure["accelerating"]
            and dom_pressure["strength_pct"] >= cfg.min_buy_pressure_strength_pct
            and dom_volume["expanding"]
            and dom_volume["strength_pct"] >= cfg.min_volume_expansion_strength_pct
            and _candle_supports_direction(support_candle, direction)
        )
        candidate.buy_pressure_ok = dom_pressure["accelerating"] and dom_pressure["strength_pct"] >= cfg.min_buy_pressure_strength_pct
        candidate.volume_ok = dom_volume["expanding"] and dom_volume["strength_pct"] >= cfg.min_volume_expansion_strength_pct

        # --- Step 2 (not yet monitoring): is the established trend still
        # gaining strength? ---
        if not candidate.monitoring:
            if dominant_healthy:
                candidate.state = "ACCEPTED"
                candidate.status = "ACCEPTED"
                async with self._lock:
                    self._candidates.pop(symbol, None)
                log.info(f"[observation] {symbol} STATE=ACCEPTED (trend healthy) — {candidate.status_line()}")
                return candidate

            # --- Step 3: exhaustion check. Only THIS specific combination
            # (dominant side flat-or-falling on BOTH pressure and volume,
            # opposite side building on BOTH) moves the candidate into
            # MONITORING — anything softer/ambiguous changes nothing and
            # is simply re-checked next tick, still OBSERVING. ---
            exhaustion = (
                not dom_pressure["accelerating"]
                and dom_pressure["strength_pct"] < cfg.min_buy_pressure_strength_pct
                and not dom_volume["expanding"]
                and dom_volume["strength_pct"] < cfg.min_volume_expansion_strength_pct
                and opp_pressure["accelerating"]
                and opp_volume["expanding"]
            )
            if exhaustion:
                candidate.monitoring = True
                candidate.exhausted_side = direction
                candidate.exhaustion_count = 1
                candidate.state = "MONITORING"
                log.info(
                    f"[observation] {symbol} STATE=MONITORING exhausted_side={direction.upper()} — "
                    f"reason: {direction} pressure/volume falling "
                    f"(pressure={dom_pressure['strength_pct']:.0f}% volume={dom_volume['strength_pct']:.0f}%), "
                    f"{opposite} pressure/volume building "
                    f"(pressure={opp_pressure['strength_pct']:.0f}% volume={opp_volume['strength_pct']:.0f}%)"
                )
            return None

        # --- Step 4: MONITORING. Only two outcomes are ever checked for
        # the ORIGINAL (exhausted) direction; anything else leaves the
        # candidate in MONITORING, not discarded, up to
        # max_observation_minutes. ---
        candidate.exhaustion_count += 1
        exhausted = candidate.exhausted_side

        recovered = (
            dom_pressure["accelerating"]
            and dom_pressure["strength_pct"] >= cfg.min_buy_pressure_strength_pct
            and dom_volume["expanding"]
            and dom_volume["strength_pct"] >= cfg.min_volume_expansion_strength_pct
            and _candle_supports_direction(support_candle, exhausted)
        )
        if recovered:
            candidate.monitoring = False
            candidate.exhausted_side = ""
            candidate.state = "TREND_RECOVERED"
            candidate.status = "ACCEPTED"
            log.info(
                f"[observation] {symbol} STATE=TREND_RECOVERED — {exhausted} pressure/volume building again, "
                f"current candle supports it — opening {exhausted.upper()}"
            )
            async with self._lock:
                self._candidates.pop(symbol, None)
            return candidate

        reversal_building = (
            opp_pressure["accelerating"]
            and opp_volume["expanding"]
            and not dom_pressure["accelerating"]
            and not dom_volume["expanding"]
        )
        if reversal_building and _reversal_confirmed(closed_candles, exhausted, cfg.reversal_velocity_pct):
            new_direction = opposite
            candidate.direction = new_direction
            candidate.monitoring = False
            candidate.exhausted_side = ""
            candidate.state = "REVERSAL_CONFIRMED"
            candidate.status = "ACCEPTED"
            log.info(
                f"[observation] {symbol} STATE=REVERSAL_CONFIRMED — price closed beyond the prior candle's "
                f"{'low' if exhausted == 'long' else 'high'} by >= {cfg.reversal_velocity_pct:.2%}, "
                f"velocity filter passed — opening {new_direction.upper()}"
            )
            async with self._lock:
                self._candidates.pop(symbol, None)
            return candidate

        # Neither outcome yet -- stays in MONITORING.
        return None
