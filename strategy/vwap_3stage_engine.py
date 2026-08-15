"""
Flow Ignition Engine — order-flow burst detection for ETH-USDT scalping.

Where vwap_3stage_engine.py decides what to do based on WHERE price sits
relative to VWAP (far above / near / far below), this engine ignores
location entirely and instead watches HOW the trade tape is behaving
right now, against its own recent normal pace. It only ever asks one
question, continuously, per symbol: is the last few seconds of executed
order flow a statistically unusual, accelerating, one-sided burst —
backed by real price movement, not just volume — compared to how this
symbol has been trading over the last few minutes?

That's the whole mechanism. There is no VWAP, no swing-high/low levels,
no candle-based trend read, and no classical lagging indicator (moving
average, RSI, MACD, Bollinger Bands, ...) anywhere in this file. For a
target this tight (see take_profit_pct below — ~0.09%, the same scale
as the $1,882.60 -> $1,884.30 example this engine was built around), a
1-minute candle can fully round-trip through the target several times
before a candle-based indicator would even see the bar close. The trade
tape itself, read on an 8-second scale, is a faster and more honest
source of truth for a move this small than anything built on candles.

THE TWO WINDOWS

Every tick pulls one trade-tape window per symbol:

  baseline_window_ms (default 3 min) -- establishes the symbol's own
      recent "normal": how fast it's trading (trades/sec), and what a
      typical buy-vs-sell imbalance looks like over an ignition-length
      slice of that normal pace.

  ignition_window_ms (default 8 sec) -- the most recent slice of that
      same window, i.e. the live burst candidate being tested.

Nothing here is a fixed, absolute threshold the way vwap_3stage_engine's
distance-from-VWAP percentages are. Every gate below is the ignition
window measured AGAINST the baseline window's own recent behavior, so
the engine self-calibrates to whatever pace/volatility ETH-USDT happens
to be trading at right now, active session or quiet one, without
needing separate configs for each.

THE GATES (see evaluate() — all must pass, cheapest checks first)

  1. Cooldown + daily cap        -- pure rate-limiting, checked first,
                                     no trade-tape call needed to reject.
  2. Regime filter                -- baseline realized range must sit in
                                     a "not dead, not chaotic" band; a
                                     0.09% target is meaningless noise in
                                     a dead market and gets swept by
                                     swings in a chaotic one.
  3. Ignition z-score              -- the ignition window's own signed
                                     buy-sell delta, z-scored against the
                                     baseline's empirical distribution of
                                     same-length-slice deltas. This is
                                     what "statistically unusual" means
                                     here — not a fixed volume number.
  4. Tape acceleration            -- ignition-window trades/sec must be
                                     meaningfully faster than the
                                     baseline's own average pace. Same
                                     delta at the same pace as always
                                     isn't a burst, it's just noise that
                                     happened to lean one way.
  5. Price displacement            -- the ignition window's own price
                                     must have actually moved in the
                                     burst's direction. Volume without
                                     movement is absorption, which is a
                                     DIFFERENT, opposite read (more like
                                     vwap_3stage_engine's exhaustion
                                     engines) — not what this scalp
                                     wants.
  6. Dominant-side trade count     -- the burst must show multiple
                                     aggressive fills, not one large
                                     block print faking an imbalance.

If, and only if, all six pass does evaluate() return a Signal — LONG on
a positive z-score burst, SHORT on a negative one. Fixed-distance
take_profit_pct / stop_loss_pct off the current price is the exit;
there is no swing-level or VWAP-based target here, deliberately — the
whole point of this engine is a small, mechanical, repeatable scalp
distance, not a level-dependent one.

MEMORY ACROSS TICKS

Same "don't chase a stale read" philosophy as vwap_3stage_engine.py —
every qualifying-or-not decision above is recomputed fresh every tick
from the CURRENT trade tape; nothing about a rejected tick is carried
into the next one. The one deliberate exception is last_signal_at per
symbol, kept purely to enforce cooldown_sec — it never influences
direction or any of the six gates above, only whether the engine is
willing to look at all yet.

OVERTRADING CONTROL

Two independent layers, per the "~20 quality signals/day, not more"
brief this engine was built for:
  - cooldown_sec: minimum gap between two signals on the SAME symbol.
  - max_signals_per_day: a hard ceiling across the whole engine, reset
    at UTC midnight, that holds regardless of how many bursts the
    market throws at it on an unusually active day.
The six gates above are what should realistically keep daily count in
the ~20 range on an actively-traded pair like ETH-USDT; the daily cap
is the backstop that guarantees it even if they're recalibrated loosely
or the market is unusually bursty. Start with the defaults below, watch
signal frequency for a few days against your own volume data, and tune
ignition_min_zscore / tape_min_acceleration up (fewer, stronger-only
signals) or down (more signals) from there — exact call frequency for a
given zscore/acceleration bar depends on live order-flow statistics
this file has no way to know in advance.

A PRACTICAL NOTE ON THE TARGET SIZE

take_profit_pct=0.09% is small enough that exchange fees and slippage
are a real fraction of the edge, not a rounding error — check your
actual OKX ETH-USDT-SWAP taker/maker fee tier and factor it into
stop_loss_pct and position sizing before running this on live capital,
and consider whether entries can be posted maker-side (post-only)
rather than always taking liquidity.

Same Signal / StrategyEngine / TradeStore / MarketDataStore contract as
vwap_3stage_engine.py, so tracker.py talks to this module exactly the
same way — switch to it with tracker.py's
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


# ---------------------------------------------------------------------------
# Pure signal functions -- all operate on raw trade-tape dicts
# ({"timestamp" (ms), "price", "qty", "side": "buy"/"sell"}), the same
# shape TradeStore.get_window() returns in vwap_3stage_engine.py.
# ---------------------------------------------------------------------------


def compute_vwap(trades: List[dict]) -> Optional[float]:
    """Volume-weighted average price across `trades`. Returns None if
    there's no volume to weight against. Same formula as
    vwap_3stage_engine.compute_vwap -- kept local so this module has no
    dependency on a sibling strategy file."""
    total_qty = sum(t["qty"] for t in trades)
    if total_qty <= 0:
        return None
    return sum(t["price"] * t["qty"] for t in trades) / total_qty


def _bucketize_by_time(trades: List[dict], bucket_count: int) -> List[List[dict]]:
    """Splits `trades` into `bucket_count` equal-DURATION chronological
    slices spanning the trades' own oldest-to-newest timestamp range
    (not equal-count slices -- a slice covering a burst of rapid trading
    should be short, and a slice covering a lull should be long, if
    that's what the data actually looks like)."""
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
    """Net signed volume: buy_qty - sell_qty across `trades`."""
    buy = sum(t["qty"] for t in trades if t["side"] == "buy")
    sell = sum(t["qty"] for t in trades if t["side"] == "sell")
    return buy - sell


def compute_realized_range_pct(baseline_trades: List[dict]) -> float:
    """(max_price - min_price) / mean_price across `baseline_trades` --
    a simple realized-volatility proxy used only as a regime filter (see
    module docstring). Returns 0.0 for an empty or zero-priced window."""
    if not baseline_trades:
        return 0.0
    prices = [t["price"] for t in baseline_trades]
    mean_price = sum(prices) / len(prices)
    if mean_price <= 0:
        return 0.0
    return (max(prices) - min(prices)) / mean_price


def compute_baseline_delta_stats(baseline_trades: List[dict], slice_count: int) -> Dict:
    """Slices `baseline_trades` into `slice_count` equal-duration slices
    (see _bucketize_by_time) and returns the mean/stdev of each slice's
    own signed buy-sell delta (_signed_delta). This is the empirical
    "what does a normal window this long usually look like" distribution
    that the live ignition window's delta gets z-scored against in
    compute_ignition_zscore -- it's what makes the burst threshold
    adaptive to this symbol's current pace/volume instead of a fixed
    absolute number.

    Returns {"mean": float, "stdev": float, "slice_count_used": int}.
    stdev is 0.0 whenever fewer than 8 non-empty slices are available --
    too small a sample to trust a stdev estimate from yet."""
    buckets = _bucketize_by_time(baseline_trades, slice_count)
    deltas = [_signed_delta(b) for b in buckets if b]
    if len(deltas) < 8:
        return {"mean": 0.0, "stdev": 0.0, "slice_count_used": len(deltas)}
    mean = statistics.fmean(deltas)
    stdev = statistics.pstdev(deltas, mu=mean)
    return {"mean": mean, "stdev": stdev, "slice_count_used": len(deltas)}


def compute_ignition_zscore(ignition_trades: List[dict], baseline_stats: Dict) -> Dict:
    """Z-scores the ignition window's own signed delta against
    `baseline_stats`'s mean/stdev. Positive = unusually buy-heavy,
    negative = unusually sell-heavy. Returns {"delta": float, "zscore":
    float}; zscore is always 0.0 (neutral, never divides by zero) when
    the baseline doesn't have measurable variance yet."""
    delta = _signed_delta(ignition_trades)
    stdev = baseline_stats["stdev"]
    if stdev <= 0:
        return {"delta": delta, "zscore": 0.0}
    return {"delta": delta, "zscore": (delta - baseline_stats["mean"]) / stdev}


def compute_tape_acceleration(
    ignition_trades: List[dict], baseline_trades: List[dict], ignition_window_ms: int, baseline_window_ms: int
) -> Dict:
    """Trades-per-second in the ignition window vs. the baseline
    window's own average pace, using the nominal (requested) window
    durations as denominators rather than trades' own observed span --
    stable even when a window happens to contain very few or very
    tightly-clustered trades. >1.0 means the tape is moving faster right
    now than its recent normal.

    Returns {"ignition_rate": trades/sec, "baseline_rate": trades/sec,
    "multiplier": ignition_rate / baseline_rate}."""
    ignition_rate = len(ignition_trades) / max(ignition_window_ms / 1000.0, 1e-9)
    baseline_rate = len(baseline_trades) / max(baseline_window_ms / 1000.0, 1e-9)
    multiplier = (ignition_rate / baseline_rate) if baseline_rate > 0 else 0.0
    return {
        "ignition_rate": round(ignition_rate, 3),
        "baseline_rate": round(baseline_rate, 3),
        "multiplier": round(multiplier, 3),
    }


def compute_micro_displacement(ignition_trades: List[dict]) -> Dict:
    """Price displacement across the ignition window itself: VWAP of its
    older half of trades vs. VWAP of its newer half (steadier than a
    raw first-print-to-last-print read, which one outlier print could
    distort). Returns {"displacement_pct": float, "direction":
    "up"/"down"/"flat"}; flat/0.0 whenever there's too little data to
    split into two meaningful halves."""
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
    """Counts individual trades on `direction`'s side ("buy" for long,
    "sell" for short) within the ignition window -- guards against one
    oversized block print faking an entire burst on its own; a real
    ignition should show multiple aggressive fills."""
    side = "buy" if direction == "long" else "sell"
    return sum(1 for t in ignition_trades if t["side"] == side)


def composite_confidence(zscore: float, acceleration: float, displacement_pct: float, cfg: "FlowIgnitionConfig") -> float:
    """Blends how far each of the three confirming reads (zscore, tape
    acceleration, displacement) cleared its OWN minimum bar into a
    single 0-1 figure. Descriptive only -- every gate in evaluate() is a
    hard pass/fail already; this never itself decides whether a signal
    fires, only how strongly it cleared once it did."""
    z_component = min(1.0, abs(zscore) / (cfg.ignition_min_zscore * 2.0))
    accel_component = min(1.0, acceleration / (cfg.tape_min_acceleration * 2.0))
    disp_component = min(1.0, abs(displacement_pct) / (cfg.min_displacement_pct * 3.0))
    return round((z_component + accel_component + disp_component) / 3.0, 3)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FlowIgnitionConfig:
    # Restricted to ETH-USDT by default -- the calibration below (window
    # lengths, thresholds, tp/sl distances) was chosen around ETH's
    # typical liquidity and ~0.05-0.15%-scale scalp moves specifically,
    # and won't transfer to a thinner or a much more volatile pair
    # unmodified. Widen deliberately, not by default.
    symbol_whitelist: Optional[frozenset] = field(default_factory=lambda: frozenset({"ETH-USDT"}))

    # --- the two trade-tape windows (see module docstring) ---
    baseline_window_ms: int = 180_000  # 3 min -- "normal pace" reference window
    ignition_window_ms: int = 8_000  # 8 sec -- live burst window under test; must be << baseline_window_ms
    min_baseline_trade_count: int = 40  # baseline too thin to trust its own mean/stdev below this
    min_ignition_trade_count: int = 6  # ignition window too thin to trust its delta below this

    # --- regime filter (see compute_realized_range_pct) ---
    min_regime_range_pct: float = 0.0006  # below this the market's too quiet for a 0.09% target to mean anything
    max_regime_range_pct: float = 0.006  # above this it's too choppy -- tiny tp/sl just get swept by noise

    # --- ignition (burst) detection ---
    ignition_min_zscore: float = 2.5  # how many baseline-slice-stdevs the burst's delta must clear
    tape_min_acceleration: float = 1.8  # ignition trades/sec must be at least this many times the baseline pace
    min_displacement_pct: float = 0.00025  # 0.025% -- price must have actually started moving with the burst
    min_dominant_trade_count: int = 4  # burst must show multiple fills, not one block print

    # --- scalp exit: fixed distance from entry, no swing/VWAP target here by design ---
    take_profit_pct: float = 0.0009  # 0.09% -- matches the $1,882.60 -> $1,884.30 example this engine targets
    stop_loss_pct: float = 0.0006  # 0.06% -- tighter than target: ~1.5:1 reward:risk, ~40% breakeven win rate
    # before fees -- see the module docstring's note on fees at this target size.

    # --- overtrading control (see module docstring) ---
    cooldown_sec: float = 180.0  # minimum gap between two signals on the SAME symbol
    max_signals_per_day: int = 20  # hard ceiling across the whole engine, resets at UTC midnight


# ---------------------------------------------------------------------------
# Per-symbol state -- deliberately thin: no direction, no accumulated
# bias, nothing but the cooldown timer and the last tick's diagnostics
# (for snapshot()/status_line(), same role as
# vwap_3stage_engine.CandidateObservation.status_line()).
# ---------------------------------------------------------------------------


@dataclass
class SymbolFlowState:
    symbol: str
    last_signal_at: float = 0.0  # 0.0 = never fired yet; the ONLY thing remembered tick-to-tick, purely for cooldown
    last_checked_at: float = 0.0

    last_zscore: float = 0.0
    last_acceleration: float = 0.0
    last_displacement_pct: float = 0.0
    last_regime_range_pct: float = 0.0
    last_reject_reason: str = ""  # which gate the most recent tick failed at, "" if it passed all of them

    def status_line(self) -> str:
        return (
            f"{self.symbol} z={self.last_zscore:+.2f} accel={self.last_acceleration:.2f}x "
            f"disp={self.last_displacement_pct:+.3%} regime={self.last_regime_range_pct:.3%} "
            f"reject={self.last_reject_reason or '-'}"
        )


class FlowIgnitionEngine(StrategyEngine):
    """Tracks one SymbolFlowState per watchlisted symbol and, every
    tick, tests the most recent slice of that symbol's trade tape for a
    statistically unusual, accelerating, price-confirmed order-flow
    burst -- see this module's docstring for the full mechanism and
    strategy.base.StrategyEngine for the interface tracker.py talks to.

    Switch to this strategy by setting tracker.py's
    STRATEGY_NAME = "flow_ignition_engine"."""

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
        self._candle_fetcher = candle_fetcher  # accepted for build(ctx)/StrategyEngine parity only -- never called;
        # this engine reads the trade tape exclusively, see module docstring for why candles are skipped entirely.
        self.config = config or FlowIgnitionConfig()
        self._states: Dict[str, SymbolFlowState] = {}
        self._lock = asyncio.Lock()
        self._signals_today = 0
        self._day_started_at = self._utc_day_start()

    @staticmethod
    def _utc_day_start(ts: Optional[float] = None) -> float:
        """Midnight UTC (as a unix timestamp) on the day containing
        `ts` (default: now) -- used only to reset the daily signal
        counter at a consistent boundary regardless of server
        timezone."""
        t = time.gmtime(ts if ts is not None else time.time())
        return float(calendar.timegm((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, 0)))

    def _roll_day_if_needed(self, now: float) -> None:
        if now >= self._day_started_at + 86_400:
            if self._signals_today:
                log.info(f"[flow_ignition] day rollover — {self._signals_today} signal(s) fired in the prior 24h")
            self._signals_today = 0
            self._day_started_at = self._utc_day_start(now)

    async def sync_watchlist(self, watchlist_symbols) -> None:
        """Starts watching any symbol newly present in the watchlist and
        drops local state for any symbol that fell off it. Same
        whitelist backstop as vwap_3stage_engine.py."""
        watchlist_symbols = set(watchlist_symbols)
        whitelist = self.config.symbol_whitelist
        if whitelist:
            rejected = watchlist_symbols - whitelist
            watchlist_symbols &= whitelist
            if rejected:
                log.debug(
                    f"[flow_ignition] ignoring {len(rejected)} non-whitelisted symbol(s) from the feed: "
                    f"{sorted(rejected)}"
                )
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
        """Runs one fresh check for `symbol` against the six gates
        described in the module docstring, cheapest/no-network checks
        first. Returns a ready-to-open market_data.Signal the moment
        every gate passes, else None. Nothing about a failed tick is
        remembered into the next one except last_signal_at, purely for
        cooldown_sec."""
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

        # Ignition window is just the tail of the same baseline sample (by wall-clock recency), not a second
        # store call -- keeps both reads perfectly time-consistent with each other.
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

        if direction == "long":
            take_profit = price * (1 + cfg.take_profit_pct)
            stop_loss = price * (1 - cfg.stop_loss_pct)
        else:
            take_profit = price * (1 - cfg.take_profit_pct)
            stop_loss = price * (1 + cfg.stop_loss_pct)

        confidence = composite_confidence(z["zscore"], accel["multiplier"], disp["displacement_pct"], cfg)

        state.last_reject_reason = ""
        state.last_signal_at = now
        self._signals_today += 1

        log.info(
            f"[flow_ignition] SIGNAL: {symbol} {direction.upper()} (#{self._signals_today}/{cfg.max_signals_per_day} today)\n"
            f"  entry={price:.6g} tp={take_profit:.6g} sl={stop_loss:.6g}\n"
            f"  zscore={z['zscore']:+.2f} acceleration={accel['multiplier']:.2f}x "
            f"displacement={disp['displacement_pct']:+.3%} regime={regime_range_pct:.3%} "
            f"dominant_trades={dominant_count}"
        )
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=price,
            take_profit=take_profit,
            stop_loss=stop_loss,
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
    """strategy.load_strategy()'s entry point -- same contract as
    vwap_3stage_engine.build()."""
    cfg = ctx.build_config(FlowIgnitionConfig)
    return FlowIgnitionEngine(ctx.trade_store, ctx.market_data, ctx.candle_fetcher, config=cfg)
