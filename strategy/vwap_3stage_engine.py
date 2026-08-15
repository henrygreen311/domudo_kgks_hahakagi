"""
MicroPulse Ignition Scalper — order-flow-imbalance + volatility-squeeze
breakout engine, tuned for sub-90-second ETH-USDT scalps where the
take-profit target can be as tight as ~0.09% (e.g. entry ≈ $1882.60,
TP ≈ $1884.30).

This is a DIFFERENT strategy shape than vwap_3stage_engine.py, not a
fork of it. That module classifies price's location relative to a long
VWAP and waits for a swing-high/low reaction — it's built to catch
multi-minute reversals/continuations off a reference level. This module
throws that away entirely: it doesn't care where price sits relative to
any long-window reference, and it never waits for a swing level to be
"reached". Instead it watches the trade tape itself, tick by tick, for
the specific signature that precedes a fast, small move:

  1. VOLATILITY SQUEEZE  - short-window price stdev compresses well
     below its own recent baseline. Compressed ranges are exactly where
     a small absolute move (a few dollars on ETH) produces a
     proportionally large, fast % move once it lets go -- which is what
     a ~0.09% TP needs to hit quickly rather than grinding toward it.

  2. ORDER FLOW IMBALANCE (OFI) - aggressor buy/sell volume ratio over a
     short window (default 12s), scored on both current dominance AND
     whether that dominance is building slice-to-slice. This is the
     "who's actually in control right now" read, using real executed
     trades rather than book snapshots.

  3. TICK-RATE ACCELERATION - trades-per-second across recent slices.
     Squeeze + imbalance without rising participation is often just thin
     order books drifting, not real intent -- this filters that out.

  4. MICRO-RANGE BREAKOUT - the tightest recent high/low (default 25s)
     acts as the trigger level. Price has to actually clear it (by a
     small buffer, to reject noise) in the OFI's direction, not just be
     near VWAP or near a swing level from candles.

  5. FORMING-CANDLE CONFIRMATION - the current (still-open) candle's
     own move must agree with direction, same spirit as
     observation_engine's candle check but applied to the live bar
     since this engine can't afford to wait for a bar to close.

All five must agree, evaluated fresh every tick with no memory of a
prior tick's partial pass (same "no locked-in direction across ticks"
philosophy as the VWAP engine) -> ENGAGE. There's no zone routing and
no separate long/short-only sub-engines: whichever side the order flow
and breakout agree on is the side taken, immediately, with a small
fixed TP/SL and a hard time-stop, because this engine is designed to
never hold a position waiting for a read to develop further.

IMPORTANT — fee/slippage sanity check: a 0.09% TP is extremely tight.
On most venues, round-trip taker fees plus slippage can easily eat
0.05-0.15% by themselves. `min_net_edge_pct` below exists so the engine
logs a loud warning if `take_profit_pct` isn't set comfortably above
your actual round-trip cost -- tune both to your venue's real fee
schedule before trusting this in size, not just to the example numbers
in this docstring.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from market_data import MarketDataStore, TradeStore, Signal, DEFAULT_SYMBOL_WHITELIST
from .base import StrategyContext, StrategyEngine

log = logging.getLogger("okx_futures.micropulse_ignition")

CandleFetcher = Callable[[str, str, int], Awaitable[List[dict]]]


def _resolve_attr(obj, candidate_names, label: str):
    """Pulls the first attribute in `candidate_names` that exists (and is
    not None) on `obj`. Exists because the loader hands this module a
    StrategyContext instance whose exact field names aren't visible from
    here -- rather than hardcoding one guess and failing with a bare
    AttributeError deep in a method call, this fails fast at construction
    time with a message naming what it looked for and what's actually on
    the object, so a naming mismatch is a one-line fix instead of a
    traceback hunt."""
    for name in candidate_names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    available = [a for a in dir(obj) if not a.startswith("_")]
    raise AttributeError(
        f"MicroPulseIgnitionEngine: couldn't find a {label} on the strategy context "
        f"(tried {list(candidate_names)}). Attributes available on context: {available}. "
        f"Adjust the candidate_names list in build()/__init__ to match your StrategyContext."
    )


# ---------------------------------------------------------------------------
# Pure signal functions
# ---------------------------------------------------------------------------


def _recent_trades(trades: List[dict], window_sec: float, now_ts: Optional[float] = None) -> List[dict]:
    """Trades whose timestamp (assumed epoch seconds; divide upstream if
    your TradeStore hands back ms) falls within the last `window_sec`
    seconds of the newest trade in the set (or of `now_ts` if given,
    so a slow-arriving batch doesn't silently widen the window)."""
    if not trades:
        return []
    ordered = sorted(trades, key=lambda t: t["timestamp"])
    cutoff = (now_ts if now_ts is not None else ordered[-1]["timestamp"]) - window_sec
    return [t for t in ordered if t["timestamp"] >= cutoff]


def _bucketize_by_time(trades: List[dict], bucket_count: int, window_sec: float, now_ts: Optional[float] = None) -> List[List[dict]]:
    """Splits the last `window_sec` seconds of `trades` into
    `bucket_count` equal-duration time slices (fixed-duration, unlike a
    trade-count split) so a burst of prints in one slice doesn't just
    relabel the clock -- it has to actually arrive in the recent slice
    to count toward "recent"."""
    buckets: List[List[dict]] = [[] for _ in range(max(bucket_count, 1))]
    window = _recent_trades(trades, window_sec, now_ts)
    if not window:
        return buckets
    end = now_ts if now_ts is not None else window[-1]["timestamp"]
    start = end - window_sec
    span = max(end - start, 1e-9)
    bucket_span = span / bucket_count
    for t in window:
        idx = min(int((t["timestamp"] - start) / bucket_span), bucket_count - 1)
        idx = max(idx, 0)
        buckets[idx].append(t)
    return buckets


def _slope_score(values: List[float], full_range: float) -> float:
    """0-100 least-squares slope score; 50 = flat, 100 = rising hard,
    0 = falling hard, normalized against the steepest plausible slope
    for values living in `full_range`."""
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


def compute_order_flow_imbalance(trades: List[dict], direction: str, bucket_count: int, window_sec: float, now_ts: Optional[float] = None) -> Dict:
    """Short-window aggressor buy/sell imbalance in `direction`'s favor,
    scored on current dominance AND whether it's building slice to
    slice. Same combination idea as a longer-window pressure read, just
    deliberately compressed onto a ~10-second horizon so it reacts to
    what's happening right now rather than the last few minutes.

    Returns {"strength_pct": 0-100, "current_ratio": 0-1, "accelerating": bool}."""
    side = "buy" if direction == "long" else "sell"
    other = "sell" if direction == "long" else "buy"
    buckets = _bucketize_by_time(trades, bucket_count, window_sec, now_ts)

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

    slope_score = _slope_score(pcts, full_range=0.5)
    level_score = pcts[-1] * 100.0
    strength_pct = round(0.5 * slope_score + 0.5 * level_score, 2)
    return {"strength_pct": strength_pct, "current_ratio": round(pcts[-1], 4), "accelerating": pcts[-1] > pcts[0]}


def compute_tick_rate_strength(trades: List[dict], bucket_count: int, window_sec: float, now_ts: Optional[float] = None) -> Dict:
    """Whether print frequency (trades per slice, regardless of side) is
    rising across the window -- a proxy for "is anyone actually showing
    up right now", used to reject a squeeze/imbalance read that's really
    just a couple of stray prints in a thin book.

    Returns {"strength_pct": 0-100, "accelerating": bool}."""
    buckets = _bucketize_by_time(trades, bucket_count, window_sec, now_ts)
    counts = [float(len(b)) for b in buckets]
    if len(counts) < 2 or all(c <= 0 for c in counts):
        return {"strength_pct": 0.0, "accelerating": False}
    max_count = max(counts) or 1.0
    strength_pct = round(_slope_score(counts, full_range=max_count), 2)
    return {"strength_pct": strength_pct, "accelerating": counts[-1] > counts[0]}


def _stdev(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return var ** 0.5


def compute_volatility_squeeze(trades: List[dict], short_window_sec: float, baseline_window_sec: float, now_ts: Optional[float] = None) -> Dict:
    """Compares short-window trade-price stdev against a longer baseline
    stdev. A low ratio means price has coiled into a tight range relative
    to its own recent behavior -- the setup this engine wants, since a
    small absolute release out of a tight coil is what clears a ~0.09%
    target quickly instead of needing a slow grind.

    Returns {"ratio": short_stdev / baseline_stdev (None if baseline has
    no volume/variance to compare against), "squeeze_score": 0-100 where
    100 = maximally compressed, "compressed": bool}."""
    baseline = _recent_trades(trades, baseline_window_sec, now_ts)
    short = _recent_trades(trades, short_window_sec, now_ts)
    if len(baseline) < 4 or len(short) < 3:
        return {"ratio": None, "squeeze_score": 0.0, "compressed": False}

    baseline_stdev = _stdev([t["price"] for t in baseline])
    short_stdev = _stdev([t["price"] for t in short])
    if baseline_stdev <= 0:
        return {"ratio": None, "squeeze_score": 0.0, "compressed": False}

    ratio = short_stdev / baseline_stdev
    # ratio 0.0 -> squeeze_score 100 (max compression); ratio >= 1.0 -> squeeze_score 0 (no compression at all)
    squeeze_score = round(max(0.0, min(1.0, 1.0 - ratio)) * 100.0, 2)
    return {"ratio": round(ratio, 4), "squeeze_score": squeeze_score, "compressed": ratio < 1.0}


def compute_micro_range(trades: List[dict], lookback_sec: float, now_ts: Optional[float] = None) -> Dict:
    """Highest/lowest traded price in the last `lookback_sec` seconds --
    the micro support/resistance this engine's breakout check uses,
    deliberately built from raw trade prints rather than closed candles
    so it stays current to the last few seconds, not the last closed bar.

    Returns {"high": ..., "low": ...} (None/None if no trades in window)."""
    window = _recent_trades(trades, lookback_sec, now_ts)
    if not window:
        return {"high": None, "low": None}
    prices = [t["price"] for t in window]
    return {"high": max(prices), "low": min(prices)}


def _split_forming_and_closed(candles: List[dict]):
    """Same idea as observation_engine's helper: separates the still-
    forming candle (confirm == "0") from closed ones (newest-first),
    treating a missing/unknown confirm flag as closed so a stale row is
    never mistaken for the live bar."""
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
    """True if `candle`'s own open->close move agrees with `direction`.
    A missing candle never counts as support -- only makes it harder to
    trigger, never easier."""
    if not candle:
        return False
    o, c = candle.get("open"), candle.get("close")
    if not o or c is None:
        return False
    return (c > o) if direction == "long" else (c < o)


def classify_ignition_direction(ofi_long: Dict, ofi_short: Dict, micro_range: Dict, last_price: float, breakout_buffer_pct: float) -> str:
    """Picks a candidate direction purely off which side's order flow is
    currently dominant AND whose micro-range edge price has actually
    cleared (by `breakout_buffer_pct`). Returns "long", "short", or ""
    if neither side has both a dominant flow read and a cleared edge."""
    high, low = micro_range.get("high"), micro_range.get("low")
    if high is None or low is None or last_price <= 0:
        return ""

    long_break = high is not None and last_price > high * (1 + breakout_buffer_pct)
    short_break = low is not None and last_price < low * (1 - breakout_buffer_pct)

    long_dominant = ofi_long["current_ratio"] > ofi_short["current_ratio"]
    short_dominant = ofi_short["current_ratio"] > ofi_long["current_ratio"]

    if long_break and long_dominant:
        return "long"
    if short_break and short_dominant:
        return "short"
    return ""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MicroPulseConfig:
    trend_candle_bar: str = "1m"  # smallest bar the venue offers; used only for the forming-candle confirmation
    candle_fetch_buffer: int = 2
    symbol_whitelist: Optional[frozenset] = field(default_factory=lambda: DEFAULT_SYMBOL_WHITELIST)

    # --- data warm-up ---
    min_data_warmup_sec: float = 20.0
    min_data_trade_count: int = 12

    # --- order flow imbalance (OFI) ---
    ofi_window_sec: float = 12.0
    ofi_bucket_count: int = 4
    min_ofi_strength_pct: float = 68.0
    require_ofi_accelerating: bool = True

    # --- tick-rate acceleration ---
    tick_rate_window_sec: float = 20.0
    tick_rate_bucket_count: int = 4
    min_tick_rate_strength_pct: float = 60.0

    # --- volatility squeeze ---
    squeeze_short_window_sec: float = 20.0
    squeeze_baseline_window_sec: float = 180.0
    min_squeeze_score: float = 55.0  # 0-100; higher = tighter compression required before ignition is even considered

    # --- micro-range breakout ---
    micro_range_lookback_sec: float = 25.0
    breakout_buffer_pct: float = 0.0004  # price must clear the range edge by this much -- rejects single-print noise

    # --- forming-candle confirmation ---
    require_candle_confirmation: bool = True

    # --- trade management: tight, fixed, time-boxed -- this engine does not hold ---
    take_profit_pct: float = 0.0009   # 0.09% -- matches the ETH-USDT example (entry ~1882.60 -> TP ~1884.30)
    stop_loss_pct: float = 0.0006     # independent of TP; tune to desk risk tolerance, not derived from it
    max_hold_seconds: float = 90.0    # hard time-stop regardless of P&L -- fast scalps only, never "wait it out"

    # --- cost sanity check (see module docstring) ---
    min_net_edge_pct: float = 0.0002  # take_profit_pct should clear round-trip fees+slippage by at least this much

    def __post_init__(self):
        if self.take_profit_pct < self.min_net_edge_pct:
            log.warning(
                "MicroPulseConfig: take_profit_pct=%.4f%% is at/under min_net_edge_pct=%.4f%% -- "
                "confirm this venue's real round-trip taker fee + expected slippage before running live.",
                self.take_profit_pct * 100, self.min_net_edge_pct * 100,
            )


# ---------------------------------------------------------------------------
# Candidate state
# ---------------------------------------------------------------------------


@dataclass
class MicroPulseCandidate:
    symbol: str
    direction: str = ""       # "" whenever the current tick doesn't qualify -- recomputed fresh every tick
    status: str = "OBSERVING"  # OBSERVING / ENGAGED / EXPIRED
    started_at: float = field(default_factory=time.time)
    last_checked_at: float = 0.0

    data_ready: bool = False

    ofi_strength_pct: float = 0.0
    ofi_accelerating: bool = False
    tick_rate_strength_pct: float = 0.0
    squeeze_score: float = 0.0
    squeeze_ratio: Optional[float] = None

    micro_high: Optional[float] = None
    micro_low: Optional[float] = None

    entry_price: float = 0.0
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.started_at

    @property
    def direction_letter(self) -> str:
        return self.direction[0].upper() if self.direction else "?"

    def status_line(self) -> str:
        base = (
            f"{self.symbol} status={self.status} direction={self.direction.upper() or '-'} "
            f"elapsed={self.elapsed_sec:.0f}s ofi={self.ofi_strength_pct:.0f}%"
            f"{'^' if self.ofi_accelerating else ''} tickrate={self.tick_rate_strength_pct:.0f}% "
            f"squeeze={self.squeeze_score:.0f}%"
        )
        if self.entry_price:
            base += f" entry={self.entry_price:.4g} tp={self.take_profit_price:.4g} sl={self.stop_loss_price:.4g}"
        if not self.data_ready:
            base += " (warming up)"
        return base


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MicroPulseIgnitionEngine(StrategyEngine):
    """Tracks one MicroPulseCandidate per watchlisted symbol. Every tick,
    pulls the recent trade tape + the current forming candle, runs the
    five checks described in the module docstring, and either emits a
    Signal (ENGAGED) or resets the candidate back to OBSERVING -- no
    partial state carries over between ticks, matching the "always a
    fresh read" philosophy used across this strategy set.
    """

    def __init__(
        self,
        context: StrategyContext,
        config: Optional[MicroPulseConfig] = None,
        *,
        trade_store: Optional[TradeStore] = None,
        market_data: Optional[MarketDataStore] = None,
        fetch_candles: Optional[CandleFetcher] = None,
    ):
        # trade_store / market_data / fetch_candles can be passed explicitly
        # (handy for tests or if the loader's build(ctx) wants to wire them
        # itself) but normally get pulled straight off the context, same as
        # the other engines in this strategy set.
        self.context = context
        self.trade_store = trade_store or _resolve_attr(
            context, ("trade_store", "trades", "trade_store_v0"), "TradeStore"
        )
        self.market_data = market_data or _resolve_attr(
            context, ("market_data", "market_data_store", "market_store"), "MarketDataStore"
        )
        self.fetch_candles = fetch_candles or _resolve_attr(
            context, ("fetch_candles", "get_candles", "candle_fetcher"), "candle fetch function"
        )
        self.cfg = config or MicroPulseConfig()
        self._candidates: Dict[str, MicroPulseCandidate] = {}

    def _candidate_for(self, symbol: str) -> MicroPulseCandidate:
        c = self._candidates.get(symbol)
        if c is None:
            c = MicroPulseCandidate(symbol=symbol)
            self._candidates[symbol] = c
        return c

    async def evaluate_symbol(self, symbol: str) -> Optional[Signal]:
        cfg = self.cfg
        candidate = self._candidate_for(symbol)
        candidate.last_checked_at = time.time()

        trades = await self.trade_store.get_recent_trades(symbol, window_sec=max(cfg.squeeze_baseline_window_sec, cfg.ofi_window_sec))
        if not trades or len(trades) < cfg.min_data_trade_count:
            candidate.data_ready = False
            candidate.direction = ""
            candidate.status = "OBSERVING"
            return None

        span_sec = trades[-1]["timestamp"] - trades[0]["timestamp"] if len(trades) > 1 else 0.0
        candidate.data_ready = span_sec >= cfg.min_data_warmup_sec
        if not candidate.data_ready:
            candidate.direction = ""
            candidate.status = "OBSERVING"
            return None

        last_price = trades[-1]["price"]
        now_ts = trades[-1]["timestamp"]

        squeeze = compute_volatility_squeeze(trades, cfg.squeeze_short_window_sec, cfg.squeeze_baseline_window_sec, now_ts)
        candidate.squeeze_score = squeeze["squeeze_score"]
        candidate.squeeze_ratio = squeeze["ratio"]
        if squeeze["squeeze_score"] < cfg.min_squeeze_score:
            candidate.direction = ""
            candidate.status = "OBSERVING"
            return None

        micro_range = compute_micro_range(trades, cfg.micro_range_lookback_sec, now_ts)
        candidate.micro_high, candidate.micro_low = micro_range["high"], micro_range["low"]

        ofi_long = compute_order_flow_imbalance(trades, "long", cfg.ofi_bucket_count, cfg.ofi_window_sec, now_ts)
        ofi_short = compute_order_flow_imbalance(trades, "short", cfg.ofi_bucket_count, cfg.ofi_window_sec, now_ts)

        direction = classify_ignition_direction(ofi_long, ofi_short, micro_range, last_price, cfg.breakout_buffer_pct)
        if not direction:
            candidate.direction = ""
            candidate.status = "OBSERVING"
            return None

        ofi = ofi_long if direction == "long" else ofi_short
        candidate.ofi_strength_pct = ofi["strength_pct"]
        candidate.ofi_accelerating = ofi["accelerating"]
        if ofi["strength_pct"] < cfg.min_ofi_strength_pct:
            candidate.direction = ""
            candidate.status = "OBSERVING"
            return None
        if cfg.require_ofi_accelerating and not ofi["accelerating"]:
            candidate.direction = ""
            candidate.status = "OBSERVING"
            return None

        tick_rate = compute_tick_rate_strength(trades, cfg.tick_rate_bucket_count, cfg.tick_rate_window_sec, now_ts)
        candidate.tick_rate_strength_pct = tick_rate["strength_pct"]
        if tick_rate["strength_pct"] < cfg.min_tick_rate_strength_pct:
            candidate.direction = ""
            candidate.status = "OBSERVING"
            return None

        if cfg.require_candle_confirmation:
            raw_candles = await self.fetch_candles(symbol, cfg.trend_candle_bar, cfg.candle_fetch_buffer + 1)
            forming, _closed = _split_forming_and_closed(raw_candles or [])
            if not _candle_supports_direction(forming, direction):
                candidate.direction = ""
                candidate.status = "OBSERVING"
                return None

        # All five checks passed on this tick -> engage.
        entry_price = last_price
        if direction == "long":
            tp = entry_price * (1 + cfg.take_profit_pct)
            sl = entry_price * (1 - cfg.stop_loss_pct)
        else:
            tp = entry_price * (1 - cfg.take_profit_pct)
            sl = entry_price * (1 + cfg.stop_loss_pct)

        candidate.direction = direction
        candidate.status = "ENGAGED"
        candidate.entry_price = entry_price
        candidate.take_profit_price = tp
        candidate.stop_loss_price = sl

        log.info("MicroPulseIgnition ENGAGE %s", candidate.status_line())

        return Signal(
            symbol=symbol,
            direction=direction,
            entry=entry_price,
            take_profit=tp,
            stop_loss=sl,
            max_hold_seconds=cfg.max_hold_seconds,
            reason="micropulse_ignition",
            meta={
                "ofi_strength_pct": ofi["strength_pct"],
                "ofi_accelerating": ofi["accelerating"],
                "tick_rate_strength_pct": tick_rate["strength_pct"],
                "squeeze_score": squeeze["squeeze_score"],
                "squeeze_ratio": squeeze["ratio"],
                "micro_high": micro_range["high"],
                "micro_low": micro_range["low"],
            },
        )

    async def run_once(self) -> List[Signal]:
        symbols = self.cfg.symbol_whitelist or DEFAULT_SYMBOL_WHITELIST
        signals: List[Signal] = []
        for symbol in symbols:
            try:
                sig = await self.evaluate_symbol(symbol)
            except Exception:
                log.exception("MicroPulseIgnitionEngine: evaluate_symbol failed for %s", symbol)
                continue
            if sig is not None:
                signals.append(sig)
        return signals

    async def run_forever(self, poll_interval_sec: float = 1.0):
        """Tight poll loop -- deliberately fast (default 1s) since every
        check in this engine is built off short (seconds-scale) windows;
        polling any slower would blur the exact ticks this strategy is
        designed to catch."""
        while True:
            for sig in await self.run_once():
                await self.context.emit_signal(sig)
            await asyncio.sleep(poll_interval_sec)


# ---------------------------------------------------------------------------
# Loader entry point
# ---------------------------------------------------------------------------


def build(ctx: StrategyContext) -> MicroPulseIgnitionEngine:
    """Factory function strategy/__init__.py's load_strategy() looks for
    on every strategy/*.py module -- without this, loading this module by
    name fails with "has no build(ctx) factory function", the same error
    you'd get from any strategy file missing it (including the original
    vwap_3stage_engine.py as pasted -- it was cut off mid-class and never
    showed one either, so add the equivalent there too if you're loading
    that one)."""
    return MicroPulseIgnitionEngine(ctx)
