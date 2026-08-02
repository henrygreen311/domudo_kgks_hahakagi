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
tick, four required signals are re-evaluated:

  1. 5-minute-candle trend over the last ~30 minutes (LONG/SHORT/SIDEWAYS)
  2. executed buy-vs-sell volume ratio (>= 70/30 on the trend's side)
  3. net aggressive delta persistence (money flow keeps favoring the trend
     direction throughout the window, not just on net)
  4. volume expansion (directional participation growing vs a longer
     baseline)

...plus cross-exchange confirmation of the same four signals (see
cross_exchange_validator.py, which reuses the pure functions below). The
trade opens the instant all of this holds simultaneously — observation
does not wait out the full window once conditions are met. If the window
elapses first, the candidate is discarded.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from market_data import MarketDataStore, TradeStore, compute_order_flow_metrics

log = logging.getLogger("okx_futures.observation")

CandleFetcher = Callable[[str, str, int], Awaitable[List[dict]]]


# ---------------------------------------------------------------------------
# Pure signal functions — shared with cross_exchange_validator.py so the
# per-exchange confirmation checks the *same* logic against the *same*
# thresholds, not a parallel reimplementation that could quietly drift.
# ---------------------------------------------------------------------------


def classify_trend(candles: List[dict], move_threshold_pct: float) -> str:
    """LONG/SHORT/SIDEWAYS from a sequence of {"ts","open","close"}
    candles (only these three keys are required — high/low aren't used),
    ordered oldest to newest internally regardless of input order. Trend
    is the net % move from the earliest candle's open to the newest
    candle's close; below `move_threshold_pct` in either direction counts
    as SIDEWAYS (reject)."""
    if not candles or len(candles) < 2:
        return "sideways"
    ordered = sorted(candles, key=lambda c: c["ts"])
    open_price = ordered[0]["open"]
    close_price = ordered[-1]["close"]
    if not open_price:
        return "sideways"
    net_change = (close_price - open_price) / open_price
    if net_change >= move_threshold_pct:
        return "long"
    if net_change <= -move_threshold_pct:
        return "short"
    return "sideways"


def compute_buy_sell_ratio(trades: List[dict]) -> Tuple[Optional[str], float]:
    """(dominant_side, ratio) from executed trades only. Every trade in
    TradeStore/SymbolState is already a taker-side (aggressor) execution,
    never resting limit volume — see market_data.TradeStore.apply_trade —
    so this is inherently "executed market trades only", satisfying that
    requirement without any extra filtering. Reuses
    compute_order_flow_metrics's aggressive_buy_pct/aggressive_sell_pct
    rather than recomputing the same ratio a second way."""
    if not trades:
        return None, 0.0
    metrics = compute_order_flow_metrics(trades)
    buy_pct, sell_pct = metrics["aggressive_buy_pct"], metrics["aggressive_sell_pct"]
    if buy_pct <= 0 and sell_pct <= 0:
        return None, 0.0
    if buy_pct >= sell_pct:
        return "long", buy_pct
    return "short", sell_pct


def compute_net_aggressive_delta_ok(trades: List[dict], direction: str, bucket_count: int) -> bool:
    """True only if aggressive money flow has continued favoring
    `direction` throughout the window, not just on net. The window is
    split into `bucket_count` equal chronological slices; EVERY slice's
    buy-sell delta must agree with `direction` (an empty slice is treated
    as "quiet", not a reversal, and doesn't fail the check — but a slice
    with real opposing flow does). A single early-window slice going the
    other way means flow reversed partway through, which isn't "continues
    pushing price in the trade direction"."""
    if not trades or bucket_count < 1:
        return False
    ordered = sorted(trades, key=lambda t: t["timestamp"])
    start, end = ordered[0]["timestamp"], ordered[-1]["timestamp"]
    span = end - start
    buckets: List[List[dict]]
    if span <= 0:
        buckets = [ordered]
    else:
        bucket_span = span / bucket_count
        buckets = [[] for _ in range(bucket_count)]
        for t in ordered:
            idx = min(int((t["timestamp"] - start) / bucket_span), bucket_count - 1)
            buckets[idx].append(t)

    for bucket in buckets:
        if not bucket:
            continue
        buy_vol = sum(t["qty"] for t in bucket if t["side"] == "buy")
        sell_vol = sum(t["qty"] for t in bucket if t["side"] == "sell")
        delta = buy_vol - sell_vol
        if direction == "long" and delta < 0:
            return False
        if direction == "short" and delta > 0:
            return False
    return True


def compute_volume_expansion_ok(
    recent_trades: List[dict],
    baseline_trades: List[dict],
    direction: str,
    recent_window_ms: float,
    baseline_window_ms: float,
    multiplier: float,
) -> bool:
    """True when directional participation (buy volume for a long
    candidate, sell volume for a short one — not raw total volume, so a
    burst of volume on the WRONG side doesn't count as "expansion") in
    the recent window has grown to at least `multiplier`x the
    recent-window-sized average over the longer baseline window."""
    if recent_window_ms <= 0 or baseline_window_ms <= 0:
        return False
    side = "buy" if direction == "long" else "sell"
    recent_vol = sum(t["qty"] for t in recent_trades if t["side"] == side)
    baseline_vol = sum(t["qty"] for t in baseline_trades if t["side"] == side)
    windows_in_baseline = baseline_window_ms / recent_window_ms
    baseline_avg = (baseline_vol / windows_in_baseline) if windows_in_baseline > 0 else 0.0
    if baseline_avg <= 0:
        return recent_vol > 0
    return recent_vol >= baseline_avg * multiplier


# ---------------------------------------------------------------------------
# Observation state
# ---------------------------------------------------------------------------


@dataclass
class ObservationConfig:
    max_observation_minutes: float = 20.0

    trend_candle_bar: str = "5m"
    trend_candle_count: int = 6  # ~30 minutes of 5m candles
    trend_move_threshold_pct: float = 0.003

    ratio_window_ms: int = 300_000  # 5 minutes of executed trades for ratio + delta
    min_buy_sell_ratio: float = 0.70

    delta_bucket_count: int = 3

    volume_recent_window_ms: int = 300_000
    volume_baseline_window_ms: int = 1_800_000  # 30 minutes
    volume_expansion_multiplier: float = 1.5

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
    trend_ok: bool = False
    buy_sell_ratio: float = 0.0
    ratio_ok: bool = False
    net_aggressive_delta_ok: bool = False
    volume_expansion_ok: bool = False
    cross_exchange_ok: bool = False
    cross_exchange_agreeing: int = 0
    cross_exchange_reason: str = ""

    entry_price: float = 0.0

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.started_at

    @property
    def local_conditions_met(self) -> bool:
        return self.trend_ok and self.ratio_ok and self.net_aggressive_delta_ok and self.volume_expansion_ok

    @property
    def all_conditions_met(self) -> bool:
        return self.local_conditions_met and self.cross_exchange_ok

    def status_line(self) -> str:
        return (
            f"{self.symbol} status={self.status} direction={self.direction.upper()} "
            f"elapsed={self.elapsed_sec:.0f}s trend={self.trend}({'OK' if self.trend_ok else 'no'}) "
            f"ratio={self.buy_sell_ratio:.2f}({'OK' if self.ratio_ok else 'no'}) "
            f"delta={'OK' if self.net_aggressive_delta_ok else 'no'} "
            f"volume={'OK' if self.volume_expansion_ok else 'no'} "
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
            candles = await self._candle_fetcher(symbol, cfg.trend_candle_bar, cfg.trend_candle_count)
        except Exception as exc:
            log.warning(f"[observation] {symbol} — could not fetch candles for the trend check: {exc}")
            candles = []
        trend = classify_trend(candles, cfg.trend_move_threshold_pct)
        candidate.trend = trend
        candidate.last_checked_at = time.time()
        candidate.entry_price = price

        if trend == "sideways":
            candidate.trend_ok = False
            candidate.ratio_ok = False
            candidate.net_aggressive_delta_ok = False
            candidate.volume_expansion_ok = False
            candidate.cross_exchange_ok = False
            return None

        direction = trend
        candidate.direction = direction
        candidate.trend_ok = True

        ratio_trades = await self._trade_store.get_window(symbol, cfg.ratio_window_ms)
        dominant_side, ratio = compute_buy_sell_ratio(ratio_trades)
        candidate.buy_sell_ratio = ratio
        candidate.ratio_ok = dominant_side == direction and ratio >= cfg.min_buy_sell_ratio
        candidate.net_aggressive_delta_ok = compute_net_aggressive_delta_ok(ratio_trades, direction, cfg.delta_bucket_count)

        recent_trades = await self._trade_store.get_window(symbol, cfg.volume_recent_window_ms)
        baseline_trades = await self._trade_store.get_window(symbol, cfg.volume_baseline_window_ms)
        candidate.volume_expansion_ok = compute_volume_expansion_ok(
            recent_trades, baseline_trades, direction,
            cfg.volume_recent_window_ms, cfg.volume_baseline_window_ms, cfg.volume_expansion_multiplier,
        )

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
