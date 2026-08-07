"""
MovementTracker — the bot's flight recorder / black box.

    Execution Engine
            |
      Trade Opened
            |
            v
     MovementTracker  <-- reads current price from MarketDataStore only
            |
      Trade Closed
            |
            v
     MovementAnalyzer -> trade_snapshots (Postgres/Supabase)

This module has exactly one job: watch a trade from the moment it opens to
the moment it closes and record how price moved. It NEVER influences
trading decisions and is intentionally decoupled from SignalGenerator, the
Rolling Evidence Accumulator/PersistenceValidator, EventConfirmationEngine,
and watchlist logic — it only reads (a) whatever ExecutionEngine tells it
when a trade opens/closes, and (b) live prices from MarketDataStore, which
every other module in this codebase already reads from too. No market
microstructure metrics (spread, book imbalance, confidence, aggressive
volume, depth, volatility) are collected here by design.

Design note on write volume: the design brief calls for 100-250ms
in-memory snapshots. Persisting a database row at that cadence for every
open trade would mean up to ~7 inserts/sec per trade (and up to
max_open_positions of those concurrently) — that's real, ongoing database
load and cost for resolution nobody will look at row-by-row. So: full
150ms-resolution stats are kept in memory (cheap — just arithmetic on a
dataclass) and used to build a bounded in-memory timeline. A single
`trade_snapshots` row is created for the trade on its first throttled tick
and then updated in place on every subsequent one (`db_snapshot_interval_sec`,
default 30s) — never a new row per tick — so each trade maintains exactly
one row while open, which then becomes the final summary row on close.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

log = logging.getLogger("okx_futures.movement_tracker")

TABLE = "trade_snapshots"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _format_duration(seconds: Optional[float]) -> Optional[str]:
    """Renders a raw seconds value as '5s', '5m 2s', or '1h 3m 10s' instead
    of a bare float — much easier to scan in position_history/trade_snapshots
    than e.g. 3286.3174872398376. Sub-second precision is dropped (whole
    seconds only) since it doesn't add anything readable at this scale.
    Note: this changes trade_duration from a numeric column to a string —
    if the DB column is currently numeric/float, its type needs to change
    to text before this can be written."""
    if seconds is None:
        return None
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _signed(value: Optional[float], sign: str) -> Optional[str]:
    """Prefixes a raw (always non-negative) magnitude with an explicit
    '+' or '-' so profit-side vs loss-side columns are distinguishable at
    a glance (e.g. maximum_favorable_excursion -> '+0.00306...',
    maximum_adverse_excursion -> '-0.00980...'). Used for excursion/USD
    magnitude columns only — every time_* column (time_to_first_profit,
    time_profitable, time_to_max_profit, etc.) uses _format_duration
    instead, matching trade_duration's '1h 3m 10s' format for consistency
    across every column that stores a duration. None is passed through
    unchanged — a trade that never went into loss genuinely has no
    time_to_first_loss/maximum_adverse_excursion, and forcing a value onto
    that would misrepresent it as real."""
    if value is None:
        return None
    return f"{sign}{abs(value)}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MovementTrackerConfig:
    tick_interval_sec: float = 0.15  # 100-250ms design target
    db_snapshot_interval_sec: float = 30.0  # throttled DB write cadence while open — one row per trade, updated in place
    timeline_max_points: int = 2000  # bounded in-memory timeline per trade


# ---------------------------------------------------------------------------
# Per-trade live state
# ---------------------------------------------------------------------------


@dataclass
class TrackedTrade:
    """Everything MovementTracker knows about one open trade. All of this
    lives in memory only until the trade closes — see MovementTracker.stop_tracking().

    `directional_pct` (used throughout) means: how far price has moved *in
    the position's favor* as a fraction of entry price. Positive = the
    trade is currently ahead, negative = currently behind. For a LONG this
    is (price - entry) / entry; for a SHORT it's the negation of that.
    """

    trade_id: Optional[int]  # position_history.id — the FK target (see TradeSnapshotStore docstring)
    symbol: str
    direction: str  # "long" | "short"
    entry_price: float
    take_profit_price: float
    contract_size: float
    size_contracts: float
    opened_at: float

    current_price: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0

    # Running peak/trough of directional_pct, used to derive MFE/MAE and
    # drawdown/run-up incrementally without re-scanning the timeline.
    _peak_directional_pct: float = field(default=0.0, repr=False)
    _trough_directional_pct: float = field(default=0.0, repr=False)

    max_favorable_excursion_pct: float = 0.0  # MFE, >= 0
    max_adverse_excursion_pct: float = 0.0  # MAE, >= 0 (magnitude of the worst adverse move)
    max_drawdown_pct: float = 0.0  # worst give-back from the best point reached, >= 0
    max_runup_pct: float = 0.0  # best recovery from the worst point reached, >= 0

    max_unrealized_profit_usdt: float = 0.0
    max_unrealized_loss_usdt: float = 0.0  # magnitude, >= 0

    time_to_first_profit_sec: Optional[float] = None
    time_to_first_loss_sec: Optional[float] = None
    time_profitable_sec: float = 0.0
    time_losing_sec: float = 0.0
    seconds_until_max_profit: float = 0.0
    seconds_until_max_loss: float = 0.0

    last_tick_at: float = 0.0
    last_db_snapshot_at: float = 0.0
    last_snapshot_row_id: Optional[int] = None

    # Bounded timeline for later chart reconstruction: (elapsed_sec, directional_pct)
    timeline: Deque[Tuple[float, float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.current_price = self.entry_price
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
        self.last_tick_at = self.opened_at

    @property
    def notional_usdt(self) -> float:
        return self.size_contracts * self.contract_size * self.entry_price

    def directional_pct(self, price: float) -> float:
        raw = (price - self.entry_price) / self.entry_price if self.entry_price else 0.0
        return raw if self.direction == "long" else -raw

    def unrealized_pnl_usdt(self, price: float) -> float:
        return self.directional_pct(price) * self.notional_usdt

    def distance_to_take_profit_pct(self, price: float) -> float:
        if self.take_profit_price in (None, 0):
            return 0.0
        return (self.take_profit_price - price) / self.take_profit_price

    def apply_tick(self, price: float, now: float, timeline_max_points: int) -> None:
        """Fold one price observation into the running stats. Pure
        arithmetic on already-allocated fields — no new objects created
        per tick except the (small, bounded) timeline entry."""
        dt = max(0.0, now - self.last_tick_at)
        elapsed = now - self.opened_at

        self.current_price = price
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)

        pct = self.directional_pct(price)
        pnl_usdt = pct * self.notional_usdt

        # MFE / MAE
        if pct > self.max_favorable_excursion_pct:
            self.max_favorable_excursion_pct = pct
            self.seconds_until_max_profit = elapsed
        if -pct > self.max_adverse_excursion_pct:
            self.max_adverse_excursion_pct = -pct
            self.seconds_until_max_loss = elapsed

        if pnl_usdt > self.max_unrealized_profit_usdt:
            self.max_unrealized_profit_usdt = pnl_usdt
        if -pnl_usdt > self.max_unrealized_loss_usdt:
            self.max_unrealized_loss_usdt = -pnl_usdt

        # Drawdown (give-back from the best point so far) / run-up
        # (recovery from the worst point so far) — both running peaks.
        if pct > self._peak_directional_pct:
            self._peak_directional_pct = pct
        drawdown_now = self._peak_directional_pct - pct
        if drawdown_now > self.max_drawdown_pct:
            self.max_drawdown_pct = drawdown_now

        if pct < self._trough_directional_pct:
            self._trough_directional_pct = pct
        runup_now = pct - self._trough_directional_pct
        if runup_now > self.max_runup_pct:
            self.max_runup_pct = runup_now

        # Timing
        if pct > 0:
            self.time_profitable_sec += dt
            if self.time_to_first_profit_sec is None:
                self.time_to_first_profit_sec = elapsed
        elif pct < 0:
            self.time_losing_sec += dt
            if self.time_to_first_loss_sec is None:
                self.time_to_first_loss_sec = elapsed

        self.timeline.append((elapsed, pct))
        while len(self.timeline) > timeline_max_points:
            self.timeline.popleft()

        self.last_tick_at = now


# ---------------------------------------------------------------------------
# Movement Analyzer — final scoring/classification, computed once on close
# ---------------------------------------------------------------------------


class MovementAnalyzer:
    """Pure functions over a closed TrackedTrade — no state, easy to unit
    test and easy to retune independently of the tracking/persistence
    machinery above."""

    @staticmethod
    def movement_score(trade: TrackedTrade, final_pct: float, duration_sec: float) -> float:
        """0-100. Rewards a trade that moved cleanly toward its favorable
        side with little drawdown and little time spent underwater — this
        is a first-pass heuristic split into named, independently-tunable
        components, not a fitted model."""
        # 1) Efficiency (30 pts): how much of the best move reached was
        # actually retained at the end.
        if trade.max_favorable_excursion_pct > 0:
            efficiency = max(0.0, min(1.0, final_pct / trade.max_favorable_excursion_pct))
        else:
            efficiency = 1.0 if final_pct >= 0 else 0.0
        efficiency_pts = 30.0 * efficiency

        # 2) Drawdown penalty (25 pts): scored against a 1% reference
        # drawdown — 0% drawdown = full credit, >=1% = zero.
        drawdown_pts = 25.0 * max(0.0, 1.0 - trade.max_drawdown_pct / 0.01)

        # 3) Time-underwater penalty (20 pts): reward staying on the
        # favorable side for more of the trade's lifetime.
        if duration_sec > 0:
            underwater_frac = trade.time_losing_sec / duration_sec
        else:
            underwater_frac = 0.0
        underwater_pts = 20.0 * max(0.0, 1.0 - underwater_frac)

        # 4) Speed-to-profit bonus (15 pts): full credit for reaching
        # profit within 60s, decaying to zero, none if it never happened.
        if trade.time_to_first_profit_sec is not None:
            speed_pts = 15.0 * max(0.0, 1.0 - trade.time_to_first_profit_sec / 60.0)
        else:
            speed_pts = 0.0

        # 5) Smoothness (10 pts): fewer favor/against sign flips across the
        # recorded timeline = a cleaner, less choppy move.
        signs = [1 if pct > 0 else (-1 if pct < 0 else 0) for _, pct in trade.timeline if pct != 0]
        flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        flip_ratio = flips / len(signs) if signs else 0.0
        smoothness_pts = 10.0 * max(0.0, 1.0 - flip_ratio)

        total = efficiency_pts + drawdown_pts + underwater_pts + speed_pts + smoothness_pts
        return round(max(0.0, min(100.0, total)), 1)

    # Thresholds (USDT, against trade.max_unrealized_loss_usdt's magnitude)
    # for classify_entry_quality below.
    EXCELLENT_ENTRY_MAX_LOSS_USDT = 0.09
    GOOD_ENTRY_MAX_LOSS_USDT = 0.2
    AVERAGE_ENTRY_MAX_LOSS_USDT = 0.4

    @staticmethod
    def classify_entry_quality(
        trade: TrackedTrade,
        net_pnl: Optional[float],
        close_reason: str,
        duration_sec: float,
    ) -> str:
        """Tiered rule based on how much unrealized loss the trade weathered
        at its worst point (trade.max_unrealized_loss_usdt, a >=0 magnitude
        in USDT — this is the same number shown as "Max Unreal. Loss" on
        the dashboard), refined by outcome once that loss is large:

          <= EXCELLENT_ENTRY_MAX_LOSS_USDT (0.09) -> "Excellent Entry"
          <= GOOD_ENTRY_MAX_LOSS_USDT (0.2)        -> "Good Entry"
          <= AVERAGE_ENTRY_MAX_LOSS_USDT (0.4)     -> "Average Entry"
          above that, and it still closed profitably -> "Lucky Win"
          above that, and it closed at a loss         -> "Bad Entry"

        The outcome check only kicks in above the Average Entry threshold:
        a trade that dug a deep hole and recovered to a real profit earned
        that via luck, not entry quality — but one that dug the same hole
        and then actually lost (e.g. stop_loss) is a bad entry, not a win
        of any kind.
        """
        max_loss = trade.max_unrealized_loss_usdt
        if max_loss <= MovementAnalyzer.EXCELLENT_ENTRY_MAX_LOSS_USDT:
            return "Excellent Entry"
        if max_loss <= MovementAnalyzer.GOOD_ENTRY_MAX_LOSS_USDT:
            return "Good Entry"
        if max_loss <= MovementAnalyzer.AVERAGE_ENTRY_MAX_LOSS_USDT:
            return "Average Entry"
        if net_pnl is not None and net_pnl > 0:
            return "Lucky Win"
        return "Bad Entry"


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------


class TradeSnapshotStore:
    """Repository for the single `trade_snapshots` table (see
    trade_snapshots.sql). `trade_id` is a foreign key into
    `position_history.id` (the table position_store.py already writes) —
    that's the closest thing this codebase has to a canonical "trades"
    table; there's no separate `trades` table to reference.

    Mirrors PositionHistoryStore's pattern: the Supabase client is
    synchronous, so every call is pushed through asyncio.to_thread.
    """

    def __init__(self, supabase_client) -> None:
        self._sb = supabase_client

    async def get_row_id_for_trade(self, trade_id: Optional[int]) -> Optional[int]:
        """Looks up the single existing trade_snapshots row for this
        trade_id, if one already exists — e.g. a snapshot written by a
        previous bot process before a restart, for a trade that's now
        being re-registered via MovementTracker.start_tracking (see
        execution_engine.py's _resume_open_row / _backfill_closed_row).
        Without this, start_tracking always begins from
        last_snapshot_row_id=None, and the later insert-or-update in
        update_snapshot/write_final_summary would insert a brand new row
        instead of continuing to update the trade's one true row —
        exactly the "one row per trade" invariant this table is supposed
        to hold (see update_snapshot's docstring)."""
        if trade_id is None:
            return None
        try:
            result = await asyncio.to_thread(
                lambda: self._sb.table(TABLE).select("id").eq("trade_id", trade_id).limit(1).execute()
            )
        except Exception:
            log.exception(f"[movement-tracker] failed to look up existing snapshot row for trade_id={trade_id}")
            return None
        data = getattr(result, "data", None) or []
        return data[0].get("id") if data else None

    async def insert_snapshot(self, trade: TrackedTrade) -> Optional[int]:
        row = {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "timestamp": _iso(trade.last_tick_at),
            "current_price": trade.current_price,
            "unrealized_pnl": trade.unrealized_pnl_usdt(trade.current_price),
            "price_change_from_entry": trade.directional_pct(trade.current_price),
            "distance_to_take_profit": trade.distance_to_take_profit_pct(trade.current_price),
            "trade_duration": _format_duration(trade.last_tick_at - trade.opened_at),
            "is_final_snapshot": False,
        }
        try:
            result = await asyncio.to_thread(lambda: self._sb.table(TABLE).insert(row).execute())
        except Exception:
            log.exception(f"[movement-tracker] failed to insert snapshot for {trade.symbol}")
            return None
        data = getattr(result, "data", None) or []
        return data[0].get("id") if data else None

    async def update_snapshot(self, row_id: int, trade: TrackedTrade) -> None:
        """Updates the trade's single existing row in place — this is what
        keeps `trade_snapshots` at exactly one row per trade instead of
        one row per tick. Only called once `insert_snapshot` has already
        created that row (see MovementTracker.run_forever)."""
        row = {
            "current_price": trade.current_price,
            "unrealized_pnl": trade.unrealized_pnl_usdt(trade.current_price),
            "price_change_from_entry": trade.directional_pct(trade.current_price),
            "distance_to_take_profit": trade.distance_to_take_profit_pct(trade.current_price),
            "trade_duration": _format_duration(trade.last_tick_at - trade.opened_at),
            "timestamp": _iso(trade.last_tick_at),
        }
        try:
            await asyncio.to_thread(lambda: self._sb.table(TABLE).update(row).eq("id", row_id).execute())
        except Exception:
            log.exception(f"[movement-tracker] failed to update snapshot for {trade.symbol} (row {row_id})")

    async def write_final_summary(
        self,
        row_id: Optional[int],
        trade: TrackedTrade,
        *,
        exit_price: Optional[float],
        final_pct: float,
        duration_sec: float,
        movement_score: float,
        entry_quality: str,
        realized_pnl: Optional[float],
        net_pnl: Optional[float],
        close_reason: str,
        closed_at: float,
    ) -> None:
        summary = {
            "symbol": trade.symbol,
            "direction": trade.direction,
            "trade_id": trade.trade_id,
            "timestamp": _iso(closed_at),
            "current_price": exit_price,
            "unrealized_pnl": 0.0,
            "price_change_from_entry": final_pct,
            "distance_to_take_profit": 0.0,
            "trade_duration": _format_duration(duration_sec),
            "entry_price": trade.entry_price,
            "exit_price": exit_price,
            "highest_price": trade.highest_price,
            "lowest_price": trade.lowest_price,
            "maximum_favorable_excursion": _signed(trade.max_favorable_excursion_pct, "+"),
            "maximum_adverse_excursion": _signed(trade.max_adverse_excursion_pct, "-"),
            "maximum_unrealized_profit": _signed(trade.max_unrealized_profit_usdt, "+"),
            "maximum_unrealized_loss": _signed(trade.max_unrealized_loss_usdt, "-"),
            "maximum_drawdown": _signed(trade.max_drawdown_pct, "-"),
            "maximum_runup": _signed(trade.max_runup_pct, "+"),
            "time_to_first_profit": _format_duration(trade.time_to_first_profit_sec),
            "time_to_first_loss": _format_duration(trade.time_to_first_loss_sec),
            "time_profitable": _format_duration(trade.time_profitable_sec),
            "time_losing": _format_duration(trade.time_losing_sec),
            "time_to_max_profit": _format_duration(trade.seconds_until_max_profit),
            "time_to_max_loss": _format_duration(trade.seconds_until_max_loss),
            "movement_score": movement_score,
            "entry_quality": entry_quality,
            "realized_profit": realized_pnl,
            "net_profit": net_pnl,
            "close_reason": close_reason,
            "is_final_snapshot": True,
        }
        try:
            if row_id is not None:
                await asyncio.to_thread(lambda: self._sb.table(TABLE).update(summary).eq("id", row_id).execute())
            else:
                await asyncio.to_thread(lambda: self._sb.table(TABLE).insert(summary).execute())
        except Exception:
            log.exception(f"[movement-tracker] failed to write final summary for {trade.symbol}")


# ---------------------------------------------------------------------------
# MovementTracker — the orchestrating class
# ---------------------------------------------------------------------------


class MovementTracker:
    """Observer only. `start_tracking`/`stop_tracking` are called by
    ExecutionEngine right after open / right after close; `run_forever` is
    a standalone background loop (started as its own asyncio task, same
    pattern as run_position_monitor in tracker.py) that ticks every
    `tick_interval_sec` and updates every currently-tracked trade from
    live prices. Nothing here reads signals, confidence, confirmations, or
    the watchlist — only MarketDataStore (for price) and whatever
    ExecutionEngine hands it directly.
    """

    def __init__(
        self,
        market_data,
        snapshot_store: Optional[TradeSnapshotStore] = None,
        config: Optional[MovementTrackerConfig] = None,
    ) -> None:
        self._market_data = market_data
        self._store = snapshot_store
        self.config = config or MovementTrackerConfig()
        self._trades: Dict[str, TrackedTrade] = {}
        self._lock = asyncio.Lock()

    async def start_tracking(self, position: Any) -> None:
        """`position` is an execution_engine.OpenPosition (duck-typed here
        to avoid a circular import — only symbol/direction/entry_price/
        take_profit_price/contract_size/size_contracts/opened_at/db_id are
        read)."""
        trade = TrackedTrade(
            trade_id=getattr(position, "db_id", None),
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            take_profit_price=position.take_profit_price,
            contract_size=position.contract_size,
            size_contracts=position.size_contracts,
            opened_at=position.opened_at,
        )

        # Re-registering a trade this process didn't itself open (a
        # still-open position resumed after a restart, or an
        # already-closed one being backfilled — see
        # execution_engine.py's reconcile_from_store) would otherwise
        # always start from last_snapshot_row_id=None and insert a
        # duplicate row on the first write. Reuse the existing row for
        # this trade_id if one's already there.
        if self._store is not None and trade.trade_id is not None:
            try:
                trade.last_snapshot_row_id = await self._store.get_row_id_for_trade(trade.trade_id)
            except Exception:
                log.exception(f"[movement-tracker] failed to check for an existing snapshot row for {position.symbol}")

        async with self._lock:
            self._trades[position.symbol] = trade
        log.info(f"[movement-tracker] now tracking {position.symbol} ({position.direction}) from entry={position.entry_price}")

    async def stop_tracking(
        self,
        symbol: str,
        *,
        exit_price: Optional[float],
        realized_pnl: Optional[float],
        closing_fee: Optional[float],
        net_pnl: Optional[float],
        close_reason: str,
        closed_at: float,
    ) -> None:
        async with self._lock:
            trade = self._trades.pop(symbol, None)
        if trade is None:
            return  # nothing was being tracked for this symbol (e.g. store never started, or already stopped)

        # One last tick right up to the close, so the final stats reflect
        # the true exit point rather than whatever the last periodic tick
        # happened to see.
        final_price = exit_price if exit_price is not None else trade.current_price
        trade.apply_tick(final_price, closed_at, self.config.timeline_max_points)

        duration_sec = closed_at - trade.opened_at
        final_pct = trade.directional_pct(final_price)
        score = MovementAnalyzer.movement_score(trade, final_pct, duration_sec)
        quality = MovementAnalyzer.classify_entry_quality(trade, net_pnl, close_reason, duration_sec)

        log.info(
            f"[movement-tracker] {symbol} closed after {duration_sec:.1f}s — "
            f"movement_score={score:.1f}/100 entry_quality={quality!r} "
            f"MFE={trade.max_favorable_excursion_pct:.2%} MAE={trade.max_adverse_excursion_pct:.2%} "
            f"max_drawdown={trade.max_drawdown_pct:.2%}"
        )

        if self._store is not None:
            await self._store.write_final_summary(
                trade.last_snapshot_row_id,
                trade,
                exit_price=final_price,
                final_pct=final_pct,
                duration_sec=duration_sec,
                movement_score=score,
                entry_quality=quality,
                realized_pnl=realized_pnl,
                net_pnl=net_pnl,
                close_reason=close_reason,
                closed_at=closed_at,
            )
        # `trade` (and its bounded timeline) goes out of scope here and is
        # garbage collected — no separate "cleanup" step needed beyond the
        # dict pop above, which already released the only reference.

    async def run_forever(self) -> None:
        """Background loop — start this once as its own asyncio task. Ticks
        every `tick_interval_sec`, updates every currently-tracked trade
        from MarketDataStore, and throttles actual DB writes to
        `db_snapshot_interval_sec` per trade.

        Does NOT feed the trailing profit-floor ratchet (tp_tracker.py).
        By explicit design, that ratchet is fed only by unrealized_pnl
        reported directly by OKX itself — the positions-channel websocket
        push in tracker.py's run_private, plus execution_engine's REST
        poll as a fallback — never by this loop's own locally-derived
        estimate (price move % × notional is a close approximation of
        OKX's own mark-price-based upl, but it isn't the same number)."""
        while True:
            await asyncio.sleep(self.config.tick_interval_sec)
            async with self._lock:
                symbols = list(self._trades.keys())

            now = time.time()
            for symbol in symbols:
                async with self._lock:
                    trade = self._trades.get(symbol)
                if trade is None:
                    continue  # closed between the snapshot above and now

                try:
                    market = await self._market_data.get(symbol)
                except Exception:
                    log.exception(f"[movement-tracker] failed to read price for {symbol}")
                    continue
                if not market:
                    continue

                price = market.get("mark_price") or market.get("last_price")
                if not price:
                    continue

                trade.apply_tick(float(price), now, self.config.timeline_max_points)

                if self._store is not None and now - trade.last_db_snapshot_at >= self.config.db_snapshot_interval_sec:
                    trade.last_db_snapshot_at = now
                    if trade.last_snapshot_row_id is None:
                        row_id = await self._store.insert_snapshot(trade)
                        if row_id is not None:
                            trade.last_snapshot_row_id = row_id
                    else:
                        await self._store.update_snapshot(trade.last_snapshot_row_id, trade)
