"""
Execution engine for OKX Demo Trading (USDT-margined perpetual swaps).

This module replaces the old paper-trading simulator with a real (demo)
execution path: it opens actual Demo Trading positions through the OKX
API, waits for the fill, computes a take-profit price from the *actual*
filled price and *actual* opening fee (never a hardcoded fee assumption),
and places that take-profit on the exchange as a standalone TP algo order.
It also tracks how many positions are open so the bot never exceeds the
configured concurrency cap.

Position mode (net vs. hedge) is configured on the `OKXFuturesClient`
instance passed in here — see okx_futures_client.py's module docstring.
It must match whatever the OKX account is actually set to.

`ExecutionEngineBase` exists so a live-trading engine can be added later by
subclassing and swapping only the parts that differ (e.g. credential
source, risk limits) while reusing the sizing/TP math and position
bookkeeping here.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from types import SimpleNamespace
from typing import Dict, List, Optional

from okx_futures_client import OKXAPIError, OKXFuturesClient
from market_data import Signal
from liquidation_guard import check_liquidation_distance, select_leverage_with_safe_liquidation, estimate_liquidation_price
from pretrade_validation import validate_liquidation_history
from tp_tracker import TPTracker, TPTrackerConfig

log = logging.getLogger("okx_futures.execution")

PERMANENTLY_UNTRADEABLE_OKX_CODES = {
    "51155",
}

@dataclass
class ExecutionConfig:
    margin_per_trade_usdt: float = 1.0
    requested_leverage: int = 100
    min_leverage_required: int = 50
    max_open_positions: int = 5
    max_total_trades: int = 10
    target_net_profit_usdt: float = 0.10
    open_type: str = "isolated"
    fill_poll_interval_sec: float = 0.5
    fill_timeout_sec: float = 15.0

    alert_move_step_pct: float = 0.5

    permanently_denied_symbols: frozenset = frozenset({"LINK-USDT-SWAP"})

    instant_liquidation_window_sec: float = 10.0
    instant_liquidation_price_move_pct: float = 0.3

    enable_liquidation_guard: bool = True

    min_liquidation_distance_pct: float = 0.012

    fallback_leverage: int = 10

    high_leverage_symbols: frozenset = frozenset({"ETH-USDT-SWAP"})

    target_stop_loss_usdt: float = 0.9

    # --- Trailing profit floor (see tp_tracker.py) ---
    # Once a position's peak unrealized profit reaches
    # trailing_tp_activation_usdt, the stop-loss is walked up to lock in
    # (peak - trailing_tp_lag_usdt) instead of sitting at the original
    # loss-side stop. Capped at target_net_profit_usdt, which is where
    # the original take-profit order already fires on its own.
    #
    # Set enable_trailing_tp=False to turn this whole feature off without
    # touching tp_tracker.py itself: maybe_ratchet_stop_loss_for and
    # _maybe_ratchet_stop_loss both become no-ops, so every position just
    # rides its original fixed stop-loss/take-profit from open to close,
    # exactly as if tp_tracker.py didn't exist. See tracker.py's
    # TP_TRACKER_ENABLED for the single switch that controls this.
    enable_trailing_tp: bool = True
    trailing_tp_activation_usdt: float = 0.20
    trailing_tp_lag_usdt: float = 0.10

    # After each ratchet, read back what OKX actually has resting
    # (orders-algo-pending) and confirm it locks in the intended profit
    # within this tolerance -- resubmitting and re-checking, up to
    # ratchet_verify_max_attempts times, if it doesn't. See
    # _ratchet_stop_loss's docstring for why this should normally
    # converge on the first attempt (OKX doesn't silently alter a
    # submitted trigger price) and what this verification actually does
    # and doesn't protect against.
    ratchet_verify_tolerance_usdt: float = 0.005
    ratchet_verify_max_attempts: int = 3

    enable_liquidation_history_check: bool = True
    liq_validation_lookback_hours: float = 5.0
    max_liq_hits_allowed: int = 1

    log_throttle_sec: float = 30.0

    candle_bar: str = "5m"

@dataclass
class OpenPosition:
    symbol: str
    direction: str
    order_id: str
    entry_price: float
    size_contracts: float
    contract_size: float
    leverage: int
    margin_usdt: float
    opening_fee: float
    take_profit_price: float
    stop_loss_price: float
    tp_order_id: Optional[str]
    price_precision: str
    stop_loss_is_profit_lock: bool = False  # True once tp_tracker.py's ratchet has moved stop_loss_price above entry (long) / below entry (short) -- see _ratchet_stop_loss and _infer_close_reason_from_exit_price
    opened_at: float = field(default_factory=time.time)
    last_alert_level: int = 0
    db_id: Optional[int] = None
    liq_price: Optional[float] = None

def _candle_limit_for_hours(hours: float, bar: str) -> int:
    """Converts a lookback window in hours into a candle count for OKX's
    /market/candles `bar` interval. Only "Nm"/"Nh" bar strings are
    parsed — the only one currently used here is "5m"."""
    bar = bar.strip().lower()
    if bar.endswith("h"):
        minutes = float(bar[:-1]) * 60
    elif bar.endswith("m"):
        minutes = float(bar[:-1])
    else:
        minutes = 5.0
    return max(1, int(round(hours * 60 / minutes)))

def _round_to_step(value: float, step_str: str, rounding=ROUND_DOWN) -> float:
    """Round `value` down (or up) to the nearest multiple of `step_str`
    (e.g. price_precision '0.1' or vol_precision '1')."""
    step = Decimal(str(step_str))
    if step <= 0:
        return value
    quant = (Decimal(str(value)) / step).to_integral_value(rounding=rounding) * step
    return float(quant)

def _decimals_from_step(step_str: str) -> int:
    """'0.0000001' -> 7, '0.01' -> 2, '1' -> 0."""
    exponent = Decimal(str(step_str)).as_tuple().exponent
    return max(-exponent, 0) if isinstance(exponent, int) else 0

def _format_price(value: float, step_str: str) -> str:
    """Format a price as a plain fixed-point decimal string for the OKX
    API — never scientific notation — with the instrument's own price
    precision.

    str(float) breaks silently for very small numbers: str(2.931e-06) ==
    '2.931e-06'. OKX's API rejects that outright (HTTP 400, code=51000,
    "Parameter tpTriggerPx error") since it expects plain decimal, not
    scientific notation. This is exactly what happened placing a PEPE
    take-profit — PEPE's price is small enough (~0.0000029) that
    Python's default float-to-str flips to exponential form. Python's
    'f' format spec always produces plain decimal regardless of
    magnitude, so formatting with an explicit decimal count avoids the
    bug entirely."""
    decimals = _decimals_from_step(step_str)
    return f"{value:.{decimals}f}"

class ExecutionEngineBase(ABC):
    """Common interface so a live-trading engine can be swapped in later
    with minimal changes to tracker.py."""

    @abstractmethod
    async def open_count(self) -> int:
        ...

    @abstractmethod
    async def has_open_position(self, symbol: str) -> bool:
        ...

    @abstractmethod
    async def try_open_trade(self, signal: Signal) -> bool:
        ...

    @abstractmethod
    async def monitor_positions(self) -> None:
        ...

class DemoFuturesExecutionEngine(ExecutionEngineBase):
    """Executes signals as real orders against OKX Demo Trading."""

    _MIN_AGE_BEFORE_CLOSE_CHECK_SEC = 5.0
    _CLOSE_CONFIRM_DELAY_SEC = 2.0

    def __init__(
        self,
        client: OKXFuturesClient,
        config: Optional[ExecutionConfig] = None,
        position_store=None,
        movement_tracker=None,
        tp_tracker: Optional[TPTracker] = None,
    ) -> None:
        self._client = client
        self.config = config or ExecutionConfig()
        self._position_store = position_store
        self._movement_tracker = movement_tracker
        self._open_positions: Dict[str, OpenPosition] = {}
        self._total_opened = 0
        self._lock = asyncio.Lock()

        # One lock per symbol with an open position, guarding the entire
        # peek->cancel->place->commit ratchet sequence in
        # _ratchet_stop_loss. Without this, the positions-websocket push
        # (run_private, fires the instant OKX recalculates -- can be
        # several times a second during a fast move) and the 5-second
        # monitor_positions() poll fallback could both see
        # tp_tracker.peek() return ratchet=True for the same symbol at
        # nearly the same moment and both start ratcheting concurrently.
        # The second call would read pos.tp_order_id before the first had
        # finished replacing it, then try to cancel an order that was
        # already gone -- exactly the "already filled, canceled or does
        # not exist" (sCode=51400) errors this was built to eliminate,
        # and in the worst case two interleaved cancel/place calls could
        # leave pos.tp_order_id and the exchange's actual resting order
        # out of sync entirely, which is what produced the follow-on
        # sCode=51280 rejection and fail-safe flatten.
        self._ratchet_locks: Dict[str, asyncio.Lock] = {}

        # Accepts an externally-constructed tracker so tracker.py's main()
        # can share ONE instance across every peak-observing source.
        # Deliberately fed ONLY by OKX's own server-computed
        # unrealized_pnl (upl) -- run_private's positions-channel push
        # (primary, arrives the instant OKX recalculates) and this
        # engine's own 5-second REST poll below (fallback). NOT fed by
        # MovementTracker's locally price-derived estimate -- see
        # tracker.py's main() for why that source was deliberately left
        # out of this tracker's wiring. Falls back to building its own if
        # none is given, so this class still works standalone.
        self._tp_tracker = tp_tracker or TPTracker(
            TPTrackerConfig(
                activation_profit_usdt=self.config.trailing_tp_activation_usdt,
                trail_lag_usdt=self.config.trailing_tp_lag_usdt,
                final_take_profit_usdt=self.config.target_net_profit_usdt,
            )
        )

        self._blacklisted_symbols: Dict[str, OKXAPIError] = {}

        # Repeated per-tick discard/pass logs (liquidation guard, liquidation
        # history, min order size, etc.) are throttled to at most once per
        # `config.log_throttle_sec` per (symbol, message-kind) key — the same
        # symbol can otherwise log the same line dozens of times a minute
        # while a candidate keeps failing the same check tick after tick.
        self._last_log_ts: Dict[str, float] = {}

    def _get_ratchet_lock(self, symbol: str) -> asyncio.Lock:
        lock = self._ratchet_locks.get(symbol)
        if lock is None:
            lock = asyncio.Lock()
            self._ratchet_locks[symbol] = lock
        return lock

    async def open_count(self) -> int:
        async with self._lock:
            return len(self._open_positions)

    def _log_throttled(self, key: str, level: int, msg: str) -> None:
        """Logs `msg` at `level` only if at least `config.log_throttle_sec`
        has passed since the last log under this same `key` — collapses
        the same recurring per-tick line (e.g. one symbol failing the same
        check over and over) down to one line per throttle window instead
        of one per tick. Not locked: worst case under concurrent calls for
        the same key is an extra line or two slipping through, which is
        fine for a rate-limiting log helper."""
        now = time.time()
        last = self._last_log_ts.get(key, 0.0)
        if now - last >= self.config.log_throttle_sec:
            self._last_log_ts[key] = now
            log.log(level, msg)

    async def is_blacklisted(self, symbol: str) -> bool:
        """True once the exchange has told us `symbol` can never be
        traded on this account (see PERMANENTLY_UNTRADEABLE_OKX_CODES) —
        callers should stop attempting it for the rest of this run."""
        async with self._lock:
            return symbol in self._blacklisted_symbols

    async def blacklisted_symbols(self) -> List[str]:
        async with self._lock:
            return sorted(self._blacklisted_symbols.keys())

    async def _blacklist(self, symbol: str, exc: OKXAPIError) -> None:
        async with self._lock:
            is_new = symbol not in self._blacklisted_symbols
            self._blacklisted_symbols[symbol] = exc
        if is_new:
            log.warning(
                f"[execution] {symbol} blacklisted for the rest of this run — "
                f"exchange rejected it as permanently untradeable, not a bot error: {exc}"
            )

    @staticmethod
    def _is_permanent_rejection(exc: OKXAPIError) -> bool:
        return exc.code in PERMANENTLY_UNTRADEABLE_OKX_CODES

    async def total_opened(self) -> int:
        async with self._lock:
            return self._total_opened

    async def trades_exhausted(self) -> bool:
        """True once the lifetime trade cap has been reached AND every
        position opened under it has since closed — the signal tracker.py
        uses to shut the bot down for manual review."""
        async with self._lock:
            return self._total_opened >= self.config.max_total_trades and len(self._open_positions) == 0

    async def has_open_position(self, symbol: str) -> bool:
        async with self._lock:
            return symbol in self._open_positions

    async def try_open_trade(self, signal: Signal) -> bool:
        cfg = self.config
        symbol = signal.symbol

        async with self._lock:
            if symbol in self._blacklisted_symbols:
                return False
            if self._total_opened >= cfg.max_total_trades:
                return False
            if symbol in self._open_positions:
                return False
            if len(self._open_positions) >= cfg.max_open_positions:
                return False
            self._open_positions[symbol] = None

        try:
            opened = await self._open_position(signal)
        except Exception:
            log.exception(f"[execution] failed to open {symbol} — releasing reserved slot")
            opened = None

        async with self._lock:
            if opened is None:
                self._open_positions.pop(symbol, None)
                return False
            self._open_positions[symbol] = opened
            self._total_opened += 1

        return True

    async def _open_position(self, signal: Signal) -> Optional[OpenPosition]:
        cfg = self.config
        symbol = signal.symbol
        direction = signal.direction

        # Authoritative guard, backed by the exchange itself rather than
        # only in-memory bookkeeping: never open a second position on a
        # pair that already has one live on OKX, even if our own
        # _open_positions dict thinks the pair is free (e.g. after a fill
        # check timed out and we lost track of an order that actually
        # went on to fill — see _wait_for_fill's caller below). Waits for
        # the existing exposure to close before this pair is touched
        # again, exactly as it should.
        try:
            existing_positions = await self._client.get_position(symbol)
        except OKXAPIError as exc:
            self._log_throttled(
                f"{symbol}:position_check_fail", logging.WARNING,
                f"[execution] {symbol} {direction.upper()} — could not verify existing exchange "
                f"position ({exc}); discarding rather than risking a duplicate"
            )
            return None
        if any(float(p.get("current_amount") or 0) > 0 for p in existing_positions):
            self._log_throttled(
                f"{symbol}:already_open_on_exchange", logging.INFO,
                f"[execution] {symbol} {direction.upper()} — a position is already open on this "
                f"pair on the exchange; waiting for it to close before opening another"
            )
            return None

        try:
            contract = await self._client.get_contract_details(symbol)
        except OKXAPIError as exc:
            log.error(f"[execution] could not fetch contract details for {symbol}: {exc}")
            return None

        max_leverage_symbol = int(float(contract.get("max_leverage", 0)))
        if max_leverage_symbol < cfg.min_leverage_required:
            self._log_throttled(
                f"{symbol}:max_lev_too_low", logging.INFO,
                f"[execution] {symbol} max leverage {max_leverage_symbol}x < required "
                f"{cfg.min_leverage_required}x — signal discarded"
            )
            return None

        contract_size = float(contract.get("contract_size", 0))
        price_precision = contract.get("price_precision", "0.01")
        vol_precision = contract.get("vol_precision", "1")
        min_volume = float(contract.get("min_volume", 1))
        if contract_size <= 0:
            log.error(f"[execution] {symbol} has invalid contract_size — skipping")
            return None

        target_leverage = cfg.requested_leverage if symbol in cfg.high_leverage_symbols else cfg.fallback_leverage
        target_leverage = min(target_leverage, max_leverage_symbol)

        if target_leverage < cfg.min_leverage_required:
            self._log_throttled(
                f"{symbol}:hardcoded_lev_too_low", logging.INFO,
                f"[execution] {symbol} {direction.upper()} — hardcoded leverage {target_leverage:.0f}x "
                f"< required {cfg.min_leverage_required}x — signal discarded"
            )
            return None

        notional_at_lev = cfg.margin_per_trade_usdt * target_leverage
        try:
            mmr = await self._client.get_position_tier_mmr(symbol, cfg.open_type, notional_at_lev)
        except OKXAPIError as exc:
            self._log_throttled(
                f"{symbol}:mmr_fetch_fail", logging.WARNING,
                f"[execution] {symbol} {direction.upper()} — could not fetch maintenance margin "
                f"rate at {target_leverage:.0f}x ({exc}); discarding rather than trading blind"
            )
            return None

        if cfg.enable_liquidation_guard:
            liq_check = select_leverage_with_safe_liquidation(
                entry_price=signal.entry_price,
                direction=direction,
                candidates=[(target_leverage, mmr)],
                min_distance_pct=cfg.min_liquidation_distance_pct,
            )
            if not liq_check.approved:
                self._log_throttled(
                    f"{symbol}:liq_guard_reject", logging.INFO,
                    f"[execution] {symbol} {direction.upper()} — {liq_check.reason}"
                )
                return None

            leverage = liq_check.leverage
            estimated_liq_price = liq_check.liquidation_price
            self._log_throttled(
                f"{symbol}:liq_guard_pass", logging.INFO,
                f"[execution] {symbol} {direction.upper()} liquidation guard passed at {leverage:.0f}x — "
                f"est. liq={liq_check.liquidation_price:.8f} "
                f"({liq_check.distance_pct:.2%} from entry {signal.entry_price})"
            )
        else:
            leverage = target_leverage
            estimated_liq_price = estimate_liquidation_price(signal.entry_price, leverage, mmr, direction)

        notional_usdt = cfg.margin_per_trade_usdt * leverage

        if cfg.enable_liquidation_history_check:
            try:
                liq_candles = await self._client.get_candles(
                    symbol,
                    bar=cfg.candle_bar,
                    limit=_candle_limit_for_hours(cfg.liq_validation_lookback_hours, cfg.candle_bar),
                )
            except OKXAPIError as exc:
                self._log_throttled(
                    f"{symbol}:liq_candle_fetch_fail", logging.WARNING,
                    f"[execution] {symbol} {direction.upper()} — could not fetch candles for the "
                    f"liquidation-history check ({exc}); discarding rather than trading without this filter"
                )
                return None

            liq_history_result = validate_liquidation_history(
                liquidation_price=estimated_liq_price,
                direction=direction,
                candles=liq_candles,
                max_hits=cfg.max_liq_hits_allowed,
                lookback_hours=cfg.liq_validation_lookback_hours,
            )
            if not liq_history_result.approved:
                self._log_throttled(
                    f"{symbol}:liq_history_reject", logging.INFO,
                    f"[execution] {symbol} {direction.upper()} — {liq_history_result.reason}"
                )
                return None
            self._log_throttled(
                f"{symbol}:liq_history_pass", logging.INFO,
                f"[execution] {symbol} {direction.upper()} liquidation-history check passed — "
                f"est. liq {estimated_liq_price:.8f} touched {liq_history_result.hits}x in the last "
                f"{cfg.liq_validation_lookback_hours:.0f}h"
            )

        qty_base = notional_usdt / signal.entry_price
        size_contracts = _round_to_step(qty_base / contract_size, vol_precision, rounding=ROUND_DOWN)

        if size_contracts < min_volume:
            self._log_throttled(
                f"{symbol}:min_order_size", logging.INFO,
                f"[execution] {symbol} minimum order size ({min_volume} contracts) exceeds what "
                f"${cfg.margin_per_trade_usdt} margin at {leverage}x can buy — signal discarded"
            )
            return None

        try:
            await self._client.submit_leverage(symbol, leverage, cfg.open_type, direction=direction)
        except OKXAPIError as exc:
            if self._is_permanent_rejection(exc):
                await self._blacklist(symbol, exc)
                return None
            log.error(f"[execution] failed to set leverage for {symbol}: {exc}")
            return None

        side = 1 if direction == "long" else 4
        client_order_id = f"bot{uuid.uuid4().hex[:12]}"
        try:
            order = await self._client.submit_order(
                symbol=symbol,
                side=side,
                size=size_contracts,
                order_type="market",
                leverage=str(leverage),
                open_type=cfg.open_type,
                client_order_id=client_order_id,
            )
        except OKXAPIError as exc:
            if self._is_permanent_rejection(exc):
                await self._blacklist(symbol, exc)
                return None
            log.error(f"[execution] order submission failed for {symbol}: {exc}")
            return None

        order_id = str(order.get("order_id"))
        log.info(f"[execution] opened {direction.upper()} {symbol} order_id={order_id} size={size_contracts}")

        filled = await self._wait_for_fill(symbol, order_id)
        if filled is None:
            log.error(
                f"[execution] {symbol} order {order_id} did not fill within timeout — cancelling it "
                f"so it can't sit live on the exchange and fill later without the bot tracking it"
            )
            filled = await self._cancel_and_verify(symbol, order_id)
            if filled is None:
                return None
            log.warning(
                f"[execution] {symbol} order {order_id} had actually filled (fully or partially) by "
                f"the time the cancel was attempted — proceeding to protect the real position with TP/SL"
            )

        deal_avg_price = float(filled.get("deal_avg_price", 0))
        deal_size = float(filled.get("deal_size", 0))
        if deal_avg_price <= 0 or deal_size <= 0:
            log.error(f"[execution] {symbol} order {order_id} reported no fill — abandoning")
            return None

        opening_fee = await self._get_opening_fee(
            symbol, order_id, notional_usdt=deal_size * contract_size * deal_avg_price
        )

        take_profit_price = self._compute_take_profit_price(
            direction=direction,
            entry_price=deal_avg_price,
            size_contracts=deal_size,
            contract_size=contract_size,
            opening_fee=opening_fee,
            price_precision=price_precision,
        )

        stop_loss_price = self._compute_stop_loss_price(
            direction=direction,
            entry_price=deal_avg_price,
            size_contracts=deal_size,
            contract_size=contract_size,
            opening_fee=opening_fee,
            price_precision=price_precision,
        )

        tp_order_id = await self._place_tp_sl(
            symbol=symbol,
            direction=direction,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            size_contracts=deal_size,
            price_precision=price_precision,
        )

        if tp_order_id is None:

            log.error(
                f"[execution] {symbol} — TP/SL could not be placed after retries; "
                f"flattening the just-opened position immediately instead of "
                f"leaving it unprotected"
            )
            close_side = 3 if direction == "long" else 2
            try:
                await self._client.submit_order(
                    symbol=symbol, side=close_side, size=deal_size, order_type="market"
                )
                log.warning(f"[execution] {symbol} — fail-safe close submitted")
            except OKXAPIError as exc:
                log.error(
                    f"[execution] {symbol} — fail-safe close ALSO failed ({exc}); "
                    f"this position is genuinely unprotected and needs manual attention"
                )
            return None

        log.info(
            f"[execution] {symbol} filled entry={deal_avg_price} size={deal_size} "
            f"opening_fee={opening_fee:.6f} take_profit={take_profit_price} stop_loss={stop_loss_price}"
        )

        position = OpenPosition(
            symbol=symbol,
            direction=direction,
            order_id=order_id,
            entry_price=deal_avg_price,
            size_contracts=deal_size,
            contract_size=contract_size,
            leverage=leverage,
            margin_usdt=cfg.margin_per_trade_usdt,
            opening_fee=opening_fee,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            tp_order_id=tp_order_id,
            price_precision=price_precision,
            liq_price=estimated_liq_price,
        )

        await self._tp_tracker.start_tracking(symbol)

        if self._position_store is not None:
            position.db_id = await self._position_store.record_open(position)

        if self._movement_tracker is not None:

            await self._movement_tracker.start_tracking(position)

        return position

    async def _cancel_and_verify(self, symbol: str, order_id: str) -> Optional[dict]:
        """Called only after _wait_for_fill gives up. Cancels the
        still-outstanding order so it can't rest on the exchange and fill
        later while the bot has already moved on — that gap is exactly
        what let multiple real orders stack up on the same pair before.

        OKX rejects cancelling an order that has already filled, so a
        failed cancel is re-checked against the order's actual state
        rather than assumed to mean "still safe to ignore": if it turns
        out the order filled (fully or partially) in the moment between
        our last poll and the cancel attempt, that fill detail is
        returned so the caller can protect the real position with TP/SL
        instead of leaving it unprotected. Returns None only once it's
        confirmed there is no fill to protect."""
        try:
            await self._client.cancel_order(symbol, order_id)
        except OKXAPIError as exc:
            log.warning(f"[execution] {symbol} order {order_id} cancel request failed ({exc}); re-checking its state")

        try:
            detail = await self._client.get_order(symbol, order_id)
        except OKXAPIError as exc:
            log.error(
                f"[execution] {symbol} order {order_id} — could not confirm final state after cancel "
                f"({exc}); treating as unfilled, but verify manually on the exchange"
            )
            return None

        if str(detail.get("state")) == "4":
            return detail
        deal_size = float(detail.get("deal_size") or 0)
        if deal_size > 0:
            return detail
        return None

    async def _wait_for_fill(self, symbol: str, order_id: str) -> Optional[dict]:
        cfg = self.config
        deadline = time.time() + cfg.fill_timeout_sec
        while time.time() < deadline:
            try:
                detail = await self._client.get_order(symbol, order_id)
            except OKXAPIError as exc:
                log.warning(f"[execution] order status check failed for {symbol} {order_id}: {exc}")
                await asyncio.sleep(cfg.fill_poll_interval_sec)
                continue
            if str(detail.get("state")) == "4":
                return detail
            await asyncio.sleep(cfg.fill_poll_interval_sec)
        return None

    async def _get_opening_fee(self, symbol: str, order_id: str, notional_usdt: float) -> float:
        """Retrieves the actual fee charged for opening the position. Trading
        fees vary by pair/VIP level/promotions, so this is always looked up
        from the exchange rather than assumed.

        The fill's fee record can lag a beat behind the order itself being
        reported as filled, so this retries a few times before concluding
        there's genuinely no fee data yet. If it still comes back empty, it
        falls back to the exchange's *quoted* taker fee rate for this pair
        (also fetched live, never hardcoded) rather than silently using 0 —
        a 0 opening fee understates the true cost and makes the take-profit
        calculated from it too tight to clear fees."""
        fee = await self._fetch_fee_from_trades(symbol, order_id)
        if fee is not None and fee > 0:
            return fee

        log.warning(
            f"[execution] {symbol} order {order_id} — no fill fee reported by the exchange yet; "
            f"falling back to the quoted taker fee rate for this pair"
        )
        return await self._estimate_fee_from_rate(symbol, notional_usdt)

    async def _fetch_fee_from_trades(
        self, symbol: str, order_id: str, attempts: int = 6, delay_sec: float = 0.5
    ) -> Optional[float]:
        for attempt in range(1, attempts + 1):
            try:
                trades = await self._client.get_trades(symbol=symbol, order_id=order_id)
            except OKXAPIError as exc:
                log.warning(f"[execution] fee lookup attempt {attempt}/{attempts} failed for {symbol} {order_id}: {exc}")
                trades = []
            total_fee = sum(float(t.get("paid_fees", 0)) for t in trades)
            if total_fee > 0:
                return total_fee
            if attempt < attempts:
                await asyncio.sleep(delay_sec)
        return None

    async def _estimate_fee_from_rate(self, symbol: str, notional_usdt: float) -> float:
        try:
            rate_info = await self._client.get_trade_fee_rate(symbol)
            taker_rate = float(rate_info.get("taker_fee_rate", 0))
        except OKXAPIError as exc:
            log.warning(f"[execution] could not fetch trade fee rate for {symbol}: {exc} — assuming 0 fee")
            return 0.0
        return notional_usdt * taker_rate

    def _compute_take_profit_price(
        self,
        direction: str,
        entry_price: float,
        size_contracts: float,
        contract_size: float,
        opening_fee: float,
        price_precision: str,
        target_net_profit_usdt: Optional[float] = None,
    ) -> float:
        """Take profit is derived entirely from real, exchange-reported
        numbers: the actual filled entry price/size and the actual opening
        fee (doubled to estimate the matching closing fee), targeting a net
        realized profit of exactly `target_net_profit_usdt` after both fees
        — no added cushion. A prior version added a slippage buffer here
        (0.20% extra required price move), which pushed the TP price out
        to ~0.30-0.39 USDT of required profit against a $0.07 target,
        defeating the scalp — removed per explicit instruction.

        `target_net_profit_usdt` defaults to `config.target_net_profit_usdt`
        (the original behavior, unchanged) but can be overridden with a
        smaller value — used by _ratchet_stop_loss to price a tighter
        profit-lock floor with this exact same fee-aware math, instead of
        duplicating it."""
        cfg = self.config
        target = target_net_profit_usdt if target_net_profit_usdt is not None else cfg.target_net_profit_usdt
        estimated_total_fees = opening_fee * 2.0
        notional = size_contracts * contract_size * entry_price
        required_gross_profit = target + estimated_total_fees
        price_move_frac = required_gross_profit / notional if notional > 0 else 0.0

        if direction == "long":
            raw_tp = entry_price * (1 + price_move_frac)
            return _round_to_step(raw_tp, price_precision, rounding=ROUND_UP)
        else:
            raw_tp = entry_price * (1 - price_move_frac)
            return _round_to_step(raw_tp, price_precision, rounding=ROUND_DOWN)

    def _compute_stop_loss_price(
        self,
        direction: str,
        entry_price: float,
        size_contracts: float,
        contract_size: float,
        opening_fee: float,
        price_precision: str,
    ) -> float:
        """Mirrors _compute_take_profit_price above, but for the loss side:
        derived from the real filled price/size and real opening fee
        (doubled to estimate the matching closing fee), targeting a net
        realized LOSS of exactly `target_stop_loss_usdt`. Fees are
        subtracted from the target here rather than added — the fees
        themselves already contribute to the loss, so less adverse price
        movement is needed to reach the same net loss than the raw target
        alone would suggest. Rounded away from entry (not toward it) in
        both directions, same conservative-rounding approach as the TP
        price: a stop-loss placed slightly too close to entry by rounding
        could trigger prematurely on ordinary noise."""
        cfg = self.config
        estimated_total_fees = opening_fee * 2.0
        notional = size_contracts * contract_size * entry_price
        required_gross_loss = max(cfg.target_stop_loss_usdt - estimated_total_fees, 0.0)
        price_move_frac = required_gross_loss / notional if notional > 0 else 0.0

        if direction == "long":
            raw_sl = entry_price * (1 - price_move_frac)
            return _round_to_step(raw_sl, price_precision, rounding=ROUND_DOWN)
        else:
            raw_sl = entry_price * (1 + price_move_frac)
            return _round_to_step(raw_sl, price_precision, rounding=ROUND_UP)

    async def _place_tp_sl(
        self,
        symbol: str,
        direction: str,
        take_profit_price: float,
        stop_loss_price: float,
        size_contracts: float,
        price_precision: str,
        attempts: int = 3,
        retry_delay_sec: float = 1.0,
    ) -> Optional[str]:
        """Places TP and SL together as a single OCO (one-cancels-other)
        algo order — whichever triggers first automatically cancels the
        other, so the position can never end up with both legs still live
        after a close."""

        close_side = 3 if direction == "long" else 2

        tp_price_str = _format_price(take_profit_price, price_precision)
        sl_price_str = _format_price(stop_loss_price, price_precision)
        last_exc: Optional[OKXAPIError] = None
        for attempt in range(1, attempts + 1):
            try:
                result = await self._client.submit_tp_sl_order(
                    symbol=symbol,
                    order_type="take_profit",
                    side=close_side,
                    trigger_price=tp_price_str,
                    executive_price=tp_price_str,
                    price_type=1,
                    size=size_contracts,
                    plan_category=2,
                    category="market",
                    stop_loss_trigger_price=sl_price_str,
                )
                return str(result.get("order_id")) if result else None
            except OKXAPIError as exc:
                last_exc = exc
                log.warning(
                    f"[execution] TP/SL placement attempt {attempt}/{attempts} failed for {symbol}: {exc}"
                )
                if attempt < attempts:
                    await asyncio.sleep(retry_delay_sec)
        log.error(f"[execution] failed to place TP/SL for {symbol} after {attempts} attempts: {last_exc}")
        return None

    async def maybe_ratchet_stop_loss_for(self, symbol: str) -> None:
        """Public, event-driven counterpart to _maybe_ratchet_stop_loss —
        call this directly from wherever a fresh OKX-sourced
        unrealized_pnl was just observed (e.g. tracker.py's run_private,
        right after the positions-channel push feeds tp_tracker.observe())
        instead of waiting for the next monitor_positions() poll.

        This is what actually closes most of the profit-floor gap: peak
        detection became instant once the positions websocket was wired
        up, but without this, the ACTION of moving the exchange order
        still only happened on monitor_positions()'s 5-second cadence —
        meaning the exchange-side stop could lag several seconds behind
        the true peak. Calling this the moment a new peak is observed
        collapses that gap to effectively zero. monitor_positions()'s own
        5-second call into _maybe_ratchet_stop_loss remains as a fallback
        sweep (harmless and cheap if this already handled it — peek()
        just returns ratchet=False).

        Skips entirely, with no exchange calls at all, if
        config.enable_trailing_tp is False — see tracker.py's
        TP_TRACKER_ENABLED."""
        if not self.config.enable_trailing_tp:
            return
        async with self._lock:
            pos = self._open_positions.get(symbol)
        if pos is None:
            return

        lock = self._get_ratchet_lock(symbol)
        if lock.locked():
            # Another observation of this same symbol (the 5-second poll
            # fallback, or a second websocket push arriving before the
            # first finished) is already mid-ratchet. Skip rather than
            # queue behind it — see this engine's __init__ for why
            # queuing here was the actual bug. Whatever new peak this
            # observation represents was already folded into
            # tp_tracker's running max via observe() below, so the
            # in-flight ratchet's own next peek() (or the very next call
            # here once it's done) will pick it up -- the caller's
            # observe() call already folded this observation's peak into
            # tp_tracker's running max before this method was reached,
            # so nothing about the peak itself is lost by skipping here.
            return

        async with lock:
            decision = await self._tp_tracker.peek(symbol)
            if not decision.ratchet:
                return
            await self._ratchet_stop_loss(symbol, pos, decision.floor_usdt)

    async def _maybe_ratchet_stop_loss(self, symbol: str, pos: OpenPosition, active: dict) -> None:
        """Feeds the TP tracker (tp_tracker.py) this 5-second REST poll's
        unrealized_pnl (OKX's own `upl`, from get_position()) as a
        fallback observation, then checks whether any observation so far
        — including run_private's positions-channel websocket push,
        which is the primary source since it arrives the instant OKX
        recalculates rather than on this poll's 5-second cadence — has
        produced a new floor that's ready to be acted on. Both sources
        feeding this tracker are genuinely OKX's own server-computed
        number; MovementTracker's locally price-derived estimate is
        deliberately NOT wired into this tracker (see tracker.py's
        main()). If so, replaces the resting stop-loss leg on the
        exchange with one at that floor. The take-profit leg
        (pos.take_profit_price) is never touched here — it stays at the
        original final target the whole time; see tp_tracker.py's module
        docstring for why this is functionally a trailing STOP despite
        "TP" in the module name.

        Skips entirely, with no exchange calls at all, if
        config.enable_trailing_tp is False — see tracker.py's
        TP_TRACKER_ENABLED."""
        if not self.config.enable_trailing_tp:
            return

        try:
            unrealized_pnl = float(active.get("unrealized_pnl") or 0.0)
        except (TypeError, ValueError):
            unrealized_pnl = None

        if unrealized_pnl is not None:
            await self._tp_tracker.observe(symbol, unrealized_pnl)

        lock = self._get_ratchet_lock(symbol)
        if lock.locked():
            # See maybe_ratchet_stop_loss_for's matching check -- the
            # websocket push handler is almost certainly the one holding
            # this right now, since it fires far more often. This poll
            # is only a fallback anyway; skip this cycle rather than
            # race it.
            return

        async with lock:
            decision = await self._tp_tracker.peek(symbol)
            if not decision.ratchet:
                return
            await self._ratchet_stop_loss(symbol, pos, decision.floor_usdt)

    def _price_to_net_profit(
        self,
        direction: str,
        price: float,
        entry_price: float,
        size_contracts: float,
        contract_size: float,
        opening_fee: float,
    ) -> float:
        """Inverse of _compute_take_profit_price: given a price level,
        returns the net profit (after estimated fees) that closing at
        exactly that price would realize. Used by _ratchet_stop_loss to
        translate OKX's actually-confirmed stop price back into a profit
        figure comparable against the tracker's floor_usdt target."""
        notional = size_contracts * contract_size * entry_price
        if not entry_price:
            return 0.0
        if direction == "long":
            price_move_frac = (price - entry_price) / entry_price
        else:
            price_move_frac = (entry_price - price) / entry_price
        gross_profit = price_move_frac * notional
        estimated_total_fees = opening_fee * 2.0
        return gross_profit - estimated_total_fees

    async def _ratchet_stop_loss(self, symbol: str, pos: OpenPosition, new_floor_usdt: float) -> None:
        """Moves the resting stop-loss to lock in new_floor_usdt, then
        verifies the change against what OKX actually confirms is resting
        (via get_pending_algo_order — the live pending-order book, not
        the lagging history endpoint), resubmitting up to
        config.ratchet_verify_max_attempts times if the OKX-confirmed
        price doesn't translate back to new_floor_usdt within
        config.ratchet_verify_tolerance_usdt.

        Worth being precise about what this can and can't catch: OKX's
        algo-order API validates a submitted trigger price and either
        accepts it exactly or rejects the order outright (sCode/sMsg) —
        it does not silently substitute a different price. So this loop
        should converge on the first attempt under normal operation; its
        real value is (a) turning "we assume the price is right" into "we
        confirmed the price OKX actually has on file", which catches a
        genuine local rounding/precision bug if one ever exists, and (b)
        making that confirmation observable in the logs rather than
        assumed. It will NOT catch or correct slippage between a stop's
        trigger firing and its resulting market order actually filling —
        that happens after this whole exchange, at a moment this code
        isn't running, and no price verification here changes it."""
        cfg = self.config
        target_floor = new_floor_usdt

        for attempt in range(1, cfg.ratchet_verify_max_attempts + 1):
            new_sl_price = self._compute_take_profit_price(
                direction=pos.direction,
                entry_price=pos.entry_price,
                size_contracts=pos.size_contracts,
                contract_size=pos.contract_size,
                opening_fee=pos.opening_fee,
                price_precision=pos.price_precision,
                target_net_profit_usdt=target_floor,
            )
            if new_sl_price == pos.stop_loss_price:
                # Rounds to the same exchange price as what's already
                # resting (price-tick precision) -- no exchange action
                # needed, but still commit so peek() doesn't keep
                # re-suggesting this same no-op ratchet every cycle until
                # a larger move produces an actually-different price.
                await self._tp_tracker.commit(symbol, new_floor_usdt)
                return

            old_algo_id = pos.tp_order_id
            try:
                await self._client.cancel_algo_order(symbol, old_algo_id)
            except OKXAPIError as exc:
                # Most likely the previous order already triggered (price
                # hit the prior floor right as this ratchet was about to
                # run) or was already cancelled some other way. Either
                # way, leave it alone here rather than treat it as a hard
                # failure — the normal close-detection in
                # monitor_positions will sort out what actually happened
                # on its own next tick.
                log.warning(
                    f"[execution] {symbol} — could not cancel the existing TP/SL order while "
                    f"ratcheting the profit floor to {target_floor:.4f} USDT ({exc}); leaving "
                    f"the current order in place"
                )
                return

            new_algo_id = await self._place_tp_sl(
                symbol=symbol,
                direction=pos.direction,
                take_profit_price=pos.take_profit_price,  # unchanged -- see module docstring in tp_tracker.py
                stop_loss_price=new_sl_price,
                size_contracts=pos.size_contracts,
                price_precision=pos.price_precision,
            )

            if new_algo_id is None:
                log.error(
                    f"[execution] {symbol} — profit-floor ratchet cancelled the old TP/SL order but "
                    f"could not place the replacement after retries; flattening the position immediately "
                    f"instead of leaving it unprotected"
                )
                close_side = 3 if pos.direction == "long" else 2
                try:
                    await self._client.submit_order(
                        symbol=symbol, side=close_side, size=pos.size_contracts, order_type="market"
                    )
                    log.warning(f"[execution] {symbol} — fail-safe close submitted after a failed ratchet")
                except OKXAPIError as exc:
                    log.error(
                        f"[execution] {symbol} — fail-safe close ALSO failed after a failed ratchet "
                        f"({exc}); this position is genuinely unprotected and needs manual attention"
                    )
                return

            pos.tp_order_id = new_algo_id
            pos.stop_loss_price = new_sl_price
            pos.stop_loss_is_profit_lock = True  # from this point on, this position's SL leg sits in profit territory, not loss territory -- see _infer_close_reason_from_exit_price

            # --- Verify what OKX actually has resting, rather than
            # trusting the just-submitted price. ---
            try:
                pending = await self._client.get_pending_algo_order(symbol, new_algo_id)
            except OKXAPIError as exc:
                log.warning(
                    f"[execution] {symbol} — could not verify the ratcheted SL via orders-algo-pending "
                    f"({exc}); trusting the locally-computed price {new_sl_price} unverified"
                )
                await self._tp_tracker.commit(symbol, new_floor_usdt)
                return

            confirmed_sl_raw = pending.get("sl_trigger_price") if pending else None
            if not confirmed_sl_raw:
                # Most likely it already triggered in the instant between
                # being placed and this check running -- nothing left to
                # verify or correct; let normal close-detection handle it.
                log.info(
                    f"[execution] {symbol} — ratcheted SL order {new_algo_id} not found pending "
                    f"(likely already triggered) -- skipping verification"
                )
                await self._tp_tracker.commit(symbol, new_floor_usdt)
                return

            confirmed_sl_price = float(confirmed_sl_raw)
            actual_locked_profit = self._price_to_net_profit(
                direction=pos.direction,
                price=confirmed_sl_price,
                entry_price=pos.entry_price,
                size_contracts=pos.size_contracts,
                contract_size=pos.contract_size,
                opening_fee=pos.opening_fee,
            )
            deviation = actual_locked_profit - new_floor_usdt

            if abs(deviation) <= cfg.ratchet_verify_tolerance_usdt:
                pos.stop_loss_price = confirmed_sl_price
                await self._tp_tracker.commit(symbol, new_floor_usdt)
                log.info(
                    f"[execution] {symbol} {pos.direction.upper()} profit floor ratcheted — stop-loss moved "
                    f"to {confirmed_sl_price} (OKX-confirmed, locks ~{actual_locked_profit:.4f} USDT net "
                    f"against a {new_floor_usdt:.4f} target), take-profit unchanged at {pos.take_profit_price}"
                )
                return

            log.warning(
                f"[execution] {symbol} attempt {attempt}/{cfg.ratchet_verify_max_attempts} — OKX confirms "
                f"stop-loss at {confirmed_sl_price}, which locks {actual_locked_profit:.4f} USDT net, but "
                f"target was {new_floor_usdt:.4f} (off by {deviation:+.4f}); resubmitting a corrected price"
            )
            target_floor = new_floor_usdt - deviation  # nudge the next attempt's target in the corrective direction

        log.error(
            f"[execution] {symbol} — profit-floor ratchet did not converge to within "
            f"{cfg.ratchet_verify_tolerance_usdt:.4f} USDT after {cfg.ratchet_verify_max_attempts} attempts; "
            f"leaving the last-placed order in place rather than looping indefinitely"
        )
        await self._tp_tracker.commit(symbol, new_floor_usdt)

    async def monitor_positions(self) -> None:
        """Checks each tracked position against the exchange. A position with
        zero remaining size has been closed by its take-profit order, its
        stop-loss order, or by exchange liquidation — either way, we free
        the slot. There is no bot-side close/timeout decision here: OKX's
        own algo orders and liquidation engine are the only things that
        can end a position. The one exception is the profit-floor ratchet
        (see _maybe_ratchet_stop_loss / tp_tracker.py) — that only ever
        moves the resting stop-loss trigger up as profit builds, it never
        force-closes a position directly except as a fail-safe if a
        ratchet's replacement order can't be placed.

        While a position stays open, it's also checked for a significant
        price move (see `_check_price_alert`) so a trade running against —
        or in favor of — us gets surfaced without flooding the log.

        A position is only eligible to be declared closed once it's been
        open at least `_MIN_AGE_BEFORE_CLOSE_CHECK_SEC`, and even then only
        after a SECOND "not found" reading a moment later confirms the
        first one. Both guards exist because of a real incident: two
        LINK-USDT-SWAP trades were marked closed (reason=unknown, no exit
        price, no realized_pnl at all) only 5-8 seconds after opening,
        with no closing trade findable anywhere — in positions-history,
        the TP algo order, or a fills scan. The only explanation that fits
        is that OKX's own position endpoint hadn't yet reflected the
        brand-new position on that single read (an exchange-side
        propagation gap), and a single empty reading was trusted as an
        authoritative close. Both positions were almost certainly still
        genuinely open and simply got orphaned from our tracking — dropped
        from `_open_positions` while still live (and still exposed) on
        the real exchange."""
        async with self._lock:
            tracked = {s: p for s, p in self._open_positions.items() if p is not None}

        now = time.time()
        for symbol, pos in tracked.items():
            if now - pos.opened_at < self._MIN_AGE_BEFORE_CLOSE_CHECK_SEC:
                continue

            try:
                positions = await self._client.get_position(symbol=symbol)
            except OKXAPIError as exc:
                log.warning(f"[execution] position check failed for {symbol}: {exc}")
                continue

            active = next((p for p in positions if float(p.get("current_amount", 0)) > 0), None)

            if active is None:

                await asyncio.sleep(self._CLOSE_CONFIRM_DELAY_SEC)
                try:
                    positions_confirm = await self._client.get_position(symbol=symbol)
                except OKXAPIError as exc:
                    log.warning(f"[execution] position re-check failed for {symbol}: {exc}")
                    continue
                active_confirm = next(
                    (p for p in positions_confirm if float(p.get("current_amount", 0)) > 0), None
                )
                if active_confirm is not None:
                    log.info(
                        f"[execution] {symbol} — first close check came back empty but the "
                        f"confirming re-check found it still open; treating as a transient read"
                    )
                    await self._check_price_alert(symbol, pos, active_confirm)
                    continue

                async with self._lock:
                    closed = self._open_positions.pop(symbol, None)
                if closed is not None:
                    await self._finalize_closed_position(symbol, closed)
                continue

            await self._check_price_alert(symbol, pos, active)
            await self._maybe_ratchet_stop_loss(symbol, pos, active)

    async def reconcile_from_store(self, position_store) -> None:
        """Run once at startup, BEFORE the bot opens any new trades — see
        tracker.py's main(). For every position_history row still marked
        status="open" (meaning the previous bot process exited — crash,
        redeploy, manual stop — while that trade was live):

          - If OKX still shows the symbol open -> re-registers it into
            this engine's own tracking (open_positions, tp_tracker,
            movement_tracker) so monitor_positions(), the profit-floor
            ratchet, and price alerts all resume on it, instead of
            silently abandoning a position that still has real money and
            a resting TP/SL order on the exchange.

          - If OKX no longer shows it open -> pulls the real close
            details from OKX's own positions-history (the exact same
            lookup a normal close already uses, via _get_close_details)
            and marks the DB row closed with real numbers, instead of
            leaving it stuck at status="open" in the DB forever.

        Also reseeds self._total_opened from the store's total row count
        (open + closed, all-time), so config.max_total_trades is actually
        respected across restarts rather than resetting to 0 every time
        the process restarts.

        Known limitation for the "still open" case: tp_tracker's
        peak-profit tracking and movement_tracker's price-movement
        history (MFE/MAE/highest_price) both restart fresh from this
        moment — neither can be reconstructed for whatever happened
        before the restart, since that history wasn't persisted
        tick-by-tick. The trailing floor only ratchets against NEW peaks
        reached from here forward; a peak that happened pre-restart is
        not retroactively protected."""
        if position_store is None:
            return

        self._total_opened = await position_store.count_all()

        rows = await position_store.get_open_rows()
        if not rows:
            log.info(f"[startup] no open position_history rows to reconcile (lifetime trades so far: {self._total_opened})")
            return

        log.info(f"[startup] reconciling {len(rows)} open position_history row(s) against OKX (lifetime trades so far: {self._total_opened})...")

        # More than one row marked status="open" for the SAME symbol
        # means either (a) a genuine bug wrote position_history twice for
        # one real trade, or (b) two bot processes really did each open
        # their own real position on the same pair around the same time
        # (e.g. overlapping runs of this bot, both passing the exchange-
        # level pre-open check in _open_position because neither one's
        # order had filled yet when the other one checked). Either way,
        # OKX itself only ever shows ONE net position per symbol, so this
        # engine can only actively track one OpenPosition per symbol too
        # -- silently letting the loop below process both, one after the
        # other, would just overwrite _open_positions[symbol] with
        # whichever row came last while leaving BOTH database rows stuck
        # at status="open" forever (since only _resume_open_row /
        # _backfill_closed_row change that, and only for the row they
        # were actually called with). That silent overwrite, with no
        # trace left behind, is what made two identical-looking "open"
        # cards show up on the dashboard even though only one was truly
        # still being managed. Surface it loudly instead.
        rows_by_symbol: Dict[str, List[dict]] = {}
        for row in rows:
            rows_by_symbol.setdefault(row.get("symbol"), []).append(row)

        for symbol, symbol_rows in rows_by_symbol.items():
            if not symbol:
                for row in symbol_rows:
                    log.warning(f"[startup] position_history row {row.get('id')} has no symbol — skipping")
                continue

            if len(symbol_rows) > 1:
                symbol_rows.sort(key=lambda r: self._parse_stored_opened_at(r), reverse=True)
                newest, *duplicates = symbol_rows
                log.error(
                    f"[startup] {symbol} has {len(symbol_rows)} position_history rows all marked "
                    f"status=\"open\" at once (ids={[r.get('id') for r in symbol_rows]}) — OKX only "
                    f"ever shows one real position per symbol, so this is either a duplicate-write "
                    f"bug or two bot processes both opened a real position on this pair around the "
                    f"same time. Resuming tracking against the most recently opened row (id="
                    f"{newest.get('id')}, opened_at={newest.get('opened_at')}) only. The other "
                    f"row(s) ({[r.get('id') for r in duplicates]}) are being left untouched in the "
                    f"database rather than guessed-closed with fabricated numbers — please check "
                    f"Supabase and OKX's own position/order history directly to see whether real "
                    f"money was actually exposed twice, and reconcile those rows manually."
                )
                row = newest
            else:
                row = symbol_rows[0]

            row_id = row.get("id")
            try:
                live_positions = await self._client.get_position(symbol=symbol)
            except OKXAPIError as exc:
                log.error(f"[startup] {symbol} — could not query OKX to reconcile row {row_id}: {exc}; leaving it as-is for now")
                continue

            live = next((p for p in live_positions if float(p.get("current_amount", 0) or 0) > 0), None)

            if live is not None:
                await self._resume_open_row(symbol, row)
            else:
                await self._backfill_closed_row(symbol, row, position_store)

    @staticmethod
    def _parse_stored_opened_at(row: dict) -> float:
        raw = row.get("opened_at")
        if not raw:
            return time.time()
        try:
            # Supabase returns timestamps as ISO 8601 strings (matching
            # position_store._iso's format at insert time).
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            log.warning(f"[startup] could not parse stored opened_at={raw!r} — using current time instead")
            return time.time()

    @staticmethod
    def _infer_stop_loss_is_profit_lock(direction: str, entry_price: float, stop_loss_price: float) -> bool:
        """A genuine loss-protecting stop is always on the losing side of
        entry; the only way stop_loss_price ends up on the winning side
        is if tp_tracker's ratchet moved it there. This DB row doesn't
        store that boolean directly (see position_store.record_open's
        columns), so it's inferred from the price relationship instead —
        which is actually more reliable than a stored flag, since it
        can't have drifted out of sync with the real price that was
        actually set."""
        if direction == "long":
            return stop_loss_price > entry_price
        if direction == "short":
            return stop_loss_price < entry_price
        return False

    async def _resume_open_row(self, symbol: str, row: dict) -> None:
        try:
            contract = await self._client.get_contract_details(symbol)
            price_precision = contract.get("price_precision", "0.01")
        except OKXAPIError as exc:
            log.warning(
                f"[startup] {symbol} — could not fetch contract details to resume monitoring ({exc}); "
                f"using a conservative default price precision"
            )
            price_precision = "0.01"

        direction = row.get("direction") or ""
        entry_price = float(row.get("entry_price") or 0)
        stop_loss_price = float(row.get("stop_loss_price") or 0)
        stop_loss_is_profit_lock = self._infer_stop_loss_is_profit_lock(direction, entry_price, stop_loss_price)
        liq_price_raw = row.get("liq_price")

        pos = OpenPosition(
            symbol=symbol,
            direction=direction,
            order_id=row.get("order_id"),
            entry_price=entry_price,
            size_contracts=float(row.get("size_contracts") or 0),
            contract_size=float(row.get("contract_size") or 0),
            leverage=int(float(row.get("leverage") or 0)),
            margin_usdt=float(row.get("margin_usdt") or 0),
            opening_fee=float(row.get("opening_fee") or 0),
            take_profit_price=float(row.get("take_profit_price") or 0),
            stop_loss_price=stop_loss_price,
            tp_order_id=row.get("tp_order_id"),
            price_precision=price_precision,
            stop_loss_is_profit_lock=stop_loss_is_profit_lock,
            opened_at=self._parse_stored_opened_at(row),
            db_id=row.get("id"),
            liq_price=float(liq_price_raw) if liq_price_raw not in (None, "") else None,
        )

        async with self._lock:
            self._open_positions[symbol] = pos

        await self._tp_tracker.start_tracking(symbol)
        if self._movement_tracker is not None:
            try:
                await self._movement_tracker.start_tracking(pos)
            except Exception:
                log.exception(f"[startup] {symbol} — failed to resume movement tracking (continuing without it)")

        log.info(
            f"[startup] {symbol} {direction.upper()} still open on OKX (position_history row id={row.get('id')}) "
            f"— resumed monitoring (entry={entry_price}, take_profit={pos.take_profit_price}, "
            f"stop_loss={pos.stop_loss_price}{' [profit-lock]' if stop_loss_is_profit_lock else ''})"
        )

    async def _backfill_closed_row(self, symbol: str, row: dict, position_store) -> None:
        direction = row.get("direction") or ""
        entry_price = float(row.get("entry_price") or 0)
        take_profit_price = float(row.get("take_profit_price") or 0)
        stop_loss_price = float(row.get("stop_loss_price") or 0)
        opening_fee = float(row.get("opening_fee") or 0)
        stop_loss_is_profit_lock = self._infer_stop_loss_is_profit_lock(direction, entry_price, stop_loss_price)

        opened_at = self._parse_stored_opened_at(row)

        # Duck-typed stand-in for the OpenPosition _get_close_details (and
        # its _infer_close_reason_from_exit_price /
        # _get_close_details_from_fills fallback) actually read from —
        # only these five attributes are ever touched by that whole path.
        stand_in = SimpleNamespace(
            symbol=symbol,
            direction=direction,
            opened_at=opened_at,
            opening_fee=opening_fee,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            stop_loss_is_profit_lock=stop_loss_is_profit_lock,
        )

        exit_price, realized_pnl, closing_fee, close_reason, net_pnl = await self._get_close_details(symbol, stand_in)

        if net_pnl is None and realized_pnl is not None and closing_fee is not None:
            net_pnl = realized_pnl - opening_fee - closing_fee
            log.warning(
                f"[startup] {symbol} — OKX's own net_pnl wasn't available while backfilling row "
                f"{row.get('id')}; using a local approximation ({net_pnl:.8f}) that can't account "
                f"for any liquidation penalty"
            )

        closed_at = time.time()

        await position_store.record_close(
            row_id=row.get("id"),
            exit_price=exit_price,
            realized_pnl=realized_pnl,
            closing_fee=closing_fee,
            close_reason=close_reason,
            closed_at=closed_at,
            net_pnl=net_pnl,
        )

        log.info(
            f"[startup] {symbol} position_history row id={row.get('id')} was marked open but OKX shows "
            f"it already closed — backfilled from OKX history (exit={exit_price}, realized_pnl={realized_pnl}, "
            f"net_pnl={net_pnl}, reason={close_reason}) and marked closed"
        )

        # This row closed entirely before this bot process started, so it
        # was never registered with MovementTracker via start_tracking
        # (unlike _resume_open_row's still-open case above) — meaning
        # stop_tracking would otherwise silently no-op (nothing tracked
        # for this symbol) and trade_snapshots would never get a row for
        # it. Feed it through a start/stop pair here purely so the final
        # summary gets written; there's no live price history to replay,
        # so movement stats (MFE/MAE/timeline) reflect this single
        # backfilled data point rather than the trade's real path.
        if self._movement_tracker is not None:
            try:
                mt_stand_in = SimpleNamespace(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    take_profit_price=take_profit_price,
                    contract_size=float(row.get("contract_size") or 0),
                    size_contracts=float(row.get("size_contracts") or 0),
                    opened_at=opened_at,
                    db_id=row.get("id"),
                )
                await self._movement_tracker.start_tracking(mt_stand_in)
                await self._movement_tracker.stop_tracking(
                    symbol,
                    exit_price=exit_price,
                    realized_pnl=realized_pnl,
                    closing_fee=closing_fee,
                    net_pnl=net_pnl,
                    close_reason=close_reason,
                    closed_at=closed_at,
                )
            except Exception:
                log.exception(
                    f"[startup] {symbol} — failed to backfill trade_snapshots for row {row.get('id')} "
                    f"(position_history backfill above already succeeded)"
                )

    async def _finalize_closed_position(self, symbol: str, closed: OpenPosition) -> None:
        await self._tp_tracker.stop_tracking(symbol)
        self._ratchet_locks.pop(symbol, None)

        exit_price, realized_pnl, closing_fee, close_reason, net_pnl = await self._get_close_details(symbol, closed)
        closed_at = time.time()

        if net_pnl is None and realized_pnl is not None and closing_fee is not None:

            net_pnl = realized_pnl - closed.opening_fee - closing_fee
            log.warning(
                f"[execution] {symbol} — OKX's own net_pnl wasn't available; using a local "
                f"approximation ({net_pnl:.8f}) that can't account for any liquidation penalty"
            )

        log.info(
            f"[execution] {symbol} position closed (entry={closed.entry_price} exit={exit_price} "
            f"take_profit={closed.take_profit_price} realized_pnl={realized_pnl} net_pnl={net_pnl} "
            f"reason={close_reason}) — slot freed ({len(self._open_positions)}/"
            f"{self.config.max_open_positions} open, {self._total_opened}/{self.config.max_total_trades} "
            f"lifetime trades)"
        )

        if self._position_store is not None:
            await self._position_store.record_close(
                row_id=closed.db_id,
                exit_price=exit_price,
                realized_pnl=realized_pnl,
                closing_fee=closing_fee,
                net_pnl=net_pnl,
                close_reason=close_reason,
                closed_at=closed_at,
            )

        if self._movement_tracker is not None:
            await self._movement_tracker.stop_tracking(
                symbol,
                exit_price=exit_price,
                realized_pnl=realized_pnl,
                closing_fee=closing_fee,
                net_pnl=net_pnl,
                close_reason=close_reason,
                closed_at=closed_at,
            )

    async def _get_close_details(self, symbol: str, closed: OpenPosition):
        """Looks up the exchange's own closed-position record to get the
        real exit price, realized PnL, closing fee, and net PnL — never
        estimated when OKX provides the real figure.

        This reads /api/v5/account/positions-history (via
        get_closed_position()), which is the endpoint OKX actually
        populates with a genuine realized-PnL field for closed swap
        positions, plus a `close_type` that says exactly how it closed,
        plus its own fully-netted `realizedPnl` (returned here as
        net_pnl — see get_closed_position()'s docstring for why this is
        preferred over reconstructing it from realized_pnl/fees locally:
        that reconstruction silently drops any liquidation penalty).
        The record can lag a beat behind the position showing as closed
        in monitor_positions(), so this retries a few times first.

        The previous version of this method sourced these numbers from
        /trade/fills instead, reading a `pnl` field from each fill —
        but OKX's /trade/fills response simply has no realized-PnL field
        for swaps, so that always evaluated to 0 regardless of whether
        the trade actually won or lost. That fallback is kept below only
        for the rare case positions-history hasn't produced a row yet,
        in which case exit_price/closing_fee can still be recovered from
        fills, but realized_pnl and net_pnl will (as before) come back
        as 0/None there — logged clearly so it isn't mistaken for a real
        zero-PnL trade.

        Retry budget: a real incident showed OKX can take longer than a
        few seconds to index a liquidation into positions-history — two
        LINK-USDT-SWAP positions were liquidated within the same second
        they opened (confirmed after the fact in the exchange's own UI:
        real liq price, real -0.02 fee, real -1.13/-1.06 realized PnL —
        all of it), but this method exhausted its old, shorter retry
        budget before that record became queryable and gave up with
        close_reason="unknown" and every number null. The position really
        was closed; we just stopped looking too early. Budget widened
        accordingly below."""
        opened_at_ms = closed.opened_at * 1000.0

        history_row = None
        for attempt in range(1, 9):
            try:
                history_row = await self._client.get_closed_position(symbol, opened_at_ms)
            except OKXAPIError as exc:
                log.warning(f"[execution] could not fetch closed-position record for {symbol}: {exc}")
                history_row = None
            if history_row is not None:
                break
            if attempt < 8:
                await asyncio.sleep(1.0)

        if history_row is not None and history_row.get("exit_price") not in (None, ""):
            close_type = str(history_row.get("close_type") or "")
            exit_price = float(history_row["exit_price"])
            if close_type in ("3", "4", "5", "6"):
                close_reason = "liquidated"
            else:
                close_reason = self._infer_close_reason_from_exit_price(closed, exit_price)

            total_fee = float(history_row["total_fee"])
            closing_fee = total_fee - closed.opening_fee
            if closing_fee < 0:
                log.warning(
                    f"[execution] {symbol} — derived closing_fee came back negative "
                    f"(total_fee={total_fee:.8f} opening_fee={closed.opening_fee:.8f}); "
                    f"clamping to 0 rather than storing a negative fee"
                )
                closing_fee = 0.0

            return (
                exit_price,
                history_row["realized_pnl"],
                closing_fee,
                close_reason,
                history_row.get("net_pnl"),
            )

        log.warning(
            f"[execution] {symbol} — no positions-history record yet; falling back to a fills scan "
            f"(realized_pnl will read as 0 there — OKX's /trade/fills has no PnL field for swaps)"
        )
        return await self._get_close_details_from_fills(symbol, closed)

    async def _get_close_details_from_fills(self, symbol: str, closed: OpenPosition):
        """Fallback used only when positions-history hasn't produced a row
        for this close yet. Recovers exit_price/closing_fee from raw
        fills, same as the original implementation — but cannot recover a
        real realized_pnl or net_pnl this way (see _get_close_details);
        net_pnl comes back None so the caller falls back to a local
        approximation. close_reason is inferred from the exit price, same
        as the positions-history path (see
        _infer_close_reason_from_exit_price)."""
        opened_at_ms = closed.opened_at * 1000.0

        closing_side = "sell" if closed.direction == "long" else "buy"

        closing_trades: List[dict] = []
        for attempt in range(1, 6):
            try:
                trades = await self._client.get_trades(symbol=symbol)
            except OKXAPIError as exc:
                log.warning(f"[execution] could not fetch closing trades for {symbol}: {exc}")
                trades = []

            closing_trades = [
                t for t in trades
                if float(t.get("create_time", 0) or 0) >= opened_at_ms and t.get("side") == closing_side
            ]
            if closing_trades:
                break
            if attempt < 5:
                await asyncio.sleep(1.0)

        if not closing_trades:
            return None, None, None, "unknown", None

        total_vol = sum(float(t.get("vol", 0) or 0) for t in closing_trades)
        if total_vol <= 0:
            return None, None, None, "unknown", None

        exit_price = sum(float(t.get("price", 0) or 0) * float(t.get("vol", 0) or 0) for t in closing_trades) / total_vol
        realized_pnl = sum(float(t.get("realised_profit", 0) or 0) for t in closing_trades)
        closing_fee = sum(float(t.get("paid_fees", 0) or 0) for t in closing_trades)

        close_reason = self._infer_close_reason_from_exit_price(closed, exit_price)
        return exit_price, realized_pnl, closing_fee, close_reason, None

    def _infer_close_reason_from_exit_price(self, closed: OpenPosition, exit_price: float) -> str:
        """Distinguishes a take-profit fill from a stop-loss fill by
        comparing the real exit price against the position's own
        take_profit_price/stop_loss_price — the values that were actually
        set on the exchange at open time (see _place_tp_sl) — rather than
        asking OKX which algo order triggered.

        This replaces the previous approach (querying
        /api/v5/trade/orders-algo-history for closed.tp_order_id and
        treating "found and effective" as take_profit, "not found" as
        liquidated), which broke in two ways once TP and SL started being
        submitted together as a single OCO order:

        1. The lookup was querying with ordType="conditional" while the
           order was actually placed with ordType="oco" — OKX filters
           this endpoint by ordType server-side, so it returned
           code=51603 "Order does not exist" for every single close, even
           genuine take-profit fills (confirmed in production: real
           trades closed exactly at take_profit_price were logged as
           reason="liquidated" because the lookup could never find them).
        2. Even with the right ordType, TP and SL now share one algoId on
           an OCO order — "effective" only says the order triggered, not
           which leg did, so there was no way to ever report
           close_reason="stop_loss" at all; a real stop-loss fill would
           have been mislabeled take_profit once the ordType above was
           fixed.

        Whichever of take_profit_price/stop_loss_price the real exit
        price landed closer to is treated as what fired. This only runs
        for closes OKX itself didn't already flag as a real liquidation
        (close_type 3-6, checked by the caller first and always
        authoritative) — a coincidental exit near the SL price during an
        actual liquidation cascade is still reported as "liquidated".

        IMPORTANT: once tp_tracker.py's ratchet has moved stop_loss_price
        into profit territory (closed.stop_loss_is_profit_lock — see
        _ratchet_stop_loss), a fill closer to that price is no longer a
        real stop-loss and must not be reported as one — it's a real,
        positive-net-pnl exit, and "stop_loss" would misrepresent a
        winning trade as a loss in position_history/trade_snapshots.
        Reported as "trailing_stop" instead in that case. (If either
        table has a CHECK constraint restricting close_reason to a fixed
        set of values, "trailing_stop" needs to be added to it before
        this will write successfully — same caveat position_store.py
        already notes for net_pnl.)"""
        if exit_price is None or exit_price <= 0:
            return "unknown"
        tp = closed.take_profit_price
        sl = closed.stop_loss_price
        if tp and sl:
            if abs(exit_price - tp) <= abs(exit_price - sl):
                return "take_profit"
            return "trailing_stop" if closed.stop_loss_is_profit_lock else "stop_loss"
        if tp:
            return "take_profit"
        if sl:
            return "trailing_stop" if closed.stop_loss_is_profit_lock else "stop_loss"
        return "unknown"

    async def _resolve_tp_execution_order_id(self, symbol: str, closed: OpenPosition) -> Optional[str]:
        """`closed.tp_order_id` is the algoId of the TP/SL OCO algo order
        placed at open time. Checks whether it has triggered ("effective")
        and, if so, returns the ordId of the order it spawned. No longer
        used to determine close_reason (see
        _infer_close_reason_from_exit_price and get_algo_order_status's
        docstring for why) — kept only in case something needs to know
        whether this specific algo order is still live/pending."""
        if not closed.tp_order_id:
            return None
        try:
            status = await self._client.get_algo_order_status(symbol, closed.tp_order_id)
        except OKXAPIError as exc:
            log.warning(f"[execution] could not fetch TP/SL algo order status for {symbol}: {exc}")
            return None
        if not status:
            return None
        if status.get("state") == "effective" and status.get("ord_id"):
            return str(status.get("ord_id"))
        return None

    async def _check_price_alert(self, symbol: str, pos: OpenPosition, active: dict) -> None:
        """Logs a movement update once the position has moved another
        `alert_move_step_pct` further from entry than the last alert,
        in either direction. Purely observational — it never touches the
        take-profit order or closes anything."""
        cfg = self.config
        step = cfg.alert_move_step_pct
        if step <= 0:
            return

        try:
            mark_price = float(active.get("mark_price", 0))
            unrealized_pnl = float(active.get("unrealized_pnl", 0))
            liquidation_price = active.get("liquidation_price")
        except (TypeError, ValueError):
            return
        if mark_price <= 0 or pos.entry_price <= 0:
            return

        if pos.direction == "long":
            move_pct = (mark_price - pos.entry_price) / pos.entry_price * 100.0
        else:
            move_pct = (pos.entry_price - mark_price) / pos.entry_price * 100.0

        level = int(move_pct / step)
        if level == pos.last_alert_level:
            return

        async with self._lock:
            current = self._open_positions.get(symbol)
            if current is None:
                return
            current.last_alert_level = level

        outlook = "favorable" if move_pct >= 0 else "adverse"
        log.info(
            f"[monitor] {symbol} {pos.direction.upper()} move {move_pct:+.2f}% ({outlook}) — "
            f"entry={pos.entry_price} mark={mark_price} unrealized_pnl={unrealized_pnl:+.4f} "
            f"take_profit={pos.take_profit_price} liquidation={liquidation_price} leverage={pos.leverage}x"
        )
