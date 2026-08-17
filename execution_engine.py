"""
Execution engine for OKX Demo Trading (USDT-margined perpetual swaps).
Opens real Demo Trading positions via the OKX API, waits for the fill,
computes TP/SL from the actual filled price and fee, and places them as
algo orders. Tracks open-position count against the concurrency cap.
Position mode (net vs hedge) is set on OKXFuturesClient, not here — must
match the account's actual setting. ExecutionEngineBase exists so a
live-trading engine can subclass and swap only what differs later,
reusing the sizing/TP/position logic.
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

    enable_funding_guard: bool = True
    funding_rate_tolerance: float = 0.0

    sl_limit_slippage_pct: float = 0.0025

    enable_trailing_tp: bool = True
    trailing_tp_activation_usdt: float = 0.20
    trailing_tp_lag_usdt: float = 0.10

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
    stop_loss_is_profit_lock: bool = False
    sl_limit_price: Optional[float] = None
    sl_breach_failsafe_submitted: bool = False
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
    """Formats a price as plain fixed-point decimal, never scientific
    notation, at the instrument's own precision. str(float) silently
    breaks for tiny numbers (str(2.931e-06) == '2.931e-06'), which OKX's
    API rejects outright. Happened for real placing a PEPE take-profit.
    Python's 'f' format spec avoids this regardless of magnitude."""
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

        self._ratchet_locks: Dict[str, asyncio.Lock] = {}

        self._tp_tracker = tp_tracker or TPTracker(
            TPTrackerConfig(
                activation_profit_usdt=self.config.trailing_tp_activation_usdt,
                trail_lag_usdt=self.config.trailing_tp_lag_usdt,
                final_take_profit_usdt=self.config.target_net_profit_usdt,
            )
        )

        self._blacklisted_symbols: Dict[str, OKXAPIError] = {}

        self._last_log_ts: Dict[str, float] = {}

    @staticmethod
    def _funding_rate_allows(direction: str, funding_rate: float, tolerance: float) -> bool:
        """True unless `funding_rate` is against `direction` by more than
        `tolerance`. OKX's sign convention: positive means longs pay
        shorts (bad for long, fine/good for short); negative means shorts
        pay longs (bad for short, fine/good for long). tolerance=0.0 (the
        default) blocks on ANY adverse rate, however small; a small
        positive tolerance (e.g. 0.0001 = 0.01%) would let through mildly
        adverse rates too small to matter over a typical hold."""
        if direction == "long":
            return funding_rate <= tolerance
        else:
            return funding_rate >= -tolerance

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

        if cfg.enable_funding_guard:
            try:
                funding = await self._client.get_funding_rate(symbol)
            except OKXAPIError as exc:
                log.warning(
                    f"[execution] {symbol} {direction.upper()} — could not fetch funding rate ({exc}); "
                    f"proceeding without this check rather than blocking a valid signal over it"
                )
            else:
                funding_rate = funding["funding_rate"]
                if not self._funding_rate_allows(direction, funding_rate, cfg.funding_rate_tolerance):
                    payer = "longs" if funding_rate > 0 else "shorts"
                    self._log_throttled(
                        f"{symbol}:funding_guard_reject", logging.INFO,
                        f"[execution] {symbol} {direction.upper()} — funding rate {funding_rate:+.4%} "
                        f"is against this direction ({payer} currently pay) — signal discarded rather "
                        f"than opening into a known funding cost"
                    )
                    return None
                self._log_throttled(
                    f"{symbol}:funding_guard_pass", logging.DEBUG,
                    f"[execution] {symbol} {direction.upper()} funding guard passed — rate {funding_rate:+.4%}"
                )

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

        tp_order_id, sl_limit_price = await self._place_tp_sl(
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
            f"opening_fee={opening_fee:.6f} take_profit={take_profit_price} stop_loss={stop_loss_price} "
            f"(SL capped at {sl_limit_price})"
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
            sl_limit_price=sl_limit_price,
            liq_price=estimated_liq_price,
        )

        await self._tp_tracker.start_tracking(symbol)

        if self._position_store is not None:
            position.db_id = await self._position_store.record_open(position)

        if self._movement_tracker is not None:

            await self._movement_tracker.start_tracking(position)

        return position

    async def _cancel_and_verify(self, symbol: str, order_id: str) -> Optional[dict]:
        """Called after _wait_for_fill gives up. Cancels the still-open
        order so it can't fill later unnoticed. If cancel fails because
        it actually filled in that gap, re-checks the order's real state
        and returns the fill detail so the caller can still protect the
        position — returns None only once confirmed there's no fill."""
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
        """Retrieves the actual opening fee from the exchange (fees vary
        by pair/VIP/promo, never assumed). Retries a few times since the
        fee record can lag behind the fill. Falls back to the live quoted
        taker rate rather than 0, since a 0 fee would make the computed
        take-profit too tight to actually clear real costs."""
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
        """Derives TP from real filled entry/size and actual opening fee
        (doubled to estimate closing fee), targeting exactly
        `target_net_profit_usdt` net after both fees, no cushion.
        Defaults to config.target_net_profit_usdt but can be overridden
        smaller — used by _ratchet_stop_loss to price a tighter
        profit-lock floor with this same fee-aware math."""
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
        """Mirrors _compute_take_profit_price for the loss side: targets a
        net LOSS of exactly target_stop_loss_usdt. Fees are subtracted
        from the target (they already contribute to the loss), so less
        adverse movement is needed than the raw target implies. Rounded
        away from entry so it can't trigger prematurely on noise."""
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

    def _compute_sl_limit_price(self, direction: str, stop_loss_price: float, price_precision: str) -> float:
        """The actual order price placed for the SL leg once it triggers
        (see ExecutionConfig.sl_limit_slippage_pct) -- a real limit price
        `sl_limit_slippage_pct` further from entry than the trigger
        itself, in the SAME (adverse) direction the trigger already is,
        so the order can still fill anywhere between the trigger and this
        price but never worse than it. Rounded further away from entry
        than plain rounding would give (ROUND_DOWN for long, ROUND_UP for
        short — same conservative direction _compute_stop_loss_price
        itself uses), so the cap is never accidentally tighter than
        configured."""
        cfg = self.config
        buffer_frac = cfg.sl_limit_slippage_pct
        if direction == "long":
            raw_limit = stop_loss_price * (1 - buffer_frac)
            return _round_to_step(raw_limit, price_precision, rounding=ROUND_DOWN)
        else:
            raw_limit = stop_loss_price * (1 + buffer_frac)
            return _round_to_step(raw_limit, price_precision, rounding=ROUND_UP)

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
    ) -> tuple:
        """Places TP and SL as one OCO algo order — whichever triggers
        first cancels the other. Returns (order_id, sl_limit_price);
        both None on failure. sl_limit_price is the SL leg's real resting
        price (see sl_limit_slippage_pct) — store on
        OpenPosition.sl_limit_price for monitor_positions' breach check."""

        close_side = 3 if direction == "long" else 2

        tp_price_str = _format_price(take_profit_price, price_precision)
        sl_price_str = _format_price(stop_loss_price, price_precision)
        sl_limit_price = self._compute_sl_limit_price(direction, stop_loss_price, price_precision)
        sl_limit_price_str = _format_price(sl_limit_price, price_precision)
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
                    stop_loss_order_price=sl_limit_price_str,
                )
                order_id = str(result.get("order_id")) if result else None
                return (order_id, sl_limit_price if order_id is not None else None)
            except OKXAPIError as exc:
                last_exc = exc
                log.warning(
                    f"[execution] TP/SL placement attempt {attempt}/{attempts} failed for {symbol}: {exc}"
                )
                if attempt < attempts:
                    await asyncio.sleep(retry_delay_sec)
        log.error(f"[execution] failed to place TP/SL for {symbol} after {attempts} attempts: {last_exc}")
        return (None, None)

    async def maybe_ratchet_stop_loss_for(self, symbol: str) -> None:
        """Event-driven counterpart to _maybe_ratchet_stop_loss — call
        from run_private right after a fresh positions-push feeds
        tp_tracker.observe(), instead of waiting for the 5s poll. Closes
        the lag between peak detection and the exchange order actually
        moving. monitor_positions()'s poll remains a harmless fallback.
        No-ops if config.enable_trailing_tp is False."""
        if not self.config.enable_trailing_tp:
            return
        async with self._lock:
            pos = self._open_positions.get(symbol)
        if pos is None:
            return

        lock = self._get_ratchet_lock(symbol)
        if lock.locked():
            return

        async with lock:
            decision = await self._tp_tracker.peek(symbol)
            if not decision.ratchet:
                return
            await self._ratchet_stop_loss(symbol, pos, decision.floor_usdt)

    async def _maybe_ratchet_stop_loss(self, symbol: str, pos: OpenPosition, active: dict) -> None:
        """Feeds tp_tracker this poll's OKX unrealized_pnl as a fallback
        observation (the websocket push is the primary, faster source),
        then acts on any new floor. Only the stop-loss leg moves; TP
        stays at the original target — see tp_tracker.py for why this is
        really a trailing stop. No-ops if enable_trailing_tp is False."""
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
        verifies against OKX's live pending-order book, resubmitting up
        to ratchet_verify_max_attempts times if it doesn't match within
        ratchet_verify_tolerance_usdt. Converges on the first attempt
        normally; value is catching a genuine local rounding bug and
        making it observable. Does NOT catch fill slippage after the
        stop actually triggers — that happens later, outside this code."""
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
                await self._tp_tracker.commit(symbol, new_floor_usdt)
                return

            old_algo_id = pos.tp_order_id
            try:
                await self._client.cancel_algo_order(symbol, old_algo_id)
            except OKXAPIError as exc:
                log.warning(
                    f"[execution] {symbol} — could not cancel the existing TP/SL order while "
                    f"ratcheting the profit floor to {target_floor:.4f} USDT ({exc}); leaving "
                    f"the current order in place"
                )
                return

            new_algo_id, new_sl_limit_price = await self._place_tp_sl(
                symbol=symbol,
                direction=pos.direction,
                take_profit_price=pos.take_profit_price,
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
            pos.sl_limit_price = new_sl_limit_price
            pos.stop_loss_is_profit_lock = True

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
            target_floor = new_floor_usdt - deviation

        log.error(
            f"[execution] {symbol} — profit-floor ratchet did not converge to within "
            f"{cfg.ratchet_verify_tolerance_usdt:.4f} USDT after {cfg.ratchet_verify_max_attempts} attempts; "
            f"leaving the last-placed order in place rather than looping indefinitely"
        )
        await self._tp_tracker.commit(symbol, new_floor_usdt)

    async def monitor_positions(self) -> None:
        """Checks each tracked position against the exchange; zero size
        means TP/SL/liquidation closed it, so the slot is freed. Only
        OKX's own orders/liquidation end a position (ratchet fail-safe
        is the one exception). Declares a close only after
        _MIN_AGE_BEFORE_CLOSE_CHECK_SEC AND a second confirming read —
        two LINK trades once got wrongly marked closed within seconds
        from a single stale exchange read, orphaning live positions."""
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
            await self._check_sl_breach(symbol, pos, active)
            await self._maybe_ratchet_stop_loss(symbol, pos, active)

    async def reconcile_from_store(self, position_store) -> None:
        """Runs once at startup before any new trades open. For every
        position_history row still marked "open" from a prior process:
        if OKX still shows it open, re-registers it into tracking so
        monitoring/ratchet/alerts resume; if OKX shows it closed, pulls
        real close details and marks the DB row closed instead of
        leaving it stuck. Also reseeds self._total_opened from the
        store's full row count so max_total_trades survives restarts.
        Limitation: tp_tracker/movement_tracker history for a resumed
        position restarts fresh — nothing pre-restart is reconstructed."""
        if position_store is None:
            return

        self._total_opened = await position_store.count_all()

        rows = await position_store.get_open_rows()
        if not rows:
            log.info(f"[startup] no open position_history rows to reconcile (lifetime trades so far: {self._total_opened})")
            return

        log.info(f"[startup] reconciling {len(rows)} open position_history row(s) against OKX (lifetime trades so far: {self._total_opened})...")

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

        sl_limit_price = (
            self._compute_sl_limit_price(direction, stop_loss_price, price_precision)
            if stop_loss_price
            else None
        )

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
            sl_limit_price=sl_limit_price,
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
        """Looks up OKX's positions-history for real exit price, realized
        PnL, closing fee, and net PnL — never estimated. Retries since
        the record can lag behind monitor_positions() seeing the close.
        Falls back to /trade/fills for exit_price/closing_fee only (no
        real realized-PnL field there for swaps). Retry budget is wide:
        two LINK positions once got liquidated within a second of
        opening and the old shorter budget gave up too early."""
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
        """Infers TP vs SL by comparing real exit price against the
        position's own take_profit_price/stop_loss_price, rather than
        asking OKX which algo leg fired — the old orders-algo-history
        lookup broke once TP/SL became one OCO order (wrong ordType
        filter, and one shared algoId can't say which leg triggered).
        Only runs for closes OKX didn't already flag as liquidation.
        If the ratchet moved stop_loss_price into profit territory
        (stop_loss_is_profit_lock), a fill there is reported
        "trailing_stop", not "stop_loss" — needs adding to any DB CHECK
        constraint on close_reason."""
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

    async def _check_sl_breach(self, symbol: str, pos: OpenPosition, active: dict) -> None:
        """Fail-safe for sl_limit_slippage_pct: the capped SL leg is a
        LIMIT order, so it can go unfilled if price gaps through it. If
        mark price has passed pos.sl_limit_price by more than a small
        buffer while the position is still open, cancels the stuck order
        and fires an uncapped market close — same protection an uncapped
        SL gives, just up to one poll cycle later. No-ops if
        sl_limit_price is unset, and fires at most once per position
        (sl_breach_failsafe_submitted)."""
        if pos.sl_limit_price is None or pos.sl_breach_failsafe_submitted:
            return

        cfg = self.config
        try:
            mark_price = float(active.get("mark_price", 0))
        except (TypeError, ValueError):
            return
        if mark_price <= 0:
            return

        confirm_buffer = pos.sl_limit_price * cfg.sl_limit_slippage_pct
        if pos.direction == "long":
            breached = mark_price < pos.sl_limit_price - confirm_buffer
        else:
            breached = mark_price > pos.sl_limit_price + confirm_buffer
        if not breached:
            return

        log.error(
            f"[execution] {symbol} {pos.direction.upper()} — mark price {mark_price} has moved past "
            f"the capped stop-loss limit ({pos.sl_limit_price}) while the position is still open; the "
            f"limit order didn't fill in time. Cancelling it and submitting an uncapped market close "
            f"as a fail-safe rather than leaving this position unprotected any longer"
        )

        async with self._lock:
            pos.sl_breach_failsafe_submitted = True

        if pos.tp_order_id:
            try:
                await self._client.cancel_algo_order(symbol, pos.tp_order_id)
            except OKXAPIError as exc:
                log.warning(
                    f"[execution] {symbol} — could not cancel the stuck TP/SL order {pos.tp_order_id} "
                    f"before the breach fail-safe close ({exc}); proceeding with the market close anyway"
                )

        close_side = 3 if pos.direction == "long" else 2
        try:
            await self._client.submit_order(
                symbol=symbol, side=close_side, size=pos.size_contracts, order_type="market"
            )
            log.warning(f"[execution] {symbol} — SL-breach fail-safe close submitted")
        except OKXAPIError as exc:
            log.error(
                f"[execution] {symbol} — SL-breach fail-safe close ALSO failed ({exc}); this position "
                f"is genuinely unprotected and needs manual attention immediately"
            )
            async with self._lock:
                pos.sl_breach_failsafe_submitted = False

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
