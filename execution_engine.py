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
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Dict, List, Optional

from okx_futures_client import OKXAPIError, OKXFuturesClient
from market_data import Signal
from liquidation_guard import check_liquidation_distance, select_leverage_with_safe_liquidation, estimate_liquidation_price
from pretrade_validation import validate_take_profit_reachable, validate_liquidation_history

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

    enable_tp_reachability_check: bool = True
    tp_validation_lookback_hours: float = 1.0
    min_tp_hits_required: int = 1
    estimated_fee_by_leverage: dict = field(default_factory=lambda: {50: 0.12, 10: 0.02})

    enable_liquidation_history_check: bool = True
    liq_validation_lookback_hours: float = 15.0
    max_liq_hits_allowed: int = 1

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
    async def try_open_trade(self, signal: Signal, signal_snapshot: Optional[dict] = None) -> bool:
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
        signal_store=None,
    ) -> None:
        self._client = client
        self.config = config or ExecutionConfig()
        self._position_store = position_store
        self._movement_tracker = movement_tracker
        self._signal_store = signal_store
        self._open_positions: Dict[str, OpenPosition] = {}
        self._total_opened = 0
        self._lock = asyncio.Lock()

        self._blacklisted_symbols: Dict[str, OKXAPIError] = {}

    async def open_count(self) -> int:
        async with self._lock:
            return len(self._open_positions)

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

    async def try_open_trade(self, signal: Signal, signal_snapshot: Optional[dict] = None) -> bool:
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

        if signal_snapshot is not None:
            await self._record_signal_history(opened, signal_snapshot)
        return True

    async def _record_signal_history(self, opened: OpenPosition, signal_snapshot: dict) -> None:
        """Persists the evidence-pipeline breakdown captured at accept
        time (see EvidenceSummary.to_signal_record()) now that the trade
        actually has a trade_id/entry_price to attach it to. Best-effort,
        same as position_store/movement_tracker calls elsewhere — a
        signal-history write failing must never take trading down with
        it."""
        if self._signal_store is None:
            return
        try:
            record = dict(signal_snapshot)
            record["trade_id"] = opened.db_id
            record["entry_price"] = opened.entry_price
            await self._signal_store.record_signal(record)
        except Exception:
            log.exception(
                f"[signal-store] failed to record signal history for "
                f"{opened.symbol} (trade_id={opened.db_id})"
            )

    async def _open_position(self, signal: Signal) -> Optional[OpenPosition]:
        cfg = self.config
        symbol = signal.symbol
        direction = signal.direction

        try:
            contract = await self._client.get_contract_details(symbol)
        except OKXAPIError as exc:
            log.error(f"[execution] could not fetch contract details for {symbol}: {exc}")
            return None

        max_leverage_symbol = int(float(contract.get("max_leverage", 0)))
        if max_leverage_symbol < cfg.min_leverage_required:
            log.info(
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
            log.info(
                f"[execution] {symbol} {direction.upper()} — hardcoded leverage {target_leverage:.0f}x "
                f"< required {cfg.min_leverage_required}x — signal discarded"
            )
            return None

        notional_at_lev = cfg.margin_per_trade_usdt * target_leverage
        try:
            mmr = await self._client.get_position_tier_mmr(symbol, cfg.open_type, notional_at_lev)
        except OKXAPIError as exc:
            log.warning(
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
                log.info(f"[execution] {symbol} {direction.upper()} — {liq_check.reason}")
                return None

            leverage = liq_check.leverage
            estimated_liq_price = liq_check.liquidation_price
            log.info(
                f"[execution] {symbol} {direction.upper()} liquidation guard passed at {leverage:.0f}x — "
                f"est. liq={liq_check.liquidation_price:.8f} "
                f"({liq_check.distance_pct:.2%} from entry {signal.entry_price})"
            )
        else:
            leverage = target_leverage
            estimated_liq_price = estimate_liquidation_price(signal.entry_price, leverage, mmr, direction)

        notional_usdt = cfg.margin_per_trade_usdt * leverage

        if cfg.enable_tp_reachability_check:
            estimated_fee = cfg.estimated_fee_by_leverage.get(
                int(leverage), max(cfg.estimated_fee_by_leverage.values())
            )
            try:
                tp_candles = await self._client.get_candles(
                    symbol,
                    bar=cfg.candle_bar,
                    limit=_candle_limit_for_hours(cfg.tp_validation_lookback_hours, cfg.candle_bar),
                )
            except OKXAPIError as exc:
                log.warning(
                    f"[execution] {symbol} {direction.upper()} — could not fetch candles for the TP "
                    f"reachability check ({exc}); discarding rather than trading without this filter"
                )
                return None

            tp_result = validate_take_profit_reachable(
                entry_price=signal.entry_price,
                direction=direction,
                target_net_profit_usdt=cfg.target_net_profit_usdt,
                estimated_fee_usdt=estimated_fee,
                notional_usdt=notional_usdt,
                candles=tp_candles,
                min_hits=cfg.min_tp_hits_required,
            )
            if not tp_result.approved:
                log.info(f"[execution] {symbol} {direction.upper()} — {tp_result.reason}")
                return None
            log.info(
                f"[execution] {symbol} {direction.upper()} TP reachability passed — planned TP "
                f"{tp_result.planned_tp_price:.8f} reached {tp_result.hits}x in the last "
                f"{cfg.tp_validation_lookback_hours:.0f}h"
            )

        if cfg.enable_liquidation_history_check:
            try:
                liq_candles = await self._client.get_candles(
                    symbol,
                    bar=cfg.candle_bar,
                    limit=_candle_limit_for_hours(cfg.liq_validation_lookback_hours, cfg.candle_bar),
                )
            except OKXAPIError as exc:
                log.warning(
                    f"[execution] {symbol} {direction.upper()} — could not fetch candles for the "
                    f"liquidation-history check ({exc}); discarding rather than trading without this filter"
                )
                return None

            liq_history_result = validate_liquidation_history(
                liquidation_price=estimated_liq_price,
                direction=direction,
                candles=liq_candles,
                max_hits=cfg.max_liq_hits_allowed,
            )
            if not liq_history_result.approved:
                log.info(f"[execution] {symbol} {direction.upper()} — {liq_history_result.reason}")
                return None
            log.info(
                f"[execution] {symbol} {direction.upper()} liquidation-history check passed — "
                f"est. liq {estimated_liq_price:.8f} touched {liq_history_result.hits}x in the last "
                f"{cfg.liq_validation_lookback_hours:.0f}h"
            )

        qty_base = notional_usdt / signal.entry_price
        size_contracts = _round_to_step(qty_base / contract_size, vol_precision, rounding=ROUND_DOWN)

        if size_contracts < min_volume:
            log.info(
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
            log.error(f"[execution] {symbol} order {order_id} did not fill within timeout")
            return None

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
            liq_price=estimated_liq_price,
        )

        if self._position_store is not None:
            position.db_id = await self._position_store.record_open(position)

        if self._movement_tracker is not None:

            await self._movement_tracker.start_tracking(position)

        return position

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
    ) -> float:
        """Take profit is derived entirely from real, exchange-reported
        numbers: the actual filled entry price/size and the actual opening
        fee (doubled to estimate the matching closing fee), targeting a net
        realized profit of exactly `target_net_profit_usdt` after both fees
        — no added cushion. A prior version added a slippage buffer here
        (0.20% extra required price move), which pushed the TP price out
        to ~0.30-0.39 USDT of required profit against a $0.07 target,
        defeating the scalp — removed per explicit instruction."""
        cfg = self.config
        estimated_total_fees = opening_fee * 2.0
        notional = size_contracts * contract_size * entry_price
        required_gross_profit = cfg.target_net_profit_usdt + estimated_total_fees
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

    async def monitor_positions(self) -> None:
        """Checks each tracked position against the exchange. A position with
        zero remaining size has been closed by its take-profit order or by
        exchange liquidation — either way, we free the slot. No stop loss and
        no timeout logic exist here by design: OKX's own liquidation
        engine is the only thing that can end a losing position.

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

    async def _finalize_closed_position(self, symbol: str, closed: OpenPosition) -> None:
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
            if close_type in ("3", "4", "5", "6"):

                close_reason = "liquidated"
            else:
                resolved_order_id = await self._resolve_tp_execution_order_id(symbol, closed)
                close_reason = "take_profit" if resolved_order_id else "liquidated"

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
                float(history_row["exit_price"]),
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
        for this close yet. Recovers exit_price/closing_fee/close_reason
        from raw fills, same as the original implementation — but cannot
        recover a real realized_pnl or net_pnl this way (see
        _get_close_details); net_pnl comes back None so the caller falls
        back to a local approximation."""
        resolved_order_id = await self._resolve_tp_execution_order_id(symbol, closed)

        opened_at_ms = closed.opened_at * 1000.0

        closing_side = "sell" if closed.direction == "long" else "buy"

        closing_trades: List[dict] = []
        for attempt in range(1, 6):
            try:
                if resolved_order_id:
                    trades = await self._client.get_trades(symbol=symbol, order_id=resolved_order_id)
                else:
                    trades = await self._client.get_trades(symbol=symbol)
            except OKXAPIError as exc:
                log.warning(f"[execution] could not fetch closing trades for {symbol}: {exc}")
                trades = []

            if resolved_order_id:
                closing_trades = trades
            else:
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

        close_reason = "take_profit" if resolved_order_id else "liquidated"
        return exit_price, realized_pnl, closing_fee, close_reason, None

    async def _resolve_tp_execution_order_id(self, symbol: str, closed: OpenPosition) -> Optional[str]:
        """`closed.tp_order_id` is the algoId of the TP algo order placed at
        open time. Checks whether it has triggered ("effective") and, if
        so, returns the ordId of the order it spawned so fills can be
        pulled precisely. Verify `get_algo_order_status`'s field names
        against a live demo response — OKX's exact linkage field for a
        triggered conditional order's child ordId should be confirmed
        before relying on this in production; the fallback path above
        (scan closing-side fills since open) still produces correct P&L
        even if this returns None."""
        if not closed.tp_order_id:
            return None
        try:
            status = await self._client.get_algo_order_status(symbol, closed.tp_order_id)
        except OKXAPIError as exc:
            log.warning(f"[execution] could not fetch TP algo order status for {symbol}: {exc}")
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
