"""
Execution engine for BitMart Demo Futures trading.

This module replaces the old paper-trading simulator with a real (demo)
execution path: it opens actual Demo Futures positions through the BitMart
API, waits for the fill, computes a take-profit price from the *actual*
filled price and *actual* opening fee (never a hardcoded fee assumption),
and places that take-profit on the exchange. It also tracks how many
positions are open so the bot never exceeds the configured concurrency cap.

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

from bitmart_futures_client import BitMartAPIError, BitMartFuturesClient
from market_data import Signal

log = logging.getLogger("bitmart_futures.execution")


@dataclass
class ExecutionConfig:
    margin_per_trade_usdt: float = 1.0
    requested_leverage: int = 100
    min_leverage_required: int = 50
    max_open_positions: int = 5
    target_net_profit_usdt: float = 0.10
    open_type: str = "isolated"
    fill_poll_interval_sec: float = 0.5
    fill_timeout_sec: float = 15.0
    # Log a price-movement update whenever a position moves another
    # `alert_move_step_pct` percentage points further from its entry price
    # (in either direction). E.g. with 0.5, alerts fire at +0.5%, +1.0%,
    # -0.5%, -1.0%, etc. — never more than once per step, so the log isn't
    # spammed on every monitor tick.
    alert_move_step_pct: float = 0.5


@dataclass
class OpenPosition:
    symbol: str
    direction: str
    order_id: str
    entry_price: float
    size_contracts: float
    contract_size: float
    leverage: int
    opening_fee: float
    take_profit_price: float
    tp_order_id: Optional[str]
    opened_at: float = field(default_factory=time.time)
    last_alert_level: int = 0


def _round_to_step(value: float, step_str: str, rounding=ROUND_DOWN) -> float:
    """Round `value` down (or up) to the nearest multiple of `step_str`
    (e.g. price_precision '0.1' or vol_precision '1')."""
    step = Decimal(str(step_str))
    if step <= 0:
        return value
    quant = (Decimal(str(value)) / step).to_integral_value(rounding=rounding) * step
    return float(quant)


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
    """Executes signals as real orders against BitMart Demo Futures Trading."""

    def __init__(self, client: BitMartFuturesClient, config: Optional[ExecutionConfig] = None) -> None:
        self._client = client
        self.config = config or ExecutionConfig()
        self._open_positions: Dict[str, OpenPosition] = {}
        self._lock = asyncio.Lock()

    async def open_count(self) -> int:
        async with self._lock:
            return len(self._open_positions)

    async def has_open_position(self, symbol: str) -> bool:
        async with self._lock:
            return symbol in self._open_positions

    # ------------------------------------------------------------------
    # Opening a trade
    # ------------------------------------------------------------------

    async def try_open_trade(self, signal: Signal) -> bool:
        cfg = self.config
        symbol = signal.symbol

        # Reserve a slot atomically so two concurrent signals can't both
        # open a position once we're at the concurrency cap.
        async with self._lock:
            if symbol in self._open_positions:
                return False
            if len(self._open_positions) >= cfg.max_open_positions:
                log.info(f"[execution] max open positions ({cfg.max_open_positions}) reached — skipping {symbol}")
                return False
            self._open_positions[symbol] = None  # placeholder reservation

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
        return True

    async def _open_position(self, signal: Signal) -> Optional[OpenPosition]:
        cfg = self.config
        symbol = signal.symbol
        direction = signal.direction

        try:
            contract = await self._client.get_contract_details(symbol)
        except BitMartAPIError as exc:
            log.error(f"[execution] could not fetch contract details for {symbol}: {exc}")
            return None

        max_leverage_symbol = int(float(contract.get("max_leverage", 0)))
        if max_leverage_symbol < cfg.min_leverage_required:
            log.info(
                f"[execution] {symbol} max leverage {max_leverage_symbol}x < required "
                f"{cfg.min_leverage_required}x — signal discarded"
            )
            return None

        leverage = min(max_leverage_symbol, cfg.requested_leverage)
        contract_size = float(contract.get("contract_size", 0))
        price_precision = contract.get("price_precision", "0.01")
        vol_precision = contract.get("vol_precision", "1")
        min_volume = float(contract.get("min_volume", 1))
        if contract_size <= 0:
            log.error(f"[execution] {symbol} has invalid contract_size — skipping")
            return None

        notional_usdt = cfg.margin_per_trade_usdt * leverage
        qty_base = notional_usdt / signal.entry_price
        size_contracts = _round_to_step(qty_base / contract_size, vol_precision, rounding=ROUND_DOWN)

        if size_contracts < min_volume:
            log.info(
                f"[execution] {symbol} minimum order size ({min_volume} contracts) exceeds what "
                f"${cfg.margin_per_trade_usdt} margin at {leverage}x can buy — signal discarded"
            )
            return None

        try:
            await self._client.submit_leverage(symbol, leverage, cfg.open_type)
        except BitMartAPIError as exc:
            log.error(f"[execution] failed to set leverage for {symbol}: {exc}")
            return None

        side = 1 if direction == "long" else 4
        client_order_id = f"bot{uuid.uuid4().hex[:12]}"
        try:
            order = await self._client.submit_order(
                symbol=symbol,
                side=side,
                size=int(size_contracts),
                order_type="market",
                leverage=str(leverage),
                open_type=cfg.open_type,
                client_order_id=client_order_id,
            )
        except BitMartAPIError as exc:
            log.error(f"[execution] order submission failed for {symbol}: {exc}")
            return None

        order_id = str(order.get("order_id"))
        log.info(f"[execution] opened {direction.upper()} {symbol} order_id={order_id} size={int(size_contracts)}")

        filled = await self._wait_for_fill(symbol, order_id)
        if filled is None:
            log.error(f"[execution] {symbol} order {order_id} did not fill within timeout")
            return None

        deal_avg_price = float(filled.get("deal_avg_price", 0))
        deal_size = float(filled.get("deal_size", 0))
        if deal_avg_price <= 0 or deal_size <= 0:
            log.error(f"[execution] {symbol} order {order_id} reported no fill — abandoning")
            return None

        opening_fee = await self._get_opening_fee(symbol, order_id)

        take_profit_price = self._compute_take_profit_price(
            direction=direction,
            entry_price=deal_avg_price,
            size_contracts=deal_size,
            contract_size=contract_size,
            opening_fee=opening_fee,
            price_precision=price_precision,
        )

        tp_order_id = await self._place_take_profit(
            symbol=symbol,
            direction=direction,
            take_profit_price=take_profit_price,
            size_contracts=int(deal_size),
        )

        log.info(
            f"[execution] {symbol} filled entry={deal_avg_price} size={deal_size} "
            f"opening_fee={opening_fee:.6f} take_profit={take_profit_price}"
        )

        return OpenPosition(
            symbol=symbol,
            direction=direction,
            order_id=order_id,
            entry_price=deal_avg_price,
            size_contracts=deal_size,
            contract_size=contract_size,
            leverage=leverage,
            opening_fee=opening_fee,
            take_profit_price=take_profit_price,
            tp_order_id=tp_order_id,
        )

    async def _wait_for_fill(self, symbol: str, order_id: str) -> Optional[dict]:
        cfg = self.config
        deadline = time.time() + cfg.fill_timeout_sec
        while time.time() < deadline:
            try:
                detail = await self._client.get_order(symbol, order_id)
            except BitMartAPIError as exc:
                log.warning(f"[execution] order status check failed for {symbol} {order_id}: {exc}")
                await asyncio.sleep(cfg.fill_poll_interval_sec)
                continue
            if str(detail.get("state")) == "4":
                return detail
            await asyncio.sleep(cfg.fill_poll_interval_sec)
        return None

    async def _get_opening_fee(self, symbol: str, order_id: str) -> float:
        """Retrieves the actual fee charged for opening the position. Trading
        fees vary by pair/VIP level/promotions, so this is always looked up
        from the exchange rather than assumed."""
        try:
            trades = await self._client.get_trades(symbol=symbol, order_id=order_id)
        except BitMartAPIError as exc:
            log.warning(f"[execution] could not fetch fill fees for {symbol} {order_id}: {exc} — assuming 0 fee")
            return 0.0
        return sum(float(t.get("paid_fees", 0)) for t in trades)

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
        realized profit of `target_net_profit_usdt` after both fees."""
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

    async def _place_take_profit(
        self, symbol: str, direction: str, take_profit_price: float, size_contracts: int
    ) -> Optional[str]:
        # Hedge-mode close sides: 3 closes a long, 2 closes a short.
        close_side = 3 if direction == "long" else 2
        price_str = str(take_profit_price)
        try:
            result = await self._client.submit_tp_sl_order(
                symbol=symbol,
                order_type="take_profit",
                side=close_side,
                trigger_price=price_str,
                executive_price=price_str,
                price_type=1,
                size=size_contracts,
                plan_category=2,
                category="market",
            )
            return str(result.get("order_id")) if result else None
        except BitMartAPIError as exc:
            log.error(f"[execution] failed to place take-profit for {symbol}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Monitoring open positions
    # ------------------------------------------------------------------

    async def monitor_positions(self) -> None:
        """Checks each tracked position against the exchange. A position with
        zero remaining size has been closed by its take-profit order or by
        exchange liquidation — either way, we free the slot. No stop loss and
        no timeout logic exist here by design: BitMart's own liquidation
        engine is the only thing that can end a losing position.

        While a position stays open, it's also checked for a significant
        price move (see `_check_price_alert`) so a trade running against —
        or in favor of — us gets surfaced without flooding the log."""
        async with self._lock:
            tracked = {s: p for s, p in self._open_positions.items() if p is not None}

        for symbol, pos in tracked.items():
            try:
                positions = await self._client.get_position(symbol=symbol)
            except BitMartAPIError as exc:
                log.warning(f"[execution] position check failed for {symbol}: {exc}")
                continue

            active = next((p for p in positions if float(p.get("current_amount", 0)) > 0), None)

            if active is None:
                async with self._lock:
                    closed = self._open_positions.pop(symbol, None)
                if closed is not None:
                    log.info(
                        f"[execution] {symbol} position closed (entry={closed.entry_price} "
                        f"take_profit={closed.take_profit_price}) — slot freed "
                        f"({len(self._open_positions)}/{self.config.max_open_positions} open)"
                    )
                continue

            await self._check_price_alert(symbol, pos, active)

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

        level = int(move_pct / step)  # truncates toward zero -> one bucket per `step`
        if level == pos.last_alert_level:
            return

        async with self._lock:
            current = self._open_positions.get(symbol)
            if current is None:
                return  # closed out from under us between the check above and here
            current.last_alert_level = level

        outlook = "favorable" if move_pct >= 0 else "adverse"
        log.info(
            f"[monitor] {symbol} {pos.direction.upper()} move {move_pct:+.2f}% ({outlook}) — "
            f"entry={pos.entry_price} mark={mark_price} unrealized_pnl={unrealized_pnl:+.4f} "
            f"take_profit={pos.take_profit_price} liquidation={liquidation_price} leverage={pos.leverage}x"
        )
