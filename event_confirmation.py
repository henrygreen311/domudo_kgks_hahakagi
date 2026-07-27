"""
Event Confirmation Layer.

Sits between SignalGenerator and the execution engine:

    SignalGenerator -> EventConfirmationEngine -> ExecutionEngine

SignalGenerator already separates prerequisite/quality checks (spread,
freshness, liquidity, connection health, book sync — gating only, never
confidence) from directional checks (volume dominance, aggressive orders,
intensity, liquidity imbalance — these DO set confidence). A Signal that
clears SignalGenerator is therefore a *directionally confident candidate*,
not yet something to trade.

This module adds the final gate: before opening a trade, require at least
one (configurably two) concrete market *event* that confirms the directional
read is actually playing out right now — a whale trade, an order-book sweep,
a burst of aggressive volume, or price momentum following a whale trade.

All data is reused from the existing stores (TradeStore, OrderFlowAnalyzer,
OrderBookStore, MarketDataStore) — this module computes new things from that
data, it doesn't duplicate collection.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from market_data import LiquidityEngine, MarketDataStore, OrderBookStore, OrderFlowAnalyzer, Signal, TradeStore, compute_order_flow_metrics

log = logging.getLogger("okx_futures.confirmation")


@dataclass
class EventConfirmationConfig:
    # Directional confidence produced by SignalGenerator must clear this bar
    # before we even look for a confirming event.
    min_signal_confidence: float = 0.70

    # How many distinct confirmation events must fire before a trade opens.
    # Set to 2 to require two independent confirmations.
    min_confirmations_required: int = 1

    # --- A. Whale trade detection ---
    enable_whale_confirmation: bool = True
    whale_multiplier: float = 5.0  # single_trade_size >= multiplier x rolling_average_trade_size
    whale_window_ms: int = 5000  # rolling window the average trade size is computed over

    # --- B. Order book sweep detection ---
    enable_sweep_confirmation: bool = True
    sweep_levels: int = 3  # number of top levels that must be consumed to count as a sweep
    sweep_window_ms: int = 500

    # --- C. Aggressive volume burst ---
    enable_aggressive_volume_confirmation: bool = True
    aggressive_volume_threshold: float = 0.80  # aggressive buy/sell vol >= 80% of total volume in the window
    aggressive_volume_window_ms: int = 500

    # --- D. Momentum confirmation (requires a whale trade to anchor "after") ---
    enable_momentum_confirmation: bool = True
    momentum_confirmation_pct: float = 0.003  # 0.3%
    momentum_confirmation_window_ms: int = 1000

    # --- Section 4: additional optional confirmation modules ---
    # Fully implemented, reusing existing data — opt-in since they're
    # explicitly called out as optional extras, not core signals.
    enable_multi_level_whale_sweep: bool = False  # whale trade that ALSO swept multiple book levels
    enable_volume_spike_confirmation: bool = False
    volume_spike_multiplier: float = 2.0  # recent-window volume >= multiplier x baseline average
    volume_spike_window_ms: int = 2000
    volume_spike_baseline_multiplier: int = 5  # baseline window = volume_spike_window_ms * this
    enable_flow_persistence_confirmation: bool = False
    flow_persistence_updates_required: int = 3  # how many of OrderFlowAnalyzer's rolling windows must agree
    enable_momentum_acceleration: bool = False  # price velocity increasing across two consecutive sub-windows

    # --- Section 4: modules that need data feeds this bot doesn't collect yet ---
    # Left here (disabled) so the shape of the config is complete and toggling
    # them on is a one-line change once the underlying feed exists. Enabling
    # them today is a no-op (logged once) rather than silently pretending to
    # confirm something we have no data for.
    enable_open_interest_confirmation: bool = False  # needs an open-interest feed
    enable_funding_rate_confirmation: bool = False  # needs a funding-rate feed
    enable_breakout_confirmation: bool = False  # needs computed resistance levels
    enable_breakdown_confirmation: bool = False  # needs computed support levels


@dataclass
class ConfirmationResult:
    confirmed: bool
    confirmations: List[str] = field(default_factory=list)


class EventConfirmationEngine:
    def __init__(
        self,
        trade_store: TradeStore,
        order_flow: OrderFlowAnalyzer,
        order_book: OrderBookStore,
        market_data: MarketDataStore,
        liquidity_engine: Optional[LiquidityEngine] = None,
        config: Optional[EventConfirmationConfig] = None,
    ) -> None:
        self._trade_store = trade_store
        self._order_flow = order_flow
        self._order_book = order_book
        self._market_data = market_data
        self._liquidity_engine = liquidity_engine  # reserved for future breakout/breakdown modules
        self.config = config or EventConfirmationConfig()
        self._unavailable_modules_logged: set = set()

    async def confirm(self, signal: Signal) -> ConfirmationResult:
        cfg = self.config
        if signal.confidence < cfg.min_signal_confidence:
            return ConfirmationResult(confirmed=False)

        symbol = signal.symbol
        direction = signal.direction
        direction_word = "buy" if direction == "long" else "sell"
        triggered: List[str] = []

        need_whale = cfg.enable_whale_confirmation or cfg.enable_momentum_confirmation or cfg.enable_multi_level_whale_sweep
        whale_trade = await self._detect_whale_trade(symbol, direction) if need_whale else None

        if cfg.enable_whale_confirmation and whale_trade is not None:
            ratio = (whale_trade["qty"] / whale_trade["avg_qty"]) if whale_trade["avg_qty"] else 0.0
            triggered.append(f"Whale {direction_word} detected ({ratio:.1f}x average)")

        need_sweep = cfg.enable_sweep_confirmation or cfg.enable_multi_level_whale_sweep
        swept_levels = await self._detect_order_book_sweep(symbol, direction) if need_sweep else None
        if cfg.enable_sweep_confirmation and swept_levels:
            triggered.append(f"Order book sweep detected ({swept_levels} levels)")

        if cfg.enable_aggressive_volume_confirmation:
            burst_pct = await self._detect_aggressive_volume_burst(symbol, direction)
            if burst_pct is not None:
                triggered.append(f"Aggressive {direction_word} volume burst ({burst_pct:.0%} of volume)")

        if cfg.enable_momentum_confirmation and whale_trade is not None:
            move_pct, elapsed_ms = await self._confirm_momentum(symbol, direction, whale_trade)
            if move_pct is not None:
                triggered.append(f"Momentum confirmed ({move_pct:+.2%} in {elapsed_ms / 1000.0:.1f}s)")

        if cfg.enable_multi_level_whale_sweep and whale_trade is not None and swept_levels:
            triggered.append(f"Whale trade swept {swept_levels} order book levels")

        if cfg.enable_volume_spike_confirmation:
            spike_ratio = await self._detect_volume_spike(symbol)
            if spike_ratio is not None:
                triggered.append(f"Volume spike detected ({spike_ratio:.1f}x recent average)")

        if cfg.enable_flow_persistence_confirmation:
            if await self._detect_flow_persistence(symbol, direction):
                triggered.append(f"Order flow imbalance persisted across {cfg.flow_persistence_updates_required} updates")

        if cfg.enable_momentum_acceleration:
            if await self._detect_momentum_acceleration(symbol, direction):
                triggered.append("Momentum acceleration detected")

        for flag, name in (
            (cfg.enable_open_interest_confirmation, "open-interest"),
            (cfg.enable_funding_rate_confirmation, "funding-rate"),
            (cfg.enable_breakout_confirmation, "breakout"),
            (cfg.enable_breakdown_confirmation, "breakdown"),
        ):
            if flag and name not in self._unavailable_modules_logged:
                self._unavailable_modules_logged.add(name)
                log.warning(
                    f"[confirmation] {name} confirmation is enabled but no {name} data feed is wired up yet — "
                    f"it will never fire until that's added"
                )

        confirmed = len(triggered) >= cfg.min_confirmations_required
        return ConfirmationResult(confirmed=confirmed, confirmations=triggered)

    # ------------------------------------------------------------------
    # A. Whale trade detection
    # ------------------------------------------------------------------

    async def _detect_whale_trade(self, symbol: str, direction: str) -> Optional[dict]:
        cfg = self.config
        trades = await self._trade_store.get_window(symbol, cfg.whale_window_ms)
        if not trades:
            return None
        total_qty = sum(t["qty"] for t in trades)
        avg_qty = total_qty / len(trades)
        if avg_qty <= 0:
            return None
        threshold = avg_qty * cfg.whale_multiplier
        side = "buy" if direction == "long" else "sell"
        candidates = [t for t in trades if t["side"] == side and t["qty"] >= threshold]
        if not candidates:
            return None
        latest = max(candidates, key=lambda t: t["timestamp"])
        return {"qty": latest["qty"], "price": latest["price"], "timestamp": latest["timestamp"], "avg_qty": avg_qty}

    # ------------------------------------------------------------------
    # B. Order book sweep detection
    # ------------------------------------------------------------------

    async def _detect_order_book_sweep(self, symbol: str, direction: str) -> Optional[int]:
        cfg = self.config
        history = await self._order_book.get_book_history(symbol, cfg.sweep_window_ms)
        if len(history) < 2:
            return None

        # Sweeping through asks is bullish (long); sweeping through bids is bearish (short).
        side_key = "asks" if direction == "long" else "bids"

        # Bid/ask updates can arrive as separate depth messages, so the very
        # first snapshot in the window may have an empty side simply because
        # that side hadn't been touched yet — not because it was swept. Use
        # the earliest snapshot that actually has data for the side we care about.
        earliest_levels = None
        for _, snap in history:
            levels = snap.get(side_key, [])
            if levels:
                earliest_levels = levels[: cfg.sweep_levels]
                break
        if not earliest_levels:
            return None

        current_book = await self._order_book.get_book(symbol)
        if not current_book:
            return None
        latest_prices = {price for price, vol in current_book.get(side_key, []) if vol > 0}
        removed = sum(1 for price, _ in earliest_levels if price not in latest_prices)
        return removed if removed >= cfg.sweep_levels else None

    # ------------------------------------------------------------------
    # C. Aggressive volume burst
    # ------------------------------------------------------------------

    async def _detect_aggressive_volume_burst(self, symbol: str, direction: str) -> Optional[float]:
        cfg = self.config
        trades = await self._trade_store.get_window(symbol, cfg.aggressive_volume_window_ms)
        if not trades:
            return None
        metrics = compute_order_flow_metrics(trades, window_sec=cfg.aggressive_volume_window_ms / 1000.0)
        if direction == "long" and metrics["aggressive_buy_pct"] >= cfg.aggressive_volume_threshold:
            return metrics["aggressive_buy_pct"]
        if direction == "short" and metrics["aggressive_sell_pct"] >= cfg.aggressive_volume_threshold:
            return metrics["aggressive_sell_pct"]
        return None

    # ------------------------------------------------------------------
    # D. Momentum confirmation (anchored to a whale trade's own fill price)
    # ------------------------------------------------------------------

    async def _confirm_momentum(self, symbol: str, direction: str, whale_trade: dict):
        cfg = self.config
        now_ms = time.time() * 1000.0
        elapsed_ms = now_ms - whale_trade["timestamp"]
        if elapsed_ms < 0 or elapsed_ms > cfg.momentum_confirmation_window_ms:
            return None, None

        market = await self._market_data.get(symbol)
        if not market:
            return None, None

        base_price = whale_trade["price"]
        if base_price <= 0:
            return None, None
        move_pct = (market["last_price"] - base_price) / base_price

        if direction == "long" and move_pct >= cfg.momentum_confirmation_pct:
            return move_pct, elapsed_ms
        if direction == "short" and move_pct <= -cfg.momentum_confirmation_pct:
            return move_pct, elapsed_ms
        return None, None

    # ------------------------------------------------------------------
    # Section 4 (implemented): volume spike, flow persistence, momentum acceleration
    # ------------------------------------------------------------------

    async def _detect_volume_spike(self, symbol: str) -> Optional[float]:
        cfg = self.config
        baseline_window_ms = cfg.volume_spike_window_ms * cfg.volume_spike_baseline_multiplier
        recent, baseline = await asyncio.gather(
            self._trade_store.get_window(symbol, cfg.volume_spike_window_ms),
            self._trade_store.get_window(symbol, baseline_window_ms),
        )
        recent_volume = sum(t["qty"] for t in recent)
        baseline_volume = sum(t["qty"] for t in baseline)
        baseline_avg_per_window = baseline_volume / cfg.volume_spike_baseline_multiplier
        if baseline_avg_per_window <= 0:
            return None
        ratio = recent_volume / baseline_avg_per_window
        return ratio if ratio >= cfg.volume_spike_multiplier else None

    async def _detect_flow_persistence(self, symbol: str, direction: str) -> bool:
        cfg = self.config
        agree_count = 0
        checked = 0
        for window_ms in self._order_flow.windows_ms:
            metrics = await self._order_flow.get(symbol, window_ms)
            if not metrics:
                continue
            checked += 1
            if direction == "long" and metrics["buy_volume"] > metrics["sell_volume"]:
                agree_count += 1
            elif direction == "short" and metrics["sell_volume"] > metrics["buy_volume"]:
                agree_count += 1
        if checked == 0:
            return False
        return agree_count >= min(cfg.flow_persistence_updates_required, checked)

    async def _detect_momentum_acceleration(self, symbol: str, direction: str) -> bool:
        cfg = self.config
        window_ms = cfg.momentum_confirmation_window_ms * 2
        trades = await self._trade_store.get_window(symbol, window_ms)
        if len(trades) < 4:
            return False

        ordered = sorted(trades, key=lambda t: t["timestamp"])
        mid_ts = ordered[0]["timestamp"] + (ordered[-1]["timestamp"] - ordered[0]["timestamp"]) / 2.0
        first_half = [t for t in ordered if t["timestamp"] <= mid_ts]
        second_half = [t for t in ordered if t["timestamp"] > mid_ts]
        if len(first_half) < 2 or len(second_half) < 2:
            return False

        first_move = (
            (first_half[-1]["price"] - first_half[0]["price"]) / first_half[0]["price"] if first_half[0]["price"] else 0.0
        )
        second_move = (
            (second_half[-1]["price"] - second_half[0]["price"]) / second_half[0]["price"] if second_half[0]["price"] else 0.0
        )

        if direction == "long":
            return second_move > first_move > 0
        return second_move < first_move < 0
