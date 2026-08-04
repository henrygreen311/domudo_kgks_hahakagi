"""
tp_tracker.py — Trailing Profit-Floor Tracker.

Why this exists: the stop-loss (target_stop_loss_usdt) was significantly
bigger than the take-profit (target_net_profit_usdt), so as few as 2-5
losing trades could erase an entire winning trade's profit — confirmed
from the trade snapshot history. Raising target_net_profit_usdt (now
0.85) fixes that ratio, but holding out for the full 0.85 on every trade
means giving back most or all of a big open gain on a sudden reversal:
several trades in that same history peaked well above what the old
(much smaller) target used to capture, but nowhere near 0.85 — a flat
0.85 target alone would have ridden many of them further back down
before exiting.

This module's one job: given a position's PEAK unrealized profit reached
so far, decide the current minimum acceptable exit profit (the "floor").
It holds no exchange connection, knows nothing about OKX, and places no
orders — it's a pure, per-symbol decision tracker that execution_engine.py
calls every monitoring tick and acts on.

IMPORTANT technical note despite the module's name: this floor is
enforced by moving the position's STOP-LOSS trigger up (from below entry,
to above entry, and progressively higher), not by moving the take-profit
trigger. The take-profit leg stays fixed at the original final target the
whole time — that's what "exit once price reaches 0.85" already refers
to, and the exchange enforces it directly with no help needed here. What
actually needed inventing is the protective side: once there's real
profit on the table, the stop should follow it up so a reversal exits at
a locked-in gain instead of riding back toward breakeven or the original
loss-side stop. Mechanically that's a trailing stop, just expressed here
in USDT-profit terms (matching how the rest of this bot already thinks
in USDT targets) rather than in price percent.

Mechanics:
  - Below `activation_profit_usdt` (0.20 by default) peak profit, no
    floor is active; the trade is governed only by the original
    stop-loss set at open.
  - Once peak profit reaches `activation_profit_usdt`, a floor activates
    at (peak - trail_lag_usdt). It only ever moves UP as a new peak is
    set, never down — a pullback that doesn't set a new peak leaves the
    floor exactly where it was:

      peak=0.20 -> floor=0.10
      peak=0.25 -> floor=0.15
      peak=0.30 -> floor=0.20
      ... (peak - 0.10, continuously, once peak >= 0.20)

  - The floor is capped at `final_take_profit_usdt` (0.85 by default).
    Once peak profit itself reaches the final target, the original fixed
    take-profit order fires on its own — this tracker doesn't need to
    (and won't) push the floor any higher than that.

Units note: this is fed the exchange's live, gross (pre-fee) unrealized
PnL each tick — e.g. OKX's own `upl` field — so `activation_profit_usdt`
and `trail_lag_usdt` are being compared directly against that gross
figure for simplicity and responsiveness, matching how these numbers were
originally specified ("price is at 0.20"). The actual stop-loss price
placed on the exchange still correctly nets out estimated fees, via
execution_engine.py's own fee-aware price math — only the peak-tracking
threshold itself is gross. At this position size the difference is a few
cents, not worth the complexity of reconciling both conventions exactly.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TPTrackerConfig:
    activation_profit_usdt: float = 0.20
    trail_lag_usdt: float = 0.10
    final_take_profit_usdt: float = 0.85


@dataclass
class TPDecision:
    """What execution_engine.py should do this tick."""

    ratchet: bool  # True -> a NEW, higher floor is ready; caller should move the stop-loss to floor_usdt
    peak_profit_usdt: float
    floor_usdt: Optional[float]  # None until activation_profit_usdt has been reached at least once


@dataclass
class _SymbolState:
    peak_profit_usdt: float = float("-inf")
    committed_floor_usdt: Optional[float] = None  # the floor last actually acted on by the caller


class TPTracker:
    """Tracks one _SymbolState per symbol with an open position and turns
    each tick's live unrealized PnL into a ratchet/hold decision. Fully
    async + locked, matching the style of the other per-symbol trackers
    in this bot (ObservationWindowManager, MovementTracker)."""

    def __init__(self, config: Optional[TPTrackerConfig] = None) -> None:
        self.config = config or TPTrackerConfig()
        self._state: Dict[str, _SymbolState] = {}
        self._lock = asyncio.Lock()

    async def start_tracking(self, symbol: str) -> None:
        """Call once, right when a position opens, so this symbol starts
        from a clean peak/floor rather than carrying over state from a
        previous trade on the same symbol."""
        async with self._lock:
            self._state[symbol] = _SymbolState()

    async def stop_tracking(self, symbol: str) -> None:
        """Call once the position closes, for any reason."""
        async with self._lock:
            self._state.pop(symbol, None)

    async def update(self, symbol: str, unrealized_pnl_usdt: float) -> TPDecision:
        """Call once per monitoring tick for an open position with its
        current gross unrealized PnL. Updates the running peak and
        returns whether a new, higher floor is ready to be acted on.

        Safe to call even if start_tracking wasn't (falls back to
        creating fresh state), so a missed/late call never raises."""
        cfg = self.config
        async with self._lock:
            state = self._state.get(symbol)
            if state is None:
                state = _SymbolState()
                self._state[symbol] = state

            if unrealized_pnl_usdt > state.peak_profit_usdt:
                state.peak_profit_usdt = unrealized_pnl_usdt

            peak = state.peak_profit_usdt
            if peak < cfg.activation_profit_usdt:
                return TPDecision(ratchet=False, peak_profit_usdt=peak, floor_usdt=state.committed_floor_usdt)

            floor = min(peak - cfg.trail_lag_usdt, cfg.final_take_profit_usdt)

            if state.committed_floor_usdt is not None and floor <= state.committed_floor_usdt:
                # No new ground gained since the last ratchet -- hold.
                return TPDecision(ratchet=False, peak_profit_usdt=peak, floor_usdt=state.committed_floor_usdt)

            state.committed_floor_usdt = floor
            return TPDecision(ratchet=True, peak_profit_usdt=peak, floor_usdt=floor)
