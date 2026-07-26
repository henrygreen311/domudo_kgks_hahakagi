"""
Repository for the `position_history` table (see position_history.sql).

Every trade the bot opens gets one row, inserted right after its
take-profit is placed, and updated in place once the exchange reports the
position closed — so `position_history` always holds the full open+close
lifecycle of every trade for later manual analysis.

The Supabase Python client is synchronous, so calls are pushed through
`asyncio.to_thread` to keep the event loop unblocked.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("bitmart_futures.position_store")

TABLE = "position_history"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class PositionHistoryStore:
    def __init__(self, supabase_client) -> None:
        self._sb = supabase_client

    async def record_open(self, position) -> Optional[int]:
        """Inserts a row for a freshly opened position and returns its id,
        which is later passed to `record_close`."""
        row = {
            "symbol": position.symbol,
            "direction": position.direction,
            "order_id": position.order_id,
            "tp_order_id": position.tp_order_id,
            "leverage": position.leverage,
            "margin_usdt": position.margin_usdt,
            "contract_size": position.contract_size,
            "size_contracts": position.size_contracts,
            "entry_price": position.entry_price,
            "take_profit_price": position.take_profit_price,
            "opening_fee": position.opening_fee,
            "opened_at": _iso(position.opened_at),
            "status": "open",
        }
        try:
            result = await asyncio.to_thread(lambda: self._sb.table(TABLE).insert(row).execute())
        except Exception:
            log.exception(f"[position-store] failed to record open for {position.symbol}")
            return None

        data = getattr(result, "data", None) or []
        if not data:
            log.warning(f"[position-store] insert for {position.symbol} returned no row id")
            return None
        return data[0].get("id")

    async def record_close(
        self,
        row_id: Optional[int],
        exit_price: Optional[float],
        realized_pnl: Optional[float],
        closing_fee: Optional[float],
        close_reason: str,
        closed_at: float,
    ) -> None:
        if row_id is None:
            log.warning("[position-store] skipping close update — no row id (insert must have failed earlier)")
            return
        update = {
            "exit_price": exit_price,
            "realized_pnl": realized_pnl,
            "closing_fee": closing_fee,
            "close_reason": close_reason,
            "closed_at": _iso(closed_at),
            "status": "closed",
        }
        try:
            await asyncio.to_thread(lambda: self._sb.table(TABLE).update(update).eq("id", row_id).execute())
        except Exception:
            log.exception(f"[position-store] failed to record close for row {row_id}")
