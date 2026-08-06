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

log = logging.getLogger("okx_futures.position_store")

TABLE = "position_history"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class PositionHistoryStore:
    def __init__(self, supabase_client) -> None:
        self._sb = supabase_client

    async def get_open_rows(self) -> list:
        """Every row still marked status="open" — used at startup
        (execution_engine.py's reconcile_from_store) to check each one
        against what OKX actually shows, since a bot restart (crash,
        redeploy, manual stop) while a trade was live otherwise leaves it
        stuck open in the DB forever with nothing watching it."""
        try:
            result = await asyncio.to_thread(
                lambda: self._sb.table(TABLE).select("*").eq("status", "open").execute()
            )
        except Exception:
            log.exception("[position-store] failed to fetch open rows for startup reconciliation")
            return []
        return getattr(result, "data", None) or []

    async def count_all(self) -> int:
        """Total row count across the table's whole history (open +
        closed, all-time) — used to reseed the bot's lifetime trade
        counter at startup so max_total_trades is genuinely respected
        across restarts, instead of resetting to 0 every time the
        process restarts."""
        try:
            result = await asyncio.to_thread(
                lambda: self._sb.table(TABLE).select("id", count="exact").execute()
            )
        except Exception:
            log.exception("[position-store] failed to count rows for startup reconciliation")
            return 0
        return getattr(result, "count", None) or 0

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
            "stop_loss_price": position.stop_loss_price,
            "liq_price": position.liq_price,
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
        net_pnl: Optional[float] = None,
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
        # net_pnl is now written directly from OKX's own realizedPnl
        # figure (see execution_engine.py's _finalize_closed_position),
        # which correctly includes any liquidation penalty. IMPORTANT:
        # if `net_pnl` on `position_history` is currently a Postgres
        # GENERATED ALWAYS AS (realized_pnl - opening_fee - closing_fee)
        # STORED column, this write will be rejected (or silently ignored,
        # depending on driver) — that formula is what under-counted losses
        # on liquidated trades in the first place, since it has no way to
        # know about liqPenalty. Run something like:
        #   ALTER TABLE position_history ALTER COLUMN net_pnl DROP EXPRESSION;
        # (or drop + re-add as a plain nullable numeric column) so this
        # value actually lands.
        if net_pnl is not None:
            update["net_pnl"] = net_pnl
        try:
            await asyncio.to_thread(lambda: self._sb.table(TABLE).update(update).eq("id", row_id).execute())
        except Exception:
            log.exception(f"[position-store] failed to record close for row {row_id}")
