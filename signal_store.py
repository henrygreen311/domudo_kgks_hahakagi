"""Repository for the `signals_histories` table (see
create_signals_histories.sql). Captures the full evidence-pipeline
breakdown (see evidence_pipeline.EvidenceSummary.to_signal_record) at the
moment a trade opens, linked to position_history via trade_id — the data
behind questions like "why do longs underperform shorts?" without having
to grep job logs by hand.

Mirrors PositionHistoryStore/TradeSnapshotStore's pattern: the Supabase
client is synchronous, so every call is pushed through asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

log = logging.getLogger(__name__)

TABLE = "signals_histories"


class SignalHistoryStore:
    def __init__(self, supabase_client) -> None:
        self._sb = supabase_client

    async def record_signal(self, record: dict) -> Optional[int]:
        """Inserts one row (built by the caller from
        EvidenceSummary.to_signal_record() plus trade_id/entry_price/
        price_at_decision/evaluated_at). Returns the new row's id, or
        None if the insert failed — logged, never raised, so a
        signal-history write can never take down trading itself (same
        contract as PositionHistoryStore.record_open/record_close and
        TradeSnapshotStore's methods)."""
        try:
            result = await asyncio.to_thread(lambda: self._sb.table(TABLE).insert(record).execute())
        except Exception:
            log.exception(
                f"[signal-store] failed to record signal history for "
                f"{record.get('symbol')} (trade_id={record.get('trade_id')})"
            )
            return None
        data = getattr(result, "data", None) or []
        return data[0].get("id") if data else None
