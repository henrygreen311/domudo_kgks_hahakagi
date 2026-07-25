import asyncio
import time
from typing import Dict, Optional


class MarketDataStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, float]] = {}
        self._lock = asyncio.Lock()

    async def update_from_ticker(self, payload: dict) -> None:
        symbol = payload.get("symbol")
        if not symbol:
            return
        try:
            last_price = float(payload.get("last_price"))
            best_bid = float(payload.get("bid_price"))
            best_ask = float(payload.get("ask_price"))
            bid_volume = float(payload.get("bid_vol"))
            ask_volume = float(payload.get("ask_vol"))
            mark_price_raw = payload.get("mark_price", payload.get("fair_price", last_price))
            mark_price = float(mark_price_raw)
        except (TypeError, ValueError):
            return

        entry = {
            "last_price": last_price,
            "mark_price": mark_price,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": best_ask - best_bid,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "last_update": time.time(),
        }

        async with self._lock:
            self._data[symbol] = entry

    async def get(self, symbol: str) -> Optional[Dict[str, float]]:
        async with self._lock:
            entry = self._data.get(symbol)
            return dict(entry) if entry is not None else None

    async def snapshot(self) -> Dict[str, Dict[str, float]]:
        async with self._lock:
            return {symbol: dict(entry) for symbol, entry in self._data.items()}

    async def symbol_count(self) -> int:
        async with self._lock:
            return len(self._data)
