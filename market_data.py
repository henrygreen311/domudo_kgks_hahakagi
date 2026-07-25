import asyncio
import time
from typing import Dict, List, Optional, Tuple


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


class OrderBookStore:
    def __init__(self, depth_levels: int = 20) -> None:
        self._books: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        self._depth_levels = depth_levels
        self._lock = asyncio.Lock()

    async def apply_depth_update(self, payload: dict) -> None:
        symbol = payload.get("symbol")
        way = payload.get("way")
        depths = payload.get("depths")
        if not symbol or way not in (1, 2) or not isinstance(depths, list):
            return

        side = "bids" if way == 1 else "asks"
        levels: List[Tuple[float, float]] = []
        for level in depths:
            try:
                price = float(level.get("price"))
                vol = float(level.get("vol"))
            except (TypeError, ValueError, AttributeError):
                continue
            levels.append((price, vol))

        levels.sort(key=lambda x: x[0], reverse=(side == "bids"))

        async with self._lock:
            book = self._books.setdefault(symbol, {"bids": [], "asks": []})
            book[side] = levels

    async def get_book(self, symbol: str) -> Optional[dict]:
        async with self._lock:
            book = self._books.get(symbol)
            if not book:
                return None
            bids = book.get("bids", [])
            asks = book.get("asks", [])

        best_bid = bids[0] if bids else None
        best_ask = asks[0] if asks else None
        bid_liquidity = sum(v for _, v in bids)
        ask_liquidity = sum(v for _, v in asks)
        spread = (best_ask[0] - best_bid[0]) if best_bid and best_ask else None

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bids": bids[: self._depth_levels],
            "asks": asks[: self._depth_levels],
            "bid_liquidity": bid_liquidity,
            "ask_liquidity": ask_liquidity,
            "spread": spread,
        }

    async def symbols(self) -> List[str]:
        async with self._lock:
            return list(self._books.keys())
