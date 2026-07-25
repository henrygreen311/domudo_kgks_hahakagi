import asyncio
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple


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

    async def remove(self, symbol: str) -> None:
        async with self._lock:
            self._books.pop(symbol, None)


DEFAULT_RANKING_WEIGHTS = {
    "volume": 0.25,
    "turnover": 0.15,
    "liquidity": 0.20,
    "activity": 0.15,
    "movement": 0.15,
    "tightness": 0.10,
}


class SymbolRanker:
    def __init__(self, top_n: int = 15, stale_after_sec: float = 30.0, weights: Optional[Dict[str, float]] = None) -> None:
        self._stats: Dict[str, Dict[str, float]] = {}
        self._lock = asyncio.Lock()
        self.top_n = top_n
        self.stale_after_sec = stale_after_sec
        self.weights = weights or DEFAULT_RANKING_WEIGHTS

    async def update_from_ticker(self, payload: dict) -> None:
        symbol = payload.get("symbol")
        if not symbol:
            return
        try:
            last_price = float(payload.get("last_price"))
            volume_24 = float(payload.get("volume_24"))
            bid_price = float(payload.get("bid_price"))
            ask_price = float(payload.get("ask_price"))
            bid_volume = float(payload.get("bid_vol"))
            ask_volume = float(payload.get("ask_vol"))
            pct_change = float(payload.get("range", 0.0))
        except (TypeError, ValueError):
            return

        spread = ask_price - bid_price
        spread_pct = (spread / last_price) if last_price else 0.0
        turnover = last_price * volume_24
        now = time.time()

        async with self._lock:
            stat = self._stats.setdefault(symbol, {"update_count": 0})
            stat["last_price"] = last_price
            stat["volume_24"] = volume_24
            stat["turnover"] = turnover
            stat["bid_volume"] = bid_volume
            stat["ask_volume"] = ask_volume
            stat["spread"] = spread
            stat["spread_pct"] = spread_pct
            stat["pct_change"] = pct_change
            stat["last_update"] = now
            stat["update_count"] += 1

    async def rank(self) -> List[Tuple[str, float]]:
        async with self._lock:
            snapshot = {symbol: dict(stat) for symbol, stat in self._stats.items()}

        now = time.time()
        candidates = [(s, st) for s, st in snapshot.items() if now - st["last_update"] <= self.stale_after_sec]
        if not candidates:
            return []

        def normalize(values: List[float]) -> List[float]:
            lo, hi = min(values), max(values)
            span = hi - lo
            return [((v - lo) / span) if span > 0 else 0.5 for v in values]

        volumes = normalize([st["volume_24"] for _, st in candidates])
        turnovers = normalize([st["turnover"] for _, st in candidates])
        liquidity = normalize([st["bid_volume"] + st["ask_volume"] for _, st in candidates])
        activity = normalize([st["update_count"] for _, st in candidates])
        movement = normalize([abs(st["pct_change"]) for _, st in candidates])
        tightness = [1.0 - v for v in normalize([st["spread_pct"] for _, st in candidates])]

        w = self.weights
        scored = []
        for i, (symbol, _) in enumerate(candidates):
            score = (
                w.get("volume", 0.0) * volumes[i]
                + w.get("turnover", 0.0) * turnovers[i]
                + w.get("liquidity", 0.0) * liquidity[i]
                + w.get("activity", 0.0) * activity[i]
                + w.get("movement", 0.0) * movement[i]
                + w.get("tightness", 0.0) * tightness[i]
            )
            scored.append((symbol, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        async with self._lock:
            for stat in self._stats.values():
                stat["update_count"] = 0

        return scored

    async def top_symbols(self) -> List[str]:
        ranked = await self.rank()
        return [symbol for symbol, _ in ranked[: self.top_n]]


def _first_present(payload: dict, keys: Tuple[str, ...]):
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


class TradeStore:
    def __init__(self, windows_ms: Tuple[int, ...] = (500, 1000, 2000, 5000)) -> None:
        self._trades: Dict[str, Deque[dict]] = defaultdict(deque)
        self._windows_ms = windows_ms
        self._max_window_ms = max(windows_ms)
        self._lock = asyncio.Lock()

    async def apply_trade(self, payload: dict) -> None:
        symbol = payload.get("symbol")
        if not symbol:
            return

        price_raw = _first_present(payload, ("price", "p"))
        qty_raw = _first_present(payload, ("vol", "size", "qty", "v"))
        try:
            price = float(price_raw)
            qty = float(qty_raw)
        except (TypeError, ValueError):
            return

        way = _first_present(payload, ("way", "side"))
        if way in (1, "1", "buy", "Buy", "BUY"):
            side = "buy"
        elif way in (2, "2", "sell", "Sell", "SELL"):
            side = "sell"
        else:
            return

        raw_ts = _first_present(payload, ("ms_t", "timestamp", "time", "t"))
        try:
            ts_ms = float(raw_ts)
            if ts_ms < 10 ** 12:
                ts_ms *= 1000.0
        except (TypeError, ValueError):
            ts_ms = time.time() * 1000.0

        trade = {"timestamp": ts_ms, "price": price, "qty": qty, "side": side}

        async with self._lock:
            dq = self._trades[symbol]
            dq.append(trade)
            cutoff = ts_ms - self._max_window_ms
            while dq and dq[0]["timestamp"] < cutoff:
                dq.popleft()

    async def get_window(self, symbol: str, window_ms: int) -> List[dict]:
        now_ms = time.time() * 1000.0
        async with self._lock:
            trades = list(self._trades.get(symbol, ()))
        cutoff = now_ms - window_ms
        return [t for t in trades if t["timestamp"] >= cutoff]

    async def symbols(self) -> List[str]:
        async with self._lock:
            return list(self._trades.keys())

    async def remove(self, symbol: str) -> None:
        async with self._lock:
            self._trades.pop(symbol, None)


def compute_order_flow_metrics(trades: List[dict]) -> dict:
    buys = [t for t in trades if t["side"] == "buy"]
    sells = [t for t in trades if t["side"] == "sell"]

    buy_volume = sum(t["qty"] for t in buys)
    sell_volume = sum(t["qty"] for t in sells)

    if sell_volume > 0:
        buy_sell_ratio = buy_volume / sell_volume
    elif buy_volume > 0:
        buy_sell_ratio = float("inf")
    else:
        buy_sell_ratio = 0.0

    return {
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "buy_sell_ratio": buy_sell_ratio,
        "delta": buy_volume - sell_volume,
        "buy_trade_count": len(buys),
        "sell_trade_count": len(sells),
        "avg_buy_size": (buy_volume / len(buys)) if buys else 0.0,
        "avg_sell_size": (sell_volume / len(sells)) if sells else 0.0,
        "largest_buy": max((t["qty"] for t in buys), default=0.0),
        "largest_sell": max((t["qty"] for t in sells), default=0.0),
    }


class OrderFlowAnalyzer:
    def __init__(self, trade_store: TradeStore, windows_ms: Tuple[int, ...] = (500, 1000, 2000, 5000)) -> None:
        self._trade_store = trade_store
        self._windows_ms = windows_ms
        self._metrics: Dict[str, Dict[int, dict]] = {}
        self._lock = asyncio.Lock()

    async def recompute(self, symbols: List[str]) -> None:
        computed: Dict[str, Dict[int, dict]] = {}
        for symbol in symbols:
            per_window = {}
            for window_ms in self._windows_ms:
                trades = await self._trade_store.get_window(symbol, window_ms)
                per_window[window_ms] = compute_order_flow_metrics(trades)
            computed[symbol] = per_window

        async with self._lock:
            self._metrics = computed

    async def get(self, symbol: str, window_ms: int) -> Optional[dict]:
        async with self._lock:
            per_window = self._metrics.get(symbol)
            return dict(per_window[window_ms]) if per_window and window_ms in per_window else None

    async def snapshot(self) -> Dict[str, Dict[int, dict]]:
        async with self._lock:
            return {symbol: {w: dict(m) for w, m in windows.items()} for symbol, windows in self._metrics.items()}
