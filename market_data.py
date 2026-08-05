import asyncio
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple


DEFAULT_SYMBOL_WHITELIST = frozenset({
    "KAITO-USDT-SWAP",
})


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

        # 24h high/low, used only by the Priority-7 overextension filter.
        # Optional: feeds that omit these simply disable that filter rather
        # than defaulting to a value that could look like a false extreme.
        high_24h = low_24h = None
        try:
            high_raw = _first_present(payload, ("high_24h", "high24h", "high"))
            if high_raw is not None:
                high_24h = float(high_raw)
            low_raw = _first_present(payload, ("low_24h", "low24h", "low"))
            if low_raw is not None:
                low_24h = float(low_raw)
        except (TypeError, ValueError):
            high_24h = low_24h = None

        entry = {
            "last_price": last_price,
            "mark_price": mark_price,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": best_ask - best_bid,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "high_24h": high_24h,
            "low_24h": low_24h,
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


class PositionUpdateStore:
    """Live per-symbol position snapshots pushed by OKX's private
    'positions' websocket channel — a real-time push feed rather than the
    REST get_position() poll used elsewhere in this bot.

    Its main purpose is to be the authoritative, lowest-latency source
    for unrealized_pnl feeding the trailing profit-floor ratchet (see
    tp_tracker.py / execution_engine.py's _maybe_ratchet_stop_loss) —
    unlike a locally-computed price-move-times-notional estimate, `upl`
    here is OKX's own server-side calculation from mark price (which
    factors in index/funding basis, not just last-trade price), and it
    arrives the instant OKX recalculates it rather than on a polling
    cadence.

    Same field shape as OKXFuturesClient.get_position()'s per-row dict
    (current_amount/mark_price/unrealized_pnl/liquidation_price/
    avg_price/raw) so any code already handling that REST shape can
    handle this one identically."""

    def __init__(self) -> None:
        self._data: Dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def apply_update(self, payload: dict) -> None:
        symbol = payload.get("symbol")
        if not symbol:
            return
        async with self._lock:
            self._data[symbol] = payload

    async def get(self, symbol: str) -> Optional[dict]:
        async with self._lock:
            entry = self._data.get(symbol)
            return dict(entry) if entry is not None else None

    async def remove(self, symbol: str) -> None:
        async with self._lock:
            self._data.pop(symbol, None)


class OrderBookStore:
    def __init__(self, depth_levels: int = 20, history_window_ms: float = 2000.0, history_top_levels: int = 10) -> None:
        self._books: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        self._depth_levels = depth_levels
        self._lock = asyncio.Lock()
        # Rolling per-symbol history of recent snapshots (timestamp_ms, {"bids": [...], "asks": [...]})
        # so higher-level consumers (e.g. sweep detection) can compare book
        # state now vs a moment ago without each maintaining their own feed.
        self._history: Dict[str, Deque[Tuple[float, dict]]] = defaultdict(deque)
        self._history_window_ms = history_window_ms
        self._history_top_levels = history_top_levels
        self._last_update: Dict[str, float] = {}

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
        now_ms = time.time() * 1000.0

        async with self._lock:
            book = self._books.setdefault(symbol, {"bids": [], "asks": []})
            book[side] = levels
            self._last_update[symbol] = now_ms

            history = self._history[symbol]
            history.append((now_ms, {"bids": list(book["bids"][: self._history_top_levels]), "asks": list(book["asks"][: self._history_top_levels])}))
            cutoff = now_ms - self._history_window_ms
            while len(history) > 1 and history[0][0] < cutoff:
                history.popleft()

    async def get_book(self, symbol: str) -> Optional[dict]:
        async with self._lock:
            book = self._books.get(symbol)
            if not book:
                return None
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            last_update = self._last_update.get(symbol)

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
            "last_update": last_update,
        }

    async def get_book_history(self, symbol: str, window_ms: float) -> List[Tuple[float, dict]]:
        """Returns (timestamp_ms, {"bids": [...], "asks": [...]}) snapshots for
        `symbol` within the last `window_ms`, oldest first. Used by the event
        confirmation layer to detect order-book sweeps (levels disappearing
        between an earlier snapshot and now)."""
        now_ms = time.time() * 1000.0
        cutoff = now_ms - window_ms
        async with self._lock:
            history = list(self._history.get(symbol, ()))
        return [snap for snap in history if snap[0] >= cutoff]

    async def symbols(self) -> List[str]:
        async with self._lock:
            return list(self._books.keys())

    async def remove(self, symbol: str) -> None:
        async with self._lock:
            self._books.pop(symbol, None)
            self._history.pop(symbol, None)
            self._last_update.pop(symbol, None)


DEFAULT_RANKING_WEIGHTS = {
    "volume": 0.25,
    "turnover": 0.15,
    "liquidity": 0.20,
    "activity": 0.15,
    "movement": 0.15,
    "tightness": 0.10,
}


class SymbolRanker:
    def __init__(
        self,
        top_n: int = 15,
        stale_after_sec: float = 30.0,
        weights: Optional[Dict[str, float]] = None,
        symbol_whitelist: Optional[frozenset] = None,
    ) -> None:
        self._stats: Dict[str, Dict[str, float]] = {}
        self._lock = asyncio.Lock()
        self.top_n = top_n
        self.stale_after_sec = stale_after_sec
        self.weights = weights or DEFAULT_RANKING_WEIGHTS
        # None disables filtering; pass frozenset() explicitly (not None) if
        # you ever want "rank nothing" rather than "rank everything".
        self.symbol_whitelist = DEFAULT_SYMBOL_WHITELIST if symbol_whitelist is None else symbol_whitelist

    async def update_from_ticker(self, payload: dict) -> None:
        symbol = payload.get("symbol")
        if not symbol:
            return
        if self.symbol_whitelist and symbol not in self.symbol_whitelist:
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


_ISO_FRACTION_RE = re.compile(r"(\.\d{6})\d+")


def _parse_timestamp_ms(raw) -> float:
    if raw is None:
        return time.time() * 1000.0
    if isinstance(raw, (int, float)):
        ts = float(raw)
        return ts * 1000.0 if ts < 10 ** 12 else ts
    if isinstance(raw, str):
        text = raw.strip()
        try:
            ts = float(text)
            return ts * 1000.0 if ts < 10 ** 12 else ts
        except ValueError:
            pass
        try:
            iso = text.replace("Z", "+00:00")
            iso = _ISO_FRACTION_RE.sub(r"\1", iso)
            return datetime.fromisoformat(iso).timestamp() * 1000.0
        except (ValueError, TypeError):
            return time.time() * 1000.0
    return time.time() * 1000.0


class TradeStore:
    def __init__(self, windows_ms: Tuple[int, ...] = (500, 1000, 2000, 5000)) -> None:
        self._trades: Dict[str, Deque[dict]] = defaultdict(deque)
        self._windows_ms = windows_ms
        self._max_window_ms = max(windows_ms)
        self._lock = asyncio.Lock()
        self._unmapped_ways_seen: set = set()
        self.on_unmapped_way = lambda way, payload: None

    async def apply_trade(self, payload: dict) -> None:
        symbol = payload.get("symbol")
        if not symbol:
            return

        price_raw = _first_present(payload, ("deal_price", "price", "p"))
        qty_raw = _first_present(payload, ("deal_vol", "vol", "size", "qty", "v"))
        try:
            price = float(price_raw)
            qty = float(qty_raw)
        except (TypeError, ValueError):
            return

        way = _first_present(payload, ("way",))
        m_flag = payload.get("m")

        side = None
        if isinstance(way, str) and way.isdigit():
            way = int(way)
        if isinstance(way, int) and 1 <= way <= 4:
            side = "buy"
        elif isinstance(way, int) and 5 <= way <= 8:
            side = "sell"
        elif isinstance(m_flag, bool):
            # m=true: buyer is maker -> seller is taker -> sell.
            # m=false: seller is maker -> buyer is taker -> buy.
            side = "sell" if m_flag else "buy"
        else:
            generic_side = _first_present(payload, ("side",))
            if generic_side in (1, "1", "buy", "Buy", "BUY"):
                side = "buy"
            elif generic_side in (2, "2", "sell", "Sell", "SELL"):
                side = "sell"

        if side is None:
            if way not in self._unmapped_ways_seen:
                self._unmapped_ways_seen.add(way)
                self.on_unmapped_way(way, payload)
            return

        raw_ts = _first_present(payload, ("created_at", "ms_t", "timestamp", "time", "t"))
        ts_ms = _parse_timestamp_ms(raw_ts)

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

    @property
    def windows_ms(self) -> Tuple[int, ...]:
        return self._windows_ms


def compute_order_flow_metrics(
    trades: List[dict],
    window_sec: Optional[float] = None,
    whale_size_multiplier: float = 3.0,
) -> dict:
    buys = [t for t in trades if t["side"] == "buy"]
    sells = [t for t in trades if t["side"] == "sell"]

    buy_volume = sum(t["qty"] for t in buys)
    sell_volume = sum(t["qty"] for t in sells)
    total_volume = buy_volume + sell_volume
    total_trades = len(trades)

    if sell_volume > 0:
        buy_sell_ratio = buy_volume / sell_volume
    elif buy_volume > 0:
        buy_sell_ratio = float("inf")
    else:
        buy_sell_ratio = 0.0

    # --- Aggressive market order detection ---
    # Trades in this feed are already taker-side executions (see TradeStore.apply_trade),
    # so buy_volume/sell_volume ARE the aggressive/market-order volumes. These are just
    # explicit, purpose-named aliases plus the derived percentages/delta.
    aggressive_buy_pct = (buy_volume / total_volume) if total_volume > 0 else 0.0
    aggressive_sell_pct = (sell_volume / total_volume) if total_volume > 0 else 0.0

    # --- Whale detection: trades meaningfully larger than the window's average size ---
    avg_trade_qty = (total_volume / total_trades) if total_trades else 0.0
    whale_threshold = avg_trade_qty * whale_size_multiplier
    whale_buys = [t for t in buys if whale_threshold > 0 and t["qty"] >= whale_threshold]
    whale_sells = [t for t in sells if whale_threshold > 0 and t["qty"] >= whale_threshold]
    whale_buy_volume = sum(t["qty"] for t in whale_buys)
    whale_sell_volume = sum(t["qty"] for t in whale_sells)

    # --- Trade speed / intensity ---
    if window_sec and window_sec > 0:
        trades_per_sec = total_trades / window_sec
        volume_per_sec = total_volume / window_sec
        buy_trades_per_sec = len(buys) / window_sec
        sell_trades_per_sec = len(sells) / window_sec
    else:
        trades_per_sec = volume_per_sec = buy_trades_per_sec = sell_trades_per_sec = 0.0

    # --- Consecutive buy/sell streaks, in chronological order ---
    ordered = sorted(trades, key=lambda t: t["timestamp"])
    longest_buy_streak = longest_sell_streak = 0
    current_streak_side = None
    current_streak_len = 0
    for t in ordered:
        if t["side"] == current_streak_side:
            current_streak_len += 1
        else:
            current_streak_side = t["side"]
            current_streak_len = 1
        if current_streak_side == "buy":
            longest_buy_streak = max(longest_buy_streak, current_streak_len)
        else:
            longest_sell_streak = max(longest_sell_streak, current_streak_len)
    current_buy_streak = current_streak_len if current_streak_side == "buy" else 0
    current_sell_streak = current_streak_len if current_streak_side == "sell" else 0

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
        # Aggressive market order detection
        "market_buy_volume": buy_volume,
        "market_sell_volume": sell_volume,
        "aggressive_buy_pct": aggressive_buy_pct,
        "aggressive_sell_pct": aggressive_sell_pct,
        "net_volume_delta": buy_volume - sell_volume,
        # Whale detection
        "whale_buy_count": len(whale_buys),
        "whale_sell_count": len(whale_sells),
        "whale_buy_volume": whale_buy_volume,
        "whale_sell_volume": whale_sell_volume,
        "whale_trade_count": len(whale_buys) + len(whale_sells),
        # Trade speed / intensity
        "trades_per_sec": trades_per_sec,
        "volume_per_sec": volume_per_sec,
        "buy_trades_per_sec": buy_trades_per_sec,
        "sell_trades_per_sec": sell_trades_per_sec,
        # Consecutive streaks
        "current_buy_streak": current_buy_streak,
        "current_sell_streak": current_sell_streak,
        "longest_buy_streak": longest_buy_streak,
        "longest_sell_streak": longest_sell_streak,
    }


class OrderFlowAnalyzer:
    def __init__(
        self,
        trade_store: TradeStore,
        windows_ms: Tuple[int, ...] = (500, 1000, 2000, 5000),
        whale_size_multiplier: float = 3.0,
    ) -> None:
        self._trade_store = trade_store
        self._windows_ms = windows_ms
        self._whale_size_multiplier = whale_size_multiplier
        self._metrics: Dict[str, Dict[int, dict]] = {}
        self._lock = asyncio.Lock()

    async def recompute(self, symbols: List[str]) -> None:
        computed: Dict[str, Dict[int, dict]] = {}
        for symbol in symbols:
            per_window = {}
            for window_ms in self._windows_ms:
                trades = await self._trade_store.get_window(symbol, window_ms)
                per_window[window_ms] = compute_order_flow_metrics(
                    trades,
                    window_sec=window_ms / 1000.0,
                    whale_size_multiplier=self._whale_size_multiplier,
                )
            computed[symbol] = per_window

        async with self._lock:
            self._metrics = computed

    async def get(self, symbol: str, window_ms: int) -> Optional[dict]:
        async with self._lock:
            per_window = self._metrics.get(symbol)
            return dict(per_window[window_ms]) if per_window and window_ms in per_window else None

    async def get_recent_trades(self, symbol: str, window_ms: int) -> List[dict]:
        """Raw trades backing the metrics above — used by the confidence
        engine's VWAP indicator (Priority 6) and time-decay calc (Priority 2),
        neither of which can be derived from the aggregated metrics dict."""
        return await self._trade_store.get_window(symbol, window_ms)

    async def snapshot(self) -> Dict[str, Dict[int, dict]]:
        async with self._lock:
            return {symbol: {w: dict(m) for w, m in windows.items()} for symbol, windows in self._metrics.items()}

    @property
    def windows_ms(self) -> Tuple[int, ...]:
        return self._windows_ms


def compute_liquidity_metrics(book: dict, distance_bands: Optional[Tuple[float, ...]] = None) -> dict:
    bid_liquidity = book.get("bid_liquidity", 0.0)
    ask_liquidity = book.get("ask_liquidity", 0.0)
    total_liquidity = bid_liquidity + ask_liquidity

    if ask_liquidity > 0:
        bid_ask_ratio = bid_liquidity / ask_liquidity
    elif bid_liquidity > 0:
        bid_ask_ratio = float("inf")
    else:
        bid_ask_ratio = 0.0

    imbalance = ((bid_liquidity - ask_liquidity) / total_liquidity) if total_liquidity > 0 else 0.0

    best_bid = book.get("best_bid")
    best_ask = book.get("best_ask")
    best_bid_size = best_bid[1] if best_bid else 0.0
    best_ask_size = best_ask[1] if best_ask else 0.0

    if imbalance > 0:
        dominance = "buyers"
    elif imbalance < 0:
        dominance = "sellers"
    else:
        dominance = "balanced"

    result = {
        "bid_liquidity": bid_liquidity,
        "ask_liquidity": ask_liquidity,
        "bid_ask_ratio": bid_ask_ratio,
        "imbalance": imbalance,
        "best_bid_size": best_bid_size,
        "best_ask_size": best_ask_size,
        "dominance": dominance,
    }

    if distance_bands:
        best_bid = book.get("best_bid")
        best_ask = book.get("best_ask")
        mid_price = ((best_bid[0] + best_ask[0]) / 2.0) if best_bid and best_ask else 0.0
        result["band_imbalance"] = (
            compute_band_imbalance(book.get("bids", []), book.get("asks", []), mid_price, distance_bands)
            if mid_price > 0
            else {}
        )

    return result


def compute_band_imbalance(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    mid_price: float,
    distance_bands: Tuple[float, ...],
) -> Dict[float, dict]:
    """Order book liquidity/imbalance within % distance bands from mid price.

    Nearby liquidity (tight bands) matters more for short-term price impact than
    liquidity sitting far from the current price, so this breaks the book down
    by proximity instead of summing it all together.
    """
    bands: Dict[float, dict] = {}
    for pct in distance_bands:
        band_bid = sum(v for p, v in bids if mid_price > 0 and (mid_price - p) / mid_price <= pct)
        band_ask = sum(v for p, v in asks if mid_price > 0 and (p - mid_price) / mid_price <= pct)
        total = band_bid + band_ask

        if band_ask > 0:
            bid_ask_ratio = band_bid / band_ask
        elif band_bid > 0:
            bid_ask_ratio = float("inf")
        else:
            bid_ask_ratio = 0.0

        if band_bid > 0:
            ask_bid_ratio = band_ask / band_bid
        elif band_ask > 0:
            ask_bid_ratio = float("inf")
        else:
            ask_bid_ratio = 0.0

        imbalance = ((band_bid - band_ask) / total) if total > 0 else 0.0

        bands[pct] = {
            "bid_liquidity": band_bid,
            "ask_liquidity": band_ask,
            "bid_ask_ratio": bid_ask_ratio,
            "ask_bid_ratio": ask_bid_ratio,
            "imbalance": imbalance,
        }
    return bands


class LiquidityEngine:
    def __init__(self, order_book: OrderBookStore) -> None:
        self._order_book = order_book

    async def get(self, symbol: str, distance_bands: Optional[Tuple[float, ...]] = None) -> Optional[dict]:
        book = await self._order_book.get_book(symbol)
        if not book:
            return None
        return compute_liquidity_metrics(book, distance_bands=distance_bands)

    async def snapshot(self, symbols: List[str]) -> Dict[str, dict]:
        result: Dict[str, dict] = {}
        for symbol in symbols:
            metrics = await self.get(symbol)
            if metrics is not None:
                result[symbol] = metrics
        return result


@dataclass
class Signal:
    symbol: str
    direction: str
    confidence: float
    entry_price: float
    take_profit: float
    stop_loss: float
    timestamp: float
    reasons: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)