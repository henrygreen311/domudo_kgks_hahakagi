import asyncio
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
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


def compute_liquidity_metrics(book: dict) -> dict:
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

    return {
        "bid_liquidity": bid_liquidity,
        "ask_liquidity": ask_liquidity,
        "bid_ask_ratio": bid_ask_ratio,
        "imbalance": imbalance,
        "best_bid_size": best_bid_size,
        "best_ask_size": best_ask_size,
        "dominance": dominance,
    }


class LiquidityEngine:
    def __init__(self, order_book: OrderBookStore) -> None:
        self._order_book = order_book

    async def get(self, symbol: str) -> Optional[dict]:
        book = await self._order_book.get_book(symbol)
        if not book:
            return None
        return compute_liquidity_metrics(book)

    async def snapshot(self, symbols: List[str]) -> Dict[str, dict]:
        result: Dict[str, dict] = {}
        for symbol in symbols:
            metrics = await self.get(symbol)
            if metrics is not None:
                result[symbol] = metrics
        return result


@dataclass
class SignalConfig:
    volume_ratio_threshold: float = 1.5
    max_spread_pct: float = 0.001
    price_trend_window_sec: float = 30.0
    min_price_move_pct: float = 0.0005
    max_data_age_sec: float = 5.0
    min_liquidity: float = 0.0
    order_flow_window_ms: int = 1000
    min_confirmations: int = 5
    take_profit_pct: float = 0.002
    stop_loss_pct: float = 0.005
    cooldown_sec: float = 5.0


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


class SignalGenerator:
    def __init__(
        self,
        market_data: MarketDataStore,
        order_flow: OrderFlowAnalyzer,
        liquidity_engine: LiquidityEngine,
        config: Optional[SignalConfig] = None,
    ) -> None:
        self._market_data = market_data
        self._order_flow = order_flow
        self._liquidity_engine = liquidity_engine
        self.config = config or SignalConfig()
        self._price_history: Dict[str, Deque[Tuple[float, float]]] = defaultdict(deque)
        self._last_signal_time: Dict[str, float] = {}

    def _update_price_history(self, symbol: str, price: float, now: float) -> None:
        history = self._price_history[symbol]
        history.append((now, price))
        cutoff = now - self.config.price_trend_window_sec
        while history and history[0][0] < cutoff:
            history.popleft()

    async def evaluate(self, symbol: str) -> Optional[Signal]:
        cfg = self.config
        now = time.time()

        last_signal_at = self._last_signal_time.get(symbol)
        if last_signal_at is not None and (now - last_signal_at) < cfg.cooldown_sec:
            return None

        market = await self._market_data.get(symbol)
        if not market:
            return None

        price = market["last_price"]
        data_age = now - market["last_update"]
        is_fresh = data_age <= cfg.max_data_age_sec
        spread_pct = (market["spread"] / price) if price else None

        self._update_price_history(symbol, price, now)
        history = self._price_history[symbol]
        price_change_pct = None
        if len(history) >= 2 and history[0][1]:
            price_change_pct = (price - history[0][1]) / history[0][1]

        flow = await self._order_flow.get(symbol, cfg.order_flow_window_ms)
        liquidity = await self._liquidity_engine.get(symbol)
        if flow is None or liquidity is None:
            return None

        liquidity_sufficient = liquidity["bid_liquidity"] >= cfg.min_liquidity and liquidity["ask_liquidity"] >= cfg.min_liquidity
        spread_tight = spread_pct is not None and spread_pct <= cfg.max_spread_pct

        buy_dominant = (flow["sell_volume"] == 0 and flow["buy_volume"] > 0) or (
            flow["sell_volume"] > 0 and flow["buy_volume"] / flow["sell_volume"] >= cfg.volume_ratio_threshold
        )
        sell_dominant = (flow["buy_volume"] == 0 and flow["sell_volume"] > 0) or (
            flow["buy_volume"] > 0 and flow["sell_volume"] / flow["buy_volume"] >= cfg.volume_ratio_threshold
        )

        long_checks = {
            "buy volume exceeds sell volume": buy_dominant,
            "bid liquidity exceeds ask liquidity": liquidity["bid_liquidity"] > liquidity["ask_liquidity"],
            "spread below threshold": spread_tight,
            "price moving upward": price_change_pct is not None and price_change_pct >= cfg.min_price_move_pct,
            "market data fresh": is_fresh,
            "liquidity sufficient": liquidity_sufficient,
        }
        short_checks = {
            "sell volume exceeds buy volume": sell_dominant,
            "ask liquidity exceeds bid liquidity": liquidity["ask_liquidity"] > liquidity["bid_liquidity"],
            "spread below threshold": spread_tight,
            "price moving downward": price_change_pct is not None and price_change_pct <= -cfg.min_price_move_pct,
            "market data fresh": is_fresh,
            "liquidity sufficient": liquidity_sufficient,
        }

        long_count = sum(long_checks.values())
        short_count = sum(short_checks.values())

        if long_count >= cfg.min_confirmations and long_count > short_count:
            direction = "long"
            checks = long_checks
            confirmations = long_count
        elif short_count >= cfg.min_confirmations and short_count > long_count:
            direction = "short"
            checks = short_checks
            confirmations = short_count
        else:
            return None

        confidence = confirmations / len(checks)
        reasons = [name for name, passed in checks.items() if passed]

        if direction == "long":
            take_profit = price * (1 + cfg.take_profit_pct)
            stop_loss = price * (1 - cfg.stop_loss_pct)
        else:
            take_profit = price * (1 - cfg.take_profit_pct)
            stop_loss = price * (1 + cfg.stop_loss_pct)

        self._last_signal_time[symbol] = now

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            timestamp=now,
            reasons=reasons,
        )


@dataclass
class PaperTrade:
    symbol: str
    direction: str
    entry_price: float
    entry_time: float
    take_profit: float
    stop_loss: float
    mfe_pct: float = 0.0
    mae_pct: float = 0.0


@dataclass
class PaperTradeResult:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    outcome: str
    mfe_pct: float
    mae_pct: float
    duration_sec: float
    gross_profit_pct: float
    fees_pct: float
    net_profit_pct: float


class PaperTradingEngine:
    def __init__(self, taker_fee_rate: float = 0.0006, max_trade_duration_sec: float = 60.0) -> None:
        self._open_trades: Dict[str, PaperTrade] = {}
        self._results: List[PaperTradeResult] = []
        self._taker_fee_rate = taker_fee_rate
        self._max_duration = max_trade_duration_sec
        self._lock = asyncio.Lock()

    async def has_open_trade(self, symbol: str) -> bool:
        async with self._lock:
            return symbol in self._open_trades

    async def open_trade(self, signal: Signal) -> bool:
        async with self._lock:
            if signal.symbol in self._open_trades:
                return False
            self._open_trades[signal.symbol] = PaperTrade(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                entry_time=signal.timestamp,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
            )
            return True

    async def update(self, symbol: str, current_price: float, now: float) -> Optional[PaperTradeResult]:
        async with self._lock:
            trade = self._open_trades.get(symbol)
            if not trade:
                return None

            if trade.direction == "long":
                move_pct = (current_price - trade.entry_price) / trade.entry_price
                hit_tp = current_price >= trade.take_profit
                hit_sl = current_price <= trade.stop_loss
            else:
                move_pct = (trade.entry_price - current_price) / trade.entry_price
                hit_tp = current_price <= trade.take_profit
                hit_sl = current_price >= trade.stop_loss

            trade.mfe_pct = max(trade.mfe_pct, move_pct)
            trade.mae_pct = min(trade.mae_pct, move_pct)

            timed_out = (now - trade.entry_time) >= self._max_duration
            if not (hit_tp or hit_sl or timed_out):
                return None

            outcome = "tp" if hit_tp else "sl" if hit_sl else "timeout"
            fees_pct = self._taker_fee_rate * 2
            net_profit_pct = move_pct - fees_pct

            result = PaperTradeResult(
                symbol=symbol,
                direction=trade.direction,
                entry_price=trade.entry_price,
                exit_price=current_price,
                entry_time=trade.entry_time,
                exit_time=now,
                outcome=outcome,
                mfe_pct=trade.mfe_pct,
                mae_pct=trade.mae_pct,
                duration_sec=now - trade.entry_time,
                gross_profit_pct=move_pct,
                fees_pct=fees_pct,
                net_profit_pct=net_profit_pct,
            )

            del self._open_trades[symbol]
            self._results.append(result)
            return result

    async def stats(self) -> dict:
        async with self._lock:
            results = list(self._results)

        total = len(results)
        wins = [r for r in results if r.net_profit_pct > 0]
        losses = [r for r in results if r.net_profit_pct <= 0]

        win_rate = (len(wins) / total) if total else 0.0
        average_profit = (sum(r.net_profit_pct for r in wins) / len(wins)) if wins else 0.0
        average_loss = (sum(r.net_profit_pct for r in losses) / len(losses)) if losses else 0.0

        gross_win = sum(r.net_profit_pct for r in wins)
        gross_loss = abs(sum(r.net_profit_pct for r in losses))
        if gross_loss > 0:
            profit_factor = gross_win / gross_loss
        elif gross_win > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        return {
            "total_signals": total,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "average_profit_pct": average_profit,
            "average_loss_pct": average_loss,
            "profit_factor": profit_factor,
            "total_fees_pct": sum(r.fees_pct for r in results),
            "net_pnl_pct": sum(r.net_profit_pct for r in results),
            "average_holding_time_sec": (sum(r.duration_sec for r in results) / total) if total else 0.0,
        }
