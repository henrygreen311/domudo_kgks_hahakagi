from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import websockets

log = logging.getLogger(__name__)

DEMO_TRADING = True

OKX_PUBLIC_WS_URL = (
    "wss://wspap.okx.com:8443/ws/v5/public"
    if DEMO_TRADING
    else "wss://ws.okx.com:8443/ws/v5/public"
)

EXCHANGE_LABELS = {
    "okx": "OKX",
    "coinbase": "Coinbase",
    "kraken": "Kraken",
    "mexc": "MEXC",
    "bitget": "Bitget",
    "gateio": "Gate.io",
    "bingx": "BingX",
}

REQUIRED_FIELDS: Dict[str, Tuple[str, ...]] = {
    "ticker": ("price", "timestamp"),
    "book": ("best_bid", "best_ask", "bid_size", "ask_size", "timestamp"),
    "trades": ("price", "size", "side", "timestamp"),
    "liquidations": ("side", "size", "price", "timestamp"),
}


def base_asset(okx_symbol: str) -> str:
    return okx_symbol.split("-")[0].upper()


SYMBOL_BUILDERS = {
    "okx": lambda base, okx_symbol: okx_symbol,
    "coinbase": lambda base, okx_symbol: f"{base}-USD",
    "kraken": lambda base, okx_symbol: f"{base}/USD",
    "mexc": lambda base, okx_symbol: f"{base}_USDT",
    "bitget": lambda base, okx_symbol: f"{base}USDT",
    "gateio": lambda base, okx_symbol: f"{base}_USDT",
    "bingx": lambda base, okx_symbol: f"{base}-USDT",
}


def to_exchange_symbol(okx_symbol: str, exchange: str) -> str:
    builder = SYMBOL_BUILDERS.get(exchange)
    if builder is None:
        raise ValueError(f"no symbol mapping registered for exchange {exchange!r}")
    return builder(base_asset(okx_symbol), okx_symbol)


def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _normalize_side(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("buy", "bid", "b"):
        return "buy"
    if s in ("sell", "ask", "a", "s"):
        return "sell"
    if s == "1":
        return "buy"
    if s == "2":
        return "sell"
    return None


def _extract_ts(raw) -> float:
    parsed = None
    if raw is not None:
        try:
            v = float(raw)
            if v > 1e15:
                parsed = v / 1e6
            elif v > 1e12:
                parsed = v / 1000.0
            elif v > 1e9:
                parsed = v
        except (TypeError, ValueError):
            try:
                parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
            except Exception:
                parsed = None
    now = time.time()
    if parsed is not None and parsed > 0 and abs(now - parsed) <= 120.0:
        return parsed
    return now


@dataclass
class CrossExchangeConfig:
    total_exchanges: int = 7
    min_agreeing: int = 5
    min_online_exchanges: int = 5

    window_size: int = 40
    trade_window_size: int = 100
    liquidation_window_size: int = 30
    snapshot_wait_timeout_sec: float = 6.0
    max_snapshot_age_sec: float = 8.0
    subscription_idle_ttl_sec: float = 300.0
    reconnect_backoff_sec: float = 3.0
    reconnect_backoff_max_sec: float = 30.0
    self_test_symbol: str = "BTC-USDT-SWAP"
    self_test_timeout_sec: float = 20.0

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "momentum": 0.30,
            "order_book_imbalance": 0.20,
            "aggressive_flow": 0.20,
            "trend_continuation": 0.15,
            "breakout": 0.10,
            "whale_activity": 0.05,
        }
    )
    neutral_band: float = 0.05


@dataclass
class Tick:
    ts: float
    last: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_sz: Optional[float] = None
    ask_sz: Optional[float] = None
    volume: Optional[float] = None


@dataclass
class TradePrint:
    ts: float
    price: Optional[float]
    size: Optional[float]
    side: Optional[str]


@dataclass
class LiquidationRecord:
    ts: float
    side: Optional[str]
    size: Optional[float]
    price: Optional[float]


@dataclass
class SymbolState:
    ticks: Deque[Tick]
    trades: Deque[TradePrint]
    liquidations: Deque[LiquidationRecord]
    last_bid: Optional[float] = None
    last_ask: Optional[float] = None
    last_bid_sz: Optional[float] = None
    last_ask_sz: Optional[float] = None
    last_seen: float = 0.0


@dataclass
class ParsedEvent:
    channel: str
    symbol: Optional[str]
    fields: dict


@dataclass
class ChannelHealth:
    subscribed: bool = False
    messages_seen: int = 0
    parsed_ok: int = 0
    parsed_fail: int = 0
    verified: bool = False
    last_error: Optional[str] = None


@dataclass
class ExchangeHealth:
    connected: bool = False
    channels: Dict[str, ChannelHealth] = field(default_factory=dict)

    def channel(self, name: str) -> ChannelHealth:
        return self.channels.setdefault(name, ChannelHealth())


@dataclass
class ExchangeOpinion:
    exchange: str
    symbol: str
    direction: str
    confidence: float
    momentum: float
    order_book_imbalance: float
    aggressive_buying: float
    aggressive_selling: float
    whale_activity: float
    breakout_confirmation: bool
    trend_continuation: float
    unavailable_metrics: Tuple[str, ...] = ()
    error: Optional[str] = None


@dataclass
class CrossExchangeResult:
    decision: str
    final_direction: Optional[str]
    agreeing_exchanges: int
    total_exchanges: int
    online_exchanges: int
    consensus_pct: float
    average_confidence: float
    reason: Optional[str]
    exchange_results: Dict[str, ExchangeOpinion]

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "final_direction": self.final_direction,
            "agreeing_exchanges": self.agreeing_exchanges,
            "total_exchanges": self.total_exchanges,
            "online_exchanges": self.online_exchanges,
            "consensus_pct": self.consensus_pct,
            "average_confidence": self.average_confidence,
            "reason": self.reason,
            "exchange_results": {
                name: {
                    "symbol": o.symbol,
                    "direction": o.direction,
                    "confidence": o.confidence,
                    "momentum": o.momentum,
                    "order_book_imbalance": o.order_book_imbalance,
                    "aggressive_buying": o.aggressive_buying,
                    "aggressive_selling": o.aggressive_selling,
                    "whale_activity": o.whale_activity,
                    "breakout_confirmation": o.breakout_confirmation,
                    "trend_continuation": o.trend_continuation,
                    "unavailable_metrics": list(o.unavailable_metrics),
                    "error": o.error,
                }
                for name, o in self.exchange_results.items()
            },
        }


def _offline_opinion(exchange: str, symbol: str, error: str) -> ExchangeOpinion:
    return ExchangeOpinion(
        exchange=exchange, symbol=symbol, direction="neutral", confidence=0.0,
        momentum=0.0, order_book_imbalance=0.0, aggressive_buying=0.0,
        aggressive_selling=0.0, whale_activity=0.0, breakout_confirmation=False,
        trend_continuation=0.0, unavailable_metrics=(), error=error,
    )


def _analyze(
    exchange: str, symbol: str, state: SymbolState, health: ExchangeHealth, cfg: CrossExchangeConfig
) -> Optional[ExchangeOpinion]:
    ticks = state.ticks
    if len(ticks) < 2:
        return None
    first, last = ticks[0], ticks[-1]
    if not first.last or not last.last:
        return None

    momentum_raw = (last.last - first.last) / first.last
    momentum = max(-1.0, min(1.0, momentum_raw / 0.003))

    unavailable: List[str] = []

    book_verified = health.channel("book").verified
    imbalance_samples = (
        [
            (t.bid_sz - t.ask_sz) / (t.bid_sz + t.ask_sz)
            for t in ticks
            if t.bid_sz and t.ask_sz and (t.bid_sz + t.ask_sz) > 0
        ]
        if book_verified
        else []
    )
    if imbalance_samples:
        order_book_imbalance = sum(imbalance_samples) / len(imbalance_samples)
    else:
        order_book_imbalance = 0.0
        unavailable.append("order_book_imbalance")

    trades_verified = health.channel("trades").verified
    trade_list = [t for t in state.trades if t.price and t.size]
    if trades_verified and trade_list:
        buy_vol = sum(t.size for t in trade_list if t.side == "buy")
        sell_vol = sum(t.size for t in trade_list if t.side == "sell")
        total_vol = buy_vol + sell_vol
        if total_vol > 0:
            aggressive_buying = buy_vol / total_vol
            aggressive_selling = sell_vol / total_vol
        else:
            aggressive_buying = aggressive_selling = 0.0
            unavailable.append("aggressive_flow")
    else:
        aggressive_buying = aggressive_selling = 0.0
        unavailable.append("aggressive_flow")
    aggressive_flow = aggressive_buying - aggressive_selling

    whale_activity = 0.0
    whale_signal_found = False
    sizes = [t.size for t in trade_list]
    if len(sizes) >= 3:
        avg = sum(sizes) / len(sizes)
        mx = max(sizes)
        if avg > 0:
            whale_activity = max(whale_activity, max(0.0, min(1.0, mx / (avg * 6.0) - 1.0)))
            whale_signal_found = True
    if state.liquidations:
        whale_activity = max(whale_activity, min(1.0, len(state.liquidations) / 5.0))
        whale_signal_found = True
    if not whale_signal_found:
        unavailable.append("whale_activity")

    prior = [t.last for t in list(ticks)[:-1]]
    breakout_up = bool(prior) and last.last > max(prior)
    breakout_down = bool(prior) and last.last < min(prior)
    breakout_confirmation = breakout_up or breakout_down

    overall_up = last.last >= first.last
    agree = total = 0
    prev = None
    for t in ticks:
        if prev is not None and t.last != prev.last:
            total += 1
            if (t.last > prev.last) == overall_up:
                agree += 1
        prev = t
    trend_continuation = (agree / total) if total else 0.5

    weights = dict(cfg.weights)
    for key in ("order_book_imbalance", "aggressive_flow", "whale_activity"):
        if key in unavailable:
            weights[key] = 0.0
    weight_total = sum(weights.values())
    if weight_total > 0:
        weights = {k: v / weight_total for k, v in weights.items()}

    direction_signal = (
        weights["momentum"] * momentum
        + weights["order_book_imbalance"] * order_book_imbalance
        + weights["aggressive_flow"] * aggressive_flow
        + weights["trend_continuation"] * (trend_continuation - 0.5) * 2.0 * (1.0 if overall_up else -1.0)
    )
    if breakout_up:
        direction_signal += weights["breakout"]
    elif breakout_down:
        direction_signal -= weights["breakout"]
    if whale_activity > 0.3:
        direction_signal *= 1.0 + weights["whale_activity"] * whale_activity

    direction_signal = max(-1.0, min(1.0, direction_signal))

    if direction_signal > cfg.neutral_band:
        direction = "long"
    elif direction_signal < -cfg.neutral_band:
        direction = "short"
    else:
        direction = "neutral"

    return ExchangeOpinion(
        exchange=exchange,
        symbol=symbol,
        direction=direction,
        confidence=round(abs(direction_signal) * 100.0, 1),
        momentum=round(momentum, 4),
        order_book_imbalance=round(order_book_imbalance, 4),
        aggressive_buying=round(aggressive_buying, 4),
        aggressive_selling=round(aggressive_selling, 4),
        whale_activity=round(whale_activity, 4),
        breakout_confirmation=breakout_confirmation,
        trend_continuation=round(trend_continuation, 4),
        unavailable_metrics=tuple(unavailable),
        error=None,
    )


class _ExchangeConnector:
    name = "base"
    ws_url = ""
    SUPPORTS_LIQUIDATIONS = False
    LIQUIDATIONS_GLOBAL = False
    # websockets' default is 1 MiB, which Coinbase's level2_batch full
    # order-book snapshot for a busy pair (BTC-USD) exceeds — this was
    # killing the connection mid-message every single time
    # (ConnectionClosedError: 1009 message too big) and looping forever
    # since the very next reconnect just failed the same way. 20 MiB
    # covers every exchange here with plenty of room.
    MAX_MESSAGE_SIZE = 20 * 1024 * 1024

    # App-level keepalive. websockets' own protocol-level ping/pong isn't
    # enough for exchanges that require a text/JSON *data frame* at the
    # application layer — Bitget and MEXC both silently drop connections
    # that never send one (confirmed against their docs: Bitget wants a
    # literal "ping" string every 30s, disconnects after 2 min of
    # silence; MEXC wants {"method": "ping"} every <=60s, recommends
    # 10-20s). None here means "no app-level ping needed" — Coinbase,
    # Kraken, and BingX don't require one per their public docs.
    PING_INTERVAL_SEC: Optional[float] = None

    def build_ping_message(self):
        """Returns a str or dict to send verbatim as the keepalive ping,
        or None if this exchange doesn't need one. Only called when
        PING_INTERVAL_SEC is set."""
        return None

    async def _ping_loop(self, ws) -> None:
        if self.PING_INTERVAL_SEC is None:
            return
        while True:
            await asyncio.sleep(self.PING_INTERVAL_SEC)
            msg = self.build_ping_message()
            if msg is None:
                continue
            try:
                await ws.send(msg if isinstance(msg, str) else json.dumps(msg))
            except Exception:
                return  # _recv_loop hitting the same dead socket triggers the reconnect

    def __init__(self, config: CrossExchangeConfig) -> None:
        self.config = config
        self._online = False
        self._subscribed: set = set()
        self._global_subscribed: set = set()
        self._pending_subscribe: "asyncio.Queue[str]" = asyncio.Queue()
        self._state: Dict[str, SymbolState] = {}
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._health = ExchangeHealth()

    @property
    def is_online(self) -> bool:
        return self._online

    @property
    def is_connected(self) -> bool:
        return self._online

    @property
    def health(self) -> ExchangeHealth:
        return self._health

    @property
    def is_verified_online(self) -> bool:
        return self._online and all(self.channel_verified(k) for k in ("ticker", "book", "trades"))

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_forever(), name=f"cross_exchange:{self.name}")
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_sweep_loop(), name=f"cross_exchange:{self.name}:idle-sweep")

    async def _idle_sweep_loop(self) -> None:
        """Frees local per-symbol memory (ticks/trades/liquidations/book
        reconstruction) for symbols not seen in subscription_idle_ttl_sec.
        Does NOT send an exchange-side unsubscribe — each exchange's
        unsubscribe payload shape would be another 7 things to get right
        and verify live, and duplicate subscribe requests are harmless
        no-ops on every exchange used here. This only fixes the actual
        "memory efficient" requirement (unbounded per-symbol state
        growth over a long-running process); if a symbol resurfaces
        later, ensure_subscribed() re-adds it and — worst case — the
        exchange silently ignores an already-active subscription it
        still had server-side."""
        interval = max(30.0, self.config.subscription_idle_ttl_sec / 5.0)
        while True:
            await asyncio.sleep(interval)
            cutoff = time.time() - self.config.subscription_idle_ttl_sec
            async with self._lock:
                idle_symbols = [s for s, state in self._state.items() if state.last_seen and state.last_seen < cutoff]
                for symbol in idle_symbols:
                    self._state.pop(symbol, None)
                    self._subscribed.discard(symbol)
                    self._on_symbol_purged(symbol)
            if idle_symbols:
                log.info(f"[cross_exchange:{self.name}] purged local state for {len(idle_symbols)} idle symbol(s)")

    def _on_symbol_purged(self, symbol: str) -> None:
        """Hook for connectors keeping extra per-symbol state alongside
        SymbolState (currently just Coinbase's reconstructed order book)."""

    async def ensure_subscribed(self, symbol: str) -> None:
        async with self._lock:
            if symbol in self._subscribed:
                return
        await self._pending_subscribe.put(symbol)

    def _symbol_state(self, symbol: str) -> SymbolState:
        return self._state.setdefault(
            symbol,
            SymbolState(
                ticks=deque(maxlen=self.config.window_size),
                trades=deque(maxlen=self.config.trade_window_size),
                liquidations=deque(maxlen=self.config.liquidation_window_size),
            ),
        )

    def get_state(self, symbol: str) -> SymbolState:
        return self._symbol_state(symbol)

    def get_ticks(self, symbol: str) -> Deque[Tick]:
        return self._symbol_state(symbol).ticks

    def last_seen(self, symbol: str) -> float:
        state = self._state.get(symbol)
        return state.last_seen if state else 0.0

    def channel_health(self, channel_key: str) -> ChannelHealth:
        return self._health.channel(channel_key)

    def channel_verified(self, channel_key: str) -> bool:
        return self._health.channels.get(channel_key, ChannelHealth()).verified

    def required_verified(self, symbol: str) -> bool:
        required_ok = all(self.channel_verified(k) for k in ("ticker", "book", "trades"))
        return required_ok and self.last_seen(symbol) > 0

    def build_ticker_sub(self, symbol: str) -> dict:
        raise NotImplementedError

    def build_book_sub(self, symbol: str) -> dict:
        raise NotImplementedError

    def build_trades_sub(self, symbol: str) -> dict:
        raise NotImplementedError

    def build_liquidations_sub(self, symbol: str) -> dict:
        raise NotImplementedError

    def parse_message(self, raw) -> List[ParsedEvent]:
        raise NotImplementedError

    def _channel_builders(self):
        builders = {
            "ticker": self.build_ticker_sub,
            "book": self.build_book_sub,
            "trades": self.build_trades_sub,
        }
        if self.SUPPORTS_LIQUIDATIONS:
            builders["liquidations"] = self.build_liquidations_sub
        return builders

    async def _run_forever(self) -> None:
        backoff = self.config.reconnect_backoff_sec
        while True:
            self._health = ExchangeHealth()
            try:
                async with websockets.connect(self.ws_url, open_timeout=10, max_size=self.MAX_MESSAGE_SIZE) as ws:
                    self._online = True
                    self._health.connected = True
                    backoff = self.config.reconnect_backoff_sec
                    log.info(f"[cross_exchange:{self.name}] connected")

                    async with self._lock:
                        already_subscribed = list(self._subscribed)
                        self._subscribed.clear()
                        self._global_subscribed.clear()
                    for symbol in already_subscribed:
                        await self._pending_subscribe.put(symbol)

                    await asyncio.gather(
                        self._subscribe_consumer(ws),
                        self._recv_loop(ws),
                        self._ping_loop(ws),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning(
                    f"[cross_exchange:{self.name}] connection lost/failed: "
                    f"{type(e).__name__}: {str(e)[:200]} — reconnecting in {backoff:.0f}s"
                )
            finally:
                self._online = False

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.config.reconnect_backoff_max_sec)

    async def _subscribe_consumer(self, ws) -> None:
        while True:
            symbol = await self._pending_subscribe.get()
            async with self._lock:
                if symbol in self._subscribed:
                    continue
                self._subscribed.add(symbol)
            ok, failed = [], []
            for channel_key, builder in self._channel_builders().items():
                if channel_key == "liquidations" and self.LIQUIDATIONS_GLOBAL:
                    async with self._lock:
                        if "liquidations" in self._global_subscribed:
                            continue
                        self._global_subscribed.add("liquidations")
                try:
                    payload = builder(symbol)
                    await ws.send(json.dumps(payload))
                    self.channel_health(channel_key).subscribed = True
                    ok.append(channel_key)
                except Exception as e:
                    self.channel_health(channel_key).last_error = f"subscribe failed: {e}"
                    failed.append(f"{channel_key}: {type(e).__name__}")
            if ok:
                log.info(f"[cross_exchange:{self.name}] {symbol} subscribed ({', '.join(ok)})")
            if failed:
                log.warning(f"[cross_exchange:{self.name}] {symbol} subscribe FAILED ({'; '.join(failed)})")

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            try:
                events = self.parse_message(raw)
            except Exception:
                continue
            for ev in events:
                self._handle_event(ev)

    def _handle_event(self, ev: ParsedEvent) -> None:
        if ev is None or ev.symbol is None:
            return
        health = self._health.channel(ev.channel)
        health.messages_seen += 1
        required = REQUIRED_FIELDS.get(ev.channel, ())
        missing = [f for f in required if ev.fields.get(f) is None]
        if missing:
            health.parsed_fail += 1
            health.last_error = f"{ev.channel} missing {', '.join(missing)}"
            return
        health.parsed_ok += 1
        health.verified = True
        if ev.channel == "ticker":
            self._apply_ticker(ev.symbol, ev.fields)
        elif ev.channel == "book":
            self._apply_book(ev.symbol, ev.fields)
        elif ev.channel == "trades":
            self._apply_trade(ev.symbol, ev.fields)
        elif ev.channel == "liquidations":
            self._apply_liquidation(ev.symbol, ev.fields)

    def _apply_ticker(self, symbol: str, fields: dict) -> None:
        state = self._symbol_state(symbol)
        tick = Tick(
            ts=fields["timestamp"],
            last=fields["price"],
            bid=state.last_bid,
            ask=state.last_ask,
            bid_sz=state.last_bid_sz,
            ask_sz=state.last_ask_sz,
            volume=fields.get("volume"),
        )
        state.ticks.append(tick)
        state.last_seen = time.time()

    def _apply_book(self, symbol: str, fields: dict) -> None:
        state = self._symbol_state(symbol)
        state.last_bid = fields.get("best_bid")
        state.last_ask = fields.get("best_ask")
        state.last_bid_sz = fields.get("bid_size")
        state.last_ask_sz = fields.get("ask_size")
        state.last_seen = time.time()

    def _apply_trade(self, symbol: str, fields: dict) -> None:
        state = self._symbol_state(symbol)
        state.trades.append(
            TradePrint(ts=fields["timestamp"], price=fields.get("price"), size=fields.get("size"), side=fields.get("side"))
        )
        state.last_seen = time.time()

    def _apply_liquidation(self, symbol: str, fields: dict) -> None:
        state = self._symbol_state(symbol)
        state.liquidations.append(
            LiquidationRecord(ts=fields["timestamp"], side=fields.get("side"), size=fields.get("size"), price=fields.get("price"))
        )
        state.last_seen = time.time()


class OKXConnector(_ExchangeConnector):
    name = "okx"
    ws_url = OKX_PUBLIC_WS_URL
    SUPPORTS_LIQUIDATIONS = True
    LIQUIDATIONS_GLOBAL = True
    PING_INTERVAL_SEC = 25.0  # OKX docs: send literal "ping" if no message sent in 30s

    def build_ping_message(self):
        return "ping"

    def build_ticker_sub(self, symbol: str) -> dict:
        return {"op": "subscribe", "args": [{"channel": "tickers", "instId": symbol}]}

    def build_book_sub(self, symbol: str) -> dict:
        return {"op": "subscribe", "args": [{"channel": "books5", "instId": symbol}]}

    def build_trades_sub(self, symbol: str) -> dict:
        return {"op": "subscribe", "args": [{"channel": "trades", "instId": symbol}]}

    def build_liquidations_sub(self, symbol: str) -> dict:
        return {"op": "subscribe", "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]}

    def parse_message(self, raw) -> List[ParsedEvent]:
        msg = json.loads(raw)
        arg = msg.get("arg", {})
        channel = arg.get("channel")
        data = msg.get("data")
        if not channel or not data:
            return []
        events: List[ParsedEvent] = []
        if channel == "tickers":
            for d in data:
                symbol = d.get("instId")
                price = _to_float(d.get("last"))
                if symbol is None or price is None:
                    continue
                events.append(
                    ParsedEvent(
                        "ticker", symbol,
                        {"price": price, "volume": _to_float(d.get("vol24h")), "timestamp": _extract_ts(d.get("ts"))},
                    )
                )
        elif channel == "books5":
            symbol = arg.get("instId")
            for d in data:
                bids = d.get("bids") or []
                asks = d.get("asks") or []
                events.append(
                    ParsedEvent(
                        "book", symbol,
                        {
                            "best_bid": _to_float(bids[0][0]) if bids else None,
                            "best_ask": _to_float(asks[0][0]) if asks else None,
                            "bid_size": _to_float(bids[0][1]) if bids else None,
                            "ask_size": _to_float(asks[0][1]) if asks else None,
                            "timestamp": _extract_ts(d.get("ts")),
                        },
                    )
                )
        elif channel == "trades":
            for d in data:
                events.append(
                    ParsedEvent(
                        "trades", d.get("instId"),
                        {
                            "price": _to_float(d.get("px")), "size": _to_float(d.get("sz")),
                            "side": _normalize_side(d.get("side")), "timestamp": _extract_ts(d.get("ts")),
                        },
                    )
                )
        elif channel == "liquidation-orders":
            for d in data:
                symbol = d.get("instId")
                details = d.get("details") or [d]
                for det in details:
                    events.append(
                        ParsedEvent(
                            "liquidations", symbol,
                            {
                                "side": _normalize_side(det.get("side")),
                                "size": _to_float(det.get("sz")),
                                "price": _to_float(det.get("bkPx") or det.get("px")),
                                "timestamp": _extract_ts(det.get("ts")),
                            },
                        )
                    )
        return events


class CoinbaseConnector(_ExchangeConnector):
    name = "coinbase"
    ws_url = "wss://ws-feed.exchange.coinbase.com"
    SUPPORTS_LIQUIDATIONS = False

    def __init__(self, config: CrossExchangeConfig) -> None:
        super().__init__(config)
        self._books: Dict[str, Dict[str, Dict[float, float]]] = {}

    def build_ticker_sub(self, symbol: str) -> dict:
        return {"type": "subscribe", "product_ids": [symbol], "channels": ["ticker"]}

    def build_book_sub(self, symbol: str) -> dict:
        return {"type": "subscribe", "product_ids": [symbol], "channels": ["level2_batch"]}

    def build_trades_sub(self, symbol: str) -> dict:
        return {"type": "subscribe", "product_ids": [symbol], "channels": ["matches"]}

    def _book_for(self, symbol: str) -> Dict[str, Dict[float, float]]:
        return self._books.setdefault(symbol, {"bids": {}, "asks": {}})

    def _on_symbol_purged(self, symbol: str) -> None:
        self._books.pop(symbol, None)

    def _best(self, side_book: Dict[float, float], highest: bool) -> Tuple[Optional[float], Optional[float]]:
        if not side_book:
            return None, None
        price = max(side_book) if highest else min(side_book)
        return price, side_book[price]

    def _book_event(self, symbol: str, ts_raw=None) -> ParsedEvent:
        book = self._book_for(symbol)
        best_bid, best_bid_sz = self._best(book["bids"], highest=True)
        best_ask, best_ask_sz = self._best(book["asks"], highest=False)
        return ParsedEvent(
            "book", symbol,
            {
                "best_bid": best_bid, "best_ask": best_ask,
                "bid_size": best_bid_sz, "ask_size": best_ask_sz,
                "timestamp": _extract_ts(ts_raw),
            },
        )

    def parse_message(self, raw) -> List[ParsedEvent]:
        msg = json.loads(raw)
        mtype = msg.get("type")
        events: List[Optional[ParsedEvent]] = []
        if mtype == "ticker":
            symbol = msg.get("product_id")
            price = _to_float(msg.get("price"))
            if symbol is None or price is None:
                return []
            events.append(
                ParsedEvent(
                    "ticker", symbol,
                    {"price": price, "volume": _to_float(msg.get("volume_24h")), "timestamp": _extract_ts(msg.get("time"))},
                )
            )
        elif mtype == "snapshot":
            symbol = msg.get("product_id")
            if symbol is None:
                return []
            book = self._book_for(symbol)
            book["bids"] = {
                p: s for p, s in ((_to_float(x[0]), _to_float(x[1])) for x in msg.get("bids", [])) if p is not None and s is not None
            }
            book["asks"] = {
                p: s for p, s in ((_to_float(x[0]), _to_float(x[1])) for x in msg.get("asks", [])) if p is not None and s is not None
            }
            events.append(self._book_event(symbol))
        elif mtype == "l2update":
            symbol = msg.get("product_id")
            if symbol is None:
                return []
            book = self._book_for(symbol)
            for side, price_raw, size_raw in msg.get("changes", []):
                price = _to_float(price_raw)
                size = _to_float(size_raw)
                if price is None or size is None:
                    continue
                target = book["bids"] if side == "buy" else book["asks"]
                if size == 0:
                    target.pop(price, None)
                else:
                    target[price] = size
            events.append(self._book_event(symbol, msg.get("time")))
        elif mtype == "match":
            symbol = msg.get("product_id")
            events.append(
                ParsedEvent(
                    "trades", symbol,
                    {
                        "price": _to_float(msg.get("price")), "size": _to_float(msg.get("size")),
                        "side": _normalize_side(msg.get("side")), "timestamp": _extract_ts(msg.get("time")),
                    },
                )
            )
        return [e for e in events if e is not None]


class KrakenConnector(_ExchangeConnector):
    name = "kraken"
    ws_url = "wss://ws.kraken.com/v2"
    SUPPORTS_LIQUIDATIONS = False

    def build_ticker_sub(self, symbol: str) -> dict:
        return {"method": "subscribe", "params": {"channel": "ticker", "symbol": [symbol]}}

    def build_book_sub(self, symbol: str) -> dict:
        return {"method": "subscribe", "params": {"channel": "book", "symbol": [symbol], "depth": 10}}

    def build_trades_sub(self, symbol: str) -> dict:
        return {"method": "subscribe", "params": {"channel": "trade", "symbol": [symbol]}}

    def parse_message(self, raw) -> List[ParsedEvent]:
        msg = json.loads(raw)
        channel = msg.get("channel")
        if channel not in ("ticker", "book", "trade") or msg.get("type") not in ("snapshot", "update"):
            return []
        data = msg.get("data") or []
        events: List[ParsedEvent] = []
        for d in data:
            symbol = d.get("symbol")
            if symbol is None:
                continue
            if channel == "ticker":
                price = _to_float(d.get("last"))
                if price is None:
                    continue
                events.append(
                    ParsedEvent(
                        "ticker", symbol,
                        {"price": price, "volume": _to_float(d.get("volume")), "timestamp": _extract_ts(None)},
                    )
                )
            elif channel == "book":
                bids = d.get("bids") or []
                asks = d.get("asks") or []
                events.append(
                    ParsedEvent(
                        "book", symbol,
                        {
                            "best_bid": _to_float(bids[0].get("price")) if bids else None,
                            "best_ask": _to_float(asks[0].get("price")) if asks else None,
                            "bid_size": _to_float(bids[0].get("qty")) if bids else None,
                            "ask_size": _to_float(asks[0].get("qty")) if asks else None,
                            "timestamp": _extract_ts(d.get("timestamp")),
                        },
                    )
                )
            elif channel == "trade":
                events.append(
                    ParsedEvent(
                        "trades", symbol,
                        {
                            "price": _to_float(d.get("price")), "size": _to_float(d.get("qty")),
                            "side": _normalize_side(d.get("side")), "timestamp": _extract_ts(d.get("timestamp")),
                        },
                    )
                )
        return events


class MEXCConnector(_ExchangeConnector):
    name = "mexc"
    ws_url = "wss://contract.mexc.com/edge"
    SUPPORTS_LIQUIDATIONS = True
    LIQUIDATIONS_GLOBAL = False
    # Confirmed via MEXC contract API docs: server disconnects if no ping
    # is received within 1 minute; docs recommend sending one every 10-20s.
    PING_INTERVAL_SEC = 15.0

    def build_ping_message(self):
        return {"method": "ping"}

    def build_ticker_sub(self, symbol: str) -> dict:
        return {"method": "sub.ticker", "param": {"symbol": symbol}}

    def build_book_sub(self, symbol: str) -> dict:
        return {"method": "sub.depth", "param": {"symbol": symbol}}

    def build_trades_sub(self, symbol: str) -> dict:
        return {"method": "sub.deal", "param": {"symbol": symbol}}

    def build_liquidations_sub(self, symbol: str) -> dict:
        return {"method": "sub.liquidate.order", "param": {"symbol": symbol}}

    def parse_message(self, raw) -> List[ParsedEvent]:
        msg = json.loads(raw)
        channel = msg.get("channel")
        if not channel or "data" not in msg:
            return []
        d = msg["data"]
        symbol = msg.get("symbol") or (d.get("symbol") if isinstance(d, dict) else None)
        if channel == "push.ticker":
            price = _to_float(d.get("lastPrice"))
            if symbol is None or price is None:
                return []
            return [
                ParsedEvent(
                    "ticker", symbol,
                    {"price": price, "volume": _to_float(d.get("volume24")), "timestamp": _extract_ts(d.get("timestamp"))},
                )
            ]
        if channel == "push.depth":
            if symbol is None:
                return []
            bids = d.get("bids") or []
            asks = d.get("asks") or []
            return [
                ParsedEvent(
                    "book", symbol,
                    {
                        "best_bid": _to_float(bids[0][0]) if bids else None,
                        "best_ask": _to_float(asks[0][0]) if asks else None,
                        "bid_size": _to_float(bids[0][1]) if bids else None,
                        "ask_size": _to_float(asks[0][1]) if asks else None,
                        "timestamp": _extract_ts(None),
                    },
                )
            ]
        if channel == "push.deal":
            items = d if isinstance(d, list) else [d]
            events = []
            for item in items:
                side_code = item.get("T")
                side = "buy" if side_code == 1 else "sell" if side_code == 2 else None
                events.append(
                    ParsedEvent(
                        "trades", symbol,
                        {"price": _to_float(item.get("p")), "size": _to_float(item.get("v")), "side": side, "timestamp": _extract_ts(item.get("t"))},
                    )
                )
            return [e for e in events if e.symbol]
        if channel == "push.liquidate.order":
            items = d if isinstance(d, list) else [d]
            events = []
            for item in items:
                side = _normalize_side(item.get("side"))
                if side is None:
                    side = "buy" if item.get("type") == 1 else "sell" if item.get("type") == 2 else None
                events.append(
                    ParsedEvent(
                        "liquidations", symbol or item.get("symbol"),
                        {
                            "side": side,
                            "size": _to_float(item.get("vol") or item.get("v")),
                            "price": _to_float(item.get("price") or item.get("p")),
                            "timestamp": _extract_ts(item.get("ts") or item.get("t")),
                        },
                    )
                )
            return [e for e in events if e.symbol]
        return []


class BitgetConnector(_ExchangeConnector):
    name = "bitget"
    ws_url = "wss://ws.bitget.com/v2/ws/public"
    SUPPORTS_LIQUIDATIONS = True
    LIQUIDATIONS_GLOBAL = False
    # This is the actual fix for the "connection lost ... no close frame
    # received or sent" / repeated-reconnect-then-immediately-fail
    # pattern observed in production: Bitget's docs are explicit —
    # clients must send a literal "ping" string every 30s or the server
    # disconnects after 2 minutes of receiving none. We never sent one,
    # so every connection was silently doomed from the moment it opened.
    PING_INTERVAL_SEC = 25.0

    def build_ping_message(self):
        return "ping"

    def build_ticker_sub(self, symbol: str) -> dict:
        return {"op": "subscribe", "args": [{"instType": "USDT-FUTURES", "channel": "ticker", "instId": symbol}]}

    def build_book_sub(self, symbol: str) -> dict:
        return {"op": "subscribe", "args": [{"instType": "USDT-FUTURES", "channel": "books15", "instId": symbol}]}

    def build_trades_sub(self, symbol: str) -> dict:
        return {"op": "subscribe", "args": [{"instType": "USDT-FUTURES", "channel": "trade", "instId": symbol}]}

    def build_liquidations_sub(self, symbol: str) -> dict:
        return {"op": "subscribe", "args": [{"instType": "USDT-FUTURES", "channel": "liquidation-order", "instId": symbol}]}

    def parse_message(self, raw) -> List[ParsedEvent]:
        msg = json.loads(raw)
        arg = msg.get("arg", {})
        channel = arg.get("channel")
        data = msg.get("data")
        if not channel or msg.get("action") not in ("snapshot", "update") or not data:
            return []
        symbol = arg.get("instId")
        events: List[ParsedEvent] = []
        if channel == "ticker":
            for d in data:
                price = _to_float(d.get("lastPr"))
                if price is None:
                    continue
                events.append(
                    ParsedEvent(
                        "ticker", symbol or d.get("instId"),
                        {"price": price, "volume": _to_float(d.get("baseVolume")), "timestamp": _extract_ts(d.get("ts"))},
                    )
                )
        elif channel in ("books15", "books5", "books"):
            for d in data:
                bids = d.get("bids") or []
                asks = d.get("asks") or []
                events.append(
                    ParsedEvent(
                        "book", symbol,
                        {
                            "best_bid": _to_float(bids[0][0]) if bids else None,
                            "best_ask": _to_float(asks[0][0]) if asks else None,
                            "bid_size": _to_float(bids[0][1]) if bids else None,
                            "ask_size": _to_float(asks[0][1]) if asks else None,
                            "timestamp": _extract_ts(d.get("ts")),
                        },
                    )
                )
        elif channel == "trade":
            for d in data:
                events.append(
                    ParsedEvent(
                        "trades", symbol,
                        {
                            "price": _to_float(d.get("price")), "size": _to_float(d.get("size")),
                            "side": _normalize_side(d.get("side")), "timestamp": _extract_ts(d.get("ts")),
                        },
                    )
                )
        elif channel == "liquidation-order":
            for d in data:
                events.append(
                    ParsedEvent(
                        "liquidations", symbol,
                        {
                            "side": _normalize_side(d.get("side")),
                            "size": _to_float(d.get("size") or d.get("baseVolume")),
                            "price": _to_float(d.get("price")),
                            "timestamp": _extract_ts(d.get("ts")),
                        },
                    )
                )
        return events


class GateioConnector(_ExchangeConnector):
    name = "gateio"
    ws_url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
    SUPPORTS_LIQUIDATIONS = True
    LIQUIDATIONS_GLOBAL = False
    # Gate.io's own docs describe this as optional ("if you want to
    # actively detect the connection status") rather than mandatory, but
    # it's cheap and matches their documented format exactly.
    PING_INTERVAL_SEC = 20.0

    def build_ping_message(self):
        return {"time": int(time.time()), "channel": "futures.ping"}

    def build_ticker_sub(self, symbol: str) -> dict:
        return {"time": int(time.time()), "channel": "futures.tickers", "event": "subscribe", "payload": [symbol]}

    def build_book_sub(self, symbol: str) -> dict:
        return {"time": int(time.time()), "channel": "futures.order_book", "event": "subscribe", "payload": [symbol, "5", "0"]}

    def build_trades_sub(self, symbol: str) -> dict:
        return {"time": int(time.time()), "channel": "futures.trades", "event": "subscribe", "payload": [symbol]}

    def build_liquidations_sub(self, symbol: str) -> dict:
        return {"time": int(time.time()), "channel": "futures.liquidates", "event": "subscribe", "payload": [symbol]}

    def parse_message(self, raw) -> List[ParsedEvent]:
        msg = json.loads(raw)
        channel = msg.get("channel")
        event = msg.get("event")
        # Confirmed via Gate.io's official Futures WebSocket docs
        # (gate.com/docs/developers/futures/): ticker/trades data
        # notifications use event:"update", but futures.order_book's data
        # notification uses event:"all" — a different value for the same
        # "here's a real message" meaning, not a subscribe ack. The old
        # code only accepted "update" here, which silently dropped every
        # order-book message while ticker/trades kept working — exactly
        # the observed "Gate.io FAIL (book)" with ticker/trades verified.
        if event not in ("update", "all") or "result" not in msg:
            return []
        result = msg["result"]
        items = result if isinstance(result, list) else [result]
        events: List[ParsedEvent] = []
        if channel == "futures.tickers":
            for d in items:
                symbol = d.get("contract")
                price = _to_float(d.get("last"))
                if symbol is None or price is None:
                    continue
                events.append(
                    ParsedEvent(
                        "ticker", symbol,
                        {"price": price, "volume": _to_float(d.get("volume_24h")), "timestamp": _extract_ts(None)},
                    )
                )
        elif channel == "futures.order_book":
            d = items[0] if items else {}
            symbol = d.get("contract")
            bids = d.get("bids") or []
            asks = d.get("asks") or []
            if symbol:
                events.append(
                    ParsedEvent(
                        "book", symbol,
                        {
                            "best_bid": _to_float(bids[0].get("p")) if bids else None,
                            "best_ask": _to_float(asks[0].get("p")) if asks else None,
                            "bid_size": _to_float(bids[0].get("s")) if bids else None,
                            "ask_size": _to_float(asks[0].get("s")) if asks else None,
                            "timestamp": _extract_ts(d.get("t")),
                        },
                    )
                )
        elif channel == "futures.trades":
            for d in items:
                symbol = d.get("contract")
                size = _to_float(d.get("size"))
                side = None
                if size is not None:
                    side = "buy" if size > 0 else "sell" if size < 0 else None
                events.append(
                    ParsedEvent(
                        "trades", symbol,
                        {
                            "price": _to_float(d.get("price")),
                            "size": abs(size) if size is not None else None,
                            "side": side,
                            "timestamp": _extract_ts(d.get("create_time_ms") or d.get("create_time")),
                        },
                    )
                )
        elif channel == "futures.liquidates":
            for d in items:
                symbol = d.get("contract")
                events.append(
                    ParsedEvent(
                        "liquidations", symbol,
                        {
                            "side": _normalize_side(d.get("side")), "size": _to_float(d.get("size")),
                            "price": _to_float(d.get("price")), "timestamp": _extract_ts(d.get("time")),
                        },
                    )
                )
        return [e for e in events if e.symbol]


class BingXConnector(_ExchangeConnector):
    name = "bingx"
    ws_url = "wss://open-api-swap.bingx.com/swap-market"
    SUPPORTS_LIQUIDATIONS = False

    def build_ticker_sub(self, symbol: str) -> dict:
        return {"id": str(uuid.uuid4()), "reqType": "sub", "dataType": f"{symbol}@ticker"}

    def build_book_sub(self, symbol: str) -> dict:
        return {"id": str(uuid.uuid4()), "reqType": "sub", "dataType": f"{symbol}@depth20"}

    def build_trades_sub(self, symbol: str) -> dict:
        return {"id": str(uuid.uuid4()), "reqType": "sub", "dataType": f"{symbol}@trade"}

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                try:
                    text = gzip.decompress(raw).decode("utf-8")
                except OSError:
                    continue
            else:
                text = raw

            if text == "Ping":
                await ws.send("Pong")
                continue

            try:
                events = self.parse_message(text)
            except Exception:
                continue
            for ev in events:
                self._handle_event(ev)

    def parse_message(self, raw) -> List[ParsedEvent]:
        msg = json.loads(raw)
        data = msg.get("data")
        data_type = str(msg.get("dataType", ""))
        if not data or not data_type:
            return []

        if data_type.endswith("@ticker"):
            symbol = data_type[: -len("@ticker")]
            price = _to_float(data.get("c"))
            if price is None:
                return []
            return [
                ParsedEvent(
                    "ticker", symbol,
                    {"price": price, "volume": _to_float(data.get("v")), "timestamp": _extract_ts(None)},
                )
            ]

        if "@depth" in data_type:
            symbol = data_type.split("@depth")[0]
            bids = data.get("bids") or []
            asks = data.get("asks") or []
            return [
                ParsedEvent(
                    "book", symbol,
                    {
                        "best_bid": _to_float(bids[0][0]) if bids else None,
                        "best_ask": _to_float(asks[0][0]) if asks else None,
                        "bid_size": _to_float(bids[0][1]) if bids else None,
                        "ask_size": _to_float(asks[0][1]) if asks else None,
                        "timestamp": _extract_ts(None),
                    },
                )
            ]

        if data_type.endswith("@trade"):
            symbol = data_type[: -len("@trade")]
            items = data if isinstance(data, list) else [data]
            events = []
            for item in items:
                is_buyer_maker = item.get("m")
                side = None
                if isinstance(is_buyer_maker, bool):
                    side = "sell" if is_buyer_maker else "buy"
                events.append(
                    ParsedEvent(
                        "trades", symbol,
                        {
                            "price": _to_float(item.get("p")), "size": _to_float(item.get("q")),
                            "side": side, "timestamp": _extract_ts(item.get("T")),
                        },
                    )
                )
            return events

        return []


def _avg_confidence(exchange_results: Dict[str, ExchangeOpinion], direction: Optional[str]) -> float:
    vals = [o.confidence for o in exchange_results.values() if o.error is None and o.direction == direction]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def _build_consensus(
    exchange_results: Dict[str, ExchangeOpinion], online_exchanges: int, cfg: CrossExchangeConfig
) -> CrossExchangeResult:
    total = len(exchange_results)

    if online_exchanges < cfg.min_online_exchanges:
        return CrossExchangeResult(
            decision="rejected", final_direction=None, agreeing_exchanges=0,
            total_exchanges=total, online_exchanges=online_exchanges,
            consensus_pct=0.0, average_confidence=0.0,
            reason=(
                f"Insufficient exchanges online ({online_exchanges}/{total}, need >= "
                f"{cfg.min_online_exchanges} reporting to evaluate the {cfg.min_agreeing}-of-{total} rule)."
            ),
            exchange_results=exchange_results,
        )

    counts = {"long": 0, "short": 0, "neutral": 0}
    for o in exchange_results.values():
        if o.error is None:
            counts[o.direction] = counts.get(o.direction, 0) + 1

    final_direction, agreeing = max(counts.items(), key=lambda kv: kv[1])
    consensus_pct = round(100.0 * agreeing / total, 2)

    if final_direction == "neutral" or agreeing < cfg.min_agreeing:
        return CrossExchangeResult(
            decision="rejected",
            final_direction=final_direction if agreeing >= cfg.min_agreeing else None,
            agreeing_exchanges=agreeing, total_exchanges=total, online_exchanges=online_exchanges,
            consensus_pct=consensus_pct, average_confidence=_avg_confidence(exchange_results, final_direction),
            reason=f"Insufficient cross-exchange confirmation (minimum required: {cfg.min_agreeing} of {total}).",
            exchange_results=exchange_results,
        )

    return CrossExchangeResult(
        decision="accepted", final_direction=final_direction,
        agreeing_exchanges=agreeing, total_exchanges=total, online_exchanges=online_exchanges,
        consensus_pct=consensus_pct, average_confidence=_avg_confidence(exchange_results, final_direction),
        reason=None, exchange_results=exchange_results,
    )


def _log_result(okx_symbol: str, candidate_direction: str, result: CrossExchangeResult) -> None:
    """Matches the requested inline style: one line listing every
    exchange as 'NAME= RESULT', comma-separated, instead of one line per
    field per exchange."""
    per_exchange = []
    for name, o in result.exchange_results.items():
        if o.error is not None:
            per_exchange.append(f"{name.upper()}= {o.error.upper()}")
        else:
            extra = f" (unavailable: {'+'.join(o.unavailable_metrics)})" if o.unavailable_metrics else ""
            per_exchange.append(f"{name.upper()}= {o.direction.upper()} {o.confidence:.0f}{extra}")

    summary = (
        f"agreement={result.agreeing_exchanges}/{result.total_exchanges} "
        f"consensus={result.consensus_pct:.2f}% "
        + (f"avg_confidence={result.average_confidence} " if result.decision == "accepted" else "")
        + f"decision={result.decision.upper()}"
        + (f" reason=\"{result.reason}\"" if result.reason else "")
    )

    log.info(f"[cross_exchange] {okx_symbol} candidate={candidate_direction.upper()}")
    log.info(f"[cross_exchange] {', '.join(per_exchange)}")
    log.info(f"[cross_exchange] {summary}")


class CrossExchangeValidator:
    def __init__(self, config: Optional[CrossExchangeConfig] = None) -> None:
        self.config = config or CrossExchangeConfig()
        self._connectors: Dict[str, _ExchangeConnector] = {
            "okx": OKXConnector(self.config),
            "coinbase": CoinbaseConnector(self.config),
            "kraken": KrakenConnector(self.config),
            "mexc": MEXCConnector(self.config),
            "bitget": BitgetConnector(self.config),
            "gateio": GateioConnector(self.config),
            "bingx": BingXConnector(self.config),
        }

    def start(self) -> None:
        for connector in self._connectors.values():
            connector.start()

    @property
    def online_count(self) -> int:
        return sum(1 for c in self._connectors.values() if c.is_verified_online)

    async def validate(self, okx_symbol: str, candidate_direction: str) -> CrossExchangeResult:
        cfg = self.config

        symbols_by_exchange: Dict[str, str] = {}
        for name, connector in self._connectors.items():
            try:
                symbols_by_exchange[name] = to_exchange_symbol(okx_symbol, name)
            except ValueError:
                continue
            await connector.ensure_subscribed(symbols_by_exchange[name])

        # Wait for enough exchanges to actually have analyzable data —
        # not just "received one message ever". _analyze() needs at
        # least 2 ticks to compute anything (momentum needs a start and
        # an end point); checking last_seen > 0 alone let this loop exit
        # after ~300ms once a handful of exchanges got their very first
        # tick, long before the 4s budget was used, which is exactly why
        # freshly-subscribed symbols were showing NO FRESH DATA /
        # INSUFFICIENT TICKS for nearly every candidate — the loop had
        # already moved on to analysis before there was anything to
        # analyze.
        min_ticks_for_analysis = 2
        deadline = time.time() + cfg.snapshot_wait_timeout_sec
        while time.time() < deadline:
            ready = sum(
                1
                for name, connector in self._connectors.items()
                if connector.is_verified_online
                and len(connector.get_state(symbols_by_exchange.get(name, "")).ticks) >= min_ticks_for_analysis
            )
            if ready >= cfg.min_online_exchanges:
                break
            await asyncio.sleep(0.1)

        exchange_results: Dict[str, ExchangeOpinion] = {}
        for name, connector in self._connectors.items():
            symbol = symbols_by_exchange.get(name)
            if symbol is None:
                exchange_results[name] = _offline_opinion(name, okx_symbol, "no symbol mapping")
                continue
            if not connector.is_verified_online:
                exchange_results[name] = _offline_opinion(name, symbol, "not verified online")
                continue

            last_seen = connector.last_seen(symbol)
            age = (time.time() - last_seen) if last_seen else float("inf")
            if age > cfg.max_snapshot_age_sec:
                exchange_results[name] = _offline_opinion(name, symbol, "no fresh data")
                continue

            opinion = _analyze(name, symbol, connector.get_state(symbol), connector.health, cfg)
            exchange_results[name] = opinion if opinion is not None else _offline_opinion(name, symbol, "insufficient ticks")

        online_exchanges = sum(1 for o in exchange_results.values() if o.error is None)
        result = _build_consensus(exchange_results, online_exchanges, cfg)
        _log_result(okx_symbol, candidate_direction, result)
        return result

    async def run_self_test(self, symbol: Optional[str] = None, timeout: Optional[float] = None) -> str:
        cfg = self.config
        test_symbol = symbol or cfg.self_test_symbol
        deadline = time.time() + (timeout or cfg.self_test_timeout_sec)

        symbols_by_exchange: Dict[str, str] = {}
        for name, connector in self._connectors.items():
            try:
                symbols_by_exchange[name] = to_exchange_symbol(test_symbol, name)
            except ValueError:
                continue
            await connector.ensure_subscribed(symbols_by_exchange[name])

        while time.time() < deadline:
            if all(
                connector.required_verified(symbols_by_exchange.get(name, ""))
                for name, connector in self._connectors.items()
            ):
                break
            await asyncio.sleep(0.2)

        lines = ["[cross_exchange] startup check:"]
        all_ready = True
        failing: List[str] = []
        for name, connector in self._connectors.items():
            label = EXCHANGE_LABELS.get(name, name.upper())
            connected = connector.is_connected
            channel_ok = {key: connected and connector.channel_verified(key) for key in ("ticker", "book", "trades")}
            parser_ok = connected and all(channel_ok.values())
            all_ready = all_ready and parser_ok

            if parser_ok:
                lines.append(f"  {label:<9} OK")
            else:
                bad = ["not connected"] if not connected else [k for k, ok in channel_ok.items() if not ok]
                lines.append(f"  {label:<9} FAIL ({', '.join(bad)})")
                failing.append(label)

        status_line = "READY" if all_ready else f"DEGRADED — failing: {', '.join(failing)}"
        lines.append(f"System: {status_line}")

        report = "\n".join(lines)
        log.info(report)
        return report
