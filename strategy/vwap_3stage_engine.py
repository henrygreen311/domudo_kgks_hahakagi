"""
orderflow_bb_scalper_engine.py

Fast ETH-USDT scalping strategy — tuned for ~20 signals/day.

Different from vwap_3stage_engine:
- No VWAP
- No swing high/low exhaustion engines
- No VWAP zone routing

Indicators used:
- Bollinger Bands (overextension / snap-back)
- RSI (overbought / oversold) — relaxed thresholds for frequency
- ATR (volatility gate)
- EMA fast/slow (context)
- Order-flow aggressor imbalance z-score

Signal is only emitted when:
1. Price is at a Bollinger Band extreme (within 0.15%)
2. RSI is oversold/overbought in the bounce direction (45/55)
3. Order-flow aggressor imbalance flips strongly in the bounce direction (≥55%)
4. ATR confirms there is enough volatility for a 0.09% TP (≥0.05%)

Target example:
  Entry ≈ 1882.60
  LONG/SHORT TP = 1884.30
  Distance = ≈ +0.09%
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from market_data import MarketDataStore, TradeStore, Signal
from .base import StrategyContext, StrategyEngine

log = logging.getLogger("okx_futures.orderflow_bb_scalper")

CandleFetcher = Callable[[str, str, int], Awaitable[List[dict]]]


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------


def compute_ema(values: List[float], period: int) -> float:
    """Last EMA value for a series of closes, oldest first."""
    if not values:
        return 0.0
    k = 2.0 / (period + 1.0)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1.0 - k)
    return ema


def compute_rsi(closes: List[float], period: int = 14) -> float:
    """Standard RSI. Returns 50.0 if not enough data."""
    if len(closes) < period + 1:
        return 50.0

    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_bollinger_bands(
    closes: List[float], period: int = 20, num_std: float = 2.0
):
    """Returns lower, mid, upper, width_pct. None if not enough data."""
    if len(closes) < period:
        return None, None, None, None

    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((c - mid) ** 2 for c in window) / period
    std = variance ** 0.5
    upper = mid + num_std * std
    lower = mid - num_std * std
    width_pct = (upper - lower) / mid if mid else 0.0
    return lower, mid, upper, width_pct


def compute_atr_pct(candles: List[dict], period: int = 14) -> Optional[float]:
    """ATR as percentage of last close. Expects candles with ts/high/low/close."""
    if len(candles) < period + 1:
        return None

    ordered = sorted(candles, key=lambda c: c.get("ts", 0))
    trs = []
    for i in range(1, len(ordered)):
        h = ordered[i].get("high")
        l = ordered[i].get("low")
        pc = ordered[i - 1].get("close")
        if h is None or l is None or pc is None:
            continue
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)

    if len(trs) < period:
        return None

    atr = sum(trs[-period:]) / period
    last_close = ordered[-1].get("close")
    if not last_close:
        return None
    return atr / last_close


def _split_forming_and_closed(candles: List[dict]):
    """Split OKX candle rows into forming candle and closed candles."""
    forming = None
    closed: List[dict] = []
    ordered = sorted(candles, key=lambda c: c.get("ts", 0), reverse=True)
    for c in ordered:
        if forming is None and str(c.get("confirm")) == "0":
            forming = c
        else:
            closed.append(c)
    return forming, closed


# ---------------------------------------------------------------------------
# Order-flow aggressor imbalance
# ---------------------------------------------------------------------------


def _classify_aggressor(trades: List[dict]) -> List[dict]:
    """
    Classify each trade as buy-aggressor or sell-aggressor using price
    direction, not just the reported side field.

    Preserves the original 'timestamp' field so downstream sorting works.
    """
    ordered = sorted(trades, key=lambda t: t["timestamp"])
    prev_price = None
    out = []
    for t in ordered:
        price = float(t["price"])
        qty = float(t.get("qty", 0.0))
        side = t.get("side", "")

        if prev_price is None or price == prev_price:
            aggressor = side  # fallback to reported side
        elif price > prev_price:
            aggressor = "buy"
        else:
            aggressor = "sell"

        out.append({
            "timestamp": t["timestamp"],   # <-- CRITICAL: keep timestamp for sorting later
            "price": price,
            "qty": qty,
            "aggressor": aggressor,
        })
        prev_price = price

    return out


def _bucketize_trades_for_flow(trades: List[dict], bucket_count: int) -> List[List[dict]]:
    """Index-based equal-count buckets."""
    if not trades or bucket_count < 1:
        return [[] for _ in range(max(bucket_count, 1))]

    ordered = sorted(trades, key=lambda t: t["timestamp"])
    n = len(ordered)
    buckets: List[List[dict]] = [[] for _ in range(bucket_count)]
    for i, t in enumerate(ordered):
        idx = min(bucket_count - 1, int(i * bucket_count / n))
        buckets[idx].append(t)
    return buckets


def compute_order_flow_imbalance(trades: List[dict], direction: str, bucket_count: int = 6) -> Dict:
    """
    Aggressor-based order-flow imbalance z-score.

    Splits trades into buckets, calculates side aggressor volume ratio per
    bucket, then compares the latest bucket to the prior buckets via z-score.

    Returns:
        strength_pct: 0-100 score
        current_ratio: latest bucket side ratio
        burst: True if latest side volume > average prior side volume
        zscore: normalized deviation of latest ratio
    """
    side = "buy" if direction == "long" else "sell"
    other = "sell" if direction == "long" else "buy"

    classified = _classify_aggressor(trades)
    buckets = _bucketize_trades_for_flow(classified, bucket_count)

    ratios = []
    vols = []
    for bucket in buckets:
        side_vol = sum(float(t["qty"]) for t in bucket if t["aggressor"] == side)
        other_vol = sum(float(t["qty"]) for t in bucket if t["aggressor"] == other)
        total = side_vol + other_vol
        ratios.append(side_vol / total if total > 0 else 0.5)
        vols.append(side_vol)

    if not ratios:
        return {
            "strength_pct": 0.0,
            "current_ratio": 0.5,
            "burst": False,
            "zscore": 0.0,
        }

    current_ratio = ratios[-1]

    if len(ratios) < 2:
        return {
            "strength_pct": 50.0,
            "current_ratio": round(current_ratio, 4),
            "burst": False,
            "zscore": 0.0,
        }

    prev_ratios = ratios[:-1]
    mean = sum(prev_ratios) / len(prev_ratios)
    std = (sum((r - mean) ** 2 for r in prev_ratios) / len(prev_ratios)) ** 0.5
    z = (current_ratio - mean) / std if std > 0 else 0.0
    z = max(-3.0, min(3.0, z))

    strength_pct = 50.0 + (z / 3.0) * 50.0
    avg_prev_vol = sum(vols[:-1]) / len(vols[:-1]) if len(vols) > 1 else 0.0
    burst = vols[-1] > avg_prev_vol if avg_prev_vol > 0 else False

    return {
        "strength_pct": round(strength_pct, 2),
        "current_ratio": round(current_ratio, 4),
        "burst": burst,
        "zscore": round(z, 4),
    }


# ---------------------------------------------------------------------------
# Config (relaxed for ~20 signals/day)
# ---------------------------------------------------------------------------


@dataclass
class OrderFlowBollingerScalperConfig:
    max_observation_minutes: float = 15.0   # longer observation window
    trend_candle_bar: str = "1m"

    bb_period: int = 20
    bb_std: float = 2.0
    bb_skin_pct: float = 0.0015             # 0.15% proximity to band extreme

    rsi_period: int = 14
    rsi_oversold: float = 45.0              # relaxed from 35
    rsi_overbought: float = 55.0            # relaxed from 65

    ema_fast_period: int = 9
    ema_slow_period: int = 21

    atr_period: int = 14
    min_atr_pct: float = 0.0005             # 0.05% min volatility

    trade_window_ms: int = 60000            # 1-minute order-flow window
    flow_bucket_count: int = 6
    flow_strength_min: float = 55.0         # relaxed from 70
    flow_ratio_min: float = 0.55            # relaxed from 0.60
    flow_burst_required: bool = False       # burst not mandatory

    tp_distance_pct: float = 0.0009         # 0.09%
    sl_distance_pct: float = 0.0006         # 0.06% initial stop

    min_data_warmup_sec: float = 10.0       # faster warm-up
    min_data_trade_count: int = 10          # fewer trades needed
    candle_fetch_buffer: int = 5

    # Accept any pair by default — override with a frozenset if you want to restrict.
    symbol_whitelist: Optional[frozenset] = None


# ---------------------------------------------------------------------------
# Candidate state
# ---------------------------------------------------------------------------


@dataclass
class CandidateObservation:
    symbol: str
    direction: str = ""
    status: str = "OBSERVING"
    started_at: float = field(default_factory=time.time)
    last_checked_at: float = 0.0

    data_ready: bool = False

    entry_price: float = 0.0

    ema_fast: float = 0.0
    ema_slow: float = 0.0
    rsi: float = 50.0

    bb_lower: float = 0.0
    bb_mid: float = 0.0
    bb_upper: float = 0.0
    bb_width_pct: float = 0.0

    atr_pct: float = 0.0

    flow_long_pct: float = 0.0
    flow_short_pct: float = 0.0
    flow_long_ratio: float = 0.0
    flow_short_ratio: float = 0.0

    engine_used: str = ""

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.started_at

    def status_line(self) -> str:
        return (
            f"{self.symbol} status={self.status} direction={self.direction or '-'} "
            f"price={self.entry_price:.6g} rsi={self.rsi:.1f} "
            f"bbL={self.bb_lower:.6g} bbU={self.bb_upper:.6g} width={self.bb_width_pct:.3%} "
            f"atr={self.atr_pct:.3%} flowL={self.flow_long_pct:.0f}% flowS={self.flow_short_pct:.0f}% "
            f"elapsed={self.elapsed_sec:.0f}s"
        )


# ---------------------------------------------------------------------------
# Strategy engine
# ---------------------------------------------------------------------------


class OrderFlowBollingerScalperEngine(StrategyEngine):
    """
    Fast scalping engine using Bollinger Band overextension + RSI exhaustion
    + aggressor order-flow imbalance reversal.

    Tuned for higher frequency (~20 signals/day) while still requiring
    price at a band extreme and directional order-flow confirmation.
    """

    name = "orderflow_bb_scalper_engine"

    def __init__(
        self,
        trade_store: TradeStore,
        market_data: MarketDataStore,
        candle_fetcher: CandleFetcher,
        config: Optional[OrderFlowBollingerScalperConfig] = None,
    ) -> None:
        self._trade_store = trade_store
        self._market_data = market_data
        self._candle_fetcher = candle_fetcher
        self.config = config or OrderFlowBollingerScalperConfig()
        self._candidates: Dict[str, CandidateObservation] = {}
        self._lock = asyncio.Lock()

    async def sync_watchlist(self, watchlist_symbols) -> None:
        watchlist_symbols = set(watchlist_symbols)
        whitelist = self.config.symbol_whitelist
        if whitelist:
            rejected = watchlist_symbols - whitelist
            watchlist_symbols &= whitelist
            if rejected:
                log.debug(
                    f"[{self.name}] ignoring non-whitelisted symbols: {sorted(rejected)}"
                )

        async with self._lock:
            for symbol in watchlist_symbols:
                if symbol not in self._candidates:
                    self._candidates[symbol] = CandidateObservation(symbol=symbol)
                    log.info(
                        f"[{self.name}] {symbol} added — fast scalping observation active"
                    )

            dropped = [s for s in self._candidates if s not in watchlist_symbols]
            for symbol in dropped:
                del self._candidates[symbol]

    async def snapshot(self) -> List[CandidateObservation]:
        async with self._lock:
            return list(self._candidates.values())

    async def evaluate(self, symbol: str) -> Optional[Signal]:
        cfg = self.config

        async with self._lock:
            candidate = self._candidates.get(symbol)

        if candidate is None or candidate.status != "OBSERVING":
            return None

        if candidate.elapsed_sec >= cfg.max_observation_minutes * 60.0:
            candidate.status = "EXPIRED"
            log.info(
                f"[{self.name}] {symbol} EXPIRED after {candidate.elapsed_sec / 60.0:.1f}m — discarding"
            )
            async with self._lock:
                self._candidates.pop(symbol, None)
            return None

        market = await self._market_data.get(symbol)
        if not market:
            return None

        price = float(market["last_price"])
        candidate.last_checked_at = time.time()
        candidate.entry_price = price

        fetch_count = max(
            cfg.bb_period + 1,
            cfg.rsi_period + 1,
            cfg.atr_period + 1,
            cfg.ema_slow_period,
        ) + cfg.candle_fetch_buffer

        try:
            raw_candles = await self._candle_fetcher(symbol, cfg.trend_candle_bar, fetch_count)
        except Exception as exc:
            log.warning(f"[{self.name}] {symbol} — candle fetch failed: {exc}")
            return None

        forming_candle, closed_candles = _split_forming_and_closed(raw_candles)
        del forming_candle  # we only use closed candles for indicators

        closed_oldest = sorted(closed_candles, key=lambda c: c.get("ts", 0))
        min_required = max(cfg.bb_period + 1, cfg.rsi_period + 1, cfg.atr_period + 1)
        if len(closed_oldest) < min_required:
            return None

        closes = [c["close"] for c in closed_oldest]

        ema_fast = compute_ema(closes, cfg.ema_fast_period)
        ema_slow = compute_ema(closes, cfg.ema_slow_period)
        rsi = compute_rsi(closes, cfg.rsi_period)
        bb_lower, bb_mid, bb_upper, bb_width_pct = compute_bollinger_bands(
            closes, cfg.bb_period, cfg.bb_std
        )
        atr_pct = compute_atr_pct(closed_oldest, cfg.atr_period)

        if bb_lower is None or bb_upper is None or atr_pct is None:
            return None

        candidate.ema_fast = ema_fast
        candidate.ema_slow = ema_slow
        candidate.rsi = rsi
        candidate.bb_lower = bb_lower
        candidate.bb_mid = bb_mid
        candidate.bb_upper = bb_upper
        candidate.bb_width_pct = bb_width_pct
        candidate.atr_pct = atr_pct

        try:
            window_trades = await self._trade_store.get_window(symbol, cfg.trade_window_ms)
        except Exception as exc:
            log.warning(f"[{self.name}] {symbol} — trade window fetch failed: {exc}")
            return None

        was_ready = candidate.data_ready
        candidate.data_ready = (
            candidate.elapsed_sec >= cfg.min_data_warmup_sec
            and len(window_trades) >= cfg.min_data_trade_count
        )
        if candidate.data_ready and not was_ready:
            log.info(
                f"[{self.name}] {symbol} order-flow warm-up complete "
                f"({len(window_trades)} trades in window)"
            )
        if not candidate.data_ready:
            return None

        # Volatility gate: skip if ATR is too small to reliably reach TP
        if atr_pct < cfg.min_atr_pct:
            candidate.engine_used = "neutral"
            candidate.direction = ""
            return None

        long_flow = compute_order_flow_imbalance(
            window_trades, "long", cfg.flow_bucket_count
        )
        short_flow = compute_order_flow_imbalance(
            window_trades, "short", cfg.flow_bucket_count
        )

        candidate.flow_long_pct = long_flow["strength_pct"]
        candidate.flow_short_pct = short_flow["strength_pct"]
        candidate.flow_long_ratio = long_flow["current_ratio"]
        candidate.flow_short_ratio = short_flow["current_ratio"]

        # Long: price near lower band, RSI oversold, buyers stepping in
        long_bb_ok = price <= bb_lower * (1.0 + cfg.bb_skin_pct)
        long_rsi_ok = rsi <= cfg.rsi_oversold
        long_flow_ok = (
            long_flow["strength_pct"] >= cfg.flow_strength_min
            and long_flow["current_ratio"] >= cfg.flow_ratio_min
            and (not cfg.flow_burst_required or long_flow["burst"])
        )

        # Short: price near upper band, RSI overbought, sellers stepping in
        short_bb_ok = price >= bb_upper * (1.0 - cfg.bb_skin_pct)
        short_rsi_ok = rsi >= cfg.rsi_overbought
        short_flow_ok = (
            short_flow["strength_pct"] >= cfg.flow_strength_min
            and short_flow["current_ratio"] >= cfg.flow_ratio_min
            and (not cfg.flow_burst_required or short_flow["burst"])
        )

        if long_bb_ok and long_rsi_ok and long_flow_ok:
            direction = "long"
            flow = long_flow
            band_label = "lower"
            rsi_label = "oversold"
            band_value = bb_lower
        elif short_bb_ok and short_rsi_ok and short_flow_ok:
            direction = "short"
            flow = short_flow
            band_label = "upper"
            rsi_label = "overbought"
            band_value = bb_upper
        else:
            candidate.direction = ""
            candidate.engine_used = ""
            return None

        candidate.direction = direction
        candidate.engine_used = "bb_rsi_flow_scalp"

        tp_distance = cfg.tp_distance_pct
        sl_distance = cfg.sl_distance_pct

        if direction == "long":
            take_profit = price * (1.0 + tp_distance)
            stop_loss = price * (1.0 - sl_distance)
        else:
            take_profit = price * (1.0 - tp_distance)
            stop_loss = price * (1.0 + sl_distance)

        log.info(
            f"[{self.name}] SIGNAL {direction.upper()} {symbol}\n"
            f"  entry={price:.6g} target={take_profit:.6g} distance={tp_distance:+.2%}\n"
            f"  Bollinger {band_label} touch price={price:.6g} band={band_value:.6g}\n"
            f"  RSI {rsi:.1f} ({rsi_label})\n"
            f"  flow_strength={flow['strength_pct']:.0f}% flow_ratio={flow['current_ratio']:.2f} burst={flow['burst']}\n"
            f"  ATR={atr_pct:.2%} BB_width={bb_width_pct:.2%}"
        )

        async with self._lock:
            self._candidates.pop(symbol, None)

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=1.0,
            entry_price=price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            timestamp=time.time(),
            reasons=[
                "engine=orderflow_bb_scalper",
                f"bb_{band_label}_touch",
                f"rsi={rsi:.1f}",
                f"flow_imbalance={flow['strength_pct']:.0f}%",
                f"flow_ratio={flow['current_ratio']:.2f}",
                f"flow_zscore={flow['zscore']:.2f}",
                f"flow_burst={flow['burst']}",
                f"atr_pct={atr_pct:.2%}",
                f"tp_distance_pct={tp_distance:.2%}",
            ],
        )


def build(ctx: StrategyContext) -> OrderFlowBollingerScalperEngine:
    """strategy.load_strategy() entry point."""
    cfg = ctx.build_config(OrderFlowBollingerScalperConfig)
    return OrderFlowBollingerScalperEngine(
        ctx.trade_store,
        ctx.market_data,
        ctx.candle_fetcher,
        config=cfg,
    )