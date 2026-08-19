"""
Support & Resistance Engine — simple, deterministic zone rejection.

Finds swing-based support/resistance zones from 5-minute candles, waits
for a completed candle to test one and reject back through it, and only
fires a Signal if the execution engine's real fee-aware TP target can be
reached before the next opposing zone. No RSI/MACD/VWAP/z-score/moving
averages — the whole read is swing geometry plus one completed candle.

Candles only, via ctx.candle_fetcher (okx_futures_client.get_candles) —
no ctx.trade_store use at all, so no REQUIRED_TRADE_WINDOW_MS is
declared (see strategy/base.py's module docstring for that convention).

THE FLOW (see evaluate())

  1. Fetch ~lookback_hours of closed 5m candles.
  2. Find swing highs/lows (a candle whose high/low is the extreme among
     its swing_fractal_width neighbors on each side) and cluster nearby
     swing prices into zones (zone_tolerance_pct). Zones need at least
     minimum_level_touches swing points to count.
  3. The most recently CLOSED candle is the only one ever checked for a
     setup — a still-forming candle is never used (see
     _split_forming_and_closed).
  4. LONG: that candle's low must have tested a support zone and its
     close must be back above it, green. SHORT: mirrors this off a
     resistance zone, red candle, close back below.
  5. TP FEASIBILITY (the critical gate): using ctx.margin_per_trade_usdt,
     ctx.default_leverage, ctx.target_net_profit_usdt, and a live-quoted
     taker fee rate, this replicates execution_engine.py's own fee-aware
     TP price math (see estimate_tp_sl_prices) — never a separate,
     invented target. If that estimated TP would sit at or beyond the
     next opposing zone's own boundary (no buffer — strict inequality
     only), the setup is rejected: tp_blocked_by_resistance /
     tp_blocked_by_support. The
     same math's SL price is sanity-checked against the zone's far edge,
     and the ratio of remaining room-to-zone vs that fixed SL distance
     must clear minimum_risk_reward.
  6. Only entry_price/take_profit/stop_loss placeholders go on the
     returned Signal, exactly like every other strategy here —
     execution_engine.py always computes the REAL TP/SL from the actual
     fill; this strategy's own TP/SL math above exists purely to decide
     WHETHER to trade, never to set the real order prices.

Cooldown + "no repeat signal off the same candle" (section 20 of the
spec this was built from) mirrors flow_ignition_engine.py's
last_signal_at pattern — the only state remembered between ticks.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from market_data import MarketDataStore, Signal, DEFAULT_SYMBOL_WHITELIST
from .base import StrategyContext, StrategyEngine

log = logging.getLogger("okx_futures.supportresistance")

CandleFetcher = Callable[[str, str, int], Awaitable[List[dict]]]


# ---------------------------------------------------------------------------
# Pure functions: candle housekeeping, swing detection, zone clustering,
# fee-aware TP/SL estimation. Nothing here talks to the network or holds
# state — evaluate() is what wires these together per tick.
# ---------------------------------------------------------------------------


def split_forming_and_closed(candles: List[dict]) -> Tuple[Optional[dict], List[dict]]:
    """Splits `candles` (any order) into (forming_or_None,
    closed_oldest_first). OKX's confirm flag: "0" = still forming,
    anything else (including missing/unknown) is treated as closed —
    conservative, never mistakes a live candle for a closed one."""
    forming = None
    closed: List[dict] = []
    for c in sorted(candles, key=lambda c: c.get("ts", 0)):
        if forming is None and str(c.get("confirm")) == "0":
            forming = c
        else:
            closed.append(c)
    return forming, closed


def find_swing_prices(closed_candles: List[dict], width: int) -> Tuple[List[float], List[float]]:
    """Classic fractal swing detection: candle i is a swing high if its
    high is the max among the `width` candles on each side of it (swing
    low mirrors this on lows). `closed_candles` must be oldest-first.
    Returns (swing_highs, swing_lows) as plain price lists — order/count
    only, no candle references kept."""
    n = len(closed_candles)
    swing_highs: List[float] = []
    swing_lows: List[float] = []
    if n < 2 * width + 1:
        return swing_highs, swing_lows

    for i in range(width, n - width):
        window = closed_candles[i - width : i + width + 1]
        high_i = closed_candles[i]["high"]
        low_i = closed_candles[i]["low"]
        if high_i >= max(c["high"] for c in window):
            swing_highs.append(high_i)
        if low_i <= min(c["low"] for c in window):
            swing_lows.append(low_i)
    return swing_highs, swing_lows


@dataclass
class Zone:
    low: float
    high: float
    touches: int

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0


def cluster_into_zones(prices: List[float], tolerance_pct: float) -> List[Zone]:
    """Greedily merges swing prices within tolerance_pct of a growing
    zone's own running average into one zone, rather than treating every
    swing price as its own level. Simple single-pass clustering, not a
    scoring system: sort ascending, extend the current zone while the
    next price is close enough to its running mean, otherwise start a
    new zone."""
    if not prices:
        return []
    ordered = sorted(prices)
    zones: List[Zone] = []
    bucket = [ordered[0]]

    def flush(b: List[float]) -> Zone:
        return Zone(low=min(b), high=max(b), touches=len(b))

    for price in ordered[1:]:
        running_mean = sum(bucket) / len(bucket)
        if abs(price - running_mean) / running_mean <= tolerance_pct:
            bucket.append(price)
        else:
            zones.append(flush(bucket))
            bucket = [price]
    zones.append(flush(bucket))
    return zones


def nearest_zone_below(zones: List[Zone], price: float) -> Optional[Zone]:
    candidates = [z for z in zones if z.high <= price]
    return max(candidates, key=lambda z: z.high) if candidates else None


def nearest_zone_above(zones: List[Zone], price: float) -> Optional[Zone]:
    candidates = [z for z in zones if z.low >= price]
    return min(candidates, key=lambda z: z.low) if candidates else None


def zone_being_tested(zones: List[Zone], price: float, tolerance_pct: float) -> Optional[Zone]:
    """The zone `price` is actively testing: either price sits inside
    the zone's [low, high] range, or is within tolerance_pct of its
    nearest edge (a brief wick through/near the level still counts as a
    test). Picks the closest qualifying zone if more than one is close
    enough."""
    best: Optional[Zone] = None
    best_dist = None
    for z in zones:
        if z.low <= price <= z.high:
            dist = 0.0
        else:
            dist = min(abs(price - z.low), abs(price - z.high))
            if dist / z.mid > tolerance_pct:
                continue
        if best_dist is None or dist < best_dist:
            best, best_dist = z, dist
    return best


def next_opposing_resistance(zones: List[Zone], entry_price: float) -> Optional[Zone]:
    """The FIRST meaningful resistance above the actual entry: closest
    zone whose LOW is strictly above entry_price. A zone whose low sits
    at or below entry overlaps (or is on the wrong side of) the entry
    and is never eligible, no matter how "near" it is in the full zone
    list."""
    candidates = [z for z in zones if z.low > entry_price]
    return min(candidates, key=lambda z: z.low) if candidates else None


def next_opposing_support(zones: List[Zone], entry_price: float) -> Optional[Zone]:
    """Mirror of next_opposing_resistance for SHORTs: closest zone whose
    HIGH is strictly below entry_price."""
    candidates = [z for z in zones if z.high < entry_price]
    return max(candidates, key=lambda z: z.high) if candidates else None


def check_tp_and_risk_reward(
    direction: str,
    entry_price: float,
    est_tp: float,
    est_sl: float,
    tested_zone: Zone,
    opposing_zone: Zone,
    minimum_risk_reward: float,
) -> Tuple[bool, str, float, float]:
    """Pure decision core called by evaluate() once the opposing zone
    and estimated TP/SL are known. No safety buffer: TP must clear the
    opposing zone's own boundary with strict inequality (TP resting
    exactly on the boundary is a reject, per spec). Risk/reward is
    measured off that same boundary, not off the estimated TP, and
    must clear minimum_risk_reward. Returns
    (accepted, reject_reason, reward_room, risk_room); reject_reason is
    "" when accepted."""
    if direction == "long":
        tp_ok = est_tp < opposing_zone.low
        reward_room = opposing_zone.low - entry_price
        risk_room = entry_price - est_sl
        sl_ok = est_sl < tested_zone.low
        blocked_reason = "tp_blocked_by_resistance"
    else:
        tp_ok = est_tp > opposing_zone.high
        reward_room = entry_price - opposing_zone.high
        risk_room = est_sl - entry_price
        sl_ok = est_sl > tested_zone.high
        blocked_reason = "tp_blocked_by_support"

    if not tp_ok:
        return False, blocked_reason, reward_room, risk_room
    if not sl_ok:
        return False, "sl_inside_opposing_zone", reward_room, risk_room
    if risk_room <= 0 or (reward_room / risk_room) < minimum_risk_reward:
        return False, "insufficient_risk_reward", reward_room, risk_room
    return True, "", reward_room, risk_room


def estimate_tp_sl_prices(
    direction: str,
    entry_price: float,
    margin_usdt: float,
    leverage: float,
    target_net_profit_usdt: float,
    target_stop_loss_usdt: float,
    taker_fee_rate: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Mirrors execution_engine.py's _compute_take_profit_price /
    _compute_stop_loss_price math exactly (same fee-aware formula), but
    with ESTIMATED inputs since no fill exists yet — purely for this
    strategy's own pre-trade feasibility gate. The real TP/SL are always
    computed later by execution_engine.py from the actual fill; nothing
    here ever overrides that. Returns (None, None) if margin/leverage
    don't give a usable notional."""
    notional = margin_usdt * leverage
    if notional <= 0:
        return None, None

    estimated_opening_fee = taker_fee_rate * notional
    estimated_total_fees = estimated_opening_fee * 2.0

    tp_price_move_frac = (target_net_profit_usdt + estimated_total_fees) / notional
    sl_required_gross_loss = max(target_stop_loss_usdt - estimated_total_fees, 0.0)
    sl_price_move_frac = sl_required_gross_loss / notional

    if direction == "long":
        tp_price = entry_price * (1 + tp_price_move_frac)
        sl_price = entry_price * (1 - sl_price_move_frac)
    else:
        tp_price = entry_price * (1 - tp_price_move_frac)
        sl_price = entry_price * (1 + sl_price_move_frac)
    return tp_price, sl_price


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SupportResistanceConfig:
    candle_bar: str = "5m"
    lookback_hours: float = 5.0
    candle_fetch_buffer: int = 5

    swing_fractal_width: int = 2
    zone_tolerance_pct: float = 0.0015
    minimum_level_touches: int = 2
    # Only exactly 1 is implemented today (see module docstring) -- kept
    # as a config field per spec rather than hardcoded, for whenever a
    # multi-candle confirmation variant gets built.
    confirmation_candle_count: int = 1

    minimum_risk_reward: float = 2.0

    cooldown_sec: float = 180.0

    # Fallbacks used only if StrategyContext didn't supply the real
    # execution-side numbers (see estimate_tp_sl_prices) -- normally
    # tracker.py always provides these, so these rarely if ever apply.
    fallback_margin_usdt: float = 5.0
    fallback_leverage: int = 10
    fallback_target_net_profit_usdt: float = 0.5
    fallback_target_stop_loss_usdt: float = 0.9
    fallback_taker_fee_rate: float = 0.0005

    symbol_whitelist: Optional[frozenset] = field(default_factory=lambda: DEFAULT_SYMBOL_WHITELIST)


@dataclass
class SymbolSRState:
    symbol: str
    direction: str = ""
    support: Optional[Zone] = None
    resistance: Optional[Zone] = None
    last_price: float = 0.0
    last_signal_at: float = 0.0
    last_signal_candle_ts: Optional[int] = None
    last_reject_reason: str = ""

    def status_line(self) -> str:
        sup = f"{self.support.low:.6g}-{self.support.high:.6g}" if self.support else "-"
        res = f"{self.resistance.low:.6g}-{self.resistance.high:.6g}" if self.resistance else "-"
        base = (
            f"{self.symbol} price={self.last_price:.6g} support={sup} resistance={res} "
            f"direction={self.direction or '-'}"
        )
        if self.last_reject_reason:
            base += f" reject={self.last_reject_reason}"
        return base


class SupportResistanceEngine(StrategyEngine):
    """Implements strategy.base.StrategyEngine — see this module's
    docstring for the full flow. Switch to this strategy by setting
    tracker.py's STRATEGY_NAME = "support_resistance_engine"."""

    name = "support_resistance_engine"

    def __init__(
        self,
        market_data: MarketDataStore,
        candle_fetcher: CandleFetcher,
        okx_client=None,
        config: Optional[SupportResistanceConfig] = None,
        margin_per_trade_usdt: Optional[float] = None,
        default_leverage: Optional[int] = None,
        target_net_profit_usdt: Optional[float] = None,
        target_stop_loss_usdt: Optional[float] = None,
    ) -> None:
        self._market_data = market_data
        self._candle_fetcher = candle_fetcher
        self._okx_client = okx_client
        self.config = config or SupportResistanceConfig()
        self._margin_usdt = margin_per_trade_usdt if margin_per_trade_usdt is not None else self.config.fallback_margin_usdt
        self._leverage = default_leverage if default_leverage is not None else self.config.fallback_leverage
        self._target_net_profit_usdt = (
            target_net_profit_usdt if target_net_profit_usdt is not None else self.config.fallback_target_net_profit_usdt
        )
        self._target_stop_loss_usdt = (
            target_stop_loss_usdt if target_stop_loss_usdt is not None else self.config.fallback_target_stop_loss_usdt
        )
        self._states: Dict[str, SymbolSRState] = {}
        self._fee_rate_cache: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def sync_watchlist(self, watchlist_symbols) -> None:
        watchlist_symbols = set(watchlist_symbols)
        whitelist = self.config.symbol_whitelist
        if whitelist:
            watchlist_symbols &= whitelist
        async with self._lock:
            for symbol in watchlist_symbols:
                if symbol not in self._states:
                    self._states[symbol] = SymbolSRState(symbol=symbol)
                    log.info(f"[support_resistance] {symbol} added — watching for S/R zone rejections")
            for symbol in [s for s in self._states if s not in watchlist_symbols]:
                del self._states[symbol]
                self._fee_rate_cache.pop(symbol, None)

    async def snapshot(self) -> List[SymbolSRState]:
        async with self._lock:
            return list(self._states.values())

    async def _get_fee_rate(self, symbol: str) -> float:
        if symbol in self._fee_rate_cache:
            return self._fee_rate_cache[symbol]
        rate = self.config.fallback_taker_fee_rate
        if self._okx_client is not None:
            try:
                info = await self._okx_client.get_trade_fee_rate(symbol)
                rate = float(info.get("taker_fee_rate", rate))
            except Exception as exc:
                log.debug(f"[support_resistance] {symbol} — could not fetch live fee rate, using fallback: {exc}")
        self._fee_rate_cache[symbol] = rate
        return rate

    async def evaluate(self, symbol: str) -> Optional[Signal]:
        cfg = self.config
        async with self._lock:
            state = self._states.get(symbol)
        if state is None:
            return None

        market = await self._market_data.get(symbol)
        if not market:
            return None
        price = market["last_price"]
        state.last_price = price

        limit = int((cfg.lookback_hours * 3600) / 300) + cfg.candle_fetch_buffer
        try:
            raw_candles = await self._candle_fetcher(symbol, cfg.candle_bar, limit)
        except Exception as exc:
            log.warning(f"[support_resistance] {symbol} — could not fetch candles: {exc}")
            return None

        _, closed = split_forming_and_closed(raw_candles)
        if len(closed) < 2 * cfg.swing_fractal_width + 2:
            state.last_reject_reason = "not_enough_candles"
            return None

        swing_highs, swing_lows = find_swing_prices(closed, cfg.swing_fractal_width)
        resistance_zones = [z for z in cluster_into_zones(swing_highs, cfg.zone_tolerance_pct) if z.touches >= cfg.minimum_level_touches]
        support_zones = [z for z in cluster_into_zones(swing_lows, cfg.zone_tolerance_pct) if z.touches >= cfg.minimum_level_touches]

        signal_candle = closed[-1]

        direction: str = ""
        tested_zone: Optional[Zone] = None
        opposing_zones: List[Zone] = []

        support = zone_being_tested(support_zones, signal_candle["low"], cfg.zone_tolerance_pct)
        resistance = zone_being_tested(resistance_zones, signal_candle["high"], cfg.zone_tolerance_pct)

        is_green = signal_candle["close"] > signal_candle["open"]
        is_red = signal_candle["close"] < signal_candle["open"]

        if support is not None and is_green and signal_candle["close"] > support.high:
            direction = "long"
            tested_zone = support
            opposing_zones = resistance_zones
        elif resistance is not None and is_red and signal_candle["close"] < resistance.low:
            direction = "short"
            tested_zone = resistance
            opposing_zones = support_zones

        state.support = support if support is not None else nearest_zone_below(support_zones, signal_candle["close"])
        state.resistance = resistance if resistance is not None else nearest_zone_above(resistance_zones, signal_candle["close"])

        if not direction:
            state.direction = ""
            state.last_reject_reason = "no_price_rejection"
            return None

        candle_ts = signal_candle.get("ts")
        if state.last_signal_candle_ts == candle_ts:
            state.last_reject_reason = "already_signaled_this_candle"
            return None
        now = time.time()
        if now - state.last_signal_at < cfg.cooldown_sec:
            state.last_reject_reason = "cooldown"
            return None

        opposing = next_opposing_resistance(opposing_zones, price) if direction == "long" else next_opposing_support(opposing_zones, price)
        if opposing is None:
            state.last_reject_reason = "no_next_resistance" if direction == "long" else "no_next_support"
            return None

        fee_rate = await self._get_fee_rate(symbol)
        est_tp, est_sl = estimate_tp_sl_prices(
            direction=direction,
            entry_price=price,
            margin_usdt=self._margin_usdt,
            leverage=self._leverage,
            target_net_profit_usdt=self._target_net_profit_usdt,
            target_stop_loss_usdt=self._target_stop_loss_usdt,
            taker_fee_rate=fee_rate,
        )
        if est_tp is None or est_sl is None:
            state.last_reject_reason = "no_notional"
            return None

        accepted, reject_reason, reward_room, risk_room = check_tp_and_risk_reward(
            direction=direction,
            entry_price=price,
            est_tp=est_tp,
            est_sl=est_sl,
            tested_zone=tested_zone,
            opposing_zone=opposing,
            minimum_risk_reward=cfg.minimum_risk_reward,
        )
        if not accepted:
            state.last_reject_reason = reject_reason
            return None

        state.direction = direction
        state.last_signal_at = now
        state.last_signal_candle_ts = candle_ts
        state.last_reject_reason = ""

        risk_reward = reward_room / risk_room
        if direction == "long":
            confirmation = "green_5m_rejection"
            reasons = [
                "engine=support_resistance",
                f"support={tested_zone.low:.6g}-{tested_zone.high:.6g}",
                f"next_resistance={opposing.low:.6g}-{opposing.high:.6g}",
                f"entry={price:.6g}",
                f"confirmation={confirmation}",
                f"tp={est_tp:.6g}",
                f"sl={est_sl:.6g}",
                f"risk_reward={risk_reward:.2f}",
            ]
        else:
            confirmation = "red_5m_rejection"
            reasons = [
                "engine=support_resistance",
                f"resistance={tested_zone.low:.6g}-{tested_zone.high:.6g}",
                f"next_support={opposing.low:.6g}-{opposing.high:.6g}",
                f"entry={price:.6g}",
                f"confirmation={confirmation}",
                f"tp={est_tp:.6g}",
                f"sl={est_sl:.6g}",
                f"risk_reward={risk_reward:.2f}",
            ]

        log.info(f"[support_resistance] ACCEPTED {symbol} {direction.upper()} — {'; '.join(reasons)}")
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=1.0,
            entry_price=price,
            take_profit=price,  # unused — execution_engine computes its own TP/SL from the real fill
            stop_loss=price,
            timestamp=now,
            reasons=reasons,
        )


def build(ctx: StrategyContext) -> SupportResistanceEngine:
    """strategy.load_strategy()'s entry point — see strategy/base.py's
    module docstring for the contract every strategy module follows."""
    cfg = ctx.build_config(SupportResistanceConfig)
    return SupportResistanceEngine(
        ctx.market_data,
        ctx.candle_fetcher,
        okx_client=ctx.okx_client,
        config=cfg,
        margin_per_trade_usdt=ctx.margin_per_trade_usdt,
        default_leverage=ctx.default_leverage,
        target_net_profit_usdt=ctx.target_net_profit_usdt,
        target_stop_loss_usdt=ctx.target_stop_loss_usdt,
    )
