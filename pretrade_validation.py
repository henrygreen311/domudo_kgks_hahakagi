"""
Additional pre-trade safety filters, checked after the liquidation-distance
guard (liquidation_guard.py) passes and before any order is submitted:

1. Take-profit reachability — the planned TP price must have actually been
   touched (candle high, for a favorable move) at least once in the last
   hour of 5-minute candles, or the trade is discarded. A TP level the
   market hasn't reached in an hour isn't a realistic target for right
   now, whatever the theoretical math says.

2. Liquidation-touch history — the estimated liquidation price must not
   have been touched more than once in the last 15 hours of 5-minute
   candles. Repeated visits to that level are a sign this symbol chops
   through that price regularly, not that it would take a single genuine
   adverse move to get there.

Like liquidation_guard.py, these are pure functions: candles and prices are
supplied by the caller (ExecutionEngine), nothing here talks to the
exchange directly.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TpValidationResult:
    approved: bool
    hits: int
    planned_tp_price: float
    reason: str = ""


@dataclass
class LiquidationHistoryResult:
    approved: bool
    hits: int
    reason: str = ""


def planned_take_profit_price(
    entry_price: float,
    direction: str,
    target_net_profit_usdt: float,
    estimated_fee_usdt: float,
    notional_usdt: float,
) -> float:
    """Same shape as ExecutionEngine._compute_take_profit_price, but using
    a flat pre-trade fee estimate (see ExecutionConfig.estimated_fee_by_leverage)
    and a notional derived from the planned, not-yet-filled entry price —
    there's no real fill or real fee to work from at this point, since
    this check runs before the order is even submitted."""
    if notional_usdt <= 0 or entry_price <= 0:
        return entry_price
    required_gross_profit = target_net_profit_usdt + estimated_fee_usdt
    price_move_frac = required_gross_profit / notional_usdt
    if direction == "long":
        return entry_price * (1 + price_move_frac)
    return entry_price * (1 - price_move_frac)


def count_price_hits(candles: List[dict], level: float, direction: str, touches: str) -> int:
    """Counts how many 5m candles reached or exceeded `level`. `touches`
    is "tp" (a favorable move for this direction) or "liq" (an adverse
    move) — that, combined with direction, determines which side of each
    candle's range to check:

        long  + tp:   candle high >= level
        long  + liq:  candle low  <= level
        short + tp:   candle low  <= level
        short + liq:  candle high >= level
    """
    if not candles or level <= 0:
        return 0
    favorable = touches == "tp"
    check_high = (direction == "long") == favorable
    hits = 0
    for c in candles:
        try:
            high = float(c["high"])
            low = float(c["low"])
        except (KeyError, TypeError, ValueError):
            continue
        if check_high:
            if high >= level:
                hits += 1
        else:
            if low <= level:
                hits += 1
    return hits


def validate_take_profit_reachable(
    entry_price: float,
    direction: str,
    target_net_profit_usdt: float,
    estimated_fee_usdt: float,
    notional_usdt: float,
    candles: List[dict],
    min_hits: int = 1,
) -> TpValidationResult:
    tp_price = planned_take_profit_price(
        entry_price, direction, target_net_profit_usdt, estimated_fee_usdt, notional_usdt
    )
    hits = count_price_hits(candles, tp_price, direction, touches="tp")
    if hits < min_hits:
        return TpValidationResult(
            approved=False,
            hits=hits,
            planned_tp_price=tp_price,
            reason=(
                f"planned take-profit {tp_price:.8f} was not reached even once in the last "
                f"hour of 5m candles (need >= {min_hits}) — not a realistic target right now"
            ),
        )
    return TpValidationResult(approved=True, hits=hits, planned_tp_price=tp_price)


def validate_liquidation_history(
    liquidation_price: float,
    direction: str,
    candles: List[dict],
    max_hits: int = 1,
) -> LiquidationHistoryResult:
    hits = count_price_hits(candles, liquidation_price, direction, touches="liq")
    if hits > max_hits:
        return LiquidationHistoryResult(
            approved=False,
            hits=hits,
            reason=(
                f"estimated liquidation price {liquidation_price:.8f} was touched {hits} times in "
                f"the last 15h of 5m candles (max allowed {max_hits}) — this level gets revisited "
                f"too often to trust a single-touch estimate"
            ),
        )
    return LiquidationHistoryResult(approved=True, hits=hits)
