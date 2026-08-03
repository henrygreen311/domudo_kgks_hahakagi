"""
Additional pre-trade safety filter, checked after the liquidation-distance
guard (liquidation_guard.py) passes and before any order is submitted:

Liquidation-touch history — the estimated liquidation price must not have
been touched more than once in the last 5 hours of 5-minute candles.
Repeated visits to that level are a sign this symbol chops through that
price regularly, not that it would take a single genuine adverse move to
get there.

The take-profit reachability check that used to run alongside this one
(requiring the planned TP price to have already been touched in the last
hour) has been removed — it's no longer imported or called anywhere in
the pipeline. `ExecutionConfig.enable_tp_reachability_check`,
`tp_validation_lookback_hours`, `min_tp_hits_required`, and
`estimated_fee_by_leverage` are gone from execution_engine.py accordingly.

Like liquidation_guard.py, this is a pure function: candles and prices are
supplied by the caller (ExecutionEngine), nothing here talks to the
exchange directly.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class LiquidationHistoryResult:
    approved: bool
    hits: int
    reason: str = ""


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


def validate_liquidation_history(
    liquidation_price: float,
    direction: str,
    candles: List[dict],
    max_hits: int = 1,
    lookback_hours: float = 5.0,
) -> LiquidationHistoryResult:
    hits = count_price_hits(candles, liquidation_price, direction, touches="liq")
    if hits > max_hits:
        return LiquidationHistoryResult(
            approved=False,
            hits=hits,
            reason=(
                f"estimated liquidation price {liquidation_price:.8f} was touched {hits} times in "
                f"the last {lookback_hours:.0f}h of 5m candles (max allowed {max_hits}) — this level "
                f"gets revisited too often to trust a single-touch estimate"
            ),
        )
    return LiquidationHistoryResult(approved=True, hits=hits)
