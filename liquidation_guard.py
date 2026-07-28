"""
Pre-trade liquidation-distance guard.

Checked once per candidate trade, right after sizing/leverage is resolved
in ExecutionEngine and before any order is submitted. Its only job: predict
how many price ticks separate the estimated entry price from the estimated
liquidation price, and say yes/no. It never talks to the exchange itself
and never depends on execution_engine.py's internals — same "compute
something from data the caller already fetched" shape as
compute_order_flow_metrics() in market_data.py.

Why this exists
----------------
OP-USDT-SWAP order #181 opened and closed in the same second (10:52:16 to
10:52:16) — liquidated 13 ticks from its entry (entry 0.08823, liq 0.08691,
tick 0.0001). A leveraged position that close to liquidation before it even
opens isn't a real trade, it's a coin-flip against noise. This guard
rejects candidates like that before they're ever submitted, instead of
discovering it after the fact from a zero-duration position row.

Liquidation-price formula
--------------------------
Standard isolated-margin approximation (ignores funding fees and the exact
moment-of-liquidation fee deduction — this is an estimate, not a
prediction of OKX's number to the tick, same disclaimer OKX's own in-app
calculator carries: "Calculations are for reference only and not based on
real market data."):

    long:  liq_price = entry_price * (1 - 1/leverage + mmr)
    short: liq_price = entry_price * (1 + 1/leverage - mmr)

Sanity-checked against the OP-USDT-SWAP liquidation above: solving for mmr
with entry=0.08823, liq=0.08691, leverage=50 gives mmr ~= 0.5%, which is a
realistic tier-1 maintenance margin rate for a low-cap alt — the formula
tracks OKX's real behavior closely enough to gate on.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass
class LiquidationCheckResult:
    approved: bool
    liquidation_price: float
    distance_ticks: float
    reason: str = ""


def estimate_liquidation_price(entry_price: float, leverage: float, mmr: float, direction: str) -> float:
    """Isolated-margin liquidation price estimate — see module docstring."""
    initial_margin_rate = 1.0 / leverage
    if direction == "long":
        return entry_price * (1.0 - initial_margin_rate + mmr)
    return entry_price * (1.0 + initial_margin_rate - mmr)


def _tick_size(tick_str) -> float:
    try:
        return float(Decimal(str(tick_str)))
    except (InvalidOperation, TypeError, ValueError):
        return 0.0


def check_liquidation_distance(
    entry_price: float,
    leverage: float,
    mmr: float,
    direction: str,
    tick_size,
    min_distance_ticks: float,
) -> LiquidationCheckResult:
    """The single entry point ExecutionEngine calls. Returns approved=False
    (with a human-readable reason) whenever the estimated liquidation price
    sits fewer than `min_distance_ticks` ticks from the estimated entry —
    or whenever any input is too broken to evaluate at all, since an
    unknown distance should never be treated as a safe one."""
    tick = _tick_size(tick_size)
    if entry_price <= 0 or leverage <= 0 or tick <= 0:
        return LiquidationCheckResult(
            approved=False, liquidation_price=0.0, distance_ticks=0.0,
            reason="invalid entry_price/leverage/tick_size — cannot evaluate, rejecting to be safe",
        )

    liq_price = estimate_liquidation_price(entry_price, leverage, mmr, direction)
    distance_ticks = abs(entry_price - liq_price) / tick

    if distance_ticks < min_distance_ticks:
        return LiquidationCheckResult(
            approved=False,
            liquidation_price=liq_price,
            distance_ticks=distance_ticks,
            reason=(
                f"estimated liquidation only {distance_ticks:.1f} ticks from entry "
                f"(need >= {min_distance_ticks:.0f}) — discarding, too close to survive normal noise"
            ),
        )
    return LiquidationCheckResult(approved=True, liquidation_price=liq_price, distance_ticks=distance_ticks)
