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
from typing import List, Optional, Tuple


@dataclass
class LiquidationCheckResult:
    approved: bool
    liquidation_price: float
    distance_ticks: float
    reason: str = ""
    # Populated by check_liquidation_distance_pct()/select_leverage_with_safe_liquidation()
    # below; left None by the original tick-based check_liquidation_distance()
    # above, which doesn't have a notion of "chosen leverage" since it's
    # only ever given one to check.
    leverage: Optional[float] = None
    distance_pct: Optional[float] = None


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


def check_liquidation_distance_pct(
    entry_price: float,
    leverage: float,
    mmr: float,
    direction: str,
    min_distance_pct: float,
) -> LiquidationCheckResult:
    """Percentage-distance counterpart to check_liquidation_distance()
    above — same estimate_liquidation_price() math, but gates on how far
    the liquidation price sits as a *fraction of entry_price* rather than
    a fixed tick count. A fixed tick count means very different things in
    relative terms across symbols and leverages — see
    select_leverage_with_safe_liquidation() below for why that mattered
    here specifically."""
    if entry_price <= 0 or leverage <= 0:
        return LiquidationCheckResult(
            approved=False, liquidation_price=0.0, distance_ticks=0.0, leverage=leverage,
            reason="invalid entry_price/leverage — cannot evaluate, rejecting to be safe",
        )

    liq_price = estimate_liquidation_price(entry_price, leverage, mmr, direction)
    distance_pct = abs(entry_price - liq_price) / entry_price

    if distance_pct < min_distance_pct:
        return LiquidationCheckResult(
            approved=False, liquidation_price=liq_price, distance_ticks=0.0,
            leverage=leverage, distance_pct=distance_pct,
            reason=(
                f"at {leverage:.0f}x, estimated liquidation is only {distance_pct:.2%} from entry "
                f"(need >= {min_distance_pct:.0%})"
            ),
        )
    return LiquidationCheckResult(
        approved=True, liquidation_price=liq_price, distance_ticks=0.0,
        leverage=leverage, distance_pct=distance_pct,
    )


def select_leverage_with_safe_liquidation(
    entry_price: float,
    direction: str,
    candidates: List[Tuple[float, float]],  # [(leverage, mmr), ...], most-preferred first
    min_distance_pct: float,
) -> LiquidationCheckResult:
    """Tries each (leverage, mmr) candidate in the order given — most
    preferred first, e.g. [(50, mmr_at_50x), (10, mmr_at_10x)] — and
    returns the first whose estimated liquidation price sits at least
    min_distance_pct away from entry_price. If none qualify, returns
    approved=False carrying the closest miss's numbers, so the rejection
    log says something concrete instead of just "nothing worked".

    Why per-symbol dynamic leverage instead of one fixed leverage for
    everything: position_history showed KAITO-USDT-SWAP liquidating
    repeatedly at 50x (entry ~1.22, liquidated after roughly a 1%
    adverse move) while ETH-USDT-SWAP survived the same 50x fine — they
    don't carry the same maintenance margin rate or typical volatility,
    so "50x" isn't the same safety margin across pairs. Blanket-forcing
    50x everywhere reliably liquidates the volatile ones; blanket-capping
    everything at a lower leverage gives up real size on pairs where 50x
    genuinely is safe (matches the original "50x is fine for ETH"
    observation). Trying 50x first and only falling back to a lower
    leverage when 50x isn't safe *for this specific symbol* keeps both.

    mmr is fetched by the caller (ExecutionEngine, which talks to the
    exchange) for each candidate's resulting notional and passed in
    already resolved — this module stays exchange-agnostic per its own
    stated design (see module docstring), so it never calls the
    exchange itself.
    """
    if not candidates:
        return LiquidationCheckResult(
            approved=False, liquidation_price=0.0, distance_ticks=0.0,
            reason="no candidate leverages available to evaluate",
        )

    closest: Optional[LiquidationCheckResult] = None
    for leverage, mmr in candidates:
        result = check_liquidation_distance_pct(entry_price, leverage, mmr, direction, min_distance_pct)
        if result.approved:
            return result
        if closest is None or (result.distance_pct or 0.0) > (closest.distance_pct or 0.0):
            closest = result

    tried = ", ".join(f"{lev:.0f}x" for lev, _ in candidates)
    closest.reason = (
        f"none of the tried leverages ({tried}) keep liquidation >= {min_distance_pct:.0%} from entry — "
        f"closest was {closest.reason}"
    )
    return closest
