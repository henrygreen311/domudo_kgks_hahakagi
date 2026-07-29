"""
Continuous, category-weighted confidence engine for SignalGenerator.

Kept as a separate module (rather than folded into market_data.py) so
normalization, indicator scoring, and category aggregation can each be
unit tested and tuned independently — per the modularity goal in the
design doc. This module does NOT touch EventConfirmationEngine or the
prerequisite/quality gate in SignalGenerator.evaluate(); those stay
exactly as they are.

ConfidenceEngine's job, and only its job, is: given already-computed
market evidence (order flow metrics, book/liquidity metrics, momentum,
raw recent trades, ticker extremes) produce a confidence score with a
full breakdown. It does not gate execution timing (see
SignalPersistenceTracker in market_data.py) and does not touch the
prerequisite gate.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

Direction = Literal["long", "short", "neutral"]


# ---------------------------------------------------------------------------
# Normalization primitives
# ---------------------------------------------------------------------------
# Every indicator reduces its raw metric to (direction, strength), where
# strength is in [0.0, 1.0] and direction says which side it favors.
# Two shapes cover almost every metric already in this codebase:
#
# 1. Ratio metrics (buy/sell volume ratio, trade intensity ratio, band
#    liquidity ratio, whale size vs. average) are unbounded on (0, inf)
#    and symmetric in log space — a ratio of 2.0 is "as extreme" as 0.5
#    is in the opposite direction. Mapping log(ratio) through tanh gives
#    smooth saturation, so a 100x whale trade doesn't blow through
#    everything downstream the same way a 5x one wouldn't — this is the
#    direct fix for the "5.1x and 100x both count as one vote" problem.
#
# 2. Already-bounded metrics (aggressive buy %, book imbalance, which are
#    naturally in [0,1] or [-1,1]) just get rescaled/clipped around their
#    neutral point — no log needed since they can't diverge.


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def ratio_strength(ratio: float, scale: float = 1.5) -> float:
    """Map a ratio in (0, inf) to a *signed* strength in [-1, 1] via
    tanh(log(ratio) / log(scale)).

    `scale` controls how fast it saturates: a ratio equal to `scale`
    maps to tanh(1) ~= 0.76. Ratios below 1.0 (favoring the denominator
    side) come out negative. Handles ratio == 0 and ratio == inf.
    """
    if scale <= 1.0:
        raise ValueError("scale must be > 1.0")
    if ratio <= 0:
        return -1.0
    if math.isinf(ratio):
        return 1.0
    return math.tanh(math.log(ratio) / math.log(scale))


def linear_from_midpoint(value: float, midpoint: float = 0.5, span: float = 0.5) -> float:
    """Map a bounded metric (e.g. a fraction in [0,1]) to a signed
    strength in [-1, 1] around a neutral midpoint.

    E.g. aggressive_buy_pct of 0.5 is neutral, 1.0 is maximally long,
    0.0 is maximally short.
    """
    if span <= 0:
        raise ValueError("span must be > 0")
    return clip((value - midpoint) / span, -1.0, 1.0)


def signed_to_direction_strength(signed: float) -> "tuple[Direction, float]":
    """Split a signed strength in [-1, 1] into (direction, magnitude in [0,1])."""
    if signed > 0:
        return "long", clip(signed, 0.0, 1.0)
    if signed < 0:
        return "short", clip(-signed, 0.0, 1.0)
    return "neutral", 0.0


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class Metric:
    """A single indicator's contribution: which category it belongs to,
    which side it favors, and how strongly — the continuous replacement
    for a single boolean entry in the old long_checks/short_checks dicts.
    """

    name: str
    category: str
    direction: Direction
    strength: float  # magnitude in [0, 1]; meaningless if direction == "neutral"
    raw_value: Optional[float] = None  # kept only for logging/debugging

    def long_strength(self) -> float:
        return self.strength if self.direction == "long" else 0.0

    def short_strength(self) -> float:
        return self.strength if self.direction == "short" else 0.0


@dataclass
class CategoryScore:
    name: str
    weight: float
    long_score: float
    short_score: float
    metrics: List[Metric] = field(default_factory=list)


@dataclass
class ConfidenceResult:
    direction: Optional[Direction]  # None if neither side clears the bar
    confidence: float  # 0.0 - 1.0
    categories: List[CategoryScore] = field(default_factory=list)
    # --- explainability extras (PRIORITY 2/4/5/7 breakdown) ---
    long_confidence: float = 0.0
    short_confidence: float = 0.0
    edge: float = 0.0
    decay_factor: float = 1.0
    agreement_penalty_applied: bool = False
    overextension_penalty_applied: bool = False
    edge_ramp_penalty_applied: bool = False  # PRIORITY 4 (revised): soft edge ramp instead of hard cutoff

    def explain(self) -> str:
        """Human-readable breakdown, in the format from the design doc."""
        lines = [f"Direction: {(self.direction or 'none').upper()}", f"Confidence: {self.confidence:.0%}", ""]
        for cat in self.categories:
            side_score = cat.long_score if self.direction == "long" else cat.short_score
            lines.append(cat.name)
            for m in cat.metrics:
                s = m.long_strength() if self.direction == "long" else m.short_strength()
                lines.append(f"  - {m.name}: {s:.2f}")
            lines.append(f"  Category Score: {side_score:.2f} (weight {cat.weight:.2f})")
            lines.append("")
        lines.append(f"LONG confidence: {self.long_confidence:.0%}")
        lines.append(f"SHORT confidence: {self.short_confidence:.0%}")
        lines.append(f"Edge: {self.edge:.0%}")
        lines.append(f"Time decay factor: {self.decay_factor:.2f}")
        if self.edge_ramp_penalty_applied:
            lines.append("Edge ramp penalty applied (edge below min but above floor)")
        if self.agreement_penalty_applied:
            lines.append("Category agreement penalty applied")
        if self.overextension_penalty_applied:
            lines.append("Overextension penalty applied")
        lines.append(f"Final Confidence: {self.confidence:.0%}")
        return "\n".join(lines)

    def reason_lines(self) -> List[str]:
        """Compact per-category breakdown used as Signal.reasons — one line
        per category that actually contributed, plus the adjustments that
        fired, so every confidence number stays traceable back to evidence.
        """
        lines: List[str] = []
        for cat in self.categories:
            if not cat.metrics:
                continue
            side_score = cat.long_score if self.direction == "long" else cat.short_score
            metric_bits = ", ".join(
                f"{m.name}={(m.long_strength() if self.direction == 'long' else m.short_strength()):.2f}"
                for m in cat.metrics
            )
            lines.append(f"{cat.name} ({side_score:.2f}, weight {cat.weight:.2f}): {metric_bits}")
        lines.append(f"long={self.long_confidence:.2f} short={self.short_confidence:.2f} edge={self.edge:.2f}")
        lines.append(f"time_decay={self.decay_factor:.2f}")
        if self.edge_ramp_penalty_applied:
            lines.append("edge_ramp_penalty=applied")
        if self.agreement_penalty_applied:
            lines.append("category_agreement_penalty=applied")
        if self.overextension_penalty_applied:
            lines.append("overextension_penalty=applied")
        return lines


# ---------------------------------------------------------------------------
# Category aggregation
# ---------------------------------------------------------------------------
# Indicator functions take arbitrary keyword args (flow/liquidity/book/etc.,
# whatever they need) and return a Metric, or None if the underlying
# feature is disabled/unavailable for this evaluation. Categories average
# their members' strengths rather than summing them, which is the specific
# mechanism that stops a group of correlated indicators (e.g. three
# different order-flow ratios) from inflating confidence just by having
# more members than a category with fewer, less-correlated indicators.

IndicatorFn = Callable[..., Optional[Metric]]


@dataclass
class Category:
    name: str
    weight: float
    indicators: List[IndicatorFn]

    def score(self, **kwargs) -> CategoryScore:
        metrics = [m for fn in self.indicators if (m := fn(**kwargs)) is not None]
        if not metrics:
            return CategoryScore(name=self.name, weight=self.weight, long_score=0.0, short_score=0.0, metrics=[])
        long_score = sum(m.long_strength() for m in metrics) / len(metrics)
        short_score = sum(m.short_strength() for m in metrics) / len(metrics)
        return CategoryScore(name=self.name, weight=self.weight, long_score=long_score, short_score=short_score, metrics=metrics)


# ---------------------------------------------------------------------------
# PRIORITY 1 — Order Flow indicators (pure aggression, no volume/frequency)
# ---------------------------------------------------------------------------
# TradeStore.apply_trade already resolves the OKX `way`/`m` (maker) fields to
# a taker-side "buy"/"sell" *before* anything reaches this module, so
# flow["aggressive_buy_pct"] / flow["aggressive_sell_pct"] are already pure
# market-order (aggressor) participation, never resting limit volume. What
# needed fixing was elsewhere: whale size, trade intensity, and streak length
# describe *how much/how fast* the market is trading, not *which side is
# aggressing* — they were previously mixed into the same category as the
# pure aggression ratio, which let a single side's *volume* of activity
# masquerade as directional evidence. They now live in the Participation
# category (below) instead, so Order Flow is exclusively about aggression.

def indicator_aggressive_participation(flow: Optional[dict] = None, **_) -> Optional[Metric]:
    if not flow:
        return None
    total = flow.get("buy_volume", 0.0) + flow.get("sell_volume", 0.0)
    if total <= 0:
        return None
    signed = linear_from_midpoint(flow["aggressive_buy_pct"], midpoint=0.5, span=0.5)
    direction, strength = signed_to_direction_strength(signed)
    return Metric("aggressive_participation", "order_flow", direction, strength, raw_value=flow["aggressive_buy_pct"])


def indicator_flow_ratio(flow: Optional[dict] = None, cfg=None, **_) -> Optional[Metric]:
    if not flow:
        return None
    ratio = flow.get("buy_sell_ratio", 0.0)
    scale = getattr(cfg, "flow_ratio_scale", 1.5) or 1.5
    signed = ratio_strength(ratio, scale=max(scale, 1.0001))
    direction, strength = signed_to_direction_strength(signed)
    return Metric("aggressive_buy_sell_ratio", "order_flow", direction, strength, raw_value=ratio)


def indicator_vwap(
    trades: Optional[List[dict]] = None,
    price: Optional[float] = None,
    cfg=None,
    **_,
) -> Optional[Metric]:
    """PRIORITY 6 — normalized, rolling VWAP indicator.

    Price above VWAP gradually strengthens LONG; below gradually
    strengthens SHORT. Strength saturates at `vwap_max_deviation_pct` away
    from VWAP and is further damped by `vwap_strength_cap` so a single
    indicator inside an already-averaged category can't dominate the
    Order Flow score.
    """
    if not cfg or not getattr(cfg, "enable_vwap_indicator", False):
        return None
    if not trades or price is None or price <= 0:
        return None

    window_ms = getattr(cfg, "vwap_window_ms", 60_000)
    now_ms = time.time() * 1000.0
    cutoff = now_ms - window_ms
    windowed = [t for t in trades if t.get("timestamp", now_ms) >= cutoff]
    total_qty = sum(t["qty"] for t in windowed)
    if total_qty <= 0:
        return None
    vwap = sum(t["price"] * t["qty"] for t in windowed) / total_qty
    if vwap <= 0:
        return None

    deviation_pct = (price - vwap) / vwap
    max_dev = getattr(cfg, "vwap_max_deviation_pct", 0.003) or 0.003
    strength_cap = getattr(cfg, "vwap_strength_cap", 0.6)
    signed = clip(deviation_pct / max_dev, -1.0, 1.0) * strength_cap
    direction, strength = signed_to_direction_strength(signed)
    return Metric("vwap", "order_flow", direction, strength, raw_value=vwap)


# ---------------------------------------------------------------------------
# Participation indicators (whale size, trade intensity, streaks) —
# corroborating evidence about *how convicted* the current move is, kept
# separate from the pure aggression ratio in Order Flow (see PRIORITY 1).
# ---------------------------------------------------------------------------

def indicator_whale_dominance(flow: Optional[dict] = None, cfg=None, **_) -> Optional[Metric]:
    if not flow or not cfg or not getattr(cfg, "enable_whale_detection", False):
        return None
    buy_v, sell_v = flow.get("whale_buy_volume", 0.0), flow.get("whale_sell_volume", 0.0)
    if buy_v <= 0 and sell_v <= 0:
        return None
    ratio = (buy_v / sell_v) if sell_v > 0 else float("inf")
    signed = ratio_strength(ratio, scale=2.0) if sell_v > 0 or buy_v > 0 else 0.0
    direction, strength = signed_to_direction_strength(signed)
    return Metric("whale_dominance", "participation", direction, strength, raw_value=ratio)


def indicator_trade_intensity(flow: Optional[dict] = None, cfg=None, **_) -> Optional[Metric]:
    if not flow or not cfg or not getattr(cfg, "enable_trade_intensity", False):
        return None
    buy_rate, sell_rate = flow.get("buy_trades_per_sec", 0.0), flow.get("sell_trades_per_sec", 0.0)
    if buy_rate <= 0 and sell_rate <= 0:
        return None
    ratio = (buy_rate / sell_rate) if sell_rate > 0 else float("inf")
    scale = max(getattr(cfg, "intensity_dominance_ratio", 1.5), 1.0001)
    signed = ratio_strength(ratio, scale=scale)
    direction, strength = signed_to_direction_strength(signed)
    return Metric("trade_intensity", "participation", direction, strength, raw_value=ratio)


def indicator_streak(flow: Optional[dict] = None, cfg=None, **_) -> Optional[Metric]:
    if not flow or not cfg or not getattr(cfg, "enable_streak_detection", False):
        return None
    buy_streak, sell_streak = flow.get("current_buy_streak", 0), flow.get("current_sell_streak", 0)
    confirm_len = getattr(cfg, "streak_confirmation_length", 4) or 4
    if buy_streak == 0 and sell_streak == 0:
        return None
    signed_len = buy_streak if buy_streak > sell_streak else -sell_streak
    signed = clip(signed_len / confirm_len, -1.0, 1.0)
    direction, strength = signed_to_direction_strength(signed)
    return Metric("streak", "participation", direction, strength, raw_value=signed_len)


# ---------------------------------------------------------------------------
# Order Book indicators — aggregate depth imbalance (directional skew of
# the whole visible book).
# ---------------------------------------------------------------------------

def indicator_book_imbalance(liquidity: Optional[dict] = None, **_) -> Optional[Metric]:
    if not liquidity:
        return None
    imbalance = liquidity.get("imbalance", 0.0)  # already signed in [-1, 1]
    direction, strength = signed_to_direction_strength(imbalance)
    return Metric("book_imbalance", "order_book", direction, strength, raw_value=imbalance)


def indicator_book_ratio(liquidity: Optional[dict] = None, cfg=None, **_) -> Optional[Metric]:
    if not liquidity:
        return None
    ratio = liquidity.get("bid_ask_ratio", 0.0)
    signed = ratio_strength(ratio, scale=1.5)
    direction, strength = signed_to_direction_strength(signed)
    return Metric("book_bid_ask_ratio", "order_book", direction, strength, raw_value=ratio)


# ---------------------------------------------------------------------------
# Liquidity indicators — near-touch / distance-banded liquidity quality,
# distinct from the aggregate depth imbalance above.
# ---------------------------------------------------------------------------

def indicator_touch_liquidity(liquidity: Optional[dict] = None, **_) -> Optional[Metric]:
    if not liquidity:
        return None
    bid_sz, ask_sz = liquidity.get("best_bid_size", 0.0), liquidity.get("best_ask_size", 0.0)
    if bid_sz <= 0 and ask_sz <= 0:
        return None
    ratio = (bid_sz / ask_sz) if ask_sz > 0 else float("inf")
    signed = ratio_strength(ratio, scale=1.5)
    direction, strength = signed_to_direction_strength(signed)
    return Metric("touch_liquidity", "liquidity", direction, strength, raw_value=ratio)


def indicator_band_imbalance(liquidity: Optional[dict] = None, cfg=None, **_) -> Optional[Metric]:
    if not liquidity or not cfg or not getattr(cfg, "enable_book_imbalance_by_distance", False):
        return None
    bands = liquidity.get("band_imbalance") or {}
    check_band = getattr(cfg, "book_imbalance_check_band", None)
    band = bands.get(check_band) if check_band in bands else next(iter(bands.values()), None)
    if not band:
        return None
    direction, strength = signed_to_direction_strength(band.get("imbalance", 0.0))
    return Metric("band_imbalance", "liquidity", direction, strength, raw_value=band.get("imbalance"))


# ---------------------------------------------------------------------------
# Momentum indicators
# ---------------------------------------------------------------------------

def indicator_price_trend(price_change_pct: Optional[float] = None, cfg=None, **_) -> Optional[Metric]:
    if price_change_pct is None:
        return None
    min_move = getattr(cfg, "min_price_move_pct", 0.0005) or 0.0005
    signed = clip(price_change_pct / (min_move * 4.0), -1.0, 1.0)
    direction, strength = signed_to_direction_strength(signed)
    return Metric("price_trend", "momentum", direction, strength, raw_value=price_change_pct)


DEFAULT_CATEGORY_WEIGHTS = {
    "order_flow": 0.30,
    "order_book": 0.20,
    "momentum": 0.20,
    "participation": 0.15,
    "liquidity": 0.15,
}


def _build_categories(weights: dict) -> List[Category]:
    return [
        Category("order_flow", weights.get("order_flow", 0.0), [
            indicator_aggressive_participation, indicator_flow_ratio, indicator_vwap,
        ]),
        Category("order_book", weights.get("order_book", 0.0), [
            indicator_book_imbalance, indicator_book_ratio,
        ]),
        Category("momentum", weights.get("momentum", 0.0), [
            indicator_price_trend,
        ]),
        Category("participation", weights.get("participation", 0.0), [
            indicator_whale_dominance, indicator_trade_intensity, indicator_streak,
        ]),
        Category("liquidity", weights.get("liquidity", 0.0), [
            indicator_touch_liquidity, indicator_band_imbalance,
        ]),
    ]


@dataclass
class ConfidenceEngineConfig:
    category_weights: dict = field(default_factory=lambda: dict(DEFAULT_CATEGORY_WEIGHTS))
    min_confidence: float = 0.55


class ConfidenceEngine:
    """Pure confidence calculation. Holds no per-symbol execution state —
    signal persistence (PRIORITY 3) is a separate concern owned by
    SignalPersistenceTracker in market_data.py, not this class.
    """

    def __init__(self, config: Optional[ConfidenceEngineConfig] = None) -> None:
        self.config = config or ConfidenceEngineConfig()
        weights = self.config.category_weights or DEFAULT_CATEGORY_WEIGHTS
        total = sum(weights.values()) or 1.0
        self._normalized_weights = {k: v / total for k, v in weights.items()}
        self._categories = _build_categories(self._normalized_weights)

    def evaluate(
        self,
        flow: Optional[dict],
        liquidity: Optional[dict],
        book: Optional[dict],
        price_change_pct: Optional[float],
        cfg,
        trades: Optional[List[dict]] = None,
        price: Optional[float] = None,
        market: Optional[dict] = None,
        now_ms: Optional[float] = None,
    ) -> ConfidenceResult:
        now_ms = now_ms if now_ms is not None else time.time() * 1000.0
        price = price if price is not None else (market or {}).get("last_price")

        kwargs = dict(flow=flow, liquidity=liquidity, book=book, price_change_pct=price_change_pct,
                      cfg=cfg, trades=trades, price=price)
        categories = [cat.score(**kwargs) for cat in self._categories]
        by_name = {c.name: c for c in categories}

        def weighted(side_attr: str) -> float:
            return sum(getattr(c, side_attr) * c.weight for c in categories)

        long_total = weighted("long_score")
        short_total = weighted("short_score")

        # --- PRIORITY 2: continuous time decay, applied symmetrically
        # before direction is chosen since staleness isn't directional. ---
        decay_factor = self._decay_factor(trades, now_ms, cfg)
        long_total *= decay_factor
        short_total *= decay_factor

        # --- PRIORITY 4: confidence edge ---
        edge = abs(long_total - short_total)
        min_edge = getattr(cfg, "min_confidence_edge", 0.15)

        if long_total <= 0.0 and short_total <= 0.0:
            direction: Optional[Direction] = None
        else:
            direction = "long" if long_total >= short_total else "short"

        confidence = long_total if direction == "long" else short_total if direction == "short" else 0.0

        # --- PRIORITY 5: category agreement ---
        agreement_penalty_applied = False
        if direction is not None:
            confidence, agreement_penalty_applied = self._apply_category_agreement(confidence, by_name, direction, cfg)

        # --- PRIORITY 7: smart overextension filter ---
        overextension_penalty_applied = False
        if direction is not None:
            confidence, overextension_penalty_applied = self._apply_overextension(
                confidence, by_name, direction, price, market, cfg,
            )

        # --- PRIORITY 4 (revised): soft edge ramp instead of a hard cutoff ---
        # A hard "edge < min_edge -> reject" rule throws away signals that
        # are correct-but-early: edge naturally starts small and widens as
        # a move develops, so a hard gate forces the bot to wait until the
        # move is already well underway (the "enters too late" complaint).
        # Below `edge_floor` the two sides are genuinely too close to call
        # and we still reject outright (this is the actual conflict-detection
        # behavior from objective 6). Between the floor and min_edge, confidence
        # is damped proportionally to how thin the edge is, rather than
        # binary pass/fail — a strong, well-corroborated signal can still
        # clear min_confidence even with a modest edge; a weak one won't.
        edge_ramp_penalty_applied = False
        edge_floor = getattr(cfg, "min_confidence_edge_floor", min_edge * 0.4)
        if direction is not None and edge < min_edge:
            if edge <= edge_floor:
                direction = None
            else:
                ramp = (edge - edge_floor) / (min_edge - edge_floor)
                confidence *= (0.6 + 0.4 * ramp)  # damp 60-100% of confidence, never fully zeroed here
                edge_ramp_penalty_applied = True

        min_confidence = getattr(cfg, "min_confidence", self.config.min_confidence)
        if direction is not None and confidence < min_confidence:
            direction = None

        return ConfidenceResult(
            direction=direction,
            confidence=clip(confidence, 0.0, 1.0),
            categories=categories,
            long_confidence=long_total,
            short_confidence=short_total,
            edge=edge,
            decay_factor=decay_factor,
            agreement_penalty_applied=agreement_penalty_applied,
            overextension_penalty_applied=overextension_penalty_applied,
            edge_ramp_penalty_applied=edge_ramp_penalty_applied,
        )

    @staticmethod
    def _decay_factor(trades: Optional[List[dict]], now_ms: float, cfg) -> float:
        half_life_ms = getattr(cfg, "confidence_half_life_ms", 2500.0) or 2500.0
        if not trades:
            # No corroborating trades at all within the analysis window:
            # treat as stale rather than silently full-strength, but not so
            # stale it effectively zeroes out order-book/liquidity-only
            # setups on quieter symbols (previously 0.5**4 ~= 0.06x, which
            # buried almost every quiet-symbol signal regardless of how
            # strong the book/liquidity evidence was). Configurable so it
            # can be tuned per symbol tier without touching this module.
            exponent = getattr(cfg, "no_trades_decay_exponent", 1.5)
            return 0.5 ** exponent
        newest_ts = max(t["timestamp"] for t in trades)
        age_ms = max(0.0, now_ms - newest_ts)
        return 0.5 ** (age_ms / half_life_ms)

    @staticmethod
    def _apply_category_agreement(confidence: float, by_name: dict, direction: Direction, cfg) -> "tuple[float, bool]":
        threshold = getattr(cfg, "category_disagreement_threshold", -0.35)
        penalty = getattr(cfg, "category_agreement_penalty", 0.15)
        weights = [c.weight for c in by_name.values()]
        mean_weight = (sum(weights) / len(weights)) if weights else 0.0

        for cat in by_name.values():
            if not cat.metrics or cat.weight < mean_weight:
                continue  # only "important" (above-average weight) categories can trigger this
            aligned_score = cat.long_score if direction == "long" else cat.short_score
            opposing_score = cat.short_score if direction == "long" else cat.long_score
            net = aligned_score - opposing_score
            if net < threshold:
                return confidence * (1.0 - penalty), True
        return confidence, False

    @staticmethod
    def _apply_overextension(
        confidence: float, by_name: dict, direction: Direction, price, market, cfg,
    ) -> "tuple[float, bool]":
        if not getattr(cfg, "enable_overextension_filter", True) or not market or price is None:
            return confidence, False
        high, low = market.get("high_24h"), market.get("low_24h")
        if high is None or low is None or high <= low:
            return confidence, False

        proximity_pct = getattr(cfg, "overextension_proximity_pct", 0.0015)
        near_high = (high - price) / high <= proximity_pct if high > 0 else False
        near_low = (price - low) / low <= proximity_pct if low > 0 else False
        near_extreme = (direction == "long" and near_high) or (direction == "short" and near_low)
        if not near_extreme:
            return confidence, False

        weak_threshold = getattr(cfg, "overextension_weak_threshold", 0.25)
        momentum = by_name.get("momentum")
        order_flow = by_name.get("order_flow")
        momentum_aligned = (momentum.long_score if direction == "long" else momentum.short_score) if momentum else 0.0
        flow_aligned = (order_flow.long_score if direction == "long" else order_flow.short_score) if order_flow else 0.0

        both_weak = momentum_aligned < weak_threshold and flow_aligned < weak_threshold
        if not both_weak:
            return confidence, False  # momentum/flow still strong -> don't penalize genuine breakouts

        penalty = getattr(cfg, "overextension_penalty", 0.25)
        return confidence * (1.0 - penalty), True