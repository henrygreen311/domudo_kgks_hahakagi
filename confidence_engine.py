"""
Continuous, category-weighted confidence engine for SignalGenerator.

Kept as a separate module (rather than folded into market_data.py) so
normalization, indicator scoring, and category aggregation can each be
unit tested and tuned independently — per the modularity goal in the
design doc. This module does NOT touch EventConfirmationEngine or the
prerequisite/quality gate in SignalGenerator.evaluate(); those stay
exactly as they are.

Build plan (incremental, per your request):
  Step 1 (this file so far): normalization primitives + core data
          structures (Metric, Category, ConfidenceResult).
  Step 2 (next): concrete indicator functions grouped into categories
          (Order Flow, Order Book, Momentum, Participation), built from
          the fields OrderFlowAnalyzer / LiquidityEngine already produce.
  Step 3 (last): wire ConfidenceEngine into SignalGenerator.evaluate(),
          replacing only the long_checks/short_checks/confidence block.
"""

from __future__ import annotations

import math
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
        lines.append(f"Final Confidence: {self.confidence:.0%}")
        return "\n".join(lines)


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