"""
Rolling Evidence Accumulator + Persistence Validator.

    SignalGenerator -> RollingEvidenceAccumulator -> PersistenceValidator -> EventConfirmationEngine -> ExecutionEngine

This module inserts between the existing directional/confidence layer and
the execution engine. It does NOT reimplement anything SignalGenerator,
ConfidenceEngine, or EventConfirmationEngine already do — it only *remembers
their outputs over a rolling 60-second window* and decides whether the
evidence has been sustained for long enough to trade, instead of trading on
a single instantaneous read.

Design notes:

- EventConfirmationEngine.confirm() is normally called once, right before
  opening a trade. Here it's called on every tick (see `evaluate_tick`
  below), and its results are folded into the rolling window. This is what
  lets "the Event Confirmation Engine continued confirming the same
  direction throughout the observation period" and "multiple confirmation
  events over the window" be checked — those aren't new detections, they're
  the existing confirm() output sampled repeatedly and counted.

- The buffer is a per-symbol deque pruned by *time*, not by count, so memory
  is bounded by (update_rate x window_seconds) regardless of how long the
  bot runs — satisfies the "efficient rolling buffer, constant memory"
  requirement.

- RollingEvidenceAccumulator only answers "what do the last 60 seconds look
  like". PersistenceValidator only answers "has that picture been true, and
  stable, for the configured persistence duration". Keeping those separate
  mirrors the existing ConfidenceEngine / SignalPersistenceTracker split in
  market_data.py, so the same mental model applies here.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from market_data import Signal
from event_confirmation import ConfirmationResult, EventConfirmationEngine

Direction = Optional[str]  # "long" | "short" | None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RollingEvidenceConfig:
    rolling_window_seconds: float = 60.0

    # --- Entry conditions (design doc items 1-7) ---
    confidence_average_threshold: float = 0.75
    confidence_peak_threshold: float = 0.85
    direction_consistency_threshold: float = 0.80
    minimum_confirmation_events: int = 2
    minimum_whale_events: int = 1
    minimum_aggressive_volume_ratio: float = 0.60
    maximum_allowed_direction_flips: int = 2
    # Order book imbalance sign reversals allowed in the window before it
    # counts as "strongly reversed" (item 6).
    maximum_allowed_liquidity_reversals: int = 1
    # Deadzone around zero so noise-level imbalance readings don't count as
    # a "side" at all (avoids phantom reversals from near-zero noise).
    liquidity_imbalance_deadzone: float = 0.05

    # --- Signal persistence (separate stage, see PersistenceValidator) ---
    signal_persistence_seconds: float = 15.0


# ---------------------------------------------------------------------------
# Internal per-tick sample and the summary derived from a window of them
# ---------------------------------------------------------------------------


@dataclass
class _Sample:
    ts_ms: float
    direction: Direction
    confidence: float
    book_imbalance_sign: Optional[int]      # -1 / 0 / 1, or None if unknown this tick
    aggressive_dominant: Direction          # which side aggressive volume favored, if any
    whale_event: bool
    sweep_event: bool
    confirmation_count: int
    signal: Optional[Signal]                # retained so a winning window has a real Signal to execute


@dataclass
class EvidenceSummary:
    symbol: str
    sample_count: int
    dominant_direction: Direction
    avg_confidence: float = 0.0
    peak_confidence: float = 0.0
    direction_consistency: float = 0.0
    direction_flips: int = 0
    confirmation_event_count: int = 0
    whale_event_count: int = 0
    sweep_event_count: int = 0
    aggressive_dominance_ratio: float = 0.0
    liquidity_reversals: int = 0
    latest_signal: Optional[Signal] = None

    def passes(self, cfg: RollingEvidenceConfig) -> bool:
        """All of the design doc's entry conditions (1-6); condition 7,
        aggressive pressure dominance, is folded into
        aggressive_dominance_ratio below."""
        if self.dominant_direction is None or self.sample_count == 0:
            return False
        return (
            self.avg_confidence >= cfg.confidence_average_threshold
            and self.peak_confidence >= cfg.confidence_peak_threshold
            and self.direction_consistency >= cfg.direction_consistency_threshold
            and self.direction_flips <= cfg.maximum_allowed_direction_flips
            and self.confirmation_event_count >= cfg.minimum_confirmation_events
            and self.whale_event_count >= cfg.minimum_whale_events
            and self.aggressive_dominance_ratio >= cfg.minimum_aggressive_volume_ratio
            and self.liquidity_reversals <= cfg.maximum_allowed_liquidity_reversals
        )

    def explain(self, cfg: RollingEvidenceConfig) -> str:
        """Human-readable breakdown, same spirit as ConfidenceResult.explain()
        in confidence_engine.py — every accept/reject should be traceable."""
        d = self.dominant_direction or "none"
        lines = [
            f"Rolling window: {self.symbol} direction={d.upper()} samples={self.sample_count}",
            f"  avg_confidence={self.avg_confidence:.2f} (need >= {cfg.confidence_average_threshold:.2f})",
            f"  peak_confidence={self.peak_confidence:.2f} (need >= {cfg.confidence_peak_threshold:.2f})",
            f"  direction_consistency={self.direction_consistency:.0%} (need >= {cfg.direction_consistency_threshold:.0%})",
            f"  direction_flips={self.direction_flips} (max {cfg.maximum_allowed_direction_flips})",
            f"  confirmation_events={self.confirmation_event_count} (need >= {cfg.minimum_confirmation_events})",
            f"  whale_events={self.whale_event_count} (need >= {cfg.minimum_whale_events})",
            f"  aggressive_dominance_ratio={self.aggressive_dominance_ratio:.0%} (need >= {cfg.minimum_aggressive_volume_ratio:.0%})",
            f"  liquidity_reversals={self.liquidity_reversals} (max {cfg.maximum_allowed_liquidity_reversals})",
            f"  PASSES={self.passes(cfg)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rolling Evidence Accumulator
# ---------------------------------------------------------------------------


class RollingEvidenceAccumulator:
    """Per-symbol, constant-memory rolling window of market evidence.

    Call `update()` on every tick of whatever cadence your data-collection
    loop already runs at (100-200ms is the design target; any cadence
    works, the buffer is pruned by wall-clock time, not tick count).
    """

    def __init__(self, config: Optional[RollingEvidenceConfig] = None) -> None:
        self.config = config or RollingEvidenceConfig()
        self._samples: Dict[str, Deque[_Sample]] = {}

    def update(
        self,
        symbol: str,
        now_ms: float,
        signal: Optional[Signal],
        confirmation: Optional[ConfirmationResult],
        book_imbalance: Optional[float] = None,
    ) -> None:
        """Fold one tick's evidence into the rolling window.

        `signal` is whatever SignalGenerator.evaluate() returned this tick
        (None if no valid directional read). `confirmation` is whatever
        EventConfirmationEngine.confirm(signal) returned this tick (None if
        there was no signal to confirm). `book_imbalance` is an optional
        signed liquidity-imbalance reading (e.g. from LiquidityEngine) used
        only for the reversal check in item 6 — omit it if unavailable and
        that check is simply skipped.
        """
        buf = self._samples.setdefault(symbol, deque())

        direction = signal.direction if signal else None
        confidence = signal.confidence if signal else 0.0

        confirmations = confirmation.confirmations if confirmation else []
        whale_event = any("whale" in c.lower() for c in confirmations)
        sweep_event = any("sweep" in c.lower() for c in confirmations)
        aggressive_dominant = direction if any("aggressive" in c.lower() for c in confirmations) else None

        sign: Optional[int] = None
        if book_imbalance is not None:
            deadzone = self.config.liquidity_imbalance_deadzone
            sign = 1 if book_imbalance > deadzone else (-1 if book_imbalance < -deadzone else 0)

        buf.append(
            _Sample(
                ts_ms=now_ms,
                direction=direction,
                confidence=confidence,
                book_imbalance_sign=sign,
                aggressive_dominant=aggressive_dominant,
                whale_event=whale_event,
                sweep_event=sweep_event,
                confirmation_count=len(confirmations),
                signal=signal,
            )
        )

        cutoff = now_ms - self.config.rolling_window_seconds * 1000.0
        while buf and buf[0].ts_ms < cutoff:
            buf.popleft()

    def summarize(self, symbol: str) -> EvidenceSummary:
        buf = self._samples.get(symbol)
        if not buf:
            return EvidenceSummary(symbol=symbol, sample_count=0, dominant_direction=None)

        directional = [s for s in buf if s.direction is not None]
        if not directional:
            return EvidenceSummary(symbol=symbol, sample_count=len(buf), dominant_direction=None)

        long_count = sum(1 for s in directional if s.direction == "long")
        short_count = len(directional) - long_count
        dominant: Direction = "long" if long_count >= short_count else "short"
        dominant_samples = [s for s in directional if s.direction == dominant]

        avg_conf = sum(s.confidence for s in dominant_samples) / len(dominant_samples)
        peak_conf = max(s.confidence for s in dominant_samples)

        # Consistency is measured against the *whole* window (including
        # no-signal ticks), not just the directional subset — a window full
        # of gaps shouldn't read as "100% consistent" just because every
        # directional sample happened to agree.
        consistency = len(dominant_samples) / len(buf)

        flips = sum(1 for a, b in zip(directional, directional[1:]) if a.direction != b.direction)

        confirmation_total = sum(s.confirmation_count for s in dominant_samples)
        whale_total = sum(1 for s in dominant_samples if s.whale_event)
        sweep_total = sum(1 for s in dominant_samples if s.sweep_event)

        aggressive_hits = sum(1 for s in dominant_samples if s.aggressive_dominant == dominant)
        aggressive_ratio = aggressive_hits / len(dominant_samples)

        signs = [s.book_imbalance_sign for s in buf if s.book_imbalance_sign is not None]
        non_zero_signs = [s for s in signs if s != 0]
        reversals = sum(1 for a, b in zip(non_zero_signs, non_zero_signs[1:]) if a != b)

        latest_signal = next((s.signal for s in reversed(dominant_samples) if s.signal is not None), None)

        return EvidenceSummary(
            symbol=symbol,
            sample_count=len(buf),
            dominant_direction=dominant,
            avg_confidence=avg_conf,
            peak_confidence=peak_conf,
            direction_consistency=consistency,
            direction_flips=flips,
            confirmation_event_count=confirmation_total,
            whale_event_count=whale_total,
            sweep_event_count=sweep_total,
            aggressive_dominance_ratio=aggressive_ratio,
            liquidity_reversals=reversals,
            latest_signal=latest_signal,
        )

    def clear(self, symbol: str) -> None:
        self._samples.pop(symbol, None)


# ---------------------------------------------------------------------------
# Persistence Validator
# ---------------------------------------------------------------------------


class PersistenceValidator:
    """Tracks how long a *passing* rolling-evidence summary has held
    continuously, per symbol+direction. Any weakening (summary stops
    passing, or the dominant direction changes) resets the timer, per the
    design doc's persistence rules.
    """

    def __init__(self) -> None:
        self._pending: Dict[str, Tuple[str, float]] = {}  # symbol -> (direction, first_passed_ms)

    def check(self, symbol: str, summary: EvidenceSummary, cfg: RollingEvidenceConfig, now_ms: float) -> bool:
        if not summary.passes(cfg) or summary.dominant_direction is None:
            self._pending.pop(symbol, None)
            return False

        direction = summary.dominant_direction
        pending = self._pending.get(symbol)
        if pending is None or pending[0] != direction:
            self._pending[symbol] = (direction, now_ms)
            return False

        held_ms = now_ms - pending[1]
        return held_ms >= cfg.signal_persistence_seconds * 1000.0

    def clear(self, symbol: str) -> None:
        self._pending.pop(symbol, None)


# ---------------------------------------------------------------------------
# Orchestration helper — the actual insertion point in the pipeline
# ---------------------------------------------------------------------------


async def evaluate_tick(
    symbol: str,
    signal,  # Optional[Signal] — pass in what SignalGenerator.evaluate() returned this tick, or None
    confirmation_engine: EventConfirmationEngine,
    accumulator: RollingEvidenceAccumulator,
    validator: PersistenceValidator,
    book_imbalance: Optional[float] = None,
    now_ms: Optional[float] = None,
) -> Optional[Signal]:
    """One tick of the new stage: fold this tick's evidence into the rolling
    window, then check whether the accumulated + persisted picture clears
    the bar to trade. Returns the Signal to execute once persistence is
    satisfied, otherwise None. Call this every 100-200ms per watchlist
    symbol; it's cheap (bounded deque append + a handful of sums over a
    window of ~300-600 samples at that cadence).

    SignalGenerator and EventConfirmationEngine are both called exactly as
    they already are elsewhere in the codebase — this function doesn't
    change how either one decides anything, it only remembers and times
    their outputs.
    """
    now_ms = now_ms if now_ms is not None else time.time() * 1000.0

    confirmation = await confirmation_engine.confirm(signal) if signal is not None else None
    accumulator.update(symbol, now_ms, signal, confirmation, book_imbalance=book_imbalance)

    summary = accumulator.summarize(symbol)
    ready = validator.check(symbol, summary, accumulator.config, now_ms)
    if not ready:
        return None

    validator.clear(symbol)
    accumulator.clear(symbol)
    return summary.latest_signal
