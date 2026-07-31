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

    # --- Scoring targets ---
    # These are no longer hard pass/fail gates (see EvidenceSummary.score()
    # below) — each is the level at which that dimension earns full credit.
    # Falling short earns partial credit rather than an outright rejection,
    # so one weak metric can be outweighed by strong ones elsewhere instead
    # of vetoing an otherwise-excellent setup.
    confidence_average_threshold: float = 0.60
    confidence_peak_threshold: float = 0.75
    direction_consistency_threshold: float = 0.60
    minimum_confirmation_events: int = 2
    minimum_whale_events: int = 1
    minimum_aggressive_volume_ratio: float = 0.60
    # Liquidity reversals score inversely (fewer = better); this is the
    # count at/above which that dimension earns zero credit rather than a
    # hard reject. 10-20 is realistic for live order-book noise even with
    # debouncing — 1 (the old default) was unreachable in practice.
    maximum_allowed_liquidity_reversals: int = 15
    # Deadzone around zero so noise-level imbalance readings don't count as
    # a "side" at all (avoids phantom reversals from near-zero noise).
    liquidity_imbalance_deadzone: float = 0.05
    # A sign flip only counts as a real reversal once the new sign has held
    # for this many consecutive samples — at a 150ms tick cadence, single-
    # tick sign flips are usually order-book microstructure noise, not an
    # actual reversal of pressure. Without this, a genuinely stable book
    # can rack up dozens of "reversals" in 60s from noise alone.
    liquidity_reversal_debounce_samples: int = 3
    # direction_flips is still computed and logged (see EvidenceSummary)
    # but no longer gates on its own — it's a close cousin of
    # direction_consistency and double-penalizing the same underlying
    # behavior worked against the "let strong signals compensate" goal.

    # Minimum number of independent SignalGenerator readings (agreeing on
    # the winning direction) that must exist in the window before scoring
    # even applies. This is NOT a scored/weighted dimension — it's a
    # sample-size floor. Without it, a single lucky tick's confidence and
    # confirmation numbers get echoed back unchanged on every subsequent
    # summarize() call for up to rolling_window_seconds, since nothing new
    # arrives to contradict them — so one strong reading can satisfy both
    # the "60s window" and the "7s persistence" checks despite reflecting
    # a few hundred milliseconds of real evidence, not 60 real seconds of
    # it. Below this count, passes() rejects outright regardless of score.
    minimum_directional_samples: int = 3

    # How long the buffer must have actually been accumulating for this
    # symbol before a trade is allowed — separate from
    # minimum_directional_samples above. That gate checks *how many*
    # readings agree; this one checks *how long* the window has genuinely
    # existed.
    #
    # NOTE: this used to default to the full rolling_window_seconds (i.e.
    # require a literal 60-second wait before a symbol could ever be
    # scored, regardless of how much evidence had already accumulated).
    # That's not what was asked for — the requirement is "gather evidence
    # in a rolling window and open once direction/whale/confidence checks
    # clear the bar," not "refuse to look until the buffer is a full
    # rolling_window_seconds old." It also turned out to be structurally
    # unreachable in practice: the buffer is pruned to rolling_window_seconds
    # on every tick (see update() below), which caps window_age_seconds at
    # just under rolling_window_seconds forever — so the old default was
    # both wrong in intent and impossible to satisfy. Default is now 0
    # (no extra time-based wait beyond minimum_directional_samples); set
    # explicitly if a genuine minimum warm-up period is wanted.
    minimum_window_age_seconds: float = 0.0

    # --- Scoring weights (must sum to 100 for score to read as a percentage) ---
    avg_confidence_weight: float = 20.0
    peak_confidence_weight: float = 20.0
    consistency_weight: float = 20.0
    whale_weight: float = 15.0
    confirmation_weight: float = 10.0
    aggressive_weight: float = 10.0
    liquidity_weight: float = 5.0
    # Minimum total score (out of 100, assuming default weights) to trade.
    min_score_threshold: float = 75.0

    # Direction-specific override of min_score_threshold above. Added
    # after analysis of the first ~40 live trades in position_history
    # showed long entries scoring just as high pre-trade (avg/peak
    # confidence, whale/event confirmations) as shorts, but landing
    # "Late Entry" 6x more often (6 of 7 Late Entry trades were long) and
    # liquidating at a 40% rate vs 25% for shorts. The pre-trade score
    # isn't distinguishing "direction confirmed" from "direction
    # confirmed after most of the move already happened" — crypto pumps
    # (longs) tend to get confirmed by trailing buy-volume signals only
    # after price has already run, while drops (shorts) show up earlier
    # in order-book/whale-sell signals. Raising the long-side bar is a
    # blunt first pass at compensating for that until signals_histories
    # (see the SQL for that table) has enough rows to fit something
    # better — e.g. a "how much of the move already happened" freshness
    # check. Re-tune these numbers once that data exists; they come from
    # "accepted longs in the sample ranged 79.9-94.2 and still
    # liquidated 40% of the time, so require meaningfully more margin
    # above the base bar," not from a proper fit.
    min_score_threshold_by_direction: Dict[str, float] = field(
        default_factory=lambda: {"long": 88.0, "short": 75.0}
    )

    def min_score_for(self, direction: str) -> float:
        return self.min_score_threshold_by_direction.get(
            direction.lower(), self.min_score_threshold
        )

    # --- Signal persistence (separate stage, see PersistenceValidator) ---
    signal_persistence_seconds: float = 7.0


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
class ScoreComponent:
    """One line of the weighted scoring breakdown — mirrors exactly what
    gets logged, so the log and the scoring math can never drift apart."""
    name: str
    value: float
    target: float
    weight: float
    points: float
    met_target: bool  # informational only — doesn't gate anything on its own
    is_ratio: bool = True  # True -> format as a percentage; False -> plain count

    def format_value(self) -> str:
        return f"{self.value:.0%}" if self.is_ratio else f"{self.value:g}"

    def format_target(self) -> str:
        return f"{self.target:.0%}" if self.is_ratio else f"{self.target:g}"


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
    directional_sample_count: int = 0  # how many raw SignalGenerator readings agreed on dominant_direction
    window_age_seconds: float = 0.0  # how long the buffer has actually been accumulating, oldest-to-newest sample

    def score_breakdown(self, cfg: RollingEvidenceConfig) -> List[ScoreComponent]:
        """Each dimension earns partial credit toward its weight, scaling
        linearly from 0 up to full credit at its target (capped at the
        weight — exceeding the target doesn't earn bonus points). Liquidity
        reversals score inversely: 0 reversals = full credit, at-or-above
        the max = zero credit. This is what lets one weak metric be offset
        by strong ones elsewhere instead of vetoing the whole signal."""

        def scaled(value: float, target: float, weight: float) -> float:
            if target <= 0:
                return weight
            return weight * max(0.0, min(1.0, value / target))

        def inverse_scaled(value: float, cap: float, weight: float) -> float:
            if cap <= 0:
                return 0.0 if value > 0 else weight
            return weight * max(0.0, min(1.0, 1.0 - value / cap))

        return [
            ScoreComponent(
                "avg confidence", self.avg_confidence, cfg.confidence_average_threshold,
                cfg.avg_confidence_weight,
                scaled(self.avg_confidence, cfg.confidence_average_threshold, cfg.avg_confidence_weight),
                self.avg_confidence >= cfg.confidence_average_threshold,
            ),
            ScoreComponent(
                "peak confidence", self.peak_confidence, cfg.confidence_peak_threshold,
                cfg.peak_confidence_weight,
                scaled(self.peak_confidence, cfg.confidence_peak_threshold, cfg.peak_confidence_weight),
                self.peak_confidence >= cfg.confidence_peak_threshold,
            ),
            ScoreComponent(
                "direction consistency", self.direction_consistency, cfg.direction_consistency_threshold,
                cfg.consistency_weight,
                scaled(self.direction_consistency, cfg.direction_consistency_threshold, cfg.consistency_weight),
                self.direction_consistency >= cfg.direction_consistency_threshold,
            ),
            ScoreComponent(
                "whale confirmations", float(self.whale_event_count), float(cfg.minimum_whale_events),
                cfg.whale_weight,
                scaled(self.whale_event_count, cfg.minimum_whale_events, cfg.whale_weight),
                self.whale_event_count >= cfg.minimum_whale_events,
                is_ratio=False,
            ),
            ScoreComponent(
                "event confirmations", float(self.confirmation_event_count), float(cfg.minimum_confirmation_events),
                cfg.confirmation_weight,
                scaled(self.confirmation_event_count, cfg.minimum_confirmation_events, cfg.confirmation_weight),
                self.confirmation_event_count >= cfg.minimum_confirmation_events,
                is_ratio=False,
            ),
            ScoreComponent(
                "aggressive volume", self.aggressive_dominance_ratio, cfg.minimum_aggressive_volume_ratio,
                cfg.aggressive_weight,
                scaled(self.aggressive_dominance_ratio, cfg.minimum_aggressive_volume_ratio, cfg.aggressive_weight),
                self.aggressive_dominance_ratio >= cfg.minimum_aggressive_volume_ratio,
            ),
            ScoreComponent(
                "liquidity reversals", float(self.liquidity_reversals), float(cfg.maximum_allowed_liquidity_reversals),
                cfg.liquidity_weight,
                inverse_scaled(self.liquidity_reversals, cfg.maximum_allowed_liquidity_reversals, cfg.liquidity_weight),
                self.liquidity_reversals <= cfg.maximum_allowed_liquidity_reversals,
                is_ratio=False,
            ),
        ]

    def score(self, cfg: RollingEvidenceConfig) -> float:
        if self.dominant_direction is None or self.sample_count == 0:
            return 0.0
        return sum(c.points for c in self.score_breakdown(cfg))

    def to_signal_record(self, cfg: RollingEvidenceConfig, raw_signal_count: int = 0) -> dict:
        """Flat dict matching signals_histories' schema (see
        create_signals_histories.sql) — the exact same numbers explain()
        would log, captured once at accept time so they exist as queryable
        rows instead of only ever living in log text. Caller (tracker.py)
        still needs to add trade_id, entry_price, price_at_decision, and
        evaluated_at once the trade has actually opened; this covers
        everything derivable from the summary itself."""
        by_name = {c.name: c for c in self.score_breakdown(cfg)}

        def value(name: str) -> float:
            return by_name[name].value

        def points(name: str) -> float:
            return by_name[name].points

        return {
            "symbol": self.symbol,
            "direction": (self.dominant_direction or "").lower(),
            "decision": "accepted",
            "raw_signal_count": raw_signal_count,
            "directional_sample_count": self.directional_sample_count,
            "direction_flips": self.direction_flips,
            "window_age_seconds": self.window_age_seconds,
            "avg_confidence_pct": value("avg confidence"),
            "peak_confidence_pct": value("peak confidence"),
            "direction_consistency_pct": value("direction consistency"),
            "whale_confirmations": int(value("whale confirmations")),
            "event_confirmations": int(value("event confirmations")),
            "aggressive_volume_pct": value("aggressive volume"),
            "liquidity_reversals": int(value("liquidity reversals")),
            "avg_confidence_score": points("avg confidence"),
            "peak_confidence_score": points("peak confidence"),
            "direction_consistency_score": points("direction consistency"),
            "whale_confirmations_score": points("whale confirmations"),
            "event_confirmations_score": points("event confirmations"),
            "aggressive_volume_score": points("aggressive volume"),
            "liquidity_reversals_score": points("liquidity reversals"),
            "total_score": self.score(cfg),
            "score_threshold_used": cfg.min_score_for(self.dominant_direction or ""),
        }

    def passes(self, cfg: RollingEvidenceConfig) -> bool:
        if self.dominant_direction is None or self.sample_count == 0:
            return False
        if self.directional_sample_count < cfg.minimum_directional_samples:
            return False
        if self.window_age_seconds < cfg.minimum_window_age_seconds:
            return False
        return self.score(cfg) >= cfg.min_score_for(self.dominant_direction)

    def explain(self, cfg: RollingEvidenceConfig) -> str:
        """Full weighted breakdown, one line per component, always shown —
        so a rejection says exactly which dimensions fell short and by how
        much, not just PASSES=False."""
        if self.dominant_direction is None or self.sample_count == 0:
            return f"{self.symbol}: no directional evidence in window — REJECTED (score=0/100)"

        if self.directional_sample_count < cfg.minimum_directional_samples:
            return (
                f"{self.symbol} — direction={self.dominant_direction.upper()} "
                f"directional_samples={self.directional_sample_count} (need >= {cfg.minimum_directional_samples}) "
                f"— REJECTED before scoring: not enough independent readings yet, "
                f"regardless of how strong the ones seen so far look"
            )

        if self.window_age_seconds < cfg.minimum_window_age_seconds:
            return (
                f"{self.symbol} — direction={self.dominant_direction.upper()} "
                f"window_age={self.window_age_seconds:.1f}s (need >= {cfg.minimum_window_age_seconds:.1f}s) "
                f"— REJECTED before scoring: still warming up, buffer hasn't spanned a full window yet"
            )

        components = self.score_breakdown(cfg)
        total = sum(c.points for c in components)
        threshold = cfg.min_score_for(self.dominant_direction)
        decision = "ACCEPTED" if total >= threshold else "REJECTED"

        lines = [f"{self.symbol} — direction={self.dominant_direction.upper()} samples={self.sample_count} flips={self.direction_flips}"]
        for c in components:
            verdict = "PASS" if c.met_target else "FAIL"
            lines.append(
                f"  {c.name}: {c.format_value()} (target {c.format_target()}) "
                f"-> {c.points:.1f}/{c.weight:.0f} pts [{verdict}]"
            )
        lines.append(f"  Total score: {total:.1f}/100 (need >= {threshold:.0f})")
        lines.append(f"  Decision: {decision}")

        if decision == "REJECTED":
            # Name the biggest shortfalls — components below target, ranked
            # by how many points they left on the table.
            shortfalls = sorted(
                (c for c in components if not c.met_target),
                key=lambda c: c.weight - c.points,
                reverse=True,
            )
            if shortfalls:
                reason = ", ".join(f"{c.name} below target" for c in shortfalls[:2])
                lines.append(f"  Reason: {reason}")
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
        # First tick timestamp per symbol, set once and never touched by
        # pruning. buf[0].ts_ms is NOT usable for "how long have we been
        # watching this symbol" — the rolling-window prune in update()
        # guarantees buf[0].ts_ms >= now_ms - rolling_window_seconds*1000
        # at all times, which caps (buf[-1] - buf[0]) at just under
        # rolling_window_seconds forever, regardless of how long the
        # symbol has actually been tracked.
        self._first_seen_ms: Dict[str, float] = {}

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
        self._first_seen_ms.setdefault(symbol, now_ms)

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

        # Consistency is measured against the *directional* samples only
        # (ticks where SignalGenerator actually produced a reading), not
        # every tick in the buffer. SignalGenerator only fires occasionally
        # relative to the 150ms accumulator cadence, so a denominator of
        # "every tick" made this unreachable regardless of signal quality —
        # real data showed 1% consistency on signals that were, in fact,
        # 100% agreeing every time they fired. Flip-flopping is still
        # caught by `direction_flips` below; this metric now answers "when
        # we had an opinion, did it stay the same" rather than "did we have
        # an opinion nearly every tick".
        consistency = len(dominant_samples) / len(directional)

        flips = sum(1 for a, b in zip(directional, directional[1:]) if a.direction != b.direction)

        confirmation_total = sum(s.confirmation_count for s in dominant_samples)
        whale_total = sum(1 for s in dominant_samples if s.whale_event)
        sweep_total = sum(1 for s in dominant_samples if s.sweep_event)

        aggressive_hits = sum(1 for s in dominant_samples if s.aggressive_dominant == dominant)
        aggressive_ratio = aggressive_hits / len(dominant_samples)

        signs = [s.book_imbalance_sign for s in buf if s.book_imbalance_sign is not None]
        non_zero_signs = [s for s in signs if s != 0]
        reversals = self._count_debounced_reversals(non_zero_signs)

        latest_signal = next((s.signal for s in reversed(dominant_samples) if s.signal is not None), None)
        first_seen_ms = self._first_seen_ms.get(symbol, buf[0].ts_ms)
        window_age_seconds = (buf[-1].ts_ms - first_seen_ms) / 1000.0

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
            directional_sample_count=len(dominant_samples),
            window_age_seconds=window_age_seconds,
        )

    def clear(self, symbol: str) -> None:
        self._samples.pop(symbol, None)
        self._first_seen_ms.pop(symbol, None)

    def _count_debounced_reversals(self, signs: List[int]) -> int:
        """Count real regime changes in a sign sequence, ignoring flips
        that don't hold for at least `liquidity_reversal_debounce_samples`
        consecutive samples. E.g. with debounce=3: [1,1,1,-1,1,1,1,1,-1,-1,-1]
        -> only the run starting at index 8 counts (1 reversal); the lone
        -1 at index 3 is noise and is ignored."""
        debounce = max(1, self.config.liquidity_reversal_debounce_samples)
        if not signs:
            return 0

        reversals = 0
        confirmed = signs[0]
        run_sign = signs[0]
        run_len = 1
        for s in signs[1:]:
            if s == run_sign:
                run_len += 1
            else:
                run_sign = s
                run_len = 1
            if run_len >= debounce and run_sign != confirmed:
                reversals += 1
                confirmed = run_sign
        return reversals


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
) -> Optional[Tuple[Signal, EvidenceSummary]]:
    """One tick of the new stage: fold this tick's evidence into the rolling
    window, then check whether the accumulated + persisted picture clears
    the bar to trade. Returns (Signal, EvidenceSummary) to execute once
    persistence is satisfied, otherwise None. Call this every 100-200ms
    per watchlist symbol; it's cheap (bounded deque append + a handful of
    sums over a window of ~300-600 samples at that cadence).

    SignalGenerator and EventConfirmationEngine are both called exactly as
    they already are elsewhere in the codebase — this function doesn't
    change how either one decides anything, it only remembers and times
    their outputs.

    The EvidenceSummary returned here is the one that actually cleared
    the bar — captured before clear() below resets the accumulator for
    this symbol. Callers needing the breakdown for persistence (e.g.
    signals_histories) must use this returned summary rather than calling
    accumulator.summarize(symbol) again afterward: by then clear() has
    already wiped it, which silently produces an empty/all-zero summary
    (dominant_direction=None) instead of raising — this was found live
    when signal_store.record_signal() started rejecting rows with an
    empty `direction` that violated signals_histories' check constraint.
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
    return summary.latest_signal, summary
