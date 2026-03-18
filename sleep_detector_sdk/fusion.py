"""FusionEngine — weighted combination of multi-source fatigue signals.

Design decisions:
- Thread-safe signal store via threading.Lock (one signal per source, last write wins).
- Stale filtering at compute() time so submit_signal() is non-blocking.
- Weighted average formula: score = sum(s.score * s.confidence * w) / sum(s.confidence * w)
  where w is the per-source weight (defaults to 1.0).
- Confidence=0 signals contribute 0 effective weight and are therefore excluded.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple

from sleep_detector_sdk.types import AlertTier, FatigueSignal, FusionResult


class FusionEngine:
    """Combine fatigue signals from multiple sources into a single FusionResult."""

    def __init__(
        self,
        stale_threshold: float = 5.0,
        tier_thresholds: Tuple[float, float] = (0.4, 0.75),
    ) -> None:
        """
        Args:
            stale_threshold: Maximum age (seconds) of a signal before it is ignored.
            tier_thresholds: (low, high) score boundaries.
                - score < low  → SILENT
                - low ≤ score < high → AUDIBLE
                - score ≥ high → CRITICAL
        """
        self._stale_threshold = stale_threshold
        self._tier_low, self._tier_high = tier_thresholds

        # Latest signal per source — protected by _lock
        self._signals: Dict[str, FatigueSignal] = {}
        # Per-source weight — read/written under _lock
        self._weights: Dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_signal(self, signal: FatigueSignal) -> None:
        """Store the latest signal for its source.  Thread-safe."""
        with self._lock:
            self._signals[signal.source] = signal

    def configure_weights(self, weights: Dict[str, float]) -> None:
        """Set per-source weights.  May be called at any time."""
        with self._lock:
            self._weights.update(weights)

    def compute(self) -> FusionResult:
        """Compute weighted fatigue score from all fresh, non-zero-confidence signals."""
        now = time.time()

        with self._lock:
            snapshot: List[FatigueSignal] = list(self._signals.values())
            weights = dict(self._weights)

        # Filter stale signals
        fresh = [
            s for s in snapshot
            if (now - s.timestamp) <= self._stale_threshold
        ]

        # Weighted average
        numerator = 0.0
        denominator = 0.0
        for sig in fresh:
            w = weights.get(sig.source, 1.0)
            effective_weight = sig.confidence * w
            numerator += sig.score * effective_weight
            denominator += effective_weight

        if denominator == 0.0:
            fatigue_score = 0.0
            active_signals: List[FatigueSignal] = []
        else:
            fatigue_score = numerator / denominator
            active_signals = fresh

        tier = self._classify_tier(fatigue_score)

        return FusionResult(
            fatigue_score=fatigue_score,
            tier=tier,
            signals=active_signals,
            timestamp=now,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_tier(self, score: float) -> AlertTier:
        if score >= self._tier_high:
            return AlertTier.CRITICAL
        if score >= self._tier_low:
            return AlertTier.AUDIBLE
        return AlertTier.SILENT
