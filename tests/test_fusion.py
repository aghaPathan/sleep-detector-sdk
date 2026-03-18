"""Tests for FusionEngine — weighted signal combination.

TDD: all tests written BEFORE implementation.
"""

import threading
import time

import pytest

from sleep_detector_sdk.fusion import FusionEngine
from sleep_detector_sdk.types import AlertTier, FatigueSignal, FusionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_signal(score: float, confidence: float, source: str, age: float = 0.0) -> FatigueSignal:
    """Return a FatigueSignal with timestamp = now - age seconds."""
    return FatigueSignal(
        score=score,
        confidence=confidence,
        source=source,
        timestamp=time.time() - age,
    )


# ---------------------------------------------------------------------------
# No-signal baseline
# ---------------------------------------------------------------------------

class TestNoSignals:
    def test_no_signals_returns_zero_score(self):
        engine = FusionEngine()
        result = engine.compute()
        assert result.fatigue_score == 0.0

    def test_no_signals_returns_silent_tier(self):
        engine = FusionEngine()
        result = engine.compute()
        assert result.tier == AlertTier.SILENT

    def test_no_signals_returns_empty_signal_list(self):
        engine = FusionEngine()
        result = engine.compute()
        assert result.signals == []

    def test_returns_fusion_result_type(self):
        engine = FusionEngine()
        result = engine.compute()
        assert isinstance(result, FusionResult)


# ---------------------------------------------------------------------------
# Single signal
# ---------------------------------------------------------------------------

class TestSingleSignal:
    def test_single_signal_score_times_confidence(self):
        engine = FusionEngine()
        signal = make_signal(score=0.8, confidence=0.5, source="ear")
        engine.submit_signal(signal)
        result = engine.compute()
        # weighted avg = (0.8 * 0.5 * 1.0) / (0.5 * 1.0) = 0.8
        assert abs(result.fatigue_score - 0.8) < 1e-6

    def test_single_signal_included_in_result(self):
        engine = FusionEngine()
        signal = make_signal(score=0.5, confidence=1.0, source="ear")
        engine.submit_signal(signal)
        result = engine.compute()
        assert signal in result.signals

    def test_single_full_confidence_passes_score_through(self):
        engine = FusionEngine()
        signal = make_signal(score=0.6, confidence=1.0, source="blink")
        engine.submit_signal(signal)
        result = engine.compute()
        assert abs(result.fatigue_score - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# Weighted fusion of two signals
# ---------------------------------------------------------------------------

class TestWeightedFusion:
    def test_two_signals_equal_weights(self):
        """0.5*0.6 (conf=1) + 1.0*0.4 (conf=1) with weights ear=0.5, blink=1.0.

        weighted_sum = (0.5 * 1.0 * 0.5) + (1.0 * 1.0 * 0.4) = 0.25 + 0.40 = 0.65
        weight_sum   = (0.5 * 1.0)       + (1.0 * 1.0)        = 0.5  + 1.0  = 1.5
        score        = 0.65 / 1.5 ≈ 0.4333…
        """
        engine = FusionEngine()
        engine.configure_weights({"ear": 0.5, "blink": 1.0})
        engine.submit_signal(make_signal(score=0.5, confidence=1.0, source="ear"))
        engine.submit_signal(make_signal(score=0.4, confidence=1.0, source="blink"))
        result = engine.compute()
        expected = (0.5 * 1.0 * 0.5 + 1.0 * 1.0 * 0.4) / (0.5 * 1.0 + 1.0 * 1.0)
        assert abs(result.fatigue_score - expected) < 1e-6

    def test_two_signals_spec_example(self):
        """Spec example: ear score=0.5 conf=1, blink score=1.0 conf=1, weights both 1.0.

        weighted_sum = (1.0 * 1.0 * 0.5) + (1.0 * 1.0 * 1.0) = 1.5
        weight_sum   = (1.0 * 1.0)       + (1.0 * 1.0)        = 2.0
        score        = 1.5 / 2.0 = 0.75
        """
        engine = FusionEngine()
        engine.configure_weights({"ear": 1.0, "blink": 1.0})
        engine.submit_signal(make_signal(score=0.5, confidence=1.0, source="ear"))
        engine.submit_signal(make_signal(score=1.0, confidence=1.0, source="blink"))
        result = engine.compute()
        assert abs(result.fatigue_score - 0.75) < 1e-6

    def test_higher_weight_dominates(self):
        engine = FusionEngine()
        engine.configure_weights({"low": 0.1, "high": 10.0})
        engine.submit_signal(make_signal(score=0.0, confidence=1.0, source="low"))
        engine.submit_signal(make_signal(score=1.0, confidence=1.0, source="high"))
        result = engine.compute()
        # high-weight source should pull score close to 1.0
        assert result.fatigue_score > 0.9


# ---------------------------------------------------------------------------
# Confidence reduces contribution
# ---------------------------------------------------------------------------

class TestConfidenceScaling:
    def test_low_confidence_reduces_contribution(self):
        engine = FusionEngine()
        engine.configure_weights({"a": 1.0, "b": 1.0})
        # signal a: score=1.0, conf=0.1  → effective weight = 0.1
        # signal b: score=0.0, conf=1.0  → effective weight = 1.0
        # score = (1.0 * 0.1 * 1.0 + 0.0 * 1.0 * 1.0) / (0.1 * 1.0 + 1.0 * 1.0)
        #       = 0.1 / 1.1 ≈ 0.0909
        engine.submit_signal(make_signal(score=1.0, confidence=0.1, source="a"))
        engine.submit_signal(make_signal(score=0.0, confidence=1.0, source="b"))
        result = engine.compute()
        expected = (1.0 * 0.1 * 1.0) / (0.1 * 1.0 + 1.0 * 1.0)
        assert abs(result.fatigue_score - expected) < 1e-6

    def test_zero_confidence_excluded(self):
        """A signal with confidence=0 contributes nothing to the weighted average."""
        engine = FusionEngine()
        engine.submit_signal(make_signal(score=1.0, confidence=0.0, source="ghost"))
        engine.submit_signal(make_signal(score=0.3, confidence=1.0, source="real"))
        result = engine.compute()
        # ghost contributes 0 effective weight; only "real" matters
        assert abs(result.fatigue_score - 0.3) < 1e-6


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

class TestTierClassification:
    def test_tier_silent_below_threshold(self):
        engine = FusionEngine(tier_thresholds=(0.4, 0.75))
        engine.submit_signal(make_signal(score=0.2, confidence=1.0, source="ear"))
        result = engine.compute()
        assert result.tier == AlertTier.SILENT

    def test_tier_silent_at_zero(self):
        engine = FusionEngine(tier_thresholds=(0.4, 0.75))
        result = engine.compute()
        assert result.tier == AlertTier.SILENT

    def test_tier_audible_at_low_threshold(self):
        engine = FusionEngine(tier_thresholds=(0.4, 0.75))
        engine.submit_signal(make_signal(score=0.4, confidence=1.0, source="ear"))
        result = engine.compute()
        assert result.tier == AlertTier.AUDIBLE

    def test_tier_audible_mid_range(self):
        engine = FusionEngine(tier_thresholds=(0.4, 0.75))
        engine.submit_signal(make_signal(score=0.6, confidence=1.0, source="ear"))
        result = engine.compute()
        assert result.tier == AlertTier.AUDIBLE

    def test_tier_audible_just_below_high_threshold(self):
        engine = FusionEngine(tier_thresholds=(0.4, 0.75))
        # score just below 0.75
        engine.submit_signal(make_signal(score=0.74, confidence=1.0, source="ear"))
        result = engine.compute()
        assert result.tier == AlertTier.AUDIBLE

    def test_tier_critical_at_high_threshold(self):
        engine = FusionEngine(tier_thresholds=(0.4, 0.75))
        engine.submit_signal(make_signal(score=0.75, confidence=1.0, source="ear"))
        result = engine.compute()
        assert result.tier == AlertTier.CRITICAL

    def test_tier_critical_high(self):
        engine = FusionEngine(tier_thresholds=(0.4, 0.75))
        engine.submit_signal(make_signal(score=1.0, confidence=1.0, source="ear"))
        result = engine.compute()
        assert result.tier == AlertTier.CRITICAL

    def test_custom_tier_thresholds(self):
        engine = FusionEngine(tier_thresholds=(0.2, 0.5))
        engine.submit_signal(make_signal(score=0.3, confidence=1.0, source="ear"))
        result = engine.compute()
        assert result.tier == AlertTier.AUDIBLE

    def test_custom_tier_thresholds_critical(self):
        engine = FusionEngine(tier_thresholds=(0.2, 0.5))
        engine.submit_signal(make_signal(score=0.5, confidence=1.0, source="ear"))
        result = engine.compute()
        assert result.tier == AlertTier.CRITICAL


# ---------------------------------------------------------------------------
# Stale signal filtering
# ---------------------------------------------------------------------------

class TestStaleSignals:
    def test_stale_signal_ignored(self):
        engine = FusionEngine(stale_threshold=0.1)
        # Signal 1 second old — older than 0.1 s threshold → stale
        stale_signal = make_signal(score=1.0, confidence=1.0, source="ear", age=1.0)
        engine.submit_signal(stale_signal)
        result = engine.compute()
        assert result.fatigue_score == 0.0
        assert result.tier == AlertTier.SILENT

    def test_stale_signal_not_in_result_signals(self):
        engine = FusionEngine(stale_threshold=0.1)
        stale_signal = make_signal(score=1.0, confidence=1.0, source="ear", age=1.0)
        engine.submit_signal(stale_signal)
        result = engine.compute()
        assert stale_signal not in result.signals

    def test_fresh_signal_not_stale(self):
        engine = FusionEngine(stale_threshold=5.0)
        signal = make_signal(score=0.8, confidence=1.0, source="ear", age=0.0)
        engine.submit_signal(signal)
        result = engine.compute()
        assert result.fatigue_score > 0.0

    def test_only_stale_leaves_empty_signals(self):
        engine = FusionEngine(stale_threshold=0.01)
        engine.submit_signal(make_signal(score=0.9, confidence=1.0, source="ear", age=1.0))
        result = engine.compute()
        assert result.signals == []


# ---------------------------------------------------------------------------
# Latest signal per source
# ---------------------------------------------------------------------------

class TestLatestSignalPerSource:
    def test_second_submit_overwrites_first(self):
        engine = FusionEngine()
        engine.submit_signal(make_signal(score=0.2, confidence=1.0, source="ear"))
        engine.submit_signal(make_signal(score=0.9, confidence=1.0, source="ear"))
        result = engine.compute()
        assert abs(result.fatigue_score - 0.9) < 1e-6

    def test_only_one_signal_per_source_in_result(self):
        engine = FusionEngine()
        engine.submit_signal(make_signal(score=0.2, confidence=1.0, source="ear"))
        engine.submit_signal(make_signal(score=0.9, confidence=1.0, source="ear"))
        result = engine.compute()
        assert len(result.signals) == 1

    def test_different_sources_both_retained(self):
        engine = FusionEngine()
        engine.submit_signal(make_signal(score=0.5, confidence=1.0, source="ear"))
        engine.submit_signal(make_signal(score=0.5, confidence=1.0, source="blink"))
        result = engine.compute()
        assert len(result.signals) == 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_submit_no_data_race(self):
        """10 threads each submit a distinct source — all should be present."""
        engine = FusionEngine()
        errors = []

        def worker(idx: int):
            try:
                engine.submit_signal(
                    make_signal(score=0.5, confidence=1.0, source=f"src_{idx}")
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        result = engine.compute()
        assert len(result.signals) == 10

    def test_concurrent_same_source_no_exception(self):
        """100 threads submit to the same source — no exception, exactly one signal."""
        engine = FusionEngine()
        errors = []

        def worker():
            try:
                engine.submit_signal(
                    make_signal(score=0.5, confidence=1.0, source="ear")
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        result = engine.compute()
        assert len(result.signals) == 1
