"""Tests for TemporalEngine — TDD: all tests written before implementation."""

import threading
import time

import pytest

from sleep_detector_sdk.temporal import TemporalEngine
from sleep_detector_sdk.types import GazeZone, TemporalState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**kwargs) -> TemporalEngine:
    """Return a stopped engine with fast defaults for unit tests."""
    defaults = {"frequency_hz": 25, "buffer_seconds": 5.0}
    defaults.update(kwargs)
    return TemporalEngine(**defaults)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_all_temporal_fields_are_none_at_start(self):
        engine = _make_engine()
        state = engine.current_state
        assert isinstance(state, TemporalState)
        assert state.t_zero is None
        assert state.t_away is None
        assert state.t_gaze is None
        assert state.t_road is None
        assert state.t_close is None

    def test_is_running_false_before_start(self):
        engine = _make_engine()
        assert engine.is_running is False

    def test_history_empty_at_start(self):
        engine = _make_engine()
        assert engine.history(1.0) == []


# ---------------------------------------------------------------------------
# Gaze transitions
# ---------------------------------------------------------------------------

class TestGazeRecording:
    def test_record_gaze_away_sets_t_away_on_first_transition_from_road(self):
        engine = _make_engine()
        before = time.time()
        engine.record_gaze(GazeZone.ROAD)       # establish ROAD baseline
        engine.record_gaze(GazeZone.IN_VEHICLE)  # transition away
        after = time.time()
        state = engine.current_state
        assert state.t_away is not None
        assert before <= state.t_away <= after

    def test_record_gaze_away_does_not_overwrite_t_away_on_repeated_calls(self):
        engine = _make_engine()
        engine.record_gaze(GazeZone.ROAD)
        engine.record_gaze(GazeZone.IN_VEHICLE)
        first_t_away = engine.current_state.t_away
        time.sleep(0.01)
        engine.record_gaze(GazeZone.IN_VEHICLE)
        assert engine.current_state.t_away == first_t_away

    def test_record_gaze_road_sets_t_road_when_returning_from_away(self):
        engine = _make_engine()
        engine.record_gaze(GazeZone.ROAD)
        engine.record_gaze(GazeZone.IN_VEHICLE)
        before = time.time()
        engine.record_gaze(GazeZone.ROAD)
        after = time.time()
        state = engine.current_state
        assert state.t_road is not None
        assert before <= state.t_road <= after

    def test_record_gaze_sets_t_gaze_on_landing_at_specific_zone(self):
        """t_gaze is set when a non-ROAD gaze lands (any zone after transition)."""
        engine = _make_engine()
        engine.record_gaze(GazeZone.ROAD)
        before = time.time()
        engine.record_gaze(GazeZone.EXTERNAL)
        after = time.time()
        state = engine.current_state
        assert state.t_gaze is not None
        assert before <= state.t_gaze <= after

    def test_no_t_away_without_prior_road_gaze(self):
        """Gaze away without establishing ROAD first should not set t_away."""
        engine = _make_engine()
        engine.record_gaze(GazeZone.IN_VEHICLE)
        assert engine.current_state.t_away is None


# ---------------------------------------------------------------------------
# Eye closure
# ---------------------------------------------------------------------------

class TestEyeRecording:
    def test_record_eye_close_sets_t_close(self):
        engine = _make_engine()
        before = time.time()
        engine.record_eye_close()
        after = time.time()
        state = engine.current_state
        assert state.t_close is not None
        assert before <= state.t_close <= after

    def test_record_eye_close_does_not_overwrite_t_close(self):
        engine = _make_engine()
        engine.record_eye_close()
        first_t_close = engine.current_state.t_close
        time.sleep(0.01)
        engine.record_eye_close()
        assert engine.current_state.t_close == first_t_close

    def test_record_eye_open_does_not_clear_t_close(self):
        engine = _make_engine()
        engine.record_eye_close()
        t_close = engine.current_state.t_close
        engine.record_eye_open()
        assert engine.current_state.t_close == t_close


# ---------------------------------------------------------------------------
# T-zero derivation
# ---------------------------------------------------------------------------

class TestTZero:
    def test_t_zero_computed_from_t_away(self):
        engine = _make_engine()
        engine.record_gaze(GazeZone.ROAD)
        engine.record_gaze(GazeZone.IN_VEHICLE)
        state = engine.current_state
        assert state.t_zero is not None
        assert state.t_zero == pytest.approx(state.t_away - 4.0, abs=1e-6)

    def test_t_zero_computed_from_t_close_when_no_t_away(self):
        engine = _make_engine()
        engine.record_eye_close()
        state = engine.current_state
        assert state.t_zero is not None
        assert state.t_zero == pytest.approx(state.t_close - 4.0, abs=1e-6)

    def test_t_zero_uses_whichever_came_first(self):
        """t_zero = min(t_away, t_close) - 4.0."""
        engine = _make_engine()
        engine.record_eye_close()
        t_close = engine.current_state.t_close
        time.sleep(0.02)
        engine.record_gaze(GazeZone.ROAD)
        engine.record_gaze(GazeZone.IN_VEHICLE)
        state = engine.current_state
        # t_close came first, so t_zero = t_close - 4.0
        assert state.t_zero == pytest.approx(t_close - 4.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all_temporal_fields(self):
        engine = _make_engine()
        engine.record_gaze(GazeZone.ROAD)
        engine.record_gaze(GazeZone.IN_VEHICLE)
        engine.record_eye_close()
        engine.reset()
        state = engine.current_state
        assert state.t_zero is None
        assert state.t_away is None
        assert state.t_gaze is None
        assert state.t_road is None
        assert state.t_close is None

    def test_reset_clears_history(self):
        engine = _make_engine(frequency_hz=100, buffer_seconds=1.0)
        engine.start()
        time.sleep(0.05)
        engine.stop()
        engine.reset()
        assert engine.history(1.0) == []


# ---------------------------------------------------------------------------
# History / ring buffer
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_returns_entries_after_ticks(self):
        engine = _make_engine(frequency_hz=100, buffer_seconds=1.0)
        engine.start()
        time.sleep(0.1)  # ~10 ticks at 100 Hz
        engine.stop()
        entries = engine.history(1.0)
        assert len(entries) >= 5

    def test_history_entries_are_temporal_state_instances(self):
        engine = _make_engine(frequency_hz=100, buffer_seconds=1.0)
        engine.start()
        time.sleep(0.05)
        engine.stop()
        for entry in engine.history(1.0):
            assert isinstance(entry, TemporalState)

    def test_history_duration_filters_to_recent_window(self):
        engine = _make_engine(frequency_hz=100, buffer_seconds=5.0)
        engine.start()
        time.sleep(0.15)
        engine.stop()
        short = engine.history(0.05)
        long_ = engine.history(1.0)
        # shorter window returns fewer or equal entries
        assert len(short) <= len(long_)


# ---------------------------------------------------------------------------
# Start / stop lifecycle
# ---------------------------------------------------------------------------

class TestStartStop:
    def test_is_running_true_after_start(self):
        engine = _make_engine()
        engine.start()
        try:
            assert engine.is_running is True
        finally:
            engine.stop()

    def test_is_running_false_after_stop(self):
        engine = _make_engine()
        engine.start()
        engine.stop()
        assert engine.is_running is False

    def test_double_start_is_idempotent(self):
        engine = _make_engine()
        engine.start()
        engine.start()
        try:
            assert engine.is_running is True
        finally:
            engine.stop()

    def test_double_stop_is_idempotent(self):
        engine = _make_engine()
        engine.start()
        engine.stop()
        engine.stop()  # should not raise
        assert engine.is_running is False


# ---------------------------------------------------------------------------
# Sampling frequency
# ---------------------------------------------------------------------------

class TestSamplingFrequency:
    def test_100hz_engine_produces_at_least_5_ticks_in_100ms(self):
        engine = _make_engine(frequency_hz=100, buffer_seconds=1.0)
        engine.start()
        time.sleep(0.1)
        engine.stop()
        assert len(engine.history(1.0)) >= 5


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_reads_do_not_raise(self):
        engine = _make_engine(frequency_hz=100, buffer_seconds=1.0)
        engine.start()
        errors = []

        def read():
            try:
                for _ in range(20):
                    _ = engine.current_state
                    _ = engine.history(1.0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=read) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        engine.stop()
        assert errors == []
