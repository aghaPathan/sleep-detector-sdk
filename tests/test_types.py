"""Tests for sleep_detector_sdk.types — enums, constants, and event dataclasses."""

import numpy as np

from sleep_detector_sdk.types import (
    EyeState,
    DetectorState,
    DEFAULT_EAR_THRESHOLD,
    DEFAULT_CLOSED_SECONDS,
    DEFAULT_ALERT_COOLDOWN,
    DrowsinessEvent,
    EyeStateEvent,
    FaceEvent,
    FaceLostEvent,
    FrameEvent,
)


class TestEyeState:
    def test_has_open_closing_closed_states(self):
        assert EyeState.OPEN.value == "open"
        assert EyeState.CLOSING.value == "closing"
        assert EyeState.CLOSED.value == "closed"

    def test_enum_members_are_exactly_three(self):
        assert len(EyeState) == 3


class TestDetectorState:
    def test_has_idle_running_stopped_states(self):
        assert DetectorState.IDLE.value == "idle"
        assert DetectorState.RUNNING.value == "running"
        assert DetectorState.STOPPED.value == "stopped"


class TestConstants:
    def test_default_ear_threshold(self):
        assert DEFAULT_EAR_THRESHOLD == 0.2

    def test_default_closed_seconds(self):
        assert DEFAULT_CLOSED_SECONDS == 5.0

    def test_default_alert_cooldown(self):
        assert DEFAULT_ALERT_COOLDOWN == 3.0


class TestDrowsinessEvent:
    def test_construction(self):
        event = DrowsinessEvent(duration=5.2, ear_value=0.15, timestamp=1000.0)
        assert event.duration == 5.2
        assert event.ear_value == 0.15
        assert event.timestamp == 1000.0


class TestEyeStateEvent:
    def test_construction(self):
        event = EyeStateEvent(
            state=EyeState.CLOSED, ear_value=0.1, timestamp=1000.0
        )
        assert event.state == EyeState.CLOSED
        assert event.ear_value == 0.1


class TestFaceEvent:
    def test_construction(self):
        landmarks = np.zeros((68, 2))
        bbox = (10, 20, 100, 150)
        event = FaceEvent(landmarks=landmarks, bbox=bbox, timestamp=1000.0)
        assert event.bbox == (10, 20, 100, 150)
        assert event.landmarks.shape == (68, 2)


class TestFaceLostEvent:
    def test_construction(self):
        event = FaceLostEvent(last_seen=999.0, timestamp=1000.0)
        assert event.last_seen == 999.0


class TestFrameEvent:
    def test_construction(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        event = FrameEvent(
            frame=frame,
            ear_value=0.25,
            eye_state=EyeState.OPEN,
            face_detected=True,
            timestamp=1000.0,
        )
        assert event.frame.shape == (480, 640, 3)
        assert event.ear_value == 0.25
        assert event.eye_state == EyeState.OPEN
        assert event.face_detected is True


# ---- New types for Next-Gen DMS SDK ----

from sleep_detector_sdk.types import (
    AlertTier,
    GazeZone,
    FatigueSignal,
    SensorMetadata,
    FusionResult,
    GazeEvent,
    TemporalState,
)


class TestAlertTier:
    def test_three_tiers_with_correct_values(self):
        assert AlertTier.SILENT.value == "silent"
        assert AlertTier.AUDIBLE.value == "audible"
        assert AlertTier.CRITICAL.value == "critical"

    def test_exactly_three_members(self):
        assert len(AlertTier) == 3


class TestGazeZone:
    def test_three_zones_with_correct_values(self):
        assert GazeZone.ROAD.value == "road"
        assert GazeZone.IN_VEHICLE.value == "in_vehicle"
        assert GazeZone.EXTERNAL.value == "external"

    def test_exactly_three_members(self):
        assert len(GazeZone) == 3


class TestFatigueSignal:
    def test_construction(self):
        sig = FatigueSignal(score=0.7, confidence=0.9, source="vision", timestamp=1000.0)
        assert sig.score == 0.7
        assert sig.confidence == 0.9
        assert sig.source == "vision"
        assert sig.timestamp == 1000.0

    def test_frozen(self):
        import pytest
        sig = FatigueSignal(score=0.5, confidence=0.8, source="steering", timestamp=500.0)
        with pytest.raises(Exception):
            sig.score = 0.1


class TestFusionResult:
    def test_construction_with_signals_list(self):
        sig = FatigueSignal(score=0.6, confidence=0.85, source="vision", timestamp=1000.0)
        result = FusionResult(
            fatigue_score=0.6,
            tier=AlertTier.AUDIBLE,
            signals=[sig],
            timestamp=1000.0,
        )
        assert result.fatigue_score == 0.6
        assert result.tier == AlertTier.AUDIBLE
        assert len(result.signals) == 1
        assert result.signals[0] is sig
        assert result.timestamp == 1000.0


class TestSensorMetadata:
    def test_construction(self):
        meta = SensorMetadata(name="front_camera", version="1.0.0", sampling_hz=30.0)
        assert meta.name == "front_camera"
        assert meta.version == "1.0.0"
        assert meta.sampling_hz == 30.0


class TestGazeEvent:
    def test_construction(self):
        event = GazeEvent(zone=GazeZone.ROAD, yaw=0.1, pitch=-0.05, timestamp=1000.0)
        assert event.zone == GazeZone.ROAD
        assert event.yaw == 0.1
        assert event.pitch == -0.05
        assert event.timestamp == 1000.0


class TestTemporalState:
    def test_construction_with_all_none(self):
        state = TemporalState(
            t_zero=None,
            t_away=None,
            t_gaze=None,
            t_road=None,
            t_close=None,
            timestamp=1000.0,
        )
        assert state.t_zero is None
        assert state.t_away is None
        assert state.t_gaze is None
        assert state.t_road is None
        assert state.t_close is None
        assert state.timestamp == 1000.0

    def test_construction_with_values(self):
        state = TemporalState(
            t_zero=0.0,
            t_away=1.5,
            t_gaze=2.0,
            t_road=0.5,
            t_close=3.0,
            timestamp=1000.0,
        )
        assert state.t_zero == 0.0
        assert state.t_away == 1.5
