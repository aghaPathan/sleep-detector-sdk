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
