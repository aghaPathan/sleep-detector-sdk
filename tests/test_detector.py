"""Tests for sleep_detector_sdk.detector — core SleepDetectorSDK class."""

import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Install a mock dlib module before importing detector
mock_dlib = MagicMock()
sys.modules.setdefault("dlib", mock_dlib)

from sleep_detector_sdk.detector import SleepDetectorSDK
from sleep_detector_sdk.types import EyeState


def _make_sdk(**kwargs):
    """Create SDK with all heavy deps mocked."""
    with patch("sleep_detector_sdk.detector.ModelManager") as MockMM:
        MockMM.return_value.resolve.return_value = "/fake/model.dat"

        mock_detector = MagicMock()
        mock_predictor = MagicMock()
        mock_dlib.get_frontal_face_detector.return_value = mock_detector
        mock_dlib.shape_predictor.return_value = mock_predictor

        sdk = SleepDetectorSDK(**kwargs)
        return sdk, mock_detector, mock_predictor


def _mock_shape_for_landmarks(ear_high=True):
    """Create a mock shape that returns landmarks via .part(i)."""
    mock_shape = MagicMock()

    def mock_part(i):
        p = MagicMock()
        if ear_high:
            # Open eyes: large vertical gap
            if 36 <= i <= 41:
                idx = i - 36
                coords = [(36, 20), (38, 26), (40, 26), (42, 20), (40, 14), (38, 14)]
                p.x, p.y = coords[idx]
            elif 42 <= i <= 47:
                idx = i - 42
                coords = [(50, 20), (52, 26), (54, 26), (56, 20), (54, 14), (52, 14)]
                p.x, p.y = coords[idx]
            else:
                p.x, p.y = 0, 0
        else:
            # Closed eyes: tiny vertical gap
            if 36 <= i <= 41:
                idx = i - 36
                coords = [
                    (36, 20), (38, 20.1), (40, 20.1),
                    (42, 20), (40, 19.9), (38, 19.9),
                ]
                p.x, p.y = int(coords[idx][0]), int(coords[idx][1])
            elif 42 <= i <= 47:
                idx = i - 42
                coords = [
                    (50, 20), (52, 20.1), (54, 20.1),
                    (56, 20), (54, 19.9), (52, 19.9),
                ]
                p.x, p.y = int(coords[idx][0]), int(coords[idx][1])
            else:
                p.x, p.y = 0, 0
        return p

    mock_shape.part = mock_part
    return mock_shape


def _mock_face():
    """Create a mock dlib face rectangle."""
    face = MagicMock()
    face.left.return_value = 10
    face.top.return_value = 20
    face.right.return_value = 110
    face.bottom.return_value = 120
    return face


class TestDetectorInit:
    def test_default_configuration(self):
        sdk, _, _ = _make_sdk()
        assert sdk.ear_threshold == 0.2
        assert sdk.closed_duration == 5.0
        assert sdk.alert_cooldown == 3.0

    def test_custom_configuration(self):
        sdk, _, _ = _make_sdk(
            ear_threshold=0.25, closed_duration=3.0, alert_cooldown=1.0
        )
        assert sdk.ear_threshold == 0.25
        assert sdk.closed_duration == 3.0
        assert sdk.alert_cooldown == 1.0


class TestDetectorState:
    def test_initial_state(self):
        sdk, _, _ = _make_sdk()
        assert sdk.is_drowsy is False
        assert sdk.current_ear == 0.0
        assert sdk.eyes_closed_duration == 0.0
        assert sdk.is_running is False


class TestDetectorCallbacks:
    def test_on_registers_callback(self):
        sdk, _, _ = _make_sdk()
        sdk.on("drowsiness_detected", lambda e: None)
        sdk.on("frame_processed", lambda e: None)


class TestProcessFrame:
    def test_emits_frame_processed_no_face(self):
        sdk, mock_detector, _ = _make_sdk()
        mock_detector.return_value = []

        events = []
        sdk.on("frame_processed", lambda e: events.append(e))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sdk.process_frame(frame)

        assert len(events) == 1
        assert events[0].face_detected is False

    def test_emits_face_detected(self):
        sdk, mock_detector, mock_predictor = _make_sdk()
        mock_detector.return_value = [_mock_face()]
        mock_predictor.return_value = _mock_shape_for_landmarks(ear_high=True)

        face_events = []
        sdk.on("face_detected", lambda e: face_events.append(e))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sdk.process_frame(frame)

        assert len(face_events) == 1
        assert face_events[0].bbox == (10, 20, 110, 120)

    def test_emits_face_lost_when_face_disappears(self):
        sdk, mock_detector, mock_predictor = _make_sdk()

        mock_detector.return_value = [_mock_face()]
        mock_predictor.return_value = _mock_shape_for_landmarks(ear_high=True)

        face_lost_events = []
        sdk.on("face_lost", lambda e: face_lost_events.append(e))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sdk.process_frame(frame)  # face present

        mock_detector.return_value = []  # face gone
        sdk.process_frame(frame)

        assert len(face_lost_events) == 1

    def test_ear_updates_on_face_detection(self):
        sdk, mock_detector, mock_predictor = _make_sdk()
        mock_detector.return_value = [_mock_face()]
        mock_predictor.return_value = _mock_shape_for_landmarks(ear_high=True)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sdk.process_frame(frame)

        assert sdk.current_ear > 0

    def test_drowsiness_detected_after_duration(self):
        sdk, mock_detector, mock_predictor = _make_sdk(closed_duration=0.1)
        mock_detector.return_value = [_mock_face()]
        mock_predictor.return_value = _mock_shape_for_landmarks(ear_high=False)

        drowsy_events = []
        sdk.on("drowsiness_detected", lambda e: drowsy_events.append(e))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        sdk.process_frame(frame)
        assert len(drowsy_events) == 0

        time.sleep(0.15)
        sdk.process_frame(frame)
        assert len(drowsy_events) == 1
        assert drowsy_events[0].duration >= 0.1

    def test_eye_state_change_emitted(self):
        sdk, mock_detector, mock_predictor = _make_sdk()
        mock_detector.return_value = [_mock_face()]

        state_events = []
        sdk.on("eye_state_change", lambda e: state_events.append(e))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Open eyes first
        mock_predictor.return_value = _mock_shape_for_landmarks(ear_high=True)
        sdk.process_frame(frame)

        # Close eyes → triggers state change
        mock_predictor.return_value = _mock_shape_for_landmarks(ear_high=False)
        sdk.process_frame(frame)

        assert len(state_events) >= 1
        states = [e.state for e in state_events]
        assert EyeState.CLOSING in states or EyeState.CLOSED in states

    def test_no_drowsiness_when_eyes_reopen(self):
        sdk, mock_detector, mock_predictor = _make_sdk(closed_duration=0.1)
        mock_detector.return_value = [_mock_face()]

        drowsy_events = []
        sdk.on("drowsiness_detected", lambda e: drowsy_events.append(e))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Close eyes briefly
        mock_predictor.return_value = _mock_shape_for_landmarks(ear_high=False)
        sdk.process_frame(frame)

        # Open eyes before duration
        mock_predictor.return_value = _mock_shape_for_landmarks(ear_high=True)
        sdk.process_frame(frame)

        time.sleep(0.15)

        # Process with open eyes — should NOT trigger drowsiness
        sdk.process_frame(frame)
        assert len(drowsy_events) == 0
