"""Tests for sleep_detector_sdk.gaze — GazeEstimator."""

import numpy as np
import pytest

from sleep_detector_sdk.gaze import GazeEstimator
from sleep_detector_sdk.types import GazeEvent, GazeZone


# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------

def _base_landmarks() -> np.ndarray:
    """68-point face landmark array with neutral/default positions.

    Coordinate system: x increases right, y increases down.
    Key points used by GazeEstimator:
      - 36-41: right eye (dlib ordering)
      - 42-47: left eye
      - 30: nose tip
      - 8: chin

    Geometry contract (default thresholds = 0.15):
      - Right eye center: x=125, y=100
      - Left  eye center: x=175, y=100
      - Eye midpoint:     x=150, y=100
      - Inter-eye dist:   50
      - Chin: y=400  → face_height = 300
      - Nose (forward):   x=150, y=130
        yaw   = (150-150)/50  = 0.0
        pitch = (130-100)/300 = 0.10  < 0.15  → ROAD ✓
    """
    pts = np.zeros((68, 2), dtype=np.float64)

    # Right eye (landmarks 36-41) — centered around x=125, y=100
    pts[36] = [105, 100]
    pts[37] = [115,  90]
    pts[38] = [125,  90]
    pts[39] = [135, 100]
    pts[40] = [125, 110]
    pts[41] = [115, 110]

    # Left eye (landmarks 42-47) — centered around x=175, y=100
    pts[42] = [165, 100]
    pts[43] = [175,  90]
    pts[44] = [185,  90]
    pts[45] = [195, 100]
    pts[46] = [185, 110]
    pts[47] = [175, 110]

    # Eye midpoint: x=150, y=100
    # Inter-eye distance (175-125) = 50

    # Nose tip (landmark 30) — directly below eye midpoint, pitch within ROAD
    # pitch = (nose_y - eye_mid_y) / face_height = (130-100)/300 = 0.10 < 0.15
    pts[30] = [150, 130]

    # Chin far below to give large face_height (face_height = 400-100 = 300)
    pts[8] = [150, 400]

    return pts


def _make_forward_landmarks() -> np.ndarray:
    """Symmetric centered face — nose aligned with eye midpoint horizontally.

    yaw   = 0.0,  pitch = 0.10  → ROAD
    """
    return _base_landmarks()


def _make_left_gaze_landmarks() -> np.ndarray:
    """Nose shifted left — simulates head turned left.

    yaw = (130-150)/50 = -0.40  → EXTERNAL (> 0.15*2=0.30)
    """
    pts = _base_landmarks()
    pts[30] = [130, 130]  # nose shifted left: yaw = -20/50 = -0.40
    return pts


def _make_down_gaze_landmarks() -> np.ndarray:
    """Nose tip lower — simulates downward head pitch into IN_VEHICLE range.

    pitch = (nose_y - eye_mid_y) / face_height
    Target: 0.15 < pitch <= 0.30  (between threshold and 2*threshold)
    pitch = (nose_y - 100) / (chin_y - 100)
    Set chin_y=400, face_height=300 → nose_y for pitch=0.20: 100 + 0.20*300 = 160
    yaw=0 (nose x centered)
    """
    pts = _base_landmarks()
    pts[30] = [150, 160]   # pitch = (160-100)/300 = 0.20 → IN_VEHICLE
    pts[8]  = [150, 400]   # chin unchanged
    return pts


def _make_extreme_gaze_landmarks() -> np.ndarray:
    """Nose far to the left — simulates extreme head rotation.

    yaw = (100-150)/50 = -1.0  → EXTERNAL (> 0.15*2)
    """
    pts = _base_landmarks()
    pts[30] = [100, 130]  # nose far left: yaw = -50/50 = -1.0
    return pts


def _make_slight_left_landmarks() -> np.ndarray:
    """Nose slightly left — yaw just over 0.15 (default threshold).

    yaw = (nose_x - 150) / 50
    Target: 0.15 < |yaw| <= 0.30 → use offset = -0.16*50 = -8 → nose_x=142
    """
    pts = _base_landmarks()
    pts[30] = [142, 130]  # yaw = (142-150)/50 = -0.16 → IN_VEHICLE with default thresh
    return pts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGazeEstimatorForwardGaze:
    def test_forward_gaze_returns_road_zone(self):
        estimator = GazeEstimator()
        landmarks = _make_forward_landmarks()
        event = estimator.estimate(landmarks)
        assert event.zone == GazeZone.ROAD

    def test_forward_gaze_returns_gaze_event(self):
        estimator = GazeEstimator()
        landmarks = _make_forward_landmarks()
        event = estimator.estimate(landmarks)
        assert isinstance(event, GazeEvent)

    def test_forward_gaze_yaw_near_zero(self):
        estimator = GazeEstimator()
        landmarks = _make_forward_landmarks()
        event = estimator.estimate(landmarks)
        assert abs(event.yaw) < 0.05, f"Forward gaze yaw should be near 0, got {event.yaw}"

    def test_forward_gaze_pitch_near_zero(self):
        estimator = GazeEstimator()
        landmarks = _make_forward_landmarks()
        event = estimator.estimate(landmarks)
        assert abs(event.pitch) < 0.5, f"Forward gaze pitch should be small, got {event.pitch}"


class TestGazeEstimatorLeftGaze:
    def test_leftward_gaze_returns_non_road(self):
        estimator = GazeEstimator()
        landmarks = _make_left_gaze_landmarks()
        event = estimator.estimate(landmarks)
        assert event.zone != GazeZone.ROAD

    def test_leftward_gaze_has_negative_yaw(self):
        """Nose shifted left → negative yaw (nose left of eye midpoint)."""
        estimator = GazeEstimator()
        landmarks = _make_left_gaze_landmarks()
        event = estimator.estimate(landmarks)
        assert event.yaw < 0, f"Left gaze should have negative yaw, got {event.yaw}"


class TestGazeEstimatorDownGaze:
    def test_downward_gaze_returns_in_vehicle(self):
        """Nose lowered past threshold but not extreme → IN_VEHICLE zone."""
        estimator = GazeEstimator()
        landmarks = _make_down_gaze_landmarks()
        event = estimator.estimate(landmarks)
        assert event.zone == GazeZone.IN_VEHICLE

    def test_downward_gaze_has_positive_pitch(self):
        """Nose below eye midpoint → positive pitch."""
        estimator = GazeEstimator()
        landmarks = _make_down_gaze_landmarks()
        event = estimator.estimate(landmarks)
        assert event.pitch > 0, f"Downward gaze should have positive pitch, got {event.pitch}"


class TestGazeEstimatorExtremeGaze:
    def test_extreme_gaze_returns_external(self):
        estimator = GazeEstimator()
        landmarks = _make_extreme_gaze_landmarks()
        event = estimator.estimate(landmarks)
        assert event.zone == GazeZone.EXTERNAL

    def test_extreme_gaze_large_yaw(self):
        estimator = GazeEstimator()
        landmarks = _make_extreme_gaze_landmarks()
        event = estimator.estimate(landmarks)
        assert abs(event.yaw) > 0.3, f"Extreme gaze should have large yaw, got {event.yaw}"


class TestGazeEstimatorCustomThresholds:
    def test_lower_threshold_catches_slight_deviation(self):
        """With threshold=0.05, a slight left deviation should trigger IN_VEHICLE."""
        estimator = GazeEstimator(yaw_threshold=0.05, pitch_threshold=0.05)
        landmarks = _make_slight_left_landmarks()
        event = estimator.estimate(landmarks)
        assert event.zone != GazeZone.ROAD

    def test_higher_threshold_ignores_slight_deviation(self):
        """With threshold=0.5, a slight left deviation stays ROAD."""
        estimator = GazeEstimator(yaw_threshold=0.5, pitch_threshold=0.5)
        landmarks = _make_slight_left_landmarks()
        event = estimator.estimate(landmarks)
        assert event.zone == GazeZone.ROAD

    def test_custom_thresholds_stored(self):
        estimator = GazeEstimator(yaw_threshold=0.1, pitch_threshold=0.2)
        assert estimator.yaw_threshold == 0.1
        assert estimator.pitch_threshold == 0.2


class TestGazeEstimatorReturnsValues:
    def test_estimate_returns_yaw_float(self):
        estimator = GazeEstimator()
        event = estimator.estimate(_make_forward_landmarks())
        assert isinstance(event.yaw, float)

    def test_estimate_returns_pitch_float(self):
        estimator = GazeEstimator()
        event = estimator.estimate(_make_forward_landmarks())
        assert isinstance(event.pitch, float)

    def test_estimate_returns_timestamp_float(self):
        estimator = GazeEstimator()
        event = estimator.estimate(_make_forward_landmarks())
        assert isinstance(event.timestamp, float)
        assert event.timestamp > 0
