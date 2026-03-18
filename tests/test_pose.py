"""Tests for sleep_detector_sdk.pose — head pose estimation and nod detection."""

import numpy as np
import pytest

from sleep_detector_sdk.pose import HeadPoseResult, NodDetector, PoseEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_centered_landmarks() -> np.ndarray:
    """68-point landmark array with a roughly forward-facing head.

    Landmark positions are computed by projecting the 6-point 3-D face model
    at zero rotation (R = I, with 180° X-flip to align model Y with image Y)
    at z=800mm using a 640×480 camera.  This guarantees solvePnP returns
    near-zero euler angles after the 180° pitch normalisation applied by the
    estimator.

    Nose tip (30): (320, 240) — image centre
    Chin     (8):  (320, 484)
    L-eye   (36):  (166, 124)
    R-eye   (45):  (474, 124)
    M-left  (48):  (216, 344)
    M-right (54):  (424, 344)
    """
    pts = np.zeros((68, 2), dtype=np.float64)
    pts[30] = [320.0, 240.0]
    pts[8]  = [320.0, 484.2]
    pts[36] = [166.0, 123.6]
    pts[45] = [474.0, 123.6]
    pts[48] = [216.2, 343.8]
    pts[54] = [423.8, 343.8]
    return pts


def _make_tilted_landmarks() -> np.ndarray:
    """68-point landmark array simulating a head tilted (rolled) to the right.

    Left eye (36) is higher on screen (lower y) than right eye (45).
    """
    pts = _make_centered_landmarks().copy()
    pts[36] = [166.0, 103.6]   # left eye: 20px higher
    pts[45] = [474.0, 143.6]   # right eye: 20px lower
    return pts


# ---------------------------------------------------------------------------
# HeadPoseResult
# ---------------------------------------------------------------------------

class TestHeadPoseResult:
    def test_is_frozen_dataclass(self):
        result = HeadPoseResult(yaw=1.0, pitch=2.0, roll=3.0)
        with pytest.raises((AttributeError, TypeError)):
            result.yaw = 99.0  # type: ignore[misc]

    def test_fields_accessible(self):
        result = HeadPoseResult(yaw=10.0, pitch=-5.0, roll=3.5)
        assert result.yaw == 10.0
        assert result.pitch == -5.0
        assert result.roll == 3.5


# ---------------------------------------------------------------------------
# PoseEstimator
# ---------------------------------------------------------------------------

class TestPoseEstimator:
    def test_returns_head_pose_result(self):
        estimator = PoseEstimator()
        landmarks = _make_centered_landmarks()
        result = estimator.estimate(landmarks, frame_shape=(480, 640))
        assert isinstance(result, HeadPoseResult)

    def test_forward_pose_angles_near_zero(self):
        """A roughly forward-facing head should have small yaw/pitch/roll."""
        estimator = PoseEstimator()
        landmarks = _make_centered_landmarks()
        result = estimator.estimate(landmarks, frame_shape=(480, 640))
        assert abs(result.yaw) < 15, f"yaw={result.yaw} should be near zero"
        assert abs(result.pitch) < 15, f"pitch={result.pitch} should be near zero"
        assert abs(result.roll) < 15, f"roll={result.roll} should be near zero"

    def test_tilted_head_has_nonzero_roll(self):
        """When left eye is higher than right eye, roll should be nonzero."""
        estimator = PoseEstimator()
        landmarks = _make_tilted_landmarks()
        result = estimator.estimate(landmarks, frame_shape=(480, 640))
        assert abs(result.roll) > 1.0, (
            f"Tilted head roll={result.roll} should be noticeably nonzero"
        )

    def test_result_has_float_attributes(self):
        estimator = PoseEstimator()
        landmarks = _make_centered_landmarks()
        result = estimator.estimate(landmarks, frame_shape=(480, 640))
        assert isinstance(result.yaw, float)
        assert isinstance(result.pitch, float)
        assert isinstance(result.roll, float)


# ---------------------------------------------------------------------------
# NodDetector
# ---------------------------------------------------------------------------

class TestNodDetector:
    def test_no_nod_when_stable(self):
        """Constant pitch should not trigger a nod."""
        detector = NodDetector(window_size=20, pitch_threshold=10.0)
        for _ in range(20):
            detector.update(0.0)
        assert detector.is_nodding is False

    def test_detects_nod_pattern(self):
        """Oscillating pitch exceeding threshold should trigger nodding."""
        detector = NodDetector(window_size=20, pitch_threshold=10.0)
        # Oscillate: 0, -15, 0, -15, 0, -15, 0, -15 — many direction changes > threshold
        pattern = [0.0, -15.0, 0.0, -15.0, 0.0, -15.0, 0.0, -15.0]
        for pitch in pattern:
            detector.update(pitch)
        assert detector.is_nodding is True

    def test_no_nod_when_changes_below_threshold(self):
        """Small oscillations that don't exceed threshold should not nod."""
        detector = NodDetector(window_size=20, pitch_threshold=10.0)
        pattern = [0.0, -3.0, 0.0, -3.0, 0.0, -3.0, 0.0, -3.0]
        for pitch in pattern:
            detector.update(pitch)
        assert detector.is_nodding is False

    def test_reset_clears_state(self):
        """After reset(), is_nodding should be False regardless of prior updates."""
        detector = NodDetector(window_size=20, pitch_threshold=10.0)
        # Prime with a nod pattern
        for pitch in [0.0, -15.0, 0.0, -15.0, 0.0, -15.0, 0.0, -15.0]:
            detector.update(pitch)
        assert detector.is_nodding is True
        detector.reset()
        assert detector.is_nodding is False

    def test_window_size_limits_history(self):
        """Only the last window_size samples should be considered."""
        detector = NodDetector(window_size=5, pitch_threshold=10.0)
        # Fill with a nod pattern first
        for pitch in [0.0, -15.0, 0.0, -15.0, 0.0, -15.0]:
            detector.update(pitch)
        # Then overwrite with stable data — window only holds last 5
        for _ in range(5):
            detector.update(0.0)
        assert detector.is_nodding is False
