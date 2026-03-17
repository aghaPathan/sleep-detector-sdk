"""Tests for sleep_detector_sdk.ear — Eye Aspect Ratio computation."""

import numpy as np
import pytest

from sleep_detector_sdk.ear import compute_ear


class TestComputeEar:
    def test_open_eye_returns_high_ear(self):
        # Simulate wide-open eye: large vertical distances, normal horizontal
        eye = np.array([
            [0, 0],    # p1 - left corner
            [1, 2],    # p2 - top-left
            [2, 2],    # p3 - top-right
            [3, 0],    # p4 - right corner
            [2, -2],   # p5 - bottom-right
            [1, -2],   # p6 - bottom-left
        ], dtype=np.float64)
        ear = compute_ear(eye)
        assert ear > 0.3, f"Open eye EAR should be > 0.3, got {ear}"

    def test_closed_eye_returns_low_ear(self):
        # Simulate closed eye: very small vertical distances
        eye = np.array([
            [0, 0],      # p1
            [1, 0.05],   # p2 - barely above center
            [2, 0.05],   # p3
            [3, 0],      # p4
            [2, -0.05],  # p5 - barely below center
            [1, -0.05],  # p6
        ], dtype=np.float64)
        ear = compute_ear(eye)
        assert ear < 0.1, f"Closed eye EAR should be < 0.1, got {ear}"

    def test_symmetric_eye_is_consistent(self):
        # Left and right eyes with mirrored landmarks should produce same EAR
        left_eye = np.array([
            [36, 20], [38, 24], [40, 24], [42, 20], [40, 16], [38, 16],
        ], dtype=np.float64)
        right_eye = np.array([
            [58, 20], [56, 24], [54, 24], [52, 20], [54, 16], [56, 16],
        ], dtype=np.float64)
        assert compute_ear(left_eye) == pytest.approx(compute_ear(right_eye), abs=1e-6)

    def test_returns_zero_when_eye_fully_flat(self):
        # All points on same horizontal line
        eye = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0], [2, 0], [1, 0],
        ], dtype=np.float64)
        assert compute_ear(eye) == pytest.approx(0.0, abs=1e-10)

    def test_ear_is_always_non_negative(self):
        rng = np.random.RandomState(42)
        for _ in range(100):
            eye = rng.rand(6, 2) * 100
            assert compute_ear(eye) >= 0
