"""Tests for the adaptive fatigue model."""
import time
import os
import tempfile

import pytest

from sleep_detector_sdk.fatigue_model import FatigueModel, CalibrationState, FatigueFeatures


class TestCalibrationState:
    def test_initial_not_calibrated(self):
        cal = CalibrationState()
        assert not cal.is_calibrated

    def test_calibrates_after_window(self):
        cal = CalibrationState(window_seconds=0.0)  # immediate
        for _ in range(30):
            cal.update(0.3)
        assert cal.is_calibrated
        mean, std = cal.baseline_ear
        assert abs(mean - 0.3) < 0.01

    def test_ignores_zero_ear(self):
        cal = CalibrationState(window_seconds=0.0)
        for _ in range(30):
            cal.update(0.0)
        assert not cal.is_calibrated  # zeros ignored

    def test_reset_clears_calibration(self):
        cal = CalibrationState(window_seconds=0.0)
        for _ in range(30):
            cal.update(0.3)
        assert cal.is_calibrated
        cal.reset()
        assert not cal.is_calibrated

    def test_stops_updating_after_calibrated(self):
        cal = CalibrationState(window_seconds=0.0)
        for _ in range(30):
            cal.update(0.3)
        mean1, _ = cal.baseline_ear
        for _ in range(30):
            cal.update(0.5)
        mean2, _ = cal.baseline_ear
        assert mean1 == mean2  # unchanged


class TestFatigueModel:
    def test_static_score_eyes_open(self):
        model = FatigueModel(static_threshold=0.2)
        score, conf = model.score(ear_value=0.3, eye_closed=False)
        assert score < 0.3  # eyes open, above threshold
        assert conf == 0.5  # uncalibrated

    def test_static_score_eyes_closed(self):
        model = FatigueModel(static_threshold=0.2)
        score, conf = model.score(ear_value=0.05, eye_closed=True)
        assert score > 0.5  # eyes closed, below threshold

    def test_confidence_increases_after_calibration(self):
        model = FatigueModel(static_threshold=0.2, calibration_window=0.0)
        # Calibrate with open eyes
        for _ in range(35):
            model.score(ear_value=0.3, eye_closed=False)
        assert model.calibration.is_calibrated
        _, conf = model.score(ear_value=0.3, eye_closed=False)
        assert conf > 0.5  # higher confidence when calibrated

    def test_adaptive_score_below_baseline(self):
        model = FatigueModel(static_threshold=0.2, calibration_window=0.0)
        for _ in range(35):
            model.score(ear_value=0.3, eye_closed=False)
        # Now test with eyes closing
        score, _ = model.score(ear_value=0.1, eye_closed=True)
        assert score > 0.0

    def test_adaptive_score_at_baseline(self):
        model = FatigueModel(static_threshold=0.2, calibration_window=0.0)
        for _ in range(35):
            model.score(ear_value=0.3, eye_closed=False)
        score, _ = model.score(ear_value=0.3, eye_closed=False)
        assert score < 0.2  # at baseline = low fatigue

    def test_save_and_load_calibration(self):
        model = FatigueModel(calibration_window=0.0)
        for _ in range(35):
            model.score(ear_value=0.28, eye_closed=False)
        assert model.calibration.is_calibrated

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            model.save(path)
            model2 = FatigueModel()
            assert not model2.calibration.is_calibrated
            model2.load(path)
            assert model2.calibration.is_calibrated
            mean, _ = model2.calibration.baseline_ear
            assert abs(mean - 0.28) < 0.01
        finally:
            os.unlink(path)

    def test_closed_ratio_increases_score(self):
        model = FatigueModel(static_threshold=0.2)
        # All closed
        for _ in range(10):
            model.score(ear_value=0.05, eye_closed=True)
        score_closed, _ = model.score(ear_value=0.05, eye_closed=True)

        model2 = FatigueModel(static_threshold=0.2)
        # All open
        for _ in range(10):
            model2.score(ear_value=0.3, eye_closed=False)
        score_open, _ = model2.score(ear_value=0.3, eye_closed=False)

        assert score_closed > score_open


class TestFatigueFeatures:
    def test_construction(self):
        f = FatigueFeatures(ear_value=0.3, ear_velocity=-0.1, eye_closed_ratio=0.2, blink_rate=15.0, timestamp=1.0)
        assert f.ear_value == 0.3
        assert f.blink_rate == 15.0
