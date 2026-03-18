"""Lightweight ML-based fatigue scoring model."""

import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATION_WINDOW = 300.0  # 5 minutes
DEFAULT_FEATURE_HISTORY = 100  # samples for calibration


@dataclass
class FatigueFeatures:
    """Feature vector extracted from detection state."""
    ear_value: float
    ear_velocity: float  # rate of change
    eye_closed_ratio: float  # fraction of recent frames with eyes closed
    blink_rate: float  # blinks per minute
    timestamp: float


class CalibrationState:
    """Per-user adaptive calibration that adjusts thresholds."""

    def __init__(self, window_seconds: float = DEFAULT_CALIBRATION_WINDOW):
        self._window = window_seconds
        self._baseline_ears: List[float] = []
        self._start_time: Optional[float] = None
        self._calibrated = False
        self._baseline_ear_mean: float = 0.0
        self._baseline_ear_std: float = 0.0

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def baseline_ear(self) -> Tuple[float, float]:
        return (self._baseline_ear_mean, self._baseline_ear_std)

    def update(self, ear_value: float) -> None:
        if self._calibrated:
            return
        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now
        if ear_value > 0:
            self._baseline_ears.append(ear_value)
        if (now - self._start_time) >= self._window and len(self._baseline_ears) >= 30:
            arr = np.array(self._baseline_ears)
            self._baseline_ear_mean = float(np.mean(arr))
            self._baseline_ear_std = float(np.std(arr))
            self._calibrated = True
            logger.info("Calibration complete: mean=%.3f, std=%.3f", self._baseline_ear_mean, self._baseline_ear_std)

    def reset(self) -> None:
        self._baseline_ears.clear()
        self._start_time = None
        self._calibrated = False
        self._baseline_ear_mean = 0.0
        self._baseline_ear_std = 0.0


class FatigueModel:
    """Adaptive fatigue scoring that improves on static thresholds.

    Uses a simple statistical model based on calibrated baseline.
    When calibrated, scores are based on deviation from the user's
    personal baseline rather than a fixed threshold.
    """

    def __init__(self, static_threshold: float = 0.2, calibration_window: float = DEFAULT_CALIBRATION_WINDOW):
        self._static_threshold = static_threshold
        self._calibration = CalibrationState(window_seconds=calibration_window)
        self._recent_ears: List[float] = []
        self._recent_states: List[bool] = []  # True = closed
        self._max_history = DEFAULT_FEATURE_HISTORY
        self._last_ear: float = 0.0
        self._last_time: float = 0.0
        self._blink_timestamps: List[float] = []

    @property
    def calibration(self) -> CalibrationState:
        return self._calibration

    def score(self, ear_value: float, eye_closed: bool) -> Tuple[float, float]:
        """Compute fatigue score and confidence.

        Returns (score: 0-1, confidence: 0-1).
        Score of 0 = fully alert, 1 = critical fatigue.
        """
        now = time.monotonic()

        # Update history
        self._recent_ears.append(ear_value)
        self._recent_states.append(eye_closed)
        if len(self._recent_ears) > self._max_history:
            self._recent_ears = self._recent_ears[-self._max_history:]
            self._recent_states = self._recent_states[-self._max_history:]

        # Track blinks (transition from closed to open)
        if len(self._recent_states) >= 2 and self._recent_states[-2] and not self._recent_states[-1]:
            self._blink_timestamps.append(now)
        cutoff = now - 60.0
        self._blink_timestamps = [t for t in self._blink_timestamps if t > cutoff]

        # Update calibration
        if not eye_closed and ear_value > 0:
            self._calibration.update(ear_value)

        # Compute velocity
        dt = now - self._last_time if self._last_time > 0 else 0.033
        ear_velocity = (ear_value - self._last_ear) / dt if dt > 0 else 0.0
        self._last_ear = ear_value
        self._last_time = now

        # Compute features
        closed_ratio = sum(self._recent_states) / len(self._recent_states) if self._recent_states else 0.0
        blink_rate = len(self._blink_timestamps)  # per minute

        if self._calibration.is_calibrated:
            return self._adaptive_score(ear_value, ear_velocity, closed_ratio, blink_rate)
        else:
            return self._static_score(ear_value, closed_ratio)

    def _static_score(self, ear_value: float, closed_ratio: float) -> Tuple[float, float]:
        """Fallback static scoring when not calibrated."""
        if self._static_threshold <= 0:
            return (0.0, 0.5)
        ear_score = max(0.0, min(1.0, 1.0 - (ear_value / self._static_threshold)))
        combined = 0.6 * ear_score + 0.4 * closed_ratio
        return (max(0.0, min(1.0, combined)), 0.5)  # lower confidence when uncalibrated

    def _adaptive_score(self, ear_value: float, ear_velocity: float, closed_ratio: float, blink_rate: float) -> Tuple[float, float]:
        """Adaptive scoring using calibrated baseline."""
        mean, std = self._calibration.baseline_ear
        if std == 0:
            std = 0.01  # avoid division by zero

        # Z-score: how far below baseline
        z_score = (mean - ear_value) / std if ear_value < mean else 0.0
        ear_score = max(0.0, min(1.0, z_score / 3.0))  # normalize to 0-1

        # Rapid closing velocity is a fatigue indicator
        velocity_score = max(0.0, min(1.0, abs(ear_velocity) * 2.0)) if ear_velocity < 0 else 0.0

        # High blink rate (>20/min) or very low (<5/min) indicates fatigue
        if blink_rate > 20:
            blink_score = min(1.0, (blink_rate - 20) / 20.0)
        elif blink_rate < 5:
            blink_score = min(1.0, (5 - blink_rate) / 5.0) * 0.3
        else:
            blink_score = 0.0

        # Weighted combination
        combined = (0.35 * ear_score + 0.25 * closed_ratio + 0.25 * velocity_score + 0.15 * blink_score)
        return (max(0.0, min(1.0, combined)), 0.85)  # higher confidence when calibrated

    def save(self, path: str) -> None:
        """Save calibration state to disk."""
        data = {
            'calibration_mean': self._calibration._baseline_ear_mean,
            'calibration_std': self._calibration._baseline_ear_std,
            'calibrated': self._calibration._calibrated,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load calibration state from disk."""
        if not os.path.exists(path):
            return
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._calibration._baseline_ear_mean = data['calibration_mean']
        self._calibration._baseline_ear_std = data['calibration_std']
        self._calibration._calibrated = data['calibrated']
