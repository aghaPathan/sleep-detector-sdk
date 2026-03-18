"""Physiological sensor plugin (ECG/HRV/Skin Conductance)."""

import logging
import math
import time
from collections import deque
from typing import Deque, List, Optional

from sleep_detector_sdk.sensors import SensorProvider
from sleep_detector_sdk.types import FatigueSignal, SensorMetadata

logger = logging.getLogger(__name__)

HRV_WINDOW_SECONDS = 30.0


class PhysiologicalProvider(SensorProvider):
    """Sensor provider for ECG, HRV, and skin conductance.

    Reads from BLE heart rate monitors via bleak library.
    Falls back to stub mode when no hardware/library available.
    """

    def __init__(self, device_address: Optional[str] = None):
        self._device_address = device_address
        self._connected = False
        self._client = None
        self._hardware_available = False

        # HRV tracking
        self._rr_intervals: Deque[float] = deque(
            maxlen=int(HRV_WINDOW_SECONDS * 3)
        )
        self._rr_timestamps: Deque[float] = deque(
            maxlen=int(HRV_WINDOW_SECONDS * 3)
        )

    def connect(self) -> None:
        self._connected = True

        try:
            import bleak  # noqa: F401

            logger.info("BLE library available, physiological sensor ready")
            self._hardware_available = True
        except ImportError:
            logger.info(
                "Physiological provider in stub mode (bleak not installed)"
            )

    def read(self) -> Optional[FatigueSignal]:
        if not self._connected:
            return None
        if not self._hardware_available:
            return None

        # In real implementation, BLE notifications would populate _rr_intervals
        # For now, compute from whatever data we have
        if len(self._rr_intervals) < 5:
            return None

        now = time.monotonic()
        hrv_score = self._compute_hrv_fatigue()
        if hrv_score is None:
            return None

        return FatigueSignal(
            score=hrv_score,
            confidence=0.8,
            source="physiological",
            timestamp=now,
        )

    def _compute_hrv_fatigue(self) -> Optional[float]:
        """Compute fatigue score from HRV metrics."""
        if len(self._rr_intervals) < 5:
            return None

        intervals = list(self._rr_intervals)

        # SDNN (Standard Deviation of NN intervals)
        mean_rr = sum(intervals) / len(intervals)
        sdnn = math.sqrt(
            sum((rr - mean_rr) ** 2 for rr in intervals) / len(intervals)
        )

        # RMSSD (Root Mean Square of Successive Differences)
        diffs = [
            intervals[i + 1] - intervals[i] for i in range(len(intervals) - 1)
        ]
        rmssd = (
            math.sqrt(sum(d ** 2 for d in diffs) / len(diffs)) if diffs else 0.0
        )

        # Low SDNN and low RMSSD indicate fatigue
        # Normal SDNN: 50-100ms, Normal RMSSD: 20-50ms
        sdnn_score = max(0.0, min(1.0, 1.0 - (sdnn / 100.0)))
        rmssd_score = max(0.0, min(1.0, 1.0 - (rmssd / 50.0)))

        return 0.5 * sdnn_score + 0.5 * rmssd_score

    def add_rr_interval(self, rr_ms: float) -> None:
        """Manually add an RR interval (for testing or external data sources)."""
        self._rr_intervals.append(rr_ms)
        self._rr_timestamps.append(time.monotonic())

    def disconnect(self) -> None:
        self._client = None
        self._connected = False
        self._hardware_available = False

    def metadata(self) -> SensorMetadata:
        return SensorMetadata(
            name="physiological", version="0.2.0", sampling_hz=25.0
        )
