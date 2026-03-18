"""Steering and vehicle data sensor plugin (CAN/OBD-II)."""

import logging
import time
from collections import deque
from typing import Deque, Optional

from sleep_detector_sdk.sensors import SensorProvider
from sleep_detector_sdk.types import FatigueSignal, SensorMetadata

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_MINUTES = 20.0
BASELINE_SAMPLE_INTERVAL = 1.0  # seconds between baseline samples


class SteeringProvider(SensorProvider):
    """Sensor provider for steering wheel angle and vehicle data.

    Reads from CAN bus (python-can) or OBD-II (obd library).
    Falls back to stub mode when no hardware/library available.
    """

    def __init__(
        self,
        baseline_minutes: float = DEFAULT_BASELINE_MINUTES,
        can_channel: str = "can0",
        can_interface: str = "socketcan",
    ):
        self.baseline_minutes = baseline_minutes
        self._can_channel = can_channel
        self._can_interface = can_interface
        self._connected = False
        self._bus = None
        self._obd_connection = None
        self._hardware_available = False

        # Baseline tracking
        self._baseline_angles: Deque[float] = deque(
            maxlen=int(baseline_minutes * 60 / BASELINE_SAMPLE_INTERVAL)
        )
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 0.0
        self._baseline_calibrated: bool = False
        self._baseline_start: Optional[float] = None
        self._last_sample_time: float = 0.0

    def connect(self) -> None:
        self._connected = True
        self._baseline_start = time.monotonic()

        # Try CAN bus first
        try:
            import can

            self._bus = can.interface.Bus(
                channel=self._can_channel, interface=self._can_interface
            )
            self._hardware_available = True
            logger.info(
                "Steering provider connected via CAN bus (%s)", self._can_channel
            )
            return
        except (ImportError, Exception) as e:
            logger.debug("CAN bus not available: %s", e)

        # Try OBD-II fallback
        try:
            import obd

            self._obd_connection = obd.OBD()
            if self._obd_connection.is_connected():
                self._hardware_available = True
                logger.info("Steering provider connected via OBD-II")
                return
            else:
                self._obd_connection = None
        except (ImportError, Exception) as e:
            logger.debug("OBD-II not available: %s", e)

        logger.info("Steering provider in stub mode (no hardware detected)")

    def read(self) -> Optional[FatigueSignal]:
        if not self._connected:
            return None

        angle = self._read_steering_angle()
        if angle is None:
            return None

        now = time.monotonic()

        # Update baseline
        if not self._baseline_calibrated and (
            now - self._last_sample_time
        ) >= BASELINE_SAMPLE_INTERVAL:
            self._baseline_angles.append(angle)
            self._last_sample_time = now

            if self._baseline_start and (
                now - self._baseline_start
            ) >= self._baseline_minutes * 60:
                if len(self._baseline_angles) >= 30:
                    import numpy as np

                    arr = np.array(list(self._baseline_angles))
                    self._baseline_mean = float(np.mean(arr))
                    self._baseline_std = float(np.std(arr))
                    self._baseline_calibrated = True
                    logger.info(
                        "Steering baseline calibrated: mean=%.1f, std=%.1f",
                        self._baseline_mean,
                        self._baseline_std,
                    )

        if not self._baseline_calibrated:
            return None  # Not enough data yet

        # Score based on deviation from baseline
        deviation = abs(angle - self._baseline_mean)
        if self._baseline_std > 0:
            z_score = deviation / self._baseline_std
        else:
            z_score = 0.0

        score = max(0.0, min(1.0, z_score / 4.0))  # Normalize
        return FatigueSignal(score=score, confidence=0.7, source="steering", timestamp=now)

    def _read_steering_angle(self) -> Optional[float]:
        """Read steering wheel angle from hardware."""
        if self._bus is not None:
            try:
                msg = self._bus.recv(timeout=0.01)
                if msg and msg.arbitration_id == 0x025:  # Common steering angle CAN ID
                    angle = (
                        int.from_bytes(msg.data[0:2], byteorder="big", signed=True)
                        * 0.1
                    )
                    return angle
            except Exception:
                pass

        if self._obd_connection is not None:
            try:
                import obd

                resp = self._obd_connection.query(obd.commands.STEERING_ANGLE)
                if not resp.is_null():
                    return resp.value.magnitude
            except Exception:
                pass

        return None

    def disconnect(self) -> None:
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:
                pass
            self._bus = None
        if self._obd_connection is not None:
            try:
                self._obd_connection.close()
            except Exception:
                pass
            self._obd_connection = None
        self._connected = False
        self._hardware_available = False

    def metadata(self) -> SensorMetadata:
        return SensorMetadata(name="steering", version="0.2.0", sampling_hz=10.0)
