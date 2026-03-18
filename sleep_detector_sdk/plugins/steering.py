"""Steering wheel sensor plugin stub."""

import logging
from typing import Optional

from sleep_detector_sdk.sensors import SensorProvider
from sleep_detector_sdk.types import FatigueSignal, SensorMetadata

logger = logging.getLogger(__name__)


class SteeringProvider(SensorProvider):
    """Stub steering wheel sensor provider.

    No real hardware is accessed; all reads return None.  Exists to validate
    the plugin contract and entry-point registration.
    """

    def __init__(self, baseline_minutes: float = 20.0) -> None:
        self.baseline_minutes = baseline_minutes
        self._connected: bool = False

    def connect(self) -> None:
        """Set connected state; logs stub mode."""
        self._connected = True
        logger.info("SteeringProvider: connected (stub mode)")

    def read(self) -> Optional[FatigueSignal]:
        """Return None — no real hardware."""
        return None

    def disconnect(self) -> None:
        """Clear connected state."""
        self._connected = False
        logger.info("SteeringProvider: disconnected")

    def metadata(self) -> SensorMetadata:
        """Return static metadata for this sensor."""
        return SensorMetadata(name="steering", version="0.1.0", sampling_hz=10.0)
