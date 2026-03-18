"""Physiological sensor plugin stub (heart rate, GSR, etc.)."""

import logging
from typing import Optional

from sleep_detector_sdk.sensors import SensorProvider
from sleep_detector_sdk.types import FatigueSignal, SensorMetadata

logger = logging.getLogger(__name__)


class PhysiologicalProvider(SensorProvider):
    """Stub physiological sensor provider.

    No real hardware is accessed; all reads return None.  Intended to validate
    the plugin contract and entry-point registration for biometric sensors
    (e.g. heart rate, galvanic skin response).
    """

    def __init__(self) -> None:
        self._connected: bool = False

    def connect(self) -> None:
        """Set connected state; logs stub mode."""
        self._connected = True
        logger.info("PhysiologicalProvider: connected (stub mode)")

    def read(self) -> Optional[FatigueSignal]:
        """Return None — no real hardware."""
        return None

    def disconnect(self) -> None:
        """Clear connected state."""
        self._connected = False
        logger.info("PhysiologicalProvider: disconnected")

    def metadata(self) -> SensorMetadata:
        """Return static metadata for this sensor."""
        return SensorMetadata(
            name="physiological", version="0.1.0", sampling_hz=25.0
        )
