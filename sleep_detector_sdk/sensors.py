"""Sensor plugin architecture — SensorProvider ABC and SensorRegistry."""

import logging
import threading
from abc import ABC, abstractmethod
from typing import List, Optional

from sleep_detector_sdk.types import FatigueSignal, SensorMetadata

logger = logging.getLogger(__name__)


class SensorProvider(ABC):
    """Abstract base class for all fatigue sensor providers.

    Concrete implementations must supply connect(), read(), disconnect(),
    and metadata().  The registry calls these methods; implementations are
    responsible for managing any internal state (e.g. connection handles).
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish the sensor connection."""

    @abstractmethod
    def read(self) -> Optional[FatigueSignal]:
        """Sample the sensor.

        Returns a FatigueSignal if a measurement is available, or None when
        the sensor has nothing to report (e.g. low-confidence frame).
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Release the sensor connection."""

    @abstractmethod
    def metadata(self) -> SensorMetadata:
        """Return static metadata describing this sensor."""


class SensorRegistry:
    """Thread-safe registry of SensorProvider instances.

    Usage::

        registry = SensorRegistry()
        registry.register(EarSensor())
        registry.connect_all()
        signals = registry.read_all()
        registry.disconnect_all()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sensors: List[SensorProvider] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, provider: SensorProvider) -> None:
        """Add *provider* to the registry.

        Raises:
            ValueError: if a sensor with the same metadata name is already
                registered.
        """
        name = provider.metadata().name
        with self._lock:
            for existing in self._sensors:
                if existing.metadata().name == name:
                    raise ValueError(
                        f"A sensor named '{name}' is already registered."
                    )
            self._sensors.append(provider)

    def unregister(self, name: str) -> None:
        """Remove the sensor identified by *name*.

        Raises:
            ValueError: if no sensor with that name is registered.
        """
        with self._lock:
            for i, sensor in enumerate(self._sensors):
                if sensor.metadata().name == name:
                    del self._sensors[i]
                    return
            raise ValueError(f"No sensor named '{name}' is registered.")

    @property
    def sensors(self) -> List[SensorProvider]:
        """Return a snapshot copy of the registered sensors list."""
        with self._lock:
            return list(self._sensors)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect_all(self) -> None:
        """Call connect() on every registered sensor.

        Exceptions from individual sensors are logged and suppressed so that
        a single failing sensor does not prevent the others from connecting.
        """
        for sensor in self.sensors:
            try:
                sensor.connect()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Error connecting sensor '%s'", sensor.metadata().name
                )

    def disconnect_all(self) -> None:
        """Call disconnect() on every registered sensor.

        Exceptions are logged and suppressed — same policy as connect_all().
        """
        for sensor in self.sensors:
            try:
                sensor.disconnect()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Error disconnecting sensor '%s'", sensor.metadata().name
                )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_all(self) -> List[FatigueSignal]:
        """Read from all registered sensors and return non-None signals.

        Sensors whose read() returns None are silently skipped.
        """
        signals: List[FatigueSignal] = []
        for sensor in self.sensors:
            result = sensor.read()
            if result is not None:
                signals.append(result)
        return signals
