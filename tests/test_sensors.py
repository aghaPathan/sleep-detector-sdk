"""Tests for the sensor plugin architecture (SensorProvider ABC + SensorRegistry)."""

import threading
import time
from typing import Optional

import pytest

from sleep_detector_sdk.sensors import SensorProvider, SensorRegistry
from sleep_detector_sdk.types import FatigueSignal, SensorMetadata


# ---------------------------------------------------------------------------
# Test helper — DummySensor
# ---------------------------------------------------------------------------

class DummySensor(SensorProvider):
    """Concrete SensorProvider for testing purposes."""

    def __init__(self, name: str = "dummy", fail_on_read: bool = False):
        self._name = name
        self._connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.read_calls = 0
        self._fail_on_read = fail_on_read

    def connect(self) -> None:
        self.connect_calls += 1
        self._connected = True

    def read(self) -> Optional[FatigueSignal]:
        self.read_calls += 1
        if self._fail_on_read:
            return None
        return FatigueSignal(
            score=0.5,
            confidence=0.9,
            source=self._name,
            timestamp=time.time(),
        )

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False

    def metadata(self) -> SensorMetadata:
        return SensorMetadata(name=self._name, version="1.0.0", sampling_hz=30.0)


class ErrorSensor(DummySensor):
    """Sensor that raises on connect/disconnect to test exception handling."""

    def connect(self) -> None:
        raise RuntimeError("connect failed")

    def disconnect(self) -> None:
        raise RuntimeError("disconnect failed")


# ---------------------------------------------------------------------------
# TestSensorProvider
# ---------------------------------------------------------------------------

class TestSensorProvider:
    def test_cannot_instantiate_abc_directly(self):
        """SensorProvider is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            SensorProvider()  # type: ignore

    def test_concrete_sensor_connect_sets_state(self):
        """Calling connect() on a DummySensor increments connect_calls."""
        sensor = DummySensor()
        assert sensor.connect_calls == 0
        sensor.connect()
        assert sensor.connect_calls == 1

    def test_concrete_sensor_read_returns_fatigue_signal(self):
        """read() returns a FatigueSignal instance."""
        sensor = DummySensor()
        sensor.connect()
        result = sensor.read()
        assert isinstance(result, FatigueSignal)
        assert result.source == "dummy"

    def test_concrete_sensor_disconnect_sets_state(self):
        """disconnect() increments disconnect_calls."""
        sensor = DummySensor()
        sensor.connect()
        sensor.disconnect()
        assert sensor.disconnect_calls == 1

    def test_concrete_sensor_metadata_returns_sensor_metadata(self):
        """metadata() returns a SensorMetadata instance with correct values."""
        sensor = DummySensor(name="test-sensor")
        meta = sensor.metadata()
        assert isinstance(meta, SensorMetadata)
        assert meta.name == "test-sensor"
        assert meta.version == "1.0.0"
        assert meta.sampling_hz == 30.0

    def test_concrete_sensor_read_can_return_none(self):
        """read() is allowed to return None (no signal available)."""
        sensor = DummySensor(fail_on_read=True)
        sensor.connect()
        result = sensor.read()
        assert result is None

    def test_concrete_sensor_full_lifecycle(self):
        """Connect → read → disconnect lifecycle works end-to-end."""
        sensor = DummySensor()
        sensor.connect()
        assert sensor.connect_calls == 1
        signal = sensor.read()
        assert signal is not None
        sensor.disconnect()
        assert sensor.disconnect_calls == 1


# ---------------------------------------------------------------------------
# TestSensorRegistry
# ---------------------------------------------------------------------------

class TestSensorRegistry:
    def test_register_and_list_sensors(self):
        """Registered sensor appears in the sensors list."""
        registry = SensorRegistry()
        sensor = DummySensor("s1")
        registry.register(sensor)
        assert sensor in registry.sensors

    def test_sensors_property_returns_copy(self):
        """Mutating the returned list does not affect the registry."""
        registry = SensorRegistry()
        sensor = DummySensor("s1")
        registry.register(sensor)
        snapshot = registry.sensors
        snapshot.clear()
        assert sensor in registry.sensors

    def test_duplicate_name_raises_value_error(self):
        """Registering two sensors with the same name raises ValueError."""
        registry = SensorRegistry()
        registry.register(DummySensor("dup"))
        with pytest.raises(ValueError, match="dup"):
            registry.register(DummySensor("dup"))

    def test_unregister_removes_sensor(self):
        """unregister() removes the named sensor from the registry."""
        registry = SensorRegistry()
        sensor = DummySensor("removable")
        registry.register(sensor)
        registry.unregister("removable")
        assert sensor not in registry.sensors

    def test_unregister_unknown_name_raises_value_error(self):
        """unregister() raises ValueError for a name not in the registry."""
        registry = SensorRegistry()
        with pytest.raises(ValueError, match="ghost"):
            registry.unregister("ghost")

    def test_connect_all_calls_connect_on_each_sensor(self):
        """connect_all() calls connect() on every registered sensor."""
        registry = SensorRegistry()
        s1 = DummySensor("a")
        s2 = DummySensor("b")
        registry.register(s1)
        registry.register(s2)
        registry.connect_all()
        assert s1.connect_calls == 1
        assert s2.connect_calls == 1

    def test_connect_all_continues_on_exception(self):
        """connect_all() does not propagate exceptions — other sensors still connect."""
        registry = SensorRegistry()
        error_sensor = ErrorSensor("err")
        good_sensor = DummySensor("good")
        registry.register(error_sensor)
        registry.register(good_sensor)
        # Should not raise
        registry.connect_all()
        assert good_sensor.connect_calls == 1

    def test_disconnect_all_calls_disconnect_on_each_sensor(self):
        """disconnect_all() calls disconnect() on every registered sensor."""
        registry = SensorRegistry()
        s1 = DummySensor("a")
        s2 = DummySensor("b")
        registry.register(s1)
        registry.register(s2)
        registry.connect_all()
        registry.disconnect_all()
        assert s1.disconnect_calls == 1
        assert s2.disconnect_calls == 1

    def test_disconnect_all_continues_on_exception(self):
        """disconnect_all() does not propagate exceptions — other sensors still disconnect."""
        registry = SensorRegistry()
        good_sensor = DummySensor("good")
        error_sensor = ErrorSensor("err")
        registry.register(good_sensor)
        registry.register(error_sensor)
        # Should not raise
        registry.disconnect_all()
        assert good_sensor.disconnect_calls == 1

    def test_read_all_returns_fatigue_signals(self):
        """read_all() returns a list of FatigueSignal from all sensors."""
        registry = SensorRegistry()
        registry.register(DummySensor("a"))
        registry.register(DummySensor("b"))
        registry.connect_all()
        signals = registry.read_all()
        assert len(signals) == 2
        assert all(isinstance(s, FatigueSignal) for s in signals)

    def test_read_all_skips_none_results(self):
        """read_all() omits sensors whose read() returns None."""
        registry = SensorRegistry()
        registry.register(DummySensor("good"))
        registry.register(DummySensor("silent", fail_on_read=True))
        registry.connect_all()
        signals = registry.read_all()
        assert len(signals) == 1
        assert signals[0].source == "good"

    def test_thread_safe_register(self):
        """10 threads concurrently registering unique sensors all succeed."""
        registry = SensorRegistry()
        errors = []

        def register_sensor(idx: int) -> None:
            try:
                registry.register(DummySensor(f"sensor-{idx}"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=register_sensor, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent registration raised: {errors}"
        assert len(registry.sensors) == 10

    def test_empty_registry_read_all_returns_empty_list(self):
        """read_all() on an empty registry returns []."""
        registry = SensorRegistry()
        assert registry.read_all() == []

    def test_empty_registry_connect_all_does_not_raise(self):
        """connect_all() on an empty registry does nothing and does not raise."""
        registry = SensorRegistry()
        registry.connect_all()  # should not raise

    def test_empty_registry_disconnect_all_does_not_raise(self):
        """disconnect_all() on an empty registry does nothing and does not raise."""
        registry = SensorRegistry()
        registry.disconnect_all()  # should not raise
