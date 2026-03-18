"""Tests for steering and physiological sensor plugin stubs."""

import pytest

from sleep_detector_sdk.sensors import SensorProvider
from sleep_detector_sdk.plugins.steering import SteeringProvider
from sleep_detector_sdk.plugins.physiological import PhysiologicalProvider


class TestSteeringProvider:
    def test_is_sensor_provider_instance(self):
        provider = SteeringProvider()
        assert isinstance(provider, SensorProvider)

    def test_metadata_has_name_steering(self):
        provider = SteeringProvider()
        meta = provider.metadata()
        assert meta.name == "steering"

    def test_metadata_version(self):
        provider = SteeringProvider()
        meta = provider.metadata()
        assert meta.version == "0.2.0"

    def test_metadata_sampling_hz(self):
        provider = SteeringProvider()
        meta = provider.metadata()
        assert meta.sampling_hz == 10.0

    def test_lifecycle_connect_sets_connected(self):
        provider = SteeringProvider()
        provider.connect()
        assert provider._connected is True

    def test_lifecycle_read_returns_none(self):
        provider = SteeringProvider()
        provider.connect()
        result = provider.read()
        assert result is None

    def test_lifecycle_disconnect_sets_connected_false(self):
        provider = SteeringProvider()
        provider.connect()
        provider.disconnect()
        assert provider._connected is False

    def test_baseline_minutes_default(self):
        provider = SteeringProvider()
        assert provider.baseline_minutes == 20.0

    def test_baseline_minutes_configurable(self):
        provider = SteeringProvider(baseline_minutes=30.0)
        assert provider.baseline_minutes == 30.0

    def test_read_without_connect_returns_none(self):
        provider = SteeringProvider()
        result = provider.read()
        assert result is None


class TestPhysiologicalProvider:
    def test_is_sensor_provider_instance(self):
        provider = PhysiologicalProvider()
        assert isinstance(provider, SensorProvider)

    def test_metadata_has_name_physiological(self):
        provider = PhysiologicalProvider()
        meta = provider.metadata()
        assert meta.name == "physiological"

    def test_metadata_sampling_hz(self):
        provider = PhysiologicalProvider()
        meta = provider.metadata()
        assert meta.sampling_hz == 25.0

    def test_lifecycle_connect_sets_connected(self):
        provider = PhysiologicalProvider()
        provider.connect()
        assert provider._connected is True

    def test_lifecycle_read_returns_none(self):
        provider = PhysiologicalProvider()
        provider.connect()
        result = provider.read()
        assert result is None

    def test_lifecycle_disconnect_sets_connected_false(self):
        provider = PhysiologicalProvider()
        provider.connect()
        provider.disconnect()
        assert provider._connected is False


# ---------------------------------------------------------------------------
# New tests for issue #16 — Real Hardware Integration
# ---------------------------------------------------------------------------


class TestSteeringProviderHardwareIntegration:
    def test_stub_mode_returns_none_when_no_hardware(self):
        """Without hardware, read() must return None even after connect."""
        provider = SteeringProvider()
        provider.connect()
        result = provider.read()
        assert result is None

    def test_baseline_calibrated_property_starts_false(self):
        """baseline_calibrated should be False on fresh instance."""
        provider = SteeringProvider()
        assert provider._baseline_calibrated is False

    def test_connect_sets_hardware_available_false_without_hardware(self):
        """In test environment with no CAN/OBD hardware, connect() should set
        _hardware_available to False and not raise."""
        provider = SteeringProvider()
        provider.connect()
        assert provider._hardware_available is False

    def test_connect_accepts_can_channel_and_interface_kwargs(self):
        """Constructor should accept can_channel and can_interface parameters."""
        provider = SteeringProvider(can_channel="vcan0", can_interface="virtual")
        assert provider._can_channel == "vcan0"
        assert provider._can_interface == "virtual"

    def test_disconnect_clears_hardware_state(self):
        """disconnect() must clear _bus, _obd_connection, and _hardware_available."""
        provider = SteeringProvider()
        provider.connect()
        provider.disconnect()
        assert provider._bus is None
        assert provider._obd_connection is None
        assert provider._hardware_available is False

    def test_metadata_version_updated_to_0_2_0(self):
        """Metadata version should reflect the hardware-capable implementation."""
        provider = SteeringProvider()
        meta = provider.metadata()
        assert meta.version == "0.2.0"

    def test_read_before_connect_returns_none(self):
        """read() must guard against calling without connect()."""
        provider = SteeringProvider()
        assert provider.read() is None

    def test_baseline_angles_deque_initialized(self):
        """Internal baseline deque must exist after construction."""
        provider = SteeringProvider()
        assert hasattr(provider, "_baseline_angles")
        assert len(provider._baseline_angles) == 0


class TestPhysiologicalProviderHardwareIntegration:
    def test_stub_mode_returns_none_when_bleak_not_available(self):
        """Without bleak installed, read() should return None."""
        provider = PhysiologicalProvider()
        provider.connect()
        result = provider.read()
        assert result is None

    def test_add_rr_interval_method_exists(self):
        """PhysiologicalProvider must expose add_rr_interval(rr_ms) method."""
        provider = PhysiologicalProvider()
        assert callable(getattr(provider, "add_rr_interval", None))

    def test_add_rr_interval_stores_data(self):
        """add_rr_interval() must persist data in the internal deque."""
        provider = PhysiologicalProvider()
        provider.add_rr_interval(800.0)
        assert len(provider._rr_intervals) == 1

    def test_compute_hrv_fatigue_returns_none_with_fewer_than_5_intervals(self):
        """_compute_hrv_fatigue must return None when insufficient data."""
        provider = PhysiologicalProvider()
        for rr in [800.0, 810.0, 790.0]:
            provider.add_rr_interval(rr)
        result = provider._compute_hrv_fatigue()
        assert result is None

    def test_compute_hrv_fatigue_returns_float_in_range(self):
        """_compute_hrv_fatigue must return a float in [0.0, 1.0] with enough data."""
        provider = PhysiologicalProvider()
        for rr in [800, 820, 790, 810, 830, 795, 815, 805, 825, 800]:
            provider.add_rr_interval(float(rr))
        result = provider._compute_hrv_fatigue()
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_hrv_computation_with_synthetic_data(self):
        """read() returns FatigueSignal with valid score when hardware_available=True
        and sufficient RR intervals have been added."""
        provider = PhysiologicalProvider()
        provider.connect()
        provider._hardware_available = True  # simulate hardware presence
        for rr in [800, 820, 790, 810, 830, 795, 815, 805, 825, 800]:
            provider.add_rr_interval(float(rr))
        signal = provider.read()
        assert signal is not None
        assert 0.0 <= signal.score <= 1.0
        assert signal.source == "physiological"

    def test_hrv_signal_confidence_is_reasonable(self):
        """FatigueSignal from physiological sensor should have confidence 0.8."""
        provider = PhysiologicalProvider()
        provider.connect()
        provider._hardware_available = True
        for rr in [800, 820, 790, 810, 830, 795, 815, 805, 825, 800]:
            provider.add_rr_interval(float(rr))
        signal = provider.read()
        assert signal is not None
        assert signal.confidence == 0.8

    def test_disconnect_clears_hardware_state(self):
        """disconnect() must clear _client and _hardware_available."""
        provider = PhysiologicalProvider()
        provider.connect()
        provider._hardware_available = True
        provider.disconnect()
        assert provider._client is None
        assert provider._hardware_available is False

    def test_metadata_version_updated_to_0_2_0(self):
        """Metadata version should reflect the hardware-capable implementation."""
        provider = PhysiologicalProvider()
        meta = provider.metadata()
        assert meta.version == "0.2.0"

    def test_device_address_parameter_accepted(self):
        """Constructor should accept optional device_address."""
        provider = PhysiologicalProvider(device_address="AA:BB:CC:DD:EE:FF")
        assert provider._device_address == "AA:BB:CC:DD:EE:FF"
