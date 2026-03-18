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
        assert meta.version == "0.1.0"

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
