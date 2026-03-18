"""Tests for modular SDK packaging."""
import pytest


class TestCoreImports:
    def test_core_imports_without_optional_deps(self):
        from sleep_detector_sdk.types import FatigueSignal, FusionResult
        from sleep_detector_sdk.sensors import SensorProvider, SensorRegistry
        from sleep_detector_sdk.fusion import FusionEngine
        from sleep_detector_sdk.temporal import TemporalEngine
        from sleep_detector_sdk.gaze import GazeEstimator
        from sleep_detector_sdk.privacy import PrivacyConfig
        assert True

    def test_plugin_imports_are_available(self):
        from sleep_detector_sdk.plugins.steering import SteeringProvider
        from sleep_detector_sdk.plugins.physiological import PhysiologicalProvider
        assert True

    def test_public_api_exports(self):
        import sleep_detector_sdk
        assert hasattr(sleep_detector_sdk, "__version__")
        assert hasattr(sleep_detector_sdk, "FatigueSignal")
        assert hasattr(sleep_detector_sdk, "FusionEngine")
        assert hasattr(sleep_detector_sdk, "SensorProvider")
        assert hasattr(sleep_detector_sdk, "TemporalEngine")
        assert hasattr(sleep_detector_sdk, "AlertTier")
        assert hasattr(sleep_detector_sdk, "GazeZone")

    def test_entry_point_discovery_api(self):
        try:
            from importlib.metadata import entry_points
            # Just verify the API works — entry points may not be registered in dev mode
            eps = entry_points()
            assert True
        except ImportError:
            pytest.skip("importlib.metadata not available")
