"""Tests for versioning."""
import re
import pytest

class TestVersion:
    def test_version_is_semver(self):
        from sleep_detector_sdk import __version__
        assert re.match(r'^\d+\.\d+\.\d+', __version__)

    def test_version_accessible(self):
        import sleep_detector_sdk
        assert hasattr(sleep_detector_sdk, "__version__")

    def test_version_is_0_2_0(self):
        from sleep_detector_sdk import __version__
        assert __version__ == "0.2.0"
