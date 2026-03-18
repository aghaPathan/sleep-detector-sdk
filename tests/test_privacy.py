"""Tests for privacy framework with PII sanitization."""
import pytest
from sleep_detector_sdk.privacy import PrivacyConfig, PERMISSION_STRINGS


class TestPrivacyConfigDefaults:
    def test_default_on_device_only_is_true(self):
        config = PrivacyConfig()
        assert config.on_device_only is True

    def test_default_retention_seconds_is_positive(self):
        config = PrivacyConfig()
        assert config.retention_seconds > 0


class TestPrivacyConfigCustom:
    def test_custom_retention_seconds(self):
        config = PrivacyConfig(retention_seconds=120.0)
        assert config.retention_seconds == 120.0


class TestPermissionStrings:
    def test_microphone_permission_string_matches_prd(self):
        expected = (
            "We need access to the microphone to analyze your sleep patterns "
            "and detect snoring."
        )
        assert PERMISSION_STRINGS["microphone"] == expected


class TestSanitizeLog:
    def test_sanitize_strips_ip_address(self):
        config = PrivacyConfig()
        result = config.sanitize_log("Connection from 192.168.1.1 failed")
        assert "192.168.1.1" not in result
        assert "[REDACTED_IP]" in result

    def test_sanitize_strips_email_address(self):
        config = PrivacyConfig()
        result = config.sanitize_log("User user@example.com logged in")
        assert "user@example.com" not in result
        assert "[REDACTED_EMAIL]" in result
