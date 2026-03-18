"""Privacy framework with PII sanitization for the sleep detector SDK."""
import re
from dataclasses import dataclass

PERMISSION_STRINGS = {
    "microphone": "We need access to the microphone to analyze your sleep patterns and detect snoring.",
    "camera": "We need access to the camera to monitor driver alertness and detect drowsiness.",
}

_IP_PATTERN = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
_EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')


@dataclass
class PrivacyConfig:
    on_device_only: bool = True
    retention_seconds: float = 3600.0
    log_pii: bool = False

    def sanitize_log(self, message: str) -> str:
        result = _IP_PATTERN.sub("[REDACTED_IP]", message)
        result = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", result)
        return result
