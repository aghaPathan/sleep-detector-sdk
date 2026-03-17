"""Alert handler interface and alert management with cooldown logic."""

import time
from abc import ABC, abstractmethod
from typing import Callable, List

from sleep_detector_sdk.types import DrowsinessEvent


class AlertHandler(ABC):
    """Abstract base class for structured alert implementations."""

    @abstractmethod
    def on_alert(self, event: DrowsinessEvent) -> None:
        """Handle a drowsiness alert event."""


class AlertManager:
    """Manages alert dispatch with cooldown and multiple handler support."""

    def __init__(self, cooldown: float):
        self._cooldown = cooldown
        self._last_alert_time: float = 0.0
        self._handlers: List[AlertHandler] = []
        self._callbacks: List[Callable] = []

    def add_handler(self, handler: AlertHandler) -> None:
        self._handlers.append(handler)

    def add_callback(self, callback: Callable) -> None:
        self._callbacks.append(callback)

    def should_alert(self, event: DrowsinessEvent) -> bool:
        if self._last_alert_time == 0.0:
            return True
        return (time.monotonic() - self._last_alert_time) >= self._cooldown

    def record_alert(self) -> None:
        self._last_alert_time = time.monotonic()

    def dispatch(self, event: DrowsinessEvent) -> None:
        for handler in self._handlers:
            handler.on_alert(event)
        for callback in self._callbacks:
            callback(event)
        self.record_alert()
