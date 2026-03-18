"""Alert handler interface and alert management with cooldown logic."""

import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from sleep_detector_sdk.types import AlertTier, DrowsinessEvent, FusionResult


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


# ---------------------------------------------------------------------------
# Task 7 — AlertProvider ABC and TieredAlertManager
# ---------------------------------------------------------------------------


class AlertProvider(ABC):
    """Abstract provider that handles triggering and cancelling a tier alert."""

    @abstractmethod
    def trigger(self, tier: AlertTier, result: FusionResult) -> None:
        """Trigger an alert for the given tier."""

    @abstractmethod
    def cancel(self) -> None:
        """Cancel / silence the active alert."""


class TieredAlertManager:
    """Manages per-tier alert providers with per-tier cooldowns."""

    def __init__(self, cooldowns: Optional[Dict[AlertTier, float]] = None) -> None:
        self._cooldowns: Dict[AlertTier, float] = cooldowns or {}
        self._providers: Dict[AlertTier, List[AlertProvider]] = {}
        self._last_dispatch: Dict[AlertTier, float] = {}
        self._current_tier: Optional[AlertTier] = None

    @property
    def current_tier(self) -> Optional[AlertTier]:
        return self._current_tier

    def register_provider(self, tier: AlertTier, provider: AlertProvider) -> None:
        self._providers.setdefault(tier, []).append(provider)

    def dispatch(self, result: FusionResult) -> None:
        tier = result.tier
        self._current_tier = tier

        cooldown = self._cooldowns.get(tier, 0.0)
        if cooldown > 0.0:
            last = self._last_dispatch.get(tier, 0.0)
            if last != 0.0 and (time.monotonic() - last) < cooldown:
                return  # still in cooldown — skip

        self._last_dispatch[tier] = time.monotonic()
        for provider in self._providers.get(tier, []):
            provider.trigger(tier, result)
