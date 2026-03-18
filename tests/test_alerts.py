"""Tests for sleep_detector_sdk.alerts — AlertHandler ABC and cooldown logic."""

import time

import pytest

from sleep_detector_sdk.alerts import AlertHandler, AlertManager
from sleep_detector_sdk.types import DrowsinessEvent


class ConcreteAlertHandler(AlertHandler):
    """Test implementation of AlertHandler."""

    def __init__(self):
        self.alerts = []

    def on_alert(self, event: DrowsinessEvent) -> None:
        self.alerts.append(event)


class TestAlertHandler:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            AlertHandler()

    def test_subclass_receives_alerts(self):
        handler = ConcreteAlertHandler()
        event = DrowsinessEvent(duration=5.0, ear_value=0.15, timestamp=1000.0)
        handler.on_alert(event)
        assert len(handler.alerts) == 1
        assert handler.alerts[0].duration == 5.0


class TestAlertManager:
    def test_respects_cooldown(self):
        manager = AlertManager(cooldown=0.1)
        event1 = DrowsinessEvent(duration=5.0, ear_value=0.15, timestamp=100.0)
        event2 = DrowsinessEvent(duration=6.0, ear_value=0.14, timestamp=100.05)

        assert manager.should_alert(event1) is True
        manager.record_alert()
        assert manager.should_alert(event2) is False

    def test_allows_alert_after_cooldown_expires(self):
        manager = AlertManager(cooldown=0.05)
        event = DrowsinessEvent(duration=5.0, ear_value=0.15, timestamp=100.0)

        assert manager.should_alert(event) is True
        manager.record_alert()
        time.sleep(0.06)
        assert manager.should_alert(event) is True

    def test_first_alert_always_allowed(self):
        manager = AlertManager(cooldown=999.0)
        event = DrowsinessEvent(duration=5.0, ear_value=0.15, timestamp=100.0)
        assert manager.should_alert(event) is True

    def test_dispatches_to_handler_and_callbacks(self):
        handler = ConcreteAlertHandler()
        callback_received = []

        manager = AlertManager(cooldown=0.0)
        manager.add_handler(handler)
        manager.add_callback(lambda e: callback_received.append(e))

        event = DrowsinessEvent(duration=5.0, ear_value=0.15, timestamp=100.0)
        manager.dispatch(event)

        assert len(handler.alerts) == 1
        assert len(callback_received) == 1


# ---------------------------------------------------------------------------
# Task 7 — TieredAlertManager
# ---------------------------------------------------------------------------

from sleep_detector_sdk.alerts import AlertProvider, TieredAlertManager  # noqa: E402
from sleep_detector_sdk.types import AlertTier, FusionResult  # noqa: E402


class DummyAlertProvider(AlertProvider):
    """Test helper that records trigger/cancel calls."""

    def __init__(self):
        self.triggered = []   # list of (tier, result) tuples
        self.cancelled = 0

    def trigger(self, tier: AlertTier, result: FusionResult) -> None:
        self.triggered.append((tier, result))

    def cancel(self) -> None:
        self.cancelled += 1


def _make_result(tier: AlertTier, score: float = 0.5) -> FusionResult:
    return FusionResult(
        fatigue_score=score,
        tier=tier,
        signals=[],
        timestamp=0.0,
    )


class TestAlertProvider:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            AlertProvider()  # type: ignore[abstract]


class TestTieredAlertManager:
    def test_dispatches_to_correct_tier(self):
        manager = TieredAlertManager()
        provider = DummyAlertProvider()
        manager.register_provider(AlertTier.SILENT, provider)

        result = _make_result(AlertTier.SILENT)
        manager.dispatch(result)

        assert len(provider.triggered) == 1
        assert provider.triggered[0][0] == AlertTier.SILENT

    def test_escalation_silent_to_audible(self):
        manager = TieredAlertManager()
        silent_provider = DummyAlertProvider()
        audible_provider = DummyAlertProvider()
        manager.register_provider(AlertTier.SILENT, silent_provider)
        manager.register_provider(AlertTier.AUDIBLE, audible_provider)

        manager.dispatch(_make_result(AlertTier.SILENT))
        assert manager.current_tier == AlertTier.SILENT

        manager.dispatch(_make_result(AlertTier.AUDIBLE))
        assert manager.current_tier == AlertTier.AUDIBLE
        assert len(audible_provider.triggered) == 1

    def test_deescalation_audible_to_silent(self):
        manager = TieredAlertManager()
        silent_provider = DummyAlertProvider()
        audible_provider = DummyAlertProvider()
        manager.register_provider(AlertTier.SILENT, silent_provider)
        manager.register_provider(AlertTier.AUDIBLE, audible_provider)

        manager.dispatch(_make_result(AlertTier.AUDIBLE))
        assert manager.current_tier == AlertTier.AUDIBLE

        manager.dispatch(_make_result(AlertTier.SILENT))
        assert manager.current_tier == AlertTier.SILENT
        assert len(silent_provider.triggered) == 1

    def test_cooldown_per_tier_blocks_second_dispatch(self):
        manager = TieredAlertManager(cooldowns={AlertTier.SILENT: 100.0})
        provider = DummyAlertProvider()
        manager.register_provider(AlertTier.SILENT, provider)

        result = _make_result(AlertTier.SILENT)
        manager.dispatch(result)
        manager.dispatch(result)   # within cooldown — should be blocked

        assert len(provider.triggered) == 1

    def test_no_provider_for_tier_is_noop(self):
        """Dispatching with no registered provider for a tier must not raise."""
        manager = TieredAlertManager()
        result = _make_result(AlertTier.CRITICAL)
        manager.dispatch(result)  # no provider registered — noop
        assert manager.current_tier == AlertTier.CRITICAL
