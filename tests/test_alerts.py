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
