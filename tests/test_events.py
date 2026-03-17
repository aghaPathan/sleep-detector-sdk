"""Tests for sleep_detector_sdk.events — event emitter and callback registry."""

from sleep_detector_sdk.events import EventEmitter


class TestEventEmitter:
    def test_register_and_emit_callback(self):
        emitter = EventEmitter()
        received = []
        emitter.on("test_event", lambda data: received.append(data))
        emitter.emit("test_event", {"value": 42})
        assert received == [{"value": 42}]

    def test_multiple_handlers_for_same_event(self):
        emitter = EventEmitter()
        results_a = []
        results_b = []
        emitter.on("evt", lambda d: results_a.append(d))
        emitter.on("evt", lambda d: results_b.append(d))
        emitter.emit("evt", "hello")
        assert results_a == ["hello"]
        assert results_b == ["hello"]

    def test_emit_unknown_event_does_nothing(self):
        emitter = EventEmitter()
        # Should not raise
        emitter.emit("nonexistent", "data")

    def test_remove_handler(self):
        emitter = EventEmitter()
        received = []
        handler = lambda d: received.append(d)
        emitter.on("evt", handler)
        emitter.off("evt", handler)
        emitter.emit("evt", "data")
        assert received == []

    def test_remove_nonexistent_handler_does_not_raise(self):
        emitter = EventEmitter()
        emitter.off("evt", lambda d: None)

    def test_handlers_called_in_registration_order(self):
        emitter = EventEmitter()
        order = []
        emitter.on("evt", lambda d: order.append("first"))
        emitter.on("evt", lambda d: order.append("second"))
        emitter.on("evt", lambda d: order.append("third"))
        emitter.emit("evt", None)
        assert order == ["first", "second", "third"]

    def test_different_events_are_independent(self):
        emitter = EventEmitter()
        a_data = []
        b_data = []
        emitter.on("a", lambda d: a_data.append(d))
        emitter.on("b", lambda d: b_data.append(d))
        emitter.emit("a", 1)
        emitter.emit("b", 2)
        assert a_data == [1]
        assert b_data == [2]
