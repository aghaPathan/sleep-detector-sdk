"""Event emitter and callback registry for the Sleep Detector SDK."""

from collections import defaultdict
from typing import Any, Callable, Dict, List


class EventEmitter:
    """Simple synchronous event emitter with callback registration."""

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)

    def on(self, event: str, handler: Callable) -> None:
        """Register a callback for an event."""
        self._handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        """Remove a callback for an event."""
        try:
            self._handlers[event].remove(handler)
        except ValueError:
            pass

    def emit(self, event: str, data: Any) -> None:
        """Emit an event, calling all registered handlers."""
        for handler in self._handlers.get(event, []):
            handler(data)
