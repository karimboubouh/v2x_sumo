"""Thread-safe event stream for the dashboard interaction log."""

from collections import deque
from dataclasses import dataclass
import threading


@dataclass(slots=True)
class SimulationEvent:
    """One human-readable event shown in the dashboard log."""

    timestamp: float
    category: str
    text: str


class EventStream:
    """Bounded, non-blocking event buffer shared across threads."""

    def __init__(self, max_events: int | None = 4096):
        self._events = deque(maxlen=max_events) if max_events is not None else deque()
        self._lock = threading.Lock()

    def publish(self, timestamp: float, category: str, text: str) -> SimulationEvent:
        """Append a new event without blocking producers."""
        event = SimulationEvent(float(timestamp), category, text)
        with self._lock:
            self._events.append(event)
        return event

    def drain(self, max_items: int = 200) -> list[SimulationEvent]:
        """Pop up to max_items events in FIFO order."""
        with self._lock:
            n = min(max_items, len(self._events))
            return [self._events.popleft() for _ in range(n)]
