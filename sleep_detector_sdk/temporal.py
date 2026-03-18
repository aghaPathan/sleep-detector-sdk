"""TemporalEngine — Euro NCAP temporal variable tracking with configurable Hz loop."""

import threading
import time
from collections import deque
from typing import List, Optional

from sleep_detector_sdk.types import GazeZone, TemporalState


class TemporalEngine:
    """Tracks Euro NCAP temporal variables T₀, T_away, T_gaze, T_road, T_close.

    A dedicated timer thread ticks at ``frequency_hz``, appending the current
    :class:`~sleep_detector_sdk.types.TemporalState` snapshot to a fixed-size
    ring buffer of ``buffer_seconds`` depth.

    Thread-safe: all mutable state is protected by a single ``threading.Lock``.

    Parameters
    ----------
    frequency_hz:
        Sampling frequency for the internal timer loop (default 25 Hz).
    buffer_seconds:
        Duration of history retained in the ring buffer (default 60 s).
    """

    def __init__(self, frequency_hz: int = 25, buffer_seconds: float = 60.0) -> None:
        self._frequency_hz = frequency_hz
        self._buffer_seconds = buffer_seconds
        self._interval = 1.0 / frequency_hz

        maxlen = int(frequency_hz * buffer_seconds)
        self._buffer: deque = deque(maxlen=maxlen)

        self._lock = threading.Lock()

        # Euro NCAP temporal variables
        self._t_away: Optional[float] = None
        self._t_gaze: Optional[float] = None
        self._t_road: Optional[float] = None
        self._t_close: Optional[float] = None
        self._last_zone: Optional[GazeZone] = None

        # Thread management
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public recording API
    # ------------------------------------------------------------------

    def record_gaze(self, zone: GazeZone) -> None:
        """Update gaze-based temporal variables based on zone transition.

        Rules
        -----
        - ROAD → non-ROAD: sets ``t_away`` (first occurrence only) and ``t_gaze``.
        - non-ROAD → non-ROAD (zone change): updates ``t_gaze``.
        - any → ROAD: sets ``t_road``.
        """
        now = time.time()
        with self._lock:
            prev = self._last_zone
            self._last_zone = zone

            if zone == GazeZone.ROAD:
                # Returning to road
                self._t_road = now
            else:
                # Moving away from or staying away from road
                if prev == GazeZone.ROAD:
                    # Transition away: set t_away once, always update t_gaze
                    if self._t_away is None:
                        self._t_away = now
                    self._t_gaze = now
                # If prev was already non-ROAD and zone changes, update t_gaze
                elif prev is not None and prev != zone:
                    self._t_gaze = now
                # If this is the very first call and not ROAD, do nothing
                # (no prior ROAD context — t_away stays None per spec)

    def record_eye_close(self) -> None:
        """Mark the onset of continuous eye closure; sets ``t_close`` once."""
        now = time.time()
        with self._lock:
            if self._t_close is None:
                self._t_close = now

    def record_eye_open(self) -> None:
        """Record eye opening (does not clear ``t_close``)."""
        # t_close records the first closure event; it is not reset on open.

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> TemporalState:
        """Return a thread-safe snapshot of the current temporal state."""
        with self._lock:
            return self._build_state()

    @property
    def is_running(self) -> bool:
        """True while the timer thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def history(self, duration: float) -> List[TemporalState]:
        """Return temporal state snapshots from the last ``duration`` seconds.

        Parameters
        ----------
        duration:
            How many seconds of history to return.

        Returns
        -------
        List[TemporalState]
            Entries in chronological order (oldest first).
        """
        cutoff = time.time() - duration
        with self._lock:
            return [s for s in self._buffer if s.timestamp >= cutoff]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the timer thread (idempotent)."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the timer thread and wait for it to finish (idempotent)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 5)
            self._thread = None

    def reset(self) -> None:
        """Clear all temporal variables and history."""
        with self._lock:
            self._t_away = None
            self._t_gaze = None
            self._t_road = None
            self._t_close = None
            self._last_zone = None
            self._buffer.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_state(self) -> TemporalState:
        """Build a TemporalState snapshot (must be called with lock held)."""
        t_zero = self._compute_t_zero()
        return TemporalState(
            t_zero=t_zero,
            t_away=self._t_away,
            t_gaze=self._t_gaze,
            t_road=self._t_road,
            t_close=self._t_close,
            timestamp=time.time(),
        )

    def _compute_t_zero(self) -> Optional[float]:
        """T₀ = (min(t_away, t_close) - 4.0) if either is set, else None."""
        candidates = [t for t in (self._t_away, self._t_close) if t is not None]
        if not candidates:
            return None
        return min(candidates) - 4.0

    def _tick(self) -> None:
        """Append current state to the ring buffer (one sample)."""
        with self._lock:
            self._buffer.append(self._build_state())

    def _run_loop(self) -> None:
        """Timer loop: fire at ``frequency_hz`` until ``stop_event`` is set."""
        while not self._stop_event.wait(timeout=self._interval):
            self._tick()
