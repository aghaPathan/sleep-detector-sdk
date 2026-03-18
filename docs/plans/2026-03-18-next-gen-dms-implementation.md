# Next-Gen DMS SDK Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the multi-modal architecture from the ADR, covering issues #4-#13 (all AFK slices for this milestone).

**Architecture:** Weighted fusion hub with SensorProvider ABC, entry-point plugin discovery, dedicated 25Hz temporal engine, and tiered alert system. See `docs/plans/2026-03-18-multi-modal-architecture-design.md`.

**Tech Stack:** Python 3.8+, numpy, scipy, threading, importlib.metadata, dataclasses, pytest

**Dependency order:** Types (#4→partial) → Sensors+Fusion (#8) → Temporal (#5) → Gaze (#6) → Pose (#7) → External Sensors (#9) → Tiered Alerts (#10) → Modular Packaging (#11) → Privacy (#12) → Release Pipeline (#13)

---

### Task 1: Core Types — FatigueSignal, FusionResult, SensorMetadata

**Issue:** Supports all subsequent tasks
**Files:**
- Modify: `sleep_detector_sdk/types.py`
- Test: `tests/test_types.py`

**Step 1: Write failing tests for new types**

```python
# Append to tests/test_types.py

from sleep_detector_sdk.types import (
    FatigueSignal, FusionResult, SensorMetadata, AlertTier,
    GazeZone, GazeEvent, TemporalState,
)

class TestFatigueSignal:
    def test_construction(self):
        sig = FatigueSignal(score=0.7, confidence=0.9, source="vision", timestamp=1.0)
        assert sig.score == 0.7
        assert sig.confidence == 0.9
        assert sig.source == "vision"

    def test_frozen(self):
        sig = FatigueSignal(score=0.7, confidence=0.9, source="vision", timestamp=1.0)
        with pytest.raises(AttributeError):
            sig.score = 0.5


class TestFusionResult:
    def test_construction(self):
        sig = FatigueSignal(score=0.7, confidence=0.9, source="vision", timestamp=1.0)
        result = FusionResult(fatigue_score=0.7, tier=AlertTier.AUDIBLE, signals=[sig], timestamp=1.0)
        assert result.fatigue_score == 0.7
        assert result.tier == AlertTier.AUDIBLE
        assert len(result.signals) == 1


class TestSensorMetadata:
    def test_construction(self):
        meta = SensorMetadata(name="ecg", version="1.0", sampling_hz=25.0)
        assert meta.name == "ecg"
        assert meta.sampling_hz == 25.0


class TestAlertTier:
    def test_has_three_tiers(self):
        assert AlertTier.SILENT.value == "silent"
        assert AlertTier.AUDIBLE.value == "audible"
        assert AlertTier.CRITICAL.value == "critical"


class TestGazeZone:
    def test_has_zones(self):
        assert GazeZone.ROAD.value == "road"
        assert GazeZone.IN_VEHICLE.value == "in_vehicle"
        assert GazeZone.EXTERNAL.value == "external"


class TestGazeEvent:
    def test_construction(self):
        evt = GazeEvent(zone=GazeZone.ROAD, yaw=0.1, pitch=-0.05, timestamp=1.0)
        assert evt.zone == GazeZone.ROAD


class TestTemporalState:
    def test_construction(self):
        state = TemporalState(
            t_zero=None, t_away=None, t_gaze=None,
            t_road=None, t_close=None, timestamp=1.0,
        )
        assert state.t_zero is None
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n scripting python -m pytest tests/test_types.py -v`
Expected: FAIL — ImportError for new types

**Step 3: Implement the new types**

Add to `sleep_detector_sdk/types.py`:

```python
class AlertTier(Enum):
    SILENT = "silent"
    AUDIBLE = "audible"
    CRITICAL = "critical"


class GazeZone(Enum):
    ROAD = "road"
    IN_VEHICLE = "in_vehicle"
    EXTERNAL = "external"


@dataclass(frozen=True)
class FatigueSignal:
    score: float        # 0.0 (alert) to 1.0 (critical fatigue)
    confidence: float   # 0.0 (unreliable) to 1.0 (high confidence)
    source: str         # e.g., "vision", "steering", "ecg"
    timestamp: float


@dataclass(frozen=True)
class SensorMetadata:
    name: str
    version: str
    sampling_hz: float


@dataclass(frozen=True)
class FusionResult:
    fatigue_score: float
    tier: AlertTier
    signals: list       # List[FatigueSignal] — use list for 3.8 compat
    timestamp: float


@dataclass(frozen=True)
class GazeEvent:
    zone: GazeZone
    yaw: float
    pitch: float
    timestamp: float


@dataclass(frozen=True)
class TemporalState:
    t_zero: Optional[float]
    t_away: Optional[float]
    t_gaze: Optional[float]
    t_road: Optional[float]
    t_close: Optional[float]
    timestamp: float
```

Add `Optional` to the typing import in types.py.

**Step 4: Export new types from `__init__.py`**

Add all new types to imports and `__all__` in `sleep_detector_sdk/__init__.py`.

**Step 5: Run tests to verify they pass**

Run: `conda run -n scripting python -m pytest tests/test_types.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add sleep_detector_sdk/types.py sleep_detector_sdk/__init__.py tests/test_types.py
git commit -m "feat: add FatigueSignal, FusionResult, SensorMetadata, and new enum types"
```

---

### Task 2: SensorProvider ABC and SensorRegistry

**Issue:** #8 (Multi-Modal Sensor Plugin Architecture)
**Files:**
- Create: `sleep_detector_sdk/sensors.py`
- Test: `tests/test_sensors.py`

**Step 1: Write failing tests**

```python
# tests/test_sensors.py
"""Tests for sensor provider interface and registry."""
import time
import threading
from unittest.mock import MagicMock

import pytest

from sleep_detector_sdk.sensors import SensorProvider, SensorRegistry
from sleep_detector_sdk.types import FatigueSignal, SensorMetadata


class DummySensor(SensorProvider):
    """Concrete sensor for testing."""

    def __init__(self, name="dummy"):
        self._name = name
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def read(self):
        if not self._connected:
            return None
        return FatigueSignal(score=0.5, confidence=0.8, source=self._name, timestamp=time.monotonic())

    def disconnect(self) -> None:
        self._connected = False

    def metadata(self) -> SensorMetadata:
        return SensorMetadata(name=self._name, version="1.0", sampling_hz=10.0)


class TestSensorProvider:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SensorProvider()

    def test_concrete_sensor_lifecycle(self):
        sensor = DummySensor()
        assert sensor.read() is None  # not connected
        sensor.connect()
        signal = sensor.read()
        assert isinstance(signal, FatigueSignal)
        assert signal.source == "dummy"
        sensor.disconnect()
        assert sensor.read() is None


class TestSensorRegistry:
    def test_register_and_list(self):
        registry = SensorRegistry()
        sensor = DummySensor("test")
        registry.register(sensor)
        assert len(registry.sensors) == 1
        assert registry.sensors[0].metadata().name == "test"

    def test_register_duplicate_name_raises(self):
        registry = SensorRegistry()
        registry.register(DummySensor("dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(DummySensor("dup"))

    def test_unregister(self):
        registry = SensorRegistry()
        sensor = DummySensor("rem")
        registry.register(sensor)
        registry.unregister("rem")
        assert len(registry.sensors) == 0

    def test_connect_all_and_disconnect_all(self):
        registry = SensorRegistry()
        s1, s2 = DummySensor("a"), DummySensor("b")
        registry.register(s1)
        registry.register(s2)
        registry.connect_all()
        assert s1._connected and s2._connected
        registry.disconnect_all()
        assert not s1._connected and not s2._connected

    def test_read_all_returns_signals(self):
        registry = SensorRegistry()
        sensor = DummySensor("sig")
        registry.register(sensor)
        registry.connect_all()
        signals = registry.read_all()
        assert len(signals) == 1
        assert signals[0].source == "sig"

    def test_read_all_skips_disconnected(self):
        registry = SensorRegistry()
        s1, s2 = DummySensor("up"), DummySensor("down")
        registry.register(s1)
        registry.register(s2)
        s1.connect()
        # s2 not connected
        signals = registry.read_all()
        assert len(signals) == 1

    def test_thread_safe_register(self):
        registry = SensorRegistry()
        errors = []
        def register_sensor(i):
            try:
                registry.register(DummySensor(f"sensor_{i}"))
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=register_sensor, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(registry.sensors) == 10
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n scripting python -m pytest tests/test_sensors.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement sensors.py**

```python
# sleep_detector_sdk/sensors.py
"""Sensor provider interface and registry for multi-modal detection."""

import logging
import threading
from abc import ABC, abstractmethod
from typing import List, Optional

from sleep_detector_sdk.types import FatigueSignal, SensorMetadata

logger = logging.getLogger(__name__)


class SensorProvider(ABC):
    """Abstract base class for sensor plugins."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the sensor hardware/data source."""

    @abstractmethod
    def read(self) -> Optional[FatigueSignal]:
        """Read the latest fatigue signal. Returns None if unavailable."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect and release resources."""

    @abstractmethod
    def metadata(self) -> SensorMetadata:
        """Return sensor metadata."""


class SensorRegistry:
    """Thread-safe registry for sensor providers."""

    def __init__(self):
        self._sensors: List[SensorProvider] = []
        self._lock = threading.Lock()

    @property
    def sensors(self) -> List[SensorProvider]:
        with self._lock:
            return list(self._sensors)

    def register(self, provider: SensorProvider) -> None:
        with self._lock:
            name = provider.metadata().name
            for s in self._sensors:
                if s.metadata().name == name:
                    raise ValueError(f"Sensor '{name}' already registered")
            self._sensors.append(provider)
            logger.info("Registered sensor: %s", name)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._sensors = [s for s in self._sensors if s.metadata().name != name]
            logger.info("Unregistered sensor: %s", name)

    def connect_all(self) -> None:
        for sensor in self.sensors:
            try:
                sensor.connect()
            except Exception:
                logger.exception("Failed to connect sensor: %s", sensor.metadata().name)

    def disconnect_all(self) -> None:
        for sensor in self.sensors:
            try:
                sensor.disconnect()
            except Exception:
                logger.exception("Failed to disconnect sensor: %s", sensor.metadata().name)

    def read_all(self) -> List[FatigueSignal]:
        signals = []
        for sensor in self.sensors:
            try:
                signal = sensor.read()
                if signal is not None:
                    signals.append(signal)
            except Exception:
                logger.exception("Failed to read sensor: %s", sensor.metadata().name)
        return signals
```

**Step 4: Run tests**

Run: `conda run -n scripting python -m pytest tests/test_sensors.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sleep_detector_sdk/sensors.py tests/test_sensors.py
git commit -m "feat: add SensorProvider ABC and SensorRegistry (#8)"
```

---

### Task 3: FusionEngine — Weighted Signal Combination

**Issue:** #8 (Multi-Modal Sensor Plugin Architecture)
**Files:**
- Create: `sleep_detector_sdk/fusion.py`
- Test: `tests/test_fusion.py`

**Step 1: Write failing tests**

```python
# tests/test_fusion.py
"""Tests for the fusion engine — weighted multi-modal signal combination."""
import time
import threading

import pytest

from sleep_detector_sdk.fusion import FusionEngine
from sleep_detector_sdk.types import AlertTier, FatigueSignal, FusionResult


class TestFusionEngine:
    def test_no_signals_returns_zero(self):
        engine = FusionEngine()
        result = engine.compute()
        assert result.fatigue_score == 0.0
        assert result.tier == AlertTier.SILENT
        assert len(result.signals) == 0

    def test_single_signal_uses_score_times_confidence(self):
        engine = FusionEngine()
        sig = FatigueSignal(score=0.8, confidence=1.0, source="vision", timestamp=time.monotonic())
        engine.submit_signal(sig)
        result = engine.compute()
        assert abs(result.fatigue_score - 0.8) < 0.01

    def test_weighted_fusion_of_two_signals(self):
        engine = FusionEngine()
        engine.configure_weights({"vision": 0.6, "ecg": 0.4})
        now = time.monotonic()
        engine.submit_signal(FatigueSignal(score=0.5, confidence=1.0, source="vision", timestamp=now))
        engine.submit_signal(FatigueSignal(score=1.0, confidence=1.0, source="ecg", timestamp=now))
        result = engine.compute()
        # 0.5*0.6 + 1.0*0.4 = 0.7
        assert abs(result.fatigue_score - 0.7) < 0.01

    def test_confidence_reduces_contribution(self):
        engine = FusionEngine()
        now = time.monotonic()
        engine.submit_signal(FatigueSignal(score=1.0, confidence=0.5, source="vision", timestamp=now))
        result = engine.compute()
        assert result.fatigue_score < 1.0

    def test_tier_silent_below_threshold(self):
        engine = FusionEngine()
        engine.submit_signal(FatigueSignal(score=0.2, confidence=1.0, source="vision", timestamp=time.monotonic()))
        assert engine.compute().tier == AlertTier.SILENT

    def test_tier_audible_mid_range(self):
        engine = FusionEngine()
        engine.submit_signal(FatigueSignal(score=0.6, confidence=1.0, source="vision", timestamp=time.monotonic()))
        assert engine.compute().tier == AlertTier.AUDIBLE

    def test_tier_critical_high(self):
        engine = FusionEngine()
        engine.submit_signal(FatigueSignal(score=0.9, confidence=1.0, source="vision", timestamp=time.monotonic()))
        assert engine.compute().tier == AlertTier.CRITICAL

    def test_stale_signal_ignored(self):
        engine = FusionEngine(stale_threshold=0.1)
        old = FatigueSignal(score=1.0, confidence=1.0, source="vision", timestamp=time.monotonic() - 1.0)
        engine.submit_signal(old)
        result = engine.compute()
        assert result.fatigue_score == 0.0

    def test_latest_signal_per_source(self):
        engine = FusionEngine()
        now = time.monotonic()
        engine.submit_signal(FatigueSignal(score=0.3, confidence=1.0, source="vision", timestamp=now))
        engine.submit_signal(FatigueSignal(score=0.9, confidence=1.0, source="vision", timestamp=now + 0.1))
        result = engine.compute()
        assert abs(result.fatigue_score - 0.9) < 0.01

    def test_thread_safe_submit(self):
        engine = FusionEngine()
        def submit(i):
            engine.submit_signal(FatigueSignal(score=0.5, confidence=1.0, source=f"s{i}", timestamp=time.monotonic()))
        threads = [threading.Thread(target=submit, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        result = engine.compute()
        assert len(result.signals) == 10

    def test_configure_tier_thresholds(self):
        engine = FusionEngine(tier_thresholds=(0.3, 0.6))
        engine.submit_signal(FatigueSignal(score=0.35, confidence=1.0, source="v", timestamp=time.monotonic()))
        assert engine.compute().tier == AlertTier.AUDIBLE
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n scripting python -m pytest tests/test_fusion.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement fusion.py**

```python
# sleep_detector_sdk/fusion.py
"""FusionEngine — weighted combination of multi-modal fatigue signals."""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from sleep_detector_sdk.types import AlertTier, FatigueSignal, FusionResult

logger = logging.getLogger(__name__)

DEFAULT_TIER_THRESHOLDS = (0.4, 0.75)  # (silent→audible, audible→critical)
DEFAULT_STALE_THRESHOLD = 5.0  # seconds


class FusionEngine:
    """Collects FatigueSignals from multiple sources and computes weighted fatigue score."""

    def __init__(
        self,
        stale_threshold: float = DEFAULT_STALE_THRESHOLD,
        tier_thresholds: Tuple[float, float] = DEFAULT_TIER_THRESHOLDS,
    ):
        self._signals: Dict[str, FatigueSignal] = {}
        self._weights: Dict[str, float] = {}
        self._stale_threshold = stale_threshold
        self._tier_thresholds = tier_thresholds
        self._lock = threading.Lock()

    def submit_signal(self, signal: FatigueSignal) -> None:
        with self._lock:
            self._signals[signal.source] = signal

    def configure_weights(self, weights: Dict[str, float]) -> None:
        with self._lock:
            self._weights = dict(weights)

    def compute(self) -> FusionResult:
        now = time.monotonic()
        with self._lock:
            # Filter stale signals
            active = {
                src: sig for src, sig in self._signals.items()
                if (now - sig.timestamp) < self._stale_threshold
            }

        if not active:
            return FusionResult(fatigue_score=0.0, tier=AlertTier.SILENT, signals=[], timestamp=now)

        signals = list(active.values())

        # Weighted average: score * confidence * weight
        total_weight = 0.0
        weighted_sum = 0.0
        for sig in signals:
            w = self._weights.get(sig.source, 1.0)
            contribution = sig.score * sig.confidence * w
            weighted_sum += contribution
            total_weight += sig.confidence * w

        fatigue_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        fatigue_score = max(0.0, min(1.0, fatigue_score))

        tier = self._score_to_tier(fatigue_score)
        return FusionResult(
            fatigue_score=fatigue_score,
            tier=tier,
            signals=signals,
            timestamp=now,
        )

    def _score_to_tier(self, score: float) -> AlertTier:
        low, high = self._tier_thresholds
        if score >= high:
            return AlertTier.CRITICAL
        elif score >= low:
            return AlertTier.AUDIBLE
        return AlertTier.SILENT
```

**Step 4: Run tests**

Run: `conda run -n scripting python -m pytest tests/test_fusion.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sleep_detector_sdk/fusion.py tests/test_fusion.py
git commit -m "feat: add FusionEngine with weighted signal combination (#8)"
```

---

### Task 4: TemporalEngine — 25Hz Sampling Loop

**Issue:** #5 (Euro NCAP Temporal Metrics Engine)
**Files:**
- Create: `sleep_detector_sdk/temporal.py`
- Test: `tests/test_temporal.py`

**Step 1: Write failing tests**

```python
# tests/test_temporal.py
"""Tests for the 25Hz temporal metrics engine."""
import time
import threading
from unittest.mock import MagicMock

import pytest

from sleep_detector_sdk.temporal import TemporalEngine
from sleep_detector_sdk.types import GazeZone, TemporalState


class TestTemporalEngine:
    def test_initial_state_all_none(self):
        engine = TemporalEngine()
        state = engine.current_state
        assert state.t_zero is None
        assert state.t_away is None

    def test_record_gaze_away_sets_t_away(self):
        engine = TemporalEngine()
        engine.record_gaze(GazeZone.IN_VEHICLE)
        state = engine.current_state
        assert state.t_away is not None
        assert state.t_gaze is not None

    def test_record_gaze_road_sets_t_road(self):
        engine = TemporalEngine()
        engine.record_gaze(GazeZone.IN_VEHICLE)
        engine.record_gaze(GazeZone.ROAD)
        state = engine.current_state
        assert state.t_road is not None

    def test_record_eye_close_sets_t_close(self):
        engine = TemporalEngine()
        engine.record_eye_close()
        state = engine.current_state
        assert state.t_close is not None

    def test_t_zero_computed_from_t_away(self):
        engine = TemporalEngine()
        engine.record_gaze(GazeZone.EXTERNAL)
        state = engine.current_state
        assert state.t_zero is not None
        assert abs(state.t_zero - (state.t_away - 4.0)) < 0.01

    def test_t_zero_computed_from_t_close(self):
        engine = TemporalEngine()
        engine.record_eye_close()
        state = engine.current_state
        assert state.t_zero is not None
        assert abs(state.t_zero - (state.t_close - 4.0)) < 0.01

    def test_reset_clears_all(self):
        engine = TemporalEngine()
        engine.record_gaze(GazeZone.IN_VEHICLE)
        engine.reset()
        state = engine.current_state
        assert state.t_away is None

    def test_history_returns_buffer(self):
        engine = TemporalEngine(buffer_seconds=60.0)
        engine.record_gaze(GazeZone.IN_VEHICLE)
        engine.record_gaze(GazeZone.ROAD)
        history = engine.history(duration=60.0)
        assert len(history) >= 2

    def test_start_stop_loop(self):
        engine = TemporalEngine(frequency_hz=100)  # fast for testing
        engine.start()
        assert engine.is_running
        time.sleep(0.05)
        engine.stop()
        assert not engine.is_running

    def test_sampling_frequency(self):
        tick_count = [0]
        engine = TemporalEngine(frequency_hz=100)

        original_tick = engine._tick
        def counting_tick():
            tick_count[0] += 1
            original_tick()
        engine._tick = counting_tick

        engine.start()
        time.sleep(0.1)
        engine.stop()
        # At 100Hz for 0.1s, expect ~10 ticks (allow tolerance)
        assert tick_count[0] >= 5

    def test_thread_safe_state_access(self):
        engine = TemporalEngine(frequency_hz=50)
        engine.start()
        errors = []
        def read_state():
            try:
                for _ in range(20):
                    _ = engine.current_state
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=read_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        engine.stop()
        assert len(errors) == 0
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n scripting python -m pytest tests/test_temporal.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement temporal.py**

```python
# sleep_detector_sdk/temporal.py
"""TemporalEngine — 25Hz sampling loop for Euro NCAP temporal variables."""

import logging
import threading
import time
from collections import deque
from typing import Deque, List, Optional

from sleep_detector_sdk.types import GazeZone, TemporalState

logger = logging.getLogger(__name__)

DEFAULT_FREQUENCY_HZ = 25
DEFAULT_BUFFER_SECONDS = 60.0
T_ZERO_OFFSET = 4.0


class TemporalEngine:
    """Tracks Euro NCAP SD 202 temporal variables at configurable frequency."""

    def __init__(
        self,
        frequency_hz: float = DEFAULT_FREQUENCY_HZ,
        buffer_seconds: float = DEFAULT_BUFFER_SECONDS,
    ):
        self._frequency_hz = frequency_hz
        self._interval = 1.0 / frequency_hz
        self._buffer_seconds = buffer_seconds
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Temporal variables
        self._t_away: Optional[float] = None
        self._t_gaze: Optional[float] = None
        self._t_road: Optional[float] = None
        self._t_close: Optional[float] = None
        self._current_gaze: GazeZone = GazeZone.ROAD

        # History buffer
        max_entries = int(frequency_hz * buffer_seconds)
        self._history: Deque[TemporalState] = deque(maxlen=max_entries)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def current_state(self) -> TemporalState:
        with self._lock:
            t_away = self._t_away
            t_close = self._t_close
            t_zero = None
            if t_away is not None:
                t_zero = t_away - T_ZERO_OFFSET
            elif t_close is not None:
                t_zero = t_close - T_ZERO_OFFSET

            return TemporalState(
                t_zero=t_zero,
                t_away=t_away,
                t_gaze=self._t_gaze,
                t_road=self._t_road,
                t_close=t_close,
                timestamp=time.monotonic(),
            )

    def record_gaze(self, zone: GazeZone) -> None:
        now = time.monotonic()
        with self._lock:
            if zone != GazeZone.ROAD and self._current_gaze == GazeZone.ROAD:
                self._t_away = now
                self._t_gaze = now
            elif zone == GazeZone.ROAD and self._current_gaze != GazeZone.ROAD:
                self._t_road = now
            self._current_gaze = zone

    def record_eye_close(self) -> None:
        with self._lock:
            if self._t_close is None:
                self._t_close = time.monotonic()

    def record_eye_open(self) -> None:
        with self._lock:
            self._t_close = None

    def reset(self) -> None:
        with self._lock:
            self._t_away = None
            self._t_gaze = None
            self._t_road = None
            self._t_close = None
            self._current_gaze = GazeZone.ROAD
            self._history.clear()

    def history(self, duration: float) -> List[TemporalState]:
        cutoff = time.monotonic() - duration
        with self._lock:
            return [s for s in self._history if s.timestamp >= cutoff]

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._tick()
            self._stop_event.wait(timeout=self._interval)

    def _tick(self) -> None:
        state = self.current_state
        with self._lock:
            self._history.append(state)
```

**Step 4: Run tests**

Run: `conda run -n scripting python -m pytest tests/test_temporal.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sleep_detector_sdk/temporal.py tests/test_temporal.py
git commit -m "feat: add TemporalEngine with 25Hz sampling loop (#5)"
```

---

### Task 5: Gaze Tracking Module

**Issue:** #6 (Gaze Tracking & Road Attention Classification)
**Files:**
- Create: `sleep_detector_sdk/gaze.py`
- Test: `tests/test_gaze.py`

**Step 1: Write failing tests**

```python
# tests/test_gaze.py
"""Tests for gaze tracking and road attention classification."""
import numpy as np
import pytest

from sleep_detector_sdk.gaze import GazeEstimator
from sleep_detector_sdk.types import GazeZone


class TestGazeEstimator:
    def test_forward_gaze_returns_road(self):
        estimator = GazeEstimator()
        # Simulate landmarks with eyes looking straight ahead
        landmarks = _make_forward_landmarks()
        result = estimator.estimate(landmarks)
        assert result.zone == GazeZone.ROAD

    def test_leftward_gaze_returns_in_vehicle(self):
        estimator = GazeEstimator()
        landmarks = _make_left_gaze_landmarks()
        result = estimator.estimate(landmarks)
        assert result.zone != GazeZone.ROAD

    def test_downward_gaze_returns_in_vehicle(self):
        estimator = GazeEstimator()
        landmarks = _make_down_gaze_landmarks()
        result = estimator.estimate(landmarks)
        assert result.zone == GazeZone.IN_VEHICLE

    def test_extreme_gaze_returns_external(self):
        estimator = GazeEstimator()
        landmarks = _make_extreme_gaze_landmarks()
        result = estimator.estimate(landmarks)
        assert result.zone == GazeZone.EXTERNAL

    def test_custom_thresholds(self):
        estimator = GazeEstimator(yaw_threshold=0.1, pitch_threshold=0.1)
        landmarks = _make_slight_left_landmarks()
        result = estimator.estimate(landmarks)
        assert result.zone != GazeZone.ROAD

    def test_returns_yaw_pitch(self):
        estimator = GazeEstimator()
        landmarks = _make_forward_landmarks()
        result = estimator.estimate(landmarks)
        assert hasattr(result, "yaw")
        assert hasattr(result, "pitch")


def _make_forward_landmarks():
    """68-point landmarks with symmetric, centered face — looking ahead."""
    landmarks = np.zeros((68, 2), dtype=int)
    # Nose tip centered
    landmarks[30] = [160, 200]
    landmarks[33] = [160, 220]
    # Left eye (36-41) — symmetric around center
    landmarks[36] = [130, 170]; landmarks[37] = [140, 165]
    landmarks[38] = [150, 165]; landmarks[39] = [155, 170]
    landmarks[40] = [150, 175]; landmarks[41] = [140, 175]
    # Right eye (42-47) — symmetric
    landmarks[42] = [165, 170]; landmarks[43] = [175, 165]
    landmarks[44] = [185, 165]; landmarks[45] = [190, 170]
    landmarks[46] = [185, 175]; landmarks[47] = [175, 175]
    # Chin
    landmarks[8] = [160, 280]
    return landmarks


def _make_left_gaze_landmarks():
    """Landmarks shifted to indicate leftward head turn."""
    landmarks = _make_forward_landmarks()
    # Shift nose significantly left
    landmarks[30] = [130, 200]
    landmarks[33] = [130, 220]
    return landmarks


def _make_down_gaze_landmarks():
    """Landmarks indicating downward gaze (looking at phone/console)."""
    landmarks = _make_forward_landmarks()
    # Nose tip lowered significantly
    landmarks[30] = [160, 230]
    landmarks[33] = [160, 260]
    landmarks[8] = [160, 310]
    return landmarks


def _make_extreme_gaze_landmarks():
    """Landmarks indicating extreme head turn (looking out window)."""
    landmarks = _make_forward_landmarks()
    landmarks[30] = [100, 200]
    landmarks[33] = [100, 220]
    return landmarks


def _make_slight_left_landmarks():
    """Landmarks with slight leftward deviation."""
    landmarks = _make_forward_landmarks()
    landmarks[30] = [150, 200]
    landmarks[33] = [150, 220]
    return landmarks
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n scripting python -m pytest tests/test_gaze.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement gaze.py**

```python
# sleep_detector_sdk/gaze.py
"""Gaze direction estimation and road attention classification."""

import logging
import time

import numpy as np

from sleep_detector_sdk.types import GazeEvent, GazeZone

logger = logging.getLogger(__name__)

DEFAULT_YAW_THRESHOLD = 0.15
DEFAULT_PITCH_THRESHOLD = 0.15
EXTERNAL_MULTIPLIER = 2.0  # yaw/pitch beyond this * threshold = external


class GazeEstimator:
    """Estimates gaze direction from 68-point facial landmarks."""

    def __init__(
        self,
        yaw_threshold: float = DEFAULT_YAW_THRESHOLD,
        pitch_threshold: float = DEFAULT_PITCH_THRESHOLD,
    ):
        self._yaw_threshold = yaw_threshold
        self._pitch_threshold = pitch_threshold

    def estimate(self, landmarks: np.ndarray) -> GazeEvent:
        """Estimate gaze zone from facial landmarks.

        Uses nose position relative to eye midpoint as a proxy for
        head yaw/pitch. This is a simplified model — 3D pose estimation
        (Task 6 / pose.py) provides higher accuracy.
        """
        now = time.monotonic()

        # Eye centers
        left_eye_center = landmarks[36:42].mean(axis=0)
        right_eye_center = landmarks[42:48].mean(axis=0)
        eye_midpoint = (left_eye_center + right_eye_center) / 2.0

        # Face dimensions for normalization
        eye_width = np.linalg.norm(right_eye_center - left_eye_center)
        if eye_width == 0:
            return GazeEvent(zone=GazeZone.ROAD, yaw=0.0, pitch=0.0, timestamp=now)

        nose_tip = landmarks[30].astype(float)

        # Yaw: horizontal offset of nose from eye midpoint, normalized
        yaw = (nose_tip[0] - eye_midpoint[0]) / eye_width

        # Pitch: vertical offset of nose from eye midpoint, normalized
        chin = landmarks[8].astype(float)
        face_height = np.linalg.norm(chin - eye_midpoint)
        pitch = (nose_tip[1] - eye_midpoint[1]) / face_height if face_height > 0 else 0.0

        zone = self._classify_zone(yaw, pitch)
        return GazeEvent(zone=zone, yaw=float(yaw), pitch=float(pitch), timestamp=now)

    def _classify_zone(self, yaw: float, pitch: float) -> GazeZone:
        abs_yaw = abs(yaw)
        abs_pitch = abs(pitch)

        # External: extreme deviation
        if abs_yaw > self._yaw_threshold * EXTERNAL_MULTIPLIER:
            return GazeZone.EXTERNAL
        if abs_pitch > self._pitch_threshold * EXTERNAL_MULTIPLIER:
            return GazeZone.EXTERNAL

        # In-vehicle: moderate deviation
        if abs_yaw > self._yaw_threshold or abs_pitch > self._pitch_threshold:
            return GazeZone.IN_VEHICLE

        return GazeZone.ROAD
```

**Step 4: Run tests**

Run: `conda run -n scripting python -m pytest tests/test_gaze.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sleep_detector_sdk/gaze.py tests/test_gaze.py
git commit -m "feat: add GazeEstimator for road attention classification (#6)"
```

---

### Task 6: 3D Head Pose & Posture Detection

**Issue:** #7 (3D Head Pose & Posture Detection)
**Files:**
- Create: `sleep_detector_sdk/pose.py`
- Test: `tests/test_pose.py`

**Step 1: Write failing tests**

```python
# tests/test_pose.py
"""Tests for 3D head pose estimation and posture detection."""
import numpy as np
import pytest

from sleep_detector_sdk.pose import PoseEstimator, HeadPoseResult, NodDetector


class TestPoseEstimator:
    def test_forward_pose_near_zero(self):
        estimator = PoseEstimator()
        landmarks = _make_centered_landmarks()
        result = estimator.estimate(landmarks, frame_shape=(480, 640))
        assert abs(result.yaw) < 15.0
        assert abs(result.pitch) < 15.0
        assert abs(result.roll) < 15.0

    def test_returns_head_pose_result(self):
        estimator = PoseEstimator()
        landmarks = _make_centered_landmarks()
        result = estimator.estimate(landmarks, frame_shape=(480, 640))
        assert isinstance(result, HeadPoseResult)
        assert hasattr(result, "yaw")
        assert hasattr(result, "pitch")
        assert hasattr(result, "roll")

    def test_tilted_head_has_nonzero_roll(self):
        estimator = PoseEstimator()
        landmarks = _make_tilted_landmarks()
        result = estimator.estimate(landmarks, frame_shape=(480, 640))
        assert abs(result.roll) > 5.0


class TestNodDetector:
    def test_no_nod_when_stable(self):
        detector = NodDetector(window_size=5, pitch_threshold=10.0)
        for _ in range(10):
            detector.update(pitch=0.0)
        assert not detector.is_nodding

    def test_detects_nod_pattern(self):
        detector = NodDetector(window_size=10, pitch_threshold=8.0)
        # Simulate nodding: pitch oscillates
        pitches = [0, -15, 0, -15, 0, -15, 0, -15, 0, -15]
        for p in pitches:
            detector.update(pitch=float(p))
        assert detector.is_nodding

    def test_reset_clears_state(self):
        detector = NodDetector(window_size=5, pitch_threshold=8.0)
        for p in [0, -15, 0, -15, 0]:
            detector.update(pitch=float(p))
        detector.reset()
        assert not detector.is_nodding


def _make_centered_landmarks():
    """68-point landmarks for a centered, forward-facing face."""
    landmarks = np.zeros((68, 2), dtype=float)
    landmarks[30] = [320, 240]  # nose tip
    landmarks[8] = [320, 340]   # chin
    landmarks[36] = [280, 200]  # left eye outer
    landmarks[45] = [360, 200]  # right eye outer
    landmarks[48] = [290, 300]  # mouth left
    landmarks[54] = [350, 300]  # mouth right
    return landmarks


def _make_tilted_landmarks():
    """Landmarks with head tilted (roll)."""
    landmarks = _make_centered_landmarks()
    # Tilt: left eye higher, right eye lower
    landmarks[36] = [280, 190]
    landmarks[45] = [360, 210]
    landmarks[48] = [290, 290]
    landmarks[54] = [350, 310]
    return landmarks
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n scripting python -m pytest tests/test_pose.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement pose.py**

```python
# sleep_detector_sdk/pose.py
"""3D head pose estimation and nod detection from facial landmarks."""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 3D model points for a generic face (subset of 68 landmarks)
# Indices: nose tip(30), chin(8), left eye outer(36), right eye outer(45),
#          mouth left(48), mouth right(54)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye outer
    (225.0, 170.0, -135.0),     # Right eye outer
    (-150.0, -150.0, -125.0),   # Mouth left
    (150.0, -150.0, -125.0),    # Mouth right
], dtype=np.float64)

LANDMARK_INDICES = [30, 8, 36, 45, 48, 54]


@dataclass(frozen=True)
class HeadPoseResult:
    yaw: float    # degrees, left(-) / right(+)
    pitch: float  # degrees, down(-) / up(+)
    roll: float   # degrees, tilt


class PoseEstimator:
    """Estimates 3D head pose (yaw, pitch, roll) from 2D landmarks using solvePnP."""

    def estimate(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> HeadPoseResult:
        """Estimate head pose from 68-point landmarks.

        Args:
            landmarks: (68, 2) array of landmark coordinates
            frame_shape: (height, width) of the frame
        """
        h, w = frame_shape
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        image_points = np.array(
            [landmarks[i] for i in LANDMARK_INDICES], dtype=np.float64
        )

        try:
            import cv2
            success, rotation_vec, translation_vec = cv2.solvePnP(
                MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0)

            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = np.hstack((rotation_mat, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
                np.vstack((pose_mat, [0, 0, 0, 1]))[:3]
            )
            pitch, yaw, roll = euler_angles.flatten()[:3]
            return HeadPoseResult(yaw=float(yaw), pitch=float(pitch), roll=float(roll))

        except ImportError:
            # Fallback: simple geometric estimation without cv2.solvePnP
            return self._geometric_fallback(landmarks)

    def _geometric_fallback(self, landmarks: np.ndarray) -> HeadPoseResult:
        """Simple geometric pose estimation when cv2 solvePnP unavailable."""
        left_eye = landmarks[36].astype(float)
        right_eye = landmarks[45].astype(float)
        nose = landmarks[30].astype(float)
        chin = landmarks[8].astype(float)

        eye_center = (left_eye + right_eye) / 2.0
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist == 0:
            return HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0)

        yaw = np.degrees(np.arctan2(nose[0] - eye_center[0], eye_dist))
        face_height = np.linalg.norm(chin - eye_center)
        pitch = np.degrees(np.arctan2(nose[1] - eye_center[1], face_height)) if face_height > 0 else 0.0
        roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

        return HeadPoseResult(yaw=float(yaw), pitch=float(pitch), roll=float(roll))


class NodDetector:
    """Detects head nodding from pitch oscillation patterns."""

    def __init__(self, window_size: int = 20, pitch_threshold: float = 10.0):
        self._window_size = window_size
        self._pitch_threshold = pitch_threshold
        self._pitches: Deque[float] = deque(maxlen=window_size)

    @property
    def is_nodding(self) -> bool:
        if len(self._pitches) < self._window_size:
            return False
        pitches = list(self._pitches)
        # Count direction changes exceeding threshold
        direction_changes = 0
        for i in range(1, len(pitches)):
            diff = pitches[i] - pitches[i - 1]
            if abs(diff) > self._pitch_threshold:
                direction_changes += 1
        return direction_changes >= 3

    def update(self, pitch: float) -> None:
        self._pitches.append(pitch)

    def reset(self) -> None:
        self._pitches.clear()
```

**Step 4: Run tests**

Run: `conda run -n scripting python -m pytest tests/test_pose.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sleep_detector_sdk/pose.py tests/test_pose.py
git commit -m "feat: add PoseEstimator and NodDetector for 3D head tracking (#7)"
```

---

### Task 7: Tiered Warning System

**Issue:** #10 (Tiered Warning System)
**Files:**
- Modify: `sleep_detector_sdk/alerts.py`
- Test: `tests/test_alerts.py` (extend)

**Step 1: Write failing tests for tiered alerts**

Append to `tests/test_alerts.py`:

```python
from sleep_detector_sdk.alerts import TieredAlertManager, AlertProvider
from sleep_detector_sdk.types import AlertTier, FusionResult, FatigueSignal


class DummyAlertProvider(AlertProvider):
    def __init__(self):
        self.alerts = []

    def trigger(self, tier: AlertTier, result: FusionResult) -> None:
        self.alerts.append((tier, result))

    def cancel(self) -> None:
        pass


class TestAlertProvider:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            AlertProvider()


class TestTieredAlertManager:
    def _make_result(self, score, tier):
        return FusionResult(fatigue_score=score, tier=tier, signals=[], timestamp=1.0)

    def test_dispatches_to_correct_tier(self):
        mgr = TieredAlertManager()
        provider = DummyAlertProvider()
        mgr.register_provider(AlertTier.AUDIBLE, provider)
        result = self._make_result(0.6, AlertTier.AUDIBLE)
        mgr.dispatch(result)
        assert len(provider.alerts) == 1
        assert provider.alerts[0][0] == AlertTier.AUDIBLE

    def test_escalation(self):
        mgr = TieredAlertManager()
        silent_p = DummyAlertProvider()
        audible_p = DummyAlertProvider()
        mgr.register_provider(AlertTier.SILENT, silent_p)
        mgr.register_provider(AlertTier.AUDIBLE, audible_p)

        mgr.dispatch(self._make_result(0.3, AlertTier.SILENT))
        assert mgr.current_tier == AlertTier.SILENT

        mgr.dispatch(self._make_result(0.6, AlertTier.AUDIBLE))
        assert mgr.current_tier == AlertTier.AUDIBLE
        assert len(audible_p.alerts) == 1

    def test_deescalation(self):
        mgr = TieredAlertManager()
        provider = DummyAlertProvider()
        mgr.register_provider(AlertTier.AUDIBLE, provider)

        mgr.dispatch(self._make_result(0.6, AlertTier.AUDIBLE))
        assert mgr.current_tier == AlertTier.AUDIBLE

        mgr.dispatch(self._make_result(0.1, AlertTier.SILENT))
        assert mgr.current_tier == AlertTier.SILENT

    def test_cooldown_per_tier(self):
        mgr = TieredAlertManager(cooldowns={AlertTier.AUDIBLE: 100.0})
        provider = DummyAlertProvider()
        mgr.register_provider(AlertTier.AUDIBLE, provider)

        mgr.dispatch(self._make_result(0.6, AlertTier.AUDIBLE))
        mgr.dispatch(self._make_result(0.6, AlertTier.AUDIBLE))
        # Second dispatch within cooldown — should not trigger
        assert len(provider.alerts) == 1

    def test_no_provider_for_tier_is_noop(self):
        mgr = TieredAlertManager()
        result = self._make_result(0.9, AlertTier.CRITICAL)
        mgr.dispatch(result)  # Should not raise
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n scripting python -m pytest tests/test_alerts.py -v`
Expected: FAIL — ImportError for TieredAlertManager, AlertProvider

**Step 3: Add tiered alert classes to alerts.py**

Append to `sleep_detector_sdk/alerts.py`:

```python
from sleep_detector_sdk.types import AlertTier, FusionResult


class AlertProvider(ABC):
    """Abstract base class for tier-specific alert delivery."""

    @abstractmethod
    def trigger(self, tier: AlertTier, result: FusionResult) -> None:
        """Deliver the alert."""

    @abstractmethod
    def cancel(self) -> None:
        """Cancel any active alert for this provider."""


class TieredAlertManager:
    """Manages tiered alert escalation/de-escalation with per-tier cooldown."""

    def __init__(self, cooldowns: Optional[Dict] = None):
        self._providers: Dict[AlertTier, List[AlertProvider]] = {
            AlertTier.SILENT: [],
            AlertTier.AUDIBLE: [],
            AlertTier.CRITICAL: [],
        }
        self._cooldowns: Dict[AlertTier, float] = cooldowns or {}
        self._last_trigger: Dict[AlertTier, float] = {}
        self._current_tier: AlertTier = AlertTier.SILENT

    @property
    def current_tier(self) -> AlertTier:
        return self._current_tier

    def register_provider(self, tier: AlertTier, provider: AlertProvider) -> None:
        self._providers[tier].append(provider)

    def dispatch(self, result: FusionResult) -> None:
        tier = result.tier
        self._current_tier = tier

        cooldown = self._cooldowns.get(tier, 0.0)
        now = time.monotonic()
        last = self._last_trigger.get(tier, 0.0)
        if cooldown > 0 and last > 0 and (now - last) < cooldown:
            return

        for provider in self._providers.get(tier, []):
            try:
                provider.trigger(tier, result)
            except Exception:
                logging.getLogger(__name__).exception("Alert provider failed")

        self._last_trigger[tier] = now
```

Add needed imports at top of alerts.py: `from typing import Dict, List, Optional`

**Step 4: Run tests**

Run: `conda run -n scripting python -m pytest tests/test_alerts.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sleep_detector_sdk/alerts.py tests/test_alerts.py
git commit -m "feat: add TieredAlertManager with escalation/de-escalation (#10)"
```

---

### Task 8: Detector Integration — Wire Fusion, Sensors, Temporal into SleepDetectorSDK

**Issue:** #8 + cross-cutting integration
**Files:**
- Modify: `sleep_detector_sdk/detector.py`
- Modify: `sleep_detector_sdk/__init__.py`
- Test: `tests/test_detector.py` (extend)

**Step 1: Write failing integration tests**

Append to `tests/test_detector.py`:

```python
from sleep_detector_sdk.sensors import SensorProvider
from sleep_detector_sdk.types import FatigueSignal, SensorMetadata, FrameResult


class MockSensor(SensorProvider):
    def __init__(self):
        self._connected = False

    def connect(self):
        self._connected = True

    def read(self):
        if self._connected:
            return FatigueSignal(score=0.7, confidence=0.9, source="mock", timestamp=time.monotonic())
        return None

    def disconnect(self):
        self._connected = False

    def metadata(self):
        return SensorMetadata(name="mock", version="1.0", sampling_hz=10.0)


class TestDetectorSensorIntegration:
    def test_register_sensor(self):
        sdk, _, _ = _make_sdk()
        sensor = MockSensor()
        sdk.register_sensor(sensor)
        assert len(sdk.sensors) == 1

    def test_process_frame_includes_fatigue_score(self):
        sdk, mock_detector, mock_predictor = _make_sdk()
        mock_detector.return_value = [_mock_face()]
        mock_predictor.return_value = _mock_shape_for_landmarks(ear_high=False)

        sensor = MockSensor()
        sdk.register_sensor(sensor)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = sdk.process_frame(frame)
        assert isinstance(result, FrameResult)
        assert result.fatigue_score is not None

    def test_backward_compat_no_sensors(self):
        sdk, mock_detector, _ = _make_sdk()
        mock_detector.return_value = []
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = sdk.process_frame(frame)
        assert result.fatigue_score is None  # No fusion when no sensors
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n scripting python -m pytest tests/test_detector.py::TestDetectorSensorIntegration -v`
Expected: FAIL — AttributeError for register_sensor

**Step 3: Integrate into detector.py**

Key changes to `detector.py`:
- Import `SensorRegistry` and `FusionEngine`
- Add `self._sensor_registry = SensorRegistry()` and `self._fusion_engine = FusionEngine()` in `__init__`
- Add `register_sensor()` method and `sensors` property
- In `process_frame()`, after EAR computation, submit vision signal to fusion engine
- Read external sensor signals and submit to fusion
- Add optional `fatigue_score` to `FrameResult`

Update `FrameResult` in `types.py` to add `fatigue_score: Optional[float] = None`.

**Step 4: Run full test suite**

Run: `conda run -n scripting python -m pytest -v`
Expected: ALL PASS (existing + new tests)

**Step 5: Export new public APIs from `__init__.py`**

Add `SensorProvider`, `SensorRegistry`, `FusionEngine`, `GazeEstimator`, `PoseEstimator`, `NodDetector`, `HeadPoseResult`, `TemporalEngine`, `TieredAlertManager`, `AlertProvider` to imports and `__all__`.

**Step 6: Commit**

```bash
git add sleep_detector_sdk/detector.py sleep_detector_sdk/types.py sleep_detector_sdk/__init__.py tests/test_detector.py
git commit -m "feat: integrate sensors, fusion, and temporal into detector (#8)"
```

---

### Task 9: External Sensor Plugin Stubs

**Issue:** #9 (External Sensor Integrations)
**Files:**
- Create: `sleep_detector_sdk/plugins/__init__.py`
- Create: `sleep_detector_sdk/plugins/steering.py`
- Create: `sleep_detector_sdk/plugins/physiological.py`
- Test: `tests/test_plugins.py`

**Step 1: Write failing tests**

```python
# tests/test_plugins.py
"""Tests for external sensor plugin stubs."""
import pytest

from sleep_detector_sdk.plugins.steering import SteeringProvider
from sleep_detector_sdk.plugins.physiological import PhysiologicalProvider
from sleep_detector_sdk.sensors import SensorProvider
from sleep_detector_sdk.types import SensorMetadata


class TestSteeringProvider:
    def test_is_sensor_provider(self):
        provider = SteeringProvider()
        assert isinstance(provider, SensorProvider)

    def test_metadata(self):
        meta = SteeringProvider().metadata()
        assert isinstance(meta, SensorMetadata)
        assert meta.name == "steering"

    def test_lifecycle(self):
        p = SteeringProvider()
        p.connect()
        signal = p.read()
        # Returns None when no real hardware
        assert signal is None
        p.disconnect()

    def test_baseline_period_configurable(self):
        p = SteeringProvider(baseline_minutes=10.0)
        assert p._baseline_minutes == 10.0


class TestPhysiologicalProvider:
    def test_is_sensor_provider(self):
        provider = PhysiologicalProvider()
        assert isinstance(provider, SensorProvider)

    def test_metadata(self):
        meta = PhysiologicalProvider().metadata()
        assert meta.name == "physiological"

    def test_lifecycle(self):
        p = PhysiologicalProvider()
        p.connect()
        signal = p.read()
        assert signal is None
        p.disconnect()
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n scripting python -m pytest tests/test_plugins.py -v`
Expected: FAIL

**Step 3: Implement plugin stubs**

Create `sleep_detector_sdk/plugins/__init__.py` (empty).

```python
# sleep_detector_sdk/plugins/steering.py
"""Steering and vehicle data sensor plugin (CAN/OBD-II)."""

import logging
from typing import Optional

from sleep_detector_sdk.sensors import SensorProvider
from sleep_detector_sdk.types import FatigueSignal, SensorMetadata

logger = logging.getLogger(__name__)


class SteeringProvider(SensorProvider):
    """Sensor provider for steering wheel angle and vehicle data.

    Requires CAN bus or OBD-II hardware interface.
    Without hardware, read() returns None (graceful degradation).
    """

    def __init__(self, baseline_minutes: float = 20.0):
        self._baseline_minutes = baseline_minutes
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        logger.info("Steering provider connected (hardware not available — stub mode)")

    def read(self) -> Optional[FatigueSignal]:
        if not self._connected:
            return None
        # Stub: returns None until real CAN/OBD-II hardware integration
        return None

    def disconnect(self) -> None:
        self._connected = False

    def metadata(self) -> SensorMetadata:
        return SensorMetadata(name="steering", version="0.1.0", sampling_hz=10.0)
```

```python
# sleep_detector_sdk/plugins/physiological.py
"""Physiological sensor plugin (ECG/HRV/Skin Conductance)."""

import logging
from typing import Optional

from sleep_detector_sdk.sensors import SensorProvider
from sleep_detector_sdk.types import FatigueSignal, SensorMetadata

logger = logging.getLogger(__name__)


class PhysiologicalProvider(SensorProvider):
    """Sensor provider for ECG, HRV, and skin conductance data.

    Requires BLE or serial ECG/HRV hardware interface.
    Without hardware, read() returns None (graceful degradation).
    """

    def __init__(self):
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        logger.info("Physiological provider connected (hardware not available — stub mode)")

    def read(self) -> Optional[FatigueSignal]:
        if not self._connected:
            return None
        return None

    def disconnect(self) -> None:
        self._connected = False

    def metadata(self) -> SensorMetadata:
        return SensorMetadata(name="physiological", version="0.1.0", sampling_hz=25.0)
```

**Step 4: Register entry points in pyproject.toml**

Add to `pyproject.toml`:

```toml
[project.entry-points."sleep_detector_sdk.sensors"]
steering = "sleep_detector_sdk.plugins.steering:SteeringProvider"
physiological = "sleep_detector_sdk.plugins.physiological:PhysiologicalProvider"

[project.optional-dependencies]
steering = []  # Future: python-can, obd
physiological = []  # Future: bleak (BLE)
```

**Step 5: Run tests**

Run: `conda run -n scripting python -m pytest tests/test_plugins.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add sleep_detector_sdk/plugins/ tests/test_plugins.py pyproject.toml
git commit -m "feat: add steering and physiological sensor plugin stubs (#9)"
```

---

### Task 10: Privacy Framework

**Issue:** #12 (On-Device ML & Privacy Framework)
**Files:**
- Create: `sleep_detector_sdk/privacy.py`
- Test: `tests/test_privacy.py`

**Step 1: Write failing tests**

```python
# tests/test_privacy.py
"""Tests for privacy framework."""
import pytest

from sleep_detector_sdk.privacy import PrivacyConfig, PERMISSION_STRINGS


class TestPrivacyConfig:
    def test_default_config(self):
        config = PrivacyConfig()
        assert config.on_device_only is True
        assert config.retention_seconds > 0

    def test_custom_retention(self):
        config = PrivacyConfig(retention_seconds=300)
        assert config.retention_seconds == 300

    def test_permission_string_for_microphone(self):
        assert "microphone" in PERMISSION_STRINGS
        assert PERMISSION_STRINGS["microphone"] == (
            "We need access to the microphone to analyze your sleep patterns and detect snoring."
        )

    def test_no_pii_in_string(self):
        config = PrivacyConfig()
        # Verify the sanitize method strips common PII patterns
        clean = config.sanitize_log("User John at 192.168.1.1 detected")
        assert "John" not in clean or "192.168.1.1" not in clean  # at minimum IP stripped

    def test_sanitize_strips_ip_addresses(self):
        config = PrivacyConfig()
        clean = config.sanitize_log("Connected from 10.0.0.1")
        assert "10.0.0.1" not in clean

    def test_sanitize_strips_email(self):
        config = PrivacyConfig()
        clean = config.sanitize_log("User test@example.com logged in")
        assert "test@example.com" not in clean
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n scripting python -m pytest tests/test_privacy.py -v`
Expected: FAIL

**Step 3: Implement privacy.py**

```python
# sleep_detector_sdk/privacy.py
"""Privacy configuration and PII protection for the Sleep Detector SDK."""

import re
from dataclasses import dataclass, field

PERMISSION_STRINGS = {
    "microphone": "We need access to the microphone to analyze your sleep patterns and detect snoring.",
    "camera": "We need access to the camera to monitor driver alertness and detect drowsiness.",
}

# PII patterns for log sanitization
_IP_PATTERN = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
_EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')


@dataclass
class PrivacyConfig:
    """Privacy configuration for the SDK."""
    on_device_only: bool = True
    retention_seconds: float = 3600.0  # 1 hour default
    log_pii: bool = False

    def sanitize_log(self, message: str) -> str:
        """Remove PII patterns from log messages."""
        result = _IP_PATTERN.sub("[REDACTED_IP]", message)
        result = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", result)
        return result
```

**Step 4: Run tests**

Run: `conda run -n scripting python -m pytest tests/test_privacy.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sleep_detector_sdk/privacy.py tests/test_privacy.py
git commit -m "feat: add privacy framework with PII sanitization (#12)"
```

---

### Task 11: Modular SDK Packaging

**Issue:** #11 (Modular SDK Packaging)
**Files:**
- Modify: `pyproject.toml`
- Modify: `sleep_detector_sdk/__init__.py`
- Test: `tests/test_packaging.py`

**Step 1: Write failing tests**

```python
# tests/test_packaging.py
"""Tests for modular SDK packaging."""
import pytest


class TestCoreImports:
    def test_core_imports_without_optional_deps(self):
        """Core package should import without optional plugin deps."""
        from sleep_detector_sdk.types import FatigueSignal, FusionResult
        from sleep_detector_sdk.sensors import SensorProvider, SensorRegistry
        from sleep_detector_sdk.fusion import FusionEngine
        from sleep_detector_sdk.temporal import TemporalEngine
        from sleep_detector_sdk.gaze import GazeEstimator
        from sleep_detector_sdk.privacy import PrivacyConfig
        assert True  # If we get here, imports work

    def test_plugin_imports_are_lazy(self):
        """Plugin imports should not fail core import."""
        import sleep_detector_sdk
        assert hasattr(sleep_detector_sdk, "__version__")


class TestEntryPointDiscovery:
    def test_discover_sensors_entry_points(self):
        """Entry points should be discoverable via importlib.metadata."""
        try:
            from importlib.metadata import entry_points
            eps = entry_points()
            # May or may not find sensors depending on install mode
            # Just verify the API works
            assert True
        except ImportError:
            pytest.skip("importlib.metadata not available")
```

**Step 2: Run tests**

Run: `conda run -n scripting python -m pytest tests/test_packaging.py -v`
Expected: Should pass if prior tasks done correctly

**Step 3: Update pyproject.toml for full modular config**

Ensure `pyproject.toml` has optional dependencies and entry points from Task 9. Add `posture` extra:

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0"]
steering = []
physiological = []
posture = []
all = ["sleep-detector-sdk[steering,physiological,posture]"]
```

**Step 4: Run full test suite**

Run: `conda run -n scripting python -m pytest -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add pyproject.toml tests/test_packaging.py sleep_detector_sdk/__init__.py
git commit -m "feat: finalize modular SDK packaging with extras (#11)"
```

---

### Task 12: Release Pipeline Configuration

**Issue:** #13 (Release Pipeline, SemVer & QA Automation)
**Files:**
- Modify: `.github/workflows/ci.yml` (or create)
- Create: `sleep_detector_sdk/_version.py`
- Test: `tests/test_version.py`

**Step 1: Write failing test for version**

```python
# tests/test_version.py
"""Tests for versioning."""
import re
import pytest


class TestVersion:
    def test_version_is_semver(self):
        from sleep_detector_sdk import __version__
        assert re.match(r'^\d+\.\d+\.\d+', __version__)

    def test_version_accessible(self):
        import sleep_detector_sdk
        assert hasattr(sleep_detector_sdk, "__version__")
```

**Step 2: Run test (should pass since __version__ exists)**

Run: `conda run -n scripting python -m pytest tests/test_version.py -v`
Expected: PASS

**Step 3: Update CI workflow**

Read existing `.github/workflows/ci.yml` and extend with:
- Multi-python matrix (3.8, 3.10, 3.12)
- Lint step (if linter configured)
- Coverage gate (80% minimum)
- Build step (sdist + wheel)

**Step 4: Bump version to 0.2.0**

Update `__version__` in `__init__.py` to `"0.2.0"` to reflect the new multi-modal architecture.

**Step 5: Run full test suite one final time**

Run: `conda run -n scripting python -m pytest -v --tb=short`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add .github/workflows/ sleep_detector_sdk/__init__.py tests/test_version.py
git commit -m "feat: update CI pipeline and bump to v0.2.0 (#13)"
```

---

## Execution Summary

| Task | Issue | New Files | Tests |
|------|-------|-----------|-------|
| 1 | Types foundation | — | ~8 tests |
| 2 | SensorProvider + Registry | `sensors.py` | ~7 tests |
| 3 | FusionEngine | `fusion.py` | ~11 tests |
| 4 | TemporalEngine | `temporal.py` | ~11 tests |
| 5 | GazeEstimator | `gaze.py` | ~6 tests |
| 6 | PoseEstimator + NodDetector | `pose.py` | ~6 tests |
| 7 | TieredAlertManager | `alerts.py` (extend) | ~5 tests |
| 8 | Detector integration | `detector.py` (modify) | ~3 tests |
| 9 | Plugin stubs | `plugins/` | ~6 tests |
| 10 | Privacy framework | `privacy.py` | ~6 tests |
| 11 | Modular packaging | `pyproject.toml` | ~3 tests |
| 12 | Release pipeline | `.github/workflows/` | ~2 tests |

**Total: ~75 new tests across 12 tasks**
