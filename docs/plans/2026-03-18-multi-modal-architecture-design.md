# ADR: Next-Gen DMS SDK — Multi-Modal Architecture

**Date**: 2026-03-18
**Status**: Accepted
**Parent PRD**: #2 (Next-Gen Driver Monitoring System SDK)
**Resolves**: #3 (Architecture Decision: Multi-Modal & Multi-Platform Strategy)

## Decisions

### 1. Platform Strategy: Python-First

**Decision**: All code stays in Python. Mobile/embedded platforms are deferred to a future milestone.

**Rationale**: The current SDK is Python-only with a Python user base. Mobile is explicitly deferred. Building a C++ core now adds build complexity with no immediate consumer. Sensor plugins are Python ABCs — simple, testable, familiar.

**Trade-off accepted**: If mobile becomes urgent, we may need to extract a C++ core later. This is acceptable — Python-first lets us validate the architecture quickly.

### 2. Data Flow: Weighted Fusion Hub

**Decision**: Each sensor produces a `FatigueSignal(score, confidence, source, timestamp)`. A central `FusionEngine` collects signals and computes a weighted fatigue score.

**Rationale**:
- Optional sensors = missing signals, not broken pipeline (graceful degradation)
- Per-sensor confidence weighting (vision high in daylight, low at night; physio always high)
- Easy testing — mock individual signals
- Clean separation: sensors produce, fusion combines, alerts consume
- Current EAR-only behavior = single vision signal with confidence 1.0

**Alternatives rejected**:
- Bus pattern: too decoupled — harder to reason about fusion ordering
- Pipeline chain: rigid ordering, breaks when sensors are optional

### 3. Temporal Metrics: Separate 25Hz Loop

**Decision**: A `TemporalEngine` runs its own dedicated 25Hz timer thread, independent of frame processing rate.

**Rationale**: The PRD mandates 25Hz minimum sampling. Tying temporal metrics to frame rate makes this guarantee fragile (cameras may run at 15fps or 30fps). A dedicated thread with its own timer is the cleanest way to guarantee the sampling contract.

**Trade-off accepted**: Extra thread adds complexity and requires careful synchronization. Mitigated by reading from FusionEngine's thread-safe state snapshot.

### 4. Module Boundaries: Entry Point Plugins

**Decision**: Core defines `sleep_detector_sdk.sensors` entry point group. Extras register via `pyproject.toml` entry points. Core discovers them at startup with `importlib.metadata.entry_points()`.

**Rationale**:
- Clean boundary between core and extras (separate install, separate deps)
- Third-party plugin support for free
- Auto-discovery without import-time coupling
- Works within monorepo now, scales to separate repos later

**Alternatives rejected**:
- Namespace packages: harder to manage, confusing for users
- Single package with extras: package grows unbounded, deps leak

## Architecture

### Core Data Flow

```
Sensors (Vision, Steering, Physiological)
    │
    ▼ FatigueSignal(score, confidence, source, timestamp)
    │
FusionEngine (weighted combination)
    │
    ▼ FusionResult(fatigue_score, tier, signals)
    │
    ├──▶ TieredAlertSystem (haptic → audible → critical)
    └──▶ TemporalEngine (25Hz, T₀/T_away/T_gaze/T_road/T_close)
```

### Threading Model

| Thread | Responsibility | Sync Mechanism |
|--------|---------------|----------------|
| Main | User code, register sensors, poll state | — |
| Vision | process_frame(), face detect, EAR/gaze/pose | Lock-guarded state |
| Temporal | 25Hz tick, compute T_* variables | Reads FusionEngine snapshot |
| Sensor (N) | Per-sensor data acquisition | Push to FusionEngine (lock) |

- `FusionEngine` uses `threading.Lock` for signal collection
- State properties remain lock-guarded (existing pattern)
- Sensor threads are daemon threads
- SIGINT/SIGTERM handlers stop all threads gracefully

### Module Map

**Core package** (`sleep-detector-sdk`):

| Module | Status | Responsibility |
|--------|--------|---------------|
| `detector.py` | Modified | + sensor registration, fusion integration |
| `ear.py` | Unchanged | Pure EAR computation |
| `camera.py` | Unchanged | Camera management |
| `events.py` | Unchanged | Event emitter |
| `alerts.py` | Modified | Tiered alert system (haptic/audible/critical) |
| `model_manager.py` | Modified | Multi-model support |
| `types.py` | Modified | + FatigueSignal, FusionResult, SensorMetadata, TemporalState |
| `cli.py` | Unchanged | CLI commands |
| `sensors.py` | **New** | SensorProvider ABC, SensorRegistry, entry point discovery |
| `fusion.py` | **New** | FusionEngine, weighted scoring, signal collection |
| `temporal.py` | **New** | TemporalEngine, 25Hz loop, Euro NCAP T_* variables |
| `gaze.py` | **New** | Gaze direction estimation, zone classification |
| `pose.py` | **New** | 3D head pose, posture tracking |
| `privacy.py` | **New** | Privacy config, permission strings, PII audit |

**Optional extras** (entry point plugins):

| Extra | Package Path | Entry Point |
|-------|-------------|-------------|
| `[steering]` | `sleep_detector_sdk.plugins.steering` | `sleep_detector_sdk.sensors` |
| `[physiological]` | `sleep_detector_sdk.plugins.physiological` | `sleep_detector_sdk.sensors` |
| `[posture]` | `sleep_detector_sdk.plugins.posture` | `sleep_detector_sdk.sensors` |

### Key Interfaces

```python
@dataclass(frozen=True)
class FatigueSignal:
    score: float        # 0.0 (alert) to 1.0 (critical fatigue)
    confidence: float   # 0.0 (unreliable) to 1.0 (high confidence)
    source: str         # e.g., "vision", "steering", "ecg"
    timestamp: float

class SensorProvider(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def read(self) -> Optional[FatigueSignal]: ...
    @abstractmethod
    def disconnect(self) -> None: ...
    @abstractmethod
    def metadata(self) -> SensorMetadata: ...

class FusionEngine:
    def submit_signal(self, signal: FatigueSignal) -> None: ...
    def compute(self) -> FusionResult: ...
    def configure_weights(self, weights: Dict[str, float]) -> None: ...
```

### Backward Compatibility

- `process_frame()` continues to work exactly as before
- Without registered sensors, FusionEngine receives only the vision signal
- `FrameResult` gains optional `fatigue_score` field (None when no fusion)
- All existing events preserved; new events added alongside
- Existing `AlertHandler` ABC still works; tiered system is opt-in
