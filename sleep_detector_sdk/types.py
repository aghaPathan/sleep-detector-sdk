"""Type definitions, enums, and constants for the Sleep Detector SDK."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class EyeState(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class DetectorState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"


DEFAULT_EAR_THRESHOLD = 0.2
DEFAULT_CLOSED_SECONDS = 5.0
DEFAULT_ALERT_COOLDOWN = 3.0


@dataclass(frozen=True)
class DrowsinessEvent:
    duration: float
    ear_value: float
    timestamp: float


@dataclass(frozen=True)
class EyeStateEvent:
    state: EyeState
    ear_value: float
    timestamp: float


@dataclass(frozen=True)
class FaceEvent:
    landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]
    timestamp: float


@dataclass(frozen=True)
class FaceLostEvent:
    last_seen: float
    timestamp: float


@dataclass(frozen=True)
class FrameEvent:
    frame: np.ndarray
    ear_value: float
    eye_state: EyeState
    face_detected: bool
    timestamp: float


@dataclass(frozen=True)
class FrameResult:
    """Result returned by process_frame() for each processed frame."""
    ear_value: float
    eye_state: EyeState
    face_detected: bool
    is_drowsy: bool
    timestamp: float
    fatigue_score: Optional[float] = None


@dataclass(frozen=True)
class FatigueSignal:
    score: float
    confidence: float
    source: str
    timestamp: float


@dataclass(frozen=True)
class SensorMetadata:
    name: str
    version: str
    sampling_hz: float


class AlertTier(Enum):
    SILENT = "silent"
    AUDIBLE = "audible"
    CRITICAL = "critical"


@dataclass(frozen=True)
class FusionResult:
    fatigue_score: float
    tier: AlertTier
    signals: list  # List[FatigueSignal] — use list for 3.8 compat
    timestamp: float


class GazeZone(Enum):
    ROAD = "road"
    IN_VEHICLE = "in_vehicle"
    EXTERNAL = "external"


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
