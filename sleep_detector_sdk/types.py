"""Type definitions, enums, and constants for the Sleep Detector SDK."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

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
