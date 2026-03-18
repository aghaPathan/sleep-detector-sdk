"""Sleep Detector SDK — Real-time drowsiness detection using EAR algorithm."""

__version__ = "0.1.0"

from sleep_detector_sdk.alerts import AlertHandler, AlertManager, AlertProvider, TieredAlertManager
from sleep_detector_sdk.ear import compute_ear
from sleep_detector_sdk.events import EventEmitter
from sleep_detector_sdk.fusion import FusionEngine
from sleep_detector_sdk.gaze import GazeEstimator
from sleep_detector_sdk.model_manager import ModelManager
from sleep_detector_sdk.pose import HeadPoseResult, NodDetector, PoseEstimator
from sleep_detector_sdk.privacy import PrivacyConfig
from sleep_detector_sdk.sensors import SensorProvider, SensorRegistry
from sleep_detector_sdk.temporal import TemporalEngine
from sleep_detector_sdk.types import (
    DEFAULT_ALERT_COOLDOWN,
    DEFAULT_CLOSED_SECONDS,
    DEFAULT_EAR_THRESHOLD,
    AlertTier,
    DetectorState,
    DrowsinessEvent,
    EyeState,
    EyeStateEvent,
    FaceEvent,
    FaceLostEvent,
    FatigueSignal,
    FrameEvent,
    FrameResult,
    FusionResult,
    GazeEvent,
    GazeZone,
    SensorMetadata,
    TemporalState,
)

# Lazy import for SleepDetectorSDK to avoid requiring dlib at import time
def __getattr__(name):
    if name == "SleepDetectorSDK":
        from sleep_detector_sdk.detector import SleepDetectorSDK
        return SleepDetectorSDK
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "SleepDetectorSDK",
    "AlertHandler",
    "AlertManager",
    "AlertProvider",
    "TieredAlertManager",
    "EventEmitter",
    "FusionEngine",
    "GazeEstimator",
    "HeadPoseResult",
    "ModelManager",
    "NodDetector",
    "PoseEstimator",
    "PrivacyConfig",
    "SensorProvider",
    "SensorRegistry",
    "TemporalEngine",
    "compute_ear",
    "AlertTier",
    "EyeState",
    "DetectorState",
    "DrowsinessEvent",
    "EyeStateEvent",
    "FaceEvent",
    "FaceLostEvent",
    "FatigueSignal",
    "FrameEvent",
    "FrameResult",
    "FusionResult",
    "GazeEvent",
    "GazeZone",
    "SensorMetadata",
    "TemporalState",
    "DEFAULT_EAR_THRESHOLD",
    "DEFAULT_CLOSED_SECONDS",
    "DEFAULT_ALERT_COOLDOWN",
]
