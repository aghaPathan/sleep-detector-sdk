"""Core SleepDetectorSDK class — orchestrates drowsiness detection."""

import logging
import signal
import threading
import time
from typing import Optional

import numpy as np

from sleep_detector_sdk.alerts import AlertManager
from sleep_detector_sdk.camera import CameraManager
from sleep_detector_sdk.ear import compute_ear
from sleep_detector_sdk.events import EventEmitter
from sleep_detector_sdk.fusion import FusionEngine
from sleep_detector_sdk.gaze import GazeEstimator
from sleep_detector_sdk.model_manager import ModelManager
from sleep_detector_sdk.sensors import SensorRegistry
from sleep_detector_sdk.temporal import TemporalEngine
from sleep_detector_sdk.types import (
    DEFAULT_ALERT_COOLDOWN,
    DEFAULT_CLOSED_SECONDS,
    DEFAULT_EAR_THRESHOLD,
    DetectorState,
    DrowsinessEvent,
    EyeState,
    EyeStateEvent,
    FaceEvent,
    FaceLostEvent,
    FatigueSignal,
    FrameEvent,
    FrameResult,
    GazeZone,
    GazeEvent,
)

logger = logging.getLogger(__name__)

# Eye landmark indices (from imutils/dlib 68-point model)
LEFT_EYE_START, LEFT_EYE_END = 42, 48
RIGHT_EYE_START, RIGHT_EYE_END = 36, 42


class SleepDetectorSDK:
    """Real-time drowsiness detection SDK using Eye Aspect Ratio."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        ear_threshold: float = DEFAULT_EAR_THRESHOLD,
        closed_duration: float = DEFAULT_CLOSED_SECONDS,
        alert_cooldown: float = DEFAULT_ALERT_COOLDOWN,
        cache_dir: Optional[str] = None,
    ):
        self._ear_threshold = ear_threshold
        self._closed_duration = closed_duration
        self._alert_cooldown = alert_cooldown

        # Resolve model path
        self._model_manager = ModelManager(cache_dir=cache_dir)
        self._model_path = self._model_manager.resolve(explicit_path=model_path)

        # Initialize dlib detector and predictor (lazy import)
        import dlib
        self._face_detector = dlib.get_frontal_face_detector()
        self._landmark_predictor = dlib.shape_predictor(self._model_path)

        # Sensor and fusion subsystems
        self._sensor_registry = SensorRegistry()
        self._fusion_engine = FusionEngine()

        # Gaze and temporal subsystems
        self._gaze_estimator = GazeEstimator()
        self._temporal_engine = TemporalEngine()
        self._last_gaze_zone: GazeZone = GazeZone.ROAD

        # Event system
        self._emitter = EventEmitter()
        self._alert_manager = AlertManager(cooldown=alert_cooldown)

        # State
        self._current_ear: float = 0.0
        self._eye_state: EyeState = EyeState.OPEN
        self._eyes_closed_since: Optional[float] = None
        self._face_was_present: bool = False
        self._last_face_seen: float = 0.0
        self._detector_state = DetectorState.IDLE
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- Public properties ---

    @property
    def ear_threshold(self) -> float:
        return self._ear_threshold

    @property
    def closed_duration(self) -> float:
        return self._closed_duration

    @property
    def alert_cooldown(self) -> float:
        return self._alert_cooldown

    @property
    def is_drowsy(self) -> bool:
        with self._lock:
            if self._eyes_closed_since is None:
                return False
            return (time.monotonic() - self._eyes_closed_since) >= self._closed_duration

    @property
    def current_ear(self) -> float:
        with self._lock:
            return self._current_ear

    @property
    def eyes_closed_duration(self) -> float:
        with self._lock:
            if self._eyes_closed_since is None:
                return 0.0
            return time.monotonic() - self._eyes_closed_since

    @property
    def is_running(self) -> bool:
        return self._detector_state == DetectorState.RUNNING

    # --- Event registration ---

    def on(self, event: str, handler) -> None:
        """Register a callback for an event."""
        if event == "drowsiness_detected":
            self._alert_manager.add_callback(handler)
        else:
            self._emitter.on(event, handler)

    def off(self, event: str, handler) -> None:
        """Remove a callback for an event."""
        self._emitter.off(event, handler)

    def add_alert_handler(self, handler) -> None:
        """Register an AlertHandler subclass."""
        self._alert_manager.add_handler(handler)

    def register_sensor(self, provider) -> None:
        """Register an external SensorProvider for multi-modal fusion."""
        self._sensor_registry.register(provider)

    @property
    def sensors(self):
        """Return registered sensor providers."""
        return self._sensor_registry.sensors

    @property
    def temporal(self):
        """Access the TemporalEngine for state queries."""
        return self._temporal_engine

    # --- Frame processing ---

    def _extract_landmarks(self, gray: np.ndarray, face) -> np.ndarray:
        """Extract 68-point facial landmarks as numpy array."""
        shape = self._landmark_predictor(gray, face)
        # Convert shape to numpy array of (x, y) coordinates
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """Process a single frame for drowsiness detection."""
        now = time.monotonic()
        gray = frame
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self._face_detector(gray)
        face_detected = len(faces) > 0

        ear_value = 0.0
        current_eye_state = self._eye_state

        if face_detected:
            face = faces[0]
            landmarks = self._extract_landmarks(gray, face)

            # Emit face_detected if face just appeared
            if not self._face_was_present:
                self._emitter.emit(
                    "face_detected",
                    FaceEvent(
                        landmarks=landmarks,
                        bbox=(face.left(), face.top(), face.right(), face.bottom()),
                        timestamp=now,
                    ),
                )

            left_eye = landmarks[LEFT_EYE_START:LEFT_EYE_END]
            right_eye = landmarks[RIGHT_EYE_START:RIGHT_EYE_END]

            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            ear_value = (left_ear + right_ear) / 2.0

            with self._lock:
                self._current_ear = ear_value

            # Gaze estimation and temporal recording
            gaze_result = self._gaze_estimator.estimate(landmarks)
            if gaze_result.zone != self._last_gaze_zone:
                if gaze_result.zone != GazeZone.ROAD and self._last_gaze_zone == GazeZone.ROAD:
                    self._emitter.emit("gaze_away", gaze_result)
                elif gaze_result.zone == GazeZone.ROAD and self._last_gaze_zone != GazeZone.ROAD:
                    self._emitter.emit("gaze_returned", gaze_result)
                self._last_gaze_zone = gaze_result.zone
            self._temporal_engine.record_gaze(gaze_result.zone)

            # Submit vision signal to fusion engine
            vision_score = (
                max(0.0, min(1.0, 1.0 - (ear_value / self._ear_threshold)))
                if self._ear_threshold > 0
                else 0.0
            )
            self._fusion_engine.submit_signal(
                FatigueSignal(score=vision_score, confidence=1.0, source="vision", timestamp=now)
            )

            # Determine eye state
            if ear_value < self._ear_threshold:
                if self._eye_state == EyeState.OPEN:
                    current_eye_state = EyeState.CLOSING
                else:
                    current_eye_state = EyeState.CLOSED

                if self._eyes_closed_since is None:
                    with self._lock:
                        self._eyes_closed_since = now
            else:
                current_eye_state = EyeState.OPEN
                with self._lock:
                    self._eyes_closed_since = None

            # Emit eye state change
            if current_eye_state != self._eye_state:
                self._emitter.emit(
                    "eye_state_change",
                    EyeStateEvent(
                        state=current_eye_state,
                        ear_value=ear_value,
                        timestamp=now,
                    ),
                )
                self._eye_state = current_eye_state

            # Feed eye state to temporal engine
            if current_eye_state == EyeState.CLOSED:
                self._temporal_engine.record_eye_close()
            elif current_eye_state == EyeState.OPEN:
                self._temporal_engine.record_eye_open()

            # Check drowsiness
            with self._lock:
                closed_since = self._eyes_closed_since
            if closed_since is not None:
                duration = now - closed_since
                if duration >= self._closed_duration:
                    event = DrowsinessEvent(
                        duration=duration,
                        ear_value=ear_value,
                        timestamp=now,
                    )
                    if self._alert_manager.should_alert(event):
                        self._alert_manager.dispatch(event)

            self._last_face_seen = now

        else:
            # No face detected
            with self._lock:
                self._current_ear = 0.0
                self._eyes_closed_since = None

            if self._face_was_present:
                self._emitter.emit(
                    "face_lost",
                    FaceLostEvent(last_seen=self._last_face_seen, timestamp=now),
                )

        self._face_was_present = face_detected

        # Read external sensors and compute fusion
        for signal in self._sensor_registry.read_all():
            self._fusion_engine.submit_signal(signal)

        fatigue_score = None
        if self._sensor_registry.sensors:
            fusion_result = self._fusion_engine.compute()
            fatigue_score = fusion_result.fatigue_score

        # Always emit frame_processed
        self._emitter.emit(
            "frame_processed",
            FrameEvent(
                frame=frame,
                ear_value=ear_value,
                eye_state=self._eye_state,
                face_detected=face_detected,
                timestamp=now,
            ),
        )

        return FrameResult(
            ear_value=ear_value,
            eye_state=self._eye_state,
            face_detected=face_detected,
            is_drowsy=self.is_drowsy,
            timestamp=now,
            fatigue_score=fatigue_score,
        )

    # --- Managed camera loop ---

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        if threading.current_thread() is not threading.main_thread():
            return
        self._prev_sigint = signal.getsignal(signal.SIGINT)
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_signal(signum, frame):
            logger.info("Received signal %s, stopping detector...", signum)
            self.stop()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if threading.current_thread() is not threading.main_thread():
            return
        if hasattr(self, "_prev_sigint"):
            signal.signal(signal.SIGINT, self._prev_sigint)
        if hasattr(self, "_prev_sigterm"):
            signal.signal(signal.SIGTERM, self._prev_sigterm)

    def start(self, camera_index: int = 0, blocking: bool = True) -> None:
        """Start the detection loop with managed camera."""
        self._detector_state = DetectorState.RUNNING
        self._stop_event.clear()
        self._temporal_engine.start()
        self._install_signal_handlers()

        if blocking:
            try:
                self._run_loop(camera_index)
            finally:
                self._restore_signal_handlers()
        else:
            self._thread = threading.Thread(
                target=self._run_loop,
                args=(camera_index,),
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the detection loop."""
        self._temporal_engine.stop()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._detector_state = DetectorState.STOPPED
        self._restore_signal_handlers()

    def _run_loop(self, camera_index: int) -> None:
        """Internal camera loop."""
        with CameraManager(camera_index=camera_index) as camera:
            while not self._stop_event.is_set():
                frame = camera.read_frame()
                if frame is None:
                    logger.warning("Failed to read frame, retrying...")
                    continue
                self.process_frame(frame)

        self._detector_state = DetectorState.STOPPED
