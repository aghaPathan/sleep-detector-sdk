"""3D head pose estimation and nod detection.

PoseEstimator uses cv2.solvePnP with a standard 6-point 3D face model to
derive yaw, pitch, and roll from 2D facial landmarks.  When OpenCV is
unavailable a simple geometric fallback is used.

NodDetector detects nodding behaviour by counting direction reversals in a
sliding window of pitch values.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# HeadPoseResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadPoseResult:
    """Euler angles (degrees) describing head orientation.

    Attributes:
        yaw:   Rotation about vertical axis  (negative = left, positive = right).
        pitch: Rotation about lateral axis   (negative = down, positive = up).
        roll:  Rotation about depth axis     (negative = left tilt, positive = right tilt).
    """

    yaw: float
    pitch: float
    roll: float


# ---------------------------------------------------------------------------
# 3-D face model landmarks (millimetres, origin at nose tip)
# Indices into dlib's 68-point scheme: 30, 8, 36, 45, 48, 54
# ---------------------------------------------------------------------------

_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # 30 — nose tip
        (0.0, -330.0, -65.0),     # 8  — chin
        (-225.0, 170.0, -135.0),  # 36 — left eye outer corner
        (225.0, 170.0, -135.0),   # 45 — right eye outer corner
        (-150.0, -150.0, -125.0), # 48 — mouth left corner
        (150.0, -150.0, -125.0),  # 54 — mouth right corner
    ],
    dtype=np.float64,
)

_LANDMARK_INDICES = [30, 8, 36, 45, 48, 54]


# ---------------------------------------------------------------------------
# PoseEstimator
# ---------------------------------------------------------------------------

class PoseEstimator:
    """Estimates 3-D head pose from 68 facial landmarks.

    Usage::

        estimator = PoseEstimator()
        result: HeadPoseResult = estimator.estimate(landmarks, frame_shape=(480, 640))
    """

    def estimate(
        self,
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> HeadPoseResult:
        """Estimate head pose from facial landmarks.

        Args:
            landmarks:   Array of shape (68, 2) containing (x, y) coordinates.
            frame_shape: ``(height, width)`` of the source frame.

        Returns:
            :class:`HeadPoseResult` with yaw, pitch and roll in degrees.
        """
        image_points = np.array(
            [landmarks[i] for i in _LANDMARK_INDICES],
            dtype=np.float64,
        )

        try:
            return self._solvepnp_estimate(image_points, frame_shape)
        except Exception:  # pragma: no cover — cv2 unavailable or degenerate input
            return self._geometric_fallback(landmarks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _solvepnp_estimate(
        self,
        image_points: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> HeadPoseResult:
        """Use cv2.solvePnP to obtain euler angles."""
        import cv2  # local import so module loads without cv2

        height, width = frame_shape
        focal_length = float(width)
        center = (width / 2.0, height / 2.0)

        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, _ = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return self._geometric_fallback_from_points(image_points, frame_shape)

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        return _rotation_matrix_to_euler(rotation_mat, normalize_for_face_model=True)

    @staticmethod
    def _geometric_fallback(landmarks: np.ndarray) -> HeadPoseResult:
        """Simple arctan-based approximation when cv2 is unavailable."""
        nose = landmarks[30].astype(float)
        chin = landmarks[8].astype(float)
        left_eye = landmarks[36].astype(float)
        right_eye = landmarks[45].astype(float)

        # Roll: angle of the eye line
        eye_delta = right_eye - left_eye
        roll = float(math.degrees(math.atan2(eye_delta[1], eye_delta[0])))

        # Pitch: vertical displacement of nose from chin midpoint (rough)
        face_height = float(np.linalg.norm(chin - nose)) or 1.0
        face_cx = (left_eye[0] + right_eye[0]) / 2.0
        pitch = float(math.degrees(math.atan2(nose[1] - face_cx, face_height)))

        # Yaw: horizontal offset of nose from eye-line midpoint
        eye_cx = (left_eye[0] + right_eye[0]) / 2.0
        eye_width = float(np.linalg.norm(right_eye - left_eye)) or 1.0
        yaw = float(math.degrees(math.atan2(nose[0] - eye_cx, eye_width)))

        return HeadPoseResult(yaw=yaw, pitch=pitch, roll=roll)

    @staticmethod
    def _geometric_fallback_from_points(
        image_points: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> HeadPoseResult:  # pragma: no cover
        """Fallback when solvePnP returns success=False."""
        nose = image_points[0]
        left_eye = image_points[2]
        right_eye = image_points[3]

        eye_delta = right_eye - left_eye
        roll = float(math.degrees(math.atan2(eye_delta[1], eye_delta[0])))
        eye_cx = (left_eye[0] + right_eye[0]) / 2.0
        eye_width = float(np.linalg.norm(right_eye - left_eye)) or 1.0
        yaw = float(math.degrees(math.atan2(nose[0] - eye_cx, eye_width)))
        pitch = 0.0

        return HeadPoseResult(yaw=yaw, pitch=pitch, roll=roll)


# ---------------------------------------------------------------------------
# Euler angle extraction
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    """Wrap *angle* to the range [-180, 180]."""
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def _rotation_matrix_to_euler(
    R: np.ndarray,
    normalize_for_face_model: bool = False,
) -> HeadPoseResult:
    """Convert a 3×3 rotation matrix to yaw/pitch/roll (degrees).

    Uses the XYZ decomposition (matching OpenCV's convention):
        pitch = atan2(R[2,1], R[2,2])   — rotation around X (nod)
        yaw   = atan2(-R[2,0], ...)     — rotation around Y (shake)
        roll  = atan2(R[1,0], R[0,0])  — rotation around Z (tilt)

    Args:
        R: 3×3 rotation matrix.
        normalize_for_face_model: When True, subtract 180° from pitch to
            account for the standard face 3-D model convention where a
            forward-facing head has pitch ≈ 180° in camera coordinates.
    """
    pitch = math.degrees(math.atan2(float(R[2, 1]), float(R[2, 2])))
    yaw = math.degrees(
        math.atan2(
            float(-R[2, 0]),
            math.sqrt(float(R[2, 1]) ** 2 + float(R[2, 2]) ** 2),
        )
    )
    roll = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))

    if normalize_for_face_model:
        # The standard 6-point face model (nose at origin, Y-up) is naturally
        # rotated ~180° around X relative to the camera frame.  Subtract that
        # offset so that a forward-facing head maps to pitch ≈ 0.
        pitch = _wrap_angle(pitch - 180.0)

    return HeadPoseResult(yaw=float(yaw), pitch=float(pitch), roll=float(roll))


# ---------------------------------------------------------------------------
# NodDetector
# ---------------------------------------------------------------------------

class NodDetector:
    """Detects nodding by counting significant pitch direction changes.

    A "nod" is declared when at least 3 direction reversals exceed
    *pitch_threshold* degrees within the sliding *window_size* samples.

    Args:
        window_size:     Number of recent pitch samples to consider.
        pitch_threshold: Minimum pitch change (degrees) to count as a reversal.
    """

    def __init__(self, window_size: int = 20, pitch_threshold: float = 10.0) -> None:
        self._window_size = window_size
        self._pitch_threshold = pitch_threshold
        self._buffer: Deque[float] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, pitch: float) -> None:
        """Append a new pitch sample to the sliding window."""
        self._buffer.append(float(pitch))

    @property
    def is_nodding(self) -> bool:
        """True when >= 3 significant direction changes are detected in the window."""
        return self._count_direction_changes() >= 3

    def reset(self) -> None:
        """Clear all accumulated pitch samples."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_direction_changes(self) -> int:
        """Count direction reversals whose magnitude exceeds the threshold."""
        samples = list(self._buffer)
        if len(samples) < 3:
            return 0

        changes = 0
        # Track the last "significant" reference value
        ref = samples[0]
        last_direction: int | None = None  # +1 or -1

        for val in samples[1:]:
            delta = val - ref
            if abs(delta) >= self._pitch_threshold:
                direction = 1 if delta > 0 else -1
                if last_direction is not None and direction != last_direction:
                    changes += 1
                last_direction = direction
                ref = val

        return changes
