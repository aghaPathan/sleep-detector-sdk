"""GazeEstimator — road attention zone classification from 68-point landmarks."""

import time

import numpy as np

from sleep_detector_sdk.types import GazeEvent, GazeZone


class GazeEstimator:
    """Estimates driver gaze zone from facial landmarks using nose-eye geometry.

    Uses nose tip position (landmark 30) relative to the eye midpoint
    (average of landmarks 36-41 and 42-47) as a proxy for head yaw and pitch.

    Yaw  = horizontal offset of nose from eye midpoint, normalized by
           inter-eye distance (distance between the two eye region midpoints).
    Pitch = vertical offset of nose from eye midpoint, normalized by
            face height (chin landmark 8 to eye midpoint y).

    Zone classification:
        abs(yaw)  > threshold*2 OR abs(pitch) > threshold*2  → EXTERNAL
        abs(yaw)  > threshold   OR abs(pitch) > threshold    → IN_VEHICLE
        else                                                  → ROAD
    """

    def __init__(self, yaw_threshold: float = 0.15, pitch_threshold: float = 0.15) -> None:
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

    def estimate(self, landmarks: np.ndarray) -> GazeEvent:
        """Estimate gaze zone from a 68-point landmark array.

        Args:
            landmarks: np.ndarray of shape (68, 2) — (x, y) coordinates.

        Returns:
            GazeEvent with zone, yaw, pitch, and timestamp.
        """
        right_eye_pts = landmarks[36:42]   # landmarks 36-41
        left_eye_pts = landmarks[42:48]    # landmarks 42-47

        right_eye_center = right_eye_pts.mean(axis=0)
        left_eye_center = left_eye_pts.mean(axis=0)

        eye_midpoint = (right_eye_center + left_eye_center) / 2.0

        inter_eye_dist = float(np.linalg.norm(left_eye_center - right_eye_center))
        if inter_eye_dist < 1e-6:
            inter_eye_dist = 1e-6

        nose = landmarks[30].astype(np.float64)
        chin = landmarks[8].astype(np.float64)

        face_height = float(abs(chin[1] - eye_midpoint[1]))
        if face_height < 1e-6:
            face_height = 1e-6

        yaw = float((nose[0] - eye_midpoint[0]) / inter_eye_dist)
        pitch = float((nose[1] - eye_midpoint[1]) / face_height)

        if abs(yaw) > self.yaw_threshold * 2 or abs(pitch) > self.pitch_threshold * 2:
            zone = GazeZone.EXTERNAL
        elif abs(yaw) > self.yaw_threshold or abs(pitch) > self.pitch_threshold:
            zone = GazeZone.IN_VEHICLE
        else:
            zone = GazeZone.ROAD

        return GazeEvent(zone=zone, yaw=yaw, pitch=pitch, timestamp=time.time())
