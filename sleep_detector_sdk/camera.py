"""Camera management wrapper for OpenCV VideoCapture."""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraManager:
    """Wraps OpenCV VideoCapture with resource management."""

    def __init__(self, camera_index: int = 0):
        self._camera_index = camera_index
        self._cap = None

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> None:
        """Open the camera device.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {self._camera_index}"
            )
        logger.info("Camera opened at index %d", self._camera_index)

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the camera.

        Returns:
            Frame as numpy array, or None if read failed.
        """
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
