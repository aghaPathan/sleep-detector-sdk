"""Tests for sleep_detector_sdk.camera — camera management wrapper."""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from sleep_detector_sdk.camera import CameraManager


class TestCameraManager:
    def test_open_calls_video_capture(self):
        with patch("sleep_detector_sdk.camera.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cv2.VideoCapture.return_value = mock_cap

            cam = CameraManager(camera_index=0)
            cam.open()

            mock_cv2.VideoCapture.assert_called_once_with(0)
            assert cam.is_opened is True

    def test_open_raises_on_failure(self):
        with patch("sleep_detector_sdk.camera.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap

            cam = CameraManager(camera_index=0)
            with pytest.raises(RuntimeError, match="Could not open camera"):
                cam.open()

    def test_read_frame_returns_numpy_array(self):
        with patch("sleep_detector_sdk.camera.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, fake_frame)
            mock_cv2.VideoCapture.return_value = mock_cap

            cam = CameraManager(camera_index=0)
            cam.open()
            frame = cam.read_frame()

            assert frame is not None
            assert frame.shape == (480, 640, 3)

    def test_read_frame_returns_none_on_failure(self):
        with patch("sleep_detector_sdk.camera.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)
            mock_cv2.VideoCapture.return_value = mock_cap

            cam = CameraManager(camera_index=0)
            cam.open()
            assert cam.read_frame() is None

    def test_release_cleans_up(self):
        with patch("sleep_detector_sdk.camera.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cv2.VideoCapture.return_value = mock_cap

            cam = CameraManager(camera_index=0)
            cam.open()
            cam.release()

            mock_cap.release.assert_called_once()
            assert cam.is_opened is False

    def test_context_manager(self):
        with patch("sleep_detector_sdk.camera.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap

            with CameraManager(camera_index=0) as cam:
                frame = cam.read_frame()
                assert frame is not None

            mock_cap.release.assert_called_once()
