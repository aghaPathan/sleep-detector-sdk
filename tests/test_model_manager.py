"""Tests for sleep_detector_sdk.model_manager — model download and path resolution."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from sleep_detector_sdk.model_manager import ModelManager


class TestModelManager:
    def test_default_cache_dir(self):
        mm = ModelManager()
        expected = os.path.join(os.path.expanduser("~"), ".sleep-detector-sdk", "models")
        assert mm.cache_dir == expected

    def test_custom_cache_dir(self):
        mm = ModelManager(cache_dir="/tmp/custom-models")
        assert mm.cache_dir == "/tmp/custom-models"

    def test_model_path_returns_path_in_cache_dir(self):
        mm = ModelManager(cache_dir="/tmp/test-cache")
        path = mm.model_path
        assert path.startswith("/tmp/test-cache")
        assert path.endswith("shape_predictor_68_face_landmarks.dat")

    def test_is_cached_returns_false_when_file_missing(self):
        mm = ModelManager(cache_dir="/tmp/nonexistent-dir-xyz")
        assert mm.is_cached is False

    def test_is_cached_returns_true_when_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = os.path.join(tmpdir, "shape_predictor_68_face_landmarks.dat")
            with open(model_file, "wb") as f:
                f.write(b"fake model data")
            mm = ModelManager(cache_dir=tmpdir)
            assert mm.is_cached is True

    def test_resolve_returns_explicit_path_if_valid(self):
        with tempfile.NamedTemporaryFile(suffix=".dat") as f:
            mm = ModelManager()
            resolved = mm.resolve(explicit_path=f.name)
            assert resolved == f.name

    def test_resolve_raises_if_explicit_path_invalid(self):
        mm = ModelManager()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            mm.resolve(explicit_path="/nonexistent/model.dat")

    def test_resolve_returns_cached_path_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = os.path.join(tmpdir, "shape_predictor_68_face_landmarks.dat")
            with open(model_file, "wb") as f:
                f.write(b"fake model data")
            mm = ModelManager(cache_dir=tmpdir)
            assert mm.resolve() == model_file

    def test_resolve_raises_when_not_cached_and_no_explicit_path(self):
        mm = ModelManager(cache_dir="/tmp/nonexistent-dir-xyz")
        with pytest.raises(FileNotFoundError, match="download-model"):
            mm.resolve()
