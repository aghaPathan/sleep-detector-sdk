"""Model file download, caching, and path resolution."""

import bz2
import logging
import os
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"
MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".sleep-detector-sdk", "models")


class ModelManager:
    """Manages the dlib facial landmark model file."""

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    @property
    def model_path(self) -> str:
        return os.path.join(self._cache_dir, MODEL_FILENAME)

    @property
    def is_cached(self) -> bool:
        return os.path.isfile(self.model_path)

    def resolve(self, explicit_path: Optional[str] = None) -> str:
        """Resolve the model file path.

        Args:
            explicit_path: If provided, use this path directly.

        Returns:
            Absolute path to the model file.

        Raises:
            FileNotFoundError: If the model file cannot be found.
        """
        if explicit_path is not None:
            if not os.path.isfile(explicit_path):
                raise FileNotFoundError(f"Model file not found: {explicit_path}")
            return explicit_path

        if self.is_cached:
            return self.model_path

        raise FileNotFoundError(
            f"Model file not found. Run 'sleep-detector-sdk download-model' "
            f"to download it, or provide a path via model_path parameter."
        )

    def download(self, progress_callback=None) -> str:
        """Download the model file from dlib.net.

        Args:
            progress_callback: Optional callable(bytes_downloaded, total_bytes).

        Returns:
            Path to the downloaded model file.
        """
        os.makedirs(self._cache_dir, exist_ok=True)
        compressed_path = self.model_path + ".bz2"

        logger.info("Downloading model from %s", MODEL_URL)
        response = urllib.request.urlopen(MODEL_URL)
        total = int(response.headers.get("Content-Length", 0))

        downloaded = 0
        with open(compressed_path, "wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total)

        logger.info("Decompressing model file")
        with open(compressed_path, "rb") as f_in:
            data = bz2.decompress(f_in.read())
        with open(self.model_path, "wb") as f_out:
            f_out.write(data)

        os.remove(compressed_path)
        logger.info("Model saved to %s", self.model_path)
        return self.model_path
