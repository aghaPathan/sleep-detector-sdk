# sleep-detector-sdk

[![PyPI version](https://img.shields.io/pypi/v/sleep-detector-sdk.svg)](https://pypi.org/project/sleep-detector-sdk/)
[![Python versions](https://img.shields.io/pypi/pyversions/sleep-detector-sdk.svg)](https://pypi.org/project/sleep-detector-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A Python SDK for real-time drowsiness and sleep detection using the Eye Aspect Ratio (EAR) algorithm. It uses dlib's 68-point facial landmark predictor to track eye openness and fires configurable events when drowsiness is detected. Works with any camera or frame source -- bring your own frames, or let the SDK manage an OpenCV camera for you.

## Overview

![Sleep Detector SDK Infographic](sleep-detector-sdk-info-graphic.png)

## Features

- Real-time Eye Aspect Ratio (EAR) computation from 68-point facial landmarks
- Event-driven architecture with callbacks for drowsiness alerts, eye state changes, face detection, and frame processing
- Pluggable `AlertHandler` ABC for structured alert implementations (logging, sound, HTTP, etc.)
- Alert cooldown to prevent duplicate notifications
- Managed camera mode (blocking or background thread) or camera-agnostic frame-by-frame processing
- Thread-safe state properties for polling-based integrations
- CLI tool for downloading the required dlib model file
- Automatic model caching in `~/.sleep-detector-sdk/models/`

## Installation

```bash
pip install sleep-detector-sdk
```

After installing, download the dlib facial landmark model (approximately 100 MB):

```bash
sleep-detector-sdk download-model
```

The model is saved to `~/.sleep-detector-sdk/models/` by default. To specify a custom directory:

```bash
sleep-detector-sdk download-model --path /path/to/models
```

### System Dependencies

`dlib` requires CMake and a C++ compiler. On most systems:

```bash
# macOS
brew install cmake

# Ubuntu / Debian
sudo apt-get install cmake build-essential

# Windows
# Install CMake from https://cmake.org/download/ and Visual Studio Build Tools
```

## Quick Start

The simplest way to get started -- open the default camera and print an alert when drowsiness is detected:

```python
from sleep_detector_sdk import SleepDetectorSDK

detector = SleepDetectorSDK()
detector.on("drowsiness_detected", lambda event: print(f"Alert! Drowsy for {event.duration:.1f}s (EAR={event.ear_value:.3f})"))

# Blocks the main thread; press Ctrl+C to stop
detector.start(camera_index=0)
```

## Usage Examples

### Frame-Based Processing (Camera-Agnostic)

Use `process_frame()` to integrate with any frame source -- a video file, IP camera, ROS topic, or test harness:

```python
import cv2
from sleep_detector_sdk import SleepDetectorSDK

detector = SleepDetectorSDK(
    ear_threshold=0.22,
    closed_duration=3.0,
    alert_cooldown=5.0,
)

detector.on("drowsiness_detected", lambda e: print(f"Drowsy! Duration: {e.duration:.1f}s"))

cap = cv2.VideoCapture("driver_video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    detector.process_frame(frame)  # accepts BGR or grayscale numpy arrays
cap.release()
```

### Managed Camera with Background Thread

Run detection in a background thread so your main thread stays free:

```python
import time
from sleep_detector_sdk import SleepDetectorSDK

detector = SleepDetectorSDK()
detector.on("drowsiness_detected", lambda e: print(f"Drowsy: {e.duration:.1f}s"))

detector.start(camera_index=0, blocking=False)

try:
    while detector.is_running:
        # Do other work, or poll state (see "State Polling" below)
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    detector.stop()
```

### Using the AlertHandler ABC

For structured alert handling, subclass `AlertHandler` instead of using plain callbacks:

```python
import logging
from sleep_detector_sdk import SleepDetectorSDK, AlertHandler, DrowsinessEvent

class LoggingAlertHandler(AlertHandler):
    def on_alert(self, event: DrowsinessEvent) -> None:
        logging.warning(
            "Drowsiness detected: duration=%.1fs, EAR=%.3f",
            event.duration,
            event.ear_value,
        )

class WebhookAlertHandler(AlertHandler):
    def __init__(self, url: str):
        self.url = url

    def on_alert(self, event: DrowsinessEvent) -> None:
        import urllib.request, json
        payload = json.dumps({"duration": event.duration, "ear": event.ear_value}).encode()
        req = urllib.request.Request(self.url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req)

detector = SleepDetectorSDK()
detector.add_alert_handler(LoggingAlertHandler())
detector.add_alert_handler(WebhookAlertHandler("https://example.com/alert"))

detector.start(camera_index=0)
```

Multiple `AlertHandler` instances and plain callbacks can be used simultaneously. All are subject to the same cooldown timer.

### State Polling

All state properties are thread-safe and can be read from any thread:

```python
import time
from sleep_detector_sdk import SleepDetectorSDK

detector = SleepDetectorSDK()
detector.start(camera_index=0, blocking=False)

while detector.is_running:
    print(
        f"EAR: {detector.current_ear:.3f} | "
        f"Drowsy: {detector.is_drowsy} | "
        f"Eyes closed: {detector.eyes_closed_duration:.1f}s"
    )
    time.sleep(0.5)

detector.stop()
```

### All Available Events

```python
from sleep_detector_sdk import SleepDetectorSDK

detector = SleepDetectorSDK()

# Fired when eyes stay closed beyond closed_duration (subject to alert_cooldown)
detector.on("drowsiness_detected", lambda e: print(f"DROWSY: {e.duration:.1f}s, EAR={e.ear_value:.3f}"))

# Fired on every eye state transition (open -> closing -> closed, or closed -> open)
detector.on("eye_state_change", lambda e: print(f"Eyes: {e.state.value}, EAR={e.ear_value:.3f}"))

# Fired when a face first appears in frame
detector.on("face_detected", lambda e: print(f"Face found at bbox={e.bbox}"))

# Fired when a previously-visible face disappears
detector.on("face_lost", lambda e: print(f"Face lost, last seen at t={e.last_seen:.2f}"))

# Fired after every frame is processed (high frequency)
detector.on("frame_processed", lambda e: print(f"Frame: face={e.face_detected}, EAR={e.ear_value:.3f}"))

detector.start(camera_index=0)
```

## Configuration Options

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` or `None` | `None` | Explicit path to the dlib 68-point landmark `.dat` file. If `None`, uses the cached model in `~/.sleep-detector-sdk/models/`. |
| `ear_threshold` | `float` | `0.2` | EAR value below which eyes are considered closed. Lower values require more lid closure to trigger. |
| `closed_duration` | `float` | `5.0` | Seconds eyes must remain closed before a drowsiness event fires. |
| `alert_cooldown` | `float` | `3.0` | Minimum seconds between consecutive drowsiness alerts. Prevents alert spam. |
| `cache_dir` | `str` or `None` | `None` | Directory for model file caching. Defaults to `~/.sleep-detector-sdk/models/`. |

## Event Reference

| Event Name | Payload Type | When Emitted |
|---|---|---|
| `drowsiness_detected` | `DrowsinessEvent` | Eyes closed for longer than `closed_duration`, subject to `alert_cooldown`. |
| `eye_state_change` | `EyeStateEvent` | Eye state transitions between `open`, `closing`, and `closed`. |
| `face_detected` | `FaceEvent` | A face appears in frame after being absent. |
| `face_lost` | `FaceLostEvent` | A previously-visible face is no longer detected. |
| `frame_processed` | `FrameEvent` | Every frame, after all detection logic runs. |

### Payload Types

**`DrowsinessEvent`** -- `duration: float`, `ear_value: float`, `timestamp: float`

**`EyeStateEvent`** -- `state: EyeState`, `ear_value: float`, `timestamp: float`

**`FaceEvent`** -- `landmarks: np.ndarray` (68x2), `bbox: Tuple[int, int, int, int]` (left, top, right, bottom), `timestamp: float`

**`FaceLostEvent`** -- `last_seen: float`, `timestamp: float`

**`FrameEvent`** -- `frame: np.ndarray`, `ear_value: float`, `eye_state: EyeState`, `face_detected: bool`, `timestamp: float`

### Enums

**`EyeState`** -- `OPEN`, `CLOSING`, `CLOSED`

**`DetectorState`** -- `IDLE`, `RUNNING`, `STOPPED`

## API Reference

### `SleepDetectorSDK`

The main class that orchestrates detection.

**Constructor:**

```python
SleepDetectorSDK(
    model_path=None,
    ear_threshold=0.2,
    closed_duration=5.0,
    alert_cooldown=3.0,
    cache_dir=None,
)
```

**Methods:**

| Method | Description |
|---|---|
| `process_frame(frame: np.ndarray)` | Process a single BGR or grayscale frame. Runs detection and emits events. |
| `start(camera_index=0, blocking=True)` | Start the managed camera loop. Set `blocking=False` to run in a background thread. |
| `stop()` | Stop the managed camera loop. Joins the background thread if running. |
| `on(event: str, handler: Callable)` | Register a callback for an event. |
| `off(event: str, handler: Callable)` | Remove a previously registered callback. |
| `add_alert_handler(handler: AlertHandler)` | Register an `AlertHandler` subclass for drowsiness alerts. |

**Properties (all thread-safe):**

| Property | Type | Description |
|---|---|---|
| `current_ear` | `float` | Most recent EAR value (0.0 if no face detected). |
| `is_drowsy` | `bool` | `True` if eyes have been closed for at least `closed_duration`. |
| `eyes_closed_duration` | `float` | Seconds the eyes have been continuously closed (0.0 if open). |
| `is_running` | `bool` | `True` if the managed camera loop is active. |
| `ear_threshold` | `float` | The configured EAR threshold. |
| `closed_duration` | `float` | The configured closed-eyes duration threshold. |
| `alert_cooldown` | `float` | The configured alert cooldown period. |

### `AlertHandler` (ABC)

Abstract base class for structured alert handling.

```python
class AlertHandler(ABC):
    @abstractmethod
    def on_alert(self, event: DrowsinessEvent) -> None: ...
```

### `compute_ear(eye: np.ndarray) -> float`

Standalone function to compute the Eye Aspect Ratio from 6 eye landmark points (shape `(6, 2)`).

### `ModelManager`

Manages model file download, caching, and path resolution. Typically used internally, but available for advanced use.

```python
from sleep_detector_sdk import ModelManager

mm = ModelManager(cache_dir="/custom/path")
mm.download(progress_callback=lambda downloaded, total: print(f"{downloaded}/{total}"))
print(mm.model_path)
print(mm.is_cached)
```

## Development

```bash
git clone https://github.com/aghaPathan/sleep-detector-sdk.git
cd sleep-detector-sdk
pip install -e ".[dev]"
sleep-detector-sdk download-model
pytest
```

Run tests with coverage:

```bash
pytest --cov=sleep_detector_sdk --cov-report=term-missing
```

## License

MIT License. See [LICENSE](LICENSE) for details.
