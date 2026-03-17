# sleep-detector-sdk

Python SDK for real-time drowsiness/sleep detection using Eye Aspect Ratio (EAR) algorithm.

## Installation

```bash
pip install sleep-detector-sdk
```

## Quick Start

```python
from sleep_detector_sdk import SleepDetectorSDK

detector = SleepDetectorSDK()
detector.on("drowsiness_detected", lambda event: print(f"Alert! Drowsy for {event.duration}s"))
detector.start(camera_index=0)
```
