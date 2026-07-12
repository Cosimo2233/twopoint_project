from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: int
    timestamp: float
    frame_bgr: np.ndarray


class CameraCapture:
    def __init__(
        self,
        camera_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        max_read_failures: int = 10,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.max_read_failures = max_read_failures
        self._capture: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._read_failures = 0

    def __enter__(self) -> CameraCapture:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    def open(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            return

        capture = cv2.VideoCapture(self.camera_index)
        if self.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps is not None:
            capture.set(cv2.CAP_PROP_FPS, self.fps)

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open camera index {self.camera_index}.")

        self._capture = capture
        self._read_failures = 0

    def read_frame(self) -> CapturedFrame:
        if self._capture is None or not self._capture.isOpened():
            self.open()

        if self._capture is None:
            raise RuntimeError("Camera is not initialized.")

        ok, frame_bgr = self._capture.read()
        if not ok or frame_bgr is None:
            self._read_failures += 1
            if self._read_failures >= self.max_read_failures:
                raise RuntimeError(
                    f"Failed to read {self._read_failures} consecutive frames "
                    f"from camera index {self.camera_index}."
                )
            raise RuntimeError(f"Failed to read frame from camera index {self.camera_index}.")

        self._read_failures = 0
        self._frame_id += 1
        return CapturedFrame(
            frame_id=self._frame_id,
            timestamp=time.time(),
            frame_bgr=frame_bgr,
        )

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
