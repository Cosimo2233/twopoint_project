from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
import threading
from typing import Any, Protocol

from twopoint_project.vision.inferencer import PointPrediction, VisionInferencer


class FrameCapture(Protocol):
    def read_frame(self) -> Any:
        ...


@dataclass(frozen=True)
class VisionFrame:
    captured: Any
    points: list[PointPrediction]


class LatestVisionQueue:
    def __init__(self) -> None:
        self._queue: Queue[VisionFrame] = Queue(maxsize=1)

    def publish(self, frame: VisionFrame) -> None:
        while True:
            try:
                self._queue.put_nowait(frame)
                return
            except Full:
                self._discard_stale_frame()

    def read_latest(self, timeout: float | None = None) -> VisionFrame:
        try:
            latest = self._queue.get(timeout=timeout)
        except Empty as exc:
            raise TimeoutError("timed out waiting for a vision frame") from exc

        while True:
            try:
                latest = self._queue.get_nowait()
            except Empty:
                return latest

    def empty(self) -> bool:
        return self._queue.empty()

    def _discard_stale_frame(self) -> None:
        try:
            self._queue.get_nowait()
        except Empty:
            pass


class VisionProducer:
    def __init__(
        self,
        *,
        capture: FrameCapture,
        inferencer: VisionInferencer,
        queue: LatestVisionQueue | None = None,
        join_timeout: float = 2.0,
    ) -> None:
        self.capture = capture
        self.inferencer = inferencer
        self.queue = queue or LatestVisionQueue()
        self.join_timeout = join_timeout
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def __enter__(self) -> VisionProducer:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="twopoint-vision-producer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.join_timeout)

    def read_latest(self, timeout: float | None = None) -> VisionFrame:
        if self._error is not None and self.queue.empty():
            raise RuntimeError("vision producer failed") from self._error
        try:
            return self.queue.read_latest(timeout=timeout)
        except TimeoutError:
            if self._error is not None:
                raise RuntimeError("vision producer failed") from self._error
            if self._stop_event.is_set():
                raise RuntimeError("vision producer stopped before producing a frame")
            raise

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                captured = self.capture.read_frame()
                points = self.inferencer.predict(captured.frame_bgr)
                self.queue.publish(VisionFrame(captured=captured, points=points))
        except BaseException as exc:
            self._error = exc
            self._stop_event.set()
