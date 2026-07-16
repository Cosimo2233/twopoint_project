from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
import time
import unittest

from twopoint_project.vision.pipeline import LatestVisionQueue, VisionFrame, VisionProducer


def make_frame(frame_id: int) -> VisionFrame:
    return VisionFrame(
        captured=Namespace(frame_id=frame_id, frame_bgr=object()),
        points=[{"label": "target_center", "x": float(frame_id), "y": 0.0, "confidence": 1.0}],
    )


class LatestVisionQueueTest(unittest.TestCase):
    def test_publish_drops_stale_frame_when_queue_is_full(self) -> None:
        queue = LatestVisionQueue()

        queue.publish(make_frame(1))
        queue.publish(make_frame(2))
        queue.publish(make_frame(3))

        latest = queue.read_latest(timeout=0)

        self.assertEqual(latest.captured.frame_id, 3)

    def test_read_nowait_latest_returns_none_when_empty(self) -> None:
        queue = LatestVisionQueue()

        self.assertIsNone(queue.read_nowait_latest())

    def test_read_nowait_latest_drains_to_latest_frame(self) -> None:
        queue = LatestVisionQueue()

        queue.publish(make_frame(1))
        queue.publish(make_frame(2))
        latest = queue.read_nowait_latest()

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.captured.frame_id, 2)
        self.assertIsNone(queue.read_nowait_latest())


class ClosingInferencer:
    def __init__(self) -> None:
        self.closed = False

    @property
    def providers(self) -> list[str]:
        return ["closing-fake"]

    def predict(self, frame_bgr: object) -> list[dict[str, float | str]]:
        return [{"label": "target_center", "x": 0.5, "y": 0.5, "confidence": 1.0}]

    def close(self) -> None:
        self.closed = True


class OneFrameCapture:
    def read_frame(self) -> object:
        return SimpleNamespace(frame_id=1, frame_bgr=object())


class VisionProducerTest(unittest.TestCase):
    def test_stop_closes_inferencer_on_producer_thread(self) -> None:
        inferencer = ClosingInferencer()
        producer = VisionProducer(capture=OneFrameCapture(), inferencer=inferencer)

        producer.start()
        producer.read_latest(timeout=1.0)
        producer.stop()
        deadline = time.monotonic() + 1.0
        while not inferencer.closed and time.monotonic() < deadline:
            time.sleep(0.001)

        self.assertTrue(inferencer.closed)


if __name__ == "__main__":
    unittest.main()
