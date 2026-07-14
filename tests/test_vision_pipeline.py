from __future__ import annotations

from argparse import Namespace
import unittest

from twopoint_project.vision.pipeline import LatestVisionQueue, VisionFrame


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


if __name__ == "__main__":
    unittest.main()
