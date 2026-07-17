from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
import threading
import time
import unittest

import numpy as np

from twopoint_project.vision.pipeline import LatestVisionQueue, VisionFrame, VisionProducer
from twopoint_project.vision.inferencer import VisionInferenceDetails


def make_frame(frame_id: int) -> VisionFrame:
    return VisionFrame(
        camera_id="test-camera",
        stream_generation=1,
        source_frame_id=frame_id,
        captured_at_monotonic_ns=1,
        inference_started_monotonic_ns=2,
        inference_finished_monotonic_ns=3,
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


class DetailedInferencer(ClosingInferencer):
    def predict_with_details(self, frame_bgr: object) -> VisionInferenceDetails:
        return VisionInferenceDetails(
            points=[{"label": "target_center", "x": 0.5, "y": 0.5, "confidence": 1.0}],
            detections=("target",),
            target_area_normalized=0.02,
            target_distance_cm=142.0,
        )


class LegacyDetailedInferencer(ClosingInferencer):
    def predict_with_details(
        self,
        frame_bgr: object,
    ) -> tuple[list[dict[str, float | str]], tuple[str, ...]]:
        return self.predict(frame_bgr), ("legacy-target",)


class SequencedDetailedInferencer(ClosingInferencer):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.release_second = threading.Event()

    def predict_with_details(self, frame_bgr: object) -> VisionInferenceDetails:
        self.call_count += 1
        if self.call_count == 1:
            return VisionInferenceDetails(
                points=self.predict(frame_bgr),
                target_area_normalized=0.02,
                target_distance_cm=142.0,
            )
        self.release_second.wait(timeout=1.0)
        return VisionInferenceDetails(points=self.predict(frame_bgr))


class OneFrameCapture:
    def __init__(self) -> None:
        self.frame_id = 0

    def read_frame(self, timeout: float | None = None) -> object:
        self.frame_id += 1
        return SimpleNamespace(
            camera_id="test-camera",
            stream_generation=1,
            frame_id=self.frame_id,
            captured_at_monotonic_ns=time.monotonic_ns(),
            frame_bgr=object(),
        )


class ImageFrameCapture(OneFrameCapture):
    def __init__(self) -> None:
        super().__init__()
        self.frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)

    def read_frame(self, timeout: float | None = None) -> object:
        captured = super().read_frame(timeout=timeout)
        captured.frame_bgr = self.frame_bgr
        return captured


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

    def test_producer_publishes_distance_with_same_frame(self) -> None:
        producer = VisionProducer(capture=ImageFrameCapture(), inferencer=DetailedInferencer())

        producer.start()
        result = producer.read_latest(timeout=1.0)
        producer.stop()

        self.assertEqual(result.source_frame_id, result.captured.frame_id)
        self.assertEqual(result.detections, ("target",))
        self.assertEqual(result.target_area_normalized, 0.02)
        self.assertEqual(result.target_distance_cm, 142.0)
        self.assertIsNotNone(result.laser_area_prediction)
        self.assertEqual(
            [point["label"] for point in result.points],
            ["target_center", "laser_point"],
        )
        assert result.laser_area_prediction is not None
        self.assertAlmostEqual(
            result.points[1]["x"],
            result.laser_area_prediction.x,
        )
        self.assertAlmostEqual(
            result.points[1]["y"],
            result.laser_area_prediction.y,
        )

    def test_legacy_details_default_distance_to_none(self) -> None:
        producer = VisionProducer(
            capture=OneFrameCapture(),
            inferencer=LegacyDetailedInferencer(),
        )

        producer.start()
        result = producer.read_latest(timeout=1.0)
        producer.stop()

        self.assertEqual(result.detections, ("legacy-target",))
        self.assertIsNone(result.target_area_normalized)
        self.assertIsNone(result.target_distance_cm)

    def test_invalid_frame_does_not_reuse_previous_distance(self) -> None:
        inferencer = SequencedDetailedInferencer()
        producer = VisionProducer(capture=ImageFrameCapture(), inferencer=inferencer)

        producer.start()
        valid = producer.read_latest(timeout=1.0)
        inferencer.release_second.set()
        invalid = producer.read_latest(timeout=1.0)
        producer.stop()

        self.assertEqual(valid.target_distance_cm, 142.0)
        self.assertIsNotNone(valid.laser_area_prediction)
        self.assertGreater(invalid.source_frame_id, valid.source_frame_id)
        self.assertIsNone(invalid.target_area_normalized)
        self.assertIsNone(invalid.target_distance_cm)
        self.assertIsNone(invalid.laser_area_prediction)
        self.assertEqual(
            [point["label"] for point in invalid.points],
            ["target_center"],
        )


if __name__ == "__main__":
    unittest.main()
