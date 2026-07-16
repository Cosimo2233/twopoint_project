from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from twopoint_project.vision.inferencer import build_vision_inferencer
from twopoint_project.vision3.infer_npu_pose import (
    LetterboxMeta,
    NativePoseDetection,
    NpuPoseInferencer,
    PoseDetection,
    PoseKeypoint,
    letterbox_rgb_uint8,
    restore_detection,
    target_center_prediction,
)


class NpuPoseTest(unittest.TestCase):
    def test_letterbox_keeps_uint8_hwc_and_converts_bgr_to_rgb(self) -> None:
        frame = np.zeros((320, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 10
        frame[:, :, 1] = 20
        frame[:, :, 2] = 30

        tensor, meta = letterbox_rgb_uint8(frame)

        self.assertEqual(tensor.shape, (640, 640, 3))
        self.assertEqual(tensor.dtype, np.uint8)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(tuple(tensor[160, 0]), (30, 20, 10))
        self.assertEqual(tuple(tensor[0, 0]), (114, 114, 114))
        self.assertEqual(meta, LetterboxMeta(1.0, 0, 160, 640, 320))

    def test_restore_detection_removes_letterbox_padding(self) -> None:
        meta = LetterboxMeta(scale=0.5, left=0, top=140, original_width=1280, original_height=720)
        keypoints = tuple(PoseKeypoint(320.0, 320.0, 0.9) for _ in range(5))
        detection = PoseDetection((100.0, 190.0, 500.0, 590.0), 0.8, keypoints)

        restored = restore_detection(detection, meta)

        self.assertEqual(restored.box, (200.0, 100.0, 1000.0, 719.0))
        self.assertEqual(restored.keypoints[0], PoseKeypoint(640.0, 360.0, 0.9))

    def test_target_center_uses_configured_keypoint(self) -> None:
        meta = LetterboxMeta(scale=1.0, left=0, top=0, original_width=640, original_height=640)
        keypoints = tuple(
            PoseKeypoint(float(index * 100), float(index * 50), 0.95 - index * 0.05)
            for index in range(5)
        )
        detection = PoseDetection((0.0, 0.0, 639.0, 639.0), 0.85, keypoints)

        prediction = target_center_prediction([detection], meta, 2)

        self.assertEqual(prediction["label"], "target_center")
        self.assertAlmostEqual(prediction["x"], 200.0 / 639.0)
        self.assertAlmostEqual(prediction["y"], 100.0 / 639.0)
        self.assertAlmostEqual(prediction["confidence"], 0.85)

    def test_no_detection_returns_zero_confidence_target(self) -> None:
        meta = LetterboxMeta(scale=1.0, left=0, top=0, original_width=640, original_height=640)

        prediction = target_center_prediction([], meta, 0)

        self.assertEqual(
            prediction,
            {"label": "target_center", "x": 0.0, "y": 0.0, "confidence": 0.0},
        )

    def test_native_structure_matches_c_abi(self) -> None:
        self.assertEqual(NativePoseDetection.keypoints.size, 15 * 4)
        self.assertEqual(NativePoseDetection.score.offset, 4 * 4)
        self.assertEqual(NativePoseDetection.keypoints.offset, 5 * 4)

    def test_factory_builds_npu_pose_backend_without_loading_hardware(self) -> None:
        inferencer = build_vision_inferencer(
            backend="npu_pose",
            npu_model_path=Path("model-bin/pose/test.nb"),
            npu_library_path=Path("build/npu/test.so"),
            img_size=640,
            npu_score_threshold=0.25,
            npu_nms_threshold=0.5,
            npu_target_keypoint_index=3,
        )

        self.assertIsInstance(inferencer, NpuPoseInferencer)
        self.assertEqual(inferencer.target_keypoint_index, 3)
        self.assertEqual(inferencer.score_threshold, 0.25)


if __name__ == "__main__":
    unittest.main()
