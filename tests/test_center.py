from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from twopoint_project.vision2.center import TargetDetectionResult, detect_target_center
from twopoint_project.vision2.infer_traditional import TraditionalVisionInferencer


@dataclass(frozen=True)
class CenterVideoDebugResult:
    output_video_path: Path
    output_points_path: Path
    frame_count: int
    found_count: int


def make_target_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    x0, y0, x1, y1 = 180, 140, 460, 350
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), 18)
    cv2.rectangle(frame, (x0 + 18, y0 + 18), (x1 - 18, y1 - 18), (245, 245, 245), -1)
    cv2.circle(frame, ((x0 + x1) // 2, (y0 + y1) // 2), 38, (0, 0, 180), 3)
    return frame


def write_synthetic_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to create test video: {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def annotate_center_frame(
    frame: np.ndarray,
    result: TargetDetectionResult,
    frame_index: int,
) -> np.ndarray:
    annotated = frame.copy()
    color = (0, 220, 0) if result.found else (0, 0, 255)
    if result.quad is not None:
        quad = result.quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [quad], isClosed=True, color=color, thickness=2)
    if result.center is not None:
        cx, cy = int(round(result.center[0])), int(round(result.center[1]))
        cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), thickness=-1)
        point_text = f"center=({cx},{cy})"
    else:
        point_text = "center=(none)"
    cv2.putText(
        annotated,
        f"frame={frame_index} found={int(result.found)} {point_text}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )
    return annotated


def process_center_video(video_path: Path, output_dir: Path) -> CenterVideoDebugResult:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)

    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = output_dir / "center_debug.avi"
    output_points_path = output_dir / "center_debug.csv"
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"failed to create output video: {output_video_path}")

    frame_count = 0
    found_count = 0
    try:
        with output_points_path.open("w", newline="", encoding="utf-8") as points_file:
            csv_writer = csv.writer(points_file)
            csv_writer.writerow(["frame", "found", "x", "y", "confidence", "reason"])
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = detect_target_center(frame)
                writer.write(annotate_center_frame(frame, result, frame_count))

                x = y = ""
                if result.found and result.center is not None:
                    found_count += 1
                    x = f"{result.center[0]:.2f}"
                    y = f"{result.center[1]:.2f}"
                csv_writer.writerow(
                    [
                        frame_count,
                        int(result.found),
                        x,
                        y,
                        f"{result.confidence:.4f}",
                        result.reason,
                    ]
                )
                frame_count += 1
    finally:
        capture.release()
        writer.release()

    return CenterVideoDebugResult(
        output_video_path=output_video_path,
        output_points_path=output_points_path,
        frame_count=frame_count,
        found_count=found_count,
    )


class CenterVision2Test(unittest.TestCase):
    def test_detect_target_center_on_synthetic_frame(self) -> None:
        result = detect_target_center(make_target_frame())

        self.assertTrue(result.found)
        self.assertIsNotNone(result.center)
        assert result.center is not None
        self.assertAlmostEqual(result.center[0], 320.0, delta=4.0)
        self.assertAlmostEqual(result.center[1], 245.0, delta=4.0)

    def test_traditional_inferencer_outputs_normalized_target_center(self) -> None:
        points = TraditionalVisionInferencer().predict(make_target_frame())

        self.assertEqual([point["label"] for point in points], ["target_center", "laser_point"])
        target = points[0]
        laser = points[1]
        self.assertGreater(target["confidence"], 0.5)
        self.assertAlmostEqual(target["x"], 320.0 / 639.0, delta=0.02)
        self.assertAlmostEqual(target["y"], 245.0 / 479.0, delta=0.02)
        self.assertEqual(laser["confidence"], 0.0)

    def test_center_video_debug_writes_annotated_video_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "input.avi"
            write_synthetic_video(video_path, [make_target_frame() for _ in range(3)])

            result = process_center_video(video_path, tmp_path)

            self.assertEqual(result.frame_count, 3)
            self.assertEqual(result.found_count, 3)
            self.assertTrue(result.output_video_path.is_file())
            self.assertGreater(result.output_video_path.stat().st_size, 0)
            with result.output_points_path.open(encoding="utf-8") as points_file:
                rows = list(csv.reader(points_file))
            self.assertEqual(rows[0], ["frame", "found", "x", "y", "confidence", "reason"])
            self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
