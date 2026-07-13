from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from twopoint_project.vision2.flash import FlashFrameResult, detect_flash_point
from twopoint_project.vision2.infer_traditional import TraditionalVisionInferencer


@dataclass(frozen=True)
class FlashVideoDebugResult:
    output_video_path: Path
    output_points_path: Path
    frame_count: int
    target_found_count: int
    flash_found_count: int


def make_flash_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    x0, y0, x1, y1 = 180, 140, 460, 350
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), 18)
    cv2.rectangle(frame, (x0 + 18, y0 + 18), (x1 - 18, y1 - 18), (245, 245, 245), -1)
    cv2.circle(frame, ((x0 + x1) // 2, (y0 + y1) // 2), 38, (0, 0, 180), 3)
    cv2.circle(frame, (350, 260), 4, (255, 0, 0), -1)
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


def csv_point(point: tuple[float, float] | None) -> tuple[str, str]:
    if point is None:
        return "", ""
    return f"{point[0]:.2f}", f"{point[1]:.2f}"


def annotate_flash_frame(frame: np.ndarray, result: FlashFrameResult, frame_index: int) -> np.ndarray:
    annotated = frame.copy()
    if result.target.quad is not None:
        quad = result.target.quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [quad], isClosed=True, color=(0, 220, 0), thickness=2)
    if result.target.center is not None:
        tx, ty = int(round(result.target.center[0])), int(round(result.target.center[1]))
        cv2.circle(annotated, (tx, ty), 3, (0, 0, 255), thickness=-1)
    if result.flash.point is not None:
        fx, fy = int(round(result.flash.point[0])), int(round(result.flash.point[1]))
        cv2.circle(annotated, (fx, fy), 3, (255, 0, 0), thickness=-1)
    cv2.putText(
        annotated,
        (
            f"frame={frame_index} target={int(result.target.found)} "
            f"flash={int(result.flash.found)}"
        ),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return annotated


def process_flash_video(video_path: Path, output_dir: Path) -> FlashVideoDebugResult:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)

    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = output_dir / "flash_debug.avi"
    output_points_path = output_dir / "flash_debug.csv"
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
    target_found_count = 0
    flash_found_count = 0
    try:
        with output_points_path.open("w", newline="", encoding="utf-8") as points_file:
            csv_writer = csv.writer(points_file)
            csv_writer.writerow(
                [
                    "frame",
                    "target_found",
                    "target_x",
                    "target_y",
                    "target_confidence",
                    "flash_found",
                    "flash_x",
                    "flash_y",
                    "flash_u",
                    "flash_v",
                    "flash_confidence",
                    "flash_reason",
                ]
            )
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = detect_flash_point(frame)
                writer.write(annotate_flash_frame(frame, result, frame_count))
                if result.target.found:
                    target_found_count += 1
                if result.flash.found:
                    flash_found_count += 1

                target_x, target_y = csv_point(result.target.center)
                flash_x, flash_y = csv_point(result.flash.point)
                flash_u, flash_v = csv_point(result.flash.normalized_point)
                csv_writer.writerow(
                    [
                        frame_count,
                        int(result.target.found),
                        target_x,
                        target_y,
                        f"{result.target.confidence:.4f}",
                        int(result.flash.found),
                        flash_x,
                        flash_y,
                        flash_u,
                        flash_v,
                        f"{result.flash.confidence:.4f}",
                        result.flash.reason,
                    ]
                )
                frame_count += 1
    finally:
        capture.release()
        writer.release()

    return FlashVideoDebugResult(
        output_video_path=output_video_path,
        output_points_path=output_points_path,
        frame_count=frame_count,
        target_found_count=target_found_count,
        flash_found_count=flash_found_count,
    )


class FlashVision2Test(unittest.TestCase):
    def test_detect_flash_point_on_synthetic_frame(self) -> None:
        result = detect_flash_point(make_flash_frame())

        self.assertTrue(result.target.found)
        self.assertTrue(result.flash.found)
        self.assertIsNotNone(result.flash.point)
        assert result.flash.point is not None
        self.assertAlmostEqual(result.flash.point[0], 350.0, delta=5.0)
        self.assertAlmostEqual(result.flash.point[1], 260.0, delta=5.0)

    def test_traditional_inferencer_outputs_both_point_labels(self) -> None:
        points = TraditionalVisionInferencer().predict(make_flash_frame())

        self.assertEqual([point["label"] for point in points], ["target_center", "laser_point"])
        self.assertGreater(points[0]["confidence"], 0.5)
        self.assertGreater(points[1]["confidence"], 0.3)
        self.assertAlmostEqual(points[1]["x"], 350.0 / 639.0, delta=0.03)
        self.assertAlmostEqual(points[1]["y"], 260.0 / 479.0, delta=0.03)

    def test_flash_video_debug_writes_annotated_video_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "input.avi"
            write_synthetic_video(video_path, [make_flash_frame() for _ in range(3)])

            result = process_flash_video(video_path, tmp_path)

            self.assertEqual(result.frame_count, 3)
            self.assertEqual(result.target_found_count, 3)
            self.assertEqual(result.flash_found_count, 3)
            self.assertTrue(result.output_video_path.is_file())
            self.assertGreater(result.output_video_path.stat().st_size, 0)
            with result.output_points_path.open(encoding="utf-8") as points_file:
                rows = list(csv.reader(points_file))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0][0], "frame")
            self.assertEqual(rows[0][5], "flash_found")


if __name__ == "__main__":
    unittest.main()
