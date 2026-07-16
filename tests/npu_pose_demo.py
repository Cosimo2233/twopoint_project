from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from twopoint_project.vision3.infer_npu_pose import NpuPoseInferencer, restore_detection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A733 YOLO11n-Pose model on one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("model-bin/pose/best_pcq_a733.nb"))
    parser.add_argument("--library", type=Path, default=Path("build/npu/libyolo11_pose_npu.so"))
    parser.add_argument("--output", type=Path, default=Path("outputs/npu_pose_demo.jpg"))
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--target-keypoint-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"failed to read image: {args.image}")

    inferencer = NpuPoseInferencer(
        args.model,
        args.library,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        target_keypoint_index=args.target_keypoint_index,
    )
    try:
        detections, meta = inferencer.predict_detections(frame)
        restored = [restore_detection(detection, meta) for detection in detections]
        print(f"providers={inferencer.providers}")
        print(f"detections={len(restored)}")
        for detection_index, detection in enumerate(restored):
            x1, y1, x2, y2 = detection.box
            cv2.rectangle(
                frame,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"target {detection.score:.3f}",
                (int(round(x1)), max(int(round(y1)) - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            print(f"detection[{detection_index}] score={detection.score:.5f} box={detection.box}")
            for keypoint_index, keypoint in enumerate(detection.keypoints):
                color = (0, 255, 0) if keypoint_index == args.target_keypoint_index else (0, 165, 255)
                point = (int(round(keypoint.x)), int(round(keypoint.y)))
                cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"k{keypoint_index}:{keypoint.confidence:.2f}",
                    (point[0] + 5, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                print(
                    f"  kpt[{keypoint_index}] x={keypoint.x:.2f} "
                    f"y={keypoint.y:.2f} confidence={keypoint.confidence:.5f}"
                )
    finally:
        inferencer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), frame):
        raise RuntimeError(f"failed to write output image: {args.output}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
