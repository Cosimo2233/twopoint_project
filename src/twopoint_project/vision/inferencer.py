from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict


DEFAULT_VISION_BACKEND = "traditional"
DEFAULT_ONNX_PATH = "model-bin/best2.onnx"
DEFAULT_NPU_MODEL_PATH = "model-bin/pose/best_pcq_a733.nb"
DEFAULT_NPU_LIBRARY_PATH = "build/npu/libyolo11_pose_npu.so"
DEFAULT_IMG_SIZE = 640
DEFAULT_NPU_SCORE_THRESHOLD = 0.4
DEFAULT_NPU_NMS_THRESHOLD = 0.45
DEFAULT_NPU_TARGET_KEYPOINT_INDEX = 0


class PointPrediction(TypedDict):
    label: str
    x: float
    y: float
    confidence: float


TargetCorners = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class VisionInferenceDetails:
    """One frame's inference outputs and optional target measurements."""

    points: list[PointPrediction]
    detections: tuple[Any, ...] = ()
    target_area_normalized: float | None = None
    target_distance_cm: float | None = None
    target_corners_normalized: TargetCorners = ()


class VisionInferencer(Protocol):
    @property
    def providers(self) -> list[str]:
        ...

    def predict(self, frame_bgr: Any) -> list[PointPrediction]:
        ...


def build_vision_inferencer(
    *,
    backend: str,
    onnx_path: str | Path = DEFAULT_ONNX_PATH,
    npu_model_path: str | Path = DEFAULT_NPU_MODEL_PATH,
    npu_library_path: str | Path = DEFAULT_NPU_LIBRARY_PATH,
    img_size: int = DEFAULT_IMG_SIZE,
    npu_score_threshold: float = DEFAULT_NPU_SCORE_THRESHOLD,
    npu_nms_threshold: float = DEFAULT_NPU_NMS_THRESHOLD,
    npu_target_keypoint_index: int = DEFAULT_NPU_TARGET_KEYPOINT_INDEX,
) -> VisionInferencer:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "traditional":
        from twopoint_project.vision2.infer_traditional import TraditionalVisionInferencer

        return TraditionalVisionInferencer()
    if normalized_backend == "onnx":
        from twopoint_project.vision.infer_twopoint_onnx import TwoPointOnnxInferencer

        return TwoPointOnnxInferencer(Path(onnx_path), img_size=img_size)
    if normalized_backend in {"npu", "npu_pose", "a733_npu"}:
        from twopoint_project.vision3.infer_npu_pose import NpuPoseInferencer

        return NpuPoseInferencer(
            Path(npu_model_path),
            Path(npu_library_path),
            img_size=img_size,
            score_threshold=npu_score_threshold,
            nms_threshold=npu_nms_threshold,
            target_keypoint_index=npu_target_keypoint_index,
        )
    raise ValueError(
        f"unsupported vision backend: {backend!r}; expected 'traditional', 'onnx', or 'npu_pose'"
    )
