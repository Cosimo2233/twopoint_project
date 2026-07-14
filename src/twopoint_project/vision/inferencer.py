from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict


DEFAULT_VISION_BACKEND = "traditional"
DEFAULT_ONNX_PATH = "model-bin/runs/twopoint/best.onnx"
DEFAULT_IMG_SIZE = 640


class PointPrediction(TypedDict):
    label: str
    x: float
    y: float
    confidence: float


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
    img_size: int = DEFAULT_IMG_SIZE,
) -> VisionInferencer:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "traditional":
        from twopoint_project.vision2.infer_traditional import TraditionalVisionInferencer

        return TraditionalVisionInferencer()
    if normalized_backend == "onnx":
        from twopoint_project.vision.infer_twopoint_onnx import TwoPointOnnxInferencer

        return TwoPointOnnxInferencer(Path(onnx_path), img_size=img_size)
    raise ValueError(
        f"unsupported vision backend: {backend!r}; expected 'traditional' or 'onnx'"
    )
