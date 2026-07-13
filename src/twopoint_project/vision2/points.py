from __future__ import annotations

from typing import TypedDict

import numpy as np


POINT_NAMES = ("target_center", "laser_point")


class PointPrediction(TypedDict):
    label: str
    x: float
    y: float
    confidence: float


def normalized_point_prediction(
    label: str,
    point: tuple[float, float] | None,
    confidence: float,
    image_width: int,
    image_height: int,
) -> PointPrediction:
    if point is None:
        return {
            "label": label,
            "x": 0.0,
            "y": 0.0,
            "confidence": 0.0,
        }

    width_scale = max(image_width - 1, 1)
    height_scale = max(image_height - 1, 1)
    return {
        "label": label,
        "x": float(np.clip(point[0] / width_scale, 0.0, 1.0)),
        "y": float(np.clip(point[1] / height_scale, 0.0, 1.0)),
        "confidence": float(confidence),
    }
