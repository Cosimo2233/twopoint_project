from __future__ import annotations

import numpy as np

from twopoint_project.vision2.center import DetectionConfig
from twopoint_project.vision2.flash import FlashConfig, detect_flash_point
from twopoint_project.vision2.points import (
    POINT_NAMES,
    PointPrediction,
    normalized_point_prediction,
)


class TraditionalVisionInferencer:
    def __init__(
        self,
        *,
        target_config: DetectionConfig | None = None,
        flash_config: FlashConfig | None = None,
    ) -> None:
        self.target_config = target_config
        self.flash_config = flash_config

    @property
    def providers(self) -> list[str]:
        return ["opencv-traditional"]

    def predict(self, frame_bgr: np.ndarray) -> list[PointPrediction]:
        result = detect_flash_point(
            frame_bgr,
            flash_config=self.flash_config,
            target_config=self.target_config,
        )
        image_height, image_width = frame_bgr.shape[:2]

        return [
            normalized_point_prediction(
                POINT_NAMES[0],
                result.target.center if result.target.found else None,
                result.target.confidence if result.target.found else 0.0,
                image_width,
                image_height,
            ),
            normalized_point_prediction(
                POINT_NAMES[1],
                result.flash.point if result.flash.found else None,
                result.flash.confidence if result.flash.found else 0.0,
                image_width,
                image_height,
            ),
        ]
