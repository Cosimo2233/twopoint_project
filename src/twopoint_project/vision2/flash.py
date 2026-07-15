from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from twopoint_project.vision2.center import (
    DetectionConfig,
    TargetDetectionResult,
    detect_target_center,
)


Point = tuple[float, float]


@dataclass(frozen=True)
class FlashConfig:
    warp_width: int = 420
    inner_margin_ratio: float = 0.10
    hue_min: int = 105
    hue_max: int = 165
    min_saturation: int = 45
    saturation_delta: int = 18
    min_value: int = 60
    value_delta: int = -5
    min_area: float = 1.0
    max_area: float = 260.0
    max_blob_width: int = 34
    max_blob_height: int = 34
    min_confidence: float = 0.38


@dataclass(frozen=True)
class FlashDetectionResult:
    found: bool
    point: Point | None
    normalized_point: Point | None
    confidence: float
    area: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "found": self.found,
            "point": _round_point(self.point),
            "normalized_point": _round_point(self.normalized_point),
            "confidence": round(self.confidence, 4),
            "area": round(self.area, 3),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FlashFrameResult:
    target: TargetDetectionResult
    flash: FlashDetectionResult

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target.to_dict(include_candidates=False),
            "flash": self.flash.to_dict(),
        }


def detect_flash_point(
    image: str | Path | np.ndarray,
    target_result: TargetDetectionResult | None = None,
    flash_config: FlashConfig | None = None,
    target_config: DetectionConfig | None = None,
) -> FlashFrameResult:
    frame = _read_image(image)
    cfg = flash_config or FlashConfig()
    target = target_result or detect_target_center(frame, target_config)

    if not target.found or target.quad is None:
        return FlashFrameResult(
            target=target,
            flash=FlashDetectionResult(
                found=False,
                point=None,
                normalized_point=None,
                confidence=0.0,
                area=0.0,
                reason="no_target",
            ),
        )

    flash = _detect_flash_in_target(frame, target.quad, cfg)
    return FlashFrameResult(target=target, flash=flash)


def _detect_flash_in_target(
    image: np.ndarray,
    quad: np.ndarray,
    cfg: FlashConfig,
) -> FlashDetectionResult:
    ordered_quad = _order_quad_points(quad)
    width, height = _quad_dimensions(ordered_quad)
    aspect = max(width, height) / max(1.0, min(width, height))
    warp_width = cfg.warp_width
    warp_height = max(80, int(round(warp_width / aspect)))
    if height > width:
        warp_width, warp_height = warp_height, warp_width

    warped, to_image = _warp_quad_with_inverse(
        image,
        ordered_quad,
        warp_width,
        warp_height,
    )
    if warped.size == 0:
        return _flash_not_found("empty_warp")

    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    h, w = warped.shape[:2]
    inner_mask = _inner_mask(h, w, cfg.inner_margin_ratio)
    if not np.any(inner_mask):
        return _flash_not_found("empty_inner_region")

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    sat_base = float(np.median(sat[inner_mask]))
    val_base = float(np.median(val[inner_mask]))
    sat_threshold = max(float(cfg.min_saturation), sat_base + cfg.saturation_delta)
    val_threshold = max(float(cfg.min_value), val_base + cfg.value_delta)

    hue_mask = (hsv[:, :, 0] >= cfg.hue_min) & (hsv[:, :, 0] <= cfg.hue_max)
    mask = (
        hue_mask
        & inner_mask
        & (sat.astype(np.float32) >= sat_threshold)
        & (val.astype(np.float32) >= val_threshold)
    )
    mask_u8 = (mask.astype(np.uint8)) * 255
    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        candidate = _score_flash_contour(
            contour,
            hsv,
            sat_threshold,
            val_threshold,
            cfg,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return _flash_not_found("no_flash_candidate")

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    best = candidates[0]
    if best["confidence"] < cfg.min_confidence:
        return FlashDetectionResult(
            found=False,
            point=None,
            normalized_point=None,
            confidence=float(best["confidence"]),
            area=float(best["area"]),
            reason="low_confidence",
        )

    warp_point = np.array([[[best["x"], best["y"]]]], dtype=np.float32)
    image_point = cv2.perspectiveTransform(warp_point, to_image).reshape(2)
    normalized = (
        float(best["x"] / max(1, w - 1)),
        float(best["y"] / max(1, h - 1)),
    )
    return FlashDetectionResult(
        found=True,
        point=(float(image_point[0]), float(image_point[1])),
        normalized_point=normalized,
        confidence=float(best["confidence"]),
        area=float(best["area"]),
        reason="ok",
    )


def _score_flash_contour(
    contour: np.ndarray,
    hsv: np.ndarray,
    sat_threshold: float,
    val_threshold: float,
    cfg: FlashConfig,
) -> dict[str, float] | None:
    area = float(cv2.contourArea(contour))
    if area < cfg.min_area or area > cfg.max_area:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    if w > cfg.max_blob_width or h > cfg.max_blob_height:
        return None

    perimeter = cv2.arcLength(contour, True)
    circularity = 1.0
    if perimeter > 0:
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))

    contour_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
    selected = contour_mask.astype(bool)
    if not np.any(selected):
        return None

    sat_mean = float(np.mean(hsv[:, :, 1][selected]))
    val_mean = float(np.mean(hsv[:, :, 2][selected]))
    sat_score = _clip01((sat_mean - sat_threshold) / 90.0)
    val_score = _clip01((val_mean - val_threshold) / 90.0)
    area_score = _clip01((area - cfg.min_area) / 16.0)
    shape_score = _clip01(circularity / 0.75)
    confidence = (
        0.34 * sat_score
        + 0.30 * val_score
        + 0.20 * shape_score
        + 0.16 * area_score
    )

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        cx = x + w / 2.0
        cy = y + h / 2.0
    else:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

    return {
        "x": float(cx),
        "y": float(cy),
        "confidence": float(_clip01(confidence)),
        "area": area,
    }


def _flash_not_found(reason: str) -> FlashDetectionResult:
    return FlashDetectionResult(
        found=False,
        point=None,
        normalized_point=None,
        confidence=0.0,
        area=0.0,
        reason=reason,
    )


def _warp_quad_with_inverse(
    image: np.ndarray,
    quad: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    if width <= 0 or height <= 0:
        return np.empty((0, 0, 3), dtype=np.uint8), np.eye(3, dtype=np.float32)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    to_warp = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    to_image = cv2.getPerspectiveTransform(dst, quad.astype(np.float32))
    warped = cv2.warpPerspective(image, to_warp, (width, height))
    return warped, to_image


def _inner_mask(h: int, w: int, margin_ratio: float) -> np.ndarray:
    margin = max(2, int(round(min(h, w) * margin_ratio)))
    mask = np.zeros((h, w), dtype=bool)
    mask[margin : h - margin, margin : w - margin] = True
    if not np.any(mask):
        mask[:, :] = True
    return mask


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if cv2.contourArea(ordered) < 0:
        ordered[[1, 3]] = ordered[[3, 1]]
    return ordered.astype(np.float32)


def _quad_dimensions(quad: np.ndarray) -> tuple[float, float]:
    top = np.linalg.norm(quad[1] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    return float(max(top, bottom)), float(max(left, right))


def _read_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image ndarray must be a BGR image with 3 channels")
        return image

    path = Path(image)
    loaded = cv2.imread(str(path))
    if loaded is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return loaded


def _round_point(point: Point | None) -> list[float] | None:
    if point is None:
        return None
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
