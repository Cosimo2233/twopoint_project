from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


Point = tuple[float, float]


@dataclass(frozen=True)
class DetectionConfig:
    """Tunable parameters for traditional OpenCV target-paper detection."""

    max_process_size: int = 960
    expected_aspect: float = 4.0 / 3.0
    aspect_tolerance: float = 0.85
    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.28
    dark_v_max: int = 115
    white_v_min: int = 105
    white_s_max: int = 120
    min_confidence: float = 0.42
    debug_warp_width: int = 360
    min_border_contrast: float = 55.0
    min_good_border_sides: int = 3
    video_confirm_frames: int = 3
    video_max_center_jump_ratio: float = 0.18


@dataclass(frozen=True)
class TargetCandidate:
    quad: np.ndarray
    center: Point
    confidence: float
    area_ratio: float
    aspect: float
    scores: dict[str, float]


@dataclass(frozen=True)
class TargetDetectionResult:
    found: bool
    center: Point | None
    quad: np.ndarray | None
    confidence: float
    candidates: list[TargetCandidate]
    image_shape: tuple[int, int]
    reason: str

    def to_dict(self, include_candidates: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "found": self.found,
            "center": _round_point(self.center),
            "quad": _round_quad(self.quad),
            "confidence": round(self.confidence, 4),
            "image_shape": self.image_shape,
            "reason": self.reason,
        }
        if include_candidates:
            payload["candidates"] = [
                {
                    "center": _round_point(candidate.center),
                    "quad": _round_quad(candidate.quad),
                    "confidence": round(candidate.confidence, 4),
                    "area_ratio": round(candidate.area_ratio, 6),
                    "aspect": round(candidate.aspect, 4),
                    "scores": {
                        key: round(value, 4)
                        for key, value in candidate.scores.items()
                    },
                }
                for candidate in self.candidates
            ]
        return payload


def detect_target_center(
    image: str | Path | np.ndarray,
    config: DetectionConfig | None = None,
) -> TargetDetectionResult:
    """Detect the target-paper center.

    The detector uses the black rectangular paper frame as the primary signal.
    Red concentric rings are treated as a validation score because they are thin
    and often weak in distant or reflective images.
    """

    cfg = config or DetectionConfig()
    original = _read_image(image)
    process, scale = _resize_for_processing(original, cfg.max_process_size)

    candidates = _detect_candidates(process, cfg)
    candidates = [_scale_candidate(candidate, 1.0 / scale) for candidate in candidates]
    candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)

    if not candidates:
        return TargetDetectionResult(
            found=False,
            center=None,
            quad=None,
            confidence=0.0,
            candidates=[],
            image_shape=original.shape[:2],
            reason="no_target_candidate",
        )

    best = candidates[0]
    found = best.confidence >= cfg.min_confidence
    return TargetDetectionResult(
        found=found,
        center=best.center if found else None,
        quad=best.quad if found else None,
        confidence=best.confidence,
        candidates=candidates,
        image_shape=original.shape[:2],
        reason="ok" if found else "low_confidence",
    )

def _detect_candidates(image: np.ndarray, cfg: DetectionConfig) -> list[TargetCandidate]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dark_by_hsv = hsv[:, :, 2] <= cfg.dark_v_max
    dark_by_gray = gray <= min(105, cfg.dark_v_max)
    dark_mask = np.where(dark_by_hsv | dark_by_gray, 255, 0).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(
        dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = float(image.shape[0] * image.shape[1])
    candidates: list[TargetCandidate] = []
    seen: list[np.ndarray] = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area <= 0:
            continue

        area_ratio = contour_area / image_area
        if area_ratio < cfg.min_area_ratio or area_ratio > cfg.max_area_ratio:
            continue

        quad_options = _quad_options_from_contour(contour)
        for quad in quad_options:
            quad = order_quad_points(quad.astype(np.float32))
            if _is_duplicate_quad(quad, seen):
                continue
            candidate = _score_quad_candidate(image, quad, area_ratio, cfg)
            if candidate is None:
                continue
            seen.append(quad)
            candidates.append(candidate)

    return candidates


def _quad_options_from_contour(contour: np.ndarray) -> list[np.ndarray]:
    options: list[np.ndarray] = []

    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        for epsilon_ratio in (0.025, 0.04, 0.065):
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                options.append(approx.reshape(4, 2).astype(np.float32))

    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    if width > 2 and height > 2:
        options.append(cv2.boxPoints(rect).astype(np.float32))

    return options


def _score_quad_candidate(
    image: np.ndarray,
    quad: np.ndarray,
    source_area_ratio: float,
    cfg: DetectionConfig,
) -> TargetCandidate | None:
    quad_area = abs(cv2.contourArea(quad.astype(np.float32)))
    image_area = float(image.shape[0] * image.shape[1])
    area_ratio = quad_area / image_area
    if area_ratio < cfg.min_area_ratio or area_ratio > cfg.max_area_ratio:
        return None

    width, height = _quad_dimensions(quad)
    if min(width, height) < 12:
        return None
    aspect = max(width, height) / max(1.0, min(width, height))
    if not (1.02 <= aspect <= cfg.expected_aspect + cfg.aspect_tolerance):
        return None

    warp_width = cfg.debug_warp_width
    warp_height = max(80, int(round(warp_width / aspect)))
    if height > width:
        warp_width, warp_height = warp_height, warp_width

    warped = _warp_quad(image, quad, warp_width, warp_height)
    if warped.size == 0:
        return None

    scores = _score_warped_target(warped, aspect, cfg)
    geometry_score = _clip01(
        1.0 - abs(aspect - cfg.expected_aspect) / cfg.aspect_tolerance
    )
    area_score = _score_area(area_ratio)

    confidence = (
        0.22 * scores["border"]
        + 0.18 * scores["paper"]
        + 0.16 * geometry_score
        + 0.14 * scores["contrast"]
        + 0.12 * scores["side_completeness"]
        + 0.10 * scores["red_ring"]
        + 0.05 * scores["hollow"]
        + 0.03 * area_score
    )

    if scores["border"] < 0.18 or scores["paper"] < 0.18:
        confidence *= 0.65
    if scores["contrast"] < 0.28:
        confidence *= 0.55
    if scores["good_sides"] < cfg.min_good_border_sides:
        if (
            scores["red_ring"] >= 0.12
            and scores["contrast"] >= 0.72
            and scores["paper"] >= 0.70
            and scores["hollow"] >= 0.55
        ):
            confidence *= 0.75
        else:
            confidence *= 0.45
    if scores["hollow"] < 0.08:
        confidence *= 0.75
    if scores["red_ring"] < 0.12 and (
        scores["contrast"] < 0.72 or scores["side_completeness"] < 0.85
    ):
        confidence *= 0.72
    if source_area_ratio < cfg.min_area_ratio:
        confidence *= 0.5

    center = _project_normalized_point_to_quad(quad, 0.5, 0.5)
    return TargetCandidate(
        quad=quad,
        center=(float(center[0]), float(center[1])),
        confidence=float(_clip01(confidence)),
        area_ratio=float(area_ratio),
        aspect=float(aspect),
        scores={
            **scores,
            "geometry": float(geometry_score),
            "area": float(area_score),
        },
    )


def _score_warped_target(
    warped: np.ndarray,
    aspect: float,
    cfg: DetectionConfig,
) -> dict[str, float]:
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    h, w = warped.shape[:2]
    margin = max(5, int(round(min(w, h) * 0.09)))

    dark = hsv[:, :, 2] <= cfg.dark_v_max
    white_paper = (hsv[:, :, 2] >= cfg.white_v_min) & (hsv[:, :, 1] <= cfg.white_s_max)
    red = (
        ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168))
        & (hsv[:, :, 1] >= 35)
        & (hsv[:, :, 2] >= 65)
    )

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:margin, :] = True
    border_mask[-margin:, :] = True
    border_mask[:, :margin] = True
    border_mask[:, -margin:] = True
    side_masks = _border_side_masks(h, w, margin)

    inner_mask = np.zeros((h, w), dtype=bool)
    inner_margin = max(margin + 3, int(round(min(w, h) * 0.18)))
    inner_mask[inner_margin : h - inner_margin, inner_margin : w - inner_margin] = True
    if not np.any(inner_mask):
        inner_mask[margin : h - margin, margin : w - margin] = True

    border_dark_ratio = _mask_ratio(dark, border_mask)
    inner_dark_ratio = _mask_ratio(dark, inner_mask)
    paper_ratio = _mask_ratio(white_paper, inner_mask)
    red_ratio = _mask_ratio(red, inner_mask)
    value = hsv[:, :, 2].astype(np.float32)
    inner_median_v = _masked_median(value, inner_mask)
    border_dark_v = _masked_percentile(value, border_mask, 25.0)
    border_contrast = max(0.0, inner_median_v - border_dark_v)

    side_dark_ratios = [_mask_ratio(dark, side_mask) for side_mask in side_masks]
    side_contrasts = [
        max(0.0, inner_median_v - _masked_percentile(value, side_mask, 25.0))
        for side_mask in side_masks
    ]
    side_scores = [
        _clip01(
            0.55 * _smoothstep(0.16, 0.42, dark_ratio)
            + 0.45 * _smoothstep(
                cfg.min_border_contrast * 0.55,
                cfg.min_border_contrast * 1.35,
                contrast,
            )
        )
        for dark_ratio, contrast in zip(side_dark_ratios, side_contrasts)
    ]
    good_sides = sum(score >= 0.45 for score in side_scores)

    sorted_side_scores = sorted(side_scores, reverse=True)
    border_score = float(np.mean(sorted_side_scores[:3])) if sorted_side_scores else 0.0
    paper_score = _smoothstep(0.24, 0.70, paper_ratio)
    hollow_score = _clip01((border_dark_ratio - inner_dark_ratio + 0.12) / 0.52)
    contrast_score = _smoothstep(
        cfg.min_border_contrast * 0.65,
        cfg.min_border_contrast * 1.45,
        border_contrast,
    )
    side_completeness_score = _clip01(good_sides / 4.0)
    red_score = _score_red_ring(red, inner_mask, red_ratio, aspect)

    return {
        "border": float(border_score),
        "paper": float(paper_score),
        "hollow": float(hollow_score),
        "contrast": float(contrast_score),
        "side_completeness": float(side_completeness_score),
        "red_ring": float(red_score),
        "border_dark_ratio": float(border_dark_ratio),
        "inner_dark_ratio": float(inner_dark_ratio),
        "paper_ratio": float(paper_ratio),
        "red_ratio": float(red_ratio),
        "border_contrast": float(border_contrast),
        "good_sides": float(good_sides),
        "top_side": float(side_scores[0]),
        "right_side": float(side_scores[1]),
        "bottom_side": float(side_scores[2]),
        "left_side": float(side_scores[3]),
    }


def _border_side_masks(h: int, w: int, margin: int) -> list[np.ndarray]:
    top = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    bottom = np.zeros((h, w), dtype=bool)
    left = np.zeros((h, w), dtype=bool)

    top[:margin, :] = True
    right[:, w - margin :] = True
    bottom[h - margin :, :] = True
    left[:, :margin] = True
    return [top, right, bottom, left]


def _masked_median(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    if selected.size == 0:
        return 0.0
    return float(np.median(selected))


def _masked_percentile(values: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    selected = values[mask]
    if selected.size == 0:
        return 0.0
    return float(np.percentile(selected, percentile))


def _score_red_ring(
    red: np.ndarray,
    inner_mask: np.ndarray,
    red_ratio: float,
    aspect: float,
) -> float:
    if red_ratio <= 0:
        return 0.0

    ys, xs = np.where(red & inner_mask)
    if len(xs) < 8:
        return _smoothstep(0.002, 0.015, red_ratio) * 0.45

    h, w = red.shape[:2]
    center = np.array([w / 2.0, h / 2.0])
    points = np.column_stack([xs, ys]).astype(np.float32)
    distances = np.linalg.norm(points - center, axis=1)
    if distances.size == 0:
        return 0.0

    spread = float(np.percentile(distances, 90) - np.percentile(distances, 10))
    radius_scale = max(1.0, min(w, h) * 0.35)
    spread_score = _clip01(spread / radius_scale)
    density_score = _smoothstep(0.003, 0.035, red_ratio)
    return float(_clip01(0.65 * density_score + 0.35 * spread_score))


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    sums = ordered.sum(axis=1)
    start = int(np.argmin(sums))
    ordered = np.roll(ordered, -start, axis=0)

    if cv2.contourArea(ordered) < 0:
        ordered[[1, 3]] = ordered[[3, 1]]
    return ordered.astype(np.float32)


def _warp_quad(image: np.ndarray, quad: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def _project_normalized_point_to_quad(quad: np.ndarray, x: float, y: float) -> np.ndarray:
    width = 1000
    height = 1000
    src = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    point = np.array([[[x * width, y * height]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point, matrix)
    return projected.reshape(2)


def _quad_dimensions(quad: np.ndarray) -> tuple[float, float]:
    top = np.linalg.norm(quad[1] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    return float(max(top, bottom)), float(max(left, right))


def _resize_for_processing(
    image: np.ndarray,
    max_size: int,
) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return image.copy(), 1.0
    scale = max_size / float(longest)
    resized = cv2.resize(
        image,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


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


def _scale_candidate(candidate: TargetCandidate, factor: float) -> TargetCandidate:
    if abs(factor - 1.0) < 1e-9:
        return candidate
    quad = candidate.quad * factor
    center = (candidate.center[0] * factor, candidate.center[1] * factor)
    return TargetCandidate(
        quad=quad,
        center=center,
        confidence=candidate.confidence,
        area_ratio=candidate.area_ratio,
        aspect=candidate.aspect,
        scores=candidate.scores,
    )


def _is_duplicate_quad(quad: np.ndarray, seen: Iterable[np.ndarray]) -> bool:
    for other in seen:
        if np.linalg.norm(quad.mean(axis=0) - other.mean(axis=0)) < 4:
            return True
    return False


def _score_area(area_ratio: float) -> float:
    if area_ratio < 0.001:
        return 0.0
    if area_ratio <= 0.08:
        return _smoothstep(0.001, 0.006, area_ratio)
    return _clip01(1.0 - (area_ratio - 0.08) / 0.20)


def _mask_ratio(mask: np.ndarray, region: np.ndarray) -> float:
    count = int(np.count_nonzero(region))
    if count == 0:
        return 0.0
    return float(np.count_nonzero(mask & region) / count)


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    x = _clip01((value - edge0) / (edge1 - edge0))
    return x * x * (3.0 - 2.0 * x)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _round_point(point: Point | None) -> list[float] | None:
    if point is None:
        return None
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def _round_quad(quad: np.ndarray | None) -> list[list[float]] | None:
    if quad is None:
        return None
    return [[round(float(x), 2), round(float(y), 2)] for x, y in quad.reshape(4, 2)]

