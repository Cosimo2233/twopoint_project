from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vision_test.center import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    DetectionConfig,
    TargetDetectionResult,
    detect_target_center,
)

logger = logging.getLogger(__name__)


Point = tuple[float, float]
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v"}


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
    draw_radius: int = 2
    hold_frames: int = 4
    smooth_alpha: float = 0.35


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


@dataclass(frozen=True)
class FlashVideoResult:
    input_path: Path
    output_video_path: Path
    output_points_path: Path
    frame_count: int
    target_found_count: int
    flash_found_count: int
    stopped_early: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "input": str(self.input_path),
            "output_video": str(self.output_video_path),
            "output_points": str(self.output_points_path),
            "frame_count": self.frame_count,
            "target_found_count": self.target_found_count,
            "flash_found_count": self.flash_found_count,
            "stopped_early": self.stopped_early,
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


def annotate_flash_detection(
    image: str | Path | np.ndarray,
    result: FlashFrameResult,
    flash_config: FlashConfig | None = None,
    *,
    draw_target: bool = True,
) -> np.ndarray:
    cfg = flash_config or FlashConfig()
    annotated = _read_image(image).copy()

    if draw_target and result.target.found and result.target.quad is not None:
        quad = result.target.quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [quad], isClosed=True, color=(0, 220, 0), thickness=2)
        if result.target.center is not None:
            tx, ty = _int_point(result.target.center)
            cv2.circle(annotated, (tx, ty), cfg.draw_radius, (0, 0, 255), thickness=-1)

    if result.flash.found and result.flash.point is not None:
        fx, fy = _int_point(result.flash.point)
        cv2.circle(annotated, (fx, fy), cfg.draw_radius, (255, 0, 0), thickness=-1)

    return annotated


def process_video(
    video_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    flash_config: FlashConfig | None = None,
    target_config: DetectionConfig | None = None,
    *,
    display: bool = True,
    window_name: str = "flash detection",
    output_name: str | None = None,
) -> FlashVideoResult:
    cfg = flash_config or FlashConfig()
    input_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"failed to open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"failed to read video size: {input_path}")

    output_video_path = _video_output_path(input_path, out_dir, output_name)
    output_points_path = output_video_path.with_suffix(".csv")
    writer = _create_video_writer(output_video_path, fps, (width, height))

    delay_ms = max(1, int(round(1000.0 / fps)))
    frame_count = 0
    target_found_count = 0
    flash_found_count = 0
    stopped_early = False
    last_valid_flash: FlashDetectionResult | None = None
    lost_flash_frames = 0

    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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

                result = detect_flash_point(
                    frame,
                    flash_config=cfg,
                    target_config=target_config,
                )
                result, last_valid_flash, lost_flash_frames = _stabilize_flash_output(
                    result,
                    last_valid_flash,
                    lost_flash_frames,
                    cfg,
                )
                annotated = annotate_flash_detection(frame, result, cfg)
                _draw_status(annotated, result, frame_count)
                writer.write(annotated)

                if result.target.found:
                    target_found_count += 1
                if result.flash.found:
                    flash_found_count += 1

                target_x, target_y = _csv_point(result.target.center)
                flash_x, flash_y = _csv_point(result.flash.point)
                flash_u, flash_v = _csv_point(result.flash.normalized_point)
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

                if display:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        stopped_early = True
                        frame_count += 1
                        break

                frame_count += 1
    finally:
        capture.release()
        writer.release()
        if display:
            cv2.destroyWindow(window_name)

    return FlashVideoResult(
        input_path=input_path,
        output_video_path=output_video_path,
        output_points_path=output_points_path,
        frame_count=frame_count,
        target_found_count=target_found_count,
        flash_found_count=flash_found_count,
        stopped_early=stopped_early,
    )


def _stabilize_flash_output(
    result: FlashFrameResult,
    last_valid_flash: FlashDetectionResult | None,
    lost_frames: int,
    cfg: FlashConfig,
) -> tuple[FlashFrameResult, FlashDetectionResult | None, int]:
    if not result.target.found:
        return result, None, 0

    if result.flash.found and result.flash.point is not None:
        current = result.flash
        if (
            last_valid_flash is not None
            and last_valid_flash.point is not None
            and cfg.smooth_alpha > 0
        ):
            current = _smooth_flash(last_valid_flash, current, cfg.smooth_alpha)
        return FlashFrameResult(result.target, current), current, 0

    if (
        last_valid_flash is not None
        and last_valid_flash.point is not None
        and lost_frames < cfg.hold_frames
    ):
        held = FlashDetectionResult(
            found=True,
            point=last_valid_flash.point,
            normalized_point=last_valid_flash.normalized_point,
            confidence=last_valid_flash.confidence,
            area=last_valid_flash.area,
            reason="temporal_hold",
        )
        return FlashFrameResult(result.target, held), last_valid_flash, lost_frames + 1

    return result, None, 0


def _smooth_flash(
    previous: FlashDetectionResult,
    current: FlashDetectionResult,
    alpha: float,
) -> FlashDetectionResult:
    if previous.point is None or current.point is None:
        return current

    px, py = previous.point
    cx, cy = current.point
    point = (
        (1.0 - alpha) * px + alpha * cx,
        (1.0 - alpha) * py + alpha * cy,
    )

    normalized_point = current.normalized_point
    if previous.normalized_point is not None and current.normalized_point is not None:
        pu, pv = previous.normalized_point
        cu, cv = current.normalized_point
        normalized_point = (
            (1.0 - alpha) * pu + alpha * cu,
            (1.0 - alpha) * pv + alpha * cv,
        )

    return FlashDetectionResult(
        found=True,
        point=point,
        normalized_point=normalized_point,
        confidence=current.confidence,
        area=current.area,
        reason=current.reason,
    )


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


def _create_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    suffix = output_path.suffix.lower()
    codec = "XVID" if suffix == ".avi" else "mp4v"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to create output video: {output_path}")
    return writer


def _video_output_path(
    input_path: Path,
    output_dir: Path,
    output_name: str | None,
) -> Path:
    if output_name:
        name = Path(output_name)
        suffix = name.suffix or ".mp4"
        stem = name.stem
    else:
        suffix = ".avi" if input_path.suffix.lower() == ".avi" else ".mp4"
        stem = f"{input_path.stem}_flash"
    return output_dir / f"{stem}{suffix}"


def _draw_status(
    image: np.ndarray,
    result: FlashFrameResult,
    frame_index: int,
) -> None:
    flash_point = "none"
    if result.flash.found and result.flash.point is not None:
        x, y = _int_point(result.flash.point)
        flash_point = f"({x}, {y})"

    lines = [
        f"frame={frame_index}",
        f"target={int(result.target.found)} conf={result.target.confidence:.2f}",
        f"flash={int(result.flash.found)} conf={result.flash.confidence:.2f}",
        f"flash_point={flash_point}",
    ]
    _draw_text_block(image, lines, (12, 12), (255, 0, 0))


def _draw_text_block(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_height = 22
    x, y = origin
    widths = [
        cv2.getTextSize(line, font, scale, thickness)[0][0]
        for line in lines
    ]
    block_width = max(widths, default=0) + 16
    block_height = line_height * len(lines) + 10
    cv2.rectangle(
        image,
        (x - 4, y - 4),
        (x + block_width, y + block_height),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.rectangle(
        image,
        (x - 4, y - 4),
        (x + block_width, y + block_height),
        color,
        thickness=1,
    )
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x + 4, y + 18 + index * line_height),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def _iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    images: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(sorted(path.glob(pattern)))
    return images


def _is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def _round_point(point: Point | None) -> list[float] | None:
    if point is None:
        return None
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def _int_point(point: Point) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def _csv_point(point: Point | None) -> tuple[str, str]:
    if point is None:
        return "", ""
    return f"{point[0]:.2f}", f"{point[1]:.2f}"


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect blue-purple laser point.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Image, image directory, or video file. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument("--output-name", default=None, help="Optional output video name.")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=FlashConfig.min_confidence)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    flash_cfg = FlashConfig(min_confidence=args.min_confidence)

    if args.input.is_file() and _is_video_path(args.input):
        result = process_video(
            args.input,
            args.output_dir,
            flash_cfg,
            display=not args.no_display,
            output_name=args.output_name,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False))
        return

    image_paths = _iter_images(args.input)
    if not image_paths:
        raise SystemExit(f"no images found: {args.input}")

    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        result = detect_flash_point(image_path, flash_config=flash_cfg)
        payload = result.to_dict()
        payload["image"] = str(image_path)
        print(json.dumps(payload, ensure_ascii=False))

        if args.debug_dir is not None:
            annotated = annotate_flash_detection(image_path, result, flash_cfg)
            cv2.imwrite(str(args.debug_dir / image_path.name), annotated)


if __name__ == "__main__":
    main()
