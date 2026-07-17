from __future__ import annotations

from math import log
import unittest

from twopoint_project.vision3.laser_area_mapping import (
    CALIBRATION_MAX_AREA_PX2,
    CALIBRATION_MIN_AREA_PX2,
    X_INTERCEPT,
    X_LOG_AREA_COEFFICIENT,
    Y_INTERCEPT,
    Y_LOG_AREA_COEFFICIENT,
    predict_laser_point_from_area,
    predict_laser_point_from_normalized_area,
)


class LaserAreaMappingTest(unittest.TestCase):
    def test_uses_combined_natural_log_formula(self) -> None:
        area = 100000.0

        prediction = predict_laser_point_from_area(
            area,
            frame_width=1280,
            frame_height=720,
            confidence=0.8,
        )

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertAlmostEqual(
            prediction.x_px,
            X_INTERCEPT + X_LOG_AREA_COEFFICIENT * log(area),
        )
        self.assertAlmostEqual(
            prediction.y_px,
            Y_INTERCEPT + Y_LOG_AREA_COEFFICIENT * log(area),
        )
        self.assertEqual(prediction.as_point_prediction()["label"], "laser_point")
        self.assertAlmostEqual(prediction.confidence, 0.8)
        self.assertFalse(prediction.area_clamped)

    def test_same_normalized_area_scales_to_runtime_resolution(self) -> None:
        calibration_area = 100000.0
        normalized_area = calibration_area / (1280 * 720)

        runtime = predict_laser_point_from_normalized_area(
            normalized_area,
            frame_width=1280,
            frame_height=720,
        )
        full = predict_laser_point_from_normalized_area(
            normalized_area,
            frame_width=1920,
            frame_height=1080,
        )

        self.assertIsNotNone(full)
        self.assertIsNotNone(runtime)
        assert full is not None and runtime is not None
        self.assertAlmostEqual(full.x_px, runtime.x_px * 1.5)
        self.assertAlmostEqual(full.y_px, runtime.y_px * 1.5)
        self.assertAlmostEqual(
            runtime.calibration_area_px2,
            calibration_area,
        )

    def test_out_of_range_area_is_clamped_not_extrapolated(self) -> None:
        below = predict_laser_point_from_area(
            CALIBRATION_MIN_AREA_PX2 / 2.0,
            frame_width=1280,
            frame_height=720,
        )
        above = predict_laser_point_from_area(
            CALIBRATION_MAX_AREA_PX2 * 2.0,
            frame_width=1280,
            frame_height=720,
        )

        self.assertIsNotNone(below)
        self.assertIsNotNone(above)
        assert below is not None and above is not None
        self.assertTrue(below.area_clamped)
        self.assertTrue(above.area_clamped)
        self.assertEqual(
            below.fitted_calibration_area_px2,
            CALIBRATION_MIN_AREA_PX2,
        )
        self.assertEqual(
            above.fitted_calibration_area_px2,
            CALIBRATION_MAX_AREA_PX2,
        )

    def test_invalid_area_does_not_produce_a_point(self) -> None:
        self.assertIsNone(
            predict_laser_point_from_area(
                0.0,
                frame_width=1280,
                frame_height=720,
            )
        )
        self.assertIsNone(
            predict_laser_point_from_normalized_area(
                float("nan"),
                frame_width=1280,
                frame_height=720,
            )
        )


if __name__ == "__main__":
    unittest.main()
