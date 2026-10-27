"""Regression tests for four-edge mappings seen from the opposite side."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import cv2
import numpy as np

import check_valid
from camera import CameraFOVCalculator


class AdaptiveOrientationValidationTests(unittest.TestCase):
    def setUp(self):
        self.pitch_quad = np.array(
            [[0, 0], [10000, 0], [10000, 6000], [0, 6000]], dtype=np.float32
        )
        self.normal_quad = np.array(
            [[200, 150], [1050, 180], [1160, 650], [100, 650]], dtype=np.float32
        )
        self.reversed_quad = np.array(
            [[1100, 150], [200, 180], [100, 650], [1160, 650]], dtype=np.float32
        )

    def _strict_config(self):
        return patch(
            "yaml.safe_load",
            return_value={"enable_orientation": True, "reprojection_eps": 5.0},
        )

    def test_four_edge_quad_supplies_actual_projection_direction(self):
        self.assertEqual(check_valid.orientation_sign_from_quad(self.normal_quad), 1)
        self.assertEqual(check_valid.orientation_sign_from_quad(self.reversed_quad), -1)
        with self.assertRaises(ValueError):
            check_valid.orientation_sign_from_quad(np.zeros((4, 2)))

    def test_reversed_mapping_keeps_visible_grids_and_fov_boundaries(self):
        image_to_pitch = cv2.getPerspectiveTransform(
            self.reversed_quad, self.pitch_quad
        )
        fov = CameraFOVCalculator(
            (720, 1280), image_to_pitch,
            pitch_bounds=(-5000, 15000, -3000, 9000),
            orientation_sign=check_valid.orientation_sign_from_quad(self.reversed_quad),
        )
        samples = np.array([[1000, 1000], [5000, 3000], [9000, 5000]])
        with self._strict_config(), redirect_stdout(io.StringIO()):
            valid_reversed = check_valid.is_valid(
                samples, fov.get_pixel_points, fov.get_pitch_points,
                1280, 720, orientation_sign=fov.orientation_sign,
            )
            rejected_legacy = check_valid.is_valid(
                samples, fov.get_pixel_points, fov.get_pitch_points, 1280, 720
            )
            visible_grids = fov.compute_visible_grids(None, 10000, 6000, 1000)
            camera_info = fov.calculate()
        self.assertTrue(valid_reversed.all())
        self.assertFalse(rejected_legacy.any())
        self.assertEqual(len(visible_grids), 60)
        self.assertEqual(len(camera_info.left_boundary), 50)
        self.assertEqual(len(camera_info.right_boundary), 50)

    def test_legacy_default_still_requires_positive_orientation(self):
        image_to_pitch = cv2.getPerspectiveTransform(
            self.normal_quad, self.pitch_quad
        )
        fov = CameraFOVCalculator((720, 1280), image_to_pitch)
        self.assertEqual(fov.orientation_sign, 1)
        with self._strict_config():
            valid = check_valid.is_valid(
                [[5000, 3000]], fov.get_pixel_points, fov.get_pitch_points,
                1280, 720,
            )
            reversed_sign = check_valid.is_valid(
                [[5000, 3000]], fov.get_pixel_points, fov.get_pitch_points,
                1280, 720, orientation_sign=-1,
            )
        self.assertEqual(valid.tolist(), [True])
        self.assertEqual(reversed_sign.tolist(), [False])

    def test_sign_flip_out_of_bounds_reprojection_and_degeneracy_still_reject(self):
        def piecewise_projection(points):
            result = np.asarray(points, dtype=np.float64).copy()
            mirrored = result[:, 1] >= 50
            result[mirrored, 0] = 200 - result[mirrored, 0]
            result[mirrored, 1] += 100
            return result

        def piecewise_inverse(pixels):
            result = np.asarray(pixels, dtype=np.float64).copy()
            mirrored = result[:, 1] >= 100
            result[mirrored, 0] = 200 - result[mirrored, 0]
            result[mirrored, 1] -= 100
            return result

        with self._strict_config():
            signs = check_valid.is_valid(
                [[10, 10], [10, 60]], piecewise_projection,
                piecewise_inverse, 300, 300, orientation_sign=-1,
            )
            out_of_bounds = check_valid.is_valid(
                [[10, 60], [250, 60]], piecewise_projection,
                piecewise_inverse, 300, 300, orientation_sign=-1,
            )
            reprojection_error = check_valid.is_valid(
                [[10, 60]], piecewise_projection,
                lambda pixels: piecewise_inverse(pixels) + 10,
                300, 300, orientation_sign=-1,
            )
            degenerate = check_valid.is_valid(
                [[10, 10]],
                lambda points: np.column_stack(
                    (np.asarray(points)[:, 0] + np.asarray(points)[:, 1],
                     np.full(len(points), 20))
                ),
                lambda pixels: np.tile([10, 10], (len(pixels), 1)),
                300, 300, orientation_sign=-1,
            )
        self.assertEqual(signs.tolist(), [False, True])
        self.assertEqual(out_of_bounds.tolist(), [True, False])
        self.assertEqual(reprojection_error.tolist(), [False])
        self.assertEqual(degenerate.tolist(), [False])


if __name__ == "__main__":
    unittest.main()
