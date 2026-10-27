"""Focused geometry tests for the pitch-view direction drawing marker.

The full camera module imports optional GPU packages, so these tests execute
the two production methods in a lightweight harness.
"""

import ast
import unittest
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def _load_marker_methods():
    source_path = Path(__file__).resolve().parents[1] / "camera.py"
    source = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    owner = next(
        node for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "CameraFOVCalculator"
    )
    selected = [
        node for node in owner.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"get_view_direction_marker", "_clip_segment_to_pitch"}
    ]
    namespace = {"np": np, "cv2": cv2, "Optional": Optional, "Tuple": Tuple}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


_METHODS = _load_marker_methods()


class MarkerHarness:
    get_view_direction_marker = _METHODS["get_view_direction_marker"]
    _clip_segment_to_pitch = staticmethod(_METHODS["_clip_segment_to_pitch"].__func__)

    def __init__(self, homography):
        self.h = 100
        self.w = 100
        self.H = np.asarray(homography, dtype=np.float64)

    def _project_points(self, image_points):
        return cv2.perspectiveTransform(
            np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2), self.H
        ).reshape(-1, 2)


class ViewDirectionMarkerTests(unittest.TestCase):
    def test_identity_points_towards_upper_image_not_bottom(self):
        marker = MarkerHarness(np.eye(3)).get_view_direction_marker(100, 100)
        self.assertIsNotNone(marker)
        anchor, forward = marker
        np.testing.assert_allclose(anchor, [50, 90], atol=1e-4)
        np.testing.assert_allclose(forward, [0, -1], atol=1e-4)

    def test_swapped_pitch_axes_are_not_assumed_to_have_near_minimum_y(self):
        homography = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        anchor, forward = MarkerHarness(homography).get_view_direction_marker(100, 100)
        np.testing.assert_allclose(anchor, [90, 50], atol=1e-4)
        np.testing.assert_allclose(forward, [-1, 0], atol=1e-4)

    def test_reversed_y_axis_points_to_increasing_pitch_y(self):
        homography = [[1, 0, 0], [0, -1, 100], [0, 0, 1]]
        anchor, forward = MarkerHarness(homography).get_view_direction_marker(100, 100)
        np.testing.assert_allclose(anchor, [50, 10], atol=1e-4)
        np.testing.assert_allclose(forward, [0, 1], atol=1e-4)

    def test_near_end_outside_pitch_clips_anchor_to_inner_border(self):
        homography = [[1, 0, 0], [0, 2, -80], [0, 0, 1]]
        anchor, forward = MarkerHarness(homography).get_view_direction_marker(100, 100)
        np.testing.assert_allclose(anchor, [50, 97], atol=1e-4)
        np.testing.assert_allclose(forward, [0, -1], atol=1e-4)

    def test_horizon_crossing_keeps_near_continuous_branch(self):
        # A steep half-pitch view puts the perspective horizon between the
        # last two centre-column samples. Joining points across that horizon
        # can fold the far end back onto the near anchor and hide the arrow.
        homography = [
            [-0.5711432564575831, 0.4161743419221724, -55.21570846195375],
            [-0.6597400214327566, -15.216582209562366, 7444.399082505429],
            [2.1279487938405644e-06, -0.002442437426189365, 1.0],
        ]
        calculator = MarkerHarness(homography)
        calculator.h, calculator.w = 1080, 1920
        marker = calculator.get_view_direction_marker(4750, 6000)
        self.assertIsNotNone(marker)
        anchor, forward = marker
        self.assertTrue(np.isfinite(anchor).all() and np.isfinite(forward).all())
        self.assertGreater(anchor[0], 0)
        self.assertGreater(anchor[1], 5000)
        np.testing.assert_allclose(np.linalg.norm(forward), 1.0, atol=1e-6)
        self.assertGreater(np.dot(forward, np.array([2375, 3000]) - anchor), 0)

    def test_non_overlapping_and_singular_projection_do_not_draw_marker(self):
        translated = [[1, 0, 0], [0, 1, 200], [0, 0, 1]]
        self.assertIsNone(MarkerHarness(translated).get_view_direction_marker(100, 100))
        self.assertIsNone(MarkerHarness(np.zeros((3, 3))).get_view_direction_marker(100, 100))


if __name__ == "__main__":
    unittest.main()
