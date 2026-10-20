"""Focused tests for the pitch diagram's projected image-edge diagnostics.

Load only the drawing functions from camera.py so these tests do not require
the optional GPU, tracking, or supervision packages at module import time.
"""

import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np


SOURCE_PATH = Path(__file__).resolve().parents[1] / "camera.py"
SOURCE = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))


def load_function(name, namespace):
    node = next(node for node in SOURCE.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace[name]


def load_camera_method(name, namespace):
    owner = next(node for node in SOURCE.body if isinstance(node, ast.ClassDef) and node.name == "Camera")
    node = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace[name]


NS = {"np": np, "cv2": cv2}
clip_segment = load_function("_clip_segment_to_pitch", NS)
edge_segments = load_function("_projected_frame_edge_segments", NS)


class ProjectedFrameEdgeTests(unittest.TestCase):
    def test_segment_crossing_pitch_is_clipped_to_real_field_rectangle(self):
        segment = clip_segment((-50, 40), (150, 40), 100, 80)
        self.assertIsNotNone(segment)
        np.testing.assert_allclose(segment[0], (0, 40))
        np.testing.assert_allclose(segment[1], (100, 40))

    def test_parallel_outside_or_nonfinite_segment_is_not_drawn(self):
        self.assertIsNone(clip_segment((-5, -1), (110, -1), 100, 80))
        self.assertIsNone(clip_segment((np.nan, 40), (50, 40), 100, 80))

    def test_same_y_samples_remain_in_video_order_without_cubic_fit(self):
        # Old X=f(Y) fit is singular here and also reordered points by Y.
        segments = edge_segments(np.asarray([[10, 20], [30, 20], [50, 20]]), 100, 80)
        self.assertEqual(len(segments), 2)
        np.testing.assert_allclose(segments[0][0], (10, 20))
        np.testing.assert_allclose(segments[0][1], (30, 20))
        np.testing.assert_allclose(segments[1][1], (50, 20))

    def test_invalid_or_horizon_jump_does_not_bridge_across_pitch(self):
        points = np.asarray([
            [10, 20], [20, 20], [np.nan, np.nan], [80, 20], [90, 20],
            [1000000, 1000000], [1000100, 1000100],
        ])
        segments = edge_segments(points, 100, 80)
        self.assertEqual(len(segments), 2)
        np.testing.assert_allclose(segments[0][0], (10, 20))
        np.testing.assert_allclose(segments[1][0], (80, 20))

    def test_direction_marker_uses_projected_near_point_not_legacy_camera_position(self):
        draw_marker = load_camera_method("_draw_view_direction_marker", NS)
        camera = types.SimpleNamespace(
            config=types.SimpleNamespace(length=100, width=80),
            fov_calculator=types.SimpleNamespace(
                get_view_direction_marker=lambda length, width:
                    (np.asarray([50.0, 70.0]), np.asarray([0.0, -1.0]))
            ),
            # Intentionally no CameraInfo.position / orientation attributes.
        )
        image = np.zeros((180, 220, 3), dtype=np.uint8)
        self.assertTrue(draw_marker(camera, image, scale=2, padding=10))
        self.assertEqual(int(image[150, 110, 1]), 255)  # green near-point centre

    def test_diagram_uses_distinct_frame_edge_colors_and_unambiguous_legend(self):
        captured = []
        labels = []
        namespace = dict(NS, imwritex=lambda _path, image: captured.append(image.copy()))
        plot = load_camera_method("_visualize_video_boundaries", namespace)

        def draw_pitch(config, scale, padding):
            return np.full(
                (int(config.width * scale) + 2 * padding,
                 int(config.length * scale) + 2 * padding, 3),
                (34, 139, 34), dtype=np.uint8,
            )

        pitch_tool = types.ModuleType("tools.pitch_annotation_tool")
        pitch_tool.draw_pitch = draw_pitch
        camera = types.SimpleNamespace(
            id="sample", name="sample", frame_shape=(100, 100),
            config=types.SimpleNamespace(length=100, width=80),
            homography_matrix=np.asarray([[0.6, 0, 20], [0, 0.8, 0], [0, 0, 1]], dtype=np.float32),
            distortion_params=None, distortion_info=None,
            enable_distortion_correction=False, store_dir=Path("/ignored"),
            _draw_view_direction_marker=lambda *_args, **_kwargs: False,
            _put_chinese_text=lambda image, label, *_args, **_kwargs:
                (labels.append(label), image)[1],
        )
        with patch.dict(sys.modules, {"tools.pitch_annotation_tool": pitch_tool}):
            plot(camera, num_samples=100, scale=2, padding=10)

        self.assertEqual(len(captured), 1)
        image = captured[0]
        self.assertTrue(np.any(np.all(image == (255, 255, 0), axis=2)))  # left image edge
        self.assertTrue(np.any(np.all(image == (255, 0, 255), axis=2)))  # right image edge
        np.testing.assert_array_equal(image[10, 10], (255, 255, 255))  # true pitch boundary
        self.assertTrue(any("不是场地标线" in label for label in labels))
        self.assertFalse(any("Camera Position" in label for label in labels))


if __name__ == "__main__":
    unittest.main()
