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
edge_line = load_function("_projected_frame_edge_line", NS)


class ProjectedFrameEdgeTests(unittest.TestCase):
    def test_segment_crossing_pitch_is_clipped_to_real_field_rectangle(self):
        segment = clip_segment((-50, 40), (150, 40), 100, 80)
        self.assertIsNotNone(segment)
        np.testing.assert_allclose(segment[0], (0, 40))
        np.testing.assert_allclose(segment[1], (100, 40))

    def test_parallel_outside_or_nonfinite_segment_is_not_drawn(self):
        self.assertIsNone(clip_segment((-5, -1), (110, -1), 100, 80))
        self.assertIsNone(clip_segment((np.nan, 40), (50, 40), 100, 80))

    def test_same_y_samples_fit_one_straight_line_without_cubic_fit(self):
        # Old X=f(Y) fit is singular here; a geometric line fit is not.
        line = edge_line(np.asarray([[10, 20], [30, 20], [50, 20]]), 100, 80)
        self.assertIsNotNone(line)
        np.testing.assert_allclose(sorted(point[0] for point in line), (10, 50))
        np.testing.assert_allclose([point[1] for point in line], (20, 20))

    def test_invalid_or_horizon_jump_does_not_bridge_across_pitch(self):
        points = np.asarray([
            [10, 20], [20, 20], [np.nan, np.nan], [80, 20], [90, 20],
            [1000000, 1000000], [np.inf, np.inf],
        ])
        self.assertIsNone(edge_line(points, 100, 80))

    def test_folded_branches_keep_only_longest_straight_run(self):
        stable = np.column_stack((np.full(30, 80.0), np.linspace(10, 70, 30)))
        folded = np.column_stack((np.linspace(0, 60, 12), np.linspace(15, 45, 12)))
        samples = np.vstack((folded, stable))
        denominators = np.concatenate((np.full(len(folded), -1.0),
                                       np.full(len(stable), 1.0)))
        line = edge_line(samples, 100, 80, denominators)
        self.assertIsNotNone(line)
        np.testing.assert_allclose([point[0] for point in line], (80, 80), atol=0.01)
        np.testing.assert_allclose(sorted(point[1] for point in line), (10, 70), atol=0.01)

    def test_direction_reversal_without_horizon_sign_still_selects_one_run(self):
        forward = np.column_stack((np.full(30, 80.0), np.linspace(10, 70, 30)))
        backward = np.column_stack((np.full(12, 80.0), np.linspace(70, 40, 12)))
        line = edge_line(np.vstack((forward, backward)), 100, 80)
        self.assertIsNotNone(line)
        np.testing.assert_allclose([point[0] for point in line], (80, 80), atol=0.01)
        np.testing.assert_allclose(sorted(point[1] for point in line), (10, 70), atol=0.01)

    def test_outside_or_too_few_projected_samples_are_not_invented(self):
        self.assertIsNone(edge_line(np.full((100, 2), 500.0), 100, 80))
        points = np.full((100, 2), np.nan)
        points[20:23] = [[80, 10], [80, 20], [80, 30]]
        self.assertIsNone(edge_line(points, 100, 80))

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

    def test_direction_marker_points_inward_even_if_projection_faces_outward(self):
        draw_marker = load_camera_method("_draw_view_direction_marker", NS)
        camera = types.SimpleNamespace(
            config=types.SimpleNamespace(length=100, width=80),
            fov_calculator=types.SimpleNamespace(
                get_view_direction_marker=lambda length, width:
                    (np.asarray([95.0, 5.0]), np.asarray([1.0, -1.0]))
            ),
        )
        image = np.zeros((180, 220, 3), dtype=np.uint8)
        arrow = []
        with patch.object(cv2, "arrowedLine", side_effect=lambda _image, start, end, *_a, **_k:
                          arrow.append((start, end))):
            self.assertTrue(draw_marker(camera, image, scale=2, padding=10))
        self.assertEqual(len(arrow), 1)
        start, end = arrow[0]
        centre = np.asarray([110.0, 90.0])  # field centre after scale/padding
        self.assertLess(np.linalg.norm(np.asarray(end) - centre),
                        np.linalg.norm(np.asarray(start) - centre))

    def test_diagram_draws_one_straight_dashed_line_per_image_edge(self):
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
            _draw_view_direction_marker=lambda *_args, **_kwargs: True,
            _put_chinese_text=lambda image, label, *_args, **_kwargs:
                (labels.append(label), image)[1],
        )
        drawn = {(255, 255, 0): [], (255, 0, 255): []}
        original_line = cv2.line

        def record_line(image, start, end, color, *args, **kwargs):
            if color in drawn:
                drawn[color].append((start, end))
            return original_line(image, start, end, color, *args, **kwargs)

        with patch.dict(sys.modules, {"tools.pitch_annotation_tool": pitch_tool}), \
             patch.object(cv2, "line", side_effect=record_line):
            plot(camera, num_samples=100, scale=2, padding=10)

        self.assertEqual(len(captured), 1)
        image = captured[0]
        self.assertTrue(np.any(np.all(image == (255, 255, 0), axis=2)))  # left image edge
        self.assertTrue(np.any(np.all(image == (255, 0, 255), axis=2)))  # right image edge
        for color, expected_x in (((255, 255, 0), 50), ((255, 0, 255), 169)):
            self.assertGreaterEqual(len(drawn[color]), 2)  # dashed, not solid
            for start, end in drawn[color]:
                self.assertLessEqual(abs(start[0] - expected_x), 1)
                self.assertLessEqual(abs(end[0] - expected_x), 1)
        np.testing.assert_array_equal(image[10, 10], (255, 255, 255))  # true pitch boundary
        self.assertEqual(labels, ["白:场线｜青/紫:画幅投影近似｜绿:视线方向"])
        self.assertEqual(image.shape[0], 220)  # compact 40-pixel footer
        self.assertFalse(any("Camera Position" in label for label in labels))

    def test_visible_grid_diagram_has_one_line_footer(self):
        captured = []
        labels = []
        namespace = dict(NS, imwritex=lambda _path, image: captured.append(image.copy()))
        plot = load_camera_method("_visualize_visible_grids_on_pitch", namespace)
        pitch_tool = types.ModuleType("tools.pitch_annotation_tool")
        pitch_tool.draw_pitch = lambda config, scale, padding: np.full(
            (int(config.width * scale) + 2 * padding,
             int(config.length * scale) + 2 * padding, 3),
            (34, 139, 34), dtype=np.uint8,
        )
        camera = types.SimpleNamespace(
            id="sample", name="sample", store_dir=Path("/ignored"),
            config=types.SimpleNamespace(length=100, width=80), grid_size=20,
            info=types.SimpleNamespace(visible_grid_cells=np.asarray([[0, 0]])),
            _draw_view_direction_marker=lambda *_args, **_kwargs: True,
            _put_chinese_text=lambda image, label, *_args, **_kwargs:
                (labels.append(label), image)[1],
        )
        with patch.dict(sys.modules, {"tools.pitch_annotation_tool": pitch_tool}):
            plot(camera, scale=2, padding=10)
        self.assertEqual(labels, ["蓝:可见区｜白:场线｜绿:视线方向"])
        self.assertEqual(captured[0].shape[0], 220)


if __name__ == "__main__":
    unittest.main()
